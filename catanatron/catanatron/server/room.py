"""A room: one game, its seats, its spectators and their signed transcripts.

Each recipient (a seat, or one spectator connection) has an :class:`Outbox`:
the hash chain of every message the server ever sent it. A seat's outbox
outlives its connections, so a client that reconnects with its seat key
gets whatever it missed and its chain carries on unbroken.

The game itself is stock Catanatron, seated with inert
:class:`~catanatron.server.config.ServerSeat` players: nothing in the server
process that decides (the server's own bots included) ever holds the game;
they get the same redacted messages a remote client gets.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
import unicodedata
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from catanatron.game import Game, is_valid_action
from catanatron.models.enums import Action, ActionRecord
from catanatron.models.player import Color
from catanatron.serialization import action_from_json, action_to_json, state_to_json
from catanatron.server.bot import ViewBot
from catanatron.server.config import GameConfig, default_action, is_over, new_game
from catanatron.server.crypto import (
    GENESIS,
    SigningKey,
    action_statement,
    envelope_hash,
    is_hex,
    master_seed,
    random_hex,
    seal,
    seed_commitment,
    state_commitment,
    SALT_BYTES,
    verify_signature,
)
from catanatron.server.messages import (
    PROTOCOL_VERSION,
    after_body,
    before_body,
    decide_body,
    step_body,
)
from catanatron.server.table import Table

logger = logging.getLogger(__name__)

MAX_NAME = 32


class Peer:
    """A live connection, written to in order by one task."""

    def __init__(self, websocket: Any):
        self.websocket = websocket
        self.queue: asyncio.Queue = asyncio.Queue()
        self.writer = asyncio.create_task(self._write())

    def send(self, payload: dict) -> None:
        self.queue.put_nowait(json.dumps(payload))

    async def _write(self) -> None:
        try:
            while True:
                await self.websocket.send(await self.queue.get())
        except Exception:  # the connection is gone; the reader cleans up
            pass

    def close(self) -> None:
        self.writer.cancel()


class Outbox:
    """Every message to one recipient: a signed hash chain."""

    def __init__(self, key: SigningKey, peer: Optional[Peer] = None):
        self.key = key
        self.to: Optional[str] = None
        self.envelopes: list = []
        self.head = GENESIS
        self.peer = peer

    def push(self, body: dict, commit: Optional[dict] = None) -> dict:
        envelope = seal(
            self.key,
            to=self.to,
            seq=len(self.envelopes),
            prev=self.head,
            commit=commit,
            body=body,
        )
        self.envelopes.append(envelope)
        self.head = envelope_hash(envelope)
        if self.peer is not None:
            self.peer.send(envelope)
        return envelope

    def attach(self, peer: Peer, last_seq: int) -> None:
        """Send ``peer`` everything after ``last_seq``, then keep it posted."""
        self.peer = peer
        for envelope in self.envelopes[max(last_seq + 1, 0) :]:
            peer.send(envelope)


@dataclass
class Pending:
    """A decision the server is waiting for."""

    index: int
    seq: int
    decide_hash: str
    playable: list
    future: asyncio.Future


@dataclass
class Seat:
    index: int
    color: Color
    kind: str = "open"  # open | remote | bot
    name: str = ""
    spec: Optional[str] = None
    public_key: Optional[str] = None
    client_seed: Optional[str] = None
    seat_key: Optional[str] = None
    outbox: Optional[Outbox] = None
    bot: Optional[ViewBot] = None
    table: Optional[Table] = None
    pending: Optional[Pending] = None

    def describe(self) -> dict:
        return {
            "seat": self.index,
            "color": self.color.value,
            "kind": self.kind,
            "name": self.name,
            "spec": self.spec,
            "public_key": self.public_key,
            "connected": self.kind == "bot"
            or (self.outbox is not None and self.outbox.peer is not None),
        }


class RoomError(Exception):
    """A request the room refuses; ``code`` tells clients why."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def clean_name(name: Any, fallback: str) -> str:
    text = "".join(
        ch
        for ch in str(name or "")
        if unicodedata.category(ch)[0] not in "CZ" or ch == " "
    ).strip()
    return text[:MAX_NAME] or fallback


class Room:
    """One game. Seats fill (remote clients or server bots); when every seat
    is taken the game starts, and when it ends the server reveals its seed."""

    def __init__(
        self,
        room_id: str,
        config: GameConfig,
        key: SigningKey,
        *,
        name: str = "",
        on_finish=None,
    ):
        self.id = room_id
        self.config = config
        self.key = key
        self.name = clean_name(name, room_id)
        self.created = time.time()
        self.status = "waiting"  # waiting | playing | finished
        self.game_id = str(uuid.uuid4())
        self.server_seed = random_hex()
        self.seed_commitment = seed_commitment(self.server_seed)
        self.seats = [Seat(i, color) for i, color in enumerate(config.colors)]
        self.spectators: list = []
        #: Outboxes of connections that have not joined or spectated yet.
        self.lobby: set = set()
        self.game: Optional[Game] = None
        self.doc: Optional[dict] = None
        self.salts: list = []
        self.commitments: list = []
        self.actions: list = []
        self.task: Optional[asyncio.Task] = None
        self.finished = asyncio.Event()
        self.on_finish = on_finish

    # ===== describing =====
    def describe(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "config": self.config.to_json(),
            "seed_commitment": self.seed_commitment,
            "seats": [seat.describe() for seat in self.seats],
            "spectators": len(self.spectators),
            "game_id": self.game_id if self.status != "waiting" else None,
        }

    def hello_body(self) -> dict:
        return {
            "type": "hello",
            "protocol_version": PROTOCOL_VERSION,
            "server_key": self.key.public_hex,
            "room": self.describe(),
        }

    def _broadcast_room(self) -> None:
        body = {"type": "room", "room": self.describe()}
        for outbox in self._outboxes(include_lobby=True):
            outbox.push(body, self._commit())

    def _outboxes(self, include_lobby: bool = False) -> list:
        boxes = [s.outbox for s in self.seats if s.outbox is not None] + self.spectators
        if include_lobby:
            boxes += [box for box in self.lobby if box.peer is not None]
        return boxes

    # ===== connections =====
    def connect(self, peer: Peer) -> Outbox:
        """A new connection: its outbox, opened with a hello."""
        outbox = Outbox(self.key, peer)
        outbox.push(self.hello_body(), self._commit())
        self.lobby.add(outbox)
        return outbox

    def disconnect(self, outbox: Outbox, peer: Peer) -> None:
        self.lobby.discard(outbox)
        if outbox in self.spectators:
            self.spectators.remove(outbox)
        if outbox.peer is peer:
            outbox.peer = None
        peer.close()
        if self.status == "waiting":
            self._broadcast_room()

    def join(self, outbox: Outbox, peer: Peer, request: dict) -> Outbox:
        """A join request on a connection whose outbox is ``outbox``.

        Returns the outbox the connection now feeds (the seat's, if it is
        taking back a seat it held)."""
        if request.get("protocol_version") != PROTOCOL_VERSION:
            raise RoomError(
                "bad_join", f"this server speaks protocol {PROTOCOL_VERSION}"
            )
        public_key = request.get("public_key")
        if not is_hex(public_key, 32):
            raise RoomError("bad_join", "public_key must be 32 bytes of hex (Ed25519)")
        if request.get("seat_key") is not None:
            return self._rejoin(outbox, peer, request)

        if self.status != "waiting":
            raise RoomError("game_started", "the game has started; spectate instead")
        client_seed = request.get("client_seed")
        if not is_hex(client_seed, 32):
            raise RoomError("bad_join", "client_seed must be 32 bytes of hex")
        seat = self._open_seat(request.get("seat"))
        seat.kind = "remote"
        seat.name = clean_name(request.get("name"), seat.color.value)
        seat.public_key = public_key
        seat.client_seed = client_seed
        seat.seat_key = random_hex(16)
        seat.outbox = outbox
        self.lobby.discard(outbox)
        outbox.to = public_key
        outbox.push(
            {
                "type": "joined",
                "room_id": self.id,
                "seat": seat.index,
                "color": seat.color.value,
                "public_key": public_key,
                "client_seed": client_seed,
                "seat_key": seat.seat_key,
            }
        )
        self._seats_changed()
        return outbox

    def _rejoin(self, outbox: Outbox, peer: Peer, request: dict) -> Outbox:
        index = request.get("seat")
        if type(index) is not int or not 0 <= index < len(self.seats):
            raise RoomError("bad_join", "a rejoin names its seat")
        seat = self.seats[index]
        if (
            seat.kind != "remote"
            or not hmac.compare_digest(str(seat.seat_key), str(request.get("seat_key")))
            or seat.public_key != request.get("public_key")
        ):
            raise RoomError("bad_join", "wrong seat key or public key for that seat")
        last_seq = request.get("last_seq", -1)
        if type(last_seq) is not int:
            raise RoomError("bad_join", "last_seq must be an integer")
        self.lobby.discard(outbox)
        old = seat.outbox
        assert old is not None
        if old.peer is not None and old.peer is not peer:
            old.peer.close()
        old.attach(peer, last_seq)
        if self.status == "waiting":
            self._broadcast_room()
        return old

    def spectate(self, outbox: Outbox, request: dict) -> None:
        self.lobby.discard(outbox)
        self.spectators.append(outbox)
        if self.game is not None and self.doc is not None:
            outbox.push(self._before(None), self._commit())
            if self.status == "finished":
                outbox.push(
                    after_body(self.game, self.doc, None, reveal=self._reveal()),
                    self._commit(),
                )
        else:
            self._broadcast_room()

    def set_bot(self, index: Any, spec: Any) -> None:
        if self.status != "waiting":
            raise RoomError("game_started", "seats are fixed once the game starts")
        if not isinstance(spec, str):
            raise RoomError("bad_bot", "spec must be a registry spec such as F or AB:2")
        seat = self._open_seat(index)
        try:
            bot = ViewBot(spec, seat.color)
        except Exception as error:  # SpecError, or a bot that fails to build
            raise RoomError("bad_bot", str(error))
        if not bot.player.is_bot:
            raise RoomError("bad_bot", f"{spec!r} is not a bot")
        seat.kind, seat.spec, seat.bot = "bot", spec, bot
        seat.name = spec
        self._seats_changed()

    def _open_seat(self, index: Any) -> Seat:
        if index is None:
            for seat in self.seats:
                if seat.kind == "open":
                    return seat
            raise RoomError("room_full", "no open seat")
        if type(index) is not int or not 0 <= index < len(self.seats):
            raise RoomError("bad_join", f"no seat {index!r}")
        if self.seats[index].kind != "open":
            raise RoomError("seat_taken", f"seat {index} is taken")
        return self.seats[index]

    def _seats_changed(self) -> None:
        self._broadcast_room()
        if self.status == "waiting" and all(s.kind != "open" for s in self.seats):
            self.status = "playing"
            self.task = asyncio.create_task(self._run())

    # ===== actions from clients =====
    def on_action(self, seat: Seat, request: dict) -> None:
        pending = seat.pending
        if pending is None or pending.future.done():
            raise RoomError("not_pending", "no decision is pending for this seat")
        statement, signature = request.get("statement"), request.get("sig")
        if not isinstance(statement, dict) or not isinstance(signature, str):
            raise RoomError("bad_action", "an action is {statement, sig}")
        expected = action_statement(
            self.game_id, pending.seq, pending.decide_hash, statement.get("action")
        )
        if statement != expected:
            raise RoomError(
                "bad_action", "the statement does not answer the pending decide"
            )
        if not verify_signature(str(seat.public_key), statement, signature):
            raise RoomError("bad_action", "bad signature")
        try:
            action = action_from_json(statement["action"])
        except Exception:
            raise RoomError(
                "bad_action", f"cannot parse action {statement['action']!r}"
            )
        assert self.game is not None
        if action.color != seat.color or not is_valid_action(
            pending.playable, self.game.state, action
        ):
            raise RoomError("bad_action", f"{statement['action']} is not legal now")
        pending.future.set_result((action, statement, signature))

    # ===== the game =====
    def _commit(self) -> Optional[dict]:
        if not self.commitments:
            return None
        index = len(self.commitments) - 1
        return {"index": index, "hash": self.commitments[index]}

    def _record_state(self) -> None:
        assert self.game is not None
        self.doc = state_to_json(self.game)
        salt = random_hex(SALT_BYTES)
        self.salts.append(salt)
        self.commitments.append(state_commitment(self.doc, salt))

    def _seat_of(self, color: Color) -> Seat:
        return self.seats[self.config.colors.index(color)]

    def _before(self, color: Optional[str]) -> dict:
        assert self.game is not None and self.doc is not None
        return before_body(
            self.game,
            self.doc,
            color,
            config=self.config.to_json(),
            seats=[
                {k: v for k, v in s.describe().items() if k != "connected"}
                for s in self.seats
            ],
            client_seeds=self._client_seeds(),
            seed_commitment=self.seed_commitment,
        )

    def _client_seeds(self) -> list:
        return [s.client_seed if s.kind == "remote" else None for s in self.seats]

    def _reveal(self) -> dict:
        return {
            "server_seed": self.server_seed,
            "salts": list(self.salts),
            "actions": list(self.actions),
        }

    async def _run(self) -> None:
        try:
            await self._play()
        except Exception:
            logger.exception("room %s crashed", self.id)
            self.status = "finished"
            for outbox in self._outboxes():
                self.error(outbox, "aborted", "the game was aborted by a server error")
        finally:
            self.finished.set()
            if self.on_finish is not None:
                self.on_finish(self)

    def _master_seed(self) -> bytes:
        return master_seed(self.server_seed, self._client_seeds())

    async def _play(self) -> None:
        self.game = new_game(self.config, self._master_seed(), self.game_id)
        self._record_state()
        self._send_all(lambda color: self._before(color))
        while not is_over(self.game):
            await self._turn()
        self.status = "finished"
        reveal = self._reveal()
        self._send_all(
            lambda color: after_body(self.game, self.doc, color, reveal=reveal)
        )

    def _send_all(self, build) -> None:
        commit = self._commit()
        for seat in self.seats:
            body = build(seat.color.value)
            if seat.outbox is not None:
                seat.outbox.push(body, commit)
            if seat.table is not None or seat.bot is not None:
                seat.table = seat.table or Table()
                seat.table.update(body)
        if self.spectators:
            body = build(None)
            for outbox in self.spectators:
                outbox.push(body, commit)

    async def _turn(self) -> None:
        game = self.game
        assert game is not None and self.doc is not None
        index = len(self.actions)
        color = game.state.current_color()
        seat = self._seat_of(color)
        playable = list(game.playable_actions)
        entry: dict = {"index": index, "color": color.value, "by": "server"}
        forced = None

        if self.config.autoplay_forced and len(playable) == 1:
            action, forced = playable[0], "only"
        else:
            decided = await (
                self._bot_decision(seat, playable)
                if seat.kind == "bot"
                else self._remote_decision(seat, playable)
            )
            if decided is None:
                action, forced = self._timeout_action(playable), "timeout"
            else:
                action, statement, signature = decided
                if statement is not None:
                    entry.update(by="client", statement=statement, sig=signature)
        entry.update(action=action_to_json(action), forced=forced)

        self._execute(action)
        self.actions.append(entry)
        self._record_state()
        self._send_all(lambda c: step_body(game, self.doc, c, forced=forced))

    def _timeout_action(self, playable: list) -> Action:
        return default_action(playable)

    def _execute(self, action: Action) -> ActionRecord:
        """Carry out ``action``. The one place the game changes."""
        assert self.game is not None
        return self.game.execute(action)

    def _timeout(self) -> float:
        return self.config.decision_timeout_ms / 1000

    async def _bot_decision(self, seat: Seat, playable: list):
        assert seat.bot is not None and seat.table is not None and self.doc is not None
        body = decide_body(
            self.game,
            self.doc,
            seat.color.value,
            timeout_ms=self.config.decision_timeout_ms,
        )
        seat.table.update(body)
        try:
            choice = await asyncio.wait_for(
                asyncio.to_thread(
                    seat.bot.decide,
                    seat.table.view(),
                    body["playable_actions"],
                    self.config.game_supply,
                ),
                self._timeout(),
            )
        except asyncio.TimeoutError:
            logger.warning("room %s: bot %s timed out", self.id, seat.spec)
            return None
        return action_from_json(choice), None, None

    async def _remote_decision(self, seat: Seat, playable: list):
        assert seat.outbox is not None and self.doc is not None
        body = decide_body(
            self.game,
            self.doc,
            seat.color.value,
            timeout_ms=self.config.decision_timeout_ms,
        )
        envelope = seat.outbox.push(body, self._commit())
        future = asyncio.get_running_loop().create_future()
        seat.pending = Pending(
            index=len(self.actions),
            seq=envelope["seq"],
            decide_hash=envelope_hash(envelope),
            playable=playable,
            future=future,
        )
        try:
            return await asyncio.wait_for(future, self._timeout())
        except asyncio.TimeoutError:
            return None
        finally:
            seat.pending = None

    def error(
        self, outbox: Outbox, code: str, message: str, request: Optional[dict] = None
    ) -> None:
        body = {"type": "error", "code": code, "message": message}
        statement = (request or {}).get("statement")
        if isinstance(statement, dict) and type(statement.get("seq")) is int:
            body["seq"] = statement["seq"]  # which decide the refusal is about
        outbox.push(body, self._commit())
