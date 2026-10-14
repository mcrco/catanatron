"""A room: one game, its seats, its spectators and their signed transcripts.

Each recipient (a seat, or one connection) has an :class:`Outbox`: the hash
chain of every message the server ever sent it. A seat's outbox outlives its
connections, so a client that reconnects with its seat key gets whatever it
missed and its chain carries on unbroken.

A room goes ``waiting`` (seats fill; each remote seat commits to a seed) ->
``revealing`` (seats are locked and every commitment is published; seats
reveal their seeds) -> ``playing`` -> ``finished``. A seat that does not
reveal (or, after a restart, commit) in time is removed and blamed, its key
and address are barred from the room, the server's seed is revealed and
replaced, every revealed seed is thrown away, and the room goes back to
``waiting`` for a fresh round; after a few rounds it is ``aborted``.

The game itself is stock Catanatron, seated with inert
:class:`~catanatron.server.config.ServerSeat` players: nothing in the server
process that decides ever holds the game. The server's own bots run in a
worker process (:mod:`catanatron.server.botproc`) and get the same redacted
messages a remote client gets.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import threading
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from catanatron.game import Game
from catanatron.models.enums import Action, ActionRecord
from catanatron.models.player import Color
from catanatron.registry import REGISTRY, parse_spec
from catanatron.serialization import action_to_json, state_to_json
from catanatron.server.bot import SERVER_PARAM_LIMITS, check_spec
from catanatron.server.botproc import BotWorker, recipe_for
from catanatron.server.config import (
    GameConfig,
    default_action,
    is_over,
    new_game,
    publicly_forced,
)
from catanatron.server.crypto import (
    GENESIS,
    SALT_BYTES,
    SigningKey,
    action_statement,
    canonical,
    client_seed_commitment,
    digest,
    envelope_hash,
    is_hex,
    join_statement,
    master_seed,
    random_hex,
    reveal_statement,
    seal,
    seed_commitment,
    state_commitment,
    submission_digest,
    verify_signature,
)
from catanatron.server.messages import (
    PROTOCOL_VERSION,
    aborted_body,
    after_body,
    before_body,
    commitments_body,
    decide_body,
    receipt_body,
    restart_body,
    step_body,
)
from catanatron.server.table import Table

logger = logging.getLogger(__name__)

MAX_NAME = 32


@dataclass(frozen=True)
class RoomLimits:
    #: Spectators per room.
    max_spectators: int = 32
    #: How long a seat taken before the start stays held for a client that
    #: disconnected (it may rejoin with its seat key meanwhile).
    seat_grace_s: float = 30.0
    #: How long seats have to reveal their seeds once the room is full (and,
    #: after a restart, to commit afresh).
    reveal_timeout_ms: int = 30_000
    #: Seed rounds before a room whose seats keep withholding is called off.
    max_rounds: int = 3
    #: Messages a connection may fall behind before it is dropped (it can
    #: rejoin and catch up).
    max_queue: int = 20_000
    #: Answers to requests (errors and receipts) one recipient may be sent;
    #: past it, bad requests close the connection instead.
    max_answers: int = 500


class Peer:
    """A live connection, written to in order by one task."""

    def __init__(self, websocket: Any, max_queue: int = 20_000, address: str = ""):
        self.websocket = websocket
        self.address = address
        self.max_queue = max_queue
        self.closed = False
        self.queue: asyncio.Queue = asyncio.Queue()
        self.writer = asyncio.create_task(self._write())

    def send(self, payload: dict) -> None:
        if self.closed:
            return
        if self.queue.qsize() >= self.max_queue:
            self.kick(4008, "too far behind; rejoin to catch up")
            return
        self.queue.put_nowait(json.dumps(payload))

    async def _write(self) -> None:
        try:
            while True:
                await self.websocket.send(await self.queue.get())
        except Exception:  # the connection is gone; the reader cleans up
            pass

    def kick(self, code: int, reason: str) -> None:
        """Close the connection (its reader then cleans up)."""
        if self.closed:
            return
        self.closed = True
        self.writer.cancel()
        asyncio.get_running_loop().create_task(self._close(code, reason))

    async def _close(self, code: int, reason: str) -> None:
        try:
            await self.websocket.close(code, reason)
        except Exception:
            pass

    def close(self) -> None:
        self.closed = True
        self.writer.cancel()


class Outbox:
    """Every message to one recipient: a signed hash chain."""

    def __init__(self, key: SigningKey, peer: Optional[Peer] = None):
        self.key = key
        self.to: Optional[str] = None
        self.envelopes: list = []
        self.head = GENESIS
        self.peer = peer
        #: Errors and receipts sent so far (bounded: see RoomLimits.max_answers).
        self.answers = 0

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
    playable: list  # the server's own Action objects
    playable_json: list
    future: asyncio.Future
    receipts: int = 0


@dataclass
class Seat:
    index: int
    color: Color
    kind: str = "open"  # open | remote | bot
    name: str = ""
    spec: Optional[str] = None
    public_key: Optional[str] = None
    #: The seat it asked for (None: any), as its join signature covers it.
    requested_seat: Optional[int] = None
    client_commitment: Optional[str] = None
    pop: Optional[str] = None
    client_seed: Optional[str] = None
    reveal_sig: Optional[str] = None
    seat_key: Optional[str] = None
    address: Optional[str] = None
    outbox: Optional[Outbox] = None
    recipe: Optional[tuple] = None
    table: Optional[Table] = None
    pending: Optional[Pending] = None
    grace: Optional[asyncio.TimerHandle] = field(default=None, repr=False)

    @property
    def connected(self) -> bool:
        return self.kind == "bot" or (
            self.outbox is not None and self.outbox.peer is not None
        )

    def public(self) -> dict:
        """What everyone is told about the seat (and what gets locked)."""
        return {
            "seat": self.index,
            "color": self.color.value,
            "kind": self.kind,
            "name": self.name,
            "spec": self.spec,
            "public_key": self.public_key,
            "requested_seat": self.requested_seat,
            "client_commitment": self.client_commitment,
            "pop": self.pop,
        }

    def describe(self) -> dict:
        return {**self.public(), "connected": self.connected}

    def forget_seed(self) -> None:
        self.client_commitment = self.pop = self.client_seed = self.reveal_sig = None
        self.requested_seat = self.index

    def free(self) -> None:
        self.kind, self.name, self.spec = "open", "", None
        self.public_key = self.requested_seat = self.address = None
        self.client_commitment = self.pop = self.client_seed = self.reveal_sig = None
        self.seat_key = self.outbox = self.recipe = self.table = None


class RoomError(Exception):
    """A request the room refuses; ``code`` tells clients why."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def clean_name(name: Any, fallback: str) -> str:
    text = "".join(
        ch
        for ch in str(name or "")[:200]
        if unicodedata.category(ch)[0] not in "CZ" or ch == " "
    ).strip()
    return text[:MAX_NAME] or fallback


class Room:
    """One game. Seats fill (remote clients or server bots); when every seat
    is taken they reveal their seeds and the game starts; when it ends the
    server reveals its seed."""

    def __init__(
        self,
        room_id: str,
        config: GameConfig,
        key: SigningKey,
        *,
        name: str = "",
        on_finish=None,
        limits: RoomLimits = RoomLimits(),
        on_blame: Optional[Callable[[Optional[str]], None]] = None,
        is_banned: Optional[Callable[[Optional[str]], bool]] = None,
    ):
        self.id = room_id
        self.config = config
        self.key = key
        self.name = clean_name(name, room_id)
        self.limits = limits
        #: When someone was last connected, and when the room ended.
        self.created = self.last_seen = time.time()
        self.ended: Optional[float] = None
        self.status = "waiting"
        self.round = 0
        self.game_id = str(uuid.uuid4())
        self.server_seed = random_hex()
        self.seed_commitment = seed_commitment(self.server_seed)
        self.seats = [Seat(i, color) for i, color in enumerate(config.colors)]
        self.spectators: list = []
        #: Outboxes of connections that have not joined or spectated yet.
        self.lobby: set = set()
        self.banned_keys: set = set()
        self.banned_addresses: set = set()
        self.on_blame = on_blame
        self.is_banned = is_banned
        self.locked: Optional[list] = None
        self.commitments: Optional[dict] = None
        self.commitments_digest: Optional[str] = None
        self.game: Optional[Game] = None
        self.doc: Optional[dict] = None
        self.salts: list = []
        self.state_commitments: list = []
        self.actions: list = []
        self.task: Optional[asyncio.Task] = None
        self.timer: Optional[asyncio.TimerHandle] = None
        self.worker = BotWorker()
        self.finished = asyncio.Event()
        self.on_finish = on_finish
        self._room_broadcast = False
        self._aborted_body: Optional[dict] = None

    # ===== describing =====
    def describe(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "round": self.round,
            "config": self.config.to_json(),
            "seed_commitment": self.seed_commitment,
            "seats": [seat.describe() for seat in self.seats],
            "spectators": len(self.spectators),
            "game_id": self.game_id if self.status in ("playing", "finished") else None,
        }

    def hello_body(self) -> dict:
        return {
            "type": "hello",
            "protocol_version": PROTOCOL_VERSION,
            "server_key": self.key.public_hex,
            "room": self.describe(),
        }

    def has_peers(self) -> bool:
        return any(s.kind == "remote" and s.connected for s in self.seats) or any(
            box.peer is not None for box in self.spectators
        )

    def bot_seats(self) -> list:
        return [s for s in self.seats if s.kind == "bot"]

    def _broadcast_room(self) -> None:
        """Tell everyone connected the room changed, once per loop tick."""
        if self.status != "waiting" or self._room_broadcast:
            return
        self._room_broadcast = True
        asyncio.get_running_loop().call_soon(self._flush_room)

    def _flush_room(self) -> None:
        self._room_broadcast = False
        if self.status != "waiting":
            return
        body = {"type": "room", "room": self.describe()}
        for outbox in self._outboxes(lobby=True):
            outbox.push(body)

    def _outboxes(self, lobby: bool = False) -> list:
        seated = [s.outbox for s in self.seats if s.outbox is not None]
        boxes = seated + self.spectators
        if lobby:
            boxes += [box for box in self.lobby if box.peer is not None]
        return boxes

    # ===== connections =====
    def connect(self, peer: Peer) -> Outbox:
        """A new connection: its outbox, opened with a hello."""
        self.last_seen = time.time()
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
            seat = self._seat_of_outbox(outbox)
            if seat is not None and self.status == "waiting":
                # Hold the seat a little while for a reconnect, then free it.
                loop = asyncio.get_running_loop()
                seat.grace = loop.call_later(
                    self.limits.seat_grace_s, self._free_if_absent, seat, outbox
                )
        peer.close()
        self._broadcast_room()

    def _free_if_absent(self, seat: Seat, outbox: Outbox) -> None:
        if self.status == "waiting" and seat.outbox is outbox and outbox.peer is None:
            seat.free()
            self._broadcast_room()

    def _seat_of_outbox(self, outbox: Outbox) -> Optional[Seat]:
        return next((s for s in self.seats if s.outbox is outbox), None)

    def join(self, outbox: Outbox, peer: Peer, request: dict) -> Seat:
        """A join request on a connection whose outbox is ``outbox``.

        Returns the seat, whose outbox the connection now feeds."""
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
            raise RoomError("game_started", "the seats are taken; spectate instead")
        if (
            public_key in self.banned_keys
            or peer.address in self.banned_addresses
            or (self.is_banned is not None and self.is_banned(peer.address))
        ):
            raise RoomError(
                "banned", "you withheld a seed here (or too often elsewhere)"
            )
        commitment = request.get("client_commitment")
        if not is_hex(commitment, 32):
            raise RoomError("bad_join", "client_commitment must be 32 bytes of hex")
        wanted = request.get("seat")
        if wanted is not None and type(wanted) is not int:
            raise RoomError("bad_join", "seat must be an integer or null")
        statement = join_statement(self.id, self.seed_commitment, wanted, commitment)
        if not verify_signature(public_key, statement, request.get("pop")):
            raise RoomError("bad_join", "pop must sign the join statement with the key")
        if any(s.public_key == public_key for s in self.seats):
            raise RoomError("duplicate_key", "that key already holds a seat here")
        seat = self._open_seat(wanted)
        seat.kind = "remote"
        seat.name = clean_name(request.get("name"), seat.color.value)
        seat.public_key = public_key
        seat.requested_seat = wanted
        seat.client_commitment = commitment
        seat.pop = request["pop"]
        seat.seat_key = random_hex(16)
        seat.address = peer.address
        seat.outbox = outbox
        self.lobby.discard(outbox)
        outbox.to = public_key
        self.last_seen = time.time()
        outbox.push(
            {
                "type": "joined",
                "room_id": self.id,
                "seat": seat.index,
                "color": seat.color.value,
                "public_key": public_key,
                "client_commitment": commitment,
                "seat_key": seat.seat_key,
                "room": self.describe(),
            }
        )
        self._seats_changed()
        return seat

    def _rejoin(self, outbox: Outbox, peer: Peer, request: dict) -> Seat:
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
            # The seat moved: the old connection is closed, so it neither
            # hangs silently nor acts for the seat any more.
            old.peer.kick(4001, "seat taken over by another connection")
        if seat.grace is not None:
            seat.grace.cancel()
            seat.grace = None
        old.attach(peer, last_seq)
        self.last_seen = time.time()
        self._broadcast_room()
        return seat

    def spectate(self, outbox: Outbox, request: dict) -> None:
        if len(self.spectators) >= self.limits.max_spectators:
            raise RoomError("room_full", "too many spectators")
        self.lobby.discard(outbox)
        self.spectators.append(outbox)
        if self.status == "waiting":
            outbox.push({"type": "room", "room": self.describe()})
        elif self.status == "revealing":
            outbox.push(self.commitments)
        elif self.game is not None and self.doc is not None:
            outbox.push(self._before(None), self._commit())
            if self.status == "finished":
                outbox.push(
                    after_body(self.game, self.doc, None, reveal=self._reveal()),
                    self._commit(),
                )
        if self.status == "aborted" and self._aborted_body is not None:
            outbox.push(self._aborted_body, self._commit())

    def set_bot(self, index: Any, spec: Any, address: Optional[str] = None) -> Seat:
        if self.status != "waiting":
            raise RoomError("game_started", "seats are fixed once every seat is taken")
        if not isinstance(spec, str) or len(spec) > 100:
            raise RoomError("bad_bot", "spec must be a registry spec such as F or AB:2")
        if index is not None and type(index) is not int:
            raise RoomError("bad_bot", "seat must be an integer")
        seat = self._open_seat(index)
        try:
            check_spec(spec, SERVER_PARAM_LIMITS)
            key = parse_spec(spec)[0]
            recipe = recipe_for(key, REGISTRY.lookup(key))
        except Exception as error:  # SpecError, or a bot that fails to build
            raise RoomError("bad_bot", str(error)[:200])
        seat.kind, seat.spec, seat.recipe, seat.name = "bot", spec, recipe, spec
        seat.address = address  # who seated it, for the per-address limit
        self._seats_changed()
        return seat

    def _open_seat(self, index: Any) -> Seat:
        if index is None:
            for seat in self.seats:
                if seat.kind == "open":
                    return seat
            raise RoomError("room_full", "no open seat")
        if not 0 <= index < len(self.seats):
            raise RoomError("bad_join", f"no seat {index!r}")
        if self.seats[index].kind != "open":
            raise RoomError("seat_taken", f"seat {index} is taken")
        return self.seats[index]

    def _seats_changed(self) -> None:
        self.last_seen = time.time()
        self._broadcast_room()
        self._maybe_lock()

    # ===== seeds: commit, then reveal =====
    def _uncommitted(self) -> list:
        return [
            s for s in self.seats if s.kind == "remote" and s.client_commitment is None
        ]

    def _maybe_lock(self) -> None:
        if self.status != "waiting" or any(s.kind == "open" for s in self.seats):
            return
        if self._uncommitted():
            # A restart left seats to commit afresh: they get the usual time.
            if self.timer is None:
                loop = asyncio.get_running_loop()
                self.timer = loop.call_later(
                    self.limits.reveal_timeout_ms / 1000, self._withheld, "commit"
                )
            return
        self._cancel_timer()
        self.status = "revealing"
        self.locked = [seat.public() for seat in self.seats]
        self.commitments = commitments_body(
            self.id,
            round=self.round,
            config=self.config.to_json(),
            seats=self.locked,
            seed_commitment=self.seed_commitment,
            reveal_timeout_ms=self.limits.reveal_timeout_ms,
        )
        self.commitments_digest = digest(self.commitments)
        for outbox in self._outboxes(lobby=True):
            outbox.push(self.commitments)
        if all(s.kind != "remote" for s in self.seats):
            self._start()
        else:
            loop = asyncio.get_running_loop()
            self.timer = loop.call_later(
                self.limits.reveal_timeout_ms / 1000, self._withheld, "reveal"
            )

    def _cancel_timer(self) -> None:
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None

    def on_commit(self, seat: Seat, peer: Peer, request: dict) -> None:
        """A seated client's fresh commitment after a restart."""
        if self.status != "waiting" or seat.client_commitment is not None:
            raise RoomError("not_committing", "no fresh commitment is due now")
        if seat.outbox is None or seat.outbox.peer is not peer:
            raise RoomError(
                "stale_connection", "this connection no longer holds the seat"
            )
        commitment = request.get("client_commitment")
        if not is_hex(commitment, 32):
            raise RoomError("bad_commit", "client_commitment must be 32 bytes of hex")
        statement = join_statement(
            self.id, self.seed_commitment, seat.index, commitment
        )
        if not verify_signature(seat.public_key, statement, request.get("pop")):
            raise RoomError(
                "bad_commit", "pop must sign the join statement for this seat"
            )
        seat.client_commitment, seat.pop = commitment, request["pop"]
        seat.requested_seat = seat.index
        self._seats_changed()

    def on_reveal(self, seat: Seat, peer: Peer, request: dict) -> None:
        if self.status != "revealing":
            raise RoomError("not_revealing", "seeds are not being revealed now")
        if seat.outbox is None or seat.outbox.peer is not peer:
            raise RoomError(
                "stale_connection", "this connection no longer holds the seat"
            )
        if seat.client_seed is not None:
            return  # a resend after a reconnect
        client_seed = request.get("client_seed")
        if (
            not is_hex(client_seed, 32)
            or client_seed_commitment(self.id, str(seat.public_key), client_seed)
            != seat.client_commitment
        ):
            raise RoomError("bad_reveal", "that seed does not open your commitment")
        signature = request.get("sig")
        if not verify_signature(
            seat.public_key, reveal_statement(self.commitments_digest), signature
        ):
            raise RoomError("bad_reveal", "sig must sign the commitments you were sent")
        seat.client_seed, seat.reveal_sig = client_seed, signature
        if all(s.client_seed is not None for s in self.seats if s.kind == "remote"):
            self._cancel_timer()
            self._start()

    def _withheld(self, what: str) -> None:
        """Seats did not reveal (or commit) in time: they are out, and the
        others start a fresh round, so withholding delays a game but cannot
        bias it (every revealed seed is discarded, the server's included)."""
        self.timer = None
        if what == "reveal" and self.status != "revealing":
            return
        if what == "commit" and self.status != "waiting":
            return
        blamed = [
            s
            for s in self.seats
            if s.kind == "remote"
            and (
                s.client_seed is None
                if what == "reveal"
                else s.client_commitment is None
            )
        ]
        blame = [s.index for s in blamed]
        for seat in blamed:
            self.banned_keys.add(seat.public_key)
            if seat.address:
                self.banned_addresses.add(seat.address)
            if self.on_blame is not None:
                self.on_blame(seat.address)
        if self.round + 1 >= self.limits.max_rounds:
            self.abort(f"seats did not {what} their seeds, round after round", blame)
            return
        old_seed = self.server_seed
        self.round += 1
        self.server_seed = random_hex()
        self.seed_commitment = seed_commitment(self.server_seed)
        for seat in blamed:
            if seat.outbox is not None and seat.outbox.peer is not None:
                seat.outbox.peer.kick(4003, "you did not reveal your seed in time")
            seat.free()
        for seat in self.seats:
            if seat.kind == "remote":
                seat.forget_seed()
        self.status = "waiting"
        self.locked = self.commitments = self.commitments_digest = None
        body = restart_body(
            self.id,
            round=self.round,
            reason=f"a seat did not {what} its seed in time",
            blame=blame,
            server_seed=old_seed,
            seed_commitment=self.seed_commitment,
            room=self.describe(),
        )
        for outbox in self._outboxes(lobby=True):
            outbox.push(body)
        self._maybe_lock()

    def _start(self) -> None:
        self.status = "playing"
        for seat in self.bot_seats():
            self.worker.declare(
                seat.index,
                parse_spec(seat.spec)[0],
                seat.recipe,
                seat.spec,
                seat.color.value,
            )
        if self.bot_seats():
            self.worker.start()
        self.task = asyncio.get_running_loop().create_task(self._run())

    def abort(self, reason: str, blame: Optional[list] = None) -> None:
        """End the room without a game result, saying why and who is to
        blame, and reveal the server seed (nothing is left to protect)."""
        if self.status in ("finished", "aborted"):
            return
        self.status = "aborted"
        self.ended = time.time()
        self._cancel_timer()
        for seat in self.seats:
            if seat.pending is not None and not seat.pending.future.done():
                seat.pending.future.cancel()
        self._aborted_body = aborted_body(
            self.id,
            reason=reason,
            blame=list(blame or []),
            server_seed=self.server_seed,
        )
        for outbox in self._outboxes(lobby=True):
            outbox.push(self._aborted_body, self._commit())
        self._stop_bots()
        self.finished.set()
        if self.on_finish is not None:
            self.on_finish(self)

    # ===== actions from clients =====
    def answer(self, outbox: Outbox, body: dict, commit: Optional[dict] = None) -> bool:
        """Push an error or receipt, within the recipient's allowance."""
        if outbox.answers >= self.limits.max_answers:
            if outbox.peer is not None:
                outbox.peer.kick(4009, "too many bad requests")
            return False
        outbox.answers += 1
        outbox.push(body, commit)
        return True

    def on_action(self, seat: Seat, peer: Peer, request: dict) -> None:
        """A submitted action. Every submission gets a signed receipt naming
        it (statement and signature together), so a client can prove that the
        server had its answer before the move was played."""
        if seat.outbox is None or seat.outbox.peer is not peer:
            raise RoomError(
                "stale_connection", "this connection no longer holds the seat"
            )
        statement, signature = request.get("statement"), request.get("sig")
        submission = submission_digest(statement, signature)
        pending = seat.pending
        accepted, reason = False, "accepted"
        if pending is None or pending.future.done():
            reason = "stale: no decision is pending for this seat"
        elif not isinstance(statement, dict) or statement.get("index") != pending.index:
            reason = "stale: not the pending decision"
        else:
            pending.receipts += 1
            if pending.receipts > 8:
                return  # the decision has had its answers' worth of receipts
            reason = self._accept(seat, pending, statement, signature)
            accepted = reason == "accepted"
        self.answer(
            seat.outbox,
            receipt_body(
                submission, pending.index if pending else None, accepted, reason
            ),
            self._commit(),
        )

    def _accept(
        self, seat: Seat, pending: Pending, statement: dict, signature: Any
    ) -> str:
        expected = action_statement(
            self.commitments_digest,
            pending.index,
            pending.seq,
            pending.decide_hash,
            statement.get("action"),
        )
        # Compare encodings, not values: 15.0 == 15 in Python, not on the wire.
        if canonical(statement) != canonical(expected):
            return "refused: the statement does not answer the pending decide"
        if not verify_signature(seat.public_key, statement, signature):
            return "refused: bad signature"
        wanted = canonical(statement["action"])
        for action, offered in zip(pending.playable, pending.playable_json):
            if canonical(offered) == wanted:
                # Play the server's own Action, never the client's encoding.
                pending.future.set_result((action, statement, signature))
                return "accepted"
        return "refused: that action was not offered"

    # ===== the game =====
    def _commit(self) -> Optional[dict]:
        if not self.state_commitments:
            return None
        index = len(self.state_commitments) - 1
        return {"index": index, "hash": self.state_commitments[index]}

    def _record_state(self) -> None:
        assert self.game is not None
        self.doc = state_to_json(self.game)
        salt = random_hex(SALT_BYTES)
        self.salts.append(salt)
        self.state_commitments.append(state_commitment(self.doc, salt))

    def _seat_of(self, color: Color) -> Seat:
        return self.seats[self.config.colors.index(color)]

    def _before(self, color: Optional[str]) -> dict:
        assert self.game is not None and self.doc is not None
        return before_body(
            self.game,
            self.doc,
            color,
            commitments=self.commitments,
            client_seeds=self._client_seeds(),
            reveals=[s.reveal_sig if s.kind == "remote" else None for s in self.seats],
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
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("room %s crashed", self.id)
            self.abort("the server failed")
        finally:
            if self.status == "finished":
                self.finished.set()
                if self.on_finish is not None:
                    self.on_finish(self)

    def _master_seed(self) -> bytes:
        return master_seed(
            self.server_seed, self._client_seeds(), self.commitments_digest
        )

    async def _play(self) -> None:
        self.game = new_game(self.config, self._master_seed(), self.game_id)
        self._record_state()
        self._send_all(lambda color: self._before(color))
        while not is_over(self.game):
            if self.status != "playing":
                return
            await self._turn()
            await asyncio.sleep(0)  # never starve the loop, even on forced moves
        if not self.game.playable_actions and self.game.state.is_initial_build_phase:
            self.abort("the map ran out of settlement spots")
            return
        self.status = "finished"
        self.ended = time.time()
        reveal = self._reveal()
        self._send_all(
            lambda color: after_body(self.game, self.doc, color, reveal=reveal)
        )
        self._stop_bots()

    def _send_all(self, build) -> None:
        commit = self._commit()
        for seat in self.seats:
            body = build(seat.color.value)
            if seat.outbox is not None:
                seat.outbox.push(body, commit)
            if seat.kind == "bot":
                seat.table = seat.table or Table()
                if body["type"] == "step" and seat.table.state is not None:
                    # Bots that watch the game (exec bots asking to observe)
                    # see each action with the view from before it.
                    self.worker.post(
                        (
                            "step",
                            seat.index,
                            seat.table.view(),
                            body["log"][0][0],
                            self.config.supply,
                        )
                    )
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

        if self._autoplay(game):
            action, forced = playable[0], "only"
        else:
            decided = await (
                self._bot_decision(seat, playable)
                if seat.kind == "bot"
                else self._remote_decision(seat, playable)
            )
            if self.status != "playing":
                return
            if isinstance(decided, str):  # no decision: why
                action, forced = self._timeout_action(playable), decided
            else:
                action, statement, signature = decided
                if statement is not None:
                    entry.update(by="client", statement=statement, sig=signature)
        entry.update(action=action_to_json(action), forced=forced)

        self._execute(action)
        self.actions.append(entry)
        self._record_state()
        self._send_all(lambda c: step_body(game, self.doc, c, forced=forced))

    def _autoplay(self, game: Game) -> bool:
        """Whether to play the move without asking (when everyone can see
        it is the only one)."""
        return self.config.autoplay_forced and publicly_forced(game)

    def _timeout_action(self, playable: list) -> Action:
        return default_action(playable)

    def _execute(self, action: Action) -> ActionRecord:
        """Carry out ``action``. The one place the game changes."""
        assert self.game is not None
        return self.game.execute(action)

    def _timeout(self) -> float:
        return self.config.decision_timeout_ms / 1000

    async def _bot_decision(self, seat: Seat, playable: list):
        assert seat.table is not None and self.doc is not None
        body = decide_body(
            self.game,
            self.doc,
            seat.color.value,
            timeout_ms=self.config.decision_timeout_ms,
        )
        seat.table.update(body)
        message = (
            "decide",
            seat.index,
            seat.table.view(),
            body["playable_actions"],
            self.config.supply,
        )
        try:
            reply = await self.worker.call(message, self._timeout())
        except asyncio.TimeoutError:
            # The worker was killed, computation and all; the next decision
            # starts a fresh one.
            logger.warning("room %s: bot %s timed out", self.id, seat.spec)
            return "timeout"
        except asyncio.CancelledError:
            raise
        except BaseException:  # the worker died; never let a bot take the room down
            logger.exception("room %s: bot %s failed", self.id, seat.spec)
            return "bot_error"
        if reply[0] != "decided":
            logger.warning("room %s: bot %s failed: %s", self.id, seat.spec, reply[1:])
            return "bot_error"
        wanted = canonical(reply[1])
        for action, offered in zip(playable, body["playable_actions"]):
            if canonical(offered) == wanted:
                return action, None, None
        return "bot_error"

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
            playable_json=body["playable_actions"],
            future=future,
        )
        try:
            return await asyncio.wait_for(future, self._timeout())
        except (asyncio.TimeoutError, asyncio.CancelledError):
            return "timeout"
        finally:
            seat.pending = None

    def _stop_bots(self) -> None:
        """Let every server bot hear the game ended (exec bots get ``after``
        and their programs are stopped), then end the worker, off the loop."""
        worker = self.worker
        if worker.process is None:
            return
        for seat in self.bot_seats():
            if seat.table is not None and seat.table.state is not None:
                worker.post(
                    (
                        "finish",
                        seat.index,
                        seat.table.view(),
                        self.config.supply,
                        seat.table.winning_color,
                    )
                )
        threading.Thread(target=worker.stop, daemon=True).start()

    def kill_bots(self) -> None:
        """At shutdown: no grace."""
        self.worker.kill()

    def error(
        self, outbox: Outbox, code: str, message: str, request: Optional[dict] = None
    ) -> None:
        self.answer(
            outbox,
            {"type": "error", "code": code, "message": message[:200]},
            self._commit(),
        )
