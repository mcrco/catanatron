"""A room: one game, its seats, its spectators and their signed transcripts.

Each recipient (a seat, or one connection) has an :class:`Outbox`: the hash
chain of every message the server ever sent it. A seat's outbox outlives its
connections, so a client that takes its seat back on a new connection (it
signs that connection's nonce with the seat's key) gets whatever it missed
and its chain carries on unbroken.

A room goes ``waiting`` (seats fill; each remote seat commits to a seed) ->
``revealing`` (seats are locked and every commitment is published, once;
seats reveal their seeds) -> ``playing`` -> ``finished``. A room whose seats
do not all reveal in time is void (``aborted``, naming them): there is no
second round of commitments, ever, so nobody can be asked to reveal twice.
See PROTOCOL.md for the whole state machine and its invariants.

The game itself is stock Catanatron, seated with inert
:class:`~catanatron.server.config.ServerSeat` players: nothing in the server
process that decides ever holds the game. The server's own bots run in a
worker process (:mod:`catanatron.server.botproc`) and get the same redacted
messages a remote client gets.
"""

from __future__ import annotations

import asyncio
import json
import logging
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
from catanatron.server.bot import SERVER_PARAM_LIMITS, PaceBook, check_spec, move_kind
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
    rejoin_statement,
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
    step_body,
)
from catanatron.server.table import Table

logger = logging.getLogger(__name__)

MAX_NAME = 32

#: How long server bots take over their real decisions, by bot and kind of
#: move. Kept in the server process (shared by every room), so that neither
#: a worker restart nor a new game starts a bot's pace from nothing.
PACE = PaceBook()

#: Client requests a room knows, named in the errors that answer them (the
#: request's own text is never echoed).
REQUESTS = ("join", "rejoin", "spectate", "seat_bot", "reveal", "action", "ping")


@dataclass(frozen=True)
class RoomLimits:
    #: Spectators per room.
    max_spectators: int = 32
    #: How long a seat taken before the start stays held for a client that
    #: disconnected (it may take it back meanwhile).
    seat_grace_s: float = 30.0
    #: How long seats have to reveal their seeds once the room is full; a
    #: room whose seats do not all reveal in time is void.
    reveal_timeout_ms: int = 30_000
    #: Messages a connection may fall behind before it is dropped (it can
    #: rejoin and catch up).
    max_queue: int = 20_000
    #: Errors one recipient may be sent; past it, a bad request closes the
    #: connection instead.
    max_answers: int = 500
    #: Submissions (reveals and actions) a seat may make, each answered by a
    #: signed receipt: this many per decision it was asked, plus the margin.
    #: Past that the connection is closed instead (an honest client answers
    #: each decision once and resends only after a reconnect, so it never
    #: gets there).
    receipts_per_decision: int = 8
    receipt_margin: int = 64


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
        #: The nonce of the hello that opened this chain (a rejoin signs it).
        self.nonce = random_hex(16)
        #: Errors sent so far (bounded: see RoomLimits.max_answers).
        self.answers = 0
        #: Whether this chain holds the room's commitments (once at most).
        self.has_commitments = False

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
    address: Optional[str] = None
    outbox: Optional[Outbox] = None
    recipe: Optional[tuple] = None
    table: Optional[Table] = None
    pending: Optional[Pending] = None
    #: Decisions this seat was asked, and receipts it was given.
    decisions: int = 0
    receipts: int = 0
    #: Receipts held back until the step that closes their decision is out.
    held: list = field(default_factory=list)
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

    def free(self) -> None:
        self.kind, self.name, self.spec = "open", "", None
        self.public_key = self.requested_seat = self.address = None
        self.client_commitment = self.pop = self.client_seed = self.reveal_sig = None
        self.outbox = self.recipe = self.table = None


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
        on_blame: Optional[Callable[[Optional[str], Optional[str]], None]] = None,
        is_banned: Optional[Callable[[Optional[str], Optional[str]], bool]] = None,
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
        self.game_id = str(uuid.uuid4())
        self.server_seed = random_hex()
        self.seed_commitment = seed_commitment(self.server_seed)
        self.seats = [Seat(i, color) for i, color in enumerate(config.colors)]
        self.spectators: list = []
        #: Outboxes of connections that have not joined or spectated yet.
        self.lobby: set = set()
        #: Told (address, key) of every seat blamed for a withheld reveal,
        #: and asked whether an (address, key) may take a seat.
        self.on_blame = on_blame
        self.is_banned = is_banned
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
            "config": self.config.to_json(),
            "seed_commitment": self.seed_commitment,
            "seats": [seat.describe() for seat in self.seats],
            "spectators": len(self.spectators),
            "game_id": self.game_id if self.status in ("playing", "finished") else None,
        }

    def hello_body(self, nonce: str) -> dict:
        return {
            "type": "hello",
            "protocol_version": PROTOCOL_VERSION,
            "server_key": self.key.public_hex,
            "nonce": nonce,
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
        outbox.push(self.hello_body(outbox.nonce), self._commit())
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

        Returns the seat, whose outbox the connection now feeds (its own
        chain: everything this connection got so far is the seat's too)."""
        if request.get("protocol_version") != PROTOCOL_VERSION:
            raise RoomError(
                "bad_join", f"this server speaks protocol {PROTOCOL_VERSION}"
            )
        if self.status != "waiting":
            raise RoomError("game_started", "the seats are taken; spectate instead")
        public_key = request.get("public_key")
        if not is_hex(public_key, 32):
            raise RoomError("bad_join", "public_key must be 32 bytes of hex (Ed25519)")
        if self.is_banned is not None and self.is_banned(peer.address, public_key):
            raise RoomError(
                "banned", "a seat of yours withheld its seed; try again later"
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
            raise RoomError(
                "duplicate_key", "that key already holds a seat here (rejoin instead)"
            )
        seat = self._open_seat(wanted)
        seat.kind = "remote"
        seat.name = clean_name(request.get("name"), seat.color.value)
        seat.public_key = public_key
        seat.requested_seat = wanted
        seat.client_commitment = commitment
        seat.pop = request["pop"]
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
                "room": self.describe(),
            }
        )
        self._seats_changed()
        return seat

    def rejoin(self, outbox: Outbox, peer: Peer, request: dict) -> Seat:
        """Take a seat's chain back on a new connection: the seat's key signs
        this connection's nonce. Its chain is resent after ``last_seq``."""
        if request.get("protocol_version") != PROTOCOL_VERSION:
            raise RoomError(
                "bad_join", f"this server speaks protocol {PROTOCOL_VERSION}"
            )
        public_key = request.get("public_key")
        seat = next(
            (
                s
                for s in self.seats
                if s.kind == "remote" and s.public_key == public_key
            ),
            None,
        )
        if seat is None or not is_hex(public_key, 32):
            raise RoomError("not_seated", "no seat here is held by that key")
        if not verify_signature(
            public_key, rejoin_statement(self.id, outbox.nonce), request.get("sig")
        ):
            raise RoomError(
                "bad_join", "sig must sign this connection's nonce with the seat's key"
            )
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
        """Watch: this connection's chain gets what every spectator gets
        (nothing twice: a connection the lock already made a spectator just
        carries on)."""
        if outbox in self.spectators:
            return
        if len(self.spectators) >= self.limits.max_spectators:
            raise RoomError("room_full", "too many spectators")
        self.lobby.discard(outbox)
        self.spectators.append(outbox)
        if self.status == "waiting":
            outbox.push({"type": "room", "room": self.describe()})
        elif self.status == "revealing":
            self._push_commitments(outbox)
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
        limits = {
            **SERVER_PARAM_LIMITS,
            # An exec bot may not wait on its program longer than the room
            # waits on it (its worker would be killed first, every time).
            "timeout_ms": self.config.decision_timeout_ms,
        }
        try:
            check_spec(spec, limits)
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
            raise RoomError("bad_join", "no such seat")
        if self.seats[index].kind != "open":
            raise RoomError("seat_taken", f"seat {index} is taken")
        return self.seats[index]

    def _seats_changed(self) -> None:
        self.last_seen = time.time()
        self._broadcast_room()
        self._maybe_lock()

    # ===== seeds: commit, then reveal, once =====
    def _maybe_lock(self) -> None:
        if self.status != "waiting" or any(s.kind == "open" for s in self.seats):
            return
        self.status = "revealing"
        self.commitments = commitments_body(
            self.id,
            config=self.config.to_json(),
            seats=[seat.public() for seat in self.seats],
            seed_commitment=self.seed_commitment,
            reveal_timeout_ms=self.limits.reveal_timeout_ms,
        )
        self.commitments_digest = digest(self.commitments)
        # Connections still looking at the room watch it from here on: the
        # game is all there is left to see.
        for outbox in list(self.lobby):
            if outbox.peer is not None:
                if len(self.spectators) >= self.limits.max_spectators:
                    break
                self.lobby.discard(outbox)
                self.spectators.append(outbox)
        for outbox in self._outboxes(lobby=True):
            self._push_commitments(outbox)
        if all(s.kind != "remote" for s in self.seats):
            self._start()
        else:
            loop = asyncio.get_running_loop()
            self.timer = loop.call_later(
                self.limits.reveal_timeout_ms / 1000, self._reveal_timeout
            )

    def _push_commitments(self, outbox: Outbox) -> None:
        if not outbox.has_commitments and self.commitments is not None:
            outbox.has_commitments = True
            outbox.push(self.commitments)

    def _cancel_timer(self) -> None:
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None

    def on_reveal(self, seat: Seat, peer: Peer, request: dict) -> None:
        """A seat's reveal. Answered by a receipt either way, so that a seat
        blamed for withholding after the server acknowledged its reveal can
        show it."""
        if self.status != "revealing":
            raise RoomError("not_revealing", "seeds are not being revealed now")
        self._check_connection(seat, peer)
        if not self._allow_receipt(seat, peer):
            return
        signature = request.get("sig")
        statement = reveal_statement(self.commitments_digest)
        submission = submission_digest(statement, signature)
        client_seed = request.get("client_seed")
        if seat.client_seed is not None:
            accepted = seat.reveal_sig == signature and seat.client_seed == client_seed
            reason = "accepted" if accepted else "refused: this seat has revealed"
        elif (
            not is_hex(client_seed, 32)
            or client_seed_commitment(self.id, str(seat.public_key), client_seed)
            != seat.client_commitment
        ):
            accepted, reason = False, "refused: that seed does not open your commitment"
        elif not verify_signature(seat.public_key, statement, signature):
            accepted, reason = False, "refused: sig must sign these commitments"
        else:
            accepted, reason = True, "accepted"
            seat.client_seed, seat.reveal_sig = client_seed, signature
        assert seat.outbox is not None
        seat.outbox.push(receipt_body(submission, None, accepted, reason))
        if all(s.client_seed is not None for s in self.seats if s.kind == "remote"):
            self._cancel_timer()
            self._start()

    def _reveal_timeout(self) -> None:
        """Seats did not reveal in time: the game is void, and they are
        blamed. There is no second round: a new game needs a new room, with
        new commitments and new seeds from everyone."""
        self.timer = None
        if self.status != "revealing":
            return
        blamed = [s for s in self.seats if s.kind == "remote" and s.client_seed is None]
        for seat in blamed:
            if self.on_blame is not None:
                self.on_blame(seat.address, seat.public_key)
        self.abort(
            "a seat did not reveal its seed in time: the game is void",
            [s.index for s in blamed],
        )

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
            # Up now, so that exec bots watch the game from its first action.
            # A worker that cannot start costs the next bot decision (its
            # call tries again and fails as a bot_error), never the room.
            try:
                self.worker.start()
            except Exception:
                logger.exception("room %s: the bot worker did not start", self.id)
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
            self._flush_held(seat)
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

    # ===== requests from clients =====
    def _check_connection(self, seat: Seat, peer: Peer) -> None:
        if seat.outbox is None or seat.outbox.peer is not peer:
            raise RoomError(
                "stale_connection", "this connection no longer holds the seat"
            )

    def _allow_receipt(self, seat: Seat, peer: Peer) -> bool:
        """Whether a submission gets its receipt; if not, the connection is
        closed (so the client knows it was not answered)."""
        allowance = (
            self.limits.receipt_margin
            + self.limits.receipts_per_decision * seat.decisions
        )
        if seat.receipts >= allowance:
            peer.kick(4009, "too many submissions")
            return False
        seat.receipts += 1
        return True

    def answer(self, outbox: Outbox, body: dict, commit: Optional[dict] = None) -> bool:
        """Push an error, within the recipient's allowance."""
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
        server had its answer before the move was played. A receipt for a
        decision that has closed (its deadline passed, or it was answered)
        is held until the step that closes it is out: a receipt that comes
        before step ``k`` always means the server had the answer in time."""
        self._check_connection(seat, peer)
        if self.game is None:
            raise RoomError("no_game", "there is no move to make before the game")
        if not self._allow_receipt(seat, peer):
            return
        statement, signature = request.get("statement"), request.get("sig")
        submission = submission_digest(statement, signature)
        index = statement.get("index") if isinstance(statement, dict) else None
        index = index if type(index) is int else None
        pending = seat.pending
        if pending is not None and index == pending.index:
            if pending.future.done():
                seat.held.append(
                    receipt_body(
                        submission, index, False, "stale: the decision has closed"
                    )
                )
                return
            reason = self._accept(seat, pending, statement, signature)
            body = receipt_body(submission, index, reason == "accepted", reason)
        else:
            body = receipt_body(
                submission, index, False, "stale: not the pending decision"
            )
        assert seat.outbox is not None
        seat.outbox.push(body, self._commit())

    def _flush_held(self, seat: Seat) -> None:
        held, seat.held = seat.held, []
        if seat.outbox is not None:
            for body in held:
                seat.outbox.push(body, self._commit())

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
        self._flush_held(seat)

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
        if len(playable) == 1:
            # One move, but nobody else can know it (it depends on the seat's
            # hidden cards): answered as slowly as this bot's real decisions
            # that end in this kind of move, from a book no worker restart
            # empties.
            kind = move_kind(body["playable_actions"][0])
            await asyncio.sleep(min(PACE.draw(seat.spec, kind), self._timeout()))
            return playable[0], None, None
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
            if self.status == "playing":  # (else the game ended under it)
                logger.exception("room %s: bot %s failed", self.id, seat.spec)
            return "bot_error"
        if reply[0] != "decided":
            logger.warning("room %s: bot %s failed: %s", self.id, seat.spec, reply[1:])
            return "bot_error"
        wanted = canonical(reply[1])
        for action, offered in zip(playable, body["playable_actions"]):
            if canonical(offered) == wanted:
                PACE.record(seat.spec, move_kind(offered), self.worker.last_call_s)
                return action, None, None
        return "bot_error"

    async def _remote_decision(self, seat: Seat, playable: list):
        """Ask the seat and wait for its answer or the deadline. The
        deadline closes the decision in one step (no receipt for this index
        can slip in before the step: see :meth:`on_action`)."""
        assert seat.outbox is not None and self.doc is not None
        body = decide_body(
            self.game,
            self.doc,
            seat.color.value,
            timeout_ms=self.config.decision_timeout_ms,
        )
        envelope = seat.outbox.push(body, self._commit())
        seat.decisions += 1
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        seat.pending = Pending(
            index=len(self.actions),
            seq=envelope["seq"],
            decide_hash=envelope_hash(envelope),
            playable=playable,
            playable_json=body["playable_actions"],
            future=future,
        )

        def deadline() -> None:
            if not future.done():
                future.set_result("timeout")

        timer = loop.call_later(self._timeout(), deadline)
        try:
            return await future
        except asyncio.CancelledError:
            return "timeout"
        finally:
            timer.cancel()
            seat.pending = None

    def _stop_bots(self) -> None:
        """Let every server bot hear the game ended (exec bots get ``after``
        and their programs are stopped), then end the worker, off the loop."""
        worker = self.worker
        for seat in self.bot_seats():  # (posts reach only a live worker)
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
        # Let go of the worker now, so that a shutdown right behind this
        # cannot kill it before its bots have finished.
        worker.stop()

    def kill_bots(self) -> None:
        """At shutdown: no grace."""
        self.worker.kill()

    def error(
        self, outbox: Outbox, code: str, message: str, request: Optional[dict] = None
    ) -> None:
        """Refuse a request: which kind of request this answers (if it is
        one this room knows), never what it said."""
        kind = request.get("type") if isinstance(request, dict) else None
        self.answer(
            outbox,
            {
                "type": "error",
                "code": code,
                "message": message[:200],
                "answers": kind if kind in REQUESTS else None,
            },
            self._commit(),
        )
