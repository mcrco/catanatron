"""An async client for catanatron-server: takes a seat, plays it, keeps proof.

::

    from catanatron.server.client import GameClient

    async def main():
        client = GameClient("ws://localhost:8765/room/ROOM_ID", name="me")
        result = await client.play(lambda view, actions: actions[0])

The client does the fairness work on its own: it checks every message's
signature and its place in the hash chain as it arrives; commits to a fresh
seed when it joins (signing the join with its key) and reveals it only once
the server has published every seat's commitment, for those commitments and
no others, ever; signs every action with the action's index; saves its
transcript when the game ends and (by default) verifies it with
:func:`catanatron.server.verify.verify_transcript`. A room whose seats do not
all reveal is void: :attr:`GameResult.aborted` says why and who is to blame.

The decision callback gets the seat's view (a ``client_view`` document, see
:class:`~catanatron.server.table.Table`) and the playable actions as JSON
(``[color, action_type, value]``), and returns one of them, an ``Action``, or
an awaitable of either. A synchronous callback runs in a worker thread, so a
slow bot does not stall the connection.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional
from urllib.parse import urlsplit, urlunsplit

import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatus

from catanatron.models.enums import Action
from catanatron.serialization import action_to_json
from catanatron.server.config import GameConfig
from catanatron.server.crypto import (
    GENESIS,
    SigningKey,
    action_statement,
    canonical,
    client_seed_commitment,
    digest,
    envelope_hash,
    is_hex,
    join_statement,
    random_hex,
    rejoin_statement,
    reveal_statement,
    seed_commitment,
    submission_digest,
    verify_envelope,
    verify_signature,
)
from catanatron.server.hashing import numbers_ok
from catanatron.server.messages import (
    COMMITMENTS_FIELDS,
    GAME_MESSAGES,
    PROTOCOL_VERSION,
    SEAT_FIELDS,
)
from catanatron.server.table import Table

logger = logging.getLogger(__name__)

TRANSCRIPT_FORMAT = "catanatron-transcript/5"

Decide = Callable[[dict, list], Any]


class ServerMisbehaved(RuntimeError):
    """A message failed its signature or hash-chain check, or contradicted
    what the client agreed to."""


class Refused(RuntimeError):
    """The server refused a request (a seat that is taken, a bad room...)."""


@dataclass
class GameResult:
    room_id: str
    game_id: Optional[str]
    color: Optional[str]
    winning_color: Optional[str]
    transcript_path: Optional[str]
    verified: Optional[bool]
    report: Any = None
    #: The ``aborted`` body, if the game was called off.
    aborted: Optional[dict] = None

    def summary(self) -> dict:
        return {
            "room_id": self.room_id,
            "game_id": self.game_id,
            "color": self.color,
            "winning_color": self.winning_color,
            "transcript": self.transcript_path,
            "verified": self.verified,
            "aborted": self.aborted,
            "errors": getattr(self.report, "errors", []),
        }


def lobby_url(url: str) -> str:
    """``ws://host:port/lobby`` for any URL on that server."""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, "/lobby", "", ""))


async def lobby_request(url: str, request: dict) -> dict:
    async with websockets.connect(lobby_url(url), max_size=None) as ws:
        await ws.send(json.dumps(request))
        reply = json.loads(await ws.recv())
    if reply.get("type") == "error":
        raise Refused(reply.get("message"))
    return reply


async def create_room(
    url: str, config: dict, seats: Optional[list] = None, name: str = ""
) -> dict:
    """Create a room; ``seats`` lists a registry spec (a server bot) or None
    (open) per seat. Returns the room's description."""
    reply = await lobby_request(
        url, {"type": "create_room", "config": config, "seats": seats, "name": name}
    )
    return reply["room"]


async def list_rooms(url: str) -> list:
    return (await lobby_request(url, {"type": "list_rooms"}))["rooms"]


class _StartOver(Exception):
    """The server holds no seat for our key, before any seed was asked for
    (a join that never landed, or a seat freed while we were away): connect
    afresh and join again."""


class GameClient:
    """Plays one seat (or watches, with ``spectate=True``) of one room.

    The client keeps the protocol's invariants on its own (see PROTOCOL.md):
    it signs exactly one reveal per room, for the first ``commitments`` it
    receives (a different second one is the server misbehaving), with the
    seed it joined with and no other; it signs at most one answer per
    decision and resends, after a reconnect, exactly what it signed and has
    no receipt for. Whatever happens, it saves the transcript it holds.
    """

    #: How long to wait, once the game is over, for receipts still owed.
    RECEIPT_WAIT_S = 2.0

    def __init__(
        self,
        url: str,
        *,
        name: str = "client",
        seat: Optional[int] = None,
        key: Optional[SigningKey] = None,
        key_path: Optional[str] = None,
        spectate: bool = False,
        server_key: Optional[str] = None,
        transcript_dir: Optional[str] = "transcripts",
        verify: bool = True,
        reconnect_attempts: int = 5,
    ):
        match = re.search(r"/room/([A-Za-z0-9_-]+)/?$", urlsplit(url).path)
        if not match:
            raise ValueError(f"expected ws://HOST:PORT/room/ROOM_ID, got {url!r}")
        self.url = url
        self.room_id = match.group(1)
        self.name = name
        self.wanted_seat = seat
        if key is None:
            key = (
                SigningKey.load_or_create(key_path)
                if key_path
                else SigningKey.generate()
            )
        self.key = key
        self.spectate = spectate
        #: Pin the server's identity; None trusts the first hello (TOFU).
        self.server_key = server_key
        self.transcript_dir = transcript_dir
        self.verify = verify
        self.reconnect_attempts = reconnect_attempts

        # One seed for this room, made here and never used for anything else.
        self.client_seed = random_hex()
        self.client_commitment = client_seed_commitment(
            self.room_id, self.key.public_hex, self.client_seed
        )
        self.table = Table()
        self.room: Optional[dict] = None
        self.seat: Optional[int] = None
        self._color: Optional[str] = None
        self.server_commitment: Optional[str] = None
        self.config: Optional[dict] = None
        #: The room's commitments: the first ever received, the only ones.
        self.commitments: Optional[dict] = None
        self.messages: list = []
        self.sent: list = []
        #: Decisions closed (by a newer message) before this client answered:
        #: its record of being timed out (index, decide seq, ms it had).
        self.unanswered: list = []
        self.head = GENESIS
        #: The decide message awaiting our answer, and the answer once made.
        self._pending: Optional[dict] = None
        self._pending_since = 0.0
        self._answer: Optional[dict] = None
        self._decision: Optional[asyncio.Task] = None
        #: Our join went out (we may hold a seat even if 'joined' was lost),
        #: on the connection whose hello carried this nonce.
        self._join_sent = False
        self._join_nonce: Optional[str] = None
        #: Our one reveal, once signed; resent as is until it has a receipt.
        self._reveal: Optional[dict] = None
        self._unreceipted: dict = {}
        self._ending: Optional[dict] = None
        self._ending_deadline = 0.0
        self._attempts = 0
        self._start_overs = 0
        #: The last move this client was asked to decide.
        self._last_decided = -1
        self._saved_path: Optional[str] = None
        #: Called with every body once the client has taken it in.
        self.on_message: Optional[Callable[[dict], None]] = None
        #: The live connection, if any.
        self.connection: Any = None

    # ----- the transcript -----
    def transcript(self) -> dict:
        return {
            "format": TRANSCRIPT_FORMAT,
            "url": self.url,
            "room_id": self.room_id,
            "server_key": self.server_key,
            "public_key": None if self.spectate else self.key.public_hex,
            "client_seed": None if self.spectate else self.client_seed,
            "reveal_sig": None if self._reveal is None else self._reveal["sig"],
            "seat": self.seat,
            "color": self.table.color,
            "messages": self.messages,
            "sent": self.sent,
            "unanswered": self.unanswered,
        }

    def save_transcript(self) -> Optional[str]:
        if not self.transcript_dir or not self.messages:
            return None
        os.makedirs(self.transcript_dir, exist_ok=True)
        who = self.table.color or ("spectator" if self.spectate else "seat")
        game = self.table.game_id or "nogame"
        path = os.path.join(self.transcript_dir, f"{self.room_id}-{game}-{who}.json")
        with open(path, "w") as file:
            json.dump(self.transcript(), file)
        self._saved_path = path
        return path

    # ----- receiving -----
    def _accept(self, envelope: dict) -> dict:
        """Check a message's signature and chain link, record it, return its body."""
        if self.server_key is None:
            body = envelope.get("body", {})
            if body.get("type") != "hello" or not is_hex(body.get("server_key"), 32):
                raise ServerMisbehaved(
                    "the first message must be a hello with the server's key"
                )
            self.server_key = body["server_key"]
        if not verify_envelope(self.server_key, envelope):
            raise ServerMisbehaved(f"bad signature on message {envelope.get('seq')}")
        if not numbers_ok(envelope):
            raise ServerMisbehaved(
                f"message {envelope.get('seq')} holds a number the protocol does not "
                "allow (a float, or beyond 2**53)"
            )
        if envelope["seq"] != len(self.messages) or envelope["prev"] != self.head:
            raise ServerMisbehaved(
                f"message {envelope['seq']} breaks the hash chain "
                f"(expected seq {len(self.messages)})"
            )
        seated = self.seat is not None and not self.spectate
        allowed = (self.key.public_hex,) if seated else (None, self.key.public_hex)
        if envelope["to"] not in allowed:
            raise ServerMisbehaved(
                f"message {envelope['seq']} was addressed to someone else"
            )
        self.messages.append(envelope)
        self.head = envelope_hash(envelope)
        return envelope["body"]

    # ----- playing -----
    async def play(self, decide: Optional[Decide] = None) -> GameResult:
        """Join, play until the game ends, save and verify the transcript.

        If it raises (the server misbehaved, refused us, or the connection
        could not be kept), the transcript received so far is saved first:
        it may be the evidence."""
        if decide is None and not self.spectate:
            raise ValueError("a seated client needs a decide callback")
        self._attempts = 0
        try:
            return await self._play(decide)
        except BaseException:
            try:
                self.save_transcript()
            except Exception:
                logger.exception("could not save the transcript")
            raise

    async def _play(self, decide: Optional[Decide]) -> GameResult:
        while True:
            try:
                return await self._session(decide)
            except _StartOver:
                self._start_overs += 1
                if self._start_overs > self.reconnect_attempts:
                    raise Refused("the server keeps forgetting our seat")
                self.messages, self.head, self.seat = [], GENESIS, None
                self._join_sent = False
                continue
            except (ConnectionClosed, OSError, InvalidStatus) as error:
                if isinstance(error, InvalidStatus) or not (
                    self._join_sent or self.spectate
                ):
                    raise
                closed = getattr(error, "rcvd", None)
                if closed is not None and closed.code == 4009:
                    # Past the seat's allowance of submissions (which an
                    # honest client never reaches): resending would only be
                    # refused again. (A flood close, 1008, is a throttle.)
                    raise
                self._attempts += 1
                if self._attempts > self.reconnect_attempts:
                    raise
                logger.info(
                    "connection lost (%s); reconnecting (%d)", error, self._attempts
                )
                await asyncio.sleep(min(0.2 * 2**self._attempts, 5))

    async def _session(self, decide: Optional[Decide]) -> GameResult:
        async with websockets.connect(self.url, max_size=None) as ws:
            self.connection = ws
            first = json.loads(await ws.recv())
            rejoining = self._join_sent and not self.spectate
            if rejoining:
                # This connection's hello opens a throw-away chain; the seat's
                # own chain resumes after our last message. Our key signs the
                # hello's nonce to show the seat is ours.
                if not verify_envelope(self.server_key, first):
                    raise ServerMisbehaved("bad signature on hello")
                nonce = (first.get("body") or {}).get("nonce")
                if not is_hex(nonce, 16):
                    raise ServerMisbehaved("the hello carries no nonce to sign")
                self._attempts = 0  # reconnected: count failures afresh
                await ws.send(
                    json.dumps(
                        {
                            "type": "rejoin",
                            "protocol_version": PROTOCOL_VERSION,
                            "public_key": self.key.public_hex,
                            "sig": self.key.sign(rejoin_statement(self.room_id, nonce)),
                            "last_seq": len(self.messages) - 1,
                        }
                    )
                )
                await self._resume(ws, decide)
            else:
                if self.spectate:
                    # A spectator's chain is per connection: start over.
                    self.messages, self.head, self.table = [], GENESIS, Table()
                body = self._accept(first)
                self._on_hello(body)
                await ws.send(json.dumps(self._join_request(body)))
                self._join_sent = not self.spectate
                self._notify(body)

            resyncing = rejoining
            loop = asyncio.get_running_loop()
            while True:
                if self._ending is not None:
                    # The game is over; collect the receipts still owed for
                    # our answers (briefly), so the transcript holds them.
                    left = self._ending_deadline - loop.time()
                    if not self._unreceipted or left <= 0:
                        return self._finish(aborted=self._ending.get("aborted"))
                    try:
                        text = await asyncio.wait_for(ws.recv(), left)
                    except (asyncio.TimeoutError, ConnectionClosed):
                        return self._finish(aborted=self._ending.get("aborted"))
                else:
                    text = await ws.recv()
                envelope = json.loads(text)
                if resyncing:
                    if not self._resumes_chain(envelope):
                        continue
                    resyncing = False
                body = self._accept(envelope)
                self._attempts = 0  # the connection works: count afresh
                if self._ending is not None and body.get("type") not in (
                    "received",
                    "error",
                ):
                    raise ServerMisbehaved(
                        f"a {str(body.get('type'))[:40]} message after the end"
                    )
                result = await self._handle(ws, body, decide)
                self._notify(body)
                if result is not None and self._ending is None:
                    self._ending = result
                    self._ending_deadline = loop.time() + self.RECEIPT_WAIT_S

    def _resumes_chain(self, envelope: dict) -> bool:
        """While rejoining: skip (after checking) what arrives on the
        throw-away chain until the seat's chain resumes. A message for our
        key that does not continue our chain means the server resent
        another chain: that is not skipped, it is the server misbehaving."""
        if (
            envelope.get("seq") == len(self.messages)
            and envelope.get("prev") == self.head
        ):
            return True
        if not verify_envelope(self.server_key, envelope):
            raise ServerMisbehaved("bad signature while rejoining")
        if self.seat is not None and envelope.get("to") == self.key.public_hex:
            raise ServerMisbehaved("the server resent another chain for this seat")
        body = envelope.get("body", {})
        if body.get("type") == "error":
            if body.get("code") == "not_seated":
                if self.commitments is not None or self._reveal is not None:
                    # Once the seats were locked, a seat is never freed.
                    raise ServerMisbehaved(
                        "the server forgot a seat after locking the seats"
                    )
                raise _StartOver()
            raise Refused(body.get("message"))
        return False

    async def _resume(self, ws, decide: Optional[Decide]) -> None:
        """After a rejoin: finish what the drop interrupted, by resending
        exactly what was signed and has no receipt (the reveal, answers).
        Resending is harmless: the server answers a stale submission with a
        receipt that says so, which no verifier holds against it."""
        for request in list(self._unreceipted.values()):
            if request["type"] == "reveal" and self.table.state is not None:
                continue  # the game started: the reveal was taken
            await ws.send(json.dumps(request))
        if (
            self._pending is not None
            and self._answer is None
            and (self._decision is None or self._decision.done())
        ):
            self._start_decision(decide)

    def _notify(self, body: dict) -> None:
        if self.on_message is not None:
            self.on_message(body)

    def _check_room(self, room: Any, where: str) -> None:
        """Every description of the room names it, its seed commitment and
        its rules, as the first hello did."""
        if (
            not isinstance(room, dict)
            or room.get("id") != self.room_id
            or room.get("seed_commitment") != self.server_commitment
            or canonical(room.get("config")) != canonical(self.config)
        ):
            raise ServerMisbehaved(
                f"{where} changes the room's seed commitment or rules"
            )

    def _on_hello(self, body: dict) -> None:
        if body.get("protocol_version") != PROTOCOL_VERSION:
            raise Refused(f"server speaks protocol {body.get('protocol_version')}")
        room = body["room"]
        if room.get("id") != self.room_id:
            raise ServerMisbehaved("the hello is for another room")
        if self.server_commitment is not None:
            # Starting over after a lost join: the room is the same room.
            self._check_room(room, "the hello")
        self.room = room
        self.server_commitment = room["seed_commitment"]
        self.config = room["config"]

    def _join_request(self, hello: dict) -> dict:
        if self.spectate:
            return {"type": "spectate", "name": self.name}
        nonce = hello.get("nonce")
        if not is_hex(nonce, 16):
            raise ServerMisbehaved("the hello carries no nonce to sign")
        self._join_nonce = nonce
        statement = join_statement(
            self.room_id,
            self.server_key,
            self.server_commitment,
            self.wanted_seat,
            self.client_commitment,
            nonce,
        )
        return {
            "type": "join",
            "protocol_version": PROTOCOL_VERSION,
            "name": self.name,
            "seat": self.wanted_seat,
            "public_key": self.key.public_hex,
            "client_commitment": self.client_commitment,
            "pop": self.key.sign(statement),
        }

    async def _handle(self, ws, body: dict, decide: Optional[Decide]):
        kind = body.get("type")
        if kind == "joined":
            if (
                self.spectate
                or body.get("public_key") != self.key.public_hex
                or body.get("client_commitment") != self.client_commitment
                or body.get("room_id") != self.room_id
                or self.seat is not None
                or type(body.get("seat")) is not int
                or (self.wanted_seat is not None and body["seat"] != self.wanted_seat)
            ):
                raise ServerMisbehaved(
                    "the server seated another key, commitment or seat"
                )
            self._check_room(body.get("room"), "the join")
            self.seat = body["seat"]
            self._color = body.get("color")
            self.room = body["room"]
        elif kind == "room":
            self._check_room(body.get("room"), "a room message")
            self.room = body["room"]
        elif kind == "commitments":
            if self.commitments is not None:
                # One commitments per room, ever: a second, different one
                # would be a second chance to reveal (and so to choose).
                if canonical(body) != canonical(self.commitments):
                    raise ServerMisbehaved(
                        "a second, different commitments for this room"
                    )
                return None
            self._check_commitments(body)
            self.commitments = body
            if not self.spectate and self.seat is not None:
                statement = reveal_statement(digest(body))
                self._reveal = {
                    "type": "reveal",
                    "client_seed": self.client_seed,
                    "sig": self.key.sign(statement),
                }
                self._unreceipted[submission_digest(statement, self._reveal["sig"])] = (
                    self._reveal
                )
                await ws.send(json.dumps(self._reveal))
        elif kind == "received":
            submission = body.get("submission")
            self._unreceipted.pop(submission, None)
            if not body.get("accepted") and submission == self._answer_digest():
                logger.warning(
                    "the server did not take our move: %s", body.get("reason")
                )
        elif kind == "error":
            await self._on_error(body)
        elif kind == "aborted":
            seed = body.get("server_seed")
            if not is_hex(seed, 32) or seed_commitment(seed) != self.server_commitment:
                raise ServerMisbehaved(
                    "the void does not reveal the seed the server committed to"
                )
            self._close_decision()
            return {"aborted": body}
        elif kind in GAME_MESSAGES:
            if kind == "before":
                if self.table.state is not None:
                    raise ServerMisbehaved("a second before")
                self._check_before(body)
            elif self.table.state is None:
                raise ServerMisbehaved(f"a {kind} message before the game")
            if kind == "decide":
                index = body.get("index")
                if type(index) is not int or index <= self._last_decided:
                    # A second decide for a move would make a second answer
                    # (and let the server choose between them).
                    raise ServerMisbehaved(f"a second decide for move {index}")
                self._last_decided = index
            self.table.update(body)
            # Any newer game message supersedes an open decision.
            self._close_decision()
            if kind == "decide":
                self._pending = self.messages[-1]
                self._pending_since = asyncio.get_running_loop().time()
                self._start_decision(decide)
            elif kind == "after":
                return {"aborted": None}
        elif kind not in ("hello",):
            raise ServerMisbehaved(f"a message of unknown type {str(kind)[:40]!r}")
        return None

    def _close_decision(self) -> None:
        """A newer message closed the open decision: if we had not answered
        it, keep a record (the server timed us out, or so it says)."""
        pending = self._pending
        if pending is not None and self._answer is None:
            elapsed = asyncio.get_running_loop().time() - self._pending_since
            self.unanswered.append(
                {
                    "index": pending["body"].get("index"),
                    "seq": pending["seq"],
                    "ms": int(elapsed * 1000),
                }
            )
        self._pending = self._answer = None

    def _answer_digest(self) -> Optional[str]:
        if self._answer is None:
            return None
        return submission_digest(self._answer["statement"], self._answer["sig"])

    def _check_commitments(self, body: dict) -> None:
        seats = body.get("seats")
        if (
            sorted(body) != sorted(COMMITMENTS_FIELDS)
            or body.get("room_id") != self.room_id
            or body.get("server_key") != self.server_key
            or body.get("seed_commitment") != self.server_commitment
            or canonical(body.get("config")) != canonical(self.config)
            or type(body.get("reveal_timeout_ms")) is not int
            or not isinstance(seats, list)
            or not all(isinstance(s, dict) for s in seats)
        ):
            raise ServerMisbehaved(
                "the commitments change the room's server, rules or seed"
            )
        keys = [s.get("public_key") for s in seats if s.get("kind") == "remote"]
        if len(keys) != len(set(keys)):
            raise ServerMisbehaved("the commitments seat one key twice")
        for index, seat in enumerate(seats):
            # Every seat, not just ours: a reveal signs them all (C), so a
            # seat whose own join does not check would be revealed into.
            if not self._seat_ok(index, seat):
                raise ServerMisbehaved(f"the commitments' seat {index} is malformed")
        if not self.spectate and self.seat is not None:
            mine = [s for s in seats if s.get("public_key") == self.key.public_hex]
            if (
                len(mine) != 1
                or mine[0].get("kind") != "remote"
                or mine[0].get("seat") != self.seat
                or mine[0].get("color") != self._color
                or mine[0].get("requested_seat") != self.wanted_seat
                or mine[0].get("client_commitment") != self.client_commitment
                or mine[0].get("nonce") != self._join_nonce
                or not verify_signature(
                    self.key.public_hex,
                    join_statement(
                        self.room_id,
                        self.server_key,
                        self.server_commitment,
                        mine[0].get("requested_seat"),
                        self.client_commitment,
                        mine[0].get("nonce"),
                    ),
                    mine[0].get("pop"),
                )
            ):
                raise ServerMisbehaved(
                    "the commitments do not seat this client as it joined"
                )

    def _seat_ok(self, index: int, seat: dict) -> bool:
        colors = [c.value for c in GameConfig.from_json(self.config).colors]
        if (
            sorted(seat) != sorted(SEAT_FIELDS)
            or type(seat.get("seat")) is not int
            or seat["seat"] != index
            or index >= len(colors)
            or seat.get("color") != colors[index]
        ):
            return False
        if seat.get("kind") == "bot":
            return all(
                seat.get(k) is None
                for k in ("public_key", "client_commitment", "nonce", "pop")
            )
        requested = seat.get("requested_seat")
        return (
            seat.get("kind") == "remote"
            and is_hex(seat.get("public_key"), 32)
            and is_hex(seat.get("client_commitment"), 32)
            and is_hex(seat.get("nonce"), 16)
            and (requested is None or (type(requested) is int and requested == index))
            and verify_signature(
                seat["public_key"],
                join_statement(
                    self.room_id,
                    self.server_key,
                    self.server_commitment,
                    requested,
                    seat["client_commitment"],
                    seat["nonce"],
                ),
                seat.get("pop"),
            )
        )

    def _check_before(self, body: dict) -> None:
        if body.get("seed_commitment") != self.server_commitment:
            raise ServerMisbehaved("the game names another seed commitment")
        if self.commitments is not None and (
            canonical(body.get("commitments")) != canonical(self.commitments)
            or canonical(body.get("config")) != canonical(self.commitments["config"])
            or canonical(body.get("seats")) != canonical(self.commitments["seats"])
        ):
            raise ServerMisbehaved("the game is not the one whose seats were committed")
        if not self.spectate:
            seeds = body.get("client_seeds") or []
            if (
                self.commitments is None
                or self._reveal is None
                or self.seat is None
                or self.seat >= len(seeds)
                or seeds[self.seat] != self.client_seed
            ):
                raise ServerMisbehaved(
                    "the game was not seeded with this client's seed"
                )
            if body.get("color") != self._color:
                raise ServerMisbehaved("the game seats this client in another color")
        elif body.get("color") is not None:
            raise ServerMisbehaved("a spectator was sent a seat's view")

    def _start_decision(self, decide: Optional[Decide]) -> None:
        if decide is None:
            raise ServerMisbehaved("asked a spectator to decide")
        self._decision = asyncio.create_task(self._decide(self._pending, decide))

    async def _decide(self, pending: dict, decide: Decide) -> None:
        """Run the callback, and answer unless a newer message superseded it.

        Runs beside the reader, so the transcript keeps up while a bot thinks
        (and a decision the server timed out is simply dropped)."""
        view, playable = self.table.view(), pending["body"]["playable_actions"]
        try:
            if inspect.iscoroutinefunction(decide):
                result = await decide(view, playable)
            else:
                result = await asyncio.to_thread(decide, view, playable)
                if inspect.isawaitable(result):
                    result = await result
        except Exception:
            logger.exception("the decision callback failed; playing the first action")
            result = playable[0]
        if self._pending is pending and self._answer is None:
            await self._answer_with(result)

    async def _answer_with(self, action: Any) -> None:
        if isinstance(action, Action):
            action = action_to_json(action)
        pending = self._pending
        assert pending is not None
        if self.commitments is None or self._reveal is None:
            # (Never sign outside the game this client revealed for; the
            # reader refuses such a decide before this can run.)
            logger.error("asked to move in a game this client never revealed for")
            return
        body = pending["body"]
        statement = action_statement(
            digest(self.commitments),
            body["index"],
            pending["seq"],
            envelope_hash(pending),
            action,
        )
        self._answer = {
            "type": "action",
            "statement": statement,
            "sig": self.key.sign(statement),
        }
        # Kept until the server's receipt names it; resent after a reconnect.
        self._unreceipted[self._answer_digest()] = self._answer
        # When it was sent (after which message): lets a verifier tell an
        # answer sent in time from one sent after the move was played.
        self.sent.append(
            {
                "statement": statement,
                "sig": self._answer["sig"],
                "sent_after": len(self.messages) - 1,
            }
        )
        try:
            await self.connection.send(json.dumps(self._answer))
        except ConnectionClosed:
            pass  # resent on reconnect

    async def _on_error(self, body: dict) -> None:
        logger.warning("server: %s", body.get("message"))
        joining = self._join_sent and self.seat is None and not self.spectate
        if (
            body.get("answers") == "join"
            and joining
            or body.get("answers") == "spectate"
            and self.spectate
        ) and body.get("code") in (
            "seat_taken",
            "room_full",
            "bad_join",
            "duplicate_key",
            "game_started",
            "banned",
        ):
            raise Refused(body.get("message"))
        # A refused answer is never followed by a different signed move: a
        # client that signed two moves for one decide would let the server
        # choose between them. The server's receipt (in this chain) is what
        # the verifier holds a refusal against.

    def _finish(self, aborted: Optional[dict] = None) -> GameResult:
        path = self.save_transcript()
        verified, report = None, None
        if self.verify:
            from catanatron.server.verify import verify_transcript

            report = verify_transcript(json.loads(json.dumps(self.transcript())))
            verified = report.ok
            if report.errors or report.invalid:
                logger.error(
                    "transcript verification failed: %s",
                    report.errors or report.invalid,
                )
        elif aborted is not None:
            verified = False
        return GameResult(
            room_id=self.room_id,
            game_id=self.table.game_id,
            color=self.table.color,
            winning_color=self.table.winning_color,
            transcript_path=path,
            verified=verified,
            report=report,
            aborted=aborted,
        )
