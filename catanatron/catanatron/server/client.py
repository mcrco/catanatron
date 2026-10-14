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
from catanatron.server.messages import (
    COMMITMENTS_FIELDS,
    GAME_MESSAGES,
    PROTOCOL_VERSION,
)
from catanatron.server.table import Table

logger = logging.getLogger(__name__)

TRANSCRIPT_FORMAT = "catanatron-transcript/4"

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
    """The server holds no seat for our key (a join that never landed, or a
    seat freed while we were away): connect afresh and join again."""


class GameClient:
    """Plays one seat (or watches, with ``spectate=True``) of one room.

    The client keeps the protocol's invariants on its own (see PROTOCOL.md):
    it signs exactly one reveal per room, for the first ``commitments`` it
    receives (a different second one is the server misbehaving), with the
    seed it joined with and no other; it signs at most one answer per
    decision and resends, after a reconnect, exactly what it signed.
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
        self.server_commitment: Optional[str] = None
        self.config: Optional[dict] = None
        #: The room's commitments: the first ever received, the only ones.
        self.commitments: Optional[dict] = None
        self.messages: list = []
        self.sent: list = []
        self.head = GENESIS
        #: The decide message awaiting our answer, and the answer once made.
        self._pending: Optional[dict] = None
        self._answer: Optional[dict] = None
        self._decision: Optional[asyncio.Task] = None
        #: Our join went out (we may hold a seat even if 'joined' was lost).
        self._join_sent = False
        #: Our one reveal, once signed; resent as is after a reconnect.
        self._reveal: Optional[dict] = None
        self._unreceipted: dict = {}
        self._ending: Optional[dict] = None
        self._ending_deadline = 0.0
        self._attempts = 0
        self._start_overs = 0
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
        }

    def save_transcript(self) -> Optional[str]:
        if not self.transcript_dir:
            return None
        os.makedirs(self.transcript_dir, exist_ok=True)
        who = self.table.color or ("spectator" if self.spectate else "seat")
        game = self.table.game_id or "nogame"
        path = os.path.join(self.transcript_dir, f"{self.room_id}-{game}-{who}.json")
        with open(path, "w") as file:
            json.dump(self.transcript(), file)
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
        """Join, play until the game ends, save and verify the transcript."""
        if decide is None and not self.spectate:
            raise ValueError("a seated client needs a decide callback")
        self._attempts = 0
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
                await ws.send(json.dumps(self._join_request()))
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
                result = await self._handle(ws, body, decide)
                self._notify(body)
                if result is not None and self._ending is None:
                    self._ending = result
                    self._ending_deadline = loop.time() + self.RECEIPT_WAIT_S

    def _resumes_chain(self, envelope: dict) -> bool:
        """While rejoining: skip (after checking) what arrives on the
        throw-away chain until the seat's chain resumes."""
        if (
            envelope.get("seq") == len(self.messages)
            and envelope.get("prev") == self.head
        ):
            return True
        if not verify_envelope(self.server_key, envelope):
            raise ServerMisbehaved("bad signature while rejoining")
        body = envelope.get("body", {})
        if body.get("type") == "error":
            if body.get("code") == "not_seated":
                raise _StartOver()
            raise Refused(body.get("message"))
        return False

    async def _resume(self, ws, decide: Optional[Decide]) -> None:
        """After a rejoin: finish what the drop interrupted, by resending
        exactly what was signed (the reveal, answers not yet acknowledged).
        Resending is harmless: the server answers a stale submission with a
        receipt that says so, which no verifier holds against it."""
        if self._reveal is not None and self.table.state is None:
            await ws.send(json.dumps(self._reveal))
        for answer in list(self._unreceipted.values()):
            await ws.send(json.dumps(answer))
        if (
            self._pending is not None
            and self._answer is None
            and (self._decision is None or self._decision.done())
        ):
            self._start_decision(decide)

    def _notify(self, body: dict) -> None:
        if self.on_message is not None:
            self.on_message(body)

    def _on_hello(self, body: dict) -> None:
        if body.get("protocol_version") != PROTOCOL_VERSION:
            raise Refused(f"server speaks protocol {body.get('protocol_version')}")
        room = body["room"]
        if room.get("id") != self.room_id:
            raise ServerMisbehaved("the hello is for another room")
        if self.server_commitment is not None and (
            room.get("seed_commitment") != self.server_commitment
            or room.get("config") != self.config
        ):
            # Starting over after a lost join: the room is the same room.
            raise ServerMisbehaved("the room changed its seed commitment or rules")
        self.room = room
        self.server_commitment = room["seed_commitment"]
        self.config = room["config"]

    def _join_request(self) -> dict:
        if self.spectate:
            return {"type": "spectate", "name": self.name}
        statement = join_statement(
            self.room_id,
            self.server_commitment,
            self.wanted_seat,
            self.client_commitment,
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
                body.get("public_key") != self.key.public_hex
                or body.get("client_commitment") != self.client_commitment
                or body.get("room_id") != self.room_id
                or self.seat is not None
            ):
                raise ServerMisbehaved("the server seated another key or commitment")
            self.seat = body["seat"]
            self.room = body["room"]
        elif kind == "room":
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
                self._reveal = {
                    "type": "reveal",
                    "client_seed": self.client_seed,
                    "sig": self.key.sign(reveal_statement(digest(body))),
                }
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
            if seed is not None and (
                not is_hex(seed, 32) or seed_commitment(seed) != self.server_commitment
            ):
                raise ServerMisbehaved("the void does not open the server's commitment")
            return {"aborted": body}
        elif kind in GAME_MESSAGES:
            if kind == "before":
                self._check_before(body)
            self.table.update(body)
            # Any newer game message supersedes an open decision.
            self._pending = self._answer = None
            if kind == "decide":
                self._pending = self.messages[-1]
                self._start_decision(decide)
            elif kind == "after":
                return {"aborted": None}
        return None

    def _answer_digest(self) -> Optional[str]:
        if self._answer is None:
            return None
        return submission_digest(self._answer["statement"], self._answer["sig"])

    def _check_commitments(self, body: dict) -> None:
        if (
            sorted(body) != sorted(COMMITMENTS_FIELDS)
            or body.get("room_id") != self.room_id
            or body.get("seed_commitment") != self.server_commitment
            or body.get("config") != self.config
            or type(body.get("reveal_timeout_ms")) is not int
            or not isinstance(body.get("seats"), list)
        ):
            raise ServerMisbehaved("the commitments change the room's rules or seed")
        if not self.spectate and self.seat is not None:
            mine = [
                s
                for s in body["seats"]
                if isinstance(s, dict) and s.get("public_key") == self.key.public_hex
            ]
            if (
                len(mine) != 1
                or mine[0].get("seat") != self.seat
                or mine[0].get("client_commitment") != self.client_commitment
                or not verify_signature(
                    self.key.public_hex,
                    join_statement(
                        self.room_id,
                        self.server_commitment,
                        mine[0].get("requested_seat"),
                        self.client_commitment,
                    ),
                    mine[0].get("pop"),
                )
            ):
                raise ServerMisbehaved(
                    "the commitments do not seat this client as it joined"
                )

    def _check_before(self, body: dict) -> None:
        if self.commitments is not None and (
            body.get("commitments") != self.commitments
            or body.get("config") != self.commitments["config"]
            or body.get("seats") != self.commitments["seats"]
        ):
            raise ServerMisbehaved("the game is not the one whose seats were committed")
        if not self.spectate:
            seeds = body.get("client_seeds") or []
            if (
                self.commitments is None
                or self.seat is None
                or self.seat >= len(seeds)
                or seeds[self.seat] != self.client_seed
            ):
                raise ServerMisbehaved(
                    "the game was not seeded with this client's seed"
                )

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
        if body.get("answers") in ("join", "spectate") and body.get("code") in (
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
        if aborted is not None:
            verified = False
        elif self.verify:
            from catanatron.server.verify import verify_transcript

            report = verify_transcript(self.transcript())
            verified = report.ok
            if not report.ok:
                logger.error("transcript verification failed: %s", report.errors)
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
