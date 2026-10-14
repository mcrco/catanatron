"""An async client for catanatron-server: takes a seat, plays it, keeps proof.

::

    from catanatron.server.client import GameClient

    async def main():
        client = GameClient("ws://localhost:8765/room/ROOM_ID", name="me")
        result = await client.play(lambda view, actions: actions[0])

The client does the fairness work on its own: it checks every message's
signature and its place in the hash chain as it arrives, contributes a fresh
seed when it joins, signs every action, saves its transcript when the game
ends and (by default) verifies it with :func:`catanatron.server.verify.verify_transcript`.

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
    envelope_hash,
    random_hex,
    verify_envelope,
)
from catanatron.server.messages import PROTOCOL_VERSION
from catanatron.server.table import Table

logger = logging.getLogger(__name__)

TRANSCRIPT_FORMAT = "catanatron-transcript/1"

Decide = Callable[[dict, list], Any]


class ServerMisbehaved(RuntimeError):
    """A message failed its signature or hash-chain check."""


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

    def summary(self) -> dict:
        return {
            "room_id": self.room_id,
            "game_id": self.game_id,
            "color": self.color,
            "winning_color": self.winning_color,
            "transcript": self.transcript_path,
            "verified": self.verified,
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


class GameClient:
    """Plays one seat (or watches, with ``spectate=True``) of one room."""

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

        self.client_seed = random_hex()
        self.table = Table()
        self.room: Optional[dict] = None
        self.seat: Optional[int] = None
        self.seat_key: Optional[str] = None
        self.messages: list = []
        self.sent: list = []
        self.head = GENESIS
        #: The decide message awaiting our answer, and the answer once made.
        self._pending: Optional[dict] = None
        self._answer: Optional[dict] = None
        self._decision: Optional[asyncio.Task] = None
        #: Called with every body, for UIs and logging.
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
            "seat": self.seat,
            "color": self.table.color,
            "messages": self.messages,
            "sent": self.sent,
        }

    def save_transcript(self) -> Optional[str]:
        if not self.transcript_dir:
            return None
        os.makedirs(self.transcript_dir, exist_ok=True)
        who = self.table.color or "spectator"
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
            if body.get("type") != "hello" or "server_key" not in body:
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
        expected_to = (
            None if (self.spectate or not self.messages) else self.key.public_hex
        )
        if envelope["to"] not in (None, expected_to):
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
        attempts = 0
        while True:
            try:
                return await self._session(decide)
            except (ConnectionClosed, OSError, InvalidStatus) as error:
                if isinstance(error, InvalidStatus) or (
                    self.seat_key is None and not self.spectate
                ):
                    raise
                attempts += 1
                if attempts > self.reconnect_attempts:
                    raise
                logger.info("connection lost (%s); reconnecting (%d)", error, attempts)
                await asyncio.sleep(min(0.2 * 2**attempts, 5))

    async def _session(self, decide: Optional[Decide]) -> GameResult:
        async with websockets.connect(self.url, max_size=None) as ws:
            self.connection = ws
            first = json.loads(await ws.recv())
            reconnecting = self.seat_key is not None or (
                self.spectate and self.messages
            )
            if reconnecting:
                if not verify_envelope(self.server_key, first):
                    raise ServerMisbehaved("bad signature on hello")
                if self.spectate:
                    # A spectator's chain is per connection: start over.
                    self.messages, self.head, self.table = [], GENESIS, Table()
                    self._accept(first)
                    await ws.send(json.dumps({"type": "spectate", "name": self.name}))
                else:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "join",
                                "protocol_version": PROTOCOL_VERSION,
                                "seat": self.seat,
                                "seat_key": self.seat_key,
                                "public_key": self.key.public_hex,
                                "last_seq": len(self.messages) - 1,
                            }
                        )
                    )
            else:
                body = self._accept(first)
                self.room = body["room"]
                if body.get("protocol_version") != PROTOCOL_VERSION:
                    raise Refused(
                        f"server speaks protocol {body.get('protocol_version')}"
                    )
                if self.spectate:
                    await ws.send(json.dumps({"type": "spectate", "name": self.name}))
                else:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "join",
                                "protocol_version": PROTOCOL_VERSION,
                                "name": self.name,
                                "seat": self.wanted_seat,
                                "public_key": self.key.public_hex,
                                "client_seed": self.client_seed,
                            }
                        )
                    )
            if reconnecting and self._pending is not None:
                if self._answer is not None:
                    await ws.send(json.dumps(self._answer))
                elif self._decision is None or self._decision.done():
                    self._start_decision(decide)

            async for text in ws:
                body = self._accept(json.loads(text))
                if self.on_message is not None:
                    self.on_message(body)
                kind = body.get("type")
                if kind == "joined":
                    self.seat, self.seat_key = body["seat"], body["seat_key"]
                elif kind == "room":
                    self.room = body["room"]
                elif kind == "error":
                    await self._on_error(body)
                elif kind in ("before", "step", "decide", "after"):
                    self.table.update(body)
                    # Any newer game message supersedes an open decision.
                    self._pending = self._answer = None
                    if kind == "decide":
                        self._pending = self.messages[-1]
                        self._start_decision(decide)
                    elif kind == "after":
                        return self._finish()
            raise ConnectionClosed(None, None)

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
        if self._pending is pending:
            await self._answer_with(result)

    async def _answer_with(self, action: Any) -> None:
        if isinstance(action, Action):
            action = action_to_json(action)
        pending = self._pending
        assert pending is not None
        statement = action_statement(
            pending["body"]["game_id"], pending["seq"], envelope_hash(pending), action
        )
        self._answer = {
            "type": "action",
            "statement": statement,
            "sig": self.key.sign(statement),
        }
        self.sent.append({"statement": statement, "sig": self._answer["sig"]})
        try:
            await self.connection.send(json.dumps(self._answer))
        except ConnectionClosed:
            pass  # resent on reconnect

    async def _on_error(self, body: dict) -> None:
        logger.warning("server: %s", body.get("message"))
        if body.get("code") in (
            "seat_taken",
            "room_full",
            "bad_join",
            "game_started",
            "aborted",
        ):
            raise Refused(body.get("message"))
        pending = self._pending
        if (
            body.get("code") == "bad_action"
            and pending is not None
            and body.get("seq") == pending["seq"]
            and self._answer is not None
            and self._answer["statement"]["action"]
            != pending["body"]["playable_actions"][0]
        ):
            # Our answer was refused and the decision is still open: play the
            # first legal action rather than stall until the timeout.
            await self._answer_with(pending["body"]["playable_actions"][0])

    def _finish(self) -> GameResult:
        path = self.save_transcript()
        verified, report = None, None
        if self.verify:
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
        )
