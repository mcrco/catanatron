"""A remote game: one seat, or a spectator's place, in a catanatron-server room.

The session is a thin layer over :class:`catanatron.server.client.GameClient`,
which does the fairness work: it checks every message's signature and hash
chain, contributes a seed, signs every move and keeps the transcript. The
session shows what the client has received (its
:class:`~catanatron.server.table.Table`), turns each ``decide`` into a choice
for the person at the keyboard, saves the transcript when the game ends and
verifies it (:func:`catanatron.server.verify.verify_transcript`).

There are no debugging controls here: the server holds the game.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit, urlunsplit

from catanatron.models.enums import Action
from catanatron.serialization import action_from_json, action_to_json, map_from_json
from catanatron.tui.board import action_target
from catanatron.tui.session import CommandError, Frame, Session

HELP = """\
Remote commands (type them after ':'):
  verify                    verify the transcript (after the game; key V)
  transcript PATH           save a copy of the transcript so far"""


@dataclass
class Pending:
    """A decision the server is waiting for from us."""

    playable: list
    future: asyncio.Future
    timeout_ms: int
    since: float


def room_url(base: str, room_id: str) -> str:
    parts = urlsplit(base)
    return urlunsplit((parts.scheme, parts.netloc, f"/room/{room_id}", "", ""))


def is_room_url(url: str) -> bool:
    return "/room/" in urlsplit(url).path


def describe_rooms(rooms: Sequence[dict]) -> str:
    if not rooms:
        return "no rooms on this server; create one with --new"
    lines = []
    for room in rooms:
        seats = ", ".join(
            f"{s['color']}:{s['kind'] if s['kind'] != 'bot' else s['spec']}"
            for s in room["seats"]
        )
        lines.append(f"{room['id']}  {room['status']:<8}  {room['name']}  [{seats}]")
    return "\n".join(lines)


class RemoteSession(Session):
    kind = "remote"
    debug = False

    def __init__(
        self,
        url: str,
        *,
        seat: Optional[int] = None,
        spectate: bool = False,
        name: str = "catanatron-tui",
        create: Optional[dict] = None,
        create_seats: Optional[list] = None,
        transcript_dir: Optional[str] = "transcripts",
        verify: bool = True,
        server_key: Optional[str] = None,
        key_path: Optional[str] = None,
    ):
        if create is None and not is_room_url(url):
            raise ValueError(
                "connect to ws://HOST:PORT/room/ROOM_ID, or create a room with --new"
            )
        self.url = url
        self.seat = seat
        self.spectate = spectate
        self.name = name
        self.create = create
        self.create_seats = create_seats
        self.transcript_dir = transcript_dir
        self.verify_at_end = verify
        self.server_key = server_key
        self.key_path = key_path

        self.client: Any = None
        self.room: Optional[dict] = None
        self.catan_map = None
        self.pending: Optional[Pending] = None
        self.result: Any = None
        self.report: Any = None
        self.transcript_path: Optional[str] = None
        self.error: Optional[str] = None
        self.done = asyncio.Event() if _has_loop() else None
        #: What happened outside the game, as (message, is_error), for the UI.
        self.messages: List[Tuple[str, bool]] = []
        self.title = f"remote game at {url}"

    # ===== running =====
    async def run(self) -> None:
        """Create the room if asked, then play (or watch) until the game ends."""
        from catanatron.server.client import GameClient, create_room

        self.done = self.done or asyncio.Event()
        try:
            if self.create is not None:
                room = await create_room(
                    self.url, self.create, self.create_seats, name=self.name
                )
                self.url = room_url(self.url, room["id"])
                self.room = room
                self.title = f"remote game at {self.url}"
                self.note(f"created room {room['id']}: {self.url}")
            self.client = GameClient(
                self.url,
                name=self.name,
                seat=self.seat,
                spectate=self.spectate,
                transcript_dir=self.transcript_dir,
                verify=False,
                server_key=self.server_key,
                key_path=self.key_path,
            )
            self.client.on_message = self._on_message
            self.changed()
            self.result = await self.client.play(
                None if self.spectate else self._decide
            )
            self.transcript_path = self.result.transcript_path
            if self.transcript_path:
                self.note(f"transcript saved to {self.transcript_path}")
            if self.verify_at_end:
                await self.verify()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
            self.note(self.error, error=True)
        finally:
            self.done.set()
            self.changed()

    def note(self, message: str, error: bool = False) -> None:
        self.messages.append((message, error))
        self.changed()

    async def verify(self) -> Any:
        """Verify the transcript received so far (normally after the game)."""
        from catanatron.server.verify import verify_transcript

        if self.client is None:
            raise CommandError("not connected")
        transcript = json.loads(json.dumps(self.client.transcript()))
        self.report = await asyncio.to_thread(verify_transcript, transcript)
        if self.report.ok:
            self.note(f"transcript verified: {_counts(self.report)}")
        else:
            self.note(
                "transcript verification FAILED: " + "; ".join(self.report.errors[:3]),
                error=True,
            )
        return self.report

    async def _decide(self, view: dict, playable: list) -> list:
        """GameClient's callback: wait for the person to choose."""
        future = asyncio.get_running_loop().create_future()
        body = next(
            (
                m["body"]
                for m in reversed(self.client.messages)
                if m["body"].get("type") == "decide"
            ),
            {},
        )
        self.pending = Pending(
            playable=playable,
            future=future,
            timeout_ms=body.get("timeout_ms", 0),
            since=time.monotonic(),
        )
        self.changed()
        try:
            return await future
        finally:
            if self.pending is not None and self.pending.future is future:
                self.pending = None

    def _on_message(self, body: dict) -> None:
        kind = body.get("type")
        if kind in ("hello", "room"):
            self.room = body["room"]
        elif kind == "before":
            self.catan_map = map_from_json(body["map"])
        elif kind == "joined":
            self.note(f"seated as {body['color']} (seat {body['seat']})")
        elif kind == "error":
            self.note(f"server: {body.get('message')}", error=True)
        if kind in ("step", "after") and self.pending is not None:
            # The server moved on without our answer (a timeout); the client
            # drops whatever we would have said.
            if not self.pending.future.done():
                self.pending.future.set_result(self.pending.playable[0])
            self.pending = None
        if kind == "step" and body.get("forced") == "timeout":
            entry = body["log"][0][0] if body.get("log") else None
            if entry is not None:
                self.note(f"{entry[0]} timed out; the server played {entry[1].lower()}")
        # The client takes the message in after this hook returns.
        asyncio.get_running_loop().call_soon(self.changed)

    # ===== the person's moves =====
    def choose(self, action: Action) -> None:
        pending = self.pending
        if pending is None or pending.future.done():
            raise CommandError("it is not your move")
        choice = action_to_json(action)
        if choice not in pending.playable:
            raise CommandError(f"{choice} is not one of the offered actions")
        pending.future.set_result(choice)
        self.pending = None
        self.changed()

    # ===== what is shown =====
    @property
    def table(self):
        return None if self.client is None else self.client.table

    def frame(self) -> Frame:
        table = self.table
        if table is None or table.state is None:
            return self._waiting_frame()
        view = table.view()
        seats = {}
        for seat in table.seats:
            if seat["color"] == table.color:
                label = "you"
            elif seat["kind"] == "bot":
                label = str(seat.get("spec"))
            else:
                label = seat.get("name") or seat["kind"]
            seats[seat["color"]] = label
        actions, actor = [], None
        if self.pending is not None and not self.pending.future.done():
            actions = [action_from_json(a) for a in self.pending.playable]
            actor = table.color
        over = table.over
        winner = table.winning_color
        current = None if over else table.current_color
        if winner is not None:
            status = f"{winner} wins"
        elif over:
            status = "the game is over"
        elif actor is not None:
            status = "your move"
        else:
            status = f"{current} to move"
        records = view["action_records"]
        recent = ()
        if records:
            target = action_target(action_from_json(records[-1][0]))
            recent = (target,) if target is not None else ()
        return Frame(
            document=view,
            catan_map=self.catan_map,
            colors=list(view["colors"]),
            seats=seats,
            viewer=table.color,
            god=False,
            current=current,
            winner=winner,
            over=over,
            actions=actions,
            actor=actor,
            status=status,
            flags=self.flags(),
            notes={},
            recent=recent,
        )

    def flags(self) -> List[str]:
        flags = []
        if self.client is not None:
            flags.append(f"room {self.client.room_id}")
        if self.pending is not None and self.pending.timeout_ms:
            left = self.pending.timeout_ms / 1000 - (
                time.monotonic() - self.pending.since
            )
            flags.append(f"{max(0, int(left))}s to decide")
        if self.report is not None:
            flags.append("verified" if self.report.ok else "VERIFICATION FAILED")
        if self.error:
            flags.append(self.error)
        return flags

    def _waiting_frame(self) -> Frame:
        room = self.room or {}
        seats = {}
        colors = []
        for seat in room.get("seats", []):
            colors.append(seat["color"])
            if seat["kind"] == "open":
                seats[seat["color"]] = "open"
            elif seat["kind"] == "bot":
                seats[seat["color"]] = f"{seat.get('spec')} (server bot)"
            else:
                seats[seat["color"]] = seat.get("name") or "remote"
        if self.error:
            status = self.error
        elif not room:
            status = f"connecting to {self.url}"
        else:
            open_seats = sum(1 for s in room.get("seats", []) if s["kind"] == "open")
            status = (
                f"room {room.get('id')}: waiting for {open_seats} more player"
                f"{'s' if open_seats != 1 else ''} (share {self.url})"
                if open_seats
                else f"room {room.get('id')}: starting"
            )
        return Frame(
            document=None,
            catan_map=None,
            colors=colors,
            seats=seats,
            viewer=None,
            god=False,
            current=None,
            winner=None,
            over=False,
            status=status,
            flags=self.flags(),
        )

    # ===== commands =====
    def help(self) -> str:
        return HELP

    def execute(self, line: str) -> Optional[str]:
        words = line.split()
        if not words:
            return None
        command, args = words[0].lower(), words[1:]
        if command == "verify":
            if self.client is None:
                raise CommandError("not connected")
            asyncio.get_running_loop().create_task(self.verify())
            return "verifying the transcript..."
        if command == "transcript":
            if len(args) != 1 or self.client is None:
                raise CommandError("usage: transcript PATH")
            with open(args[0], "w") as file:
                json.dump(self.client.transcript(), file)
            return f"transcript saved to {args[0]}"
        raise CommandError(f"unknown command {command!r}; try help")


def _counts(report: Any) -> str:
    counts = getattr(report, "counts", {}) or {}
    return ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) or "ok"


def _has_loop() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False
