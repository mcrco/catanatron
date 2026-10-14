"""What the terminal UI talks to: a session, local or remote.

The app draws :class:`Frame`\\ s and hands the person's choices back to the
session; it never touches a game. :class:`~catanatron.tui.local.LocalSession`
runs a stock Catanatron game in this process (bots, hot seat and debugging
controls); :class:`~catanatron.tui.remote.RemoteSession` plays one seat (or
watches) of a catanatron-server room.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from catanatron.models.enums import Action
from catanatron.models.map import CatanMap


class CommandError(ValueError):
    """A command that cannot run, with a message for the person."""


@dataclass
class Frame:
    """Everything the UI shows, as the current viewer may see it."""

    #: The viewer's state document (``client_view``, ``public_view`` or, in
    #: god view, the full document), with ``map`` and ``action_records``.
    #: None before a remote game starts.
    document: Optional[dict]
    catan_map: Optional[CatanMap]
    #: Seat colors in turn order, and who plays each.
    colors: List[str]
    seats: Dict[str, str]
    #: Whose hand the document shows (None: nobody's, or everybody's if god).
    viewer: Optional[str]
    god: bool
    current: Optional[str]
    winner: Optional[str]
    over: bool
    #: The actions the person at the keyboard may play now, and for whom.
    actions: List[Action] = field(default_factory=list)
    actor: Optional[str] = None
    #: One line about what is happening ("BLUE (F) is thinking").
    status: str = ""
    #: Debug state shown under the status line (forced chance, autoplay...).
    flags: List[str] = field(default_factory=list)
    #: Notes on log entries, by entry index.
    notes: Dict[int, List[str]] = field(default_factory=dict)
    #: Board targets of the latest action, for highlighting.
    recent: Tuple = ()


class Session:
    """The interface the app uses. Methods run on the app's event loop."""

    kind = "?"
    #: Whether debugging controls (undo, forced chance, god view) exist.
    debug = False
    title = "catanatron"
    #: Called by the session whenever what it would show has changed.
    on_change: Optional[Callable[[], None]] = None

    def frame(self) -> Frame:
        raise NotImplementedError

    def choose(self, action: Action) -> None:
        """The person plays ``action`` (one of ``frame().actions``)."""
        raise NotImplementedError

    def execute(self, line: str) -> Optional[str]:
        """Run a typed command; returns a message to show."""
        raise CommandError(f"unknown command {line.split()[0]!r}; try help")

    def help(self) -> str:
        return ""

    def changed(self) -> None:
        if self.on_change is not None:
            self.on_change()
