"""The Textual front end of catanatron-tui.

The app draws :class:`~catanatron.tui.session.Frame`\\ s from a session and
hands the person's choices back. Local sessions are also driven from here:
bots think in worker threads, one move at a time, at the chosen speed.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical, VerticalScroll
from textual.geometry import Region
from textual.message import Message
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Footer, Header, Input, OptionList, RichLog, Static
from textual.widgets.option_list import Option

from catanatron.models.enums import Action
from catanatron.tui.actions import describe, grouped, node_label, tile_label
from catanatron.tui.board import (
    SIZES,
    BoardPicture,
    BoardState,
    Target,
    draw_board,
    legal_targets,
)
from catanatron.tui.panels import log_lines, players_panel, status_panel
from catanatron.tui.session import CommandError, Frame, Session

SPEEDS = (0.0, 0.05, 0.15, 0.3, 0.6, 1.0, 2.0)
#: The least time between two redraws, in seconds.
REDRAW_INTERVAL = 0.05
DIRECTIONS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}

KEYS_HELP = """\
Keys
  up/down, enter   choose a move from the list (grouped by kind)
  /                filter the list (escape clears)
  b                pick on the board: arrows move between targets, enter
                   picks, escape goes back. Clicking a node, road or tile
                   picks it too; elsewhere it says what is there
  :                type a command (below)
  z                board size: auto, small, medium, large
  i                node ids on/off
  ?                this help
  q, ctrl+q        quit
Local games
  space            pause / resume the bots
  n                one bot move
  + / -            bots faster / slower
  u / r            undo / redo one action
  U                undo to your last choice
  g                god view (every hand)
  v                cycle whose view is shown
  d, c, t          set the next dice, development card, steal
  s, l             save, load (JSON: the position and its history)
Remote games
  V                verify the transcript

App commands: step [N], auto on|off, speed SECONDS, zoom SIZE,
ids on|off, filter TEXT (or /TEXT), help, quit."""


class BoardView(Widget, can_focus=True):
    """The board picture. Clicks and keys become messages for the app."""

    BINDINGS = [
        Binding("up", "move('up')", "Up", show=False),
        Binding("down", "move('down')", "Down", show=False),
        Binding("left", "move('left')", "Left", show=False),
        Binding("right", "move('right')", "Right", show=False),
        Binding("enter,space", "pick", "Pick", show=False),
        Binding("escape", "leave", "Back to the list", show=False),
    ]

    class Picked(Message):
        def __init__(self, target: Target) -> None:
            super().__init__()
            self.target = target

    class Moved(Message):
        def __init__(self, direction: str) -> None:
            super().__init__()
            self.direction = direction

    class Left(Message):
        pass

    class Hovered(Message):
        def __init__(self, target: Optional[Target]) -> None:
            super().__init__()
            self.target = target

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.picture: Optional[BoardPicture] = None
        self.cursor: Optional[Target] = None
        self._hovered: Optional[Target] = None

    def show(self, picture: BoardPicture) -> None:
        self.picture = picture
        self.styles.width = max(picture.width, 1)
        self.styles.height = max(picture.height, 1)
        self.refresh(layout=True)

    def render(self) -> Text:
        return self.picture.text if self.picture is not None else Text("")

    def on_click(self, event: events.Click) -> None:
        target = self._target(event)
        if target is not None:
            self.post_message(self.Picked(target))

    def on_mouse_move(self, event: events.MouseMove) -> None:
        target = self._target(event)
        if target != self._hovered:
            self._hovered = target
            self.post_message(self.Hovered(target))

    def _target(self, event) -> Optional[Target]:
        if self.picture is None:
            return None
        offset = event.get_content_offset(self)
        if offset is None:
            return None
        return self.picture.target_at(offset.x, offset.y)

    def action_move(self, direction: str) -> None:
        self.post_message(self.Moved(direction))

    def action_pick(self) -> None:
        if self.cursor is not None:
            self.post_message(self.Picked(self.cursor))

    def action_leave(self) -> None:
        self.post_message(self.Left())

    def on_focus(self) -> None:
        self.post_message(self.Moved(""))

    def on_blur(self) -> None:
        self.post_message(self.Left())


class HelpScreen(ModalScreen):
    BINDINGS = [
        Binding("escape,q,question_mark", "dismiss", "Close"),
    ]
    DEFAULT_CSS = """
    HelpScreen { align: center middle; }
    #help { width: 90; max-width: 100%; height: auto; max-height: 90%;
            border: round $accent; padding: 0 1; background: $panel; }
    """

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="help") as scroll:
            scroll.border_title = "catanatron-tui (escape closes)"
            yield Static(Text(self.text))


class CatanApp(App):
    """Play, watch and debug Catanatron games in the terminal."""

    TITLE = "catanatron-tui"
    CSS = """
    #main { height: 1fr; }
    #board-scroll { width: 1fr; height: 1fr; border: round $panel; align: center top; }
    #board-scroll:focus-within { border: round $accent; }
    BoardView { padding: 0; }
    #side { width: 64; max-width: 55%; min-width: 44; }
    #status { height: auto; padding: 0 1; }
    #message { height: auto; padding: 0 1; color: $text-muted; }
    #players { height: auto; padding: 0 1; margin: 1 0; }
    #actions { height: 1fr; min-height: 6; border: round $panel; }
    #actions:focus { border: round $accent; }
    #log { height: 1fr; min-height: 5; border: round $panel; }
    #command { display: none; }
    #command.open { display: block; }
    """

    BINDINGS = [
        Binding("question_mark", "help", "Help"),
        Binding("colon", "command", "Command"),
        Binding("b", "board", "Board"),
        Binding("space", "autoplay", "Pause/Play"),
        Binding("n", "step", "Step"),
        Binding("u", "undo", "Undo"),
        Binding("r", "redo", "Redo"),
        Binding("U", "back", "Undo turn", show=False),
        Binding("g", "god", "God view"),
        Binding("v", "cycle_view", "View", show=False),
        Binding("d", "prefill('dice ')", "Dice"),
        Binding("c", "prefill('card ')", "Card", show=False),
        Binding("t", "prefill('steal ')", "Steal", show=False),
        Binding("plus,equals_sign", "speed(-1)", "Faster", show=False),
        Binding("minus", "speed(1)", "Slower", show=False),
        Binding("s", "prefill('save ')", "Save"),
        Binding("l", "prefill('load ')", "Load", show=False),
        Binding("V", "verify", "Verify"),
        Binding("z", "zoom", "Zoom"),
        Binding("i", "ids", "Ids", show=False),
        Binding("slash", "prefill('/')", "Filter", show=False),
        Binding("q", "quit", "Quit"),
    ]
    LOCAL_ONLY = {
        "autoplay",
        "step",
        "undo",
        "redo",
        "back",
        "god",
        "cycle_view",
        "speed",
    }
    LOCAL_PREFILLS = {"dice ", "card ", "steal ", "save ", "load "}

    def __init__(
        self,
        session: Session,
        *,
        board_size: str = "auto",
        speed: float = 0.3,
        autoplay: bool = True,
        commands: Tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self.session = session
        self.zoom = board_size
        self.delay = speed
        self.autoplay = autoplay
        self.node_ids = True
        self.startup_commands = tuple(commands)
        self._frame: Optional[Frame] = None
        self.thinking: Optional[Tuple[object, float]] = None
        self.steps_left = 0
        self.cursor: Optional[Target] = None
        self.hovered: Optional[Target] = None
        self.filter_text = ""
        self.filter_target: Optional[Target] = None
        self.messages: List[str] = []
        self._timer = None
        self._shown_actions: Dict[str, Action] = {}
        self._actions_key: Optional[tuple] = None
        self._log_key: Optional[tuple] = None
        self._log_count = 0
        self._log_last = None
        self._sizes: Dict[str, Tuple[int, int]] = {}
        self._sizes_map = None
        self._picture: Optional[BoardPicture] = None
        self._picture_key: Optional[tuple] = None
        self._redraw_timer = None
        self._last_redraw = 0.0

    # ===== layout =====
    def compose(self) -> ComposeResult:
        # Kept as attributes: the app's own lookups search the active screen,
        # which is not this one while the help is open.
        self.board_view = BoardView(id="board")
        self.board_scroll = ScrollableContainer(self.board_view, id="board-scroll")
        self.status_view = Static(id="status")
        self.message_view = Static(id="message")
        self.players_view = Static(id="players")
        self.actions_view = OptionList(id="actions")
        # Lines are wrapped here (hanging indent), not by the log.
        self.log_view = RichLog(id="log", wrap=False, markup=False, auto_scroll=True)
        self.command_input = Input(
            placeholder="command (help lists them); escape to cancel", id="command"
        )
        yield Header()
        with Horizontal(id="main"):
            yield self.board_scroll
            with Vertical(id="side"):
                yield self.status_view
                yield self.message_view
                yield self.players_view
                yield self.actions_view
                yield self.log_view
        yield self.command_input
        yield Footer()

    @property
    def local(self) -> bool:
        return self.session.kind == "local"

    def on_mount(self) -> None:
        self.title = "catanatron-tui"
        self.sub_title = self.session.title
        self.session.on_change = self._session_changed
        self.actions_view.focus()
        if self.local:
            for command in self.startup_commands:
                self.run_command(command)
        else:
            self.run_worker(self.session.run(), group="remote", exit_on_error=False)
        self.set_interval(0.5, self._tick)
        self.refresh_view()
        self.schedule()

    def on_resize(self, event: events.Resize) -> None:
        self._log_key = None  # rewrap the log
        self.refresh_view()

    def check_action(self, action: str, parameters: tuple) -> Optional[bool]:
        if not self.local:
            if action in self.LOCAL_ONLY:
                return False
            if (
                action == "prefill"
                and parameters
                and parameters[0] in self.LOCAL_PREFILLS
            ):
                return False
        elif action == "verify":
            return False
        return True

    # ===== drawing =====
    @property
    def frame(self) -> Optional[Frame]:
        """What the session shows now (computed at most once per change)."""
        if self._frame is None:
            self._frame = self.session.frame()
        return self._frame

    def _session_changed(self) -> None:
        self.refresh_view()

    def _alive(self) -> bool:
        """Whether there is a screen to draw on (not so while shutting down,
        when a timer, a bot's thread or a server message can still arrive)."""
        try:
            return (
                self.is_running
                and bool(self.screen_stack)
                and self.board_view.is_mounted
            )
        except Exception:
            return False

    def refresh_view(self, now: bool = False) -> None:
        """Redraw everything, soon: changes that come in quick succession
        (bots at full speed, a burst of server messages) share one redraw."""
        self._frame = None
        if not self._alive():
            return
        if now:
            self._redraw()
        elif self._redraw_timer is None:
            wait = self._last_redraw + REDRAW_INTERVAL - time.monotonic()
            self._redraw_timer = self.set_timer(max(wait, 0.001), self._redraw)

    def _redraw(self) -> None:
        if self._redraw_timer is not None:
            self._redraw_timer.stop()
            self._redraw_timer = None
        if not self._alive():
            return
        self._last_redraw = time.monotonic()
        frame = self.frame
        self.sub_title = frame.status
        self._update_status()
        self.players_view.update(players_panel(frame))
        self._update_actions(frame)
        self._update_log(frame)
        self.refresh_board()

    def _update_status(self) -> None:
        if not self._alive():
            return
        frame = self.frame
        if frame is None:
            return
        if not self.local:
            frame.flags = self.session.flags()
        self.status_view.update(status_panel(frame, self._extra_flags()))

    def _extra_flags(self) -> List[str]:
        if not self.local:
            return []
        flags = []
        if self.thinking is not None:
            turn, since = self.thinking
            flags.append(
                f"{turn.color.value} ({turn.agent.label}) thinking "
                f"{time.monotonic() - since:.1f}s"
            )
        if not self.autoplay:
            flags.append("bots paused (space resumes, n steps)")
        else:
            flags.append(f"bot delay {self.delay:g}s")
        return flags

    def show_message(self, message: Optional[str], *, error: bool = False) -> None:
        if not message:
            return
        self.messages.append(message)
        if not self._alive():
            return
        style = "bold #ff6b6b" if error else "italic #a0c0ff"
        self.message_view.update(Text(message, style=style))

    def _tick(self) -> None:
        """Keep the clocks moving (thinking time, decision countdowns)."""
        if self.thinking is not None or (
            not self.local and getattr(self.session, "pending", None) is not None
        ):
            self._update_status()

    def screen_text(self) -> str:
        """The screen as plain text (for tests and text screenshots)."""
        import io

        from rich.console import Console

        width, height = self.size
        console = Console(
            width=width,
            height=height,
            file=io.StringIO(),
            record=True,
            legacy_windows=False,
            safe_box=False,
        )
        console.print(self.screen._compositor.render_update(full=True))
        return "\n".join(line.rstrip() for line in console.export_text().splitlines())

    def refresh_board(self) -> None:
        frame = self.frame
        board = self.board_view
        if frame is None or frame.catan_map is None or frame.document is None:
            self._picture = self._picture_key = None
            board.picture = None
            board.refresh()
            return
        targets = self._targets(frame)
        highlights = set(targets) if frame.actor is not None else set()
        if self.filter_target is not None:
            highlights = {self.filter_target}
        focused = self.focused is board
        if focused and (
            self.cursor is None or self.cursor not in self._cursor_candidates()
        ):
            self.cursor = self._first_candidate()
        cursor = self.cursor if focused else None
        size = self._board_size(frame)
        pieces = frame.document["board"]
        key = (
            id(frame.catan_map),
            repr(pieces["buildings"]),
            repr(pieces["roads"]),
            repr(pieces["robber_coordinate"]),
            frozenset(highlights),
            cursor,
            tuple(frame.recent),
            size,
            self.node_ids,
        )
        if key == self._picture_key:
            return
        picture = draw_board(
            frame.catan_map,
            BoardState.from_document(pieces),
            size=size,
            highlights=highlights,
            cursor=cursor,
            recent=frame.recent,
            node_ids=self.node_ids,
        )
        self._picture, self._picture_key = picture, key
        board.cursor = cursor
        board.show(picture)

    def _board_size(self, frame: Frame) -> str:
        if self.zoom != "auto":
            return self.zoom
        catan_map = frame.catan_map
        if self._sizes_map is not catan_map:
            self._sizes = {}
            for size in SIZES:
                picture = draw_board(catan_map, size=size)
                self._sizes[size] = (picture.width, picture.height)
            self._sizes_map = catan_map
        container = self.board_scroll
        width, height = container.content_size
        for size in reversed(SIZES):
            w, h = self._sizes[size]
            if w <= width and h <= height:
                return size
        return "small"

    def _targets(self, frame: Frame) -> Dict[Target, List[Action]]:
        return legal_targets(frame.actions) if frame.actor is not None else {}

    def _update_actions(self, frame: Frame) -> None:
        options = self.actions_view
        actions = frame.actions
        if self.filter_target is not None:
            on_target = self._targets(frame).get(self.filter_target)
            if on_target:
                actions = on_target
            else:
                self.filter_target = None
        key = (
            tuple(str(a) for a in actions),
            frame.actor,
            frame.current,
            frame.over,
            self.filter_text,
            self.filter_target,
            frame.document is not None,
        )
        if key == self._actions_key:
            return
        self._actions_key = key
        self._shown_actions = {}
        entries: List = []
        if frame.actor is None:
            options.border_title = None
            if frame.document is not None and not frame.over:
                who = frame.current or "?"
                entries.append(
                    Option(
                        Text(
                            f"{who} ({frame.seats.get(who, '?')}) to move",
                            style="italic",
                        ),
                        disabled=True,
                    )
                )
        else:
            options.border_title = f"{frame.actor} to play"
            catan_map = frame.catan_map
            needle = self.filter_text.lower()
            for title, group in grouped(actions, catan_map):
                labels = [(a, describe(a, catan_map)) for a in group]
                if needle:
                    labels = [(a, t) for a, t in labels if needle in t.lower()]
                if not labels:
                    continue
                entries.append(Option(Text(title, style="bold #8a8a8a"), disabled=True))
                for action, label in labels:
                    option_id = f"a{len(self._shown_actions)}"
                    self._shown_actions[option_id] = action
                    entries.append(Option(Text("  " + label), id=option_id))
            if needle or self.filter_target is not None:
                what = self.filter_text or "on the board pick"
                options.border_subtitle = f"filter: {what} (escape clears)"
            else:
                options.border_subtitle = None
            if not self._shown_actions:
                entries.append(
                    Option(
                        Text("nothing matches the filter", style="italic"),
                        disabled=True,
                    )
                )
        options.clear_options()
        options.add_options(entries)
        first = next(
            (
                i
                for i, o in enumerate(entries)
                if isinstance(o, Option) and not o.disabled
            ),
            None,
        )
        if first is not None:
            options.highlighted = first

    def _update_log(self, frame: Frame) -> None:
        log = self.log_view
        if frame.document is None or frame.catan_map is None:
            return
        records = frame.document["action_records"]
        key = (frame.viewer, frame.god, frame.catan_map)
        stale = (
            key != self._log_key
            or self._log_count > len(records)
            or (self._log_count and records[self._log_count - 1] != self._log_last)
        )
        if stale:
            log.clear()
            self._log_key = key
            self._log_count = 0
        width = max(log.scrollable_content_region.width, 30)
        for line in log_lines(
            frame.document, frame.catan_map, frame.notes, self._log_count
        ):
            for part in _hanging(line, width, self.console):
                log.write(part)
        self._log_count = len(records)
        self._log_last = records[-1] if records else None

    # ===== choosing =====
    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        action = self._shown_actions.get(event.option.id or "")
        if action is not None:
            self.choose(action)

    def choose(self, action: Action) -> None:
        self.filter_target = None
        self.filter_text = ""
        try:
            self.session.choose(action)
        except CommandError as error:
            self.show_message(str(error), error=True)
        # At once: the list must not offer the old moves a moment longer.
        self.refresh_view(now=True)
        self.schedule()

    def on_board_view_picked(self, message: BoardView.Picked) -> None:
        target = message.target
        self.cursor = target
        frame = self.frame
        if frame is None:
            return
        on_target = self._targets(frame).get(target)
        if not on_target:
            self.show_message(self.describe_target(target))
            self.refresh_board()
            return
        if len(on_target) == 1:
            self.choose(on_target[0])
            return
        # Several moves here (robber victims): choose among them in the list.
        self.filter_target = target
        self.refresh_view()
        self.actions_view.focus()

    def on_board_view_moved(self, message: BoardView.Moved) -> None:
        candidates = self._cursor_candidates()
        if not candidates or self._picture is None:
            return
        anchors = self._picture.anchors
        if self.cursor not in anchors or self.cursor not in candidates:
            self.cursor = self._first_candidate()
        elif message.direction:
            self.cursor = _next_target(
                self.cursor, DIRECTIONS[message.direction], candidates, anchors
            )
        self.refresh_board()
        self._scroll_to(self.cursor)
        if self.cursor is not None:
            self.show_message(self.describe_target(self.cursor))

    def on_board_view_left(self, message: BoardView.Left) -> None:
        board = self.board_view
        if self.focused is board:
            self.actions_view.focus()
        self.refresh_board()

    def on_board_view_hovered(self, message: BoardView.Hovered) -> None:
        if message.target is not None:
            self.board_view.tooltip = self.describe_target(message.target)
        else:
            self.board_view.tooltip = None

    def _cursor_candidates(self) -> List[Target]:
        if self._picture is None or self.frame is None:
            return []
        legal = list(self._targets(self.frame))
        if legal:
            return [t for t in legal if t in self._picture.anchors]
        return [t for t in self._picture.anchors if t[0] in ("node", "tile")]

    def _first_candidate(self) -> Optional[Target]:
        candidates = self._cursor_candidates()
        if not candidates or self._picture is None:
            return None
        anchors = self._picture.anchors
        return min(candidates, key=lambda t: (anchors[t][1], anchors[t][0]))

    def _scroll_to(self, target: Optional[Target]) -> None:
        if (
            target is None
            or self._picture is None
            or target not in self._picture.anchors
        ):
            return
        x, y = self._picture.anchors[target]
        container = self.board_scroll
        board = self.board_view
        offset = board.virtual_region.offset
        container.scroll_to_region(
            Region(offset.x + x - 4, offset.y + y - 2, 9, 5), animate=False
        )

    def describe_target(self, target: Target) -> str:
        frame = self.frame
        if frame is None or frame.catan_map is None or frame.document is None:
            return ""
        catan_map = frame.catan_map
        state = BoardState.from_document(frame.document["board"])
        kind, value = target
        if kind == "node":
            text = f"node {value}: {node_label(catan_map, value)}"
            building = state.buildings.get(value)
            if building:
                text += f"; {building[0]} {building[1].lower()}"
            return text
        if kind == "edge":
            owner = state.roads.get(value)
            return f"edge {value[0]}-{value[1]}" + (f": {owner} road" if owner else "")
        robber = " (robber)" if tuple(value) == state.robber else ""
        return f"tile {tile_label(catan_map, value)}{robber}"

    # ===== bots (local) =====
    def schedule(self) -> None:
        """Let forced moves and bots play, at the chosen speed."""
        if not self.local:
            return
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        session = self.session
        if session.over:
            self.steps_left = 0
        if self.thinking is not None or session.over:
            return
        stepping = self.steps_left > 0
        if session.person_to_move():
            self.steps_left = 0
            return
        human = session.agent(session.current_color).human
        if not (self.autoplay or stepping or human):
            return
        delay = 0 if human or stepping else self.delay
        self._timer = self.set_timer(max(delay, 0.001), self._advance)

    def _advance(self) -> None:
        self._timer = None
        if not self._alive():
            return
        session = self.session
        if self.thinking is not None or session.over:
            return
        forced = session.forced_action()
        if forced is not None:
            self.steps_left = max(0, self.steps_left - 1)
            try:
                session.play(forced)
            except CommandError as error:
                self.show_message(str(error), error=True)
            self.schedule()
            return
        turn = session.bot_turn()
        if turn is None:
            return
        self.thinking = (turn, time.monotonic())
        self._think(turn)
        self.refresh_view()

    @work(thread=True, group="bot")
    def _think(self, turn) -> None:
        start = time.perf_counter()
        try:
            action, error = turn.think(), None
        except Exception as exc:  # report it, keep the app alive
            action, error = None, exc
        seconds = time.perf_counter() - start
        try:
            self.call_from_thread(self._thought, turn, action, error, seconds)
        except RuntimeError:  # the app is gone
            pass

    def _thought(self, turn, action, error, seconds) -> None:
        self.thinking = None
        if not self._alive():
            return
        if error is not None:
            self.autoplay = False
            self.steps_left = 0
            self.show_message(
                f"{turn.color.value} ({turn.agent.label}) failed: {type(error).__name__}: {error}",
                error=True,
            )
            self.refresh_view()
            return
        self.steps_left = max(0, self.steps_left - 1)
        try:
            self.session.finish(turn, action, seconds)
        except CommandError as error:
            self.show_message(str(error), error=True)
        self.refresh_view()
        self.schedule()

    # ===== key actions =====
    def action_help(self) -> None:
        self.push_screen(HelpScreen(KEYS_HELP + "\n\n" + self.session.help()))

    def action_command(self) -> None:
        self.action_prefill("")

    def action_prefill(self, text: str) -> None:
        command = self.command_input
        command.add_class("open")
        command.value = text
        command.cursor_position = len(text)
        command.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        line = event.value.strip()
        self._close_command()
        if line:
            self.run_command(line)

    def on_key(self, event: events.Key) -> None:
        if event.key != "escape":
            return
        command = self.command_input
        if command.has_class("open"):
            self._close_command()
            event.stop()
        elif self.filter_text or self.filter_target is not None:
            self.filter_text = ""
            self.filter_target = None
            self.refresh_view()
            event.stop()

    def _close_command(self) -> None:
        command = self.command_input
        command.remove_class("open")
        command.value = ""
        self.actions_view.focus()

    def run_command(self, line: str) -> None:
        """Run a command line: the app's own, else the session's."""
        try:
            message = self._app_command(line)
            if message is None:
                message = self.session.execute(line)
        except CommandError as error:
            self.show_message(str(error), error=True)
            return
        except Exception as error:  # a command must never take the app down
            self.show_message(f"{type(error).__name__}: {error}", error=True)
            return
        self.show_message(message)
        self.refresh_view()
        self.schedule()

    def _app_command(self, line: str) -> Optional[str]:
        if line.startswith("/"):
            self.filter_text = line[1:].strip()
            self.filter_target = None
            return (
                f"filter: {self.filter_text}" if self.filter_text else "filter cleared"
            )
        words = line.split()
        command, args = words[0].lower(), words[1:]
        if command in ("help", "?"):
            self.action_help()
            return ""
        if command in ("quit", "exit", "q"):
            self.exit()
            return ""
        if command == "filter":
            return self._app_command("/" + " ".join(args))
        if command == "zoom":
            if len(args) != 1 or args[0] not in ("auto", *SIZES):
                raise CommandError("usage: zoom auto|small|medium|large")
            self.zoom = args[0]
            return f"board size {self.zoom}"
        if command == "ids":
            self.node_ids = not args or args[0] in ("on", "yes", "1")
            return "node ids " + ("on" if self.node_ids else "off")
        if not self.local:
            return None
        if command == "step":
            self.steps_left = _number(args, 1)
            self._advance_now()
            return ""
        if command == "auto":
            self.autoplay = not args or args[0] in ("on", "yes", "1")
            return "bots " + ("play on" if self.autoplay else "paused")
        if command == "speed":
            try:
                self.delay = max(0.0, float(args[0]))
            except (IndexError, ValueError):
                raise CommandError(
                    "usage: speed SECONDS (delay between bot moves)"
                ) from None
            return f"bot delay {self.delay:g}s"
        return None

    def _advance_now(self) -> None:
        if self.thinking is None and not self.session.over:
            if self._timer is not None:
                self._timer.stop()
                self._timer = None
            self._advance()

    def action_board(self) -> None:
        self.board_view.focus()

    def action_autoplay(self) -> None:
        self.autoplay = not self.autoplay
        self.show_message("bots " + ("play on" if self.autoplay else "paused"))
        self.refresh_view()
        self.schedule()

    def action_step(self) -> None:
        if self.session.person_to_move():
            self.show_message("it is a person's move", error=True)
            return
        self.steps_left = 1
        self._advance_now()

    def action_undo(self) -> None:
        self._history("undo")

    def action_redo(self) -> None:
        self._history("redo")

    def action_back(self) -> None:
        self._history("back")

    def _history(self, command: str) -> None:
        # Taking moves back would be pointless with bots replaying them at once.
        self.autoplay = False
        self.steps_left = 0
        self.run_command(command)

    def action_god(self) -> None:
        self.run_command("god")

    def action_cycle_view(self) -> None:
        session = self.session
        modes = ["follow", *[c.value for c in session.colors], "public", "god"]
        current = session.view_mode
        mode = (
            modes[(modes.index(current) + 1) % len(modes)]
            if current in modes
            else "follow"
        )
        self.run_command(f"view {mode}")

    def action_speed(self, step: int) -> None:
        index = min(range(len(SPEEDS)), key=lambda i: abs(SPEEDS[i] - self.delay))
        self.delay = SPEEDS[max(0, min(len(SPEEDS) - 1, index + step))]
        self.show_message(f"bot delay {self.delay:g}s")
        self.refresh_view()

    def action_zoom(self) -> None:
        sizes = ("auto", *SIZES)
        self.zoom = sizes[(sizes.index(self.zoom) + 1) % len(sizes)]
        self.show_message(f"board size {self.zoom}")
        self.refresh_board()

    def action_ids(self) -> None:
        self.node_ids = not self.node_ids
        self.refresh_board()

    def action_verify(self) -> None:
        self.run_command("verify")


def _hanging(line: Text, width: int, console, indent: int = 6) -> List[Text]:
    """``line`` wrapped to ``width``, continuation lines indented."""
    lines = list(line.wrap(console, width, no_wrap=False))
    if len(lines) <= 1:
        return lines
    rest = Text(" ").join(lines[1:])
    wrapped = [lines[0]]
    for part in rest.wrap(console, width - indent):
        wrapped.append(Text(" " * indent) + part)
    return wrapped


def _next_target(current, direction, candidates, anchors):
    """The nearest candidate in ``direction`` from ``current`` (rows count
    double: a cell is about twice as tall as it is wide)."""
    x, y = anchors[current]
    dx, dy = direction
    best, best_score = current, None
    for target in candidates:
        if target == current:
            continue
        tx, ty = anchors[target]
        ex, ey = tx - x, (ty - y) * 2
        along = ex * dx + ey * dy
        if along <= 0:
            continue
        across = abs(ex * dy - ey * dx)
        score = along + 2 * across
        if best_score is None or score < best_score:
            best, best_score = target, score
    return best


def _number(args, default: int) -> int:
    if not args:
        return default
    try:
        return max(1, int(args[0]))
    except ValueError:
        raise CommandError(f"expected a number, got {args[0]!r}") from None
