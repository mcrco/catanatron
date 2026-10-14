"""A local game: stock Catanatron in this process, with debugging controls.

Every seat is played by an :class:`~catanatron.tui.agents.Agent` (a person,
a registry bot on its view, an oracle, an external callback). The session
keeps an undo/redo history of snapshots, can force the next chance outcomes
(dice, development card, stolen resource), shows any seat's view or every
hand, and saves and loads positions with their history
(:mod:`catanatron.tui.positions`).

Bots think off the UI thread: :meth:`LocalSession.bot_turn` packs what the
seat may see into a :class:`BotTurn` (copies only), ``BotTurn.think()`` runs
anywhere, and :meth:`LocalSession.finish` plays the answer unless the game
moved on meanwhile (an undo, a load). Everything else runs on the caller's
thread. :meth:`step` and :meth:`run` do it all synchronously, for scripts and
tests.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from catanatron.game import TURNS_LIMIT, Game
from catanatron.models.enums import (
    DEVELOPMENT_CARDS,
    RESOURCES,
    Action,
    ActionRecord,
    ActionType,
)
from catanatron.models.player import Color
from catanatron.serialization import (
    client_view,
    public_view,
    state_from_json,
    state_to_json,
)
from catanatron.tui.agents import (
    Agent,
    Request,
    Seat,
    canonical_order,
    make_agent,
    oracle_copy,
)
from catanatron.tui.board import action_target
from catanatron.tui.positions import (
    GameSettings,
    SavedSession,
    colors_for,
    new_game,
    read_saved,
    replay,
    supply_of,
    write_saved,
)
from catanatron.tui.session import CommandError, Frame, Session

MAX_HISTORY = 20_000

HELP = """\
Local commands (type them after ':'):
  dice A B | dice off       force the next roll (e.g. dice 6 6)
  card NAME | card off      force the next development card drawn (knight,
                            year_of_plenty, monopoly, road_building, victory_point)
  steal RES | steal off     force what the next robber steals (wood ... ore)
  undo [N] | redo [N]       take back / replay N actions (keys u, r)
  back                      undo to the last time a person had a choice (key U)
  view follow|god|public|COLOR
                            whose cards to show (key v cycles, g toggles god view)
  seat COLOR SPEC           who plays COLOR from now on: human, F, AB:2,
                            oracle:F, a decide_fns name...
  hand COLOR wood=2 ore=1   set a hand (cards come from / go to the bank)
  robber TILE_ID            move the robber (no steal)
  save PATH | load PATH     the position and its history, as JSON
  new [SEED]                a new game with the same rules and seats"""

ALIASES = {
    "wd": "WOOD",
    "br": "BRICK",
    "sh": "SHEEP",
    "wh": "WHEAT",
    "or": "ORE",
    "k": "KNIGHT",
    "yop": "YEAR_OF_PLENTY",
    "mono": "MONOPOLY",
    "rb": "ROAD_BUILDING",
    "vp": "VICTORY_POINT",
}


def card_name(text: str, names: Sequence[str]) -> str:
    """``wheat``, ``WH`` or ``wh`` -> ``WHEAT`` (from ``names``)."""
    upper = ALIASES.get(text.lower(), text.upper())
    if upper not in names:
        raise CommandError(
            f"unknown {text!r}; expected one of {', '.join(n.lower() for n in names)}"
        )
    return upper


@dataclass
class Snapshot:
    game: Game
    random_state: Any


@dataclass
class Chance:
    """Outcomes forced on the next roll, purchase and steal."""

    dice: Optional[Tuple[int, int]] = None
    card: Optional[str] = None
    steal: Optional[str] = None


@dataclass
class BotTurn:
    """A bot's decision to make, packed so it can be made on any thread."""

    version: int
    color: Color
    agent: Agent
    request: Request

    def think(self) -> Action:
        return self.agent.decide(self.request)


class LocalSession(Session):
    kind = "local"
    debug = True

    def __init__(
        self,
        game: Game,
        seats: Mapping[Color, str],
        *,
        seed: int = 0,
        settings: Optional[GameSettings] = None,
        decide_fns: Optional[Mapping[str, Any]] = None,
        initial: Optional[dict] = None,
        history: Sequence[Snapshot] = (),
        notes: Optional[Mapping[int, List[str]]] = None,
        autoplay_forced: bool = True,
    ):
        self.game = game
        self.seed = seed
        self.settings = settings
        self.decide_fns = dict(decide_fns or {})
        self.autoplay_forced = autoplay_forced
        self.specs: Dict[Color, str] = {}
        self.agents: Dict[Color, Agent] = {}
        for color in game.state.colors:
            self._seat(color, seats.get(color, "F"))
        self.supply = supply_of(game)
        self.initial = initial if initial is not None else state_to_json(game)
        self.undo_stack: List[Snapshot] = list(history)
        self.redo_stack: List[Snapshot] = []
        self.notes: Dict[int, List[str]] = {
            k: list(v) for k, v in (notes or {}).items()
        }
        self.chance = Chance()
        self.view_mode = "follow"
        #: Bumped by every change, so a stale bot decision is dropped.
        self.version = 0
        self._last_human: Optional[Color] = None
        self._decisions = random.Random(f"{seed}:decisions")
        self.title = self._title()

    # ===== construction =====
    @classmethod
    def new(
        cls,
        seats: Optional[Sequence[str]] = None,
        *,
        settings: Optional[GameSettings] = None,
        seed: Optional[int] = None,
        decide_fns: Optional[Mapping[str, Any]] = None,
        autoplay_forced: bool = True,
    ) -> "LocalSession":
        """A fresh game. ``seats`` lists one spec per seat (RED, BLUE, ORANGE,
        WHITE, GREEN, BROWN in that order); default: you against F bots."""
        if settings is None:
            settings = GameSettings(num_players=len(seats) if seats else 4)
        if seats is None:
            seats = ["human"] + ["F"] * (settings.num_players - 1)
        if len(seats) != settings.num_players:
            raise ValueError(
                f"{len(seats)} seat specs for {settings.num_players} players"
            )
        seed = random.randrange(1 << 31) if seed is None else seed
        colors = colors_for(settings.num_players)
        for spec in seats:
            make_agent(spec, Color.RED, decide_fns=decide_fns)  # fail early
        game = new_game(settings, [Seat(c) for c in colors], seed)
        return cls(
            game,
            dict(zip(colors, seats)),
            seed=seed,
            settings=settings,
            decide_fns=decide_fns,
            autoplay_forced=autoplay_forced,
        )

    @classmethod
    def load(
        cls,
        path: str,
        seats: Optional[Sequence[str]] = None,
        *,
        decide_fns: Optional[Mapping[str, Any]] = None,
        seed: Optional[int] = None,
        autoplay_forced: bool = True,
    ) -> "LocalSession":
        """A session from a saved file (see :mod:`catanatron.tui.positions`).
        ``seats`` overrides the file's, in color order."""
        return cls.from_saved(
            read_saved(path),
            seats,
            decide_fns=decide_fns,
            seed=seed,
            autoplay_forced=autoplay_forced,
        )

    @classmethod
    def from_saved(
        cls,
        saved: SavedSession,
        seats: Optional[Sequence[str]] = None,
        *,
        decide_fns: Optional[Mapping[str, Any]] = None,
        seed: Optional[int] = None,
        autoplay_forced: bool = True,
    ) -> "LocalSession":
        doc = saved.state
        colors = [Color[c] for c in doc["colors"]]
        players = [Seat(c) for c in colors]
        by_seat = sorted(colors, key=list(Color).index)
        if seats is not None:
            if len(seats) != len(colors):
                raise ValueError(f"{len(seats)} seat specs for {len(colors)} players")
            specs = dict(zip(by_seat, seats))
        elif saved.seats:
            specs = {Color[c]: s for c, s in saved.seats.items()}
        else:
            specs = {c: ("human" if i == 0 else "F") for i, c in enumerate(by_seat)}

        initial, history, notes = doc, [], dict(saved.notes)
        if saved.initial is not None and saved.initial is not doc:
            try:
                _, steps = replay(saved.initial, doc, players)
                initial = saved.initial
                history = [Snapshot(game, state) for game, state in steps]
            except (ValueError, KeyError) as error:
                notes.setdefault(len(doc["action_records"]), []).append(
                    f"history not restored: {error}"
                )
        game = state_from_json(doc, players)
        if seed is None:
            seed = doc["game"].get("seed") or 0
        return cls(
            game,
            specs,
            seed=seed,
            decide_fns=decide_fns,
            initial=initial,
            history=history,
            notes=notes,
            autoplay_forced=autoplay_forced,
        )

    def _seat(self, color: Color, spec: str) -> None:
        seed = random.Random(f"{self.seed}:{color.value}").getrandbits(32)
        self.agents[color] = make_agent(
            spec, color, seed=seed, decide_fns=self.decide_fns
        )
        self.specs[color] = spec

    def _title(self) -> str:
        n = len(self.colors)
        tiles = len(self.game.state.board.map.land_tiles)
        return f"local game, {n} players, {tiles} land tiles"

    # ===== state =====
    @property
    def colors(self) -> Tuple[Color, ...]:
        return tuple(self.game.state.colors)

    @property
    def current_color(self) -> Color:
        return self.game.state.current_color()

    @property
    def winner(self) -> Optional[Color]:
        return self.game.winning_color()

    @property
    def over(self) -> bool:
        return (
            self.winner is not None
            or self.game.state.num_turns >= TURNS_LIMIT
            or not self.game.playable_actions
        )

    @property
    def humans(self) -> List[Color]:
        return [c for c in self.colors if self.agents[c].human]

    def agent(self, color: Color) -> Agent:
        return self.agents[color]

    def legal_actions(self) -> List[Action]:
        return [] if self.over else list(self.game.playable_actions)

    def person_to_move(self) -> bool:
        """Whether the game waits for a person's choice."""
        return (
            not self.over
            and self.agents[self.current_color].human
            and self.forced_action() is None
        )

    def forced_action(self) -> Optional[Action]:
        """The move to play without asking anyone: the only legal action (of a
        bot, or of a person when ``autoplay_forced``)."""
        if self.over:
            return None
        actions = self.game.playable_actions
        if len(actions) != 1:
            return None
        if self.agents[self.current_color].human and not self.autoplay_forced:
            return None
        return actions[0]

    # ===== playing =====
    def choose(self, action: Action) -> None:
        self.play(action)

    def play(
        self, action: Action, *, result: Any = None, notes: Sequence[str] = ()
    ) -> ActionRecord:
        """Play ``action`` for the seat to move. ``result`` forces its chance
        outcome (dice ``(a, b)``, a card, a stolen resource); otherwise any
        pending forced outcome applies."""
        if self.over:
            raise CommandError("the game is over")
        if action not in self.game.playable_actions:
            raise CommandError(f"{action} is not legal now")
        record, chance_notes = self._outcome(action, result)
        snapshot = Snapshot(self.game.copy(), self.game.random.getstate())
        index = len(self.game.state.action_records)
        try:
            executed = self.game.execute(action, action_record=record)
        except Exception as error:
            self._restore(snapshot)
            raise CommandError(f"cannot play {action}: {error}") from error
        self.undo_stack.append(snapshot)
        if len(self.undo_stack) > MAX_HISTORY:
            del self.undo_stack[0]
        self.redo_stack.clear()
        for key in [k for k in self.notes if k >= index]:
            del self.notes[key]
        if chance_notes or notes:
            self.notes[index] = [*chance_notes, *notes]
        self._bump()
        return executed

    def _outcome(
        self, action: Action, result: Any
    ) -> Tuple[Optional[ActionRecord], List[str]]:
        kind = action.action_type
        if result is not None:
            self._check_outcome(action, result)
            return ActionRecord(action, result), [
                f"outcome set: {_outcome_text(result)}"
            ]
        if kind == ActionType.ROLL and self.chance.dice is not None:
            dice, self.chance.dice = self.chance.dice, None
            return ActionRecord(action, dice), [f"dice set to {dice[0]}+{dice[1]}"]
        if kind == ActionType.BUY_DEVELOPMENT_CARD and self.chance.card is not None:
            card, self.chance.card = self.chance.card, None
            if card in self.game.state.development_listdeck:
                return ActionRecord(action, card), [f"card set to {card.lower()}"]
            return None, [f"no {card.lower()} left in the deck; drew at random"]
        victim = action.value[1] if kind == ActionType.MOVE_ROBBER else None
        if victim is not None and self.chance.steal is not None:
            resource, self.chance.steal = self.chance.steal, None
            if self._holds(victim, resource):
                return ActionRecord(action, resource), [
                    f"steal set to {resource.lower()}"
                ]
            return None, [f"{victim.value} has no {resource.lower()}; stole at random"]
        return None, []

    def _holds(self, color: Color, resource: str) -> bool:
        index = self.game.state.color_to_index[color]
        return self.game.state.player_state[f"P{index}_{resource}_IN_HAND"] > 0

    def _check_outcome(self, action: Action, result: Any) -> None:
        kind = action.action_type
        if kind == ActionType.ROLL:
            if not (
                isinstance(result, tuple)
                and len(result) == 2
                and all(isinstance(d, int) and 1 <= d <= 6 for d in result)
            ):
                raise CommandError("dice are two numbers from 1 to 6")
        elif kind == ActionType.BUY_DEVELOPMENT_CARD:
            if result not in self.game.state.development_listdeck:
                raise CommandError(f"no {str(result).lower()} left in the deck")
        elif kind == ActionType.MOVE_ROBBER and action.value[1] is not None:
            if result not in RESOURCES or not self._holds(action.value[1], result):
                raise CommandError(f"{action.value[1].value} has no {result}")
        else:
            raise CommandError(f"{kind.value} has no chance outcome to set")

    def bot_turn(self) -> Optional[BotTurn]:
        """The bot decision to make now, or None if it is not a bot's turn."""
        if self.over:
            return None
        color = self.current_color
        agent = self.agents[color]
        if agent.human:
            return None
        doc = state_to_json(self.game)
        if agent.oracle:
            view = client_view(doc)
            game = oracle_copy(self.game, self._decisions.getrandbits(63))
        else:
            view = client_view(doc, color)
            game = None
        request = Request(
            color, view, canonical_order(self.game.playable_actions), self.supply, game
        )
        return BotTurn(self.version, color, agent, request)

    def finish(
        self, turn: BotTurn, action: Action, seconds: Optional[float] = None
    ) -> bool:
        """Play a bot's decision; False if the game moved on since it was asked."""
        if turn.version != self.version:
            return False
        notes = (
            [f"{turn.agent.label} thought {seconds:.2f}s"]
            if seconds and seconds >= 0.05
            else []
        )
        self.play(action, notes=notes)
        return True

    def step(self) -> Optional[ActionRecord]:
        """Play one move that needs no person (forced, or a bot's), synchronously."""
        forced = self.forced_action()
        if forced is not None:
            return self.play(forced)
        turn = self.bot_turn()
        if turn is None:
            return None
        start = time.perf_counter()
        action = turn.think()
        self.finish(turn, action, time.perf_counter() - start)
        return self.game.state.action_records[-1]

    def run(self, limit: int = 100_000) -> int:
        """Step until a person must choose, the game ends or ``limit`` moves."""
        played = 0
        while played < limit and self.step() is not None:
            played += 1
        return played

    # ===== history =====
    def undo(self, count: int = 1) -> int:
        done = 0
        while done < count and self.undo_stack:
            self.redo_stack.append(Snapshot(self.game, self.game.random.getstate()))
            self._restore(self.undo_stack.pop())
            done += 1
        if done:
            self._bump()
        return done

    def redo(self, count: int = 1) -> int:
        done = 0
        while done < count and self.redo_stack:
            self.undo_stack.append(Snapshot(self.game, self.game.random.getstate()))
            self._restore(self.redo_stack.pop())
            done += 1
        if done:
            self._bump()
        return done

    def back(self) -> int:
        """Undo to the last point where a person had a choice."""
        done = self.undo(1)
        while self.undo_stack and not self.person_to_move():
            done += self.undo(1)
        return done

    def _restore(self, snapshot: Snapshot) -> None:
        # One random stream for the whole session, positioned as it was.
        rng = self.game.random
        rng.setstate(snapshot.random_state)
        game = snapshot.game
        game.random = rng
        game.state.random = rng
        self.game = game

    def _bump(self) -> None:
        self.version += 1
        self.changed()

    # ===== forced chance =====
    def set_dice(self, dice: Optional[Tuple[int, int]]) -> None:
        if dice is not None and not all(1 <= d <= 6 for d in dice):
            raise CommandError("dice are two numbers from 1 to 6")
        self.chance.dice = dice
        self.changed()

    def set_card(self, card: Optional[str]) -> None:
        if card is not None and card not in DEVELOPMENT_CARDS:
            raise CommandError(f"unknown card {card!r}")
        self.chance.card = card
        self.changed()

    def set_steal(self, resource: Optional[str]) -> None:
        if resource is not None and resource not in RESOURCES:
            raise CommandError(f"unknown resource {resource!r}")
        self.chance.steal = resource
        self.changed()

    # ===== seats and views =====
    def set_seat(self, color: Color, spec: str) -> None:
        if color not in self.colors:
            raise CommandError(f"{color.value} is not seated")
        try:
            self._seat(color, spec)
        except Exception as error:  # SpecError, or a bot that fails to build
            raise CommandError(str(error)) from error
        self._bump()

    def set_view(self, mode: str) -> None:
        mode = mode.strip()
        if mode.lower() in ("follow", "god", "public"):
            self.view_mode = mode.lower()
        else:
            self.view_mode = self.color(mode).value
        self.changed()

    def god(self) -> bool:
        return self.view_mode == "god" or (
            self.view_mode == "follow" and not self.humans
        )

    def viewer(self) -> Optional[Color]:
        """Whose hand is shown (None: nobody's, or everybody's in god view)."""
        if self.view_mode in ("god", "public"):
            return None
        if self.view_mode != "follow":
            return Color[self.view_mode]
        humans = self.humans
        if not humans:
            return None
        if not self.over and self.current_color in humans:
            self._last_human = self.current_color
        return self._last_human if self._last_human in humans else humans[0]

    def color(self, text: str) -> Color:
        names = [c.value for c in self.colors]
        if text.isdigit() and int(text) < len(names):
            return sorted(self.colors, key=list(Color).index)[int(text)]
        if text.upper() not in names:
            raise CommandError(f"{text!r} is not seated; seats are {', '.join(names)}")
        return Color[text.upper()]

    # ===== what is shown =====
    def frame(self) -> Frame:
        doc = state_to_json(self.game)
        viewer = self.viewer()
        god = self.god()
        if god:
            document = doc
        elif viewer is not None:
            document = client_view(doc, viewer)
        else:
            document = public_view(doc)
        over = self.over
        winner = self.winner
        current = self.current_color
        actions: List[Action] = []
        actor = None
        if not over and self.agents[current].human:
            actions = list(self.game.playable_actions)
            actor = current.value
        if winner is not None:
            status = f"{winner.value} wins"
        elif over:
            status = "the game is over (turn limit)"
        elif actor is not None:
            status = f"{actor} to play ({self.specs[current]})"
        else:
            status = f"{current.value} ({self.agents[current].label}) to move"
        flags = []
        if self.chance.dice:
            flags.append(f"next roll {self.chance.dice[0]}+{self.chance.dice[1]}")
        if self.chance.card:
            flags.append(f"next card {self.chance.card.lower()}")
        if self.chance.steal:
            flags.append(f"next steal {self.chance.steal.lower()}")
        records = self.game.state.action_records
        recent = ()
        if records:
            target = action_target(records[-1].action)
            recent = (target,) if target is not None else ()
        return Frame(
            document=document,
            catan_map=self.game.state.board.map,
            colors=[c.value for c in self.colors],
            seats={c.value: self.agents[c].label for c in self.colors},
            viewer=None if god else (viewer.value if viewer else None),
            god=god,
            current=None if over else current.value,
            winner=winner.value if winner else None,
            over=over,
            actions=actions,
            actor=actor,
            status=status,
            flags=flags,
            notes=self.notes,
            recent=recent,
        )

    # ===== files =====
    def saved(self) -> SavedSession:
        return SavedSession(
            state=state_to_json(self.game),
            initial=self.initial,
            seats={c.value: s for c, s in self.specs.items()},
            notes=self.notes,
        )

    def save(self, path: str) -> None:
        write_saved(path, self.saved())

    def adopt(self, other: "LocalSession") -> None:
        """Become ``other`` (a loaded or new game), keeping the UI's hooks."""
        on_change, version = self.on_change, self.version
        self.__dict__.update(other.__dict__)
        self.on_change = on_change
        self.version = version  # keeps counting, so older bot answers stay stale
        self._bump()

    # ===== edits =====
    def edit_hand(self, color: Color, counts: Mapping[str, int]) -> None:
        """Set some of ``color``'s resource counts; the bank pays or takes the difference."""
        game = self.game.copy()
        state = game.state
        index = state.color_to_index[color]
        for resource, count in counts.items():
            key = f"P{index}_{resource}_IN_HAND"
            delta = count - state.player_state[key]
            bank_index = RESOURCES.index(resource)
            if count < 0 or state.resource_freqdeck[bank_index] < delta:
                raise CommandError(f"the bank cannot pay {delta} {resource.lower()}")
            state.player_state[key] = count
            state.resource_freqdeck[bank_index] -= delta
        self._replace(game, f"{color.value}'s hand edited")

    def edit_robber(self, tile_id: int) -> None:
        tiles = self.game.state.board.map.tiles_by_id
        if tile_id not in tiles:
            raise CommandError(f"no land tile #{tile_id}")
        game = self.game.copy()
        coordinate = next(
            c for c, t in game.state.board.map.land_tiles.items() if t.id == tile_id
        )
        game.state.board.robber_coordinate = coordinate
        self._replace(game, f"robber moved to #{tile_id}")

    def _replace(self, game: Game, note: str) -> None:
        from catanatron.models.actions import generate_playable_actions

        game.playable_actions = generate_playable_actions(game.state)
        self.undo_stack.append(Snapshot(self.game, self.game.random.getstate()))
        self.redo_stack.clear()
        index = len(game.state.action_records)
        self.notes.setdefault(index, []).append(f"edit: {note}")
        self.game = game
        self._bump()

    # ===== commands =====
    def help(self) -> str:
        return HELP

    def execute(self, line: str) -> Optional[str]:
        words = line.split()
        if not words:
            return None
        command, args = words[0].lower(), words[1:]
        handler = getattr(self, f"_cmd_{command}", None)
        if handler is None:
            raise CommandError(f"unknown command {command!r}; try help")
        return handler(args)

    def _cmd_dice(self, args):
        if args in (["off"], []):
            self.set_dice(None)
            return "the next roll is random"
        try:
            if len(args) == 1 and len(args[0]) == 2:
                dice = (int(args[0][0]), int(args[0][1]))
            else:
                dice = tuple(int(a) for a in " ".join(args).replace(",", " ").split())
        except ValueError:
            raise CommandError(
                "usage: dice A B (two numbers from 1 to 6), or dice off"
            ) from None
        if len(dice) != 2:
            raise CommandError("usage: dice A B (two numbers from 1 to 6), or dice off")
        self.set_dice(dice)  # type: ignore[arg-type]
        return f"the next roll will be {dice[0]}+{dice[1]}"

    def _cmd_card(self, args):
        if args in (["off"], []):
            self.set_card(None)
            return "the next card is drawn at random"
        card = card_name("_".join(args), DEVELOPMENT_CARDS)
        self.set_card(card)
        warning = (
            "" if card in self.game.state.development_listdeck else " (none left now)"
        )
        return f"the next card drawn will be {card.lower()}{warning}"

    def _cmd_steal(self, args):
        if args in (["off"], []):
            self.set_steal(None)
            return "the next steal is random"
        resource = card_name(args[0], RESOURCES)
        self.set_steal(resource)
        return f"the next steal will take {resource.lower()} if the victim has one"

    def _cmd_undo(self, args):
        done = self.undo(_count(args))
        return f"undid {done}" if done else "nothing to undo"

    def _cmd_redo(self, args):
        done = self.redo(_count(args))
        return f"redid {done}" if done else "nothing to redo"

    def _cmd_back(self, args):
        done = self.back()
        return f"undid {done}" if done else "nothing to undo"

    def _cmd_view(self, args):
        if len(args) != 1:
            raise CommandError("usage: view follow|god|public|COLOR")
        self.set_view(args[0])
        return f"view: {self.view_mode.lower()}"

    def _cmd_god(self, args):
        self.set_view("follow" if self.view_mode == "god" else "god")
        return "god view " + ("on" if self.view_mode == "god" else "off")

    def _cmd_seat(self, args):
        if len(args) != 2:
            raise CommandError("usage: seat COLOR SPEC")
        color = self.color(args[0])
        self.set_seat(color, args[1])
        return f"{color.value} is played by {self.agents[color].label}"

    def _cmd_hand(self, args):
        if len(args) < 2:
            raise CommandError("usage: hand COLOR wood=2 ore=1 ...")
        color = self.color(args[0])
        counts = {}
        for item in args[1:]:
            name, _, value = item.partition("=")
            try:
                counts[card_name(name, RESOURCES)] = int(value)
            except ValueError:
                raise CommandError(f"bad count {item!r}") from None
        self.edit_hand(color, counts)
        return f"{color.value}'s hand set"

    def _cmd_robber(self, args):
        if len(args) != 1 or not args[0].lstrip("#").isdigit():
            raise CommandError("usage: robber TILE_ID")
        self.edit_robber(int(args[0].lstrip("#")))
        return f"robber on #{args[0].lstrip('#')}"

    def _cmd_save(self, args):
        if len(args) != 1:
            raise CommandError("usage: save PATH")
        self.save(args[0])
        return f"saved {args[0]}"

    def _cmd_load(self, args):
        if len(args) != 1:
            raise CommandError("usage: load PATH")
        try:
            other = LocalSession.load(
                args[0],
                decide_fns=self.decide_fns,
                autoplay_forced=self.autoplay_forced,
            )
        except (OSError, ValueError, KeyError) as error:
            raise CommandError(f"cannot load {args[0]}: {error}") from error
        self.adopt(other)
        return f"loaded {args[0]} ({len(self.undo_stack)} actions of history)"

    def _cmd_new(self, args):
        seed = _count(args, default=self.seed + 1)
        settings = self.settings or GameSettings(
            num_players=len(self.colors),
            vps_to_win=self.game.vps_to_win,
            discard_limit=self.game.state.discard_limit,
            friendly_robber=self.game.state.friendly_robber,
        )
        seats = [self.specs[c] for c in sorted(self.colors, key=list(Color).index)]
        other = LocalSession.new(
            seats,
            settings=settings,
            seed=seed,
            decide_fns=self.decide_fns,
            autoplay_forced=self.autoplay_forced,
        )
        self.adopt(other)
        return f"new game, seed {seed}"


def parse_outcome(text: str) -> Any:
    """``dice=3,4``, ``card=knight`` or ``steal=ore`` as an action's outcome."""
    key, _, value = text.partition("=")
    key = key.lower()
    if key == "dice":
        parts = value.replace("+", ",").split(",")
        try:
            dice = tuple(int(p) for p in parts)
        except ValueError:
            raise CommandError(f"bad outcome {text!r}") from None
        if len(dice) != 2 or not all(1 <= d <= 6 for d in dice):
            raise CommandError("dice are two numbers from 1 to 6")
        return dice
    if key == "card":
        return card_name(value, DEVELOPMENT_CARDS)
    if key == "steal":
        return card_name(value, RESOURCES)
    raise CommandError(f"bad outcome {text!r}; use dice=3,4, card=knight or steal=ore")


def _outcome_text(result: Any) -> str:
    if isinstance(result, tuple):
        return "+".join(str(d) for d in result)
    return str(result).lower()


def _count(args: Sequence[str], default: int = 1) -> int:
    if not args:
        return default
    try:
        value = int(args[0])
    except ValueError:
        raise CommandError(f"expected a number, got {args[0]!r}") from None
    if value < 0:
        raise CommandError("expected a non-negative number")
    return value
