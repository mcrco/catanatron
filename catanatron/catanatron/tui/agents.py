"""Who plays the seats of a local game, and what each of them gets to see.

A seat is given by a spec:

``human`` (or ``H``)
    A person at this terminal. Several human seats make a hot-seat game.
any registry spec: ``F``, ``AB:2``, ``R``, a bot registered with ``--bot``...
    The bot decides from its seat's view only. It is the server's
    :class:`~catanatron.server.bot.ViewBot`: each decision is made on a game
    sampled from the view (other hands dealt consistently with what the seat
    knows, a fresh deck order and random stream), so lookahead cannot see
    hidden cards or future rolls. This is how bots play on catanatron-server.
``oracle:SPEC``
    The same bot on a copy of the true game: every hand and the deck's order
    (Catanatron's classic in-process setting). The copy gets its own random
    stream, so the bot's lookahead neither sees nor disturbs the real dice.
a name from ``decide_fns`` (and ``oracle:NAME``)
    An external bot (see :class:`CallbackAgent`), e.g. a neural network.

The game itself never holds a deciding player: its seats are inert
:class:`Seat` placeholders and the session asks each seat's agent, handing
it only what that agent may see (:class:`Request`).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import random
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence, Union

from catanatron.game import Game
from catanatron.models.decks import Supply
from catanatron.models.enums import Action
from catanatron.models.player import Color, Player, RandomPlayer
from catanatron.registry import REGISTRY, SpecError
from catanatron.serialization import action_from_json, action_to_json

#: ``decide(view, playable_actions) -> action``: the same callback
#: catanatron-server's GameClient takes. ``view`` is the seat's
#: ``client_view`` document (with ``map`` and ``action_records``),
#: ``playable_actions`` are ``[color, action_type, value]`` triples, and the
#: answer is one of them or an ``Action`` (or an awaitable of either).
DecideFn = Callable[[dict, list], Any]

HUMAN_SPECS = ("human", "h")
ORACLE_PREFIX = "oracle:"


class Seat(Player):
    """A seat in a local game. Agents decide for it, never the game."""

    def decide(self, game, playable_actions):
        raise RuntimeError("seats of a catanatron-tui game are played by agents")


@dataclass
class Request:
    """What a seat's agent is handed when it must decide."""

    color: Color
    #: The seat's ``client_view`` (with ``map`` and ``action_records``).
    #: For an oracle, ``client_view`` with every hand face up.
    view: dict
    #: The legal actions, as JSON triples in canonical order.
    playable_actions: list
    #: The game's bank, deck and pieces (what sampling a world needs).
    supply: Supply
    #: A copy of the true game, for oracles only.
    game: Optional[Game] = None


class Agent:
    """Something that plays a seat."""

    label: str = "?"
    human: bool = False
    #: Whether the session hands this agent the true game.
    oracle: bool = False

    def decide(self, request: Request) -> Action:
        raise NotImplementedError


class HumanAgent(Agent):
    human = True

    def __init__(self, label: str = "human"):
        self.label = label


class ViewBotAgent(Agent):
    """A registry bot deciding on games sampled from its view (the server's ViewBot)."""

    def __init__(self, spec: str, color: Color, seed: Optional[int] = None):
        from catanatron.server.bot import ViewBot

        self.label = spec
        self.bot = ViewBot(spec, color, seed=seed)

    def decide(self, request: Request) -> Action:
        choice = self.bot.decide(request.view, request.playable_actions, request.supply)
        return action_from_json(choice)


class PlayerAgent(Agent):
    """A Catanatron ``Player``: on a world sampled from its view, or (oracle)
    on a copy of the true game."""

    def __init__(
        self,
        player: Player,
        label: str,
        *,
        oracle: bool = False,
        seed: Optional[int] = None,
    ):
        self.player = player
        self.label = label
        self.oracle = oracle
        self.rng = random.Random(seed)

    def decide(self, request: Request) -> Action:
        if self.oracle:
            if request.game is None:
                raise RuntimeError(f"{self.label} needs the true game")
            game = request.game
        else:
            from catanatron.server.sampling import sample_game

            game = sample_game(request.view, request.color, request.supply, self.rng)
        action = self.player.decide(game, game.playable_actions)
        if action_to_json(action) not in request.playable_actions:
            raise RuntimeError(f"{self.label} chose {action}, which is not legal")
        return action


class CallbackAgent(Agent):
    """An external bot: ``decide(view, playable_actions) -> action``.

    The callback is the one catanatron-server's ``GameClient.play`` takes, so
    a bot written for the server plays here unchanged. With ``oracle`` the
    view shows every hand (``client_view`` without a perspective).
    """

    def __init__(self, decide: DecideFn, label: str, *, oracle: bool = False):
        self.fn = decide
        self.label = label
        self.oracle = oracle

    def decide(self, request: Request) -> Action:
        result = self.fn(request.view, list(request.playable_actions))
        if inspect.isawaitable(result):
            result = asyncio.run(_awaited(result))
        if isinstance(result, Action):
            choice = action_to_json(result)
        else:
            choice = json.loads(json.dumps(result))  # tuples -> lists
        if choice not in request.playable_actions:
            raise RuntimeError(f"{self.label} answered {result!r}, which is not legal")
        return action_from_json(choice)


async def _awaited(awaitable):
    return await awaitable


def is_human_spec(spec: str) -> bool:
    return spec.strip().lower() in HUMAN_SPECS


def make_agent(
    spec: str,
    color: Color,
    *,
    seed: Optional[int] = None,
    decide_fns: Optional[Mapping[str, Union[DecideFn, Player]]] = None,
) -> Agent:
    """The agent for ``color`` from its seat spec (see the module docstring)."""
    text = spec.strip()
    if is_human_spec(text):
        return HumanAgent()
    oracle = text.lower().startswith(ORACLE_PREFIX)
    inner = text[len(ORACLE_PREFIX) :] if oracle else text
    if not inner:
        raise SpecError(f"empty seat spec {spec!r}")

    external = (decide_fns or {}).get(inner)
    if external is not None:
        if isinstance(external, Player):
            return PlayerAgent(external, text, oracle=oracle, seed=seed)
        if not callable(external):
            raise SpecError(f"decide_fns[{inner!r}] is neither a Player nor callable")
        return CallbackAgent(external, text, oracle=oracle)

    _register_builtins()
    player = REGISTRY.build(inner, color)
    if not player.is_bot:
        if oracle:
            raise SpecError(f"{inner!r} is not a bot")
        return HumanAgent()
    if oracle:
        return PlayerAgent(player, text, oracle=True, seed=seed)
    return ViewBotAgent(inner, color, seed=seed)


def check_seat_spec(spec: str, decide_fns: Optional[Mapping[str, Any]] = None) -> None:
    """Raise SpecError unless ``spec`` makes an agent."""
    make_agent(spec, Color.RED, decide_fns=decide_fns)


def _register_builtins() -> None:
    from catanatron.players import register_builtins

    register_builtins()


def canonical_order(actions: Sequence[Action]) -> list:
    """``actions`` as JSON triples, in the order catanatron-server offers them
    (sorted by their canonical JSON), so a bot sees the same list locally as
    over the wire."""
    triples = [action_to_json(a) for a in actions]
    return sorted(
        triples,
        key=lambda t: json.dumps(t, sort_keys=True, separators=(",", ":")),
    )


def oracle_copy(game: Game, seed: int) -> Game:
    """A copy of ``game`` for an oracle: true hands and deck, its own random
    stream and placeholder players (so playouts never reach a real seat)."""
    copy = game.copy()
    rng = random.Random(seed)
    copy.random = rng
    copy.state.random = rng
    copy.state.players = [RandomPlayer(color) for color in copy.state.colors]
    return copy
