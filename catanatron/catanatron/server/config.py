"""A room's game: its rules, and how a game is built from them and a seed.

:func:`new_game` is shared by the server and the verifier: given the same
config, master seed and game id it builds the same game, which is what lets a
transcript be replayed in stock Catanatron. Nothing here needs the
``cryptography`` package, so local tools (the TUI) use it too.
"""

from __future__ import annotations

import functools
import random
from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence

from catanatron.game import TURNS_LIMIT, Game
from catanatron.models.decks import BASE_SUPPLY, EXTENSION_SUPPLY, Supply
from catanatron.models.enums import Action, ActionPrompt, ActionType
from catanatron.models.map import CatanMap, Layout, build_map, layout_graph
from catanatron.models.player import Color, Player
from catanatron.server.hashing import derive_seed
from catanatron.server.maps import MapError, normalize_map, parse_map
from catanatron.state_functions import player_has_rolled, player_num_dev_cards

MAP_TYPES = ("BASE", "MINI", "TOURNAMENT")
NUMBER_PLACEMENTS = ("official_spiral", "random")
SUPPLIES = {"base": BASE_SUPPLY, "extension": EXTENSION_SUPPLY}


class ConfigError(ValueError):
    """A game config that cannot be played."""


@dataclass(frozen=True)
class GameConfig:
    """Everything that defines a room's game, apart from who plays it."""

    num_players: int = 4
    #: BASE, MINI or TOURNAMENT; ignored when ``map`` is given.
    map_type: str = "BASE"
    #: An explicit map spec (see :mod:`catanatron.server.maps`), or None.
    map: Optional[dict] = None
    number_placement: str = "official_spiral"
    vps_to_win: int = 10
    discard_limit: int = 7
    friendly_robber: bool = False
    #: "base" or "extension" (the 5-6 player bank and deck); None picks by
    #: the number of players.
    supply: Optional[str] = None
    #: How long a seat may take over one decision before the server plays
    #: :func:`default_action` for it.
    decision_timeout_ms: int = 60_000
    #: Play a seat's move without asking it when everyone can see it has
    #: only one (see :func:`publicly_forced`).
    autoplay_forced: bool = True

    def __post_init__(self):
        if self.supply is None and type(self.num_players) is int:
            name = "extension" if self.num_players > 4 else "base"
            object.__setattr__(self, "supply", name)

    @classmethod
    def from_json(cls, payload: Any) -> "GameConfig":
        """Validate a config from a request (unknown keys are rejected)."""
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise ConfigError("a config is a JSON object")
        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise ConfigError(f"unknown config keys: {sorted(unknown)}")
        try:
            config = cls(**payload)
        except TypeError as error:
            raise ConfigError(str(error))
        config.validate()
        if config.map is not None:
            config = cls(**{**asdict(config), "map": normalize_map(config.map)})
        return config

    def validate(self) -> None:
        if _int(self.num_players) not in range(2, 7):
            raise ConfigError("num_players must be 2 to 6")
        if self.map_type not in MAP_TYPES:
            raise ConfigError(f"map_type must be one of {MAP_TYPES}")
        if self.number_placement not in NUMBER_PLACEMENTS:
            raise ConfigError(f"number_placement must be one of {NUMBER_PLACEMENTS}")
        if _int(self.vps_to_win) not in range(3, 31):
            raise ConfigError("vps_to_win must be 3 to 30")
        if _int(self.discard_limit) not in range(3, 101):
            raise ConfigError("discard_limit must be 3 to 100")
        if self.supply not in SUPPLIES:
            raise ConfigError(f"supply must be one of {sorted(SUPPLIES)}")
        if _int(self.decision_timeout_ms) not in range(100, 24 * 3600 * 1000 + 1):
            raise ConfigError("decision_timeout_ms must be 100 ms to a day")
        for name in ("friendly_robber", "autoplay_forced"):
            if not isinstance(getattr(self, name), bool):
                raise ConfigError(f"{name} must be true or false")
        try:
            catan_map = self.build_map(random.Random(0))
        except MapError as error:
            raise ConfigError(f"map: {error}")
        seats = settlement_spots(catan_map.layout, 2 * self.num_players)
        if seats < 2 * self.num_players:
            raise ConfigError(
                f"this map can run out of settlement spots after {seats} "
                f"settlements; {self.num_players} players need "
                f"{2 * self.num_players} to start"
            )

    def to_json(self) -> dict:
        return asdict(self)

    @property
    def colors(self) -> list:
        """Seat ``i`` plays ``colors[i]``. Turn order is drawn at the start."""
        return list(Color)[: self.num_players]

    @property
    def game_supply(self) -> Supply:
        return SUPPLIES[self.supply or "base"]

    def build_map(self, rng: random.Random) -> CatanMap:
        """The board: the explicit map, or the template shuffled by ``rng``."""
        if self.map is not None:
            return parse_map(self.map)
        return build_map(self.map_type, self.number_placement, rng=rng)  # type: ignore[arg-type]


def _int(value: Any) -> int:
    if type(value) is not int:
        raise ConfigError(f"expected an integer, got {value!r}")
    return value


@functools.lru_cache(maxsize=256)
def settlement_spots(layout: Layout, enough: int = 13) -> int:
    """The fewest settlements that can use up a board's spots, exactly (the
    smallest set of nodes, no two neighbors, that leaves no node free: the
    independent domination number), or ``enough`` if it is at least that.

    Players placing adversarially can reach that minimum, so a board seats
    ``n`` players only if it is at least ``2 * n``. Branch and bound: some
    spot in the closed neighborhood of the first free node must be taken,
    and each placement frees at most ``1 + max degree`` spots.
    """
    graph = layout_graph(layout)
    neighbors = {v: frozenset(graph.neighbors(v)) for v in graph.nodes}
    reach = 1 + max((len(n) for n in neighbors.values()), default=0)
    best = [enough]

    def search(free: frozenset, placed: int) -> None:
        if not free:
            best[0] = min(best[0], placed)
            return
        if placed + -(-len(free) // reach) >= best[0]:
            return  # cannot beat the best found
        # Branch on the free node with the fewest free choices around it.
        node = min(free, key=lambda v: (len(neighbors[v] & free), v))
        for choice in sorted({node} | (neighbors[node] & free)):
            search(free - {choice} - neighbors[choice], placed + 1)

    search(frozenset(graph.nodes), 0)
    return best[0]


class ServerSeat(Player):
    """A seat in a server's game. Its decisions come over the wire, never
    from the game, and nothing seated in the game ever sees the game."""

    def decide(self, game, playable_actions):
        raise RuntimeError("server seats are played by their clients")


def build_catan_map(config: GameConfig, rng: random.Random) -> CatanMap:
    return config.build_map(rng)


def new_game(
    config: GameConfig,
    master: bytes,
    game_id: str,
    players: Optional[Sequence[Player]] = None,
) -> Game:
    """The game a room plays, from its master seed.

    Chance comes from two streams derived from the master seed: one for the
    board, one that the game owns for everything else (turn order, the
    deck's shuffle, dice and steals).
    """
    catan_map = config.build_map(random.Random(derive_seed(master, "map")))
    game = Game(
        (
            list(players)
            if players is not None
            else [ServerSeat(c) for c in config.colors]
        ),
        seed=derive_seed(master, "game"),
        discard_limit=config.discard_limit,
        friendly_robber=config.friendly_robber,
        vps_to_win=config.vps_to_win,
        catan_map=catan_map,
        supply=config.game_supply,
    )
    game.id = game_id
    return game


def is_over(game: Game) -> bool:
    return (
        game.winning_color() is not None
        or game.state.num_turns >= TURNS_LIMIT
        or not game.playable_actions
    )


#: Prompts whose legal actions follow from the board and public counts alone.
PUBLIC_PROMPTS = (
    ActionPrompt.BUILD_INITIAL_SETTLEMENT,
    ActionPrompt.BUILD_INITIAL_ROAD,
    ActionPrompt.MOVE_ROBBER,
)


def publicly_forced(game: Game) -> bool:
    """Whether everyone can see that the seat to move has one legal action.

    Only then may the server play it without asking: autoplaying a move
    whose being forced depends on hidden cards (a roll from someone holding
    only victory point cards, a discard from a one-resource hand) would tell
    the table what is in that hand. Holds for the initial placements, robber
    moves without the friendly robber (victims are chosen by public card
    counts), free roads, and the roll of a seat that holds no development
    card.
    """
    if len(game.playable_actions) != 1:
        return False
    state = game.state
    prompt = state.current_prompt
    if prompt in PUBLIC_PROMPTS:
        return prompt != ActionPrompt.MOVE_ROBBER or not state.friendly_robber
    if prompt != ActionPrompt.PLAY_TURN:
        return False
    if state.is_road_building:
        return True
    color = state.current_color()
    return (
        not player_has_rolled(state, color) and player_num_dev_cards(state, color) == 0
    )


#: What a seat that ran out of time plays, in order of preference: the moves
#: that keep the game going without committing the seat to anything.
DEFAULT_PREFERENCE = (ActionType.ROLL, ActionType.END_TURN)


def default_action(playable_actions: Sequence[Action]) -> Action:
    """The move the server plays for a seat that timed out: the first
    preferred kind of move, else the first legal action (Catanatron lists
    them in the same order in every process). Deterministic, so a verifier
    can check the server did not pick a convenient one."""
    for action_type in DEFAULT_PREFERENCE:
        for action in playable_actions:
            if action.action_type == action_type:
                return action
    return playable_actions[0]
