"""A room's game: its rules, and how a game is built from them and a seed.

:func:`new_game` is shared by the server and the verifier: given the same
config, master seed and game id it builds the same game, which is what lets a
transcript be replayed in stock Catanatron.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence

from catanatron.game import TURNS_LIMIT, Game
from catanatron.models.decks import BASE_SUPPLY, EXTENSION_SUPPLY, Supply
from catanatron.models.enums import Action, ActionType
from catanatron.models.map import CatanMap, build_map
from catanatron.models.player import Color, Player
from catanatron.serialization import action_to_json
from catanatron.server.crypto import canonical, derive_seed
from catanatron.server.maps import MapError, normalize_map, parse_map

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
    #: "base" or "extension" (the 5-6 player bank and deck).
    supply: str = "base"
    #: How long a seat may take over one decision before the server plays
    #: :func:`default_action` for it.
    decision_timeout_ms: int = 60_000
    #: Play a seat's only legal action without asking it.
    autoplay_forced: bool = True

    @classmethod
    def from_json(cls, payload: Any) -> "GameConfig":
        """Validate a config from a request (unknown keys are rejected)."""
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise ConfigError("a config is a JSON object")
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(payload) - known
        if unknown:
            raise ConfigError(f"unknown config keys: {sorted(unknown)}")
        values = dict(payload)
        num_players = values.get("num_players", 4)
        values.setdefault("supply", "extension" if _int(num_players) > 4 else "base")
        try:
            config = cls(**values)
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
        if self.map is not None:
            try:
                parse_map(self.map)
            except MapError as error:
                raise ConfigError(f"map: {error}")

    def to_json(self) -> dict:
        return asdict(self)

    @property
    def colors(self) -> list:
        """Seat ``i`` plays ``colors[i]``. Turn order is drawn at the start."""
        return list(Color)[: self.num_players]

    @property
    def game_supply(self) -> Supply:
        return SUPPLIES[self.supply]


def _int(value: Any) -> int:
    if type(value) is not int:
        raise ConfigError(f"expected an integer, got {value!r}")
    return value


class ServerSeat(Player):
    """A seat in a server's game. Its decisions come over the wire, never
    from the game, and nothing seated in the game ever sees the game."""

    def decide(self, game, playable_actions):
        raise RuntimeError("server seats are played by their clients")


def build_catan_map(config: GameConfig, rng: random.Random) -> CatanMap:
    if config.map is not None:
        return parse_map(config.map)
    return build_map(config.map_type, config.number_placement, rng=rng)  # type: ignore[arg-type]


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
    catan_map = build_catan_map(config, random.Random(derive_seed(master, "map")))
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


def ordered(actions: Sequence[Action]) -> list:
    """``actions`` in a canonical order.

    Catanatron generates some actions from sets of strings (port resources,
    robbery victims), whose order changes with Python's hash seed. Whatever
    has to be the same in the server and in a verifier running in another
    process uses this order instead.
    """
    return sorted(actions, key=lambda action: canonical(action_to_json(action)))


#: What a seat that ran out of time plays, in order of preference: the moves
#: that keep the game going without committing the seat to anything.
DEFAULT_PREFERENCE = (ActionType.ROLL, ActionType.END_TURN)


def default_action(playable_actions: Sequence[Action]) -> Action:
    """The move the server plays for a seat that timed out: the first
    preferred kind of move, else the first in canonical order. Deterministic,
    so a verifier can check the server did not pick a convenient one."""
    candidates = ordered(playable_actions)
    for action_type in DEFAULT_PREFERENCE:
        for action in candidates:
            if action.action_type == action_type:
                return action
    return candidates[0]
