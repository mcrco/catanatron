"""Games and positions for local sessions: new games, files and history.

A saved session is one JSON file::

    {"format": "catanatron-tui/1",
     "seats": {"RED": "human", "BLUE": "F", ...},
     "initial": <state document where the session's history starts>,
     "state": <state document now>,
     "notes": {"17": ["dice forced to 6+6"], ...}}

Both documents are ``state_to_json`` output, so the file is a position (load
it and play on from ``state``) and a full game log at once: the actions from
``initial`` to ``state`` are ``state["action_records"]`` past the initial
ones, each with its outcome (dice, card drawn, card stolen), and loading the
file replays them to rebuild the undo history. A bare ``state_to_json``
document and a catanatron-server archive record (its ``final_state``) load
as positions without history.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from catanatron.game import Game
from catanatron.models.decks import BASE_SUPPLY, EXTENSION_SUPPLY, Supply
from catanatron.models.enums import (
    CITY,
    DEVELOPMENT_CARDS,
    RESOURCES,
    ROAD,
    SETTLEMENT,
    Action,
    ActionType,
)
from catanatron.models.map import CatanMap, build_map
from catanatron.models.player import Color, Player
from catanatron.serialization import (
    action_record_from_json,
    state_from_json,
    state_to_json,
)

FORMAT = "catanatron-tui/1"
SUPPLIES = {"base": BASE_SUPPLY, "extension": EXTENSION_SUPPLY}
MAP_TYPES = ("BASE", "MINI", "TOURNAMENT")
NUMBER_PLACEMENTS = ("official_spiral", "random")
#: The actions whose log entry carries their outcome in the action's value.
CHANCE_IN_VALUE = (ActionType.ROLL, ActionType.BUY_DEVELOPMENT_CARD)


@dataclass
class GameSettings:
    """The rules of a new game. The keys are those of a catanatron-server
    room config (``GameConfig``), so one ``--config`` works for both."""

    num_players: int = 4
    #: BASE, MINI or TOURNAMENT; ignored when ``map`` is given.
    map_type: str = "BASE"
    #: An explicit map spec (``{"tiles": [...]}``, see catanatron.server.maps).
    map: Optional[dict] = None
    number_placement: str = "official_spiral"
    vps_to_win: int = 10
    discard_limit: int = 7
    friendly_robber: bool = False
    #: "base" or "extension"; None picks by the number of players.
    supply: Optional[str] = None

    @classmethod
    def from_json(cls, payload: Optional[dict]) -> "GameSettings":
        payload = dict(payload or {})
        known = set(cls.__dataclass_fields__)
        # Server-only keys are accepted and ignored, so one config serves both.
        ignored = {"decision_timeout_ms", "autoplay_forced"}
        unknown = set(payload) - known - ignored
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        settings = cls(**{k: v for k, v in payload.items() if k in known})
        settings.validate()
        return settings

    def validate(self) -> None:
        if not 2 <= self.num_players <= 6:
            raise ValueError("num_players must be 2 to 6")
        if self.map is None and self.map_type not in MAP_TYPES:
            raise ValueError(f"map_type must be one of {MAP_TYPES}")
        if self.number_placement not in NUMBER_PLACEMENTS:
            raise ValueError(f"number_placement must be one of {NUMBER_PLACEMENTS}")
        if self.supply is not None and self.supply not in SUPPLIES:
            raise ValueError(f"supply must be one of {sorted(SUPPLIES)}")

    @property
    def game_supply(self) -> Supply:
        name = self.supply or ("extension" if self.num_players > 4 else "base")
        return SUPPLIES[name]

    def to_json(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def build_map(self, rng: random.Random) -> CatanMap:
        if self.map is not None:
            from catanatron.server.maps import parse_map

            return parse_map(self.map)
        return build_map(self.map_type, self.number_placement, rng=rng)  # type: ignore[arg-type]


def colors_for(num_players: int) -> List[Color]:
    """Seat ``i`` plays ``colors[i]``, as on catanatron-server."""
    return list(Color)[:num_players]


def new_game(settings: GameSettings, players: Sequence[Player], seed: int) -> Game:
    """A fresh game; ``seed`` draws the board (if random) and every chance
    event. Turn order is drawn too, as in any Catanatron game."""
    return Game(
        list(players),
        seed=seed,
        discard_limit=settings.discard_limit,
        friendly_robber=settings.friendly_robber,
        vps_to_win=settings.vps_to_win,
        catan_map=settings.build_map(random.Random(seed)),
        supply=settings.game_supply,
    )


def document_of(game: Game) -> dict:
    """``state_to_json(game)``, detached from the game.

    ``state_to_json`` hands out the live ``player_state`` dict; a document
    that outlives the moment (a saved initial position, a bot's view on
    another thread) must not change as the game goes on.
    """
    doc = state_to_json(game)
    doc["player_state"] = dict(doc["player_state"])
    return doc


def supply_of(game: Game) -> Supply:
    """The game's bank, deck and pieces, recovered from the state.

    Nothing leaves the game: every resource is in the bank or a hand, every
    development card in the deck, a hand or played, every piece on the board
    or available. So a position needs no separate record of its supply.
    """
    state = game.state
    colors = state.colors
    bank = list(state.resource_freqdeck)
    deck = [state.development_listdeck.count(card) for card in DEVELOPMENT_CARDS]
    for index in range(len(colors)):
        key = f"P{index}_"
        for i, resource in enumerate(RESOURCES):
            bank[i] += state.player_state[f"{key}{resource}_IN_HAND"]
        for i, card in enumerate(DEVELOPMENT_CARDS):
            deck[i] += state.player_state[f"{key}{card}_IN_HAND"]
            deck[i] += state.player_state[f"{key}PLAYED_{card}"]
    first = colors[0]
    built = state.buildings_by_color.get(first, {})
    return Supply(
        resource_bank=tuple(bank),
        development_deck=tuple(deck),
        roads=state.player_state["P0_ROADS_AVAILABLE"] + len(built.get(ROAD, [])),
        settlements=state.player_state["P0_SETTLEMENTS_AVAILABLE"]
        + len(built.get(SETTLEMENT, [])),
        cities=state.player_state["P0_CITIES_AVAILABLE"] + len(built.get(CITY, [])),
    )


@dataclass
class SavedSession:
    state: dict
    initial: Optional[dict] = None
    seats: Optional[Dict[str, str]] = None
    notes: Dict[int, List[str]] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "format": FORMAT,
            "seats": self.seats,
            "initial": self.initial,
            "state": self.state,
            "notes": {str(k): v for k, v in sorted(self.notes.items())},
        }


def read_saved(path: str) -> SavedSession:
    """A session file, a bare state document or a server archive record."""
    with open(path) as file:
        payload = json.load(file)
    return parse_saved(payload)


def parse_saved(payload: Any) -> SavedSession:
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object")
    if payload.get("format") == FORMAT:
        return SavedSession(
            state=payload["state"],
            initial=payload.get("initial"),
            seats=payload.get("seats"),
            notes={int(k): list(v) for k, v in (payload.get("notes") or {}).items()},
        )
    if "format" in payload:
        raise ValueError(f"unknown format {payload['format']!r}")
    if "schema_version" in payload:
        return SavedSession(state=payload)
    if isinstance(payload.get("final_state"), dict):
        return SavedSession(state=payload["final_state"])
    raise ValueError(
        "not a catanatron-tui file, a state document or a server archive record"
    )


def write_saved(path: str, saved: SavedSession) -> None:
    with open(path, "w") as file:
        json.dump(saved.to_json(), file)


def intent(action: Action) -> Action:
    """The action as it was offered, before the engine filled in its outcome
    (a roll's dice, a purchase's card)."""
    if action.action_type in CHANCE_IN_VALUE:
        return Action(action.color, action.action_type, None)
    return action


def replay(
    initial: dict, state: dict, players: Sequence[Player]
) -> Tuple[Game, List[Tuple[Game, Any]]]:
    """Replay ``state``'s log from ``initial``.

    Returns the replayed game and, for every action after ``initial``, the
    game just before it with the random state then (an undo history).
    Raises ValueError if the log does not apply or does not lead to ``state``.
    """
    game = state_from_json(json.loads(json.dumps(initial)), players)
    start = len(initial["action_records"])
    records = state["action_records"]
    if records[:start] != initial["action_records"]:
        raise ValueError("the state does not continue the initial position")
    history = []
    for index, payload in enumerate(records[start:], start):
        record = action_record_from_json(payload)
        action = intent(record.action)
        if action not in game.playable_actions:
            raise ValueError(f"action {index} ({payload[0]}) is not legal there")
        history.append((game.copy(), game.random.getstate()))
        game.execute(action, action_record=record)
    replayed = state_to_json(game)
    for key in ("random_state", "game"):
        replayed.pop(key)
    expected = {k: v for k, v in state.items() if k not in ("random_state", "game")}
    if json.loads(json.dumps(replayed)) != expected:
        raise ValueError("replaying the log does not lead to the saved state")
    return game, history
