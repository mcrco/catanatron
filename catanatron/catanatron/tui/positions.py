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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from catanatron.game import Game
from catanatron.models.decks import Supply
from catanatron.models.enums import (
    CITY,
    DEVELOPMENT_CARDS,
    RESOURCES,
    ROAD,
    SETTLEMENT,
    Action,
    ActionType,
)
from catanatron.models.player import Color, Player
from catanatron.server.config import GameConfig
from catanatron.serialization import (
    action_record_from_json,
    state_from_json,
    state_to_json,
)

FORMAT = "catanatron-tui/1"
#: The actions whose log entry carries their outcome in the action's value.
CHANCE_IN_VALUE = (ActionType.ROLL, ActionType.BUY_DEVELOPMENT_CARD)
#: The rules of a new game: catanatron-server's room config, so one
#: ``--config`` works for both (server-only keys are simply not used here).
GameSettings = GameConfig


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
