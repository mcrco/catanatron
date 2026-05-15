"""Fixed integer tensor ABI for the C++/CUDA Catanatron engine.

This module intentionally models transition state, not the existing Gym
observation tensor. It is small enough to round-trip in tests and explicit
enough to become the CUDA memory contract later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is optional outside gym installs
    np = None  # type: ignore

from catanatron.models.enums import ActionPrompt, CITY, ROAD, SETTLEMENT
from catanatron.models.player import Color
from catanatron.state_functions import player_key


MAX_PLAYERS = 4
NUM_NODES = 54
NUM_EDGES = 72
PLAYER_FIELDS = 5
NO_OWNER = -1

COLOR_TO_ID = {
    Color.RED: 0,
    Color.BLUE: 1,
    Color.ORANGE: 2,
    Color.WHITE: 3,
}

PROMPT_TO_ID = {
    ActionPrompt.BUILD_INITIAL_SETTLEMENT: 0,
    ActionPrompt.BUILD_INITIAL_ROAD: 1,
    ActionPrompt.PLAY_TURN: 2,
    ActionPrompt.DISCARD: 3,
    ActionPrompt.MOVE_ROBBER: 4,
    ActionPrompt.DECIDE_TRADE: 5,
    ActionPrompt.DECIDE_ACCEPTEES: 6,
}

BUILDING_TO_ID = {
    None: 0,
    SETTLEMENT: 1,
    CITY: 2,
}

STATIC_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (0, 5),
    (0, 20),
    (1, 2),
    (1, 6),
    (2, 3),
    (2, 9),
    (3, 4),
    (3, 12),
    (4, 5),
    (4, 15),
    (5, 16),
    (6, 7),
    (6, 23),
    (7, 8),
    (7, 24),
    (8, 9),
    (8, 27),
    (9, 10),
    (10, 11),
    (10, 29),
    (11, 12),
    (11, 32),
    (12, 13),
    (13, 14),
    (13, 34),
    (14, 15),
    (14, 37),
    (15, 17),
    (16, 18),
    (16, 21),
    (17, 18),
    (17, 39),
    (18, 40),
    (19, 20),
    (19, 21),
    (19, 46),
    (20, 22),
    (21, 43),
    (22, 23),
    (22, 49),
    (23, 52),
    (24, 25),
    (24, 53),
    (25, 26),
    (26, 27),
    (27, 28),
    (28, 29),
    (29, 30),
    (30, 31),
    (31, 32),
    (32, 33),
    (33, 34),
    (34, 35),
    (35, 36),
    (36, 37),
    (37, 38),
    (38, 39),
    (39, 41),
    (40, 42),
    (40, 44),
    (41, 42),
    (43, 44),
    (43, 47),
    (45, 46),
    (45, 47),
    (46, 48),
    (48, 49),
    (49, 50),
    (50, 51),
    (51, 52),
    (52, 53),
)


@dataclass(frozen=True)
class TensorState:
    scalars: Any
    player: Any
    nodes: Any
    edges: Any


def empty_tensor_state():
    """Allocate an empty transition-state tensor bundle."""
    if np is None:
        return TensorState(
            scalars=[0] * 8,
            player=[[0] * PLAYER_FIELDS for _ in range(MAX_PLAYERS)],
            nodes=[[NO_OWNER, 0] for _ in range(NUM_NODES)],
            edges=[NO_OWNER] * NUM_EDGES,
        )

    return TensorState(
        scalars=np.zeros((8,), dtype=np.int16),
        player=np.zeros((MAX_PLAYERS, PLAYER_FIELDS), dtype=np.int16),
        nodes=np.full((NUM_NODES, 2), [NO_OWNER, 0], dtype=np.int16),
        edges=np.full((NUM_EDGES,), NO_OWNER, dtype=np.int16),
    )


def python_game_to_tensor(game) -> TensorState:
    """Convert a Python Catanatron Game into the canonical transition ABI."""
    tensor = empty_tensor_state()
    state = game.state
    colors = list(state.colors)

    _set(tensor.scalars, 0, len(colors))
    _set(tensor.scalars, 1, state.current_player_index)
    _set(tensor.scalars, 2, state.current_turn_index)
    _set(tensor.scalars, 3, PROMPT_TO_ID[state.current_prompt])
    _set(tensor.scalars, 4, int(state.is_initial_build_phase))
    _set(tensor.scalars, 5, state.num_turns)
    _set(tensor.scalars, 6, getattr(game, "vps_to_win", 10))
    _set(tensor.scalars, 7, int(getattr(state, "friendly_robber", False)))

    for slot, color in enumerate(colors):
        key = player_key(state, color)
        _set2(tensor.player, slot, 0, COLOR_TO_ID[color])
        _set2(tensor.player, slot, 1, state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"])
        _set2(tensor.player, slot, 2, state.player_state[f"{key}_ROADS_AVAILABLE"])
        _set2(tensor.player, slot, 3, state.player_state[f"{key}_SETTLEMENTS_AVAILABLE"])
        _set2(tensor.player, slot, 4, state.player_state[f"{key}_CITIES_AVAILABLE"])

    for node_id, (color, building) in state.board.buildings.items():
        _set2(tensor.nodes, node_id, 0, COLOR_TO_ID[color])
        _set2(tensor.nodes, node_id, 1, BUILDING_TO_ID[building])

    edge_to_index = {edge: i for i, edge in enumerate(STATIC_EDGES)}
    for edge, color in state.board.roads.items():
        normalized = tuple(sorted(edge))
        if normalized in edge_to_index:
            _set(tensor.edges, edge_to_index[normalized], COLOR_TO_ID[color])

    return tensor


def cpp_snapshot_to_tensor(snapshot: Mapping[str, Any]) -> TensorState:
    """Convert a C++ pybind snapshot dict into the canonical transition ABI."""
    tensor = empty_tensor_state()
    colors = list(snapshot["colors"])

    _set(tensor.scalars, 0, len(colors))
    _set(tensor.scalars, 1, snapshot["current_player_index"])
    _set(tensor.scalars, 2, snapshot["current_turn_index"])
    _set(tensor.scalars, 3, snapshot["current_prompt_id"])
    _set(tensor.scalars, 4, int(snapshot["is_initial_build_phase"]))
    _set(tensor.scalars, 5, snapshot["num_turns"])

    for slot, color_id in enumerate(colors):
        _set2(tensor.player, slot, 0, color_id)
        _set2(tensor.player, slot, 1, snapshot["victory_points"][slot])
        _set2(tensor.player, slot, 2, snapshot["roads_available"][slot])
        _set2(tensor.player, slot, 3, snapshot["settlements_available"][slot])
        _set2(tensor.player, slot, 4, snapshot["cities_available"][slot])

    for node_id, owner in enumerate(snapshot["node_owner"]):
        _set2(tensor.nodes, node_id, 0, owner)
        _set2(tensor.nodes, node_id, 1, snapshot["node_building"][node_id])

    for edge_id, owner in enumerate(snapshot["edge_owner"]):
        _set(tensor.edges, edge_id, owner)

    return tensor


def tensor_to_debug_dict(tensor: TensorState) -> Dict[str, Any]:
    """Return a stable, readable view for parity assertions."""
    return {
        "scalars": _tolist(tensor.scalars),
        "player": _tolist(tensor.player),
        "occupied_nodes": [
            (node_id, row[0], row[1])
            for node_id, row in enumerate(_tolist(tensor.nodes))
            if row[0] != NO_OWNER
        ],
        "occupied_edges": [
            (edge_id, owner)
            for edge_id, owner in enumerate(_tolist(tensor.edges))
            if owner != NO_OWNER
        ],
    }


def _set(values, index: int, value: int) -> None:
    values[index] = value


def _set2(values, row: int, column: int, value: int) -> None:
    values[row][column] = value


def _tolist(values):
    if hasattr(values, "tolist"):
        return values.tolist()
    return values
