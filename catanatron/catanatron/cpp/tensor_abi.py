"""Fixed integer tensor ABI for the C++/CUDA Catanatron engine.

This module intentionally models transition state, not the existing Gym
observation tensor. It is small enough to round-trip in tests and explicit
enough to become the CUDA memory contract later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

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
NUM_TILES = 19
NUM_RESOURCES = 5
NUM_SCALARS = 9
PLAYER_FIELDS = 11
TILE_FIELDS = 3
NO_OWNER = -1

SCALAR_NUM_PLAYERS = 0
SCALAR_CURRENT_PLAYER = 1
SCALAR_CURRENT_TURN = 2
SCALAR_PROMPT = 3
SCALAR_INITIAL_BUILD = 4
SCALAR_NUM_TURNS = 5
SCALAR_VPS_TO_WIN = 6
SCALAR_DISCARD_LIMIT = 7
SCALAR_ROBBER_TILE = 8

PLAYER_COLOR = 0
PLAYER_VPS = 1
PLAYER_ROADS_LEFT = 2
PLAYER_SETTLEMENTS_LEFT = 3
PLAYER_CITIES_LEFT = 4
PLAYER_HAS_ROLLED = 5
PLAYER_RESOURCE_START = 6

TILE_RESOURCE = 0
TILE_NUMBER = 1
TILE_HAS_ROBBER = 2

COLOR_TO_ID = {
    Color.RED: 0,
    Color.BLUE: 1,
    Color.ORANGE: 2,
    Color.WHITE: 3,
}

RESOURCE_TO_ID = {
    None: -1,
    "WOOD": 0,
    "BRICK": 1,
    "SHEEP": 2,
    "WHEAT": 3,
    "ORE": 4,
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
    tiles: Any
    bank: Any


def empty_tensor_state():
    """Allocate an empty transition-state tensor bundle."""
    if np is None:
        return TensorState(
            scalars=[0] * NUM_SCALARS,
            player=[[0] * PLAYER_FIELDS for _ in range(MAX_PLAYERS)],
            nodes=[[NO_OWNER, 0] for _ in range(NUM_NODES)],
            edges=[NO_OWNER] * NUM_EDGES,
            tiles=[[0] * TILE_FIELDS for _ in range(NUM_TILES)],
            bank=[0] * NUM_RESOURCES,
        )

    return TensorState(
        scalars=np.zeros((NUM_SCALARS,), dtype=np.int16),
        player=np.zeros((MAX_PLAYERS, PLAYER_FIELDS), dtype=np.int16),
        nodes=np.full((NUM_NODES, 2), [NO_OWNER, 0], dtype=np.int16),
        edges=np.full((NUM_EDGES,), NO_OWNER, dtype=np.int16),
        tiles=np.zeros((NUM_TILES, TILE_FIELDS), dtype=np.int16),
        bank=np.zeros((NUM_RESOURCES,), dtype=np.int16),
    )


def python_game_to_tensor(game) -> TensorState:
    """Convert a Python Catanatron Game into the canonical transition ABI."""
    tensor = empty_tensor_state()
    state = game.state
    colors = list(state.colors)

    robber_tile_id = next(
        tile.id
        for coordinate, tile in state.board.map.land_tiles.items()
        if coordinate == state.board.robber_coordinate
    )

    _set(tensor.scalars, SCALAR_NUM_PLAYERS, len(colors))
    _set(tensor.scalars, SCALAR_CURRENT_PLAYER, state.current_player_index)
    _set(tensor.scalars, SCALAR_CURRENT_TURN, state.current_turn_index)
    _set(tensor.scalars, SCALAR_PROMPT, PROMPT_TO_ID[state.current_prompt])
    _set(tensor.scalars, SCALAR_INITIAL_BUILD, int(state.is_initial_build_phase))
    _set(tensor.scalars, SCALAR_NUM_TURNS, state.num_turns)
    _set(tensor.scalars, SCALAR_VPS_TO_WIN, getattr(game, "vps_to_win", 10))
    _set(tensor.scalars, SCALAR_DISCARD_LIMIT, getattr(state, "discard_limit", 7))
    _set(tensor.scalars, SCALAR_ROBBER_TILE, robber_tile_id)

    for slot, color in enumerate(colors):
        key = player_key(state, color)
        _set2(tensor.player, slot, PLAYER_COLOR, COLOR_TO_ID[color])
        _set2(tensor.player, slot, PLAYER_VPS, state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"])
        _set2(tensor.player, slot, PLAYER_ROADS_LEFT, state.player_state[f"{key}_ROADS_AVAILABLE"])
        _set2(tensor.player, slot, PLAYER_SETTLEMENTS_LEFT, state.player_state[f"{key}_SETTLEMENTS_AVAILABLE"])
        _set2(tensor.player, slot, PLAYER_CITIES_LEFT, state.player_state[f"{key}_CITIES_AVAILABLE"])
        _set2(tensor.player, slot, PLAYER_HAS_ROLLED, int(state.player_state[f"{key}_HAS_ROLLED"]))
        for resource_offset, resource in enumerate(["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]):
            _set2(
                tensor.player,
                slot,
                PLAYER_RESOURCE_START + resource_offset,
                state.player_state[f"{key}_{resource}_IN_HAND"],
            )

    for index, count in enumerate(state.resource_freqdeck):
        _set(tensor.bank, index, count)

    for tile_id in range(NUM_TILES):
        tile = state.board.map.tiles_by_id[tile_id]
        _set2(tensor.tiles, tile_id, TILE_RESOURCE, RESOURCE_TO_ID[tile.resource])
        _set2(tensor.tiles, tile_id, TILE_NUMBER, tile.number or 0)
        _set2(tensor.tiles, tile_id, TILE_HAS_ROBBER, int(tile_id == robber_tile_id))

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

    _set(tensor.scalars, SCALAR_NUM_PLAYERS, len(colors))
    _set(tensor.scalars, SCALAR_CURRENT_PLAYER, snapshot["current_player_index"])
    _set(tensor.scalars, SCALAR_CURRENT_TURN, snapshot["current_turn_index"])
    _set(tensor.scalars, SCALAR_PROMPT, snapshot["current_prompt_id"])
    _set(tensor.scalars, SCALAR_INITIAL_BUILD, int(snapshot["is_initial_build_phase"]))
    _set(tensor.scalars, SCALAR_NUM_TURNS, snapshot["num_turns"])
    _set(tensor.scalars, SCALAR_VPS_TO_WIN, snapshot["vps_to_win"])
    _set(tensor.scalars, SCALAR_DISCARD_LIMIT, snapshot["discard_limit"])
    _set(tensor.scalars, SCALAR_ROBBER_TILE, snapshot["robber_tile_id"])

    for slot, color_id in enumerate(colors):
        _set2(tensor.player, slot, PLAYER_COLOR, color_id)
        _set2(tensor.player, slot, PLAYER_VPS, snapshot["victory_points"][slot])
        _set2(tensor.player, slot, PLAYER_ROADS_LEFT, snapshot["roads_available"][slot])
        _set2(tensor.player, slot, PLAYER_SETTLEMENTS_LEFT, snapshot["settlements_available"][slot])
        _set2(tensor.player, slot, PLAYER_CITIES_LEFT, snapshot["cities_available"][slot])
        _set2(tensor.player, slot, PLAYER_HAS_ROLLED, snapshot["has_rolled"][slot])
        for resource_offset, count in enumerate(snapshot["player_resources"][slot]):
            _set2(tensor.player, slot, PLAYER_RESOURCE_START + resource_offset, count)

    for index, count in enumerate(snapshot["resource_bank"]):
        _set(tensor.bank, index, count)

    for tile_id in range(NUM_TILES):
        _set2(tensor.tiles, tile_id, TILE_RESOURCE, snapshot["tile_resource"][tile_id])
        _set2(tensor.tiles, tile_id, TILE_NUMBER, snapshot["tile_number"][tile_id])
        _set2(tensor.tiles, tile_id, TILE_HAS_ROBBER, int(tile_id == snapshot["robber_tile_id"]))

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
        "tiles": _tolist(tensor.tiles),
        "bank": _tolist(tensor.bank),
    }


def _set(values, index: int, value: int) -> None:
    values[index] = value


def _set2(values, row: int, column: int, value: int) -> None:
    values[row][column] = value


def _tolist(values):
    if hasattr(values, "tolist"):
        return values.tolist()
    return values
