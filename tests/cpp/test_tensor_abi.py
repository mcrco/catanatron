from catanatron.cpp.tensor_abi import (
    NO_OWNER,
    PROMPT_TO_ID,
    STATIC_EDGES,
    python_game_to_tensor,
    tensor_to_debug_dict,
)
from catanatron.game import Game
from catanatron.models.enums import ActionPrompt, SETTLEMENT
from catanatron.models.player import Color, SimplePlayer


def test_python_game_to_tensor_initial_state():
    game = Game([SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)], seed=1)

    tensor = python_game_to_tensor(game)
    debug = tensor_to_debug_dict(tensor)

    assert debug["scalars"][0] == 2
    assert debug["scalars"][1] == 0
    assert debug["scalars"][3] == PROMPT_TO_ID[ActionPrompt.BUILD_INITIAL_SETTLEMENT]
    assert debug["player"][0][:5] == [0, 0, 15, 5, 4]
    assert debug["player"][1][:5] == [1, 0, 15, 5, 4]
    assert debug["occupied_nodes"] == []
    assert debug["occupied_edges"] == []


def test_python_game_to_tensor_after_initial_action():
    game = Game([SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)], seed=1)
    game.play_tick()

    tensor = python_game_to_tensor(game)
    debug = tensor_to_debug_dict(tensor)

    assert debug["scalars"][3] == PROMPT_TO_ID[ActionPrompt.BUILD_INITIAL_ROAD]
    assert debug["player"][0][:5] == [0, 1, 15, 4, 4]
    assert debug["occupied_nodes"] == [(0, 0, 1)]
    assert all(owner == NO_OWNER for _, owner in enumerate(tensor.edges))


def test_static_edges_match_base_board_contract():
    assert len(STATIC_EDGES) == 72
    assert STATIC_EDGES[0] == (0, 1)
    assert STATIC_EDGES[-1] == (52, 53)
