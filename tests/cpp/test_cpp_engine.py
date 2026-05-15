import pytest

from catanatron.cpp import AVAILABLE

pytestmark = pytest.mark.skipif(
    not AVAILABLE,
    reason="C++ pybind extension is not built in this environment",
)

if AVAILABLE:
    from catanatron.cpp import Action, Color, Game, Player
    from catanatron.cpp.tensor_abi import cpp_snapshot_to_tensor, tensor_to_debug_dict


def test_cpp_engine_initial_playable_actions():
    game = Game([Player(Color.RED), Player(Color.BLUE)])

    actions = game.generate_playable_actions()

    assert len(actions) == 54
    assert actions[0].value0 == 0


def test_cpp_engine_scripted_initial_build_phase_snapshot():
    game = Game([Player(Color.RED), Player(Color.BLUE)])
    script = [
        Action.build_settlement(Color.RED, 0),
        Action.build_road(Color.RED, 0, 1),
        Action.build_settlement(Color.BLUE, 2),
        Action.build_road(Color.BLUE, 2, 3),
        Action.build_settlement(Color.BLUE, 4),
        Action.build_road(Color.BLUE, 3, 4),
        Action.build_settlement(Color.RED, 6),
        Action.build_road(Color.RED, 6, 23),
    ]

    for action in script:
        game.execute(action)

    snapshot = game.snapshot()
    tensor = cpp_snapshot_to_tensor(snapshot)
    debug = tensor_to_debug_dict(tensor)

    assert snapshot["current_prompt"] == "PLAY_TURN"
    assert snapshot["current_player_index"] == 0
    assert snapshot["is_initial_build_phase"] is False
    assert debug["occupied_nodes"] == [
        (0, 0, 1),
        (2, 1, 1),
        (4, 1, 1),
        (6, 0, 1),
    ]
    assert debug["occupied_edges"] == [(0, 0), (5, 1), (7, 1), (13, 0)]
