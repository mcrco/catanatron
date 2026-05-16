import pytest

from catanatron.cpp import AVAILABLE

pytestmark = pytest.mark.skipif(
    not AVAILABLE,
    reason="C++ pybind extension is not built in this environment",
)

if AVAILABLE:
    from catanatron.cpp import Action, Color, Game, Player
    from catanatron.cpp.tensor_abi import (
        cpp_snapshot_to_tensor,
        python_game_to_tensor,
        tensor_to_debug_dict,
    )

from catanatron.game import Game as PythonGame
from catanatron.models.enums import Action as PythonAction
from catanatron.models.enums import ActionRecord as PythonActionRecord
from catanatron.models.enums import ActionType as PythonActionType
from catanatron.models.player import Color as PythonColor
from catanatron.models.player import SimplePlayer


def _python_initial_game():
    game = PythonGame([SimplePlayer(PythonColor.RED), SimplePlayer(PythonColor.BLUE)], seed=1)
    script = [
        PythonAction(PythonColor.RED, PythonActionType.BUILD_SETTLEMENT, 0),
        PythonAction(PythonColor.RED, PythonActionType.BUILD_ROAD, (0, 1)),
        PythonAction(PythonColor.BLUE, PythonActionType.BUILD_SETTLEMENT, 2),
        PythonAction(PythonColor.BLUE, PythonActionType.BUILD_ROAD, (2, 3)),
        PythonAction(PythonColor.BLUE, PythonActionType.BUILD_SETTLEMENT, 4),
        PythonAction(PythonColor.BLUE, PythonActionType.BUILD_ROAD, (3, 4)),
        PythonAction(PythonColor.RED, PythonActionType.BUILD_SETTLEMENT, 6),
        PythonAction(PythonColor.RED, PythonActionType.BUILD_ROAD, (6, 23)),
    ]
    for action in script:
        game.execute(action)
    return game


def _cpp_initial_game():
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
    return game


def test_cpp_engine_initial_playable_actions():
    game = Game([Player(Color.RED), Player(Color.BLUE)])

    actions = game.generate_playable_actions()

    assert len(actions) == 54
    assert actions[0].value0 == 0


def test_cpp_engine_dynamic_game_config_reaches_snapshot_and_tensor():
    game = Game([Player(Color.RED), Player(Color.BLUE)], vps_to_win=12, discard_limit=9)

    snapshot = game.snapshot()
    debug = tensor_to_debug_dict(cpp_snapshot_to_tensor(snapshot))

    assert snapshot["vps_to_win"] == 12
    assert snapshot["discard_limit"] == 9
    assert debug["scalars"][6] == 12
    assert debug["scalars"][7] == 9


def test_cpp_engine_scripted_initial_build_phase_snapshot():
    game = _cpp_initial_game()

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
    assert debug["bank"] == [17, 19, 18, 19, 16]
    assert debug["player"][0][6:11] == [1, 0, 1, 0, 1]
    assert debug["player"][1][6:11] == [1, 0, 0, 0, 2]


def test_cpp_engine_initial_tensor_matches_python_reference():
    python_tensor = python_game_to_tensor(_python_initial_game())
    cpp_tensor = cpp_snapshot_to_tensor(_cpp_initial_game().snapshot())

    assert tensor_to_debug_dict(cpp_tensor) == tensor_to_debug_dict(python_tensor)


def test_cpp_engine_explicit_roll_resource_payout_matches_python_reference():
    python_game = _python_initial_game()
    cpp_game = _cpp_initial_game()

    python_roll = PythonAction(PythonColor.RED, PythonActionType.ROLL, None)
    python_record = PythonActionRecord(python_roll, (3, 3))
    python_game.execute(python_roll, action_record=python_record)
    cpp_game.execute(Action.roll(Color.RED, 3, 3))

    assert tensor_to_debug_dict(cpp_snapshot_to_tensor(cpp_game.snapshot())) == tensor_to_debug_dict(
        python_game_to_tensor(python_game)
    )


def test_cpp_engine_paid_builds_and_turn_cleanup():
    game = _cpp_initial_game()
    game.execute(Action.roll(Color.RED, 3, 3))
    game.set_player_resources(Color.RED, [5, 5, 5, 5, 5])

    game.execute(Action.build_road(Color.RED, 0, 5))
    game.execute(Action.build_road(Color.RED, 5, 16))
    game.execute(Action.build_settlement(Color.RED, 16))
    game.execute(Action.build_city(Color.RED, 0))
    game.execute(Action.end_turn(Color.RED))

    snapshot = game.snapshot()

    assert snapshot["current_player_index"] == 1
    assert snapshot["has_rolled"][0] == 0
    assert snapshot["roads_available"][0] == 11
    assert snapshot["settlements_available"][0] == 3
    assert snapshot["cities_available"][0] == 3
    assert snapshot["victory_points"][0] == 4
    assert snapshot["player_resources"][0] == [2, 2, 4, 2, 2]
    assert snapshot["resource_bank"] == [20, 22, 19, 22, 18]


def test_cpp_engine_rejects_normal_build_before_roll():
    game = _cpp_initial_game()
    game.set_player_resources(Color.RED, [5, 5, 5, 5, 5])

    with pytest.raises(ValueError):
        game.execute(Action.build_road(Color.RED, 0, 5))
