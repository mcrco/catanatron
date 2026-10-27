"""OpenHex's friendly robber (``robber_mode="friendly_public"``): a tile with an
opponent of the mover at 2 public points or fewer is off limits unless every
tile is; hidden victory point cards never count, and a knight's tile is
judged on the points before the knight's Largest Army award. Catanatron's
own friendly robber (``friendly_robber=True``) is unchanged."""

import pytest

from catanatron.game import Game
from catanatron.models.actions import generate_playable_actions, robber_possibilities
from catanatron.models.enums import SETTLEMENT, Action, ActionPrompt, ActionRecord, ActionType
from catanatron.models.player import Color, SimplePlayer
from catanatron.serialization import client_view, state_from_json, state_to_json
from catanatron.state_functions import build_city, player_key
from tests.utils import build_initial_placements


def two_players(robber_mode):
    """Two settlements each (2 public points), RED or BLUE to roll first."""
    game = Game([SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)], seed=1, robber_mode=robber_mode)
    build_initial_placements(game)
    return game


def seats(game):
    mover = game.state.current_color()
    return mover, next(c for c in game.state.colors if c != mover)


def tiles_of(game, color):
    """The land tiles ``color`` has a building on."""
    board = game.state.board
    return {
        coordinate
        for coordinate, tile in board.map.land_tiles.items()
        if any(board.buildings.get(n, (None,))[0] == color for n in tile.nodes.values())
    }


def roll_seven(game):
    roll = Action(game.state.current_color(), ActionType.ROLL, None)
    game.execute(roll, action_record=ActionRecord(roll, (1, 6)))
    assert game.state.current_prompt == ActionPrompt.MOVE_ROBBER


def robber_tiles(game):
    return {a.value[0] for a in game.playable_actions if a.action_type == ActionType.MOVE_ROBBER}


def all_tiles(game):
    return set(game.state.board.map.land_tiles) - {game.state.board.robber_coordinate}


def give_victory_point_card(game, color):
    key = player_key(game.state, color)
    game.state.player_state[f"{key}_VICTORY_POINT_IN_HAND"] += 1
    game.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] += 1


def test_modes():
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    assert Game(players).state.robber_mode == "normal"
    assert Game(players, friendly_robber=True).state.robber_mode == "friendly"
    game = Game(players, robber_mode="friendly_public")
    assert game.state.robber_mode == "friendly_public" and game.friendly_robber
    with pytest.raises(ValueError):
        Game(players, robber_mode="openhex")
    with pytest.raises(ValueError):
        Game(players, friendly_robber=True, robber_mode="normal")


def test_two_public_points_shield_a_seat_whatever_cards_it_hides():
    for hidden in (0, 1):
        games = {mode: two_players(mode) for mode in ("normal", "friendly", "friendly_public")}
        for game in games.values():
            for _ in range(hidden):
                give_victory_point_card(game, seats(game)[1])
            roll_seven(game)
        normal = games["normal"]
        opponent = seats(normal)[1]
        assert robber_tiles(normal) == all_tiles(normal)
        shielded = all_tiles(normal) - tiles_of(normal, opponent)
        assert robber_tiles(games["friendly_public"]) == shielded
        # Catanatron's own rule counts the hidden card: 3 actual points open the seat.
        assert robber_tiles(games["friendly"]) == (all_tiles(normal) if hidden else shielded)


def test_three_public_points_open_a_seats_tiles():
    game = two_players("friendly_public")
    opponent = seats(game)[1]
    node = game.state.buildings_by_color[opponent][SETTLEMENT][0]
    game.state.board.build_city(opponent, node)
    build_city(game.state, opponent, node)
    roll_seven(game)
    assert robber_tiles(game) == all_tiles(game)


def test_every_tile_shielded_opens_them_all():
    game = two_players("friendly_public")
    mover, opponent = seats(game)
    roll_seven(game)
    board = game.state.board
    # An opponent building on every tile (placed without the rules; its points stay 2).
    for tile in board.map.land_tiles.values():
        if not any(board.buildings.get(n, (None,))[0] == opponent for n in tile.nodes.values()):
            free = next(n for n in tile.nodes.values() if n not in board.buildings)
            board.buildings[free] = (opponent, SETTLEMENT)
    actions = robber_possibilities(game.state, mover)
    assert {a.value[0] for a in actions} == all_tiles(game)
    game.state.robber_mode = "normal"
    assert actions == robber_possibilities(game.state, mover)


def knight_that_takes_the_army(robber_mode):
    """The opponent holds Largest Army (3 knights) on two settlements: 4 public
    points. The mover plays its fourth knight before rolling and takes it."""
    game = two_players(robber_mode)
    mover, opponent = seats(game)
    state = game.state.player_state
    them, us = player_key(game.state, opponent), player_key(game.state, mover)
    state[f"{them}_PLAYED_KNIGHT"] = 3
    state[f"{them}_HAS_ARMY"] = True
    state[f"{them}_VICTORY_POINTS"] += 2
    state[f"{them}_ACTUAL_VICTORY_POINTS"] += 2
    state[f"{us}_PLAYED_KNIGHT"] = 3
    state[f"{us}_KNIGHT_IN_HAND"] = 1
    state[f"{us}_KNIGHT_OWNED_AT_START"] = True
    game.playable_actions = generate_playable_actions(game.state)
    game.execute(Action(mover, ActionType.PLAY_KNIGHT_CARD, None))
    assert state[f"{us}_HAS_ARMY"] and state[f"{them}_VICTORY_POINTS"] == 2
    return game, mover, opponent


def test_a_knights_tile_is_judged_on_the_points_before_its_award():
    game, mover, opponent = knight_that_takes_the_army("friendly_public")
    assert game.state.largest_army_taken_from == opponent
    # 2 public points now, 4 before the award: the knight may rob the seat it beat.
    assert robber_tiles(game) == all_tiles(game)
    # Catanatron's own rule awards first, and shields the seat.
    friendly, _, _ = knight_that_takes_the_army("friendly")
    assert friendly.state.largest_army_taken_from is None
    assert robber_tiles(friendly) == all_tiles(friendly) - tiles_of(friendly, opponent)

    # The award's seat is part of the position: documents and views carry it.
    doc = state_to_json(game)
    assert doc["largest_army_taken_from"] == opponent.value
    assert client_view(doc, mover.value)["largest_army_taken_from"] == opponent.value
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    rebuilt = state_from_json(doc, players)
    assert robber_tiles(rebuilt) == robber_tiles(game)
    assert state_to_json(rebuilt) == doc
    assert game.copy().state.largest_army_taken_from == opponent

    # Once the robber moved, the seat counts its 2 points again.
    game.execute(next(a for a in game.playable_actions if a.value[0] in tiles_of(game, opponent)))
    assert game.state.largest_army_taken_from is None
    roll_seven(game)
    assert robber_tiles(game) == all_tiles(game) - tiles_of(game, opponent)


def test_a_knight_that_takes_no_award_from_anyone_marks_nothing():
    game = two_players("friendly_public")
    mover, opponent = seats(game)
    key = player_key(game.state, mover)
    game.state.player_state[f"{key}_PLAYED_KNIGHT"] = 2
    game.state.player_state[f"{key}_KNIGHT_IN_HAND"] = 1
    game.state.player_state[f"{key}_KNIGHT_OWNED_AT_START"] = True
    game.playable_actions = generate_playable_actions(game.state)
    game.execute(Action(mover, ActionType.PLAY_KNIGHT_CARD, None))
    assert game.state.player_state[f"{key}_HAS_ARMY"]  # the first army: nobody lost it
    assert game.state.largest_army_taken_from is None
    assert robber_tiles(game) == all_tiles(game) - tiles_of(game, opponent)


def test_documents_of_other_robbers_carry_no_new_fields():
    for mode in ("normal", "friendly"):
        doc = state_to_json(two_players(mode))
        assert "robber_mode" not in doc["game"] and "largest_army_taken_from" not in doc
        assert doc["game"]["friendly_robber"] is (mode == "friendly")
        rebuilt = state_from_json(doc, [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)])
        assert rebuilt.state.robber_mode == mode
    doc = state_to_json(two_players("friendly_public"))
    assert doc["game"]["robber_mode"] == "friendly_public" and doc["game"]["friendly_robber"] is True
    assert doc["largest_army_taken_from"] is None
