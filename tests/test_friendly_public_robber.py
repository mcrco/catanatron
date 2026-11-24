"""OpenHex's friendly robber (``robber_mode="friendly_public"``): a tile with a
building of any seat at 2 public points or fewer is off limits, the mover's
own included; hidden victory point cards never count. With every tile off
limits the robber is not moved: a 7 goes on with the roller's turn after the
discards, a knight is played without a robber move. A knight's robber is
judged on the points after the knight's Largest Army award. Catanatron's own
friendly robber (``friendly_robber=True``) is unchanged."""

import pytest

from catanatron.game import Game
from catanatron.models.actions import generate_playable_actions, robber_possibilities
from catanatron.models.enums import SETTLEMENT, Action, ActionPrompt, ActionRecord, ActionType
from catanatron.models.player import Color, SimplePlayer
from catanatron.serialization import client_view, state_from_json, state_to_json
from catanatron.state_functions import build_city, player_deck_replenish, player_key
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


def roll_seven(game, prompt=ActionPrompt.MOVE_ROBBER):
    roll = Action(game.state.current_color(), ActionType.ROLL, None)
    game.execute(roll, action_record=ActionRecord(roll, (1, 6)))
    assert game.state.current_prompt == prompt


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


def make_city(game, color):
    """One of ``color``'s settlements becomes a city: 3 public points."""
    node = game.state.buildings_by_color[color][SETTLEMENT][0]
    game.state.board.build_city(color, node)
    build_city(game.state, color, node)


def cover_every_tile(game, color):
    """A building of ``color`` on every land tile (placed without the rules:
    its points stay what they were)."""
    board = game.state.board
    for tile in board.map.land_tiles.values():
        if not any(board.buildings.get(n, (None,))[0] == color for n in tile.nodes.values()):
            free = next(n for n in tile.nodes.values() if n not in board.buildings)
            board.buildings[free] = (color, SETTLEMENT)
    assert tiles_of(game, color) == set(board.map.land_tiles)
    game.playable_actions = generate_playable_actions(game.state)


def hold_army(game, color, knights=3):
    state, key = game.state.player_state, player_key(game.state, color)
    state[f"{key}_PLAYED_KNIGHT"] = knights
    state[f"{key}_HAS_ARMY"] = True
    state[f"{key}_VICTORY_POINTS"] += 2
    state[f"{key}_ACTUAL_VICTORY_POINTS"] += 2


def play_knight(game, color, played_before):
    """``color`` plays a knight, having played ``played_before`` already."""
    state, key = game.state.player_state, player_key(game.state, color)
    state[f"{key}_PLAYED_KNIGHT"] = played_before
    state[f"{key}_KNIGHT_IN_HAND"] = 1
    state[f"{key}_KNIGHT_OWNED_AT_START"] = True
    game.playable_actions = generate_playable_actions(game.state)
    game.execute(Action(color, ActionType.PLAY_KNIGHT_CARD, None))
    assert state[f"{key}_PLAYED_KNIGHT"] == played_before + 1
    assert state[f"{key}_KNIGHT_IN_HAND"] == 0
    assert state[f"{key}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"]


def public_points(game, color):
    return game.state.player_state[f"{player_key(game.state, color)}_VICTORY_POINTS"]


def kinds(game):
    return {a.action_type for a in game.playable_actions}


# ===== which tiles are shielded =====
def test_two_public_points_shield_a_seat_whatever_cards_it_hides():
    for hidden in (0, 1):
        games = {mode: two_players(mode) for mode in ("normal", "friendly", "friendly_public")}
        for game in games.values():
            for _ in range(hidden):
                give_victory_point_card(game, seats(game)[1])
            roll_seven(game)
        normal = games["normal"]
        mover, opponent = seats(normal)
        assert robber_tiles(normal) == all_tiles(normal)
        # Both seats are at 2 public points: the mover's own tiles are shielded too.
        assert tiles_of(normal, mover) - tiles_of(normal, opponent)
        assert robber_tiles(games["friendly_public"]) == (
            all_tiles(normal) - tiles_of(normal, opponent) - tiles_of(normal, mover)
        )
        # Catanatron's own rule shields opponents only, and counts the hidden
        # card: 3 actual points open the seat.
        assert robber_tiles(games["friendly"]) == (
            all_tiles(normal) if hidden else all_tiles(normal) - tiles_of(normal, opponent)
        )


def test_three_public_points_open_an_opponents_tiles():
    game = two_players("friendly_public")
    mover, opponent = seats(game)
    make_city(game, opponent)
    roll_seven(game)
    # The opponent's tiles are open, except where the mover (at 2) also sits.
    assert robber_tiles(game) == all_tiles(game) - tiles_of(game, mover)
    assert robber_tiles(game) & tiles_of(game, opponent)


def test_the_movers_own_buildings_shield_at_two_points_and_not_at_three():
    game = two_players("friendly_public")
    mover, opponent = seats(game)
    make_city(game, opponent)  # 3 points: the opponent shields nothing
    own_only = tiles_of(game, mover) - tiles_of(game, opponent) - {game.state.board.robber_coordinate}
    assert own_only  # tiles with only the mover's building
    assert public_points(game, mover) == 2
    assert not own_only & {a.value[0] for a in robber_possibilities(game.state, mover)}
    make_city(game, mover)
    assert public_points(game, mover) == 3
    roll_seven(game)
    assert own_only <= robber_tiles(game) and robber_tiles(game) == all_tiles(game)


def test_hidden_points_of_the_mover_do_not_open_its_tiles():
    game = two_players("friendly_public")
    mover, opponent = seats(game)
    make_city(game, opponent)
    give_victory_point_card(game, mover)  # 3 actual points, 2 public
    roll_seven(game)
    assert robber_tiles(game) == all_tiles(game) - tiles_of(game, mover)


# ===== no tile open: the robber is not moved =====
@pytest.mark.parametrize("coverer", ["opponent", "mover"])
def test_a_seven_with_every_tile_shielded_owes_no_robber_move(coverer):
    game = two_players("friendly_public")
    mover, opponent = seats(game)
    if coverer == "mover":
        make_city(game, opponent)  # the opponent shields nothing: the mover's buildings do it all
    cover_every_tile(game, opponent if coverer == "opponent" else mover)
    assert robber_possibilities(game.state, mover) == []
    robber, hands = game.state.board.robber_coordinate, dict(game.state.player_state)
    roll_seven(game, ActionPrompt.PLAY_TURN)
    # The roller's turn goes on as after a robber move: nothing moved, nobody robbed.
    assert game.state.current_color() == mover
    assert game.state.board.robber_coordinate == robber
    assert ActionType.MOVE_ROBBER not in kinds(game) and ActionType.ROLL not in kinds(game)
    assert ActionType.END_TURN in kinds(game)
    assert all(game.state.player_state[k] == v for k, v in hands.items() if k.endswith("_IN_HAND"))
    assert [r.action.action_type for r in game.state.action_records[-1:]] == [ActionType.ROLL]
    game.execute(Action(mover, ActionType.END_TURN, None))
    assert game.state.current_color() == opponent and ActionType.ROLL in kinds(game)


@pytest.mark.parametrize("discarders", ["opponent", "mover", "both"])
def test_a_seven_with_every_tile_shielded_still_takes_the_discards(discarders):
    game = two_players("friendly_public")
    mover, opponent = seats(game)
    cover_every_tile(game, opponent)
    robber = game.state.board.robber_coordinate
    over = [c for c, name in ((mover, "mover"), (opponent, "opponent")) if discarders in (name, "both")]
    for color in game.state.colors:
        key = player_key(game.state, color)
        for resource in ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"):
            game.state.player_state[f"{key}_{resource}_IN_HAND"] = 0
        player_deck_replenish(game.state, color, "WOOD", 9 if color in over else 7)
    roll_seven(game, ActionPrompt.DISCARD)
    discarded = []
    while game.state.current_prompt == ActionPrompt.DISCARD:
        assert kinds(game) == {ActionType.DISCARD_RESOURCE}
        discarded.append(game.state.current_color())
        game.execute(game.playable_actions[0])
    # 9 cards discard 4, 7 discard none; then no robber move: the roller plays on.
    assert sorted(discarded, key=game.state.colors.index) == sorted(
        [c for c in over for _ in range(4)], key=game.state.colors.index
    )
    for color in game.state.colors:
        wood = game.state.player_state[f"{player_key(game.state, color)}_WOOD_IN_HAND"]
        assert wood == (5 if color in over else 7)
    assert game.state.current_prompt == ActionPrompt.PLAY_TURN
    assert game.state.current_color() == mover and not game.state.is_discarding
    assert game.state.discard_counts == [0, 0]
    assert game.state.board.robber_coordinate == robber
    assert ActionType.MOVE_ROBBER not in kinds(game) and ActionType.END_TURN in kinds(game)


def test_a_seven_with_one_tile_open_asks_for_that_robber_move():
    game = two_players("friendly_public")
    mover, opponent = seats(game)
    board = game.state.board
    # A tile nobody has built on, and not the robber's: the one left open.
    coordinate, tile = next(
        (c, t)
        for c, t in board.map.land_tiles.items()
        if c != board.robber_coordinate and not any(n in board.buildings for n in t.nodes.values())
    )
    kept = set(tile.nodes.values())
    for other in board.map.land_tiles.values():
        if other is not tile and not any(
            board.buildings.get(n, (None,))[0] == opponent for n in other.nodes.values()
        ):
            free = next(n for n in other.nodes.values() if n not in board.buildings and n not in kept)
            board.buildings[free] = (opponent, SETTLEMENT)
    roll_seven(game)
    assert robber_tiles(game) == {coordinate}
    game.execute(game.playable_actions[0])
    assert board.robber_coordinate == coordinate and game.state.current_prompt == ActionPrompt.PLAY_TURN


def test_a_knight_with_every_tile_shielded_is_played_and_moves_no_robber():
    game = two_players("friendly_public")
    mover, opponent = seats(game)
    cover_every_tile(game, opponent)
    robber = game.state.board.robber_coordinate
    play_knight(game, mover, played_before=0)
    # No robber prompt: the turn goes on where it was, at the roll.
    assert game.state.current_prompt == ActionPrompt.PLAY_TURN
    assert game.state.current_color() == mover and game.state.board.robber_coordinate == robber
    assert kinds(game) == {ActionType.ROLL}  # the turn's development card is spent
    assert game.state.largest_army_taken_from is None
    assert game.state.action_records[-1].action.action_type == ActionType.PLAY_KNIGHT_CARD


def test_a_knight_with_every_tile_shielded_still_earns_largest_army():
    game = two_players("friendly_public")
    mover, opponent = seats(game)
    cover_every_tile(game, opponent)
    robber = game.state.board.robber_coordinate
    # Played after the roll, as the third knight: Largest Army is due.
    roll = Action(mover, ActionType.ROLL, None)
    game.execute(roll, action_record=ActionRecord(roll, (6, 6)))
    play_knight(game, mover, played_before=2)
    assert game.state.player_state[f"{player_key(game.state, mover)}_HAS_ARMY"]
    assert public_points(game, mover) == 4
    assert game.state.current_prompt == ActionPrompt.PLAY_TURN
    assert game.state.board.robber_coordinate == robber
    assert ActionType.MOVE_ROBBER not in kinds(game) and ActionType.END_TURN in kinds(game)
    assert ActionType.PLAY_KNIGHT_CARD not in kinds(game)


def test_a_knight_that_wins_the_game_with_every_tile_shielded_wins_it():
    game = Game([SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)], seed=1, robber_mode="friendly_public",
                vps_to_win=4)
    build_initial_placements(game)
    mover, opponent = seats(game)
    cover_every_tile(game, opponent)
    play_knight(game, mover, played_before=2)
    assert game.winning_color() == mover


# ===== a knight's robber is judged after its Largest Army award =====
def knight_that_takes_the_army(robber_mode):
    """The opponent holds Largest Army (3 knights) on two settlements: 4 public
    points. The mover plays its fourth knight before rolling and takes it."""
    game = two_players(robber_mode)
    mover, opponent = seats(game)
    hold_army(game, opponent)
    play_knight(game, mover, played_before=3)
    state = game.state.player_state
    assert state[f"{player_key(game.state, mover)}_HAS_ARMY"]
    assert public_points(game, opponent) == 2 and public_points(game, mover) == 4
    return game, mover, opponent


def test_a_knights_robber_is_judged_on_the_points_after_its_award():
    game, mover, opponent = knight_that_takes_the_army("friendly_public")
    assert game.state.current_prompt == ActionPrompt.MOVE_ROBBER
    assert game.state.largest_army_taken_from is None
    # After the award the opponent is at 2 (shielded) and the mover at 4 (its
    # own tiles open). Before it, it was the other way round.
    assert robber_tiles(game) == all_tiles(game) - tiles_of(game, opponent)
    assert robber_tiles(game) & tiles_of(game, mover)
    # A 7 at the same points is judged the same.
    game.execute(game.playable_actions[0])
    roll_seven(game)
    assert robber_tiles(game) == all_tiles(game) - tiles_of(game, opponent)


def test_a_knight_without_an_award_leaves_the_mover_shielding_its_own_tiles():
    game = two_players("friendly_public")
    mover, opponent = seats(game)
    play_knight(game, mover, played_before=0)
    assert public_points(game, mover) == 2
    assert game.state.current_prompt == ActionPrompt.MOVE_ROBBER
    assert robber_tiles(game) == all_tiles(game) - tiles_of(game, opponent) - tiles_of(game, mover)


def test_the_first_largest_army_opens_the_movers_own_tiles_for_that_knight():
    game = two_players("friendly_public")
    mover, opponent = seats(game)
    play_knight(game, mover, played_before=2)
    assert public_points(game, mover) == 4 and game.state.largest_army_taken_from is None
    assert robber_tiles(game) == all_tiles(game) - tiles_of(game, opponent)


def test_an_award_that_opens_the_only_tiles_brings_the_robber_prompt():
    # Every tile has a building of the mover (2 points): nothing is open before
    # the knight. Its Largest Army lifts the mover to 4, and every tile opens.
    game = two_players("friendly_public")
    mover, opponent = seats(game)
    make_city(game, opponent)
    cover_every_tile(game, mover)
    assert robber_possibilities(game.state, mover) == []
    play_knight(game, mover, played_before=2)
    assert game.state.current_prompt == ActionPrompt.MOVE_ROBBER
    assert robber_tiles(game) == all_tiles(game)


def test_an_award_that_shields_every_tile_leaves_no_robber_prompt():
    # The opponent holds Largest Army (4 points) and sits on every tile; the
    # mover (a city: 3 points) shields nothing. Every tile is open before the
    # knight; it takes the award, the opponent is back at 2, and none is.
    game = two_players("friendly_public")
    mover, opponent = seats(game)
    hold_army(game, opponent)
    make_city(game, mover)
    cover_every_tile(game, opponent)
    assert {a.value[0] for a in robber_possibilities(game.state, mover)} == all_tiles(game)
    robber = game.state.board.robber_coordinate
    play_knight(game, mover, played_before=3)
    assert public_points(game, opponent) == 2 and public_points(game, mover) == 5
    assert game.state.current_prompt == ActionPrompt.PLAY_TURN
    assert game.state.board.robber_coordinate == robber and kinds(game) == {ActionType.ROLL}


# ===== the field of the older rule =====
def test_largest_army_taken_from_still_loads_and_shields_nothing():
    game, mover, opponent = knight_that_takes_the_army("friendly_public")
    doc = state_to_json(game)
    assert doc["largest_army_taken_from"] is None
    assert client_view(doc, mover.value)["largest_army_taken_from"] is None
    # A document saved under the older rule, at such a knight's robber prompt.
    old = {**doc, "largest_army_taken_from": opponent.value}
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    rebuilt = state_from_json(old, players)
    assert rebuilt.state.largest_army_taken_from == opponent
    assert state_to_json(rebuilt) == old
    assert rebuilt.copy().state.largest_army_taken_from == opponent
    assert robber_tiles(rebuilt) == robber_tiles(game)  # no +2 for the seat
    rebuilt.execute(rebuilt.playable_actions[0])
    assert rebuilt.state.largest_army_taken_from is None


# ===== Catanatron's own friendly robber: as it was =====
def test_the_classic_friendly_robber_shields_opponents_only_and_awards_first():
    game, mover, opponent = knight_that_takes_the_army("friendly")
    assert game.state.current_prompt == ActionPrompt.MOVE_ROBBER
    assert game.state.largest_army_taken_from is None
    assert robber_tiles(game) == all_tiles(game) - tiles_of(game, opponent)
    # A mover at 2 points does not shield its own tiles.
    game = two_players("friendly")
    mover, opponent = seats(game)
    roll_seven(game)
    assert robber_tiles(game) == all_tiles(game) - tiles_of(game, opponent)
    assert robber_tiles(game) & tiles_of(game, mover)


@pytest.mark.parametrize("by", ["seven", "knight"])
def test_the_classic_friendly_robber_keeps_its_fallback(by):
    game = two_players("friendly")
    mover, opponent = seats(game)
    cover_every_tile(game, opponent)
    if by == "seven":
        roll_seven(game)
    else:
        play_knight(game, mover, played_before=0)
        assert game.state.current_prompt == ActionPrompt.MOVE_ROBBER
    # Every tile but the robber's shielded: all are open, as under the normal robber.
    assert robber_tiles(game) == all_tiles(game)
    actions = robber_possibilities(game.state, mover)
    game.state.robber_mode = "normal"
    assert actions == robber_possibilities(game.state, mover)


def test_the_normal_robber_always_moves():
    game = two_players("normal")
    mover, opponent = seats(game)
    cover_every_tile(game, opponent)
    roll_seven(game)
    assert robber_tiles(game) == all_tiles(game)


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
