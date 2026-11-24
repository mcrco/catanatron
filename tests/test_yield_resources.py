from catanatron.apply_action import yield_resources
from catanatron.game import Game
from catanatron.models.board import Board
from catanatron.models.enums import CITY, RESOURCES, Action, ActionRecord, ActionType
from catanatron.models.player import Color, SimplePlayer
from catanatron.models.decks import (
    freqdeck_count,
    freqdeck_draw,
    starting_resource_bank,
)
from tests.utils import build_initial_placements


def test_yield_resources():
    board = Board()
    resource_freqdeck = starting_resource_bank()

    tile = board.map.land_tiles[(0, 0, 0)]
    if tile.resource is None:  # is desert
        tile = board.map.land_tiles[(-1, 0, 1)]

    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    payout, depleted = yield_resources(board, resource_freqdeck, tile.number)
    assert len(depleted) == 0
    assert freqdeck_count(payout[Color.RED], tile.resource) >= 1  # type: ignore


def test_yield_resources_two_settlements():
    board = Board()
    resource_freqdeck = starting_resource_bank()

    tile, edge2, node2 = board.map.land_tiles[(0, 0, 0)], (4, 5), 5
    if tile.resource is None:  # is desert
        tile, edge2, node2 = board.map.land_tiles[(-1, 0, 1)], (4, 15), 15

    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    board.build_road(Color.RED, (3, 4))
    board.build_road(Color.RED, edge2)
    board.build_settlement(Color.RED, node2)
    payout, depleted = yield_resources(board, resource_freqdeck, tile.number)
    assert len(depleted) == 0
    assert freqdeck_count(payout[Color.RED], tile.resource) >= 2  # type: ignore


def test_yield_resources_two_players_and_city():
    board = Board()
    resource_freqdeck = starting_resource_bank()

    tile, edge1, edge2, red_node, blue_node = (
        board.map.land_tiles[(0, 0, 0)],
        (2, 3),
        (3, 4),
        4,
        0,
    )
    if tile.resource is None:  # is desert
        tile, edge1, edge2, red_node, blue_node = (
            board.map.land_tiles[(1, -1, 0)],
            (9, 2),
            (9, 8),
            8,
            6,
        )

    # red has one settlements and one city on tile
    board.build_settlement(Color.RED, 2, initial_build_phase=True)
    board.build_road(Color.RED, edge1)
    board.build_road(Color.RED, edge2)
    board.build_settlement(Color.RED, red_node)
    board.build_city(Color.RED, red_node)

    # blue has a city in tile
    board.build_settlement(Color.BLUE, blue_node, initial_build_phase=True)
    board.build_city(Color.BLUE, blue_node)
    payout, depleted = yield_resources(board, resource_freqdeck, tile.number)
    assert len(depleted) == 0
    assert freqdeck_count(payout[Color.RED], tile.resource) >= 3  # type: ignore
    assert freqdeck_count(payout[Color.BLUE], tile.resource) >= 2  # type: ignore


# ===== a short bank =====
def one_tile_board():
    """A board, a tile with a resource, and its six nodes in order around it.
    No other tile yields on the tile's number."""
    board = Board()
    tile = board.map.land_tiles[(0, 0, 0)]
    if tile.resource is None:  # is desert
        tile = board.map.land_tiles[(-1, 0, 1)]
    for other in board.map.land_tiles.values():
        if other is not tile and other.number == tile.number:
            other.number = 2 if tile.number != 2 else 3
    return board, tile, list(tile.nodes.values())


def bank_with(resource, count):
    bank = starting_resource_bank()
    freqdeck_draw(bank, freqdeck_count(bank, resource) - count, resource)
    assert freqdeck_count(bank, resource) == count
    return bank


def test_a_sole_claimant_takes_what_a_short_bank_has_left():
    board, tile, nodes = one_tile_board()
    board.build_settlement(Color.RED, nodes[0], initial_build_phase=True)
    board.build_city(Color.RED, nodes[0])
    bank = bank_with(tile.resource, 1)

    payout, depleted = yield_resources(board, bank, tile.number)
    # The city claims 2, the bank has 1: not paid in full, but RED takes the 1.
    assert depleted == [tile.resource]
    assert freqdeck_count(payout[Color.RED], tile.resource) == 1
    assert sum(payout[Color.RED]) == 1


def test_a_sole_claimant_gets_nothing_from_an_empty_bank():
    board, tile, nodes = one_tile_board()
    board.build_settlement(Color.RED, nodes[0], initial_build_phase=True)
    payout, depleted = yield_resources(board, bank_with(tile.resource, 0), tile.number)
    assert depleted == [tile.resource]
    assert sum(payout.get(Color.RED, [0])) == 0


def test_two_claimants_of_a_short_bank_get_nothing():
    board, tile, nodes = one_tile_board()
    board.build_settlement(Color.RED, nodes[0], initial_build_phase=True)
    board.build_settlement(Color.BLUE, nodes[2], initial_build_phase=True)
    board.build_city(Color.BLUE, nodes[2])
    # 3 claimed, 2 in the bank: neither is paid, not even in part.
    payout, depleted = yield_resources(board, bank_with(tile.resource, 2), tile.number)
    assert depleted == [tile.resource]
    assert sum(payout.get(Color.RED, [0])) == 0 and sum(payout.get(Color.BLUE, [0])) == 0
    # With 3 in the bank both are paid in full.
    payout, depleted = yield_resources(board, bank_with(tile.resource, 3), tile.number)
    assert depleted == []
    assert freqdeck_count(payout[Color.RED], tile.resource) == 1
    assert freqdeck_count(payout[Color.BLUE], tile.resource) == 2


def test_one_player_with_two_buildings_on_the_number_is_one_claimant():
    board, tile, nodes = one_tile_board()
    board.build_settlement(Color.RED, nodes[0], initial_build_phase=True)
    board.build_settlement(Color.RED, nodes[2], initial_build_phase=True)
    board.build_city(Color.RED, nodes[2])
    # 3 claimed by RED alone, 2 in the bank: RED takes the 2.
    payout, depleted = yield_resources(board, bank_with(tile.resource, 2), tile.number)
    assert depleted == [tile.resource]
    assert freqdeck_count(payout[Color.RED], tile.resource) == 2


def test_each_resource_is_judged_on_its_own():
    board, tile, nodes = one_tile_board()
    # A second tile on the same number, of another resource, with two claimants.
    other = next(
        t
        for t in board.map.land_tiles.values()
        if t.resource not in (None, tile.resource) and not set(t.nodes.values()) & set(nodes)
    )
    other.number = tile.number
    far = list(other.nodes.values())
    board.build_settlement(Color.RED, nodes[0], initial_build_phase=True)
    board.build_settlement(Color.RED, far[0], initial_build_phase=True)
    board.build_settlement(Color.BLUE, far[2], initial_build_phase=True)
    bank = bank_with(tile.resource, 0)
    payout, depleted = yield_resources(board, bank, tile.number)
    # The empty resource pays nobody; the other, in stock, pays both.
    assert depleted == [tile.resource]
    assert freqdeck_count(payout[Color.RED], tile.resource) == 0
    assert freqdeck_count(payout[Color.RED], other.resource) == 1
    assert freqdeck_count(payout[Color.BLUE], other.resource) == 1


def test_a_roll_pays_a_sole_claimant_the_banks_remainder():
    game = Game([SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)], seed=1)
    build_initial_placements(game)
    state, board = game.state, game.state.board
    roller = state.current_color()
    # A tile only one seat has built on, not under the robber.
    tile, owner = next(
        (t, owners.pop())
        for c, t in board.map.land_tiles.items()
        if t.resource is not None
        and c != board.robber_coordinate
        and len(owners := {board.buildings[n][0] for n in t.nodes.values() if n in board.buildings}) == 1
    )
    for other in board.map.land_tiles.values():
        if other is not tile and other.number == tile.number:
            other.number = 2 if tile.number != 2 else 3
    node = next(n for n in tile.nodes.values() if n in board.buildings)
    board.buildings[node] = (owner, CITY)  # claims 2 (with any second building, more)
    index = RESOURCES.index(tile.resource)
    state.resource_freqdeck[index] = 1
    key = f"P{state.colors.index(owner)}_{tile.resource}_IN_HAND"
    before = state.player_state[key]
    dice = next((a, tile.number - a) for a in range(1, 7) if 1 <= tile.number - a <= 6)
    roll = Action(roller, ActionType.ROLL, None)
    game.execute(roll, action_record=ActionRecord(roll, dice))
    assert state.player_state[key] == before + 1
    assert state.resource_freqdeck[index] == 0
