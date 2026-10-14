import pytest

from catanatron.game import Game
from catanatron.models.decks import (
    BASE_SUPPLY,
    EXTENSION_SUPPLY,
    Supply,
    starting_devcard_bank,
    starting_resource_bank,
)
from catanatron.models.enums import DEVELOPMENT_CARDS
from catanatron.models.player import Color, RandomPlayer
from catanatron.features import build_production_features
from catanatron.players.value import DEFAULT_WEIGHTS, base_fn, value_production
from catanatron.state_functions import player_key

SIX_COLORS = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE, Color.GREEN, Color.BROWN]


def test_base_supply_matches_the_starting_decks():
    assert BASE_SUPPLY.resource_freqdeck() == starting_resource_bank()
    assert BASE_SUPPLY.development_listdeck() == starting_devcard_bank()


def test_extension_supply_has_the_five_six_player_counts():
    assert sum(EXTENSION_SUPPLY.resource_bank) == 24 * 5
    deck = EXTENSION_SUPPLY.development_listdeck()
    assert len(deck) == 34
    assert [deck.count(card) for card in DEVELOPMENT_CARDS] == [20, 3, 3, 3, 5]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(resource_bank=(19, 19, 19, 19)),
        dict(development_deck=(14, 2, 2, 2, -1)),
        dict(settlements=1),
        dict(roads=1),
        dict(cities=-1),
    ],
)
def test_invalid_supplies_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Supply(**kwargs)


def test_a_game_starts_from_its_supply():
    supply = Supply(
        resource_bank=(24, 24, 24, 24, 24),
        development_deck=(20, 3, 3, 3, 5),
        roads=12,
        settlements=4,
        cities=3,
    )
    game = Game([RandomPlayer(color) for color in SIX_COLORS], seed=1, supply=supply)
    state = game.state
    assert state.resource_freqdeck == [24, 24, 24, 24, 24]
    assert sorted(state.development_listdeck) == sorted(supply.development_listdeck())
    for color in SIX_COLORS:
        key = player_key(state, color)
        assert state.player_state[f"{key}_ROADS_AVAILABLE"] == 12
        assert state.player_state[f"{key}_SETTLEMENTS_AVAILABLE"] == 4
        assert state.player_state[f"{key}_CITIES_AVAILABLE"] == 3


def test_base_supply_shuffles_like_before():
    """The default supply must not change any seeded game."""
    players = [RandomPlayer(color) for color in SIX_COLORS[:4]]
    default = Game(players, seed=5)
    explicit = Game(players, seed=5, supply=BASE_SUPPLY)
    assert default.state.development_listdeck == explicit.state.development_listdeck


def test_six_player_games_play_past_the_initial_build_phase():
    game = Game(
        [RandomPlayer(color) for color in SIX_COLORS], seed=3, supply=EXTENSION_SUPPLY
    )
    winner = game.play()
    assert len(game.state.colors) == 6
    assert not game.state.is_initial_build_phase
    if winner is not None:
        key = player_key(game.state, winner)
        assert game.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] >= game.vps_to_win


def _play_initial_phase(game):
    while game.state.is_initial_build_phase:
        game.execute(game.playable_actions[0])


def test_all_enemies_matches_next_enemy_with_two_players():
    game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=2)
    _play_initial_phase(game)
    for color in game.state.colors:
        assert base_fn(enemies="all")(game, color) == base_fn()(game, color)


def test_all_enemies_averages_every_opponent():
    game = Game([RandomPlayer(color) for color in SIX_COLORS[:3]], seed=2)
    _play_initial_phase(game)
    color = game.state.colors[0]
    sample = build_production_features(True)(game, color)
    next_enemy = value_production(sample, "P1", False)
    mean_enemy = (value_production(sample, "P1", False) + value_production(sample, "P2", False)) / 2
    difference = base_fn(enemies="all")(game, color) - base_fn()(game, color)
    assert difference == pytest.approx(
        DEFAULT_WEIGHTS["enemy_production"] * (mean_enemy - next_enemy), abs=1.0
    )


def test_unknown_enemy_scopes_are_rejected():
    with pytest.raises(ValueError):
        base_fn(enemies="nearest")
