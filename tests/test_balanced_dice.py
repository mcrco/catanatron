"""OpenHex's balanced dice (``dice_mode="balanced"``, catanatron.models.dice):
weights on hand-computed states, the deck's refills, draws that follow the
odds, and games that roll them. Fair-dice games roll as they always have."""

import hashlib
import json
import random
from collections import Counter

import pytest

from catanatron.game import Game
from catanatron.models.dice import CARDS, FULL_DECK, BalancedDice, card_bit
from catanatron.models.enums import Action, ActionRecord, ActionType
from catanatron.models.player import Color, RandomPlayer, SimplePlayer
from catanatron.serialization import client_view, state_from_json, state_to_json
from tests.utils import build_initial_placements

#: Cards per total 2..12 in a full deck.
FULL_COUNTS = [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]


def deck_of(cards):
    return sum(1 << card_bit(card) for card in cards)


# ===== the weights =====
def test_a_fresh_deck_weighs_each_total_by_its_cards():
    dice = BalancedDice.fresh(2)
    assert dice.remaining == 36 and dice.counts() == FULL_COUNTS
    # No balancing yet: the 7's multiplier is 5/5, every total weighs c_k * 50 * 5.
    assert dice.seven_multiplier(0) == (5, 5)
    assert dice.sum_weights(0) == [c * 50 * 5 for c in FULL_COUNTS]
    assert dice.sum_weights(None) == [c * 50 for c in FULL_COUNTS]
    assert dice.sum_probabilities(1)[5] == pytest.approx(6 / 36)


def test_a_seven_starts_a_streak_and_joins_the_window():
    dice = BalancedDice.fresh(2)
    dice.apply((3, 4), 0)
    assert dice.remaining == 35 and dice.recent == [7]
    assert dice.sevens == [1, 0] and (dice.streak_player, dice.streak_count) == (0, 1)
    # Fewer than n=2 sevens: imbalance 1; streak -0.4 for its owner, +0.4 for the other.
    assert dice.seven_multiplier(0) == (3, 5)
    assert dice.seven_multiplier(1) == (7, 5)
    # The 7 has 5 cards left and appears once among the last totals: 5 * (50 - 17).
    assert dice.sum_weights(0)[5] == 5 * 33 * 3
    assert dice.sum_weights(1)[5] == 5 * 33 * 7
    assert dice.sum_weights(1)[6] == 5 * 50 * 5  # an 8: untouched
    dice.apply((1, 1), 1)
    assert (dice.streak_player, dice.streak_count) == (None, 0)
    assert dice.recent == [7, 2]


def test_the_imbalance_term_and_the_clamp():
    # 4 sevens, 3 of them seat 0's: 1 + (1/2 - 3/4) * 2 = 1/2 for seat 0, 3/2 for seat 1.
    dice = BalancedDice(2, sevens=[3, 1])
    assert dice.seven_multiplier(0) == (10, 20)
    assert dice.seven_multiplier(1) == (30, 20)
    # Seat 1 rolled all four, the last four in a row: 2 + 1.6 clamps to 2, 0 - 1.6 to 0.
    dice = BalancedDice(2, sevens=[0, 4], streak_player=1, streak_count=4)
    assert dice.seven_multiplier(0) == (40, 20)
    assert dice.seven_multiplier(1) == (0, 20)
    assert dice.sum_weights(1)[5] == 0 and dice.sum_probabilities(1)[5] == 0
    # One seat: no seven balancing.
    assert BalancedDice(1, sevens=[3]).seven_multiplier(0) == (1, 1)


def test_three_recent_appearances_zero_a_total():
    dice = BalancedDice(2, recent=[8, 8, 8, 5, 6])
    weights = dice.sum_weights(0)
    assert weights[6] == 0
    assert weights[3] == 4 * 33 * 5 and weights[4] == 5 * 33 * 5


def test_the_window_holds_the_last_five_totals():
    dice = BalancedDice.fresh(2)
    for card in [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 6)]:
        dice.apply(card, 0)
    assert dice.recent == [3, 4, 5, 6, 8]


# ===== the deck =====
def test_the_deck_refills_before_a_draw_at_13_cards():
    left = CARDS[:13]
    dice = BalancedDice(2, deck=deck_of(left), recent=[6], sevens=[2, 1], streak_player=1, streak_count=1)
    assert dice.remaining == 13 and dice.drawable() == FULL_DECK
    assert dice.counts() == FULL_COUNTS
    assert dice.can_draw((6, 6))  # long drawn, but the deck refills first
    dice.apply((6, 6), 0)
    # Only the deck reset: the window, the sevens and the streak carried over.
    assert dice.remaining == 35 and not dice.can_draw((6, 6))
    assert dice.recent == [6, 12] and dice.sevens == [2, 1] and dice.streak_count == 0


def test_a_card_no_longer_in_the_deck_is_refused_and_changes_nothing():
    dice = BalancedDice(2, deck=deck_of(CARDS[:14]))
    assert dice.drawable() == dice.deck  # 14 cards: no refill
    before = dice.copy()
    with pytest.raises(ValueError, match="not in the balanced deck"):
        dice.apply((6, 6), 0)
    with pytest.raises(ValueError):
        dice.apply((0, 7), 0)
    with pytest.raises(ValueError):
        dice.apply((1, 1), 2)  # not a seat
    assert dice == before


def test_draws_follow_the_weights_and_are_uniform_within_a_total():
    dice = BalancedDice(2, deck=deck_of(CARDS[5:30]), recent=[7, 6], sevens=[1, 0], streak_player=0,
                        streak_count=1)
    rng = random.Random(0)
    draws = Counter(dice.draw(rng, 0) for _ in range(40_000))
    expected = dice.sum_probabilities(0)
    totals = Counter()
    for card, count in draws.items():
        assert dice.can_draw(card)
        totals[sum(card)] += count
    for index, p in enumerate(expected):
        assert abs(totals[index + 2] / 40_000 - p) < 0.01
    sixes = [draws[card] for card in CARDS if sum(card) == 6 and dice.drawable() >> card_bit(card) & 1]
    assert max(sixes) - min(sixes) < 0.25 * max(sixes)


def test_json_round_trips_and_names_the_streak_by_color():
    dice = BalancedDice(2, deck=deck_of(CARDS[3:]), recent=[7, 7], sevens=[0, 2], streak_player=1, streak_count=2)
    doc = dice.to_json(["RED", "BLUE"])
    assert doc["streak_player"] == "BLUE" and len(doc["deck"]) == 33
    assert BalancedDice.from_json(json.loads(json.dumps(doc)), ["RED", "BLUE"]) == dice
    with pytest.raises(ValueError):
        BalancedDice.from_json({**doc, "deck": doc["deck"] + [doc["deck"][0]]}, ["RED", "BLUE"])


def test_states_no_game_reaches_are_refused():
    with pytest.raises(ValueError):
        BalancedDice(2, recent=[1])
    with pytest.raises(ValueError):
        BalancedDice(2, sevens=[1])
    with pytest.raises(ValueError):
        BalancedDice(2, streak_player=0, streak_count=0)
    with pytest.raises(ValueError):
        BalancedDice(2, recent=[7] * 6)


# ===== in a game =====
def balanced_game(seed=1):
    players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
    return Game(players, seed=seed, dice_mode="balanced")


def test_a_balanced_game_rolls_from_its_deck():
    game = balanced_game()
    assert game.state.balanced_dice == BalancedDice.fresh(2)
    build_initial_placements(game)
    seen = []
    while len(seen) < 30:
        kinds = {a.action_type: a for a in game.playable_actions}
        if ActionType.ROLL in kinds:
            color = game.state.current_color()
            before = game.state.balanced_dice.copy()
            record = game.execute(kinds[ActionType.ROLL])
            assert before.can_draw(record.result)
            expected = before.copy()
            expected.apply(record.result, game.state.colors.index(color))
            assert game.state.balanced_dice == expected
            seen.append(record.result)
        else:  # end the turn, or discard / move the robber first
            game.execute(kinds.get(ActionType.END_TURN, game.playable_actions[0]))
    # A deck yields 23 draws (36 down to 13) before it refills: no card repeats among them.
    assert len(set(seen[:23])) == 23


def test_the_deck_refills_when_the_main_phase_starts():
    game = balanced_game()
    # A dice state declared during the initial placement (as a loaded one can be).
    game.state.balanced_dice = BalancedDice(2, deck=deck_of(CARDS[:20]), recent=[5], sevens=[1, 0])
    build_initial_placements(game)
    assert game.state.balanced_dice == BalancedDice(2, recent=[5], sevens=[1, 0])


def test_a_replayed_roll_the_deck_cannot_give_is_refused():
    game = balanced_game()
    build_initial_placements(game)
    color = game.state.current_color()
    roll = Action(color, ActionType.ROLL, None)
    game.state.balanced_dice.deck = deck_of(CARDS[1:])  # (1, 1) is gone
    with pytest.raises(ValueError, match="not in the balanced deck"):
        game.execute(roll, action_record=ActionRecord(roll, (1, 1)))
    assert not game.state.player_state[f"P{game.state.colors.index(color)}_HAS_ROLLED"]
    record = game.execute(roll, action_record=ActionRecord(roll, (2, 5)))
    assert record.result == (2, 5) and game.state.balanced_dice.sevens[game.state.colors.index(color)] == 1


def test_documents_carry_the_dice_and_copies_do_not_share_them():
    game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=3, dice_mode="balanced")
    for _ in range(200):
        game.play_tick()
    doc = state_to_json(game)
    assert doc["game"]["dice_mode"] == "balanced"
    assert BalancedDice.from_json(doc["balanced_dice"], doc["colors"]) == game.state.balanced_dice
    rebuilt = state_from_json(doc, [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)])
    assert state_to_json(rebuilt) == doc
    # Public: every view shows it.
    assert client_view(doc, "RED")["balanced_dice"] == doc["balanced_dice"]
    copy = game.copy()
    copy.state.balanced_dice.apply(copy.state.balanced_dice.draw(random.Random(0), 0), 0)
    assert game.state.balanced_dice == BalancedDice.from_json(doc["balanced_dice"], doc["colors"])


def test_unknown_dice_modes_are_refused():
    with pytest.raises(ValueError, match="dice_mode"):
        Game([SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)], dice_mode="loaded")


#: Whole random games (document digests without the random game id), recorded
#: with the code before robber_mode and dice_mode existed (bb3710d).
OLD_GAMES = {
    (2, 1, False): "aeb023015310cf69",
    (2, 1, True): "1e99ce66d4d6bf8d",
    (4, 2, False): "b278dac4726bf2ad",
    (4, 2, True): "bd55b1f29754cb34",
}


@pytest.mark.parametrize("players, seed, friendly_robber", sorted(OLD_GAMES))
def test_fair_dice_games_play_and_serialize_as_before(players, seed, friendly_robber):
    game = Game([RandomPlayer(c) for c in list(Color)[:players]], seed=seed, friendly_robber=friendly_robber)
    game.play()
    doc = state_to_json(game)
    assert "dice_mode" not in doc["game"] and "robber_mode" not in doc["game"]
    assert "balanced_dice" not in doc and "largest_army_taken_from" not in doc
    doc["game"].pop("id")
    digest = hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:16]
    assert digest == OLD_GAMES[(players, seed, friendly_robber)]
