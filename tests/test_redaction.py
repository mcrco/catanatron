"""A seat's view must not depend on anything that seat cannot know.

The property, checked over random games of two to six players: take the
authoritative document, re-deal everything the seat cannot see (the other
seats' resources and development cards, the deck's order, the random
stream, other seats' draws and the steals it was not party to) in any way
consistent with what it *can* see, and the seat's view must not change. A
view that changed would be telling the seat which deal is the real one.
"""

import copy
import json
import random

import pytest

from catanatron.game import Game
from catanatron.models.decks import BASE_SUPPLY, EXTENSION_SUPPLY
from catanatron.models.enums import DEVELOPMENT_CARDS, RESOURCES
from catanatron.models.player import Color, RandomPlayer
from catanatron.serialization import client_view, public_view, state_to_json

PLAYABLE_CARDS = DEVELOPMENT_CARDS[:-1]


def random_documents(num_players, seed, samples=6):
    """Documents of one random game at a few random points."""
    colors = list(Color)[:num_players]
    supply = EXTENSION_SUPPLY if num_players > 4 else BASE_SUPPLY
    game = Game([RandomPlayer(c) for c in colors], seed=seed, supply=supply)
    rng = random.Random(seed)
    stops = sorted(rng.sample(range(1, 900), samples))
    documents = []
    for tick in range(stops[-1] + 1):
        if game.winning_color() is not None:
            break
        if tick in stops:
            documents.append(state_to_json(game))
        game.play_tick()
    documents.append(state_to_json(game))
    return documents


def redeal(doc, seat, rng):
    """Another document that ``seat`` (None: nobody) cannot tell from ``doc``."""
    doc = copy.deepcopy(doc)
    colors = doc["colors"]
    state = doc["player_state"]
    others = [i for i in range(len(colors)) if i != seat]

    # Resources: each hand's size and each resource's total held by the
    # others are known; who holds which is not.
    pool = [
        resource
        for i in others
        for resource in RESOURCES
        for _ in range(state[f"P{i}_{resource}_IN_HAND"])
    ]
    rng.shuffle(pool)
    for i in others:
        size = sum(state[f"P{i}_{resource}_IN_HAND"] for resource in RESOURCES)
        hand, pool = pool[:size], pool[size:]
        for resource in RESOURCES:
            state[f"P{i}_{resource}_IN_HAND"] = hand.count(resource)

    # Development cards: the deck and the others' hands form one unseen pool.
    pool = list(doc["development_listdeck"]) + [
        card
        for i in others
        for card in DEVELOPMENT_CARDS
        for _ in range(state[f"P{i}_{card}_IN_HAND"])
    ]
    rng.shuffle(pool)
    for i in others:
        size = sum(state[f"P{i}_{card}_IN_HAND"] for card in DEVELOPMENT_CARDS)
        hand, pool = pool[:size], pool[size:]
        for card in DEVELOPMENT_CARDS:
            state[f"P{i}_{card}_IN_HAND"] = hand.count(card)
        for card in PLAYABLE_CARDS:
            state[f"P{i}_{card}_OWNED_AT_START"] = rng.random() < 0.5
        state[f"P{i}_ACTUAL_VICTORY_POINTS"] = state[
            f"P{i}_VICTORY_POINTS"
        ] + hand.count("VICTORY_POINT")
    doc["development_listdeck"] = pool

    # The future, and the seed it came from.
    doc["game"]["seed"] = rng.getrandbits(64)
    doc["random_state"] = [3, [rng.getrandbits(32) for _ in range(625)], None]

    # Others' draws, and steals this seat was not party to.
    mine = None if seat is None else colors[seat]
    for record in doc["action_records"]:
        action, _ = record
        if action[1] == "BUY_DEVELOPMENT_CARD" and action[0] != mine:
            card = rng.choice(DEVELOPMENT_CARDS)
            action[2], record[1] = card, card
        elif (
            action[1] == "MOVE_ROBBER"
            and action[2][1] is not None
            and mine not in (action[0], action[2][1])
        ):
            record[1] = rng.choice(RESOURCES)
    return doc


CASES = [(n, seed) for n in range(2, 7) for seed in (1, 2)]


@pytest.mark.parametrize("num_players,seed", CASES)
def test_a_seats_view_is_the_same_in_every_world_it_cannot_tell_apart(
    num_players, seed
):
    rng = random.Random(seed)
    for doc in random_documents(num_players, seed):
        for seat, color in enumerate(doc["colors"]):
            view = client_view(doc, color)
            for _ in range(3):
                other = redeal(doc, seat, rng)
                assert client_view(other, color) == view
            json.dumps(view)  # and it is plain JSON


@pytest.mark.parametrize("num_players,seed", CASES)
def test_the_public_view_is_the_same_in_every_world_an_onlooker_cannot_tell_apart(
    num_players, seed
):
    rng = random.Random(seed)
    for doc in random_documents(num_players, seed):
        view = public_view(doc)
        for _ in range(3):
            assert public_view(redeal(doc, None, rng)) == view


def test_the_redeal_does_change_the_hidden_state():
    """Guards the property tests against a redeal that changes nothing."""
    rng = random.Random(0)
    doc = random_documents(4, 3)[-1]
    assert any(redeal(doc, 0, rng) != doc for _ in range(5))
    assert any(
        client_view(redeal(doc, 0, rng)) != client_view(doc) for _ in range(5)
    ), "the omniscient view does see the difference"


def test_the_public_view_hides_every_hand():
    doc = random_documents(3, 5)[-1]
    view = public_view(doc)
    for i in range(3):
        assert f"P{i}_WOOD_IN_HAND" not in view["player_state"]
        assert f"P{i}_KNIGHT_IN_HAND" not in view["player_state"]
        assert f"P{i}_ACTUAL_VICTORY_POINTS" not in view["player_state"]
        assert f"P{i}_NUM_RESOURCES_IN_HAND" in view["player_state"]
    for action, result in view["action_records"]:
        if action[1] == "BUY_DEVELOPMENT_CARD":
            assert (action[2], result) == (None, None)
        if action[1] == "MOVE_ROBBER":
            assert result is None
