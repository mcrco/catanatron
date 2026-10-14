"""Complete a seat's view into a full game, for bots that need one.

Catanatron's bots decide with a whole ``Game`` in hand. A seat on the wire
only has its view, so it hands its bot a game sampled from what the view
leaves open: the other seats' resources (each hand's size is public, and so
is each resource's total across their hands: the starting bank minus the
bank minus your own hand), their development cards (dealt from the unseen
pool, redrawn if a deal would give a seat enough victory points to have
already won), the deck's order and the random stream. The bot's lookahead
therefore never sees hidden cards or future rolls.

No card tracking: with three or more players, steals and production make
some splits likelier than others, which uniform dealing ignores.
"""

from __future__ import annotations

import copy
import random
from typing import Optional, Sequence

from catanatron.game import Game
from catanatron.models.decks import Supply
from catanatron.models.enums import DEVELOPMENT_CARDS, RESOURCES
from catanatron.models.player import Color, Player, RandomPlayer
from catanatron.serialization import state_from_json

PLAYABLE_CARDS = DEVELOPMENT_CARDS[:-1]
MAX_REDRAWS = 200


def sample_game(
    view: dict,
    color: Color,
    supply: Supply,
    rng: random.Random,
    players: Optional[Sequence[Player]] = None,
) -> Game:
    """A full game consistent with ``color``'s ``view``.

    ``view`` is a seated ``client_view`` (with its ``map`` and
    ``action_records``); ``players`` seat the result (placeholders by
    default).
    """
    document = copy.deepcopy(view)
    colors = document["colors"]
    seat = colors.index(color.value)
    state = document["player_state"]
    others = [i for i in range(len(colors)) if i != seat]

    bank = document["resource_freqdeck"]
    held = [
        supply.resource_bank[index] - bank[index] - state[f"P{seat}_{resource}_IN_HAND"]
        for index, resource in enumerate(RESOURCES)
    ]
    pool = [resource for resource, count in zip(RESOURCES, held) for _ in range(count)]
    sizes = {i: state.pop(f"P{i}_NUM_RESOURCES_IN_HAND") for i in others}
    if len(pool) != sum(sizes.values()):
        raise ValueError("the view's bank and hand sizes do not add up")
    rng.shuffle(pool)
    for i in others:
        hand, pool = pool[: sizes[i]], pool[sizes[i] :]
        for resource in RESOURCES:
            state[f"P{i}_{resource}_IN_HAND"] = hand.count(resource)

    unseen = [
        card
        for card, count in document.pop("unseen_development_cards").items()
        for _ in range(count)
    ]
    counts = {i: state.pop(f"P{i}_NUM_DEVELOPMENT_CARDS_IN_HAND") for i in others}
    vps_to_win = document["game"]["vps_to_win"]
    for _ in range(MAX_REDRAWS):
        rng.shuffle(unseen)
        hands, rest = {}, unseen
        for i in others:
            hands[i], rest = rest[: counts[i]], rest[counts[i] :]
        if all(
            state[f"P{i}_VICTORY_POINTS"] + hands[i].count("VICTORY_POINT") < vps_to_win
            for i in others
        ):
            break
    for i in others:
        for card in DEVELOPMENT_CARDS:
            state[f"P{i}_{card}_IN_HAND"] = hands[i].count(card)
        # Only the seat to move plays cards, so earlier turns' flags suffice.
        for card in PLAYABLE_CARDS:
            state[f"P{i}_{card}_OWNED_AT_START"] = hands[i].count(card) > 0
        state[f"P{i}_ACTUAL_VICTORY_POINTS"] = state[f"P{i}_VICTORY_POINTS"] + hands[
            i
        ].count("VICTORY_POINT")
    document["development_listdeck"] = rest
    document.pop("development_deck_size", None)

    seed = rng.getrandbits(63)
    version, internal, gauss = random.Random(seed).getstate()
    document["random_state"] = [version, list(internal), gauss]
    document["game"] = {**document["game"], "seed": seed}
    seated = (
        list(players)
        if players is not None
        else [RandomPlayer(Color[name]) for name in colors]
    )
    return state_from_json(document, seated)
