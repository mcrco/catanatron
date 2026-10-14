"""Providers helper functions to deal with representations of decks of cards

We use a histogram / 'frequency list' to represent decks (aliased 'freqdeck').
This representation is concise, easy to copy, access and fast to compare.
"""

from dataclasses import dataclass
from typing import Iterable, List, Tuple

from catanatron.models.enums import (
    KNIGHT,
    MONOPOLY,
    ROAD_BUILDING,
    VICTORY_POINT,
    YEAR_OF_PLENTY,
    WOOD,
    BRICK,
    SHEEP,
    WHEAT,
    ORE,
    FastDevCard,
    FastResource,
)


ROAD_COST_FREQDECK = [1, 1, 0, 0, 0]
SETTLEMENT_COST_FREQDECK = [1, 1, 1, 1, 0]
CITY_COST_FREQDECK = [0, 0, 0, 2, 3]
DEVELOPMENT_CARD_COST_FREQDECK = [0, 0, 1, 1, 1]


# ===== ListDecks
def starting_resource_bank():
    """Returns freqdeck of resource cards"""
    return [19, 19, 19, 19, 19]


RESOURCE_FREQDECK_INDEXES = {WOOD: 0, BRICK: 1, SHEEP: 2, WHEAT: 3, ORE: 4}


def freqdeck_can_draw(freqdeck, amount: int, card: FastResource):
    return freqdeck[RESOURCE_FREQDECK_INDEXES[card]] >= amount


def freqdeck_draw(freqdeck, amount: int, card: FastResource):
    freqdeck[RESOURCE_FREQDECK_INDEXES[card]] -= amount


def freqdeck_replenish(freqdeck, amount: int, card: FastResource):
    freqdeck[RESOURCE_FREQDECK_INDEXES[card]] += amount


def freqdeck_count(freqdeck, card: FastResource):
    return freqdeck[RESOURCE_FREQDECK_INDEXES[card]]


def freqdeck_from_listdeck(listdeck: Iterable[FastResource]):
    freqdeck = [0, 0, 0, 0, 0]
    for resource in listdeck:
        freqdeck_replenish(freqdeck, 1, resource)
    return freqdeck


def starting_devcard_proba(card: FastDevCard):
    starting_deck = starting_devcard_bank()
    return starting_deck.count(card) / len(starting_deck)


def starting_devcard_bank():
    """Returns listdeck of devcards"""
    return (
        [KNIGHT] * 14
        + [YEAR_OF_PLENTY] * 2
        + [ROAD_BUILDING] * 2
        + [MONOPOLY] * 2
        + [VICTORY_POINT] * 5
    )


@dataclass(frozen=True)
class Supply:
    """Card and piece counts, which differ between the base game and the 5-6
    player extension. The defaults are the base game's."""

    #: Starting bank as a freqdeck: [WOOD, BRICK, SHEEP, WHEAT, ORE].
    resource_bank: Tuple[int, int, int, int, int] = (19, 19, 19, 19, 19)
    #: Starting deck in DEVELOPMENT_CARDS order:
    #: KNIGHT, YEAR_OF_PLENTY, MONOPOLY, ROAD_BUILDING, VICTORY_POINT.
    development_deck: Tuple[int, int, int, int, int] = (14, 2, 2, 2, 5)
    #: Pieces each player starts with.
    roads: int = 15
    settlements: int = 5
    cities: int = 4

    def __post_init__(self):
        if len(self.resource_bank) != 5 or len(self.development_deck) != 5:
            raise ValueError("resource_bank and development_deck need five counts each")
        if min(self.resource_bank) < 0 or min(self.development_deck) < 0:
            raise ValueError("card counts must be non-negative")
        if self.roads < 2 or self.settlements < 2 or self.cities < 0:
            raise ValueError("players need at least two roads and two settlements")

    def resource_freqdeck(self) -> List[int]:
        return list(self.resource_bank)

    def development_listdeck(self) -> List[FastDevCard]:
        """Unshuffled, in starting_devcard_bank's order so seeds shuffle alike."""
        knights, year_of_plenty, monopoly, road_building, victory_points = (
            self.development_deck
        )
        return (
            [KNIGHT] * knights
            + [YEAR_OF_PLENTY] * year_of_plenty
            + [ROAD_BUILDING] * road_building
            + [MONOPOLY] * monopoly
            + [VICTORY_POINT] * victory_points
        )


BASE_SUPPLY = Supply()
#: The 5-6 player extension adds five of each resource and nine development
#: cards (six knights and one of each progress card).
EXTENSION_SUPPLY = Supply(
    resource_bank=(24, 24, 24, 24, 24),
    development_deck=(20, 3, 3, 3, 5),
)


def draw_from_listdeck(list1: List, amount: int, card: int):
    i = 0
    while i < amount:
        index = list1.index(card)
        del list1[index]
        i += 1


def freqdeck_add(list1, list2):
    return [a + b for a, b in zip(list1, list2)]


def freqdeck_subtract(list1, list2):
    return [a - b for a, b in zip(list1, list2)]


def freqdeck_contains(list1, list2):
    """True if list1 >= list2 element-wise"""
    return all([a >= b for a, b in zip(list1, list2)])
