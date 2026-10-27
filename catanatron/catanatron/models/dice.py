"""OpenHex's balanced dice: a 36-card deck with recency and seven balancing.

A game with ``dice_mode="balanced"`` rolls these instead of two fair dice
(OpenHex's 1v1 rules). This follows cppanatron's ``BalancedDice`` exactly
(same integer weights, same state). A roll by ``roller`` (the seat whose
turn it is):

1. If at most 13 cards remain, the deck is first refilled to all 36. Only
   the deck resets: the recent totals, the sevens and the streak carry over.
   A game starts with a full deck and no history, and the deck is refilled
   again when the main phase starts (as OpenHex does).
2. Total k in 2..12 weighs c_k * max(0, 1 - 0.34 a_k), with c_k of the cards
   left of total k and a_k the times k is among the last five totals.
3. With two or more seats, the 7's weight is multiplied by
   clamp(imbalance + streak, 0, 2): streak = 0.4 per 7 of the current run of
   7s, negated when the roller owns the run; imbalance = 1 + (1/n - s/S) /
   (1/n) once the 7s rolled so far S reach the n seats (s the roller's),
   else 1.
4. The total is drawn by weight, then the card uniformly among that total's
   cards left. (All-zero weights would draw a card uniformly; at least 14
   cards span three totals, which rules it out.)
5. The card leaves the deck, the total joins the window (dropping the oldest
   of six), a 7 counts for the roller and extends its run or starts one of 1,
   and any other total ends the run.

All of it is public: every roll is, and the deck has no order (a card's
chance depends only on its total), so views show it whole. The weights are
integers (0.34 = 17/50, 0.4 = 2/5), so the odds are exact.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple

DICE_MODES = ("random", "balanced")

#: The 36 cards, one per ordered pair of faces; card ``i`` is deck bit ``i``.
CARDS: Tuple[Tuple[int, int], ...] = tuple(
    (first, second) for first in range(1, 7) for second in range(1, 7)
)
FULL_DECK = (1 << len(CARDS)) - 1
TOTALS = tuple(range(2, 13))
#: A draw first refills the deck when at most this many cards remain.
RESHUFFLE_AT = 13
#: Totals remembered for the recency penalty.
WINDOW = 5
SEVEN = TOTALS.index(7)
#: The deck bits of each total.
_TOTAL_MASKS = tuple(
    sum(1 << bit for bit, card in enumerate(CARDS) if sum(card) == total)
    for total in TOTALS
)


def card_bit(card: Sequence[int]) -> int:
    """The deck bit of a ``(die1, die2)`` card; ValueError for other faces."""
    try:
        first, second = card
    except (TypeError, ValueError):
        raise ValueError(f"a roll is two dice, not {card!r}") from None
    if not all(type(d) is int and 1 <= d <= 6 for d in (first, second)):
        raise ValueError(f"dice faces must be 1 to 6, not {card!r}")
    return (first - 1) * 6 + (second - 1)


class BalancedDice:
    """The balanced dice between two rolls. Seats index ``State.colors``;
    a roller of None applies no seven balancing."""

    __slots__ = ("deck", "recent", "sevens", "streak_player", "streak_count")

    def __init__(
        self,
        seats: int,
        deck: int = FULL_DECK,
        recent: Sequence[int] = (),
        sevens: Optional[Sequence[int]] = None,
        streak_player: Optional[int] = None,
        streak_count: int = 0,
    ):
        #: The cards left, as deck bits.
        self.deck = deck
        #: The last totals, oldest first (at most WINDOW).
        self.recent: List[int] = list(recent)
        #: 7s rolled per seat.
        self.sevens: List[int] = list(sevens) if sevens is not None else [0] * seats
        #: The seat owning the current run of 7s (None for none), and its length.
        self.streak_player = streak_player
        self.streak_count = streak_count
        self.validate(seats)

    @classmethod
    def fresh(cls, seats: int) -> "BalancedDice":
        """A full deck and no history."""
        return cls(seats)

    def validate(self, seats: int) -> None:
        if not 1 <= seats <= 6 or len(self.sevens) != seats:
            raise ValueError("balanced dice need one sevens count for each of 1 to 6 seats")
        if type(self.deck) is not int or self.deck & ~FULL_DECK or self.deck < 0:
            raise ValueError("the balanced deck holds only the 36 cards")
        if len(self.recent) > WINDOW or any(
            type(t) is not int or t not in TOTALS for t in self.recent
        ):
            raise ValueError("balanced dice remember at most five totals of 2 to 12")
        if any(type(c) is not int or c < 0 for c in self.sevens):
            raise ValueError("sevens per seat must be non-negative integers")
        owned = self.streak_player is not None
        if (
            type(self.streak_count) is not int
            or owned != (self.streak_count > 0)
            or (owned and self.streak_player not in range(seats))
        ):
            raise ValueError(
                "a run of 7s has a seat for an owner exactly when its length is positive"
            )

    @property
    def seats(self) -> int:
        return len(self.sevens)

    def copy(self) -> "BalancedDice":
        # No validation: game copies (bots' lookahead) make many.
        dice = BalancedDice.__new__(BalancedDice)
        dice.deck = self.deck
        dice.recent = list(self.recent)
        dice.sevens = list(self.sevens)
        dice.streak_player = self.streak_player
        dice.streak_count = self.streak_count
        return dice

    def __eq__(self, other) -> bool:
        if not isinstance(other, BalancedDice):
            return NotImplemented
        return all(getattr(self, k) == getattr(other, k) for k in self.__slots__)

    def __repr__(self) -> str:
        return (
            f"BalancedDice(remaining={self.remaining}, recent={self.recent}, "
            f"sevens={self.sevens}, streak_player={self.streak_player}, "
            f"streak_count={self.streak_count})"
        )

    # ===== the next draw =====
    @property
    def remaining(self) -> int:
        """Cards left (before any refill)."""
        return self.deck.bit_count()

    def drawable(self) -> int:
        """The deck the next draw takes from: the full deck when it refills first."""
        return FULL_DECK if self.remaining <= RESHUFFLE_AT else self.deck

    def counts(self) -> List[int]:
        """Cards per total 2..12 in :meth:`drawable`."""
        cards = self.drawable()
        return [(cards & mask).bit_count() for mask in _TOTAL_MASKS]

    def can_draw(self, card: Sequence[int]) -> bool:
        try:
            return bool(self.drawable() >> card_bit(card) & 1)
        except ValueError:
            return False

    def seven_multiplier(self, roller: Optional[int]) -> Tuple[int, int]:
        """The factor on the 7's weight when ``roller`` rolls, as an exact
        fraction (numerator, denominator); 1 without seven balancing."""
        seats = self.seats
        if roller is not None and roller not in range(seats):
            raise ValueError(f"the roller {roller!r} is not a seat")
        if roller is None or seats < 2:
            return 1, 1
        # streak = 0.4 * count * direction = 2 * signed / 5.
        signed = 0
        if self.streak_count > 0 and self.streak_player is not None:
            signed = (
                -self.streak_count
                if self.streak_player == roller
                else self.streak_count
            )
        total = sum(self.sevens)
        if total >= seats:
            # 1 + (1/n - s/S) / (1/n) = (2S - n s) / S, plus 2 signed / 5: over 5S.
            denominator = 5 * total
            numerator = 5 * (2 * total - seats * self.sevens[roller]) + 2 * signed * total
        else:
            denominator = 5
            numerator = 5 + 2 * signed
        return min(max(numerator, 0), 2 * denominator), denominator

    def sum_weights(self, roller: Optional[int]) -> List[int]:
        """Integer weights of totals 2..12, proportional to their chances
        (cppanatron's ``BalancedDice.sum_weights``, the same integers)."""
        multiplier, denominator = self.seven_multiplier(roller)
        cards = self.counts()
        appearances = [0] * len(TOTALS)
        for total in self.recent:
            appearances[total - 2] += 1
        # c_k * (50 - 17 a_k)+ * (the 7's multiplier, or its denominator).
        weights = [
            cards[i]
            * max(0, 50 - 17 * appearances[i])
            * (multiplier if i == SEVEN else denominator)
            for i in range(len(TOTALS))
        ]
        return weights if any(weights) else cards

    def sum_probabilities(self, roller: Optional[int]) -> List[float]:
        """P(next total = 2..12) when ``roller`` rolls."""
        weights = self.sum_weights(roller)
        total = sum(weights)
        return [w / total for w in weights]

    def draw(self, rng: random.Random, roller: Optional[int]) -> Tuple[int, int]:
        """A card for ``roller`` from ``rng``, without changing the state: the
        total by weight, then one of its cards uniformly."""
        weights = self.sum_weights(roller)
        pick = rng.randrange(sum(weights))
        index = 0
        while pick >= weights[index]:
            pick -= weights[index]
            index += 1
        cards = self.drawable() & _TOTAL_MASKS[index]
        bits = [bit for bit in range(len(CARDS)) if cards >> bit & 1]
        return CARDS[bits[rng.randrange(len(bits))]]

    # ===== after a draw =====
    def apply(self, card: Sequence[int], roller: Optional[int]) -> None:
        """The state after ``roller`` drew ``card``. ValueError, changing
        nothing, for a card the deck no longer holds or a roller that is not
        a seat."""
        bit = card_bit(card)
        if roller is not None and roller not in range(self.seats):
            raise ValueError(f"the roller {roller!r} is not a seat")
        cards = self.drawable()
        if not cards >> bit & 1:
            raise ValueError(
                f"dice {tuple(card)} are not in the balanced deck: that card was "
                f"drawn since the last reshuffle ({cards.bit_count()} cards left)"
            )
        self.deck = cards & ~(1 << bit)
        total = card[0] + card[1]
        self.recent = (self.recent + [total])[-WINDOW:]
        if roller is None:
            return
        if total == 7:
            self.sevens[roller] += 1
            if self.streak_player == roller:
                self.streak_count += 1
            else:
                self.streak_player, self.streak_count = roller, 1
        else:
            self.streak_player, self.streak_count = None, 0

    def refill(self) -> None:
        """Only the deck resets (the main phase starts)."""
        self.deck = FULL_DECK

    # ===== JSON =====
    def to_json(self, colors: Sequence[str]) -> dict:
        """``colors``: the seats' color names, in seat order."""
        return {
            "deck": [list(card) for bit, card in enumerate(CARDS) if self.deck >> bit & 1],
            "recent": list(self.recent),
            "sevens": list(self.sevens),
            "streak_player": (
                None if self.streak_player is None else colors[self.streak_player]
            ),
            "streak_count": self.streak_count,
        }

    @classmethod
    def from_json(cls, doc: dict, colors: Sequence[str]) -> "BalancedDice":
        deck = 0
        for card in doc["deck"]:
            bit = card_bit(card)
            if deck >> bit & 1:
                raise ValueError("the balanced deck holds each card once")
            deck |= 1 << bit
        player = doc["streak_player"]
        return cls(
            len(colors),
            deck,
            doc["recent"],
            doc["sevens"],
            None if player is None else list(colors).index(player),
            doc["streak_count"],
        )
