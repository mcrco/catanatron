"""Actions and log entries in words, and the legal actions in groups.

Everything here works on Catanatron ``Action``s (log entries arrive as JSON
and go through ``action_from_json`` first) and a ``CatanMap``, which is
enough to say what a node produces or which tile a robber move targets.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence, Tuple

from rich.text import Text

from catanatron.models.enums import Action
from catanatron.models.map import CatanMap
from catanatron.serialization import action_from_json
from catanatron.tui.board import PIPS, RESOURCES, player_style, resource_style

#: The groups of the action list, in order: (title, action types).
GROUPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("Turn", ("ROLL", "END_TURN")),
    ("Build", ("BUILD_SETTLEMENT", "BUILD_CITY", "BUILD_ROAD")),
    (
        "Development cards",
        (
            "BUY_DEVELOPMENT_CARD",
            "PLAY_KNIGHT_CARD",
            "PLAY_ROAD_BUILDING",
            "PLAY_YEAR_OF_PLENTY",
            "PLAY_MONOPOLY",
        ),
    ),
    (
        "Trade",
        (
            "MARITIME_TRADE",
            "OFFER_TRADE",
            "ACCEPT_TRADE",
            "REJECT_TRADE",
            "CONFIRM_TRADE",
            "CANCEL_TRADE",
        ),
    ),
    ("Robber", ("MOVE_ROBBER",)),
    ("Discard", ("DISCARD_RESOURCE",)),
)
_ORDER = {name: i for i, (_, names) in enumerate(GROUPS) for name in names}
_TYPE_ORDER = {name: j for _, names in GROUPS for j, name in enumerate(names)}


def resource_name(resource: Optional[str]) -> str:
    return "any" if resource is None else RESOURCES[resource][0]


def resource_text(resource: Optional[str], name: Optional[str] = None) -> Text:
    return Text(name or resource_name(resource), style=resource_style(resource))


def color_text(color: Optional[str], text: Optional[str] = None) -> Text:
    return Text(text or str(color), style=f"bold {player_style(color)}")


def tile_label(catan_map: CatanMap, coordinate) -> str:
    """``#7 wheat 5``, or ``#4 desert``."""
    tile = catan_map.land_tiles.get(tuple(coordinate))
    if tile is None:
        return str(tuple(coordinate))
    if tile.resource is None:
        return f"#{tile.id} desert"
    return f"#{tile.id} {resource_name(tile.resource)} {tile.number}"


def node_label(catan_map: CatanMap, node: int) -> str:
    """What a node touches: ``wh8 or5 sh10, 11 pips, 3:1 port``."""
    tiles = sorted(
        catan_map.adjacent_tiles.get(node, []),
        key=lambda t: (-PIPS.get(t.number, 0), t.id),
    )
    parts = [
        "desert" if t.resource is None else f"{RESOURCES[t.resource][1]}{t.number}"
        for t in tiles
    ]
    pips = sum(PIPS.get(t.number, 0) for t in tiles if t.resource is not None)
    text = " ".join(parts) + f", {pips} pips"
    for resource, nodes in catan_map.port_nodes.items():
        if node in nodes:
            text += (
                ", 3:1 port"
                if resource is None
                else f", 2:1 {resource_name(resource)} port"
            )
    return text


def describe(action: Action, catan_map: CatanMap) -> str:
    """A legal action as a menu entry: ``Settlement at 23 (wh8 or5, 9 pips)``."""
    kind, value = action.action_type.value, action.value
    if kind == "ROLL":
        return "Roll the dice"
    if kind == "END_TURN":
        return "End turn"
    if kind == "BUILD_SETTLEMENT":
        return f"Settlement at {value} ({node_label(catan_map, value)})"
    if kind == "BUILD_CITY":
        return f"City at {value} ({node_label(catan_map, value)})"
    if kind == "BUILD_ROAD":
        a, b = sorted(value)
        return f"Road {a}-{b}"
    if kind == "BUY_DEVELOPMENT_CARD":
        return "Buy a development card"
    if kind == "PLAY_KNIGHT_CARD":
        return "Play a knight"
    if kind == "PLAY_ROAD_BUILDING":
        return "Play road building"
    if kind == "PLAY_YEAR_OF_PLENTY":
        return "Year of plenty: " + " + ".join(resource_name(r) for r in value)
    if kind == "PLAY_MONOPOLY":
        return f"Monopoly on {resource_name(value)}"
    if kind == "MARITIME_TRADE":
        given = [r for r in value[:4] if r is not None]
        return f"Trade {len(given)} {resource_name(given[0])} for 1 {resource_name(value[4])}"
    if kind == "MOVE_ROBBER":
        coordinate, victim = value
        victim = getattr(victim, "value", victim)
        steal = f", steal from {victim}" if victim else ", steal nothing"
        return f"Robber to {tile_label(catan_map, coordinate)}{steal}"
    if kind == "DISCARD_RESOURCE":
        return "Discard " + ("a card" if value is None else resource_name(value))
    return f"{kind.replace('_', ' ').capitalize()} {'' if value is None else value}".strip()


def group_of(action: Action) -> str:
    index = _ORDER.get(action.action_type.value)
    return GROUPS[index][0] if index is not None else "Other"


def _natural_key(text: str) -> tuple:
    return tuple(
        int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)
    )


def sort_key(action: Action, catan_map: CatanMap) -> tuple:
    kind = action.action_type.value
    return (
        _ORDER.get(kind, len(GROUPS)),
        _TYPE_ORDER.get(kind, 99),
        _natural_key(describe(action, catan_map)),
    )


def grouped(
    actions: Iterable[Action], catan_map: CatanMap
) -> List[Tuple[str, List[Action]]]:
    """``actions`` in display order, as (group title, actions) pairs."""
    groups: dict = {}
    for action in sorted(actions, key=lambda a: sort_key(a, catan_map)):
        groups.setdefault(group_of(action), []).append(action)
    return list(groups.items())


def ordered(actions: Iterable[Action], catan_map: CatanMap) -> List[Action]:
    """``actions`` in the order the action list shows them."""
    return [a for _, group in grouped(actions, catan_map) for a in group]


# ===== the log =====
def describe_record(payload: Sequence, catan_map: CatanMap) -> Text:
    """One log entry (``[action, result]`` as JSON, possibly redacted)."""
    action = action_from_json(payload[0])
    result = payload[1]
    color = action.color.value
    kind, value = action.action_type.value, action.value
    text = color_text(color)
    text.append(" ")
    if kind == "ROLL":
        dice = result if result is not None else value
        if dice is not None:
            total = dice[0] + dice[1]
            text.append(f"rolled {dice[0]}+{dice[1]} = ")
            text.append(str(total), style="bold #ff5f5f" if total == 7 else "bold")
        else:
            text.append("rolled")
    elif kind == "END_TURN":
        text.append("ended the turn", style="#8a8a8a")
    elif kind == "BUILD_SETTLEMENT":
        text.append(f"built a settlement at {value}")
    elif kind == "BUILD_CITY":
        text.append(f"built a city at {value}")
    elif kind == "BUILD_ROAD":
        a, b = sorted(value)
        text.append(f"built a road {a}-{b}")
    elif kind == "BUY_DEVELOPMENT_CARD":
        card = result or value
        text.append("bought a development card")
        if card:
            text.append(f" ({card.replace('_', ' ').lower()})", style="italic")
    elif kind == "PLAY_KNIGHT_CARD":
        text.append("played a knight")
    elif kind == "PLAY_ROAD_BUILDING":
        text.append("played road building")
    elif kind == "PLAY_YEAR_OF_PLENTY":
        text.append("played year of plenty: ")
        for i, resource in enumerate(value):
            if i:
                text.append(" + ")
            text.append_text(resource_text(resource))
    elif kind == "PLAY_MONOPOLY":
        text.append("played monopoly on ")
        text.append_text(resource_text(value))
    elif kind == "MARITIME_TRADE":
        given = [r for r in value[:4] if r is not None]
        text.append(f"traded {len(given)} ")
        text.append_text(resource_text(given[0]))
        text.append(" for 1 ")
        text.append_text(resource_text(value[4]))
    elif kind == "MOVE_ROBBER":
        coordinate, victim = value
        text.append(f"moved the robber to {tile_label(catan_map, coordinate)}")
        if victim is not None:
            text.append(" and stole ")
            if result is not None:
                text.append_text(resource_text(result))
            else:
                text.append("a card")
            text.append(" from ")
            text.append_text(color_text(victim.value))
    elif kind == "DISCARD_RESOURCE":
        text.append("discarded ")
        discarded = value if value is not None else result
        if discarded is None:
            text.append("a card")
        else:
            text.append_text(resource_text(discarded))
    else:
        text.append(describe(action, catan_map).lower())
    return text
