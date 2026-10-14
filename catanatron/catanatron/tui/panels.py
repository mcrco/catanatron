"""The side panels, drawn from a state document: status, players and log.

The document is whatever the viewer may see (``client_view``,
``public_view`` or the full document in god view), so a panel shows a hand
card by card where the document has the cards and as a count where it only
has the count.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from rich.table import Table
from rich.text import Text

from catanatron.models.enums import DEVELOPMENT_CARDS, RESOURCES
from catanatron.models.map import CatanMap
from catanatron.tui.actions import color_text, describe_record
from catanatron.tui.board import RESOURCES as RESOURCE_STYLES
from catanatron.tui.session import Frame

DIM = "#8a8a8a"
CARD_LETTERS = {
    "KNIGHT": "K",
    "YEAR_OF_PLENTY": "Y",
    "MONOPOLY": "M",
    "ROAD_BUILDING": "R",
    "VICTORY_POINT": "V",
}


def turn_number(document: dict) -> Optional[int]:
    """The 1-based number of the turn being played (None during setup)."""
    if document["is_initial_build_phase"]:
        return None
    return document["num_turns"] - 2 * (len(document["colors"]) - 1) + 1


def prompt_text(document: dict) -> str:
    prompt = document["current_prompt"]
    index = document["current_player_index"]
    state = document["player_state"]
    if prompt == "BUILD_INITIAL_SETTLEMENT":
        return "place a settlement"
    if prompt == "BUILD_INITIAL_ROAD":
        return "place a road"
    if prompt == "DISCARD":
        count = document["discard_counts"][index]
        return f"discard {count} card{'s' if count != 1 else ''}"
    if prompt == "MOVE_ROBBER":
        return "move the robber"
    if prompt == "PLAY_TURN":
        if document.get("is_road_building"):
            return f"place free roads ({document['free_roads_available']} left)"
        if not state[f"P{index}_HAS_ROLLED"]:
            return "roll (or play a card)"
        return "build, trade, play a card or end turn"
    return prompt.replace("_", " ").lower()


def last_roll(document: dict) -> Optional[Sequence[int]]:
    for payload, result in reversed(document["action_records"]):
        if payload[1] == "ROLL":
            return result or payload[2]
    return None


def deck_size(document: dict) -> int:
    if "development_deck_size" in document:
        return document["development_deck_size"]
    deck = document["development_listdeck"]
    return sum(deck.values()) if isinstance(deck, dict) else len(deck)


def status_panel(frame: Frame, extra: Iterable[str] = ()) -> Text:
    """Whose move it is and what to do, the last roll, the deck and the
    debugging flags."""
    text = Text()
    document = frame.document
    if document is None:
        text.append(frame.status or "waiting for the game to start", style="italic")
        return text
    turn = turn_number(document)
    if frame.winner is not None:
        text.append_text(color_text(frame.winner))
        text.append(" wins!", style="bold")
        text.append(f"  after {turn} turns" if turn else "", style=DIM)
    elif frame.over:
        text.append("The game is over", style="bold")
    else:
        text.append("setup" if turn is None else f"turn {turn}", style=DIM)
        text.append("  ")
        text.append_text(color_text(frame.current))
        text.append(f" ({frame.seats.get(frame.current, '?')}): ", style=DIM)
        text.append(prompt_text(document))
    roll = last_roll(document)
    text.append("\n")
    if roll is not None:
        text.append("last roll ", style=DIM)
        total = roll[0] + roll[1]
        text.append(
            f"{roll[0]}+{roll[1]}={total}",
            style="bold #ff5f5f" if total == 7 else "bold",
        )
        text.append("  ·  ", style=DIM)
    deck = document.get("development_listdeck")
    text.append(f"{deck_size(document)} development cards left", style=DIM)
    if isinstance(deck, list) and deck:
        text.append(f" (next: {deck[-1].lower().replace('_', ' ')})", style=DIM)
    flags = list(frame.flags) + list(extra)
    if frame.god:
        flags.insert(0, "god view")
    elif frame.viewer is not None:
        flags.insert(0, f"viewing as {frame.viewer}")
    else:
        flags.insert(0, "spectator view")
    if flags:
        text.append("\n" + "  ·  ".join(flags), style="#a0c0ff")
    return text


def players_panel(frame: Frame) -> Table:
    """One row per seat: who plays it, points, cards, army, road, pieces;
    then the bank's cards and the development deck."""
    table = Table(
        box=None,
        padding=(0, 1),
        collapse_padding=True,
        pad_edge=False,
        expand=False,
        header_style=f"bold {DIM}",
        show_edge=False,
    )
    table.add_column("", width=1, no_wrap=True)
    table.add_column("seat", no_wrap=True, max_width=18, overflow="ellipsis")
    table.add_column("VP", justify="right", no_wrap=True)
    table.add_column(Text("wd br sh wh or", style=f"bold {DIM}"), no_wrap=True)
    table.add_column("dev", no_wrap=True)
    table.add_column("kn", justify="right", no_wrap=True)
    table.add_column("rd", justify="right", no_wrap=True)
    table.add_column("r/s/c", justify="right", no_wrap=True)
    document = frame.document
    if document is None:
        return table
    state = document["player_state"]
    for index, color in enumerate(document["colors"]):
        p = f"P{index}_"
        marker = (
            "▶" if color == frame.current else ("•" if color == frame.viewer else "")
        )
        seat = color_text(color, color.title())
        seat.append(f" {_short(frame.seats.get(color, ''), 11)}", style=DIM)
        vp = Text(str(state[p + "VICTORY_POINTS"]))
        actual = state.get(p + "ACTUAL_VICTORY_POINTS")
        if actual is not None and actual != state[p + "VICTORY_POINTS"]:
            vp.append(f"({actual})", style=DIM)
        if color == frame.winner:
            vp.stylize("bold #ffd75f")
        table.add_row(
            marker,
            seat,
            vp,
            _hand(state, p),
            _dev_cards(state, p),
            Text(str(state[p + "PLAYED_KNIGHT"]))
            + Text("A" if state[p + "HAS_ARMY"] else " ", style="bold #ffd75f"),
            Text(str(state[p + "LONGEST_ROAD_LENGTH"]))
            + Text("L" if state[p + "HAS_ROAD"] else " ", style="bold #ffd75f"),
            f"{state[p + 'ROADS_AVAILABLE']}/{state[p + 'SETTLEMENTS_AVAILABLE']}"
            f"/{state[p + 'CITIES_AVAILABLE']}",
        )
    bank = Text()
    for i, ((name, short, style), count) in enumerate(
        zip(RESOURCE_STYLES.values(), document["resource_freqdeck"])
    ):
        bank.append(" " if i else "")
        bank.append(f"{count:>2}", style=style)
    table.add_row(
        "", Text("bank", style=DIM), "", bank, Text(str(deck_size(document)), style=DIM)
    )
    return table


def _short(label: str, width: int = 12) -> str:
    return label if len(label) <= width else label[: width - 1] + "…"


def _hand(state: dict, p: str) -> Text:
    text = Text()
    if p + "WOOD_IN_HAND" not in state:
        count = state[p + "NUM_RESOURCES_IN_HAND"]
        text.append(f"{count} card{'s' if count != 1 else ''}", style="italic")
        return text
    counts = [state[f"{p}{resource}_IN_HAND"] for resource in RESOURCES]
    for i, ((name, short, style), count) in enumerate(
        zip(RESOURCE_STYLES.values(), counts)
    ):
        text.append(" " if i else "")
        text.append(f"{count:>2}", style=style if count else "#555555")
    text.append(f" ={sum(counts)}", style="bold")
    return text


def _dev_cards(state: dict, p: str) -> Text:
    if p + "KNIGHT_IN_HAND" not in state:
        return Text(str(state[p + "NUM_DEVELOPMENT_CARDS_IN_HAND"]), style="italic")
    held = [
        f"{state[f'{p}{card}_IN_HAND']}{CARD_LETTERS[card]}"
        for card in DEVELOPMENT_CARDS
        if state[f"{p}{card}_IN_HAND"]
    ]
    return Text(" ".join(held) if held else "-")


def log_lines(
    document: dict, catan_map: CatanMap, notes: Dict[int, List[str]], start: int = 0
) -> List[Text]:
    """The log from entry ``start`` on, one line per entry (plus its notes),
    with the turn number on the first entry of each turn."""
    records = document["action_records"]
    setup = 4 * len(document["colors"])  # two settlements and two roads each
    labels = turn_labels(records, setup)
    lines = []
    for index in range(start, len(records)):
        line = Text(f"{labels[index]:>5} ", style=DIM)
        line.append_text(describe_record(records[index], catan_map))
        for note in notes.get(index, []):
            line.append(f"  [{note}]", style="italic #a0c0ff")
        lines.append(line)
    return lines


def turn_labels(records: Sequence, setup: int) -> List[str]:
    """Per log entry: ``setup`` or the turn number on the first entry of a
    phase or turn, blank elsewhere."""
    labels, turn = [], 1
    for index, (payload, _) in enumerate(records):
        if index == 0:
            labels.append("setup")
        elif index < setup:
            labels.append("")
        elif index == setup or records[index - 1][0][1] == "END_TURN":
            labels.append(str(turn))
        else:
            labels.append("")
        if index >= setup and payload[1] == "END_TURN":
            turn += 1
    return labels
