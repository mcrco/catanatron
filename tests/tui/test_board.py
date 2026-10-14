"""The board drawing: every shape, every size, 2-6 players' pieces."""

import random

import pytest

from catanatron.models.map import build_map
from catanatron.models.player import Color
from catanatron.tui.board import (
    PLAYER_COLORS,
    SIZES,
    BoardState,
    draw_board,
    label_width,
    legal_targets,
)
from catanatron.models.enums import Action, ActionType

from tests.tui.helpers import SHAPES, load_map, snapshot

TEMPLATES = ["BASE", "MINI", "TOURNAMENT"]


def make_map(name):
    if name in TEMPLATES:
        return build_map(name, rng=random.Random(0))
    return load_map(name)


def desert(catan_map):
    return next(c for c, t in catan_map.land_tiles.items() if t.resource is None)


def lines_of(picture):
    lines = picture.text.plain.split("\n")
    assert len(lines) == picture.height
    assert all(len(line) == picture.width for line in lines)
    return lines


def cells_of(picture, target):
    return sorted(
        (cell for cell, t in picture.cells.items() if t == target),
        key=lambda c: (c[1], c[0]),
    )


def read(lines, cells):
    return "".join(lines[row][col] for col, row in cells)


def style_at(picture, col, row):
    """The style of one cell, as a string (all spans that cover it)."""
    offset = row * (picture.width + 1) + col
    return " ".join(
        str(span.style)
        for span in picture.text.spans
        if span.start <= offset < span.end
    )


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("name", TEMPLATES + SHAPES)
def test_every_node_edge_tile_and_port_is_drawn_where_it_is_picked(name, size):
    catan_map = make_map(name)
    robber = desert(catan_map)
    picture = draw_board(catan_map, BoardState(robber=robber), size=size)
    lines = lines_of(picture)
    width = label_width(catan_map)

    for node in catan_map.land_nodes:
        cells = cells_of(picture, ("node", node))
        assert len(cells) == width and len({row for _, row in cells}) == 1
        assert read(lines, cells).strip() == str(node)
        assert picture.anchors[("node", node)] in cells

    for edge in catan_map.land_edges:
        target = ("edge", tuple(sorted(edge)))
        cells = cells_of(picture, target)
        assert cells, f"edge {edge} is not drawn"
        assert set(read(lines, cells)) <= set("_/\\")
        assert picture.anchors[target] in cells

    for coordinate, tile in catan_map.land_tiles.items():
        target = ("tile", coordinate)
        col, row = picture.anchors[target]
        assert picture.target_at(col, row) == target
        cells = cells_of(picture, target)
        tile_text = "\n".join(
            "".join(lines[r][c] for c, cr in cells if cr == r)
            for r in sorted({r for _, r in cells})
        )
        assert f"#{tile.id}" in tile_text
        assert ("ROBBER" in tile_text) == (coordinate == robber)
        if tile.resource is None:
            assert "desert" in tile_text
        else:
            assert f" {tile.number}" in tile_text

    ports = sum(1 for t in catan_map.tiles.values() if type(t).__name__ == "Port")
    plain = picture.text.plain
    assert plain.count("3:1") + plain.count("2:1") == ports


@pytest.mark.parametrize("num_players", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("name", ["BASE", "extension", "hexagon3"])
def test_pieces_are_drawn_in_their_owners_colors(name, num_players):
    catan_map = make_map(name)
    colors = list(Color)[:num_players]
    nodes = sorted(catan_map.land_nodes)
    edges = sorted(tuple(sorted(e)) for e in catan_map.land_edges)
    buildings, roads = {}, {}
    for i, color in enumerate(colors):
        buildings[nodes[7 * i]] = (color.value, "SETTLEMENT")
        buildings[nodes[7 * i + 3]] = (color.value, "CITY")
        roads[edges[9 * i]] = color.value
    state = BoardState(buildings=buildings, roads=roads, robber=desert(catan_map))
    for size in SIZES:
        picture = draw_board(catan_map, state, size=size)
        for node, (color, kind) in buildings.items():
            col, row = cells_of(picture, ("node", node))[-1]
            style = style_at(picture, col, row)
            assert PLAYER_COLORS[color] in style, (node, color, style)
            assert ("on " + PLAYER_COLORS[color] in style) == (kind == "CITY")
        for edge, color in roads.items():
            col, row = picture.anchors[("edge", edge)]
            assert PLAYER_COLORS[color] in style_at(picture, col, row)


def test_highlights_cursor_and_hidden_ids():
    catan_map = make_map("BASE")
    node, edge = 12, (12, 13)
    picture = draw_board(
        catan_map,
        highlights=[("node", node), ("edge", edge)],
        cursor=("node", node),
        node_ids=False,
    )
    lines = lines_of(picture)
    assert read(lines, cells_of(picture, ("node", node))).strip() == "12"
    assert read(lines, cells_of(picture, ("node", 13))).strip() == "·"
    col, row = picture.anchors[("node", node)]
    assert "reverse" in style_at(picture, col, row)
    col, row = picture.anchors[("edge", edge)]
    assert "#ffd75f" in style_at(picture, col, row)


def test_legal_targets_group_actions_by_board_target():
    red = Color.RED
    actions = [
        Action(red, ActionType.BUILD_SETTLEMENT, 3),
        Action(red, ActionType.BUILD_ROAD, (5, 4)),
        Action(red, ActionType.MOVE_ROBBER, ((0, 0, 0), Color.BLUE)),
        Action(red, ActionType.MOVE_ROBBER, ((0, 0, 0), None)),
        Action(red, ActionType.END_TURN, None),
    ]
    targets = legal_targets(actions)
    assert set(targets) == {("node", 3), ("edge", (4, 5)), ("tile", (0, 0, 0))}
    assert len(targets[("tile", (0, 0, 0))]) == 2


def test_snapshots():
    base = make_map("BASE")
    state = BoardState(
        buildings={
            12: ("RED", "SETTLEMENT"),
            20: ("BLUE", "CITY"),
            40: ("WHITE", "SETTLEMENT"),
            8: ("ORANGE", "CITY"),
        },
        roads={(12, 13): "RED", (19, 20): "BLUE", (8, 27): "ORANGE"},
        robber=(0, 0, 0),
    )
    snapshot("base_medium", draw_board(base, state, size="medium").text.plain)
    snapshot("base_small", draw_board(base, state, size="small").text.plain)
    extension = make_map("extension")
    snapshot(
        "extension_small",
        draw_board(
            extension, BoardState(robber=desert(extension)), size="small"
        ).text.plain,
    )
    hexagon = make_map("hexagon3")
    snapshot(
        "hexagon3_large",
        draw_board(
            hexagon, BoardState(robber=desert(hexagon)), size="large"
        ).text.plain,
    )
