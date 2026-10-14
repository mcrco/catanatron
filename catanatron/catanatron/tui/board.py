"""The board as text, for maps of any shape.

Each land tile is a flat-top hexagon with its node ids at the corners
(``medium`` size; ``small`` and ``large`` have shorter and longer sides)::

        43______21
       /          \\
      /    #14     \\
    44   wheat 8    16
      \\   •••••    /
       \\          /
        40______18

Settlements and cities are their node id in the owner's color (cities in
reverse video), roads are the sides between two nodes drawn in the owner's
color, and the robber sits on its tile. The dots count the ways to roll the
tile's number. Ports are labelled in the water off the coast they serve.

Catanatron's hexes are pointy-top; this picture is the board turned 30
degrees counterclockwise, which lets the sides be drawn with ``_``, ``/`` and
``\\``. Node, edge and tile ids are unchanged. Nothing here depends on the
map's shape: each tile is placed from its cube coordinate, so templates,
explicit maps (``CatanMap.from_tiles``) and islands all draw the same way.

:func:`draw_board` also returns where every node, edge and tile was drawn, so
a UI can map a mouse click or a cursor to a :data:`Target`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Collection, Dict, Iterable, Mapping, Optional, Tuple, Union

from rich.style import Style
from rich.text import Text

from catanatron.models.enums import EdgeRef, NodeRef
from catanatron.models.map import CatanMap
from catanatron.models.tiles import LandTile, Port

#: A pickable thing on the board: ``("node", 12)``, ``("edge", (3, 4))``
#: (node ids in increasing order) or ``("tile", (0, -1, 1))``.
Target = Tuple[str, Union[int, Tuple[int, ...]]]

PLAYER_COLORS = {
    "RED": "#ff4d4d",
    "BLUE": "#4d8dff",
    "ORANGE": "#ff9f1a",
    "WHITE": "#f2f2f2",
    "GREEN": "#3ddc84",
    "BROWN": "#c27c4a",
}
#: Name, short name and color of each resource.
RESOURCES = {
    "WOOD": ("wood", "wd", "#3e9e5b"),
    "BRICK": ("brick", "br", "#d9734e"),
    "SHEEP": ("sheep", "sh", "#a8d672"),
    "WHEAT": ("wheat", "wh", "#e8c547"),
    "ORE": ("ore", "or", "#a3acb9"),
}
DESERT_STYLE = "#c8ad7f"
OUTLINE_STYLE = "#56606e"
EMPTY_NODE_STYLE = "#7d8794"
TILE_ID_STYLE = "#7d8794"
PORT_STYLE = "#6fb3e0"
NUMBER_STYLE = "bold #e6e6e6"
HOT_NUMBER_STYLE = "bold #ff5f5f"
ROBBER_STYLE = "bold #ffffff on #9b1c1c"
HIGHLIGHT_STYLE = "bold #ffd75f"
CURSOR_STYLE = Style(reverse=True, bold=True)
RECENT_STYLE = Style(underline=True)

#: Ways to roll each number with two dice.
PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}

#: Corner slots of the turned hexagon, in NodeRef order (NORTH ... NORTHWEST).
UL, UR, E, LR, LL, W = range(6)
#: Each EdgeRef (EAST ... NORTHEAST) as the corner slots it joins.
EDGE_CORNERS = ((UR, E), (E, LR), (LR, LL), (LL, W), (W, UL), (UL, UR))
NODE_REFS = tuple(NodeRef)
EDGE_REFS = tuple(EdgeRef)

SIZES = ("small", "medium", "large")
SLANTS = {"small": 1, "medium": 2, "large": 3}


def player_style(color: Optional[str]) -> str:
    return PLAYER_COLORS.get(color or "", "#ffffff")


def resource_style(resource: Optional[str]) -> str:
    return DESERT_STYLE if resource is None else RESOURCES[resource][2]


@dataclass(frozen=True)
class HexGeometry:
    """Character offsets of a flat-top hex, from its center cell.

    ``slant`` is the height in rows of each sloping side; the top and bottom
    sides are ``3 * slant`` underscores. ``label`` is the width of a node id.
    """

    slant: int
    label: int = 2

    @property
    def underscores(self) -> int:
        return 3 * self.slant

    @property
    def top(self) -> int:
        return -(self.slant + 1)

    @property
    def first_underscore(self) -> int:
        return -(self.underscores // 2) + 1

    @property
    def west(self) -> int:
        return self.first_underscore - 2 * self.label - self.slant

    @property
    def east(self) -> int:
        return self.first_underscore + self.underscores + self.label + self.slant

    def corners(self) -> Tuple[Tuple[int, int], ...]:
        """(column, row) of each corner label's first character, in slot order."""
        left = self.first_underscore - self.label
        right = self.first_underscore + self.underscores
        top, bottom = self.top, -self.top
        return (
            (left, top),
            (right, top),
            (self.east, 0),
            (right, bottom),
            (left, bottom),
            (self.west, 0),
        )

    def side(self, edge: int) -> list:
        """The cells and glyphs of one side, by EdgeRef index."""
        rows = range(1, self.slant + 1)
        inner_west = self.west + self.label - 1
        if edge == 5:  # NORTHEAST: the top side.
            return [
                (self.first_underscore + i, self.top, "_")
                for i in range(self.underscores)
            ]
        if edge == 2:  # SOUTHWEST: the bottom side.
            return [
                (self.first_underscore + i, -self.top, "_")
                for i in range(self.underscores)
            ]
        if edge == 0:  # EAST: upper right.
            return [(self.east - k, -k, "\\") for k in rows]
        if edge == 1:  # SOUTHEAST: lower right.
            return [(self.east - k, k, "/") for k in rows]
        if edge == 3:  # WEST: lower left.
            return [(inner_west + k, k, "\\") for k in rows]
        return [(inner_west + k, -k, "/") for k in rows]  # NORTHWEST: upper left.

    def interior(self):
        """The cells strictly inside the outline: (column range, row) pairs."""
        for row in range(-self.slant, self.slant + 1):
            k = abs(row)
            if k == 0:
                yield range(self.west + self.label, self.east), row
            else:
                yield range(self.west + self.label + k, self.east - k), row

    def center(self, coordinate) -> Tuple[int, int]:
        """Column and row of a tile's center cell."""
        _, y, z = coordinate
        across = self.underscores + 2 * self.label + self.slant
        rise = self.slant + 1
        return -across * y, rise * y + 2 * rise * z


@dataclass(frozen=True)
class BoardState:
    """What sits on the board: buildings, roads and the robber.

    Built from the ``board`` section of a state document (the full one or a
    seat's view: the board is public), see :meth:`from_document`.
    """

    #: node -> (color, "SETTLEMENT" | "CITY")
    buildings: Mapping[int, Tuple[str, str]] = field(default_factory=dict)
    #: (a, b) with a < b -> color
    roads: Mapping[Tuple[int, int], str] = field(default_factory=dict)
    robber: Optional[Tuple[int, int, int]] = None

    @classmethod
    def from_document(cls, board: dict) -> "BoardState":
        return cls(
            buildings={node: (color, kind) for node, color, kind in board["buildings"]},
            roads={tuple(sorted(edge)): color for edge, color in board["roads"]},
            robber=tuple(board["robber_coordinate"]),
        )


@dataclass
class BoardPicture:
    """A drawn board and where everything on it is."""

    text: Text
    width: int
    height: int
    #: (column, row) -> the target drawn there, for mouse clicks.
    cells: Dict[Tuple[int, int], Target]
    #: target -> the (column, row) that represents it, for cursors.
    anchors: Dict[Target, Tuple[int, int]]

    def target_at(self, column: int, row: int) -> Optional[Target]:
        return self.cells.get((column, row))


def label_width(catan_map: CatanMap) -> int:
    return max(2, len(str(max(catan_map.land_nodes, default=0))))


def draw_board(
    catan_map: CatanMap,
    state: BoardState = BoardState(),
    *,
    size: str = "medium",
    highlights: Collection[Target] = (),
    cursor: Optional[Target] = None,
    recent: Collection[Target] = (),
    node_ids: bool = True,
) -> BoardPicture:
    """Draw ``catan_map`` with ``state``'s pieces on it.

    ``highlights`` are drawn as legal targets (yellow), ``cursor`` in reverse
    video and ``recent`` (say, the last move's pieces) underlined. Without
    ``node_ids``, empty nodes are dots.
    """
    if size not in SLANTS:
        raise ValueError(f"size must be one of {SIZES}, got {size!r}")
    geometry = HexGeometry(SLANTS[size], label_width(catan_map))
    canvas = _Canvas()
    highlights = set(highlights)
    recent = set(recent)
    land = [(c, t) for c, t in catan_map.tiles.items() if isinstance(t, LandTile)]
    land_nodes = catan_map.land_nodes

    def extra(target: Target) -> Optional[Style]:
        style = None
        if target in recent:
            style = RECENT_STYLE
        if target == cursor:
            style = CURSOR_STYLE if style is None else style + CURSOR_STYLE
        return style

    # Outlines first, then tile contents, then roads over the outlines and
    # node labels over everything.
    for coordinate, tile in land:
        col, row = geometry.center(coordinate)
        for edge in range(6):
            for dx, dy, glyph in geometry.side(edge):
                canvas.put_default(col + dx, row + dy, glyph, OUTLINE_STYLE)
    for coordinate, tile in land:
        _draw_tile(canvas, geometry, coordinate, tile, state, highlights, extra, size)

    for coordinate, tile in land:
        col, row = geometry.center(coordinate)
        for index, ref in enumerate(EDGE_REFS):
            edge = tuple(sorted(tile.edges[ref]))
            target: Target = ("edge", edge)
            owner = state.roads.get(edge)
            if owner is not None:
                style: Union[str, Style] = f"bold {player_style(owner)}"
            elif target in highlights:
                style = HIGHLIGHT_STYLE
            else:
                style = OUTLINE_STYLE
            style = _with(style, extra(target))
            for dx, dy, glyph in geometry.side(index):
                canvas.put(col + dx, row + dy, glyph, style, target)
            canvas.anchor(target, *_middle(geometry, col, row, index))

    for coordinate, tile in land:
        col, row = geometry.center(coordinate)
        for slot, ((dx, dy), ref) in enumerate(zip(geometry.corners(), NODE_REFS)):
            node = tile.nodes[ref]
            if node not in land_nodes:
                continue
            target = ("node", node)
            building = state.buildings.get(node)
            if building is not None:
                color = player_style(building[0])
                if building[1] == "CITY":
                    style = f"bold #000000 on {color}"
                else:
                    style = f"bold {color} on #2b2b2b"
            elif target in highlights:
                style = f"underline {HIGHLIGHT_STYLE}"
            else:
                style = EMPTY_NODE_STYLE
            style = _with(style, extra(target))
            # Left ends of a horizontal side (UL, LL, E) hug the underscores
            # to their right; right ends (UR, LR, W) those to their left.
            align = ">" if slot in (UL, LL, E) else "<"
            if node_ids or building is not None or target in highlights:
                text = f"{node:{align}{geometry.label}}"
            else:
                text = f"{'·':{align}{geometry.label}}"
            canvas.put_text(col + dx, row + dy, text, style, target)
            canvas.anchor(target, col + dx + (geometry.label - 1) // 2, row + dy)

    for coordinate, tile in catan_map.tiles.items():
        if isinstance(tile, Port):
            _draw_port(canvas, geometry, coordinate, tile, size)
    return canvas.picture()


def _draw_tile(canvas, geometry, coordinate, tile, state, highlights, extra, size):
    col, row = geometry.center(coordinate)
    target: Target = ("tile", tuple(coordinate))
    for columns, dy in geometry.interior():
        for column in columns:
            canvas.claim(col + column, row + dy, target)
    canvas.anchor(target, col, row)
    marker = extra(target)

    id_style: Union[str, Style] = (
        HIGHLIGHT_STYLE if target in highlights else TILE_ID_STYLE
    )
    canvas.centered(col, row - 1, f"#{tile.id}", _with(id_style, marker), target)
    if tile.resource is None:
        canvas.centered(col, row, "desert", _with(DESERT_STYLE, marker), target)
    else:
        # Small hexes are too narrow for "brick 10" between two node ids.
        name = RESOURCES[tile.resource][1 if size == "small" else 0]
        number_style = HOT_NUMBER_STYLE if tile.number in (6, 8) else NUMBER_STYLE
        start = col - (len(name) + 3 - 1) // 2
        canvas.put_text(
            start, row, name, _with(RESOURCES[tile.resource][2], marker), target
        )
        canvas.put_text(
            start + len(name) + 1,
            row,
            f"{tile.number:<2}",
            _with(number_style, marker),
            target,
        )
        canvas.centered(col, row + 1, "•" * PIPS[tile.number], number_style, target)
    if state.robber is not None and tuple(coordinate) == tuple(state.robber):
        # Below the pips where there is room, over them where there is not.
        robber_row = row + 2 if geometry.slant >= 2 else row + 1
        canvas.centered(col, robber_row, "ROBBER", ROBBER_STYLE, target)


def _draw_port(canvas, geometry, coordinate, tile, size) -> None:
    """A port's label, in the water as close to its coast as it fits."""
    col, row = geometry.center(coordinate)
    corners = geometry.corners()
    a, b = EDGE_CORNERS[EDGE_REFS.index(EdgeRef[tile.direction.name])]
    coast_col = col + (corners[a][0] + corners[b][0] + geometry.label) // 2
    coast_row = row + (corners[a][1] + corners[b][1]) // 2
    if tile.resource is None:
        label, style = "3:1", PORT_STYLE
    else:
        name, short, color = RESOURCES[tile.resource]
        label, style = f"2:1 {name if size == 'large' else short}", f"bold {color}"
    first, label_row = 0, row
    for fraction in (0.35, 0.5, 0.65, 0.8, 1.0, 1.2):
        label_row = round(coast_row + (row - coast_row) * fraction)
        first = round(coast_col + (col - coast_col) * fraction) - (len(label) - 1) // 2
        if canvas.is_free(range(first - 1, first + len(label) + 1), label_row):
            break
    canvas.put_text(first, label_row, label, style, None)


def _middle(geometry, col, row, edge):
    cells = geometry.side(edge)
    dx, dy, _ = cells[len(cells) // 2]
    return col + dx, row + dy


def _with(style, extra: Optional[Style]):
    if extra is None:
        return style
    base = Style.parse(style) if isinstance(style, str) else style
    return base + extra


class _Canvas:
    """Styled characters at (column, row), with the target each belongs to."""

    def __init__(self):
        self.chars: Dict[Tuple[int, int], Tuple[str, object]] = {}
        self.targets: Dict[Tuple[int, int], Target] = {}
        self.anchors: Dict[Target, Tuple[int, int]] = {}

    def put(self, col, row, char, style, target) -> None:
        self.chars[(col, row)] = (char, style)
        if target is not None:
            self.targets[(col, row)] = target

    def put_default(self, col, row, char, style) -> None:
        self.chars.setdefault((col, row), (char, style))

    def put_text(self, col, row, text, style, target) -> None:
        for offset, char in enumerate(text):
            self.put(col + offset, row, char, style, target)

    def centered(self, col, row, text, style, target) -> None:
        self.put_text(col - (len(text) - 1) // 2, row, text, style, target)

    def claim(self, col, row, target) -> None:
        self.targets.setdefault((col, row), target)

    def anchor(self, target, col, row) -> None:
        self.anchors.setdefault(target, (col, row))

    def is_free(self, columns: Iterable[int], row: int) -> bool:
        return all(
            (c, row) not in self.chars and (c, row) not in self.targets for c in columns
        )

    def picture(self) -> BoardPicture:
        """Rows of styled characters, trimmed to the drawn area."""
        text = Text(no_wrap=True, overflow="crop")
        if not self.chars:
            return BoardPicture(text, 0, 0, {}, {})
        min_col = min(c for c, _ in self.chars)
        max_col = max(c for c, _ in self.chars)
        min_row = min(r for _, r in self.chars)
        max_row = max(r for _, r in self.chars)
        for row in range(min_row, max_row + 1):
            if row != min_row:
                text.append("\n")
            run, run_style = [], None
            for col in range(min_col, max_col + 1):
                char, style = self.chars.get((col, row), (" ", ""))
                if style != run_style and run:
                    text.append("".join(run), style=run_style or "")
                    run = []
                run_style = style
                run.append(char)
            if run:
                text.append("".join(run), style=run_style or "")

        def shift(cell):
            return cell[0] - min_col, cell[1] - min_row

        return BoardPicture(
            text=text,
            width=max_col - min_col + 1,
            height=max_row - min_row + 1,
            cells={shift(cell): t for cell, t in self.targets.items()},
            anchors={t: shift(cell) for t, cell in self.anchors.items()},
        )


def legal_targets(actions) -> Dict[Target, list]:
    """The board targets of ``actions`` (Catanatron Actions), and the actions on each.

    Settlements and cities target nodes, roads edges, robber moves tiles.
    """
    targets: Dict[Target, list] = {}
    for action in actions:
        target = action_target(action)
        if target is not None:
            targets.setdefault(target, []).append(action)
    return targets


def action_target(action) -> Optional[Target]:
    name = action.action_type.value
    if name in ("BUILD_SETTLEMENT", "BUILD_CITY"):
        return ("node", action.value)
    if name == "BUILD_ROAD":
        return ("edge", tuple(sorted(action.value)))
    if name == "MOVE_ROBBER":
        return ("tile", tuple(action.value[0]))
    return None
