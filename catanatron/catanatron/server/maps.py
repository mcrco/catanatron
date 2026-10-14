"""Explicit maps: a JSON list of tiles, and the board geometry clients draw.

A map spec is::

    {"tiles": [
        {"coordinate": [0, 0, 0], "type": "land", "resource": "WOOD", "number": 6},
        {"coordinate": [1, -1, 0], "type": "land", "resource": null},   # desert
        {"coordinate": [3, -3, 0], "type": "port", "direction": "WEST",
         "port_resource": "SHEEP"},                                    # 2:1
        {"coordinate": [2, -3, 1], "type": "water"},
        ...
    ]}

Coordinates are Catanatron's cube coordinates; a port's ``direction`` is the
way it faces (towards the land it serves), and a null ``port_resource`` is a
3:1 port. Node ids are numbered in the order the tiles are listed, exactly as
``CatanMap.from_tiles`` numbers a template's, so the same spec builds the same
map in any engine that follows that order.

Any shape works: the board's node graph is built from the land tiles
(``CatanMap.graph``), so land that only meets across water is not adjacent.
"""

from __future__ import annotations

from typing import Any

from catanatron.models.coordinate_system import UNIT_VECTORS, Direction, add
from catanatron.models.enums import RESOURCES
from catanatron.models.map import PORT_DIRECTION_TO_NODEREFS, CatanMap
from catanatron.models.tiles import LandTile, Port, Water
from catanatron.serialization import TILES, map_from_json

NUMBERS = (2, 3, 4, 5, 6, 8, 9, 10, 11, 12)
MAX_TILES = 200


class MapError(ValueError):
    """A map spec that cannot be played."""


def parse_map(spec: Any) -> CatanMap:
    """Validate a map spec and build the map."""
    return map_from_json(_tiles_document(spec))


def normalize_map(spec: Any) -> dict:
    """The spec in canonical form (what a room publishes in its config)."""
    return map_spec(parse_map(spec))


def _tiles_document(spec: Any) -> dict:
    if not isinstance(spec, dict) or not isinstance(spec.get("tiles"), list):
        raise MapError('a map spec is {"tiles": [...]}')
    tiles = spec["tiles"]
    if not 1 <= len(tiles) <= MAX_TILES:
        raise MapError(f"a map has 1 to {MAX_TILES} tiles, got {len(tiles)}")

    entries, seen, land, ports = [], set(), set(), []
    for position, tile in enumerate(tiles):
        where = f"tile {position}"
        if not isinstance(tile, dict):
            raise MapError(f"{where}: expected an object")
        coordinate = tile.get("coordinate")
        if (
            not isinstance(coordinate, list)
            or len(coordinate) != 3
            or not all(type(c) is int for c in coordinate)
            or sum(coordinate) != 0
        ):
            raise MapError(f"{where}: coordinate must be three integers summing to 0")
        coordinate = tuple(coordinate)
        if coordinate in seen:
            raise MapError(f"{where}: {list(coordinate)} appears twice")
        seen.add(coordinate)
        kind = str(tile.get("type", "")).lower()
        entry: dict = {"coordinate": list(coordinate)}
        if kind == "land":
            resource = tile.get("resource")
            if resource in (None, "DESERT"):
                entry.update(type="LAND", resource=None, number=None)
            elif resource in RESOURCES:
                number = tile.get("number")
                if type(number) is not int or number not in NUMBERS:
                    raise MapError(f"{where}: number must be one of {NUMBERS}")
                entry.update(type="LAND", resource=resource, number=number)
            else:
                raise MapError(f"{where}: unknown resource {resource!r}")
            land.add(coordinate)
        elif kind == "water":
            entry.update(type="WATER")
        elif kind == "port":
            resource = tile.get("port_resource", tile.get("resource"))
            if resource is not None and resource not in RESOURCES:
                raise MapError(f"{where}: unknown port resource {resource!r}")
            try:
                direction = Direction(str(tile.get("direction")).upper())
            except ValueError:
                raise MapError(
                    f"{where}: direction must be one of "
                    f"{[d.value for d in Direction]}"
                )
            entry.update(type="PORT", resource=resource, direction=direction.value)
            ports.append((where, coordinate, direction))
        else:
            raise MapError(f"{where}: type must be land, water or port")
        entries.append(entry)

    for where, coordinate, direction in ports:
        if add(coordinate, UNIT_VECTORS[direction]) not in land:
            raise MapError(f"{where}: the port faces {direction.value}, not land")
    if not any(e["type"] == "LAND" and e["resource"] is None for e in entries):
        raise MapError("a map needs a desert: the robber starts there")
    if not any(e["type"] == "LAND" and e["resource"] is not None for e in entries):
        raise MapError("a map needs at least one resource tile")

    next_land_id = next_port_id = 0
    for entry in entries:
        if entry["type"] == "LAND":
            entry["id"], next_land_id = next_land_id, next_land_id + 1
        elif entry["type"] == "PORT":
            entry["id"], next_port_id = next_port_id, next_port_id + 1
    return {"template": TILES, "tiles": entries}


def map_spec(catan_map: CatanMap) -> dict:
    """The spec of any map (inverse of :func:`parse_map`)."""
    tiles = []
    for coordinate, tile in catan_map.tiles.items():
        entry: dict = {"coordinate": list(coordinate)}
        if isinstance(tile, LandTile):
            entry.update(type="land", resource=tile.resource, number=tile.number)
        elif isinstance(tile, Port):
            entry.update(
                type="port",
                direction=tile.direction.value,
                port_resource=tile.resource,
            )
        else:
            entry.update(type="water")
        tiles.append(entry)
    return {"tiles": tiles}


def board_geometry(catan_map: CatanMap) -> list:
    """What a client needs to draw the board, tile by tile.

    Each tile carries its node ids by corner (``NORTH`` ... ``NORTHWEST``,
    pointy-top hexes) and its edges by side, so a client places node ``n``
    at any tile corner that names it. Ports also name the two nodes they
    serve.
    """
    tiles = []
    for coordinate, tile in catan_map.tiles.items():
        entry: dict = {
            "coordinate": list(coordinate),
            "nodes": {ref.value: node for ref, node in tile.nodes.items()},
            "edges": {ref.value: list(edge) for ref, edge in tile.edges.items()},
        }
        if isinstance(tile, LandTile):
            entry.update(
                type="land", id=tile.id, resource=tile.resource, number=tile.number
            )
        elif isinstance(tile, Port):
            a, b = PORT_DIRECTION_TO_NODEREFS[tile.direction]
            entry.update(
                type="port",
                id=tile.id,
                direction=tile.direction.value,
                port_resource=tile.resource,
                port_nodes=[tile.nodes[a], tile.nodes[b]],
            )
        else:
            assert isinstance(tile, Water)
            entry.update(type="water")
        tiles.append(entry)
    return tiles
