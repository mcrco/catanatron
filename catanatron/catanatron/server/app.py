"""The WebSocket server: a lobby, rooms, and the browser client.

Routes (one port, HTTP and WebSocket):

    GET /                  the browser client (static/index.html)
    GET /static/FILE       its assets
    GET /api/rooms         rooms, as JSON
    GET /api/players       the registry's bots, as JSON
    GET /api/server        the server's public key and protocol version
    WS  /lobby             create_room, list_rooms, players
    WS  /room/ROOM_ID      join a seat, or spectate, and play
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import secrets
import signal
import sys
from http import HTTPStatus
from typing import Any, Optional, Sequence

from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from catanatron.registry import REGISTRY, describe
from catanatron.server.config import ConfigError, GameConfig
from catanatron.server.crypto import SigningKey
from catanatron.server.messages import PROTOCOL_VERSION
from catanatron.server.room import Outbox, Peer, Room, RoomError

logger = logging.getLogger(__name__)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".json": "application/json",
}
MAX_MESSAGE = 1 << 20


class GameServer:
    def __init__(
        self,
        key: Optional[SigningKey] = None,
        *,
        archive_dir: Optional[str] = None,
        max_rooms: int = 500,
        room_class: type = Room,
    ):
        from catanatron.players import register_builtins

        register_builtins()
        self.key = key or SigningKey.generate()
        self.archive_dir = archive_dir
        self.max_rooms = max_rooms
        self.room_class = room_class
        self.rooms: dict = {}

    # ===== rooms =====
    def create_room(
        self, config: Any, seats: Optional[list] = None, name: str = ""
    ) -> Room:
        game_config = GameConfig.from_json(config)
        seats = list(seats or [])
        if len(seats) > game_config.num_players:
            raise ConfigError(
                f"{len(seats)} seat specs for {game_config.num_players} seats"
            )
        self._evict()
        room_id = secrets.token_urlsafe(6).replace("-", "x").replace("_", "y")
        room = self.room_class(
            room_id, game_config, self.key, name=name, on_finish=self._archive
        )
        for index, spec in enumerate(seats):
            if spec is not None:
                room.set_bot(index, spec)
        self.rooms[room_id] = room
        return room

    def _evict(self) -> None:
        if len(self.rooms) < self.max_rooms:
            return
        finished = sorted(
            (r for r in self.rooms.values() if r.status == "finished"),
            key=lambda r: r.created,
        )
        for room in finished[: max(1, len(self.rooms) - self.max_rooms + 1)]:
            del self.rooms[room.id]
        if len(self.rooms) >= self.max_rooms:
            raise ConfigError("the server is full")

    def _archive(self, room: Room) -> None:
        """Keep the server's own record of a finished game."""
        if not self.archive_dir or room.game is None:
            return
        os.makedirs(self.archive_dir, exist_ok=True)
        record = {
            "room": room.describe(),
            "server_key": self.key.public_hex,
            "reveal": room._reveal(),
            "commitments": room.commitments,
            "final_state": room.doc,
        }
        path = os.path.join(self.archive_dir, f"{room.id}-{room.game_id}.json")
        with open(path, "w") as file:
            json.dump(record, file)

    # ===== HTTP =====
    def process_request(self, connection: ServerConnection, request: Request):
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return None
        path = request.path.split("?", 1)[0]
        if path == "/":
            return _file_response("index.html")
        if path.startswith("/static/"):
            return _file_response(path[len("/static/") :])
        if path == "/api/rooms":
            return _json_response([r.describe() for r in self.rooms.values()])
        if path.startswith("/api/rooms/"):
            room = self.rooms.get(path[len("/api/rooms/") :])
            if room is None:
                return _json_response({"error": "no such room"}, HTTPStatus.NOT_FOUND)
            return _json_response(room.describe())
        if path == "/api/players":
            return _json_response(_players())
        if path == "/api/server":
            return _json_response(
                {
                    "server_key": self.key.public_hex,
                    "protocol_version": PROTOCOL_VERSION,
                }
            )
        return _json_response({"error": "not found"}, HTTPStatus.NOT_FOUND)

    # ===== WebSocket =====
    async def handler(self, websocket: ServerConnection) -> None:
        path = websocket.request.path.split("?", 1)[0] if websocket.request else ""
        if path == "/lobby":
            await self._lobby(websocket)
        elif path.startswith("/room/"):
            room = self.rooms.get(path[len("/room/") :].strip("/"))
            if room is None:
                await websocket.close(4004, "no such room")
                return
            await self._room(websocket, room)
        else:
            await websocket.close(4004, "unknown path")

    async def _lobby(self, websocket: ServerConnection) -> None:
        async for text in websocket:
            try:
                request = json.loads(text)
                kind = request.get("type")
                if kind == "create_room":
                    room = self.create_room(
                        request.get("config"),
                        request.get("seats"),
                        str(request.get("name") or ""),
                    )
                    reply = {"type": "room_created", "room": room.describe()}
                elif kind == "list_rooms":
                    reply = {
                        "type": "rooms",
                        "rooms": [r.describe() for r in self.rooms.values()],
                    }
                elif kind == "players":
                    reply = {"type": "players", "players": _players()}
                else:
                    reply = {"type": "error", "message": f"unknown request {kind!r}"}
            except (ValueError, AttributeError, RoomError) as error:
                reply = {"type": "error", "message": str(error)}
            await websocket.send(json.dumps(reply))

    async def _room(self, websocket: ServerConnection, room: Room) -> None:
        peer = Peer(websocket)
        outbox: Outbox = room.connect(peer)
        seat, watching = None, False
        try:
            async for text in websocket:
                request = None
                try:
                    request = json.loads(text)
                    if not isinstance(request, dict):
                        raise RoomError("bad_request", "expected a JSON object")
                    kind = request.get("type")
                    if kind == "join" and seat is None and not watching:
                        outbox = room.join(outbox, peer, request)
                        seat = next(s for s in room.seats if s.outbox is outbox)
                    elif kind == "spectate" and seat is None and not watching:
                        room.spectate(outbox, request)
                        watching = True
                    elif kind == "action" and seat is not None:
                        room.on_action(seat, request)
                    elif kind == "seat_bot":
                        room.set_bot(request.get("seat"), request.get("spec"))
                    elif kind == "ping":
                        pass
                    else:
                        raise RoomError("bad_request", f"unexpected {kind!r} here")
                except RoomError as error:
                    room.error(outbox, error.code, str(error), request)
                except ValueError as error:
                    room.error(outbox, "bad_request", f"not JSON: {error}")
        except ConnectionClosed:
            pass
        finally:
            room.disconnect(outbox, peer)


def _players() -> list:
    return [
        describe(key, cls)
        for key, cls in sorted(REGISTRY.items())
        if getattr(cls, "IS_BOT", True)
    ]


def _json_response(payload: Any, status: HTTPStatus = HTTPStatus.OK) -> Response:
    body = json.dumps(payload).encode()
    return _response(status, body, "application/json")


def _file_response(name: str) -> Response:
    # Only plain file names in the static directory: no traversal.
    if "/" in name or "\\" in name or name.startswith("."):
        return _json_response({"error": "not found"}, HTTPStatus.NOT_FOUND)
    path = os.path.join(STATIC_DIR, name)
    extension = os.path.splitext(name)[1]
    if not os.path.isfile(path) or extension not in CONTENT_TYPES:
        return _json_response({"error": "not found"}, HTTPStatus.NOT_FOUND)
    with open(path, "rb") as file:
        return _response(HTTPStatus.OK, file.read(), CONTENT_TYPES[extension])


def _response(status: HTTPStatus, body: bytes, content_type: str) -> Response:
    headers = Headers(
        [
            ("Content-Type", content_type),
            ("Content-Length", str(len(body))),
            ("Cache-Control", "no-cache"),
            ("Connection", "close"),
        ]
    )
    return Response(status.value, status.phrase, headers, body)


async def run_server(server: GameServer, host: str = "127.0.0.1", port: int = 8765):
    """Start serving; returns the websockets server (``.close()`` to stop)."""
    return await serve(
        server.handler,
        host,
        port,
        process_request=server.process_request,
        max_size=MAX_MESSAGE,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="catanatron-server",
        description="Serve Catanatron games over WebSocket, verifiably fair.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--key",
        default=os.path.join(os.path.expanduser("~"), ".catanatron", "server_key"),
        help="file with the server's Ed25519 identity key (created if missing)",
    )
    parser.add_argument(
        "--archive", help="directory to keep finished games' reveals in"
    )
    parser.add_argument(
        "--bot",
        action="append",
        default=[],
        metavar="[NAME=]SOURCE",
        help="register a bot class for seats, as in catanatron-play",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    server = GameServer(SigningKey.load_or_create(args.key), archive_dir=args.archive)
    for declaration in args.bot:
        name, _, source = declaration.rpartition("=")
        REGISTRY.register_source(source, name or None)

    async def run() -> None:
        ws_server = await run_server(server, args.host, args.port)
        print(
            f"catanatron-server on http://{args.host}:{args.port}/ "
            f"(key {server.key.public_hex})",
            file=sys.stderr,
        )
        stop = asyncio.get_running_loop().create_future()
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_running_loop().add_signal_handler(sig, stop.set_result, None)
        await stop
        ws_server.close()
        await ws_server.wait_closed()

    asyncio.run(run())
    return 0
