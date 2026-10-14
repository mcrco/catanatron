"""The WebSocket server: a lobby, rooms, and the browser client.

Routes (one port, HTTP and WebSocket):

    GET /                  the browser client (static/index.html)
    GET /static/FILE       its assets
    GET /api/rooms         rooms, as JSON
    GET /api/players       the registry's bots, as JSON
    GET /api/server        the server's public key, protocol version and the
                           SHA-256 of the app.js it serves
    WS  /lobby             create_room, list_rooms, players
    WS  /room/ROOM_ID      join a seat, or spectate, and play

The API answers any origin, so a copy of the browser client served from
elsewhere (see the README: that is what makes browser seats verifiable) can
use it.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import ipaddress
import json
import logging
import os
import secrets
import signal
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any, Optional, Sequence

from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from catanatron.registry import REGISTRY, describe
from catanatron.server.config import ConfigError, GameConfig
from catanatron.server.crypto import SigningKey
from catanatron.server.hashing import MAX_SAFE_INTEGER
from catanatron.server.messages import PROTOCOL_VERSION
from catanatron.server.room import Outbox, Peer, Room, RoomError, RoomLimits

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


@dataclass(frozen=True)
class ServerLimits:
    max_rooms: int = 500
    max_connections: int = 2000
    connections_per_ip: int = 64
    connections_per_room: int = 64
    #: Rooms one address may create per minute, and keep waiting at once.
    rooms_per_minute_per_ip: int = 10
    waiting_rooms_per_ip: int = 10
    #: Waiting rooms nobody is connected to are dropped after this long.
    idle_waiting_s: float = 600.0
    #: Games in which no remote seat has been connected for this long are
    #: aborted.
    abandoned_playing_s: float = 1800.0
    #: Finished and aborted rooms are forgotten after this long.
    finished_ttl_s: float = 3600.0
    #: Games with server bots (each has a bot worker process) at once, and
    #: server bot seats one address may have in rooms not yet ended.
    max_bot_rooms: int = 32
    bot_seats_per_ip: int = 12
    #: An address blamed this often within an hour (for withholding seeds)
    #: may not take seats anywhere for that hour; a blamed key may not take
    #: a seat for that hour at all.
    blames_per_ip: int = 3
    #: Messages per second a connection may send (and its burst).
    messages_per_s: float = 30.0
    message_burst: int = 60
    room: RoomLimits = field(default_factory=RoomLimits)


class TooMuch(Exception):
    """A request over one of the server's limits."""


class _Floats(ValueError):
    pass


def parse_request(text: Any) -> dict:
    """A client message: a JSON object whose numbers are all integers that
    every client can represent exactly."""

    def no_float(token):
        raise _Floats("numbers must be integers")

    def safe_int(token):
        if len(token) > 17 or abs(int(token)) > MAX_SAFE_INTEGER:
            raise _Floats("integers must be at most 2**53 - 1")
        return int(token)

    try:
        request = json.loads(
            text, parse_float=no_float, parse_constant=no_float, parse_int=safe_int
        )
    except _Floats as error:
        raise RoomError("bad_request", str(error))
    except (ValueError, TypeError, RecursionError) as error:
        raise RoomError("bad_request", f"not JSON: {error}")
    if not isinstance(request, dict):
        raise RoomError("bad_request", "expected a JSON object")
    return request


class GameServer:
    def __init__(
        self,
        key: Optional[SigningKey] = None,
        *,
        archive_dir: Optional[str] = None,
        limits: ServerLimits = ServerLimits(),
        room_class: type = Room,
        trusted_proxies: Sequence[str] = (),
    ):
        from catanatron.players import register_builtins

        register_builtins()
        self.key = key or SigningKey.generate()
        self.archive_dir = archive_dir
        self.limits = limits
        self.room_class = room_class
        #: Peers (a reverse proxy) whose X-Forwarded-For is believed: behind
        #: one, every per-address limit and ban would otherwise apply to the
        #: proxy, so to everyone at once.
        self.trusted_proxies = frozenset(
            _normalize(address) for address in trusted_proxies
        )
        self.rooms: dict = {}
        self.blames: dict = defaultdict(deque)
        self.banned_keys: dict = {}
        self.connections = 0
        self.by_ip: dict = defaultdict(int)
        self.by_room: dict = defaultdict(int)
        self.created_by: dict = defaultdict(deque)
        self.room_owner: dict = {}
        self.janitor: Optional[asyncio.Task] = None
        self.app_sha256 = _file_sha256("app.js")

    # ===== rooms =====
    def create_room(
        self,
        config: Any,
        seats: Optional[list] = None,
        name: str = "",
        address: Optional[str] = None,
    ) -> Room:
        game_config = GameConfig.from_json(config)
        if game_config.friendly_robber:
            raise ConfigError(
                "friendly_robber is not offered here: whether a robbery is legal "
                "would depend on (and reveal) hidden victory point cards"
            )
        if seats is not None and not isinstance(seats, list):
            raise ConfigError("seats must be a list")
        seats = list(seats or [])
        if len(seats) > game_config.num_players:
            raise ConfigError(
                f"{len(seats)} seat specs for {game_config.num_players} seats"
            )
        if address is not None:
            self._admit_creation(address)
        self.admit_bots(address, sum(1 for spec in seats if spec is not None))
        self._evict()
        room_id = secrets.token_urlsafe(6).replace("-", "x").replace("_", "y")
        room = self.room_class(
            room_id,
            game_config,
            self.key,
            name=name,
            on_finish=self._archive,
            limits=self.limits.room,
            on_blame=self._blame,
            is_banned=self._banned,
        )
        for index, spec in enumerate(seats):
            if spec is not None:
                room.set_bot(index, spec, address)
        self.rooms[room_id] = room
        if address is not None:
            self.room_owner[room_id] = address
            self.created_by[address].append(time.time())
        return room

    def _admit_creation(self, address: str) -> None:
        recent = self.created_by[address]
        while recent and recent[0] < time.time() - 60:
            recent.popleft()
        if len(recent) >= self.limits.rooms_per_minute_per_ip:
            raise TooMuch("too many rooms created from your address; wait a minute")
        waiting = sum(
            1
            for room_id, owner in self.room_owner.items()
            if owner == address
            and room_id in self.rooms
            and self.rooms[room_id].status in ("waiting", "revealing")
        )
        if waiting >= self.limits.waiting_rooms_per_ip:
            raise TooMuch("you have too many rooms waiting for players")

    def admit_bots(self, address: Optional[str], count: int) -> None:
        """Refuse more server bot seats than the limits allow: each game with
        bots runs a worker process, and one address must not fill them all."""
        if not count:
            return
        live = [
            r for r in self.rooms.values() if r.status not in ("finished", "aborted")
        ]
        if sum(1 for r in live if r.bot_seats()) >= self.limits.max_bot_rooms:
            raise TooMuch("too many games with server bots are running; try later")
        if address is not None:
            mine = sum(
                1 for r in live for seat in r.bot_seats() if seat.address == address
            )
            if mine + count > self.limits.bot_seats_per_ip:
                raise TooMuch("you have too many server bots seated")

    def _blame(self, address: Optional[str], key: Optional[str] = None) -> None:
        now = time.time()
        if address:
            self.blames[address].append(now)
        if key:
            self.banned_keys[key] = now

    def _banned(self, address: Optional[str], key: Optional[str] = None) -> bool:
        now = time.time()
        if key and now - self.banned_keys.get(key, -1e18) < 3600:
            return True
        recent = self.blames.get(address) if address else None
        if not recent:
            return False
        while recent and recent[0] < now - 3600:
            recent.popleft()
        return len(recent) >= self.limits.blames_per_ip

    def _evict(self) -> None:
        self.sweep()
        if len(self.rooms) >= self.limits.max_rooms:
            done = sorted(
                (r for r in self.rooms.values() if r.status in ("finished", "aborted")),
                key=lambda r: r.created,
            )
            for room in done[: len(self.rooms) - self.limits.max_rooms + 1]:
                self._forget(room)
        if len(self.rooms) >= self.limits.max_rooms:
            raise TooMuch("the server is full")

    def _forget(self, room: Room) -> None:
        self.rooms.pop(room.id, None)
        self.room_owner.pop(room.id, None)

    def sweep(self, now: Optional[float] = None) -> None:
        """Expire idle waiting rooms, abandoned games and old results."""
        now = time.time() if now is None else now
        limits = self.limits
        for room in list(self.rooms.values()):
            if room.has_peers():
                room.last_seen = now
            absent = now - room.last_seen
            if room.status == "waiting" and absent > limits.idle_waiting_s:
                room.abort("nobody came")
                self._forget(room)
            elif room.status == "playing" and absent > limits.abandoned_playing_s:
                if any(s.kind == "remote" for s in room.seats):
                    room.abort("every remote seat left the game")
            elif (
                room.status in ("finished", "aborted")
                and room.ended is not None
                and now - room.ended > limits.finished_ttl_s
            ):
                self._forget(room)

    async def _janitor(self) -> None:
        while True:
            await asyncio.sleep(10)
            try:
                self.sweep()
            except Exception:
                logger.exception("sweeping rooms failed")

    def start(self) -> None:
        if self.janitor is None:
            self.janitor = asyncio.get_running_loop().create_task(self._janitor())

    def shutdown(self) -> None:
        """Abort every game still going (revealing the server seed) and stop
        every bot worker."""
        if self.janitor is not None:
            self.janitor.cancel()
        for room in list(self.rooms.values()):
            # Killed first, so that the abort has no worker to give grace to
            # (a room that ended earlier keeps the grace it was given).
            room.kill_bots()
            room.abort("the server is shutting down")
            if room.task is not None:
                room.task.cancel()

    def _archive(self, room: Room) -> None:
        """Keep the server's own record of a finished game."""
        if not self.archive_dir or room.game is None:
            return
        os.makedirs(self.archive_dir, exist_ok=True)
        record = {
            "room": room.describe(),
            "server_key": self.key.public_hex,
            "reveal": room._reveal(),
            "commitments": room.state_commitments,
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
        if path.count("/") == 1 and os.path.splitext(path)[1] in (
            ".js",
            ".css",
            ".html",
        ):
            # The page loads its assets by relative path, so that a copy of
            # static/ opened from a file works the same.
            return _file_response(path[1:])
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
                    "app_sha256": self.app_sha256,
                }
            )
        return _json_response({"error": "not found"}, HTTPStatus.NOT_FOUND)

    # ===== WebSocket =====
    def address_of(self, websocket: ServerConnection) -> str:
        """Who is on the other end, for per-address limits and bans: the
        peer, or (when the peer is a trusted proxy) the last address it says
        it forwarded for. IPv6 addresses count by /64 (one host's share)."""
        peer = _normalize(_address(websocket))
        if peer in self.trusted_proxies and websocket.request is not None:
            forwarded = websocket.request.headers.get_all("X-Forwarded-For")
            hops = [h.strip() for value in forwarded for h in value.split(",")]
            for hop in reversed(hops):
                address = _normalize(hop)
                if address not in self.trusted_proxies:
                    return address
        return peer

    async def handler(self, websocket: ServerConnection) -> None:
        path = websocket.request.path.split("?", 1)[0] if websocket.request else ""
        address = self.address_of(websocket)
        room = None
        if path.startswith("/room/"):
            room = self.rooms.get(path[len("/room/") :].strip("/"))
            if room is None:
                await websocket.close(4004, "no such room")
                return
        elif path != "/lobby":
            await websocket.close(4004, "unknown path")
            return
        limits = self.limits
        if (
            self.connections >= limits.max_connections
            or self.by_ip[address] >= limits.connections_per_ip
            or (
                room is not None
                and self.by_room[room.id] >= limits.connections_per_room
            )
        ):
            await websocket.close(1013, "too many connections; try again later")
            return
        self.connections += 1
        self.by_ip[address] += 1
        if room is not None:
            self.by_room[room.id] += 1
        try:
            if room is None:
                await self._lobby(websocket, address)
            else:
                await self._room(websocket, room, address)
        finally:
            self.connections -= 1
            self.by_ip[address] -= 1
            if not self.by_ip[address]:
                del self.by_ip[address]
            if room is not None:
                self.by_room[room.id] -= 1
                if not self.by_room[room.id]:
                    del self.by_room[room.id]

    async def _lobby(self, websocket: ServerConnection, address: str) -> None:
        async for text in websocket:
            try:
                request = parse_request(text)
                kind = request.get("type")
                if kind == "create_room":
                    room = self.create_room(
                        request.get("config"),
                        request.get("seats"),
                        str(request.get("name") or ""),
                        address=address,
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
                    reply = {"type": "error", "message": "unknown request type"}
            except (ValueError, AttributeError, RoomError, TooMuch) as error:
                # Messages say what was wrong, not what was sent (and are
                # short whatever they say).
                reply = {"type": "error", "message": str(error)[:200]}
            await websocket.send(json.dumps(reply))

    async def _room(
        self, websocket: ServerConnection, room: Room, address: str
    ) -> None:
        peer = Peer(websocket, room.limits.max_queue, address)
        outbox: Outbox = room.connect(peer)
        seat, watching = None, False
        tokens, last = float(self.limits.message_burst), time.monotonic()
        try:
            async for text in websocket:
                now = time.monotonic()
                tokens = min(
                    float(self.limits.message_burst),
                    tokens + (now - last) * self.limits.messages_per_s,
                )
                last = now
                if peer.closed:
                    continue  # closing: drain what is in flight, unread
                if tokens < 1:
                    # Close, and keep draining so the close handshake is read.
                    peer.kick(1008, "too many messages")
                    continue
                tokens -= 1
                request: Optional[dict] = None
                try:
                    request = parse_request(text)
                    kind = request.get("type")
                    if kind in ("join", "rejoin") and seat is None and not watching:
                        if kind == "join":
                            seat = room.join(outbox, peer, request)
                        else:
                            seat = room.rejoin(outbox, peer, request)
                        assert seat.outbox is not None
                        outbox = seat.outbox
                    elif kind == "spectate" and seat is None and not watching:
                        room.spectate(outbox, request)
                        watching = True
                    elif kind == "reveal" and seat is not None:
                        room.on_reveal(seat, peer, request)
                    elif kind == "action" and seat is not None:
                        room.on_action(seat, peer, request)
                    elif kind == "seat_bot":
                        try:
                            self.admit_bots(peer.address, 1)
                        except TooMuch as error:
                            raise RoomError("too_many_bots", str(error))
                        room.set_bot(
                            request.get("seat"), request.get("spec"), peer.address
                        )
                    elif kind == "ping":
                        pass
                    else:
                        raise RoomError(
                            "bad_request", f"unexpected {str(kind)[:40]!r} here"
                        )
                except RoomError as error:
                    if (
                        seat is not None
                        and seat.outbox is not None
                        and seat.outbox.peer is not peer
                    ):
                        continue  # a superseded connection: nothing more to say
                    room.error(outbox, error.code, str(error), request)
        except ConnectionClosed:
            pass
        finally:
            room.disconnect(outbox, peer)


def _address(websocket: ServerConnection) -> str:
    remote = getattr(websocket, "remote_address", None)
    if isinstance(remote, (tuple, list)) and remote:
        return str(remote[0])
    return str(remote)


def _normalize(address: str) -> str:
    """An address as limits count it: IPv4 as is, IPv6 by its /64."""
    try:
        ip = ipaddress.ip_address(address.strip().strip("[]"))
    except ValueError:
        return address.strip()[:64]
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return str(ip.ipv4_mapped)
        return str(ipaddress.IPv6Network((ip, 64), strict=False))
    return str(ip)


def _players() -> list:
    return [
        describe(key, cls)
        for key, cls in sorted(REGISTRY.items())
        if getattr(cls, "IS_BOT", True)
    ]


def _file_sha256(name: str) -> Optional[str]:
    try:
        with open(os.path.join(STATIC_DIR, name), "rb") as file:
            return hashlib.sha256(file.read()).hexdigest()
    except OSError:
        return None


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
            ("Access-Control-Allow-Origin", "*"),
            ("Connection", "close"),
        ]
    )
    return Response(status.value, status.phrase, headers, body)


async def run_server(server: GameServer, host: str = "127.0.0.1", port: int = 8765):
    """Start serving; returns the websockets server (``.close()`` to stop)."""
    server.start()
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
    parser.add_argument(
        "--trusted-proxy",
        action="append",
        default=[],
        metavar="ADDRESS",
        help="a reverse proxy in front of the server: take client addresses "
        "from its X-Forwarded-For (otherwise every per-address limit and ban "
        "applies to the proxy, so to everyone behind it)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    server = GameServer(
        SigningKey.load_or_create(args.key),
        archive_dir=args.archive,
        trusted_proxies=args.trusted_proxy,
    )
    for declaration in args.bot:
        name, _, source = declaration.rpartition("=")
        REGISTRY.register_source(source, name or None)

    async def run() -> None:
        ws_server = await run_server(server, args.host, args.port)
        print(
            f"catanatron-server on http://{args.host}:{args.port}/\n"
            f"  server key {server.key.public_hex}\n"
            f"  app.js sha256 {server.app_sha256}",
            file=sys.stderr,
        )
        stop = asyncio.get_running_loop().create_future()
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_running_loop().add_signal_handler(sig, stop.set_result, None)
        await stop
        # Bot decisions run in daemon threads, so nothing here waits on them.
        server.shutdown()
        ws_server.close()
        try:
            await asyncio.wait_for(ws_server.wait_closed(), 5)
        except asyncio.TimeoutError:
            pass

    asyncio.run(run())
    return 0
