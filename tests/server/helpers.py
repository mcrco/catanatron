"""Running a real server on localhost for a test."""

import asyncio
import contextlib
import json

import websockets

from catanatron.models.player import Color
from catanatron.server.app import GameServer, run_server
from catanatron.server.bot import ViewBot
from catanatron.server.client import GameClient
from catanatron.server.crypto import (
    SigningKey,
    client_seed_commitment,
    join_statement,
    random_hex,
)
from catanatron.server.messages import PROTOCOL_VERSION

TEST_TIMEOUT = 300


def run(coroutine, timeout=TEST_TIMEOUT):
    async def bounded():
        return await asyncio.wait_for(coroutine, timeout)

    return asyncio.run(bounded())


@contextlib.asynccontextmanager
async def serving(**kwargs):
    """A GameServer listening on a free localhost port; yields (server, base URL)."""
    server = GameServer(**kwargs)
    ws_server = await run_server(server, "127.0.0.1", 0)
    port = ws_server.sockets[0].getsockname()[1]
    try:
        yield server, f"ws://127.0.0.1:{port}"
    finally:
        server.shutdown()
        ws_server.close()
        await ws_server.wait_closed()


class LaxClient(GameClient):
    """A client that does not check, as it plays, that the game is the one it
    joined: so a dishonest server's game runs to the end and what is tested
    is the verifier."""

    def _check_commitments(self, body):
        pass

    def _check_before(self, body):
        pass


def bot_client(url, spec="F", client_class=GameClient, **kwargs):
    """A client whose seat is played by a registry bot from its view."""
    kwargs.setdefault("transcript_dir", None)
    client = client_class(url, name=spec, **kwargs)
    bots = {}

    def decide(view, playable_actions):
        color = client.table.color
        bot = bots.get(color) or bots.setdefault(
            color, ViewBot(spec, Color[color], seed=1)
        )
        return bot.decide(view, playable_actions, client.table.supply)

    return client, decide


async def raw_connect(url):
    """A bare connection to a room: (websocket, hello body)."""
    ws = await websockets.connect(url)
    hello = json.loads(await ws.recv())["body"]
    return ws, hello


def join_request(hello, key=None, seat=None, seed=None, pop_key=None, name="raw"):
    """A protocol-correct join (unless ``pop_key`` signs it instead)."""
    key = key or SigningKey.generate()
    seed = seed or random_hex()
    room = hello["room"]
    commitment = client_seed_commitment(room["id"], key.public_hex, seed)
    statement = join_statement(
        room["id"],
        hello["server_key"],
        room["seed_commitment"],
        seat,
        commitment,
        hello["nonce"],
    )
    request = {
        "type": "join",
        "protocol_version": PROTOCOL_VERSION,
        "name": name,
        "seat": seat,
        "public_key": key.public_hex,
        "client_commitment": commitment,
        "pop": (pop_key or key).sign(statement),
    }
    return request, key, seed


async def next_body(ws, *kinds, timeout=10):
    """The next message body of one of ``kinds`` (any, if none given)."""
    while True:
        body = json.loads(await asyncio.wait_for(ws.recv(), timeout))["body"]
        if not kinds or body.get("type") in kinds:
            return body
