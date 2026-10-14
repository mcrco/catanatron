"""Running a real server on localhost for a test."""

import asyncio
import contextlib

from catanatron.models.player import Color
from catanatron.server.app import GameServer, run_server
from catanatron.server.bot import ViewBot
from catanatron.server.client import GameClient

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
        ws_server.close()
        await ws_server.wait_closed()


def bot_client(url, spec="F", **kwargs):
    """A GameClient whose seat is played by a registry bot from its view."""
    kwargs.setdefault("transcript_dir", None)
    client = GameClient(url, name=spec, **kwargs)
    bots = {}

    def decide(view, playable_actions):
        color = client.table.color
        bot = bots.get(color) or bots.setdefault(
            color, ViewBot(spec, Color[color], seed=1)
        )
        return bot.decide(view, playable_actions, client.table.supply)

    return client, decide
