"""Server bots in worker processes: killable, capped, and off the event loop.

Each room with server bots gets one worker process (rooms decide one move at
a time, so one process serves all of a room's bots). A decision that
overruns its time kills the worker, which stops the computation for real
(threads cannot be stopped) along with anything the bot started; the next
decision starts a fresh worker. Workers are forked from a forkserver that
preloads Catanatron, so starting one is cheap, and they are daemonic and
killed on shutdown. Inside a worker, bots that would fan out into their own
process pools (GreedyPlayouts) run single-process instead.

The parent talks to a worker over a pipe, one pickled tuple per message::

    ("seat", index, key, recipe, spec, color, seed)   declare a seat's bot
    ("decide", index, view, playable, supply)          -> ("decided", choice)
                                                       or ("failed", reason)
    ("step", index, view_before, action, supply)       an exec bot observes
    ("finish", index, view, supply, winner)            after(); stop the bot
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import multiprocessing
import os
import threading
import time

logger = logging.getLogger(__name__)

_CONTEXT = None


def context():
    """The multiprocessing context workers are made with (a forkserver
    preloading this module, so a worker starts in milliseconds and a fork
    never copies the server's threads)."""
    global _CONTEXT
    if _CONTEXT is None:
        method = (
            "forkserver"
            if "forkserver" in multiprocessing.get_all_start_methods()
            else "spawn"
        )
        _CONTEXT = multiprocessing.get_context(method)
        if method == "forkserver":
            # Workers fork from here with Catanatron and its bots imported.
            _CONTEXT.set_forkserver_preload(
                [
                    "catanatron.server.botproc",
                    "catanatron.server.bot",
                    "catanatron.server.sampling",
                    "catanatron.players",
                ]
            )
    return _CONTEXT


def recipe_for(key: str, player_class) -> tuple:
    """How a worker rebuilds a registry entry: as registered from a source,
    imported by name, or built in."""
    from catanatron.registry import REGISTRY

    source = getattr(REGISTRY, "sources", {}).get(key)
    if source is not None:
        return ("source",) + tuple(source)
    module = player_class.__module__
    if module.startswith("catanatron.players") or module == "catanatron.models.player":
        return ("builtin",)
    if module == "__main__" or module.startswith("catanatron_source_"):
        raise ValueError(f"{key}: its class cannot be loaded in a worker process")
    return ("import", module, player_class.__qualname__)


def _install(key: str, recipe: tuple) -> None:
    from catanatron.players import register_builtins
    from catanatron.registry import REGISTRY

    register_builtins()
    kind = recipe[0]
    if kind == "source":
        source, base_dir = recipe[1], recipe[2]
        REGISTRY.register_source(source, key, base_dir)
    elif kind == "import":
        target = importlib.import_module(recipe[1])
        for part in recipe[2].split("."):
            target = getattr(target, part)
        REGISTRY.register(key, target, replace=True)


def worker_main(conn) -> None:
    """A worker's loop: one request at a time, until the pipe closes."""
    from catanatron.models.player import Color
    from catanatron.players import playouts
    from catanatron.server.bot import ViewBot
    from catanatron.server.config import SUPPLIES

    playouts.USE_MULTIPROCESSING = False  # no process pools inside a worker
    bots: dict = {}
    while True:
        try:
            message = conn.recv()
        except (EOFError, OSError):
            break
        kind, index = message[0], message[1] if len(message) > 1 else None
        try:
            if kind == "ping":
                conn.send(("pong",))
            elif kind == "seat":
                _, index, key, recipe, spec, color, seed = message
                _install(key, recipe)
                bots[index] = ViewBot(spec, Color[color], seed=seed)
            elif kind == "decide":
                _, index, view, playable, supply = message
                choice = bots[index].decide(view, playable, SUPPLIES[supply])
                conn.send(("decided", choice))
            elif kind == "step":
                _, index, view, action, supply = message
                bots[index].observe(view, action, SUPPLIES[supply])
            elif kind == "finish":
                _, index, view, supply, winner = message
                bot = bots.pop(index, None)
                if bot is not None:
                    bot.finish(view, SUPPLIES[supply], winner)
        except BaseException as error:  # a bot's SystemExit is a failed decision
            if kind == "decide":
                try:
                    conn.send(("failed", f"{type(error).__name__}: {error}"[:200]))
                except (OSError, ValueError):
                    break
    for bot in bots.values():
        bot.close()


class BotWorker:
    """One room's worker process, as seen from the server's event loop."""

    def __init__(self):
        self.process = None
        self.conn = None
        self.seats: dict = {}  # index -> the "seat" message, replayed on restart

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.is_alive()

    def declare(
        self, index: int, key: str, recipe: tuple, spec: str, color: str
    ) -> None:
        self.seats[index] = (
            "seat",
            index,
            key,
            recipe,
            spec,
            color,
            int.from_bytes(os.urandom(4), "big"),
        )
        if self.alive:
            self._send(self.seats[index])

    def start(self) -> None:
        if self.alive:
            return
        ctx = context()
        parent, child = ctx.Pipe()
        process = ctx.Process(
            target=worker_main, args=(child,), daemon=True, name="catanatron-bots"
        )
        process.start()
        child.close()
        self.process, self.conn = process, parent
        for message in self.seats.values():
            self._send(message)

    def _send(self, message) -> bool:
        try:
            self.conn.send(message)
            return True
        except (OSError, ValueError, AttributeError):
            self.kill()
            return False

    def post(self, message) -> None:
        """Send something that expects no reply (if the worker is up)."""
        if self.alive:
            self._send(message)

    #: How long a fresh worker may take to start (not counted against the
    #: decision that needed it).
    STARTUP_S = 30.0

    async def call(self, message, timeout: float):
        """Send ``message`` and wait for the reply; on timeout the worker is
        killed (and restarted for the next call)."""
        if not self.alive:
            self.start()
            # Seats are declared; the pong says the worker has them ready.
            await self._exchange(("ping",), self.STARTUP_S)
        return await self._exchange(message, timeout)

    async def _exchange(self, message, timeout: float):
        if not self._send(message):
            raise RuntimeError("the bot worker died")
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        fd = self.conn.fileno()

        def readable():
            if not future.done():
                future.set_result(None)

        loop.add_reader(fd, readable)
        try:
            await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            self.kill()
            raise
        finally:
            try:
                loop.remove_reader(fd)
            except (ValueError, OSError):
                pass
        try:
            return self.conn.recv()
        except (EOFError, OSError, AttributeError):
            self.kill()
            raise RuntimeError("the bot worker died")

    def kill(self) -> None:
        process, conn = self.process, self.conn
        self.process = self.conn = None
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass
        if process is not None and process.is_alive():
            process.kill()
            # Reap it off the event loop.
            threading.Thread(target=process.join, daemon=True).start()

    def stop(self, grace: float = 2.0) -> None:
        """Let the worker finish its bots (exec bots get ``after``), then
        make sure it is gone. Blocking; call it off the event loop."""
        process, conn = self.process, self.conn
        self.process = self.conn = None
        if conn is not None:
            try:
                conn.close()  # the worker's loop ends on EOF
            except OSError:
                pass
        if process is not None:
            process.join(grace)
            if process.is_alive():
                process.kill()
                process.join(0.5)


def pace(timings, rng) -> None:
    """Sleep as long as a typical real decision took (see ViewBot)."""
    if timings:
        time.sleep(rng.choice(list(timings)))
