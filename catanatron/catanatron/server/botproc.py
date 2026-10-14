"""Server bots in worker processes: killable, capped, and off the event loop.

Each room with server bots gets one worker process (rooms decide one move at
a time, so one process serves all of a room's bots). A decision that
overruns its time kills the worker, which stops the computation for real
(threads cannot be stopped) along with anything the bot started; the next
decision starts a fresh worker. Workers are forked from a forkserver that
preloads Catanatron, so starting one is cheap; they are daemonic, get a few
seconds to finish their bots when a game ends, and are killed on shutdown.
Inside a worker, bots that would fan out into their own process pools
(GreedyPlayouts) run single-process instead.

The parent talks to a worker over a pipe, one pickled tuple per message::

    ("seat", index, key, recipe, spec, color, seed)   declare a seat's bot
    ("decide", index, view, playable, supply)          -> ("decided", choice)
                                                       or ("failed", reason)
    ("step", index, view_before, action, supply)       an exec bot observes
    ("finish", index, view, supply, winner)            after(); stop the bot
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import logging
import multiprocessing
import os
import sys
from multiprocessing import forkserver, resource_tracker
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

_CONTEXT = None
_CONTEXT_LOCK = threading.Lock()


@contextlib.contextmanager
def _real_stdio():
    """Standard streams with real file descriptors, for the helper
    processes' sake.

    The resource tracker hands its child ``sys.stderr.fileno()``. An app
    that captures output (Textual, pytest) replaces the streams with objects
    whose fileno() is invalid, and starting the tracker then fails ("bad
    value(s) in fds_to_keep"). For as long as the helpers start, the streams
    are the process's originals, or /dev/null when those are gone too.
    """
    saved = sys.stdout, sys.stderr
    opened = []
    for name, original in (("stdout", sys.__stdout__), ("stderr", sys.__stderr__)):
        if _has_fd(getattr(sys, name)):
            continue
        if _has_fd(original):
            setattr(sys, name, original)
        else:
            stream = open(os.devnull, "w")
            opened.append(stream)
            setattr(sys, name, stream)
    try:
        yield
    finally:
        sys.stdout, sys.stderr = saved
        for stream in opened:
            stream.close()


def _has_fd(stream) -> bool:
    try:
        return stream is not None and stream.fileno() >= 0
    except Exception:  # io.UnsupportedOperation, ValueError, AttributeError...
        return False


def context():
    """The multiprocessing context workers are made with (a forkserver
    preloading Catanatron, so a worker starts in milliseconds and a fork
    never copies the server's threads; spawn where there is no forkserver).

    The resource tracker and the forkserver are started here, once, with
    real standard streams (see :func:`_real_stdio`), whatever the host app
    has done to ``sys.stdout``/``sys.stderr`` since.
    """
    global _CONTEXT
    with _CONTEXT_LOCK:
        if _CONTEXT is None:
            method = (
                "forkserver"
                if "forkserver" in multiprocessing.get_all_start_methods()
                else "spawn"
            )
            ctx = multiprocessing.get_context(method)
            with _real_stdio():
                if method == "forkserver":
                    # Workers fork from here with Catanatron and its bots
                    # imported.
                    ctx.set_forkserver_preload(
                        [
                            "catanatron.server.botproc",
                            "catanatron.server.bot",
                            "catanatron.server.sampling",
                            "catanatron.players",
                        ]
                    )
                    forkserver.ensure_running()
                else:
                    resource_tracker.ensure_running()
            _CONTEXT = ctx
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
                _reply(conn, ("pong",))
            elif kind == "seat":
                _, index, key, recipe, spec, color, seed = message
                _install(key, recipe)
                bots[index] = ViewBot(spec, Color[color], seed=seed)
            elif kind == "decide":
                _, index, view, playable, supply = message
                choice = bots[index].decide(view, playable, SUPPLIES[supply])
                _reply(conn, ("decided", choice))
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
                _reply(conn, ("failed", f"{type(error).__name__}: {error}"[:200]))
    for bot in bots.values():
        bot.close()


def _reply(conn, message) -> None:
    """Answer the server, if it still listens. A server that stopped
    listening (the game ended mid-decision) may have queued ``finish``
    messages behind the decision: the loop goes on to them, and ends when
    the pipe runs dry."""
    try:
        conn.send(message)
    except (OSError, ValueError):
        pass


class BotWorker:
    """One room's worker process, as seen from the server's event loop."""

    def __init__(self):
        self.process = None
        self.conn = None
        self.seats: dict = {}  # index -> the "seat" message, replayed on restart
        self.stopped = False  # by stop(): the game is over, no worker restarts
        self._waiting = None  # (future, fd) while a call waits on the pipe

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
        if self.stopped:
            raise RuntimeError("the game is over")
        ctx = context()
        parent, child = ctx.Pipe()
        process = ctx.Process(
            target=worker_main, args=(child,), daemon=True, name="catanatron-bots"
        )
        try:
            with _real_stdio():  # in case a helper process has to restart
                process.start()
        except BaseException:
            parent.close()
            raise
        finally:
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
            try:
                self.start()
            except Exception as error:
                self.kill()
                raise RuntimeError(f"the bot worker did not start: {error}") from error
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
        self._waiting = future, fd
        try:
            await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            self.kill()
            raise
        finally:
            self._unwait()
        try:
            return self.conn.recv()
        except (EOFError, OSError, AttributeError):
            self.kill()
            raise RuntimeError("the bot worker died")

    def _unwait(self) -> None:
        """Take the pipe out of the event loop (before it is closed: a
        closed descriptor's number can be reused), and wake whoever waited
        on it."""
        waiting, self._waiting = self._waiting, None
        if waiting is None:
            return
        future, fd = waiting
        try:
            asyncio.get_running_loop().remove_reader(fd)
        except (RuntimeError, ValueError, OSError):
            pass
        if not future.done():
            future.set_exception(RuntimeError("the bot worker was stopped"))

    def kill(self) -> None:
        self._unwait()
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

    #: How long a stopping worker has to finish its bots (an exec bot's
    #: ``after`` may wait on a decision still running) before it is killed.
    STOP_GRACE_S = 5.0

    def stop(self, grace: Optional[float] = None) -> None:
        """The game is over: let the worker finish its bots (exec bots get
        ``after``), then make sure it is gone. It is let go at once (a later
        :meth:`kill` cannot cut its grace short, and :meth:`start` refuses)
        and winds down on a thread of its own."""
        self._unwait()
        process, conn = self.process, self.conn
        self.process = self.conn = None
        self.stopped = True
        if process is not None or conn is not None:
            threading.Thread(
                target=_wind_down,
                args=(process, conn, self.STOP_GRACE_S if grace is None else grace),
                daemon=True,
            ).start()


def _wind_down(process, conn, grace: float) -> None:
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
