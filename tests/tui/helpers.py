"""Driving catanatron-tui headlessly in tests."""

import asyncio
import json
import os
import random
import socket
import time

from catanatron.server.maps import parse_map

MAPS_DIR = os.path.join(os.path.dirname(__file__), "..", "maps")
SHAPES = ["hexagon3", "extension", "rectangle5x4", "blob25", "blob25_stringy"]
SNAPSHOTS = os.path.join(os.path.dirname(__file__), "snapshots")
TIMEOUT = 120


def map_spec_file(name):
    with open(os.path.join(MAPS_DIR, name + ".json")) as file:
        return {"tiles": json.load(file)["tiles"]}


def load_map(name):
    return parse_map(map_spec_file(name))


def run(coroutine, timeout=TIMEOUT):
    async def bounded():
        return await asyncio.wait_for(coroutine, timeout)

    return asyncio.run(bounded())


async def settle(pilot, condition, timeout=30.0):
    """Let the app run until ``condition()`` holds."""
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting for the app")
        await pilot.pause(0.01)


async def idle(pilot, app, timeout=30.0):
    """Until the local game waits for a person (or is over), and is drawn."""
    session = app.session
    await settle(
        pilot,
        lambda: app.thinking is None
        and (session.over or session.person_to_move())
        and app._timer is None,
        timeout,
    )
    await pilot.pause(0.1)  # one batched redraw


def free_port():
    """A free localhost port in 20000-29999."""
    rng = random.Random()
    for port in rng.sample(range(20000, 30000), 200):
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("no free port")


def snapshot(name, text):
    """Compare ``text`` with the stored snapshot (UPDATE_SNAPSHOTS=1 rewrites it)."""
    path = os.path.join(SNAPSHOTS, name + ".txt")
    text = "\n".join(line.rstrip() for line in text.splitlines()) + "\n"
    if os.environ.get("UPDATE_SNAPSHOTS"):
        os.makedirs(SNAPSHOTS, exist_ok=True)
        with open(path, "w") as file:
            file.write(text)
    assert os.path.exists(path), f"no snapshot {path}; run with UPDATE_SNAPSHOTS=1"
    with open(path) as file:
        assert file.read() == text, f"{name} differs from {path}"
