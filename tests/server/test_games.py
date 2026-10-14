"""Whole games over real WebSockets on localhost."""

import asyncio
import json
import os
import subprocess
import sys

import pytest

from catanatron.server.client import GameClient, create_room
from catanatron.server.maps import map_spec
from catanatron.server.verify import verify_transcripts
from catanatron.models.map import build_map

from tests.server.helpers import bot_client, run, serving

FAST = {"vps_to_win": 6, "decision_timeout_ms": 20_000}
MAPS_DIR = os.path.join(os.path.dirname(__file__), "..", "maps")


@pytest.mark.parametrize("num_players", [2, 3, 4, 5, 6])
def test_server_bots_play_a_whole_game_that_a_spectator_can_verify(num_players):
    async def scenario():
        async with serving() as (server, base):
            seats = ["F", "R"] * 3
            room = await create_room(
                base, {"num_players": num_players, **FAST}, seats[:num_players]
            )
            assert room["status"] == "playing", "a room full of bots starts at once"
            watcher = GameClient(
                f"{base}/room/{room['id']}", spectate=True, transcript_dir=None
            )
            result = await watcher.play()
            return server, room, watcher, result

    server, room, watcher, result = run(scenario())
    assert result.verified, result.report.errors
    assert result.color is None
    game = server.rooms[room["id"]].game
    assert result.winning_color == (
        game.winning_color().value if game.winning_color() else None
    )
    assert len(game.state.colors) == num_players
    assert result.report.counts["actions"] == len(game.state.action_records)


def test_remote_bots_play_and_every_transcript_verifies(tmp_path):
    async def scenario():
        async with serving() as (server, base):
            room = await create_room(
                base, {"num_players": 4, **FAST}, [None, None, None, "F"]
            )
            url = f"{base}/room/{room['id']}"
            players = [
                bot_client(url, spec, transcript_dir=str(tmp_path))
                for spec in ("F", "R", "W")
            ]
            watcher = GameClient(url, spectate=True, transcript_dir=str(tmp_path))
            watching = asyncio.create_task(watcher.play())
            results = await asyncio.gather(*(c.play(d) for c, d in players))
            return players, results, await watching

    players, results, watched = run(scenario())
    for result in [*results, watched]:
        assert result.verified, result.report.errors
    assert sorted(r.color for r in results) == ["BLUE", "ORANGE", "RED"]
    assert len({r.game_id for r in results}) == 1
    assert all(r.report.counts["own moves"] > 0 for r in results)

    # The transcripts were saved, and agree with one another.
    transcripts = [json.loads(open(r.transcript_path).read()) for r in results]
    reports = verify_transcripts(transcripts)
    assert all(r.ok for r in reports), [r.errors for r in reports]
    seeds = {t["client_seed"] for t in transcripts}
    assert len(seeds) == 3, "every client contributed its own seed"

    # And from the shell, in a process whose string hashing differs from the
    # server's (Catanatron builds some actions from sets of strings).
    for hash_seed in ("0", "1"):
        verified = subprocess.run(
            [sys.executable, "-m", "catanatron.server.verify"]
            + [r.transcript_path for r in results],
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert verified.returncode == 0, verified.stdout + verified.stderr


def test_a_client_that_reconnects_keeps_its_seat_and_its_chain():
    async def scenario():
        async with serving() as (server, base):
            room = await create_room(base, {"num_players": 2, **FAST}, [None, "R"])
            client, decide = bot_client(f"{base}/room/{room['id']}", "R")
            drops = []

            def on_message(body):
                if body["type"] == "step" and body["index"] in (12, 40):
                    drops.append(body["index"])
                    asyncio.get_running_loop().create_task(client.connection.close())

            client.on_message = on_message
            return await client.play(decide), drops

    result, drops = run(scenario())
    assert drops == [12, 40]
    assert result.verified, result.report.errors


def test_a_seat_that_times_out_gets_the_default_action_flagged():
    async def scenario():
        async with serving() as (server, base):
            config = {"num_players": 2, "vps_to_win": 4, "decision_timeout_ms": 1500}
            room = await create_room(base, config, [None, "F"])
            client, decide = bot_client(f"{base}/room/{room['id']}", "F")
            slow = {"n": 0}

            async def sometimes_slow(view, playable_actions):
                slow["n"] += 1
                if slow["n"] in (3, 4):
                    await asyncio.sleep(2.0)
                return await asyncio.to_thread(decide, view, playable_actions)

            result = await client.play(sometimes_slow)
            return server.rooms[room["id"]], result

    room, result = run(scenario())
    assert result.verified, result.report.errors
    timeouts = [a for a in room.actions if a["forced"] == "timeout"]
    assert len(timeouts) == 2
    assert all(a["by"] == "server" for a in timeouts)


def test_a_game_on_an_explicit_map():
    spec = map_spec(build_map("BASE"))
    # Turn every 2:1 port into a 3:1 port, and swap two numbers.
    for tile in spec["tiles"]:
        if tile["type"] == "port":
            tile["port_resource"] = None
    numbered = [t for t in spec["tiles"] if t.get("number")]
    numbered[0]["number"], numbered[1]["number"] = (
        numbered[1]["number"],
        numbered[0]["number"],
    )

    async def scenario():
        async with serving() as (server, base):
            room = await create_room(
                base, {"num_players": 3, "map": spec, **FAST}, ["F", "R", "F"]
            )
            watcher = GameClient(
                f"{base}/room/{room['id']}", spectate=True, transcript_dir=None
            )
            return server.rooms[room["id"]], await watcher.play()

    room, result = run(scenario())
    assert result.verified, result.report.errors
    assert room.config.map == spec
    assert room.game.state.board.map.port_nodes.keys() == {None}


@pytest.mark.parametrize("name", ["extension", "blob25_stringy"])
def test_a_verified_game_on_an_irregular_map(name):
    with open(os.path.join(MAPS_DIR, name + ".json")) as file:
        spec = {"tiles": json.load(file)["tiles"]}
    num_players = 6 if name == "extension" else 3

    async def scenario():
        async with serving() as (server, base):
            config = {"num_players": num_players, "map": spec, **FAST}
            room = await create_room(base, config, [None] + ["F"] * (num_players - 1))
            client, decide = bot_client(f"{base}/room/{room['id']}", "F")
            return server.rooms[room["id"]], await client.play(decide)

    room, result = run(scenario())
    assert result.verified, result.report.errors
    assert result.report.counts["client signatures"] > 0
    assert not room.game.state.board.map.is_base_layout()
    assert len(room.game.state.board.map.land_tiles) == (
        30 if name == "extension" else 25
    )


def test_seats_are_refused_once_taken_or_started():
    async def scenario():
        async with serving() as (server, base):
            room = await create_room(base, {"num_players": 2, **FAST}, ["R"])
            url = f"{base}/room/{room['id']}"
            first, decide = bot_client(url, "R", seat=1)
            playing = asyncio.create_task(first.play(decide))
            while first.seat is None:
                await asyncio.sleep(0.01)
            second, decide2 = bot_client(url, "R", seat=1)
            with pytest.raises(Exception, match="taken|started"):
                await second.play(decide2)
            return await playing

    assert run(scenario()).verified
