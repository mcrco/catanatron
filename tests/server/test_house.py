"""House mode: the server is a place to play its operator's bots.

Only connections showing the house token open rooms and seat bots, a room
takes at most one seat from anyone else, and a game its person has left for
good is void.
"""

import asyncio
import json
import urllib.request

import pytest

from catanatron.server.client import Refused, create_room

from tests.server.helpers import bot_client, join_request, next_body, raw_connect, run, serving

TOKEN = "t" * 32


@pytest.fixture
def house_env(monkeypatch):
    monkeypatch.setenv("CATANATRON_HOUSE_TOKEN", TOKEN)


def _get(base, path):
    with urllib.request.urlopen(base.replace("ws://", "http://") + path) as reply:
        return json.loads(reply.read())


def test_only_the_house_opens_rooms(monkeypatch):
    async def scenario():
        async with serving(house_token=TOKEN) as (server, base):
            with pytest.raises(Refused, match="opened by the house"):
                await create_room(base, {"num_players": 2}, [None, None])
            monkeypatch.setenv("CATANATRON_HOUSE_TOKEN", "x" * 32)
            with pytest.raises(Refused, match="opened by the house"):
                await create_room(base, {"num_players": 2}, [None, None])
            monkeypatch.setenv("CATANATRON_HOUSE_TOKEN", TOKEN)
            room = await create_room(base, {"num_players": 2}, [None, None])
            info = await asyncio.to_thread(_get, base, "/api/server")
            return room, info

    room, info = run(scenario())
    assert room["status"] == "waiting"
    assert info["house"] is True


def test_a_server_without_a_token_is_not_in_house_mode():
    async def scenario():
        async with serving() as (server, base):
            await create_room(base, {"num_players": 2}, [None, None])
            return await asyncio.to_thread(_get, base, "/api/server")

    assert run(scenario())["house"] is False


def test_a_house_room_takes_one_person_and_no_bots_from_people(house_env):
    async def scenario():
        async with serving(house_token=TOKEN) as (server, base):
            room = await create_room(base, {"num_players": 4}, [None] * 4)
            url = f"{base}/room/{room['id']}"
            # The house sits...
            house_ws, hello = await raw_connect(url, {"X-Catanatron-House": TOKEN})
            request, _, _ = join_request(hello)
            await house_ws.send(json.dumps(request))
            assert (await next_body(house_ws, "joined", "error"))["type"] == "joined"
            # ...a person may not seat a bot...
            ws, hello = await raw_connect(url)
            await ws.send(json.dumps({"type": "seat_bot", "seat": 1, "spec": "R"}))
            refused_bot = await next_body(ws, "error")
            # ...takes one seat...
            request, _, _ = join_request(hello)
            await ws.send(json.dumps(request))
            joined = await next_body(ws, "joined", "error")
            # ...and a second person is turned away.
            other, hello = await raw_connect(url)
            request, _, _ = join_request(hello)
            await other.send(json.dumps(request))
            second = await next_body(other, "joined", "error")
            listed = await asyncio.to_thread(_get, base, "/api/rooms")
            for socket in (house_ws, ws, other):
                await socket.close()
            return refused_bot, joined, second, listed

    refused_bot, joined, second, listed = run(scenario())
    assert refused_bot["code"] == "bad_bot"
    assert joined["type"] == "joined"
    assert (second["type"], second["code"]) == ("error", "room_full")
    assert listed[0]["house_seats"] == [0]


def test_a_game_its_person_left_is_void(house_env, monkeypatch):
    async def scenario():
        async with serving(house_token=TOKEN) as (server, base):
            room = await create_room(base, {"num_players": 2, "decision_timeout_ms": 600_000}, [None, None])
            url = f"{base}/room/{room['id']}"
            client, decide = bot_client(url)
            house = asyncio.create_task(client.play(decide))
            while not server.rooms[room["id"]].seats[0].connected:
                await asyncio.sleep(0.05)
            monkeypatch.delenv("CATANATRON_HOUSE_TOKEN")
            person, decide = bot_client(url)
            task = asyncio.create_task(person.play(decide))
            live = server.rooms[room["id"]]
            while live.status != "playing":
                await asyncio.sleep(0.05)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            server.sweep()
            assert live.status == "playing"  # not yet: they may come back
            server.sweep(now=server.away_since[live.id] + server.limits.human_away_s + 1)
            result = await asyncio.wait_for(house, 30)
            return live.status, result

    status, result = run(scenario())
    assert status == "aborted"
    assert result.aborted and "left" in result.aborted["reason"]
