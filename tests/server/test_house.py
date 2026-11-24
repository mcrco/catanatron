"""House mode: the server is a place to play its operator's bots.

Only connections showing the house token open rooms and seat bots, people
take the seats the house leaves open (and, where the house allows it, ask it
for bots), and a game its person has left for good is void.
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


def test_a_house_room_takes_people_in_its_open_seats_and_no_bots_from_them(house_env):
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
            # ...and a second person takes another.
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
    assert second["type"] == "joined" and second["seat"] != joined["seat"]
    assert listed[0]["house_seats"] == [0]
    assert "bots_on_request" not in listed[0]


async def _room_body(ws, timeout=10):
    return (await next_body(ws, "room", timeout=timeout))["room"]


def test_people_ask_the_house_for_bots_where_it_allows_them(house_env):
    async def scenario():
        async with serving(house_token=TOKEN) as (server, base):
            room = await create_room(base, {"num_players": 4}, [None] * 4, bots_on_request=True)
            plain = await create_room(base, {"num_players": 3}, [None] * 3)
            url = f"{base}/room/{room['id']}"
            house_ws, hello = await raw_connect(url, {"X-Catanatron-House": TOKEN})
            request, _, _ = join_request(hello)
            await house_ws.send(json.dumps(request))
            await next_body(house_ws, "joined")
            # The house asks for nothing; a person not seated asks for nothing.
            await house_ws.send(json.dumps({"type": "add_bot"}))
            by_house = await next_body(house_ws, "error")
            ws, hello = await raw_connect(url)
            await ws.send(json.dumps({"type": "add_bot"}))
            unseated = await next_body(ws, "error")
            request, _, _ = join_request(hello)
            await ws.send(json.dumps(request))
            await next_body(ws, "joined")
            # Two open seats: two requests, then no more.
            for _ in range(3):
                await ws.send(json.dumps({"type": "add_bot"}))
            seen = await _room_body(house_ws)
            while seen["bot_requests"] < 2:  # the person's join, then the requests (maybe one push)
                seen = await _room_body(house_ws)
            third = await next_body(ws, "error")
            # A house seat answers one.
            bot_ws, hello = await raw_connect(url, {"X-Catanatron-House": TOKEN})
            request, _, _ = join_request(hello)
            await bot_ws.send(json.dumps(request))
            await next_body(bot_ws, "joined")
            answered = await _room_body(house_ws)
            listed = await asyncio.to_thread(_get, base, f"/api/rooms/{room['id']}")
            # A room opened without bots_on_request takes no requests.
            purl = f"{base}/room/{plain['id']}"
            pws, hello = await raw_connect(purl)
            request, _, _ = join_request(hello)
            await pws.send(json.dumps(request))
            await next_body(pws, "joined")
            await pws.send(json.dumps({"type": "add_bot"}))
            refused = await next_body(pws, "error")
            for socket in (house_ws, ws, bot_ws, pws):
                await socket.close()
            return by_house, unseated, seen, third, answered, listed, refused

    by_house, unseated, seen, third, answered, listed, refused = run(scenario())
    assert by_house["code"] == unseated["code"] == "bad_request"
    assert (seen["bot_requests"], seen["bots_wanted"]) == (2, 2)
    assert third["code"] == "room_full"
    assert (answered["bot_requests"], answered["bots_wanted"]) == (2, 1)
    assert listed["bots_on_request"] is True and listed["house_seats"] == [0, 2]
    assert refused["code"] == "bad_request"


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
