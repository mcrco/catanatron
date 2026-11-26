"""The archive's index: finished games by the names people sat under, their
replays, and the analyses the house sends back."""

import asyncio
import gzip
import json
import urllib.error
import urllib.request

import pytest

from catanatron.server.client import Refused, create_room, lobby_request
from catanatron.server.games import GameStore, name_key, rebuild

from tests.server.helpers import bot_client, run, serving

TOKEN = "t" * 32
FAST = {"vps_to_win": 6, "decision_timeout_ms": 20_000}


def _get(base, path, headers=None):
    request = urllib.request.Request(base.replace("ws://", "http://") + path, headers=headers or {})
    with urllib.request.urlopen(request) as reply:
        body = reply.read()
        if reply.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
        return json.loads(body)


def _status(base, path, headers=None):
    try:
        _get(base, path, headers)
        return 200
    except urllib.error.HTTPError as error:
        return error.code


def test_names_fold_case_and_spacing_and_defaults_are_nobody():
    assert name_key("  Marco ") == name_key("marco") == "marco"
    assert name_key("Ada  Lovelace") == "ada lovelace"
    assert name_key("BLUE") is None and name_key("") is None and name_key("Guest") is None
    assert name_key(None) is None


def test_a_finished_game_is_listed_under_its_players_names_and_replays(tmp_path):
    async def scenario():
        async with serving(archive_dir=str(tmp_path)) as (server, base):
            room = await create_room(base, {"num_players": 2, **FAST}, ["F", None])
            client, decide = bot_client(f"{base}/room/{room['id']}")
            client.name = "Ada"
            await client.play(decide)
            listed = await asyncio.to_thread(_get, base, "/api/games?player=ada")
            everyone = await asyncio.to_thread(_get, base, "/api/games")
            nobody = await asyncio.to_thread(_get, base, "/api/games?player=bob")
            game_id = listed["games"][0]["game_id"]
            replay = await asyncio.to_thread(_get, base, f"/api/games/{game_id}", {"Accept-Encoding": "gzip"})
            plain = await asyncio.to_thread(_get, base, f"/api/games/{game_id}")
            missing = await asyncio.to_thread(_status, base, "/api/games/nope")
            return listed, everyone, nobody, replay, plain, missing

    listed, everyone, nobody, replay, plain, missing = run(scenario())
    (game,) = listed["games"]
    assert [g["game_id"] for g in everyone["games"]] == [game["game_id"]]
    assert nobody["games"] == []
    names = {s["name"]: s["kind"] for s in game["seats"]}
    assert names["Ada"] == "person" and names["F"] == "bot"
    assert game["winner"] in {s["color"] for s in game["seats"]}
    assert replay == plain and missing == 404
    # The page asks for a review only of games the house reviews (people played them).
    assert game["people"] == replay["game"]["people"] == 1
    assert len(replay["frames"]) == len(replay["log"]) + 1 == game["actions"] + 1
    end = replay["frames"][-1]
    winner = end["colors"].index(game["winner"])
    assert end["player_state"][f"P{winner}_ACTUAL_VICTORY_POINTS"] >= 6
    # Every hand face up, in the page's shape.
    assert all(f"P{i}_WOOD_IN_HAND" in end["player_state"] for i in range(2))
    assert "P0_NUM_RESOURCES_IN_HAND" in end["player_state"]
    assert replay["analysis"] is None


def test_the_index_rebuilds_from_the_archive_files(tmp_path):
    async def scenario():
        async with serving(archive_dir=str(tmp_path)) as (server, base):
            room = await create_room(base, {"num_players": 3, **FAST}, ["F", "R", None])
            client, decide = bot_client(f"{base}/room/{room['id']}")
            client.name = "Marco"
            await client.play(decide)

    run(scenario())
    (tmp_path / "games.sqlite").unlink()
    store = GameStore(str(tmp_path))
    (game,) = store.games("MARCO")
    record = store.record(game["game_id"])
    docs = rebuild(record)
    assert len(docs) == len(record["final_state"]["action_records"]) + 1


def test_the_house_takes_the_queue_and_sends_analyses(tmp_path):
    house = {"X-Catanatron-House": TOKEN}

    async def scenario():
        async with serving(archive_dir=str(tmp_path), house_token=TOKEN) as (server, base):
            room = await create_room(base, {"num_players": 2, **FAST}, [None, None], house_token=TOKEN)
            bot, bot_decide = bot_client(f"{base}/room/{room['id']}")
            person, person_decide = bot_client(f"{base}/room/{room['id']}", spec="R")
            person.name = "ada"
            await asyncio.gather(bot.play(bot_decide), person.play(person_decide))
            forbidden = await asyncio.to_thread(_status, base, "/api/analysis/queue")
            queue = await asyncio.to_thread(_get, base, "/api/analysis/queue", house)
            game_id = queue["games"][0]
            record = await asyncio.to_thread(_get, base, f"/api/games/{game_id}/record", house)
            hidden = await asyncio.to_thread(_status, base, f"/api/games/{game_id}/record")
            with pytest.raises(Refused, match="from the house"):
                await lobby_request(base, {"type": "submit_analysis", "game_id": game_id,
                                           "analysis": {"summary": {}}}, house_token="x" * 32)
            with pytest.raises(Refused, match="summary"):
                await lobby_request(base, {"type": "submit_analysis", "game_id": game_id, "analysis": {}},
                                    house_token=TOKEN)
            analysis = {"version": 1, "summary": {"label": "F", "accuracy": {"RED": 91}},
                        "evals": [[0, [500, 500]]], "moves": []}
            await lobby_request(base, {"type": "submit_analysis", "game_id": game_id, "analysis": analysis},
                                house_token=TOKEN)
            after = await asyncio.to_thread(_get, base, "/api/analysis/queue", house)
            listed = await asyncio.to_thread(_get, base, "/api/games?player=Ada")
            replay = await asyncio.to_thread(_get, base, f"/api/games/{game_id}")
            return forbidden, queue, record, hidden, after, listed, replay, analysis

    forbidden, queue, record, hidden, after, listed, replay, analysis = run(scenario())
    assert forbidden == 403 and hidden == 403
    assert len(queue["games"]) == 1 and record["final_state"]["action_records"]
    assert after["games"] == []
    assert listed["games"][0]["analysis"] == analysis["summary"]
    assert replay["analysis"] == analysis


def test_the_rebuild_offers_every_logged_move_at_its_step(tmp_path):
    TRADES = {"OFFER_TRADE", "ACCEPT_TRADE", "REJECT_TRADE", "CONFIRM_TRADE", "CANCEL_TRADE"}

    async def scenario():
        async with serving(archive_dir=str(tmp_path)) as (server, base):
            room = await create_room(base, {"num_players": 4, **FAST}, ["F", "R", "F", None])
            client, decide = bot_client(f"{base}/room/{room['id']}")
            await client.play(decide)

    run(scenario())
    store = GameStore(str(tmp_path))
    (game,) = store.games()
    record = store.record(game["game_id"])
    playable: list = []
    docs = rebuild(record, playable)
    log = record["final_state"]["action_records"]
    assert len(playable) == len(log) == len(docs) - 1
    for index, ((action, _), legal) in enumerate(zip(log, playable)):
        if action[1] in TRADES:
            continue
        color, kind, value = action
        wanted = [color, kind, None if kind in ("BUY_DEVELOPMENT_CARD", "ROLL") else value]
        assert json.loads(json.dumps(wanted)) in json.loads(json.dumps(legal)), (index, action)
