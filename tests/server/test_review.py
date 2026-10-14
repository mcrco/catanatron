"""Regression tests for the findings of the adversarial review of the server.

Each dishonest server is a Room that cheats in one way and signs the result
as usual. A LaxClient (which skips the client's own live checks) lets such a
game run to the end, so that the verifier's verdict is what is tested; the
ordinary client is also shown to refuse the game outright where it can.
"""

import asyncio
import dataclasses
import json
import os
import signal
import subprocess
import sys
import time

import pytest
import websockets

from catanatron.models.enums import Action, ActionType
from catanatron.models.player import Player
from catanatron.registry import REGISTRY
from catanatron.serialization import action_to_json
from catanatron.server.app import ServerLimits
from catanatron.server.client import (
    GameClient,
    Refused,
    ServerMisbehaved,
    create_room,
)
from catanatron.server.config import ConfigError, GameConfig
from catanatron.server.crypto import digest, random_hex
from catanatron.server.messages import PROTOCOL_VERSION, decide_body
from catanatron.server.room import Room, RoomError, RoomLimits
from catanatron.server.verify import verify_transcript

from tests.server.helpers import (
    LaxClient,
    bot_client,
    join_request,
    next_body,
    raw_connect,
    run,
    serving,
)

CONFIG = {"num_players": 3, "vps_to_win": 5, "decision_timeout_ms": 20_000}


def play(room_class=Room, client_class=LaxClient, config=CONFIG, remote=2, **server):
    """Remote bot clients in the first ``remote`` seats, server F bots in the
    rest. Returns (clients, results or exceptions, room)."""

    async def scenario():
        async with serving(room_class=room_class, **server) as (srv, base):
            seats = [None] * remote + ["F"] * (config["num_players"] - remote)
            room = await create_room(base, config, seats)
            url = f"{base}/room/{room['id']}"
            clients = [
                bot_client(url, "F", client_class=client_class, seat=i, verify=False)
                for i in range(remote)
            ]
            results = await asyncio.gather(
                *(c.play(d) for c, d in clients), return_exceptions=True
            )
            return [c for c, _ in clients], results, srv.rooms.get(room["id"])

    return run(scenario())


def errors_of(client):
    return verify_transcript(json.loads(json.dumps(client.transcript()))).errors


def any_error(errors, *fragments):
    return any(all(f in e for f in fragments) for e in errors)


# ===== A. seeds: commit, then reveal =====
class RecordingRoom(Room):
    seen_at_lock = None

    def _maybe_lock(self):
        if self.status == "waiting" and all(s.kind != "open" for s in self.seats):
            RecordingRoom.seen_at_lock = [s.client_seed for s in self.seats]
        super()._maybe_lock()


def test_seeds_are_committed_before_any_is_revealed():
    clients, results, room = play(RecordingRoom, GameClient)
    assert RecordingRoom.seen_at_lock == [None, None, None], "no seed known at the lock"
    for client, result in zip(clients, results):
        assert result.verified is None or result.verified
        assert verify_transcript(client.transcript()).ok
        kinds = [m["body"]["type"] for m in client.messages]
        assert (
            kinds.index("joined") < kinds.index("commitments") < kinds.index("before")
        )
        assert "client_seed" not in json.dumps(
            client.messages[: kinds.index("commitments")]
        )


class SilentClient(GameClient):
    """Joins, then never reveals its seed."""

    async def _handle(self, ws, body, decide):
        if body.get("type") == "commitments":
            self.commitments = body
            return None
        return await super()._handle(ws, body, decide)


def test_a_seat_that_does_not_reveal_is_replaced_and_the_others_commit_afresh():
    """Withholding no longer voids the game: the withholder is out (and
    barred), every seed is replaced, and the room fills again."""

    async def scenario():
        limits = ServerLimits(room=RoomLimits(reveal_timeout_ms=400))
        async with serving(limits=limits) as (server, base):
            room = await create_room(base, CONFIG, [None, None, "F"])
            url = f"{base}/room/{room['id']}"
            honest, decide = bot_client(url, seat=0, verify=False)
            silent, decide2 = bot_client(url, client_class=SilentClient, seat=1)
            playing = asyncio.create_task(honest.play(decide))
            withheld = await asyncio.gather(
                silent.play(decide2), return_exceptions=True
            )
            # The withholder's key may not come back.
            again, decide3 = bot_client(url, key=silent.key, seat=1)
            with pytest.raises(Refused, match="withheld"):
                await again.play(decide3)
            room_obj = server.rooms[room["id"]]
            # Its address is barred from this room too: allow a second one in.
            room_obj.banned_addresses.clear()
            late, decide4 = bot_client(url, seat=1, verify=False)
            results = await asyncio.gather(playing, late.play(decide4))
            return honest, late, results, room_obj, withheld

    honest, late, results, room, withheld = run(scenario())
    assert room.status == "finished" and room.round == 1
    kinds = [m["body"]["type"] for m in honest.messages]
    assert kinds.count("commitments") == 2 and kinds.count("restart") == 1
    for client in (honest, late):
        report = verify_transcript(client.transcript())
        assert report.ok, report.errors


def test_a_room_whose_seats_keep_withholding_is_called_off():
    async def scenario():
        limits = ServerLimits(room=RoomLimits(reveal_timeout_ms=300, max_rounds=1))
        async with serving(limits=limits) as (server, base):
            room = await create_room(base, CONFIG, [None, None, "F"])
            url = f"{base}/room/{room['id']}"
            honest, decide = bot_client(url, seat=0, verify=False)
            silent, decide2 = bot_client(url, client_class=SilentClient, seat=1)
            results = await asyncio.gather(
                honest.play(decide), silent.play(decide2), return_exceptions=True
            )
            return results[0], honest

    result, honest = run(scenario())
    assert result.aborted["blame"] == [1]
    assert result.verified is False
    errors = verify_transcript(honest.transcript()).errors
    assert any_error(errors, "aborted", "blamed seats: [1]")


def test_a_reveal_must_open_its_commitment():
    async def scenario():
        async with serving() as (server, base):
            room = await create_room(base, {"num_players": 2}, [None, "F"])
            ws, hello = await raw_connect(f"{base}/room/{room['id']}")
            request, key, seed = join_request(hello)
            await ws.send(json.dumps(request))
            await next_body(ws, "commitments")
            await ws.send(json.dumps({"type": "reveal", "client_seed": random_hex()}))
            refusal = await next_body(ws, "error")
            await ws.close()
            return refusal

    assert run(scenario())["code"] == "bad_reveal"


def test_a_join_needs_proof_of_the_key_and_a_key_not_already_seated():
    async def scenario():
        async with serving() as (server, base):
            room = await create_room(base, {"num_players": 3}, [None, None, None])
            url = f"{base}/room/{room['id']}"
            ws1, hello = await raw_connect(url)
            request, key, _ = join_request(hello)
            await ws1.send(json.dumps(request))
            await next_body(ws1, "joined")

            ws2, hello2 = await raw_connect(url)
            stolen, _, _ = join_request(hello2, key=key, pop_key=None)
            stolen["pop"] = "00" * 64  # cannot sign with a key it does not hold
            await ws2.send(json.dumps(stolen))
            no_proof = await next_body(ws2, "error")

            # Even with a valid proof, one key holds one seat.
            again, _, _ = join_request(hello2, key=key)
            await ws2.send(json.dumps(again))
            duplicate = await next_body(ws2, "error")
            await ws1.close()
            await ws2.close()
            return no_proof, duplicate

    no_proof, duplicate = run(scenario())
    assert no_proof["code"] == "bad_join" and "pop" in no_proof["message"]
    assert duplicate["code"] == "duplicate_key"


class SeedSwapRoom(Room):
    """Publishes the two remote seats' seeds the other way round."""

    def _client_seeds(self):
        seeds = super()._client_seeds()
        seeds[0], seeds[1] = seeds[1], seeds[0]
        return seeds


def test_seeds_that_do_not_open_their_commitments_are_caught():
    clients, results, _ = play(SeedSwapRoom)
    for client in clients:
        assert any_error(errors_of(client), "seed does not open its commitment")
    strict, results, _ = play(SeedSwapRoom, GameClient)
    assert all(isinstance(r, ServerMisbehaved) for r in results)


# ===== B. the game must be the one agreed to =====
class ConfigSwapRoom(Room):
    """Changes the rules once every seed is in."""

    async def _play(self):
        self.config = dataclasses.replace(self.config, vps_to_win=3)
        await super()._play()


class SeatSwapRoom(Room):
    """Swaps the two remote seats once every seed is in."""

    async def _play(self):
        a, b = self.locked[0], self.locked[1]
        for key in ("kind", "name", "public_key", "client_commitment"):
            a[key], b[key] = b[key], a[key]
        await super()._play()


@pytest.mark.parametrize("room_class", [ConfigSwapRoom, SeatSwapRoom])
def test_a_game_other_than_the_committed_one_is_caught(room_class):
    clients, results, _ = play(room_class)
    for client in clients:
        errors = errors_of(client)
        assert errors, "the game is not the committed one"
        assert (
            any_error(errors, "committed")
            or any_error(errors, "published")
            or any_error(errors, "malformed")
            or any_error(errors, "vps_to_win")
        ), errors
    strict, results, _ = play(room_class, GameClient)
    # The live client refuses the game when it can tell from the messages; a
    # change it cannot see there fails verification at the end.
    assert all(
        isinstance(r, ServerMisbehaved) or errors_of(c) for c, r in zip(strict, results)
    )


# ===== C. statements, refusals and the message schedule =====
class ReplayRoom(Room):
    """Reuses BLUE's old signed END_TURN instead of asking BLUE."""

    replayed = 0

    async def _remote_decision(self, seat, playable):
        old = getattr(self, "_old", None)
        if seat.color.value == "BLUE" and old is not None:
            action = next(
                (a for a in playable if action_to_json(a) == old[0]["action"]), None
            )
            if action is not None and self.replayed < 3:
                self.replayed += 1
                return action, old[0], old[1]
        decided = await super()._remote_decision(seat, playable)
        if seat.color.value == "BLUE" and not isinstance(decided, str):
            if decided[1]["action"][1] == "END_TURN":
                self._old = (decided[1], decided[2])
        return decided


def test_a_signature_cannot_be_replayed_for_another_move():
    clients, _, room = play(ReplayRoom)
    assert room.replayed
    red = errors_of(clients[0])
    assert any_error(red, "signed statement is for another move")


class CoercingRoom(Room):
    """Refuses RED's first legal answer with a made-up error."""

    coerced = False

    def on_action(self, seat, peer, request):
        if seat.color.value == "RED" and not self.coerced and seat.pending is not None:
            self.coerced = True
            raise RoomError("bad_action", "is not legal now")
        super().on_action(seat, peer, request)


def test_a_refused_legal_move_is_not_replaced_and_is_reported():
    config = {**CONFIG, "decision_timeout_ms": 1500}
    clients, results, room = play(CoercingRoom, GameClient, config=config)
    red = clients[0]
    assert room.coerced
    by_index = {}
    for item in red.sent:
        by_index.setdefault(item["statement"]["index"], set()).add(
            digest(item["statement"])
        )
    assert all(
        len(statements) == 1 for statements in by_index.values()
    ), "never a second move"
    assert any_error(errors_of(red), "never acknowledged this client's legal")


class ResolicitRoom(Room):
    """Asks RED twice for the same move."""

    asked = False

    async def _remote_decision(self, seat, playable):
        if seat.color.value == "RED" and not self.asked:
            self.asked = True
            seat.outbox.push(
                decide_body(
                    self.game,
                    self.doc,
                    "RED",
                    timeout_ms=self.config.decision_timeout_ms,
                ),
                self._commit(),
            )
        return await super()._remote_decision(seat, playable)


class DuplicateStepRoom(Room):
    """Sends RED one step twice."""

    doubled = False

    def _send_all(self, build):
        super()._send_all(build)
        red = self.seats[0].outbox
        if not self.doubled and len(self.actions) == 12 and red is not None:
            self.doubled = True
            red.push(build("RED"), self._commit())


class StrayDecideRoom(Room):
    """Sends RED a decide during another seat's turn."""

    strayed = False

    async def _remote_decision(self, seat, playable):
        if seat.color.value == "BLUE" and not self.strayed:
            self.strayed = True
            red = self.seats[0].outbox
            red.push(
                decide_body(self.game, self.doc, "RED", timeout_ms=1000), self._commit()
            )
        return await super()._remote_decision(seat, playable)


@pytest.mark.parametrize(
    "room_class,message",
    [
        (ResolicitRoom, "a second decide"),
        (DuplicateStepRoom, "out of order"),
        (StrayDecideRoom, "not to move"),
    ],
)
def test_the_message_schedule_is_checked(room_class, message):
    clients, _, _ = play(room_class)
    assert any_error(errors_of(clients[0]), message), errors_of(clients[0])


def test_two_different_signed_moves_for_one_index_are_reported():
    clients, _, _ = play()
    red = clients[0]
    assert verify_transcript(red.transcript()).ok
    statement = dict(red.sent[0]["statement"])
    statement["action"] = ["RED", "END_TURN", None]
    red.sent.append({"statement": statement, "sig": red.key.sign(statement)})
    assert any_error(errors_of(red), "signed 2 different moves")


def test_numbers_must_be_integers():
    async def scenario():
        async with serving() as (server, base):
            room = await create_room(base, {"num_players": 2}, [None, "F"])
            ws, hello = await raw_connect(f"{base}/room/{room['id']}")
            request, _, _ = join_request(hello)
            text = json.dumps(request).replace('"seat": null', '"seat": 0.0')
            await ws.send(text)
            refusal = await next_body(ws, "error")
            await ws.close()
            return refusal

    assert "integers" in run(scenario())["message"]
    from catanatron.models.map import build_map
    from catanatron.server.maps import map_spec

    spec = map_spec(build_map("BASE"))
    GameConfig.from_json({"map": spec})
    numbered = next(t for t in spec["tiles"] if t.get("number"))
    numbered["number"] = float(numbered["number"])  # 6.0 == 6, but not on the wire
    with pytest.raises(ConfigError, match="number"):
        GameConfig.from_json({"map": spec})


# ===== D. only offered actions =====
def test_moves_that_were_not_offered_are_refused():
    async def scenario():
        async with serving() as (server, base):
            config = {"num_players": 2, "vps_to_win": 4, "decision_timeout_ms": 300}
            room = await create_room(base, config, [None, "F"])
            client = GameClient(f"{base}/room/{room['id']}", transcript_dir=None)
            offers = []

            def decide(view, playable):
                if any(a[1] == "END_TURN" for a in playable) and len(offers) < 3:
                    offers.append(1)
                    return ["RED", "OFFER_TRADE", [1, 0, 0, 0, 0, 0, 0, 0, 0, 19]]
                return playable[0]

            result = await client.play(decide)
            return result, server.rooms[room["id"]], offers

    result, room, offers = run(scenario())
    assert offers
    assert not any(a["action"][1] == "OFFER_TRADE" for a in room.actions)
    assert result.verified, result.report.errors  # refusing an illegal move is fine


class OffMenuRoom(Room):
    """Has its bot make a trade offer the game never offered."""

    offered = False

    async def _bot_decision(self, seat, playable):
        state = self.game.state
        if not self.offered and any(
            a.action_type == ActionType.END_TURN for a in playable
        ):
            self.offered = True
            color = state.current_color()
            return (
                Action(color, ActionType.OFFER_TRADE, (1, 0, 0, 0, 0, 0, 0, 0, 0, 1)),
                None,
                None,
            )
        return await super()._bot_decision(seat, playable)


def test_moves_not_offered_by_the_game_fail_verification():
    config = {**CONFIG, "decision_timeout_ms": 2000}
    clients, _, room = play(OffMenuRoom, config=config, remote=1)
    assert room.offered
    assert any_error(errors_of(clients[0]), "not one of the moves the game offered")


# ===== E. forced moves: only when everyone can see it =====
def test_publicly_forced_only_when_the_options_are_public():
    from catanatron.game import Game
    from catanatron.models.enums import ActionRecord
    from catanatron.models.player import Color, RandomPlayer
    from catanatron.server.config import publicly_forced
    from catanatron.state_functions import player_key

    game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=2)
    while game.state.is_initial_build_phase:
        game.play_tick()
    state = game.state
    assert [a.action_type for a in game.playable_actions] == [ActionType.ROLL]
    assert publicly_forced(game), "no development cards: everyone knows it must roll"

    key = player_key(state, state.current_color())
    state.player_state[f"{key}_VICTORY_POINT_IN_HAND"] = 1
    assert len(game.playable_actions) == 1
    assert not publicly_forced(game), "a hidden card might have been a knight"
    state.player_state[f"{key}_VICTORY_POINT_IN_HAND"] = 0

    # A discard from a one-resource hand is forced, but only its holder knows.
    for color in state.colors:
        k = player_key(state, color)
        for resource in ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"):
            state.player_state[f"{k}_{resource}_IN_HAND"] = (
                8 if resource == "WOOD" else 0
            )
    roll = game.playable_actions[0]
    game.execute(roll, action_record=ActionRecord(roll, (3, 4)))
    assert [a.action_type for a in game.playable_actions] == [
        ActionType.DISCARD_RESOURCE
    ]
    assert not publicly_forced(game)


# ===== F. friendly robber and offered-only bots =====
def test_rooms_refuse_the_friendly_robber():
    async def scenario():
        async with serving() as (server, base):
            with pytest.raises(Refused, match="friendly_robber"):
                await create_room(
                    base, {"num_players": 2, "friendly_robber": True}, ["F", "F"]
                )

    run(scenario())


def test_a_view_bot_only_picks_among_the_offered_actions():
    from catanatron.game import Game
    from catanatron.models.player import Color, RandomPlayer
    from catanatron.models.decks import BASE_SUPPLY
    from catanatron.serialization import client_view, state_to_json
    from catanatron.server.bot import ViewBot

    game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=1)
    for _ in range(40):
        game.play_tick()
    while len(game.playable_actions) < 3:
        game.play_tick()
    color = game.state.current_color()
    view = client_view(state_to_json(game), color)
    offered = [action_to_json(a) for a in game.playable_actions][-2:]
    bot = ViewBot("F", color, seed=0)
    for _ in range(5):
        assert bot.decide(view, offered, BASE_SUPPLY) in offered


# ===== G. bots =====
class CrashingBot(Player):
    def decide(self, game, playable_actions):
        raise RuntimeError("this bot always fails")


class SlowBot(Player):
    def decide(self, game, playable_actions):
        time.sleep(3)
        return playable_actions[0]


class ExitingBot(Player):
    def decide(self, game, playable_actions):
        raise SystemExit("a bot that quits the process")


@pytest.fixture
def test_bots():
    REGISTRY.register("CRASHTEST", CrashingBot, replace=True)
    REGISTRY.register("SLOWTEST", SlowBot, replace=True)
    REGISTRY.register("EXITTEST", ExitingBot, replace=True)
    yield
    for key in ("CRASHTEST", "SLOWTEST", "EXITTEST"):
        REGISTRY.pop(key, None)


def watch(seats, config):
    """A spectator's view of a game of server bots; also returns the room and
    every bot worker process the room used."""

    async def scenario():
        async with serving() as (server, base):
            room = await create_room(base, config, seats)
            watcher = GameClient(
                f"{base}/room/{room['id']}", spectate=True, transcript_dir=None
            )
            room_obj = server.rooms[room["id"]]
            workers = {}  # pid -> most processes it ever had under it

            async def sample():
                import psutil

                while True:
                    process = room_obj.worker.process
                    if process is not None and process.pid:
                        try:
                            under = len(
                                psutil.Process(process.pid).children(recursive=True)
                            )
                        except psutil.Error:
                            under = 0
                        workers[process.pid] = max(workers.get(process.pid, 0), under)
                    await asyncio.sleep(0.01)

            sampler = asyncio.create_task(sample())
            result = await watcher.play()
            sampler.cancel()
            return result, room_obj, workers

    return run(scenario())


def gone(pid):
    import psutil

    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def test_a_failing_bot_does_not_end_the_game(test_bots):
    result, room, _ = watch(["CRASHTEST", "F"], {"num_players": 2, "vps_to_win": 3})
    assert result.verified, result.report.errors
    assert room.status == "finished"
    assert any(a["forced"] == "bot_error" for a in room.actions)


def test_a_bot_raising_system_exit_does_not_take_the_server_down(test_bots):
    result, room, _ = watch(["EXITTEST", "F"], {"num_players": 2, "vps_to_win": 3})
    assert result.verified, result.report.errors
    assert room.status == "finished"
    assert any(a["forced"] == "bot_error" for a in room.actions)


def test_a_bot_that_overruns_is_killed_with_its_worker(test_bots):
    config = {"num_players": 2, "vps_to_win": 3, "decision_timeout_ms": 1000}
    result, room, workers = watch(["SLOWTEST", "F"], config)
    assert result.verified, result.report.errors
    assert any(a["forced"] == "timeout" for a in room.actions)
    assert len(workers) > 1, "each overrun killed the worker and started another"
    for _ in range(100):
        if all(gone(pid) for pid in workers):
            break
        time.sleep(0.05)
    assert all(gone(pid) for pid in workers), "no worker outlives its game"


def test_bots_in_workers_do_not_fork_process_pools():
    from catanatron.players import playouts
    from catanatron.server import botproc

    assert playouts.USE_MULTIPROCESSING  # the default, outside the server
    result, room, workers = watch(
        ["G:2", "R"], {"num_players": 2, "vps_to_win": 3, "decision_timeout_ms": 5000}
    )
    assert result.verified, result.report.errors
    assert room.status == "finished"
    assert workers, "the G bot ran in a worker"
    assert max(workers.values()) == 0, "G's process pool would be the worker's children"
    assert botproc.context().get_start_method() in ("forkserver", "spawn")


@pytest.mark.parametrize("spec", ["AB:5", "M:5000", "G:1000"])
def test_expensive_server_bots_are_refused(spec):
    async def scenario():
        async with serving() as (server, base):
            with pytest.raises(Refused, match="at most"):
                await create_room(base, {"num_players": 2}, [spec, "F"])

    run(scenario())


EXEC_BOT = """
import json, sys
log = open(sys.argv[1], "a")
for line in sys.stdin:
    message = json.loads(line)
    log.write(message["type"] + "\\n"); log.flush()
    if message["type"] == "hello":
        print(json.dumps({"protocol_version": 1, "name": "logger"}), flush=True)
    elif message["type"] == "decide":
        print(json.dumps({"action": message["playable_actions"][0]}), flush=True)
log.write("closed\\n")
"""


def test_exec_bots_seated_by_the_server_hear_before_and_after(tmp_path):
    script, log = tmp_path / "bot.py", tmp_path / "log.txt"
    script.write_text(EXEC_BOT)
    REGISTRY.register_source(f"exec:{sys.executable} {script} {log}", "EXECLOG")
    try:
        result, room, _ = watch(["EXECLOG", "R"], {"num_players": 2, "vps_to_win": 3})
    finally:
        REGISTRY.pop("EXECLOG", None)
    assert result.verified
    for _ in range(400):  # finishing runs on its own thread
        if log.exists() and "after" in log.read_text():
            break
        time.sleep(0.05)
    lines = log.read_text().split()
    assert lines[:2] == ["hello", "before"]
    # (The program may or may not get to log "closed" before it is stopped.)
    assert [line for line in lines if line != "closed"][-1] == "after"
    assert "decide" in lines


SLOW_BOT_FILE = """
import time
from catanatron.models.player import Player

class Sleepy(Player):
    def decide(self, game, playable_actions):
        time.sleep(60)
        return playable_actions[0]
"""


def test_the_server_exits_promptly_on_sigterm(tmp_path):
    bot = tmp_path / "sleepy.py"
    bot.write_text(SLOW_BOT_FILE)
    port = 23000 + os.getpid() % 2000
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "catanatron.server",
            "--port",
            str(port),
            "--key",
            str(tmp_path / "key"),
            "--bot",
            f"SLEEPY={bot}",
        ],
        stderr=subprocess.PIPE,
    )
    try:
        base = f"ws://127.0.0.1:{port}"

        async def create():
            for _ in range(100):
                try:
                    return await create_room(
                        base,
                        {"num_players": 2, "decision_timeout_ms": 100},
                        ["SLEEPY", "SLEEPY"],
                    )
                except OSError:
                    await asyncio.sleep(0.1)

        run(create())
        time.sleep(1.0)  # several sleepy decisions are now running
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=15) == 0
    finally:
        if process.poll() is None:
            process.kill()


# ===== H. limits =====
def test_connections_per_room_are_capped():
    async def scenario():
        limits = ServerLimits(connections_per_room=2)
        async with serving(limits=limits) as (server, base):
            room = await create_room(base, {"num_players": 2}, [None, None])
            url = f"{base}/room/{room['id']}"
            a, _ = await raw_connect(url)
            b, _ = await raw_connect(url)
            c = await websockets.connect(url)
            with pytest.raises(websockets.ConnectionClosed) as closed:
                await asyncio.wait_for(c.recv(), 5)
            await a.close()
            await b.close()
            return closed.value.rcvd.code

    assert run(scenario()) == 1013


def test_connections_that_have_not_joined_see_the_seats_change():
    """A page looking at a waiting room keeps its seat picker current."""

    async def scenario():
        async with serving() as (server, base):
            room = await create_room(base, {"num_players": 3}, [None, None, None])
            url = f"{base}/room/{room['id']}"
            idle, _ = await raw_connect(url)
            joiner, hello = await raw_connect(url)
            request, _, _ = join_request(hello)
            await joiner.send(json.dumps(request))
            await next_body(joiner, "joined")
            update = await next_body(idle, "room")
            await idle.close()
            await joiner.close()
            return update

    update = run(scenario())
    assert [s["kind"] for s in update["room"]["seats"]].count("remote") == 1


def test_room_creation_is_rate_limited():
    async def scenario():
        limits = ServerLimits(rooms_per_minute_per_ip=2)
        async with serving(limits=limits) as (server, base):
            await create_room(base, {"num_players": 2}, [None, None])
            await create_room(base, {"num_players": 2}, [None, None])
            with pytest.raises(Refused, match="too many rooms"):
                await create_room(base, {"num_players": 2}, [None, None])

    run(scenario())


def test_idle_rooms_expire_and_results_are_forgotten():
    async def scenario():
        async with serving() as (server, base):
            waiting = await create_room(base, {"num_players": 2}, [None, None])
            played = await create_room(
                base, {"num_players": 2, "vps_to_win": 3}, ["R", "R"]
            )
            room = server.rooms[played["id"]]
            await asyncio.wait_for(room.finished.wait(), 60)
            now = time.time()
            server.sweep(now + server.limits.idle_waiting_s + 1)
            assert waiting["id"] not in server.rooms
            assert played["id"] in server.rooms
            server.sweep(now + server.limits.finished_ttl_s + 1)
            assert played["id"] not in server.rooms

    run(scenario())


def test_an_abandoned_seat_is_freed_before_the_start():
    async def scenario():
        limits = ServerLimits(room=RoomLimits(seat_grace_s=0.2))
        async with serving(limits=limits) as (server, base):
            created = await create_room(base, {"num_players": 2}, [None, None])
            room = server.rooms[created["id"]]
            ws, hello = await raw_connect(f"{base}/room/{created['id']}")
            request, _, _ = join_request(hello, seat=0)
            await ws.send(json.dumps(request))
            await next_body(ws, "joined")
            assert room.seats[0].kind == "remote"
            await ws.close()
            await asyncio.sleep(0.6)
            return room.seats[0].kind

    assert run(scenario()) == "open"


def test_a_rejoin_closes_the_old_connection():
    async def scenario():
        async with serving() as (server, base):
            created = await create_room(base, {"num_players": 2}, [None, None])
            url = f"{base}/room/{created['id']}"
            old, hello = await raw_connect(url)
            request, key, _ = join_request(hello, seat=0)
            await old.send(json.dumps(request))
            joined = await next_body(old, "joined")
            new, _ = await raw_connect(url)
            await new.send(
                json.dumps(
                    {
                        "type": "join",
                        "protocol_version": PROTOCOL_VERSION,
                        "seat": 0,
                        "seat_key": joined["seat_key"],
                        "public_key": key.public_hex,
                        "last_seq": -1,
                    }
                )
            )
            await next_body(new, "joined")  # the seat's chain, resent
            with pytest.raises(websockets.ConnectionClosed) as closed:
                while True:
                    await asyncio.wait_for(old.recv(), 5)
            await new.close()
            return closed.value.rcvd.code

    assert run(scenario()) == 4001


@pytest.mark.parametrize("num_players,ok", [(3, True), (4, False)])
def test_maps_that_cannot_seat_everyone_are_refused(num_players, ok):
    payload = {"num_players": num_players, "map_type": "MINI"}
    if ok:
        GameConfig.from_json(payload)
    else:
        with pytest.raises(ConfigError, match="settlement spots"):
            GameConfig.from_json(payload)


# ===== the client library =====
def test_a_client_survives_many_brief_drops():
    async def scenario():
        async with serving() as (server, base):
            room = await create_room(
                base, {"num_players": 2, "vps_to_win": 5}, [None, "F"]
            )
            client, decide = bot_client(f"{base}/room/{room['id']}", "F")
            drops = []

            def on_message(body):
                assert (
                    body["type"] not in ("step", "decide")
                    or client.table.index == body["index"]
                ), "the hook runs after the table took the message in"
                if body["type"] == "step" and body["index"] % 4 == 0 and len(drops) < 8:
                    drops.append(body["index"])
                    asyncio.get_running_loop().create_task(client.connection.close())

            client.on_message = on_message
            return await client.play(decide), drops

    result, drops = run(scenario())
    assert len(drops) == 8
    assert result.verified, result.report.errors


# ===== J. the verifier never crashes =====
def test_catanatron_verify_reports_unreadable_and_malformed_files(tmp_path):
    clients, _, _ = play()
    good = tmp_path / "good.json"
    good.write_text(json.dumps(clients[0].transcript()))
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not json")
    deep = tmp_path / "deep.json"
    deep.write_text("[" * 100_000 + "]" * 100_000)
    odd = tmp_path / "odd.json"
    transcript = clients[0].transcript()
    transcript["messages"][0]["body"]["room"]["id"] = ["a", "list"]
    odd.write_text(json.dumps(transcript))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "catanatron.server.verify",
            str(good),
            str(bad_json),
            str(deep),
            str(odd),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert f"{good}: OK" in result.stdout
    for path in (bad_json, deep, odd):
        assert f"{path}: FAILED" in result.stdout
