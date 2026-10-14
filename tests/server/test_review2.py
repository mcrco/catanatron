"""Regression tests for the second review of the server: every attack
variant that survived protocol 2, and the defects its fixes introduced.

Dishonest servers are Rooms that cheat in one way and sign the result as
usual; dishonest or unlucky clients are GameClient subclasses.
"""

import asyncio
import gzip
import json
import os
import subprocess
import sys
import time
import uuid

import pytest
import websockets

from catanatron.models.player import Color
from catanatron.registry import REGISTRY
from catanatron.serialization import action_to_json
from catanatron.server.app import GameServer, ServerLimits, run_server
from catanatron.server.bot import ViewBot
from catanatron.server.client import GameClient, create_room
from catanatron.server.config import ConfigError, GameConfig
from catanatron.server.crypto import SigningKey, random_hex, seed_commitment
from catanatron.server.room import Room, RoomError, RoomLimits
from catanatron.server.verify import verify_transcript, verify_transcripts

from tests.server.helpers import (
    LaxClient,
    bot_client,
    join_request,
    next_body,
    raw_connect,
    run,
    serving,
)

HERE = os.path.dirname(__file__)
CONFIG = {"num_players": 2, "vps_to_win": 5, "decision_timeout_ms": 20_000}


def errors_of(client):
    return verify_transcript(json.loads(json.dumps(client.transcript()))).errors


def report_of(client):
    return verify_transcript(json.loads(json.dumps(client.transcript())))


def any_error(errors, *fragments):
    return any(all(f in e for f in fragments) for e in errors)


async def fixed_room(server, room_id, server_seed, config, room_class=Room):
    """A room with an id and server seed of our choosing (as a dishonest
    server could reuse them), registered on ``server``."""
    room = room_class(room_id, GameConfig.from_json(config), server.key)
    room.server_seed = server_seed
    room.seed_commitment = seed_commitment(server_seed)
    server.rooms[room_id] = room
    return room


# ===== 1. freshness: nothing carries over from another game =====
class ImpersonationRoom(Room):
    """Seats a player's key itself, with that player's join, seed and reveal
    from an earlier game in a room with the same id and server seed, and
    replays the player's signed moves from that game."""

    victim: dict = {}
    replayed = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seat, v = self.seats[1], self.victim
        seat.kind, seat.name, seat.public_key = "remote", "famous-bot", v["public_key"]
        seat.requested_seat, seat.client_commitment, seat.pop = (
            1,
            v["commitment"],
            v["pop"],
        )
        seat.client_seed, seat.reveal_sig = v["seed"], v["reveal"]

    async def _remote_decision(self, seat, playable):
        if seat.index != 1:
            return await super()._remote_decision(seat, playable)
        for item in self.victim["sent"]:
            statement = item["statement"]
            match = [a for a in playable if action_to_json(a) == statement["action"]]
            if match:
                self.replayed += 1
                return match[0], statement, item["sig"]
        return "timeout"


def test_a_players_signatures_cannot_be_carried_into_another_game():
    seed = random_hex()

    async def scenario():
        async with serving() as (server, base):
            await fixed_room(server, "fixedroom", seed, CONFIG)
            url = f"{base}/room/fixedroom"
            red, d1 = bot_client(url, "F", seat=0, verify=False)
            blue, d2 = bot_client(url, "F", seat=1, verify=False)
            await asyncio.gather(red.play(d1), blue.play(d2))
            first = server.rooms["fixedroom"]
            ImpersonationRoom.victim = {
                "public_key": blue.key.public_hex,
                "commitment": first.seats[1].client_commitment,
                "pop": first.seats[1].pop,
                "seed": first.seats[1].client_seed,
                "reveal": first.seats[1].reveal_sig,
                "sent": blue.sent,
            }
            del server.rooms["fixedroom"]
            second = await fixed_room(
                server, "fixedroom", seed, CONFIG, ImpersonationRoom
            )
            red2, d3 = bot_client(
                url, "F", client_class=LaxClient, seat=0, verify=False
            )
            await red2.play(d3)
            return report_of(blue), red2, second

    first_report, red2, second = run(scenario())
    assert first_report.ok, first_report.errors
    errors = errors_of(red2)
    assert any_error(errors, "did not reveal its seed for these commitments"), errors
    if second.replayed:
        assert any_error(errors, "signed statement is for another move or game"), errors


def test_a_server_naming_different_games_to_different_recipients_is_caught():
    """One key, one room id, one seed commitment: two games, each shown to a
    different spectator under its own game id."""
    key, seed = SigningKey.generate(), random_hex()
    config = {"num_players": 2, "vps_to_win": 4, "decision_timeout_ms": 5000}

    async def one(game_id):
        server = GameServer(key)
        ws_server = await run_server(server, "127.0.0.1", 0)
        port = ws_server.sockets[0].getsockname()[1]
        try:
            room = await fixed_room(server, "splitroom", seed, config)
            room.game_id = game_id
            room.set_bot(0, "R")
            watcher = GameClient(
                f"ws://127.0.0.1:{port}/room/splitroom",
                spectate=True,
                transcript_dir=None,
                verify=False,
            )
            watching = asyncio.create_task(watcher.play())
            await asyncio.sleep(0.3)
            room.set_bot(1, "R")
            await watching
            return watcher.transcript()
        finally:
            server.shutdown()
            ws_server.close()
            await ws_server.wait_closed()

    async def scenario():
        return await one("game-A"), await one("game-B")

    a, b = run(scenario())
    assert verify_transcript(a).ok and verify_transcript(b).ok, "each alone looks fine"
    reports = verify_transcripts([a, b])
    for report in reports:
        assert any_error(
            report.errors, "equivocation", "another game id"
        ), report.errors


# ===== 2. refusals and coercion =====
class RefusingRoom(Room):
    """Refuses RED's first legal answer (with a receipt saying so)."""

    coerced = False

    def _accept(self, seat, pending, statement, signature):
        if seat.color.value == "RED" and not self.coerced:
            self.coerced = True
            return "refused: is not legal now"
        return super()._accept(seat, pending, statement, signature)


class SilentRoom(Room):
    """Ignores RED's first answer: no receipt, no error."""

    coerced = False

    def on_action(self, seat, peer, request):
        if seat.color.value == "RED" and not self.coerced and seat.pending is not None:
            self.coerced = True
            return
        super().on_action(seat, peer, request)


class ErrorInsteadRoom(Room):
    """Answers RED's first answer with a signed error instead of a receipt."""

    coerced = False

    def on_action(self, seat, peer, request):
        if seat.color.value == "RED" and not self.coerced and seat.pending is not None:
            self.coerced = True
            raise RoomError("not_pending", "no decision is pending for this seat")
        super().on_action(seat, peer, request)


@pytest.mark.parametrize(
    "room_class,message",
    [
        (RefusingRoom, "had this client's legal, signed move"),
        (SilentRoom, "never acknowledged"),
        (ErrorInsteadRoom, "never acknowledged"),
    ],
)
def test_a_legal_move_the_server_overrules_is_reported(room_class, message):
    config = {**CONFIG, "decision_timeout_ms": 1500}

    async def scenario():
        async with serving(room_class=room_class) as (server, base):
            room = await create_room(base, config, [None, "F"])
            client, decide = bot_client(f"{base}/room/{room['id']}", "F", verify=False)
            await client.play(decide)
            return client, server.rooms[room["id"]]

    client, room = run(scenario())
    assert room.coerced
    assert any_error(errors_of(client), message), errors_of(client)


class DroppingClient(GameClient):
    """Its connection drops right after it answers, once: before the step
    that plays the answer arrives."""

    dropped = False

    async def _answer_with(self, action):
        await super()._answer_with(action)
        if not self.dropped and self._pending["body"]["index"] > 2:
            self.dropped = True
            await self.connection.close()


class LostAnswerClient(GameClient):
    """Its first answer is lost with the connection (never reaches the
    server); it is resent after the reconnect."""

    lost = False

    async def _answer_with(self, action):
        if not self.lost and self._pending["body"]["index"] > 2:
            self.lost = True
            send, self.connection.send = self.connection.send, _swallow
            try:
                await super()._answer_with(action)
            finally:
                self.connection.send = send
            await self.connection.close()
            return
        await super()._answer_with(action)


async def _swallow(_):
    return None


class LateClient(GameClient):
    """One answer reaches the server only after its deadline."""

    delayed = False

    async def _answer_with(self, action):
        if not self.delayed and self._pending["body"]["index"] > 2:
            self.delayed = True
            send = self.connection.send

            async def later(text):
                await asyncio.sleep(1.5)
                await send(text)

            self.connection.send = later
            try:
                await super()._answer_with(action)
            finally:
                self.connection.send = send
            return
        await super()._answer_with(action)


class FramingClient(GameClient):
    """Sends a badly signed copy of its answer first, to frame the server
    with the refusal."""

    async def _answer_with(self, action):
        await super()._answer_with(action)
        bad = dict(self._answer, sig="00" * 64)
        self.sent.append(
            {"statement": bad["statement"], "sig": bad["sig"], "sent_after": 0}
        )
        await self.connection.send(json.dumps(bad))
        await self.connection.send(json.dumps(self._answer))


@pytest.mark.parametrize(
    "client_class,timeout,warned",
    [
        (DroppingClient, 20_000, False),
        (LostAnswerClient, 20_000, False),
        (LateClient, 1000, True),
        (FramingClient, 20_000, False),
    ],
)
def test_an_honest_server_is_not_accused(client_class, timeout, warned):
    config = {**CONFIG, "decision_timeout_ms": timeout}

    async def scenario():
        async with serving() as (server, base):
            room = await create_room(base, config, [None, "F"])
            client, decide = bot_client(
                f"{base}/room/{room['id']}",
                "F",
                client_class=client_class,
                verify=False,
            )
            await client.play(decide)
            return client

    client = run(scenario())
    report = report_of(client)
    assert report.ok, report.errors
    if warned:
        assert any("late" in w for w in report.warnings), report.warnings


# ===== 3. withheld reveals: see test_review.py (restart rounds, bans) =====


# ===== 4. forced moves =====
class HiddenAutoplayRoom(Room):
    """Plays every one-option move without asking, hidden or not."""

    def _autoplay(self, game):
        return len(game.playable_actions) == 1


class AskForcedRoom(Room):
    """Asks for every move, even the ones everyone knows are forced."""

    def _autoplay(self, game):
        return False


@pytest.mark.parametrize(
    "room_class,message",
    [
        (HiddenAutoplayRoom, "played as forced, but nobody could know it was"),
        (AskForcedRoom, "a decide for a move everyone knew was forced"),
    ],
)
def test_forced_moves_follow_what_the_table_can_see(room_class, message):
    config = {"num_players": 2, "vps_to_win": 6, "decision_timeout_ms": 20_000}

    async def scenario():
        async with serving(room_class=room_class) as (server, base):
            room = await create_room(base, config, [None, "R"])
            client, decide = bot_client(
                f"{base}/room/{room['id']}", "R", client_class=LaxClient, verify=False
            )
            await client.play(decide)
            return client

    client = run(scenario())
    assert any_error(errors_of(client), message), errors_of(client)


def test_a_bot_answers_a_forced_move_as_slowly_as_a_real_one():
    from catanatron.game import Game
    from catanatron.models.decks import BASE_SUPPLY
    from catanatron.models.player import RandomPlayer
    from catanatron.serialization import client_view, state_to_json

    game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=1)
    color = game.state.current_color()
    view = client_view(state_to_json(game), color)
    bot = ViewBot("R", color)
    only = [action_to_json(game.playable_actions[0])]
    began = time.monotonic()
    bot.decide(view, only, BASE_SUPPLY)
    assert time.monotonic() - began < 0.05, "no history yet: no pace to keep"
    bot.timings.extend([0.25] * 5)
    began = time.monotonic()
    bot.decide(view, only, BASE_SUPPLY)
    assert time.monotonic() - began >= 0.25


# ===== 5. bots =====
EXEC_BOT = """
import json, sys
log = open(sys.argv[1], "a")
observe = len(sys.argv) > 2
for line in sys.stdin:
    message = json.loads(line)
    log.write(message["type"] + "\\n"); log.flush()
    if message["type"] == "hello":
        print(json.dumps({"protocol_version": 1, "name": "logger", "observe": observe}), flush=True)
    elif message["type"] == "decide":
        print(json.dumps({"action": message["playable_actions"][0]}), flush=True)
"""


@pytest.fixture
def exec_bot(tmp_path):
    def make(name, observe=False):
        script, log = tmp_path / f"{name}.py", tmp_path / f"{name}.log"
        script.write_text(EXEC_BOT)
        command = f"exec:{sys.executable} {script} {log}" + (
            " observe" if observe else ""
        )
        REGISTRY.register_source(command, name)
        made.append(name)
        return log

    made = []
    yield make
    for name in made:
        REGISTRY.pop(name, None)
        REGISTRY.sources.pop(name, None)


def wait_for(path, text, seconds=20):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if path.exists() and text in path.read_text().split():
            return True
        time.sleep(0.05)
    return False


def test_exec_bots_that_observe_get_every_step(exec_bot):
    log = exec_bot("EXECOBS", observe=True)

    async def scenario():
        async with serving() as (server, base):
            room = await create_room(
                base, {"num_players": 2, "vps_to_win": 3}, ["EXECOBS", "R"]
            )
            await asyncio.wait_for(server.rooms[room["id"]].finished.wait(), 120)

    run(scenario())
    assert wait_for(log, "after")
    lines = log.read_text().split()
    assert lines.count("step") > 10


def test_exec_bots_hear_after_when_a_game_is_called_off(exec_bot):
    log = exec_bot("EXECABORT")

    async def scenario():
        async with serving() as (server, base):
            config = {"num_players": 2, "vps_to_win": 10, "decision_timeout_ms": 5000}
            room = await create_room(base, config, ["EXECABORT", "F"])
            room_obj = server.rooms[room["id"]]
            while len(room_obj.actions) < 20:
                await asyncio.sleep(0.05)
            room_obj.abort("called off by the test")

    run(scenario())
    assert wait_for(log, "after"), log.read_text()


CAPTURED_STDIO_WORKER = """
import asyncio, io, sys

class Captured(io.StringIO):  # as Textual's run_test leaves sys.stderr
    def fileno(self):
        return -1

sys.stdout, sys.stderr = Captured(), Captured()
from catanatron.server.botproc import BotWorker

async def main():
    worker = BotWorker()
    reply = await worker.call(("ping",), 60)
    worker.stop()
    return reply

if __name__ == "__main__":
    reply = asyncio.run(main())
    sys.__stdout__.write(repr(reply))
"""


def test_bot_workers_start_while_the_host_app_captures_stdio(tmp_path):
    # The resource tracker would be handed sys.stderr.fileno() (-1 here) and
    # refuse to start: "bad value(s) in fds_to_keep".
    script = tmp_path / "captured.py"
    script.write_text(CAPTURED_STDIO_WORKER)
    done = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=240
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout == "('pong',)"


def test_a_room_whose_bot_worker_cannot_start_plays_on(monkeypatch):
    from catanatron.server import botproc

    def broken(self):
        raise OSError("no more processes")

    monkeypatch.setattr(botproc.BotWorker, "start", broken)

    async def scenario():
        async with serving() as (server, base):
            room = await create_room(base, CONFIG, ["F", "F"])
            room_obj = server.rooms[room["id"]]
            deadline = time.monotonic() + 120
            while len(room_obj.actions) < 10 and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
            assert room_obj.status == "playing"
            assert len(room_obj.actions) >= 10
            forced = {entry["forced"] for entry in room_obj.actions}
            assert forced <= {"bot_error", "only"} and "bot_error" in forced

    run(scenario())


SLOW_G_SERVER = """
import os, sys
os.environ["CATANATRON_TEST_MARK"] = sys.argv[1]
from catanatron.server.app import main
if __name__ == "__main__":
    sys.exit(main(sys.argv[2:]))
"""


def marked_processes(mark):
    import psutil

    found = []
    for process in psutil.process_iter(["pid"]):
        try:
            if process.environ().get("CATANATRON_TEST_MARK") == mark:
                found.append(process.pid)
        except psutil.Error:
            continue
    return found


def test_the_server_exits_and_leaves_no_bot_processes_behind(tmp_path):
    mark = uuid.uuid4().hex
    script = tmp_path / "server.py"
    script.write_text(SLOW_G_SERVER)
    port = 25000 + os.getpid() % 2000
    process = subprocess.Popen(
        [
            sys.executable,
            str(script),
            mark,
            "--port",
            str(port),
            "--key",
            str(tmp_path / "key"),
        ],
        stderr=subprocess.DEVNULL,
    )
    try:

        async def create():
            for _ in range(100):
                try:
                    return await create_room(
                        f"ws://127.0.0.1:{port}",
                        {"num_players": 2, "decision_timeout_ms": 30_000},
                        ["G:100", "G:100"],
                    )
                except OSError:
                    await asyncio.sleep(0.1)

        run(create())
        time.sleep(2.0)  # the G bots are deep in their playouts now
        assert len(marked_processes(mark)) > 1, "the server runs bot workers"
        process.terminate()
        assert process.wait(timeout=15) == 0
        deadline = time.time() + 10
        while marked_processes(mark) and time.time() < deadline:
            time.sleep(0.1)
        assert marked_processes(mark) == [], "no bot process outlives the server"
    finally:
        if process.poll() is None:
            process.kill()


def test_one_address_cannot_fill_the_server_with_bot_games():
    async def scenario():
        limits = ServerLimits(bot_seats_per_ip=4, max_bot_rooms=3)
        async with serving(limits=limits) as (server, base):
            config = {"num_players": 2, "decision_timeout_ms": 60_000}
            await create_room(base, config, [None, "F"])
            await create_room(base, config, [None, "F"])
            await create_room(base, config, [None, "F"])
            from catanatron.server.client import Refused

            with pytest.raises(Refused, match="too many"):
                await create_room(base, config, [None, "F"])

    run(scenario())


# ===== 7. the client library =====
def test_reconnects_while_another_seat_thinks_do_not_use_up_the_client():
    async def scenario():
        async with serving() as (server, base):
            config = {"num_players": 2, "vps_to_win": 4, "decision_timeout_ms": 20_000}
            room = await create_room(base, config, [None, "F"])
            client, bot_decide = bot_client(f"{base}/room/{room['id']}", "F")
            drops = 0

            async def decide(view, playable):
                # Our seat holds the game until the connection has flapped
                # eight times, so that a quick game cannot end first.
                while drops < 8:
                    await asyncio.sleep(0.05)
                return await asyncio.to_thread(bot_decide, view, playable)

            async def flap():
                # Each connection, once the game is on and the client has
                # verified the server's hello on it, is dropped: eight drops,
                # more than the client's five reconnect attempts.
                nonlocal drops
                dropped = None
                while drops < 8:
                    await asyncio.sleep(0.02)
                    ws = client.connection
                    if ws is None or ws is dropped or client.table.state is None:
                        continue
                    if client._attempts:  # back, but not yet in
                        continue
                    await asyncio.sleep(0.1)
                    dropped = ws
                    drops += 1
                    await ws.close()

            flapping = asyncio.create_task(flap())
            result = await client.play(decide)
            flapping.cancel()
            return result, drops

    result, drops = run(scenario())
    assert drops == 8
    assert result.verified, result.report.errors


def test_catanatron_verify_reports_a_corrupt_gzip_as_failed(tmp_path):
    good = tmp_path / "good.json.gz"
    with gzip.open(good, "wt") as file:
        file.write("{}")
    corrupt = tmp_path / "corrupt.json.gz"
    data = bytearray(good.read_bytes())
    data[12:20] = b"\xff" * 8
    corrupt.write_bytes(bytes(data))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "catanatron.server.verify",
            "--json",
            str(corrupt),
            str(good),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    reports = json.loads(result.stdout)
    assert len(reports) == 2 and not reports[0]["ok"] and not reports[1]["ok"]


# ===== 8. maps =====
def test_a_map_that_runs_out_of_spots_is_refused_exactly():
    with open(os.path.join(HERE, "..", "maps", "eleven_spots.json")) as file:
        spec = json.load(file)
    GameConfig.from_json({"num_players": 5, "map": spec})
    with pytest.raises(ConfigError, match="after 11 settlements"):
        GameConfig.from_json({"num_players": 6, "map": spec})


# ===== new defects: what a client may make the server keep =====
def test_errors_do_not_echo_requests_and_bad_requests_are_bounded():
    async def scenario():
        limits = ServerLimits(
            room=RoomLimits(max_answers=20), messages_per_s=1000.0, message_burst=1000
        )
        async with serving(limits=limits) as (server, base):
            room = await create_room(base, {"num_players": 2}, [None, None])
            ws, hello = await raw_connect(f"{base}/room/{room['id']}")
            request, _, _ = join_request(hello)
            await ws.send(json.dumps(request))
            await next_body(ws, "joined")
            await ws.send(json.dumps({"type": "x" * 100_000}))
            error = await next_body(ws, "error")
            closed = None
            try:
                for _ in range(100):
                    await ws.send(json.dumps({"type": "nonsense"}))
                    await asyncio.sleep(0)
                for _ in range(200):
                    await asyncio.wait_for(ws.recv(), 5)
            except websockets.ConnectionClosed as error_:
                closed = error_.rcvd.code if error_.rcvd else None
            seat = server.rooms[room["id"]].seats[0]
            return error, closed, seat.outbox.answers

    error, closed, answers = run(scenario())
    assert len(error["message"]) < 300
    assert closed == 4009
    assert answers <= 20


def test_a_flooding_connection_is_closed():
    async def scenario():
        limits = ServerLimits(messages_per_s=5.0, message_burst=10)
        async with serving(limits=limits) as (server, base):
            room = await create_room(base, {"num_players": 2}, [None, None])
            ws, _ = await raw_connect(f"{base}/room/{room['id']}")
            try:
                for _ in range(50):
                    await ws.send(json.dumps({"type": "ping"}))
                await asyncio.wait_for(ws.wait_closed(), 5)
            except websockets.ConnectionClosed:
                pass
            return ws.close_code

    assert run(scenario()) == 1008
