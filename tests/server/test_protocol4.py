"""Protocol 4: the invariants of PROTOCOL.md, attacked.

One commitments per room (a withheld reveal voids the room; there is no
second round), rejoining by a signature over the connection's nonce, atomic
deadlines, a receipt (or a closed connection) for every submission, and the
verifier's split between what anyone can be shown and what only the client
knows. Each dishonest server is a Room that cheats in one way and signs the
result as usual.
"""

import asyncio
import itertools
import json
import os
import sys
import time
import uuid

import pytest
import websockets
from websockets.exceptions import ConnectionClosedError

from catanatron.registry import REGISTRY
from catanatron.server.app import GameServer, ServerLimits, _normalize, run_server
from catanatron.server.client import (
    GameClient,
    ServerMisbehaved,
    create_room,
)
from catanatron.server.config import GameConfig
from catanatron.server.crypto import (
    SigningKey,
    random_hex,
    rejoin_statement,
    seed_commitment,
)
from catanatron.server.messages import PROTOCOL_VERSION, receipt_body
from catanatron.server.room import Room, RoomLimits
from catanatron.server.verify import verify_transcript, verify_transcripts

from tests.server.helpers import (
    bot_client,
    join_request,
    next_body,
    raw_connect,
    run,
    serving,
)

CONFIG = {"num_players": 2, "vps_to_win": 4, "decision_timeout_ms": 20_000}


def report_of(client):
    return verify_transcript(json.loads(json.dumps(client.transcript())))


def any_error(errors, *fragments):
    return any(all(f in e for f in fragments) for e in errors)


class SilentClient(GameClient):
    """Joins, then never reveals its seed."""

    async def _handle(self, ws, body, decide):
        if body.get("type") == "commitments":
            self.commitments = body
            return None
        return await super()._handle(ws, body, decide)


class RecordingReveals(Room):
    """Remembers every reveal signature each key sent."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reveals: dict = {}

    def on_reveal(self, seat, peer, request):
        self.reveals.setdefault(seat.public_key, set()).add(request.get("sig"))
        super().on_reveal(seat, peer, request)


# ===== I1/I2: one commitments, one reveal =====
class GrindRoom(RecordingReveals):
    """Round 3's attack: once every seat has revealed over C1 it knows every
    seed, so it issues a second commitments C' (differing only in a field
    nobody needs) and drops the seats, hoping their reconnect signs a reveal
    over C'."""

    ground = False

    def _start(self):
        if self.ground:
            return super()._start()
        self.ground = True
        self.status = "revealing"
        self.commitments = dict(
            self.commitments,
            reveal_timeout_ms=self.commitments["reveal_timeout_ms"] + 1,
        )
        from catanatron.server.crypto import digest

        self.commitments_digest = digest(self.commitments)
        for seat in self.seats:
            if seat.kind == "remote":
                seat.client_seed = seat.reveal_sig = None
                if seat.outbox.peer is not None:
                    seat.outbox.peer.kick(1011, "drop")  # first the drop...
                seat.outbox.push(self.commitments)  # ...then C', on the chain


def test_a_second_commitments_is_never_revealed_into():
    async def scenario():
        async with serving(room_class=GrindRoom) as (server, base):
            room = await create_room(base, CONFIG, [None, "F"])
            client, decide = bot_client(f"{base}/room/{room['id']}", verify=False)
            with pytest.raises(ServerMisbehaved, match="second, different commitments"):
                await client.play(decide)
            return client, server.rooms[room["id"]]

    client, room = run(scenario())
    assert room.ground
    # One reveal, ever: whatever was resent after the drop is the same
    # signature, over the first commitments.
    assert room.reveals[client.key.public_hex] == {client._reveal["sig"]}


class DoubleCommitRoom(Room):
    """Sends the (same) commitments twice on every chain."""

    def _push_commitments(self, outbox):
        super()._push_commitments(outbox)
        outbox.push(self.commitments)


def test_a_transcript_with_two_commitments_fails():
    async def scenario():
        async with serving(room_class=DoubleCommitRoom) as (server, base):
            room = await create_room(base, CONFIG, [None, "F"])
            client, decide = bot_client(f"{base}/room/{room['id']}", verify=False)
            await client.play(decide)  # an identical copy changes nothing live
            return client

    client = run(scenario())
    assert len(client.commitments) and client._reveal is not None
    errors = report_of(client).errors
    assert any_error(errors, "a second commitments message"), errors


class RestartRoom(Room):
    """Sends protocol 3's restart message before the commitments."""

    def _maybe_lock(self):
        if self.status == "waiting" and all(s.kind != "open" for s in self.seats):
            for outbox in self._outboxes(lobby=True):
                outbox.push({"type": "restart", "room_id": self.id, "blame": []})
        super()._maybe_lock()


def test_a_restart_message_fails():
    async def scenario():
        async with serving(room_class=RestartRoom) as (server, base):
            room = await create_room(base, CONFIG, [None, "F"])
            client, decide = bot_client(f"{base}/room/{room['id']}", verify=False)
            with pytest.raises(ServerMisbehaved, match="unknown type"):
                await client.play(decide)  # the live client refuses it
            return client

    errors = report_of(run(scenario())).errors
    assert any_error(errors, "unexpected 'restart' before the game"), errors


# ===== I4/I7: voids =====
class BadVoidRoom(Room):
    """Voids the room with a seed that is not the one it committed to."""

    def abort(self, reason, blame=None):
        self.server_seed = random_hex()
        super().abort(reason, blame)


class TrustingVoids(GameClient):
    """Takes a void in without checking it (so the verifier is tested)."""

    async def _handle(self, ws, body, decide):
        if body.get("type") == "aborted":
            return {"aborted": body}
        return await super()._handle(ws, body, decide)


def test_a_void_must_open_the_seed_commitment():
    async def scenario():
        limits = ServerLimits(room=RoomLimits(reveal_timeout_ms=400))
        async with serving(room_class=BadVoidRoom, limits=limits) as (server, base):
            room = await create_room(base, {**CONFIG, "num_players": 3}, [None] * 3)
            url = f"{base}/room/{room['id']}"
            strict, d1 = bot_client(url, seat=0, verify=False)
            lax, d2 = bot_client(url, client_class=TrustingVoids, seat=1, verify=False)
            silent, d3 = bot_client(url, client_class=SilentClient, seat=2)
            return (
                await asyncio.gather(
                    strict.play(d1),
                    lax.play(d2),
                    silent.play(d3),
                    return_exceptions=True,
                ),
                lax,
            )

    (strict, lax_result, _), lax = run(scenario())
    assert isinstance(strict, ServerMisbehaved)
    report = verify_transcript(lax.transcript())
    assert report.void is not None
    assert any_error(report.errors, "void does not reveal the seed"), report.errors


class BlameTheHonestRoom(Room):
    """Acknowledges every reveal, then voids the room blaming seat 0."""

    def _start(self):
        self.abort("a seat did not reveal its seed in time: the game is void", [0])


class IgnoreSeatZeroReveal(Room):
    """Never answers seat 0's reveal, then blames it (a lie it cannot be
    held to: the client's word against the server's)."""

    def on_reveal(self, seat, peer, request):
        if seat.index != 0:
            super().on_reveal(seat, peer, request)


@pytest.mark.parametrize(
    "room_class,provable", [(BlameTheHonestRoom, True), (IgnoreSeatZeroReveal, False)]
)
def test_blaming_a_seat_whose_reveal_was_acknowledged_is_caught(room_class, provable):
    async def scenario():
        limits = ServerLimits(room=RoomLimits(reveal_timeout_ms=600))
        async with serving(room_class=room_class, limits=limits) as (server, base):
            room = await create_room(base, CONFIG, [None, "F"])
            client, decide = bot_client(f"{base}/room/{room['id']}", verify=False)
            result = await client.play(decide)
            return client, result

    client, result = run(scenario())
    assert result.aborted is not None and result.aborted["blame"] == [0]
    report = report_of(client)
    assert report.void is not None
    if provable:
        assert any_error(report.errors, "after acknowledging its reveal"), report.errors
    else:
        assert not report.errors, report.errors
        assert any("not provable" in w for w in report.warnings), report.warnings


# ===== I9: rejoining =====
class LoseJoinedClient(GameClient):
    """The link dies as 'joined' arrives: the client never sees it."""

    lost = False

    def _accept(self, envelope):
        if not self.lost and envelope.get("body", {}).get("type") == "joined":
            self.lost = True
            asyncio.ensure_future(self.connection.close())
            raise ConnectionClosedError(None, None)
        return super()._accept(envelope)


def test_a_client_that_loses_joined_takes_its_seat_back():
    """Round 3: the client had no seat key, could not rejoin, and was
    blamed (and its address banned) as a withholder."""

    async def scenario():
        limits = ServerLimits(room=RoomLimits(reveal_timeout_ms=3000))
        async with serving(limits=limits) as (server, base):
            room = await create_room(base, CONFIG, [None, "F"])
            client, decide = bot_client(
                f"{base}/room/{room['id']}", client_class=LoseJoinedClient, verify=False
            )
            result = await client.play(decide)
            return client, result, server

    client, result, server = run(scenario())
    assert client.lost and result.aborted is None
    assert report_of(client).ok
    assert not server.blames and not server.banned_keys


def test_a_rejoin_signature_is_good_on_one_connection_only():
    async def scenario():
        async with serving() as (server, base):
            created = await create_room(base, {"num_players": 3}, [None, None, None])
            url = f"{base}/room/{created['id']}"
            first, hello = await raw_connect(url)
            request, key, _ = join_request(hello, seat=0)
            await first.send(json.dumps(request))
            await next_body(first, "joined")

            second, hello2 = await raw_connect(url)
            rejoin = {
                "type": "rejoin",
                "protocol_version": PROTOCOL_VERSION,
                "public_key": key.public_hex,
                "sig": key.sign(rejoin_statement(created["id"], hello2["nonce"])),
                "last_seq": -1,
            }
            await second.send(json.dumps(rejoin))
            taken = await next_body(second, "hello", "error")  # the seat's chain

            third, _ = await raw_connect(url)
            await third.send(json.dumps(rejoin))  # replayed on another connection
            replayed = await next_body(third, "error")
            # The pop is public (every commitments carries it): replaying the
            # join takes nothing either.
            await third.send(json.dumps(request))
            copied = await next_body(third, "error")
            for ws in (first, second, third):
                await ws.close()
            return taken, replayed, copied

    taken, replayed, copied = run(scenario())
    assert taken["type"] == "hello"
    assert replayed["code"] == "bad_join" and replayed["answers"] == "rejoin"
    assert copied["code"] == "bad_join"  # the pop names another connection


# ===== I13: every submission answered =====
def test_every_submission_gets_a_receipt_or_a_closed_connection():
    async def scenario():
        limits = ServerLimits(
            room=RoomLimits(receipt_margin=4, receipts_per_decision=1)
        )
        async with serving(limits=limits) as (server, base):
            created = await create_room(base, CONFIG, [None, "R"])
            ws, hello = await raw_connect(f"{base}/room/{created['id']}")
            request, key, seed = join_request(hello)
            await ws.send(json.dumps(request))
            commitments = await next_body(ws, "commitments")
            from catanatron.server.crypto import digest, reveal_statement

            sig = key.sign(reveal_statement(digest(commitments)))
            await ws.send(
                json.dumps({"type": "reveal", "client_seed": seed, "sig": sig})
            )
            await next_body(ws, "before")
            junk = {"type": "action", "statement": {"index": 0}, "sig": "00" * 64}
            for _ in range(20):
                await ws.send(json.dumps(junk))
            with pytest.raises(websockets.ConnectionClosed) as closed:
                while True:
                    await next_body(ws)
            room = server.rooms[created["id"]]
            return closed.value.rcvd.code, room.seats[0]

    code, seat = run(scenario())
    assert code == 4009
    # Every submission the seat was allowed has its receipt on the seat's
    # chain (the reveal's included; a rejoin gets them), then the connection
    # was closed: never a submission silently dropped.
    chain = [e["body"] for e in seat.outbox.envelopes]
    receipts = [b for b in chain if b["type"] == "received"]
    assert len(receipts) == seat.receipts >= 4
    assert not any(b["type"] == "error" for b in chain)


def test_receipts_do_not_use_up_the_error_allowance():
    """Round 3: receipts counted against max_answers, so a long game was
    kicked on every move once it passed the limit."""

    async def scenario():
        limits = ServerLimits(room=RoomLimits(max_answers=3))
        async with serving(limits=limits) as (server, base):
            room = await create_room(base, CONFIG, [None, "R"])
            client, decide = bot_client(f"{base}/room/{room['id']}", "R", verify=False)
            await client.play(decide)
            return client, server.rooms[room["id"]]

    client, room = run(scenario())
    assert room.seats[0].receipts > 3
    report = report_of(client)
    assert report.ok and not report.warnings, (report.errors, report.warnings)


class PregameActionClient(GameClient):
    """Sends a (meaningless) action as soon as it is seated."""

    async def _handle(self, ws, body, decide):
        result = await super()._handle(ws, body, decide)
        if body.get("type") == "joined":
            bad = {"type": "action", "statement": {"index": 0}, "sig": "00" * 64}
            await ws.send(json.dumps(bad))
        return result


def test_an_action_before_the_game_gets_an_error_and_verifies():
    async def scenario():
        async with serving() as (server, base):
            room = await create_room(base, CONFIG, [None, "R"])
            client, decide = bot_client(
                f"{base}/room/{room['id']}",
                "R",
                client_class=PregameActionClient,
                verify=False,
            )
            await client.play(decide)
            return client

    client = run(scenario())
    kinds = [m["body"]["type"] for m in client.messages]
    errors = [m["body"] for m in client.messages if m["body"]["type"] == "error"]
    assert (
        errors and errors[0]["code"] == "no_game" and errors[0]["answers"] == "action"
    )
    assert kinds.index("error") < kinds.index("before")
    report = report_of(client)
    assert report.ok, report.errors


# ===== I14: deadlines =====
def test_answers_at_the_deadline_never_frame_an_honest_server():
    """Round 3: an answer read in the loop pass between the deadline and the
    step got a receipt before the step, which the verifier took as proof
    that the server had it in time."""
    timeout_ms = 150
    offsets = itertools.cycle([-0.004, -0.002, -0.001, 0.0, 0.001, 0.002, 0.004])

    async def scenario():
        config = {**CONFIG, "vps_to_win": 3, "decision_timeout_ms": timeout_ms}
        async with serving() as (server, base):
            room = await create_room(base, config, [None, "R"])
            client = GameClient(
                f"{base}/room/{room['id']}", transcript_dir=None, verify=False
            )

            async def decide(view, playable):
                await asyncio.sleep(timeout_ms / 1000 + next(offsets))
                return playable[-1]

            await client.play(decide)
            return client, server.rooms[room["id"]]

    client, room = run(scenario())
    timeouts = [a for a in room.actions if a["forced"] == "timeout"]
    assert timeouts, "some answers did miss the deadline"
    report = report_of(client)
    assert report.ok, report.errors


class AcceptButIgnoreRoom(Room):
    """Tells RED its first legal answer was accepted, holds that receipt
    until after the step, and plays the default instead."""

    lied = False

    def on_action(self, seat, peer, request):
        pending = seat.pending
        if (
            not self.lied
            and seat.color.value == "RED"
            and pending is not None
            and not pending.future.done()
            and len(pending.playable) > 1
        ):
            from catanatron.server.crypto import submission_digest

            self.lied = True
            submission = submission_digest(request.get("statement"), request.get("sig"))
            seat.held.append(receipt_body(submission, pending.index, True, "accepted"))
            pending.future.set_result("timeout")
            return
        super().on_action(seat, peer, request)


def test_an_accepted_answer_the_server_did_not_play_is_caught():
    async def scenario():
        async with serving(room_class=AcceptButIgnoreRoom) as (server, base):
            room = await create_room(base, CONFIG, [None, "R"])
            client, decide = bot_client(f"{base}/room/{room['id']}", "R", verify=False)
            await client.play(decide)
            return client, server.rooms[room["id"]]

    client, room = run(scenario())
    assert room.lied
    errors = report_of(client).errors
    assert any_error(errors, "had this client's legal, signed move"), errors


# ===== I15: equivocation =====
async def _fixed(server, room_id, server_seed, config, room_class=Room):
    room = room_class(room_id, GameConfig.from_json(config), server.key)
    room.server_seed = server_seed
    room.seed_commitment = seed_commitment(server_seed)
    server.rooms[room_id] = room
    return room


async def _watch_room(key, seed, setup, watchers=1, room_class=Room, limits=None):
    """Spectator transcripts of one room (id 'eqroom') hosted under ``key``
    with ``seed``; ``setup(room, base)`` seats it."""
    server = GameServer(key, **({"limits": limits} if limits else {}))
    ws_server = await run_server(server, "127.0.0.1", 0)
    port = ws_server.sockets[0].getsockname()[1]
    base = f"ws://127.0.0.1:{port}"
    try:
        room = await _fixed(server, "eqroom", seed, CONFIG, room_class)
        clients = [
            GameClient(
                f"{base}/room/eqroom", spectate=True, transcript_dir=None, verify=False
            )
            for _ in range(watchers)
        ]
        tasks = [asyncio.create_task(c.play()) for c in clients]
        await asyncio.sleep(0.3)
        extra = await setup(room, base)
        await asyncio.gather(*tasks)
        if extra is not None:
            await extra
        return [c.transcript() for c in clients]
    finally:
        server.shutdown()
        ws_server.close()
        await ws_server.wait_closed()


def test_every_pair_of_transcripts_of_a_room_is_compared():
    """Round 3: members were compared with the group's first transcript
    only, so [A1, A2, B] failed A1 and B but passed A2."""
    key, seed = SigningKey.generate(), random_hex()

    async def bots(room, base):
        room.set_bot(0, "R")
        room.set_bot(1, "R")

    async def scenario():
        a = await _watch_room(key, seed, bots, watchers=2)
        b = await _watch_room(key, seed, bots, watchers=1)
        return a + b

    a1, a2, b = run(scenario())
    for order in itertools.permutations([a1, a2, b]):
        reports = verify_transcripts(list(order))
        for report in reports:
            assert any_error(report.errors, "equivocation"), report.errors


def test_a_void_for_some_and_a_game_for_others_is_caught():
    key, seed = SigningKey.generate(), random_hex()
    limits = ServerLimits(room=RoomLimits(reveal_timeout_ms=400))

    async def withheld(room, base):
        room.set_bot(1, "R")
        silent, decide = bot_client(
            f"{base}/room/eqroom", client_class=SilentClient, seat=0
        )
        return asyncio.create_task(silent.play(decide))

    async def played(room, base):
        room.set_bot(0, "R")
        room.set_bot(1, "R")

    async def scenario():
        void = await _watch_room(key, seed, withheld, limits=limits)
        game = await _watch_room(key, seed, played)
        return void + game

    void, game = run(scenario())
    assert verify_transcript(void).void is not None
    assert verify_transcript(game).ok
    for report in verify_transcripts([void, game]):
        assert any_error(report.errors, "equivocation"), report.errors


# ===== bots: their programs die with them =====
EXEC_HANG = """
import json, subprocess, sys, time
mark = sys.argv[1]
for line in sys.stdin:
    message = json.loads(line)
    if message["type"] == "hello":
        print(json.dumps({"protocol_version": 1, "name": "hang"}), flush=True)
    elif message["type"] == "decide":
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)", mark])
        time.sleep(600)
"""


def _marked(mark):
    import psutil

    found = []
    for process in psutil.process_iter(["cmdline"]):
        try:
            if mark in (process.info["cmdline"] or []):
                found.append(process.pid)
        except psutil.Error:
            continue
    return found


def test_exec_bot_programs_die_with_the_server(tmp_path):
    """Round 3: a worker killed with SIGKILL left its exec bot's program
    (and whatever that started) running, past the server's own exit."""
    mark = uuid.uuid4().hex
    script = tmp_path / "hang.py"
    script.write_text(EXEC_HANG)
    REGISTRY.register_source(f"exec:{sys.executable} {script} {mark}", "EXECHANG")

    async def scenario():
        server = GameServer()
        ws_server = await run_server(server, "127.0.0.1", 0)
        try:
            config = {**CONFIG, "decision_timeout_ms": 30_000}
            room = server.create_room(config, ["EXECHANG:timeout_ms=20000", "R"])
            deadline = time.monotonic() + 60
            while len(_marked(mark)) < 2 and time.monotonic() < deadline:
                await asyncio.sleep(0.2)  # the program and its child are up
            running = _marked(mark)
            assert room.status == "playing"
        finally:
            server.shutdown()
            ws_server.close()
            await ws_server.wait_closed()
        return running

    try:
        running = run(scenario())
        assert len(running) >= 2, running
        deadline = time.monotonic() + 10
        while _marked(mark) and time.monotonic() < deadline:
            time.sleep(0.2)
        assert not _marked(mark), "the bot's processes outlived the server"
    finally:
        for pid in _marked(mark):
            os.kill(pid, 9)
        REGISTRY.pop("EXECHANG", None)
        REGISTRY.sources.pop("EXECHANG", None)


def test_an_exec_bot_may_not_wait_longer_than_the_room(tmp_path):
    script = tmp_path / "hang.py"
    script.write_text(EXEC_HANG)
    REGISTRY.register_source(f"exec:{sys.executable} {script} x", "EXECWAIT")

    async def scenario():
        server = GameServer()
        config = {**CONFIG, "decision_timeout_ms": 300}
        with pytest.raises(Exception, match="timeout_ms may be at most 300"):
            server.create_room(config, ["EXECWAIT:timeout_ms=30000", None])
        server.create_room(config, ["EXECWAIT:timeout_ms=300", None])
        server.shutdown()

    try:
        run(scenario())
    finally:
        REGISTRY.pop("EXECWAIT", None)
        REGISTRY.sources.pop("EXECWAIT", None)


# ===== addresses behind a proxy =====
class _Request:
    def __init__(self, headers):
        self.headers = websockets.datastructures.Headers(headers)
        self.path = "/"


class _Socket:
    def __init__(self, peer, headers=()):
        self.remote_address = (peer, 1234)
        self.request = _Request(list(headers))


def test_addresses_come_from_a_trusted_proxy_only():
    server = GameServer(trusted_proxies=["127.0.0.1"])
    forwarded = [("X-Forwarded-For", "203.0.113.9, 10.1.1.1")]
    # Behind the trusted proxy: the last hop it names.
    assert server.address_of(_Socket("127.0.0.1", forwarded)) == "10.1.1.1"
    # Anyone else's header is ignored.
    assert server.address_of(_Socket("192.0.2.7", forwarded)) == "192.0.2.7"
    # IPv6 counts by /64 (one host's share), v4-mapped as v4.
    assert (
        _normalize("2001:db8::1")
        == _normalize("2001:db8::ffff")
        != _normalize("2001:db8:0:1::1")
    )
    assert _normalize("::ffff:192.0.2.1") == "192.0.2.1"


def test_a_withholder_behind_a_proxy_bans_only_itself():
    """Round 3: behind a proxy, one withholder got the proxy's address
    banned from the room, and three got everyone banned from the server."""

    async def withhold(base, address):
        created = await create_room(base, CONFIG, [None, "F"])
        ws = await websockets.connect(
            f"{base}/room/{created['id']}",
            additional_headers={"X-Forwarded-For": address},
        )
        hello = json.loads(await ws.recv())["body"]
        request, _, _ = join_request(hello)
        await ws.send(json.dumps(request))
        await next_body(ws, "aborted", timeout=30)
        await ws.close()

    async def scenario():
        limits = ServerLimits(room=RoomLimits(reveal_timeout_ms=300))
        async with serving(limits=limits, trusted_proxies=["127.0.0.1"]) as (
            server,
            base,
        ):
            for _ in range(3):
                await withhold(base, "10.0.0.2")
            created = await create_room(base, CONFIG, [None, "F"])
            ws = await websockets.connect(
                f"{base}/room/{created['id']}",
                additional_headers={"X-Forwarded-For": "10.0.0.9"},
            )
            hello = json.loads(await ws.recv())["body"]
            request, _, _ = join_request(hello)
            await ws.send(json.dumps(request))
            joined = await next_body(ws, "joined", "error")
            await ws.close()
            return dict(server.blames), joined

    blames, joined = run(scenario())
    assert set(blames) == {"10.0.0.2"} and len(blames["10.0.0.2"]) == 3
    assert joined["type"] == "joined"


# ===== the lobby =====
def test_lobby_errors_do_not_echo_requests():
    async def ask(base, request):
        async with websockets.connect(f"{base}/lobby", max_size=None) as ws:
            await ws.send(json.dumps(request))
            return await ws.recv()

    async def scenario():
        async with serving() as (server, base):
            big = "x" * 300_000
            return [
                await ask(base, {"type": big}),
                await ask(base, {"type": "create_room", "config": {big: 1}}),
                await ask(
                    base, {"type": "create_room", "config": {"num_players": big}}
                ),
                await ask(
                    base,
                    {
                        "type": "create_room",
                        "config": {
                            "map": {
                                "tiles": [
                                    {
                                        "coordinate": [0, 0, 0],
                                        "type": "land",
                                        "resource": big,
                                    }
                                ]
                            }
                        },
                    },
                ),
            ]

    for reply in run(scenario()):
        assert len(reply) < 400 and "xxxx" not in reply, reply[:200]
        assert json.loads(reply)["type"] == "error"


# ===== framing an honest server (round 3's variants) =====
class FabricatingClient(GameClient):
    """Lists a signed, legal answer in `sent` that it never sends."""

    fabricated = False

    async def _answer_with(self, action):
        if not self.fabricated and self._pending["body"]["index"] >= 4:
            self.fabricated = True
            send, self.connection.send = self.connection.send, _swallow
            try:
                await super()._answer_with(action)
            finally:
                self.connection.send = send
            self._unreceipted.clear()  # nor resends it
            return
        await super()._answer_with(action)


async def _swallow(_):
    return None


class NineClient(GameClient):
    """Sends eight badly signed copies of one answer, then the real one
    (round 3: the ninth submission for a decision went unanswered)."""

    done = False

    async def _answer_with(self, action):
        await super()._answer_with(action)
        if self.done or self._pending["body"]["index"] < 4:
            return
        self.done = True
        bad = dict(self._answer, sig="00" * 64)
        for _ in range(8):
            self.sent.append(
                {"statement": bad["statement"], "sig": bad["sig"], "sent_after": 0}
            )
            await self.connection.send(json.dumps(bad))
        await self.connection.send(json.dumps(self._answer))


@pytest.mark.parametrize(
    "client_class,timeout,warned",
    [(FabricatingClient, 1500, True), (NineClient, 20_000, False)],
)
def test_a_client_cannot_frame_an_honest_server(client_class, timeout, warned):
    async def scenario():
        config = {**CONFIG, "decision_timeout_ms": timeout}
        async with serving() as (server, base):
            room = await create_room(base, config, [None, "R"])
            client, decide = bot_client(
                f"{base}/room/{room['id']}",
                "R",
                client_class=client_class,
                verify=False,
            )
            await client.play(decide)
            return client

    report = report_of(run(scenario()))
    assert report.ok, report.errors
    if warned:
        assert any("never acknowledged" in w for w in report.warnings), report.warnings
