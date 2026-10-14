"""Protocol 5: round 5's attacks on PROTOCOL.md, one test (or more) each.

The verifier must check a whole chain whatever a void says, classify what
the header claims as the client's word, and never let a dishonest client
or spectator make an honest server's transcript fail. The Python client
must check every description of the room, keep its evidence, and refuse
what the browser refuses. Each dishonest server is a Room that cheats in
one way and signs the result as usual; some chains are re-signed directly
with the server's key, to build exactly the dishonest transcript wanted.
"""

import asyncio
import copy
import json

import pytest
import websockets

from catanatron.server.app import ServerLimits, parse_request
from catanatron.server.client import (
    GameClient,
    ServerMisbehaved,
    create_room,
)
from catanatron.server.config import ConfigError, GameConfig
from catanatron.server.crypto import (
    SigningKey,
    random_hex,
    rejoin_statement,
    submission_digest,
)
from catanatron.server.messages import PROTOCOL_VERSION, aborted_body, receipt_body
from catanatron.server.room import Outbox, Room, RoomError, RoomLimits
from catanatron.server.verify import verify_transcript, verify_transcripts

from tests.server.helpers import (
    resign,
    LaxClient,
    bot_client,
    join_request,
    next_body,
    raw_connect,
    run,
    serving,
)

CONFIG = {"num_players": 2, "vps_to_win": 4, "decision_timeout_ms": 20_000}


def report_of(transcript):
    return verify_transcript(json.loads(json.dumps(transcript)))


def any_error(errors, *fragments):
    return any(all(f in e for f in fragments) for e in errors)


def play_honest(key, room_class=Room, spectate=True, config=CONFIG, limits=None):
    """A finished game under ``key``: RED (a remote seat), a spectator."""

    async def scenario():
        kwargs = {"key": key, "room_class": room_class}
        if limits is not None:
            kwargs["limits"] = limits
        async with serving(**kwargs) as (server, base):
            room = await create_room(base, config, [None, "R"])
            url = f"{base}/room/{room['id']}"
            red, decide = bot_client(url, "R", verify=False)
            watcher = GameClient(url, spectate=True, transcript_dir=None, verify=False)
            tasks = [red.play(decide)] + ([watcher.play()] if spectate else [])
            await asyncio.gather(*tasks, return_exceptions=True)
            return red, watcher, server.rooms[room["id"]]

    return run(scenario())


@pytest.fixture(scope="module")
def honest():
    key = SigningKey.generate()
    red, watcher, room = play_honest(key)
    return key, red.transcript(), watcher.transcript(), room


# ===== the verifier: a void ends nothing it does not end =====
def test_a_void_after_the_game_hides_nothing(honest):
    """Round 5: an aborted appended after a complete after (with a forged
    step before it) verified VOID, no errors."""
    key, red, _, _ = honest
    forged = copy.deepcopy(red)
    step = next(m for m in forged["messages"] if m["body"]["type"] == "step")
    step["body"]["state"]["player_state"]["P0_VICTORY_POINTS"] += 7
    after = next(m for m in forged["messages"] if m["body"]["type"] == "after")
    void = aborted_body(
        red["room_id"],
        game_id=after["body"]["game_id"],
        reason="called off",
        blame=[],
        server_seed=after["body"]["reveal"]["server_seed"],
    )
    forged["messages"].append({**forged["messages"][-1], "body": void})
    forged["messages"] = resign(key, forged["messages"])
    report = report_of(forged)
    assert any_error(report.errors, "a void after the game finished"), report.errors
    assert any_error(report.errors, "message"), "the forged view is still found"
    assert report.verdict == "FAILED"


def _cut_at(transcript, kind):
    messages = transcript["messages"]
    at = next(i for i, m in enumerate(messages) if m["body"]["type"] == kind)
    return at


def test_messages_after_a_void_are_violations(honest):
    """Round 5: [.., commitments, aborted, commitments#2, restart] and
    [.., before, commitments#2, aborted] verified VOID."""
    key, red, _, _ = honest
    seed = next(m for m in red["messages"] if m["body"]["type"] == "after")["body"][
        "reveal"
    ]["server_seed"]
    base = copy.deepcopy(red)
    at = _cut_at(base, "before")
    messages = base["messages"][:at]
    commitments = next(m for m in messages if m["body"]["type"] == "commitments")
    void = aborted_body(
        red["room_id"], game_id=None, reason="void", blame=[], server_seed=seed
    )
    second = copy.deepcopy(commitments)
    second["body"]["reveal_timeout_ms"] += 1
    chain = messages + [
        {**messages[-1], "body": void},
        second,
        {**messages[-1], "body": {"type": "restart", "room_id": red["room_id"]}},
    ]
    transcript = dict(base, messages=resign(key, chain))
    report = report_of(transcript)
    assert any_error(report.errors, "a commitments message after the end")
    assert any_error(report.errors, "unexpected 'restart'")


def test_a_rigged_game_after_a_void_is_not_a_void(honest):
    """Round 5: the server voided once every seed was in, then played a game
    from another master seed; the page played it and the verifier said
    VOID. Now the game after the void is a violation, and a void after the
    start is checked against the start."""
    key, red, _, _ = honest
    seed = next(m for m in red["messages"] if m["body"]["type"] == "after")["body"][
        "reveal"
    ]["server_seed"]
    at = _cut_at(red, "before")
    messages = copy.deepcopy(red["messages"])
    void = {
        **messages[at - 1],
        "body": aborted_body(
            red["room_id"], game_id=None, reason="void", blame=[], server_seed=seed
        ),
    }
    chain = messages[:at] + [void] + messages[at:]
    report = report_of(dict(red, messages=resign(key, chain)))
    assert any_error(report.errors, "a before message after the end"), report.errors


class RiggedStartRoom(Room):
    """Derives the game from another master seed, then calls it off."""

    def _master_seed(self):
        return b"\x01" * 32

    async def _turn(self):
        if len(self.actions) == 6:
            self.abort("called off")
            return
        await super()._turn()


def test_a_void_after_the_start_checks_the_start():
    key = SigningKey.generate()
    red, watcher, room = play_honest(key, room_class=RiggedStartRoom)
    for transcript in (red.transcript(), watcher.transcript()):
        report = report_of(transcript)
        assert report.void is not None
        assert any_error(report.errors, "the game as it started"), report.errors


class CheatThenVoidRoom(Room):
    """Acknowledges RED's legal move as accepted, plays the default, then
    calls the game off (or, with ``erroring``, answers the move with an
    error instead)."""

    erroring = False
    cheated_at = None

    def on_action(self, seat, peer, request):
        pending = seat.pending
        if (
            self.cheated_at is None
            and pending is not None
            and not pending.future.done()
            and len(pending.playable) > 1
        ):
            self.cheated_at = pending.index
            pending.future.set_result("timeout")
            if self.erroring:
                raise RoomError("bad_action", "not now")
            submission = submission_digest(request.get("statement"), request.get("sig"))
            seat.outbox.push(
                receipt_body(submission, pending.index, True, "accepted"),
                self._commit(),
            )
            return
        super().on_action(seat, peer, request)

    async def _turn(self):
        if self.cheated_at is not None and len(self.actions) > self.cheated_at + 3:
            self.abort("the server is shutting down")
            return
        await super()._turn()


class ErrorThenVoidRoom(CheatThenVoidRoom):
    erroring = True


@pytest.mark.parametrize(
    "room_class,message",
    [
        (CheatThenVoidRoom, "had this client's legal, signed move"),
        (ErrorThenVoidRoom, "an action answered with an error"),
    ],
)
def test_a_cheat_is_caught_though_the_game_is_then_void(room_class, message):
    """Round 5: the receipt checks ran on finished games only."""
    key = SigningKey.generate()
    red, _, room = play_honest(key, room_class=room_class, spectate=False)
    assert room.cheated_at is not None and room.status == "aborted"
    report = report_of(red.transcript())
    assert report.void is not None
    assert any_error(report.errors, message), report.errors
    # And in what a silent server would leave (the chain cut after the step).
    transcript = red.transcript()
    step_at = next(
        i
        for i, m in enumerate(transcript["messages"])
        if m["body"]["type"] == "step" and m["body"]["index"] == room.cheated_at + 1
    )
    cut = dict(transcript, messages=transcript["messages"][: step_at + 1])
    report = report_of(cut)
    assert report.incomplete and any_error(report.errors, message), report.errors


def test_a_spectator_cannot_make_an_honest_server_fail():
    """Round 5: a spectator's action got 'unexpected action here' (answers
    'action') on its chain, which the verifier took as provable."""

    async def scenario():
        async with serving() as (server, base):
            room = await create_room(base, CONFIG, ["R", "R"])
            ws = await websockets.connect(f"{base}/room/{room['id']}")
            hello = json.loads(await ws.recv())
            await ws.send(json.dumps({"type": "spectate"}))
            messages = [hello]
            sent = False
            while True:
                envelope = json.loads(await asyncio.wait_for(ws.recv(), 60))
                messages.append(envelope)
                kind = envelope["body"]["type"]
                if kind == "step" and not sent:
                    sent = True
                    await ws.send(
                        json.dumps({"type": "action", "statement": {}, "sig": "00"})
                    )
                if kind == "after":
                    break
            await ws.close()
            return {
                "server_key": hello["body"]["server_key"],
                "room_id": room["id"],
                "public_key": None,
                "client_seed": None,
                "messages": messages,
            }

    transcript = run(scenario())
    errors = [m["body"] for m in transcript["messages"] if m["body"]["type"] == "error"]
    assert errors and errors[0]["answers"] == "action"
    report = report_of(transcript)
    assert report.ok, report.errors


def test_a_connection_that_only_heard_the_void_agrees_with_the_seat():
    """Round 5: a lurker's [hello, aborted] against the seat's chain was read
    as 'a void before the game and a game'. The void names its game."""

    async def scenario():
        async with serving() as (server, base):
            room = await create_room(
                base, {**CONFIG, "decision_timeout_ms": 600}, [None, "R"]
            )
            url = f"{base}/room/{room['id']}"
            red = GameClient(url, transcript_dir=None, verify=False)

            async def slow(view, playable):
                await asyncio.sleep(5)  # times out: the game goes on slowly
                return playable[0]

            playing = asyncio.create_task(red.play(slow))
            room_obj = server.rooms[room["id"]]
            while len(room_obj.actions) < 4:
                await asyncio.sleep(0.05)
            lurker = await websockets.connect(url)
            hello = json.loads(await lurker.recv())
            room_obj.abort("the server is shutting down")
            void = json.loads(await asyncio.wait_for(lurker.recv(), 10))
            await lurker.close()
            await playing
            return red.transcript(), {
                "server_key": hello["body"]["server_key"],
                "room_id": room["id"],
                "public_key": None,
                "client_seed": None,
                "messages": [hello, void],
            }

    seat, lurker = run(scenario())
    reports = verify_transcripts([seat, lurker])
    for report in reports:
        assert report.void is not None and not report.errors, report.errors


class BlameAfterStartRoom(Room):
    """Starts the game with seat 0's seed and reveal, then voids it blaming
    seat 0 for withholding."""

    async def _turn(self):
        if len(self.actions) == 5:
            self.abort("a seat did not reveal its seed in time: the game is void", [0])
            return
        await super()._turn()


def test_a_blame_the_servers_own_before_belies_is_caught_by_everyone():
    key = SigningKey.generate()
    red, watcher, _ = play_honest(key, room_class=BlameAfterStartRoom)
    for transcript in (red.transcript(), watcher.transcript()):
        report = report_of(transcript)
        assert any_error(report.errors, "the server's own before holds"), report.errors


class BlameBeforeLockRoom(Room):
    """Blames a seat for withholding before ever asking for its seed."""

    def join(self, outbox, peer, request):
        seat = super().join(outbox, peer, request)
        asyncio.get_running_loop().call_later(
            0.2, self.abort, "a seat did not reveal its seed", [seat.index]
        )
        return seat


def test_blaming_a_seat_never_asked_for_its_seed_is_caught():
    async def scenario():
        async with serving(room_class=BlameBeforeLockRoom) as (server, base):
            room = await create_room(
                base, {**CONFIG, "num_players": 3}, [None, None, "R"]
            )
            client = GameClient(
                f"{base}/room/{room['id']}", transcript_dir=None, verify=False
            )
            await client.play(lambda view, playable: playable[0])
            return client

    report = report_of(run(scenario()).transcript())
    assert any_error(report.errors, "nobody was asked to reveal"), report.errors


class CoercingRoom(Room):
    """Times RED out of every decision the moment it asks."""

    async def _remote_decision(self, seat, playable):
        task = asyncio.ensure_future(super()._remote_decision(seat, playable))
        await asyncio.sleep(0)
        if seat.pending is not None and not seat.pending.future.done():
            seat.pending.future.set_result("timeout")
        return await task


def test_coercion_to_the_default_leaves_a_trace():
    """Round 5: decisions closed before the client could answer left no
    record and no warning."""
    key = SigningKey.generate()
    red, _, room = play_honest(key, room_class=CoercingRoom, spectate=False)
    transcript = red.transcript()
    assert transcript["unanswered"], "the client recorded being timed out"
    report = report_of(transcript)
    assert report.ok, report.errors
    timeouts = [w for w in report.warnings if "played the default" in w]
    assert timeouts and "ms" in timeouts[0]


class SwappedSeatsRoom(Room):
    """Seats whoever asks for seat 0 at seat 1, and the other way round."""

    def _open_seat(self, index):
        if index in (0, 1):
            index = 1 - index
        return super()._open_seat(index)


def test_the_seat_a_client_asked_for_is_the_seat_it_gets():
    async def scenario():
        limits = ServerLimits(room=RoomLimits(reveal_timeout_ms=500))
        async with serving(room_class=SwappedSeatsRoom, limits=limits) as (
            server,
            base,
        ):
            room = await create_room(base, CONFIG, [None, None])
            url = f"{base}/room/{room['id']}"
            strict, d1 = bot_client(url, "R", seat=0, verify=False)
            lax, d2 = bot_client(url, "R", client_class=LaxJoin, seat=1, verify=False)
            return (
                await asyncio.gather(
                    strict.play(d1), lax.play(d2), return_exceptions=True
                ),
                lax,
            )

    (strict, _), lax = run(scenario())
    assert isinstance(strict, ServerMisbehaved)
    report = report_of(lax.transcript())
    assert any_error(report.errors, "malformed") or report.invalid, report.errors


class LaxJoin(LaxClient):
    """Takes any seat it is given (so the verifier is what is tested)."""

    wanted_seat_checked = False

    async def _handle(self, ws, body, decide):
        if body.get("type") == "joined":
            wanted, self.wanted_seat = self.wanted_seat, None
            try:
                return await super()._handle(ws, body, decide)
            finally:
                self.wanted_seat = wanted
        return await super()._handle(ws, body, decide)


class FloatRoom(Room):
    """Signs a float into every step."""

    def _send_all(self, build):
        super()._send_all(
            lambda color: (
                {**build(color), "num_turns": 1.0}
                if build(color)["type"] == "step"
                else build(color)
            )
        )


def test_a_float_the_server_signed_is_a_violation(honest):
    """A float reads differently in JavaScript and Python: the live client
    refuses it, and in a transcript it is the server's doing."""
    key, red, _, _ = honest
    forged = copy.deepcopy(red)
    step = next(m for m in forged["messages"] if m["body"]["type"] == "step")
    step["body"]["index"] = float(step["body"]["index"])
    forged["messages"] = resign(key, forged["messages"])
    report = report_of(forged)
    assert any_error(report.errors, "does not allow"), report.errors

    async def scenario():
        async with serving(room_class=FloatRoom) as (server, base):
            room = await create_room(base, CONFIG, [None, "R"])
            client, decide = bot_client(f"{base}/room/{room['id']}", "R", verify=False)
            with pytest.raises(ServerMisbehaved, match="does not allow"):
                await client.play(decide)

    run(scenario())


def test_one_room_cannot_be_served_under_two_keys(honest):
    """Round 5: the commitments did not name the server key, so a twin room
    under another key could carry the same pops and reveals. Re-signed
    under another key, the reveals no longer verify; and two keys for one
    room id are flagged (as a warning: anyone can reuse an id)."""
    key, red, watcher, _ = honest
    other = SigningKey.generate()
    twin = copy.deepcopy(red)
    for envelope in twin["messages"]:
        body = envelope["body"]
        if body["type"] == "hello":
            body["server_key"] = other.public_hex
        if body["type"] == "commitments":
            body["server_key"] = other.public_hex
        if body["type"] == "before":
            body["commitments"]["server_key"] = other.public_hex
    twin["messages"] = resign(other, twin["messages"])
    twin["server_key"] = other.public_hex
    report = report_of(twin)
    assert any_error(report.errors, "malformed") or any_error(
        report.errors, "did not reveal its seed for these commitments"
    ), report.errors
    reports = verify_transcripts([watcher, twin])
    assert any("another server key" in w for w in reports[0].warnings)
    assert reports[0].ok


def test_a_header_that_belies_the_chain_is_not_an_accusation(honest):
    _, red, _, _ = honest
    for edit in (
        {"client_seed": random_hex()},
        {"public_key": None},
        {"public_key": SigningKey.generate().public_hex},
        {"room_id": "elsewhere"},
    ):
        report = report_of({**red, **edit})
        assert report.invalid and not report.errors, (edit, report.errors)


# ===== the Python client =====
class BoolCommitmentsRoom(Room):
    """Puts the commitments in the before with False where 0 was (equal
    in Python, another C on the wire)."""

    def _before(self, color):
        body = super()._before(color)
        commitments = copy.deepcopy(body["commitments"])
        commitments["seats"][0]["seat"] = False
        return {**body, "commitments": commitments}


class NoSeedVoidRoom(Room):
    def abort(self, reason, blame=None):
        self.server_seed = None
        super().abort(reason, blame)


class FakeJoinedRoom(Room):
    def join(self, outbox, peer, request):
        real = outbox.push

        def push(body, commit=None):
            if body.get("type") == "joined":
                body = copy.deepcopy(body)
                body["room"]["seed_commitment"] = random_hex()
            return real(body, commit)

        outbox.push = push
        try:
            return super().join(outbox, peer, request)
        finally:
            outbox.push = real


@pytest.mark.parametrize(
    "room_class,message",
    [
        (BoolCommitmentsRoom, "not the one whose seats were committed"),
        (FakeJoinedRoom, "changes the room's seed commitment"),
    ],
)
def test_the_python_client_refuses_what_the_verifier_would(room_class, message):
    async def scenario():
        async with serving(room_class=room_class) as (server, base):
            room = await create_room(base, CONFIG, [None, "R"])
            client, decide = bot_client(f"{base}/room/{room['id']}", "R", verify=False)
            with pytest.raises(ServerMisbehaved, match=message):
                await client.play(decide)

    run(scenario())


def test_a_void_without_the_server_seed_is_refused():
    async def scenario():
        limits = ServerLimits(room=RoomLimits(reveal_timeout_ms=300))
        async with serving(room_class=NoSeedVoidRoom, limits=limits) as (server, base):
            room = await create_room(
                base, {**CONFIG, "num_players": 3}, [None, None, "R"]
            )
            url = f"{base}/room/{room['id']}"
            client, decide = bot_client(url, "R", seat=0, verify=False)
            silent = GameClient(url, seat=1, transcript_dir=None, verify=False)
            silent._handle = _silent_handle(silent)
            results = await asyncio.gather(
                client.play(decide),
                silent.play(lambda v, p: p[0]),
                return_exceptions=True,
            )
            return results

    results = run(scenario())
    assert isinstance(results[0], ServerMisbehaved)


def _silent_handle(client):
    real = client._handle

    async def handle(ws, body, decide):
        if body.get("type") == "commitments":
            client.commitments = body
            return None
        return await real(ws, body, decide)

    return handle


class KindBotRoom(Room):
    """Publishes the client's own seat as a bot's."""

    def _push_commitments(self, outbox):
        if self.commitments is not None and not getattr(self, "_bent", False):
            self._bent = True
            self.commitments = copy.deepcopy(self.commitments)
            self.commitments["seats"][0]["kind"] = "bot"
        super()._push_commitments(outbox)


def test_the_python_client_checks_its_own_seat_fully():
    async def scenario():
        async with serving(room_class=KindBotRoom) as (server, base):
            room = await create_room(base, CONFIG, [None, "R"])
            client, decide = bot_client(f"{base}/room/{room['id']}", "R", verify=False)
            with pytest.raises(ServerMisbehaved, match="seat 0 is malformed"):
                await client.play(decide)

    run(scenario())


class ForgetAfterRevealRoom(Room):
    """Acknowledges the client's reveal, drops it, and then says it holds
    no seat (hoping it starts over and reveals again)."""

    def on_reveal(self, seat, peer, request):
        super().on_reveal(seat, peer, request)
        if seat.index == 0 and seat.outbox.peer is not None:
            asyncio.get_running_loop().call_later(
                0.2, seat.outbox.peer.kick, 1011, "drop"
            )

    def rejoin(self, outbox, peer, request):
        raise RoomError("not_seated", "no seat here is held by that key")


def test_a_client_keeps_its_evidence_when_the_server_forgets_its_seat(tmp_path):
    """Round 5: the client started over (wiping its chain) on not_seated
    after revealing; and saved nothing when play() raised."""

    async def scenario():
        async with serving(room_class=ForgetAfterRevealRoom) as (server, base):
            room = await create_room(
                base, {**CONFIG, "num_players": 3}, [None, None, "R"]
            )
            url = f"{base}/room/{room['id']}"
            client = GameClient(url, seat=0, transcript_dir=str(tmp_path), verify=False)
            other = GameClient(url, seat=1, transcript_dir=None, verify=False)
            other._handle = _silent_handle(other)
            results = await asyncio.gather(
                client.play(lambda v, p: p[0]),
                asyncio.wait_for(other.play(lambda v, p: p[0]), 5),
                return_exceptions=True,
            )
            return results[0], client

    raised, client = run(scenario())
    assert isinstance(raised, ServerMisbehaved), raised
    saved = list(tmp_path.iterdir())
    assert saved, "the transcript was saved though play() raised"
    transcript = json.loads(saved[0].read_text())
    kinds = [m["body"]["type"] for m in transcript["messages"]]
    assert "commitments" in kinds and "received" in kinds


class FakeJoinErrorRoom(Room):
    """Pushes a join refusal onto a seat's chain in the middle of the game."""

    async def _turn(self):
        if len(self.actions) == 6:
            seat = self.seats[0]
            seat.outbox.push(
                {
                    "type": "error",
                    "code": "banned",
                    "message": "go away",
                    "answers": "join",
                },
                self._commit(),
            )
        await super()._turn()


def test_a_join_refusal_mid_game_does_not_end_the_client():
    key = SigningKey.generate()
    red, _, room = play_honest(key, room_class=FakeJoinErrorRoom, spectate=False)
    assert room.status == "finished"
    assert red.table.winning_color is not None


class OtherChainRoom(Room):
    """Answers a rejoin with a brand-new chain for the seat (from seq 0)."""

    def rejoin(self, outbox, peer, request):
        key = request.get("public_key")
        fresh = Outbox(self.key)
        fresh.to = key
        peer.send(fresh.push(self.hello_body(fresh.nonce)))
        return super().rejoin(outbox, peer, request)


def test_a_resent_chain_that_is_not_ours_is_refused():
    """Round 5: the client skipped such messages forever."""

    async def scenario():
        async with serving(room_class=OtherChainRoom) as (server, base):
            room = await create_room(
                base, {**CONFIG, "num_players": 3}, [None, None, "R"]
            )
            client = GameClient(
                f"{base}/room/{room['id']}", transcript_dir=None, verify=False
            )
            playing = asyncio.create_task(client.play(lambda v, p: p[0]))
            while client.seat is None:
                await asyncio.sleep(0.05)
            await client.connection.close()
            with pytest.raises(ServerMisbehaved, match="another chain"):
                await asyncio.wait_for(playing, 30)

    run(scenario())


# ===== the server =====
def test_a_public_join_cannot_be_replayed_to_reseat_a_freed_key():
    """Round 5: anyone could replay a freed seat's published join, seating
    its key without its holder; the key was then blamed and banned."""

    async def scenario():
        limits = ServerLimits(room=RoomLimits(seat_grace_s=0.2))
        async with serving(limits=limits) as (server, base):
            room = await create_room(
                base, {**CONFIG, "num_players": 3}, [None, None, "R"]
            )
            url = f"{base}/room/{room['id']}"
            first, hello = await raw_connect(url)
            request, key, _ = join_request(hello, seat=0)
            await first.send(json.dumps(request))
            await next_body(first, "joined")
            await first.close()
            await asyncio.sleep(0.6)  # the seat is freed
            assert server.rooms[room["id"]].seats[0].kind == "open"
            second, _ = await raw_connect(url)
            await second.send(json.dumps(request))  # the published join, replayed
            refusal = await next_body(second, "error", "joined")
            await second.close()
            return refusal, server.rooms[room["id"]]

    refusal, room = run(scenario())
    assert refusal["type"] == "error" and refusal["code"] == "bad_join"
    assert room.seats[0].kind == "open"


def test_a_rejoin_racing_the_lock_carries_one_chain():
    """Round 5: a reconnecting seat's new connection was made a spectator at
    the lock, then its rejoin attached the seat's chain too: two chains on
    one socket, and an honest client accusing an honest server."""

    async def scenario():
        async with serving() as (server, base):
            created = await create_room(
                base, {**CONFIG, "num_players": 3}, [None, None, "R"]
            )
            url = f"{base}/room/{created['id']}"
            room = server.rooms[created["id"]]
            first, hello = await raw_connect(url)
            request, key, _ = join_request(hello, seat=0)
            await first.send(json.dumps(request))
            await next_body(first, "joined")
            await first.close()
            again, hello2 = await raw_connect(url)  # in the lobby
            other, hello3 = await raw_connect(url)
            request2, _, _ = join_request(hello3, seat=1)
            await other.send(json.dumps(request2))
            await next_body(again, "commitments")  # the lock made it a spectator
            spectators_at_lock = len(room.spectators)
            await again.send(
                json.dumps(
                    {
                        "type": "rejoin",
                        "protocol_version": PROTOCOL_VERSION,
                        "public_key": key.public_hex,
                        "sig": key.sign(
                            rejoin_statement(created["id"], hello2["nonce"])
                        ),
                        "last_seq": 1,
                    }
                )
            )
            await asyncio.sleep(0.5)
            spectators = len(room.spectators)
            for ws in (again, other):
                await ws.close()
            return spectators_at_lock, spectators

    at_lock, after = run(scenario())
    assert at_lock == 1 and after == 0


def test_an_accepted_reveal_resent_on_every_reconnect_is_not_charged():
    """Round 5: an honest client on a flaky link resent its receipted reveal
    on every reconnect until the allowance ran out (4009, for good)."""

    async def scenario():
        limits = ServerLimits(
            room=RoomLimits(receipt_margin=3, reveal_timeout_ms=60_000)
        )
        async with serving(limits=limits) as (server, base):
            created = await create_room(
                base, {**CONFIG, "num_players": 3}, [None, None, "R"]
            )
            url = f"{base}/room/{created['id']}"
            ws, hello = await raw_connect(url)
            request, key, seed = join_request(hello, seat=0)
            await ws.send(json.dumps(request))
            other, hello2 = await raw_connect(url)
            await other.send(json.dumps(join_request(hello2, seat=1)[0]))
            commitments = await next_body(ws, "commitments")
            from catanatron.server.crypto import digest, reveal_statement

            reveal = {
                "type": "reveal",
                "client_seed": seed,
                "sig": key.sign(reveal_statement(digest(commitments))),
            }
            receipts = []
            for _ in range(10):
                await ws.send(json.dumps(reveal))
                receipts.append(await next_body(ws, "received"))
            await ws.close()
            await other.close()
            return receipts

    receipts = run(scenario())
    assert len(receipts) == 10 and all(r["accepted"] for r in receipts)


def test_connections_coming_and_going_do_not_grow_the_chains():
    """Round 5: every bare connect/close pushed an identical 'room' message
    to every chain, without bound."""

    async def scenario():
        async with serving() as (server, base):
            created = await create_room(
                base, {**CONFIG, "num_players": 3}, [None, None, None]
            )
            url = f"{base}/room/{created['id']}"
            watcher, _ = await raw_connect(url)
            await watcher.send(json.dumps({"type": "spectate"}))
            await next_body(watcher, "room")
            room = server.rooms[created["id"]]
            box = room.spectators[0]
            before = len(box.envelopes)
            for _ in range(30):
                ws, _ = await raw_connect(url)
                await ws.close()
            await asyncio.sleep(0.3)
            grown = len(box.envelopes) - before
            await watcher.close()
            return grown

    assert run(scenario()) <= 1


def test_numbers_beyond_every_clients_reach_are_refused():
    """Round 5: an integer above 2**53 in a map put a number in every signed
    hello that a browser could not represent."""
    with pytest.raises(RoomError):
        parse_request(json.dumps({"type": "ping", "x": 2**60}))
    with pytest.raises(RoomError):
        parse_request('{"type": "ping", "x": 5.0}')
    tiles = [
        {"coordinate": [0, 0, 0], "type": "land", "resource": None},
        {"coordinate": [1, -1, 0], "type": "land", "resource": "WOOD", "number": 6},
        {"coordinate": [2**40, -(2**40), 0], "type": "water"},
    ]
    with pytest.raises(ConfigError, match="at most"):
        GameConfig.from_json({"num_players": 2, "map": {"tiles": tiles}})


class OtherSeatBadPopRoom(Room):
    """Publishes the server bot's seat with a pop (and a key) it cannot
    have: a seat that is not what it claims, which the client must refuse
    before revealing (its reveal signs every seat)."""

    def _push_commitments(self, outbox):
        if self.commitments is not None and not getattr(self, "_bent", False):
            self._bent = True
            self.commitments = copy.deepcopy(self.commitments)
            self.commitments["seats"][1]["pop"] = "00" * 64
        super()._push_commitments(outbox)


def test_the_python_client_checks_every_seat_before_revealing():
    async def scenario():
        async with serving(room_class=OtherSeatBadPopRoom) as (server, base):
            room = await create_room(base, CONFIG, [None, "R"])
            client, decide = bot_client(f"{base}/room/{room['id']}", "R", verify=False)
            with pytest.raises(ServerMisbehaved, match="seat 1 is malformed"):
                await client.play(decide)
            return client

    client = run(scenario())
    assert client._reveal is None, "nothing was revealed into those commitments"


class VoidMidGameRoom(Room):
    async def _turn(self):
        if len(self.actions) == 8:
            self.abort("the server is shutting down")
            return
        await super()._turn()


def test_recipients_of_a_game_voided_midway_are_still_compared():
    """Round 5: in a room voided mid-game, the seat and a spectator were
    given different states, and both verified VOID with no equivocation (the
    state commitments came only from the replay, which a void skipped)."""
    key = SigningKey.generate()
    red, watcher, _ = play_honest(key, room_class=VoidMidGameRoom)
    seat, spectator = red.transcript(), copy.deepcopy(watcher.transcript())
    assert verify_transcripts([seat, spectator])[0].void is not None
    for envelope in spectator["messages"]:
        commit = envelope.get("commit")
        if commit and commit["index"] == 3:
            commit["hash"] = "ee" * 32
    spectator["messages"] = resign(key, spectator["messages"])
    for report in verify_transcripts([seat, spectator]):
        assert any_error(
            report.errors, "equivocation", "another state 3"
        ), report.errors
