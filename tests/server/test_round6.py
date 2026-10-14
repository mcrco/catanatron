"""Round 6's attacks on protocol 5, one test (or more) each.

A void reveals the server seed: the verifier replays everything a chain
shows of the game from it (the public log), for seats and for spectators
that came in mid-game. The verifier is total on what the server signed
(a crash there is the server's doing, never 'invalid'). The clients and
the verifier agree on every value (hex, strings, colors), and neither
signs twice for one move.
"""

import asyncio
import copy
import json
import shutil

import pytest
import websockets

from catanatron.server.app import GameServer, ServerLimits, parse_request
from catanatron.server.client import GameClient, ServerMisbehaved, create_room
from catanatron.server.crypto import SigningKey, is_hex, random_hex
from catanatron.server.room import Room, RoomError, RoomLimits
from catanatron.server.verify import verify_transcript, verify_transcripts

from tests.server.helpers import (
    bot_client,
    join_request,
    next_body,
    raw_connect,
    resign,
    run,
    serving,
)
from tests.server.test_browser import FakeServer, ScriptedPage, _seated_page

CONFIG = {"num_players": 2, "vps_to_win": 4, "decision_timeout_ms": 20_000}
needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")


def report_of(transcript):
    return verify_transcript(json.loads(json.dumps(transcript)))


def any_error(errors, *fragments):
    return any(all(f in e for f in fragments) for e in errors)


def voided_game(key, void_at=14, late_at=6, room_class=Room, config=CONFIG):
    """A game voided at move ``void_at`` (the server shutting down): RED's
    transcript, a spectator's from the start, one that came in at move
    ``late_at``, and a lurker's (a connection that only reads)."""

    async def scenario():
        async with serving(key=key, room_class=room_class) as (server, base):
            room = await create_room(base, config, [None, "R"])
            url = f"{base}/room/{room['id']}"
            room_obj = server.rooms[room["id"]]
            red, decide = bot_client(url, "R", verify=False)
            early = GameClient(url, spectate=True, transcript_dir=None, verify=False)
            tasks = [
                asyncio.create_task(red.play(decide)),
                asyncio.create_task(early.play()),
            ]
            while len(room_obj.actions) < late_at:
                await asyncio.sleep(0.01)
            late = GameClient(url, spectate=True, transcript_dir=None, verify=False)
            tasks.append(asyncio.create_task(late.play()))
            lurker = await websockets.connect(url)
            lurked = [json.loads(await lurker.recv())]
            while len(room_obj.actions) < void_at:
                await asyncio.sleep(0.01)
            room_obj.abort("the server is shutting down")
            lurked.append(json.loads(await asyncio.wait_for(lurker.recv(), 10)))
            await lurker.close()
            await asyncio.gather(*tasks, return_exceptions=True)
            lurk = {
                "server_key": key.public_hex,
                "room_id": room["id"],
                "public_key": None,
                "client_seed": None,
                "messages": lurked,
            }
            return red.transcript(), early.transcript(), late.transcript(), lurk

    return run(scenario())


@pytest.fixture(scope="module")
def voided():
    key = SigningKey.generate()
    return (key,) + tuple(voided_game(key))


# ===== the verifier, after a void =====
def test_a_spectator_that_came_in_midgame_verifies_a_void(voided):
    """Round 6 (a regression): its mid-game before was compared with the
    game's start, and an honest server's void FAILED."""
    _, red, early, late, lurk = voided
    assert late["messages"][1]["body"]["index"] > 0
    for transcript in (red, early, late, lurk):
        report = report_of(transcript)
        assert report.void is not None and not report.errors, report.errors
    for report in verify_transcripts([red, early, late, lurk]):
        assert not report.errors, report.errors


def _first(transcript, kind, after=0):
    return next(
        i
        for i, m in enumerate(transcript["messages"])
        if i >= after and m["body"]["type"] == kind
    )


@pytest.mark.parametrize("who", ["red", "early", "late"])
def test_a_lie_before_a_void_is_proven_by_the_void_s_seed(voided, who):
    """Round 6: a forged roll (a view lie, or a rigged step that keeps the
    state consistent) and then a void verified VOID: the void reveals the
    seed, and the public log replays the game."""
    key, red, early, late, _ = voided
    transcript = copy.deepcopy({"red": red, "early": early, "late": late}[who])
    messages = transcript["messages"]
    at = next(
        i
        for i, m in enumerate(messages)
        if m["body"]["type"] == "step" and m["body"]["log"][0][0][1] == "ROLL"
    )
    roll = messages[at]["body"]["log"][0]
    roll[1] = [7 - roll[1][0], 7 - roll[1][1]]
    transcript["messages"] = resign(key, messages)
    report = report_of(transcript)
    assert report.void is not None
    assert any_error(report.errors, f"message {at}"), report.errors


@pytest.mark.parametrize(
    "forced,message",
    [
        ("bot_error", "a bot error on a seat that is not a bot"),
        ("only", "played as forced"),
        ("nonsense", "unknown forced reason"),
    ],
)
def test_moves_the_server_could_not_make_are_caught_in_a_void(voided, forced, message):
    """Round 6: in a void or cut transcript, a remote seat's move flagged
    as a bot error (or 'only' after a decide) was not an error."""
    key, red, *_ = voided
    transcript = copy.deepcopy(red)
    messages = transcript["messages"]
    decide_at = _first(transcript, "decide", after=_first(transcript, "before") + 3)
    index = messages[decide_at]["body"]["index"]
    step_at = next(
        i
        for i, m in enumerate(messages)
        if m["body"]["type"] == "step" and m["body"]["index"] == index + 1
    )
    messages[step_at]["body"]["forced"] = forced
    transcript["messages"] = resign(key, messages)
    for cut in (
        transcript,
        dict(transcript, messages=transcript["messages"][: step_at + 1]),
    ):
        report = report_of(cut)
        assert any_error(report.errors, message), report.errors


def test_a_move_made_for_a_seat_that_was_never_asked_is_caught_in_a_void(voided):
    key, red, *_ = voided
    transcript = copy.deepcopy(red)
    messages = transcript["messages"]
    decide_at = _first(transcript, "decide", after=_first(transcript, "before") + 3)
    del messages[decide_at]
    transcript["messages"] = resign(key, messages)
    report = report_of(transcript)
    assert any_error(report.errors, "never asked"), report.errors


# ===== the verifier is total =====
def test_what_the_server_signed_never_makes_a_transcript_invalid(voided):
    """Round 6: a signed type that is not a string (or a supply that is a
    list) crashed the verifier, and INVALID hid the server's lie."""
    key, red, *_ = voided
    for edit in ("type", "supply", "index", "submission"):
        transcript = copy.deepcopy(red)
        messages = transcript["messages"]
        if edit == "type":
            messages.insert(3, {**messages[2], "body": {"type": []}})
        elif edit == "supply":
            for m in messages:
                for room in (m["body"].get("room"), m["body"].get("commitments")):
                    if isinstance(room, dict) and "config" in room:
                        room["config"]["supply"] = []
                if m["body"].get("type") in ("commitments", "before"):
                    m["body"]["config"]["supply"] = []
        elif edit == "index":
            d = _first(transcript, "decide")
            messages[d]["body"]["index"] = {"x": 1}
        else:
            messages.append(
                {**messages[-1], "body": {"type": "received", "submission": [1]}}
            )
        transcript["messages"] = resign(key, messages)
        report = report_of(transcript)
        assert report.errors and not report.invalid, (edit, report.invalid)


def test_another_protocol_s_transcript_is_not_an_accusation(voided):
    key, red, *_ = voided
    transcript = copy.deepcopy(red)
    transcript["messages"][0]["body"]["protocol_version"] = 4
    transcript["messages"] = resign(key, transcript["messages"])
    report = report_of(transcript)
    assert report.invalid and not report.errors
    assert any("not a protocol 5 transcript" in e for e in report.invalid)


def test_the_commitments_must_seat_this_client_on_its_own_connection(voided):
    """Round 6: the server kept a pop from the client's earlier connection;
    the client refused live, the verifier said INCOMPLETE."""
    key, red, *_ = voided
    transcript = copy.deepcopy(red)
    transcript["messages"][0]["body"]["nonce"] = random_hex(16)
    transcript["messages"] = resign(key, transcript["messages"])
    report = report_of(transcript)
    assert any_error(
        report.errors, "do not seat this client as it joined"
    ), report.errors


# ===== I15: what a lurker's void says =====
def test_a_lurker_s_void_is_compared_in_full(voided):
    """Round 6: a void naming another game, a different hello commitment,
    and a rigged start were not compared with a lurker's transcript."""
    key, red, early, late, lurk = voided
    other = copy.deepcopy(lurk)
    other["messages"][1]["body"]["game_id"] = "another-game"
    other["messages"] = resign(key, other["messages"])
    reports = verify_transcripts([red, other])
    assert all(
        any_error(r.errors, "equivocation", "a void of another game") for r in reports
    )
    commit = copy.deepcopy(lurk)
    commit["messages"][0]["commit"]["hash"] = "ee" * 32
    commit["messages"] = resign(key, commit["messages"])
    reports = verify_transcripts([red, commit])
    assert all(any_error(r.errors, "equivocation", "another state") for r in reports)


class RiggedSilentRoom(Room):
    """Derives the game from another master seed, then calls it off, the
    void reaching only who is still listening."""

    def _master_seed(self):
        return b"\x02" * 32


def test_a_void_another_transcript_holds_replays_a_game_cut_short():
    key = SigningKey.generate()
    red, early, _, lurk = voided_game(key, room_class=RiggedSilentRoom)
    void_at = _first(red, "aborted")
    silent = dict(red, messages=red["messages"][:void_at])  # the server went silent
    alone = report_of(silent)
    assert alone.incomplete and not alone.errors
    reports = verify_transcripts([silent, lurk])
    assert any_error(reports[0].errors, "the game as it started"), reports[0].errors


def test_room_ids_are_long_and_unique(monkeypatch):
    """Round 6: 48-bit ids, with no uniqueness check, would one day make two
    honest rooms' transcripts look like equivocation."""
    from catanatron.server import app

    ids = iter(["same" * 6, "same" * 6, "other" * 5])
    asked = []

    def token(n):
        asked.append(n)
        return next(ids)

    monkeypatch.setattr(app.secrets, "token_urlsafe", token)

    async def scenario():
        server = GameServer()
        a = server.create_room({"num_players": 2}, [None, None])
        b = server.create_room({"num_players": 2}, [None, None])
        server.shutdown()
        return a.id, b.id

    a, b = run(scenario())
    assert a != b and min(asked) >= 16, "128 random bits, and never one in use"


# ===== voids and blame =====
class RefuseRevealRoom(Room):
    """Answers seat 0's reveal with an error while seeds are revealed, then
    lets the reveal timeout blame it."""

    def on_reveal(self, seat, peer, request):
        if seat.index == 0:
            raise RoomError("not_revealing", "seeds are not being revealed now")
        super().on_reveal(seat, peer, request)


def test_a_reveal_refused_by_an_error_then_blamed_is_caught():
    async def scenario():
        limits = ServerLimits(room=RoomLimits(reveal_timeout_ms=500))
        async with serving(room_class=RefuseRevealRoom, limits=limits) as (
            server,
            base,
        ):
            room = await create_room(base, CONFIG, [None, "R"])
            client = GameClient(
                f"{base}/room/{room['id']}", transcript_dir=None, verify=False
            )
            result = await client.play(lambda v, p: p[0])
            return client, result

    client, result = run(scenario())
    assert result.aborted["blame"] == [0]
    report = report_of(client.transcript())
    assert any_error(
        report.errors, "answered this seat's reveal with an error"
    ), report.errors


class BlameNobodyRoom(Room):
    """Calls the room off before it locks, blaming a seat."""

    blame = [9]

    def join(self, outbox, peer, request):
        seat = super().join(outbox, peer, request)
        if seat.index == 1:
            asyncio.get_running_loop().call_later(
                0.2, self.abort, "a seat did not reveal its seed", list(self.blame)
            )
        return seat


class BlameSeatZeroRoom(BlameNobodyRoom):
    blame = [0]


@pytest.mark.parametrize(
    "room_class,message",
    [
        (BlameNobodyRoom, "a seat the room does not have"),
        (BlameSeatZeroRoom, "nobody was asked to reveal"),
    ],
)
def test_a_void_cannot_blame_a_seat_that_was_never_asked(room_class, message):
    """Round 6: blame [9], or blame on another seat of a room that never
    locked, verified as a plain void."""

    async def scenario():
        async with serving(room_class=room_class) as (server, base):
            room = await create_room(
                base, {**CONFIG, "num_players": 3}, [None, None, None]
            )
            url = f"{base}/room/{room['id']}"
            a = GameClient(url, seat=0, transcript_dir=None, verify=False)
            b = GameClient(url, seat=1, transcript_dir=None, verify=False)
            await asyncio.gather(a.play(lambda v, p: p[0]), b.play(lambda v, p: p[0]))
            return a, b

    a, b = run(scenario())
    assert any_error(report_of(b.transcript()).errors, message)


# ===== the Python client =====
class SecondDecideRoom(Room):
    """Asks RED twice for its first decision."""

    asked = False

    async def _remote_decision(self, seat, playable):
        if not self.asked:
            self.asked = True
            seat.outbox.push({**self._decide_body(seat)}, self._commit())
        return await super()._remote_decision(seat, playable)

    def _decide_body(self, seat):
        from catanatron.server.messages import decide_body

        return decide_body(
            self.game,
            self.doc,
            seat.color.value,
            timeout_ms=self.config.decision_timeout_ms,
        )


class EarlyDecideRoom(Room):
    """Sends a decide right after the join."""

    def join(self, outbox, peer, request):
        seat = super().join(outbox, peer, request)
        outbox.push(
            {
                "type": "decide",
                "index": 0,
                "playable_actions": [["RED", "END_TURN", None]],
            }
        )
        return seat


class NoStringTypeRoom(Room):
    def join(self, outbox, peer, request):
        seat = super().join(outbox, peer, request)
        outbox.push({"type": []})
        return seat


class BadColorRoom(Room):
    """Labels the bot's seat with the client's color in the commitments."""

    def _push_commitments(self, outbox):
        if self.commitments is not None and not getattr(self, "_bent", False):
            self._bent = True
            self.commitments = copy.deepcopy(self.commitments)
            self.commitments["seats"][1]["color"] = "RED"
        super()._push_commitments(outbox)


class OtherViewRoom(Room):
    """Shows the RED seat BLUE's view in its before."""

    def _before(self, color):
        return super()._before("BLUE" if color == "RED" else color)


@pytest.mark.parametrize(
    "room_class,message",
    [
        (SecondDecideRoom, "a second decide"),
        (EarlyDecideRoom, "before the game"),
        (NoStringTypeRoom, "unknown type"),
        (BadColorRoom, "seat 1 is malformed"),
        (OtherViewRoom, "another color"),
    ],
)
def test_the_python_client_refuses_what_it_must_not_sign_under(room_class, message):
    async def scenario():
        async with serving(room_class=room_class) as (server, base):
            room = await create_room(base, CONFIG, [None, "R"])
            client, decide = bot_client(f"{base}/room/{room['id']}", "R", verify=False)
            with pytest.raises(ServerMisbehaved, match=message):
                await client.play(decide)
            return client

    client = run(scenario())
    if room_class in (SecondDecideRoom, EarlyDecideRoom):
        indices = [item["statement"]["index"] for item in client.sent]
        assert len(indices) == len(set(indices)), "never two answers for one move"


def test_hex_is_strict_everywhere():
    """Round 6: bytes.fromhex took ' ab ', so the server published a padded
    pop the browser called malformed, and the browser's seat was blamed."""
    for value in (
        " " + "ab" * 32,
        "ab" * 32 + " ",
        "ab " * 32,
        "AB" * 32,
        "ab" * 31 + "a",
    ):
        assert not is_hex(value, 32)
    assert is_hex("ab" * 32, 32)

    async def scenario():
        async with serving() as (server, base):
            room = await create_room(
                base, {**CONFIG, "num_players": 3}, [None, None, "R"]
            )
            ws, hello = await raw_connect(f"{base}/room/{room['id']}")
            request, _, _ = join_request(hello)
            request["pop"] = " " + request["pop"] + " "
            await ws.send(json.dumps(request))
            answer = await next_body(ws, "error", "joined")
            await ws.close()
            return answer

    assert run(scenario())["code"] == "bad_join"


def test_strings_no_client_can_encode_are_refused():
    with pytest.raises(RoomError):
        parse_request('{"type": "join", "name": "\\ud800"}')


# ===== the browser =====
@needs_node
def test_the_page_signs_nothing_it_could_not_keep():
    """Round 6: with its storage full, the page sent its reveal anyway; after
    a reload it signed a second reveal over other commitments."""

    async def scenario():
        server = FakeServer()
        async with ScriptedPage(server.room.id) as page:
            await page.do(op="open")
            box = server.chain()
            await page.recv(server.hello(box))
            join = (await page.do(op="sit", seat=0))["sends"][0]
            await page.recv(server.joined(box, join))
            await page.do(op="full")
            return await page.recv(box.push(server.commitments()))

    out = run(scenario())
    assert not [s for s in out["sends"] if s["type"] == "reveal"]
    assert any("could not store" in p for p in out["problems"])


@needs_node
def test_after_a_reload_the_page_keeps_the_rules_it_sat_under():
    """Round 6: after a reload while the room waited, the page took other
    rules from the server's new seat chain."""

    async def scenario():
        server = FakeServer()
        async with ScriptedPage(server.room.id) as page:
            await page.do(op="open")
            box = server.chain()
            await page.recv(server.hello(box))
            join = (await page.do(op="sit", seat=0))["sends"][0]
            await page.recv(server.joined(box, join))
            await page.do(op="open")  # reload
            await page.recv(server.hello(server.chain()))
            fresh = server.chain()
            hello = server.room.hello_body(fresh.nonce)
            hello["room"]["config"]["vps_to_win"] = 3
            return await page.recv(fresh.push(hello))

    out = run(scenario())
    assert any("rules" in p for p in out["problems"]), out["problems"]


@needs_node
def test_the_page_refuses_lone_surrogates_and_wrong_colors():
    async def scenario(variant):
        server = FakeServer()
        async with ScriptedPage(server.room.id) as page:
            await page.do(op="open")
            box = server.chain()
            if variant == "surrogate":
                hello = server.room.hello_body(box.nonce)
                text = json.dumps(server.hello(box)).replace(
                    hello["room"]["name"], hello["room"]["name"] + "\\ud800", 1
                )
                return await page.do(op="recv", text=text)
            await page.recv(server.hello(box))
            join = (await page.do(op="sit", seat=0))["sends"][0]
            await page.recv(server.joined(box, join))
            commitments = server.commitments()
            commitments["seats"][1]["color"] = "RED"
            return await page.recv(box.push(commitments))

    surrogate = run(scenario("surrogate"))
    assert surrogate["messages"] == 0 and surrogate["problems"]
    color = run(scenario("color"))
    assert not [s for s in color["sends"] if s["type"] == "reveal"]
    assert any("malformed" in p for p in color["problems"])


@needs_node
def test_the_page_answers_one_decide_per_move_even_after_a_reload():
    """Round 6: a second decide for one move (or the same decide replayed
    after a reload) made the page sign a second answer."""

    async def scenario():
        server = FakeServer()
        async with ScriptedPage(server.room.id) as page:
            box, join, commitments, reveal = await _seated_page(page, server)
            body, commit = server.before(
                commitments, reveal["client_seed"], reveal["sig"]
            )
            await page.recv(box.push(body, commit))
            decide = {
                "type": "decide",
                "game_id": body["game_id"],
                "color": "RED",
                "index": 0,
                "state": body["state"],
                "log_start": 0,
                "log": [],
                "playable_actions": [["RED", "BUILD_SETTLEMENT", 0]],
                "timeout_ms": 20000,
                "current_color": "RED",
                "winning_color": None,
            }
            await page.recv(box.push(decide, commit))
            first = await page.do(op="act")
            second = await page.recv(box.push(decide, commit))
            await page.do(op="act")
            again = await page.do(op="act")
            await page.do(op="open")  # reload: the chain is replayed
            await page.recv(server.hello(server.chain()))
            for envelope in box.envelopes[:-1]:
                await page.recv(envelope)
            replayed = await page.do(op="act")
            return first, second, again, replayed

    first, second, again, replayed = run(scenario())
    assert [s["type"] for s in first["sends"]] == ["action"]
    assert any("second decide" in p for p in second["problems"])
    assert not [s for s in again["sends"] if s["type"] == "action"]
    assert not [s for s in replayed["sends"] if s["type"] == "action"]
    notes = replayed["saved"].get("unanswered") or []
    assert not notes, "an answered decide is not 'unanswered'"


def test_a_chain_addressed_to_a_key_it_never_seated_is_the_server_s_doing(voided):
    """Found by the fuzzer: with the join's type changed, a seat's chain has
    no join, and the verifier called the transcript INVALID (not this
    client's) although the server signed messages to the client's key."""
    key, red, *_ = voided
    transcript = copy.deepcopy(red)
    joined = _first(transcript, "joined")
    transcript["messages"][joined]["body"]["type"] = "x"
    transcript["messages"] = resign(key, transcript["messages"])
    report = report_of(transcript)
    assert report.errors and not report.invalid, report.invalid
