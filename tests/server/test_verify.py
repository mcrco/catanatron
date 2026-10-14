"""The verifier passes an honest server and catches a dishonest one.

Each dishonest server is a Room that cheats in one way and signs the result
as usual, so what is tested is the scheme, not merely the signatures.
"""

import asyncio
import copy
import json
import random

import pytest

from catanatron.apply_action import roll_dice
from catanatron.models.enums import ActionRecord, ActionType
from catanatron.serialization import action_to_json
from catanatron.server.client import create_room
from catanatron.server.config import default_action
from catanatron.server.crypto import SigningKey
from catanatron.server.room import Room
from catanatron.server.verify import verify_transcript

from tests.server.helpers import bot_client, run, serving

CONFIG = {"num_players": 3, "vps_to_win": 6, "decision_timeout_ms": 20_000}


def play(room_class=Room, remote=2, spec="F"):
    """A game with ``remote`` client seats (RED first) and server bots in the
    rest; returns the clients' transcripts and the server's room."""

    async def scenario():
        async with serving(room_class=room_class) as (server, base):
            seats = [None] * remote + ["F"] * (CONFIG["num_players"] - remote)
            room = await create_room(base, CONFIG, seats)
            url = f"{base}/room/{room['id']}"
            clients = [
                bot_client(url, spec, seat=i, verify=False) for i in range(remote)
            ]
            await asyncio.gather(*(c.play(d) for c, d in clients))
            return [c.transcript() for c, _ in clients], server.rooms[room["id"]]

    return run(scenario())


def errors_of(transcript):
    report = verify_transcript(json.loads(json.dumps(transcript)))
    return report.errors


@pytest.fixture(scope="module")
def honest():
    return play()


def test_an_honest_servers_transcripts_verify(honest):
    transcripts, room = honest
    for transcript in transcripts:
        report = verify_transcript(transcript)
        assert report.ok, report.errors
        assert report.counts["views"] > 10
        assert report.counts["client signatures"] > 0


# ===== (a) a tampered view =====
class TamperingRoom(Room):
    """Tells RED that the next seat holds one card more than it does."""

    def _send_all(self, build):
        def tampered(color):
            body = build(color)
            if color == "RED" and body["type"] == "step" and body["index"] == 30:
                body = copy.deepcopy(body)
                colors = body["state"]["colors"]
                other = (colors.index("RED") + 1) % len(colors)
                body["state"]["player_state"][f"P{other}_NUM_RESOURCES_IN_HAND"] += 1
            return body

        super()._send_all(tampered)


def test_a_tampered_view_is_caught():
    transcripts, _ = play(TamperingRoom)
    red, blue = transcripts
    assert any("step at state 30" in e and "NUM_RESOURCES" in e for e in errors_of(red))
    assert errors_of(blue) == [], "only RED was lied to"


# ===== (b) a dice roll altered after the commitment =====
class LoadedDiceRoom(Room):
    """Overrides the first roll after the initial placements."""

    loaded = False

    def _execute(self, action):
        if action.action_type == ActionType.ROLL and not self.loaded:
            self.loaded = True
            peek = random.Random()
            peek.setstate(self.game.random.getstate())
            honest = roll_dice(peek)
            loaded = (6, 6) if sum(honest) != 12 else (3, 4)
            return self.game.execute(action, action_record=ActionRecord(action, loaded))
        return super()._execute(action)


def test_an_altered_dice_roll_is_caught():
    transcripts, _ = play(LoadedDiceRoom)
    for transcript in transcripts:
        errors = errors_of(transcript)
        assert any("does not open" in e for e in errors), errors
        assert any(".log[0]" in e for e in errors), "the log shows other dice"


class IgnoredSeedRoom(Room):
    """Publishes the clients' seeds but seeds the game without them."""

    def _master_seed(self):
        from catanatron.server.crypto import master_seed

        return master_seed(self.server_seed, [None] * len(self.seats))


def test_a_game_not_seeded_from_the_clients_seeds_is_caught():
    transcripts, _ = play(IgnoredSeedRoom)
    for transcript in transcripts:
        assert errors_of(transcript), "the replay from the real seed must diverge"


# ===== (c) forged actions =====
class SubstitutingRoom(Room):
    """Plays another move than the one RED signed, keeping RED's signature."""

    forged = False

    async def _remote_decision(self, seat, playable):
        decided = await super()._remote_decision(seat, playable)
        if decided and seat.color.value == "RED" and not self.forged:
            action, statement, signature = decided
            others = [a for a in playable if a != action]
            if others:
                self.forged = True
                return others[0], statement, signature
        return decided


class ForgingRoom(Room):
    """Signs a move for RED with a key that is not RED's."""

    forged = False
    impostor = SigningKey.generate()

    async def _remote_decision(self, seat, playable):
        decided = await super()._remote_decision(seat, playable)
        if decided and seat.color.value == "RED" and not self.forged:
            action, statement, _ = decided
            others = [a for a in playable if a != action]
            if others:
                self.forged = True
                statement = {**statement, "action": action_to_json(others[0])}
                return others[0], statement, self.impostor.sign(statement)
        return decided


class FalseTimeoutRoom(Room):
    """Claims RED timed out, and plays a move of its choosing for it."""

    forged = False

    async def _remote_decision(self, seat, playable):
        decided = await super()._remote_decision(seat, playable)
        if decided and seat.color.value == "RED" and not self.forged:
            if playable[-1] != default_action(playable):
                self.forged = True
                return None  # recorded as a timeout
        return decided

    def _timeout_action(self, playable):
        return playable[-1]


@pytest.mark.parametrize(
    "room_class,message",
    [
        (SubstitutingRoom, "signed statement is for another move"),
        (ForgingRoom, "not signed by its key"),
        (FalseTimeoutRoom, "a timeout must play the default action"),
    ],
)
def test_a_forged_move_is_caught_by_everyone(room_class, message):
    transcripts, room = play(room_class)
    assert room.forged
    for transcript in transcripts:
        assert any(message in e for e in errors_of(transcript)), errors_of(transcript)


# ===== the transcript itself =====
def test_an_edited_transcript_fails_its_signatures(honest):
    transcripts, _ = honest
    edited = copy.deepcopy(transcripts[0])
    step = next(m for m in edited["messages"] if m["body"]["type"] == "step")
    step["body"]["state"]["player_state"]["P0_VICTORY_POINTS"] += 1
    assert any("bad server signature" in e for e in errors_of(edited))


def test_a_transcript_claiming_another_seed_fails(honest):
    transcripts, _ = honest
    edited = copy.deepcopy(transcripts[0])
    edited["client_seed"] = "11" * 32
    errors = errors_of(edited)
    assert any("another key or seed" in e for e in errors)
    assert any("seed did not go into the game" in e for e in errors)


def test_someone_elses_transcript_does_not_verify_as_mine(honest):
    transcripts, _ = honest
    edited = copy.deepcopy(transcripts[0])
    edited["public_key"] = transcripts[1]["public_key"]
    assert any("addressed to" in e for e in errors_of(edited))


def test_an_unfinished_transcript_cannot_be_verified_yet(honest):
    transcripts, _ = honest
    edited = copy.deepcopy(transcripts[0])
    edited["messages"] = edited["messages"][:40]
    assert any("not ended" in e for e in errors_of(edited))
