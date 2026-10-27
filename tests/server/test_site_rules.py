"""Rooms under OpenHex's 1v1 rules: ``robber_mode: friendly_public`` (the
friendly robber on public points) and ``dice_mode: balanced``. Rooms open,
play and verify under them; Catanatron's own friendly robber is still
refused; configs and transcripts from before the two keys existed are
unchanged."""

import asyncio
import json
import os
import random

import pytest

from catanatron.models.dice import BalancedDice
from catanatron.models.enums import ActionPrompt, ActionRecord, ActionType
from catanatron.serialization import state_to_json
from catanatron.server.client import GameClient, Refused, create_room
from catanatron.server.config import ConfigError, GameConfig, new_game, publicly_forced
from catanatron.server.room import Room
from catanatron.server.verify import load_transcript, verify_transcript, verify_transcripts

from tests.server.helpers import bot_client, run, serving

#: OpenHex's 1v1 (catanrl's base-2p-fr): 15 points, discard above 9.
SITE = {
    "num_players": 2,
    "vps_to_win": 15,
    "discard_limit": 9,
    "robber_mode": "friendly_public",
    "dice_mode": "balanced",
    "decision_timeout_ms": 20_000,
}
#: Every key a room config had before robber_mode and dice_mode.
OLD_KEYS = {
    "num_players", "map_type", "map", "number_placement", "vps_to_win", "discard_limit",
    "friendly_robber", "supply", "decision_timeout_ms", "autoplay_forced",
}
FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


# ===== configs =====
def test_the_site_config_round_trips_and_names_its_modes():
    config = GameConfig.from_json(SITE)
    assert config.robber_mode == "friendly_public" and config.dice_mode == "balanced"
    assert config.game_robber_mode == "friendly_public"
    assert config.to_json()["robber_mode"] == "friendly_public"
    assert config.to_json()["dice_mode"] == "balanced"
    assert GameConfig.from_json(config.to_json()) == config


def test_old_configs_keep_their_canonical_form():
    for payload in ({}, {"num_players": 2, "vps_to_win": 15, "discard_limit": 9}, {"friendly_robber": True}):
        config = GameConfig.from_json(payload)
        assert set(config.to_json()) == OLD_KEYS
        assert config.robber_mode == "normal" and config.dice_mode == "random"
    # The defaults said out loud are the same config, in the same canonical form.
    explicit = GameConfig.from_json({"robber_mode": "normal", "dice_mode": "random"})
    assert explicit == GameConfig.from_json({}) and set(explicit.to_json()) == OLD_KEYS
    assert GameConfig.from_json({"friendly_robber": True}).game_robber_mode == "friendly"


@pytest.mark.parametrize(
    "payload",
    [
        {"robber_mode": "friendly"},  # Catanatron's own is friendly_robber
        {"robber_mode": "FRIENDLY_PUBLIC"},
        {"robber_mode": True},
        {"dice_mode": "loaded"},
        {"dice_mode": None},
        {"friendly_robber": True, "robber_mode": "friendly_public"},
    ],
)
def test_bad_modes_are_rejected(payload):
    with pytest.raises(ConfigError):
        GameConfig.from_json(payload)


def test_a_rooms_game_plays_the_configs_rules():
    game = new_game(GameConfig.from_json(SITE), b"\0" * 32, "g")
    state = game.state
    assert state.robber_mode == "friendly_public" and state.friendly_robber
    assert state.dice_mode == "balanced" and state.balanced_dice == BalancedDice.fresh(2)
    assert game.vps_to_win == 15 and state.discard_limit == 9
    old = new_game(GameConfig.from_json({"num_players": 2}), b"\0" * 32, "g")
    assert old.state.robber_mode == "normal" and old.state.balanced_dice is None


def test_a_lone_robber_move_is_publicly_forced_on_public_points_only():
    for config, forced in ((SITE, True), ({"num_players": 2, "friendly_robber": True}, False)):
        game = new_game(GameConfig.from_json(config), b"\1" * 32, "g")
        while game.state.current_prompt != ActionPrompt.MOVE_ROBBER:
            actions = game.playable_actions
            roll = next((a for a in actions if a.action_type == ActionType.ROLL), None)
            if roll is not None and game.state.balanced_dice is None:
                game.execute(roll, action_record=ActionRecord(roll, (1, 6)))
            elif roll is not None and game.state.balanced_dice.can_draw((1, 6)):
                game.execute(roll, action_record=ActionRecord(roll, (1, 6)))
            else:
                game.execute(actions[0])
        game.playable_actions = game.playable_actions[:1]
        assert publicly_forced(game) is forced


# ===== rooms =====
def test_rooms_open_under_the_site_rules_and_old_configs_still_open():
    async def scenario():
        async with serving() as (server, base):
            site = await create_room(base, SITE, [None, None])
            old = await create_room(base, {"num_players": 2}, [None, None])
            with pytest.raises(Refused, match="friendly_public"):
                await create_room(base, {"num_players": 2, "friendly_robber": True}, [None, None])
            with pytest.raises(Refused, match="robber_mode"):
                await create_room(base, {"num_players": 2, "robber_mode": "friendly"}, [None, None])
            return site, old

    site, old = run(scenario())
    assert site["config"] == GameConfig.from_json(SITE).to_json()
    assert old["config"] == GameConfig.from_json({"num_players": 2}).to_json()
    assert set(old["config"]) == OLD_KEYS


def test_a_game_under_the_site_rules_verifies(tmp_path):
    """A remote bot against a server bot, watched; every transcript verifies
    (the verifier replays the balanced dice and the friendly robber from the
    revealed seeds), and the game rolled the balanced dice."""

    async def scenario():
        async with serving() as (server, base):
            room = await create_room(base, SITE, [None, "F"])
            url = f"{base}/room/{room['id']}"
            player, decide = bot_client(url, "F", transcript_dir=str(tmp_path))
            watcher = GameClient(url, spectate=True, transcript_dir=str(tmp_path))
            watching = asyncio.create_task(watcher.play())
            result = await player.play(decide)
            return server.rooms[room["id"]].game, result, await watching

    game, played, watched = run(scenario())
    for result in (played, watched):
        assert result.verified, result.report.errors
        assert result.winning_color is not None
    assert game.state.robber_mode == "friendly_public"
    rolls = [r.result for r in game.state.action_records if r.action.action_type == ActionType.ROLL]
    assert len(rolls) > 23  # past the first refill of the deck
    dice = BalancedDice.fresh(2)
    for record in game.state.action_records:
        if record.action.action_type == ActionType.ROLL:
            dice.apply(record.result, game.state.colors.index(record.action.color))
    assert dice == game.state.balanced_dice
    assert state_to_json(game)["balanced_dice"] == dice.to_json([c.value for c in game.state.colors])
    transcripts = [json.loads(open(r.transcript_path).read()) for r in (played, watched)]
    assert all(r.ok for r in verify_transcripts(transcripts))


class LoadedBalancedDiceRoom(Room):
    """Overrides the first roll after the initial placements with another
    card of the balanced deck."""

    loaded = False

    def _execute(self, action):
        if action.action_type == ActionType.ROLL and not self.loaded:
            self.loaded = True
            dice = self.game.state.balanced_dice
            peek = random.Random()
            peek.setstate(self.game.random.getstate())
            honest = dice.draw(peek, self.game.state.colors.index(action.color))
            loaded = (6, 6) if sum(honest) != 12 else (3, 4)
            return self.game.execute(action, action_record=ActionRecord(action, loaded))
        return super()._execute(action)


def test_a_loaded_balanced_roll_is_caught():
    async def scenario():
        async with serving(room_class=LoadedBalancedDiceRoom) as (server, base):
            room = await create_room(base, {**SITE, "vps_to_win": 5}, [None, "F"])
            player, decide = bot_client(f"{base}/room/{room['id']}", "F", verify=False)
            await player.play(decide)
            return player.transcript()

    errors = verify_transcript(json.loads(json.dumps(run(scenario())))).errors
    assert any("does not open" in e for e in errors), errors
    assert any(".log[0]" in e for e in errors), "the log shows other dice"


# ===== transcripts from before =====
def test_a_transcript_from_before_the_modes_still_verifies():
    """A 2p game the server played before robber_mode and dice_mode existed
    (catanatron bb3710d, protocol 5): its config, documents and views are
    what this code builds, so it verifies unchanged."""
    transcript = load_transcript(os.path.join(FIXTURES, "before-rule-modes-2p.json.gz"))
    joined = next(m["body"] for m in transcript["messages"] if m["body"]["type"] == "joined")
    assert set(joined["room"]["config"]) == OLD_KEYS
    report = verify_transcript(transcript)
    assert report.ok, report.errors + report.invalid
    assert report.counts["actions"] > 100
