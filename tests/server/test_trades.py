"""Domestic trades in a room with ``trades``: a seat that may offer is told so
in its decide, the room plays a valid offer it was not listed, the others
answer, the offerer confirms, and every transcript (each seat's and a
spectator's) verifies. Rooms without trades are what they were."""

import asyncio
import random

from catanatron.game import Game
from catanatron.models.enums import ActionPrompt, ActionType
from catanatron.models.player import Color, RandomPlayer
from catanatron.server.client import GameClient, create_room
from catanatron.server.config import (
    MAX_OFFERS_PER_TURN,
    GameConfig,
    offer_action,
    offer_allowed,
)
from catanatron.server.verify import verify_transcript
from catanatron.state_functions import get_player_freqdeck

from tests.server.helpers import bot_client, run, serving

KINDS = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")


def _rolled_game(seed=3):
    """A 3-player game at its first turn after the roll with cards in hand."""
    game = Game([RandomPlayer(c) for c in (Color.RED, Color.BLUE, Color.ORANGE)], seed=seed)
    while game.state.is_initial_build_phase:
        game.play_tick()
    while True:
        roll = [a for a in game.playable_actions if a.action_type == ActionType.ROLL]
        if roll:
            game.execute(roll[0])
        state = game.state
        color = state.current_color()
        if state.current_prompt == ActionPrompt.PLAY_TURN and sum(get_player_freqdeck(state, color)):
            return game
        game.play_tick()


def _offer_json(game, give_kind, ask_kind, give=1, ask=1):
    value = [0] * 10
    value[KINDS.index(give_kind)] = give
    value[5 + KINDS.index(ask_kind)] = ask
    return [game.state.current_color().value, "OFFER_TRADE", value]


def test_the_offer_rule():
    game = _rolled_game()
    color = game.state.current_color()
    hand = get_player_freqdeck(game.state, color)
    have = KINDS[next(i for i, n in enumerate(hand) if n)]
    other = next(k for k in KINDS if k != have)
    missing = next((k for i, k in enumerate(KINDS) if not hand[i]), None)

    assert not offer_allowed(game, False)
    assert offer_allowed(game, True)
    assert offer_action(game, False, _offer_json(game, have, other)) is None
    action = offer_action(game, True, _offer_json(game, have, other))
    assert action is not None and action.action_type == ActionType.OFFER_TRADE
    # Malformed or unpayable offers are no moves.
    assert offer_action(game, True, _offer_json(game, have, have)) is None  # the same kind both sides
    assert offer_action(game, True, _offer_json(game, have, other, give=hand[KINDS.index(have)] + 1)) is None
    if missing is not None:
        assert offer_action(game, True, _offer_json(game, missing, have)) is None
    bad = _offer_json(game, have, other)
    bad[2][0] = 1.0  # not an integer
    assert offer_action(game, True, bad) is None
    assert offer_action(game, True, ["BLUE" if color.value != "BLUE" else "RED", "OFFER_TRADE", bad[2]]) is None
    # At most MAX_OFFERS_PER_TURN offers a turn: each, rejected by all, returns to the turn.
    for _ in range(MAX_OFFERS_PER_TURN):
        game.execute(offer_action(game, True, _offer_json(game, have, other)))
        while game.state.current_prompt == ActionPrompt.DECIDE_TRADE:
            game.execute(next(a for a in game.playable_actions if a.action_type == ActionType.REJECT_TRADE))
        assert game.state.current_prompt == ActionPrompt.PLAY_TURN
    assert not offer_allowed(game, True)


def test_a_config_without_trades_is_unchanged():
    assert "trades" not in GameConfig.from_json({"num_players": 3}).to_json()
    assert GameConfig.from_json({"num_players": 3, "trades": True}).to_json()["trades"] is True


def _trading(spec, seed):
    """A bot client that also trades: offers a random one-for-one when it
    may, accepts half the offers it can pay, confirms the first acceptee."""
    rng = random.Random(seed)
    stats = {"offers": 0, "accepts": 0, "confirms": 0, "flags": 0}

    def wrap(client, decide):
        def view_color(view):
            return client.table.color

        def trade_decide(view, playable):
            kinds = {a[1] for a in playable}
            if "CONFIRM_TRADE" in kinds:
                stats["confirms"] += 1
                return next(a for a in playable if a[1] == "CONFIRM_TRADE")
            if "REJECT_TRADE" in kinds:
                accept = [a for a in playable if a[1] == "ACCEPT_TRADE"]
                if accept and rng.random() < 0.5:
                    stats["accepts"] += 1
                    return accept[0]
                return next(a for a in playable if a[1] == "REJECT_TRADE")
            if view.get("offer_trade"):
                stats["flags"] += 1
                seat = view["colors"].index(view_color(view))
                hand = [view["player_state"][f"P{seat}_{k}_IN_HAND"] for k in KINDS]
                held = [i for i, n in enumerate(hand) if n]
                if held and rng.random() < 0.5:
                    give = rng.choice(held)
                    ask = rng.choice([i for i in range(5) if i != give])
                    value = [0] * 10
                    value[give], value[5 + ask] = 1, 1
                    stats["offers"] += 1
                    return [view_color(view), "OFFER_TRADE", value]
            return decide(view, playable)

        return trade_decide

    return wrap, stats


def test_a_room_with_trades_plays_offers_and_every_transcript_verifies():
    async def scenario():
        async with serving() as (server, base):
            room = await create_room(base, {"num_players": 3, "vps_to_win": 6, "trades": True}, [None] * 3)
            url = f"{base}/room/{room['id']}"
            seats, stats = [], []
            for seed in range(3):
                client, decide = bot_client(url, transcript_dir=None)
                wrap, counts = _trading("F", seed)
                seats.append((client, wrap(client, decide)))
                stats.append(counts)
            watcher = GameClient(url, spectate=True, transcript_dir=None)
            results = await asyncio.gather(*(c.play(d) for c, d in seats), watcher.play())
            return server.rooms[room["id"]], seats, watcher, results, stats

    live, seats, watcher, results, stats = run(scenario())
    assert all(r.winning_color for r in results[:3])
    kinds = [entry["action"][1] for entry in live.actions]
    assert "OFFER_TRADE" in kinds and "CONFIRM_TRADE" in kinds, sum(s["offers"] for s in stats)
    assert sum(s["flags"] for s in stats) > 0
    for client, _ in seats:
        report = verify_transcript(client.transcript())
        assert report.ok, report.errors[:3]
    report = verify_transcript(watcher.transcript())
    assert report.ok, report.errors[:3]


def test_a_room_without_trades_refuses_offers_and_says_nothing_of_them():
    async def scenario():
        async with serving() as (server, base):
            room = await create_room(base, {"num_players": 2, "vps_to_win": 4}, [None, None])
            url = f"{base}/room/{room['id']}"
            seen = []
            clients = []
            for seed in range(2):
                client, decide = bot_client(url, transcript_dir=None)

                def offering(view, playable, decide=decide, client=client):
                    seen.append("offer_trade" in view)
                    return decide(view, playable)

                clients.append((client, offering))
            results = await asyncio.gather(*(c.play(d) for c, d in clients))
            return results, seen, [c.messages for c, _ in clients]

    results, seen, messages = run(scenario())
    assert all(r.winning_color for r in results)
    assert seen and not any(seen)
    decides = [m["body"] for chain in messages for m in chain if m["body"].get("type") == "decide"]
    assert decides and not any("offer_trade" in body for body in decides)
