"""Local sessions without a terminal: seats, views, history, chance, files."""

import json
import random

import pytest

from catanatron.models.decks import BASE_SUPPLY, EXTENSION_SUPPLY
from catanatron.models.enums import RESOURCES, Action, ActionType
from catanatron.models.player import Color, Player
from catanatron.tui.agents import make_agent
from catanatron.tui.local import LocalSession, parse_outcome
from catanatron.tui.positions import GameSettings, document_of, supply_of
from catanatron.tui.session import CommandError

from tests.tui.helpers import SHAPES, map_spec_file

FAST = GameSettings(num_players=2, vps_to_win=5)


def documents_equal(a, b):
    return json.loads(json.dumps(a)) == json.loads(json.dumps(b))


def play_randomly(session, rng, limit=5000):
    """Play to the end, a person choosing at random; returns their choices."""
    choices = 0
    while not session.over and choices < limit:
        session.run()
        if session.person_to_move():
            frame = session.frame()
            assert frame.actor == session.current_color.value
            assert set(frame.actions) == set(session.game.playable_actions)
            session.choose(rng.choice(frame.actions))
            choices += 1
    return choices


def roll_now(session):
    """Play bots until the seat to move can roll."""
    for _ in range(500):
        roll = Action(session.current_color, ActionType.ROLL, None)
        if roll in session.game.playable_actions:
            return roll
        if session.step() is None:
            session.choose(session.game.playable_actions[0])
    raise AssertionError("no roll came up")


# ===== playing =====
def test_a_person_plays_bots_to_the_end_and_is_only_asked_real_choices():
    session = LocalSession.new(["human", "F"], settings=FAST, seed=1)
    asked = play_randomly(session, random.Random(0))
    assert session.over and session.winner is not None
    assert asked > 0
    frame = session.frame()
    assert frame.winner == session.winner.value and not frame.actions


@pytest.mark.parametrize("num_players", [2, 3, 4, 5, 6])
def test_bots_finish_games_of_any_size_on_any_shape(num_players):
    name = SHAPES[num_players % len(SHAPES)]
    settings = GameSettings(
        num_players=num_players, map=map_spec_file(name), vps_to_win=6
    )
    seats = (["F", "R"] * 3)[:num_players]
    session = LocalSession.new(seats, settings=settings, seed=num_players)
    session.run()
    assert session.over
    assert len(session.colors) == num_players
    expected = EXTENSION_SUPPLY if num_players > 4 else BASE_SUPPLY
    assert supply_of(session.game) == expected


def test_seat_specs():
    assert make_agent("human", Color.RED).human
    assert make_agent("H", Color.RED).human
    bot = make_agent("AB:1", Color.RED)
    assert not bot.human and not bot.oracle and bot.label == "AB:1"
    oracle = make_agent("oracle:F", Color.RED)
    assert oracle.oracle and oracle.label == "oracle:F"
    with pytest.raises(ValueError):
        make_agent("NOSUCHBOT", Color.RED)
    with pytest.raises(ValueError):
        LocalSession.new(["human", "F", "F"], settings=FAST)


# ===== what bots see =====
def test_bots_decide_from_their_view_and_oracles_from_everything():
    seen = {"mine": [], "peek": []}

    def spy(name):
        def decide(view, playable_actions):
            seen[name].append((view, playable_actions))
            return playable_actions[0]

        return decide

    session = LocalSession.new(
        ["mine", "oracle:peek", "F"],
        settings=GameSettings(num_players=3, vps_to_win=4),
        seed=3,
        decide_fns={"mine": spy("mine"), "peek": spy("peek")},
    )
    session.run(limit=300)
    assert seen["mine"] and seen["peek"]
    me = session.colors.index(Color.RED)
    for view, playable in seen["mine"]:
        state = view["player_state"]
        assert f"P{me}_WOOD_IN_HAND" in state
        others = [i for i in range(3) if i != me]
        assert all(f"P{i}_WOOD_IN_HAND" not in state for i in others)
        assert all(f"P{i}_NUM_RESOURCES_IN_HAND" in state for i in others)
        assert "random_state" not in view and "development_listdeck" not in view
        assert "seed" not in view["game"]
        assert view["map"] and "action_records" in view
        # The server's order: Catanatron's own (the same in every process).
        keys = [json.dumps(a, sort_keys=True, separators=(",", ":")) for a in playable]
        assert len(set(keys)) == len(keys) and all(len(a) == 3 for a in playable)
    for view, _ in seen["peek"]:
        state = view["player_state"]
        assert all(f"P{i}_WOOD_IN_HAND" in state for i in range(3))
        assert "random_state" not in view


class Recorder(Player):
    """A Player that remembers the games it was handed."""

    def __init__(self, color=Color.RED, params=None):
        super().__init__(color, params)
        self.games = []

    def decide(self, game, playable_actions):
        self.games.append(game)
        return playable_actions[0]


def test_players_decide_on_sampled_worlds_and_oracles_on_the_true_game():
    sampled, oracle = Recorder(), Recorder()
    session = LocalSession.new(
        ["sampled", "oracle:true"],
        settings=FAST,
        seed=5,
        decide_fns={"sampled": sampled, "true": oracle},
    )
    while len(oracle.games) < 20 and not session.over:
        turn = session.bot_turn()
        if turn is None or session.forced_action() is not None:
            session.step()
            continue
        hands = document_of(session.game)["player_state"]
        session.finish(turn, turn.think())
        handed = turn.request.game
        if turn.agent.oracle:
            assert handed is not None and handed is not session.game
            assert handed.state.player_state == hands
            # Its own random stream: lookahead cannot disturb the real dice.
            assert handed.state.random is not session.game.state.random
        else:
            assert handed is None
    assert sampled.games and oracle.games


def test_a_bot_decision_is_dropped_when_the_game_moved_on():
    session = LocalSession.new(["F", "F"], settings=FAST, seed=2)
    session.run(limit=10)
    turn = session.bot_turn()
    action = turn.think()
    session.undo()
    assert session.finish(turn, action) is False


# ===== history =====
def test_undo_and_redo_restore_the_game_exactly():
    session = LocalSession.new(
        ["F", "R", "F"], settings=GameSettings(num_players=3), seed=4
    )
    session.run(limit=150)
    before = document_of(session.game)
    assert session.undo(60) == 60
    middle = document_of(session.game)
    assert session.redo(60) == 60
    assert documents_equal(document_of(session.game), before)
    # Replaying from the middle draws the same chance as the first time.
    session.undo(60)
    assert documents_equal(document_of(session.game), middle)
    session.redo(1)
    assert session.undo(1) == 1
    record_count = len(session.game.state.action_records)
    session.run(limit=60)
    assert len(session.game.state.action_records) == record_count + 60
    assert session.redo(1) == 0, "a new move clears the redo stack"


def test_back_undoes_to_the_last_choice_of_a_person():
    session = LocalSession.new(["human", "F"], settings=FAST, seed=8)
    rng = random.Random(1)
    for _ in range(5):
        session.run()
        session.choose(rng.choice(session.frame().actions))
    session.run()
    choice_point = None
    while session.undo_stack:
        session.undo()
        if session.person_to_move():
            choice_point = document_of(session.game)
            break
    session.redo(10_000)
    session.back()
    assert documents_equal(document_of(session.game), choice_point)


# ===== chance =====
def test_forced_dice_card_and_steal():
    session = LocalSession.new(["human", "human"], settings=FAST, seed=6)
    rng = random.Random(2)
    roll = roll_now(session)
    session.execute("dice 6 6")
    record = session.play(roll)
    assert record.result == (6, 6)
    assert (
        "dice set to 6+6" in session.notes[len(session.game.state.action_records) - 1]
    )
    assert session.chance.dice is None

    # Give the seat to move what a development card costs, then force the draw.
    while not session.over:
        color = session.current_color
        if Action(color, ActionType.END_TURN, None) in session.game.playable_actions:
            break
        session.choose(rng.choice(session.game.playable_actions))
    session.execute(f"hand {color.value} sheep=1 wheat=1 ore=1")
    session.execute("card monopoly")
    buy = Action(color, ActionType.BUY_DEVELOPMENT_CARD, None)
    assert buy in session.game.playable_actions
    assert session.play(buy).result == "MONOPOLY"

    # Steal: give the victim one of everything, then a seven for the thief.
    session.play(Action(color, ActionType.END_TURN, None))
    thief = session.current_color
    victim = next(c for c in session.colors if c != thief)
    session.execute(f"hand {victim.value} wood=1 brick=1 sheep=1 wheat=1 ore=1")
    session.execute("dice 3 4")
    session.play(roll_now(session))
    while session.game.state.current_prompt.value == "DISCARD":
        session.choose(session.game.playable_actions[0])
    session.execute("steal ore")
    robbery = next(
        a
        for a in session.game.playable_actions
        if a.action_type == ActionType.MOVE_ROBBER and a.value[1] == victim
    )
    assert session.play(robbery).result == "ORE"


def test_explicit_outcomes_are_checked():
    session = LocalSession.new(["human", "human"], settings=FAST, seed=6)
    roll = roll_now(session)
    with pytest.raises(CommandError):
        session.play(roll, result=(7, 1))
    assert session.play(roll, result=parse_outcome("dice=2,5")).result == (2, 5)
    assert parse_outcome("card=yop") == "YEAR_OF_PLENTY"
    assert parse_outcome("steal=wh") == "WHEAT"
    for bad in ("dice=1", "card=castle", "nothing=1", "dice 6 7"):
        with pytest.raises(CommandError):
            parse_outcome(bad)
    for bad in ("dice 7 1", "dice 1", "card castle", "steal gold", "seat RED XYZ"):
        with pytest.raises(CommandError):
            session.execute(bad)
    assert "66" not in str(session.execute("dice 66"))
    assert session.chance.dice == (6, 6)
    session.execute("dice off")
    assert session.chance.dice is None


# ===== edits, views, seats =====
def test_edits_keep_the_bank_whole_and_can_be_undone():
    session = LocalSession.new(["human", "F"], settings=FAST, seed=7)
    session.run()
    supply = supply_of(session.game)
    before = document_of(session.game)
    color = session.colors[0]
    session.execute(f"hand {color.value} wood=5 ore=0")
    assert supply_of(session.game) == supply
    index = session.game.state.color_to_index[color]
    assert session.game.state.player_state[f"P{index}_WOOD_IN_HAND"] == 5
    with pytest.raises(CommandError):
        session.execute(f"hand {color.value} wood=50")
    tile = next(
        t for t in session.game.state.board.map.land_tiles.values() if t.resource
    )
    session.execute(f"robber {tile.id}")
    assert (
        session.game.state.board.robber_coordinate
        in session.game.state.board.map.land_tiles
    )
    session.undo(2)
    assert documents_equal(document_of(session.game), before)


def test_views_and_seats():
    session = LocalSession.new(
        ["human", "F", "F"], settings=GameSettings(num_players=3), seed=9
    )
    session.run()
    frame = session.frame()
    assert frame.viewer == "RED" and not frame.god
    red = frame.colors.index("RED")
    assert all(
        (f"P{i}_WOOD_IN_HAND" in frame.document["player_state"]) == (i == red)
        for i in range(3)
    )
    session.execute("god")
    frame = session.frame()
    assert frame.god and all(
        f"P{i}_WOOD_IN_HAND" in frame.document["player_state"] for i in range(3)
    )
    session.execute("view public")
    assert not any(
        f"P{i}_WOOD_IN_HAND" in session.frame().document["player_state"]
        for i in range(3)
    )
    session.execute("view BLUE")
    assert session.frame().viewer == "BLUE"
    session.execute("seat RED F")
    assert session.frame().seats["RED"] == "F"
    session.execute("view follow")
    assert session.frame().god, "with no person seated, follow shows everything"
    session.run()
    assert session.over


# ===== files =====
def test_save_and_load_keep_the_position_seats_notes_and_history(tmp_path):
    session = LocalSession.new(["human", "AB:1"], settings=FAST, seed=10)
    rng = random.Random(3)
    for _ in range(8):
        session.run()
        session.choose(rng.choice(session.frame().actions))
    session.execute("dice 5 5")
    session.run()
    path = str(tmp_path / "game.json")
    assert "saved" in session.execute(f"save {path}")

    loaded = LocalSession.load(path)
    assert documents_equal(document_of(loaded.game), document_of(session.game))
    assert loaded.specs == session.specs
    assert loaded.notes == session.notes
    assert len(loaded.undo_stack) == len(session.game.state.action_records)
    # The history replays to the very first position.
    loaded.undo(10_000)
    assert documents_equal(document_of(loaded.game), session.initial)
    loaded.redo(10_000)
    assert documents_equal(document_of(loaded.game), document_of(session.game))

    # And a session can load in place, keeping its hooks.
    calls = []
    other = LocalSession.new(["human", "F"], settings=FAST, seed=11)
    other.on_change = lambda: calls.append(1)
    assert "loaded" in other.execute(f"load {path}")
    assert calls and documents_equal(document_of(other.game), document_of(session.game))


def test_load_bare_documents_and_archives(tmp_path):
    session = LocalSession.new(["F", "F"], settings=FAST, seed=12)
    session.run(limit=80)
    doc = document_of(session.game)
    bare = tmp_path / "state.json"
    bare.write_text(json.dumps(doc))
    archive = tmp_path / "archive.json"
    archive.write_text(json.dumps({"room": {}, "final_state": doc}))
    for path in (bare, archive):
        loaded = LocalSession.load(str(path), ["human", "F"])
        assert documents_equal(document_of(loaded.game), doc)
        assert not loaded.undo_stack
        assert loaded.agents[Color.RED].human
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"hello": 1}))
    with pytest.raises(ValueError):
        LocalSession.load(str(bad))
    with pytest.raises(CommandError):
        session.execute(f"load {bad}")


def test_new_game_keeps_the_rules_and_seats():
    session = LocalSession.new(
        ["human", "F", "R"],
        settings=GameSettings(num_players=3, map=map_spec_file("blob25"), vps_to_win=7),
        seed=13,
    )
    session.execute("new 99")
    assert session.seed == 99 and session.game.vps_to_win == 7
    assert len(session.game.state.board.map.land_tiles) == 25
    assert [session.specs[c] for c in (Color.RED, Color.BLUE, Color.ORANGE)] == [
        "human",
        "F",
        "R",
    ]


def test_settings_validate():
    assert GameSettings.from_json({"num_players": 5}).game_supply == EXTENSION_SUPPLY
    assert GameSettings.from_json({"decision_timeout_ms": 5000}).num_players == 4
    for bad in ({"num_players": 7}, {"colour": 1}, {"supply": "x"}, {"map_type": "X"}):
        with pytest.raises(ValueError):
            GameSettings.from_json(bad)
    assert set(RESOURCES) == {"WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"}
