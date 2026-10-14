"""The terminal app on local games, driven headlessly with Textual's Pilot."""

import json
import random

import pytest

pytest.importorskip("textual")

from catanatron.models.enums import ActionType
from catanatron.serialization import state_to_json
from catanatron.tui.app import BoardView, CatanApp, HelpScreen
from catanatron.tui.local import LocalSession
from catanatron.tui.positions import GameSettings

from tests.tui.helpers import SHAPES, idle, map_spec_file, run, settle

FAST = GameSettings(num_players=2, vps_to_win=5)
SIZE = (160, 50)


def same(a, b):
    return json.loads(json.dumps(a)) == json.loads(json.dumps(b))


def make_app(seats=("human", "F"), settings=FAST, seed=1, **kwargs):
    session = LocalSession.new(list(seats), settings=settings, seed=seed)
    kwargs.setdefault("speed", 0)
    return CatanApp(session, **kwargs), session


async def command(pilot, app, line):
    await pilot.press("colon")
    assert app.focused is app.command_input
    app.command_input.value = line
    await pilot.press("enter")
    await pilot.pause()


def test_a_person_plays_a_whole_game_against_a_bot_with_the_keyboard():
    app, session = make_app(seed=3)
    rng = random.Random(0)

    async def scenario():
        async with app.run_test(size=SIZE) as pilot:
            choices = 0
            while True:
                await idle(pilot, app)
                if session.over:
                    break
                assert app.frame.actor == session.current_color.value
                for _ in range(rng.randrange(4)):
                    await pilot.press("down")
                await pilot.press("enter")
                choices += 1
            await pilot.pause(0.2)
            return choices, app.screen_text()

    choices, screen = run(scenario())
    assert (
        session.winner is not None and choices >= 4
    )  # two settlements, two roads at least
    assert f"{session.winner.value} wins!" in screen


def test_undo_back_and_redo_keys():
    app, session = make_app(seed=4)

    async def scenario():
        async with app.run_test(size=SIZE) as pilot:
            await idle(pilot, app)
            choice_point = state_to_json(session.game)
            await pilot.press("enter")
            await idle(pilot, app)
            after = state_to_json(session.game)
            await pilot.press("u")
            assert not app.autoplay, "taking moves back pauses the bots"
            assert len(session.redo_stack) == 1
            await pilot.press("U")
            assert same(state_to_json(session.game), choice_point)
            await pilot.press("r")
            assert (
                len(session.game.state.action_records)
                == len(choice_point["action_records"]) + 1
            )
            await command(pilot, app, "redo 1000")
            assert same(state_to_json(session.game), after)
            await pilot.press("space")
            assert app.autoplay

    run(scenario())


def test_forced_dice_step_and_god_view():
    app, session = make_app(seats=("F", "F"), seed=5, autoplay=False)

    async def scenario():
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause()
            assert session.frame().god, "nobody seated: everything is shown"
            await command(pilot, app, "dice 2 3")
            assert "next roll 2+3" in app.screen_text()
            await command(pilot, app, "view public")
            assert "spectator view" in app.screen_text()
            await pilot.press("g")
            assert session.frame().god
            start = len(session.game.state.action_records)
            await pilot.press("n")  # one bot move
            await settle(pilot, lambda: app.thinking is None)
            assert len(session.game.state.action_records) == start + 1
            await command(pilot, app, "step 200")
            await settle(pilot, lambda: app.steps_left == 0 and app.thinking is None)
            rolls = [
                r.result
                for r in session.game.state.action_records
                if r.action.action_type == ActionType.ROLL
            ]
            assert rolls and rolls[0] == (2, 3)

    run(scenario())


def test_save_and_load_in_the_app(tmp_path):
    app, session = make_app(seed=6)
    path = str(tmp_path / "game.json")

    async def scenario():
        async with app.run_test(size=SIZE) as pilot:
            await idle(pilot, app)
            await pilot.press("enter")
            await idle(pilot, app)
            saved = state_to_json(session.game)
            await command(pilot, app, f"save {path}")
            for _ in range(3):
                await pilot.press("enter")
                await idle(pilot, app)
            assert not same(state_to_json(session.game), saved)
            await command(pilot, app, f"load {path}")
            await idle(pilot, app)
            assert app.messages[-1].startswith("loaded")
            return saved

    saved = run(scenario())
    # Loading resumes at the saved position (and bots play on from there).
    records = json.loads(json.dumps(state_to_json(session.game)["action_records"]))
    assert records[: len(saved["action_records"])] == saved["action_records"]
    with open(path) as file:
        assert json.load(file)["format"] == "catanatron-tui/1"


def test_picking_on_the_board_by_click_and_by_keyboard():
    app, session = make_app(seed=7)

    async def scenario():
        async with app.run_test(size=SIZE) as pilot:
            await idle(pilot, app)
            assert session.game.state.current_prompt.value == "BUILD_INITIAL_SETTLEMENT"
            color = session.current_color
            node = max(a.value for a in session.game.playable_actions)
            x, y = app._picture.anchors[("node", node)]
            await pilot.click(BoardView, offset=(x, y))
            await pilot.pause()
            assert session.game.state.board.buildings.get(node) == (color, "SETTLEMENT")

            # The road: on the board with the keyboard.
            assert session.game.state.current_prompt.value == "BUILD_INITIAL_ROAD"
            await pilot.press("b")
            assert app.focused is app.board_view
            await pilot.press("right")
            await pilot.press("down")
            target = app.cursor
            assert target is not None and target[0] == "edge"
            await pilot.press("enter")
            await pilot.pause()
            assert session.game.state.board.roads.get(target[1]) == color

            # Clicking where nothing can be built only says what is there.
            await idle(pilot, app)
            busy = next(n for n in session.game.state.board.buildings)
            x, y = app._picture.anchors[("node", busy)]
            records = len(session.game.state.action_records)
            await pilot.click(BoardView, offset=(x, y))
            await pilot.pause()
            assert app.messages[-1].startswith(f"node {busy}:")
            assert len(session.game.state.action_records) == records

            # Hovering says what is there too.
            coordinate = next(iter(session.game.state.board.map.land_tiles))
            x, y = app._picture.anchors[("tile", coordinate)]
            await pilot.hover(BoardView, offset=(x, y))
            await pilot.pause()
            assert str(app.board_view.tooltip).startswith("tile #")

    run(scenario())


@pytest.mark.parametrize(
    "num_players,shape",
    [(2, "BASE"), (3, SHAPES[0]), (4, SHAPES[2]), (5, SHAPES[3]), (6, SHAPES[1])],
)
def test_every_shape_and_player_count_draws_and_fits(num_players, shape):
    settings = GameSettings(
        num_players=num_players, map=None if shape == "BASE" else map_spec_file(shape)
    )
    seats = ["human"] + ["F"] * (num_players - 1)
    app, session = make_app(seats, settings=settings, seed=num_players)

    async def scenario():
        async with app.run_test(size=(140, 42)) as pilot:
            await idle(pilot, app)
            screen = app.screen_text()
            board = app.board_view
            picture = app._picture
            return screen, board.size, picture

    screen, size, picture = run(scenario())
    for color in session.colors:
        assert color.value.title() in screen
    assert (size.width, size.height) == (picture.width, picture.height)
    land_nodes = session.game.state.board.map.land_nodes
    assert {t[1] for t in picture.anchors if t[0] == "node"} == set(land_nodes)


def test_help_screen_and_failing_bots():
    def broken(view, playable_actions):
        raise RuntimeError("no idea")

    session = LocalSession.new(
        ["human", "broken"], settings=FAST, seed=8, decide_fns={"broken": broken}
    )
    app = CatanApp(session, speed=0)

    async def scenario():
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("question_mark")
            assert isinstance(app.screen, HelpScreen)
            # The game goes on underneath.
            await settle(pilot, lambda: not app.autoplay or session.person_to_move())
            await pilot.press("escape")
            assert not isinstance(app.screen, HelpScreen)
            for _ in range(50):
                await settle(
                    pilot, lambda: session.person_to_move() or not app.autoplay
                )
                if not app.autoplay:
                    break
                await pilot.press("enter")
            # The bot's error pauses the bots and says what went wrong.
            assert not app.autoplay
            assert any("failed: RuntimeError: no idea" in m for m in app.messages)
            assert not session.agents[session.current_color].human

    run(scenario())


def test_hot_seat_follows_the_seat_to_move():
    app, session = make_app(seats=("human", "human"), seed=9)

    async def scenario():
        async with app.run_test(size=SIZE) as pilot:
            viewers = set()
            for _ in range(12):
                await idle(pilot, app)
                assert app.frame.viewer == session.current_color.value
                viewers.add(app.frame.viewer)
                await pilot.press("enter")
            return viewers

    assert run(scenario()) == {"RED", "BLUE"}


def test_a_robber_tile_with_two_victims_narrows_the_list():
    session = LocalSession.new(
        ["human"] * 3, settings=GameSettings(num_players=3), seed=10
    )
    board = session.game.state.board
    tile_coordinate, tile = next(
        (c, t)
        for c, t in board.map.land_tiles.items()
        if t.resource is not None and c != board.robber_coordinate
    )
    corners = list(tile.nodes.values())
    # Everyone settles on that tile first (north, south-east and south-west
    # corners are two roads apart), then anywhere.
    for node in (corners[0], corners[2], corners[4]):
        session.choose(
            next(a for a in session.game.playable_actions if a.value == node)
        )
        session.choose(session.game.playable_actions[0])
    while session.game.state.is_initial_build_phase:
        session.choose(session.game.playable_actions[0])
    thief = session.current_color
    for color in session.colors:
        session.execute(f"hand {color.value} wood=1 brick=1")
    session.execute("dice 1 6")
    app = CatanApp(session, speed=0)

    async def scenario():
        async with app.run_test(size=SIZE) as pilot:
            # Rolling is the only move, so the app plays it: a seven.
            await settle(pilot, lambda: session.person_to_move())
            await pilot.pause(0.2)
            assert session.game.state.current_prompt.value == "MOVE_ROBBER"
            x, y = app._picture.anchors[("tile", tile_coordinate)]
            await pilot.click(BoardView, offset=(x, y))
            await pilot.pause(0.2)
            shown = list(app._shown_actions.values())
            assert len(shown) == 2 and all(a.value[0] == tile_coordinate for a in shown)
            assert app.focused is app.actions_view
            await pilot.press("down", "enter")
            await pilot.pause(0.2)
            return shown[1]

    chosen = run(scenario())
    last = session.game.state.action_records[-1]
    assert last.action == chosen and last.action.color == thief
    assert session.game.state.board.robber_coordinate == tile_coordinate
