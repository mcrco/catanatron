"""Remote games: the app as a catanatron-server client, on a real local server."""

import asyncio
import contextlib
import json
import os

import pytest

pytest.importorskip("textual")
pytest.importorskip("websockets")
pytest.importorskip("cryptography")

from catanatron.server.app import GameServer, run_server
from catanatron.server.client import create_room
from catanatron.server.verify import verify_transcript
from catanatron.tui.app import CatanApp
from catanatron.tui.cli import build_parser, remote_session
from catanatron.tui.remote import RemoteSession, describe_rooms

from tests.tui.helpers import choices_shown, free_port, run, settle

FAST = {"num_players": 2, "vps_to_win": 4, "decision_timeout_ms": 60_000}
SIZE = (160, 50)


@contextlib.asynccontextmanager
async def serving():
    """A GameServer on a free localhost port in 20000-29999."""
    server = GameServer()
    for _ in range(5):
        port = free_port()
        try:
            ws_server = await run_server(server, "127.0.0.1", port)
            break
        except OSError:
            continue
    try:
        yield server, f"ws://127.0.0.1:{port}"
    finally:
        ws_server.close()
        await ws_server.wait_closed()


def test_play_a_server_game_to_the_end_and_verify_it(tmp_path):
    async def scenario():
        async with serving() as (server, base):
            session = RemoteSession(
                base,
                create=FAST,
                create_seats=[None, "F"],
                seat=0,
                transcript_dir=str(tmp_path),
            )
            app = CatanApp(session)
            async with app.run_test(size=SIZE) as pilot:
                choices = 0
                while not session.done.is_set():
                    await pilot.pause(0.01)
                    pending = session.pending
                    if pending is not None and choices_shown(app):
                        # Whatever is highlighted: the first option of the list.
                        await pilot.press("enter")
                        choices += 1
                        await settle(
                            pilot, lambda p=pending: session.pending is not p, 30
                        )
                await settle(pilot, lambda: session.report is not None, 60)
                await pilot.pause(0.2)
                # No debugging controls on a server's game.
                records = len(session.table.log)
                for key in ("u", "g", "d", "n"):
                    await pilot.press(key)
                await pilot.pause()
                assert len(session.table.log) == records
                assert app.focused is not app.command_input
                await pilot.press("colon")
                app.command_input.value = "dice 6 6"
                await pilot.press("enter")
                await pilot.pause()
                assert "unknown command" in app.messages[-1]
                return server, session, app.screen_text(), choices

    server, session, screen, choices = run(scenario(), timeout=300)
    assert choices > 4
    assert session.result.color == "RED"
    room = server.rooms[session.client.room_id]
    winner = room.game.winning_color()
    assert session.result.winning_color == winner.value
    assert session.report.ok, session.report.errors
    assert "verified" in screen and f"{winner.value} wins!" in screen
    path = session.transcript_path
    assert path and os.path.dirname(path) == str(tmp_path)
    with open(path) as file:
        assert verify_transcript(json.load(file)).ok


def test_spectate_a_room_of_server_bots():
    async def scenario():
        async with serving() as (server, base):
            room = await create_room(base, {**FAST, "num_players": 3}, ["F", "R", "F"])
            url = f"{base}/room/{room['id']}"
            session = RemoteSession(url, spectate=True, transcript_dir=None)
            app = CatanApp(session)
            async with app.run_test(size=SIZE) as pilot:
                seen_frames = 0
                while not session.done.is_set():
                    await pilot.pause(0.05)
                    frame = app.frame
                    if frame.document is not None:
                        seen_frames += 1
                        assert frame.viewer is None and not frame.actions
                        state = frame.document["player_state"]
                        assert not any(k.endswith("_WOOD_IN_HAND") for k in state)
                await settle(pilot, lambda: session.report is not None, 60)
                return session, seen_frames

    session, seen_frames = run(scenario(), timeout=300)
    assert seen_frames > 0
    assert session.result.color is None
    assert session.report.ok, session.report.errors


def test_join_an_open_seat_by_url_while_a_bot_client_waits():
    """Two clients of one room: the app takes seat 1 while a remote bot sits at 0."""
    from tests.server.helpers import bot_client

    async def scenario():
        async with serving() as (server, base):
            room = await create_room(base, FAST, [None, None])
            url = f"{base}/room/{room['id']}"
            bot, decide = bot_client(url, "F", seat=0)
            playing = asyncio.create_task(bot.play(decide))
            session = RemoteSession(url, seat=1, transcript_dir=None)
            app = CatanApp(session)
            async with app.run_test(size=SIZE) as pilot:
                await settle(pilot, lambda: session.room is not None)
                while not session.done.is_set():
                    await pilot.pause(0.01)
                    pending = session.pending
                    if pending is not None and choices_shown(app):
                        await pilot.press("enter")
                        await settle(
                            pilot, lambda p=pending: session.pending is not p, 30
                        )
                await settle(pilot, lambda: session.report is not None, 60)
            return session, await playing

    session, bot_result = run(scenario(), timeout=300)
    assert session.result.color == "BLUE"
    assert session.report.ok and bot_result.verified


def test_remote_command_line_options(tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        [
            "--connect",
            "ws://localhost:1/",
            "--new",
            "--seats",
            "F,human,open,AB:2",
            "--vps",
            "8",
            "--transcripts",
            str(tmp_path),
        ]
    )
    session = remote_session(args)
    assert session.create == {"vps_to_win": 8, "num_players": 4}
    assert session.create_seats == ["F", None, None, "AB:2"]
    assert session.seat == 1
    for bad in (["--seats", "human,human"], ["--seats", "F,F"]):
        with pytest.raises(ValueError):
            remote_session(parser.parse_args(["--connect", "ws://x:1", "--new", *bad]))
    with pytest.raises(ValueError):
        RemoteSession("ws://localhost:1")  # neither a room nor --new
    assert "no rooms" in describe_rooms([])
    text = describe_rooms(
        [
            {
                "id": "abc",
                "status": "waiting",
                "name": "x",
                "seats": [
                    {"color": "RED", "kind": "open"},
                    {"color": "BLUE", "kind": "bot", "spec": "F"},
                ],
            }
        ]
    )
    assert "abc" in text and "RED:open" in text and "BLUE:F" in text
