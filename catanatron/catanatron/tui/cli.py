"""``catanatron-tui``: play, watch and debug Catanatron games in the terminal.

Local games (stock Catanatron in this process):

    catanatron-tui                                   # you (RED) against three F bots
    catanatron-tui --seats human,AB:2,F,R --vps 8
    catanatron-tui --seats human,human,F             # hot seat
    catanatron-tui --seats F,F,F,F                   # watch bots (god view)
    catanatron-tui --map tests/maps/extension.json --seats human,F,F,F,F,F
    catanatron-tui --load game.json                  # resume a saved game

Remote games (a catanatron-server room):

    catanatron-tui --connect ws://localhost:8765/room/ROOM_ID
    catanatron-tui --connect ws://localhost:8765/room/ROOM_ID --spectate
    catanatron-tui --connect ws://localhost:8765 --new --seats human,F,open,AB:2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import List, Optional, Sequence

from catanatron.models.player import Color

DEFAULT_TRANSCRIPTS = "transcripts"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="catanatron-tui",
        description="Play, watch and debug Catanatron games in the terminal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n\n".join(__doc__.split("\n\n")[1:]),
    )
    seats = parser.add_argument_group("seats")
    seats.add_argument(
        "--seats",
        help="comma-separated seat specs in color order (RED, BLUE, ORANGE, WHITE, "
        "GREEN, BROWN): human, a registry bot (F, AB:2, R, ...), oracle:SPEC (the "
        "bot sees every hand). Remote with --new: human is your seat, open an open "
        "seat, a spec a server bot. Default: human,F,F,F",
    )
    seats.add_argument(
        "--seat",
        action="append",
        default=[],
        metavar="COLOR=SPEC",
        help="set one seat (local; repeatable), e.g. --seat BLUE=AB:3",
    )
    seats.add_argument(
        "--bot",
        action="append",
        default=[],
        metavar="[NAME=]SOURCE",
        help="register a bot class for seat specs, as in catanatron-play",
    )

    rules = parser.add_argument_group("rules (new games)")
    rules.add_argument("--map", metavar="FILE", help="explicit map spec (JSON tiles)")
    rules.add_argument("--map-type", choices=("BASE", "MINI", "TOURNAMENT"))
    rules.add_argument("--number-placement", choices=("official_spiral", "random"))
    rules.add_argument("--vps", type=int, metavar="N", help="points to win (10)")
    rules.add_argument("--discard-limit", type=int, metavar="N", help="(7)")
    rules.add_argument("--friendly-robber", action="store_true", default=None)
    rules.add_argument("--supply", choices=("base", "extension"))
    rules.add_argument(
        "--config", metavar="JSON", help="rules as a catanatron-server room config"
    )
    rules.add_argument("--seed", type=int, help="board and dice seed (local)")

    local = parser.add_argument_group("local games")
    local.add_argument(
        "--load", metavar="FILE", help="start from a saved game or position"
    )
    local.add_argument(
        "--speed",
        type=float,
        default=0.3,
        metavar="SECONDS",
        help="delay between bot moves (default %(default)s)",
    )
    local.add_argument(
        "--paused", action="store_true", help="start with the bots paused"
    )
    local.add_argument(
        "--ask-forced",
        action="store_true",
        help="ask a person even when they have only one legal move",
    )
    local.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="COMMAND",
        help="run a command at start (repeatable), e.g. --run 'dice 6 6'",
    )

    remote = parser.add_argument_group("remote games")
    remote.add_argument(
        "--connect",
        metavar="URL",
        help="catanatron-server room (or server, with --new)",
    )
    remote.add_argument(
        "--new", action="store_true", help="create a room with --seats and the rules"
    )
    remote.add_argument(
        "--spectate", action="store_true", help="watch instead of playing"
    )
    remote.add_argument(
        "--sit",
        metavar="SEAT",
        help="the seat to take: index or color (default: first open)",
    )
    remote.add_argument(
        "--name", default="catanatron-tui", help="your name at the table"
    )
    remote.add_argument(
        "--transcripts", default=DEFAULT_TRANSCRIPTS, help="where to save transcripts"
    )
    remote.add_argument(
        "--no-verify", action="store_true", help="do not verify at the end"
    )
    remote.add_argument("--server-key", help="pin the server's public key (hex)")
    remote.add_argument("--key", help="file with your Ed25519 key (created if missing)")

    parser.add_argument(
        "--board-size",
        choices=("auto", "small", "medium", "large"),
        default="auto",
        help="hex size (default: the largest that fits)",
    )
    return parser


def rules_config(args) -> dict:
    """The room-config dict the rule options describe."""
    config = json.loads(args.config) if args.config else {}
    if not isinstance(config, dict):
        raise ValueError("--config must be a JSON object")
    if args.map:
        with open(args.map) as file:
            spec = json.load(file)
        config["map"] = {"tiles": spec["tiles"]}
    for option, key in (
        ("map_type", "map_type"),
        ("number_placement", "number_placement"),
        ("vps", "vps_to_win"),
        ("discard_limit", "discard_limit"),
        ("friendly_robber", "friendly_robber"),
        ("supply", "supply"),
    ):
        value = getattr(args, option)
        if value is not None:
            config[key] = value
    return config


def split_seats(text: Optional[str]) -> Optional[List[str]]:
    if not text:
        return None
    return [part.strip() for part in text.split(",") if part.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        from catanatron.tui.app import CatanApp
    except ImportError as error:  # pragma: no cover - depends on the install
        parser.exit(
            2,
            f"catanatron-tui needs Textual: pip install 'catanatron[tui]' ({error})\n",
        )

    from catanatron.registry import REGISTRY, SpecError

    from catanatron.players import register_builtins

    register_builtins()
    for declaration in args.bot:
        name, _, source = declaration.rpartition("=")
        try:
            REGISTRY.register_source(source, name or None)
        except SpecError as error:
            parser.error(str(error))

    try:
        if args.connect:
            session = remote_session(args)
        else:
            session = local_session(args)
    except (ValueError, OSError, SpecError, KeyError) as error:
        parser.exit(2, f"catanatron-tui: {error}\n")

    app = CatanApp(
        session,
        board_size=args.board_size,
        speed=args.speed,
        autoplay=not args.paused,
        commands=tuple(args.run),
    )
    app.run()
    return 0


def local_session(args):
    from catanatron.tui.local import LocalSession
    from catanatron.tui.positions import GameSettings

    seats = split_seats(args.seats)
    if args.load:
        session = LocalSession.load(
            args.load, seats, seed=args.seed, autoplay_forced=not args.ask_forced
        )
    else:
        config = rules_config(args)
        if seats is not None:
            config.setdefault("num_players", len(seats))
        settings = GameSettings.from_json(config)
        session = LocalSession.new(
            seats,
            settings=settings,
            seed=args.seed,
            autoplay_forced=not args.ask_forced,
        )
    for item in args.seat:
        color, _, spec = item.partition("=")
        if not spec:
            raise ValueError(f"--seat expects COLOR=SPEC, got {item!r}")
        session.set_seat(session.color(color), spec)
    return session


def remote_session(args):
    from catanatron.tui.remote import RemoteSession, describe_rooms, is_room_url

    url = args.connect
    if not args.new and not is_room_url(url):
        from catanatron.server.client import list_rooms

        rooms = asyncio.run(list_rooms(url))
        raise ValueError(
            "pick a room (--connect ws://HOST:PORT/room/ID) or create one with --new:\n"
            + describe_rooms(rooms)
        )
    create, create_seats, seat = None, None, _seat_index(args.sit)
    if args.new:
        create = rules_config(args)
        specs = split_seats(args.seats) or ["human", "F", "F", "F"]
        create.setdefault("num_players", len(specs))
        if len(specs) != create["num_players"]:
            raise ValueError(f"{len(specs)} seats for {create['num_players']} players")
        mine = [i for i, s in enumerate(specs) if s.lower() in ("human", "me", "you")]
        if len(mine) > 1:
            raise ValueError("one human seat per terminal in a remote game")
        if not mine and not args.spectate:
            raise ValueError("mark your seat as human (or use --spectate)")
        create_seats = [
            None if s.lower() in ("human", "me", "you", "open") else s for s in specs
        ]
        if mine:
            seat = mine[0]
    return RemoteSession(
        url,
        seat=seat,
        spectate=args.spectate,
        name=args.name,
        create=create,
        create_seats=create_seats,
        transcript_dir=args.transcripts,
        verify=not args.no_verify,
        server_key=args.server_key,
        key_path=args.key,
    )


def _seat_index(text: Optional[str]) -> Optional[int]:
    if text is None:
        return None
    if text.isdigit():
        return int(text)
    try:
        return list(Color).index(Color[text.upper()])
    except KeyError:
        raise ValueError(
            f"--sit expects a seat index or a color, got {text!r}"
        ) from None


if __name__ == "__main__":
    sys.exit(main())
