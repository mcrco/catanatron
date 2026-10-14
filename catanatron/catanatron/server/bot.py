"""Registry bots that play from a view: on the server, or as a remote client.

:class:`ViewBot` wraps any Catanatron registry player (``F``, ``AB:2``,
``--bot`` imports...). It never sees the real game: each decision is made on
a game sampled from the seat's view (:mod:`catanatron.server.sampling`).
The server seats its own bots this way, and ``catanatron-play-bot`` runs one
as a remote client::

    catanatron-play-bot --spec AB:2 ws://localhost:8765/room/ROOM_ID
    catanatron-play-bot --spec F --new --players 4 --fill R ws://localhost:8765
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
from typing import Optional, Sequence

from catanatron.models.decks import Supply
from catanatron.models.player import Color
from catanatron.registry import REGISTRY, SpecError
from catanatron.serialization import action_to_json
from catanatron.server.sampling import sample_game

logger = logging.getLogger(__name__)


def _registry():
    from catanatron.players import register_builtins

    register_builtins()
    return REGISTRY


class ViewBot:
    """A registry player deciding on games sampled from a seat's view."""

    def __init__(self, spec: str, color: Color, seed: Optional[int] = None):
        self.spec = spec
        self.color = color
        self.player = _registry().build(spec, color)
        self.rng = random.Random(seed)

    def decide(self, view: dict, playable_actions: list, supply: Supply) -> list:
        """One of ``playable_actions`` (as JSON) for ``view``."""
        if len(playable_actions) == 1:
            return playable_actions[0]
        game = sample_game(view, self.color, supply, self.rng)
        action = action_to_json(self.player.decide(game, game.playable_actions))
        if action not in playable_actions:
            # The sampled world differs from the real one only in hidden
            # information, which legality never depends on; but be safe.
            logger.warning(
                "%s chose %s, which is not legal; playing the first", self.spec, action
            )
            return playable_actions[0]
        return action


def check_spec(spec: str) -> None:
    """Raise SpecError unless ``spec`` builds a bot."""
    player = _registry().build(spec, Color.RED)
    if not player.is_bot:
        raise SpecError(f"{spec!r} is not a bot")


# ===== catanatron-play-bot =====
def main(argv: Optional[Sequence[str]] = None) -> int:
    from catanatron.server.client import GameClient, create_room, lobby_url

    parser = argparse.ArgumentParser(
        prog="catanatron-play-bot",
        description="Play a registry bot in a catanatron-server room, from its view only.",
    )
    parser.add_argument(
        "url", help="ws://HOST:PORT/room/ROOM_ID, or ws://HOST:PORT with --new"
    )
    parser.add_argument("--spec", default="F", help="registry spec, e.g. F, AB:2, R")
    parser.add_argument("--name", help="name shown at the table (default: the spec)")
    parser.add_argument(
        "--seat", type=int, help="seat to take (default: the first open one)"
    )
    parser.add_argument(
        "--key", help="file with this client's Ed25519 key (created if missing)"
    )
    parser.add_argument(
        "--transcripts", default="transcripts", help="where to save transcripts"
    )
    parser.add_argument(
        "--no-verify", action="store_true", help="skip verifying the transcript"
    )
    parser.add_argument(
        "--new", action="store_true", help="create a room first, then join it"
    )
    parser.add_argument(
        "--players", type=int, default=4, help="with --new: number of seats"
    )
    parser.add_argument(
        "--fill",
        default=None,
        help="with --new: registry spec for the server to seat in every other seat",
    )
    parser.add_argument(
        "--config", default="{}", help="with --new: game config as JSON"
    )
    parser.add_argument(
        "--bot",
        action="append",
        default=[],
        metavar="[NAME=]SOURCE",
        help="register a bot class, as in catanatron-play",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    registry = _registry()
    for declaration in args.bot:
        name, _, source = declaration.rpartition("=")
        registry.register_source(source, name or None)
    try:
        check_spec(args.spec)
    except SpecError as error:
        parser.error(str(error))

    async def run() -> int:
        url = args.url
        if args.new:
            config = {"num_players": args.players, **json.loads(args.config)}
            seats = [None] + [args.fill] * (config["num_players"] - 1)
            room = await create_room(lobby_url(url), config, seats)
            url = f"{url.rstrip('/')}/room/{room['id']}"
            print(f"created room {room['id']}: {url}", file=sys.stderr)

        bots: dict = {}

        def decide(view, playable_actions):
            table = client.table
            bot = bots.get(table.color)
            if bot is None:
                bot = bots[table.color] = ViewBot(args.spec, Color[table.color])
            return bot.decide(view, playable_actions, table.supply)

        client = GameClient(
            url,
            name=args.name or args.spec,
            seat=args.seat,
            key_path=args.key,
            transcript_dir=args.transcripts,
            verify=not args.no_verify,
        )
        result = await client.play(decide)
        print(json.dumps(result.summary(), indent=2))
        return 0 if result.verified in (True, None) else 1

    return asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())
