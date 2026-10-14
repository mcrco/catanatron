"""Registry bots that play from a view: on the server, or as a remote client.

:class:`ViewBot` wraps any Catanatron registry player (``F``, ``AB:2``,
``--bot`` imports...). It never sees the real game: each decision is made on
a game sampled from the seat's view (:mod:`catanatron.server.sampling`), and
only among the actions the seat was offered. The server seats its own bots
this way (in worker processes, :mod:`catanatron.server.botproc`), and
``catanatron-play-bot`` runs one as a remote client::

    catanatron-play-bot --spec AB:2 ws://localhost:8765/room/ROOM_ID
    catanatron-play-bot --spec F --new --players 4 --fill R ws://localhost:8765
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import random
import sys
import time
from collections import deque
from typing import Optional, Sequence

from catanatron.models.decks import Supply
from catanatron.models.player import Color
from catanatron.registry import REGISTRY, SpecError
from catanatron.observer import GameObserver
from catanatron.serialization import action_from_json, action_to_json
from catanatron.server.botproc import pace
from catanatron.server.sampling import sample_game

logger = logging.getLogger(__name__)

#: The most a bot seated by the server may spend, by param name: a room's
#: creator picks its bots, and one room must not starve the others.
SERVER_PARAM_LIMITS = {"depth": 3, "num_simulations": 200, "num_playouts": 100}


def _registry():
    from catanatron.players import register_builtins

    register_builtins()
    return REGISTRY


class ViewBot:
    """A registry player deciding on games sampled from a seat's view.

    A seat with only one legal move answers it as slowly as it answers real
    decisions (a duration drawn from its own past ones): otherwise the speed
    of the answer would tell the table that the seat's options were few,
    which can depend on its hidden cards (a roll from a hand of victory point
    cards only, a discard from a one-resource hand).
    """

    #: How many past decision durations a bot draws its pace from.
    PACE_HISTORY = 64

    def __init__(self, spec: str, color: Color, seed: Optional[int] = None):
        self.spec = spec
        self.color = color
        self.player = _registry().build(spec, color)
        self.rng = random.Random(seed)
        self.pace_rng = random.Random()
        self.timings: deque = deque(maxlen=self.PACE_HISTORY)
        self.started = False

    def start(self, view: dict, supply: Supply) -> None:
        """Exec bots learn the board from before(); the sampled world shows
        exactly what the seat may see."""
        if not self.started:
            self.started = True
            self.player.before(sample_game(view, self.color, supply, self.rng))

    def decide(self, view: dict, playable_actions: list, supply: Supply) -> list:
        """One of ``playable_actions`` (as JSON) for ``view``."""
        if len(playable_actions) == 1:
            pace(self.timings, self.pace_rng)
            return playable_actions[0]
        began = time.monotonic()
        self.start(view, supply)
        game = sample_game(view, self.color, supply, self.rng)
        offered = {json.dumps(a): a for a in playable_actions}
        options = [
            action
            for action in game.playable_actions
            if json.dumps(action_to_json(action)) in offered
        ]
        if not options:
            logger.warning("%s: no offered action in the sampled world", self.spec)
            return playable_actions[0]
        choice = action_to_json(self.player.decide(game, options))
        self.timings.append(time.monotonic() - began)
        if json.dumps(choice) not in offered:
            logger.warning("%s chose %s, which was not offered", self.spec, choice)
            return playable_actions[0]
        return offered[json.dumps(choice)]

    def observe(self, view: dict, action: list, supply: Supply) -> None:
        """Someone acted (``view`` is from before the action): players that
        watch the game (exec bots that asked to observe) get step()."""
        if type(self.player).step is GameObserver.step:
            return
        self.start(view, supply)
        if not getattr(self.player, "_wants_step", True):
            return
        game = sample_game(view, self.color, supply, self.rng)
        self.player.step(game, action_from_json(action))

    def finish(self, view: dict, supply: Supply, winning_color: Optional[str]) -> None:
        """The game is over: tell the player (exec bots get ``after``) and
        let it go (exec bots' programs are stopped)."""
        try:
            if self.started:
                game = sample_game(view, self.color, supply, self.rng)
                winner = Color[winning_color] if winning_color else None
                game.winning_color = lambda: winner  # the view's, not the sample's
                self.player.after(game)
        finally:
            self.close()

    def close(self) -> None:
        close = getattr(self.player, "close", None)
        if callable(close):
            close()


def check_spec(spec: str, limits: Optional[dict] = None) -> None:
    """Raise SpecError unless ``spec`` builds a bot within ``limits``."""
    player = _registry().build(spec, Color.RED)
    if not player.is_bot:
        raise SpecError(f"{spec!r} is not a bot")
    params = player.params
    if limits and dataclasses.is_dataclass(params):
        for name, limit in limits.items():
            value = getattr(params, name, None)
            if type(value) is int and value > limit:
                raise SpecError(f"{spec!r}: {name} may be at most {limit} here")


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
        for bot in bots.values():
            table = client.table
            bot.finish(table.view(), table.supply, table.winning_color)
        print(json.dumps(result.summary(), indent=2))
        return 0 if result.verified in (True, None) else 1

    return asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())
