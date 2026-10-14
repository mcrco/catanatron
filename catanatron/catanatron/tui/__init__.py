"""A terminal UI for playing, watching and debugging Catanatron games.

``catanatron-tui`` (or ``python -m catanatron.tui``) plays local games (stock
Catanatron in this process, against registry bots or your own) and remote
ones (a seat in a catanatron-server room). See README.md next to this file.

From Python, :func:`run` opens a local game; ``decide_fns`` plugs in bots
that are not in the registry::

    from catanatron.tui import run

    def my_bot(view, playable_actions):
        # view: this seat's client_view (its own hand, others' counts, the map,
        # the log); playable_actions: [color, action_type, value] triples.
        return playable_actions[0]

    run(["human", "F", "mine"], decide_fns={"mine": my_bot})

A ``decide_fns`` entry is either such a callback (the one
``catanatron.server.client.GameClient.play`` takes, so a server bot plays
here unchanged) or a Catanatron ``Player`` (it then decides on a game sampled
from its view). ``oracle:NAME`` gives it every hand instead.

The package needs Textual for the app (the ``tui`` extra) and
``websockets``/``cryptography`` for remote games (the ``server`` extra);
:mod:`~catanatron.tui.session`, :mod:`~catanatron.tui.local` and
:mod:`~catanatron.tui.board` work without them.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union

__all__ = ["run", "connect"]


def run(
    seats: Optional[Sequence[str]] = None,
    *,
    decide_fns: Optional[Mapping[str, Any]] = None,
    config: Union[dict, Any, None] = None,
    seed: Optional[int] = None,
    load: Optional[str] = None,
    board_size: str = "auto",
    speed: float = 0.3,
    autoplay: bool = True,
    commands: Sequence[str] = (),
):
    """Open a local game in the terminal; returns its session when the app exits.

    ``seats`` lists one spec per seat (RED, BLUE, ORANGE, WHITE, GREEN, BROWN
    in that order): ``human``, a registry spec (``F``, ``AB:2``...),
    ``oracle:SPEC`` or a ``decide_fns`` name. ``config`` holds the rules (a
    dict with catanatron-server's room config keys, or a ``GameSettings``);
    ``load`` resumes a saved game instead. ``commands`` run at start, as if
    typed (e.g. ``"dice 6 6"``).
    """
    from catanatron.tui.app import CatanApp
    from catanatron.tui.local import LocalSession
    from catanatron.tui.positions import GameSettings

    if load is not None:
        session = LocalSession.load(load, seats, decide_fns=decide_fns, seed=seed)
    else:
        if config is None or isinstance(config, dict):
            payload = dict(config or {})
            if seats is not None:
                payload.setdefault("num_players", len(seats))
            config = GameSettings.from_json(payload)
        session = LocalSession.new(
            seats, settings=config, seed=seed, decide_fns=decide_fns
        )
    CatanApp(
        session,
        board_size=board_size,
        speed=speed,
        autoplay=autoplay,
        commands=tuple(commands),
    ).run()
    return session


def connect(url: str, **kwargs):
    """Play (or, with ``spectate=True``, watch) a catanatron-server room in
    the terminal. ``kwargs`` go to :class:`~catanatron.tui.remote.RemoteSession`;
    returns it when the app exits (``.result``, ``.report``)."""
    from catanatron.tui.app import CatanApp
    from catanatron.tui.remote import RemoteSession

    board_size = kwargs.pop("board_size", "auto")
    session = RemoteSession(url, **kwargs)
    CatanApp(session, board_size=board_size).run()
    return session
