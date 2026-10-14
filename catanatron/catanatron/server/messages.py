"""The bodies of the server's messages.

The server builds every game-message body with these functions, and the
verifier rebuilds them from its own replay and compares: so whatever a body
contains is exactly what an honest server would have sent at that point, and
nothing more. Only fields that cannot be derived from the game (who sits
where, their keys and seeds) are taken from the commitments the client saw
before it revealed its seed.

The vocabulary is :mod:`catanatron.protocol`'s, extended for a networked
table:

    hello        server -> client, first message on every connection (with
                 a nonce a rejoining client signs)
    joined       the seat this connection now holds (and the room)
    room         seats changed before the game started
    commitments  every seat is taken: the locked seats with their keys,
                 seed commitments and join signatures, and the config; seats
                 now reveal their seeds (signing the commitments' digest).
                 Sent once per room: there is no second round
    before       the game starts (or a spectator arrives): board, seats, seeds
    decide       the seat must answer with a signed action
    received     the server's receipt for a submitted reveal or action
    step         somebody acted; the new state and that action's log entry
    after        the game is over, with the reveal: server seed, salts, actions
    aborted      the game is void (why, who is to blame, the server seed)
    error        a request was refused

See PROTOCOL.md next to this module for the state machine and invariants.

Every game message (``before``, ``decide``, ``step``, ``after``) carries
``state``: the recipient's view of the game (``client_view`` for a seat,
``public_view`` for a spectator) without the static ``map`` (sent once, in
``before``) and without the action log, which comes incrementally: ``log``
holds the log entries from index ``log_start`` on that the recipient has not
been sent yet, already redacted for it.
"""

from __future__ import annotations

from typing import Optional

from catanatron.game import Game
from catanatron.serialization import action_to_json, client_view, public_view
from catanatron.server.maps import board_geometry

#: Bumped when a change would break clients written against the old shape.
#: 2: seeds are committed, then revealed; statements carry the action's index.
#: 3: statements and reveals sign the commitments' digest; restart rounds;
#: receipts for submitted actions.
#: 4: one commitments per room (a withheld reveal voids the room; no restart
#: rounds); rejoining signs the connection's nonce (no seat keys); receipts
#: for reveals too.
#: 5: the commitments name the server's key (so C does); a join signs the
#: server's key and its connection's nonce; a void names its game.
PROTOCOL_VERSION = 5

#: Every field of a commitments body, and nothing else.
COMMITMENTS_FIELDS = (
    "type",
    "room_id",
    "server_key",
    "config",
    "seats",
    "seed_commitment",
    "reveal_timeout_ms",
)

#: Every field of a seat in the commitments.
SEAT_FIELDS = (
    "seat",
    "color",
    "kind",
    "name",
    "spec",
    "public_key",
    "requested_seat",
    "client_commitment",
    "nonce",
    "pop",
)

GAME_MESSAGES = ("before", "decide", "step", "after")


def view_for(doc: dict, color: Optional[str]) -> dict:
    """What ``color`` (a spectator, if None) may see of the document."""
    return client_view(doc, color) if color is not None else public_view(doc)


def _split(view: dict):
    state = {k: v for k, v in view.items() if k not in ("map", "action_records")}
    return state, view["action_records"]


def _status(game: Game) -> dict:
    winner = game.winning_color()
    return {
        "current_color": game.state.current_color().value,
        "winning_color": winner.value if winner else None,
    }


def playable_json(game: Game) -> list:
    """The legal actions, in Catanatron's order (the same in every process)."""
    return [action_to_json(a) for a in game.playable_actions]


def commitments_body(
    room_id: str,
    *,
    server_key: str,
    config: dict,
    seats: list,
    seed_commitment: str,
    reveal_timeout_ms: int,
) -> dict:
    return {
        "type": "commitments",
        "room_id": room_id,
        "server_key": server_key,
        "config": config,
        "seats": seats,
        "seed_commitment": seed_commitment,
        "reveal_timeout_ms": reveal_timeout_ms,
    }


def receipt_body(
    submission: str, index: Optional[int], accepted: bool, reason: str
) -> dict:
    return {
        "type": "received",
        "submission": submission,
        "index": index,
        "accepted": accepted,
        "reason": reason,
    }


def before_body(
    game: Game,
    doc: dict,
    color: Optional[str],
    *,
    commitments: dict,
    client_seeds: list,
    reveals: list,
) -> dict:
    view = view_for(doc, color)
    state, records = _split(view)
    return {
        "type": "before",
        "protocol_version": PROTOCOL_VERSION,
        "room_id": commitments["room_id"],
        "game_id": game.id,
        "color": color,
        "commitments": commitments,
        "config": commitments["config"],
        "seats": commitments["seats"],
        "client_seeds": client_seeds,
        "reveals": reveals,
        "seed_commitment": commitments["seed_commitment"],
        "map": view["map"],
        "board": board_geometry(game.state.board.map),
        "index": len(records),
        "state": state,
        "log_start": 0,
        "log": records,
        **_status(game),
    }


def decide_body(game: Game, doc: dict, color: str, *, timeout_ms: int) -> dict:
    state, records = _split(view_for(doc, color))
    return {
        "type": "decide",
        "game_id": game.id,
        "color": color,
        "index": len(records),
        "state": state,
        "log_start": len(records),
        "log": [],
        "playable_actions": playable_json(game),
        "timeout_ms": timeout_ms,
        **_status(game),
    }


def step_body(
    game: Game, doc: dict, color: Optional[str], *, forced: Optional[str]
) -> dict:
    state, records = _split(view_for(doc, color))
    return {
        "type": "step",
        "game_id": game.id,
        "index": len(records),
        "state": state,
        "log_start": len(records) - 1,
        "log": records[-1:],
        "forced": forced,
        **_status(game),
    }


def after_body(game: Game, doc: dict, color: Optional[str], *, reveal: dict) -> dict:
    state, records = _split(view_for(doc, color))
    return {
        "type": "after",
        "game_id": game.id,
        "index": len(records),
        "state": state,
        "log_start": len(records),
        "log": [],
        "num_turns": game.state.num_turns,
        "reveal": reveal,
        **_status(game),
    }


def aborted_body(
    room_id: str,
    *,
    game_id: Optional[str],
    reason: str,
    blame: list,
    server_seed: Optional[str],
) -> dict:
    """``game_id`` is the game's if it had started (None: void before it)."""
    return {
        "type": "aborted",
        "room_id": room_id,
        "game_id": game_id,
        "reason": reason,
        "blame": blame,
        "server_seed": server_seed,
    }
