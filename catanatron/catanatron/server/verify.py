"""Check a game's transcript by replaying it in stock Catanatron.

A transcript is what one client received (every signed message, in order)
plus what it says of itself (its public key, its seed, what it signed).
:func:`verify_transcript` checks the invariants of PROTOCOL.md (next to this
module) that one recipient's transcript can show, over the whole chain:

1. every message is signed by the server's key, numbered in order and
   chained by hash; the header's claims (key, seed, room) agree with what
   the server signed (``joined``), or the transcript is not this client's;
2. the schedule: one hello first; the room, the join and one
   ``commitments`` before the game; one ``before``; decides and steps after
   it; one ``after`` or one ``aborted`` to end it, and nothing but receipts
   and errors after that; no unknown message types;
3. the seeds and the binding: the seed commitment is the same everywhere it
   appears; the commitments name the room, the server's key, the hello's
   rules and every seat (its key, seed commitment, the seat it asked for,
   and a pop over the nonce of the connection it joined on); ``before``
   carries exactly those commitments, every revealed seed opens its
   commitment and every reveal signs their digest;
4. replaying the revealed actions from the master seed in stock Catanatron,
   every state commitment opens to the replayed state and every message body
   is *exactly* what an honest server builds for this recipient; a game
   called off after its start is checked as far as the void's revealed seed
   allows (its start);
5. this client's moves, from the signed decides, receipts and steps alone
   (so also in a void or unfinished game): no legal signed answer the server
   acknowledged in time (or as accepted) was replaced, no action was
   answered with an error, no move was timed out without being asked.

The report sorts what it finds (see :class:`Report`): ``errors`` are what
the server's own signatures prove against it, to anyone; ``invalid`` says
why the transcript is not evidence at all; ``warnings`` are what only this
client knows; ``void`` and ``incomplete`` are verdicts, not accusations.

``catanatron-verify T1.json [T2.json ...]`` runs it from the shell; given
several transcripts of one room under one server key it also checks every
pair of them against each other: the same seed commitment, commitments,
game, seeds, reveal or void, and the same state commitment at each point
(no equivocation).
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from catanatron.serialization import action_to_json, state_to_json
from catanatron.server.config import (
    ConfigError,
    GameConfig,
    default_action,
    is_over,
    new_game,
    publicly_forced,
)
from catanatron.server.hashing import numbers_ok
from catanatron.server.crypto import (
    GENESIS,
    action_statement,
    canonical,
    client_seed_commitment,
    digest,
    envelope_hash,
    is_hex,
    join_statement,
    master_seed,
    reveal_statement,
    seed_commitment,
    state_commitment,
    submission_digest,
    verify_envelope,
    verify_signature,
)
from catanatron.server.messages import (
    COMMITMENTS_FIELDS,
    PROTOCOL_VERSION,
    SEAT_FIELDS,
    after_body,
    before_body,
    decide_body,
    step_body,
)

MAX_ERRORS = 50
FORCED = (None, "only", "timeout", "bot_error")

#: Message types, and where on a chain each may come.
PRE_GAME = ("room", "joined", "commitments")
IN_GAME = ("decide", "step")
ANSWERS = ("received", "error")
KNOWN = ("hello", "before", "after", "aborted") + PRE_GAME + IN_GAME + ANSWERS


@dataclass
class Report:
    """What one transcript shows.

    ``errors``: what the server's own signatures prove against it, to
    anyone. ``invalid``: why the transcript is not evidence at all (a broken
    signature or chain, a header that does not match the chain, a malformed
    file): anyone can write one. ``warnings``: what only this client knows
    (an answer the server never acknowledged, a deadline). ``void``: the
    room was called off (its ``aborted`` body). ``incomplete``: the
    transcript stops before the game did; what it holds is still checked.
    """

    errors: list = field(default_factory=list)
    invalid: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    counts: dict = field(default_factory=dict)
    game_id: Optional[str] = None
    color: Optional[str] = None
    #: State commitments by index, as the envelopes carry them.
    commitments: dict = field(default_factory=dict)
    #: What every recipient of this room must have been told alike, by name.
    shared: Optional[dict] = None
    void: Optional[dict] = None
    incomplete: Optional[str] = None

    @property
    def ok(self) -> bool:
        """A finished game, every check passed."""
        return (
            not self.errors
            and not self.invalid
            and self.void is None
            and self.incomplete is None
        )

    @property
    def verdict(self) -> str:
        if self.errors:
            return "FAILED"
        if self.invalid:
            return "INVALID"
        if self.void is not None:
            return "VOID"
        if self.incomplete is not None:
            return "INCOMPLETE"
        return "OK"

    def fail(self, message: str) -> None:
        if len(self.errors) < MAX_ERRORS:
            self.errors.append(message)

    def reject(self, message: str) -> None:
        if len(self.invalid) < MAX_ERRORS:
            self.invalid.append(message)

    def warn(self, message: str) -> None:
        if len(self.warnings) < MAX_ERRORS:
            self.warnings.append(message)

    def count(self, what: str) -> None:
        self.counts[what] = self.counts.get(what, 0) + 1

    def summary(self) -> dict:
        return {
            "ok": self.ok,
            "verdict": self.verdict,
            "void": self.void,
            "incomplete": self.incomplete,
            "game_id": self.game_id,
            "color": self.color,
            "errors": self.errors,
            "invalid": self.invalid,
            "warnings": self.warnings,
            "checked": self.counts,
        }


class _Abort(Exception):
    pass


def _plain(obj: Any) -> Any:
    """JSON's view of a value (tuples become lists), for comparisons."""
    return json.loads(json.dumps(obj))


def _same(a: Any, b: Any) -> bool:
    """Equal as JSON documents, types included (15.0 is not 15)."""
    try:
        return canonical(a) == canonical(b)
    except (TypeError, ValueError):
        return False


#: Moves whose log entry fills in what chance decided (the dice, the card
#: drawn): the move that was asked for has no value.
OUTCOME_IN_LOG = ("ROLL", "BUY_DEVELOPMENT_CARD")


def _same_move(asked: Any, logged: Any) -> bool:
    """Whether a step's log entry plays the move ``asked`` for."""
    if _same(asked, logged):
        return True
    return (
        isinstance(asked, list)
        and isinstance(logged, list)
        and len(asked) == len(logged) == 3
        and asked[1] in OUTCOME_IN_LOG
        and asked[2] is None
        and asked[:2] == logged[:2]
    )


def load_transcript(path: str) -> dict:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as file:  # type: ignore[operator]
        return json.load(file)


def verify_transcript(transcript: Any) -> Report:
    report = Report()
    try:
        _Verifier(transcript, report).run()
    except _Abort:
        pass
    except RecursionError:
        report.reject("malformed transcript: nested too deeply")
    except Exception as error:  # a malformed transcript must fail, not crash
        report.reject(f"malformed transcript: {type(error).__name__}: {error}")
    return report


class _Verifier:
    def __init__(self, transcript: Any, report: Report):
        self.transcript = transcript
        self.report = report

    def fail(self, message: str) -> None:
        self.report.fail(message)

    def require(self, condition: bool, message: str) -> None:
        """A provable violation after which nothing more can be checked."""
        if not condition:
            self.fail(message)
            raise _Abort()

    def reject(self, message: str) -> None:
        """The transcript is not evidence: stop."""
        self.report.reject(message)
        raise _Abort()

    # ===== 1. the chain, and whose it is =====
    def run(self) -> None:
        t = self.transcript
        if not isinstance(t, dict):
            self.reject("a transcript is a JSON object")
        raw = t.get("messages")
        if not isinstance(raw, list) or not raw:
            self.reject("the transcript has no messages")
        self.messages = [json.loads(m) if isinstance(m, str) else m for m in raw]
        self.server_key = t.get("server_key")
        if not is_hex(self.server_key, 32):
            self.reject("server_key is not an Ed25519 key")
        head = GENESIS
        for position, envelope in enumerate(self.messages):
            if not isinstance(envelope, dict) or not isinstance(
                envelope.get("body"), dict
            ):
                self.reject(f"message {position} is not an envelope")
            if not verify_envelope(self.server_key, envelope):
                self.reject(f"message {position}: bad server signature")
            if envelope.get("seq") != position or envelope.get("prev") != head:
                self.reject(f"message {position}: breaks the hash chain")
            head = envelope_hash(envelope)
            self.report.count("signatures")
        self.bodies = [envelope["body"] for envelope in self.messages]
        for position, body in enumerate(self.bodies):
            if not numbers_ok(self.messages[position]):
                self.fail(
                    f"message {position}: the server signed a number the protocol "
                    "does not allow (a float, or beyond 2**53)"
                )
        self.check_identity()
        self.check_schedule()
        self.check_setup()
        self.check_game()
        self.check_own_moves()
        if self.terminal is None:
            self.report.incomplete = (
                "the transcript ends before the game did: nothing after its last "
                "message is evidence"
            )

    def check_identity(self) -> None:
        """Whose transcript this is, from what the server signed; the
        header (written by the client) must agree, or the transcript is not
        this client's (or was edited): not evidence, never an accusation."""
        t, hello = self.transcript, self.bodies[0]
        if hello.get("type") != "hello" or hello.get("server_key") != self.server_key:
            self.reject("the transcript does not start with the server's hello")
        room = hello.get("room")
        if not isinstance(room, dict) or not isinstance(room.get("id"), str):
            self.reject("the hello describes no room")
        self.room_id = room["id"]
        if t.get("room_id") not in (None, self.room_id):
            self.reject("the transcript's header names another room than its messages")
        joins = [p for p, b in enumerate(self.bodies) if b.get("type") == "joined"]
        self.joined_at = joins[0] if joins else None
        self.joined = self.bodies[joins[0]] if joins else {}
        if len(joins) > 1:
            self.fail(f"message {joins[1]}: this chain was seated twice")
        self.me = t.get("public_key")
        self.seated = self.me is not None
        if not self.seated:
            if joins:
                self.reject("the header says spectator, but the chain holds a seat")
            return
        if self.joined.get("public_key") != self.me:
            self.reject(
                "the header's key is not the one this chain seated: not this "
                "client's transcript, or edited"
            )
        if not is_hex(self.joined.get("client_commitment"), 32) or (
            client_seed_commitment(self.room_id, self.me, t.get("client_seed"))
            != self.joined["client_commitment"]
        ):
            self.reject(
                "the header's client_seed does not open this client's commitment: "
                "not this client's transcript, or edited"
            )
        self.my_seed = t["client_seed"]
        self.my_seat = self.joined.get("seat")
        for position in range(self.joined_at + 1, len(self.messages)):
            if self.messages[position].get("to") != self.me:
                self.fail(
                    f"message {position}: addressed to {self.messages[position].get('to')}, "
                    "not this client"
                )

    # ===== 2. what comes where =====
    def check_schedule(self) -> None:
        """Message types, over the whole chain: one hello first; the room,
        the join and one commitments before the game; one before; decides
        and steps after it; one after or one aborted to end it, and nothing
        but receipts and errors after that."""
        positions: dict = {}
        for position, body in enumerate(self.bodies):
            positions.setdefault(body.get("type"), []).append(position)
        end = len(self.bodies)
        self.before_at = positions.get("before", [None])[0]
        after_at = positions.get("after", [None])[0]
        aborted_at = positions.get("aborted", [None])[0]
        ends = [p for p in (after_at, aborted_at) if p is not None]
        self.terminal = min(ends) if ends else None
        self.after_at = after_at
        self.aborted_at = aborted_at
        if positions.get("hello") != [0]:
            self.fail("a second hello in the chain")
        for kind in ("before", "after", "aborted", "commitments"):
            if len(positions.get(kind, [])) > 1:
                self.fail(
                    f"message {positions[kind][1]}: a second {kind} message "
                    "(a room has one: there is no second round)"
                )
        if after_at is not None and aborted_at is not None:
            later = max(after_at, aborted_at)
            self.fail(
                f"message {later}: "
                + (
                    "a void after the game finished"
                    if aborted_at > after_at
                    else "an after following a void"
                )
            )
        start = self.before_at if self.before_at is not None else end
        for position, body in enumerate(self.bodies[1:], start=1):
            kind = body.get("type")
            if kind not in KNOWN:
                where = "before the game" if position < start else "in the chain"
                self.fail(f"message {position}: unexpected {str(kind)[:40]!r} {where}")
            elif self.terminal is not None and position > self.terminal:
                if kind not in ANSWERS and position not in (after_at, aborted_at):
                    self.fail(f"message {position}: a {kind} message after the end")
            elif kind in PRE_GAME and position > start:
                self.fail(f"message {position}: a {kind} message during the game")
            elif kind in IN_GAME + ("after",) and position < start:
                self.fail(f"message {position}: a {kind} message before the game")

    # ===== 3. seeds and binding =====
    def check_room(self, room: Any, position: int) -> None:
        if (
            not isinstance(room, dict)
            or room.get("id") != self.room_id
            or room.get("seed_commitment") != self.seed_commitment
            or not _same(room.get("config"), self.config_json)
        ):
            self.fail(
                f"message {position}: the room changed its rules or seed commitment"
            )

    def check_setup(self) -> None:
        hello = self.bodies[0]
        self.require(
            hello.get("protocol_version") == PROTOCOL_VERSION,
            f"protocol {hello.get('protocol_version')!r} (this verifier speaks {PROTOCOL_VERSION})",
        )
        room = hello["room"]
        self.config_json = room.get("config")
        self.seed_commitment = room.get("seed_commitment")
        self.require(
            is_hex(self.seed_commitment, 32), "the room has no seed commitment"
        )
        self.report.shared = {
            "room_id": self.room_id,
            "server_key": self.server_key,
            "seed_commitment": self.seed_commitment,
        }
        try:
            self.config = GameConfig.from_json(self.config_json)
        except ConfigError as error:
            self.fail(f"invalid config: {error}")
            raise _Abort()
        if not _same(self.config.to_json(), self.config_json):
            self.fail("the config is not in its canonical form")

        self.committed = None  # (position, body)
        self.reveal_receipts: list = []  # (position, body) before the game
        stop = min(
            p
            for p in (self.before_at, self.terminal, len(self.bodies))
            if p is not None
        )
        for position in range(1, stop):
            body = self.bodies[position]
            kind = body.get("type")
            if kind == "room":
                self.check_room(body.get("room"), position)
            elif kind == "joined":
                self.check_room(body.get("room"), position)
                if body.get("room_id") != self.room_id:
                    self.fail(f"message {position}: a join to another room")
            elif kind == "commitments" and self.committed is None:
                self.check_commitments(position, body)
                self.committed = (position, body)
            elif kind == "received":
                self.reveal_receipts.append((position, body))
        if self.committed is not None:
            self.report.shared["commitments"] = self.committed[1]
            #: What every statement and reveal of this game signs.
            self.C = digest(self.committed[1])
            self.seats = self.committed[1]["seats"]

    def check_commitments(self, position: int, body: dict) -> None:
        if (
            sorted(body) != sorted(COMMITMENTS_FIELDS)
            or body.get("room_id") != self.room_id
            or body.get("server_key") != self.server_key
            or body.get("seed_commitment") != self.seed_commitment
            or not _same(body.get("config"), self.config_json)
            or type(body.get("reveal_timeout_ms")) is not int
        ):
            self.fail(
                f"message {position}: the commitments change the room, its server, "
                "its rules or its seed commitment"
            )
        self.check_seats(body.get("seats"), f"message {position}")
        if not self.seated:
            return
        if self.joined_at is None or position < self.joined_at:
            self.fail(
                f"message {position}: the commitments came before this seat's join"
            )
            return
        mine = [s for s in body["seats"] if s.get("public_key") == self.me]
        if (
            len(mine) != 1
            or mine[0].get("seat") != self.my_seat
            or mine[0].get("color") != self.joined.get("color")
            or mine[0].get("client_commitment") != self.joined.get("client_commitment")
        ):
            self.fail(
                f"message {position}: the commitments do not seat this client as it joined"
            )

    def check_seats(self, seats: Any, where: str) -> None:
        colors = [c.value for c in self.config.colors]
        self.require(
            isinstance(seats, list) and len(seats) == len(colors),
            f"{where}: the seats do not match the config",
        )
        keys = []
        for index, seat in enumerate(seats):
            self.require(
                isinstance(seat, dict) and sorted(seat) == sorted(SEAT_FIELDS),
                f"{where}: seat {index} is malformed",
            )
            ok = (
                type(seat.get("seat")) is int
                and seat["seat"] == index
                and seat.get("color") == colors[index]
            )
            if seat.get("kind") == "remote":
                requested = seat.get("requested_seat")
                ok = ok and is_hex(seat.get("public_key"), 32)
                ok = ok and is_hex(seat.get("client_commitment"), 32)
                ok = ok and is_hex(seat.get("nonce"), 16)
                # The seat it asked for, if it asked for one, is the seat it got.
                ok = ok and (
                    requested is None or (type(requested) is int and requested == index)
                )
                # The key holder joined this room, of this server, under its
                # seed commitment, on its own connection.
                ok = ok and verify_signature(
                    seat.get("public_key"),
                    join_statement(
                        self.room_id,
                        self.server_key,
                        self.seed_commitment,
                        requested,
                        seat.get("client_commitment"),
                        seat.get("nonce"),
                    ),
                    seat.get("pop"),
                )
                keys.append(seat.get("public_key"))
            elif seat.get("kind") == "bot":
                ok = ok and all(
                    seat.get(k) is None
                    for k in ("public_key", "client_commitment", "pop", "nonce")
                )
            else:
                ok = False
            self.require(ok, f"{where}: seat {index} is malformed")
        self.require(len(keys) == len(set(keys)), f"{where}: two seats share a key")

    # ===== 4. the game, and the void =====
    def check_game(self) -> None:
        if self.before_at is not None:
            self.check_before()
        if self.aborted_at is not None:
            self.check_void(self.aborted_at)
        if self.before_at is None:
            if self.seated and self.terminal is not None and self.aborted_at is None:
                self.fail("the game never started")
            return
        for position in range(self.before_at, len(self.messages)):
            commit = self.messages[position].get("commit")
            if isinstance(commit, dict) and type(commit.get("index")) is int:
                index, hash_ = commit["index"], commit.get("hash")
                if self.report.commitments.setdefault(index, hash_) != hash_:
                    self.fail(
                        f"message {position}: another commitment for state {index}"
                    )
        if self.after_at is not None and self.after_at > self.before_at:
            self.check_reveal()
            self.replay()
        elif self.aborted_at is not None and self.aborted_at > self.before_at:
            self.check_start_from_void()

    def check_before(self) -> None:
        before = self.bodies[self.before_at]
        self.before = before
        commitments = before.get("commitments")
        self.require(
            isinstance(commitments, dict), "the before message carries no commitments"
        )
        if self.seated and self.committed is None:
            self.fail(
                "the seats' seed commitments were not published between joining and the game"
            )
        if self.committed is not None and not _same(self.committed[1], commitments):
            self.fail(
                "the game's commitments are not the ones published before the seeds were revealed"
            )
        if self.committed is None:
            self.check_commitments(self.before_at, commitments)
            self.report.shared["commitments"] = commitments
            self.C = digest(commitments)
            self.seats = commitments.get("seats")
        self.require(
            before.get("room_id") == self.room_id
            and isinstance(before.get("game_id"), str),
            "the before message names another room, or no game",
        )
        self.report.game_id = before["game_id"]
        if (
            not _same(before.get("config"), self.config_json)
            or not _same(before.get("seats"), self.seats)
            or before.get("seed_commitment") != self.seed_commitment
        ):
            self.fail("the game's config or seats are not the committed ones")
        seeds, reveals = before.get("client_seeds"), before.get("reveals")
        self.require(
            isinstance(seeds, list)
            and isinstance(reveals, list)
            and len(seeds) == len(reveals) == len(self.seats),
            "seats, client seeds and reveals do not line up",
        )
        self.client_seeds = seeds
        for seat, seed, signature in zip(self.seats, seeds, reveals):
            if seat["kind"] == "remote":
                if (
                    not is_hex(seed, 32)
                    or client_seed_commitment(self.room_id, seat["public_key"], seed)
                    != seat["client_commitment"]
                ):
                    self.fail(
                        f"seat {seat['seat']}'s seed does not open its commitment"
                    )
                if not verify_signature(
                    seat["public_key"], reveal_statement(self.C), signature
                ):
                    self.fail(
                        f"seat {seat['seat']}'s key did not reveal its seed for these commitments"
                    )
            elif seed is not None or signature is not None:
                self.fail(f"seat {seat['seat']} is a server bot but contributed a seed")
        self.color = before.get("color")
        self.report.color = self.color
        if self.seated:
            if self.color != self.joined.get("color"):
                self.fail("this client is not seated in the game as it joined")
            if before.get("index") != 0:
                self.fail("a seat's game must start from the first move")
        elif self.color is not None:
            self.fail("a spectator's transcript carries a seat's view")
        self.report.shared["game_id"] = before["game_id"]
        self.report.shared["before"] = {
            k: before.get(k)
            for k in ("room_id", "game_id", "commitments", "client_seeds", "reveals")
        }

    def check_reveal(self) -> None:
        self.after = self.bodies[self.after_at]
        reveal = self.after.get("reveal")
        self.require(isinstance(reveal, dict), "the after message reveals nothing")
        self.server_seed = reveal.get("server_seed")
        self.require(
            is_hex(self.server_seed, 32)
            and seed_commitment(self.server_seed) == self.seed_commitment,
            "the revealed server seed does not open the seed commitment",
        )
        self.actions, self.salts = reveal.get("actions"), reveal.get("salts")
        self.require(
            isinstance(self.actions, list) and isinstance(self.salts, list),
            "the reveal lists no actions or salts",
        )
        self.require(
            len(self.salts) == len(self.actions) + 1
            and all(is_hex(salt, 16) for salt in self.salts),
            f"{len(self.salts)} salts revealed for {len(self.actions)} actions",
        )
        self.report.shared["reveal"] = reveal

    def check_start_from_void(self) -> None:
        """A game called off after it started: the void reveals the server
        seed, so the game's start (board, turn order, the first view) can be
        checked even without the rest of the reveal."""
        seed = self.bodies[self.aborted_at].get("server_seed")
        if not is_hex(seed, 32) or seed_commitment(seed) != self.seed_commitment:
            return  # (an error already, from check_void)
        game = new_game(
            self.config,
            master_seed(seed, self.client_seeds, self.C),
            self.before["game_id"],
        )
        expected = before_body(
            game,
            state_to_json(game),
            self.color,
            commitments=self.before["commitments"],
            client_seeds=self.client_seeds,
            reveals=self.before["reveals"],
        )
        if not _same(self.before, _plain(expected)):
            self.fail(
                f"message {self.before_at} (the game's start): "
                f"{_difference(self.before, _plain(expected))}"
            )
        self.report.count("views")

    def check_void(self, position: int) -> None:
        """The room was called off: what the void itself claims must hold,
        and it must blame honestly what it can be held to."""
        body = self.bodies[position]
        self.report.void = body
        self.report.shared["aborted"] = body
        revealed = body.get("server_seed")
        if (
            not is_hex(revealed, 32)
            or seed_commitment(revealed) != self.seed_commitment
        ):
            self.fail("the void does not reveal the seed the server committed to")
        game_id = body.get("game_id")
        if self.before_at is not None and self.before_at < position:
            names_its_game = game_id == self.bodies[self.before_at].get("game_id")
        elif self.seated:
            # Every seat gets the before: a seat's void before it is a void
            # before the game.
            names_its_game = game_id is None
        else:
            # A connection that saw no before may hear of a game's void.
            names_its_game = game_id is None or isinstance(game_id, str)
        if body.get("room_id") != self.room_id or not names_its_game:
            self.fail("the void names another room or game than this chain's")
        blame = body.get("blame")
        if not isinstance(blame, list):
            self.fail("the void names no list of seats to blame")
            return
        seats = getattr(self, "seats", None)
        if seats is not None and not all(
            type(i) is int and 0 <= i < len(seats) and seats[i]["kind"] == "remote"
            for i in blame
        ):
            self.fail("the void blames a seat that is not a remote seat")
            return
        if self.before_at is not None and self.before_at < position:
            # The server's own before holds these seats' seeds and reveals.
            for i in blame:
                seed, signature = self.client_seeds[i], self.before["reveals"][i]
                if (
                    is_hex(seed, 32)
                    and client_seed_commitment(
                        self.room_id, seats[i]["public_key"], seed
                    )
                    == seats[i]["client_commitment"]
                    and verify_signature(
                        seats[i]["public_key"], reveal_statement(self.C), signature
                    )
                ):
                    self.fail(
                        f"the void blames seat {i} for withholding its seed, which "
                        "the server's own before holds, revealed"
                    )
            return
        if not self.seated or self.my_seat not in blame:
            return
        if self.committed is None or self.committed[0] > position:
            self.fail(
                "the void blames this seat for withholding a seed it was never "
                "asked to reveal (no commitments came before it)"
            )
            return
        signature = self.transcript.get("reveal_sig")
        if not verify_signature(self.me, reveal_statement(self.C), signature):
            self.report.warn(
                "the server blamed this seat for withholding its seed (this "
                "transcript holds no reveal of it to show otherwise)"
            )
            return
        mine = submission_digest(reveal_statement(self.C), signature)
        acknowledged = [
            p
            for p, receipt in self.reveal_receipts
            if receipt.get("submission") == mine and receipt.get("accepted") is True
        ]
        if acknowledged:
            self.fail(
                f"the server blamed this seat for withholding its seed after "
                f"acknowledging its reveal (receipt in message {acknowledged[0]})"
            )
        else:
            self.report.warn(
                "the server blamed this seat for withholding its seed; this client "
                "says it revealed, which the server never acknowledged (not provable)"
            )

    # ===== 5. the replay =====
    def replay(self) -> None:
        report = self.report
        master = master_seed(self.server_seed, self.client_seeds, self.C)
        self.game = new_game(self.config, master, self.before["game_id"])
        self.index = 0
        self.doc = state_to_json(self.game)
        self.by_color = {seat["color"]: seat for seat in self.seats}
        self.last_seq: dict = {}
        self.decides: dict = {}  # index -> envelope
        self.forced_at: dict = {}  # index -> whether publicly forced

        next_step = None
        for position in range(self.before_at, self.after_at + 1):
            envelope, body = self.messages[position], self.bodies[position]
            kind, commit = body.get("type"), envelope.get("commit")
            if kind not in ("before", "decide", "step", "after"):
                continue  # receipts, errors (and what the schedule flagged)
            if not isinstance(commit, dict) or type(commit.get("index")) is not int:
                self.fail(f"message {position}: no state commitment during the game")
                continue
            target = commit["index"]
            self.advance(target, position)
            if state_commitment(self.doc, self.salts[target]) != commit.get("hash"):
                self.fail(
                    f"message {position}: the state commitment does not open to the "
                    f"replayed state {target}"
                )
            report.count("commitments")

            if kind == "before":
                expected = before_body(
                    self.game,
                    self.doc,
                    self.color,
                    commitments=self.before["commitments"],
                    client_seeds=self.client_seeds,
                    reveals=self.before["reveals"],
                )
                next_step = target + 1
            elif kind == "decide":
                if (
                    self.color is None
                    or body.get("color") != self.color
                    or self.game.state.current_color().value != self.color
                ):
                    self.fail(
                        f"message {position}: a decide when this seat was not to move"
                    )
                    continue
                if target in self.decides:
                    self.fail(f"message {position}: a second decide for move {target}")
                    continue
                if self.config.autoplay_forced and publicly_forced(self.game):
                    self.fail(
                        f"message {position}: a decide for a move everyone knew was forced"
                    )
                self.decides[target] = envelope
                expected = decide_body(
                    self.game,
                    self.doc,
                    self.color,
                    timeout_ms=self.config.decision_timeout_ms,
                )
            elif kind == "step":
                if body.get("index") != next_step or target != next_step:
                    self.fail(
                        f"message {position}: step {body.get('index')} out of order (expected {next_step})"
                    )
                    continue
                next_step += 1
                forced = self.actions[target - 1].get("forced")
                expected = step_body(self.game, self.doc, self.color, forced=forced)
            else:  # after
                if target != len(self.actions) or not is_over(self.game):
                    self.fail("the game ended before the replay did")
                if next_step != len(self.actions) + 1:
                    self.fail("steps were withheld: the after came early")
                expected = after_body(
                    self.game, self.doc, self.color, reveal=self.after["reveal"]
                )
            if not _same(body, _plain(expected)):
                self.fail(
                    f"message {position} ({kind} at state {target}): "
                    f"{_difference(body, _plain(expected))}"
                )
            report.count("views")
        if self.index != len(self.actions):
            self.fail("revealed actions past the last message")
        if self.seated:
            for index, entry in enumerate(self.actions):
                if entry.get("color") != self.color:
                    continue
                if self.forced_at.get(index) and entry.get("forced") == "only":
                    continue
                if index not in self.decides:
                    self.fail(
                        f"action {index}: this seat's move was made without asking it"
                    )
            report.counts["own moves"] = sum(
                1 for a in self.actions if a.get("color") == self.color
            )

    def advance(self, target: int, position: int) -> None:
        if target < self.index or target > len(self.actions):
            self.fail(f"message {position} refers to state {target}, out of order")
            raise _Abort()
        while self.index < target:
            self.apply(self.index, self.actions[self.index])
            self.index += 1
            self.doc = state_to_json(self.game)

    def apply(self, index: int, entry: Any) -> None:
        current = self.game.state.current_color().value
        self.require(isinstance(entry, dict), f"action {index} is malformed")
        if entry.get("index") != index or entry.get("color") != current:
            self.fail(f"action {index}: not {current}'s move")
            raise _Abort()
        offered = [action_to_json(a) for a in self.game.playable_actions]
        matches = [
            a
            for a, js in zip(self.game.playable_actions, offered)
            if _same(js, entry.get("action"))
        ]
        if not matches:
            self.fail(
                f"action {index}: {entry.get('action')!r} is not one of the moves the game offered"
            )
            raise _Abort()
        action = matches[0]
        self.forced_at[index] = self.config.autoplay_forced and publicly_forced(
            self.game
        )
        self.authorize(index, entry, action, self.by_color[current])
        try:
            self.game.execute(action)
        except Exception as error:
            self.fail(f"action {index}: illegal in the replay ({error})")
            raise _Abort()
        self.report.count("actions")

    def authorize(self, index: int, entry: dict, action, seat: dict) -> None:
        forced, by = entry.get("forced"), entry.get("by")
        playable = list(self.game.playable_actions)
        if forced not in FORCED:
            self.fail(f"action {index}: unknown forced reason {forced!r}")
        elif forced == "only":
            if not self.forced_at[index] or action != playable[0]:
                self.fail(
                    f"action {index}: played as forced, but nobody could know it was"
                )
        elif forced in ("timeout", "bot_error"):
            if action != default_action(playable):
                self.fail(f"action {index}: a {forced} must play the default action")
            if forced == "bot_error" and seat["kind"] != "bot":
                self.fail(f"action {index}: a bot error on a seat that is not a bot")
        elif seat["kind"] == "bot":
            if by != "server":
                self.fail(f"action {index}: a server bot's move must be the server's")
        else:
            self.authorize_signed(index, entry, seat)

    def authorize_signed(self, index: int, entry: dict, seat: dict) -> None:
        statement, signature = entry.get("statement"), entry.get("sig")
        signed = (
            entry.get("by") == "client"
            and isinstance(statement, dict)
            and verify_signature(seat["public_key"], statement, signature)
        )
        if not signed:
            self.fail(
                f"action {index}: {seat['color']}'s move is not signed by its key"
            )
            return
        seq, decide = statement.get("seq"), statement.get("decide")
        expected = action_statement(self.C, index, seq, decide, entry.get("action"))
        if (
            not _same(statement, expected)
            or type(seq) is not int
            or not is_hex(decide, 32)
        ):
            self.fail(
                f"action {index}: the signed statement is for another move or game"
            )
            return
        if seq <= self.last_seq.get(seat["seat"], -1):
            self.fail(f"action {index}: {seat['color']}'s statements are not in order")
        self.last_seq[seat["seat"]] = seq
        if self.seated and seat["public_key"] == self.me:
            decide_envelope = (
                self.messages[seq] if 0 <= seq < len(self.messages) else None
            )
            if (
                decide_envelope is None
                or decide_envelope["body"].get("type") != "decide"
                or envelope_hash(decide_envelope) != decide
                or decide_envelope["body"].get("index") != index
            ):
                self.fail(
                    f"action {index}: this client's signed move answers a decide it did "
                    "not get at that point"
                )
        self.report.count("client signatures")

    # ===== 6. this client's own moves, from what the server signed =====
    def check_own_moves(self) -> None:
        """This client's moves against what it signed and what the server
        acknowledged, from the signed decides, receipts and steps alone (no
        reveal needed, so a game that is void or unfinished is checked too).

        Provable to anyone: a receipt for a legal, correctly signed answer
        to decide k that comes before the step for move k (the server never
        signs one between the deadline and that step), or a receipt saying
        it was accepted, wherever it is, proves the server had the answer;
        a step that plays another move is then the server's doing. During
        the game an action is answered by a receipt, never by an error; and
        a seat cannot be timed out of a move it was not asked.

        Not provable (warnings): that this client sent an answer the server
        never acknowledged (``sent`` is only the client's word); when an
        answer arrived relative to the deadline; every move the server
        played for this seat as a timeout (deadlines are its word).
        """
        if not self.seated or self.before_at is None:
            return
        end = self.terminal if self.terminal is not None else len(self.bodies)
        color = self.joined.get("color")
        decides: dict = {}  # index -> (position, envelope)
        steps: dict = {}  # index -> (position, body)
        receipts: dict = {}  # submission -> [(position, body)]
        for position in range(self.before_at + 1, len(self.bodies)):
            body = self.bodies[position]
            kind = body.get("type")
            if kind == "decide" and position < end:
                decides.setdefault(
                    body.get("index"), (position, self.messages[position])
                )
            elif kind == "step" and position < end and type(body.get("index")) is int:
                steps.setdefault(body["index"] - 1, (position, body))
            elif kind == "received":
                receipts.setdefault(body.get("submission"), []).append((position, body))
            elif (
                kind == "error"
                and body.get("answers") == "action"
                and self.joined_at is not None
                and position < end
            ):
                # During the game every action gets a receipt (one that
                # names it), never an error that names nothing.
                self.fail(
                    f"message {position}: an action answered with an error, not a receipt"
                )

        signed: dict = {}  # index -> {submission digest: entry}
        for item in self.transcript.get("sent") or []:
            if not isinstance(item, dict):
                continue
            statement, signature = item.get("statement"), item.get("sig")
            if (
                not isinstance(statement, dict)
                or type(statement.get("index")) is not int
                or not verify_signature(self.me, statement, signature)
            ):
                continue  # only what this key really signed counts
            signed.setdefault(statement["index"], {})[
                submission_digest(statement, signature)
            ] = item
        unanswered = {
            u.get("index"): u
            for u in self.transcript.get("unanswered") or []
            if isinstance(u, dict)
        }

        for index, (position, body) in sorted(steps.items()):
            log = body.get("log") or [[None]]
            played = log[0][0] if isinstance(log[0], list) and log[0] else None
            if not isinstance(played, list) or played[:1] != [color]:
                continue
            if body.get("forced") == "timeout":
                if index not in decides:
                    self.fail(
                        f"action {index}: the server timed this seat out of a move it "
                        "never asked it to make"
                    )
                else:
                    note = unanswered.get(index)
                    had = (
                        f" (this client says it had the decide for {note.get('ms')} ms)"
                        if note and type(note.get("ms")) is int
                        else ""
                    )
                    self.report.warn(
                        f"action {index}: the server played the default for this seat, "
                        f"saying its time ran out{had} (deadlines are the server's word)"
                    )

        for index, (decide_at, envelope) in sorted(decides.items()):
            if type(index) is not int:
                continue
            items = signed.get(index, {})
            if len({digest(item["statement"]) for item in items.values()}) > 1:
                self.report.warn(
                    f"action {index}: this client signed different moves for one "
                    "decision (so it cannot hold the server to any of them)"
                )
                continue
            legal = {
                key: item
                for key, item in items.items()
                if self._legal_answer(envelope, index, item["statement"])
            }
            if not legal or index not in steps:
                continue
            step_at, step = steps[index]
            log = step.get("log") or [[None]]
            played = log[0][0] if isinstance(log[0], list) and log[0] else None
            if any(
                _same_move(item["statement"]["action"], played)
                for item in legal.values()
            ):
                continue
            seen = sorted((p, b) for key in legal for p, b in receipts.get(key, []))
            in_time = [p for p, b in seen if p < step_at]
            accepted = [p for p, b in seen if b.get("accepted") is True]
            if in_time or accepted:
                self.fail(
                    f"action {index}: the server had this client's legal, signed move "
                    f"(receipt in message {(in_time or accepted)[0]}) and played another"
                )
            elif not seen:
                self.report.warn(
                    f"action {index}: this client says it sent a legal, signed move "
                    "the server never acknowledged, and the server played another "
                    "(not provable: only this client's word)"
                )
            else:
                self.report.warn(
                    f"action {index}: this client's move reached the server after the "
                    "move was played (late)"
                )

    def _legal_answer(self, decide: dict, index: int, statement: dict) -> bool:
        expected = action_statement(
            self.C, index, decide["seq"], envelope_hash(decide), statement.get("action")
        )
        offered = decide["body"].get("playable_actions") or []
        return _same(statement, expected) and any(
            _same(statement.get("action"), a) for a in offered
        )


def _difference(got: Any, want: Any, path: str = "") -> str:
    """Where two JSON values first differ, for error messages."""
    if isinstance(got, dict) and isinstance(want, dict):
        for key in sorted(set(got) | set(want)):
            if not _same(got.get(key, _MISSING), want.get(key, _MISSING)):
                return _difference(
                    got.get(key, _MISSING), want.get(key, _MISSING), f"{path}.{key}"
                )
    if isinstance(got, list) and isinstance(want, list) and len(got) == len(want):
        for i, (a, b) in enumerate(zip(got, want)):
            if not _same(a, b):
                return _difference(a, b, f"{path}[{i}]")
    return f"{path or 'body'} is {_short(got)}, an honest server sends {_short(want)}"


_MISSING = "<missing>"


def _short(value: Any) -> str:
    text = json.dumps(value)
    return text if len(text) < 80 else text[:77] + "..."


def verify_transcripts(transcripts: Sequence[Any]) -> list:
    """Verify each transcript, then check every pair of transcripts of one
    room against each other: under one server key, one room has one seed
    commitment, one commitments message, one game (its id, seeds, reveals
    and reveal) or one void, and one state commitment at each point. Every
    transcript of a pair that disagrees fails, whatever the order given.
    (Two server keys for one room id are only a warning: anyone can run a
    server with any room id, and the commitments name the key, so nothing
    one key signed can stand in for the other's.)"""
    reports = [verify_transcript(t) for t in transcripts]
    groups: dict = {}
    for report in reports:
        if report.shared is not None and not report.invalid:
            groups.setdefault(report.shared["room_id"], []).append(report)
    for members in groups.values():
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                if a.shared["server_key"] != b.shared["server_key"]:
                    note = (
                        "another transcript has this room id under another server key"
                    )
                    for report in (a, b):
                        if note not in report.warnings:
                            report.warn(note)
                    continue
                problems = _disagreements(a, b)
                if problems:
                    message = "equivocation: recipients of one room got " + ", ".join(
                        problems
                    )
                    for report in (a, b):
                        if message not in report.errors:
                            report.fail(message)
    return reports


def _disagreements(a: Report, b: Report) -> list:
    problems = []
    x, y = a.shared, b.shared
    if x["seed_commitment"] != y["seed_commitment"]:
        problems.append("other seed commitments")
    if "commitments" in x and "commitments" in y:
        if not _same(x["commitments"], y["commitments"]):
            problems.append("other commitments")
    if "game_id" in x and "game_id" in y:
        if x["game_id"] != y["game_id"]:
            problems.append("another game id")
        if not _same(x["before"], y["before"]):
            problems.append("other seeds or reveals")
    if "reveal" in x and "reveal" in y and not _same(x["reveal"], y["reveal"]):
        problems.append("another reveal")
    if "aborted" in x and "aborted" in y and not _same(x["aborted"], y["aborted"]):
        problems.append("another void")
    for one, other in ((x, y), (y, x)):
        void = one.get("aborted")
        if void is None:
            continue
        if void.get("game_id") is None and "game_id" in other:
            problems.append("a void before the game and a game")
            break
        if "reveal" in other:
            problems.append("a void and a finished game")
            break
    for index, digest_ in a.commitments.items():
        if b.commitments.get(index, digest_) != digest_:
            problems.append(f"another state {index}")
            break
    return problems


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="catanatron-verify",
        description="Replay catanatron-server transcripts and check the server was honest.",
    )
    parser.add_argument(
        "transcripts", nargs="+", help="transcript files (.json or .json.gz)"
    )
    parser.add_argument("--json", action="store_true", help="print the reports as JSON")
    args = parser.parse_args(argv)
    loaded, unreadable = [], {}
    for path in args.transcripts:
        try:
            loaded.append(load_transcript(path))
        except Exception as error:  # zlib.error, UnicodeError...: not a crash
            loaded.append(None)
            unreadable[path] = f"cannot read: {type(error).__name__}: {error}"
    reports = verify_transcripts(loaded)
    for path, report in zip(args.transcripts, reports):
        if path in unreadable:
            report.invalid = [unreadable[path]]
    if args.json:
        print(json.dumps([r.summary() for r in reports], indent=2))
    else:
        for path, report in zip(args.transcripts, reports):
            print(
                f"{path}: {report.verdict} "
                f"(game {report.game_id}, {report.color or 'spectator'})"
            )
            if report.counts:
                print(
                    "  checked: "
                    + ", ".join(f"{v} {k}" for k, v in report.counts.items())
                )
            if report.void is not None:
                print(
                    f"  void: {report.void.get('reason')!r}"
                    + (
                        f" (blamed seats: {report.void.get('blame')})"
                        if report.void.get("blame")
                        else ""
                    )
                )
            if report.incomplete is not None:
                print(f"  incomplete: {report.incomplete}")
            for problem in report.invalid:
                print(f"  not evidence: {problem}")
            for error in report.errors:
                print(f"  error: {error}")
            for warning in report.warnings:
                print(f"  warning: {warning}")
    return 0 if all(r.ok for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
