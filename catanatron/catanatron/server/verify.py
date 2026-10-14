"""Check a game's transcript by replaying it in stock Catanatron.

A transcript is what one client received (every signed message, in order)
plus what it knows of itself (its public key, its seed, what it signed).
:func:`verify_transcript` checks the invariants of PROTOCOL.md (next to this
module) that one recipient's transcript can show:

1. every message is signed by the server's key, numbered in order, chained
   to the previous one by hash, and (once seated) addressed to this client;
2. the seeds: the server committed to its seed in the ``hello``; this client
   joined with a commitment to its own seed; one ``commitments`` message (and
   only one) fixed every seat, key and seed commitment and the config (the
   ones the ``hello`` advertised) before any seed was revealed; every
   revealed seed opens its commitment, this client's included; the revealed
   server seed opens the server's;
3. the game is the committed one: ``before`` carries exactly the commitments
   this client received, and the master seed is derived from all of it;
4. replaying the revealed actions from that seed in stock Catanatron, every
   action is one the game offered, every state commitment opens to the
   replayed state with its revealed salt, and every message body is
   *exactly* what an honest server builds from the replayed state for this
   recipient;
5. the schedule: one ``before``, then one ``step`` per action in order, a
   ``decide`` exactly when this seat had a move that was not publicly
   forced (and at most one per move), one ``after`` last;
6. every move by a remote seat is signed by that seat's key over the
   commitments and the action's index (so a signature cannot be replayed for
   another move); every move of this client answers the ``decide`` it got
   at that point; every move the server made for a seat is one it may make
   (the only legal one, when everyone could see it was; or the default
   after a timeout); and no legal move of this client that the server had
   acknowledged in time (or at all, as accepted) was replaced.

A void room (``aborted``) is not a game: the report says so (``void``) and
checks only what the void itself claims (the revealed seed opens the
commitment; a seat whose reveal the server acknowledged is not blamed).

Errors are what the server's own signatures prove against it, to anyone;
warnings are what only this client knows (that it sent an answer the server
never acknowledged, or acknowledged late).

``catanatron-verify T1.json [T2.json ...]`` runs it from the shell; given
several transcripts of one room (under one server key) it also checks every
pair of them against each other: the same commitments, game, seeds, reveal
or void, and the same state commitment at each point (no equivocation).
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
    after_body,
    before_body,
    decide_body,
    step_body,
)

MAX_ERRORS = 50
FORCED = (None, "only", "timeout", "bot_error")


@dataclass
class Report:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    counts: dict = field(default_factory=dict)
    game_id: Optional[str] = None
    color: Optional[str] = None
    commitments: dict = field(default_factory=dict)
    #: What every recipient of this room must have been told alike, by name.
    shared: Optional[dict] = None
    #: The ``aborted`` body if the room is void (no game to verify).
    void: Optional[dict] = None

    @property
    def ok(self) -> bool:
        """A finished game, every check passed."""
        return not self.errors and self.void is None

    def fail(self, message: str) -> None:
        if len(self.errors) < MAX_ERRORS:
            self.errors.append(message)

    def count(self, what: str) -> None:
        self.counts[what] = self.counts.get(what, 0) + 1

    def summary(self) -> dict:
        return {
            "ok": self.ok,
            "void": self.void,
            "game_id": self.game_id,
            "color": self.color,
            "errors": self.errors,
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
        report.fail("malformed transcript: nested too deeply")
    except Exception as error:  # a malformed transcript must fail, not crash
        report.fail(f"malformed transcript: {type(error).__name__}: {error}")
    return report


class _Verifier:
    def __init__(self, transcript: Any, report: Report):
        self.transcript = transcript
        self.report = report

    def fail(self, message: str) -> None:
        self.report.fail(message)

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.fail(message)
            raise _Abort()

    # ===== 1. the chain =====
    def run(self) -> None:
        t = self.transcript
        self.require(isinstance(t, dict), "a transcript is a JSON object")
        raw = t.get("messages")
        self.require(isinstance(raw, list) and raw, "the transcript has no messages")
        self.messages = [json.loads(m) if isinstance(m, str) else m for m in raw]
        self.server_key = t.get("server_key")
        self.require(is_hex(self.server_key, 32), "server_key is not an Ed25519 key")
        self.me = t.get("public_key")
        self.my_seed = t.get("client_seed")
        self.seated = self.me is not None
        if self.seated:
            self.require(is_hex(self.me, 32), "public_key is not an Ed25519 key")
            self.require(is_hex(self.my_seed, 32), "client_seed is not 32 bytes of hex")

        head, joined = GENESIS, False
        for position, envelope in enumerate(self.messages):
            self.require(
                isinstance(envelope, dict) and isinstance(envelope.get("body"), dict),
                f"message {position} is not an envelope",
            )
            if not verify_envelope(self.server_key, envelope):
                self.fail(f"message {position}: bad server signature")
            if envelope.get("seq") != position or envelope.get("prev") != head:
                self.fail(f"message {position}: breaks the hash chain")
            head = envelope_hash(envelope)
            if self.seated and joined and envelope.get("to") != self.me:
                self.fail(
                    f"message {position}: addressed to {envelope.get('to')}, not this client"
                )
            joined = joined or envelope["body"].get("type") == "joined"
            self.report.count("signatures")
        if self.report.errors:
            raise _Abort()  # nothing below can be trusted
        self.bodies = [envelope["body"] for envelope in self.messages]
        self.check_setup()
        self.replay()
        self.check_own_moves()

    # ===== 2./3. seeds and binding =====
    def check_setup(self) -> None:
        hello = self.bodies[0]
        self.require(
            hello.get("type") == "hello" and hello.get("server_key") == self.server_key,
            "the transcript does not start with the server's hello",
        )
        self.require(
            hello.get("protocol_version") == PROTOCOL_VERSION,
            f"protocol {hello.get('protocol_version')!r} (this verifier speaks {PROTOCOL_VERSION})",
        )
        room = hello.get("room")
        self.require(isinstance(room, dict), "the hello describes no room")
        self.room_id = room.get("id")
        self.config_json = room.get("config")
        self.require(isinstance(self.room_id, str), "the room has no id")
        commitment = room.get("seed_commitment")
        self.require(is_hex(commitment, 32), "the room has no seed commitment")
        self.server_commitment = commitment
        if self.transcript.get("room_id") not in (None, self.room_id):
            self.fail("the transcript is for another room than its messages")
        self.report.shared = {
            "group": [self.server_key, self.room_id],
            "seed_commitment": commitment,
        }
        try:
            self.config = GameConfig.from_json(self.config_json)
        except ConfigError as error:
            self.fail(f"invalid config: {error}")
            raise _Abort()
        if not _same(self.config.to_json(), self.config_json):
            self.fail("the config is not in its canonical form")

        kinds = [b.get("type") for b in self.bodies]
        positions: dict = {}
        for i, kind in enumerate(kinds):
            positions.setdefault(kind, []).append(i)
        self.require(positions.get("hello") == [0], "a second hello in the chain")
        befores = positions.get("before", [])
        self.require(
            len(befores) <= 1, f"{len(befores)} before messages (one is honest)"
        )
        self.before_at = befores[0] if befores else len(self.bodies)
        aborted_at = positions.get("aborted", [len(self.bodies)])[0]
        if len(positions.get("aborted", [])) > 1:
            self.fail("the room was called off twice")

        # Before the game: the room as it fills, this client's seat, and the
        # one commitments message (every seat, fixed before any reveal).
        self.committed = None  # (position, body)
        self.joined: dict = {}
        joined_at = None
        self.reveal_receipts: list = []  # (position, body) before the game
        for position in range(1, min(self.before_at, aborted_at)):
            body = self.bodies[position]
            kind = body.get("type")
            if kind == "room":
                r = body.get("room") or {}
                if (
                    r.get("id") != self.room_id
                    or r.get("seed_commitment") != commitment
                    or not _same(r.get("config"), self.config_json)
                ):
                    self.fail(
                        f"message {position}: the room changed its rules or seed commitment"
                    )
            elif kind == "joined":
                if joined_at is not None:
                    self.fail("this client was seated twice")
                joined_at, self.joined = position, body
            elif kind == "commitments":
                if self.committed is not None:
                    self.fail(
                        f"message {position}: a second commitments message (a room "
                        "has one: there is no second round)"
                    )
                    continue
                if (
                    sorted(body) != sorted(COMMITMENTS_FIELDS)
                    or body.get("room_id") != self.room_id
                    or body.get("seed_commitment") != commitment
                    or not _same(body.get("config"), self.config_json)
                    or type(body.get("reveal_timeout_ms")) is not int
                ):
                    self.fail(
                        f"message {position}: the commitments change the room, its rules or its seed commitment"
                    )
                self.committed = (position, body)
            elif kind == "received":
                self.reveal_receipts.append((position, body))
            elif kind == "error":
                pass
            elif kind in ("decide", "step", "after"):
                self.fail(f"message {position}: a {kind} message before the game")
            else:
                self.fail(f"message {position}: unexpected {kind!r} before the game")
        if self.committed is not None:
            self.report.shared["commitments"] = self.committed[1]
            #: What every statement and reveal of this game signs.
            self.C = digest(self.committed[1])
        if self.seated:
            self.require(joined_at is not None, "this client was not seated")
            if (
                self.joined.get("room_id") != self.room_id
                or self.joined.get("public_key") != self.me
                or self.joined.get("client_commitment")
                != client_seed_commitment(self.room_id, self.me, self.my_seed)
            ):
                self.fail(
                    "the server acknowledged another key or seed than this client's"
                )
            if self.committed is not None and self.committed[0] < joined_at:
                self.fail("the commitments came before this client was seated")
            self.my_seat = self.joined.get("seat")

        if befores:
            started = self.bodies[self.before_at]
            self.report.shared["game_id"] = started.get("game_id")
            self.report.shared["before"] = {
                k: started.get(k)
                for k in (
                    "room_id",
                    "game_id",
                    "commitments",
                    "client_seeds",
                    "reveals",
                )
            }
        if aborted_at < len(self.bodies):
            self.check_void(aborted_at)
            raise _Abort()

        afters = positions.get("after", [])
        self.require(bool(befores), "the game never started")
        self.require(
            len(afters) == 1, "the game has not ended: nothing was revealed yet"
        )
        self.after_at = afters[0]
        before = self.bodies[self.before_at]
        self.before = before
        commitments = before.get("commitments")
        self.require(
            isinstance(commitments, dict), "the before message carries no commitments"
        )
        if self.seated:
            self.require(
                self.committed is not None,
                "the seats' seed commitments were not published between joining and the game",
            )
        if self.committed is not None and not _same(self.committed[1], commitments):
            self.fail(
                "the game's commitments are not the ones published before the seeds were revealed"
            )
        if (
            sorted(commitments) != sorted(COMMITMENTS_FIELDS)
            or commitments.get("type") != "commitments"
            or commitments.get("room_id") != self.room_id
            or commitments.get("seed_commitment") != commitment
            or not _same(commitments.get("config"), self.config_json)
        ):
            self.fail(
                "the game's commitments change the room, its rules or its seed commitment"
            )
        self.report.shared.setdefault("commitments", commitments)
        #: What every statement and reveal of this game signs: fresh for this
        #: game, because it names every seat's fresh seed commitment.
        self.C = digest(commitments)
        self.seats = commitments.get("seats")
        self.check_seats(self.seats, "commitments")
        self.require(
            before.get("room_id") == self.room_id
            and isinstance(before.get("game_id"), str),
            "the before message names another room, or no game",
        )
        self.report.game_id = before["game_id"]
        if (
            not _same(before.get("config"), self.config_json)
            or not _same(before.get("seats"), self.seats)
            or before.get("seed_commitment") != commitment
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
            mine = [s for s in self.seats if s["public_key"] == self.me]
            self.require(
                len(mine) == 1
                and mine[0]["seat"] == self.my_seat
                and mine[0]["color"] == self.color,
                "this client is not seated in the game as it joined",
            )
            if mine[0]["client_commitment"] != client_seed_commitment(
                self.room_id, self.me, self.my_seed
            ):
                self.fail(
                    "the game's commitment for this client is not to this client's seed"
                )
            if seeds[self.my_seat] != self.my_seed:
                self.fail("this client's seed did not go into the game's seed")
            if before.get("index") != 0:
                self.fail("a seat's game must start from the first move")
        elif self.color is not None:
            self.fail("a spectator's transcript carries a seat's view")

        self.after = self.bodies[self.after_at]
        reveal = self.after.get("reveal")
        self.require(isinstance(reveal, dict), "the after message reveals nothing")
        self.server_seed = reveal.get("server_seed")
        if (
            not is_hex(self.server_seed, 32)
            or seed_commitment(self.server_seed) != commitment
        ):
            self.fail("the revealed server seed does not open the seed commitment")
            raise _Abort()
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

    def check_void(self, position: int) -> None:
        """The room was called off: what the void itself claims must hold."""
        body = self.bodies[position]
        self.report.void = body
        self.report.shared["aborted"] = body
        if self.before_at < position:
            self.report.game_id = self.bodies[self.before_at].get("game_id")
        revealed = body.get("server_seed")
        if (
            not is_hex(revealed, 32)
            or seed_commitment(revealed) != self.server_commitment
        ):
            self.fail("the void does not reveal the seed the server committed to")
        if not isinstance(body.get("blame"), list):
            self.fail("the void names no list of seats to blame")
            return
        if not self.seated or self.my_seat not in body["blame"]:
            return
        signature = self.transcript.get("reveal_sig")
        if self.committed is None or not verify_signature(
            self.me, reveal_statement(self.C), signature
        ):
            self.report.warnings.append(
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
            self.report.warnings.append(
                "the server blamed this seat for withholding its seed; this client "
                "says it revealed, which the server never acknowledged (not provable)"
            )

    def check_seats(self, seats: Any, where: str) -> None:
        colors = [c.value for c in self.config.colors]
        self.require(
            isinstance(seats, list) and len(seats) == len(colors),
            f"the {where} seats do not match the config",
        )
        keys = []
        for index, seat in enumerate(seats):
            self.require(isinstance(seat, dict), f"{where}: seat {index} is malformed")
            ok = seat.get("seat") == index and seat.get("color") == colors[index]
            if seat.get("kind") == "remote":
                ok = ok and is_hex(seat.get("public_key"), 32)
                ok = ok and is_hex(seat.get("client_commitment"), 32)
                requested = seat.get("requested_seat")
                ok = ok and (requested is None or type(requested) is int)
                # The key holder joined this room, under this round's server
                # commitment, with this seed commitment.
                ok = ok and verify_signature(
                    seat.get("public_key"),
                    join_statement(
                        self.room_id,
                        self.server_commitment,
                        requested,
                        seat.get("client_commitment"),
                    ),
                    seat.get("pop"),
                )
                keys.append(seat.get("public_key"))
            elif seat.get("kind") == "bot":
                ok = (
                    ok
                    and seat.get("public_key") is None
                    and seat.get("client_commitment") is None
                    and seat.get("pop") is None
                )
            else:
                ok = False
            self.require(ok, f"{where}: seat {index} is malformed")
        self.require(len(keys) == len(set(keys)), f"{where}: two seats share a key")

    # ===== 4./5./6. the replay =====
    def replay(self) -> None:
        report = self.report
        master = master_seed(self.server_seed, self.client_seeds, self.C)
        self.game = new_game(self.config, master, self.before["game_id"])
        self.index = 0
        self.doc = state_to_json(self.game)
        self.by_color = {seat["color"]: seat for seat in self.seats}
        self.last_seq: dict = {}
        self.decides: dict = {}  # index -> envelope
        self.offered: dict = {}  # index -> playable actions (JSON) at that point
        self.forced_at: dict = {}  # index -> whether publicly forced
        self.receipts: list = []  # (position, body) of the server's receipts
        self.steps_at: dict = {}  # index -> position of the step that played it

        next_step = None
        for position in range(self.before_at, len(self.messages)):
            envelope, body = self.messages[position], self.bodies[position]
            kind, commit = body.get("type"), envelope.get("commit")
            if (
                kind in ("hello", "joined", "room", "commitments", "before")
                and position != self.before_at
            ):
                self.fail(f"message {position}: a {kind} message during the game")
                continue
            if position > self.after_at and kind not in ("error", "received"):
                self.fail(f"message {position}: a {kind} message after the game ended")
                continue
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
            report.commitments[target] = commit.get("hash")
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
                self.steps_at[target - 1] = position
                forced = self.actions[target - 1].get("forced")
                expected = step_body(self.game, self.doc, self.color, forced=forced)
            elif kind == "after":
                if target != len(self.actions) or not is_over(self.game):
                    self.fail("the game ended before the replay did")
                if next_step != len(self.actions) + 1:
                    self.fail("steps were withheld: the after came early")
                expected = after_body(
                    self.game, self.doc, self.color, reveal=self.after["reveal"]
                )
            elif kind == "received":
                self.receipts.append((position, body))
                continue
            elif kind == "error":
                if body.get("answers") == "action" and position < self.after_at:
                    # During the game every action gets a receipt (one that
                    # names it), never an error that names nothing.
                    self.fail(
                        f"message {position}: an action answered with an error, "
                        "not a receipt"
                    )
                continue
            else:
                self.fail(f"message {position}: unexpected {kind!r} during the game")
                continue
            if not _same(body, _plain(expected)):
                self.fail(
                    f"message {position} ({kind} at state {target}): "
                    f"{_difference(body, _plain(expected))}"
                )
            report.count("views")
        if self.index != len(self.actions):
            self.fail("revealed actions past the last message")

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
        self.offered[index] = offered
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
        if seat["public_key"] == self.me:
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

    # ===== this client's own moves =====
    def check_own_moves(self) -> None:
        """This client's moves against what it signed and what the server
        acknowledged.

        Provable to anyone: the server signs a receipt for every submission
        it answers, and its messages to a recipient are in order. A receipt
        for a legal, correctly signed answer to decide k that comes before
        the step for move k proves the server had that answer before it
        played the move (the server never signs one between the deadline and
        that step); so does a receipt that says the answer was accepted,
        wherever it is. Playing anything else then is the server's doing.

        Not provable (warnings): that this client sent an answer the server
        never acknowledged (``sent`` is only the client's word), and when an
        answer arrived relative to the deadline (one acknowledged only after
        the move was played was late, or so the server says).
        """
        if not self.seated:
            return
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
        for index, items in signed.items():
            distinct = {digest(item["statement"]) for item in items.values()}
            if len(distinct) > 1:
                self.fail(
                    f"action {index}: this client signed {len(distinct)} different moves"
                )

        receipts: dict = {}
        for position, body in self.receipts:
            receipts.setdefault(body.get("submission"), []).append((position, body))

        for index, entry in enumerate(self.actions):
            if entry.get("color") != self.color:
                continue
            if self.forced_at.get(index) and entry.get("forced") == "only":
                continue
            if index not in self.decides:
                self.fail(
                    f"action {index}: this seat's move was made without asking it"
                )
                continue
            legal = {
                key: item
                for key, item in signed.get(index, {}).items()
                if self._legal_answer(index, item["statement"])
            }
            played = entry.get("statement") if entry.get("by") == "client" else None
            if not legal or any(
                played is not None and _same(item["statement"], played)
                for item in legal.values()
            ):
                continue
            step_at = self.steps_at.get(index, len(self.messages))
            seen = sorted(
                (p, body) for key in legal for p, body in receipts.get(key, [])
            )
            in_time = [p for p, body in seen if p < step_at]
            accepted = [p for p, body in seen if body.get("accepted") is True]
            if in_time or accepted:
                self.fail(
                    f"action {index}: the server had this client's legal, signed move "
                    f"(receipt in message {(in_time or accepted)[0]}) and played another"
                )
            elif not seen:
                self.report.warnings.append(
                    f"action {index}: this client says it sent a legal, signed move "
                    "the server never acknowledged, and the server played another "
                    "(not provable: only this client's word)"
                )
            else:
                self.report.warnings.append(
                    f"action {index}: this client's move reached the server after the "
                    "move was played (late)"
                )
        self.report.counts["own moves"] = sum(
            1 for a in self.actions if a.get("color") == self.color
        )

    def _legal_answer(self, index: int, statement: dict) -> bool:
        decide = self.decides.get(index)
        if decide is None:
            return False
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
    room (under one server key) against each other: one room has one seed
    commitment, one commitments message, one game (its id, seeds, reveals
    and reveal) or one void, and one state commitment at each point. Every
    transcript of a pair that disagrees fails, whatever the order given."""
    reports = [verify_transcript(t) for t in transcripts]
    groups: dict = {}
    for report in reports:
        if report.shared is not None:
            groups.setdefault(canonical(report.shared["group"]), []).append(report)
    for members in groups.values():
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
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
        if "aborted" in one and "game_id" not in one and "game_id" in other:
            problems.append("a void before the game and a game")
            break
        if "aborted" in one and "reveal" in other:
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
        except Exception as error:  # zlib.error, UnicodeError...: FAILED, not a crash
            loaded.append(None)
            unreadable[path] = f"cannot read: {type(error).__name__}: {error}"
    reports = verify_transcripts(loaded)
    for path, report in zip(args.transcripts, reports):
        if path in unreadable:
            report.errors = [unreadable[path]]
    if args.json:
        print(json.dumps([r.summary() for r in reports], indent=2))
    else:
        for path, report in zip(args.transcripts, reports):
            verdict = (
                "OK"
                if report.ok
                else (
                    "VOID"
                    if report.void is not None and not report.errors
                    else "FAILED"
                )
            )
            print(
                f"{path}: {verdict} (game {report.game_id}, {report.color or 'spectator'})"
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
            for error in report.errors:
                print(f"  error: {error}")
            for warning in report.warnings:
                print(f"  warning: {warning}")
    return 0 if all(r.ok for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
