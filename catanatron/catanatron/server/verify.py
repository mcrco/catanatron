"""Check a game's transcript by replaying it in stock Catanatron.

A transcript is what one client received (every signed message, in order)
plus what it knows of itself (its public key, its seed, what it signed).
:func:`verify_transcript` checks, for that client:

1. every message is signed by the server's key, numbered in order, chained
   to the previous one by hash, and (once seated) addressed to this client;
2. the seeds: the server committed to its seed in the ``hello``; this client
   joined with a commitment to its own seed; the ``commitments`` message
   fixed every seat, key and seed commitment and the config (the ones the
   ``hello`` advertised) before any seed was revealed; every revealed seed
   opens its commitment, this client's included; the revealed server seed
   opens the server's;
3. the game is the committed one: ``before`` has exactly the committed
   config and seats, and the master seed is derived from all of it;
4. replaying the revealed actions from that seed in stock Catanatron, every
   action is one the game offered, every state commitment opens to the
   replayed state with its revealed salt, and every message body is
   *exactly* what an honest server builds from the replayed state for this
   recipient;
5. the schedule: one ``before``, then one ``step`` per action in order, a
   ``decide`` exactly when this seat had a move that was not publicly
   forced (and at most one per move), one ``after`` last;
6. every move by a remote seat is signed by that seat's key over the room,
   the game and the action's index (so a signature cannot be replayed for
   another move); every move of this client answers the ``decide`` it got
   at that point; every move the server made for a seat is one it may make
   (the only legal one, when everyone could see it was; or the default
   after a timeout); and no move of this client was refused although it
   was legal.

``catanatron-verify T1.json [T2.json ...]`` runs it from the shell; given
several transcripts of one game it also checks that every recipient got the
same game: the same ``before`` seats, config and seeds, the same reveal, and
the same state commitment at each point (no equivocation).
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
    master_seed,
    seed_commitment,
    state_commitment,
    verify_envelope,
    verify_signature,
)
from catanatron.server.messages import (
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
    #: What every recipient of this game must have been told alike.
    shared: Optional[dict] = None

    @property
    def ok(self) -> bool:
        return not self.errors

    def fail(self, message: str) -> None:
        if len(self.errors) < MAX_ERRORS:
            self.errors.append(message)

    def count(self, what: str) -> None:
        self.counts[what] = self.counts.get(what, 0) + 1

    def summary(self) -> dict:
        return {
            "ok": self.ok,
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
        self.server_commitment = room.get("seed_commitment")
        self.config_json = room.get("config")
        self.require(isinstance(self.room_id, str), "the room has no id")
        self.require(
            is_hex(self.server_commitment, 32), "the room has no seed commitment"
        )
        if self.transcript.get("room_id") not in (None, self.room_id):
            self.fail("the transcript is for another room than its messages")
        try:
            self.config = GameConfig.from_json(self.config_json)
        except ConfigError as error:
            self.fail(f"invalid config: {error}")
            raise _Abort()
        if not _same(self.config.to_json(), self.config_json):
            self.fail("the config is not in its canonical form")

        kinds = [b.get("type") for b in self.bodies]
        for body in self.bodies[1:]:
            if body.get("type") == "room":
                r = body.get("room") or {}
                if (
                    r.get("id") != self.room_id
                    or r.get("seed_commitment") != self.server_commitment
                    or not _same(r.get("config"), self.config_json)
                ):
                    self.fail(
                        "a room message changed the room, its rules or its seed commitment"
                    )

        aborted = [b for b in self.bodies if b.get("type") == "aborted"]
        if aborted:
            body = aborted[0]
            revealed = body.get("server_seed")
            if revealed is not None and (
                not is_hex(revealed, 32)
                or seed_commitment(revealed) != self.server_commitment
            ):
                self.fail(
                    "the aborted message reveals a seed that does not open the commitment"
                )
            self.fail(
                f"the game was aborted: {body.get('reason')!r}"
                + (f" (blamed seats: {body.get('blame')})" if body.get("blame") else "")
            )
            raise _Abort()

        positions = {
            kind: [i for i, k in enumerate(kinds) if k == kind] for kind in set(kinds)
        }
        self.require(positions.get("hello") == [0], "a second hello in the chain")
        befores = positions.get("before", [])
        afters = positions.get("after", [])
        self.require(
            len(befores) == 1, f"{len(befores)} before messages (one is honest)"
        )
        self.require(
            len(afters) == 1, "the game has not ended: nothing was revealed yet"
        )
        self.before_at, self.after_at = befores[0], afters[0]
        commitments = positions.get("commitments", [])
        self.require(len(commitments) <= 1, "more than one commitments message")

        self.committed = None
        if self.seated:
            joins = positions.get("joined", [])
            self.require(len(joins) == 1, "this client was not seated exactly once")
            joined = self.bodies[joins[0]]
            mine = client_seed_commitment(self.room_id, self.me, self.my_seed)
            if (
                joined.get("room_id") != self.room_id
                or joined.get("public_key") != self.me
                or joined.get("client_commitment") != mine
            ):
                self.fail(
                    "the server acknowledged another key or seed commitment than this client's"
                )
            self.require(
                len(commitments) == 1 and joins[0] < commitments[0] < self.before_at,
                "the seats' seed commitments were not published between joining and the game",
            )
            self.my_seat = joined.get("seat")
        if commitments:
            self.committed = self.bodies[commitments[0]]
            c = self.committed
            if (
                c.get("room_id") != self.room_id
                or c.get("seed_commitment") != self.server_commitment
                or not _same(c.get("config"), self.config_json)
            ):
                self.fail(
                    "the commitments change the room, its rules or its seed commitment"
                )
            self.check_seats(c.get("seats"), "commitments")

        before = self.bodies[self.before_at]
        self.before = before
        self.require(
            before.get("room_id") == self.room_id
            and isinstance(before.get("game_id"), str),
            "the before message names another room, or no game",
        )
        self.report.game_id = before["game_id"]
        if not _same(before.get("config"), self.config_json):
            self.fail("the game's config is not the one the room advertised")
        if before.get("seed_commitment") != self.server_commitment:
            self.fail("the seed commitment changed during the game")
        self.seats = before.get("seats")
        self.check_seats(self.seats, "before")
        if self.committed is not None and not _same(
            self.seats, self.committed.get("seats")
        ):
            self.fail(
                "the game's seats are not the ones committed before the seeds were revealed"
            )

        seeds = before.get("client_seeds")
        self.require(
            isinstance(seeds, list) and len(seeds) == len(self.seats),
            "seats and client seeds do not line up",
        )
        self.client_seeds = seeds
        for seat, seed in zip(self.seats, seeds):
            if seat["kind"] == "remote":
                if (
                    not is_hex(seed, 32)
                    or client_seed_commitment(self.room_id, seat["public_key"], seed)
                    != seat["client_commitment"]
                ):
                    self.fail(
                        f"seat {seat['seat']}'s seed does not open its commitment"
                    )
            elif seed is not None:
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
            or seed_commitment(self.server_seed) != self.server_commitment
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
        self.report.shared = {
            "before": {
                k: before.get(k)
                for k in (
                    "room_id",
                    "game_id",
                    "config",
                    "seats",
                    "client_seeds",
                    "seed_commitment",
                )
            },
            "reveal": reveal,
        }

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
                keys.append(seat.get("public_key"))
            elif seat.get("kind") == "bot":
                ok = (
                    ok
                    and seat.get("public_key") is None
                    and seat.get("client_commitment") is None
                )
            else:
                ok = False
            self.require(ok, f"{where}: seat {index} is malformed")
        self.require(len(keys) == len(set(keys)), f"{where}: two seats share a key")

    # ===== 4./5./6. the replay =====
    def replay(self) -> None:
        report = self.report
        master = master_seed(
            self.server_seed,
            self.client_seeds,
            {"room_id": self.room_id, "config": self.config_json, "seats": self.seats},
        )
        self.game = new_game(self.config, master, self.before["game_id"])
        self.index = 0
        self.doc = state_to_json(self.game)
        self.by_color = {seat["color"]: seat for seat in self.seats}
        self.last_seq: dict = {}
        self.decides: dict = {}  # index -> envelope
        self.offered: dict = {}  # index -> playable actions (JSON) at that point
        self.forced_at: dict = {}  # index -> whether publicly forced
        self.errors_seen: list = []  # (position, body) of errors during the game

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
            if position > self.after_at and kind != "error":
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
                    room_id=self.room_id,
                    config=self.config_json,
                    seats=self.seats,
                    client_seeds=self.client_seeds,
                    seed_commitment=self.server_commitment,
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
            elif kind == "after":
                if target != len(self.actions) or not is_over(self.game):
                    self.fail("the game ended before the replay did")
                if next_step != len(self.actions) + 1:
                    self.fail("steps were withheld: the after came early")
                expected = after_body(
                    self.game, self.doc, self.color, reveal=self.after["reveal"]
                )
            elif kind == "error":
                self.errors_seen.append((position, body))
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
        expected = action_statement(
            self.room_id,
            self.before["game_id"],
            index,
            seq,
            decide,
            entry.get("action"),
        )
        if (
            not _same(statement, expected)
            or type(seq) is not int
            or not is_hex(decide, 32)
        ):
            self.fail(
                f"action {index}: the signed statement is for another move, game or room"
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
        if not self.seated:
            return
        sent = self.transcript.get("sent") or []
        statements: dict = {}
        for item in sent:
            statement = item.get("statement") if isinstance(item, dict) else None
            if isinstance(statement, dict) and type(statement.get("index")) is int:
                statements.setdefault(statement["index"], {})[
                    digest(statement)
                ] = statement
        for index, signed in statements.items():
            if len(signed) > 1:
                self.fail(
                    f"action {index}: this client signed {len(signed)} different moves"
                )

        refused = {
            body.get("refused"): position
            for position, body in self.errors_seen
            if body.get("code") == "bad_action" and is_hex(body.get("refused"), 32)
        }
        for index, signed in statements.items():
            for key, statement in signed.items():
                if key in refused and self._legal_answer(index, statement):
                    self.fail(
                        f"action {index}: the server refused this client's legal, "
                        f"correctly signed move (message {refused[key]})"
                    )

        for index, entry in enumerate(self.actions):
            if entry.get("color") != self.color:
                continue
            if self.forced_at.get(index) and entry.get("forced") == "only":
                continue
            if index not in self.decides:
                self.fail(
                    f"action {index}: this seat's move was made without asking it"
                )
            elif entry.get("forced") == "timeout" and any(
                self._legal_answer(index, s) for s in statements.get(index, {}).values()
            ):
                self.report.warnings.append(
                    f"action {index}: the server timed this client out although it "
                    "answered (late, or ignored)"
                )
        if self.seated:
            self.report.counts["own moves"] = sum(
                1 for a in self.actions if a.get("color") == self.color
            )

    def _legal_answer(self, index: int, statement: dict) -> bool:
        decide = self.decides.get(index)
        if decide is None:
            return False
        expected = action_statement(
            self.room_id,
            self.before["game_id"],
            index,
            decide["seq"],
            envelope_hash(decide),
            statement.get("action"),
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
    """Verify each transcript, then check they saw one game alike."""
    reports = [verify_transcript(t) for t in transcripts]
    seen: dict = {}
    shared: dict = {}
    for report in reports:
        game = (
            report.game_id if isinstance(report.game_id, str) else repr(report.game_id)
        )
        for index, digest_ in report.commitments.items():
            if seen.setdefault((game, index), digest_) != digest_:
                report.fail(
                    f"equivocation: state {index} was committed differently to another recipient"
                )
        if report.shared is not None:
            first = shared.setdefault(game, report.shared)
            if not _same(first["before"], report.shared["before"]):
                report.fail(
                    "equivocation: another recipient got other seats, config or seeds"
                )
            if not _same(first["reveal"], report.shared["reveal"]):
                report.fail("equivocation: another recipient got another reveal")
    return reports


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
        except (OSError, ValueError, RecursionError, EOFError) as error:
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
            verdict = "OK" if report.ok else "FAILED"
            print(
                f"{path}: {verdict} (game {report.game_id}, {report.color or 'spectator'})"
            )
            if report.counts:
                print(
                    "  checked: "
                    + ", ".join(f"{v} {k}" for k, v in report.counts.items())
                )
            for error in report.errors:
                print(f"  error: {error}")
            for warning in report.warnings:
                print(f"  warning: {warning}")
    return 0 if all(r.ok for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
