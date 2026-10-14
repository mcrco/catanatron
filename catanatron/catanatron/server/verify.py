"""Check a game's transcript by replaying it in stock Catanatron.

A transcript is what one client received (every signed message, in order)
plus what it knows of itself (its public key and the seed it contributed).
:func:`verify_transcript` checks, for that client:

1. every message is signed by the server's key, numbered in order, chained
   to the previous one by hash, and (once seated) addressed to this client;
2. the server committed to its seed before this client contributed one
   (``hello`` precedes ``joined``), the revealed seed opens that commitment,
   and this client's seed is the one that went into the game's seed;
3. replaying the revealed actions from that seed in stock Catanatron, every
   action is legal, every state commitment opens to the replayed state with
   its revealed salt, and every message's body is *exactly* what an honest
   server builds from the replayed state for this recipient (its redacted
   view, its log entries, the playable actions it was offered, the board);
4. every move by a remote seat is signed by that seat's key (and every move
   by this client answers a ``decide`` it received at that point), and every
   move the server made on a seat's behalf is one it was allowed to make
   (the only legal action, or :func:`~catanatron.server.config.default_action`
   after a timeout).

``catanatron-verify T1.json [T2.json ...]`` runs it from the shell; given
several transcripts of one game it also checks that the server committed to
the same state at each point for everyone (no equivocation).
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from catanatron.serialization import action_from_json, state_to_json
from catanatron.server.config import (
    ConfigError,
    GameConfig,
    default_action,
    is_over,
    new_game,
)
from catanatron.server.crypto import (
    GENESIS,
    envelope_hash,
    is_hex,
    master_seed,
    seed_commitment,
    state_commitment,
    verify_envelope,
    verify_signature,
)
from catanatron.server.messages import (
    after_body,
    before_body,
    decide_body,
    step_body,
)

MAX_ERRORS = 50


@dataclass
class Report:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    counts: dict = field(default_factory=dict)
    game_id: Optional[str] = None
    color: Optional[str] = None
    commitments: dict = field(default_factory=dict)

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


def load_transcript(path: str) -> dict:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as file:  # type: ignore[operator]
        return json.load(file)


def verify_transcript(transcript: dict) -> Report:
    report = Report()
    try:
        _verify(transcript, report)
    except _Abort:
        pass
    except Exception as error:  # a malformed transcript must fail, not crash
        report.fail(f"malformed transcript: {type(error).__name__}: {error}")
    return report


def _verify(transcript: dict, report: Report) -> None:
    messages = [
        json.loads(m) if isinstance(m, str) else m for m in transcript["messages"]
    ]
    server_key = transcript["server_key"]
    me = transcript.get("public_key")
    my_seed = transcript.get("client_seed")
    seated = me is not None

    # 1. Signatures, sequence numbers, the hash chain, the recipient.
    head, joined_at = GENESIS, None
    for position, envelope in enumerate(messages):
        if not verify_envelope(server_key, envelope):
            report.fail(f"message {position}: bad server signature")
        if envelope.get("seq") != position or envelope.get("prev") != head:
            report.fail(f"message {position}: breaks the hash chain")
        head = envelope_hash(envelope)
        body = envelope["body"]
        if body.get("type") == "joined" and joined_at is None:
            joined_at = position
        if seated and joined_at is not None and envelope["to"] != me:
            report.fail(
                f"message {position}: addressed to {envelope['to']}, not this client"
            )
        report.count("signatures")
    if report.errors:
        raise _Abort()  # nothing below can be trusted
    bodies = [envelope["body"] for envelope in messages]

    # 2. Commitment before contribution.
    hello = bodies[0] if bodies else {}
    if hello.get("type") != "hello" or hello.get("server_key") != server_key:
        report.fail("the transcript does not start with the server's hello")
        raise _Abort()
    commitment = hello["room"]["seed_commitment"]
    for body in bodies:
        if body.get("type") in ("room", "before") and (
            body.get("room", body).get("seed_commitment") != commitment
        ):
            report.fail("the seed commitment changed during the game")
    if seated:
        if joined_at is None:
            report.fail("this client never joined a seat")
            raise _Abort()
        joined = bodies[joined_at]
        if joined["public_key"] != me or joined["client_seed"] != my_seed:
            report.fail(
                "the server acknowledged another key or seed than this client's"
            )

    befores = [i for i, b in enumerate(bodies) if b.get("type") == "before"]
    afters = [i for i, b in enumerate(bodies) if b.get("type") == "after"]
    if not befores:
        report.fail("the game never started")
        raise _Abort()
    if not afters:
        report.fail("the game has not ended: nothing was revealed yet")
        raise _Abort()
    before, after = bodies[befores[0]], bodies[afters[-1]]
    reveal = after["reveal"]
    report.game_id = before["game_id"]
    color = before["color"]
    report.color = color

    server_seed = reveal["server_seed"]
    if not is_hex(server_seed) or seed_commitment(server_seed) != commitment:
        report.fail("the revealed server seed does not open the seed commitment")
    seats, client_seeds = before["seats"], before["client_seeds"]
    if len(seats) != len(client_seeds):
        report.fail("seats and client seeds do not line up")
        raise _Abort()
    for seat, seed in zip(seats, client_seeds):
        if (seat["kind"] == "remote") != (seed is not None) or (
            seed is not None and not is_hex(seed, 32)
        ):
            report.fail(
                f"seat {seat['seat']}: a remote seat contributes one seed, a bot none"
            )
    if seated:
        mine = [s for s in seats if s["public_key"] == me]
        if len(mine) != 1 or mine[0]["color"] != color:
            report.fail("this client's key is not seated as its color")
            raise _Abort()
        if client_seeds[mine[0]["seat"]] != my_seed:
            report.fail("this client's seed did not go into the game's seed")
    elif color is not None:
        report.fail("a spectator's transcript carries a seat's view")

    try:
        config = GameConfig.from_json(before["config"])
    except ConfigError as error:
        report.fail(f"invalid config: {error}")
        raise _Abort()
    if _plain(config.to_json()) != before["config"]:
        report.fail("the config is not in its canonical form")
    by_color = {seat["color"]: seat for seat in seats}
    if [s["color"] for s in seats] != [c.value for c in config.colors]:
        report.fail("seats do not match the config's colors")
        raise _Abort()

    # 3./4. Replay.
    game = new_game(config, master_seed(server_seed, client_seeds), before["game_id"])
    actions, salts = reveal["actions"], reveal["salts"]
    if len(salts) != len(actions) + 1:
        report.fail(f"{len(salts)} salts revealed for {len(actions)} actions")
        raise _Abort()
    my_decides = {
        envelope["seq"]: envelope
        for envelope in messages
        if envelope["body"].get("type") == "decide"
    }
    answered = {}
    for sent in transcript.get("sent", []):
        decide = my_decides.get(sent["statement"].get("seq"))
        if decide is not None:
            answered[decide["body"]["index"]] = sent

    replay = {"index": 0, "doc": state_to_json(game)}

    def advance(target: int) -> None:
        if target < replay["index"] or target > len(actions):
            report.fail(f"a message refers to state {target}, out of order")
            raise _Abort()
        while replay["index"] < target:
            _apply(game, replay["index"], actions[replay["index"]])
            replay["index"] += 1
            replay["doc"] = state_to_json(game)

    def _apply(game, index: int, entry: dict) -> None:
        current = game.state.current_color().value
        if entry.get("index") != index or entry.get("color") != current:
            report.fail(f"action {index}: not {current}'s move")
            raise _Abort()
        try:
            action = action_from_json(entry["action"])
        except Exception:
            report.fail(f"action {index}: cannot parse {entry.get('action')!r}")
            raise _Abort()
        playable = list(game.playable_actions)
        _authorize(index, entry, action, playable, by_color[current])
        try:
            game.execute(action)
        except Exception as error:
            report.fail(f"action {index}: illegal in the replay ({error})")
            raise _Abort()
        report.count("actions")

    def _authorize(index, entry, action, playable, seat) -> None:
        forced, by = entry.get("forced"), entry.get("by")
        if forced == "only":
            if (
                not config.autoplay_forced
                or len(playable) != 1
                or action != playable[0]
            ):
                report.fail(
                    f"action {index}: claimed to be the only legal move, but is not"
                )
        elif forced == "timeout":
            if action != default_action(playable):
                report.fail(f"action {index}: a timeout must play the default action")
            if seat["public_key"] == me and index in answered:
                report.warnings.append(
                    f"action {index}: the server timed this client out although it "
                    "answered (late, or ignored)"
                )
        elif forced is not None:
            report.fail(f"action {index}: unknown forced reason {forced!r}")
        elif seat["kind"] == "bot":
            if by != "server":
                report.fail(f"action {index}: a server bot's move must be the server's")
        else:
            statement, signature = entry.get("statement"), entry.get("sig")
            if (
                by != "client"
                or not isinstance(statement, dict)
                or not verify_signature(seat["public_key"], statement, str(signature))
            ):
                report.fail(
                    f"action {index}: {seat['color']}'s move is not signed by its key"
                )
                return
            if (
                statement.get("action") != entry["action"]
                or statement.get("game_id") != before["game_id"]
            ):
                report.fail(f"action {index}: the signed statement is for another move")
            if seat["public_key"] == me:
                decide = my_decides.get(statement.get("seq"))
                if (
                    decide is None
                    or envelope_hash(decide) != statement.get("decide")
                    or decide["body"]["index"] != index
                ):
                    report.fail(
                        f"action {index}: this client's signed move answers a decide "
                        "it did not get at that point"
                    )
            report.count("client signatures")

    started = False
    for position, envelope in enumerate(messages):
        body, commit = envelope["body"], envelope["commit"]
        kind = body.get("type")
        started = started or kind == "before"
        if commit is None:
            if started:
                report.fail(f"message {position}: no state commitment during the game")
            continue
        # (A spectator arriving mid-game gets commitments from its hello on.)
        advance(commit["index"])
        if state_commitment(replay["doc"], salts[commit["index"]]) != commit["hash"]:
            report.fail(
                f"message {position}: the state commitment does not open to the "
                f"replayed state {commit['index']}"
            )
        report.commitments[commit["index"]] = commit["hash"]
        report.count("commitments")

        if kind == "before":
            expected = before_body(
                game,
                replay["doc"],
                color,
                config=before["config"],
                seats=seats,
                client_seeds=client_seeds,
                seed_commitment=commitment,
            )
        elif kind == "decide":
            if color is None or body.get("color") != color:
                report.fail(f"message {position}: a decide for another seat")
                continue
            expected = decide_body(
                game, replay["doc"], color, timeout_ms=config.decision_timeout_ms
            )
        elif kind == "step":
            index = commit["index"]
            forced = actions[index - 1].get("forced") if index > 0 else None
            expected = step_body(game, replay["doc"], color, forced=forced)
        elif kind == "after":
            if commit["index"] != len(actions) or not is_over(game):
                report.fail("the game ended before the replay did")
            expected = after_body(game, replay["doc"], color, reveal=reveal)
        else:
            continue
        if body != _plain(expected):
            report.fail(
                f"message {position} ({kind} at state {commit['index']}): "
                f"{_difference(body, _plain(expected))}"
            )
        report.count("views")

    if replay["index"] != len(actions):
        report.fail("revealed actions past the last message")
    if seated:
        report.counts["own moves"] = sum(
            1 for a in actions if by_color[a["color"]]["public_key"] == me
        )


def _difference(got: Any, want: Any, path: str = "") -> str:
    """Where two JSON values first differ, for error messages."""
    if isinstance(got, dict) and isinstance(want, dict):
        for key in sorted(set(got) | set(want)):
            if got.get(key, _MISSING) != want.get(key, _MISSING):
                return _difference(
                    got.get(key, _MISSING), want.get(key, _MISSING), f"{path}.{key}"
                )
    if isinstance(got, list) and isinstance(want, list) and len(got) == len(want):
        for i, (a, b) in enumerate(zip(got, want)):
            if a != b:
                return _difference(a, b, f"{path}[{i}]")
    return f"{path or 'body'} is {_short(got)}, an honest server sends {_short(want)}"


_MISSING = "<missing>"


def _short(value: Any) -> str:
    text = json.dumps(value)
    return text if len(text) < 80 else text[:77] + "..."


def verify_transcripts(transcripts: Sequence[dict]) -> list:
    """Verify each transcript, then check they saw one game committed alike."""
    reports = [verify_transcript(t) for t in transcripts]
    seen: dict = {}
    for report in reports:
        for index, digest in report.commitments.items():
            key = (report.game_id, index)
            if seen.setdefault(key, digest) != digest:
                report.fail(
                    f"equivocation: state {index} was committed differently to "
                    "another recipient"
                )
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
    reports = verify_transcripts([load_transcript(p) for p in args.transcripts])
    if args.json:
        print(json.dumps([r.summary() for r in reports], indent=2))
    else:
        for path, report in zip(args.transcripts, reports):
            verdict = "OK" if report.ok else "FAILED"
            print(
                f"{path}: {verdict} (game {report.game_id}, {report.color or 'spectator'})"
            )
            print(
                "  checked: " + ", ".join(f"{v} {k}" for k, v in report.counts.items())
            )
            for error in report.errors:
                print(f"  error: {error}")
            for warning in report.warnings:
                print(f"  warning: {warning}")
    return 0 if all(r.ok for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
