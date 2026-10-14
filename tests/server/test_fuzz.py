"""A mutation fuzzer for the verdict rules of PROTOCOL.md section 4.

1. A corpus of honest transcripts is recorded once from a real server
   whose key the test holds: seats, spectators from the start and from
   mid-game, lurkers; games finished, voided midway, voided before they
   started; a seat that reconnects, a seat whose every answer is
   duplicated on the wire.
2. Each transcript is mutated at random (a seeded generator):
   * server lies, re-signed with the server's key, each built so that it
     contradicts something provable (a game message's value where the seed
     is known, a state commitment, the void's seed, game or blame, the seed
     commitment, the commitments, a duplicated, missing or reordered game
     message, a message type, an address, a float);
   * client edits: the header (key, seed, room, seat, color...), what it
     says it sent, its notes;
   * network effects: the chain cut short (and, in the corpus, answers
     duplicated or resent after reconnects).
3. The verdict rules are asserted: an honest transcript has no errors; a
   server lie always has errors (and never because the verifier crashed);
   a client edit or a network effect never has errors; the verifier never
   raises.

Python and the browser must also read every value alike: the same
canonical JSON, the same hex, the same refusal of floats, unsafe integers
and lone surrogates (app.js runs under Node on the same generated corpus).

The default run uses a fixed seed and a small budget; ``-m fuzz`` (with
CATANATRON_FUZZ_BUDGET, default 2000) runs a larger one.
"""

import asyncio
import copy
import json
import os
import random
import shutil
import subprocess

import pytest

from catanatron.server.app import parse_request
from catanatron.server.client import GameClient, create_room
from catanatron.server.crypto import (
    SigningKey,
    canonical,
    is_hex,
    random_hex,
    seed_commitment,
)
from catanatron.server.room import RoomError
from catanatron.server.verify import verify_transcript

from tests.server.helpers import bot_client, resign, run, serving

SEED = 20261014
CONFIG = {"num_players": 2, "vps_to_win": 3, "decision_timeout_ms": 20_000}
APP_JS = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "catanatron",
    "catanatron",
    "server",
    "static",
    "app.js",
)
GAME = ("before", "decide", "step", "after")


# ===== the corpus =====
class DroppingClient(GameClient):
    """Its connection drops right after one answer: it rejoins and resends."""

    dropped = False

    async def _answer_with(self, action):
        await super()._answer_with(action)
        if not self.dropped and self._pending["body"]["index"] > 4:
            self.dropped = True
            await self.connection.close()


class EchoClient(GameClient):
    """Every answer goes out twice (the network duplicates it)."""

    async def _answer_with(self, action):
        await super()._answer_with(action)
        if self._answer is not None:
            await self.connection.send(json.dumps(self._answer))


def _lurker_transcript(key, room_id, messages):
    return {
        "server_key": key.public_hex,
        "room_id": room_id,
        "public_key": None,
        "client_seed": None,
        "messages": messages,
    }


async def _record(key):
    """Honest transcripts: [(name, transcript, client key or None)]."""
    from websockets.asyncio.client import connect

    corpus = []
    async with serving(key=key) as (server, base):
        # A finished game: RED reconnects once, BLUE's answers are doubled.
        room = await create_room(base, CONFIG, [None, None])
        url = f"{base}/room/{room['id']}"
        red, d1 = bot_client(
            url, "R", client_class=DroppingClient, seat=0, verify=False
        )
        blue, d2 = bot_client(url, "R", client_class=EchoClient, seat=1, verify=False)
        early = GameClient(url, spectate=True, transcript_dir=None, verify=False)
        tasks = [
            asyncio.create_task(red.play(d1)),
            asyncio.create_task(blue.play(d2)),
            asyncio.create_task(early.play()),
        ]
        room_obj = server.rooms[room["id"]]
        while len(room_obj.actions) < 8 and room_obj.status != "finished":
            await asyncio.sleep(0.01)
        late = GameClient(url, spectate=True, transcript_dir=None, verify=False)
        tasks.append(asyncio.create_task(late.play()))
        await asyncio.gather(*tasks)
        corpus += [
            ("finished-red", red.transcript(), red.key),
            ("finished-blue", blue.transcript(), blue.key),
            ("finished-spectator", early.transcript(), None),
            ("finished-late", late.transcript(), None),
        ]

        # A game voided midway (the server shutting down).
        room = await create_room(base, CONFIG, [None, "R"])
        url = f"{base}/room/{room['id']}"
        room_obj = server.rooms[room["id"]]
        red, d1 = bot_client(url, "R", verify=False)
        early = GameClient(url, spectate=True, transcript_dir=None, verify=False)
        tasks = [asyncio.create_task(red.play(d1)), asyncio.create_task(early.play())]
        while len(room_obj.actions) < 6:
            await asyncio.sleep(0.01)
        late = GameClient(url, spectate=True, transcript_dir=None, verify=False)
        tasks.append(asyncio.create_task(late.play()))
        async with connect(url) as lurker:
            lurked = [json.loads(await lurker.recv())]
            while len(room_obj.actions) < 16:
                await asyncio.sleep(0.01)
            room_obj.abort("the server is shutting down")
            lurked.append(json.loads(await asyncio.wait_for(lurker.recv(), 10)))
        await asyncio.gather(*tasks)
        corpus += [
            ("void-red", red.transcript(), red.key),
            ("void-spectator", early.transcript(), None),
            ("void-late", late.transcript(), None),
            ("void-lurker", _lurker_transcript(key, room["id"], lurked), None),
        ]
    return corpus


async def _record_withheld(key):
    """A room voided before its game: one seat withholds its reveal."""
    from catanatron.server.app import ServerLimits
    from catanatron.server.room import RoomLimits

    limits = ServerLimits(room=RoomLimits(reveal_timeout_ms=400))
    async with serving(key=key, limits=limits) as (server, base):
        room = await create_room(base, {**CONFIG, "num_players": 3}, [None, None, "R"])
        url = f"{base}/room/{room['id']}"
        honest = GameClient(url, seat=0, transcript_dir=None, verify=False)
        silent = GameClient(url, seat=1, transcript_dir=None, verify=False)
        real = silent._handle

        async def withhold(ws, body, decide):
            if body.get("type") == "commitments":
                silent.commitments = body
                return None
            return await real(ws, body, decide)

        silent._handle = withhold
        watcher = GameClient(url, spectate=True, transcript_dir=None, verify=False)
        await asyncio.gather(
            honest.play(lambda v, p: p[0]),
            silent.play(lambda v, p: p[0]),
            watcher.play(),
        )
        return [
            ("withheld-seat", honest.transcript(), honest.key),
            ("withheld-spectator", watcher.transcript(), None),
        ]


@pytest.fixture(scope="module")
def corpus():
    key = SigningKey.generate()
    recorded = run(_record(key)) + run(_record_withheld(key))
    return key, [(name, json.loads(json.dumps(t)), k) for name, t, k in recorded]


def test_the_honest_corpus_verifies(corpus):
    _, transcripts = corpus
    for name, transcript, _ in transcripts:
        report = verify_transcript(transcript)
        assert not report.errors and not report.invalid, (
            name,
            report.errors,
            report.invalid,
        )
        expected = "OK" if name.startswith("finished") else "VOID"
        assert report.verdict == expected, (name, report.verdict)


# ===== mutations =====
def _kinds(transcript):
    return [m["body"].get("type") for m in transcript["messages"]]


def _seed_known(transcript):
    kinds = _kinds(transcript)
    if "before" not in kinds:
        return False
    return "after" in kinds or (
        "aborted" in kinds and kinds.index("aborted") > kinds.index("before")
    )


def _leaves(value, path=()):
    """Every (path, leaf) in a JSON value; lists and dicts are leaves too
    (they can lose or swap elements)."""
    yield path, value
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _leaves(value[key], path + (key,))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _leaves(item, path + (i,))


def _set(root, path, value):
    target = root
    for step in path[:-1]:
        target = target[step]
    target[path[-1]] = value


def _changed(rng, value):
    """Another value of (mostly) the same kind."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + rng.choice([-2, -1, 1, 2, 7])
    if isinstance(value, str):
        if is_hex(value) and value:
            return random_hex(len(value) // 2) if len(value) >= 2 else "00"
        return value + rng.choice(["x", "Z", "é"]) if value else "x"
    if value is None:
        return rng.choice([0, "x", False])
    if isinstance(value, list):
        if len(value) >= 2 and rng.random() < 0.5:
            i, j = rng.sample(range(len(value)), 2)
            out = list(value)
            out[i], out[j] = out[j], out[i]
            return out
        return value[:-1] if value else [0]
    if isinstance(value, dict):
        if value and rng.random() < 0.5:
            out = dict(value)
            out.pop(rng.choice(sorted(out)))
            return out
        return {**value, "zz": 1}
    return "x"


def _mutate_leaf(rng, body, allowed=lambda path: True):
    candidates = [(p, v) for p, v in _leaves(body) if p and allowed(p)]
    rng.shuffle(candidates)
    for path, value in candidates[:20]:
        new = _changed(rng, value)
        mutated = copy.deepcopy(body)
        _set(mutated, path, new)
        if canonical(mutated) != canonical(body):
            return mutated, "/".join(map(str, path))
    return None, None


def _positions(transcript, kinds):
    return [i for i, k in enumerate(_kinds(transcript)) if k in kinds]


# Each lie returns (messages, description) or None when it does not apply.
def lie_game_value(rng, t):
    if not _seed_known(t):
        return None
    kinds = _kinds(t)
    end = kinds.index("aborted") if "after" not in kinds else len(kinds)
    spots = [i for i in _positions(t, GAME) if i < end]
    i = rng.choice(spots)
    messages = copy.deepcopy(t["messages"])
    body, where = _mutate_leaf(rng, messages[i]["body"], lambda p: p[0] != "type")
    if body is None:
        return None
    messages[i]["body"] = body
    return messages, f"message {i} {kinds[i]} {where}"


def lie_commit(rng, t):
    if "after" not in _kinds(t):
        return None
    i = rng.choice(_positions(t, GAME))
    messages = copy.deepcopy(t["messages"])
    messages[i]["commit"]["hash"] = random_hex()
    return messages, f"message {i} commit"


def lie_void(rng, t):
    spots = _positions(t, ("aborted",))
    if not spots:
        return None
    i = spots[0]
    messages = copy.deepcopy(t["messages"])
    body = messages[i]["body"]
    kinds = _kinds(t)
    started = "before" in kinds and kinds.index("before") < i
    choices = ["seed"]
    if started or t.get("public_key") is not None:
        choices.append("game")
    if started:
        choices.append("blame")
    what = rng.choice(choices)
    if what == "seed":
        body["server_seed"] = random_hex()
    elif what == "game":
        body["game_id"] = "another-game"
    else:
        before = messages[kinds.index("before")]["body"]
        remote = [s["seat"] for s in before["seats"] if s["kind"] == "remote"]
        body["blame"] = sorted(set(body["blame"]) | {rng.choice(remote)})
    return messages, f"void {what}"


def lie_seed_commitment(rng, t):
    messages = copy.deepcopy(t["messages"])
    messages[0]["body"]["room"]["seed_commitment"] = seed_commitment(random_hex())
    return messages, "hello seed commitment"


def lie_commitments(rng, t):
    spots = _positions(t, ("commitments",))
    if not spots or "before" not in _kinds(t):
        return None
    i = spots[0]
    messages = copy.deepcopy(t["messages"])
    body, where = _mutate_leaf(rng, messages[i]["body"], lambda p: p[0] != "type")
    if body is None:
        return None
    messages[i]["body"] = body
    return messages, f"commitments {where}"


def lie_duplicate(rng, t):
    once = ("hello", "joined", "commitments", "before", "after", "aborted", "decide")
    spots = _positions(t, once)
    if _seed_known(t):
        spots += _positions(t, ("step",))
    i = rng.choice(spots)
    messages = copy.deepcopy(t["messages"])
    messages.insert(i + 1, copy.deepcopy(messages[i]))
    return messages, f"message {i} twice"


def lie_omit(rng, t):
    if not _seed_known(t):
        return None
    kinds = _kinds(t)
    end = kinds.index("after") if "after" in kinds else kinds.index("aborted")
    spots = [i for i in _positions(t, ("step",)) if i < end - 1]
    if t.get("public_key") is not None:
        stepped = {
            t["messages"][i]["body"]["index"] - 1 for i in _positions(t, ("step",))
        }
        spots += [
            i
            for i in _positions(t, ("decide",))
            if i < end and t["messages"][i]["body"]["index"] in stepped
        ]
    if not spots:
        return None
    i = rng.choice(spots)
    messages = copy.deepcopy(t["messages"])
    del messages[i]
    return messages, f"message {i} ({kinds[i]}) dropped"


def lie_reorder(rng, t):
    if not _seed_known(t):
        return None
    kinds = _kinds(t)
    end = kinds.index("after") if "after" in kinds else kinds.index("aborted")
    spots = [
        i
        for i in range(kinds.index("before") + 1, end - 1)
        if kinds[i] in ("step", "decide") and kinds[i + 1] in ("step", "decide")
    ]
    if not spots:
        return None
    i = rng.choice(spots)
    messages = copy.deepcopy(t["messages"])
    messages[i], messages[i + 1] = messages[i + 1], messages[i]
    return messages, f"messages {i} and {i + 1} swapped"


def lie_type(rng, t):
    i = rng.randrange(1, len(t["messages"]))
    messages = copy.deepcopy(t["messages"])
    messages[i]["body"]["type"] = rng.choice([[], {}, 5, None, "restart", "x"])
    return messages, f"message {i} type"


def lie_address(rng, t):
    if t.get("public_key") is None:
        return None
    kinds = _kinds(t)
    spots = list(range(kinds.index("joined") + 1, len(kinds)))
    if not spots:
        return None
    i = rng.choice(spots)
    messages = copy.deepcopy(t["messages"])
    messages[i]["to"] = SigningKey.generate().public_hex
    return messages, f"message {i} addressed elsewhere"


def lie_float(rng, t):
    spots = [
        (i, p)
        for i, m in enumerate(t["messages"])
        for p, v in _leaves(m["body"])
        if type(v) is int and p
    ]
    i, path = rng.choice(spots)
    messages = copy.deepcopy(t["messages"])
    target = messages[i]["body"]
    for step in path[:-1]:
        target = target[step]
    target[path[-1]] = float(target[path[-1]]) + 0.5
    return messages, f"message {i} float at {path}"


LIES = [
    lie_game_value,
    lie_game_value,
    lie_game_value,
    lie_commit,
    lie_void,
    lie_seed_commitment,
    lie_commitments,
    lie_duplicate,
    lie_omit,
    lie_reorder,
    lie_type,
    lie_address,
    lie_float,
]


def edit_header(rng, t, key):
    field = rng.choice(
        [
            "client_seed",
            "public_key",
            "room_id",
            "seat",
            "color",
            "format",
            "url",
            "reveal_sig",
        ]
    )
    edited = copy.deepcopy(t)
    edited[field] = rng.choice(
        [random_hex(), None, 7, "x", SigningKey.generate().public_hex]
    )
    return edited, f"header {field}"


def edit_sent(rng, t, key):
    edited = copy.deepcopy(t)
    sent = edited.setdefault("sent", [])
    what = rng.choice(["drop", "double", "junk", "forged", "resigned", "tampered"])
    if what == "drop" and sent:
        del sent[rng.randrange(len(sent))]
    elif what == "double" and sent:
        sent.append(copy.deepcopy(rng.choice(sent)))
    elif what == "junk":
        sent.append(
            rng.choice([7, "x", None, {"statement": 5}, {"statement": {"index": "a"}}])
        )
    elif what == "forged":
        other = SigningKey.generate()
        statement = {"index": rng.randrange(50), "action": ["RED", "END_TURN", None]}
        sent.append({"statement": statement, "sig": other.sign(statement)})
    elif what == "resigned" and key is not None and sent:
        statement = copy.deepcopy(rng.choice(sent)["statement"])
        statement["action"] = ["RED", "END_TURN", None]
        sent.append({"statement": statement, "sig": key.sign(statement)})
    elif what == "tampered" and sent:
        item = rng.choice(sent)
        item["statement"] = {
            **item["statement"],
            "index": item["statement"]["index"] + 1,
        }
    else:
        return None
    return edited, f"sent {what}"


def edit_notes(rng, t, key):
    edited = copy.deepcopy(t)
    edited["unanswered"] = rng.choice(
        [
            [{"index": rng.randrange(40), "ms": rng.randrange(5)}],
            "x",
            [5, None, {"index": "a"}],
        ]
    )
    return edited, "notes"


def network_cut(rng, t, key):
    edited = copy.deepcopy(t)
    edited["messages"] = edited["messages"][: rng.randrange(1, len(edited["messages"]))]
    return edited, "cut"


EDITS = [edit_header, edit_sent, edit_sent, edit_notes]
NETWORK = [network_cut]


def _fuzz(corpus, budget, seed):
    key, transcripts = corpus
    rng = random.Random(seed)
    findings = []
    counts = {"lie": 0, "edit": 0, "network": 0}
    for n in range(budget):
        name, transcript, client_key = rng.choice(transcripts)
        roll = rng.random()
        if roll < 0.5:
            made = rng.choice(LIES)(rng, transcript)
            if made is None:
                continue
            messages, what = made
            mutated = dict(transcript, messages=resign(key, messages))
            report = verify_transcript(json.loads(json.dumps(mutated)))
            counts["lie"] += 1
            crashed = [e for e in report.errors if "cannot read" in e]
            if not report.errors or (crashed and "type" not in what):
                findings.append(("lie", name, what, report.verdict, report.errors[:2]))
        else:
            kind = "edit" if roll < 0.85 else "network"
            made = rng.choice(EDITS if kind == "edit" else NETWORK)(
                rng, transcript, client_key
            )
            if made is None:
                continue
            mutated, what = made
            report = verify_transcript(json.loads(json.dumps(mutated)))
            counts[kind] += 1
            if report.errors:
                findings.append((kind, name, what, report.verdict, report.errors[:2]))
    return findings, counts


def test_the_verdict_rules_hold_under_mutation(corpus):
    findings, counts = _fuzz(
        corpus, budget=int(os.environ.get("CATANATRON_FUZZ_QUICK", 160)), seed=SEED
    )
    assert counts["lie"] > 30 and counts["edit"] > 20 and counts["network"] > 5, counts
    assert not findings, findings[:10]


@pytest.mark.fuzz
def test_the_verdict_rules_hold_under_a_large_budget(corpus):
    budget = int(os.environ.get("CATANATRON_FUZZ_BUDGET", 2000))
    seed = int(os.environ.get("CATANATRON_FUZZ_SEED", SEED + 1))
    findings, counts = _fuzz(corpus, budget=budget, seed=seed)
    assert not findings, findings[:10]


# ===== Python and the browser read every value alike =====
PARITY_HARNESS = r"""
const fs = require("fs");
const vm = require("vm");
globalThis.window = { crypto: globalThis.crypto, addEventListener() {} };
globalThis.document = { querySelector() { return null; } };
globalThis.location = { hash: "", protocol: "http:", host: "localhost" };
vm.runInThisContext(fs.readFileSync(process.argv[process.argv.length - 1], "utf8"));
vm.runInThisContext(`
  const input = JSON.parse(fs.readFileSync(0, "utf8"));
  const out = {
    canonical: input.values.map((v) => canonical(v)),
    hex: input.hexes.map((h) => [HEX64.test(h), NONCE.test(h)]),
    numbers: input.texts.map((t) => numbersOk(t)),
    strings: input.texts.map((t) => { try { return stringsOk(JSON.parse(t)); } catch (e) { return null; } }),
  };
  process.stdout.write(JSON.stringify(out));
`);
"""


def _text(rng):
    pieces = []
    for _ in range(rng.randrange(0, 6)):
        pick = rng.random()
        if pick < 0.3:
            pieces.append(chr(rng.randrange(0x20, 0x7F)))
        elif pick < 0.45:
            pieces.append(chr(rng.randrange(0x00, 0x20)))
        elif pick < 0.6:
            pieces.append(rng.choice(['"', "\\", "/", " ", " ", "\x7f"]))
        elif pick < 0.8:
            pieces.append(chr(rng.randrange(0xA0, 0xD7FF)))
        elif pick < 0.95:
            pieces.append(chr(rng.randrange(0x10000, 0x10FFFF)))
        else:
            pieces.append(chr(rng.randrange(0xE000, 0xFFFF)))
    return "".join(pieces)


def _value(rng, depth=0):
    pick = rng.random()
    if depth > 3 or pick < 0.3:
        return rng.choice(
            [
                rng.randrange(-(2**53) + 1, 2**53),
                rng.randrange(-1000, 1000),
                2**53 - 1,
                -(2**53) + 1,
                0,
                True,
                False,
                None,
                _text(rng),
            ]
        )
    if pick < 0.6:
        return [_value(rng, depth + 1) for _ in range(rng.randrange(0, 4))]
    return {_text(rng): _value(rng, depth + 1) for _ in range(rng.randrange(0, 4))}


def _hexish(rng):
    n = rng.choice([16, 32])
    digits = random_hex(n)
    return rng.choice(
        [
            digits,
            " " + digits,
            digits + " ",
            digits[:-1],
            digits.upper(),
            digits[:10] + " " + digits[10:-1],
            digits[:-2] + "\n",
            digits[:-1] + "g",
            "",
        ]
    )


def _wire_text(rng):
    """A JSON object as a server (or a client) might put it on the wire."""
    value = {"type": "x", "v": _value(rng)}
    text = json.dumps(value)
    pick = rng.random()
    if pick < 0.2:
        return text.replace("}", ', "f": 5.0}', 1)
    if pick < 0.3:
        return text.replace("}", ', "f": 1e3}', 1)
    if pick < 0.45:
        return text.replace("}", f', "f": {2**53 + rng.randrange(1, 1000)}}}', 1)
    if pick < 0.55:
        return text.replace("}", ', "f": "\\ud800x"}', 1)
    if pick < 0.6:
        return text.replace("}", ', "f": "\\udc00"}', 1)
    return text


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_python_and_the_browser_read_every_value_alike():
    rng = random.Random(SEED)
    values = [_value(rng) for _ in range(300)]
    hexes = [_hexish(rng) for _ in range(200)]
    texts = [_wire_text(rng) for _ in range(300)]
    result = subprocess.run(
        ["node", "-e", PARITY_HARNESS, APP_JS],
        input=json.dumps({"values": values, "hexes": hexes, "texts": texts}),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    for value, js in zip(values, out["canonical"]):
        assert canonical(value).decode() == js, value
    for text, (js64, js16) in zip(hexes, out["hex"]):
        assert (is_hex(text, 32), is_hex(text, 16)) == (js64, js16), repr(text)
    for text, numbers_ok, strings_ok in zip(texts, out["numbers"], out["strings"]):
        try:
            parse_request(text)
            python_ok = True
        except RoomError:
            python_ok = False
        assert python_ok == (numbers_ok and strings_ok is True), text
