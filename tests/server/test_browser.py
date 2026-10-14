"""The browser client's crypto agrees with the server's, byte for byte.

Runs static/app.js under Node (skipped without it) with just enough of a
browser stubbed out to load it, then checks its canonical JSON, its hash
chain and its Ed25519 signatures against the Python side on a real game.
"""

import asyncio
import json
import os
import shutil
import subprocess
import urllib.request

import pytest

from catanatron.server.client import GameClient, create_room, list_rooms
from catanatron.server.crypto import envelope_hash, verify_signature
from catanatron.server.verify import verify_transcript

from tests.server.helpers import run, serving

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

NODE_HARNESS = r"""
const fs = require("fs");
const vm = require("vm");
globalThis.window = { crypto: globalThis.crypto, addEventListener() {} };
globalThis.document = { querySelector() { return null; } };
globalThis.location = { hash: "", protocol: "http:", host: "localhost" };
vm.runInThisContext(fs.readFileSync(process.argv[process.argv.length - 1], "utf8"));
const input = JSON.parse(fs.readFileSync(0, "utf8"));
(async () => {
  const out = { hashes: [], valid: [], identity: null, signature: null };
  for (const env of input.envelopes) {
    out.hashes.push(await sha256Hex(canonical(headOf(env))));
    out.valid.push(await verifyEnvelope(input.server_key, env));
  }
  const identity = await newIdentity();
  out.identity = identity.publicHex;
  out.signature = await signHex(identity.privateKey, input.statement);
  out.canonical = canonical(input.statement);
  process.stdout.write(JSON.stringify(out));
})();
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_the_browser_hashes_verifies_and_signs_like_the_server():
    async def scenario():
        async with serving() as (server, base):
            room = await create_room(
                base, {"num_players": 2, "vps_to_win": 4}, ["R", "R"]
            )
            watcher = GameClient(
                f"{base}/room/{room['id']}", spectate=True, transcript_dir=None
            )
            await watcher.play()
            return watcher

    watcher = run(scenario())
    envelopes = watcher.messages[:40]
    statement = {
        "game_id": "g",
        "seq": 3,
        "decide": "ab" * 32,
        "action": ["RED", "MOVE_ROBBER", [[0, -1, 1], "BLUE"]],
        "note": "é ü — unicode survives canonicalization",
    }
    payload = {
        "envelopes": envelopes,
        "server_key": watcher.server_key,
        "statement": statement,
    }
    result = subprocess.run(
        ["node", "-e", NODE_HARNESS, APP_JS],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    assert out["hashes"] == [envelope_hash(e) for e in envelopes]
    assert all(out["valid"])
    assert verify_signature(out["identity"], statement, out["signature"])


SESSION_HARNESS = r"""
const fs = require("fs");
const vm = require("vm");
const [host, roomId, seat, appJs] = process.argv.slice(-4);
const store = new Map();
globalThis.window = { crypto: globalThis.crypto, addEventListener() {} };
globalThis.document = { querySelector() { return null; } };
globalThis.location = { hash: "", protocol: "http:", host };
globalThis.localStorage = {
  getItem: (k) => (store.has(k) ? store.get(k) : null),
  setItem: (k, v) => store.set(k, String(v)),
  removeItem: (k) => store.delete(k),
};
vm.runInThisContext(fs.readFileSync(appJs, "utf8"));
vm.runInThisContext(`
  let asked = false, dropped = false;
  render = function () {
    if (session.room && !session.seated && !asked) { asked = true; session.takeSeat(${seat}); }
    if (session.problems.length) { console.error(session.problems); process.exit(2); }
    if (session.game && session.game.index > 30 && !dropped) { dropped = true; session.ws.close(); }
    if (session.pending && !session.answered) {
      const actions = session.pending.body.playable_actions;
      session.act(actions[actions.length - 1]);
    }
    if (session.game && session.game.over) {
      session.close();
      process.stdout.write(JSON.stringify(session.transcript()), () => process.exit(0));
      render = () => {};
    }
  };
  session = new Session(${JSON.stringify(roomId)});
  session.connect();
`);
"""


def _node_has_websocket():
    if shutil.which("node") is None:
        return False
    probe = subprocess.run(
        ["node", "--experimental-websocket", "-e", "new WebSocket('ws://127.0.0.1:9')"],
        capture_output=True,
        timeout=30,
    )
    return (
        probe.returncode == 0 or "WebSocket is not defined" not in probe.stderr.decode()
    )


@pytest.mark.skipif(not _node_has_websocket(), reason="needs Node.js with WebSocket")
def test_the_browser_client_plays_a_game_whose_transcript_verifies():
    async def scenario():
        async with serving() as (server, base):
            config = {"num_players": 3, "vps_to_win": 5, "decision_timeout_ms": 20_000}
            room = await create_room(base, config, [None, "F", "R"])
            host = base[len("ws://") :]
            process = await asyncio.create_subprocess_exec(
                "node",
                "--experimental-websocket",
                "-e",
                SESSION_HARNESS,
                host,
                room["id"],
                "0",
                APP_JS,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            return process.returncode, stdout, stderr

    code, stdout, stderr = run(scenario())
    assert code == 0, stderr.decode()[-2000:]
    transcript = json.loads(stdout)
    assert all(isinstance(m, str) for m in transcript["messages"]), "kept verbatim"
    report = verify_transcript(transcript)
    assert report.ok, report.errors
    assert report.counts["client signatures"] > 0
    assert report.color == "RED"


def test_the_lobby_and_the_http_api():
    async def scenario():
        async with serving() as (server, base):
            room = await create_room(base, {"num_players": 2}, [None, "F"], name="t")
            rooms = await list_rooms(base)
            http = base.replace("ws://", "http://")
            index = await asyncio.to_thread(_get, f"{http}/")
            api = json.loads(await asyncio.to_thread(_get, f"{http}/api/rooms"))
            players = json.loads(await asyncio.to_thread(_get, f"{http}/api/players"))
            script = await asyncio.to_thread(_get, f"{http}/static/app.js")
            with pytest.raises(urllib.error.HTTPError):
                await asyncio.to_thread(_get, f"{http}/static/../app.py")
            return room, rooms, index, api, players, script

    room, rooms, index, api, players, script = run(scenario())
    assert [r["id"] for r in rooms] == [room["id"]]
    assert api[0]["seats"][1]["kind"] == "bot"
    assert room["seats"][0]["kind"] == "open"
    assert "<html" in index.lower() and "app.js" in index
    assert "WebSocket" in script
    assert {"F", "AB", "R"} <= {p["key"] for p in players}


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.read().decode()
