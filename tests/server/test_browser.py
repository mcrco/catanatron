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

from tests.server.helpers import bot_client, join_request, raw_connect, run, serving

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
const [host, roomId, seat, mode, appJs] = process.argv.slice(-5);
const store = new Map();
const noSubtle = mode === "watch-nosubtle";
globalThis.window = {
  crypto: noSubtle ? { getRandomValues: (a) => crypto.getRandomValues(a) } : globalThis.crypto,
  addEventListener() {},
};
globalThis.document = { querySelector() { return null; } };
globalThis.location = { hash: "", protocol: "http:", host };
globalThis.localStorage = {
  getItem: (k) => (store.has(k) ? store.get(k) : null),
  setItem: (k, v) => store.set(k, String(v)),
  removeItem: (k) => store.delete(k),
};
vm.runInThisContext(fs.readFileSync(appJs, "utf8"));
vm.runInThisContext(`
  const mode = ${JSON.stringify(mode)};
  let asked = false, dropped = false, lost = false, again = false;
  const sits = !["watch-nosubtle", "watch-room"].includes(mode);
  render = function () {
    if (mode === "watch-room") {
      // Never sits: the seat picker must still follow the room.
      const kinds = session.room ? session.room.seats.map((s) => s.kind) : [];
      if (kinds.includes("remote")) {
        process.stdout.write(JSON.stringify(kinds), () => process.exit(0));
        render = () => {};
      }
      return;
    }
    if (mode === "unseated-reconnect" && session.room && !dropped) {
      // The page loses its connection before it sits.
      dropped = true; session.ws.close(); return;
    }
    if (sits && session.room && !session.seated && !asked && !session.first) {
      asked = true;
      if (mode === "doubleclick-bot") session.seatBot(${seat}, "NOSUCHBOT");  // fails
      session.takeSeat(${seat});
      if (mode === "doubleclick") session.takeSeat(${seat});  // a double click
    }
    if (mode === "doubleclick-bot" && !again && session.problems.some((p) => p.includes("NOSUCHBOT"))) {
      // The bot's error comes back while the join is in flight; click again.
      again = true;
      session.problems = session.problems.filter((p) => !p.includes("NOSUCHBOT"));
      session.takeSeat(${seat});
    }
    if (session.problems.length) { console.error(session.problems); process.exit(2); }
    if (mode === "play" && session.game && session.game.index > 30 && !dropped) {
      dropped = true; session.ws.close();
    }
    if (session.pending && !session.answered) {
      const actions = session.pending.body.playable_actions;
      if (mode === "doubleclick" && !lost && actions.length > 1) {
        // The connection drops just as the answer goes out: it must be resent.
        lost = true; session.ws.close();
      }
      session.act(actions[actions.length - 1]);
    }
    if ((session.game && session.game.over) || session.aborted) {
      session.close();
      const out = mode === "watch-nosubtle"
        ? { over: session.game.over, unverified: session.unverified, verified: session.verifiedCount }
        : session.transcript();
      process.stdout.write(JSON.stringify(out), () => process.exit(0));
      render = () => {};
    }
  };
  session = new Session(${JSON.stringify(roomId)});
  session.connect();
`);
"""


async def run_session(base, room_id, mode, seat="0"):
    process = await asyncio.create_subprocess_exec(
        "node",
        "--experimental-websocket",
        "-e",
        SESSION_HARNESS,
        base[len("ws://") :],
        room_id,
        seat,
        mode,
        APP_JS,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    assert process.returncode == 0, stderr.decode()[-2000:]
    return json.loads(stdout)


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
            return await run_session(base, room["id"], "play")

    transcript = run(scenario())
    assert all(isinstance(m, str) for m in transcript["messages"]), "kept verbatim"
    report = verify_transcript(transcript)
    assert report.ok, report.errors
    assert report.counts["client signatures"] > 0
    assert report.color == "RED"


@pytest.mark.skipif(not _node_has_websocket(), reason="needs Node.js with WebSocket")
def test_a_double_click_and_a_lost_answer_do_not_cost_the_seat():
    async def scenario():
        async with serving() as (server, base):
            config = {"num_players": 2, "vps_to_win": 4, "decision_timeout_ms": 20_000}
            room = await create_room(base, config, [None, "R"])
            transcript = await run_session(base, room["id"], "doubleclick")
            return transcript, server.rooms[room["id"]]

    transcript, room = run(scenario())
    assert verify_transcript(transcript).ok
    red = [a for a in room.actions if a["color"] == "RED"]
    assert red and not any(
        a["forced"] == "timeout" for a in red
    ), "the answer was resent"


@pytest.mark.skipif(not _node_has_websocket(), reason="needs Node.js with WebSocket")
def test_a_click_after_an_unrelated_error_does_not_take_a_second_key():
    async def scenario():
        async with serving() as (server, base):
            config = {"num_players": 2, "vps_to_win": 4, "decision_timeout_ms": 20_000}
            room = await create_room(base, config, [None, None])
            other, decide = bot_client(f"{base}/room/{room['id']}", "R", seat=1)
            playing = asyncio.create_task(other.play(decide))
            transcript = await run_session(base, room["id"], "doubleclick-bot")
            await playing
            return transcript

    assert verify_transcript(run(scenario())).ok


@pytest.mark.skipif(not _node_has_websocket(), reason="needs Node.js with WebSocket")
def test_a_page_that_reconnects_before_sitting_starts_a_fresh_chain():
    async def scenario():
        async with serving() as (server, base):
            config = {"num_players": 3, "vps_to_win": 4, "decision_timeout_ms": 20_000}
            room = await create_room(base, config, [None, None, "R"])
            page = asyncio.create_task(
                run_session(base, room["id"], "unseated-reconnect")
            )
            await asyncio.sleep(0.5)  # the page has dropped by now
            other, decide = bot_client(f"{base}/room/{room['id']}", "R", seat=1)
            playing = asyncio.create_task(other.play(decide))
            transcript = await page
            await playing
            return transcript

    assert verify_transcript(run(scenario())).ok


@pytest.mark.skipif(not _node_has_websocket(), reason="needs Node.js with WebSocket")
def test_a_page_looking_at_a_waiting_room_sees_seats_taken():
    async def scenario():
        async with serving() as (server, base):
            room = await create_room(base, {"num_players": 3}, [None, None, None])
            page = asyncio.create_task(run_session(base, room["id"], "watch-room"))
            await asyncio.sleep(0.5)
            ws, hello = await raw_connect(f"{base}/room/{room['id']}")
            request, _, _ = join_request(hello, seat=1)
            await ws.send(json.dumps(request))
            kinds = await asyncio.wait_for(page, 30)
            await ws.close()
            return kinds

    assert run(scenario()) == ["open", "remote", "open"]


ESCAPE_HARNESS = r"""
const fs = require("fs");
const vm = require("vm");
const sinks = (globalThis.sinks = []);
function element(name) {
  const node = {
    name, value: "", hidden: false, textContent: "", className: "", style: {},
    set innerHTML(v) { sinks.push(String(v)); },
    get innerHTML() { return ""; },
    insertAdjacentHTML(_, v) { sinks.push(String(v)); },
    append() {}, replaceChildren() {}, setAttribute() {}, addEventListener() {},
    querySelector() { return element(name + " *"); }, querySelectorAll() { return []; },
    classList: { add() {} }, get lastChild() { return element(name + ">*"); },
    children: [],
  };
  return node;
}
const elements = {};
globalThis.window = { crypto: globalThis.crypto, addEventListener() {} };
globalThis.document = {
  querySelector(selector) { return elements[selector] || (elements[selector] = element(selector)); },
  createElement(tag) { return element(tag); },
  createElementNS(_, tag) { return element(tag); },
};
globalThis.location = { hash: "", protocol: "file:", host: "", search: "?server=127.0.0.1:1" };
globalThis.localStorage = { getItem() { return null; }, setItem() {}, removeItem() {} };
vm.runInThisContext(fs.readFileSync(process.argv[process.argv.length - 1], "utf8"));
const EVIL = '<img src=x onerror="window.PWNED=1">';
vm.runInThisContext(`
(async () => {
  const EVIL = ${JSON.stringify(EVIL)};
  const room = {
    id: EVIL, name: EVIL, status: EVIL, round: EVIL, seed_commitment: EVIL,
    config: { num_players: EVIL },
    seats: [{ seat: 0, color: EVIL, kind: EVIL, name: EVIL, connected: false },
            { seat: 1, color: "RED", kind: "open", name: "", connected: false }],
  };
  globalThis.fetch = async (url) => ({ json: async () =>
    url.includes("/api/rooms") ? [room] : [{ key: EVIL, name: EVIL }] });
  document.querySelector("#create [name=num_players]").value = "2";
  await refreshRooms();
  players = [{ key: EVIL, name: EVIL }];
  renderSeatSpecs();
  serverInfo = { app_sha256: EVIL };
  const state = {
    colors: ["RED", "BLUE"], player_state: {}, resource_freqdeck: [EVIL, EVIL, EVIL, EVIL, EVIL],
    development_deck_size: EVIL, board: { buildings: [], roads: [], robber_coordinate: [0, 0, 0] },
    current_prompt: EVIL,
  };
  for (const i of [0, 1]) {
    for (const key of ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]) state.player_state["P" + i + "_" + key + "_IN_HAND"] = EVIL;
    for (const key of ["KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING", "VICTORY_POINT"]) state.player_state["P" + i + "_" + key + "_IN_HAND"] = EVIL;
    for (const key of ["VICTORY_POINTS", "ACTUAL_VICTORY_POINTS", "NUM_RESOURCES_IN_HAND",
      "NUM_DEVELOPMENT_CARDS_IN_HAND", "PLAYED_KNIGHT", "LONGEST_ROAD_LENGTH", "HAS_ROAD", "HAS_ARMY"]) state.player_state["P" + i + "_" + key] = EVIL;
  }
  session = {
    room, seated: false, spectating: false, joining: false, saved: null, messages: [],
    verifiedCount: EVIL, unverified: EVIL, serverKey: EVIL, problems: [EVIL], aborted: null,
    game: { color: "RED", current: "RED", over: false, seats: room.seats, state,
            log: [[["RED", "ROLL", [EVIL, EVIL]], [EVIL, EVIL]]] },
    takeSeat() {}, seatBot() {}, spectate() {}, transcript() { return {}; },
  };
  renderSeating();
  renderMe();
  renderPlayers();
  renderFairness();
  session.game = null;
  renderSeating();
  process.stdout.write(JSON.stringify(sinks));
})().catch((error) => { console.error(error); process.exit(3); });
`);
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_the_page_never_renders_server_data_as_markup():
    """Everything a (possibly hostile) server sends is escaped: an
    independent copy of the page must not run the server's script."""
    result = subprocess.run(
        ["node", "-e", ESCAPE_HARNESS, APP_JS],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    sinks = json.loads(result.stdout)
    assert sinks, "the harness rendered something"
    assert not [s for s in sinks if "<img" in s], "unescaped server data"
    assert any("&lt;img" in s for s in sinks), "the data was shown, escaped"


def test_the_page_loads_its_assets_by_relative_path():
    """So that a copy opened from a file (or served by any host) works."""
    index = open(os.path.join(os.path.dirname(APP_JS), "index.html")).read()
    assert "/static/" not in index
    assert 'src="app.js"' in index and 'href="style.css"' in index


@pytest.mark.skipif(not _node_has_websocket(), reason="needs Node.js with WebSocket")
def test_a_browser_without_subtle_crypto_can_still_watch():
    async def scenario():
        async with serving() as (server, base):
            created = await create_room(
                base, {"num_players": 2, "vps_to_win": 4}, ["R", "R"]
            )
            await asyncio.wait_for(server.rooms[created["id"]].finished.wait(), 60)
            return await run_session(base, created["id"], "watch-nosubtle")

    out = run(scenario())
    assert out["over"] and out["unverified"] >= 3 and out["verified"] == 0


SIGNATURE_HARNESS = r"""
const fs = require("fs");
const vm = require("vm");
globalThis.window = { crypto: globalThis.crypto, addEventListener() {} };
globalThis.document = { querySelector() { return null; } };
globalThis.location = { hash: "", protocol: "http:", host: "localhost" };
vm.runInThisContext(fs.readFileSync(process.argv[process.argv.length - 1], "utf8"));
const input = JSON.parse(fs.readFileSync(0, "utf8"));
(async () => {
  const env = input.envelope, key = input.server_key;
  const unsigned = { ...env }; delete unsigned.sig;
  let badKey = null;
  try { checkedKey("ab".repeat(31)); } catch (error) { badKey = String(error.message); }
  process.stdout.write(JSON.stringify({
    good: await verifyEnvelope(key, env),
    unsigned: await verifyEnvelope(key, unsigned),
    garbled: await verifyEnvelope(key, { ...env, sig: "zz" }),
    short_key: await verifyEnvelope("ab".repeat(31), env),
    bad_key: badKey,
  }));
})();
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_the_browser_treats_unsigned_or_badly_keyed_messages_as_invalid():
    async def scenario():
        async with serving() as (server, base):
            room = await create_room(
                base, {"num_players": 2, "vps_to_win": 3}, ["R", "R"]
            )
            watcher = GameClient(
                f"{base}/room/{room['id']}", spectate=True, transcript_dir=None
            )
            await watcher.play()
            return watcher

    watcher = run(scenario())
    payload = {"envelope": watcher.messages[1], "server_key": watcher.server_key}
    result = subprocess.run(
        ["node", "-e", SIGNATURE_HARNESS, APP_JS],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    assert out["good"] is True
    assert (
        out["unsigned"] is False
        and out["garbled"] is False
        and out["short_key"] is False
    )
    assert "not an Ed25519 key" in out["bad_key"]


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
            assert await asyncio.to_thread(_get, f"{http}/app.js") == script
            assert "body" in await asyncio.to_thread(_get, f"{http}/style.css")
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
