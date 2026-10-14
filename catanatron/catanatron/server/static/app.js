// Browser client for catanatron-server. No build step, no dependencies.
//
// It speaks the same WebSocket protocol as catanatron.server.client: checks
// the server's Ed25519 signature and the hash chain on every message,
// contributes a random seed when it takes a seat, signs every action with
// a key generated in this browser (WebCrypto Ed25519), and keeps the
// transcript, which it stores in IndexedDB for `catanatron-verify`.
"use strict";

const PROTOCOL_VERSION = 5;
const GENESIS = "0".repeat(64);
const RESOURCES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"];
const DEV_CARDS = ["KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING", "VICTORY_POINT"];
const PLAYER_COLORS = {
  RED: "#d32f2f", BLUE: "#1e63c4", ORANGE: "#f08a00",
  WHITE: "#f7f7f7", GREEN: "#2e9d3f", BROWN: "#7b4a2a",
};
const TILE_COLORS = {
  WOOD: "#2f7d32", BRICK: "#c2593a", SHEEP: "#9bcf63",
  WHEAT: "#efc94c", ORE: "#8f9aa5", DESERT: "#e3cb9a",
};
const SIZE = 50;
const W = Math.sqrt(3) * SIZE;
const CORNERS = {
  NORTH: [0, -SIZE], NORTHEAST: [W / 2, -SIZE / 2], SOUTHEAST: [W / 2, SIZE / 2],
  SOUTH: [0, SIZE], SOUTHWEST: [-W / 2, SIZE / 2], NORTHWEST: [-W / 2, -SIZE / 2],
};
const $ = (selector) => document.querySelector(selector);

// ===== bytes, hashes, canonical JSON =====
// Must produce the bytes of Python's json.dumps(sort_keys=True,
// separators=(",", ":"), ensure_ascii=False) for what the protocol signs.
// Keys sort by code point, as Python's sort_keys does (JavaScript's own sort
// compares UTF-16 units, which differs above U+FFFF).
function byCodePoint(a, b) {
  const x = Array.from(a), y = Array.from(b);
  for (let i = 0; i < Math.min(x.length, y.length); i++) {
    const d = x[i].codePointAt(0) - y[i].codePointAt(0);
    if (d) return d;
  }
  return x.length - y.length;
}
function canonical(value) {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return "[" + value.map(canonical).join(",") + "]";
  return "{" + Object.keys(value).sort(byCodePoint)
    .map((k) => JSON.stringify(k) + ":" + canonical(value[k])).join(",") + "}";
}
// The protocol's numbers are integers every client represents exactly. A
// float or a larger integer on the wire would parse here into something
// Python reads otherwise (5.0 is 5 here, not there), so such a message is
// refused before it is parsed, whatever its signature says.
function numbersOk(text) {
  const n = text.length;
  for (let i = 0; i < n; i++) {
    const c = text[i];
    if (c === '"') {
      for (i++; i < n && text[i] !== '"'; i++) if (text[i] === "\\") i++;
      continue;
    }
    if (c === "-" || (c >= "0" && c <= "9")) {
      let j = c === "-" ? i + 1 : i;
      while (j < n && text[j] >= "0" && text[j] <= "9") j++;
      if (j < n && (text[j] === "." || text[j] === "e" || text[j] === "E")) return false;
      if (!Number.isSafeInteger(Number(text.slice(i, j)))) return false;
      i = j - 1;
    }
  }
  return true;
}
const utf8 = (text) => new TextEncoder().encode(text);
const toHex = (bytes) => Array.from(new Uint8Array(bytes), (b) => b.toString(16).padStart(2, "0")).join("");
function fromHex(hex) {
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = parseInt(hex.substr(2 * i, 2), 16);
  return out;
}
const randomHex = (n) => toHex(crypto.getRandomValues(new Uint8Array(n)));
const subtle = window.crypto && window.crypto.subtle;
async function sha256Hex(text) { return toHex(await subtle.digest("SHA-256", utf8(text))); }
const headOf = (env) => ({ to: env.to, seq: env.seq, prev: env.prev, commit: env.commit, body: env.body });

let cryptoProblem = null;
let noEd25519 = false;
const HEX64 = /^[0-9a-f]{64}$/;
// The errors that answer a join (any other error leaves a join in flight).
const JOIN_REFUSALS = new Set(["seat_taken", "room_full", "bad_join", "duplicate_key", "game_started", "banned"]);
async function checkCrypto() {
  if (!subtle) {
    cryptoProblem = "WebCrypto is unavailable (open the page over https or on localhost): you can watch, unverified, but not play.";
    return;
  }
  try {
    await subtle.generateKey({ name: "Ed25519" }, true, ["sign", "verify"]);
  } catch (error) {
    noEd25519 = true;
    cryptoProblem = "This browser has no WebCrypto Ed25519 (needs Chrome 137+, Firefox 129+ or Safari 17+): you can watch, unverified, but not play.";
  }
}

// ===== keys =====
async function newIdentity() {
  const pair = await subtle.generateKey({ name: "Ed25519" }, true, ["sign", "verify"]);
  return {
    publicHex: toHex(await subtle.exportKey("raw", pair.publicKey)),
    privatePkcs8: toHex(await subtle.exportKey("pkcs8", pair.privateKey)),
    privateKey: pair.privateKey,
  };
}
async function loadIdentity(saved) {
  const privateKey = await subtle.importKey("pkcs8", fromHex(saved.privatePkcs8), { name: "Ed25519" }, false, ["sign"]);
  return { publicHex: saved.publicHex, privatePkcs8: saved.privatePkcs8, privateKey };
}
async function signHex(privateKey, obj) {
  return toHex(await subtle.sign({ name: "Ed25519" }, privateKey, utf8(canonical(obj))));
}
const serverKeys = {};
// null: this browser cannot check Ed25519 at all (the page says so). Any
// other failure (no signature, a malformed one, a key that will not import)
// counts as a bad signature.
async function verifyEnvelope(serverKeyHex, env) {
  if (!subtle || noEd25519) return null;
  try {
    if (!HEX64.test(serverKeyHex) || typeof env.sig !== "string" || !/^[0-9a-f]{128}$/.test(env.sig)) return false;
    serverKeys[serverKeyHex] = serverKeys[serverKeyHex] ||
      await subtle.importKey("raw", fromHex(serverKeyHex), { name: "Ed25519" }, false, ["verify"]);
    return await subtle.verify({ name: "Ed25519" }, serverKeys[serverKeyHex], fromHex(env.sig), utf8(canonical(headOf(env))));
  } catch (error) {
    return false;
  }
}
function checkedKey(key) {
  if (typeof key !== "string" || !HEX64.test(key)) throw new Error("the server's key is not an Ed25519 key");
  return key;
}
async function verifyHex(publicHex, obj, sigHex) {
  if (!subtle || noEd25519) return null;
  try {
    if (!HEX64.test(publicHex) || typeof sigHex !== "string" || !/^[0-9a-f]{128}$/.test(sigHex)) return false;
    const key = await subtle.importKey("raw", fromHex(publicHex), { name: "Ed25519" }, false, ["verify"]);
    return await subtle.verify({ name: "Ed25519" }, key, fromHex(sigHex), utf8(canonical(obj)));
  } catch (error) {
    return false;
  }
}
async function sha256OfHex(hex) { return toHex(await subtle.digest("SHA-256", fromHex(hex))); }
async function clientSeedCommitment(roomId, publicKey, clientSeed) {
  return sha256Hex(canonical({
    domain: "catanatron-client-seed/1", room_id: roomId, public_key: publicKey, client_seed: clientSeed,
  }));
}

// ===== transcripts in IndexedDB =====
function openDb() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open("catanatron-transcripts", 1);
    request.onupgradeneeded = () => request.result.createObjectStore("transcripts", { keyPath: "id" });
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}
// A stored transcript is only ever replaced by one that extends it (the
// same messages and more, at least the same signed answers). A poorer copy
// (reopening a finished room replays it without what was signed) is not
// stored, and a different one (a server naming the same game twice) is
// stored beside it: nothing this page could prove is ever erased.
function extendsTranscript(a, b) {
  const prefix = b.messages.every((m, i) => a.messages[i] === m);
  const sent = new Set((a.sent || []).map((x) => canonical(x)));
  return prefix && (b.sent || []).every((x) => sent.has(canonical(x)));
}
async function saveTranscript(id, transcript) {
  try {
    const db = await openDb();
    const store = () => db.transaction("transcripts", "readwrite").objectStore("transcripts");
    const existing = await new Promise((resolve) => {
      const request = store().get(id);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => resolve(null);
    });
    if (existing) {
      let old = null;
      try { old = JSON.parse(existing.json); } catch (error) { /* replace a broken copy */ }
      if (old && !extendsTranscript(transcript, old)) {
        if (extendsTranscript(old, transcript)) return; // a poorer copy
        id = `${id}-${Date.now()}`; // another story: keep both
      }
    }
    store().put({ id, created: Date.now(), json: JSON.stringify(transcript) });
  } catch (error) { console.warn("could not store the transcript", error); }
}
async function listTranscripts() {
  try {
    const db = await openDb();
    return await new Promise((resolve) => {
      const request = db.transaction("transcripts").objectStore("transcripts").getAll();
      request.onsuccess = () => resolve(request.result.sort((a, b) => b.created - a.created));
      request.onerror = () => resolve([]);
    });
  } catch (error) { return []; }
}
function download(name, text) {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([text], { type: "application/json" }));
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

// ===== lobby =====
let players = [];
async function showLobby() {
  $("#room").hidden = true;
  $("#lobby").hidden = false;
  if (session) session.close();
  session = null;
  await refreshRooms();
  if (!players.length) players = await (await fetch(apiUrl("/api/players"))).json();
  renderSeatSpecs();
  const list = $("#transcripts");
  list.innerHTML = "";
  for (const entry of await listTranscripts()) {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.textContent = "Download";
    button.onclick = () => download(entry.id + ".json", entry.json);
    item.append(entry.id + " ", button);
    list.append(item);
  }
  if (!list.children.length) list.innerHTML = '<li class="muted">none yet</li>';
}
async function refreshRooms() {
  const rooms = await (await fetch(apiUrl("/api/rooms"))).json();
  const body = $("#rooms tbody");
  body.innerHTML = "";
  for (const room of rooms.reverse()) {
    const row = document.createElement("tr");
    const seats = room.seats.map((s) => (s.kind === "open" ? "open" : `${s.name} (${s.kind})`)).join(", ");
    setHtml(row, html`<td>${room.name}</td><td>${room.status}</td><td>${room.config.num_players}</td><td>${seats}</td><td></td>`);
    const open = document.createElement("button");
    open.textContent = room.status === "waiting" ? "Open" : "Watch";
    open.onclick = () => { location.hash = "room=" + room.id; };
    row.lastChild.append(open);
    body.append(row);
  }
  if (!rooms.length) body.innerHTML = '<tr><td colspan="5" class="muted">no rooms yet</td></tr>';
}
function renderSeatSpecs() {
  const count = Number($("#create [name=num_players]").value);
  const box = $("#seat-specs");
  const previous = Array.from(box.querySelectorAll("select")).map((s) => s.value);
  box.innerHTML = "";
  const colors = Object.keys(PLAYER_COLORS);
  for (let i = 0; i < count; i++) {
    const select = document.createElement("select");
    options(select, [["", "open (a person or a remote bot)"], ...players.map((p) => [p.key, `bot: ${p.key} (${p.name})`])]);
    select.value = previous[i] !== undefined ? previous[i] : (i === 0 ? "" : "F");
    const row = document.createElement("div");
    row.className = "seat-row";
    row.innerHTML = `<span class="swatch" style="background:${PLAYER_COLORS[colors[i]]}"></span>Seat ${i} (${colors[i]})`;
    row.append(select);
    box.append(row);
  }
}
function lobbyRequest(request) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl("/lobby"));
    ws.onopen = () => ws.send(JSON.stringify(request));
    ws.onmessage = (event) => { ws.close(); resolve(JSON.parse(event.data)); };
    ws.onerror = () => reject(new Error("cannot reach the server"));
  });
}
async function createRoom(event) {
  event.preventDefault();
  const form = event.target;
  const config = {
    num_players: Number(form.num_players.value),
    map_type: form.map_type.value,
    number_placement: form.number_placement.value,
    vps_to_win: Number(form.vps_to_win.value),
    discard_limit: Number(form.discard_limit.value),
    decision_timeout_ms: Math.round(Number(form.timeout.value) * 1000),
  };
  $("#create-error").textContent = "";
  if (form.map.value.trim()) {
    try { config.map = JSON.parse(form.map.value); } catch (error) {
      $("#create-error").textContent = "the map is not valid JSON"; return;
    }
  }
  const seats = Array.from($("#seat-specs").querySelectorAll("select")).map((s) => s.value || null);
  const reply = await lobbyRequest({ type: "create_room", config, seats, name: form.name.value });
  if (reply.type === "error") { $("#create-error").textContent = reply.message; return; }
  location.hash = "room=" + reply.room.id;
}
// A copy of this page served from elsewhere (a file, your own host) names
// the game server with ?server=HOST:PORT (and &tls=1 for wss/https).
const QUERY = new URLSearchParams(location.search || "");
const SERVER = QUERY.get("server") || location.host;
const TLS = QUERY.get("server") ? QUERY.get("tls") === "1" : location.protocol === "https:";
const wsUrl = (path) => (TLS ? "wss://" : "ws://") + SERVER + path;
const apiUrl = (path) => (TLS ? "https://" : "http://") + SERVER + path;
// Everything the server sends is data, never markup: interpolate it with
// html`...`, which escapes every value (an html`` fragment stays as is).
class SafeHtml { constructor(text) { this.text = text; } toString() { return this.text; } }
function html(strings, ...values) {
  let out = strings[0];
  values.forEach((value, i) => {
    const parts = Array.isArray(value) ? value : [value];
    out += parts.map((part) => (part instanceof SafeHtml ? part.text : escapeHtml(part))).join("") + strings[i + 1];
  });
  return new SafeHtml(out);
}
function setHtml(element, fragment) { element.innerHTML = fragment.text; }
function addHtml(element, fragment) { element.insertAdjacentHTML("beforeend", fragment.text); }
function options(select, entries) {
  // [value, label] pairs, built as DOM nodes (never markup).
  select.replaceChildren(...entries.map(([value, label]) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    return option;
  }));
}
function escapeHtml(text) {
  return String(text === undefined || text === null ? "" : text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ===== a room =====
let session = null;

// Seat records live in localStorage, one per game server and room: a copy
// of this page that talks to several servers never hands one server's seat
// (its key, its unrevealed seed) to another.
function seatStorageKey(serverKey, roomId) { return `catanatron-seat-${serverKey}-${roomId}`; }
function loadSeat(key) {
  try { return JSON.parse(localStorage.getItem(key) || "null"); } catch (error) { return null; }
}
function storeSeat(key, saved) {
  try {
    if (saved) localStorage.setItem(key, JSON.stringify(saved));
    else localStorage.removeItem(key);
  } catch (error) { console.warn("could not store the seat", error); }
}
const COMMITMENTS_FIELDS = ["config", "reveal_timeout_ms", "room_id", "seats", "seed_commitment", "server_key", "type"];
const SEAT_FIELDS = ["client_commitment", "color", "kind", "name", "nonce", "pop", "public_key", "requested_seat", "seat", "spec"];
// How long to wait, once the game is over, for receipts still owed.
const RECEIPT_WAIT_MS = 5000;
const NONCE = /^[0-9a-f]{32}$/;

class Session {
  // The page keeps the protocol's invariants on its own (PROTOCOL.md): it
  // signs exactly one reveal per room and server key, for the first
  // commitments it gets and with the seed it joined with; it plays and
  // signs only under those commitments; it signs one answer per decision
  // and resends, after a reconnect, exactly that. A seat record whose game
  // ended is never used to sit again.
  constructor(roomId) {
    this.roomId = roomId;
    this.storageKey = null; // known once the server's key is (first hello)
    this.saved = null; // the seat record (see takeSeat)
    this.identity = null;
    this.spectating = false;
    this.joining = false;
    this.started = Date.now();
    this.resetTranscript();
    this.room = null;
    this.game = null; // the table: before's static parts, state, log
    this.pending = null; // { env, hash, body }
    this.answered = false;
    this.answer = null; // the signed action for the pending decide
    this.answerDigest = null;
    this.closed = false;
    this.verifiedCount = 0;
    this.unverified = 0;
    this.problems = [];
    this.saveId = null;
    this.aborted = null;
    this.finishTimer = null;
    this.lateReconnects = 0;
  }
  resetTranscript() {
    this.messages = []; // raw text, exactly as received
    this.head = GENESIS;
    this.joinedAt = undefined;
    this.commitments = null; // the commitments accepted on this chain
    this.C = null; // their digest, which every move signs
    this.ended = null; // "after" or "aborted": the chain's end
  }
  // A join went out from this page (it may hold a seat even if "joined" was lost).
  get joinSent() { return Boolean(this.saved && this.saved.identity && !this.saved.closed); }
  get seated() { return Boolean(this.saved && this.saved.joined); }
  get sent() { return this.saved ? this.saved.sent : []; }
  get unreceipted() { return this.saved && !this.saved.closed ? this.saved.unreceipted : []; }
  persist() { if (this.storageKey) storeSeat(this.storageKey, this.saved); }
  remember(changes) { this.saved = { ...this.saved, ...changes }; this.persist(); }

  connect() {
    this.ws = new WebSocket(wsUrl("/room/" + this.roomId));
    this.first = true;
    this.resyncing = false;
    this.queue = Promise.resolve();
    this.ws.onmessage = (event) => {
      // One message at a time, in order: checking a signature is async.
      this.queue = this.queue.then(() => this.receive(event.data)).catch((error) => {
        console.error(error);
        this.problems.push(String(error.message || error));
        render();
      });
    };
    this.ws.onclose = () => {
      if (this.closed || this.done) return;
      // Over but owed receipts: a few tries to collect them, no more.
      if ((this.aborted || (this.game && this.game.over)) && ++this.lateReconnects > 3) return;
      setTimeout(() => { if (!this.closed) this.connect(); }, 1500);
    };
  }
  // Over, with nothing owed: no reason to reconnect.
  get done() {
    return Boolean((this.aborted || (this.game && this.game.over)) && !this.unreceipted.length);
  }
  close() { this.closed = true; if (this.ws) this.ws.close(); }
  send(payload) {
    if (!this.ws || this.ws.readyState !== 1) return false;
    this.ws.send(JSON.stringify(payload));
    return true;
  }

  async receive(text) {
    if (!numbersOk(text)) throw new Error("a message with a number the protocol does not allow (a float, or beyond 2**53)");
    const env = JSON.parse(text);
    if (this.first) {
      this.first = false;
      const body = env.body;
      const key = checkedKey(body.server_key);
      if (this.serverKey && key !== this.serverKey) throw new Error("the server's key changed between connections");
      if (!this.storageKey) {
        this.serverKey = key;
        this.storageKey = seatStorageKey(key, this.roomId);
        this.saved = loadSeat(this.storageKey);
        if (this.saved && this.saved.seedCommitment) this.serverCommitment = this.saved.seedCommitment;
      }
      if (this.joinSent) {
        // Taking back our seat: this connection's hello opens a throw-away
        // chain; our key signs its nonce, and the seat's own chain resumes
        // after our last message.
        if ((await verifyEnvelope(this.serverKey, env)) === false) throw new Error("bad signature on hello");
        if (!NONCE.test(body.nonce || "")) throw new Error("the hello carries no nonce to sign");
        if (!this.identity) this.identity = await loadIdentity(this.saved.identity);
        this.resyncing = true;
        const sig = await signHex(this.identity.privateKey, { domain: "catanatron-rejoin/1", room_id: this.roomId, nonce: body.nonce });
        this.send({
          type: "rejoin", protocol_version: PROTOCOL_VERSION, public_key: this.identity.publicHex,
          sig, last_seq: this.messages.length - 1,
        });
        this.resume();
        return;
      }
      // A connection of our own: a new chain (watching, or before sitting).
      // (A seat record whose game ended is history: it is never sat in again.)
      this.resetTranscript();
      if (this.saved && this.saved.closed) this.spectating = true;
      if (this.spectating) {
        this.game = null;
        await this.accept(text, env);
        this.send({ type: "spectate" });
        return;
      }
      this.joining = false;
      await this.accept(text, env);
      if (body.protocol_version !== PROTOCOL_VERSION) throw new Error(`the server speaks protocol ${body.protocol_version}`);
      this.checkRoom(body.room, "the hello", !this.serverCommitment);
      this.room = body.room;
      this.serverCommitment = body.room.seed_commitment;
      this.helloNonce = body.nonce;
      if (body.room.status !== "waiting") this.spectate(); // too late to sit
      render();
      return;
    }
    if (this.resyncing) {
      if (!(env.seq === this.messages.length && env.prev === this.head)) {
        // Still the throw-away chain: check it, but keep it out of ours. A
        // message for our key that does not continue our chain is another
        // chain for this seat: refused.
        if ((await verifyEnvelope(this.serverKey, env)) === false) throw new Error("bad signature while rejoining");
        if (this.seated && env.to === this.saved.identity.publicHex) {
          this.resyncing = false;
          this.ws.close();
          throw new Error("the server resent another chain for this seat");
        }
        if (env.body.type === "error") this.rejoinRefused(env.body);
        return;
      }
      this.resyncing = false;
    }
    await this.accept(text, env);
    await this.handle(env);
  }

  rejoinRefused(body) {
    if (body.code === "not_seated" && !(this.saved && (this.saved.reveal || this.saved.joined))) {
      // Our join never landed (or the seat was freed while we were away,
      // before any seed was asked for): forget it and look at the room afresh.
      this.saved = null;
      this.persist();
      this.identity = null;
      this.resetTranscript();
      this.ws.close();
      return;
    }
    this.problems.push(body.message);
    render();
  }

  // Every description of the room names it, its seed commitment and its
  // rules as they were when this page first saw it (or sat).
  checkRoom(room, where, first) {
    if (!room || room.id !== this.roomId) throw new Error(`${where} names another room`);
    if (first) return;
    if (room.seed_commitment !== this.serverCommitment ||
        (this.room && canonical(room.config) !== canonical(this.room.config))) {
      throw new Error(`${where} changes the room's seed commitment or rules`);
    }
  }

  async accept(text, env) {
    const valid = await verifyEnvelope(this.serverKey, env);
    if (valid === false) throw new Error(`message ${env.seq}: bad server signature`);
    if (env.seq !== this.messages.length) throw new Error(`message ${env.seq}: out of order`);
    if (subtle && env.prev !== this.head) throw new Error(`message ${env.seq}: breaks the hash chain`);
    if (this.joinedAt !== undefined && env.to !== this.saved.identity.publicHex) {
      throw new Error(`message ${env.seq}: addressed to someone else`);
    }
    if (valid) this.verifiedCount++; else this.unverified++;
    this.messages.push(text);
    // Without SubtleCrypto there is no SHA-256 here: follow the sequence
    // numbers only (the page says the session is unverified).
    this.head = subtle ? await sha256Hex(canonical(headOf(env))) : null;
    env.hash = this.head;
  }

  async handle(env) {
    const body = env.body;
    if (this.ended && body.type !== "received" && body.type !== "error") {
      this.problems.push(`a ${String(body.type).slice(0, 40)} message after the ${this.ended}`);
      render();
      return;
    }
    switch (body.type) {
      case "joined": {
        this.joining = false;
        const mine = this.saved && this.identity && !this.saved.joined && this.joinedAt === undefined &&
          body.public_key === this.identity.publicHex && body.room_id === this.roomId &&
          body.client_commitment === this.saved.clientCommitment &&
          (this.saved.requested === null || body.seat === this.saved.requested);
        if (!mine && !(this.saved && this.saved.joined && this.joinedAt === undefined &&
            body.public_key === this.identity.publicHex && body.seat === this.saved.seat &&
            body.client_commitment === this.saved.clientCommitment)) {
          this.problems.push("the server seated another key, seed commitment or seat than this page's");
          break;
        }
        this.checkRoom(body.room, "the join");
        this.joinedAt = env.seq;
        this.remember({ joined: true, seat: body.seat, color: body.color });
        this.room = body.room;
        break;
      }
      case "hello": // the seat chain's first message, resent after a reload
        this.checkRoom(body.room, "the seat's hello");
        this.room = body.room;
        break;
      case "room":
        this.checkRoom(body.room, "a room message");
        this.room = body.room;
        break;
      case "commitments":
        await this.onCommitments(body);
        break;
      case "received":
        this.onReceipt(body);
        break;
      case "error":
        if (body.code !== "not_revealing") this.problems.push(body.message);
        // Only an answer to the join itself ends the wait for it.
        if (this.joining && body.answers === "join" && JOIN_REFUSALS.has(body.code)) {
          this.joining = false;
          this.saved = null; // that key and seed were never seated: forget them
          this.persist();
          this.identity = null;
        }
        break;
      case "aborted":
        if (subtle && (!HEX64.test(body.server_seed || "") ||
            (await sha256OfHex(body.server_seed)) !== this.serverCommitment)) {
          this.problems.push("the void does not reveal the seed the server committed to");
        }
        this.aborted = body;
        this.ended = "void";
        this.closePending();
        if (this.saved) this.remember({ closed: true });
        this.finish();
        break;
      case "before":
        if (!this.startGame(body)) break;
        this.update(body);
        break;
      case "step": case "decide": case "after":
        if (!this.game) {
          this.problems.push(`a ${body.type} message before the game`);
          break;
        }
        this.update(body);
        this.closePending();
        if (body.type === "decide") this.pending = { env, hash: env.hash, body, received: Date.now() };
        if (body.type === "after") {
          this.ended = "game";
          if (this.saved) this.remember({ closed: true });
          this.finish();
        }
        break;
      default:
        this.problems.push(`unexpected ${String(body.type).slice(0, 40)} message`);
    }
    render();
  }

  // A seated page plays only the game of the commitments it revealed for,
  // received on this chain; nobody plays a game whose seed commitment or
  // seats are not the room's.
  startGame(body) {
    if (body.seed_commitment !== this.serverCommitment) {
      this.problems.push("the game names another seed commitment");
      return false;
    }
    if (this.commitments && canonical(body.commitments) !== canonical(this.commitments)) {
      this.problems.push("the game is not the one whose seats were committed");
      return false;
    }
    if (this.seated) {
      if (!this.commitments || !this.saved.reveal || this.saved.reveal.commitments !== this.C) {
        this.problems.push("the game is not the one this page revealed its seed for");
        return false;
      }
      if ((body.client_seeds || [])[this.saved.seat] !== this.saved.clientSeed) {
        this.problems.push("the game was not seeded with this page's seed");
        return false;
      }
    }
    return true;
  }

  closePending() {
    // A newer message closed the open decision: if this page had not
    // answered it, keep a record (the server timed it out, or so it says).
    if (this.pending && !this.answered && this.seated) {
      const note = { index: this.pending.body.index, seq: this.pending.env.seq, ms: Date.now() - this.pending.received };
      this.remember({ unanswered: [...(this.saved.unanswered || []), note] });
    }
    this.pending = null;
    this.answered = false;
    this.answer = null;
  }

  async onCommitments(body) {
    const fields = Object.keys(body).sort().join(",");
    if (fields !== COMMITMENTS_FIELDS.join(",") || body.room_id !== this.roomId ||
        body.server_key !== this.serverKey || body.seed_commitment !== this.serverCommitment ||
        !Number.isInteger(body.reveal_timeout_ms) || !Array.isArray(body.seats) ||
        (this.room && canonical(body.config) !== canonical(this.room.config))) {
      this.problems.push("the commitments change the room's server, rules or seed");
      return;
    }
    if (this.commitments) {
      // One commitments per room, ever: a second, different one would be a
      // second chance to reveal (and so to choose).
      if (canonical(body) !== canonical(this.commitments)) this.problems.push("a second, different commitments for this room");
      return;
    }
    const C = subtle ? await sha256Hex(canonical(body)) : null;
    if (this.seated && this.saved.reveal && this.saved.reveal.commitments !== C) {
      // A reload replaying the chain: this page revealed already, for other
      // commitments. These are refused, and nothing is signed under them.
      this.problems.push("a second, different commitments for this room");
      return;
    }
    this.commitments = body;
    this.C = C;
    if (!this.seated) {
      if (!this.joining) this.spectating = true; // the server made us a spectator
      return;
    }
    if (this.saved.reveal) {
      if (!this.game) this.send(this.revealRequest());
      return;
    }
    const problem = await this.mySeatProblem(body);
    if (problem) {
      this.problems.push(problem);
      return;
    }
    const sig = await signHex(this.identity.privateKey, { domain: "catanatron-reveal/1", commitments: C });
    const digest = await sha256Hex(canonical({ statement: { domain: "catanatron-reveal/1", commitments: C }, sig }));
    this.remember({ reveal: { commitments: C, sig, digest } });
    this.send(this.revealRequest());
  }
  async mySeatProblem(body) {
    const keys = body.seats.filter((s) => s && s.kind === "remote").map((s) => s.public_key);
    if (new Set(keys).size !== keys.length) return "the commitments seat one key twice";
    // Every seat, not just ours: a reveal signs them all.
    for (const [index, seat] of body.seats.entries()) {
      if (!seat || Object.keys(seat).sort().join(",") !== SEAT_FIELDS.join(",") || seat.seat !== index) {
        return `the commitments' seat ${index} is malformed`;
      }
      if (seat.kind === "bot") {
        if (seat.public_key !== null || seat.client_commitment !== null || seat.nonce !== null || seat.pop !== null) {
          return `the commitments' seat ${index} is malformed`;
        }
        continue;
      }
      const requested = seat.requested_seat;
      const statement = {
        domain: "catanatron-join/2", room_id: this.roomId, server_key: this.serverKey,
        seed_commitment: this.serverCommitment, seat: requested,
        client_commitment: seat.client_commitment, nonce: seat.nonce,
      };
      if (seat.kind !== "remote" || !(requested === null || requested === index) ||
          !NONCE.test(seat.nonce || "") || !(await verifyHex(seat.public_key, statement, seat.pop))) {
        return `the commitments' seat ${index} is malformed`;
      }
    }
    const mine = body.seats.filter((s) => s && s.public_key === this.saved.identity.publicHex);
    const seat = mine[0];
    if (mine.length !== 1 || Object.keys(seat).sort().join(",") !== SEAT_FIELDS.join(",") ||
        seat.kind !== "remote" || seat.seat !== this.saved.seat || seat.requested_seat !== this.saved.requested ||
        seat.client_commitment !== this.saved.clientCommitment || seat.nonce !== this.saved.nonce) {
      return "the commitments do not seat this page as it joined";
    }
    const statement = {
      domain: "catanatron-join/2", room_id: this.roomId, server_key: this.serverKey,
      seed_commitment: this.serverCommitment, seat: seat.requested_seat,
      client_commitment: seat.client_commitment, nonce: seat.nonce,
    };
    if (!(await verifyHex(this.saved.identity.publicHex, statement, seat.pop))) {
      return "the commitments do not carry this page's own join signature";
    }
    if (!this.identity) this.identity = await loadIdentity(this.saved.identity);
    return null;
  }
  revealRequest() {
    return { type: "reveal", client_seed: this.saved.clientSeed, sig: this.saved.reveal.sig };
  }

  onReceipt(body) {
    if (this.saved) {
      const left = this.saved.unreceipted.filter((u) => u.digest !== body.submission);
      if (left.length !== this.saved.unreceipted.length) this.remember({ unreceipted: left });
      if (this.saved.reveal && this.saved.reveal.digest === body.submission && body.accepted) {
        this.remember({ reveal: { ...this.saved.reveal, receipted: true } });
      }
    }
    // A refusal of the answer we are waiting on (not of an older one resent).
    if (!body.accepted && body.submission === this.answerDigest && this.pending) {
      this.problems.push("the server did not take your move: " + body.reason);
    }
    if (this.saveId) this.finish(); // the saved transcript gets the receipt too
  }

  resume() {
    // After a rejoin: finish what the drop interrupted, by resending exactly
    // what was signed and has no receipt yet. Resending is harmless (the
    // server's receipt says "stale" for what it already had).
    if (this.saved.reveal && !this.saved.reveal.receipted && !this.game) this.send(this.revealRequest());
    for (const owed of this.unreceipted) this.send(owed.answer);
  }

  update(body) {
    if (body.type === "before") {
      this.game = {
        gameId: body.game_id, color: body.color, config: body.config, seats: body.seats,
        clientSeeds: body.client_seeds, board: body.board, log: [], over: false,
      };
    }
    const game = this.game;
    if (!game) return;
    game.log.splice(body.log_start, game.log.length - body.log_start, ...body.log);
    game.state = body.state;
    game.index = body.index;
    game.current = body.current_color;
    game.winner = body.winning_color;
    if (body.type === "after") game.over = true;
  }

  async takeSeat(index) {
    if (cryptoProblem) { alert(cryptoProblem); return; }
    if (this.joining || this.joinSent || !this.storageKey || !NONCE.test(this.helloNonce || "")) return; // one join at a time
    this.joining = true;
    render();
    this.identity = await newIdentity();
    const clientSeed = randomHex(32);
    const clientCommitment = await clientSeedCommitment(this.roomId, this.identity.publicHex, clientSeed);
    const statement = {
      domain: "catanatron-join/2", room_id: this.roomId, server_key: this.serverKey,
      seed_commitment: this.serverCommitment, seat: index, client_commitment: clientCommitment,
      nonce: this.helloNonce,
    };
    const pop = await signHex(this.identity.privateKey, statement);
    // Kept before the join goes out: if "joined" is lost, the key takes the
    // seat back. A fresh key and seed for every seat, never used again.
    this.saved = {
      identity: { publicHex: this.identity.publicHex, privatePkcs8: this.identity.privatePkcs8 },
      clientSeed, clientCommitment, pop, nonce: this.helloNonce, requested: index,
      seedCommitment: this.serverCommitment, joined: false, sent: [], unreceipted: [], unanswered: [],
    };
    this.persist();
    this.send({
      type: "join", protocol_version: PROTOCOL_VERSION, name: $("#my-name") ? $("#my-name").value : "",
      seat: index, public_key: this.identity.publicHex, client_commitment: clientCommitment, pop,
    });
  }
  spectate() {
    this.spectating = true;
    this.send({ type: "spectate" });
  }
  seatBot(index, spec) {
    if (this.joining) return; // its errors would be mistaken for the join's
    this.send({ type: "seat_bot", seat: index, spec });
  }

  async act(action) {
    if (!this.pending || this.answered || this.ended) return;
    // Only under the commitments this page revealed for.
    if (!this.saved || !this.saved.reveal || this.C !== this.saved.reveal.commitments) return;
    this.answered = true;
    const pending = this.pending;
    const statement = {
      domain: "catanatron-action/1", commitments: this.C, index: pending.body.index,
      seq: pending.env.seq, decide: pending.hash, action,
    };
    const sig = await signHex(this.identity.privateKey, statement);
    this.answer = { type: "action", statement, sig };
    this.answerDigest = await sha256Hex(canonical({ statement, sig }));
    // Kept with the seat (a reload keeps them): what was sent, for the
    // verifier, and what is owed a receipt, resent until it gets one.
    this.remember({
      sent: [...this.saved.sent, { statement, sig, sent_after: this.messages.length - 1 }],
      unreceipted: [...this.saved.unreceipted, { digest: this.answerDigest, answer: this.answer }],
    });
    this.send(this.answer);
    render();
  }

  transcript() {
    const seated = this.seated && this.joinedAt !== undefined;
    return {
      format: "catanatron-transcript/5",
      url: wsUrl("/room/" + this.roomId),
      room_id: this.roomId,
      server_key: this.serverKey,
      public_key: seated ? this.saved.identity.publicHex : null,
      client_seed: seated ? this.saved.clientSeed : null,
      reveal_sig: seated && this.saved.reveal ? this.saved.reveal.sig : null,
      seat: seated ? this.saved.seat : null,
      color: this.game ? this.game.color : null,
      messages: this.messages,
      sent: seated ? this.sent : [],
      unanswered: seated ? this.saved.unanswered || [] : [],
    };
  }
  finish() {
    // Saved once the receipts still owed have come (or a while has passed),
    // and again whenever one comes later; never over a fuller copy.
    const who = this.seated ? `${this.saved.color || "seat"}-${this.saved.identity.publicHex.slice(0, 8)}`
      : `spectator-${this.started}`;
    this.saveId = `${String(this.serverKey).slice(0, 16)}-${this.roomId}-${this.game ? this.game.gameId : "void"}-${who}`;
    clearTimeout(this.finishTimer);
    const owed = this.saved && this.saved.unreceipted ? this.saved.unreceipted.length : 0;
    if (owed && !this.finishWaited) {
      this.finishWaited = true;
      this.finishTimer = setTimeout(() => saveTranscript(this.saveId, this.transcript()), RECEIPT_WAIT_MS);
      return;
    }
    saveTranscript(this.saveId, this.transcript());
  }
  newRoomLikeThis() {
    // A void room cannot be played again: a new room, new seeds from everyone.
    const room = this.room;
    const seats = room.seats.map((s) => (s.kind === "bot" ? s.spec : null));
    lobbyRequest({ type: "create_room", config: room.config, seats, name: room.name }).then((reply) => {
      if (reply.type === "error") { this.problems.push(reply.message); render(); return; }
      location.hash = "room=" + reply.room.id;
    });
  }
}

async function openRoom(roomId) {
  $("#lobby").hidden = true;
  $("#room").hidden = false;
  if (session) session.close();
  session = new Session(roomId);
  session.connect();
  render();
}

// ===== rendering =====
function render() {
  if (!session) return;
  renderSeating();
  if (session.game) {
    renderBoard();
    renderMe();
    renderActions();
    renderPlayers();
    renderLog();
  } else {
    $("#board").innerHTML = "";
    for (const id of ["#me", "#actions", "#players"]) $(id).innerHTML = "";
    $("#log").innerHTML = "";
    $("#prompt").textContent = session.aborted ? promptText()
      : session.commitments ? "Every seat is taken: revealing seeds..." : "Waiting for every seat to be taken...";
    if (session.aborted && session.room) {
      $("#actions").append(button("New room with these settings", () => session.newRoomLikeThis(), "primary"));
    }
  }
  renderFairness();
}

function swatch(color) { return PLAYER_COLORS[color] || "#999"; }
function storedName() { try { return localStorage.getItem("catanatron-name") || ""; } catch (error) { return ""; } }

function seatInfo(color) {
  const seats = session.game ? session.game.seats : (session.room ? session.room.seats : []);
  return seats.find((s) => s.color === color) || {};
}

function renderSeating() {
  const box = $("#seating");
  const room = session.room;
  if (!room || session.game) {
    setHtml(box, room ? html`<h3>${room.name}</h3><div class="muted">room ${room.id}</div>` : html``);
    return;
  }
  setHtml(box, html`<h3>${room.name} <span class="muted">(${room.status})</span></h3>`);
  const canSit = !session.joinSent && !session.spectating && !session.joining && room.status === "waiting";
  if (canSit) {
    addHtml(box, html`<label>Your name <input id="my-name" maxlength="32" value="${storedName()}"></label>`);
    $("#my-name").onchange = (e) => localStorage.setItem("catanatron-name", e.target.value);
  }
  for (const seat of room.seats) {
    const row = document.createElement("div");
    row.className = "seat-row";
    const who = seat.kind === "open" ? html`<span class="muted">open</span>`
      : html`${seat.name} <span class="muted">(${seat.kind}${seat.kind === "remote" && !seat.connected ? ", away" : ""})</span>`;
    setHtml(row, html`<span class="swatch" style="background:${swatch(seat.color)}"></span>${seat.color}: ${who}`);
    if (seat.kind === "open" && room.status === "waiting") {
      if (canSit) row.append(button("Sit here", () => session.takeSeat(seat.seat), "primary"));
      const select = document.createElement("select");
      options(select, players.map((p) => [p.key, p.key]));
      select.value = "F";
      const add = button("Add bot", () => session.seatBot(seat.seat, select.value));
      add.disabled = session.joining; // its errors would be mistaken for the join's
      row.append(select, add);
    }
    box.append(row);
  }
  if (session.joining) addHtml(box, html`<p>Taking the seat...</p>`);
  if (session.seated) addHtml(box, html`<p>You are seat ${session.saved.seat}. The game starts when every seat is taken.</p>`);
  else if (!session.spectating && !session.joining && !session.joinSent) box.append(button("Just watch", () => session.spectate()));
  if (!players.length) fetch(apiUrl("/api/players")).then((r) => r.json()).then((p) => { players = p; renderSeating(); });
}

function button(label, onclick, className) {
  const element = document.createElement("button");
  element.textContent = label;
  element.onclick = onclick;
  if (className) element.className = className;
  return element;
}

const pixel = (coordinate) => [W * (coordinate[0] + coordinate[2] / 2), 1.5 * SIZE * coordinate[2]];
const svg = (tag, attrs, text) => {
  const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs || {})) element.setAttribute(key, value);
  if (text !== undefined) element.textContent = text;
  return element;
};
const PIPS = { 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1 };

function boardActions() {
  // Legal moves that are a click on the board, keyed by what is clicked.
  const out = { nodes: {}, edges: {}, tiles: {} };
  if (!session.pending || session.answered) return out;
  for (const action of session.pending.body.playable_actions) {
    const [, type, value] = action;
    if (type === "BUILD_SETTLEMENT" || type === "BUILD_CITY") out.nodes[value] = action;
    else if (type === "BUILD_ROAD") out.edges[[...value].sort((a, b) => a - b).join(",")] = action;
    else if (type === "MOVE_ROBBER") (out.tiles[value[0].join(",")] = out.tiles[value[0].join(",")] || []).push(action);
  }
  return out;
}

function renderBoard() {
  const game = session.game;
  const board = $("#board");
  board.innerHTML = "";
  const positions = {};
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const tile of game.board) {
    const [cx, cy] = pixel(tile.coordinate);
    for (const [ref, node] of Object.entries(tile.nodes)) {
      positions[node] = [cx + CORNERS[ref][0], cy + CORNERS[ref][1]];
    }
    minX = Math.min(minX, cx - W / 2); maxX = Math.max(maxX, cx + W / 2);
    minY = Math.min(minY, cy - SIZE); maxY = Math.max(maxY, cy + SIZE);
  }
  board.setAttribute("viewBox", `${minX - 10} ${minY - 10} ${maxX - minX + 20} ${maxY - minY + 20}`);
  const clicks = boardActions();
  const robber = game.state.board.robber_coordinate.join(",");

  for (const tile of game.board) {
    const [cx, cy] = pixel(tile.coordinate);
    const points = Object.values(CORNERS).map(([dx, dy]) => `${cx + dx},${cy + dy}`).join(" ");
    const fill = tile.type === "land" ? TILE_COLORS[tile.resource || "DESERT"] : "#6fb3dd";
    const key = tile.coordinate.join(",");
    const hex = svg("polygon", { points, fill, stroke: "#5b5b5b", "stroke-width": tile.type === "land" ? 2 : 0.5 });
    board.append(hex);
    if (tile.type === "land" && tile.number) {
      board.append(svg("circle", { cx, cy, r: 15, fill: "#fdf6e3", stroke: "#333" }));
      const hot = tile.number === 6 || tile.number === 8;
      board.append(svg("text", { x: cx, y: cy + 5, "text-anchor": "middle", "font-size": 15, "font-weight": 700, fill: hot ? "#c53030" : "#222" }, tile.number));
      board.append(svg("text", { x: cx, y: cy + 13, "text-anchor": "middle", "font-size": 7, fill: hot ? "#c53030" : "#222" }, "•".repeat(PIPS[tile.number])));
    }
    if (tile.type === "port") {
      for (const node of tile.port_nodes) {
        const [nx, ny] = positions[node];
        board.append(svg("line", { x1: cx, y1: cy, x2: nx, y2: ny, stroke: "#7a5c3a", "stroke-width": 4 }));
      }
      board.append(svg("circle", { cx, cy, r: 16, fill: "#fff", stroke: "#7a5c3a" }));
      const label = tile.port_resource ? "2:1" : "3:1";
      board.append(svg("text", { x: cx, y: cy - 1, "text-anchor": "middle", "font-size": 10, "font-weight": 700 }, label));
      if (tile.port_resource) board.append(svg("text", { x: cx, y: cy + 10, "text-anchor": "middle", "font-size": 7 }, tile.port_resource));
    }
    if (key === robber) {
      board.append(svg("circle", { cx: cx - 22, cy: cy - 14, r: 9, fill: "#222", stroke: "#fff", "stroke-width": 1.5 }));
      board.append(svg("text", { x: cx - 22, y: cy - 10, "text-anchor": "middle", "font-size": 10, fill: "#fff" }, "R"));
    }
    if (clicks.tiles[key]) {
      hex.setAttribute("stroke", "#ffd400");
      hex.setAttribute("stroke-width", 5);
      hex.classList.add("clickable");
      hex.onclick = () => chooseRobber(clicks.tiles[key]);
    }
  }

  const drawn = new Set();
  for (const [[a, b], color] of game.state.board.roads) {
    const id = [a, b].sort((x, y) => x - y).join(",");
    if (drawn.has(id)) continue;
    drawn.add(id);
    const [x1, y1] = positions[a], [x2, y2] = positions[b];
    board.append(svg("line", { x1, y1, x2, y2, stroke: "#222", "stroke-width": 9, "stroke-linecap": "round" }));
    board.append(svg("line", { x1, y1, x2, y2, stroke: PLAYER_COLORS[color], "stroke-width": 6, "stroke-linecap": "round" }));
  }
  for (const [id, action] of Object.entries(clicks.edges)) {
    const [a, b] = id.split(",").map(Number);
    const [x1, y1] = positions[a], [x2, y2] = positions[b];
    const line = svg("line", { x1, y1, x2, y2, stroke: "#ffd400", "stroke-width": 8, "stroke-linecap": "round", opacity: 0.8, class: "clickable" });
    line.onclick = () => session.act(action);
    board.append(line);
  }
  for (const [node, color, kind] of game.state.board.buildings) {
    const [x, y] = positions[node];
    const shape = kind === "CITY"
      ? svg("rect", { x: x - 10, y: y - 10, width: 20, height: 20, rx: 3 })
      : svg("circle", { cx: x, cy: y, r: 8 });
    shape.setAttribute("fill", PLAYER_COLORS[color]);
    shape.setAttribute("stroke", "#222");
    shape.setAttribute("stroke-width", 2);
    board.append(shape);
  }
  for (const [node, action] of Object.entries(clicks.nodes)) {
    const [x, y] = positions[node];
    const spot = svg("circle", { cx: x, cy: y, r: action[1] === "BUILD_CITY" ? 13 : 8, fill: "#ffd400", opacity: 0.85, stroke: "#222", class: "clickable" });
    spot.onclick = () => session.act(action);
    board.append(spot);
  }
  $("#prompt").textContent = promptText();
}

function chooseRobber(actions) {
  if (actions.length === 1) { session.act(actions[0]); return; }
  const victims = actions.map((a) => a[2][1] || "nobody").join(", ");
  const pick = prompt(`Steal from whom? (${victims})`, actions[0][2][1] || "");
  const action = actions.find((a) => (a[2][1] || "nobody") === (pick || "nobody").toUpperCase());
  if (action) session.act(action);
}

function promptText() {
  if (session.aborted) return `The game is void: ${session.aborted.reason}` +
    (session.aborted.blame.length ? ` (seats ${session.aborted.blame.join(", ")} did not reveal their seeds)` : "");
  const game = session.game;
  if (game.over) {
    return game.winner ? `${game.winner} (${seatInfo(game.winner).name || ""}) wins!` : "Game over (turn limit).";
  }
  const whose = `${game.current} (${seatInfo(game.current).name || ""})`;
  if (session.pending && !session.answered) {
    const left = Math.max(0, Math.round((session.pending.body.timeout_ms - (Date.now() - session.pending.received)) / 1000));
    return `Your move: ${prettyPrompt(game.state.current_prompt)} (${left}s)`;
  }
  return `${whose}: ${prettyPrompt(game.state.current_prompt)}`;
}
const prettyPrompt = (prompt) => String(prompt || "").toLowerCase().replace(/_/g, " ");

function seatIndex(color) { return session.game.state.colors.indexOf(color); }
function ps(color, key) { return session.game.state.player_state[`P${seatIndex(color)}_${key}`]; }

function renderMe() {
  const game = session.game;
  const box = $("#me");
  if (!game.color) {
    box.innerHTML = '<h3>Spectating</h3><div class="muted">Every hand is hidden from you.</div>';
    return;
  }
  const cards = RESOURCES.map((r) => html`<span>${r.toLowerCase()} ${ps(game.color, r + "_IN_HAND")}</span>`);
  const held = DEV_CARDS.map((c) => [c, ps(game.color, c + "_IN_HAND")]).filter(([, n]) => n > 0)
    .map(([c, n]) => html`<span>${c.toLowerCase().replace(/_/g, " ")} ${n}</span>`);
  const devs = held.length ? held : html`<span class="muted">none</span>`;
  setHtml(box, html`<h3><span class="swatch" style="background:${swatch(game.color)}"></span>You are ${game.color} · ${ps(game.color, "ACTUAL_VICTORY_POINTS")} VP</h3><div class="cards">${cards}</div><div class="cards">Development cards: ${devs}</div>`);
}

function actionLabel(action) {
  const [, type, value] = action;
  switch (type) {
    case "ROLL": return "Roll the dice";
    case "END_TURN": return "End turn";
    case "BUY_DEVELOPMENT_CARD": return "Buy a development card";
    case "PLAY_KNIGHT_CARD": return "Play a knight";
    case "PLAY_ROAD_BUILDING": return "Play road building";
    case "PLAY_YEAR_OF_PLENTY": return "Year of plenty: " + value.join(" + ").toLowerCase();
    case "PLAY_MONOPOLY": return "Monopoly: " + value.toLowerCase();
    case "DISCARD_RESOURCE": return "Discard " + String(value).toLowerCase();
    case "MARITIME_TRADE": {
      const give = value.slice(0, 4).filter((r) => r);
      return `Trade ${give.length} ${give[0].toLowerCase()} for 1 ${value[4].toLowerCase()}`;
    }
    case "BUILD_SETTLEMENT": return `Settlement at node ${value}`;
    case "BUILD_CITY": return `City at node ${value}`;
    case "BUILD_ROAD": return `Road ${value.join("-")}`;
    case "MOVE_ROBBER": return `Robber to ${value[0].join(",")}` + (value[1] ? `, steal from ${value[1]}` : "");
    case "ACCEPT_TRADE": return "Accept the trade";
    case "REJECT_TRADE": return "Reject the trade";
    case "CONFIRM_TRADE": return `Trade with ${value[10]}`;
    case "CANCEL_TRADE": return "Cancel the trade";
    default: return type + " " + JSON.stringify(value);
  }
}
const ON_BOARD = new Set(["BUILD_SETTLEMENT", "BUILD_CITY", "BUILD_ROAD", "MOVE_ROBBER"]);

function renderActions() {
  const box = $("#actions");
  box.innerHTML = "";
  if (!session.pending) {
    box.innerHTML = '<h3>Actions</h3><div class="muted">' + (session.game.over ? "The game is over." : "Not your move.") + "</div>";
    return;
  }
  box.insertAdjacentHTML("beforeend", "<h3>Your move</h3>");
  if (session.answered) { box.insertAdjacentHTML("beforeend", '<div class="muted">Sent.</div>'); return; }
  const actions = session.pending.body.playable_actions;
  const offBoard = actions.filter((a) => !ON_BOARD.has(a[1]));
  for (const action of offBoard) box.append(button(actionLabel(action), () => session.act(action), "primary"));
  const onBoard = actions.filter((a) => ON_BOARD.has(a[1]));
  if (onBoard.length) {
    box.insertAdjacentHTML("beforeend", '<div class="muted">Click a highlighted spot on the board, or:</div>');
    const details = document.createElement("details");
    details.innerHTML = `<summary>${onBoard.length} board moves as a list</summary>`;
    const list = document.createElement("div");
    list.className = "list";
    for (const action of onBoard) list.append(button(actionLabel(action), () => session.act(action)));
    details.append(list);
    box.append(details);
  }
}

function renderPlayers() {
  const game = session.game;
  const box = $("#players");
  const state = game.state;
  setHtml(box, html`<h3>Table</h3>`);
  for (const color of state.colors) {
    const mine = color === game.color;
    const resources = mine ? RESOURCES.reduce((n, r) => n + ps(color, r + "_IN_HAND"), 0) : ps(color, "NUM_RESOURCES_IN_HAND");
    const devs = mine ? DEV_CARDS.reduce((n, c) => n + ps(color, c + "_IN_HAND"), 0) : ps(color, "NUM_DEVELOPMENT_CARDS_IN_HAND");
    const badges = [ps(color, "HAS_ROAD") ? "longest road" : "", ps(color, "HAS_ARMY") ? "largest army" : ""].filter(Boolean).join(", ");
    const seat = seatInfo(color);
    const row = document.createElement("div");
    row.className = "player" + (color === game.current && !game.over ? " current" : "");
    setHtml(row, html`<div><span class="swatch" style="background:${swatch(color)}"></span><b>${seat.name || color}</b> <span class="muted">${seat.kind || ""}</span></div><div><b>${ps(color, "VICTORY_POINTS")}</b> VP</div><div class="muted">${resources} cards · ${devs} dev · ${ps(color, "PLAYED_KNIGHT")} knights · road ${ps(color, "LONGEST_ROAD_LENGTH")}</div><div class="muted">${badges}</div>`);
    box.append(row);
  }
  const bank = state.resource_freqdeck.map((n, i) => html`<span>${RESOURCES[i].toLowerCase()} ${n}</span>`);
  const deck = state.development_deck_size !== undefined ? state.development_deck_size : "?";
  addHtml(box, html`<div class="cards">Bank: ${bank}<span>dev deck ${deck}</span></div>`);
  const roll = [...game.log].reverse().find(([a]) => a[1] === "ROLL");
  if (roll) addHtml(box, html`<div>Last roll: <b>${roll[1][0]} + ${roll[1][1]} = ${Number(roll[1][0]) + Number(roll[1][1])}</b> (${roll[0][0]})</div>`);
}

function logText([action, result]) {
  const [color, type, value] = action;
  switch (type) {
    case "ROLL": return `${color} rolled ${result[0]} + ${result[1]} = ${result[0] + result[1]}`;
    case "BUY_DEVELOPMENT_CARD": return `${color} bought a development card` + (value ? ` (${value.toLowerCase()})` : "");
    case "MOVE_ROBBER": return `${color} moved the robber to ${value[0].join(",")}` +
      (value[1] ? `, stealing from ${value[1]}` + (result ? ` (${result.toLowerCase()})` : "") : "");
    case "END_TURN": return `${color} ended the turn`;
    default: return `${color}: ${actionLabel(action).toLowerCase()}`;
  }
}

function renderLog() {
  const list = $("#log");
  list.innerHTML = "";
  const log = session.game.log;
  for (let i = log.length - 1; i >= Math.max(0, log.length - 80); i--) {
    const item = document.createElement("li");
    item.value = i + 1;
    item.textContent = logText(log[i]);
    list.append(item);
  }
}

function renderFairness() {
  const box = $("#fairness");
  const s = session;
  const short = (value) => String(value || "").slice(0, 16);
  const lines = [html`<h3>Fairness</h3>`];
  if (s.serverCommitment) lines.push(html`<div class="muted">Seed commitment: <code>${short(s.serverCommitment)}…</code></div>`);
  const key = s.serverKey ? html`; server key <code>${short(s.serverKey)}…</code>` : html``;
  lines.push(html`<div>${s.messages.length} messages, ${s.verifiedCount} signatures checked here${key}</div>`);
  if (s.unverified) lines.push(html`<div class="error">${s.unverified} messages could not be checked in this browser: this session is unverified.</div>`);
  if (serverInfo.app_sha256) {
    lines.push(html`<div class="muted">The server says it serves app.js with SHA-256 <code>${short(serverInfo.app_sha256)}…</code>. This page only protects you from the server if it (and index.html) did not come from that server: see the README.</div>`);
  }
  if (cryptoProblem) lines.push(html`<div class="error">${cryptoProblem}</div>`);
  for (const problem of s.problems.slice(-3)) lines.push(html`<div class="error">${problem}</div>`);
  setHtml(box, html`${lines}`);
  if (s.aborted || (s.game && s.game.over)) {
    addHtml(box, html`<div>The server revealed its seed. Check the game with <code>catanatron-verify</code>:</div>`);
    box.append(button("Download transcript", () => download((s.saveId || "transcript") + ".json", JSON.stringify(s.transcript())), "primary"));
  }
}

// ===== routing =====
function route() {
  const match = location.hash.match(/room=([A-Za-z0-9_-]+)/);
  if (match) openRoom(match[1]);
  else showLobby();
}
window.addEventListener("hashchange", route);
let serverInfo = {};
window.addEventListener("load", async () => {
  await checkCrypto();
  fetch(apiUrl("/api/server")).then((r) => r.json()).then((info) => { serverInfo = info; }).catch(() => {});
  $("#crypto-status").textContent = cryptoProblem ? "watch-only browser" : "signing with Ed25519 (WebCrypto)";
  $("#create").addEventListener("submit", createRoom);
  $("#create [name=num_players]").addEventListener("change", renderSeatSpecs);
  $("#refresh").onclick = refreshRooms;
  $("#home").onclick = (event) => { event.preventDefault(); location.hash = ""; };
  setInterval(() => { if (session && session.pending && !session.answered) $("#prompt").textContent = promptText(); }, 1000);
  route();
});
