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
  WOOD: "#3a7d44", BRICK: "#c65d3b", SHEEP: "#9bc53d",
  WHEAT: "#e9b949", ORE: "#8e9aa6", DESERT: "#dcc48f",
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
// A message for the next time the lobby is shown (a table that was busy).
let lobbyNotice = "";
async function showLobby() {
  compactSeats();
  $("#room").hidden = true;
  $("#lobby").hidden = false;
  $("#room-title").textContent = "";
  closeModal();
  if (session) session.close();
  session = null;
  // A house server opens every room itself: no room form, no plain list.
  const house = Boolean(serverInfo.house);
  $("#create-panel").hidden = house;
  $("#rooms-panel").hidden = house;
  $("#tables").hidden = !house;
  $("#lobby-notice").hidden = !lobbyNotice;
  $("#lobby-notice").textContent = lobbyNotice;
  lobbyNotice = "";
  $("#lobby-name").value = storedName();
  await refreshRooms();
  if (!house) {
    if (!players.length) players = await (await fetch(apiUrl("/api/players"))).json();
    renderSeatSpecs();
  }
  const list = $("#transcripts");
  list.innerHTML = "";
  for (const entry of await listTranscripts()) {
    const item = document.createElement("li");
    const link = document.createElement("a");
    link.href = "#";
    link.textContent = entry.id;
    link.onclick = (event) => { event.preventDefault(); download(entry.id + ".json", entry.json); };
    item.append(link);
    list.append(item);
  }
  if (!list.children.length) list.innerHTML = '<li class="muted">none</li>';
}
async function refreshRooms() {
  const all = await (await fetch(apiUrl("/api/rooms"))).json();
  renderTables(all);
  renderLive(all);
  // Finished and void rooms stay listed for an hour (to watch or verify);
  // hide them unless asked.
  const showDone = $("#show-finished").checked;
  const rooms = showDone ? all : all.filter((room) => room.status !== "finished" && room.status !== "aborted");
  const hidden = all.length - rooms.length;
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
  if (!rooms.length) {
    body.innerHTML = `<tr><td colspan="5" class="muted">${hidden ? "no open or running rooms" : "no rooms yet"}</td></tr>`;
  }
  $("#hidden-count").textContent = hidden && !showDone ? `(${hidden} hidden)` : "";
}
const isHouseRoom = (room) => Array.isArray(room.house_seats) && room.house_seats.length > 0;
// A house room a person can sit in now: every seat but one held by the
// house (and connected), the last one open.
function readyRoom(room) {
  return isHouseRoom(room) && room.status === "waiting" && room.house_seats.length === room.seats.length - 1 &&
    room.seats.every((s, i) => (room.house_seats.includes(i) ? s.connected : s.kind === "open"));
}
// The house's tables, by room name: what they play, who the bots are, how
// many games are on, and a room to sit in if there is one.
function tableKinds(rooms) {
  const kinds = new Map();
  for (const room of rooms) {
    if (!isHouseRoom(room)) continue;
    let kind = kinds.get(room.name);
    if (!kind) kinds.set(room.name, (kind = { name: room.name, config: room.config, bots: [], playing: 0, open: null }));
    for (const index of room.house_seats) {
      const name = room.seats[index] && room.seats[index].name;
      if (name && !kind.bots.includes(name)) kind.bots.push(name);
    }
    if (room.status === "playing" || room.status === "revealing") kind.playing++;
    if (!kind.open && readyRoom(room)) kind.open = room;
  }
  return [...kinds.values()].sort((a, b) => a.config.num_players - b.config.num_players);
}
function rulesText(config) {
  const parts = [config.num_players === 2 ? "1v1" : `${config.num_players} players`, `${config.vps_to_win} points`];
  if (config.discard_limit !== undefined && config.discard_limit !== 7) parts.push(`discard over ${config.discard_limit}`);
  return parts.join(" · ");
}
function renderTables(rooms) {
  const box = $("#tables");
  box.innerHTML = "";
  const kinds = tableKinds(rooms);
  for (const kind of kinds) {
    const card = document.createElement("div");
    card.className = "table-card";
    const n = Number(kind.config.num_players) || 0;
    const bots = kind.bots.join(", ");
    const status = kind.open ? html`<span class="status open">open</span>`
      : html`<span class="status">busy${kind.playing ? html` · ${kind.playing} playing` : html``}</span>`;
    setHtml(card, html`<h3>${kind.name}</h3><div class="meta">${rulesText(kind.config)}</div>
      <div class="bots">vs ${n - 1 > 1 ? html`${n - 1} × ` : html``}${bots}</div><div class="foot">${status}</div>`);
    const play = button("Play", () => playTable(kind.name), "primary big");
    play.disabled = !kind.open;
    card.querySelector(".foot").append(play);
    box.append(card);
  }
  if (!kinds.length) box.innerHTML = '<div class="panel muted">No tables open yet.</div>';
}
function renderLive(rooms) {
  const list = $("#live");
  list.innerHTML = "";
  for (const room of rooms.filter((r) => r.status === "playing").reverse()) {
    const item = document.createElement("li");
    const seats = room.seats.map((s) => html`<span class="swatch" style="background:${swatch(s.color)}"></span>${s.name || s.color} `);
    setHtml(item, html`<b>${room.name}</b><span class="grow">${seats}</span>`);
    item.append(button("Watch", () => { location.hash = "room=" + room.id; }));
    list.append(item);
  }
  if (!list.children.length) list.innerHTML = '<li class="muted">none</li>';
}
function saveName() {
  const field = $("#lobby-name");
  if (field && field.value !== undefined) {
    try { localStorage.setItem("catanatron-name", field.value.trim()); } catch (error) { /* not kept */ }
  }
}
// Sit at an open table of this kind (the room to sit in is looked up now:
// the lobby's list may be stale).
async function playTable(name) {
  saveName();
  let rooms = [];
  try { rooms = await (await fetch(apiUrl("/api/rooms"))).json(); } catch (error) { /* shown below */ }
  const room = rooms.find((r) => r.name === name && readyRoom(r));
  if (!room) {
    lobbyNotice = `${name} is busy. A table opens when a game ends.`;
    if (location.hash) location.hash = ""; else showLobby();
    return;
  }
  location.hash = `room=${room.id}&sit=1&table=${encodeURIComponent(name)}`;
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
// True only if the record is stored (read back): the page sends what a
// record holds (a join, a reveal, a move) only once it is kept, so a reload
// can never make it sign again what it signed before.
function storeSeat(key, saved) {
  try {
    if (saved) {
      const text = JSON.stringify(saved);
      localStorage.setItem(key, text);
      return localStorage.getItem(key) === text;
    }
    localStorage.removeItem(key);
    return true;
  } catch (error) {
    console.warn("could not store the seat", error);
    return false;
  }
}
// Seat records of ended games keep only what stops the page from sitting
// in them again: the transcripts are in IndexedDB, and the store must not
// fill up.
function compactSeats() {
  try {
    for (let i = localStorage.length - 1; i >= 0; i--) {
      const key = localStorage.key(i);
      if (!key || !key.startsWith("catanatron-seat-")) continue;
      const record = loadSeat(key);
      if (record && record.closed && record.identity) {
        localStorage.setItem(key, JSON.stringify({ closed: true, seedCommitment: record.seedCommitment }));
      }
    }
  } catch (error) { /* a store we cannot read: nothing to compact */ }
}
const COLORS = Object.keys(PLAYER_COLORS); // seat i plays COLORS[i]
// Lone surrogates: Python cannot encode them, so no honest server signs one.
function stringsOk(value) {
  if (typeof value === "string") return !/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?:^|[^\uD800-\uDBFF])[\uDC00-\uDFFF]/.test(value);
  if (Array.isArray(value)) return value.every(stringsOk);
  if (value && typeof value === "object") return Object.entries(value).every(([k, v]) => stringsOk(k) && stringsOk(v));
  return true;
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
    this.lastDecided = -1; // the last move this page was asked to decide
    this.halted = false; // a seat record could not be stored: sign nothing more
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
  persist() {
    if (!this.storageKey || storeSeat(this.storageKey, this.saved)) return true;
    this.problems.push("this browser could not store the seat (its storage is full?): nothing more is signed or sent");
    this.halted = true;
    return false;
  }
  remember(changes) { this.saved = { ...this.saved, ...changes }; return this.persist(); }

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
    if (!stringsOk(env)) throw new Error("a message with text no honest server signs (a lone surrogate)");
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
    const rules = this.saved && this.saved.config ? this.saved.config : (this.room ? canonical(this.room.config) : null);
    if (room.seed_commitment !== this.serverCommitment || (rules !== null && canonical(room.config) !== rules)) {
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
        // (A reveal resent after the game started is answered not_revealing:
        // harmless then; while seeds are revealed, it is the server refusing.)
        if (body.code !== "not_revealing" || (this.commitments && !this.game)) this.problems.push(body.message);
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
        if (body.type === "decide" && !(Number.isInteger(body.index) && body.index > this.lastDecided)) {
          // A second decide for a move would make a second answer (and let
          // the server choose between them).
          this.problems.push(`a second decide for move ${body.index}`);
          break;
        }
        this.update(body);
        this.closePending();
        if (body.type === "decide") {
          this.lastDecided = body.index;
          this.pending = { env, hash: env.hash, body, received: Date.now() };
          // Replayed after a reload: answered already, if the record says so.
          if (this.seated && this.sent.some((s) => s.statement && s.statement.seq === env.seq)) this.answered = true;
        }
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
      if (body.color !== this.saved.color) {
        this.problems.push("the game seats this page in another color");
        return false;
      }
    } else if (body.color !== null) {
      this.problems.push("a spectator was sent a seat's view");
      return false;
    }
    return true;
  }

  closePending() {
    // A newer message closed the open decision: if this page had not
    // answered it, keep a record (the server timed it out, or so it says).
    const notes = this.saved ? this.saved.unanswered || [] : [];
    if (this.pending && !this.answered && this.seated && !notes.some((n) => n.index === this.pending.body.index)) {
      const note = { index: this.pending.body.index, seq: this.pending.env.seq, ms: Date.now() - this.pending.received };
      this.remember({ unanswered: [...notes, note] });
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
    if (this.halted || !this.remember({ reveal: { commitments: C, sig, digest } })) return;
    this.send(this.revealRequest());
  }
  async mySeatProblem(body) {
    const keys = body.seats.filter((s) => s && s.kind === "remote").map((s) => s.public_key);
    if (new Set(keys).size !== keys.length) return "the commitments seat one key twice";
    // Every seat, not just ours: a reveal signs them all.
    for (const [index, seat] of body.seats.entries()) {
      if (!seat || Object.keys(seat).sort().join(",") !== SEAT_FIELDS.join(",") || seat.seat !== index ||
          seat.color !== COLORS[index]) {
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
        seat.kind !== "remote" || seat.seat !== this.saved.seat || seat.color !== this.saved.color ||
        seat.requested_seat !== this.saved.requested ||
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
    if (this.halted) return;
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
      seedCommitment: this.serverCommitment, config: canonical(this.room.config),
      joined: false, sent: [], unreceipted: [], unanswered: [],
    };
    if (!this.persist()) {
      this.saved = null;
      this.joining = false;
      render();
      return;
    }
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
    if (!this.pending || this.answered || this.ended || this.halted) return;
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
    const kept = this.remember({
      sent: [...this.saved.sent, { statement, sig, sent_after: this.messages.length - 1 }],
      unreceipted: [...this.saved.unreceipted, { digest: this.answerDigest, answer: this.answer }],
    });
    if (!kept) return;
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

// ===== the room page =====
const RES_COLORS = { WOOD: "#2f7a3d", BRICK: "#c2410c", SHEEP: "#84cc16", WHEAT: "#eab308", ORE: "#64748b" };
const RES_LABEL = { WOOD: "wood", BRICK: "brick", SHEEP: "sheep", WHEAT: "wheat", ORE: "ore" };
const DEV_NAME = {
  KNIGHT: "Knight", YEAR_OF_PLENTY: "Year of Plenty", MONOPOLY: "Monopoly",
  ROAD_BUILDING: "Road Building", VICTORY_POINT: "Victory Point",
};
const DEV_PLAY = { KNIGHT: "PLAY_KNIGHT_CARD", ROAD_BUILDING: "PLAY_ROAD_BUILDING", YEAR_OF_PLENTY: "PLAY_YEAR_OF_PLENTY", MONOPOLY: "PLAY_MONOPOLY" };
const COSTS = {
  road: ["WOOD", "BRICK"], settlement: ["WOOD", "BRICK", "SHEEP", "WHEAT"],
  city: ["WHEAT", "WHEAT", "ORE", "ORE", "ORE"], dev: ["SHEEP", "WHEAT", "ORE"],
};
const BUILD_TYPES = { road: "BUILD_ROAD", settlement: "BUILD_SETTLEMENT", city: "BUILD_CITY" };
// A resource, as a colored word (or a swatch when there is no room for one).
const res = (resource) => html`<span style="color:${RES_COLORS[resource] || "#999"};font-weight:600">${RES_LABEL[resource] || String(resource).toLowerCase()}</span>`;
const dot = (resource) => html`<span class="res res-${resource}"></span>`;
// What the page shows besides the protocol: the build mode, the open
// dialog, whether it was sent here to sit (and at which kind of table),
// and which seats the house holds.
const ui = {
  build: null, modal: null, autoSit: false, table: "", sitTried: false, retries: 0,
  seenEnd: false, houseSeats: [], houseRoom: null,
};

async function openRoom(roomId) {
  $("#lobby").hidden = true;
  $("#room").hidden = false;
  closeModal();
  Object.assign(ui, { build: null, sitTried: false, seenEnd: false, houseSeats: [], houseRoom: null });
  if (session) session.close();
  session = new Session(roomId);
  session.connect();
  render();
}

// ===== rendering =====
function render() {
  if (!session) return;
  renderSeating();
  maybeSit();
  loadHouseSeats();
  if (session.game) {
    renderBoard();
    renderMe();
    renderActions();
    renderPlayers();
    renderBank();
    renderLog();
    renderDice();
  } else {
    $("#board").innerHTML = "";
    for (const id of ["#me", "#actions", "#players", "#bank", "#dice"]) $(id).innerHTML = "";
    $("#log").innerHTML = "";
    $("#prompt").textContent = session.aborted ? promptText() : "";
  }
  renderTimer();
  renderModal();
  renderFairness();
}

function swatch(color) { return PLAYER_COLORS[color] || "#999"; }
// Readable on white (a white seat's name is shown in grey).
function inkOf(color) { return color === "WHITE" ? "#6b7280" : swatch(color); }
function storedName() { try { return localStorage.getItem("catanatron-name") || ""; } catch (error) { return ""; } }

function seatInfo(color) {
  const seats = session.game ? session.game.seats : (session.room ? session.room.seats : []);
  return seats.find((s) => s.color === color) || {};
}
// (Seat i plays COLORS[i]; a game's seat list names colors, not indexes.)
const isBot = (seat) => seat.kind === "bot" || ui.houseSeats.includes(seat.seat !== undefined ? seat.seat : COLORS.indexOf(seat.color));
// A seat's name, with its color when another seat has the same name (the
// house's bots all go by their model's name).
function seatName(color) {
  const seat = seatInfo(color);
  const name = seat.name || String(color || "").toLowerCase();
  const seats = session.game ? session.game.seats : (session.room ? session.room.seats : []);
  const twins = seats.filter((s) => s.name === seat.name).length > 1;
  return twins ? `${name} (${String(color).toLowerCase()})` : name;
}
function nameOf(color) {
  if (session.game && color === session.game.color) return "You";
  return seatName(color);
}
function avatarHtml(seat, color) {
  const name = seat.name || String(color || "");
  const initial = Array.from(name.trim())[0] || "?";
  return html`<span class="avatar${color === "WHITE" ? " light" : ""}" style="background:${swatch(color)}">${initial.toUpperCase()}</span>`;
}
const botTag = (seat) => (isBot(seat) ? html` <span class="tag">bot</span>` : html``);

// Which seats the house holds (the server's list says; a room message does
// not): asked once the room is full, when it no longer changes.
function loadHouseSeats() {
  // (The page's copy of the room stops at its last room message, while it
  // was still waiting: the commitments or the game say it is full.)
  if (!serverInfo.house || !session.room || !(session.commitments || session.game)) return;
  const id = session.roomId;
  if (ui.houseRoom === id) return;
  ui.houseRoom = id;
  fetch(apiUrl("/api/rooms/" + id)).then((r) => r.json()).then((room) => {
    if (session && session.roomId === id && Array.isArray(room.house_seats)) {
      ui.houseSeats = room.house_seats;
      render();
    }
  }).catch(() => {});
}

function renderSeating() {
  const box = $("#seating");
  const room = session.room;
  if (room) $("#room-title").textContent = room.name || "";
  if (!room || session.game || session.aborted) {
    box.innerHTML = "";
    return;
  }
  const house = Boolean(serverInfo.house);
  const canSit = !session.joinSent && !session.spectating && !session.joining && room.status === "waiting";
  setHtml(box, html`<h3>${room.name}</h3><div class="muted">${rulesText(room.config)}</div>`);
  if (canSit || (session.joining && !session.seated)) {
    addHtml(box, html`<label>Name <input id="my-name" maxlength="32" placeholder="Guest" value="${storedName()}"></label>`);
    $("#my-name").onchange = (e) => localStorage.setItem("catanatron-name", e.target.value);
  }
  for (const seat of room.seats) {
    const row = document.createElement("div");
    row.className = "seat-row";
    const mine = session.seated && session.saved.seat === seat.seat;
    const away = seat.kind === "remote" && !seat.connected ? html` <span class="muted">(away)</span>` : html``;
    const who = seat.kind === "open" ? html`<span class="muted">open</span>`
      : html`<b>${seat.name}</b>${mine ? html` <span class="muted">(you)</span>` : seat.kind === "bot" || house ? html` <span class="tag">bot</span>` : html``}${away}`;
    setHtml(row, html`<span class="swatch" style="background:${swatch(seat.color)}"></span><span class="grow">${who}</span>`);
    if (seat.kind === "open" && room.status === "waiting") {
      if (canSit) row.append(button("Sit here", () => session.takeSeat(seat.seat), "primary"));
      if (!house) {
        const select = document.createElement("select");
        options(select, players.map((p) => [p.key, p.key]));
        select.value = "F";
        const add = button("Add bot", () => session.seatBot(seat.seat, select.value));
        add.disabled = session.joining; // its errors would be mistaken for the join's
        row.append(select, add);
      }
    }
    box.append(row);
  }
  let status = html``;
  if (session.joining) status = html`<p><span class="spinner"></span>Joining…</p>`;
  else if (session.commitments) status = html`<p><span class="spinner"></span>Starting…</p>`;
  else if (session.seated) status = html`<p><span class="spinner"></span>Waiting for the other seats…</p>`;
  else if (session.spectating) status = html`<p class="muted">Watching.</p>`;
  addHtml(box, status);
  for (const problem of session.problems.slice(-1)) addHtml(box, html`<p class="error">${problem}</p>`);
  const foot = document.createElement("div");
  foot.className = "foot";
  if (!session.spectating && !session.joining && !session.joinSent) foot.append(button("Watch", () => session.spectate()));
  foot.append(button("Lobby", () => { location.hash = ""; }));
  box.append(foot);
  if (!house && !players.length) fetch(apiUrl("/api/players")).then((r) => r.json()).then((p) => { players = p; renderSeating(); });
}

// Sent here to sit ("Play" in the lobby): take the open seat, or, if
// someone took it first, look for another table of the same kind.
function maybeSit() {
  const s = session;
  if (!ui.autoSit || !s.room || s.game || s.aborted) return;
  if (!ui.sitTried) {
    if (s.spectating || s.room.status !== "waiting") { retryTable(); return; }
    if (s.joinSent || s.joining || !s.storageKey || !NONCE.test(s.helloNonce || "")) return;
    ui.sitTried = true;
    s.takeSeat(null);
    try { history.replaceState(null, "", `#room=${s.roomId}&table=${encodeURIComponent(ui.table)}`); } catch (error) { /* cosmetic */ }
    return;
  }
  if (!s.joining && !s.joinSent) retryTable(); // the join was refused
}
function retryTable() {
  ui.autoSit = false;
  if (ui.table && ui.retries++ < 3) setTimeout(() => playTable(ui.table), 400);
}

function button(label, onclick, className) {
  const element = document.createElement("button");
  element.textContent = label;
  element.onclick = onclick;
  if (className) element.className = className;
  return element;
}

// ===== the board =====
const pixel = (coordinate) => [W * (coordinate[0] + coordinate[2] / 2), 1.5 * SIZE * coordinate[2]];
const svg = (tag, attrs, text) => {
  const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs || {})) element.setAttribute(key, value);
  if (text !== undefined) element.textContent = text;
  return element;
};
const PIPS = { 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1 };
const hexPoints = (cx, cy, r) => Object.values(CORNERS).map(([dx, dy]) => `${cx + dx * r / SIZE},${cy + dy * r / SIZE}`).join(" ");
const shape = (x, y, points) => points.map(([dx, dy]) => `${x + dx},${y + dy}`).join(" ");
const SETTLEMENT = [[-8, 7], [-8, -2], [0, -10], [8, -2], [8, 7]];
const CITY = [[-13, 9], [-13, -3], [-6, -10], [1, -3], [1, -1], [13, -1], [13, 9]];

function boardActions() {
  // Legal moves that are a click on the board, keyed by what is clicked.
  const out = { nodes: {}, edges: {}, tiles: {} };
  if (!session.pending || session.answered) return out;
  const actions = session.pending.body.playable_actions;
  // Shown at once when the board is all there is to decide (placing, the
  // robber, a free road); otherwise only the build the player picked.
  const all = actions.every((a) => ON_BOARD.has(a[1]));
  for (const action of actions) {
    const [, type, value] = action;
    if (!all && type !== "MOVE_ROBBER" && BUILD_TYPES[ui.build] !== type) continue;
    if (type === "BUILD_SETTLEMENT" || type === "BUILD_CITY") out.nodes[value] = action;
    else if (type === "BUILD_ROAD") out.edges[[...value].sort((a, b) => a - b).join(",")] = action;
    else if (type === "MOVE_ROBBER") (out.tiles[value[0].join(",")] = out.tiles[value[0].join(",")] || []).push(action);
  }
  return out;
}

// The sum of this turn's roll, if the current player has rolled.
function currentRoll() {
  const log = session.game.log;
  for (let i = log.length - 1; i >= 0; i--) {
    const [action, result] = log[i];
    if (action[1] === "END_TURN") return null;
    if (action[1] === "ROLL" && Array.isArray(result)) return Number(result[0]) + Number(result[1]);
  }
  return null;
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
  board.setAttribute("viewBox", `${minX - 6} ${minY - 6} ${maxX - minX + 12} ${maxY - minY + 12}`);
  const clicks = boardActions();
  const robber = game.state.board.robber_coordinate.join(",");
  const rolled = currentRoll();

  for (const tile of game.board) {
    if (tile.type !== "land") continue;
    const [cx, cy] = pixel(tile.coordinate);
    const resource = tile.resource || "DESERT";
    board.append(svg("polygon", { points: hexPoints(cx, cy, SIZE + 3), fill: "#e8d9b0" }));
    board.append(svg("polygon", { points: hexPoints(cx, cy, SIZE - 2), fill: TILE_COLORS[resource] }));
    board.append(svg("polygon", { points: hexPoints(cx, cy, SIZE - 9), fill: "none", stroke: "rgba(255,255,255,0.18)", "stroke-width": 3 }));
    if (tile.number) {
      const hot = tile.number === 6 || tile.number === 8;
      const lit = rolled !== null && tile.number === rolled;
      board.append(svg("circle", { cx, cy, r: 14, fill: lit ? "#fde68a" : "#fbf6e6", stroke: lit ? "#d97706" : "#b9ab84", "stroke-width": lit ? 3 : 1.5 }));
      board.append(svg("text", { x: cx, y: cy + 4, "text-anchor": "middle", "font-size": 14, "font-weight": 700, fill: hot ? "#b91c1c" : "#1f2937" }, tile.number));
      board.append(svg("text", { x: cx, y: cy + 11, "text-anchor": "middle", "font-size": 6, fill: hot ? "#b91c1c" : "#1f2937" }, "•".repeat(PIPS[tile.number] || 0)));
    }
    const key = tile.coordinate.join(",");
    if (key === robber) {
      board.append(svg("polygon", { points: hexPoints(cx, cy, SIZE - 2), fill: "rgba(0,0,0,0.3)" }));
      const [rx, ry] = tile.number ? [cx + 19, cy + 2] : [cx, cy];
      board.append(svg("path", { d: `M${rx - 8} ${ry + 13} L${rx + 8} ${ry + 13} L${rx + 5} ${ry - 1} Q${rx} ${ry - 6} ${rx - 5} ${ry - 1} Z`, fill: "#1f2937", stroke: "#e5e7eb", "stroke-width": 1.2 }));
      board.append(svg("circle", { cx: rx, cy: ry - 8, r: 5.5, fill: "#1f2937", stroke: "#e5e7eb", "stroke-width": 1.2 }));
    }
    if (clicks.tiles[key]) {
      const pick = svg("polygon", { points: hexPoints(cx, cy, SIZE - 4), fill: "rgba(255,255,255,0.12)", stroke: "#fff", "stroke-width": 3, "stroke-dasharray": "6 4", class: "tile-pick" });
      pick.onclick = () => chooseRobber(clicks.tiles[key]);
      board.append(pick);
    }
  }
  for (const tile of game.board) {
    if (tile.type !== "port") continue;
    const [cx, cy] = pixel(tile.coordinate);
    for (const node of tile.port_nodes) {
      const [nx, ny] = positions[node];
      board.append(svg("line", { x1: cx + (nx - cx) * 0.35, y1: cy + (ny - cy) * 0.35, x2: nx, y2: ny, stroke: "#8a6a3f", "stroke-width": 5, "stroke-linecap": "round" }));
    }
    board.append(svg("circle", { cx, cy, r: 15, fill: "#fbf6e6", stroke: "#8a6a3f", "stroke-width": 2 }));
    board.append(svg("text", { x: cx, y: cy + (tile.port_resource ? -1 : 4), "text-anchor": "middle", "font-size": 10, "font-weight": 700, fill: "#1f2937" }, tile.port_resource ? "2:1" : "3:1"));
    if (tile.port_resource) board.append(svg("circle", { cx, cy: cy + 7, r: 4, fill: RES_COLORS[tile.port_resource] || "#999" }));
  }

  const drawn = new Set();
  for (const [[a, b], color] of game.state.board.roads) {
    const id = [a, b].sort((x, y) => x - y).join(",");
    if (drawn.has(id)) continue;
    drawn.add(id);
    const [x1, y1, x2, y2] = inset(positions[a], positions[b], 0.14);
    board.append(svg("line", { x1, y1, x2, y2, stroke: "#1d1d1d", "stroke-width": 10, "stroke-linecap": "round" }));
    board.append(svg("line", { x1, y1, x2, y2, stroke: PLAYER_COLORS[color], "stroke-width": 7, "stroke-linecap": "round" }));
  }
  for (const [id, action] of Object.entries(clicks.edges)) {
    const [a, b] = id.split(",").map(Number);
    const [x1, y1, x2, y2] = inset(positions[a], positions[b], 0.18);
    const line = svg("line", { x1, y1, x2, y2, stroke: "#fff", "stroke-width": 8, "stroke-linecap": "round", opacity: 0.9, class: "spot" });
    line.onclick = () => { ui.build = null; session.act(action); };
    board.append(line);
  }
  for (const [node, color, kind] of game.state.board.buildings) {
    const [x, y] = positions[node];
    board.append(svg("polygon", {
      points: shape(x, y, kind === "CITY" ? CITY : SETTLEMENT), fill: PLAYER_COLORS[color],
      stroke: "#1d1d1d", "stroke-width": 1.8, "stroke-linejoin": "round",
    }));
  }
  for (const [node, action] of Object.entries(clicks.nodes)) {
    const [x, y] = positions[node];
    const city = action[1] === "BUILD_CITY";
    const spot = svg("circle", {
      cx: x, cy: y, r: city ? 14 : 7, fill: city ? "rgba(255,255,255,0.35)" : "#fff",
      stroke: city ? "#fff" : "#333", "stroke-width": city ? 3 : 1.5, opacity: 0.95, class: "spot",
    });
    spot.onclick = () => { ui.build = null; session.act(action); };
    board.append(spot);
  }
  $("#prompt").textContent = promptText();
}
// A road's ends, pulled in from the nodes (so roads stop short of buildings).
function inset([x1, y1], [x2, y2], t) {
  return [x1 + (x2 - x1) * t, y1 + (y2 - y1) * t, x2 - (x2 - x1) * t, y2 - (y2 - y1) * t];
}

function chooseRobber(actions) {
  if (actions.length === 1) { session.act(actions[0]); return; }
  openModal({ kind: "robber", actions });
}

const PIP_CELLS = { 1: [4], 2: [0, 8], 3: [0, 4, 8], 4: [0, 2, 6, 8], 5: [0, 2, 4, 6, 8], 6: [0, 2, 3, 5, 6, 8] };
function dieHtml(value, red) {
  const cells = [];
  for (let i = 0; i < 9; i++) cells.push((PIP_CELLS[value] || []).includes(i) ? html`<i></i>` : html`<span></span>`);
  return html`<div class="die${red ? " red" : ""}">${cells}</div>`;
}
function renderDice() {
  const roll = [...session.game.log].reverse().find(([a]) => a[1] === "ROLL");
  if (!roll || !Array.isArray(roll[1])) { $("#dice").innerHTML = ""; return; }
  const [a, b] = roll[1].map(Number);
  setHtml($("#dice"), html`${dieHtml(a, true)}${dieHtml(b)}<span class="dice-total">${a + b}</span>`);
}

// ===== prompts =====
const THEIR_PROMPTS = {
  BUILD_INITIAL_SETTLEMENT: "placing", BUILD_INITIAL_ROAD: "placing",
  PLAY_TURN: "playing", DISCARD: "discarding", MOVE_ROBBER: "moving the robber",
  DECIDE_TRADE: "deciding on a trade", DECIDE_ACCEPTEES: "choosing a trade partner",
};
function myPromptText() {
  const game = session.game;
  const actions = session.pending.body.playable_actions;
  const has = (type) => actions.some((a) => a[1] === type);
  switch (game.state.current_prompt) {
    case "BUILD_INITIAL_SETTLEMENT": return "Place a settlement";
    case "BUILD_INITIAL_ROAD": return "Place a road";
    case "DISCARD": return "Discard: pick a card";
    case "MOVE_ROBBER": return "Move the robber";
    case "PLAY_TURN":
      if (has("ROLL")) return "Roll";
      if (actions.every((a) => a[1] === "BUILD_ROAD")) return "Place a free road";
      return ui.build ? `Pick a spot for the ${ui.build}` : "Your turn";
    default: return prettyPrompt(game.state.current_prompt);
  }
}
function promptText() {
  if (session.aborted) return `Void: ${session.aborted.reason}`;
  const game = session.game;
  if (game.over) {
    if (!game.winner) return "Game over";
    return game.winner === game.color ? "You win" : `${nameOf(game.winner)} wins`;
  }
  if (session.pending && !session.answered) return myPromptText();
  const prompt = THEIR_PROMPTS[game.state.current_prompt] || prettyPrompt(game.state.current_prompt);
  return game.current === game.color ? "Sent" : `${nameOf(game.current)} is ${prompt}…`;
}
const prettyPrompt = (prompt) => String(prompt || "").toLowerCase().replace(/_/g, " ");
function renderTimer() {
  const bar = $("#timer"), fill = $("#timer-fill"), prompt = $("#prompt");
  const live = session && session.game && session.pending && !session.answered && !session.game.over;
  bar.className = live ? "on" : "";
  prompt.className = live ? "mine" : "";
  if (!live) return;
  const total = session.pending.body.timeout_ms || 1;
  const left = Math.max(0, total - (Date.now() - session.pending.received));
  fill.style.width = `${(100 * left) / total}%`;
  fill.className = left < 15000 ? "low" : "";
}

// ===== the hand and the action bar =====
function seatIndex(color) { return session.game.state.colors.indexOf(color); }
function ps(color, key) { return session.game.state.player_state[`P${seatIndex(color)}_${key}`]; }
// Points as far as this page can see them: a seat's own (or anyone's once
// the game is over, if the server shows them), else the public count.
function vpOf(color) {
  const game = session.game;
  const actual = ps(color, "ACTUAL_VICTORY_POINTS");
  return (color === game.color || game.over) && actual !== undefined && actual !== null ? actual : ps(color, "VICTORY_POINTS");
}
// The same, as shown: a winner whose victory point cards stay hidden has at
// least the points to win.
function vpText(color) {
  const game = session.game;
  const vp = vpOf(color), target = Number(game.config && game.config.vps_to_win);
  return game.over && color === game.winner && target && vp < target ? `${target}+` : String(vp);
}
function pendingActions() { return session.pending && !session.answered ? session.pending.body.playable_actions : []; }

function renderMe() {
  const game = session.game;
  const box = $("#me");
  if (!game.color) {
    setHtml(box, html`<div class="label">Watching</div>`);
    return;
  }
  const actions = pendingActions();
  const discard = {};
  for (const a of actions) if (a[1] === "DISCARD_RESOURCE") discard[a[2]] = a;
  const cards = RESOURCES.map((r) => {
    const n = ps(game.color, r + "_IN_HAND");
    const cls = `card res-${r}${n ? "" : " empty"}${discard[r] ? " pick" : ""}`;
    return html`<div class="${cls}" data-res="${r}">${RES_LABEL[r]}<span class="count">${n}</span></div>`;
  });
  const devs = DEV_CARDS.map((c) => [c, ps(game.color, c + "_IN_HAND")]).filter(([, n]) => n > 0).map(([c, n]) => {
    const playable = DEV_PLAY[c] && actions.some((a) => a[1] === DEV_PLAY[c]);
    return html`<div class="card dev${playable ? " pick" : ""}" data-dev="${c}">${DEV_NAME[c]}<span class="count">${n}</span></div>`;
  });
  setHtml(box, html`<div class="label">${vpOf(game.color)} points</div>${cards}<span class="cards-gap"></span>${devs}`);
  for (const element of box.querySelectorAll(".card.pick")) {
    element.classList.add("clickable");
    const resource = element.getAttribute("data-res"), dev = element.getAttribute("data-dev");
    element.onclick = () => (resource ? session.act(discard[resource]) : playDev(dev));
  }
}
function playDev(card) {
  const options = pendingActions().filter((a) => a[1] === DEV_PLAY[card]);
  if (!options.length) return;
  if (card === "YEAR_OF_PLENTY") openModal({ kind: "yop" });
  else if (card === "MONOPOLY") openModal({ kind: "monopoly" });
  else session.act(options[0]);
}
function act(type) {
  const action = pendingActions().find((a) => a[1] === type);
  if (action) { ui.build = null; session.act(action); }
}

const ACTION_TYPES = new Set([
  "ROLL", "END_TURN", "BUY_DEVELOPMENT_CARD", "MARITIME_TRADE", "BUILD_ROAD", "BUILD_SETTLEMENT", "BUILD_CITY",
  "MOVE_ROBBER", "DISCARD_RESOURCE", ...Object.values(DEV_PLAY),
]);
function actionLabel(action) {
  const [, type, value] = action;
  switch (type) {
    case "ROLL": return "Roll";
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
  const game = session.game;
  if (!game.color || game.over) return;
  const actions = pendingActions();
  const has = (type) => actions.some((a) => a[1] === type);
  const boardOnly = actions.length > 0 && actions.every((a) => ON_BOARD.has(a[1]));
  if (ui.build && !has(BUILD_TYPES[ui.build])) ui.build = null;
  const add = (label, cost, enabled, onclick, extra) => {
    const element = button("", onclick, "act" + (extra ? " " + extra : ""));
    const costs = cost ? html`<span class="cost">${cost.map(dot)}</span>` : html``;
    setHtml(element, html`<span>${label}</span>${costs}`);
    element.disabled = !enabled;
    box.append(element);
  };
  add("Trade", null, has("MARITIME_TRADE"), () => openModal({ kind: "trade" }));
  add("Dev card", COSTS.dev, has("BUY_DEVELOPMENT_CARD"), () => act("BUY_DEVELOPMENT_CARD"));
  for (const [mode, label] of [["road", "Road"], ["settlement", "Settlement"], ["city", "City"]]) {
    add(label, COSTS[mode], has(BUILD_TYPES[mode]) && !boardOnly,
      () => { ui.build = ui.build === mode ? null : mode; render(); }, ui.build === mode ? "on" : "");
  }
  if (has("ROLL")) add("Roll", null, true, () => act("ROLL"), "go");
  else add("End turn", null, has("END_TURN"), () => act("END_TURN"), "go");
  for (const action of actions.filter((a) => !ACTION_TYPES.has(a[1]))) {
    box.append(button(actionLabel(action), () => session.act(action), "primary"));
  }
}

// ===== the side panels =====
function renderPlayers() {
  const game = session.game;
  const box = $("#players");
  const state = game.state;
  setHtml(box, html`<h3>Players</h3>`);
  for (const color of state.colors) {
    const mine = color === game.color;
    const resources = mine ? RESOURCES.reduce((n, r) => n + ps(color, r + "_IN_HAND"), 0) : ps(color, "NUM_RESOURCES_IN_HAND");
    const devs = mine ? DEV_CARDS.reduce((n, c) => n + ps(color, c + "_IN_HAND"), 0) : ps(color, "NUM_DEVELOPMENT_CARDS_IN_HAND");
    const seat = seatInfo(color);
    const badges = [];
    if (ps(color, "HAS_ROAD")) badges.push(html`<span class="badge">Longest road</span>`);
    if (ps(color, "HAS_ARMY")) badges.push(html`<span class="badge">Largest army</span>`);
    const row = document.createElement("div");
    row.className = "player" + (color === game.current && !game.over ? " current" : "");
    setHtml(row, html`${avatarHtml(seat, color)}
      <div class="who"><span>${seatName(color)}</span>${mine ? html`<span class="tag">you</span>` : botTag(seat)}</div>
      <div class="vp">${vpText(color)}<small>pts</small></div>
      <div class="stats"><span>${resources} cards</span><span>${devs} dev</span><span>${ps(color, "PLAYED_KNIGHT")} knights</span><span>road ${ps(color, "LONGEST_ROAD_LENGTH")}</span></div>
      ${badges.length ? html`<div class="badges">${badges}</div>` : html``}`);
    box.append(row);
  }
}
function renderBank() {
  const state = session.game.state;
  const bank = state.resource_freqdeck.map((n, i) => html`<span>${dot(RESOURCES[i])} ${n}</span>`);
  const deck = state.development_deck_size !== undefined ? state.development_deck_size : "?";
  setHtml($("#bank"), html`<h3>Bank</h3><div class="row">${bank}<span class="muted">${deck} dev</span></div>`);
}

function who(color) { return html`<span class="who" style="color:${inkOf(color)}">${nameOf(color)}</span>`; }
function logHtml([action, result]) {
  const [color, type, value] = action;
  const by = who(color);
  switch (type) {
    case "ROLL": {
      const sum = Number(result[0]) + Number(result[1]);
      return html`${by} rolled <b>${sum}</b> <span class="muted">(${result[0]}+${result[1]})</span>`;
    }
    case "BUILD_SETTLEMENT": return html`${by} built a settlement`;
    case "BUILD_CITY": return html`${by} built a city`;
    case "BUILD_ROAD": return html`${by} built a road`;
    case "BUY_DEVELOPMENT_CARD": return html`${by} bought a development card${value ? html` <span class="muted">(${DEV_NAME[value] || value})</span>` : html``}`;
    case "PLAY_KNIGHT_CARD": return html`${by} played a knight`;
    case "PLAY_ROAD_BUILDING": return html`${by} played road building`;
    case "PLAY_YEAR_OF_PLENTY": return html`${by} took ${res(value[0])} and ${res(value[1])}`;
    case "PLAY_MONOPOLY": return html`${by} monopolized ${res(value)}`;
    case "MOVE_ROBBER": {
      const stole = value[1] ? html`, stealing ${result ? res(result) : "a card"} from ${who(value[1])}` : html``;
      return html`${by} moved the robber${stole}`;
    }
    case "DISCARD_RESOURCE": return html`${by} discarded ${value ? res(value) : "a card"}`;
    case "MARITIME_TRADE": {
      const give = value.slice(0, 4).filter((r) => r);
      return html`${by} traded ${give.length} ${res(give[0])} for 1 ${res(value[4])}`;
    }
    case "END_TURN": return null;
    default: return html`${by}: ${actionLabel(action).toLowerCase()}`;
  }
}
function renderLog() {
  const list = $("#log");
  const stick = list.scrollTop + list.clientHeight >= list.scrollHeight - 40;
  list.innerHTML = "";
  const log = session.game.log;
  for (let i = Math.max(0, log.length - 150); i < log.length; i++) {
    let line = null;
    try { line = logHtml(log[i]); } catch (error) { line = html`${String(log[i][0][0])}: ${String(log[i][0][1]).toLowerCase()}`; }
    if (!line) continue;
    const item = document.createElement("li");
    setHtml(item, line);
    list.append(item);
  }
  if (stick) list.scrollTop = list.scrollHeight;
}

function renderFairness() {
  const box = $("#fairness");
  const s = session;
  const short = (value) => String(value || "").slice(0, 16);
  const ok = !s.unverified && !s.problems.length && !cryptoProblem;
  setHtml($("#fairness-summary"), ok ? html`Verified · ${s.verifiedCount} signatures` : html`<span class="error">Verification: see details</span>`);
  const lines = [];
  if (s.serverCommitment) lines.push(html`<div><span>Seed commitment</span><code>${short(s.serverCommitment)}…</code></div>`);
  if (s.serverKey) lines.push(html`<div><span>Server key</span><code>${short(s.serverKey)}…</code></div>`);
  lines.push(html`<div><span>Messages</span><span>${s.messages.length}, ${s.verifiedCount} signatures checked</span></div>`);
  if (serverInfo.app_sha256) lines.push(html`<div><span>app.js</span><code>${short(serverInfo.app_sha256)}…</code></div>`);
  if (s.unverified) lines.push(html`<div class="error">${s.unverified} messages could not be checked in this browser.</div>`);
  if (cryptoProblem) lines.push(html`<div class="error">${cryptoProblem}</div>`);
  for (const problem of s.problems.slice(-3)) lines.push(html`<div class="error">${problem}</div>`);
  setHtml(box, html`${lines}`);
  if (s.aborted || (s.game && s.game.over)) {
    box.append(button("Download transcript", () => download((s.saveId || "transcript") + ".json", JSON.stringify(s.transcript()))));
  }
}

// ===== dialogs =====
// A dialog for a decision closes by itself once that decision is gone.
function openModal(modal) {
  ui.modal = { ...modal, seq: modal.kind === "over" || modal.kind === "void" ? null : session.pending && session.pending.env.seq };
  render();
}
function closeModal() {
  ui.modal = null;
  const box = $("#modal");
  if (box) { box.hidden = true; box.innerHTML = ""; }
}
function renderModal() {
  const s = session;
  const box = $("#modal");
  if (!ui.seenEnd && ((s.game && s.game.over) || s.aborted)) {
    ui.seenEnd = true;
    ui.modal = { kind: s.aborted ? "void" : "over", seq: null };
  }
  const m = ui.modal;
  if (m && m.seq !== null && (!s.pending || s.answered || s.pending.env.seq !== m.seq)) ui.modal = null;
  if (!ui.modal) { box.hidden = true; box.innerHTML = ""; return; }
  box.hidden = false;
  const inner = document.createElement("div");
  inner.className = "box";
  box.replaceChildren(inner);
  box.onclick = (event) => { if (event.target === box) closeModal(); };
  const choices = document.createElement("div");
  choices.className = "options";
  const choose = (label, action) => {
    const element = button("", () => { closeModal(); session.act(action); });
    setHtml(element, label);
    choices.append(element);
  };
  const foot = document.createElement("div");
  foot.className = "foot";
  const pending = pendingActions();
  switch (ui.modal.kind) {
    case "robber":
      setHtml(inner, html`<h2>Steal from</h2>`);
      for (const action of ui.modal.actions) {
        const victim = action[2][1];
        choose(victim ? html`${avatarHtml(seatInfo(victim), victim)} ${nameOf(victim)} <span class="muted">${ps(victim, "NUM_RESOURCES_IN_HAND")} cards</span>`
          : html`Nobody`, action);
      }
      inner.append(choices);
      foot.append(button("Cancel", closeModal));
      break;
    case "yop":
      setHtml(inner, html`<h2>Year of Plenty</h2>`);
      for (const action of pending.filter((a) => a[1] === "PLAY_YEAR_OF_PLENTY")) {
        choose(html`${dot(action[2][0])}${dot(action[2][1])} ${RES_LABEL[action[2][0]]} + ${RES_LABEL[action[2][1]]}`, action);
      }
      inner.append(choices);
      foot.append(button("Cancel", closeModal));
      break;
    case "monopoly":
      setHtml(inner, html`<h2>Monopoly</h2>`);
      for (const action of pending.filter((a) => a[1] === "PLAY_MONOPOLY")) {
        choose(html`${dot(action[2])} ${RES_LABEL[action[2]]}`, action);
      }
      inner.append(choices);
      foot.append(button("Cancel", closeModal));
      break;
    case "trade": {
      setHtml(inner, html`<h2>Trade with the bank</h2>`);
      const byGive = new Map();
      for (const action of pending.filter((a) => a[1] === "MARITIME_TRADE")) {
        const give = action[2].slice(0, 4).filter((r) => r);
        const key = `${give.length} ${give[0]}`;
        if (!byGive.has(key)) byGive.set(key, { give, actions: [] });
        byGive.get(key).actions.push(action);
      }
      for (const { give, actions } of byGive.values()) {
        const row = document.createElement("div");
        row.className = "trade-row";
        setHtml(row, html`<span class="give">${give.length} ${res(give[0])} for</span>`);
        for (const action of actions) {
          const element = button("", () => { closeModal(); session.act(action); });
          setHtml(element, html`${dot(action[2][4])} ${RES_LABEL[action[2][4]]}`);
          row.append(element);
        }
        inner.append(row);
      }
      foot.append(button("Cancel", closeModal));
      break;
    }
    case "over":
    case "void": {
      const game = s.game;
      if (ui.modal.kind === "void") {
        setHtml(inner, html`<h2>Game void</h2><p class="muted">${s.aborted ? s.aborted.reason : ""}</p>`);
      } else {
        setHtml(inner, html`<p class="result">${promptText()}</p>`);
        const table = document.createElement("table");
        table.className = "final";
        const rows = [...game.state.colors].sort((a, b) => vpOf(b) - vpOf(a));
        setHtml(table, html`${rows.map((color) => html`<tr><td>${avatarHtml(seatInfo(color), color)}</td><td>${seatName(color)}${botTag(seatInfo(color))}</td><td style="text-align:right"><b>${vpText(color)}</b> pts</td></tr>`)}`);
        inner.append(table);
      }
      const name = s.room && s.room.name;
      if (serverInfo.house && name) foot.append(button("Play again", () => { ui.retries = 0; playTable(name); }, "primary"));
      foot.append(button("Board", closeModal));
      foot.append(button("Lobby", () => { location.hash = ""; }));
      break;
    }
    default:
      ui.modal = null;
  }
  inner.append(foot);
}

// ===== routing =====
function route() {
  const params = new URLSearchParams(location.hash.replace(/^#/, ""));
  const roomId = params.get("room");
  if (roomId && /^[A-Za-z0-9_-]+$/.test(roomId)) {
    ui.autoSit = params.get("sit") === "1" && !cryptoProblem;
    ui.table = params.get("table") || "";
    openRoom(roomId);
  } else {
    showLobby();
  }
}
window.addEventListener("hashchange", route);
let serverInfo = {};
window.addEventListener("load", async () => {
  await checkCrypto();
  try { serverInfo = await (await fetch(apiUrl("/api/server"))).json(); } catch (error) { serverInfo = {}; }
  $("#crypto-status").textContent = cryptoProblem ? "watch only" : "";
  $("#create").addEventListener("submit", createRoom);
  $("#create [name=num_players]").addEventListener("change", renderSeatSpecs);
  $("#refresh").onclick = refreshRooms;
  $("#show-finished").onchange = refreshRooms;
  $("#lobby-name").onchange = saveName;
  $("#home").onclick = (event) => { event.preventDefault(); location.hash = ""; };
  document.addEventListener("keydown", (event) => { if (event.key === "Escape" && ui.modal) closeModal(); });
  setInterval(() => {
    if (session && session.game && session.pending && !session.answered) {
      $("#prompt").textContent = promptText();
      renderTimer();
    }
  }, 1000);
  // The lobby keeps itself current (tables open and close as games end).
  setInterval(() => { if (!session && !$("#lobby").hidden) refreshRooms().catch(() => {}); }, 5000);
  route();
});
