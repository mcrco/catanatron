// Browser client for catanatron-server. No build step, no dependencies.
//
// It speaks the same WebSocket protocol as catanatron.server.client: checks
// the server's Ed25519 signature and the hash chain on every message,
// contributes a random seed when it takes a seat, signs every action with
// a key generated in this browser (WebCrypto Ed25519), and keeps the
// transcript, which it stores in IndexedDB for `catanatron-verify`.
"use strict";

const PROTOCOL_VERSION = 3;
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
function canonical(value) {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return "[" + value.map(canonical).join(",") + "]";
  return "{" + Object.keys(value).sort()
    .map((k) => JSON.stringify(k) + ":" + canonical(value[k])).join(",") + "}";
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
async function saveTranscript(id, transcript) {
  try {
    const db = await openDb();
    db.transaction("transcripts", "readwrite").objectStore("transcripts")
      .put({ id, created: Date.now(), json: JSON.stringify(transcript) });
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

class Session {
  constructor(roomId) {
    this.roomId = roomId;
    this.storageKey = "catanatron-seat-" + roomId;
    this.saved = JSON.parse(localStorage.getItem(this.storageKey) || "null");
    this.identity = null;
    this.spectating = false;
    this.joining = false;
    this.resetTranscript();
    this.room = null;
    this.commitments = null;
    this.game = null; // the table: before's static parts, state, log
    this.pending = null; // { env, hash, body }
    this.answered = false;
    this.answer = null; // the signed action for the pending decide
    this.unreceipted = new Map(); // submission digest -> signed action not yet acknowledged
    this.commitRequest = null; // a fresh commitment after a restart, until accepted
    this.revealRequest = null;
    this.C = null; // the digest of this game's commitments, which every move signs
    this.closed = false;
    this.verifiedCount = 0;
    this.unverified = 0;
    this.problems = [];
    this.saveId = null;
    this.aborted = null;
  }
  resetTranscript() {
    this.messages = []; // raw text, exactly as received
    this.sent = [];
    this.head = GENESIS;
    this.serverKey = null;
  }
  get seated() { return Boolean(this.saved && this.saved.seatKey); }

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
      if (this.closed || this.aborted || (this.game && this.game.over)) return;
      setTimeout(() => { if (!this.closed) this.connect(); }, 1500);
    };
  }
  close() { this.closed = true; if (this.ws) this.ws.close(); }
  send(payload) {
    if (!this.ws || this.ws.readyState !== 1) return false;
    this.ws.send(JSON.stringify(payload));
    return true;
  }

  async receive(text) {
    const env = JSON.parse(text);
    if (this.first) {
      this.first = false;
      const body = env.body;
      if (this.seated) {
        // Taking back our seat: this connection's hello opens a throw-away
        // chain; the seat's own chain resumes after our last message.
        if (!this.identity) this.identity = await loadIdentity(this.saved.identity);
        if (!this.serverKey) this.serverKey = checkedKey(body.server_key);
        if ((await verifyEnvelope(this.serverKey, env)) === false) throw new Error("bad signature on hello");
        this.resyncing = true;
        this.send({
          type: "join", protocol_version: PROTOCOL_VERSION, seat: this.saved.seat,
          seat_key: this.saved.seatKey, public_key: this.identity.publicHex,
          last_seq: this.messages.length - 1,
        });
        this.resume();
        return;
      }
      if (this.spectating) {
        this.resetTranscript();
        this.game = null;
        this.commitments = null;
        await this.accept(text, env);
        this.send({ type: "spectate" });
        return;
      }
      if (this.messages.length) {
        // Reconnected before taking a seat: this connection is a new chain.
        this.resetTranscript();
        this.joining = false;
      }
      await this.accept(text, env);
      if (body.protocol_version !== PROTOCOL_VERSION) throw new Error(`the server speaks protocol ${body.protocol_version}`);
      this.room = body.room;
      this.serverCommitment = body.room.seed_commitment;
      if (body.room.status !== "waiting") this.spectate(); // too late to sit
      render();
      return;
    }
    if (this.resyncing) {
      if (!(env.seq === this.messages.length && env.prev === this.head)) {
        // Still the throw-away chain: check it, but keep it out of ours.
        if ((await verifyEnvelope(this.serverKey, env)) === false) throw new Error("bad signature while rejoining");
        if (env.body.type === "error") this.problems.push(env.body.message);
        return;
      }
      this.resyncing = false;
    }
    await this.accept(text, env);
    await this.handle(env);
  }

  async accept(text, env) {
    if (!this.serverKey) this.serverKey = checkedKey(env.body.server_key);
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
    switch (body.type) {
      case "joined": {
        this.joining = false;
        const pending = this.saved || {};
        const mine = this.identity && body.public_key === this.identity.publicHex &&
          body.client_commitment === pending.clientCommitment;
        if (!mine) {
          this.problems.push("the server seated another key or seed commitment than this page's");
          break;
        }
        this.joinedAt = env.seq;
        this.saved = { ...pending, seat: body.seat, seatKey: body.seat_key, color: body.color };
        localStorage.setItem(this.storageKey, JSON.stringify(this.saved));
        this.room = body.room;
        break;
      }
      case "hello": // the seat chain's first message, resent after a reload
        this.room = body.room;
        this.serverCommitment = body.room.seed_commitment;
        break;
      case "room":
        this.room = body.room;
        break;
      case "restart":
        await this.restart(body);
        break;
      case "commitments":
        this.commitments = body;
        this.commitRequest = null;
        this.C = await sha256Hex(canonical(body));
        if (this.seated && !this.game) {
          const mine = body.seats.find((s) => s.public_key === this.saved.identity.publicHex);
          if (!mine || mine.seat !== this.saved.seat || mine.client_commitment !== this.saved.clientCommitment) {
            this.problems.push("the commitments do not seat this page as it joined");
            break;
          }
          if (!this.identity) this.identity = await loadIdentity(this.saved.identity);
          const sig = await signHex(this.identity.privateKey, { domain: "catanatron-reveal/1", commitments: this.C });
          this.revealRequest = { type: "reveal", client_seed: this.saved.clientSeed, sig };
          this.send(this.revealRequest);
        }
        break;
      case "received":
        this.unreceipted.delete(body.submission);
        if (!body.accepted && this.pending && body.index === this.pending.body.index) {
          this.problems.push("the server did not take your move: " + body.reason);
        }
        break;
      case "error":
        if (body.code !== "not_revealing" && body.code !== "not_committing") this.problems.push(body.message);
        // Only an answer to the join itself ends the wait for it.
        if (this.joining && JOIN_REFUSALS.has(body.code)) this.joining = false;
        break;
      case "aborted":
        this.aborted = body;
        this.finish();
        break;
      case "before": case "step": case "decide": case "after":
        if (body.type === "before" && this.commitments &&
            (canonical(body.config) !== canonical(this.commitments.config) ||
             canonical(body.seats) !== canonical(this.commitments.seats))) {
          this.problems.push("the game is not the one whose seats were committed");
        }
        this.update(body);
        this.pending = null;
        this.answered = false;
        this.answer = null;
        if (body.type === "decide") this.pending = { env, hash: env.hash, body, received: Date.now() };
        if (body.type === "after") this.finish();
        break;
    }
    render();
  }

  resume() {
    // After a rejoin: finish what the drop interrupted. Resending is harmless
    // (the server's receipt says "stale" for what it already had), and every
    // answer it never acknowledged is resent, so each gets a receipt.
    if (this.commitRequest) this.send(this.commitRequest);
    if (this.revealRequest && !this.game) this.send(this.revealRequest);
    for (const answer of this.unreceipted.values()) this.send(answer);
  }

  async restart(body) {
    // A seat withheld its seed: the server must reveal the seed it had
    // committed to; then we commit afresh to a new seed of our own.
    if (body.room_id !== this.roomId || !HEX64.test(body.server_seed || "") ||
        (subtle && (await sha256OfHex(body.server_seed)) !== this.serverCommitment)) {
      throw new Error("the restart does not open the server's seed commitment");
    }
    this.serverCommitment = body.seed_commitment;
    this.room = body.room;
    this.commitments = null;
    this.revealRequest = null;
    if (!this.seated) return;
    if ((body.blame || []).includes(this.saved.seat)) {
      this.problems.push("the server says this page did not reveal its seed in time");
      return;
    }
    if ((this.saved.round || 0) >= body.round) {
      // A restart already handled (the chain is being replayed after a
      // reload): resend that round's commitment, harmless if it arrived.
      this.commitRequest = { type: "commit", client_commitment: this.saved.clientCommitment, pop: this.saved.pop };
      this.send(this.commitRequest);
      return;
    }
    if (!this.identity) this.identity = await loadIdentity(this.saved.identity);
    const clientSeed = randomHex(32);
    const clientCommitment = await clientSeedCommitment(this.roomId, this.identity.publicHex, clientSeed);
    const pop = await signHex(this.identity.privateKey, {
      domain: "catanatron-join/1", room_id: this.roomId, seed_commitment: this.serverCommitment,
      seat: this.saved.seat, client_commitment: clientCommitment,
    });
    this.saved = { ...this.saved, clientSeed, clientCommitment, pop, round: body.round };
    localStorage.setItem(this.storageKey, JSON.stringify(this.saved));
    this.commitRequest = { type: "commit", client_commitment: clientCommitment, pop };
    this.send(this.commitRequest);
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
    if (this.joining || this.seated) return; // one join at a time
    this.joining = true;
    render();
    this.identity = await newIdentity();
    const clientSeed = randomHex(32);
    const clientCommitment = await clientSeedCommitment(this.roomId, this.identity.publicHex, clientSeed);
    const statement = {
      domain: "catanatron-join/1", room_id: this.roomId, seed_commitment: this.serverCommitment,
      seat: index, client_commitment: clientCommitment,
    };
    const pop = await signHex(this.identity.privateKey, statement);
    this.saved = {
      identity: { publicHex: this.identity.publicHex, privatePkcs8: this.identity.privatePkcs8 },
      clientSeed, clientCommitment, pop, round: this.room.round || 0,
    };
    localStorage.setItem(this.storageKey, JSON.stringify(this.saved));
    this.send({
      type: "join", protocol_version: PROTOCOL_VERSION, name: $("#my-name") ? $("#my-name").value : "",
      seat: index, public_key: this.identity.publicHex, client_commitment: clientCommitment, pop,
    });
  }
  spectate() {
    this.spectating = true;
    this.send({ type: "spectate" });
  }
  seatBot(index, spec) { this.send({ type: "seat_bot", seat: index, spec }); }

  async act(action) {
    if (!this.pending || this.answered) return;
    this.answered = true;
    const pending = this.pending;
    const statement = {
      domain: "catanatron-action/1", commitments: this.C, index: pending.body.index,
      seq: pending.env.seq, decide: pending.hash, action,
    };
    const sig = await signHex(this.identity.privateKey, statement);
    this.sent.push({ statement, sig, sent_after: this.messages.length - 1 });
    this.answer = { type: "action", statement, sig };
    this.unreceipted.set(await sha256Hex(canonical({ statement, sig })), this.answer);
    this.send(this.answer); // resent after a reconnect until acknowledged
    render();
  }

  transcript() {
    return {
      format: "catanatron-transcript/3",
      url: wsUrl("/room/" + this.roomId),
      room_id: this.roomId,
      server_key: this.serverKey,
      public_key: this.seated ? this.saved.identity.publicHex : null,
      client_seed: this.seated ? this.saved.clientSeed : null,
      seat: this.seated ? this.saved.seat : null,
      color: this.game ? this.game.color : null,
      messages: this.messages,
      sent: this.sent,
    };
  }
  finish() {
    const who = this.game ? this.game.color || "spectator" : "seat";
    this.saveId = `${this.roomId}-${this.game ? this.game.gameId : "aborted"}-${who}`;
    saveTranscript(this.saveId, this.transcript());
  }
}

async function openRoom(roomId) {
  $("#lobby").hidden = true;
  $("#room").hidden = false;
  if (session) session.close();
  session = new Session(roomId);
  if (session.saved && !session.saved.seatKey) {
    // We asked for a seat but never heard back: start over.
    localStorage.removeItem(session.storageKey);
    session.saved = null;
  }
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
  const canSit = !session.seated && !session.spectating && !session.joining && room.status === "waiting";
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
      row.append(select, button("Add bot", () => session.seatBot(seat.seat, select.value)));
    }
    box.append(row);
  }
  if (session.joining) addHtml(box, html`<p>Taking the seat...</p>`);
  if (session.seated) addHtml(box, html`<p>You are seat ${session.saved.seat}. The game starts when every seat is taken.</p>`);
  else if (!session.spectating) box.append(button("Just watch", () => session.spectate()));
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
  if (session.aborted) return `The game was called off: ${session.aborted.reason}` +
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
  if (s.room) lines.push(html`<div class="muted">Seed commitment: <code>${short(s.room.seed_commitment)}…</code></div>`);
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
