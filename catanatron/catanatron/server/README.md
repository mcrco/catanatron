# catanatron-server

A networked game server on stock Catanatron: rooms of 2–6 seats taken by people
(in a browser), remote bots (any language that speaks JSON over WebSocket) and
server-side registry bots, plus spectators. Every game is **verifiable**: each
client keeps a signed transcript and can replay the game afterwards to check
that the server dealt fairly and showed it nothing but the truth.

```bash
pip install -e "catanatron[server]"      # websockets + cryptography
```

## Running

```bash
catanatron-server --port 8765                  # or: python -m catanatron.server
```

Open <http://localhost:8765/> for the browser client: create a room (players,
map, points to win, discard limit, seconds per decision, optionally an explicit
map), pick a server bot or "open" for each seat, then sit down or watch.

Options: `--host`, `--key FILE` (the server's Ed25519 identity, created on first
use, default `~/.catanatron/server_key`), `--archive DIR` (keep each finished
game's reveal, commitments and final state), `--bot NAME=SOURCE` (register your
own bot classes for seats, as in `catanatron-play`).

A bot from the registry, as a remote client (it sees only its seat's view):

```bash
catanatron-play-bot --spec AB:2 ws://localhost:8765/room/ROOM_ID
# create a 4-seat room, the other seats played by the server's F bots, and join it
catanatron-play-bot --spec F --new --players 4 --fill F ws://localhost:8765
```

It saves its transcript under `./transcripts/` and verifies it when the game ends.
Anyone can verify a transcript later (several transcripts of one game are also
checked against one another):

```bash
catanatron-verify transcripts/ROOM-GAME-RED.json
```

Your own client in Python:

```python
import asyncio
from catanatron.server.client import GameClient, create_room

async def main():
    room = await create_room("ws://localhost:8765", {"num_players": 3}, [None, "F", "F"])
    client = GameClient(f"ws://localhost:8765/room/{room['id']}", name="me")
    result = await client.play(lambda view, playable_actions: playable_actions[0])
    print(result.summary())   # winner, transcript path, verified

asyncio.run(main())
```

`view` is the seat's `client_view` document (see below) and `playable_actions`
are `[color, action_type, value]` triples; return one of them (or an `Action`).
Synchronous callbacks run in a thread; `async` ones are awaited. To run a
Catanatron registry bot on views, use `catanatron.server.bot.ViewBot`, which
decides on a full game sampled from the view (`catanatron.server.sampling`):
other hands are dealt at random consistently with what the seat knows, and the
deck order and random stream are fresh, so the bot's lookahead cannot see
hidden cards or future rolls.

## Rooms and games

A room's config (`catanatron.server.config.GameConfig`, all optional):

| key | default | |
|---|---|---|
| `num_players` | 4 | 2–6; seat *i* plays RED, BLUE, ORANGE, WHITE, GREEN, BROWN in that order. Turn order is drawn at random. |
| `map_type` | `BASE` | `BASE`, `MINI` or `TOURNAMENT` |
| `map` | null | an explicit map, see below |
| `number_placement` | `official_spiral` | or `random` |
| `vps_to_win` | 10 | |
| `discard_limit` | 7 | |
| `friendly_robber` | false | |
| `supply` | `base` (`extension` for 5–6) | bank, development deck and pieces |
| `decision_timeout_ms` | 60000 | after which the server plays the default action for the seat |
| `autoplay_forced` | true | play a seat's only legal action without asking |

An explicit map lists every tile, in the order that numbers the nodes (the order
`CatanMap.from_tiles` walks):

```json
{"tiles": [
  {"coordinate": [0, 0, 0], "type": "land", "resource": "WHEAT", "number": 6},
  {"coordinate": [1, -1, 0], "type": "land", "resource": null},
  {"coordinate": [3, -3, 0], "type": "port", "direction": "WEST", "port_resource": "SHEEP"},
  {"coordinate": [2, -3, 1], "type": "water"}
]}
```

`resource: null` is the desert (a map needs one: the robber starts there), a
port's `direction` is the way it faces (towards land) and `port_resource: null`
is a 3:1 port. `catanatron.server.maps.map_spec(catan_map)` produces this for
any map. Any shape works (hexagons, the 5–6 player board, rectangles,
islands): the board's node graph is built from the land tiles, so land that
only meets across water is not adjacent (no road, no distance rule there).

When every seat is taken (a remote client, or a bot via the room's `seats` or
a `seat_bot` request) the game starts. A seat that does not answer within the
timeout is played `default_action` (roll, else end the turn, else the first
legal action in canonical order) and the move is flagged `"forced": "timeout"`;
an only legal action is played for the seat and flagged `"only"`.

Server bots are seated exactly like remote ones: they get the same redacted
messages and decide on sampled games, never on the real one.

## Protocol

HTTP and WebSocket share one port:

| | |
|---|---|
| `GET /`, `/static/…` | the browser client |
| `GET /api/rooms`, `/api/rooms/ID` | rooms |
| `GET /api/players` | the registry's bots |
| `GET /api/server` | `{"server_key", "protocol_version"}` |
| `WS /lobby` | `{"type": "create_room", "config", "seats", "name"}` → `{"type": "room_created", "room"}`; `{"type": "list_rooms"}` → `{"type": "rooms", "rooms"}`; `{"type": "players"}` |
| `WS /room/ID` | a seat or a spectator (below) |

Every message from the server on a room connection is a signed envelope:

```json
{"to": "<recipient's public key, or null>", "seq": 7, "prev": "<hash of message 6>",
 "commit": {"index": 12, "hash": "<state commitment>"} ,
 "body": {"type": "step", ...}, "sig": "<Ed25519 signature>"}
```

`seq` counts the messages to this recipient, `prev` is `sha256(canonical(head))`
of the previous one (64 zeros for the first), `commit` binds the full game
state after `index` actions (null before the game), and `sig` signs
`canonical({to, seq, prev, commit, body})`. `canonical` is JSON with sorted keys,
no whitespace, UTF-8 (`json.dumps(sort_keys=True, separators=(",", ":"),
ensure_ascii=False)`); signed bodies contain only strings, integers, booleans and
null. All hashes, keys, seeds and signatures are lowercase hex.

The vocabulary extends `catanatron.protocol`'s (hello / before / decide / step /
after):

| body `type` | when | fields |
|---|---|---|
| `hello` | first message of every connection | `protocol_version`, `server_key`, `room` (config, seats, status, `seed_commitment`) |
| `joined` | you took a seat | `seat`, `color`, `public_key`, `client_seed`, `seat_key` (keep it to reconnect) |
| `room` | seats changed before the start | `room` |
| `before` | the game starts (or you arrive as a spectator) | `game_id`, `color` (null for spectators), `config`, `seats` (with public keys), `client_seeds`, `seed_commitment`, `map`, `board`, `index`, `state`, `log_start`, `log`, `current_color`, `winning_color` |
| `decide` | your move | `index`, `state`, `playable_actions` (canonical order), `timeout_ms`, … |
| `step` | someone acted | `index`, `state`, `log_start`, `log` (that action's entry), `forced` (null, `"only"` or `"timeout"`), … |
| `after` | the game is over | `index`, `state`, `num_turns`, `winning_color`, `reveal`: `server_seed`, `salts`, `actions` |
| `error` | a request was refused | `code`, `message`, `seq` (the decide an action was for) |

`state` is your view of the game: Catanatron's `client_view(doc, your_color)`
(for spectators `public_view(doc)`) without `map` (sent once, in `before`) and
without `action_records`, which arrive incrementally as `log` (entries
`log_start…index-1`, already redacted for you: your own development-card draws
and the steals you were party to are shown, others' are not; dice, discards,
trades and everything else are public). `board` is the drawable geometry: per
tile its coordinate, type, resource/number or port, and node ids by corner
(pointy-top hexes, `NORTH … NORTHWEST`) and edges by side.
`catanatron.server.table.Table` reassembles messages into the full view.

Client → server on `/room/ID`:

```json
{"type": "join", "protocol_version": 1, "name": "me", "seat": 0,
 "public_key": "<Ed25519 public key>", "client_seed": "<32 random bytes>"}
{"type": "join", "protocol_version": 1, "seat": 0, "seat_key": "...",
 "public_key": "...", "last_seq": 41}                       // reconnect: resend after 41
{"type": "spectate"}
{"type": "seat_bot", "seat": 2, "spec": "AB:2"}              // before the start
{"type": "action", "statement": {"game_id": "...", "seq": 57, "decide": "<hash of decide 57>",
 "action": ["RED", "BUILD_ROAD", [3, 4]]}, "sig": "<signature of canonical(statement)>"}
```

A client sends its seed only after it has received the server's signed `hello`,
which commits the server to its seed. A reconnecting client sends the seat key
it got in `joined` and the same public key; the server resends every message
after `last_seq`, so its hash chain continues unbroken.

## Fairness

### What the scheme guarantees

1. **Chance is unbiased and was fixed in advance.** When a room is created the
   server draws `server_seed` and publishes `H(server_seed)` in every `hello`.
   Each remote seat then contributes a fresh `client_seed` when it joins (server
   bots contribute none: their randomness would be the server's anyway). The
   board and the game's `random.Random` — turn order, the deck's shuffle, dice,
   steals — are derived from `H("catanatron-seed/1" ‖ canonical({server_seed,
   client_seeds in seat order}))`. Players cannot predict chance (they do not
   know `server_seed` until the end) and the server cannot steer it (it
   committed before seeing any client seed). Every chance event in Catanatron
   already comes from the game's own stream (`State.random`); the server never
   lets anything else draw from it — no bot ever holds the real game — so
   replaying the actions reproduces every roll and draw.
2. **Every view was the honest redaction of one committed game.** Every message
   is signed, numbered and hash-chained per recipient, and carries
   `H(canonical(full state) ‖ salt)` for the state it was built from. At the end
   the server reveals `server_seed`, every salt and the full action log.
   `verify_transcript` replays the game in stock Catanatron and checks that every
   commitment opens to the replayed state, and that every message body is
   *exactly* what an honest server builds for that recipient from that state:
   its view, its log entries, the actions it was offered, the board. A server
   that alters a roll, a deal or a hand after committing, shows a seat a
   doctored view, or seeds the game without the clients' seeds, fails.
3. **Moves cannot be forged.** Clients sign every action together with the hash
   of the `decide` message it answers; the reveal carries those signatures. The
   verifier checks every remote seat's moves against that seat's key (and that
   each of your moves answers a `decide` you received at that point). Moves the
   server makes for a seat must be the only legal one or, flagged as a timeout,
   the deterministic default.
4. **No equivocation.** Given several transcripts of one game, the verifier also
   checks that every recipient was sent the same commitment for each state.

### What it does not guarantee (threat model)

* **The server knows all hidden state during the game.** It holds the deck,
  every hand and (from its seed) every future roll. A server that colludes with
  a player can leak any of it, and nothing in the transcripts would show it.
  Removing this needs mental-poker style dealing (cards encrypted under every
  player's key, chance by joint coin flips), which is future work.
* **The server can refuse to play.** It can abort a game it does not like
  before the reveal (selective abort), or never admit a client. Clients should
  treat a game without a reveal as void.
* **Timeouts are the server's word.** A server can claim a seat timed out and
  play the default move for it. The verifier only checks that the forced move
  is the default and warns when the client's own transcript shows it had
  answered; it cannot prove when a message arrived.
* **Server bots are the server's.** Their moves are not signed by anyone else,
  and a dishonest server could let them peek at hidden state. Their seats are
  marked `bot` in every transcript.
* **Identity is trust-on-first-use.** The Python client trusts the key in the
  first `hello` unless given `server_key=`; pin it (it is printed at startup
  and served at `/api/server`) when that matters. Room creation and `seat_bot`
  are open to anyone who can reach the server.
* **Bots see sampled worlds, not beliefs.** `ViewBot` deals hidden cards
  uniformly among consistent hands, ignoring what steals and production reveal.

### The browser

The page checks every signature and the hash chain with WebCrypto, generates
its Ed25519 key in the browser (kept in `localStorage` with the seat key and
its seed, so a reload takes the seat back), signs every action, and stores each
finished transcript in IndexedDB ("Saved transcripts" in the lobby) for
`catanatron-verify`. WebCrypto Ed25519 needs a recent browser (Chrome 137+,
Firefox 129+, Safari 17+) and a secure context: open the page on `localhost` or
behind TLS (`https://`/`wss://`). Without it the page can watch but not play.

## Layout

| module | |
|---|---|
| `app.py` | the server: HTTP routes, lobby, room connections, `catanatron-server` |
| `room.py` | a room: seats, outboxes (per-recipient hash chains), the game loop |
| `messages.py` | message bodies, built from the game (shared by server and verifier) |
| `config.py` | `GameConfig`, `new_game(config, master_seed, game_id)`, `default_action` |
| `maps.py` | explicit map specs, board geometry |
| `crypto.py` | canonical JSON, Ed25519, seeds and commitments, envelopes |
| `client.py` | `GameClient`, `create_room`, `list_rooms` |
| `table.py` | reassembles a recipient's view from its messages |
| `sampling.py`, `bot.py` | registry bots on views, `catanatron-play-bot` |
| `verify.py` | `verify_transcript`, `catanatron-verify` |
| `static/` | the browser client |
