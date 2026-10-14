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
| `friendly_robber` | false | refused by the server: whether a robbery is legal would depend on (and reveal) hidden victory point cards |
| `supply` | `base` (`extension` for 5–6) | bank, development deck and pieces |
| `decision_timeout_ms` | 60000 | after which the server plays the default action for the seat |
| `autoplay_forced` | true | play a seat's move without asking when everyone can see it has one legal action (see below) |

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
any map. A map (or player count) whose settlement spots could run out before
every seat has placed its two settlements is refused. Any shape works (hexagons, the 5–6 player board, rectangles,
islands): the board's node graph is built from the land tiles, so land that
only meets across water is not adjacent (no road, no distance rule there).

A room fills (remote clients, or server bots from the room's `seats` or a
`seat_bot` request); each remote seat commits to a seed when it joins. When
every seat is taken the seats are locked, the server publishes every seat's
key, seed commitment and join signature (`commitments`), the seats reveal their
seeds, and the game starts. A seat that does not reveal in time
(`reveal_timeout`, 30 s) is removed and blamed, its key and address may not
take a seat in that room again (and an address blamed three times in an hour
may take no seat on the server for that hour), the server reveals its seed and
commits to a new one, every revealed seed is discarded, and the remaining seats
commit afresh (`restart`); the freed seat can be taken by someone else. After
three rounds the room is called off (`aborted`, naming the seats).

During the game a seat that does not answer within the timeout is played
`default_action` (roll, else end the turn, else the first legal action), and
the move is flagged `"forced": "timeout"` (`"bot_error"` if a server bot
failed). A move is played without asking, flagged `"only"`, only when everyone
can see it is the only legal one: an initial placement or robber move with one
option, a free road with one place to go, or the roll of a seat that holds no
development card. A seat with one legal action whose singleness depends on its
hidden cards (a roll from a hand of only victory point cards, a discard from a
one-resource hand) is asked like any other, and bots answer such a move as
slowly as they answer real decisions (a duration drawn from their own past
ones), so that neither the answer nor its timing reveals the hand.

Server bots are seated exactly like remote ones: they get the same redacted
messages, decide on sampled games (never the real one) and pick only among the
actions they were offered. They run in a worker process per game (forked from
a forkserver that preloads Catanatron), off the server's event loop: a decision
that overruns its time kills the worker, stopping the computation and anything
it started, and the next decision starts a fresh worker; bots that would fan
out into process pools (GreedyPlayouts) run single-process there; a bot that
raises anything, `SystemExit` included, plays the default. Bot parameters are
capped for server seats (`depth` 3, `num_simulations` 200, `num_playouts` 100),
games with server bots are capped server-wide (32) and server bots per address
(12). Exec bots (`--bot NAME=exec:...`) get the protocol's `before`, `step` (if
they asked to observe) and `after`, also when a game is called off, and are
stopped when the game ends. A script that starts a `GameServer` itself must
guard its entry point with `if __name__ == "__main__":` (worker processes
import the main module).

The server limits connections (in total, per address and per room), messages
per second per connection, spectators per room, answers (errors and receipts)
per recipient, rooms created per address per minute and rooms waiting per
address; drops waiting rooms nobody is connected to after 10 minutes and games
every remote seat has left after 30 minutes; forgets finished rooms after an
hour; and frees a seat taken before the start if its client stays away for 30
seconds. Error messages never echo what the client sent. A map (or player
count) whose settlement spots could run out before every seat has placed its
two settlements is refused (computed exactly). `catanatron-server` exits
promptly on SIGINT/SIGTERM, calling off every game in progress (which reveals
the server seed) and killing every bot worker.

## Protocol

HTTP and WebSocket share one port:

| | |
|---|---|
| `GET /`, `/app.js`, `/style.css`, `/static/…` | the browser client |
| `GET /api/rooms`, `/api/rooms/ID` | rooms |
| `GET /api/players` | the registry's bots |
| `GET /api/server` | `{"server_key", "protocol_version", "app_sha256"}` |
| `WS /lobby` | `{"type": "create_room", "config", "seats", "name"}` → `{"type": "room_created", "room"}`; `{"type": "list_rooms"}` → `{"type": "rooms", "rooms"}`; `{"type": "players"}` |
| `WS /room/ID` | a seat or a spectator (below) |

The API answers any origin, so a copy of the browser client served from
elsewhere can use it. Client messages must be JSON objects whose numbers are
all integers (a float anywhere is refused).

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
null. All hashes, keys, seeds and signatures are lowercase hex. `H(x)` below is
`sha256(canonical(x))`.

The vocabulary extends `catanatron.protocol`'s (hello / before / decide / step /
after); this is protocol version 3:

| body `type` | when | fields |
|---|---|---|
| `hello` | first message of every connection | `protocol_version`, `server_key`, `room` (id, round, config, seats, status, `seed_commitment`) |
| `joined` | you took a seat | `room_id`, `seat`, `color`, `public_key`, `client_commitment`, `seat_key` (keep it to reconnect), `room` |
| `room` | seats changed before the lock (to every connection) | `room` |
| `commitments` | every seat is taken: reveal your seed | `room_id`, `round`, `config`, `seats` (seat, color, kind, name, spec, `public_key`, `requested_seat`, `client_commitment`, `pop`), `seed_commitment`, `reveal_timeout_ms` |
| `restart` | a seat withheld its seed: commit afresh | `room_id`, `round` (the new one), `reason`, `blame`, `server_seed` (the old one, revealed), `seed_commitment` (the new one), `room` |
| `before` | the game starts (or you arrive as a spectator) | `room_id`, `game_id`, `color` (null for spectators), `commitments` (the whole body), `config`, `seats`, `client_seeds`, `reveals` (each seat's reveal signature), `seed_commitment`, `map`, `board`, `index`, `state`, `log_start`, `log`, `current_color`, `winning_color` |
| `decide` | your move | `index`, `state`, `playable_actions` (Catanatron's order, the same in every process), `timeout_ms`, … |
| `received` | a receipt for every action you submit | `submission` (H of {statement, sig}), `index` (the pending decision's, or null), `accepted`, `reason` (e.g. `stale`, `refused: …`) |
| `step` | someone acted | `index`, `state`, `log_start`, `log` (that action's entry), `forced` (null, `"only"`, `"timeout"` or `"bot_error"`), … |
| `after` | the game is over | `index`, `state`, `num_turns`, `winning_color`, `reveal`: `server_seed`, `salts`, `actions` |
| `aborted` | the game is called off | `reason`, `blame`, `server_seed` |
| `error` | a request was refused | `code`, `message` (never echoing the request) |

`state` is your view of the game: Catanatron's `client_view(doc, your_color)`
(for spectators `public_view(doc)`) without `map` (sent once, in `before`) and
without `action_records`, which arrive incrementally as `log` (entries
`log_start…index-1`, already redacted for you: your own development-card draws
and the steals you were party to are shown, others' are not; dice, discards,
trades and everything else are public). `board` is the drawable geometry: per
tile its coordinate, type, resource/number or port, and node ids by corner
(pointy-top hexes, `NORTH … NORTHWEST`) and edges by side.
`catanatron.server.table.Table` reassembles messages into the full view.

Client → server on `/room/ID` (`C` is `H(commitments body)`):

```json
{"type": "join", "protocol_version": 3, "name": "me", "seat": 0,
 "public_key": "<Ed25519 public key>",
 "client_commitment": "<H({domain: 'catanatron-client-seed/1', room_id, public_key, client_seed})>",
 "pop": "<sig of {domain: 'catanatron-join/1', room_id, seed_commitment, seat, client_commitment}>"}
{"type": "commit", "client_commitment": "...", "pop": "..."}   // after a restart (seat = your seat)
{"type": "reveal", "client_seed": "<32 random bytes>",
 "sig": "<sig of {domain: 'catanatron-reveal/1', commitments: C}>"}   // after `commitments`
{"type": "join", "protocol_version": 3, "seat": 0, "seat_key": "...",
 "public_key": "...", "last_seq": 41}                       // reconnect: resend after 41
{"type": "spectate"}
{"type": "seat_bot", "seat": 2, "spec": "AB:2"}              // before the lock
{"type": "action", "statement": {"domain": "catanatron-action/1", "commitments": "C",
 "index": 57, "seq": 81, "decide": "<hash of decide 81>", "action": ["RED", "BUILD_ROAD", [3, 4]]},
 "sig": "<sig of canonical(statement)>"}
```

`pop` proves the joining client holds its key and binds its seed commitment to
the server's commitment it saw; a key already seated in the room is refused.
The `commitments` publish every pop, and each seat signs `C` when it reveals,
so every verifier can check that each key took part in this very game (`C`
names every seat's fresh seed commitment). The server accepts an action only if
its statement is byte for byte the one for the pending `decide` and its action
is one of the `playable_actions` that decide offered; it plays its own copy of
that action, and answers every submission, accepted or not, with a signed
`received`. One answer per decide: a refused move is never followed by another
signed move. A reconnecting client sends the seat key it got in `joined` and
the same public key; the server closes the seat's previous connection and
resends every message after `last_seq`, so the chain continues unbroken
(messages on the new connection's own throw-away chain are not part of the
seat's transcript); the client then resends whatever it was in the middle of
(a commitment, a reveal, any answer the server has not acknowledged).

## Fairness

### What the scheme guarantees

For a client running honest client code, against a server that may cheat:

1. **No one can steer chance after seeing the others' contributions.** The
   server publishes `H(server_seed)` in its `hello`. Each remote seat joins with
   a commitment to a fresh seed (bound to the room and its key). When every seat
   is taken the server publishes every seat's key, commitment and join
   signature and the config, signed and hash-chained; only then do seats reveal
   their seeds (signing that list's digest `C`), and the verifier checks every
   opening and signature. The board and the game's `random.Random` (turn order,
   the deck's shuffle, dice, steals) come from `H(server_seed, every revealed
   seed, C)`. So as long as *your* seed is unpredictable and you committed to
   it, chance is unpredictable to everyone else, the server and any other seat
   (which may be the server's own sock puppet, or a player colluding with it)
   included: all of them fixed their contributions before seeing yours.
   Withholding a reveal does not void the game any more: the withholder is
   removed and every seed (the server's too) is replaced. A withholder that is
   not the server cannot compute the outcome it would refuse (it does not know
   the server's seed), so it can only delay; a server colluding with
   withholders could reroll at most `max_rounds` (3) times, as it could by
   cancelling and recreating rooms.
2. **You played the game you joined.** The game's commitments must be the ones
   published before the seeds (no rules changed, no seat swapped, relabelled or
   added), and every key in them must have signed this round's join and this
   game's `C`, so no one can be seated in a game they did not take part in.
3. **Every view was the honest redaction of one committed game.** Every message
   is signed, numbered and hash-chained per recipient, and carries
   `H(canonical(full state) ‖ salt)` for the state it was built from. At the end
   the server reveals `server_seed`, every salt and the full action log.
   `verify_transcript` replays the game in stock Catanatron and checks that every
   commitment opens to the replayed state, and that every message body is
   *exactly* what an honest server builds for that recipient from that state:
   its view, its log entries, the actions it was offered, the board. It also
   checks the schedule: one `before`, one `step` per action in order (nothing
   withheld or repeated), a `decide` exactly when it was your move and the move
   was not publicly forced, at most one per move, one `after`.
4. **Moves cannot be forged or replayed, and your moves are not silently
   overruled.** Every signed statement names `C` and the move's index, so it
   cannot be replayed at another point or in another game (even one hosted
   under the same room and game ids). Every move must be one the game offered.
   Moves the server makes for a seat must be publicly forced or, flagged, the
   deterministic default. And for each of your moves: if your transcript holds
   your legal, correctly signed answer and the server played something else,
   the verifier reports it when the server's receipt for that answer precedes
   the step that played the move (it had your answer in time), and also when
   the server never acknowledged the answer at all (an honest server
   acknowledges every submission, and your client resends unacknowledged
   answers). A receipt only *after* that step means your answer was late: a
   warning, never an accusation. A receipt names statement and signature
   together, so a client cannot frame the server with a badly signed copy.
5. **No equivocation.** Given several transcripts of one room and server
   commitment (under one server key), the verifier checks that every recipient
   got the same game id, commitments, seeds and reveal, and the same commitment
   for each state; every disagreeing transcript fails.

### What it does not guarantee (threat model)

* **The server knows all hidden state during the game.** It holds the deck,
  every hand and (from its seed) every future roll. A server that colludes with
  a player can leak any of it, and nothing in the transcripts would show it.
  Removing this needs mental-poker style dealing (cards encrypted under every
  player's key, chance by joint coin flips), which is future work.
* **Any seat may be the server's.** Seats are keys, not people; nothing ties a
  seat to an identity. Guarantee 1 holds anyway (that is what the commitments
  are for), but a sock puppet can play, withhold and delay.
* **Withholding delays.** A seat (or a series of throwaway keys from several
  addresses) can delay a room's start by withholding reveals, up to the rounds
  limit, after which the room is called off. Bans by key and address, and the
  server-wide blame limit, make this cost addresses.
* **The server can refuse to play.** It can call off a game before the reveal
  (selective abort), or never admit a client. Treat a game without a reveal as
  void.
* **Deadlines are the server's word.** Whether an answer arrived in time is not
  provable: the verifier proves misbehaviour only from the order of the
  server's own signed messages (a receipt before the step) or from a missing
  receipt; a server that times you out and later acknowledges your answer as
  "stale" produces only a warning.
* **Timing leaks.** How long a person takes to answer can hint at a hand. Bots
  pace their forced answers like real ones, which hides the gap in
  distribution, not in every single case.
* **Server bots are the server's.** Their moves are not signed by anyone else,
  and a dishonest server could let them peek at hidden state. Their seats are
  marked `bot` in every transcript.
* **Identity is trust-on-first-use.** The Python client trusts the key in the
  first `hello` unless given `server_key=`; pin it (it is printed at startup
  and served at `/api/server`) when that matters. `catanatron-verify` checks a
  transcript against the key inside it. Room creation and `seat_bot` are open
  to anyone who can reach the server (within the limits above).
* **Bots see sampled worlds, not beliefs.** `ViewBot` deals hidden cards
  uniformly among consistent hands, ignoring what steals and production reveal.

### The browser

The page checks every signature and the hash chain with WebCrypto, generates
its Ed25519 key in the browser (kept in `localStorage` with the seat key and
its seed, so a reload takes the seat back), commits to and reveals its seed,
signs every action, resends unacknowledged answers after a reconnect, and
stores each finished transcript in IndexedDB ("Saved transcripts" in the lobby)
for `catanatron-verify`. Everything the server sends is treated as data: it is
escaped before display, never interpreted as markup.

**But a page served by the game server protects you from nothing the server
does.** The server that serves the page could serve one that picks your seed,
signs moves you did not make, or drops what would expose it from the saved
transcript. Browser seats are only verifiable when the page (`index.html`,
`app.js` and `style.css`, all three) comes from a source independent of the
game server:

* a copy you host or keep yourself (the `static/` directory of a release you
  trust, opened from a file or served by your own web server; it loads its
  assets by relative path), pointed at the game server with
  `?server=HOST:PORT` (add `&tls=1` for `wss://`/`https://`); or
* a copy whose hashes you check before playing: the server's `/api/server` and
  startup line give the SHA-256 of its `app.js`, but a malicious server can lie
  there, so compare what you actually download (`curl -s URL | sha256sum`, for
  the page and its assets) with the hashes of the release you trust.

Otherwise treat a browser seat as trusting the server, and use the Python
client (or `catanatron-play-bot`) to play verifiably.

WebCrypto Ed25519 needs a recent browser (Chrome 137+, Firefox 129+, Safari
17+) and a secure context: open the page on `localhost`, from a file, or behind
TLS. Without it the page can watch but not play, and says the session is
unverified (it follows the messages' sequence numbers only).

## Layout

| module | |
|---|---|
| `app.py` | the server: HTTP routes, lobby, room connections, `catanatron-server` |
| `room.py` | a room: seats, outboxes (per-recipient hash chains), the game loop |
| `messages.py` | message bodies, built from the game (shared by server and verifier) |
| `config.py` | `GameConfig`, `new_game(config, master_seed, game_id)`, `default_action` |
| `maps.py` | explicit map specs, board geometry |
| `hashing.py` | canonical JSON, hashes, seeds and commitments (standard library only) |
| `crypto.py` | Ed25519 keys and signatures, envelopes, the signed statements |
| `client.py` | `GameClient`, `create_room`, `list_rooms` |
| `table.py` | reassembles a recipient's view from its messages |
| `sampling.py`, `bot.py` | registry bots on views, `catanatron-play-bot` |
| `botproc.py` | the worker processes server bots run in |
| `verify.py` | `verify_transcript`, `catanatron-verify` |
| `static/` | the browser client |
