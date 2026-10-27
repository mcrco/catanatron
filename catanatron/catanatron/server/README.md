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
own bot classes for seats, as in `catanatron-play`), `--trusted-proxy ADDRESS`
(a reverse proxy in front of the server, e.g. the one terminating TLS: client
addresses are then taken from its `X-Forwarded-For`; without it, behind a
proxy, every per-address limit and ban below applies to the proxy, so to
everyone at once).

A bot from the registry, as a remote client (it sees only its seat's view):

```bash
catanatron-play-bot --spec AB:2 ws://localhost:8765/room/ROOM_ID
# create a 4-seat room, the other seats played by the server's F bots, and join it
catanatron-play-bot --spec F --new --players 4 --fill F ws://localhost:8765
```

It saves its transcript under `./transcripts/` and verifies it when the game ends.
Anyone can verify a transcript later (several transcripts of one room are also
checked against one another). The verdict is `OK` (a finished game, every check
passed), `FAILED` (the server's own signatures prove it cheated), `INVALID`
(the file is not evidence: a broken signature, or a header the chain belies),
`VOID` (the room was called off) or `INCOMPLETE` (the transcript stops early;
what it holds is still checked):

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
| `friendly_robber` | false | Catanatron's own friendly robber; refused by the server: whether a robbery is legal would depend on (and reveal) hidden victory point cards |
| `robber_mode` | `normal` | or `friendly_public`: OpenHex's friendly robber, which counts public points only (see below) |
| `dice_mode` | `random` | or `balanced`: OpenHex's balanced dice (see below) |
| `supply` | `base` (`extension` for 5–6) | bank, development deck and pieces |
| `decision_timeout_ms` | 60000 | after which the server plays the default action for the seat |
| `autoplay_forced` | true | play a seat's move without asking when everyone can see it has one legal action (see below) |

`robber_mode` and `dice_mode` are left out of a config (and of the
transcripts' configs) at their defaults, so rooms and transcripts from before
they existed are unchanged. Together with `num_players: 2`, `vps_to_win: 15`
and `discard_limit: 9` they are OpenHex's 1v1 rules:

- `robber_mode: friendly_public` (catanatron.models.actions.robber_possibilities):
  a tile with an opponent of the mover at 2 public points or fewer
  (settlements, cities, Longest Road, Largest Army; never victory point cards)
  is off limits, unless every tile but the robber's is, when all are open. A
  knight's tile is judged on the points before the knight's Largest Army
  award (OpenHex validates it first), so the seat the knight took the award
  from keeps its 2 points for that robber move (the document's
  `largest_army_taken_from`). Victims are not filtered. Since it reads public
  points only, the legal moves reveal nothing, and a lone robber move is
  publicly forced.
- `dice_mode: balanced` (catanatron.models.dice): a 36-card deck of face pairs,
  refilled to 36 before a roll when 13 or fewer are left (and when the main
  phase starts); a total weighs its cards left times max(0, 1 - 0.34 x its
  count among the last 5 totals), the 7 further times
  clamp(imbalance + 0.4 per 7 of the current run, negative for the run's
  owner, 0, 2). The dice's state is public (the document's `balanced_dice`,
  in every view), and the game's random stream draws from it, so the verifier
  replays the rolls from the revealed seeds as it does fair ones.

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
key, seed commitment and join signature (`commitments`, once per room), the
seats reveal their seeds, and the game starts. A room whose seats do not all
reveal in time (`reveal_timeout`, 30 s) is void (`aborted`, naming the seats
that did not reveal, revealing the server seed): there is no second round,
since a second set of commitments would be a second chance to reveal. The
blamed keys may not take a seat on the server for an hour, and an address
blamed three times in an hour may take none for that hour. To play anyway,
open a new room (the browser offers one with the same settings): new
commitments, new seeds from everyone. [PROTOCOL.md](PROTOCOL.md) is the
protocol's state machine and the invariants clients and the verifier keep.

During the game a seat that does not answer within the timeout is played
`default_action` (roll, else end the turn, else the first legal action), and
the move is flagged `"forced": "timeout"` (`"bot_error"` if a server bot
failed). A move is played without asking, flagged `"only"`, only when everyone
can see it is the only legal one: an initial placement or robber move with one
option, a free road with one place to go, or the roll of a seat that holds no
development card. A seat with one legal action whose singleness depends on its
hidden cards (a roll from a hand of only victory point cards, a discard from a
one-resource hand) is asked like any other, and bots answer such a move as
slowly as they answer real decisions ending in the same kind of move (a
duration drawn from the same bot's past decisions of that kind; for server
bots the room keeps these, server-wide, so no worker restart resets them), so
that the timing tells the table no more than the move does.

Server bots are seated exactly like remote ones: they get the same redacted
messages, decide on sampled games (never the real one) and pick only among the
actions they were offered. They run in a worker process per game (forked from
a forkserver that preloads Catanatron), off the server's event loop, each in a
process group of its own: a decision that overruns its time kills the worker's
group, stopping the computation and anything it started (an exec bot's
program and its children), and the next decision starts a fresh worker; bots
that would fan out into process pools (GreedyPlayouts) run single-process
there; a bot that raises anything, `SystemExit` included, plays the default.
Bot parameters are capped for server seats (`depth` 3, `num_simulations` 200,
`num_playouts` 100; an exec bot's `timeout_ms` at most the room's decision
time), games with server bots are capped server-wide (32) and server bots per
address (12). Exec bots (`--bot NAME=exec:...`) get the protocol's `before`,
`step` (if they asked to observe) and `after`, also when a game is called off,
and are stopped when the game ends (given 5 s, then killed with their group). A script that starts a `GameServer` itself must
guard its entry point with `if __name__ == "__main__":` (worker processes
import the main module).

The server limits connections (in total, per address and per room), messages
per second per connection (a flood closes the connection), spectators per
room, errors per recipient, submissions per seat (a receipt for each, up to
eight per decision the seat was asked plus a margin; past that the connection
is closed instead), rooms created per address per minute and rooms waiting per
address; drops waiting rooms nobody is connected to after 10 minutes and games
every remote seat has left after 30 minutes; forgets finished rooms after an
hour; and frees a seat taken before the start if its client stays away for 30
seconds. Limits and bans count IPv6 addresses by /64. Error messages say what
was wrong, never echoing what the client sent. A map (or player
count) whose settlement spots could run out before every seat has placed its
two settlements is refused (computed exactly). `catanatron-server` exits
promptly on SIGINT/SIGTERM, calling off every game in progress (which reveals
the server seed) and killing every bot worker with its process group (exec
bots' programs included).

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
all integers of magnitude at most 2^53 − 1 (a float, or a larger integer, is
refused), so that every client reads every signed body alike; map coordinates
are at most 100 from the center.

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
after); this is protocol version 5 (see [PROTOCOL.md](PROTOCOL.md)):

| body `type` | when | fields |
|---|---|---|
| `hello` | first message of every connection | `protocol_version`, `server_key`, `nonce` (a rejoin signs it), `room` (id, config, seats, status, `seed_commitment`) |
| `joined` | you took a seat | `room_id`, `seat`, `color`, `public_key`, `client_commitment`, `room` |
| `room` | seats changed before the lock (to every connection) | `room` |
| `commitments` | every seat is taken: reveal your seed (once per room) | `room_id`, `server_key`, `config`, `seats` (seat, color, kind, name, spec, `public_key`, `requested_seat`, `client_commitment`, `nonce`, `pop`), `seed_commitment`, `reveal_timeout_ms` |
| `before` | the game starts (or you arrive as a spectator) | `room_id`, `game_id`, `color` (null for spectators), `commitments` (the whole body), `config`, `seats`, `client_seeds`, `reveals` (each seat's reveal signature), `seed_commitment`, `map`, `board`, `index`, `state`, `log_start`, `log`, `current_color`, `winning_color` |
| `decide` | your move | `index`, `state`, `playable_actions` (Catanatron's order, the same in every process), `timeout_ms`, … |
| `received` | a receipt for every reveal and action you submit | `submission` (H of {statement, sig}), `index` (the action's, or null), `accepted`, `reason` (e.g. `stale`, `refused: …`) |
| `step` | someone acted | `index`, `state`, `log_start`, `log` (that action's entry), `forced` (null, `"only"`, `"timeout"` or `"bot_error"`), … |
| `after` | the game is over | `index`, `state`, `num_turns`, `winning_color`, `reveal`: `server_seed`, `salts`, `actions` |
| `aborted` | the room is void | `game_id` (null if it had not started), `reason`, `blame`, `server_seed` |
| `error` | a request was refused | `code`, `message` (never echoing the request), `answers` (which kind of request it answers) |

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
{"type": "join", "protocol_version": 5, "name": "me", "seat": 0,
 "public_key": "<Ed25519 public key>",
 "client_commitment": "<H({domain: 'catanatron-client-seed/1', room_id, public_key, client_seed})>",
 "pop": "<sig of {domain: 'catanatron-join/2', room_id, server_key, seed_commitment, seat,
          client_commitment, nonce}>"}          // nonce: this connection's hello's
{"type": "reveal", "client_seed": "<32 random bytes>",
 "sig": "<sig of {domain: 'catanatron-reveal/1', commitments: C}>"}   // once, after `commitments`
{"type": "rejoin", "protocol_version": 5, "public_key": "...",
 "sig": "<sig of {domain: 'catanatron-rejoin/1', room_id, nonce}>",
 "last_seq": 41}                     // reconnect: the hello's nonce; resend after 41
{"type": "spectate"}
{"type": "seat_bot", "seat": 2, "spec": "AB:2"}              // before the lock
{"type": "action", "statement": {"domain": "catanatron-action/1", "commitments": "C",
 "index": 57, "seq": 81, "decide": "<hash of decide 81>", "action": ["RED", "BUILD_ROAD", [3, 4]]},
 "sig": "<sig of canonical(statement)>"}
```

`pop` proves the joining client holds its key and binds its seed commitment to
the server (its key and seed commitment), the seat it asked for and the
connection it joins on; a key already seated in the room is refused. The
`commitments` publish every pop (a pop seats its key on its own connection
only, so publishing it gives nobody a seat), and each seat signs `C` when it
reveals, so every verifier can check that each key took part in this very
game (`C` names the server's key and every seat's fresh seed commitment). The server accepts an action only if
its statement is byte for byte the one for the pending `decide` and its action
is one of the `playable_actions` that decide offered; it plays its own copy of
that action, and answers every submission (reveal or action), accepted or not,
with a signed `received`, or closes the connection; a decision that has closed
(its deadline passed) gets its receipts only after its `step`. One answer per
decide: a refused move is never followed by another signed move. A
reconnecting client signs the new connection's `nonce` with its seat's key
(no seat key to lose: a client whose `joined` never arrived takes its seat
back the same way); the server closes the seat's previous connection and
resends every message after `last_seq`, so the chain continues unbroken
(messages on the new connection's own throw-away chain are not part of the
seat's transcript); the client then resends exactly what it had signed and
was in the middle of (its reveal, any answer the server has not
acknowledged).

## Fairness

### What the scheme guarantees

For a client running honest client code, against a server that may cheat (the
invariants behind each are in [PROTOCOL.md](PROTOCOL.md)):

1. **No one can steer chance after seeing the others' contributions.** The
   server publishes `H(server_seed)` in its `hello`. Each remote seat joins with
   a commitment to a fresh seed (bound to the room and its key). When every seat
   is taken the server publishes every seat's key, commitment and join
   signature and the config, once, signed and hash-chained; only then do seats
   reveal their seeds (signing that list's digest `C`), and the verifier checks
   every opening and signature. The board and the game's `random.Random` (turn
   order, the deck's shuffle, dice, steals) come from `H(server_seed, every
   revealed seed, C)`. So as long as *your* seed is unpredictable and you
   committed to it, chance is unpredictable to everyone else, the server and
   any other seat (which may be the server's own sock puppet, or a player
   colluding with it) included: all of them fixed their contributions before
   seeing yours. A client reveals once per room, for the first commitments it
   receives, and never uses a seed twice: there is no way to make it reveal
   into a second set of commitments. A seat that withholds its reveal voids
   the room, which is the one way left to "reroll" (see the threat model).
2. **You played the game you joined.** The game's commitments must be the ones
   published before the seeds (no rules changed, no seat swapped, relabelled or
   added), and every key in them must have signed its join and this game's
   `C`, so no one can be seated in a game they did not take part in.
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
4. **Moves cannot be forged or replayed, and a move you can show the server
   had is not silently overruled.** Every signed statement names `C` and the
   move's index, so it cannot be replayed at another point or in another game
   (even one hosted under the same room and game ids). Every move must be one
   the game offered. Moves the server makes for a seat must be publicly forced
   or, flagged, the deterministic default. The server answers every
   submission with a receipt naming it (statement and signature together, so a
   badly signed copy proves nothing), and signs no receipt for a decision
   between its deadline and its step: a receipt before the step (or any
   receipt saying `accepted`) for your legal, signed answer proves the server
   had it in time, and playing anything else is an error anyone can check. That
   the server never acknowledged an answer, or acknowledged it only after the
   step, rests on your word (you write `sent`): a warning, for you.
5. **No equivocation.** Given several transcripts of one room (under one
   server key), the verifier checks every pair: one seed commitment, one set
   of commitments, one game (its id, seeds and reveals), one reveal or one
   void, one state commitment at each point; every disagreeing transcript
   fails, whatever order the transcripts are given in. Since `C` names the
   server's key, one room cannot be served under two keys with the same
   seats' signatures.

What a transcript proves is checked over the whole chain, whatever it holds:
a void or an early end does not hide what came before it, and nothing but
receipts may follow the end. What only the client can say (its header, what
it sent, how long it had) is never turned into an accusation: a transcript
whose header the signed chain belies is not evidence, and a missing receipt
or a timeout is a warning.

### What it does not guarantee (threat model)

* **The server knows all hidden state during the game.** It holds the deck,
  every hand and (from its seed) every future roll. A server that colludes with
  a player can leak any of it, and nothing in the transcripts would show it.
  Removing this needs mental-poker style dealing (cards encrypted under every
  player's key, chance by joint coin flips), which is future work.
* **Any seat may be the server's.** Seats are keys, not people; nothing ties a
  seat to an identity. Guarantee 1 holds anyway (that is what the commitments
  are for), but a sock puppet can play, withhold and delay.
* **The server can void a game it dislikes.** It learns the outcome as soon as
  every seed is in, and a seat it controls can then withhold its reveal (or the
  server can call the game off). Each such reroll is a visible void, naming the
  withholder, and the next try needs a new room with new commitments and new
  seeds from every seat: exactly as costly as cancelling and recreating rooms,
  which no protocol can stop (selective abort). Treat a server whose rooms are
  often void, or a game without a reveal, as suspect. A withholder that is not
  the server cannot compute the outcome it would refuse (it does not know the
  server's seed), so it can only cost the others a new room; blamed keys and,
  past three blames an hour, addresses are kept out for an hour.
* **Deadlines are the server's word.** Whether an answer arrived in time is not
  provable: the verifier proves misbehaviour only from the server's own signed
  messages (a receipt before the step, an `accepted` receipt, an action
  answered with an error); a missing or late receipt is only a warning, since a
  client could claim to have sent anything. The same holds for a void that
  blames a seat whose reveal the server never acknowledged.
* **Timing leaks.** How long a person takes to answer can hint at a hand. Bots
  pace their hidden one-move answers like their real decisions of the same
  kind of move, which hides the gap as far as the move's kind goes, not in
  every single case; and until a bot has made real decisions of a kind (the
  server keeps this across games), it is paced from its decisions of any kind
  or a fixed range.
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
its Ed25519 key in the browser, commits to and reveals its seed (once per
room), signs every action (one per move, also after a reload), resends
unacknowledged answers after a reconnect, and stores each finished transcript
in IndexedDB ("Saved transcripts" in the lobby) for `catanatron-verify`, after
waiting a few seconds for the receipts it is owed, never replacing a stored
transcript with a poorer copy. It sends a join, a reveal or a move only once
the seat record that holds it is stored: with its storage full, it stops and
says so, rather than sign something a reload could make it sign again. Its seat records (key, seed, the
server's seed commitment, its reveal, signed answers) are kept in
`localStorage` per game server key and room, so a reload takes the seat back
and keeps what was signed, and a copy of the page used with several servers
never hands one server's seat to another. After a reload it plays only the
game it revealed its seed for, and a seat record whose game ended is never
sat in again. It refuses floats and integers beyond 2^53 on the wire (which
JavaScript and Python would read differently), and sorts keys by code point,
as Python does.
Everything the server sends is treated as data: it is escaped before display,
never interpreted as markup.

**But a page served by the game server protects you from nothing the server
does.** The server that serves the page could serve one that picks your seed,
signs moves you did not make, or drops what would expose it from the saved
transcript. Browser seats are only verifiable when the page (`index.html`,
`app.js` and `style.css`, all three) comes from a source independent of the
game server: a copy you host or keep yourself (the `static/` directory of a
release you trust, opened from a file or served by your own web server; it
loads its assets by relative path), pointed at the game server with
`?server=HOST:PORT` (add `&tls=1` for `wss://`/`https://`). Checking the
hashes of what the game server serves does not help: it can serve your
checking tool one file and your browser another. (A copy you downloaded,
checked against a release you trust, and then keep and open yourself is
fine: that is the first case.)

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
| `PROTOCOL.md` | the protocol: state machine, invariants, what is provable |
| `static/` | the browser client |
