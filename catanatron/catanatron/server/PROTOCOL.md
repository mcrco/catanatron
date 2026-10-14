# catanatron-server protocol 4

This is the specification: the room's state machine, who signs what, what
each party keeps, and the invariants that clients (`client.py`, the browser's
`static/app.js`) and the verifier (`verify.py`) enforce. The README explains
how to run things; where the two disagree, this file is right and the code
is wrong.

Notation: `canonical(x)` is JSON with sorted keys, no whitespace, UTF-8;
`H(x) = sha256(canonical(x))`, lowercase hex. `sign_K(x)` is an Ed25519
signature by key `K` over `canonical(x)`. `S` is the server's key.

## 1. Messages

Every message from the server on a room connection is an envelope
`{to, seq, prev, commit, body, sig}` with `sig = sign_S({to, seq, prev,
commit, body})`. Each recipient has its own chain: `seq` counts from 0, `prev`
is `H(head)` of the previous envelope (64 zeros first), `to` is the
recipient's public key once it holds a seat (null before, and for
spectators), `commit` is `{index, hash}` with `hash = sha256(canonical(full
state after index actions) ‖ salt_index)` once the game exists (null before).

Server bodies:

| body | sent | content |
|---|---|---|
| `hello` | first message of every connection | `protocol_version` 4, `server_key`, `nonce` (16 random bytes, fresh per connection), `room` (id, name, status, config, `seed_commitment`, seats) |
| `joined` | the join was accepted | `room_id`, `seat`, `color`, `public_key`, `client_commitment`, `room` |
| `room` | seats changed while `waiting` | `room` |
| `commitments` | once, when every seat is taken | exactly `type`, `room_id`, `config`, `seats` (per seat: `seat`, `color`, `kind`, `name`, `spec`, `public_key`, `requested_seat`, `client_commitment`, `pop`), `seed_commitment`, `reveal_timeout_ms` |
| `received` | the answer to a `reveal` or an `action` | `submission`, `index` (the action's claimed index, or null), `accepted`, `reason` |
| `before` | the game starts (or a spectator arrives) | the whole `commitments` body, `client_seeds`, `reveals`, `game_id`, the board and the recipient's view |
| `decide` | the recipient's move | `index`, view, `playable_actions`, `timeout_ms` |
| `step` | a move was played | `index`, view, the move's log entry, `forced` (null, `only`, `timeout`, `bot_error`) |
| `after` | the game is over | view, `reveal`: `server_seed`, every salt, every action |
| `aborted` | the room is void | `room_id`, `reason`, `blame` (seat indices), `server_seed` |
| `error` | a request was refused | `code`, `message` (never echoing the request), `answers` (the request's type, if it is one the room knows) |

Client requests on `/room/ID` (`C = H(commitments body)`):

| request | signed statement |
|---|---|
| `join {protocol_version, name, seat, public_key K, client_commitment, pop}` | `pop = sign_K({domain: "catanatron-join/1", room_id, seed_commitment, seat, client_commitment})`, where `client_commitment = H({domain: "catanatron-client-seed/1", room_id, public_key: K, client_seed})` |
| `rejoin {protocol_version, public_key K, sig, last_seq}` | `sig = sign_K({domain: "catanatron-rejoin/1", room_id, nonce})`, `nonce` from this connection's `hello` |
| `reveal {client_seed, sig}` | `sig = sign_K({domain: "catanatron-reveal/1", commitments: C})` |
| `action {statement, sig}` | `statement = {domain: "catanatron-action/1", commitments: C, index, seq, decide: H(head of the decide envelope), action}`, `sig = sign_K(statement)` |
| `spectate`, `seat_bot {seat, spec}`, `ping` | none |

A receipt's `submission` is `H({statement, sig})`: for an action, its
statement; for a reveal, the reveal statement above. Statement and signature
together, so a receipt for a badly signed copy is not a receipt for the real
thing.

The game's master seed is `sha256("catanatron-seed/1" ‖ canonical({server_seed,
client_seeds (seat order, null for server bots), binding: C}))`; the board,
turn order, deck and every roll come from it, in stock Catanatron.

## 2. A room's state machine

```
           join / seat_bot / leave            every seat taken
 waiting ───────────────────────────► waiting ─────────────────► revealing
                                                 (commitments, once)   │
                                                                       │ every remote seat revealed
                                             reveal_timeout passes     ▼
            aborted ◄──────────────────────────────────────────── playing ──► finished
          (blame, server seed)      (also: shutdown, abandoned,          (after: server seed,
                                     the server failed)                   salts, actions)
```

* **waiting.** Connections get `hello` (and `room` updates). A `join` takes an
  open seat: the server checks the pop, that the key holds no seat here, and
  that neither key nor address is banned, then sends `joined` on that
  connection's chain, which from then on is the seat's chain. `seat_bot` puts
  a server bot in an open seat. A seat whose connection drops is held for
  `seat_grace_s` (it can `rejoin`), then freed.
* **revealing.** When no seat is open the server builds the one `commitments`
  body (the locked seats with their keys, commitments and pops, the config,
  its own seed commitment) and sends it, once, to every seat, every spectator
  and every connection still looking at the room (which become spectators:
  the game is all there is left to see). Each remote seat answers with its
  `reveal`; each reveal gets a `received` (accepted or not). With no remote
  seat the game starts at once.
* **playing.** When every remote seat has revealed, the server derives the
  game from the master seed and sends `before`. Then for each move: a
  publicly forced move is played (`forced: "only"`); otherwise the seat gets a
  `decide` (a server bot decides in its worker; a one-move decision that is
  not publicly forced is answered by the room, paced like that bot's real
  decisions of that kind), and the server waits for an accepted `action` or
  the deadline, plays the answer or the default (`forced: "timeout"` /
  `"bot_error"`), and sends `step` to everyone.
* **finished.** `after` reveals the server seed, every salt and every action.
* **aborted (void).** A room whose remote seats have not all revealed
  `reveal_timeout_ms` after the commitments is void: one `aborted` to every
  chain, naming the seats that did not reveal and revealing the server seed.
  Their keys may not take a seat on this server for an hour; an address
  blamed three times in an hour may not take seats for that hour. There is no
  second round: a new game needs a new room, with new commitments and new
  seeds from everyone. (Shutdown, a game every remote seat abandoned, or a
  server failure also end a room as `aborted`, without blame.)

What each party keeps:

* **Server**, per room: its seed; the commitments; per seat the key,
  commitment, pop, seed and reveal signature; every outbox (chain); the
  salts and actions. Server-wide: blamed keys and addresses (an hour), the
  pace book of its bots.
* **Python client** (`GameClient`): its key, its one seed and commitment, the
  chain it received, the commitments (the first, the only), its reveal, what
  it signed (`sent`, with the message it was sent after), answers not yet
  receipted. The transcript is the chain plus `public_key`, `client_seed`,
  `reveal_sig`, `sent`.
* **Browser**: the same, in a seat record in `localStorage` keyed by the
  server's key and the room (`catanatron-seat-<server key>-<room>`: key pair,
  seed, commitment, pop, seat, the reveal, `sent`, unreceipted answers),
  written before the join goes out; finished transcripts in IndexedDB, never
  replaced by a poorer copy.

## 3. Invariants

Each invariant names what the clients check live, what the verifier checks
afterwards, and the tests (`tests/server/...`) that attack it.

**I1. One reveal per room, one seed per join.** A seat signs exactly one
reveal per room and server key: for the first `commitments` it receives, with
the seed it joined with. It never signs a reveal for another commitments
body, never reveals into a room again after a void, and never uses a seed
for a second room.
* Clients: `GameClient` makes its seed when it is created and keeps the first
  commitments; a different second one raises `ServerMisbehaved` (an identical
  copy changes nothing and is ignored); after a reconnect it resends its
  stored reveal, never a fresh signature. The browser stores the reveal (and
  the digest it signed) in the seat record and refuses to sign for another
  digest; each `takeSeat` makes a fresh key and seed.
* Verifier: I2, and every reveal signature in `before` verifies over the one
  commitments' digest.
* Tests: `test_protocol4.py::test_a_second_commitments_is_never_revealed_into`,
  `test_browser.py::test_a_regrinding_server_is_refused_by_the_page`,
  `test_review.py::test_a_seat_that_does_not_reveal_voids_the_room_and_is_blamed`.

**I2. One commitments, one before, no rounds.** A chain holds at most one
`commitments` and at most one `before`, and no message type outside the
table above (protocol 3's `restart` included).
* Clients: as I1; the browser reports unknown message types.
* Verifier: fails a second `commitments`, a second `before`, and any unknown
  type.
* Tests: `test_protocol4.py::test_a_transcript_with_two_commitments_fails`,
  `test_protocol4.py::test_a_restart_message_fails`.

**I3. Everything signed names C.** Reveals sign `C`; action statements name
`C`, the move's index, and the `decide` they answer (its `seq` and hash).
* Clients: sign only over `H` of the commitments they kept.
* Verifier: reveal signatures over `C`; every played client move's statement
  equals `action_statement(C, index, seq, decide, action)`; this client's
  moves answer the decide it got at that point.
* Tests: `test_review2.py::test_a_players_signatures_cannot_be_carried_into_another_game`,
  `test_review.py::test_a_signature_cannot_be_replayed_for_another_move`.

**I4. One seed commitment per room.** `hello`, `room`, `commitments` and
`before` carry the same `seed_commitment`, and the seed revealed by `after` or
`aborted` opens it.
* Clients: `GameClient` checks commitments, `before` and `aborted`
  (`ServerMisbehaved`); the browser checks commitments and `aborted`.
* Verifier: the same, for every message.
* Tests: `test_protocol4.py::test_a_void_must_open_the_seed_commitment`.

**I5. The game is the committed one.** `before.commitments` is the
commitments body this recipient got; its config and seats are those; every
remote seed opens its seat's commitment; this client's seed is its own; the
game replays from the master seed of all of it.
* Clients: `GameClient` checks `before` against its commitments and its seed;
  the browser too.
* Verifier: all of it, then the replay.
* Tests: `test_review.py::test_seeds_that_do_not_open_their_commitments_are_caught`,
  `test_review.py::test_a_game_other_than_the_committed_one_is_caught`,
  `test_verify.py::test_a_game_not_seeded_from_the_clients_seeds_is_caught`.

**I6. Seated as joined.** `joined` names this client's key and commitment;
the commitments seat that key once, at that seat, with that commitment and a
pop that verifies.
* Clients: both check `joined` and the commitments.
* Verifier: the same, plus every seat's pop under the room's seed commitment.
* Tests: `test_review.py::test_a_join_needs_proof_of_the_key_and_a_key_not_already_seated`,
  `test_verify.py::test_someone_elses_transcript_does_not_verify_as_mine`.

**I7. A void is announced, signed and blamed, and blames honestly what it
can be held to.** A room whose remote seats do not all reveal in time ends
with one `aborted` to every chain, naming the seats that did not reveal and
revealing the server seed. The server answers every reveal with a receipt,
and never blames a seat whose reveal it acknowledged as accepted.
* Clients: `GameClient.play` returns with `GameResult.aborted`; the browser
  shows the void and offers a new room with the same settings.
* Verifier: `report.void`; the seed opens the commitment; a blamed seat whose
  transcript holds its reveal and an accepted receipt for it before the void
  is an error; a blamed seat without that receipt is a warning.
* Tests: `test_review.py::test_a_seat_that_does_not_reveal_voids_the_room_and_is_blamed`,
  `test_protocol4.py::test_blaming_a_seat_whose_reveal_was_acknowledged_is_caught`.

**I8. One chain per recipient.** Every message is signed by `S`, numbered
from 0 and hash-chained, and addressed to the seat's key once it is seated.
* Clients: checked on every message as it arrives.
* Verifier: the same; a transcript that fails it is not evidence of anything.
* Tests: `test_verify.py::test_an_edited_transcript_fails_its_signatures`,
  `test_review.py::test_a_client_survives_many_brief_drops` (the chain
  survives reconnects).

**I9. A seat's chain is taken back only by its key, on that connection.** A
`rejoin` must sign the nonce of the connection it is sent on with the seat's
key; the server then closes the seat's previous connection and resends its
chain after `last_seq`. A client whose `joined` was lost still knows its key,
so it takes its seat back; nobody else can (a pop is public, a nonce is
fresh per connection).
* Clients: `GameClient` rejoins whenever its join went out; on `not_seated`
  it joins afresh (the join never landed). The browser writes its seat record
  before the join goes out and does the same.
* Verifier: not concerned (a rejoin's own chain is not part of a transcript).
* Tests: `test_protocol4.py::test_a_client_that_loses_joined_takes_its_seat_back`,
  `test_protocol4.py::test_a_rejoin_signature_is_good_on_one_connection_only`,
  `test_review.py::test_a_rejoin_closes_the_old_connection`,
  `test_browser.py::test_a_page_that_loses_joined_takes_its_seat_back`.

**I10. Every view is the honest redaction of one committed game.** Each
body is exactly what an honest server builds from the replayed state for its
recipient, and each state commitment opens with its revealed salt.
* Verifier: the replay. Tests: `test_verify.py` (tampered views, altered
  dice), `test_review.py::test_moves_not_offered_by_the_game_fail_verification`.

**I11. The schedule.** One `before`; one `step` per action, in order; a
`decide` exactly when this seat had a move that was not publicly forced, at
most one per move; one `after`.
* Verifier. Tests: `test_review.py::test_the_message_schedule_is_checked`,
  `test_review2.py::test_forced_moves_follow_what_the_table_can_see`.

**I12. Every move is authorised.** A remote seat's move is its own signed
statement; a server move is publicly forced (`only`, the only legal action)
or flagged `timeout`/`bot_error` and the default action; a server bot's move
is the server's.
* Verifier. Tests: `test_verify.py::test_a_forged_move_is_caught_by_everyone`.

**I13. Every submission is answered.** Every `reveal` while seeds are being
revealed, and every `action` once the game exists, gets exactly one signed
`received` naming it, or the connection is closed (code 4009, past the seat's allowance of
`receipt_margin + receipts_per_decision × decisions asked`, which an honest
client never reaches). During the game an action is never answered with an
`error`. Receipts do not count against the error allowance (`max_answers`).
An `action` before the game is answered with an `error` (`no_game`), and so
is a `reveal` outside `revealing` (`not_revealing`: a resend after the game
started).
* Clients: `GameClient` resends every answer not yet receipted after a
  reconnect, and waits `RECEIPT_WAIT_S` at the end for receipts still owed; the
  browser keeps unreceipted answers in its seat record (a reload keeps them)
  and reconnects after the end while it is owed receipts.
* Verifier: an `error` answering an `action` during the game is an error;
  receipts or errors before the game are ignored.
* Tests: `test_protocol4.py::test_every_submission_gets_a_receipt_or_a_closed_connection`,
  `test_protocol4.py::test_receipts_do_not_use_up_the_error_allowance`,
  `test_protocol4.py::test_an_action_before_the_game_gets_an_error_and_verifies`,
  `test_review2.py::test_a_legal_move_the_server_overrules_is_reported`.

**I14. Deadlines are atomic.** A decision closes (its deadline passes, or an
answer is accepted) and its `step` is pushed with no receipt naming that
decision in between: a submission that arrives in that window gets its
receipt after the step. So a receipt before step `k` means the server had
the submission while decision `k` was open.
* Verifier: a legal, correctly signed answer of this client to decide `k`
  that was not played, with a receipt before step `k` or a receipt saying
  `accepted` anywhere, is an error.
* Tests: `test_protocol4.py::test_answers_at_the_deadline_never_frame_an_honest_server`,
  `test_protocol4.py::test_an_accepted_answer_the_server_did_not_play_is_caught`,
  `test_review2.py::test_a_legal_move_the_server_overrules_is_reported`.

**I15. No equivocation within a room.** Any two transcripts of one room
under one server key agree on the seed commitment, the commitments, the game
(id, seeds, reveals), its reveal or its void, and the state commitment at
each index. Every pair is compared, so the verdict does not depend on the
order the transcripts are given in.
* Verifier: `verify_transcripts` (and `catanatron-verify` with several files).
* Tests: `test_review2.py::test_a_server_naming_different_games_to_different_recipients_is_caught`,
  `test_protocol4.py::test_every_pair_of_transcripts_of_a_room_is_compared`,
  `test_protocol4.py::test_a_void_for_some_and_a_game_for_others_is_caught`.

## 4. What is provable, and to whom

A transcript whose signatures and chain check (I8) is a record of what the
server signed, to whom, in what order. Everything below is about such a
transcript.

**Provable to anyone (verifier errors).** Whatever contradicts the server's
own signatures: a broken invariant I1–I7 or I10–I15 inside one transcript, or
I15 across two. In particular: a second commitments; a game other than the
committed one; a view that is not the honest redaction; a move nobody
authorised; a void whose seed does not open the commitment; a blame on a
seat whose reveal the server acknowledged; an action answered by an error
during the game; a legal signed answer the server acknowledged in time (or
as accepted) and did not play; two recipients of one room told different
things.

**Known only to the client (verifier warnings).** What rests on the client's
word, because the client writes `sent` itself and decides what it sends:
* that it sent an answer the server never acknowledged (a missing receipt);
* that an answer was late (acknowledged only after its step), or on time
  although the server timed it out: deadlines are the server's word;
* that it revealed although the server blamed it, when the server never
  acknowledged the reveal;
* that the server went silent or closed connections.

**Not evidence.** A transcript that fails I8 (anyone can write one), and the
parts of a transcript after which it stops (an unfinished game proves only
what it holds). The live client does know: it raises `ServerMisbehaved` the
moment a message fails a check, which is why it checks as it plays.

Tests that a client cannot turn the unprovable into an accusation:
`test_protocol4.py::test_a_client_cannot_frame_an_honest_server` (an answer
listed in `sent` but never sent; eight badly signed copies before the real
one), `test_review2.py::test_an_honest_server_is_not_accused` (drops, lost
and late answers, a badly signed copy first),
`test_review2.py::test_an_unacknowledged_move_is_the_clients_word_only`.

## 5. What the server can still do

* **Void a game it dislikes.** The server learns the outcome as soon as every
  seed is in; a seat it controls can then withhold its reveal, voiding the
  room (or the server can call it off). Every such reroll is a visible void,
  naming the withholder, and the next game needs a new room with new
  commitments and new seeds from every seat. Rerolling is therefore exactly
  as costly to the server as cancelling and recreating rooms, which the
  threat model already admits (selective abort): treat a server whose rooms
  are often void as dishonest. What protocol 3's restart rounds allowed
  beyond that (a second commitments within one room, so a reveal of a seed
  the server had already seen, or unlimited rounds) is gone: there is one
  commitments per room, and clients reveal once.
* Everything the README's threat model lists: it knows all hidden state
  during the game, any seat may be its puppet, deadlines and timing are its
  word, its own bots are its own.
