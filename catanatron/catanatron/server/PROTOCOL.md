# catanatron-server protocol 5

This is the specification: the room's state machine, who signs what, what
each party keeps, and the invariants that clients (`client.py`, the browser's
`static/app.js`) and the verifier (`verify.py`) enforce. The README explains
how to run things; where the two disagree, this file is right and the code
is wrong.

Notation: `canonical(x)` is JSON with keys sorted by code point, no
whitespace, UTF-8; `H(x) = sha256(canonical(x))`, lowercase hex.
`sign_K(x)` is an Ed25519 signature by key `K` over `canonical(x)`. `S` is
the server's key. Every number in the protocol is an integer of magnitude at
most 2^53 − 1, which every client represents exactly; the server refuses
anything else from clients (and bounds what clients may put in signed
bodies, such as map coordinates), and clients and the verifier refuse it
from the server.

Changes from protocol 4: the commitments (so `C`) name the server's key; a
join's proof of possession signs the server's key and the nonce of the
connection it joins on (so a published pop seats nothing); a void names its
game; the verifier checks whole chains whatever a void says.

## 1. Messages

Every message from the server on a room connection is an envelope
`{to, seq, prev, commit, body, sig}` with `sig = sign_S({to, seq, prev,
commit, body})`. Each recipient has its own chain: `seq` counts from 0, `prev`
is `H(head)` of the previous envelope (64 zeros first), `to` is the
recipient's public key once it holds a seat (null before, and for
spectators), `commit` is `{index, hash}` with `hash = sha256(canonical(full
state after index actions) ‖ salt_index)` once the game exists (null before).
A connection carries exactly one chain at a time.

Server bodies:

| body | sent | content |
|---|---|---|
| `hello` | first message of every connection | `protocol_version` 5, `server_key`, `nonce` (16 random bytes, fresh per connection), `room` (id, name, status, config, `seed_commitment`, seats) |
| `joined` | the join was accepted | `room_id`, `seat`, `color`, `public_key`, `client_commitment`, `room` |
| `room` | the seats changed while `waiting` | `room` (pushed only when the seats changed since the last one) |
| `commitments` | once, when every seat is taken | exactly `type`, `room_id`, `server_key`, `config`, `seats` (per seat exactly: `seat`, `color`, `kind`, `name`, `spec`, `public_key`, `requested_seat`, `client_commitment`, `nonce`, `pop`), `seed_commitment`, `reveal_timeout_ms` |
| `received` | the answer to a `reveal` or an `action` | `submission`, `index` (the action's claimed index, or null), `accepted`, `reason` |
| `before` | the game starts (or a spectator arrives) | the whole `commitments` body, `client_seeds`, `reveals`, `game_id`, the board and the recipient's view |
| `decide` | the recipient's move | `index`, view, `playable_actions`, `timeout_ms` |
| `step` | a move was played | `index`, view, the move's log entry, `forced` (null, `only`, `timeout`, `bot_error`) |
| `after` | the game is over | view, `reveal`: `server_seed`, every salt, every action |
| `aborted` | the room is void | `room_id`, `game_id` (the game's, or null if it had not started), `reason`, `blame` (seat indices), `server_seed` |
| `error` | a request was refused | `code`, `message` (never echoing the request), `answers` (the request's type, if it is one the room knows) |

Client requests on `/room/ID` (`C = H(commitments body)`; `N` is the nonce of
the `hello` on the connection the request is sent on):

| request | signed statement |
|---|---|
| `join {protocol_version, name, seat, public_key K, client_commitment, pop}` | `pop = sign_K({domain: "catanatron-join/2", room_id, server_key: S, seed_commitment, seat, client_commitment, nonce: N})`, where `client_commitment = H({domain: "catanatron-client-seed/1", room_id, public_key: K, client_seed})` |
| `rejoin {protocol_version, public_key K, sig, last_seq}` | `sig = sign_K({domain: "catanatron-rejoin/1", room_id, nonce: N})` |
| `reveal {client_seed, sig}` | `sig = sign_K({domain: "catanatron-reveal/1", commitments: C})` |
| `action {statement, sig}` | `statement = {domain: "catanatron-action/1", commitments: C, index, seq, decide: H(head of the decide envelope), action}`, `sig = sign_K(statement)` |
| `spectate`, `seat_bot {seat, spec}`, `ping` | none |

A receipt's `submission` is `H({statement, sig})`: for an action, its
statement; for a reveal, the reveal statement above. Statement and signature
together, so a receipt for a badly signed copy is not a receipt for the real
thing.

The game's master seed is `sha256("catanatron-seed/1" ‖ canonical({server_seed,
client_seeds (seat order, null for server bots), binding: C}))`; the board,
turn order, deck and every roll come from it, in stock Catanatron. `C` names
the room, the server's key, the rules, every seat with its key, fresh seed
commitment and join, and the server's seed commitment.

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

* **waiting.** Connections get `hello` (and `room` when the seats change). A
  `join` takes an open seat: the server checks the pop (over this
  connection's nonce), that the key holds no seat here, and that neither key
  nor address is banned, then sends `joined` on that connection's chain,
  which from then on is the seat's chain. `seat_bot` puts a server bot in an
  open seat. A seat whose connection drops is held for `seat_grace_s` (it
  can `rejoin`), then freed; a freed key can sit again only by a fresh join,
  signed over a fresh nonce.
* **revealing.** When no seat is open the server builds the one `commitments`
  body and sends it, once, to every seat, every spectator and every
  connection still looking at the room (which become spectators: the game is
  all there is left to see; one that then rejoins a seat stops being one, so
  that its connection carries only the seat's chain). Each remote seat
  answers with its `reveal`; each reveal gets a `received` (an accepted
  reveal resent after a reconnect is not charged against the seat's
  allowance). With no remote seat the game starts at once.
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
  server failure also end a room as `aborted`, without blame; its `game_id`
  says whether the game had started.) `after` and `aborted` each end a chain:
  only receipts and errors may follow them.

What each party keeps:

* **Server**, per room: its seed; the commitments; per seat the key,
  commitment, join nonce, pop, seed and reveal signature; every outbox
  (chain); the salts and actions. Server-wide: blamed keys and addresses (an
  hour), the pace book of its bots.
* **Python client** (`GameClient`): its key, its one seed and commitment, the
  nonce it joined with, the chain it received, the commitments (the first,
  the only), its reveal, what it signed (`sent`, with the message it was sent
  after), answers and the reveal not yet receipted, the decisions it was
  timed out of (`unanswered`: index, decide seq, how long it had). The
  transcript is the chain plus `public_key`, `client_seed`, `reveal_sig`,
  `sent`, `unanswered`; it is saved when the game ends, and also whenever
  `play()` raises.
* **Browser**: the same, in a seat record in `localStorage` keyed by the
  server's key and the room (`catanatron-seat-<server key>-<room>`: key
  pair, seed, commitment, pop, join nonce, requested seat, the server's seed
  commitment, seat, the reveal and the digest it signed, `sent`, unreceipted
  answers, `unanswered`, and whether the game ended), written before the join
  goes out; finished transcripts in IndexedDB, only ever replaced by one that
  extends them. A record whose game ended is history: the page never sits in
  it again (it watches instead).

## 3. Invariants

Each invariant names what the clients check live, what the verifier checks
afterwards, and the tests (`tests/server/...`) that attack it.

**I1. One reveal per room, one seed per join, one game per reveal.** A seat
signs exactly one reveal per room and server key: for the first
`commitments` it receives, with the seed it joined with. It never signs a
reveal for another commitments body, never plays or signs a move under any
other commitments, never reveals into a room again after a void, and never
uses a seed for a second room.
* Clients: `GameClient` makes its seed when it is created and keeps the first
  commitments; a different second one raises `ServerMisbehaved` (an identical
  copy changes nothing and is ignored); after a reconnect it resends its
  stored reveal while it has no receipt, never a fresh signature; once the
  seats are locked, a server that says it holds no seat for its key is
  misbehaving (it never starts over then). The browser stores the reveal and
  its digest in the seat record, refuses to sign for another digest, plays
  only a `before` whose commitments it accepted on that chain and revealed
  for, signs moves only under that digest, and never sits again in a record
  whose game ended; each `takeSeat` makes a fresh key and seed.
* Verifier: I2, and every reveal signature in `before` verifies over the one
  commitments' digest.
* Tests: `test_protocol4.py::test_a_second_commitments_is_never_revealed_into`,
  `test_browser.py::test_a_regrinding_server_is_refused_by_the_page`,
  `test_browser.py::test_after_a_reload_the_page_plays_only_the_game_it_revealed_for`,
  `test_browser.py::test_a_void_ends_the_page_s_chain_and_its_seat`,
  `test_round5.py::test_a_client_keeps_its_evidence_when_the_server_forgets_its_seat`,
  `test_review.py::test_a_seat_that_does_not_reveal_voids_the_room_and_is_blamed`.

**I2. The schedule of a chain.** A chain starts with its `hello`, holds at
most one `commitments`, one `before` and one end (`after` or `aborted`), keeps
pre-game messages before the game and game messages in it, has nothing but
receipts and errors after its end, and no message type outside the table
above (protocol 3's `restart` included). This holds over the whole chain,
whatever a void says.
* Clients: as I1; both stop taking game messages after an end (the browser
  reports anything else).
* Verifier: all of it, over every message.
* Tests: `test_protocol4.py::test_a_transcript_with_two_commitments_fails`,
  `test_protocol4.py::test_a_restart_message_fails`,
  `test_round5.py::test_messages_after_a_void_are_violations`,
  `test_round5.py::test_a_void_after_the_game_hides_nothing`,
  `test_round5.py::test_a_rigged_game_after_a_void_is_not_a_void`.

**I3. Everything signed names C, and C names the server.** Reveals sign `C`;
action statements name `C`, the move's index, and the `decide` they answer
(its `seq` and hash); `C` names the server's key, so nothing signed for one
server key stands for another.
* Clients: sign only over `H` of the commitments they kept, whose
  `server_key` is the one their chain is signed with.
* Verifier: the commitments name the transcript's server key; reveal
  signatures over `C`; every played client move's statement equals
  `action_statement(C, index, seq, decide, action)`; this client's moves
  answer the decide it got at that point.
* Tests: `test_review2.py::test_a_players_signatures_cannot_be_carried_into_another_game`,
  `test_review.py::test_a_signature_cannot_be_replayed_for_another_move`,
  `test_round5.py::test_one_room_cannot_be_served_under_two_keys`.

**I4. One seed commitment per room.** `hello`, `room`, `joined`'s room,
`commitments` and `before` carry the same `seed_commitment` (and the same
room id and rules), and the seed revealed by `after` or `aborted` opens it.
A void that reveals no seed is a violation.
* Clients: `GameClient` checks every room description, the commitments,
  `before` and `aborted` (`ServerMisbehaved`); the browser checks them all
  against the seed commitment it saw first (kept in the seat record across
  reloads) and shows that value.
* Verifier: the same, for every message.
* Tests: `test_protocol4.py::test_a_void_must_open_the_seed_commitment`,
  `test_round5.py::test_a_void_without_the_server_seed_is_refused`,
  `test_round5.py::test_the_python_client_refuses_what_the_verifier_would`,
  `test_browser.py::test_the_page_checks_its_own_seat_and_every_room_description`.

**I5. The game is the committed one.** `before.commitments` is, byte for
byte, the commitments body this recipient got; its config and seats are
those; every remote seed opens its seat's commitment; this client's seed is
its own; the game replays from the master seed of all of it. A game called
off after it started still has to start as the void's revealed seed says.
* Clients: `GameClient` compares canonical bytes (never Python value
  equality, under which 30000 == 30000.0) and checks its seed; the browser
  too.
* Verifier: all of it, then the replay; with only a void, the game's start.
* Tests: `test_review.py::test_seeds_that_do_not_open_their_commitments_are_caught`,
  `test_review.py::test_a_game_other_than_the_committed_one_is_caught`,
  `test_verify.py::test_a_game_not_seeded_from_the_clients_seeds_is_caught`,
  `test_round5.py::test_the_python_client_refuses_what_the_verifier_would`,
  `test_round5.py::test_a_void_after_the_start_checks_the_start`.

**I6. Seated as joined, by its own join.** `joined` names this client's key,
commitment and the seat it asked for (if it asked); the commitments seat
that key once, as a remote seat, at that seat, with that commitment, its
requested seat, the nonce of the connection it joined on and a pop over all
of it. A pop is public, but it seats its key on one connection only:
replaying it seats nothing.
* Clients: both check `joined` (one, for their key, commitment and seat),
  their own seat in the commitments in full (kind, nonce, requested seat),
  and, before revealing (a reveal signs every seat), every seat: its exact
  fields, a unique key, and a pop that verifies.
* Verifier: the same, plus every seat's exact fields and pop, and
  `requested_seat` equal to the seat when given.
* Tests: `test_review.py::test_a_join_needs_proof_of_the_key_and_a_key_not_already_seated`,
  `test_round5.py::test_the_python_client_checks_its_own_seat_fully`,
  `test_round5.py::test_the_python_client_checks_every_seat_before_revealing`,
  `test_round5.py::test_the_seat_a_client_asked_for_is_the_seat_it_gets`,
  `test_round5.py::test_a_public_join_cannot_be_replayed_to_reseat_a_freed_key`,
  `test_browser.py::test_the_page_checks_its_own_seat_and_every_room_description`.

**I7. A void is announced, signed and blamed, and blames honestly what it
can be held to.** A room whose remote seats do not all reveal in time ends
with one `aborted` to every chain, naming the seats that did not reveal and
revealing the server seed. The server answers every reveal with a receipt.
It never blames a seat whose reveal it acknowledged as accepted, whose seed
and reveal its own `before` carries, or that it never sent the commitments.
* Clients: `GameClient.play` returns with `GameResult.aborted` (and verifies
  the transcript); the browser shows the void and offers a new room.
* Verifier: `report.void`; the seed opens the commitment; the blamed seats
  are remote seats; a blame on a seat whose seed and reveal the chain's
  `before` holds is an error in any transcript; in the blamed seat's own
  transcript, a blame without commitments before it, or after an accepted
  receipt for its reveal, is an error; otherwise a warning.
* Tests: `test_review.py::test_a_seat_that_does_not_reveal_voids_the_room_and_is_blamed`,
  `test_protocol4.py::test_blaming_a_seat_whose_reveal_was_acknowledged_is_caught`,
  `test_round5.py::test_a_blame_the_servers_own_before_belies_is_caught_by_everyone`,
  `test_round5.py::test_blaming_a_seat_never_asked_for_its_seed_is_caught`.

**I8. One chain per recipient, and the transcript is its own.** Every
message is signed by `S`, numbered from 0 and hash-chained, and addressed to
the seat's key once it is seated; every number in it is an integer every
client represents exactly. The transcript's header (its key, seed and room)
agrees with what the server signed.
* Clients: checked on every message as it arrives; the browser refuses a
  float or an unsafe integer before parsing (JavaScript would read 5.0 as 5,
  Python not).
* Verifier: a broken signature or chain, or a header the chain belies, makes
  the transcript **invalid** (not evidence: anyone can write one), never an
  accusation; a number the server signed but the protocol does not allow is
  an error.
* Tests: `test_verify.py::test_an_edited_transcript_fails_its_signatures`,
  `test_round5.py::test_a_header_that_belies_the_chain_is_not_an_accusation`,
  `test_round5.py::test_a_float_the_server_signed_is_a_violation`,
  `test_round5.py::test_numbers_beyond_every_clients_reach_are_refused`,
  `test_browser.py::test_the_page_refuses_numbers_python_reads_otherwise`.

**I9. A seat's chain is taken back only by its key, on that connection, and
is the only chain it carries.** A `rejoin` must sign the nonce of the
connection it is sent on with the seat's key; the server then closes the
seat's previous connection, stops treating the new one as a spectator, and
resends the seat's chain after `last_seq`. A client whose `joined` was lost
still knows its key, so it takes its seat back; nobody else can. A resent
chain that does not continue the client's is refused, not skipped.
* Clients: `GameClient` rejoins whenever its join went out; on `not_seated`
  before any commitments it joins afresh; a message for its key that does not
  continue its chain is `ServerMisbehaved`. The browser writes its seat
  record before the join goes out and does the same.
* Verifier: not concerned (a rejoin's own chain is not part of a transcript).
* Tests: `test_protocol4.py::test_a_client_that_loses_joined_takes_its_seat_back`,
  `test_protocol4.py::test_a_rejoin_signature_is_good_on_one_connection_only`,
  `test_review.py::test_a_rejoin_closes_the_old_connection`,
  `test_browser.py::test_a_page_that_loses_joined_takes_its_seat_back`,
  `test_round5.py::test_a_rejoin_racing_the_lock_carries_one_chain`,
  `test_round5.py::test_a_resent_chain_that_is_not_ours_is_refused`.

**I10. Every view is the honest redaction of one committed game.** Each
body is exactly what an honest server builds from the replayed state for its
recipient, and each state commitment opens with its revealed salt.
* Verifier: the replay. Tests: `test_verify.py` (tampered views, altered
  dice), `test_review.py::test_moves_not_offered_by_the_game_fail_verification`,
  `test_round5.py::test_a_void_after_the_game_hides_nothing`.

**I11. The game's schedule.** One `before`; one `step` per action, in order;
a `decide` exactly when this seat had a move that was not publicly forced,
at most one per move; one `after`.
* Verifier. Tests: `test_review.py::test_the_message_schedule_is_checked`,
  `test_review2.py::test_forced_moves_follow_what_the_table_can_see`.

**I12. Every move is authorised.** A remote seat's move is its own signed
statement; a server move is publicly forced (`only`, the only legal action)
or flagged `timeout`/`bot_error` and the default action; a server bot's move
is the server's; a seat is never timed out of a move it was not asked.
* Verifier. Tests: `test_verify.py::test_a_forged_move_is_caught_by_everyone`.

**I13. Every submission is answered.** Every `reveal` while seeds are being
revealed, and every `action` from a seat once the game exists, gets exactly
one signed `received` naming it, or the connection is closed (code 4009,
past the seat's allowance of `receipt_margin + receipts_per_decision ×
decisions asked`, which an honest client never reaches: an accepted reveal
resent after a reconnect is not charged). During the game a seat's action is
never answered with an `error`. Receipts do not count against the error
allowance (`max_answers`). An `action` before the game is answered with an
`error` (`no_game`), and so is a `reveal` outside `revealing`; what a
spectator sends is answered however the server likes, and proves nothing.
* Clients: `GameClient` resends every answer (and its reveal) not yet
  receipted after a reconnect, and waits `RECEIPT_WAIT_S` at the end for
  receipts still owed; the browser keeps unreceipted answers in its seat
  record (a reload keeps them) and reconnects after the end while it is owed
  receipts. A join refusal is a refusal only while a join is outstanding.
* Verifier: in a seated transcript, an `error` answering an `action` between
  the join and the chain's end is an error; receipts or errors before the
  game are ignored.
* Tests: `test_protocol4.py::test_every_submission_gets_a_receipt_or_a_closed_connection`,
  `test_protocol4.py::test_receipts_do_not_use_up_the_error_allowance`,
  `test_protocol4.py::test_an_action_before_the_game_gets_an_error_and_verifies`,
  `test_review2.py::test_a_legal_move_the_server_overrules_is_reported`,
  `test_round5.py::test_a_spectator_cannot_make_an_honest_server_fail`,
  `test_round5.py::test_an_accepted_reveal_resent_on_every_reconnect_is_not_charged`,
  `test_round5.py::test_a_join_refusal_mid_game_does_not_end_the_client`.

**I14. Deadlines are atomic.** A decision closes (its deadline passes, or an
answer is accepted) and its `step` is pushed with no receipt naming that
decision in between: a submission that arrives in that window gets its
receipt after the step. So a receipt before step `k` means the server had
the submission while decision `k` was open.
* Clients: both keep a record of every decision closed before they answered
  (`unanswered`).
* Verifier, from the signed decides, receipts and steps alone (so in a void
  or unfinished game too): a legal, correctly signed answer of this client to
  decide `k` that step `k` did not play, with a receipt before step `k` or a
  receipt saying `accepted` anywhere, is an error; every move played for this
  seat as a timeout is a warning (with how long the client says it had).
* Tests: `test_protocol4.py::test_answers_at_the_deadline_never_frame_an_honest_server`,
  `test_protocol4.py::test_an_accepted_answer_the_server_did_not_play_is_caught`,
  `test_review2.py::test_a_legal_move_the_server_overrules_is_reported`,
  `test_round5.py::test_a_cheat_is_caught_though_the_game_is_then_void`,
  `test_round5.py::test_coercion_to_the_default_leaves_a_trace`.

**I15. No equivocation within a room.** Any two transcripts of one room
under one server key agree on the seed commitment, the commitments, the game
(id, seeds, reveals), its reveal or its void (which names its game, so a
connection that only heard the void agrees with one that saw the game), and
the state commitment at each index. Every pair is compared, so the verdict
does not depend on the order the transcripts are given in. Two server keys
for one room id are a warning only: anyone can run a server with any room
id, and since `C` names the key, nothing one key's room holds can stand in
the other's.
* Verifier: `verify_transcripts` (and `catanatron-verify` with several files).
* Tests: `test_review2.py::test_a_server_naming_different_games_to_different_recipients_is_caught`,
  `test_protocol4.py::test_every_pair_of_transcripts_of_a_room_is_compared`,
  `test_protocol4.py::test_a_void_for_some_and_a_game_for_others_is_caught`,
  `test_round5.py::test_a_connection_that_only_heard_the_void_agrees_with_the_seat`,
  `test_round5.py::test_recipients_of_a_game_voided_midway_are_still_compared`,
  `test_round5.py::test_one_room_cannot_be_served_under_two_keys`.

Also kept, though not a fairness invariant: the room pushes `room` only when
its seats change, so connections coming and going cannot make every chain
grow (`test_round5.py::test_connections_coming_and_going_do_not_grow_the_chains`).

## 4. What is provable, and to whom

The verifier's report sorts everything it finds:

* **errors** — provable to anyone: whatever contradicts the server's own
  signatures in a transcript that is valid (I8): a broken invariant I2–I7 or
  I10–I15 inside one transcript, or I15 across two. In particular: a second
  commitments; a message after the chain's end; a game other than the
  committed one; a view that is not the honest redaction; a move nobody
  authorised; a seat timed out of a move it was not asked; a void whose
  seed does not open the commitment, or that blames a seat the server's own
  messages show revealed (or never asked); an action answered by an error
  during the game; a legal signed answer the server acknowledged in time (or
  as accepted) and did not play; a number the protocol does not allow; two
  recipients of one room told different things. These are checked over
  everything the transcript holds, finished or not, void or not.
* **invalid** — not evidence: a broken signature or chain, a malformed file,
  a header (key, seed, room) that the signed chain belies. Anyone can write
  one; it accuses nobody. (The live client knows better: it raises
  `ServerMisbehaved` the moment a message fails a check, which is why it
  checks as it plays.)
* **warnings** — what rests on the client's word, because the client writes
  `sent` and `unanswered` itself and decides what it sends: that it sent an
  answer the server never acknowledged; that an answer was late, or on time
  although the server timed it out (deadlines are the server's word; every
  timeout of this seat is listed); that it revealed although the server
  blamed it and never acknowledged the reveal; that the server went silent
  or closed connections; that another server key uses this room id.
* **void** — the room was called off (its signed `aborted`): not a game, not
  an accusation by itself.
* **incomplete** — the transcript stops before its chain's end: a missing
  tail is not evidence (anyone can cut a transcript), but everything it does
  hold is checked.

`ok` means a finished game with no errors and a valid transcript; the
command line prints `OK`, `FAILED` (errors), `INVALID`, `VOID` or
`INCOMPLETE`.

Tests that a client or spectator cannot turn the unprovable into an
accusation: `test_protocol4.py::test_a_client_cannot_frame_an_honest_server`
(an answer listed in `sent` but never sent; eight badly signed copies before
the real one), `test_review2.py::test_an_honest_server_is_not_accused`
(drops, lost and late answers, a badly signed copy first),
`test_review2.py::test_an_unacknowledged_move_is_the_clients_word_only`,
`test_round5.py::test_a_header_that_belies_the_chain_is_not_an_accusation`,
`test_round5.py::test_a_spectator_cannot_make_an_honest_server_fail`,
`test_verify.py::test_an_unfinished_transcript_cannot_be_verified_yet`.

## 5. What the server can still do

* **Void a game it dislikes.** The server learns the outcome as soon as every
  seed is in; a seat it controls can then withhold its reveal, voiding the
  room (or the server can call it off). Every such reroll is a visible void,
  naming the withholder, and the next game needs a new room with new
  commitments and new seeds from every seat (clients never carry a seed or a
  seat record into it). Rerolling is therefore exactly as costly to the
  server as cancelling and recreating rooms, which the threat model already
  admits (selective abort): treat a server whose rooms are often void as
  dishonest.
* **Go silent.** A server can stop sending a client its chain; the client's
  transcript then ends early (incomplete), and what it holds still counts.
  Clients save what they hold whenever they stop.
* Everything the README's threat model lists: it knows all hidden state
  during the game, any seat may be its puppet, deadlines and timing are its
  word, its own bots are its own.
