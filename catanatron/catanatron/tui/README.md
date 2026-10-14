# catanatron-tui

A terminal UI for playing, watching and debugging Catanatron games: you
against registry bots (or your own), hot seat, bots against bots, on any map
shape with 2–6 players, with debugging controls (undo/redo, forced dice and
cards, every hand face up, saved positions). It also plays a seat in a
[catanatron-server](../server/README.md) room, or watches one, and verifies
the game's transcript at the end.

```bash
pip install -e "catanatron[tui]"            # Textual
pip install -e "catanatron[tui,server]"     # and remote games
```

```
 ⭘                                                 catanatron-tui — RED to play (human)
╭──────────────────────────────────────────────────────────────────────────╮ turn 48  RED (human): build, trade, play a card or end turn
│                                50______51                                │ last roll 4+6=10  ·  25 development cards left
│                               /          \                               │ viewing as RED  ·  next roll 6+6  ·  bot delay 0s
│                        3:1   /    #17     \   2:1 sh                     │ the next roll will be 6+6
│                    48______49    desert    52______53                    │
│                   /          \            /          \                   │   seat        VP wd br sh wh or    dev kn rd  r/s/c
│                  /    #16     \          /    #18     \                  │   Blue F       3 8 cards           0    0  4  9/4/3
│        45______46   sheep 6    22______23    ore 2     24______25        │   White F      3 11 cards          0    0  3 11/4/3
│   3:1 /          \   •••••    /          \     •      /          \ 2:1 or│   Orange AB:2  5 7 cards           0    0 5L  9/4/3
│      /    #15     \          /     #6     \          /     #7     \      │ ▶ Red human    2  0  0  1  7  1 =9 -    0  1 13/3/4
│    47   wood 3     19______20   wheat 9     6______7    sheep 5    26    │   bank           15 15 12  8 10    25
│      \     ••     /          \    ••••    /          \    ••••    /      │
│       \          /     #5     \          /     #1     \          /       │╭─ RED to play ────────────────────────────────────────────────╮
│        43______21   wheat 4     0______1    sheep 10    8______27        ││ Turn                                                         │
│       /          \    •••     /          \    •••     /          \       ││   End turn                                                   │
│      /    #14     \          /     #0     \          /     #8     \      ││ Development cards                                            │
│    44    ore 8     16______5    wheat 11    2______9    wood 8     28    ││   Buy a development card                                     │
│      \   •••••    /          \     ••     /          \   •••••    /      ││ Trade                                                        │
│2:1 wh \  ROBBER  /     #4     \          /     #2     \          / 2:1 br││   Trade 4 wheat for 1 brick                                  │
│        40______18    ore 5      4______3    wheat 3    10______29        ││   Trade 4 wheat for 1 ore                                    │
│       /          \    ••••    /          \     ••     /          \       ││   Trade 4 wheat for 1 sheep                                  │
│      /    #13     \          /     #3     \          /     #9     \      ││   Trade 4 wheat for 1 wood                                   │
│    42   brick 10   17______15   brick 6    12______11   wood 4     30    ││                                                              │
│      \    •••     /          \   •••••    /          \    •••     /      ││                                                              │
│       \          /    #12     \          /    #10     \          /       ││                                                              │
│        41______39   sheep 9    14______13   wood 11    32______31        ││                                                              │
│                  \    ••••    /          \     ••     /                  │╰──────────────────────────────────────────────────────────────╯
│               3:1 \          /    #11     \          / 2:1 wd            │╭──────────────────────────────────────────────────────────────╮
│                    38______37   brick 12   34______33                    ││   43 ORANGE rolled 6+2 = 8                                   │
│                              \     •      /                              ││      ORANGE ended the turn                                   │
│                               \          /                               ││   44 RED rolled 4+2 = 6                                      │
│                                36______35                                ││      RED ended the turn                                      │
│                                    3:1                                   ││   45 BLUE rolled 6+5 = 11                                    │
│                                                                          ││      BLUE built a road 27-28                                 │
│                                                                          ││      BLUE ended the turn                                     │
│                                                                          ││   46 WHITE rolled 1+3 = 4                                    │
│                                                                          ││      WHITE ended the turn                                    │
│                                                                          ││   47 ORANGE rolled 4+5 = 9                                   │
│                                                                          ││      ORANGE ended the turn                                   │
│                                                                          ││   48 RED rolled 4+6 = 10                                     │
│                                                                          ││      RED traded 4 brick for 1 sheep                          │
╰──────────────────────────────────────────────────────────────────────────╯╰──────────────────────────────────────────────────────────────╯
 ? Help  : Command  b Board  space Pause/Play  n Step  u Undo  r Redo  g God view  d Dice  s Save  z Zoom  q Quit
```

(In a terminal the pieces are in their owners' colors: settlements are the
node id in the color, cities the id in reverse video, roads the colored sides;
the legal targets of the move being chosen are yellow, and the last move is
underlined.)

## Local games

```bash
catanatron-tui                                    # you (RED) against three F bots
catanatron-tui --seats human,AB:2,F,R --vps 8     # one spec per seat, in color order
catanatron-tui --seats human,human,F              # hot seat
catanatron-tui --seats F,AB:2,F,F                 # watch bots (every hand is shown)
catanatron-tui --map-type MINI --seats human,F
catanatron-tui --map tests/maps/extension.json --seats human,F,F,F,F,F
catanatron-tui --load game.json                   # resume a saved game or position
catanatron-tui --seats human,F --run 'dice 6 6'   # commands to run at start
```

Seats are listed in color order: RED, BLUE, ORANGE, WHITE, GREEN, BROWN. Turn
order is drawn at random, as in any Catanatron game. A seat is one of

| spec | who plays it |
|---|---|
| `human` (or `H`) | a person at this terminal; several make a hot-seat game |
| a registry spec: `F`, `AB:2`, `R`, `M:500`, … | that bot, **seeing only its seat's view** (below) |
| `oracle:SPEC` | that bot on a copy of the true game: every hand and the deck's order |
| a name from `decide_fns` (Python), `oracle:NAME` | your own bot (below) |

Bots registered from files work as in `catanatron-play`:
`--bot MYBOT=./mybots.py#MyPlayer --seats human,MYBOT`.

The rules: `--map FILE` (an explicit map spec, see the server README; the
files in `tests/maps/` and catanrl's map generators produce them), `--map-type
BASE|MINI|TOURNAMENT`, `--number-placement`, `--vps`, `--discard-limit`,
`--friendly-robber`, `--supply base|extension` (the 5–6 player bank and deck
are the default for 5–6 players), or all of them as `--config JSON` with the
keys of a catanatron-server room config. `--seed N` fixes the board and every
roll and draw.

### What bots see

A bot decides from its seat's view only, exactly as on catanatron-server: the
seat's `client_view` (its own hand; the others' hand sizes, played cards and
public points; the deck's unseen pool, not its composition or order), turned
into a full game for the bot by the server's
[`ViewBot`](../server/bot.py) and [`sample_game`](../server/sampling.py):
the other hands are dealt at random consistently with what the seat knows,
and the deck order and random stream are fresh, so lookahead sees neither
hidden cards nor future rolls. `oracle:SPEC` is the comparison point: the
bot decides on a copy of the real game (all hands, the deck's order), with
its own random stream, so it cannot see or disturb the real dice.

### Your own bots

```python
from catanatron.tui import run

def my_bot(view, playable_actions):
    # view: the seat's client_view document (with "map" and "action_records");
    # playable_actions: [color, action_type, value] triples, in the order the
    # server offers them. Return one of them, or an Action.
    return playable_actions[0]

run(["human", "F", "mine"], decide_fns={"mine": my_bot})
```

This is the callback `catanatron.server.client.GameClient.play` takes, so a
bot written for the server plays here unchanged (async callbacks too). A
`decide_fns` value may also be a Catanatron `Player`: it then decides on a
game sampled from its view, like registry bots, or on the true game as
`oracle:NAME`. An oracle callback gets `client_view` with every hand face up.

`run()` takes the rules as `config=` (a dict of room-config keys or a
`GameSettings`), `seed=`, `load=` (a saved file), `speed=`, `autoplay=` and
`commands=`, and returns the session when the app exits (`.game` is the
Catanatron game). Without a terminal, `catanatron.tui.local.LocalSession` is
the same game for scripts and tests: `.run()` plays bots until a person must
choose, `.frame().actions` lists the choices, `.choose(action)` plays one.

### Debugging controls

| key | command | |
|---|---|---|
| `space` | `auto on\|off` | pause / resume the bots |
| `n` | `step [N]` | one bot move (N moves) |
| `+` `-` | `speed SECONDS` | delay between bot moves |
| `u` / `r` | `undo [N]` / `redo [N]` | take back / replay actions (pauses the bots) |
| `U` | `back` | undo to the last time a person had a choice |
| `g` | `god` | god view: every hand, the deck's next card |
| `v` | `view follow\|god\|public\|COLOR` | whose view is shown (follow: the person to move) |
| `d` | `dice A B` / `dice off` | the next roll |
| `c` | `card NAME` | the next development card drawn (`knight`, `yop`, `mono`, `rb`, `vp`) |
| `t` | `steal RES` | what the next robber steals (if the victim has it) |
| | `seat COLOR SPEC` | who plays a seat from now on (take over a bot, hand yours over) |
| | `hand COLOR wood=2 ore=1` | set a hand (the bank pays the difference) |
| | `robber TILE_ID` | move the robber (no steal) |
| `s` / `l` | `save PATH` / `load PATH` | the game as JSON (below) |
| | `new [SEED]` | a new game with the same rules and seats |

Forced outcomes apply to the next roll, purchase or steal by anyone and are
noted in the log. Moves with forced outcomes and edits go through the undo
history like any move; `load` and `new` replace the game and its history.

**Files.** `save` writes one JSON file that is both a position and a full
game log: `state` (Catanatron's `state_to_json`, everything including the
deck order and the random stream), `initial` (where the session started),
the seats and the log notes. The actions between the two are
`state["action_records"]` with their outcomes, and `load` replays them, so the
whole game can be stepped back through with `u`/`r`. `--load` also accepts a
bare `state_to_json` document and a catanatron-server archive record (its
`final_state`). Bots resume from the saved random stream.

## Remote games

```bash
catanatron-server --port 8765 &
# create a room: you, a server F bot, an open seat for someone else, a server AB:2
catanatron-tui --connect ws://localhost:8765 --new --seats human,F,open,AB:2 --vps 8
# join an open seat of an existing room (or --sit COLOR|INDEX for a given one)
catanatron-tui --connect ws://localhost:8765/room/ROOM_ID
# watch
catanatron-tui --connect ws://localhost:8765/room/ROOM_ID --spectate
```

`--connect` with a server URL and no `--new` lists the rooms. The TUI is a
`GameClient`: it checks every message's signature and hash chain as it
arrives, contributes its seed, signs your moves, saves the transcript to
`--transcripts` (default `./transcripts/`) when the game ends and verifies
it (`--no-verify` skips that; `V` or `:verify` runs it again). The result
shows in the status line, and `catanatron-verify` can check the file later.
The server holds the game, so there are no debugging controls; the decision
countdown shows next to the room id. `--server-key` pins the server's
identity and `--key FILE` keeps your signing key.

## Choosing moves

The legal moves are listed by kind (turn, build, development cards, trade,
robber, discard); arrows and `enter` play one, `/TEXT` filters the list. On
the board, click a node, road or tile (or press `b`, move with the arrows
and press `enter`) to build there or move the robber there; if a robber move
has several victims, the list narrows to them. Clicking anything else says
what is there (a node's production and port, a building's owner), and so
does hovering. A seat's only legal move (a lone roll, say) is played for it.

## The board

Each land tile is a flat-top hexagon: the board is turned 30° from
Catanatron's pointy-top layout, which lets the sides be drawn with `_`, `/`
and `\`; node, edge and tile ids are Catanatron's. Tiles are placed from
their cube coordinates, so any map shape draws the same way. `z` cycles the
hex size (`auto` picks the largest that fits; small boards abbreviate
resources), `i` hides empty nodes' ids, and boards larger than the window
scroll. The 30-tile 5–6 player board in a 140×44 terminal:

```
╭──────────────────────────────────────────────────────────────────────────╮ turn 31  ORANGE (F): discard 3 cards
│                             2:1 wd                                       │ last roll 3+4=7  ·  33 development cards left (next: knight)
│                            10___11                                       │ god view  ·  bots paused (space resumes, n steps)
│                           /  #2   \                                      │
│                     6___7  sh 5    12___22 2:1 br                        │
│            2:1 sh /  #1   \ ••••  /  #6   \                              │   seat     VP wd br sh wh or    dev kn rd  r/s/c
│             0___1  sh 11    8___13 wd 5    23___35                       │   Blue F    4  1  1  1  1  0 =4 -    1  2 12/5/2
│           /  #0   \  ••   /  #5   \ ••••  / #11   \                      │ ▶ Orange F  4  0  1  0  2  5 =8 -    0  2 12/3/3
│         5  or 11    2___9  wd 2    20___24 desert  36___50 3:1           │   White F   2  3  0  3  1  0 =7 -    0  2 12/3/4
│           \  ••   /  #4   \  •    / #10   \       / #17   \              │   Red F     2  2  0  1  0  6 =9 -    0  1 13/3/4
│       3:1   4___3  or 6    18___21 br 8    33___37 br 8    51            │   Green F   2  2  0  3  2  1 =8 -    0  2 12/3/4
│           /  #3   \ROBBER /  #9   \•••••  / #16   \•••••  /              │   Brown F   2  0  0  1  1  1 =3 -    0  3 11/3/4
│         17 br 12   14___19 wh 10   31___34 wh 9    48___52               │   bank        16 22 15 17 11    33
│           \  •    /  #8   \ •••   / #15   \ ••••  / #22   \              │
│            16___15 br 10   29___32 wh 9    46___49 wd 3    62            │
│           /  #7   \ •••   / #14   \ ••••  / #21   \  ••   /              │
│         28 desert  25___30 wd 6    44___47 sh 4    60___63 3:1           │
│           \       / #13   \•••••  / #20   \ •••   / #26   \              │
│       3:1  27___26 sh 11   42___45 sh 9    58___61 or 8    71            │
│           / #12   \  ••   / #19   \ ••••  / #25   \•••••  /              │
│         41 wh 10   38___43 wh 12   56___59 or 5    69___72               │
│           \ •••   / #18   \  •    / #24   \ ••••  / #29   \              │
│            40___39 sh 3    53___57 wd 2    67___70 wd 4    78            │
│              3:1  \  ••   / #23   \  •    / #28   \ •••   /              │
│                    55___54 br 3    64___68 wh 6    76___79 2:1 wh        │
│                           \  ••   / #27   \•••••  /                      │
│                            66___65 or 4    73___77                       │
│                            2:1 or \ •••   / 2:1 sh                       │
│                                    75___74                               │
```

The panels: VP shows public points, and actual points in parentheses when
you may see them; the hand is per resource for hands you may see (yours, or
all in god view) and a count otherwise, likewise development cards (`1K 2V`:
one knight, two victory points); `kn` is knights played (`A`: largest army),
`rd` the longest road (`L`: holds it), `r/s/c` the roads, settlements and
cities left; the last row is the bank and the development deck.

## Layout

| module | |
|---|---|
| `board.py` | the board as text, for any map; where each node, edge and tile is drawn |
| `actions.py` | moves and log entries in words; the action groups |
| `panels.py` | status, players and log panels, from a state document |
| `session.py` | the interface the app talks to (`Frame`, `Session`) |
| `local.py` | `LocalSession`: the in-process game, history, forced chance, edits |
| `agents.py` | who plays a local seat, and what each agent is handed |
| `positions.py` | rules, new games, the file format, replaying a saved log |
| `remote.py` | `RemoteSession`: a `GameClient` seat or spectator |
| `app.py` | the Textual app |
| `cli.py` | `catanatron-tui` |

Tests: `pytest tests/tui` (headless, with Textual's Pilot; the remote tests
start a real server on localhost).

## Limitations

* Domestic (player-to-player) trades are not offered: Catanatron's bots do
  not make them and its move generator does not list them.
* Hot seat shows the hand of whoever is to move; there is no "pass the
  device" screen between people.
* Bots sample hidden cards uniformly from what is consistent with the view
  (no card tracking), as on the server.
* A bot that is thinking cannot be interrupted; an undo meanwhile simply
  drops its answer.
* The board is drawn 30° turned from Catanatron's web UI.
