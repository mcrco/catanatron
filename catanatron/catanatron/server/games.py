"""The server's record of finished games: an index over the archive, by the
names people sat under, a replay rebuilt from each game's record, and the
analyses the house sends back.

The archive (``--archive DIR``) keeps one JSON record per finished game
(``GameServer._archive``): the room as it was described, the reveal and the
game's final document. :class:`GameStore` indexes those files in an SQLite
database next to them (``DIR/games.sqlite``), built from the files on
startup and kept current as games finish, so the files stay the record and
the database can always be thrown away and rebuilt.

Names are labels, not accounts: a seat's name is what its person typed (the
key it signed with is new for every room), and :func:`name_key` folds case
and spacing so that ``Marco`` and ``marco`` list together. Seats with no
name, or named after their color (the page's default), are not listed under
any name. House seats (the house's bots, ``house_seats`` in the room) are
listed by their model name like anyone else, but never count as people.

A replay is the game's whole document at every step, every hand face up
(the game is over, and its reveal makes everything public already):
:func:`rebuild` replays the final document's own log on a fresh game with
every chance outcome forced from the log, and checks that it ends where the
record does. Analyses come from the house (``submit_analysis`` on the lobby,
see PROTOCOL.md); the server only stores them and serves them with the
replay.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import re
import sqlite3
import threading
import time
from typing import Any, Iterable, Optional

from catanatron.game import Game
from catanatron.models.actions import generate_playable_actions
from catanatron.models.enums import DEVELOPMENT_CARDS, RESOURCES
from catanatron.models.player import Color
from catanatron.serialization import (
    action_record_from_json,
    action_to_json,
    client_view,
    map_from_json,
    state_to_json,
)
from catanatron.server.config import GameConfig, ServerSeat
from catanatron.server.maps import board_geometry

logger = logging.getLogger(__name__)

GAME_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
COLOR_NAMES = {c.value.casefold() for c in Color}
#: Analyses are stored as sent, up to this size (the lobby's message limit is 1 MiB).
MAX_ANALYSIS = 900_000
#: An analysis that failed this often is not offered again.
MAX_TRIES = 3
#: Games shorter than this many actions (a game left during the setup) get no analysis.
MIN_ANALYZED_ACTIONS = 20
REPLAY_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    room_id TEXT NOT NULL,
    file TEXT NOT NULL,
    name TEXT,
    finished_at REAL NOT NULL,
    num_players INTEGER NOT NULL,
    config TEXT NOT NULL,
    seats TEXT NOT NULL,
    winner TEXT,
    actions INTEGER NOT NULL,
    turns INTEGER NOT NULL,
    people INTEGER NOT NULL,
    analysis TEXT,
    analysis_tries INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS players (
    game_id TEXT NOT NULL,
    color TEXT NOT NULL,
    name_key TEXT NOT NULL,
    PRIMARY KEY (game_id, color)
);
CREATE INDEX IF NOT EXISTS players_by_name ON players (name_key);
CREATE INDEX IF NOT EXISTS games_by_time ON games (finished_at);
"""


def name_key(name: Any) -> Optional[str]:
    """The key a name lists under, or None for no name (empty, a color, "guest")."""
    if not isinstance(name, str):
        return None
    key = " ".join(name.split()).casefold()
    if not key or key in COLOR_NAMES or key == "guest":
        return None
    return key[:64]


def _winner(final: dict) -> Optional[str]:
    """Catanatron's rule: the seat to move once it has the points."""
    colors = final["colors"]
    index = final["current_player_index"]
    target = final["game"]["vps_to_win"]
    if final["player_state"].get(f"P{index}_ACTUAL_VICTORY_POINTS", 0) >= target:
        return colors[index]
    return None


def summarize(record: dict, house_keys: Iterable[str] = ()) -> dict:
    """What the index keeps of an archive record. ``house_keys``: the
    public keys the house is known to sit with, for records from before
    rooms named their house seats."""
    room, final = record["room"], record["final_state"]
    house_seats = room.get("house_seats")
    known = set(house_keys)
    by_color = {s["color"]: s for s in room["seats"]}
    seats = []
    for index, color in enumerate(final["colors"]):
        seat = by_color.get(color, {})
        if seat.get("kind") == "bot":
            kind = "bot"
        elif house_seats is not None:
            kind = "house" if seat.get("seat") in house_seats else "person"
        else:
            # Older records: a key the house reused, or the name the room is
            # named after ("1v1 vs graph-m"); the house's bots go by it.
            label = (room.get("name") or "").rsplit(" vs ", 1)
            named = len(label) == 2 and seat.get("name") == label[1].strip()
            kind = "house" if seat.get("public_key") in known or named else "person"
        seats.append(
            {
                "color": color,
                "seat": seat.get("seat"),
                "name": seat.get("name") or "",
                "kind": kind,
                "points": final["player_state"].get(f"P{index}_ACTUAL_VICTORY_POINTS", 0),
            }
        )
    return {
        "game_id": room.get("game_id") or final["game"]["id"],
        "room_id": room["id"],
        "name": room.get("name") or "",
        "finished_at": float(record.get("finished_at") or 0.0),
        "num_players": len(final["colors"]),
        "config": room.get("config") or {},
        "seats": seats,
        "winner": _winner(final),
        "actions": len(final["action_records"]),
        "turns": int(final.get("num_turns", 0)),
        "people": sum(1 for s in seats if s["kind"] == "person"),
    }


# ===== replays =====
def rebuild(record: dict, playable: Optional[list] = None) -> list:
    """The game's document before each action and after the last (one more
    than the log has entries), replayed from the final document's log with
    every outcome forced. Raises ValueError when the replay does not end in
    the recorded final state. ``playable``: a list to append the legal
    actions before each logged action to (JSON, Catanatron's order; a trade
    offer is never listed)."""
    final = record["final_state"]
    config = GameConfig.from_json(record["room"]["config"])
    order = [Color[c] for c in final["colors"]]
    # The dice are forced from the log, so a balanced deck (whose shuffle
    # the record does not keep) is not dealt: the documents say what the
    # room played by.
    game = Game(
        [ServerSeat(c) for c in order],
        discard_limit=config.discard_limit,
        friendly_robber=config.friendly_robber,
        vps_to_win=config.vps_to_win,
        catan_map=map_from_json(final["map"]),
        supply=config.game_supply,
        robber_mode=config.game_robber_mode,
    )
    state = game.state
    by_color = {p.color: p for p in state.players}
    state.players = [by_color[c] for c in order]
    state.colors = tuple(order)
    state.color_to_index = {c: i for i, c in enumerate(order)}
    game.playable_actions = generate_playable_actions(state)  # the seat to move changed
    game.id = final["game"]["id"]
    docs = [state_to_json(game)]
    for entry in final["action_records"]:
        if playable is not None:
            playable.append([action_to_json(a) for a in game.playable_actions])
        forced = action_record_from_json(entry)
        game.execute(forced.action, validate_action=False, action_record=forced)
        docs.append(state_to_json(game))
    last = json.loads(json.dumps(docs[-1]))
    for key in ("board", "player_state", "buildings_by_color", "resource_freqdeck", "num_turns"):
        if last[key] != final[key]:
            raise ValueError(f"the replay does not end in the recorded game ({key} differs)")
    for doc in docs:
        doc["game"]["dice_mode"] = config.dice_mode
    return docs


def frame_of(doc: dict) -> dict:
    """A replay frame: the document every hand face up, in the shape the page
    reads (a spectator's ``state``, plus each hand's cards)."""
    view = client_view(doc)
    state = {k: v for k, v in view.items() if k not in ("map", "action_records", "balanced_dice")}
    players = dict(state["player_state"])
    for index in range(len(doc["colors"])):
        prefix = f"P{index}_"
        players[prefix + "NUM_RESOURCES_IN_HAND"] = sum(players[f"{prefix}{r}_IN_HAND"] for r in RESOURCES)
        players[prefix + "NUM_DEVELOPMENT_CARDS_IN_HAND"] = sum(
            players[f"{prefix}{c}_IN_HAND"] for c in DEVELOPMENT_CARDS
        )
    state["player_state"] = players
    state["development_deck_size"] = len(doc["development_listdeck"])
    return state


def replay_of(record: dict) -> dict:
    """Everything the page needs to replay a game: the summary, the board,
    the whole log and a frame per step (frame i is the game before log entry
    i; the last is the end)."""
    docs = rebuild(record)
    final = record["final_state"]
    catan_map = map_from_json(final["map"])
    return {
        "version": REPLAY_VERSION,
        "game": summarize(record),
        "map": final["map"],
        "board": board_geometry(catan_map),
        "log": final["action_records"],
        "frames": [frame_of(doc) for doc in docs],
    }


# ===== the store =====
class GameStore:
    """The index of ``archive_dir`` (see the module doc). Thread-safe; every
    call is short (one SQLite statement or one file)."""

    def __init__(self, archive_dir: str):
        self.dir = archive_dir
        os.makedirs(os.path.join(archive_dir, "analysis"), exist_ok=True)
        os.makedirs(os.path.join(archive_dir, "replays"), exist_ok=True)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(os.path.join(archive_dir, "games.sqlite"), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.scan()

    def close(self) -> None:
        with self.lock:
            self.db.close()

    def _files(self) -> list:
        return sorted(
            name
            for name in os.listdir(self.dir)
            if name.endswith(".json") and os.path.isfile(os.path.join(self.dir, name))
        )

    def _load(self, name: str) -> dict:
        with open(os.path.join(self.dir, name)) as file:
            record = json.load(file)
        if not record.get("finished_at"):
            record["finished_at"] = os.path.getmtime(os.path.join(self.dir, name))
        return record

    def scan(self) -> int:
        """Index every archived game not indexed yet; returns how many. Keys
        that sat in more than one room are the house's (a person's browser
        makes a key per room), for records older than ``house_seats``."""
        with self.lock:
            known = {row[0] for row in self.db.execute("SELECT file FROM games")}
        names = [n for n in self._files() if n not in known]
        if not names:
            return 0
        records = {}
        for name in names:
            try:
                records[name] = self._load(name)
            except (OSError, ValueError) as error:
                logger.warning("archive %s is unreadable: %s", name, error)
        seen: dict = {}
        for name in self._files():
            try:
                record = records.get(name) or self._load(name)
            except (OSError, ValueError):
                continue
            for seat in record.get("room", {}).get("seats", []):
                key = seat.get("public_key")
                if key:
                    seen.setdefault(key, set()).add(name)
        repeated = {key for key, files in seen.items() if len(files) > 1}
        added = 0
        for name, record in records.items():
            try:
                self.add(record, name, house_keys=repeated)
                added += 1
            except (KeyError, TypeError, ValueError) as error:
                logger.warning("archive %s cannot be indexed: %s", name, error)
        return added

    def add(self, record: dict, file: str, house_keys: Iterable[str] = ()) -> dict:
        summary = summarize(record, house_keys)
        with self.lock, self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO games (game_id, room_id, file, name, finished_at, num_players, "
                "config, seats, winner, actions, turns, people) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    summary["game_id"],
                    summary["room_id"],
                    file,
                    summary["name"],
                    summary["finished_at"],
                    summary["num_players"],
                    json.dumps(summary["config"]),
                    json.dumps(summary["seats"]),
                    summary["winner"],
                    summary["actions"],
                    summary["turns"],
                    summary["people"],
                ),
            )
            self.db.execute("DELETE FROM players WHERE game_id = ?", (summary["game_id"],))
            for seat in summary["seats"]:
                key = name_key(seat["name"])
                if key is not None:
                    self.db.execute(
                        "INSERT OR REPLACE INTO players (game_id, color, name_key) VALUES (?,?,?)",
                        (summary["game_id"], seat["color"], key),
                    )
        return summary

    @staticmethod
    def _row(row: sqlite3.Row) -> dict:
        analysis = json.loads(row["analysis"]) if row["analysis"] else None
        return {
            "game_id": row["game_id"],
            "room_id": row["room_id"],
            "name": row["name"],
            "finished_at": row["finished_at"],
            "num_players": row["num_players"],
            "config": json.loads(row["config"]),
            "seats": json.loads(row["seats"]),
            "winner": row["winner"],
            "actions": row["actions"],
            "turns": row["turns"],
            "people": row["people"],
            "analysis": analysis,
        }

    def games(self, player: Optional[str] = None, limit: int = 50, before: Optional[float] = None) -> list:
        """Newest first: the games a name sat in (with ``player``), else every
        game a person played."""
        limit = max(1, min(int(limit), 200))
        args: list = []
        if player is not None:
            key = name_key(player)
            if key is None:
                return []
            sql = (
                "SELECT g.* FROM games g JOIN players p ON p.game_id = g.game_id "
                "WHERE p.name_key = ?"
            )
            args.append(key)
        else:
            sql = "SELECT g.* FROM games g WHERE g.people > 0"
        if before is not None:
            sql += " AND g.finished_at < ?"
            args.append(float(before))
        sql += " ORDER BY g.finished_at DESC LIMIT ?"
        args.append(limit)
        with self.lock:
            rows = self.db.execute(sql, args).fetchall()
        seen, out = set(), []
        for row in rows:  # (one name in two seats lists the game once)
            if row["game_id"] not in seen:
                seen.add(row["game_id"])
                out.append(self._row(row))
        return out

    def summary(self, game_id: str) -> Optional[dict]:
        if not GAME_ID.match(game_id or ""):
            return None
        with self.lock:
            row = self.db.execute("SELECT * FROM games WHERE game_id = ?", (game_id,)).fetchone()
        return self._row(row) if row else None

    def record(self, game_id: str) -> Optional[dict]:
        if not GAME_ID.match(game_id or ""):
            return None
        with self.lock:
            row = self.db.execute("SELECT file FROM games WHERE game_id = ?", (game_id,)).fetchone()
        if row is None:
            return None
        return self._load(row["file"])

    def replay(self, game_id: str) -> Optional[bytes]:
        """The replay as gzipped JSON (built once, then kept), with the
        game's analysis when there is one."""
        summary = self.summary(game_id)
        if summary is None:
            return None
        path = os.path.join(self.dir, "replays", f"{game_id}.v{REPLAY_VERSION}.json.gz")
        if os.path.exists(path):
            with open(path, "rb") as file:
                base = json.loads(gzip.decompress(file.read()))
        else:
            base = replay_of(self.record(game_id))
            tmp = path + ".tmp"
            with open(tmp, "wb") as file:
                file.write(gzip.compress(json.dumps(base, separators=(",", ":")).encode(), 6))
            os.replace(tmp, path)
        base["game"] = summary
        base["analysis"] = self.analysis(game_id)
        return gzip.compress(json.dumps(base, separators=(",", ":")).encode(), 6)

    # ===== analyses =====
    def analysis(self, game_id: str) -> Optional[dict]:
        path = os.path.join(self.dir, "analysis", f"{game_id}.json")
        if not GAME_ID.match(game_id or "") or not os.path.exists(path):
            return None
        with open(path) as file:
            return json.load(file)

    def queue(self, limit: int = 10) -> list:
        """Games the house may analyze: played by a person, long enough, not
        analyzed, not failed too often; newest first."""
        with self.lock:
            rows = self.db.execute(
                "SELECT game_id FROM games WHERE people > 0 AND analysis IS NULL AND actions >= ? "
                "AND analysis_tries < ? ORDER BY finished_at DESC LIMIT ?",
                (MIN_ANALYZED_ACTIONS, MAX_TRIES, max(1, min(int(limit), 100))),
            ).fetchall()
        return [row["game_id"] for row in rows]

    def submit(self, game_id: Any, analysis: Any = None, error: Any = None) -> None:
        """Store an analysis (a JSON object with a ``summary``), or count a
        failure. Raises ValueError for a game it does not know or a payload
        it will not keep."""
        if not isinstance(game_id, str) or self.summary(game_id) is None:
            raise ValueError("no such game")
        if analysis is None:
            with self.lock, self.db:
                self.db.execute(
                    "UPDATE games SET analysis_tries = analysis_tries + 1 WHERE game_id = ?", (game_id,)
                )
            logger.warning("analysis of %s failed: %s", game_id, str(error)[:200])
            return
        if not isinstance(analysis, dict) or not isinstance(analysis.get("summary"), dict):
            raise ValueError("an analysis is an object with a summary")
        text = json.dumps(analysis, separators=(",", ":"))
        if len(text) > MAX_ANALYSIS:
            raise ValueError("the analysis is too large")
        path = os.path.join(self.dir, "analysis", f"{game_id}.json")
        with open(path + ".tmp", "w") as file:
            file.write(text)
        os.replace(path + ".tmp", path)
        with self.lock, self.db:
            self.db.execute(
                "UPDATE games SET analysis = ? WHERE game_id = ?",
                (json.dumps(analysis["summary"], separators=(",", ":")), game_id),
            )


def stamp(record: dict) -> dict:
    """The archive record with the time it was written."""
    record["finished_at"] = time.time()
    return record
