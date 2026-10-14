"""What one recipient knows of a game, assembled from the messages it got.

Game messages carry the state without the map (sent once) and the log
incrementally (see :mod:`catanatron.server.messages`). :class:`Table` puts
them back together into exactly the document ``client_view`` (or
``public_view``) produces, which is what decision callbacks receive.
"""

from __future__ import annotations

from typing import Optional

from catanatron.models.decks import Supply
from catanatron.models.player import Color
from catanatron.server.config import SUPPLIES
from catanatron.server.messages import GAME_MESSAGES


class Table:
    def __init__(self) -> None:
        self.game_id: Optional[str] = None
        self.color: Optional[str] = None
        self.config: dict = {}
        self.seats: list = []
        self.client_seeds: list = []
        self.map: Optional[dict] = None
        self.board: list = []
        self.state: Optional[dict] = None
        self.log: list = []
        self.index = 0
        self.current_color: Optional[str] = None
        self.winning_color: Optional[str] = None
        self.over = False

    def update(self, body: dict) -> None:
        """Take in a game message (others are ignored)."""
        kind = body.get("type")
        if kind not in GAME_MESSAGES:
            return
        if kind == "before":
            self.game_id = body["game_id"]
            self.color = body["color"]
            self.config = body["config"]
            self.seats = body["seats"]
            self.client_seeds = body["client_seeds"]
            self.map = body["map"]
            self.board = body["board"]
        start = body["log_start"]
        if start > len(self.log):
            raise ValueError(f"log gap: have {len(self.log)}, got entries from {start}")
        self.log[start:] = body["log"]
        self.state = body["state"]
        self.index = body["index"]
        self.current_color = body["current_color"]
        self.winning_color = body["winning_color"]
        self.over = kind == "after"

    def view(self) -> dict:
        """The recipient's ``client_view``/``public_view`` of the game now."""
        if self.state is None:
            raise ValueError("the game has not started")
        return {**self.state, "map": self.map, "action_records": list(self.log)}

    @property
    def supply(self) -> Supply:
        return SUPPLIES[self.config.get("supply", "base")]

    @property
    def my_color(self) -> Optional[Color]:
        return Color[self.color] if self.color else None
