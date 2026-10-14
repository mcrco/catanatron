"""Playable actions come in the same order in every process.

Some actions used to be built from sets, whose order follows the process's
string hash seed and, before Python 3.12, hash(None), an address that ASLR
moves. Seeded games against bots that pick by index or break ties by order
then played out differently in every process.
"""

import json
import os
import subprocess
import sys

PROGRAM = """
import json, random
from catanatron.game import Game
from catanatron.models.actions import (
    inner_maritime_trade_possibilities,
    year_of_plenty_possibilities,
)
from catanatron.models.player import Color, RandomPlayer

out = {
    "trades": list(inner_maritime_trade_possibilities([4, 3, 2, 4, 0], [19] * 5, {None, "ORE"})),
    "plenty": [a.value for a in year_of_plenty_possibilities(Color.RED, [19, 1, 0, 19, 2])],
}
random.seed(0)
game = Game([RandomPlayer(c) for c in [Color.RED, Color.BLUE, Color.ORANGE]], seed=3)
game.play()
out["game"] = [str(record) for record in game.state.action_records]
print(json.dumps(out))
"""


def run(hash_seed):
    env = dict(os.environ, PYTHONHASHSEED=hash_seed)
    result = subprocess.run(
        [sys.executable, "-c", PROGRAM],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    return result.stdout


def test_action_order_and_seeded_games_do_not_depend_on_the_process():
    outputs = {run(seed) for seed in ("1", "2", "random", "random")}
    assert len(outputs) == 1
    out = json.loads(outputs.pop())
    assert len(out["game"]) > 100
    assert out["trades"][0] == ["WOOD", "WOOD", "WOOD", None, "BRICK"]
