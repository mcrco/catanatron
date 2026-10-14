"""catanatron-tui's command line, for local games."""

import json

import pytest

from catanatron.models.player import Color
from catanatron.tui.cli import build_parser, local_session, rules_config

from tests.tui.helpers import MAPS_DIR


def session_for(*argv):
    return local_session(build_parser().parse_args(list(argv)))


def test_seats_rules_and_maps():
    session = session_for(
        "--seats",
        "human,AB:1,oracle:F",
        "--map",
        f"{MAPS_DIR}/blob25.json",
        "--vps",
        "7",
        "--discard-limit",
        "9",
        "--seed",
        "4",
        "--seat",
        "BLUE=R",
    )
    assert [session.specs[c] for c in (Color.RED, Color.BLUE, Color.ORANGE)] == [
        "human",
        "R",
        "oracle:F",
    ]
    assert session.game.vps_to_win == 7 and session.game.state.discard_limit == 9
    assert len(session.game.state.board.map.land_tiles) == 25
    assert session.seed == 4


def test_config_json_and_defaults():
    session = session_for(
        "--config", json.dumps({"num_players": 5, "map_type": "BASE"})
    )
    assert len(session.colors) == 5
    assert session.specs[Color.RED] == "human"
    assert all(session.specs[c] == "F" for c in session.colors if c != Color.RED)
    args = build_parser().parse_args(["--map-type", "MINI", "--friendly-robber"])
    assert rules_config(args) == {"map_type": "MINI", "friendly_robber": True}


def test_load_with_seat_overrides(tmp_path):
    first = session_for("--seats", "F,F", "--vps", "5", "--seed", "1")
    first.run(limit=40)
    path = str(tmp_path / "game.json")
    first.save(path)
    session = session_for("--load", path, "--seat", "RED=human")
    assert session.specs[Color.RED] == "human" and session.specs[Color.BLUE] == "F"
    assert len(session.undo_stack) == 40


@pytest.mark.parametrize(
    "argv",
    [
        ["--seats", "human,NOSUCHBOT"],
        ["--seats", "human,F", "--config", '{"num_players": 3}'],
        ["--seat", "PURPLE=F"],
        ["--seat", "RED"],
        ["--config", '{"colour": 1}'],
    ],
)
def test_bad_arguments(argv):
    with pytest.raises(ValueError):
        session_for(*argv)
