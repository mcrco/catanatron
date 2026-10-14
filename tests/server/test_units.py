"""Configs, maps, the crypto helpers and view sampling, without a network."""

import random

import pytest

from catanatron.game import Game
from catanatron.models.decks import BASE_SUPPLY, EXTENSION_SUPPLY
from catanatron.models.map import build_map
from catanatron.models.player import Color, RandomPlayer
from catanatron.serialization import client_view, state_to_json
from catanatron.server.config import ConfigError, GameConfig, default_action, new_game
from catanatron.server.crypto import (
    SigningKey,
    canonical,
    master_seed,
    seal,
    seed_commitment,
    verify_envelope,
)
from catanatron.server.maps import MapError, board_geometry, map_spec, parse_map
from catanatron.server.sampling import sample_game


# ===== config =====
def test_config_defaults_and_supply():
    assert GameConfig.from_json({}).supply == "base"
    assert GameConfig.from_json({"num_players": 5}).supply == "extension"
    assert GameConfig.from_json({"num_players": 6}).game_supply == EXTENSION_SUPPLY
    assert GameConfig.from_json({"num_players": 3}).colors == [
        Color.RED,
        Color.BLUE,
        Color.ORANGE,
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {"num_players": 1},
        {"num_players": 7},
        {"num_players": "4"},
        {"map_type": "HUGE"},
        {"vps_to_win": 2},
        {"supply": "deluxe"},
        {"decision_timeout_ms": 1.5},
        {"friendly_robber": "yes"},
        {"colour": "RED"},
        [],
    ],
)
def test_invalid_configs_are_rejected(payload):
    with pytest.raises(ConfigError):
        GameConfig.from_json(payload)


def test_config_round_trips():
    config = GameConfig.from_json({"num_players": 3, "vps_to_win": 8})
    assert GameConfig.from_json(config.to_json()) == config


# ===== maps =====
def test_a_map_spec_round_trips():
    catan_map = build_map("BASE")
    spec = map_spec(catan_map)
    rebuilt = parse_map(spec)
    assert map_spec(rebuilt) == spec
    assert rebuilt.port_nodes == catan_map.port_nodes
    assert rebuilt.land_nodes == catan_map.land_nodes


def test_any_shape_is_accepted():
    spec = _base_spec()
    spec["tiles"].reverse()  # numbers the nodes differently: another layout
    spec["tiles"] = [t for t in spec["tiles"] if t["type"] != "water"]
    catan_map = parse_map(spec)
    assert not catan_map.is_base_layout()
    assert map_spec(catan_map) == spec


def test_the_mini_layout_is_accepted():
    assert (
        parse_map(map_spec(build_map("MINI"))).land_nodes
        == build_map("MINI").land_nodes
    )


def _base_spec():
    return map_spec(build_map("BASE"))


@pytest.mark.parametrize(
    "edit,message",
    [
        (lambda s: s["tiles"][0].update(coordinate=[1, 1, 1]), "summing to 0"),
        (lambda s: s["tiles"][1].update(coordinate=[0, 0, 0]), "twice"),
        (lambda s: s["tiles"][0].update(type="lava"), "type"),
        (
            lambda s: [
                t.update(resource="WOOD", number=6)
                for t in s["tiles"]
                if t["type"] == "land"
            ],
            "desert",
        ),
        (lambda s: _first(s, "port").update(direction="EAST"), "not land"),
        (lambda s: _first(s, "port").update(port_resource="GOLD"), "port resource"),
        (lambda s: _numbered(s).update(number=7), "number"),
    ],
)
def test_bad_map_specs_are_rejected(edit, message):
    spec = _base_spec()
    edit(spec)
    with pytest.raises(MapError, match=message):
        parse_map(spec)


def _first(spec, kind):
    return next(t for t in spec["tiles"] if t["type"] == kind)


def _numbered(spec):
    return next(t for t in spec["tiles"] if t.get("number"))


def test_board_geometry_places_every_node_and_names_port_nodes():
    catan_map = build_map("BASE")
    tiles = board_geometry(catan_map)
    nodes = {n for t in tiles for n in t["nodes"].values()}
    assert catan_map.land_nodes <= nodes
    port_nodes = {n for t in tiles if t["type"] == "port" for n in t["port_nodes"]}
    assert port_nodes == set().union(*catan_map.port_nodes.values())


# ===== games from seeds =====
def test_a_game_is_a_function_of_its_config_and_master_seed():
    config = GameConfig.from_json({"num_players": 4, "number_placement": "random"})
    one = state_to_json(new_game(config, b"a" * 32, "g"))
    assert one == state_to_json(new_game(config, b"a" * 32, "g"))
    other = state_to_json(new_game(config, b"b" * 32, "g"))
    assert one["map"] != other["map"], "the board comes from the seed too"
    assert one["development_listdeck"] != other["development_listdeck"]


def test_the_default_action_prefers_rolling_and_ending_the_turn():
    game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=1)
    assert default_action(game.playable_actions) == game.playable_actions[0]
    while game.state.is_initial_build_phase:
        game.play_tick()
    assert default_action(game.playable_actions).action_type.value in (
        "ROLL",
        "PLAY_KNIGHT_CARD",
    )


# ===== crypto =====
def test_canonical_json_is_order_independent():
    assert canonical({"b": 1, "a": [1, {"d": 2, "c": "é"}]}) == canonical(
        {"a": [1, {"c": "é", "d": 2}], "b": 1}
    )
    assert canonical({"a": "é"}) == '{"a":"é"}'.encode()


def test_a_sealed_message_verifies_and_a_changed_one_does_not():
    key = SigningKey.generate()
    envelope = seal(key, to=None, seq=0, prev="0" * 64, commit=None, body={"x": 1})
    assert verify_envelope(key.public_hex, envelope)
    assert not verify_envelope(key.public_hex, {**envelope, "body": {"x": 2}})
    assert not verify_envelope(SigningKey.generate().public_hex, envelope)


def test_keys_round_trip_through_files(tmp_path):
    path = str(tmp_path / "key")
    key = SigningKey.load_or_create(path)
    assert SigningKey.load_or_create(path).public_hex == key.public_hex


def test_the_master_seed_depends_on_every_seed_and_their_order():
    server = "ab" * 32
    seeds = ["01" * 32, None, "02" * 32]
    base = master_seed(server, seeds)
    assert master_seed(server, list(reversed(seeds))) != base
    assert master_seed(server, ["03" * 32, None, "02" * 32]) != base
    assert master_seed("cd" * 32, seeds) != base
    assert len(seed_commitment(server)) == 64


# ===== sampling =====
@pytest.mark.parametrize("num_players", [2, 3, 4, 5, 6])
def test_a_sampled_game_looks_exactly_like_the_view(num_players):
    colors = list(Color)[:num_players]
    supply = EXTENSION_SUPPLY if num_players > 4 else BASE_SUPPLY
    game = Game([RandomPlayer(c) for c in colors], seed=num_players, supply=supply)
    rng = random.Random(0)
    for tick in range(400):
        if game.winning_color() is not None:
            break
        game.play_tick()
        if tick % 97 == 0:
            doc = state_to_json(game)
            for color in colors:
                view = client_view(doc, color)
                sampled = sample_game(view, color, supply, rng)
                assert client_view(state_to_json(sampled), color) == view
                if game.state.current_color() == color:
                    # (Another seat's options depend on its hidden hand.)
                    assert sampled.playable_actions == game.playable_actions


def test_configs_and_games_need_no_cryptography():
    """Local tools (the TUI) build games from configs without the server's
    optional dependencies."""
    import subprocess
    import sys

    program = """
import sys
sys.modules["cryptography"] = None  # importing it now fails
from catanatron.server.config import GameConfig, new_game
from catanatron.server.hashing import master_seed
game = new_game(GameConfig.from_json({"num_players": 3}), master_seed("ab" * 32, [None] * 3), "g")
assert len(game.state.colors) == 3
"""
    subprocess.run([sys.executable, "-c", program], check=True, timeout=120)
