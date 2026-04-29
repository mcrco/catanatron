from __future__ import annotations

import importlib.util
import logging
import os
import sys
from dataclasses import dataclass
from hashlib import sha1
from typing import Callable, Iterable, Sequence

from catanatron.models.player import Color, Player


VALID_MAP_TEMPLATES = ("BASE", "MINI", "TOURNAMENT")


@dataclass(frozen=True)
class PlayerFactoryContext:
    """Game setup information available to custom web player factories."""

    player_key: str
    color: Color
    player_keys: Sequence[str]
    map_template: str
    discard_limit: int
    vps_to_win: int
    friendly_robber: bool

    @property
    def num_players(self) -> int:
        return len(self.player_keys)


PlayerFactory = Callable[[Color, PlayerFactoryContext], Player]


@dataclass(frozen=True)
class PlayerRegistration:
    key: str
    label: str
    factory: PlayerFactory
    description: str = ""
    min_players: int = 2
    max_players: int = 4
    map_templates: Sequence[str] = VALID_MAP_TEMPLATES

    def to_json(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "description": self.description,
            "min_players": self.min_players,
            "max_players": self.max_players,
            "map_templates": list(self.map_templates),
        }

    def validate_context(self, context: PlayerFactoryContext) -> None:
        if not self.min_players <= context.num_players <= self.max_players:
            raise ValueError(
                f"{self.label} supports {self.min_players}-{self.max_players} players, "
                f"got {context.num_players}."
            )
        if context.map_template not in self.map_templates:
            supported = ", ".join(self.map_templates)
            raise ValueError(f"{self.label} supports maps: {supported}.")


class PlayerRegistry:
    def __init__(self) -> None:
        self._registrations: dict[str, PlayerRegistration] = {}
        self._loaded_module_paths: set[str] = set()

    def register(
        self,
        *,
        key: str,
        label: str,
        factory: PlayerFactory,
        description: str = "",
        min_players: int = 2,
        max_players: int = 4,
        map_templates: Sequence[str] = VALID_MAP_TEMPLATES,
    ) -> None:
        if key in self._registrations:
            raise ValueError(f"Player key '{key}' is already registered.")
        self._registrations[key] = PlayerRegistration(
            key=key,
            label=label,
            factory=factory,
            description=description,
            min_players=min_players,
            max_players=max_players,
            map_templates=tuple(map_templates),
        )

    def create(self, context: PlayerFactoryContext) -> Player:
        registration = self._registrations.get(context.player_key)
        if registration is None:
            raise ValueError(f"Unknown player key '{context.player_key}'.")
        registration.validate_context(context)
        return registration.factory(context.color, context)

    def list_players(self) -> list[dict]:
        return [registration.to_json() for registration in self._registrations.values()]


def load_external_player_modules(registry: PlayerRegistry, module_paths: Iterable[str]) -> None:
    for module_path in module_paths:
        path = module_path.strip()
        if not path:
            continue

        abs_path = os.path.abspath(path)
        if abs_path in registry._loaded_module_paths:
            continue
        module_hash = sha1(abs_path.encode("utf-8")).hexdigest()
        module_name = f"catanatron_web_player_{module_hash}"
        spec = importlib.util.spec_from_file_location(module_name, abs_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load player module at '{abs_path}'.")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        register_players = getattr(module, "register_players", None)
        if register_players is None:
            raise ValueError(f"Player module '{abs_path}' must define register_players(registry).")
        register_players(registry)
        registry._loaded_module_paths.add(abs_path)
        logging.info("Loaded Catanatron web player module: %s", abs_path)


def module_paths_from_env(env_value: str | None) -> list[str]:
    if not env_value:
        return []
    # os.pathsep handles normal shell usage; commas make single-env-var Docker edits easier.
    paths: list[str] = []
    for chunk in env_value.split(os.pathsep):
        paths.extend(part.strip() for part in chunk.split(","))
    return [path for path in paths if path]
