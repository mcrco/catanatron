"""Python entrypoint for the C++ Catanatron parity engine.

The extension is optional at import time so pure-Python users can keep using
the package before building the pybind11 module.
"""

try:
    from catanatron._cpp_engine import (  # type: ignore
        Action,
        ActionPrompt,
        ActionType,
        Color,
        Edge,
        Game,
        Player,
        static_edges,
    )

    AVAILABLE = True
except ImportError:  # pragma: no cover - depends on local build state
    AVAILABLE = False

    Action = None  # type: ignore
    ActionPrompt = None  # type: ignore
    ActionType = None  # type: ignore
    Color = None  # type: ignore
    Edge = None  # type: ignore
    Game = None  # type: ignore
    Player = None  # type: ignore

    def static_edges():
        raise RuntimeError(
            "The C++ engine extension is not built. Install with `pip install -e .` "
            "in an environment with pybind11."
        )


__all__ = [
    "AVAILABLE",
    "Action",
    "ActionPrompt",
    "ActionType",
    "Color",
    "Edge",
    "Game",
    "Player",
    "static_edges",
]
