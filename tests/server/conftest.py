"""The server tests' options: ``-m fuzz`` runs the fuzzer's large budget,
which is skipped otherwise."""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "fuzz: the verdict fuzzer's large budget (run with -m fuzz)"
    )


def pytest_collection_modifyitems(config, items):
    if "fuzz" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason="the large fuzz budget runs with -m fuzz")
    for item in items:
        if "fuzz" in item.keywords:
            item.add_marker(skip)
