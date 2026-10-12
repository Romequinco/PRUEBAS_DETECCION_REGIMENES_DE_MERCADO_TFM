"""Tests de ``ingesta``.

- Por defecto, sin red: parsean muestras guardadas en ``tests/ingesta/<tema>/fixtures/``.
- ``@pytest.mark.live``: tocan la red; se saltan salvo con ``INGESTA_LIVE=1``.
"""
import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("INGESTA_LIVE") == "1":
        return
    saltar = pytest.mark.skip(reason="test live: exportar INGESTA_LIVE=1 para ejecutarlo")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(saltar)
