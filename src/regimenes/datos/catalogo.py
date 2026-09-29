"""catalogo.py — Carga del catalogo de series (``configs/catalog.yaml``).

``load_catalog`` lee el global ``CATALOG`` de este modulo en tiempo de llamada, de modo
que los tests pueden parchear ``regimenes.datos.catalogo.CATALOG``.
"""
from __future__ import annotations

import yaml

from regimenes.rutas import CATALOG as _CATALOG_RUTA

CATALOG = _CATALOG_RUTA


def load_catalog() -> dict:
    return yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
