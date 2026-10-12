"""Rutas de la Fase 2. Ningún otro módulo de ``ingesta`` calcula rutas.

Cada tema tiene su propio hueco, en el repositorio y en el disco:

- ``results/ingesta/<tema>/``: salidas pequeñas versionadas (fichas JSON, catálogos, tablas).
- ``<INGESTA_CORPUS_DIR>/<tema>/``: corpus pesado, FUERA del repositorio. La ruta va en la variable
  de entorno o en ``.env`` (``INGESTA_CORPUS_DIR``; por defecto ``./corpus`` en la raíz del repo,
  ignorado por git).
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = ROOT / ".env"
RESULTS_INGESTA = ROOT / "results" / "ingesta"

TEMAS = ("mercados_series", "bancos_riesgo", "empresas", "noticias_alt")


def env(nombre: str, defecto: str | None = None) -> str | None:
    """Lee una variable de ``.env`` o del entorno (no la imprime)."""
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(f"{nombre}="):
                valor = line.split("=", 1)[1].strip()
                if valor:
                    return valor
    return os.environ.get(nombre, defecto)


def _tema_valido(tema: str) -> str:
    if tema not in TEMAS and tema != "comun":
        raise ValueError(f"tema desconocido {tema!r}; usa uno de {TEMAS}")
    return tema


def resultados_dir(tema: str) -> Path:
    """Salidas versionadas de un tema: ``results/ingesta/<tema>/``."""
    p = RESULTS_INGESTA / _tema_valido(tema)
    p.mkdir(parents=True, exist_ok=True)
    return p


def corpus_dir(tema: str | None = None) -> Path:
    """Corpus pesado (fuera del repo); con ``tema``, su subcarpeta."""
    base = Path(env("INGESTA_CORPUS_DIR", str(ROOT / "corpus")))
    p = base / _tema_valido(tema) if tema else base
    p.mkdir(parents=True, exist_ok=True)
    return p
