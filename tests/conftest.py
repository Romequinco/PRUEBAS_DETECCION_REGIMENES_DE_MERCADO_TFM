"""Configuracion comun de pytest para ``tests/`` (espejo de ``src/regimenes``).

Estructura (ADR-004)::

    tests/datos/        -> regimenes.datos
    tests/features/     -> regimenes.features
    tests/detectores/   -> regimenes.detectores
    tests/evaluacion/   -> regimenes.evaluacion
    tests/benchmark/    -> regimenes.benchmark
    tests/fusion/       -> regimenes.fusion
    tests/sinteticos/   -> regimenes.sinteticos
    tests/test_paquete_unificado.py -> guardas del paquete completo

Marcadores (declarados en ``pyproject.toml``):

- ``datos``: necesita ``data/raw`` o ``data/processed`` (gitignored). Si faltan,
  el propio test se salta con motivo; ``make test-rapido`` los deselecciona.
- ``resultados``: lee ``results/benchmark`` versionado (metricas, ranking,
  manifest). Pueden fallar de forma transitoria mientras un benchmark reescribe
  esa carpeta; ``make test-rapido`` tambien los deselecciona.

Las dependencias opcionales (``torch`` para D12, ``jumpmodels`` para D09) se
tratan con ``pytest.importorskip`` en los tests que las necesitan, de modo que
CI (sin extras ``[deep,jump]``) las salta limpiamente.

Los nombres de archivo de test son unicos en todo ``tests/`` (modo de import
por defecto ``prepend`` sin ``__init__.py``): no repetir un basename.
"""
from __future__ import annotations

import os

# Backend sin ventana para cualquier figura creada en tests (CI sin display).
os.environ.setdefault("MPLBACKEND", "Agg")

from regimenes import rutas  # noqa: E402


def _hay(path) -> str:
    return "si" if path.exists() and any(path.iterdir()) else "no"


def pytest_report_header(config) -> list[str]:
    raw = rutas.DATA_RAW
    processed = rutas.DATA_PROCESSED
    panels = rutas.RESULTS_BENCHMARK / "panels"
    return [
        f"regimenes: raiz={rutas.ROOT}",
        "datos locales: "
        f"data/raw={_hay(raw) if raw.is_dir() else 'no'} "
        f"data/processed={'si' if (processed / 'pistaA_diaria.parquet').exists() else 'no'} "
        f"results/benchmark/panels={_hay(panels) if panels.is_dir() else 'no'}",
    ]
