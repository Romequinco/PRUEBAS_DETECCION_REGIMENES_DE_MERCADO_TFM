"""Rutas centralizadas del repositorio (unica fuente de verdad).

Todo el paquete se refiere al disco a traves de este modulo; ningun otro modulo
calcula rutas con ``Path(__file__).parents[...]``. El paquete se instala en modo
editable (``pip install -e .``), de modo que ``ROOT`` es la raiz del checkout.

Directorios de datos y resultados (fuera del paquete):

- ``configs/``            catalog.yaml, benchmark_spec.yaml, sinteticos.yaml
- ``data/raw``            descargas crudas (gitignored salvo procedencia)
- ``data/processed``      paneles pistaA/pistaB (gitignored)
- ``data/sinteticos``     trayectorias sinteticas generadas (gitignored)
- ``results/benchmark``   salida de ``python -m regimenes.benchmark``
- ``results/fusion``      subcarpetas d07_d08/ y d02_d06/
- ``results/detectores``, ``results/sinteticos``, ``results/pseudolive``
"""

from __future__ import annotations

from pathlib import Path

# src/regimenes/rutas.py -> parents[0]=regimenes, [1]=src, [2]=raiz del repo
ROOT: Path = Path(__file__).resolve().parents[2]

CONFIGS: Path = ROOT / "configs"
CATALOG: Path = CONFIGS / "catalog.yaml"
BENCHMARK_SPEC: Path = CONFIGS / "benchmark_spec.yaml"
SINTETICOS_CONFIG: Path = CONFIGS / "sinteticos.yaml"

DATA: Path = ROOT / "data"
DATA_RAW: Path = DATA / "raw"
DATA_PROCESSED: Path = DATA / "processed"
DATA_SINTETICOS: Path = DATA / "sinteticos"

RESULTS: Path = ROOT / "results"
RESULTS_BENCHMARK: Path = RESULTS / "benchmark"
RESULTS_FUSION: Path = RESULTS / "fusion"
RESULTS_FUSION_D07_D08: Path = RESULTS_FUSION / "d07_d08"
RESULTS_FUSION_D02_D06: Path = RESULTS_FUSION / "d02_d06"
RESULTS_DETECTORES: Path = RESULTS / "detectores"
RESULTS_SINTETICOS: Path = RESULTS / "sinteticos"
RESULTS_PSEUDOLIVE: Path = RESULTS / "pseudolive"

DOCS: Path = ROOT / "docs"
ENV_FILE: Path = ROOT / ".env"

__all__ = [
    "ROOT", "CONFIGS", "CATALOG", "BENCHMARK_SPEC", "SINTETICOS_CONFIG",
    "DATA", "DATA_RAW", "DATA_PROCESSED", "DATA_SINTETICOS",
    "RESULTS", "RESULTS_BENCHMARK", "RESULTS_FUSION",
    "RESULTS_FUSION_D07_D08", "RESULTS_FUSION_D02_D06",
    "RESULTS_DETECTORES", "RESULTS_SINTETICOS", "RESULTS_PSEUDOLIVE",
    "DOCS", "ENV_FILE",
]
