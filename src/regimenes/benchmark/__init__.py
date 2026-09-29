"""Benchmark v2 de detectores.

- ``ejecucion``  paneles, gate, run_one/run_job/run_benchmark, consolidación.
- ``cache``      rutas de artefactos, huella de caché y verificación de procedencia.
- ``cli``        argumentos y ``main``; ``python -m regimenes.benchmark``.

El ranking de detección ADR-003 vive en ``regimenes.evaluacion.ranking`` y NO se
re-exporta aquí (``evaluacion.ranking`` depende de ``benchmark.cache``).
"""

from regimenes.benchmark.cache import (
    CACHE_SCHEMA_VERSION,
    DEFAULT_OUTPUT,
    PROCESSED,
    ROOT,
    load_benchmark_spec,
    load_metrics,
    metrics_provenance,
    processed_available,
    processed_paths,
)
from regimenes.benchmark.ejecucion import (
    DEFAULT_TRAIN_DAYS,
    common_track_index,
    configure_evaluation,
    consolidate_run,
    load_track_panel,
    preflight,
    run_benchmark,
    run_job,
    run_one,
)
from regimenes.benchmark.cli import main

__all__ = [
    "ROOT", "PROCESSED", "DEFAULT_OUTPUT", "DEFAULT_TRAIN_DAYS", "CACHE_SCHEMA_VERSION",
    "load_benchmark_spec", "configure_evaluation", "processed_paths", "processed_available",
    "load_track_panel", "common_track_index", "preflight", "run_one", "run_job",
    "run_benchmark", "consolidate_run", "metrics_provenance", "load_metrics", "main",
]
