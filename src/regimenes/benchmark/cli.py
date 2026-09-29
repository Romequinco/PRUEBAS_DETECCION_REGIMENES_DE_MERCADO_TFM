"""CLI del benchmark: ``python -m regimenes.benchmark`` (antes ``python -m src.benchmark``)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from regimenes.benchmark.cache import DEFAULT_OUTPUT
from regimenes.benchmark.ejecucion import consolidate_run, run_benchmark, run_job
from regimenes.detectores import detector_specs


def _rango_ids() -> str:
    """Rango de IDs del registro (p. ej. ``D01..D13``), para que la ayuda no se
    desfase al anadir detectores."""
    ids = sorted({s.detector_id for t in ("A", "B") for s in detector_specs(t)})
    return f"{ids[0]}..{ids[-1]}" if ids else "D01.."


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m regimenes.benchmark",
        description="Benchmark v2 de detectores (secuencial, paralelo o por detector).",
    )
    parser.add_argument("--track", nargs="+", default=["A", "B"], help="pistas: A, B o ambas")
    parser.add_argument("--detector", nargs="+", default=None,
                        help=f"IDs del registro ({_rango_ids()}). Con --jobs 1 solo se ejecutan esas "
                             "combinaciones y NO se escribe manifest/run_status "
                             "(usa --consolidate al final); con --jobs > 1 se "
                             "reparten en procesos y este proceso consolida.")
    parser.add_argument("--jobs", type=int, default=1, help="procesos en paralelo (matriz completa)")
    parser.add_argument("--threads-per-job", type=int, default=1,
                        help="hilos BLAS/OpenMP por proceso con --jobs > 1 (0 = no tocar)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="directorio de salida")
    parser.add_argument("--force", action="store_true", help="ignora la cache y reajusta")
    parser.add_argument("--cache-only", action="store_true", help="solo verifica la cache")
    parser.add_argument("--consolidate", action="store_true",
                        help="reescribe manifest.json y run_status.csv desde status/*.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    tracks = [track.upper() for track in args.track]
    output = Path(args.output)
    if args.consolidate:
        status = consolidate_run(output, tracks=tracks)
    elif args.detector and args.jobs <= 1 and not args.cache_only:
        rows = [
            run_job(track, detector_id, output_dir=output, force=args.force)
            for track in tracks
            for detector_id in args.detector
            if any(spec.detector_id == detector_id.upper() for spec in detector_specs(track))
        ]
        status = pd.DataFrame(rows)
    else:
        _, status = run_benchmark(
            tracks=tracks, detector_ids=args.detector, output_dir=output, force=args.force,
            cache_only=args.cache_only, n_jobs=args.jobs,
            threads_per_job=args.threads_per_job or None,
        )
    if not status.empty:
        cols = [c for c in ("pista", "id", "estado", "segundos", "detalle") if c in status]
        print(status[cols].to_string(index=False))
    return int(bool(len(status)) and status["estado"].eq("error").any())
