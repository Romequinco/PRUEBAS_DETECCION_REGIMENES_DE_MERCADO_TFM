"""Huella de caché, rutas de artefactos y verificación de procedencia del benchmark.

Contiene las rutas de salida (``DEFAULT_OUTPUT``
= ``results/benchmark``), la huella reproducible por contenido
(:func:`_cache_fingerprint`), la verificación de la caché y la lectura de
métricas (:func:`load_metrics`, :func:`metrics_provenance`).

No importa ``regimenes.evaluacion.ranking`` (el ranking depende de este módulo).
"""

from __future__ import annotations

import ast
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
from importlib import metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

from regimenes.detectores import DetectorSpec, detector_specs
from regimenes.rutas import BENCHMARK_SPEC, DATA_PROCESSED, DATA_RAW, RESULTS_BENCHMARK, ROOT


# Alias públicos conservados (antes constantes de src/benchmark.py).
PROCESSED = DATA_PROCESSED
DEFAULT_OUTPUT = RESULTS_BENCHMARK
# v3: huella por CONTENIDO (parquet/yaml/texto normalizado) y detector_base resuelto.
CACHE_SCHEMA_VERSION = 3

# Paquete instalado (layout src/): de aquí salen los archivos de código hasheados.
_PACKAGE_DIR = ROOT / "src" / "regimenes"


def _normalized_text_bytes(path: Path) -> bytes:
    """Bytes de un archivo de texto con fin de línea normalizado (CRLF → LF).

    ``git`` con ``core.autocrlf`` reescribe los finales de línea al hacer checkout
    en Windows: el contenido es el mismo pero los bytes no.
    """
    return Path(path).read_bytes().replace(b"\r\n", b"\n")


def _parquet_content_digest(path: Path) -> str:
    """SHA-256 del CONTENIDO de un parquet, no de sus bytes.

    Dos parquet con los mismos datos pueden diferir en bytes (versión de pyarrow,
    compresión, metadatos de escritura, orden de row groups). Se hashea el
    esquema (nombres y dtypes de columnas e índice) y el valor de cada fila con
    ``pandas.util.hash_pandas_object`` (determinista, independiente de la
    plataforma para un mismo pandas; la versión de pandas ya entra en la huella).
    """
    frame = pd.read_parquet(path)
    digest = hashlib.sha256()
    schema = {
        "columns": [str(col) for col in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "index_names": [str(name) for name in frame.index.names],
        "index_dtype": str(frame.index.dtype),
        "shape": list(frame.shape),
    }
    digest.update(json.dumps(schema, sort_keys=True).encode("utf-8"))
    row_hashes = pd.util.hash_pandas_object(frame, index=True, categorize=True)
    digest.update(np.ascontiguousarray(row_hashes.to_numpy(dtype="<u8")).tobytes())
    return digest.hexdigest()


def _yaml_content_digest(path: Path) -> str:
    """SHA-256 del YAML ya parseado: comentarios y formato no invalidan la caché."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@lru_cache(maxsize=None)
def _sha256_file_version(path_text: str, size: int, mtime_ns: int) -> str:
    """Hash de contenido cacheado mientras tamaño y mtime del archivo no cambien."""
    del size, mtime_ns
    path = Path(path_text)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return _parquet_content_digest(path)
    if suffix in {".yaml", ".yml"}:
        return _yaml_content_digest(path)
    return hashlib.sha256(_normalized_text_bytes(path)).hexdigest()


def _sha256_file(path: Path) -> str:
    path = Path(path).resolve()
    stat = path.stat()
    return _sha256_file_version(str(path), stat.st_size, stat.st_mtime_ns)


def _relative_hashes(paths: Iterable[Path]) -> dict[str, str]:
    return {
        path.resolve().relative_to(ROOT).as_posix(): _sha256_file(path)
        for path in sorted({Path(item).resolve() for item in paths})
    }


def _resolved_detector_base() -> Path:
    """Archivo de ``regimenes.detectores.base`` (única ``RegimeDetector``).

    Desde la unificación (ADR-004) solo existe una copia y ningún módulo toca
    la ruta de importación: se resuelve con el mecanismo de importación estándar.
    """
    loaded = sys.modules.get("regimenes.detectores.base")
    loaded_file = getattr(loaded, "__file__", None)
    if loaded_file:
        return Path(loaded_file).resolve()
    spec = importlib.util.find_spec("regimenes.detectores.base")
    if spec is None or not spec.origin:
        raise FileNotFoundError("No se encontró detectores/base.py para la huella de caché.")
    return Path(spec.origin).resolve()


def _runtime_versions() -> dict[str, str]:
    versions = {"python": ".".join(map(str, sys.version_info[:3]))}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "hmmlearn", "arch", "torch", "jumpmodels"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


# Módulos donde vive la lógica que entra en ``_benchmark_model_sha256``.
_MODEL_SOURCE_MODULES = (
    "regimenes.benchmark.ejecucion",
    "regimenes.benchmark.cache",
    "regimenes.evaluacion.metricas",
)


def _benchmark_model_sha256() -> str:
    """Hash de la lógica que puede cambiar predicciones o métricas del modelo.

    Excluye deliberadamente carga de resultados, ranking, manifiesto y mensajes:
    modificar esas capas no debe forzar horas de reajuste numérico.
    """
    # Mismo conjunto y mismo orden que antes de la unificación; ahora viven en
    # tres módulos y se leen por AST sin importarlos (no crea ciclos).
    ordered_names = (
        "configure_evaluation",
        "processed_paths",
        "_sp500_path",
        "load_track_panel",
        "common_track_index",
        "_dependency_for",
        "run_one",
        # Columnas ``det_*`` de detección por evento que run_one escribe en el CSV.
        "_longest_true_run",
        "event_detection_table",
        "detection_summary",
    )
    selected_names = set(ordered_names)
    nodes: dict[str, ast.AST] = {}
    for module in _MODEL_SOURCE_MODULES:
        spec = importlib.util.find_spec(module)
        if spec is None or not spec.origin:
            continue
        tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
        for node in tree.body:
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in selected_names
            ):
                nodes.setdefault(node.name, node)
    selected = [nodes[name] for name in ordered_names if name in nodes]
    found = set(nodes)
    if found != selected_names:
        raise RuntimeError(
            f"No se pudo construir la huella del benchmark: faltan {sorted(selected_names - found)}"
        )
    canonical = ast.dump(
        ast.Module(body=selected, type_ignores=[]),
        annotate_fields=True,
        include_attributes=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_benchmark_spec() -> dict:
    path = BENCHMARK_SPEC
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def processed_paths(track: str) -> tuple[Path, Path]:
    track = track.upper()
    if track not in {"A", "B"}:
        raise ValueError("track debe ser 'A' o 'B'.")
    prefix = f"pista{track}"
    return PROCESSED / f"{prefix}_diaria.parquet", PROCESSED / f"{prefix}_mensual.parquet"


def processed_available(track: str | None = None) -> bool:
    tracks = [track.upper()] if track else ["A", "B"]
    return all(path.exists() for item in tracks for path in processed_paths(item))


def _sp500_path() -> Path:
    candidates = sorted(DATA_RAW.glob("*/SP500.parquet"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            "No se encontró de forma unívoca data/raw/<fuente>/SP500.parquet. "
            "Regenera la descarga antes del benchmark."
        )
    return candidates[0]


def _cache_code_paths() -> list[Path]:
    detectors_dir = _PACKAGE_DIR / "detectores"
    # Implementaciones oficiales por familia (f1_reglas … f7_deep), sin la
    # variante HSMM que no entra en la matriz oficial.
    official_detectors = sorted(
        path for path in detectors_dir.glob("f*/*.py")
        if path.name != "hsmm_tstudent.py"
    )
    return [
        _PACKAGE_DIR / "evaluacion" / "walk_forward.py",
        _PACKAGE_DIR / "evaluacion" / "metricas.py",
        detectors_dir / "__init__.py",
        detectors_dir / "registry.py",
        _resolved_detector_base(),
        *official_detectors,
    ]


def _cache_fingerprint(
    track: str,
    spec: DetectorSpec,
    train_days: int,
    *,
    runtime_versions: dict[str, str] | None = None,
) -> str:
    """Huella reproducible de configuración, código, datos y entorno."""
    track = track.upper()
    daily_path, monthly_path = processed_paths(track)
    payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "track": track,
        "train_days": int(train_days),
        "detector": asdict(spec),
        "runtime_versions": runtime_versions or _runtime_versions(),
        "benchmark_model_sha256": _benchmark_model_sha256(),
        "code_sha256": _relative_hashes(_cache_code_paths()),
        "detector_base_resolved": _resolved_detector_base().relative_to(ROOT).as_posix(),
        "input_sha256": _relative_hashes([
            BENCHMARK_SPEC,
            daily_path,
            monthly_path,
            _sp500_path(),
        ]),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_matches(metric_path: Path, panel_path: Path, expected: str) -> bool:
    """Solo acepta una caché completa y ligada a la huella actual."""
    if not metric_path.is_file() or not panel_path.is_file():
        return False
    try:
        cached = pd.read_csv(metric_path, usecols=["cache_fingerprint"])
    except (ValueError, OSError, pd.errors.ParserError):
        return False
    values = cached["cache_fingerprint"].dropna().astype(str).unique()
    return len(values) == 1 and values[0] == expected


def _metric_path(output_dir: Path, track: str, detector_id: str) -> Path:
    return output_dir / "metrics" / f"{track}_{detector_id}.csv"


def _panel_path(output_dir: Path, track: str, detector_id: str) -> Path:
    return output_dir / "panels" / f"{track}_{detector_id}.parquet"


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _write_atomic(path: Path, writer) -> None:
    """Escribe ``path`` a través de un temporal en el mismo directorio + replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    try:
        writer(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _status_path(output_dir: Path, track: str, detector_id: str) -> Path:
    return Path(output_dir) / "status" / f"{track}_{detector_id}.json"


def _write_manifest(
    output_dir: Path, train_days: dict[str, int], tracks: Iterable[str]
) -> None:
    benchmark_path = BENCHMARK_SPEC
    selected_tracks = tuple(dict.fromkeys(track.upper() for track in tracks))
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "benchmark_spec_sha256": _sha256_file(benchmark_path),
        "runtime_versions": _runtime_versions(),
        "train_days": train_days,
        "tracks": {
            track: [asdict(spec) for spec in detector_specs(track)]
            for track in ("A", "B")
        },
        "cache_fingerprints": {
            track: {
                spec.detector_id: _cache_fingerprint(
                    track, spec, train_days[track]
                )
                for spec in detector_specs(track)
            }
            for track in selected_tracks
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _read_verified_cache(
    output_dir: Path,
    tracks: Iterable[str],
    selected: set[str],
    train_days: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee solo las cachés verificadas por huella; no ajusta ni escribe nada."""
    statuses: list[dict] = []
    metric_frames: list[pd.DataFrame] = []
    for track in tracks:
        for spec in detector_specs(track):
            if spec.detector_id not in selected:
                continue
            metric_path = _metric_path(output_dir, track, spec.detector_id)
            panel_path = _panel_path(output_dir, track, spec.detector_id)
            try:
                expected = _cache_fingerprint(track, spec, train_days[track])
                valid = _cache_matches(metric_path, panel_path, expected)
                detail = (
                    f"caché verificada: {_display_path(metric_path)}" if valid
                    else "sin caché verificable (huella distinta o panel OOS ausente)"
                )
            except (FileNotFoundError, OSError) as exc:
                valid = False
                detail = f"no se pudo calcular la huella: {type(exc).__name__}: {exc}"
            if valid:
                metric_frames.append(pd.read_csv(metric_path))
            statuses.append({
                "pista": track,
                "id": spec.detector_id,
                "estado": "cache" if valid else "sin_cache",
                "segundos": 0.0,
                "detalle": detail,
            })
    metrics_all = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    return metrics_all, pd.DataFrame(statuses)


def metrics_provenance(output_dir: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    """Trazabilidad de cada CSV de métricas frente a ``manifest.json``.

    No necesita datos locales: comprueba que la huella escrita en cada CSV es la
    que el manifiesto registró para esa pista/detector (es decir, que las
    métricas versionadas proceden de la ejecución documentada) y si el panel OOS
    correspondiente existe en disco. Es una comprobación más débil que
    ``load_metrics(require_current=True)``, que además recalcula la huella con
    el código y los datos locales.
    """
    output_dir = Path(output_dir)
    try:
        manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = {}
    recorded = manifest.get("cache_fingerprints", {})
    rows = []
    for path in sorted((output_dir / "metrics").glob("[AB]_D*.csv")):
        frame = pd.read_csv(path)
        track = str(frame["pista"].iloc[0]).upper()
        detector_id = str(frame["id"].iloc[0]).upper()
        csv_fp = (
            str(frame["cache_fingerprint"].iloc[0])
            if "cache_fingerprint" in frame else None
        )
        manifest_fp = recorded.get(track, {}).get(detector_id)
        rows.append({
            "pista": track,
            "id": detector_id,
            "huella_csv": csv_fp,
            "huella_manifest": manifest_fp,
            "coincide_manifest": csv_fp is not None and csv_fp == manifest_fp,
            "panel_oos_presente": _panel_path(output_dir, track, detector_id).is_file(),
        })
    out = pd.DataFrame(rows)
    out.attrs["manifest_generated_utc"] = manifest.get("generated_utc")
    out.attrs["manifest_runtime_versions"] = manifest.get("runtime_versions")
    return out


def load_metrics(
    output_dir: Path = DEFAULT_OUTPUT, *, require_current: bool = True
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    files = sorted((output_dir / "metrics").glob("[AB]_D*.csv"))
    if not files:
        raise FileNotFoundError(
            "No hay métricas v2. Ejecuta primero el benchmark del notebook 04."
        )
    recorded_runtime: dict[str, str] | None = None
    if require_current:
        try:
            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            if manifest.get("cache_schema_version") == CACHE_SCHEMA_VERSION:
                recorded_runtime = manifest["runtime_versions"]
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            recorded_runtime = None

    frames: list[pd.DataFrame] = []
    stale: list[str] = []
    for path in files:
        frame = pd.read_csv(path)
        frames.append(frame)
        if not require_current:
            continue
        try:
            track = str(frame["pista"].iloc[0]).upper()
            detector_id = str(frame["id"].iloc[0]).upper()
            train_days = int(frame["train_days"].iloc[0])
            spec = next(
                item for item in detector_specs(track)
                if item.detector_id == detector_id
            )
            if recorded_runtime is None:
                raise ValueError("manifiesto sin entorno reproducible")
            expected = _cache_fingerprint(
                track,
                spec,
                train_days,
                runtime_versions=recorded_runtime,
            )
            panel_path = _panel_path(output_dir, track, detector_id)
            valid = _cache_matches(path, panel_path, expected)
        except (KeyError, IndexError, OSError, StopIteration, TypeError, ValueError):
            valid = False
        if not valid:
            stale.append(path.name)
    if stale:
        names = ", ".join(stale)
        raise RuntimeError(
            "Hay métricas ausentes u obsoletas: "
            f"{names}. Ejecuta de nuevo el benchmark antes de comparar."
        )
    return pd.concat(frames, ignore_index=True)
