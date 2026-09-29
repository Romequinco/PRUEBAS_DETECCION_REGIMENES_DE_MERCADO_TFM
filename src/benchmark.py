"""Ejecutor reproducible y reanudable del benchmark de detectores v2.

Además de ejecutar la matriz pista × detector (secuencial o en paralelo por
procesos), este módulo contiene el **criterio de ranking de detección** de
ADR-003 (:func:`rank_detection`) y el ranking descriptivo heredado
(:func:`rank_within_track`, ``rank_medio``), que se conserva solo para comparar.

Uso por línea de comandos (desde la raíz del repo)::

    # matriz completa en paralelo (4 procesos); manifest y run_status al final
    python -m src.benchmark --track A B --jobs 4
    # un único detector (p. ej. en otra terminal); no toca manifest/run_status
    python -m src.benchmark --track A --detector D04
    # consolidar manifest.json y run_status.csv a partir de status/*.json
    python -m src.benchmark --consolidate --track A B
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
from importlib import metadata
import importlib.machinery
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from typing import Iterable, Iterator
import warnings

import numpy as np
import pandas as pd
import yaml

from src import evaluation as ev
from src.detectors import DetectorSpec, detector_specs


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DEFAULT_OUTPUT = ROOT / "results" / "benchmark_v2"
DEFAULT_TRAIN_DAYS = {"A": 252 * 8, "B": 252 * 3}
# v3: huella por CONTENIDO (parquet/yaml/texto normalizado) y detector_base resuelto.
CACHE_SCHEMA_VERSION = 3
LEGACY_ROOT = ROOT / "capa1_exploracion"

# ``src`` es un paquete de espacio de nombres (sin __init__.py) que también existe
# en ``capa1_exploracion/src``. ``DetectorSpec.factory`` inserta capa1_exploracion
# al principio de sys.path, y a partir de ese momento cualquier ``src.<mod>`` aún
# no importado se resolvería en la copia de Capa 1. El juez debe ser siempre el
# de la raíz: si no lo es, el proceso se ha arrancado con un sys.path contaminado.
if Path(ev.__file__).resolve() != (ROOT / "src" / "evaluation.py").resolve():
    raise ImportError(
        f"src.evaluation se resolvió en {ev.__file__}, no en {ROOT / 'src'}. "
        "Arranca el proceso sin capa1_exploracion al principio de sys.path."
    )


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
    """Copia de ``detector_base.py`` que importan realmente los detectores.

    Los detectores de ``capa1_exploracion/detectors`` hacen
    ``from src.detector_base import RegimeDetector``. Como ``src`` es un paquete
    de espacio de nombres y ``DetectorSpec.factory`` antepone ``capa1_exploracion``
    a sys.path, gana ``capa1_exploracion/src/detector_base.py`` (comprobado:
    ``sys.modules['src.detector_base'].__file__`` tras crear un detector). Si ya
    está importado se usa ese archivo; si no, se reproduce el mismo orden.
    """
    loaded = sys.modules.get("src.detector_base")
    loaded_file = getattr(loaded, "__file__", None)
    if loaded_file:
        return Path(loaded_file).resolve()
    search = [str(LEGACY_ROOT / "src"), str(ROOT / "src")]
    spec = importlib.machinery.PathFinder.find_spec("detector_base", search)
    if spec is None or not spec.origin:
        raise FileNotFoundError("No se encontró detector_base.py para la huella de caché.")
    return Path(spec.origin).resolve()


def _runtime_versions() -> dict[str, str]:
    versions = {"python": ".".join(map(str, sys.version_info[:3]))}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "hmmlearn", "arch", "torch", "jumpmodels"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _benchmark_model_sha256() -> str:
    """Hash de la lógica que puede cambiar predicciones o métricas del modelo.

    Excluye deliberadamente carga de resultados, ranking, manifiesto y mensajes:
    modificar esas capas no debe forzar horas de reajuste numérico.
    """
    selected_names = {
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
    }
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    selected = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in selected_names
    ]
    found = {node.name for node in selected}
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
    path = ROOT / "data" / "benchmark_spec.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def configure_evaluation(track: str, benchmark: dict | None = None) -> None:
    """Carga en el juez solo las labels congeladas de una pista."""
    track = track.upper()
    benchmark = benchmark or load_benchmark_spec()
    crisis = {
        name: tuple(bounds)
        for name, bounds in benchmark["crisis_windows"][f"pista_{track}"].items()
    }
    false_positive = {
        name: tuple(info["ventana"])
        for name, info in benchmark["false_positive_windows"].items()
        if isinstance(info, dict)
        and info.get("en_catalogo") is False
        and "ventana" in info
    }
    troughs = {
        name: benchmark["drawdown_troughs"][name]
        for name in crisis
    }
    ev.configure_event_windows(crisis, false_positive, troughs)


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
    candidates = sorted((ROOT / "data" / "raw").glob("*/SP500.parquet"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            "No se encontró de forma unívoca data/raw/<fuente>/SP500.parquet. "
            "Regenera la descarga antes del benchmark."
        )
    return candidates[0]


def _cache_code_paths() -> list[Path]:
    legacy_detectors = LEGACY_ROOT / "detectors"
    official_detectors = [
        path for path in legacy_detectors.glob("*.py")
        if path.name != "hsmm_tstudent.py"
    ]
    # Se hashean las DOS copias de detector_base (raíz y Capa 1): hoy gana la de
    # Capa 1 (ver ``_resolved_detector_base``), pero un cambio de sys.path podría
    # invertirlo. Además la huella registra cuál se resuelve.
    detector_bases = [
        path for path in (ROOT / "src" / "detector_base.py",
                          LEGACY_ROOT / "src" / "detector_base.py")
        if path.is_file()
    ]
    return [
        ROOT / "src" / "evaluation.py",
        ROOT / "src" / "detectors" / "__init__.py",
        ROOT / "src" / "detectors" / "registry.py",
        _resolved_detector_base(),
        *detector_bases,
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
            ROOT / "data" / "benchmark_spec.yaml",
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


def load_track_panel(track: str) -> pd.DataFrame:
    """Une los bloques diario y mensual ya alineados por 03_preprocesado.

    Las columnas ``*_edad_dias`` son metadatos de disponibilidad, no señales del
    detector, y se excluyen. Los NaN no se imputan.
    """
    daily_path, monthly_path = processed_paths(track)
    missing = [path for path in (daily_path, monthly_path) if not path.exists()]
    if missing:
        names = ", ".join(str(path.relative_to(ROOT)) for path in missing)
        raise FileNotFoundError(
            f"Faltan {names}. Ejecuta `python -m src.ingest.download` y después "
            "03_preprocesado.ipynb."
        )
    daily = pd.read_parquet(daily_path).sort_index()
    monthly = pd.read_parquet(monthly_path).sort_index()
    monthly = monthly[[c for c in monthly if not c.endswith("_edad_dias")]]
    duplicated = set(daily).intersection(monthly)
    if duplicated:
        raise ValueError(f"Columnas duplicadas diaria/mensual: {sorted(duplicated)}")
    panel = pd.concat([daily, monthly], axis=1, join="inner").sort_index()
    panel.index = pd.to_datetime(panel.index)
    # Los modelos de varianza necesitan el retorno en escala natural, no su z-score
    # expanding. Se deriva causalmente de la espina SP500 cruda y se trata como
    # columna técnica común, no como una feature seleccionada por el EDA.
    sp500_path = _sp500_path()
    sp500_frame = pd.read_parquet(sp500_path)
    if "SP500" not in sp500_frame:
        raise KeyError(f"{sp500_path} no contiene la columna SP500.")
    sp500 = sp500_frame["SP500"].sort_index().astype(float)
    raw_return = np.log(sp500 / sp500.shift(1)).rename("SP500_ret")
    panel["SP500_ret"] = raw_return.reindex(panel.index)
    return panel.replace([np.inf, -np.inf], np.nan)


def common_track_index(track: str, panel: pd.DataFrame | None = None) -> pd.DatetimeIndex:
    """Índice completo común a los doce detectores de una pista.

    Sin este recorte previo, un detector univariante empezaría antes que uno
    multivariante por tener un warm-up más corto. El benchmark exige exactamente
    las mismas fechas OOS: se elimina una sola vez el warm-up de la unión de todas
    las features declaradas y después cada modelo selecciona sus columnas.
    """
    track = track.upper()
    panel = load_track_panel(track) if panel is None else panel
    required = sorted({
        feature
        for spec in detector_specs(track)
        for feature in spec.features
    })
    missing = sorted(set(required) - set(panel.columns))
    if missing:
        raise KeyError(f"{track}: faltan features del índice común: {missing}")
    return panel.loc[:, required].dropna(how="any").index


def _dependency_for(detector_id: str) -> str | None:
    return {"D06": "arch", "D09": "jumpmodels"}.get(detector_id)


def preflight(track: str, specs: Iterable[DetectorSpec] | None = None) -> pd.DataFrame:
    """Comprueba datos, columnas y dependencias sin ajustar ningún modelo."""
    track = track.upper()
    specs = list(specs or detector_specs(track))
    if not processed_available(track):
        return pd.DataFrame([
            {
                "pista": track,
                "id": spec.detector_id,
                "datos_ok": False,
                "features_ok": False,
                "dependencia_ok": False,
                "detalle": "faltan paneles data/processed",
            }
            for spec in specs
        ])
    panel = load_track_panel(track)
    rows = []
    for spec in specs:
        missing = sorted(set(spec.features) - set(panel.columns))
        dependency = _dependency_for(spec.detector_id)
        dep_ok = dependency is None or importlib.util.find_spec(dependency) is not None
        rows.append({
            "pista": track,
            "id": spec.detector_id,
            "datos_ok": True,
            "features_ok": not missing,
            "dependencia_ok": dep_ok,
            "detalle": (
                f"faltan features: {missing}" if missing
                else f"falta dependencia: {dependency}" if not dep_ok
                else "OK"
            ),
        })
    return pd.DataFrame(rows)


def _metric_path(output_dir: Path, track: str, detector_id: str) -> Path:
    return output_dir / "metrics" / f"{track}_{detector_id}.csv"


def _panel_path(output_dir: Path, track: str, detector_id: str) -> Path:
    return output_dir / "panels" / f"{track}_{detector_id}.parquet"


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _write_manifest(
    output_dir: Path, train_days: dict[str, int], tracks: Iterable[str]
) -> None:
    benchmark_path = ROOT / "data" / "benchmark_spec.yaml"
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


def run_one(
    track: str,
    spec: DetectorSpec,
    *,
    output_dir: Path = DEFAULT_OUTPUT,
    train_days: int | None = None,
    force: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Ejecuta y guarda un detector; devuelve su fila métrica y su estado."""
    track = track.upper()
    output_dir = Path(output_dir)
    metric_path = _metric_path(output_dir, track, spec.detector_id)
    panel_path = _panel_path(output_dir, track, spec.detector_id)
    min_train = int(train_days or DEFAULT_TRAIN_DAYS[track])
    cache_fingerprint = _cache_fingerprint(track, spec, min_train)
    if not force and _cache_matches(metric_path, panel_path, cache_fingerprint):
        cached = pd.read_csv(metric_path)
        return cached, {
            "pista": track,
            "id": spec.detector_id,
            "estado": "cache",
            "segundos": 0.0,
            "detalle": f"caché verificada: {_display_path(metric_path)}",
        }

    dependency = _dependency_for(spec.detector_id)
    if dependency and importlib.util.find_spec(dependency) is None:
        raise ModuleNotFoundError(
            f"{spec.detector_id} requiere {dependency!r}. Ejecuta: pip install -r requirements.txt"
        )

    configure_evaluation(track)
    full_panel = load_track_panel(track)
    missing = sorted(set(spec.features) - set(full_panel.columns))
    if missing:
        raise KeyError(f"{track}/{spec.detector_id}: faltan features {missing}")
    common_index = common_track_index(track, full_panel)
    X = full_panel.loc[common_index, list(spec.features)]
    if len(X) <= min_train:
        raise ValueError(
            f"{track}/{spec.detector_id}: {len(X)} filas válidas <= train inicial {min_train}."
        )
    market_returns = full_panel["SP500_ret"].reindex(X.index)

    started = time.perf_counter()
    wf_panel = ev.walk_forward(
        spec.factory,
        X,
        market_returns=market_returns,
        train_size=min_train,
        min_train=min_train,
        step=spec.step,
        expanding=spec.expanding,
    )

    # Ajuste final solo para score/AIC/BIC. Las métricas de comportamiento utilizan
    # exclusivamente las etiquetas OOS ya fijadas en wf_panel.
    final_detector = spec.factory()
    # Algunos detectores fijan dentro de ``fit`` un orden económico provisional
    # sin retornos. Se sobrescribe justo después con SP500_ret; silenciamos ese
    # aviso redundante para que no parezca un problema del resultado OOS.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*label_states_economically.*")
        warnings.filterwarnings("ignore", message=r".*market_returns.*")
        final_detector.fit(X)
    final_detector.label_states_economically(X, market_returns=market_returns)
    result = ev.evaluate(
        final_detector,
        wf_panel,
        market_returns=market_returns,
        X_full=X,
    )
    elapsed = time.perf_counter() - started
    metrics = ev.results_table([result])
    # Detección por evento (ADR-003): se guarda en el CSV porque los paneles OOS
    # no se versionan y el ranking debe poder reconstruirse solo desde metrics/.
    crisis_flags = wf_panel["state"].eq(final_detector.crisis_state)
    for key, value in detection_summary(crisis_flags, dict(ev.CRISIS_WINDOWS)).items():
        metrics[key] = value
    metadata = {
        "id": spec.detector_id,
        "pista": track,
        "familia": spec.family,
        "variante_v2": spec.variant,
        "n_features_input": len(spec.features),
        "features_input": "|".join(spec.features),
        "train_days": min_train,
        "refit_step": spec.step,
        "expanding": spec.expanding,
        "cache_fingerprint": cache_fingerprint,
        "elapsed_seconds": elapsed,
    }
    for key, value in reversed(list(metadata.items())):
        metrics.insert(0, key, value)

    # ``stability_panel`` vive en attrs y ya se ha resumido en la métrica de
    # estabilidad. PyArrow no puede serializar un DataFrame dentro de metadata.
    panel_to_save = wf_panel.copy()
    panel_to_save.attrs = {}
    # Escritura atómica (tmp + replace): un proceso interrumpido no deja un CSV o
    # un panel a medias que luego pase por caché. El panel va primero: el CSV
    # (con la huella) es lo que da la combinación por terminada.
    _write_atomic(panel_path, lambda tmp: panel_to_save.to_parquet(tmp))
    _write_atomic(metric_path, lambda tmp: metrics.to_csv(tmp, index=False))
    return metrics, {
        "pista": track,
        "id": spec.detector_id,
        "estado": "ok",
        "segundos": elapsed,
        "detalle": _display_path(metric_path),
    }


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


def _spec_for(track: str, detector_id: str) -> DetectorSpec:
    track = track.upper()
    detector_id = detector_id.upper()
    for spec in detector_specs(track):
        if spec.detector_id == detector_id:
            return spec
    raise KeyError(f"No existe el detector {detector_id} en la pista {track}.")


def run_job(
    track: str,
    detector_id: str,
    *,
    output_dir: Path = DEFAULT_OUTPUT,
    train_days: int | None = None,
    force: bool = False,
) -> dict:
    """Ejecuta UNA combinación pista/detector y deja su estado en ``status/``.

    Es la unidad de trabajo del modo paralelo y de la CLI por detector. Solo
    escribe archivos propios de la combinación (``metrics/<P>_<D>.csv``,
    ``panels/<P>_<D>.parquet`` y ``status/<P>_<D>.json``), así que varias
    instancias pueden correr a la vez sobre el mismo ``output_dir`` sin
    carreras. ``manifest.json`` y ``run_status.csv`` los escribe un único
    proceso: :func:`run_benchmark` o :func:`consolidate_run`.
    """
    track = track.upper()
    detector_id = detector_id.upper()
    output_dir = Path(output_dir)
    started = datetime.now(timezone.utc).isoformat()
    try:
        _, status = run_one(
            track,
            _spec_for(track, detector_id),
            output_dir=output_dir,
            train_days=train_days,
            force=force,
        )
    except Exception as exc:  # noqa: BLE001 - el estado debe persistir
        status = {
            "pista": track,
            "id": detector_id,
            "estado": "error",
            "segundos": np.nan,
            "detalle": f"{type(exc).__name__}: {exc}",
        }
    status = {
        **status,
        "inicio_utc": started,
        "fin_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
    }
    payload = json.dumps(status, ensure_ascii=False, indent=2, default=str)
    _write_atomic(
        _status_path(output_dir, track, detector_id),
        lambda tmp: Path(tmp).write_text(payload, encoding="utf-8"),
    )
    return status


@contextmanager
def _clean_sys_path_for_workers() -> Iterator[None]:
    """sys.path que heredarán los procesos hijos (``spawn`` copia el del padre).

    Si el padre ya creó algún detector, ``capa1_exploracion`` está al principio
    de sys.path y el hijo resolvería ``src.evaluation`` en la copia de Capa 1.
    Se retira mientras vive el pool y se garantiza la raíz del repo.
    """
    original = list(sys.path)
    legacy = {str(LEGACY_ROOT), str(LEGACY_ROOT.resolve())}
    sys.path[:] = [item for item in sys.path if item not in legacy]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        yield
    finally:
        sys.path[:] = original


@contextmanager
def _thread_limits(threads: int | None) -> Iterator[None]:
    """Limita hilos BLAS/OpenMP de los hijos para no sobre-suscribir la CPU."""
    if not threads:
        yield
        return
    names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    previous = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ[name] = str(int(threads))
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _write_run_status(output_dir: Path, statuses: list[dict]) -> None:
    columns = ["pista", "id", "estado", "segundos", "detalle"]
    frame = pd.DataFrame(statuses)
    if frame.empty:
        frame = pd.DataFrame(columns=columns)
    frame = frame[[c for c in columns if c in frame] + [c for c in frame if c not in columns]]
    frame = frame.sort_values(["pista", "id"]).reset_index(drop=True)
    _write_atomic(Path(output_dir) / "run_status.csv",
                  lambda tmp: frame.to_csv(tmp, index=False))


def _collect_metrics(output_dir: Path, statuses: list[dict]) -> pd.DataFrame:
    frames = [
        pd.read_csv(_metric_path(output_dir, s["pista"], s["id"]))
        for s in statuses
        if s.get("estado") in {"ok", "cache"}
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_benchmark(
    *,
    tracks: Iterable[str] = ("A", "B"),
    detector_ids: Iterable[str] | None = None,
    output_dir: Path = DEFAULT_OUTPUT,
    train_days: dict[str, int] | None = None,
    force: bool = False,
    continue_on_error: bool = True,
    cache_only: bool = False,
    n_jobs: int = 1,
    threads_per_job: int | None = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ejecuta la matriz pista×detector con checkpoint después de cada ajuste.

    ``cache_only=True`` es un modo de solo lectura: nunca ajusta modelos ni
    escribe ``manifest.json``/``run_status.csv``. Cada combinación cuya caché no
    se verifique con la huella actual queda con estado ``sin_cache``. Sirve para
    abrir el notebook 04 en otra máquina sin lanzar horas de cálculo ni
    sobrescribir los artefactos versionados.

    ``n_jobs > 1`` reparte las combinaciones en procesos independientes
    (``spawn``), empezando por los detectores lentos. Cada proceso escribe solo
    su CSV, su panel y su ``status/<P>_<D>.json``; este proceso (el único
    escritor) escribe ``manifest.json`` al inicio y ``run_status.csv`` a medida
    que terminan. ``threads_per_job`` fija los hilos BLAS/OpenMP de cada hijo
    en modo paralelo (``None`` = no tocar el entorno); en modo secuencial no se
    usa.
    """
    tracks = [track.upper() for track in tracks]
    selected = {item.upper() for item in (detector_ids or [f"D{i:02d}" for i in range(1, 13)])}
    output_dir = Path(output_dir)
    train_days = {**DEFAULT_TRAIN_DAYS, **(train_days or {})}
    if force and cache_only:
        raise ValueError("force=True y cache_only=True son incompatibles.")
    if cache_only:
        return _read_verified_cache(output_dir, tracks, selected, train_days)
    n_jobs = max(1, int(n_jobs))
    jobs = [
        (track, spec)
        for track in tracks
        for spec in detector_specs(track)
        if spec.detector_id in selected
    ]
    _write_manifest(output_dir, train_days, tracks)
    statuses: list[dict] = []

    if n_jobs == 1:
        for track, spec in jobs:
            status = run_job(track, spec.detector_id, output_dir=output_dir,
                             train_days=train_days[track], force=force)
            statuses.append(status)
            _write_run_status(output_dir, statuses)
            if status["estado"] == "error" and not continue_on_error:
                raise RuntimeError(f"{track}/{spec.detector_id}: {status['detalle']}")
        return _collect_metrics(output_dir, statuses), pd.DataFrame(statuses)

    # Lentos primero (planificación "longest processing time first").
    ordered = sorted(jobs, key=lambda item: (not item[1].slow, item[0], item[1].detector_id))
    failure: dict | None = None
    with _clean_sys_path_for_workers(), _thread_limits(threads_per_job):
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(n_jobs, max(len(ordered), 1)),
                                 mp_context=context) as pool:
            pending = {
                pool.submit(run_job, track, spec.detector_id, output_dir=output_dir,
                            train_days=train_days[track], force=force): (track, spec)
                for track, spec in ordered
            }
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    track, spec = pending.pop(future)
                    try:
                        status = future.result()
                    except Exception as exc:  # noqa: BLE001 - p. ej. proceso hijo muerto
                        status = {"pista": track, "id": spec.detector_id, "estado": "error",
                                  "segundos": np.nan, "detalle": f"{type(exc).__name__}: {exc}"}
                    statuses.append(status)
                    _write_run_status(output_dir, statuses)
                    if status["estado"] == "error" and not continue_on_error and failure is None:
                        failure = status
                        for other in pending:
                            other.cancel()
                if failure is not None:
                    pending = {f: v for f, v in pending.items() if not f.cancelled()}
    if failure is not None:
        raise RuntimeError(f"{failure['pista']}/{failure['id']}: {failure['detalle']}")
    return _collect_metrics(output_dir, statuses), pd.DataFrame(statuses)


def consolidate_run(
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    tracks: Iterable[str] = ("A", "B"),
    train_days: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Reconstruye ``manifest.json`` y ``run_status.csv`` desde ``status/*.json``.

    Para ejecuciones lanzadas detector a detector con la CLI (p. ej. en varias
    terminales): se llama una vez al final, cuando ya no escribe nadie.
    """
    output_dir = Path(output_dir)
    tracks = [track.upper() for track in tracks]
    train_days = {**DEFAULT_TRAIN_DAYS, **(train_days or {})}
    statuses = []
    for path in sorted((output_dir / "status").glob("[AB]_D*.json")):
        status = json.loads(path.read_text(encoding="utf-8"))
        if str(status.get("pista", "")).upper() in tracks:
            statuses.append(status)
    _write_manifest(output_dir, train_days, tracks)
    _write_run_status(output_dir, statuses)
    return pd.DataFrame(statuses)


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


def add_comparison_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Añade agregados comparables sin mezclar pistas.

    - ``mean_crisis_coverage``: media NO ponderada, por evento, de ``cov_<crisis>``
      (fracción de días OOS de la ventana [pico, suelo] marcados como crisis). Las
      crisis sin días OOS en esa pista son NaN y se omiten; una ventana solo
      parcialmente OOS (p. ej. ``bear_1969_70`` en A: 11 de 370 días) cuenta como
      un evento completo. Un volmageddon de 10 días pesa igual que la GFC.
    - ``mean_trap_activation``: media de ``fa_<trampa>`` (fracción de días de la
      trampa marcados como crisis; menor es mejor).
    """
    out = metrics.copy()
    coverage = [
        c for c in out
        if c.startswith("cov_") and not c.endswith(("_lo", "_hi"))
    ]
    traps = [c for c in out if c.startswith("fa_")]
    out["mean_crisis_coverage"] = out[coverage].mean(axis=1, skipna=True)
    out["mean_trap_activation"] = out[traps].mean(axis=1, skipna=True)
    return out


def rank_within_track(metrics: pd.DataFrame) -> pd.DataFrame:
    """[LEGACY] Ranking descriptivo sobre cinco ejes, calculado dentro de cada pista.

    Desde ADR-003 el ranking principal es :func:`rank_detection`; este se
    conserva (``rank_medio``/``puesto_pista`` de ``metrics_master_v2.csv``, y
    ``rank_medio_legacy``/``puesto_legacy`` en ``ranking_v2.csv``) solo para
    comparar con el criterio anterior.

    Cada eje se ordena por separado (1 = mejor); los empates reciben el rango
    medio (``method="average"``) y los NaN van al fondo (p. ej. un
    ``false_alarm_rate`` NaN porque el detector nunca marca crisis).
    ``rank_medio`` es la media simple de los cinco rangos y ``puesto_pista`` su
    rango con ``method="min"`` (los empates comparten puesto).

    Limitación conocida: tres de los cinco ejes (trampas, switching y
    estabilidad) premian la inactividad. Un detector trivial que nunca o siempre
    marca crisis quedaría en mitad de la tabla; el notebook 05 lo muestra con
    líneas base triviales. Es un mapa descriptivo, no una función de utilidad.
    """
    out = add_comparison_metrics(metrics)
    directions = {
        "mean_crisis_coverage": False,
        "false_alarm_rate": True,
        "mean_trap_activation": True,
        "switching_rate": True,
        "label_stability": False,
    }
    # ``mean_regime_duration`` se conserva en la tabla como diagnóstico, pero no
    # entra en el promedio: para una secuencia fija es una transformación monótona
    # del número de cambios y duplicaría exactamente el peso de ``switching_rate``.
    rank_cols = []
    for metric, ascending in directions.items():
        col = f"rank_{metric}"
        out[col] = out.groupby("pista")[metric].rank(
            ascending=ascending, method="average", na_option="bottom"
        )
        rank_cols.append(col)
    out["rank_medio"] = out[rank_cols].mean(axis=1)
    out["puesto_pista"] = out.groupby("pista")["rank_medio"].rank(method="min")
    return out.sort_values(["pista", "puesto_pista", "id"])


# --------------------------------------------------------------------------- #
# Criterio de ranking de detección (ADR-003)
# --------------------------------------------------------------------------- #
# Parámetros explícitos del criterio. Cualquier cambio debe registrarse en la ADR.
DETECTION_DEFAULTS: dict[str, float] = {
    # Un evento cuenta como detectado si hay >= ``min_run`` sesiones OOS
    # CONSECUTIVAS marcadas como crisis dentro de su ventana [pico, suelo]
    # (o todas, si la ventana tiene menos sesiones OOS). 3 = el gatillo de
    # persistencia de ``evaluation.lead_lag`` y de la regla de cambio del TFM.
    "min_run": 3,
    # F-beta entre recall por evento y precisión diaria (beta > 1 prima recall).
    "beta": 1.0,
    # Restricción de falsas alarmas: precisión / tasa base > ``lift_min``
    # (equivale a FAR < FAR de marcar crisis siempre).
    "lift_min": 1.0,
    # Filtros mínimos de persistencia ("switching absurdo"): duración media de
    # régimen >= 1 semana hábil y episodios de crisis de al menos ``min_run``
    # sesiones de media (más cortos que el propio gatillo de detección = parpadeo).
    "min_mean_duration": 5.0,
    "min_crisis_run": 3.0,
    # Empate si el score coincide a estos decimales → desempate por persistencia.
    "score_decimals": 3,
}

DETECTION_LEVELS: dict[int, str] = {
    0: "elegible",
    1: "precision_no_supera_azar",
    2: "parpadeo",
    3: "degenerado",
}


def _longest_true_run(values: np.ndarray) -> int:
    """Longitud de la racha más larga de True consecutivos."""
    values = np.asarray(values, dtype=bool)
    if not values.any():
        return 0
    padded = np.concatenate(([0], values.astype(np.int8), [0]))
    steps = np.diff(padded)
    starts = np.flatnonzero(steps == 1)
    ends = np.flatnonzero(steps == -1)
    return int((ends - starts).max())


def event_detection_table(
    crisis_flags: pd.Series,
    windows: dict[str, tuple[str, str]],
    *,
    min_run: int = 3,
) -> pd.DataFrame:
    """Detección por evento a partir de la señal OOS de crisis (bool por fecha).

    Por crisis: sesiones OOS dentro de ``[pico, suelo]``, sesiones marcadas,
    racha máxima y ``detectada`` (1/0; NaN si la ventana cae fuera del OOS).
    Detectada = racha de ``min(min_run, dias_oos)`` sesiones consecutivas.
    """
    flags = pd.Series(crisis_flags).astype(bool)
    index = pd.DatetimeIndex(flags.index)
    values = flags.to_numpy()
    rows = []
    for name, (start, end) in windows.items():
        mask = (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
        segment = values[mask]
        n_days = int(mask.sum())
        longest = _longest_true_run(segment)
        rows.append({
            "crisis": name,
            "dias_oos": n_days,
            "dias_marcados": int(segment.sum()),
            "racha_max": longest,
            "detectada": float(longest >= min(int(min_run), n_days)) if n_days else np.nan,
        })
    return pd.DataFrame(rows).set_index("crisis")


def detection_summary(
    crisis_flags: pd.Series,
    windows: dict[str, tuple[str, str]],
    *,
    min_run: int = 3,
) -> dict[str, float]:
    """Métricas de detección (columnas ``det_*``) de una señal OOS de crisis.

    - ``det_event_recall``: eventos detectados / eventos con días OOS.
    - ``det_precision``: días marcados dentro de alguna ventana / días marcados
      (= ``1 − false_alarm_rate``; NaN si nunca marca).
    - ``det_base_rate``: fracción de días OOS dentro de ventanas (= precisión de
      marcar crisis siempre); ``det_lift_precision`` = precisión / tasa base.
    - ``det_marked_rate``: fracción de días OOS marcados; ``det_day_recall``:
      fracción de días de ventana marcados (cobertura ponderada por duración).
    - ``det_mean_crisis_run``: duración media (sesiones) de los episodios de crisis.
    - ``det_ev_<crisis>``: 1/0/NaN por evento.
    """
    flags = pd.Series(crisis_flags).astype(bool)
    index = pd.DatetimeIndex(flags.index)
    values = flags.to_numpy()
    table = event_detection_table(flags, windows, min_run=min_run)
    evaluated = table["dias_oos"] > 0
    inside = np.zeros(len(values), dtype=bool)
    for start, end in windows.values():
        inside |= (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
    n_days = len(values)
    n_marked = int(values.sum())
    true_positive = int((values & inside).sum())
    base_rate = float(inside.mean()) if n_days else np.nan
    precision = true_positive / n_marked if n_marked else np.nan
    n_events = int(evaluated.sum())
    n_detected = int(table.loc[evaluated, "detectada"].sum())
    n_episodes = int(np.sum(np.diff(np.concatenate(([0], values.astype(np.int8)))) == 1))
    out: dict[str, float] = {
        "det_min_run": int(min_run),
        "det_n_eventos": n_events,
        "det_n_detectados": n_detected,
        "det_event_recall": n_detected / n_events if n_events else np.nan,
        "det_precision": precision,
        "det_base_rate": base_rate,
        "det_lift_precision": (
            precision / base_rate if n_marked and base_rate and base_rate > 0 else np.nan
        ),
        "det_marked_rate": n_marked / n_days if n_days else np.nan,
        "det_day_recall": true_positive / int(inside.sum()) if inside.any() else np.nan,
        "det_mean_crisis_run": n_marked / n_episodes if n_episodes else np.nan,
    }
    for name, value in table["detectada"].items():
        out[f"det_ev_{name}"] = value
    return out


def track_crisis_windows(track: str, benchmark: dict | None = None) -> dict[str, tuple[str, str]]:
    """Ventanas ``[pico, suelo]`` congeladas de una pista (las del juez)."""
    benchmark = benchmark or load_benchmark_spec()
    return {
        name: tuple(map(str, bounds))
        for name, bounds in benchmark["crisis_windows"][f"pista_{track.upper()}"].items()
    }


def track_false_positive_windows(benchmark: dict | None = None) -> dict[str, tuple[str, str]]:
    benchmark = benchmark or load_benchmark_spec()
    return {
        name: tuple(map(str, info["ventana"]))
        for name, info in benchmark["false_positive_windows"].items()
        if isinstance(info, dict) and info.get("en_catalogo") is False and "ventana" in info
    }


def track_oos_index(
    track: str,
    metrics: pd.DataFrame,
    output_dir: Path = DEFAULT_OUTPUT,
) -> tuple[pd.DatetimeIndex, bool]:
    """Índice OOS común de una pista y si es exacto.

    Exacto: el de cualquier panel OOS de la pista cuyo largo coincide con
    ``n_oos`` (el gate de 05 exige el mismo índice a los 12 detectores).
    Aproximado: sesiones lunes–viernes entre ``oos_start`` y ``oos_end``.
    """
    track = track.upper()
    part = metrics[metrics["pista"].astype(str).str.upper() == track]
    n_oos = int(part["n_oos"].iloc[0]) if "n_oos" in part and len(part) else None
    for path in sorted((Path(output_dir) / "panels").glob(f"{track}_D*.parquet")):
        try:
            index = pd.DatetimeIndex(pd.read_parquet(path, columns=["state"]).index)
        except (OSError, ValueError, KeyError):
            continue
        if n_oos is None or len(index) == n_oos:
            return index, True
    if not len(part):
        raise KeyError(f"No hay métricas de la pista {track}.")
    return pd.bdate_range(part["oos_start"].iloc[0], part["oos_end"].iloc[0]), False


def _detection_from_coverage(
    row: pd.Series,
    oos_index: pd.DatetimeIndex,
    windows: dict[str, tuple[str, str]],
    min_run: int,
) -> dict[str, float]:
    """Aproximación de ``detection_summary`` desde las columnas del CSV.

    Sin panel OOS no se conoce la racha máxima: se exige ``round(cov·n) >=
    min_run`` sesiones marcadas en la ventana, no necesariamente consecutivas
    (cota superior del criterio exacto; coincide con él en 13/14 paneles
    disponibles en la revisión de 2026-09-29).
    """
    inside = np.zeros(len(oos_index), dtype=bool)
    n_days = len(oos_index)
    true_positive = 0.0
    out: dict[str, float] = {"det_min_run": int(min_run)}
    detected: list[float] = []
    for name, (start, end) in windows.items():
        mask = (oos_index >= pd.Timestamp(start)) & (oos_index <= pd.Timestamp(end))
        inside |= mask
        n_window = int(mask.sum())
        coverage = row.get(f"cov_{name}", np.nan)
        if n_window == 0 or pd.isna(coverage):
            out[f"det_ev_{name}"] = np.nan
            continue
        marked = round(float(coverage) * n_window)
        true_positive += float(coverage) * n_window
        flag = float(marked >= min(int(min_run), n_window))
        out[f"det_ev_{name}"] = flag
        detected.append(flag)
    far = row.get("false_alarm_rate", np.nan)
    precision = 1.0 - float(far) if pd.notna(far) else np.nan
    base_rate = float(inside.mean()) if n_days else np.nan
    if pd.isna(far):
        marked_rate = 0.0  # evaluation.false_alarm_rate es NaN solo si nunca marca
    elif precision > 0:
        marked_rate = min(1.0, true_positive / precision / n_days)
    else:
        marked_rate = np.nan
    out.update({
        "det_n_eventos": len(detected),
        "det_n_detectados": int(np.sum(detected)),
        "det_event_recall": float(np.mean(detected)) if detected else np.nan,
        "det_precision": precision,
        "det_base_rate": base_rate,
        "det_lift_precision": precision / base_rate if pd.notna(precision) and base_rate else np.nan,
        "det_marked_rate": marked_rate,
        "det_day_recall": true_positive / inside.sum() if inside.any() else np.nan,
        # nº de episodios de crisis ≈ cambios/2 (exacto con 2 estados; con K > 2
        # los cambios entre estados no-crisis lo hacen conservador).
        "det_mean_crisis_run": _approx_crisis_run(marked_rate, row.get("switching_rate", np.nan), n_days),
    })
    return out


def _approx_crisis_run(marked_rate: float, switching_rate: float, n_days: int) -> float:
    if pd.isna(marked_rate) or marked_rate <= 0:
        return np.nan
    if pd.isna(switching_rate):
        return np.nan
    if switching_rate <= 0:
        return float(marked_rate * n_days)
    return float(max(1.0, 2.0 * marked_rate / switching_rate))


def add_detection_metrics(
    metrics: pd.DataFrame,
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    min_run: int = int(DETECTION_DEFAULTS["min_run"]),
    benchmark: dict | None = None,
) -> pd.DataFrame:
    """Completa las columnas ``det_*`` y ``fuente_deteccion`` fila a fila.

    Prioridad de la fuente: (1) ``csv`` si el CSV ya trae ``det_*`` con el mismo
    ``min_run`` (benchmark re-ejecutado con este módulo); (2) ``panel`` si
    existe ``panels/<P>_<D>.parquet`` (exacto; crisis = ``n_states − 1``);
    (3) ``cobertura_aprox`` desde ``cov_*`` y ``false_alarm_rate`` (sufijo
    ``_bdate`` si además el índice OOS es aproximado).
    """
    out = metrics.copy().reset_index(drop=True)
    benchmark = benchmark or load_benchmark_spec()
    output_dir = Path(output_dir)
    if "fuente_deteccion" not in out:
        out["fuente_deteccion"] = pd.Series([None] * len(out), dtype=object)
    updates: dict[int, dict[str, float]] = {}
    sources: dict[int, str] = {}
    for track in out["pista"].astype(str).str.upper().unique():
        windows = track_crisis_windows(track, benchmark)
        oos: tuple[pd.DatetimeIndex, bool] | None = None
        for i in out.index[out["pista"].astype(str).str.upper() == track]:
            row = out.loc[i]
            has_csv = (
                pd.notna(row.get("det_event_recall", np.nan))
                and pd.notna(row.get("det_min_run", np.nan))
                and int(row["det_min_run"]) == int(min_run)
            )
            if has_csv:
                sources[i] = row["fuente_deteccion"] if isinstance(row["fuente_deteccion"], str) else "csv"
                continue
            panel_path = _panel_path(output_dir, track, str(row["id"]))
            if panel_path.is_file() and pd.notna(row.get("n_states", np.nan)):
                panel = pd.read_parquet(panel_path, columns=["state"])
                flags = panel["state"].eq(int(row["n_states"]) - 1)
                updates[i] = detection_summary(flags, windows, min_run=min_run)
                sources[i] = "panel"
                continue
            if oos is None:
                oos = track_oos_index(track, out, output_dir)
            updates[i] = _detection_from_coverage(row, oos[0], windows, min_run)
            sources[i] = "cobertura_aprox" if oos[1] else "cobertura_aprox_bdate"
    if updates:
        update_frame = pd.DataFrame.from_dict(updates, orient="index")
        for col in update_frame:
            if col not in out:
                out[col] = np.nan
            out.loc[update_frame.index, col] = update_frame[col].astype(float)
    for i, source in sources.items():
        out.at[i, "fuente_deteccion"] = source
    return out


def detection_score(precision, recall, beta: float = 1.0):
    """F-beta entre precisión diaria y recall por evento (0 si no detecta nada).

    Es el *composite F-score* de Garg et al. (2021): recall a nivel de evento y
    precisión a nivel de instante. NaN de precisión (nunca marca) → 0.
    """
    precision = pd.Series(precision, dtype=float) if not np.isscalar(precision) else precision
    recall = pd.Series(recall, dtype=float) if not np.isscalar(recall) else recall
    b2 = float(beta) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        score = (1 + b2) * precision * recall / (b2 * precision + recall)
    if np.isscalar(score):
        return 0.0 if not np.isfinite(score) else float(score)
    return score.where(np.isfinite(score), 0.0)


def rank_detection(
    metrics: pd.DataFrame,
    *,
    output_dir: Path = DEFAULT_OUTPUT,
    benchmark: dict | None = None,
    min_run: int = int(DETECTION_DEFAULTS["min_run"]),
    beta: float = DETECTION_DEFAULTS["beta"],
    lift_min: float = DETECTION_DEFAULTS["lift_min"],
    min_mean_duration: float = DETECTION_DEFAULTS["min_mean_duration"],
    min_crisis_run: float = DETECTION_DEFAULTS["min_crisis_run"],
    score_decimals: int = int(DETECTION_DEFAULTS["score_decimals"]),
) -> pd.DataFrame:
    """Ranking de detección (ADR-003), dentro de cada pista.

    1. **Niveles** (se ordenan antes que el score):
       0 ``elegible``; 1 ``precision_no_supera_azar`` (lift ≤ ``lift_min``);
       2 ``parpadeo`` (duración media de régimen < ``min_mean_duration`` o
       duración media de los episodios de crisis < ``min_crisis_run``);
       3 ``degenerado`` (salida constante: ``switching_rate == 0`` o nunca marca).
    2. **Score** dentro de cada nivel: ``score_deteccion`` = F-beta entre
       ``det_event_recall`` y ``det_precision``.
    3. **Desempate** (score igual a ``score_decimals``): menor
       ``switching_rate`` y después mayor ``label_stability``.

    ``puesto_deteccion`` comparte puesto solo si nivel, score redondeado y
    desempates coinciden. Se añaden ``rank_medio_legacy`` y ``puesto_legacy``
    (:func:`rank_within_track`) solo como referencia descriptiva.
    """
    out = add_detection_metrics(metrics, output_dir, min_run=min_run, benchmark=benchmark)
    out = add_comparison_metrics(out)
    legacy = rank_within_track(out)[["pista", "id", "rank_medio", "puesto_pista"]].rename(
        columns={"rank_medio": "rank_medio_legacy", "puesto_pista": "puesto_legacy"}
    )
    out = out.merge(legacy, on=["pista", "id"], how="left")
    out["score_deteccion"] = detection_score(out["det_precision"], out["det_event_recall"], beta)
    switching = pd.to_numeric(out["switching_rate"], errors="coerce")
    duration = pd.to_numeric(out["mean_regime_duration"], errors="coerce")
    marked = pd.to_numeric(out["det_marked_rate"], errors="coerce")
    lift = pd.to_numeric(out["det_lift_precision"], errors="coerce")
    degenerate = (
        switching.fillna(0).eq(0)
        | out["det_precision"].isna()
        | marked.le(0)
        | marked.ge(1)
    )
    crisis_run = pd.to_numeric(out["det_mean_crisis_run"], errors="coerce")
    flicker = (
        duration.lt(min_mean_duration) | duration.isna()
        | crisis_run.lt(min_crisis_run) | crisis_run.isna()
    )
    weak = ~lift.gt(lift_min)
    out["nivel"] = np.select([degenerate, flicker, weak], [3, 2, 1], default=0)
    out["nivel_etiqueta"] = out["nivel"].map(DETECTION_LEVELS)
    out["elegible"] = out["nivel"].eq(0)

    keys = pd.DataFrame({
        "nivel": out["nivel"],
        "score": -out["score_deteccion"].round(score_decimals),
        "switching": switching.fillna(np.inf),
        "stability": -pd.to_numeric(out["label_stability"], errors="coerce").fillna(-np.inf),
    })
    order = pd.concat([out[["pista", "id"]], keys], axis=1).sort_values(
        ["pista", "nivel", "score", "switching", "stability", "id"]
    )
    places = pd.Series(np.nan, index=order.index)
    for _, part in order.groupby("pista", sort=False):
        key_cols = part[["nivel", "score", "switching", "stability"]]
        changed = key_cols.ne(key_cols.shift()).any(axis=1).to_numpy()
        position = np.arange(1, len(part) + 1, dtype=float)
        places.loc[part.index] = pd.Series(np.where(changed, position, np.nan)).ffill().to_numpy()
    out["puesto_deteccion"] = places.astype(int)
    out.attrs["detection_params"] = {
        "min_run": int(min_run), "beta": float(beta), "lift_min": float(lift_min),
        "min_mean_duration": float(min_mean_duration), "min_crisis_run": float(min_crisis_run),
        "score_decimals": int(score_decimals),
    }
    return out.sort_values(["pista", "puesto_deteccion", "id"]).reset_index(drop=True)


RANKING_COLUMNS = [
    "pista", "puesto_deteccion", "id", "detector", "familia", "nivel_etiqueta",
    "score_deteccion", "det_event_recall", "det_n_detectados", "det_n_eventos",
    "det_precision", "det_lift_precision", "det_base_rate", "det_marked_rate",
    "det_day_recall", "det_mean_crisis_run", "mean_crisis_coverage", "false_alarm_rate", "mean_trap_activation",
    "switching_rate", "mean_regime_duration", "label_stability", "elapsed_seconds",
    "fuente_deteccion", "rank_medio_legacy", "puesto_legacy",
]


def pareto_mask(frame: pd.DataFrame, maximize: Iterable[str]) -> np.ndarray:
    """True para las filas no dominadas (todas las columnas se maximizan)."""
    values = frame[list(maximize)].to_numpy(dtype=float)
    values = np.where(np.isnan(values), -np.inf, values)
    keep = np.ones(len(values), dtype=bool)
    for i in range(len(values)):
        dominated = (values >= values[i]).all(axis=1) & (values > values[i]).any(axis=1)
        keep[i] = not dominated.any()
    return keep


# --------------------------------------------------------------------------- #
# Líneas base triviales y nulo de azar persistente
# --------------------------------------------------------------------------- #
def _markov_flags(n: int, rate: float, mean_run: float, rng: np.random.Generator) -> np.ndarray:
    """Señal binaria de Markov estacionaria: fracción ``rate`` y rachas de crisis
    de duración media ``mean_run`` (rachas de calma: ``mean_run·(1−rate)/rate``)."""
    if rate <= 0:
        return np.zeros(n, dtype=bool)
    if rate >= 1:
        return np.ones(n, dtype=bool)
    crisis_run = max(float(mean_run), 1.0)
    calm_run = max(crisis_run * (1 - rate) / rate, 1.0)
    flags = np.zeros(n, dtype=bool)
    state = bool(rng.random() < rate)
    position = 0
    while position < n:
        length = int(rng.geometric(1.0 / (crisis_run if state else calm_run)))
        flags[position:position + length] = state
        position += length
        state = not state
    return flags


def _baseline_row(
    flags: np.ndarray,
    oos_index: pd.DatetimeIndex,
    track: str,
    name: str,
    windows: dict[str, tuple[str, str]],
    fp_windows: dict[str, tuple[str, str]],
    min_run: int,
) -> dict:
    states = pd.Series(np.asarray(flags, dtype=int), index=oos_index)
    constant = states.nunique() <= 1
    row: dict = {
        "pista": track, "id": name, "detector": name.lower(), "familia": "Línea base",
        "n_states": 2, "n_oos": len(states),
        "false_alarm_rate": ev.false_alarm_rate(states, 1, windows),
        "switching_rate": ev.switching_rate(states),
        "mean_regime_duration": ev.mean_regime_duration(states),
        # Un modelo constante no cambia al reentrenar; uno aleatorio no tiene
        # estabilidad definida (NaN → al fondo en el eje heredado).
        "label_stability": 1.0 if constant else np.nan,
        "fuente_deteccion": "linea_base",
    }
    row.update({f"cov_{k}": v for k, v in ev.crisis_coverage(states, 1, windows).items()})
    row.update({f"fa_{k}": v for k, v in ev.false_alarm_in_windows(states, 1, fp_windows).items()})
    row.update(detection_summary(states.eq(1), windows, min_run=min_run))
    return row


def trivial_baselines(
    track: str,
    oos_index: pd.DatetimeIndex,
    *,
    rate: float,
    mean_run: float,
    benchmark: dict | None = None,
    min_run: int = int(DETECTION_DEFAULTS["min_run"]),
    n_sims: int = 200,
    seed: int = 20260929,
) -> pd.DataFrame:
    """Cuatro detectores sin información, con el mismo esquema que las métricas.

    - ``SIEMPRE_CRISIS`` / ``SIEMPRE_CALMA``: salidas constantes.
    - ``AZAR_IID``: Bernoulli diario con tasa ``rate`` (una realización fija).
    - ``AZAR_PERSISTENTE``: Markov con tasa ``rate`` y rachas medias
      ``mean_run``; fila = **media** de ``n_sims`` realizaciones, con la
      precisión fijada en su valor esperado (la tasa base, lift = 1): una señal
      independiente de las crisis no puede, en esperanza, concentrarse en ellas.
    """
    track = track.upper()
    benchmark = benchmark or load_benchmark_spec()
    windows = track_crisis_windows(track, benchmark)
    fp_windows = track_false_positive_windows(benchmark)
    n = len(oos_index)
    rng = np.random.default_rng(seed)
    rows = [
        _baseline_row(np.ones(n, bool), oos_index, track, "SIEMPRE_CRISIS", windows, fp_windows, min_run),
        _baseline_row(np.zeros(n, bool), oos_index, track, "SIEMPRE_CALMA", windows, fp_windows, min_run),
        _baseline_row(rng.random(n) < rate, oos_index, track, "AZAR_IID", windows, fp_windows, min_run),
    ]
    sims = pd.DataFrame([
        _baseline_row(_markov_flags(n, rate, mean_run, rng), oos_index, track,
                      "AZAR_PERSISTENTE", windows, fp_windows, min_run)
        for _ in range(int(n_sims))
    ])
    expected = sims.iloc[0].to_dict()
    numeric = sims.select_dtypes("number").mean()
    expected.update(numeric.to_dict())
    base_rate = float(expected["det_base_rate"])
    expected.update({
        "det_precision": base_rate,
        "det_lift_precision": 1.0,
        "false_alarm_rate": 1.0 - base_rate,
        "label_stability": np.nan,
        "fuente_deteccion": f"linea_base_media_{int(n_sims)}_sim",
    })
    rows.append(expected)
    return pd.DataFrame(rows)


def persistent_random_null(
    oos_index: pd.DatetimeIndex,
    windows: dict[str, tuple[str, str]],
    *,
    rate: float,
    mean_run: float,
    n_sims: int = 300,
    seed: int = 0,
    min_run: int = int(DETECTION_DEFAULTS["min_run"]),
    beta: float = DETECTION_DEFAULTS["beta"],
) -> pd.DataFrame:
    """Distribución nula de (recall, precisión, lift, score) para una señal de
    Markov independiente de las crisis con la misma tasa y persistencia."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(int(n_sims)):
        flags = pd.Series(_markov_flags(len(oos_index), rate, mean_run, rng), index=oos_index)
        summary = detection_summary(flags, windows, min_run=min_run)
        rows.append({
            "det_event_recall": summary["det_event_recall"],
            "det_precision": summary["det_precision"],
            "det_lift_precision": summary["det_lift_precision"],
        })
    null = pd.DataFrame(rows)
    null["score_deteccion"] = detection_score(null["det_precision"], null["det_event_recall"], beta)
    return null


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.benchmark",
        description="Benchmark v2 de detectores (secuencial, paralelo o por detector).",
    )
    parser.add_argument("--track", nargs="+", default=["A", "B"], help="pistas: A, B o ambas")
    parser.add_argument("--detector", nargs="+", default=None,
                        help="IDs (D01..D12). Con --jobs 1 solo se ejecutan esas "
                             "combinaciones y NO se escribe manifest/run_status "
                             "(usa --consolidate al final); con --jobs > 1 se "
                             "reparten en procesos y este proceso consolida.")
    parser.add_argument("--jobs", type=int, default=1, help="procesos en paralelo (matriz completa)")
    parser.add_argument("--threads-per-job", type=int, default=1,
                        help="hilos BLAS/OpenMP por proceso con --jobs > 1 (0 = no tocar)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="directorio de salida")
    parser.add_argument("--force", action="store_true", help="ignora la caché y reajusta")
    parser.add_argument("--cache-only", action="store_true", help="solo verifica la caché")
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


if __name__ == "__main__":
    # Se reimporta como ``src.benchmark`` para que las funciones que viajan a los
    # procesos hijos no se serialicen como ``__main__.*``.
    from src.benchmark import main as _module_main

    raise SystemExit(_module_main())
