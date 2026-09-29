"""Ejecutor reproducible y reanudable del benchmark de detectores v2."""

from __future__ import annotations

import ast
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
from importlib import metadata
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Iterable
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
CACHE_SCHEMA_VERSION = 2


@lru_cache(maxsize=None)
def _sha256_file_version(path_text: str, size: int, mtime_ns: int) -> str:
    """Hash cacheado mientras tamaño y mtime del archivo no cambien."""
    del size, mtime_ns
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    path = Path(path).resolve()
    stat = path.stat()
    return _sha256_file_version(str(path), stat.st_size, stat.st_mtime_ns)


def _relative_hashes(paths: Iterable[Path]) -> dict[str, str]:
    return {
        path.resolve().relative_to(ROOT).as_posix(): _sha256_file(path)
        for path in sorted({Path(item).resolve() for item in paths})
    }


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
    legacy_detectors = ROOT / "capa1_exploracion" / "detectors"
    official_detectors = [
        path for path in legacy_detectors.glob("*.py")
        if path.name != "hsmm_tstudent.py"
    ]
    return [
        ROOT / "src" / "evaluation.py",
        ROOT / "src" / "detectors" / "registry.py",
        ROOT / "capa1_exploracion" / "src" / "detector_base.py",
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

    metric_path.parent.mkdir(parents=True, exist_ok=True)
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metric_path, index=False)
    # ``stability_panel`` vive en attrs y ya se ha resumido en la métrica de
    # estabilidad. PyArrow no puede serializar un DataFrame dentro de metadata.
    panel_to_save = wf_panel.copy()
    panel_to_save.attrs = {}
    panel_to_save.to_parquet(panel_path)
    return metrics, {
        "pista": track,
        "id": spec.detector_id,
        "estado": "ok",
        "segundos": elapsed,
        "detalle": _display_path(metric_path),
    }


def run_benchmark(
    *,
    tracks: Iterable[str] = ("A", "B"),
    detector_ids: Iterable[str] | None = None,
    output_dir: Path = DEFAULT_OUTPUT,
    train_days: dict[str, int] | None = None,
    force: bool = False,
    continue_on_error: bool = True,
    cache_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ejecuta la matriz pista×detector con checkpoint después de cada ajuste.

    ``cache_only=True`` es un modo de solo lectura: nunca ajusta modelos ni
    escribe ``manifest.json``/``run_status.csv``. Cada combinación cuya caché no
    se verifique con la huella actual queda con estado ``sin_cache``. Sirve para
    abrir el notebook 04 en otra máquina sin lanzar horas de cálculo ni
    sobrescribir los artefactos versionados.
    """
    tracks = [track.upper() for track in tracks]
    selected = set(detector_ids or [f"D{i:02d}" for i in range(1, 13)])
    output_dir = Path(output_dir)
    train_days = {**DEFAULT_TRAIN_DAYS, **(train_days or {})}
    if force and cache_only:
        raise ValueError("force=True y cache_only=True son incompatibles.")
    if cache_only:
        return _read_verified_cache(output_dir, tracks, selected, train_days)
    _write_manifest(output_dir, train_days, tracks)
    statuses: list[dict] = []
    metric_frames: list[pd.DataFrame] = []
    status_path = output_dir / "run_status.csv"

    for track in tracks:
        for spec in detector_specs(track):
            if spec.detector_id not in selected:
                continue
            try:
                metrics, status = run_one(
                    track,
                    spec,
                    output_dir=output_dir,
                    train_days=train_days[track],
                    force=force,
                )
                metric_frames.append(metrics)
            except Exception as exc:  # noqa: BLE001 - el estado debe persistir y continuar
                status = {
                    "pista": track,
                    "id": spec.detector_id,
                    "estado": "error",
                    "segundos": np.nan,
                    "detalle": f"{type(exc).__name__}: {exc}",
                }
                if not continue_on_error:
                    statuses.append(status)
                    pd.DataFrame(statuses).to_csv(status_path, index=False)
                    raise
            statuses.append(status)
            pd.DataFrame(statuses).to_csv(status_path, index=False)

    metrics_all = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    return metrics_all, pd.DataFrame(statuses)


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
    """Ranking descriptivo sobre cinco ejes, calculado dentro de cada pista.

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
