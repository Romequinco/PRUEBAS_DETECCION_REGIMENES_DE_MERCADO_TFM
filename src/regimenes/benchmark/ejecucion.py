"""Ejecutor reproducible y reanudable del benchmark de detectores v2.

Ejecuta la matriz pista × detector (secuencial o en paralelo por procesos
``spawn``). La huella de caché y la lectura de métricas viven en
``regimenes.benchmark.cache``; el **criterio de ranking de detección** de
ADR-003 (``rank_detection``) y el ranking descriptivo heredado
(``rank_within_track``) en ``regimenes.evaluacion.ranking``.

Uso por línea de comandos (desde la raíz del repo)::

    # matriz completa en paralelo (4 procesos); manifest y run_status al final
    python -m regimenes.benchmark --track A B --jobs 4
    # un único detector (p. ej. en otra terminal); no toca manifest/run_status
    python -m regimenes.benchmark --track A --detector D04
    # consolidar manifest.json y run_status.csv a partir de status/*.json
    python -m regimenes.benchmark --consolidate --track A B
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import time
from typing import Iterable, Iterator
import warnings

import numpy as np
import pandas as pd

from regimenes import evaluacion as ev
from regimenes.benchmark.cache import (
    DEFAULT_OUTPUT,
    ROOT,
    _cache_fingerprint,
    _cache_matches,
    _display_path,
    _metric_path,
    _panel_path,
    _read_verified_cache,
    _sp500_path,
    _status_path,
    _write_atomic,
    _write_manifest,
    load_benchmark_spec,
    processed_available,
    processed_paths,
)
from regimenes.detectores import DetectorSpec, detector_specs
from regimenes.evaluacion.metricas import detection_summary


DEFAULT_TRAIN_DAYS = {"A": 252 * 8, "B": 252 * 3}


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
            f"Faltan {names}. Ejecuta `python -m regimenes.datos` y después "
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
    with _thread_limits(threads_per_job):
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
