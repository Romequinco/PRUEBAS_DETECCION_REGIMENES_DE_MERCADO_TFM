"""Evaluación de variantes detector × menú sobre un panel dado (ADR-008).

Repite el protocolo de :func:`regimenes.benchmark.ejecucion.run_one` (mismo juez, mismo
walk-forward, mismo ajuste final y misma detección por evento) pero:

- recibe el panel y el índice OOS en vez de leerlos de ``data/processed``, para poder
  comparar menús sobre el MISMO panel (por ejemplo, uno construido en memoria);
- no escribe nada ni usa la caché por huella.

Vive aparte de ``ejecucion`` a propósito: ``run_one``, ``load_track_panel`` y
``common_track_index`` forman parte de la huella de la caché del benchmark, y tocarlos
invalidaría los 24 resultados oficiales.
"""

from __future__ import annotations

import argparse
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from regimenes import evaluacion as ev
from regimenes import features as ft
from regimenes import rutas
from regimenes.benchmark.cache import _sp500_path
from regimenes.benchmark.ejecucion import DEFAULT_TRAIN_DAYS, common_track_index, configure_evaluation
from regimenes.detectores.menus import variant_specs
from regimenes.detectores.registry import DetectorSpec
from regimenes.evaluacion.metricas import detection_summary
from regimenes.evaluacion.ranking import detection_score


def evaluar_variante(track: str, spec: DetectorSpec, panel: pd.DataFrame, index: pd.DatetimeIndex,
                     *, train_days: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Métricas (una fila, esquema de ``run_one`` + ``score_deteccion``) y panel OOS.

    ``panel`` debe traer las columnas de ``spec.features`` y ``SP500_ret``; ``index`` es
    el índice común de la comparación (el mismo para todas las variantes que se comparan).
    """
    track = track.upper()
    min_train = int(train_days or DEFAULT_TRAIN_DAYS[track])
    configure_evaluation(track)
    faltan = sorted(set(spec.features) - set(panel.columns))
    if faltan:
        raise KeyError(f"{track}/{spec.detector_id}: faltan columnas {faltan}")
    X = panel.loc[index, list(spec.features)]
    if X.isna().any().any():
        raise ValueError(f"{track}/{spec.detector_id}: NaN dentro del índice común")
    market_returns = panel["SP500_ret"].reindex(X.index)

    started = time.perf_counter()
    wf_panel = ev.walk_forward(spec.factory, X, market_returns=market_returns, train_size=min_train,
                               min_train=min_train, step=spec.step, expanding=spec.expanding)
    final = spec.factory()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*label_states_economically.*")
        warnings.filterwarnings("ignore", message=r".*market_returns.*")
        final.fit(X)
    final.label_states_economically(X, market_returns=market_returns)
    result = ev.evaluate(final, wf_panel, market_returns=market_returns, X_full=X)
    metrics = ev.results_table([result])
    flags = wf_panel["state"].eq(final.crisis_state)
    for key, value in detection_summary(flags, dict(ev.CRISIS_WINDOWS)).items():
        metrics[key] = value
    metrics["score_deteccion"] = detection_score(metrics["det_precision"], metrics["det_event_recall"])
    for key, value in reversed(list({
        "id": spec.detector_id, "pista": track, "n_features_input": len(spec.features),
        "features_input": "|".join(spec.features), "train_days": min_train, "refit_step": spec.step,
        "elapsed_seconds": time.perf_counter() - started,
    }.items())):
        metrics.insert(0, key, value)
    wf_panel = wf_panel.copy()
    wf_panel.attrs = {}
    return metrics, wf_panel


# --------------------------------------------------------------------------- #
# Comparación N frente a T sobre un mismo panel construido en memoria (prueba de humo)
# --------------------------------------------------------------------------- #
# Los detectores más lentos (D04, D09) quedan fuera: esto es una prueba de humo.
TRABAJOS_HUMO = (("B", "D02"), ("B", "D10"), ("B", "D03"), ("B", "D08"), ("A", "D02"), ("A", "D03"))
_PANELES: dict[str, tuple[pd.DataFrame, pd.DatetimeIndex]] = {}


def panel_en_memoria(track: str) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Panel de la pista con ``SP500_ret`` y termómetros, e índice común, desde ``data/raw``.

    No lee ni escribe ``data/processed`` ni verifica sha256: sirve para comparar menús entre
    sí sobre los mismos datos, no para reproducir las cifras oficiales.
    """
    track = track.upper()
    if track in _PANELES:
        return _PANELES[track]
    cov = pd.read_csv(rutas.DATA_RAW / "coverage_report.csv")
    fuente = dict(zip(cov["nombre"], cov["fuente"]))
    spec = yaml.safe_load(rutas.BENCHMARK_SPEC.read_text(encoding="utf-8"))[f"pista_{track}"]
    raw = {}
    for nombre in spec["series_features"]:
        ruta = rutas.DATA_RAW / str(fuente.get(nombre)) / f"{nombre}.parquet"
        if ruta.exists():
            raw[nombre] = ft.seleccionar_raw(pd.read_parquet(ruta), nombre)
    bloques = ft.construir_paneles(raw, spec["ventana_inicio"], spec["ventana_fin"])
    mensual = bloques["mensual"][[c for c in bloques["mensual"] if not c.endswith("_edad_dias")]]
    panel = bloques["diaria"].join(mensual)
    sp500 = pd.read_parquet(_sp500_path())["SP500"].sort_index().astype(float)
    panel["SP500_ret"] = np.log(sp500 / sp500.shift(1)).reindex(panel.index)
    termometros, _ = ft.termometros_pista(raw, track)
    panel = panel.join(termometros)
    # Mismo índice para N y T: el común de los 12 originales, sin NaN en los termómetros.
    index = common_track_index(track, panel)
    index = index[panel.loc[index, list(termometros.columns)].notna().all(axis=1)]
    _PANELES[track] = (panel, index)
    return _PANELES[track]


def _trabajo_humo(args: tuple[str, str, str, str]) -> str:
    track, detector, menu, salida = args
    destino = Path(salida) / f"{track}_{detector}_{menu}.csv"
    if destino.exists():
        return f"{destino.name}: ya estaba"
    warnings.filterwarnings("ignore")
    panel, index = panel_en_memoria(track)
    spec = next(s for s in variant_specs(track, menu) if s.detector_id.split("-")[0] == detector)
    metrics, wf = evaluar_variante(track, spec, panel, index)
    metrics.insert(0, "menu", menu)
    metrics.insert(0, "base", detector)
    wf.to_parquet(destino.with_suffix(".parquet"))
    metrics.to_csv(destino, index=False)      # el CSV va al final: marca el trabajo como terminado
    return f"{destino.name}: score {metrics.loc[0, 'score_deteccion']:.3f}"


def comparar_menus_humo(salida: Path, *, jobs: int = 3) -> None:
    """Ejecuta N y T de ``TRABAJOS_HUMO`` en paralelo; reanudable (salta los CSV ya escritos)."""
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    trabajos = [(t, d, m, str(salida)) for t, d in TRABAJOS_HUMO for m in ("N", "T")]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for linea in pool.map(_trabajo_humo, trabajos):
            print(linea, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--salida", type=Path, default=rutas.RESULTS_DETECTORES / "menus_humo")
    parser.add_argument("--jobs", type=int, default=3)
    argumentos = parser.parse_args()
    comparar_menus_humo(argumentos.salida, jobs=argumentos.jobs)
