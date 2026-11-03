"""Ejecución OFICIAL del menú T frente al N (ADR-008 §4–§6), reanudable y en paralelo.

Sustituye a la prueba de humo de :mod:`regimenes.benchmark.variantes` (paneles en memoria
desde crudos incompletos) por la ruta oficial:

- **Panel.** ``data/processed`` verificado, cargado exactamente como lo carga el benchmark
  (:func:`~regimenes.benchmark.ejecucion.load_track_panel`, que añade ``SP500_ret``), más
  los termómetros de :func:`regimenes.features.termometros_pista` (historia previa
  completa, ADR-008 §2.1) calculados desde los crudos de ``data/raw``. Antes de usarlo se
  comprueba el sha256 de los parquet frente a ``data/processed/_meta.json`` y que no falte
  ninguna serie cruda ni ninguna columna de ``configs/bloques.yaml``.
- **Índice OOS (ADR-008 §7, último punto).** El índice de la comparación es el del
  benchmark: :func:`~regimenes.benchmark.ejecucion.common_track_index` sobre el panel
  oficial. Comprobado el 2026-11-02 sobre los datos verificados: los termómetros no tienen
  ningún NaN dentro de ese índice en ninguna de las dos pistas (A: 16.149 días, 1962-03-29
  a 2026-05-29, OOS 14.133; B: 4.815 días, 2007-04-11 a 2026-05-29, OOS 4.059), así que
  **N y T comparten exactamente el mismo OOS y N no se recorta respecto a v2**. Decisión
  conservadora si algún día deja de cumplirse (otros crudos, otro ``bloques.yaml``): no se
  recorta en silencio; :func:`indice_oficial` falla con los días perdidos y sus fechas, y
  habría que re-ejecutar N sobre el índice recortado en vez de tomarlo de v2.
- **N.** Con el mismo índice, N no se re-ejecuta: su fila sale de
  ``results/benchmark/metrics/<P>_<D>.csv`` (v2), pasada al esquema de los menús, y su
  panel OOS es ``results/benchmark/panels/<P>_<D>.parquet``. ``--verificar-n`` re-ejecuta N
  para un detector (por defecto B/D02, el más rápido) sobre el panel oficial y lo compara
  celda a celda con v2.

Salida (``results/detectores/menus/``): un CSV por trabajo ``<P>_<D>_<M>.csv`` con el
esquema de ``menus_humo`` (se escribe al final: es la marca de terminado) y su panel OOS
``<P>_<D>_<M>.parquet`` (no se versiona, ``.gitignore``: ``results/detectores/**/*.parquet``);
``_entrada/<P>.parquet`` con las columnas que leen los procesos hijos (tampoco se versiona)
e ``indice_<P>.json`` con la comprobación del índice; ``comparacion_N_T.csv`` al final.

Memoria: el proceso principal construye el panel oficial UNA vez por pista y guarda solo
las columnas necesarias, ya recortadas al índice (≈ 5 MB en A); cada hijo lee ese parquet
y no vuelve a tocar crudos ni a construir termómetros. Cada hijo atiende un solo trabajo
(``max_tasks_per_child=1``) y libera su memoria al terminar.

Uso (desde la raíz del repo)::

    # menú T completo, 10 procesos, 1 hilo BLAS por proceso, lentos primero
    python -m regimenes.benchmark.menus_oficial --pista A B --jobs 10
    # solo algunos detectores
    python -m regimenes.benchmark.menus_oficial --pista A B --detector D02 D10 --jobs 2
    # re-ejecuta N de B/D02 y lo compara con v2
    python -m regimenes.benchmark.menus_oficial --pista B --verificar-n D02
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import gc
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import yaml

from regimenes import features as ft
from regimenes import rutas
from regimenes.benchmark.cache import (
    _cache_fingerprint,
    _cache_matches,
    _metric_path,
    _panel_path,
    _write_atomic,
    processed_paths,
)
from regimenes.benchmark.ejecucion import (
    DEFAULT_TRAIN_DAYS,
    _thread_limits,
    common_track_index,
    load_track_panel,
)
from regimenes.benchmark.variantes import crudos_pista, evaluar_variante
from regimenes.detectores.menus import ADMITIDOS, columnas_menu, variant_specs
from regimenes.evaluacion.ranking import detection_score


SALIDA = rutas.RESULTS_DETECTORES / "menus"
V2 = rutas.RESULTS_BENCHMARK
DETECTORES_T = ADMITIDOS["T"]                       # D02, D03, D04, D08, D09, D10
# Columnas de las filas de v2 que no tienen sentido en una variante (procedencia de la caché).
_SOLO_V2 = ("familia", "variante_v2", "expanding", "cache_fingerprint")


# --------------------------------------------------------------------------- #
# Panel e índice oficiales
# --------------------------------------------------------------------------- #
def verificar_processed(track: str) -> dict[str, str]:
    """sha256 (bytes) de los paneles de ``data/processed`` frente a ``_meta.json``.

    Falla si alguno no coincide: el menú T solo se ejecuta sobre los paneles que escribió
    ``03_preprocesado`` (los mismos que usó v2).
    """
    meta = json.loads((rutas.DATA_PROCESSED / "_meta.json").read_text(encoding="utf-8"))
    esperado = meta["checksums_sha256"]
    out = {}
    for ruta in processed_paths(track):
        real = hashlib.sha256(ruta.read_bytes()).hexdigest()
        if esperado.get(ruta.name) != real:
            raise ValueError(f"{ruta.name}: sha256 {real} no coincide con _meta.json "
                             f"({esperado.get(ruta.name)}). Re-ejecuta 03_preprocesado.")
        out[ruta.name] = real
    return out


def termometros_oficiales(track: str) -> pd.DataFrame:
    """Termómetros de la pista desde ``data/raw`` con historia previa completa (ADR-008 §2.1).

    Exige el pool completo: ni series crudas ausentes ni columnas de ``bloques.yaml``
    omitidas (la prueba de humo iba sin ``CAPE_z``, ``FF_RMW_z`` y
    ``FF_industry_dispersion_z``).
    """
    track = track.upper()
    spec = yaml.safe_load(rutas.BENCHMARK_SPEC.read_text(encoding="utf-8"))[f"pista_{track}"]
    raw = crudos_pista(track)
    faltan = [n for n in spec["series_features"] if n not in raw]
    if faltan:
        raise FileNotFoundError(f"{track}: faltan crudos en data/raw: {faltan}")
    termometros, diag = ft.termometros_pista(raw, track)
    del raw
    ausentes = {k: sorted({c for fila in d["ausentes"] for c in fila}) for k, d in diag.items()}
    ausentes = {k: v for k, v in ausentes.items() if v}
    if ausentes:
        raise ValueError(f"{track}: termómetros con columnas ausentes {ausentes}")
    return termometros


def indice_oficial(track: str, panel: pd.DataFrame, columnas_t: list[str]
                   ) -> tuple[pd.DatetimeIndex, dict]:
    """Índice común del benchmark y comprobación de que los termómetros no lo recortan.

    Devuelve ``common_track_index(track, panel)`` y un informe con los días que perderían
    los termómetros. Si hay alguno, falla (decisión conservadora del docstring del módulo):
    N se tomaría de v2 sobre un OOS distinto del de T.
    """
    track = track.upper()
    indice = common_track_index(track, panel)
    t = panel[columnas_t].reindex(indice)
    perdidos = indice[t.isna().any(axis=1).to_numpy()]
    train = DEFAULT_TRAIN_DAYS[track]
    informe = {
        "pista": track,
        "n_dias_indice": len(indice),
        "inicio": str(indice[0].date()),
        "fin": str(indice[-1].date()),
        "train_days": train,
        "n_oos": len(indice) - train,
        "oos_inicio": str(indice[train].date()),
        "n_dias_perdidos_por_termometros": len(perdidos),
        "fechas_perdidas": [str(d.date()) for d in perdidos],
        "nan_por_termometro": {c: int(v) for c, v in t.isna().sum().items()},
        "mismo_indice_que_v2": len(perdidos) == 0,
    }
    if len(perdidos):
        raise ValueError(
            f"{track}: los termómetros tienen NaN en {len(perdidos)} días del índice del "
            f"benchmark ({informe['fechas_perdidas'][:10]}...). N y T no compartirían OOS: "
            "hay que re-ejecutar N sobre el índice recortado en vez de tomarlo de v2.")
    return indice, informe


def panel_oficial(track: str) -> tuple[pd.DataFrame, pd.DatetimeIndex, dict]:
    """Panel oficial (``load_track_panel`` + termómetros), índice común e informe."""
    track = track.upper()
    sha = verificar_processed(track)
    panel = load_track_panel(track)
    termometros = termometros_oficiales(track)
    columnas_t = list(columnas_menu(track, "T"))
    if list(termometros.columns) != columnas_t:
        raise ValueError(f"{track}: termómetros {list(termometros.columns)} != menú T {columnas_t}")
    repetidas = sorted(set(columnas_t) & set(panel.columns))
    if repetidas:
        raise ValueError(f"{track}: los termómetros pisarían columnas del panel: {repetidas}")
    panel = panel.join(termometros)
    indice, informe = indice_oficial(track, panel, columnas_t)
    informe["sha256_processed"] = sha
    return panel, indice, informe


def columnas_entrada(track: str) -> list[str]:
    """Columnas que necesitan los hijos: features de N (los 12) y de T, y ``SP500_ret``."""
    cols = {c for s in variant_specs(track, "N") for c in s.features}
    cols |= set(columnas_menu(track, "T"))
    return sorted(cols | {"SP500_ret"})


def escribir_entrada(track: str, salida: Path = SALIDA) -> Path:
    """Construye el panel oficial una vez y guarda solo lo que leen los hijos.

    ``_entrada/<P>.parquet``: columnas de :func:`columnas_entrada` recortadas al índice
    oficial (así el índice viaja con el archivo). ``indice_<P>.json``: el informe de
    :func:`indice_oficial` y si la fila N de v2 sigue ligada a su huella de caché.
    """
    track = track.upper()
    salida = Path(salida)
    panel, indice, informe = panel_oficial(track)
    entrada = panel.loc[indice, columnas_entrada(track)].copy()
    del panel
    gc.collect()
    informe["huella_v2_vigente"] = {d: huella_v2_vigente(track, d) for d in DETECTORES_T}
    ruta = salida / "_entrada" / f"{track}.parquet"
    _write_atomic(ruta, lambda tmp: entrada.to_parquet(tmp))
    texto = json.dumps(informe, ensure_ascii=False, indent=2)
    _write_atomic(salida / f"indice_{track}.json", lambda tmp: Path(tmp).write_text(texto, encoding="utf-8"))
    return ruta


def leer_entrada(ruta: Path) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    entrada = pd.read_parquet(ruta)
    entrada.index = pd.DatetimeIndex(entrada.index)
    return entrada, entrada.index


# --------------------------------------------------------------------------- #
# N tomado de v2
# --------------------------------------------------------------------------- #
def huella_v2_vigente(track: str, detector: str, v2: Path = V2) -> bool:
    """¿La fila de v2 sigue ligada a la huella actual (datos, código y entorno)?

    Es la misma comprobación con la que el benchmark acepta su caché. Si es ``False`` (p. ej.
    otra versión de numpy) los datos pueden seguir siendo los mismos: ``--verificar-n`` lo
    dirime re-ejecutando.
    """
    spec = next(s for s in variant_specs(track, "N") if s.detector_id == detector)
    return _cache_matches(_metric_path(v2, track, detector), _panel_path(v2, track, detector),
                          _cache_fingerprint(track, spec, DEFAULT_TRAIN_DAYS[track.upper()]))


def fila_n_v2(track: str, detector: str, v2: Path = V2) -> pd.DataFrame:
    """Fila de v2 (``metrics/<P>_<D>.csv``) en el esquema de los menús.

    Esquema de :func:`evaluar_variante` con ``base`` y ``menu`` delante: se quitan las
    columnas de procedencia de la caché y se añade ``score_deteccion``.
    """
    track, detector = track.upper(), detector.upper()
    fila = pd.read_csv(_metric_path(Path(v2), track, detector))
    fila = fila.drop(columns=[c for c in _SOLO_V2 if c in fila])
    fila["score_deteccion"] = detection_score(fila["det_precision"], fila["det_event_recall"])
    fila.insert(0, "menu", "N")
    fila.insert(0, "base", detector)
    return fila


# --------------------------------------------------------------------------- #
# Trabajos
# --------------------------------------------------------------------------- #
def ruta_trabajo(salida: Path, track: str, detector: str, menu: str) -> Path:
    return Path(salida) / f"{track.upper()}_{detector.upper()}_{menu.upper()}.csv"


def tiempos_v2(v2: Path = V2) -> dict[tuple[str, str], float]:
    """Segundos de cada pista/detector en ``run_status.csv`` de v2 (para el orden LPT)."""
    ruta = Path(v2) / "run_status.csv"
    if not ruta.exists():
        return {}
    rs = pd.read_csv(ruta)
    return {(p, d): float(s) for p, d, s in zip(rs["pista"], rs["id"], rs["segundos"])}


def ordenar_lpt(trabajos: list[tuple[str, str, str]], tiempos: dict[tuple[str, str], float]
                ) -> list[tuple[str, str, str]]:
    """Lentos primero (*longest processing time*): el tiempo de N en v2 aproxima el de T.

    Un trabajo sin tiempo conocido va delante (el caso conservador)."""
    return sorted(trabajos, key=lambda t: (-tiempos.get((t[0], t[1]), np.inf), t))


def _inicializar_hijo(hilos: int | None) -> None:
    """Hilos BLAS/OpenMP del hijo. Las variables de entorno las fija el padre antes del
    ``spawn`` (``_thread_limits``); ``threadpool_limits`` cubre las librerías ya cargadas."""
    warnings.filterwarnings("ignore")
    if hilos:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=int(hilos))


def ejecutar_trabajo(track: str, detector: str, menu: str, entrada: Path, salida: Path) -> dict:
    """Un trabajo pista × detector × menú; escribe panel OOS y, al final, el CSV.

    No hace nada si el CSV ya existe (reanudación). Escritura atómica: un proceso que muere
    no deja un CSV a medias que luego pase por terminado.
    """
    destino = ruta_trabajo(salida, track, detector, menu)
    if destino.exists():
        return {"trabajo": destino.stem, "estado": "ya_estaba"}
    panel, indice = leer_entrada(Path(entrada))
    spec = next(s for s in variant_specs(track, menu) if s.detector_id.split("-")[0] == detector)
    metrics, wf = evaluar_variante(track, spec, panel, indice)
    del panel
    metrics.insert(0, "menu", menu)
    metrics.insert(0, "base", detector)
    _write_atomic(destino.with_suffix(".parquet"), lambda tmp: wf.to_parquet(tmp))
    _write_atomic(destino, lambda tmp: metrics.to_csv(tmp, index=False))   # marca de terminado
    return {"trabajo": destino.stem, "estado": "ok",
            "segundos": float(metrics.loc[0, "elapsed_seconds"]),
            "score": float(metrics.loc[0, "score_deteccion"])}


def _trabajo_seguro(args: tuple[str, str, str, str, str]) -> dict:
    track, detector, menu, entrada, salida = args
    try:
        return ejecutar_trabajo(track, detector, menu, Path(entrada), Path(salida))
    except Exception as exc:  # noqa: BLE001 - un fallo no debe tumbar al resto
        return {"trabajo": f"{track}_{detector}_{menu}", "estado": "error",
                "detalle": f"{type(exc).__name__}: {exc}"}


def escribir_n_v2(tracks: list[str], detectores: list[str], salida: Path = SALIDA,
                  v2: Path = V2) -> list[Path]:
    """Copia las filas N de v2 al esquema de los menús (``<P>_<D>_N.csv``); salta las hechas."""
    hechas = []
    for track in tracks:
        for detector in detectores:
            destino = ruta_trabajo(salida, track, detector, "N")
            if not destino.exists():
                fila = fila_n_v2(track, detector, v2)
                _write_atomic(destino, lambda tmp, f=fila: f.to_csv(tmp, index=False))
            hechas.append(destino)
    return hechas


def ejecutar_menu_t(tracks: list[str], detectores: list[str], *, salida: Path = SALIDA,
                    jobs: int = 1, threads_per_job: int | None = 1) -> list[dict]:
    """Prepara la entrada de cada pista y lanza los trabajos T pendientes en orden LPT."""
    salida = Path(salida)
    tracks = [t.upper() for t in tracks]
    detectores = [d.upper() for d in detectores]
    entradas = {}
    for track in tracks:
        entradas[track] = escribir_entrada(track, salida)
        gc.collect()
    escribir_n_v2(tracks, detectores, salida)
    pendientes = [(t, d, "T") for t in tracks for d in detectores
                  if not ruta_trabajo(salida, t, d, "T").exists()]
    pendientes = ordenar_lpt(pendientes, tiempos_v2())
    print(f"{len(pendientes)} trabajos pendientes (lentos primero): "
          f"{[f'{t}_{d}' for t, d, _ in pendientes]}", flush=True)
    estados: list[dict] = []
    if not pendientes:
        return estados
    args = [(t, d, m, str(entradas[t]), str(salida)) for t, d, m in pendientes]
    with _thread_limits(threads_per_job):
        contexto = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max(1, min(int(jobs), len(args))), mp_context=contexto,
                                 initializer=_inicializar_hijo, initargs=(threads_per_job,),
                                 max_tasks_per_child=1) as pool:
            futuros = {pool.submit(_trabajo_seguro, a): a for a in args}
            while futuros:
                hechos, _ = wait(futuros, return_when=FIRST_COMPLETED)
                for futuro in hechos:
                    t, d, m, *_ = futuros.pop(futuro)
                    try:
                        estado = futuro.result()
                    except Exception as exc:  # noqa: BLE001 - p. ej. hijo muerto por memoria
                        estado = {"trabajo": f"{t}_{d}_{m}", "estado": "error",
                                  "detalle": f"{type(exc).__name__}: {exc}"}
                    estados.append(estado)
                    print(time.strftime("%H:%M:%S"), estado, flush=True)
    return estados


def comparar(salida: Path = SALIDA, *, escribir: bool = True) -> pd.DataFrame:
    """``comparacion_N_T.csv``: score de T frente a N por pista y detector (lo que haya)."""
    salida = Path(salida)
    filas = [pd.read_csv(p) for p in sorted(salida.glob("[AB]_D*_[NT].csv"))]
    if not filas:
        return pd.DataFrame()
    todo = pd.concat(filas, ignore_index=True)
    cols = ["score_deteccion", "det_n_detectados", "det_precision", "det_event_recall", "n_oos",
            "elapsed_seconds"]
    tabla = todo.pivot_table(index=["pista", "base"], columns="menu", values=cols, aggfunc="first")
    tabla.columns = [f"{v}_{m}" for v, m in tabla.columns]
    tabla = tabla.reset_index()
    if {"score_deteccion_N", "score_deteccion_T"} <= set(tabla):
        tabla["delta_score"] = tabla["score_deteccion_T"] - tabla["score_deteccion_N"]
    if escribir:
        _write_atomic(salida / "comparacion_N_T.csv", lambda tmp: tabla.to_csv(tmp, index=False))
    return tabla


def mcs_t_union_n(track: str, salida: Path = SALIDA, v2: Path = V2, *,
                  incluir_baselines: bool = False, **kwargs) -> pd.DataFrame:
    """MCS del criterio (c) del ADR-008 §6 sobre el conjunto «T ∪ N» de una pista.

    N son los 12 originales (paneles de v2) y T las 6 variantes del menú T (paneles de
    ``salida``), todos sobre el mismo índice OOS. Misma pérdida y remuestreo que el MCS
    definitivo (``evaluacion.mcs.mcs_senales``). Sin baselines por defecto: el criterio
    habla de detectores; con ``incluir_baselines`` se añaden los baselines sin
    información (``SIEMPRE_*``, ``AZAR_*``) como sensibilidad, sin ``VOL_P90``.
    """
    from regimenes.evaluacion import mcs
    from regimenes.evaluacion import ranking as rk

    track = track.upper()
    paneles_v2 = Path(v2) / "panels"
    pan_n = mcs.cargar_paneles(track, paneles_v2)
    pan_t = mcs.cargar_paneles(track, Path(salida), [f"{d}_T" for d in DETECTORES_T])
    ref = next(iter(pan_n.values())).index
    distintos = [k for k, p in {**pan_n, **pan_t}.items() if not p.index.equals(ref)]
    if distintos:
        raise ValueError(f"{track}: los paneles {distintos} no tienen el índice OOS de v2")
    sen = pd.concat([
        mcs.senales_crisis(pan_n, n_states=mcs.n_states_metricas(track, paneles_v2, list(pan_n))),
        mcs.senales_crisis(pan_t, n_states=mcs.n_states_metricas(track, Path(salida), list(pan_t))),
    ], axis=1)
    if incluir_baselines:
        F = sen.to_numpy()
        base = mcs.senales_baselines(track, pd.DatetimeIndex(sen.index),
                                     rate=float(np.median(F.mean(axis=0))),
                                     mean_run=float(np.median([mcs._racha_media(F[:, j])
                                                               for j in range(F.shape[1])])))
        sen = pd.concat([sen, base], axis=1)
    _, res = mcs.mcs_senales(sen, rk.track_crisis_windows(track), **kwargs)
    return res.tabla


def evaluar_criterios(comp: pd.DataFrame, tablas_mcs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Criterios (a) y (c) a partir de ``comparar`` y de las tablas MCS «T ∪ N» por pista.

    (a) exige los 6 multivariantes comparados y que mejoren estrictamente más de la mitad;
    un empate (``delta_score == 0``) cuenta como «no mejora».
    """
    filas = []
    for track, tabla in tablas_mcs.items():
        c = (comp[comp["pista"] == track].dropna(subset=["delta_score"])
             if {"pista", "delta_score"} <= set(comp) else comp.iloc[0:0])
        n_mejoran = int((c["delta_score"] > 0).sum()) if len(c) else 0
        en_mcs = [str(i) for i in tabla.index[tabla["incluido"].astype(bool)]]
        t_en_mcs = [i for i in en_mcs if i.endswith("_T")]
        filas.append({
            "pista": track, "n_comparados": len(c), "n_mejoran_T": n_mejoran,
            "a_mayoria_mejora": len(c) == len(DETECTORES_T) and n_mejoran > len(DETECTORES_T) / 2,
            "mcs_miembros": "|".join(en_mcs), "c_algun_T_en_mcs": bool(t_en_mcs),
            "T_en_mcs": "|".join(t_en_mcs),
        })
    out = pd.DataFrame(filas)
    out["b_bloques_frente_a_azar"] = "pendiente (Fase 2, 2.5)"
    return out


def criterios_adr008(salida: Path = SALIDA) -> pd.DataFrame:
    """Criterios (a) y (c) del ADR-008 §6 por pista; (b) necesita las 100 reparticiones
    aleatorias de columnas a bloques (Fase 2, punto 2.5) y queda pendiente.

    Escribe ``comparacion_N_T.csv``, ``mcs_T_union_N_{A,B}.csv`` y ``criterios_adr008.csv``.
    """
    salida = Path(salida)
    comp = comparar(salida)
    tablas = {track: mcs_t_union_n(track, salida) for track in ("A", "B")}
    for track, tabla in tablas.items():
        _write_atomic(salida / f"mcs_T_union_N_{track}.csv", lambda tmp, t=tabla: t.to_csv(tmp))
    out = evaluar_criterios(comp, tablas)
    _write_atomic(salida / "criterios_adr008.csv", lambda tmp: out.to_csv(tmp, index=False))
    return out


# --------------------------------------------------------------------------- #
# Verificación de N
# --------------------------------------------------------------------------- #
def comparar_con_v2(track: str, detector: str, metrics: pd.DataFrame, wf: pd.DataFrame,
                    v2: Path = V2) -> dict:
    """Diferencias celda a celda entre una re-ejecución de N y v2 (fila y panel OOS)."""
    ref = fila_n_v2(track, detector, v2)
    # La re-ejecución pasa por el mismo viaje a CSV que la fila de v2: así se comparan los
    # mismos flotantes y no el redondeo del lector de CSV.
    metrics = pd.read_csv(io.StringIO(metrics.to_csv(index=False)))
    comunes = [c for c in ref.columns if c in metrics.columns and c != "elapsed_seconds"]
    distintas = []
    for c in comunes:
        a, b = ref.loc[0, c], metrics.loc[0, c]
        if pd.isna(a) and pd.isna(b):
            continue
        if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
            if float(a) == float(b):
                continue
        elif str(a) == str(b):
            continue
        distintas.append((c, a, b))
    panel_v2 = pd.read_parquet(_panel_path(Path(v2), track, detector))
    try:
        pd.testing.assert_frame_equal(wf, panel_v2, check_exact=True, check_freq=False)
        panel_igual, detalle_panel = True, ""
    except AssertionError as exc:
        panel_igual, detalle_panel = False, str(exc)[:500]
    return {"pista": track, "detector": detector, "columnas_comparadas": len(comunes),
            "columnas_distintas": distintas, "panel_identico": panel_igual,
            "detalle_panel": detalle_panel,
            "reproduce_v2": not distintas and panel_igual}


def verificar_n(track: str, detector: str = "D02", *, salida: Path = SALIDA,
                threads: int | None = 1) -> dict:
    """Re-ejecuta N de un detector sobre el panel oficial y lo compara con v2.

    Deja la fila, el panel y el veredicto en ``<salida>/verificacion_n/`` (no toca v2)."""
    track, detector = track.upper(), detector.upper()
    destino = Path(salida) / "verificacion_n"
    entrada = escribir_entrada(track, salida)
    _inicializar_hijo(threads)
    panel, indice = leer_entrada(entrada)
    spec = next(s for s in variant_specs(track, "N") if s.detector_id == detector)
    metrics, wf = evaluar_variante(track, spec, panel, indice)
    metrics.insert(0, "menu", "N")
    metrics.insert(0, "base", detector)
    veredicto = comparar_con_v2(track, detector, metrics, wf)
    nombre = f"{track}_{detector}_N"
    _write_atomic(destino / f"{nombre}.parquet", lambda tmp: wf.to_parquet(tmp))
    _write_atomic(destino / f"{nombre}.csv", lambda tmp: metrics.to_csv(tmp, index=False))
    texto = json.dumps(veredicto, ensure_ascii=False, indent=2, default=str)
    _write_atomic(destino / f"{nombre}.json", lambda tmp: Path(tmp).write_text(texto, encoding="utf-8"))
    return veredicto


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pista", nargs="+", default=["A", "B"], choices=["A", "B", "a", "b"])
    parser.add_argument("--detector", nargs="+", default=list(DETECTORES_T),
                        help=f"bases del menú T ({' '.join(DETECTORES_T)})")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--threads-per-job", type=int, default=1,
                        help="hilos BLAS/OpenMP por proceso (0 = no tocar el entorno)")
    parser.add_argument("--salida", type=Path, default=SALIDA)
    parser.add_argument("--verificar-n", metavar="DETECTOR", default=None,
                        help="re-ejecuta N de este detector en cada --pista y lo compara con v2")
    args = parser.parse_args(argv)
    pistas = [p.upper() for p in args.pista]
    hilos = args.threads_per_job or None
    if args.verificar_n:
        for pista in pistas:
            print(json.dumps(verificar_n(pista, args.verificar_n, salida=args.salida, threads=hilos),
                             ensure_ascii=False, indent=2, default=str), flush=True)
        return
    detectores = [d.upper() for d in args.detector]
    ajenos = sorted(set(detectores) - set(DETECTORES_T))
    if ajenos:
        parser.error(f"detectores fuera del menú T: {ajenos}")
    estados = ejecutar_menu_t(pistas, detectores, salida=args.salida, jobs=args.jobs,
                              threads_per_job=hilos)
    tabla = comparar(args.salida)
    if not tabla.empty:
        print(tabla.to_string(index=False), flush=True)
    errores = [e for e in estados if e["estado"] == "error"]
    if errores:
        raise SystemExit(f"{len(errores)} trabajos con error: {errores}")


if __name__ == "__main__":
    main()
