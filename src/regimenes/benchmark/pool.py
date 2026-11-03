"""Fase 2 · pool sin comprimir (ADR-010): preparar, piloto de coste, ejecutar, estado y lectura.

Ejecuta TODAS las variantes de :func:`regimenes.detectores.pool.especificaciones.specs_pool`
(las de ``pool_crudo``: D10-P, D10-R, D14-R, D18-P; y las de ``shu_supervisados``: D15, D16,
D17) con el protocolo de v2: mismo walk-forward (``evaluacion.walk_forward``, sin tocar), mismo
``train_days`` (8 años en A, 3 en B), el ``step`` y la ventana de cada ``SpecPool``, mismo juez
(``configure_evaluation``), mismo ajuste final y misma detección por evento que
:func:`regimenes.benchmark.variantes.evaluar_variante`, y ``score_deteccion``.

- **Panel.** ``data/processed`` verificado por sha256 frente a ``_meta.json``
  (:func:`~regimenes.benchmark.menus_oficial.verificar_processed`), cargado como el benchmark
  (``load_track_panel``, con ``SP500_ret``) y, en B, unido al panel R
  (``data/processed/pistaB_rentabilidades.parquet``, verificado con su ficha; se construye desde
  ``data/raw`` si no existe).
- **Índice OOS.** El del benchmark: ``common_track_index`` sobre el panel oficial, y se comprueba
  día a día que ``indice[train_days:]`` es el índice de los 12 paneles de v2
  (``results/benchmark/panels``). Los NaN de una variante se permiten en el calentamiento (antes
  del OOS: columnas de P que empiezan tarde, regla de disponibilidad del ADR-010 §2.1) y
  **no en el OOS**, salvo en el menú R, cuyos días sin cotización (festivos de FRED en las
  divisas, 2 días de DXY) se dejan en NaN por el propio ADR-010 §2.2; quedan contados en
  ``preparacion_<P>.json``.
- **Huella de código.** sha256 de ``detectores/pool/*.py``, de este módulo, de
  ``features/rentabilidades.py`` y de ``configs/bloques.yaml``, registrada al preparar; ejecutar
  y piloto se niegan a seguir si cambió (como en :mod:`regimenes.benchmark.ablaciones`).

Salida (``results/detectores/pool/``):

- ``_entrada/<P>.parquet`` (no se versiona) y ``preparacion_<P>.json`` (índice, NaN por
  variante, huella, sha256 de la entrada).
- ``<P>_<id>.csv`` (marca de terminado; esquema de ``results/detectores/menus`` con ``base`` y
  ``menu`` delante) y su panel OOS ``<P>_<id>.parquet`` (``.gitignore``:
  ``results/detectores/**/*.parquet``); ``reajustes/<P>_<id>.csv``: una fila por reajuste
  (columnas que entran, orden canónico y, en D18, columnas con peso > 0).
- ``piloto.csv``: coste medido y extrapolado por pista y variante; ``ejecucion.log``;
  ``_errores/<P>_<id>.json``.
- ``leer``: ``comparacion_pares.csv`` (§5 a), ``mcs_NUPR_<P>.csv`` (§5 b),
  ``criterio_adr010.csv``, ``orden_canonico.csv``, ``d18_columnas_<P>.csv``,
  ``d18_bloques_<P>.csv`` y, si hay supervisados, ``mcs_supervisados_<P>.csv`` y
  ``techo_supervisados.csv`` (§6).

Uso (desde la raíz del repo)::

    python -m regimenes.benchmark.pool preparar --pista A B
    python -m regimenes.benchmark.pool piloto --pista A B --jobs 2 --reajustes 3
    python -m regimenes.benchmark.pool ejecutar --pista A B --jobs 4 --horas 7.5
    python -m regimenes.benchmark.pool estado --pista A B
    python -m regimenes.benchmark.pool leer
    # subconjuntos (admite comodines)
    python -m regimenes.benchmark.pool ejecutar --pista B --detector "D10-*" D14-R
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
import fnmatch
import gc
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
from typing import Any, Callable, Sequence
import warnings

import numpy as np
import pandas as pd

from regimenes import evaluacion as ev
from regimenes import rutas
from regimenes.benchmark import menus_oficial as mo
from regimenes.benchmark.cache import _write_atomic
from regimenes.benchmark.ejecucion import (
    DEFAULT_TRAIN_DAYS,
    _thread_limits,
    common_track_index,
    configure_evaluation,
    load_track_panel,
)
from regimenes.detectores.pool.especificaciones import SpecPool
from regimenes.evaluacion.metricas import detection_summary
from regimenes.evaluacion.ranking import detection_score
from regimenes.features import rentabilidades as rent

SALIDA = rutas.RESULTS_DETECTORES / "pool"
V2 = rutas.RESULTS_BENCHMARK
PISTAS = ("A", "B")
SUPERVISADOS = ("D16", "D17")                 # categoría aparte (ADR-010 §3.2 y §6)
MENUS_NAN_OOS = ("R",)                        # NaN en el OOS permitidos (ADR-010 §2.2)
# Pares del criterio (a) del §5: variante del pool -> detector de v2 (menú N).
PARES = {"D10-P": "D10", "D10-R": "D10", "D14-R": "D10", "D18-P": "D09"}
# Comparaciones que se reportan sin peso en el veredicto (§5, notas): base -> par de v2.
PARES_INFORMATIVOS = {"D15": "D09"}
COSTE_DESCONOCIDO = 3600.0                    # segundos de un trabajo sin piloto
MAX_ROTURAS = 3
N_PILOTO = 3                                  # reajustes medidos al principio y al final


# --------------------------------------------------------------------------- #
# Variantes
# --------------------------------------------------------------------------- #
def base_de(detector_id: str) -> str:
    return detector_id.split("-")[0]


def menu_de(spec_o_id: SpecPool | str) -> str:
    if isinstance(spec_o_id, SpecPool) and spec_o_id.menu:
        return spec_o_id.menu.upper()
    texto = spec_o_id.detector_id if isinstance(spec_o_id, SpecPool) else str(spec_o_id)
    return texto.split("-")[1].upper() if "-" in texto else "N"


def es_supervisado(detector_id: str) -> bool:
    return base_de(detector_id) in SUPERVISADOS


def todas_las_specs(track: str) -> list[SpecPool]:
    """``specs_pool(track)``; si el módulo de D15–D17 aún no existe, avisa y sigue con
    ``pool_crudo`` (para poder probar una parte sin la otra)."""
    from regimenes.detectores.pool import especificaciones as esp

    try:
        return esp.specs_pool(track)
    except ModuleNotFoundError as exc:
        if "shu_supervisados" not in str(exc):
            raise
        from regimenes.detectores.pool.pool_crudo import especificaciones_pool_crudo

        warnings.warn("regimenes.detectores.pool.shu_supervisados no existe: solo D10/D14/D18",
                      stacklevel=2)
        return especificaciones_pool_crudo(track)


def seleccionar(specs: Sequence[SpecPool], patrones: Sequence[str] | None) -> list[SpecPool]:
    if not patrones:
        return list(specs)
    return [s for s in specs
            if any(fnmatch.fnmatchcase(s.detector_id.upper(), p.upper()) for p in patrones)]


def spec_de(track: str, detector_id: str) -> SpecPool:
    for s in todas_las_specs(track):
        if s.detector_id.upper() == detector_id.upper():
            return s
    raise KeyError(f"{track}: variante desconocida {detector_id!r}")


# --------------------------------------------------------------------------- #
# Huella y rutas
# --------------------------------------------------------------------------- #
def sha256(ruta: Path) -> str:
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def huella_codigo() -> dict[str, str]:
    """sha256 del código y la configuración que fijan los resultados de la Fase 2."""
    src = rutas.ROOT / "src" / "regimenes"
    archivos = sorted((src / "detectores" / "pool").glob("*.py"))
    archivos += [Path(__file__), Path(rent.__file__), rutas.CONFIGS / "bloques.yaml"]
    return {Path(a).resolve().relative_to(rutas.ROOT).as_posix(): sha256(a) for a in archivos}


def ruta_entrada(salida: Path, track: str) -> Path:
    return Path(salida) / "_entrada" / f"{track.upper()}.parquet"


def ruta_preparacion(salida: Path, track: str) -> Path:
    return Path(salida) / f"preparacion_{track.upper()}.json"


def ruta_trabajo(salida: Path, track: str, detector_id: str) -> Path:
    return Path(salida) / f"{track.upper()}_{detector_id}.csv"


def ruta_reajustes(salida: Path, track: str, detector_id: str) -> Path:
    return Path(salida) / "reajustes" / f"{track.upper()}_{detector_id}.csv"


def ruta_error(salida: Path, track: str, detector_id: str, tipo: str = "ejecutar") -> Path:
    sufijo = "" if tipo == "ejecutar" else f"_{tipo}"
    return Path(salida) / "_errores" / f"{track.upper()}_{detector_id}{sufijo}.json"


def leer_preparacion(salida: Path, track: str) -> dict[str, Any]:
    ruta = ruta_preparacion(salida, track)
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}


def comprobar_huella(prep: dict[str, Any], track: str, salida: Path) -> None:
    """Falla si ``preparacion_<P>.json`` falta o es de otro código."""
    if not prep:
        raise FileNotFoundError(f"{track}: falta {ruta_preparacion(salida, track).name}; "
                                "ejecuta antes `preparar`")
    registrada, actual = prep.get("huella_codigo") or {}, huella_codigo()
    if registrada != actual:
        distintos = sorted(k for k in set(actual) | set(registrada) if registrada.get(k) != actual.get(k))
        raise ValueError(f"{track}: el código cambió desde preparar ({distintos}). Borra en "
                         f"{Path(salida)} los resultados de la pista ({track}_*.csv, {track}_*.parquet, "
                         f"reajustes/{track}_*, preparacion_{track}.json) y vuelve a preparar.")


# --------------------------------------------------------------------------- #
# Panel, índice y entrada
# --------------------------------------------------------------------------- #
def panel_r() -> pd.DataFrame:
    """Panel R verificado; se construye desde ``data/raw`` si aún no existe."""
    if not rent.PANEL_R.exists() or not rent.META_R.exists():
        rent.escribir_panel_r()
    return rent.leer_panel_r()


def indice_v2(track: str, v2: Path = V2) -> pd.DatetimeIndex:
    """Índice OOS común a los 12 paneles de v2 de la pista (falla si difieren)."""
    from regimenes.evaluacion import mcs

    paneles = mcs.cargar_paneles(track, Path(v2) / "panels")
    return pd.DatetimeIndex(next(iter(paneles.values())).index)


def comprobar_indice(track: str, indice: pd.DatetimeIndex, oos_v2: pd.DatetimeIndex) -> dict[str, Any]:
    """``indice[train_days:]`` debe ser, día a día, el OOS de v2."""
    train = DEFAULT_TRAIN_DAYS[track.upper()]
    oos = pd.DatetimeIndex(indice[train:])
    if not oos.equals(pd.DatetimeIndex(oos_v2)):
        solo_aqui = oos.difference(oos_v2)
        solo_v2 = pd.DatetimeIndex(oos_v2).difference(oos)
        raise ValueError(f"{track}: el OOS no es el de v2 ({len(oos)} frente a {len(oos_v2)} días; "
                         f"solo aquí {[str(d.date()) for d in solo_aqui[:5]]}, solo en v2 "
                         f"{[str(d.date()) for d in solo_v2[:5]]})")
    return {"n_dias_indice": len(indice), "inicio": str(indice[0].date()), "fin": str(indice[-1].date()),
            "train_days": train, "n_oos": len(oos), "oos_inicio": str(oos[0].date()),
            "mismo_oos_que_v2": True}


def informe_nan(track: str, spec: SpecPool, X: pd.DataFrame, train_days: int | None = None) -> dict[str, Any]:
    """NaN de una variante en el calentamiento y en el OOS; falla si hay en el OOS (salvo R)."""
    train = int(train_days or DEFAULT_TRAIN_DAYS[track.upper()])
    oos = X.iloc[train:]
    nan_oos = {c: int(v) for c, v in oos.isna().sum().items() if v}
    primeros = {c: str(X[c].first_valid_index().date()) for c in X
                if X[c].first_valid_index() is not None and X[c].first_valid_index() > X.index[0]}
    vacias = [c for c in X if X[c].first_valid_index() is None]
    if vacias:
        raise ValueError(f"{track}/{spec.detector_id}: columnas sin ningún dato {vacias}")
    if nan_oos and menu_de(spec) not in MENUS_NAN_OOS:
        raise ValueError(f"{track}/{spec.detector_id}: NaN dentro del OOS {nan_oos}")
    return {"n_columnas": X.shape[1], "nan_calentamiento": int(X.iloc[:train].isna().sum().sum()),
            "columnas_que_empiezan_tarde": primeros, "nan_oos_por_columna": nan_oos,
            "filas_oos_incompletas": int(oos.isna().any(axis=1).sum())}


def construir_panel(track: str) -> tuple[pd.DataFrame, pd.DatetimeIndex, dict[str, str]]:
    """Panel oficial (+ R en B), índice común del benchmark y sha256 de ``data/processed``."""
    track = track.upper()
    sha = mo.verificar_processed(track)
    panel = load_track_panel(track)
    if track == "B":
        r = panel_r()
        repetidas = sorted(set(r.columns) & set(panel.columns))
        if repetidas:
            raise ValueError(f"B: el panel R pisaría columnas del panel: {repetidas}")
        if not r.index.equals(pd.DatetimeIndex(panel.index)):
            r = r.reindex(panel.index)
        panel = panel.join(r)
        sha[rent.PANEL_R.name] = sha256(rent.PANEL_R)
    return panel, common_track_index(track, panel), sha


def preparar(tracks: Sequence[str], *, salida: Path = SALIDA, v2: Path = V2) -> dict[str, Path]:
    """Entrada de cada pista (columnas de todas las variantes, recortadas al índice oficial) y
    ``preparacion_<P>.json``. Comprueba el índice frente a v2 y los NaN de cada variante."""
    salida = Path(salida)
    out = {}
    for track in [t.upper() for t in tracks]:
        panel, indice, sha = construir_panel(track)
        info = {"pista": track, **comprobar_indice(track, indice, indice_v2(track, v2))}
        specs = todas_las_specs(track)
        cols = sorted({c for s in specs for c in s.features} | {"SP500_ret"})
        faltan = sorted(set(cols) - set(panel.columns))
        if faltan:
            raise KeyError(f"{track}: faltan columnas en el panel oficial: {faltan}")
        entrada = panel.loc[indice, cols].copy()
        del panel
        gc.collect()
        info["variantes"] = {s.detector_id: {"menu": menu_de(s), "step": s.step, "expanding": s.expanding,
                                             **informe_nan(track, s, entrada[list(s.features)])}
                             for s in specs}
        ruta = ruta_entrada(salida, track)
        _write_atomic(ruta, lambda tmp, e=entrada: e.to_parquet(tmp))
        info.update({"sha256_processed": sha, "entrada": {"archivo": ruta.name, "sha256": sha256(ruta)},
                     "huella_codigo": huella_codigo(),
                     "fecha": time.strftime("%Y-%m-%d %H:%M:%S")})
        texto = json.dumps(info, ensure_ascii=False, indent=1, default=str)
        _write_atomic(ruta_preparacion(salida, track),
                      lambda tmp, x=texto: Path(tmp).write_text(x, encoding="utf-8"))
        print(f"{track}: entrada {entrada.shape}, {len(specs)} variantes, OOS {info['n_oos']} días "
              f"igual que v2", flush=True)
        out[track] = ruta
    return out


def leer_entrada(salida: Path, track: str, prep: dict[str, Any]) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    ruta = ruta_entrada(salida, track)
    if sha256(ruta) != prep["entrada"]["sha256"]:
        raise ValueError(f"{ruta.name}: sha256 distinto del registrado; vuelve a preparar")
    entrada = pd.read_parquet(ruta)
    entrada.index = pd.DatetimeIndex(entrada.index)
    return entrada, entrada.index


# --------------------------------------------------------------------------- #
# Evaluación de una variante
# --------------------------------------------------------------------------- #
class Recolector:
    """Factoría que guarda el resumen de cada reajuste del walk-forward.

    ``walk_forward`` crea un detector nuevo por reajuste y termina con el anterior antes de
    pedir el siguiente; aquí se resume el anterior (``resumen_reajuste()`` si lo expone, si no
    su orden canónico) y se suelta, para no acumular modelos en memoria."""

    def __init__(self, factory: Callable[[], Any]) -> None:
        self.factory = factory
        self.filas: list[dict[str, Any]] = []
        self._ultimo = None

    def _resumir(self) -> None:
        det, self._ultimo = self._ultimo, None
        if det is None:
            return
        f = getattr(det, "resumen_reajuste", None)
        fila = dict(f()) if callable(f) else {}
        orden = getattr(det, "_canonical_order", None)
        if "orden_canonico" not in fila and orden is not None:
            fila["orden_canonico"] = "|".join(str(int(o)) for o in orden)
        self.filas.append(fila)

    def __call__(self):
        self._resumir()
        self._ultimo = self.factory()
        return self._ultimo

    def cerrar(self) -> pd.DataFrame:
        self._resumir()
        tabla = pd.DataFrame(self.filas)
        tabla.insert(0, "reajuste", range(len(tabla)))
        return tabla


def matriz(track: str, spec: SpecPool, panel: pd.DataFrame, index: pd.DatetimeIndex,
           train_days: int | None = None) -> pd.DataFrame:
    faltan = sorted(set(spec.features) - set(panel.columns))
    if faltan:
        raise KeyError(f"{track}/{spec.detector_id}: faltan columnas {faltan}")
    X = panel.loc[index, list(spec.features)]
    informe_nan(track, spec, X, train_days)
    return X


def evaluar_pool(track: str, spec: SpecPool, panel: pd.DataFrame, index: pd.DatetimeIndex,
                 *, train_days: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Métricas (una fila), panel OOS y tabla de reajustes de una variante.

    Mismo protocolo que :func:`regimenes.benchmark.variantes.evaluar_variante`, salvo que admite
    NaN en el calentamiento (y en el OOS del menú R), que el detector trata con su regla de
    disponibilidad, y que recoge el resumen de cada reajuste."""
    track = track.upper()
    min_train = int(train_days or DEFAULT_TRAIN_DAYS[track])
    configure_evaluation(track)
    X = matriz(track, spec, panel, index, min_train)
    market_returns = panel["SP500_ret"].reindex(X.index)
    rec = Recolector(spec.factory)
    started = time.perf_counter()
    wf_panel = ev.walk_forward(rec, X, market_returns=market_returns, train_size=min_train,
                               min_train=min_train, step=spec.step, expanding=spec.expanding)
    reajustes = rec.cerrar()
    esperado = pd.DatetimeIndex(index[min_train:])
    if not pd.DatetimeIndex(wf_panel.index).equals(esperado):
        raise ValueError(f"{track}/{spec.detector_id}: el índice OOS no es el oficial "
                         f"({len(wf_panel)} frente a {len(esperado)})")
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
    informe = getattr(final, "informe_oos", None)
    if callable(informe):
        for key, value in informe(X, esperado).items():
            metrics[key] = value
    metrics["n_reajustes"] = len(reajustes)
    if "orden_canonico" in reajustes:
        metrics["n_reajustes_orden_invertido"] = int(
            (reajustes["orden_canonico"].dropna() != "|".join(str(i) for i in range(final.n_states))).sum())
    for key, value in reversed(list({
        "base": base_de(spec.detector_id), "menu": menu_de(spec),
        "id": spec.detector_id, "pista": track, "n_features_input": len(spec.features),
        "features_input": "|".join(spec.features), "train_days": min_train, "refit_step": spec.step,
        "expanding": spec.expanding, "elapsed_seconds": time.perf_counter() - started,
    }.items())):
        metrics.insert(0, key, value)
    wf_panel = wf_panel.copy()
    wf_panel.attrs = {}
    return metrics, wf_panel, reajustes


# --------------------------------------------------------------------------- #
# Trabajos (un proceso por trabajo)
# --------------------------------------------------------------------------- #
def ejecutar_trabajo(track: str, detector_id: str, salida: Path) -> dict:
    """Una variante de una pista: panel OOS, reajustes y, al final, el CSV (marca de terminado)."""
    track = track.upper()
    destino = ruta_trabajo(salida, track, detector_id)
    if destino.exists():
        return {"trabajo": destino.stem, "estado": "ya_estaba"}
    prep = leer_preparacion(salida, track)
    comprobar_huella(prep, track, salida)
    panel, indice = leer_entrada(Path(salida), track, prep)
    spec = spec_de(track, detector_id)
    metrics, wf, reajustes = evaluar_pool(track, spec, panel, indice)
    del panel
    _write_atomic(destino.with_suffix(".parquet"), lambda tmp: wf.to_parquet(tmp))
    _write_atomic(ruta_reajustes(salida, track, detector_id), lambda tmp: reajustes.to_csv(tmp, index=False))
    _write_atomic(destino, lambda tmp: metrics.to_csv(tmp, index=False))     # marca de terminado
    ruta_error(salida, track, detector_id).unlink(missing_ok=True)
    return {"trabajo": destino.stem, "estado": "ok",
            "segundos": round(float(metrics.loc[0, "elapsed_seconds"]), 1),
            "score": float(metrics.loc[0, "score_deteccion"])}


def _tramo(spec: SpecPool, X: pd.DataFrame, mr: pd.Series, train: int, n_reajustes: int, final: bool
           ) -> tuple[float, float, int]:
    """Segundos, tamaño medio del train y reajustes de un tramo del walk-forward del piloto."""
    step = spec.step
    n = len(X)
    if final:
        corte = max(train, n - n_reajustes * step)
        Xt, inicio = X, corte
    else:
        Xt, inicio = X.iloc[:min(n, train + n_reajustes * step)], train
    t0 = time.perf_counter()
    # train_size fija la ventana móvil (si no es expansiva) y min_train el primer reajuste medido.
    wf = ev.walk_forward(spec.factory, Xt, market_returns=mr.reindex(Xt.index), train_size=train,
                         min_train=inicio, step=step, expanding=spec.expanding)
    segundos = time.perf_counter() - t0
    k = int(wf["fold"].nunique())
    tamanos = [inicio + i * step if spec.expanding else train for i in range(k)]
    return segundos, float(np.mean(tamanos)), k


def coste_extrapolado(seg_ini: float, m_ini: float, k_ini: int, seg_fin: float, m_fin: float, k_fin: int,
                      tamanos: Sequence[int]) -> float:
    """Coste total con un coste por reajuste lineal en el tamaño del train, ajustado a los dos
    tramos medidos (al principio y al final), más el ajuste final (un reajuste con todo)."""
    c_ini, c_fin = seg_ini / max(k_ini, 1), seg_fin / max(k_fin, 1)
    pendiente = (c_fin - c_ini) / (m_fin - m_ini) if m_fin > m_ini else 0.0
    costes = [max(0.0, c_ini + pendiente * (m - m_ini)) for m in tamanos]
    return float(sum(costes) + max(costes[-1] if costes else 0.0, c_fin))


def ejecutar_piloto(track: str, detector_id: str, salida: Path, n_reajustes: int = N_PILOTO) -> dict:
    """Mide ``n_reajustes`` al principio y al final del walk-forward y extrapola el coste."""
    track = track.upper()
    prep = leer_preparacion(salida, track)
    comprobar_huella(prep, track, salida)
    panel, indice = leer_entrada(Path(salida), track, prep)
    spec = spec_de(track, detector_id)
    configure_evaluation(track)
    X = matriz(track, spec, panel, indice)
    mr = panel["SP500_ret"]
    train = DEFAULT_TRAIN_DAYS[track]
    seg_i, m_i, k_i = _tramo(spec, X, mr, train, n_reajustes, final=False)
    seg_f, m_f, k_f = _tramo(spec, X, mr, train, n_reajustes, final=True)
    n_oos = len(X) - train
    total_reaj = int(np.ceil(n_oos / spec.step))
    tamanos = [train + i * spec.step if spec.expanding else train for i in range(total_reaj)]
    est = coste_extrapolado(seg_i, m_i, k_i, seg_f, m_f, k_f, tamanos)
    return {"trabajo": f"{track}_{detector_id}", "estado": "ok", "pista": track, "id": detector_id,
            "n_reajustes_total": total_reaj, "reajustes_medidos": k_i + k_f,
            "seg_por_reajuste_inicio": round(seg_i / max(k_i, 1), 3),
            "seg_por_reajuste_final": round(seg_f / max(k_f, 1), 3),
            "segundos_estimados": round(est, 1), "horas_estimadas": round(est / 3600, 3),
            "fecha": time.strftime("%Y-%m-%d %H:%M:%S")}


def _seguro(funcion: Callable[..., dict], args: tuple) -> dict:
    try:
        return funcion(*args)
    except Exception as exc:  # noqa: BLE001 - un fallo no debe tumbar al resto
        return {"trabajo": f"{args[0]}_{args[1]}", "estado": "error",
                "detalle": f"{type(exc).__name__}: {exc}"}


def _trabajo_seguro(args: tuple[str, str, str]) -> dict:
    return _seguro(ejecutar_trabajo, (args[0], args[1], Path(args[2])))


def _piloto_seguro(args: tuple[str, str, str, int]) -> dict:
    return _seguro(ejecutar_piloto, (args[0], args[1], Path(args[2]), int(args[3])))


# --------------------------------------------------------------------------- #
# Coste, log y pool de procesos
# --------------------------------------------------------------------------- #
def leer_piloto(salida: Path = SALIDA) -> pd.DataFrame:
    ruta = Path(salida) / "piloto.csv"
    return pd.read_csv(ruta) if ruta.exists() else pd.DataFrame(columns=["pista", "id", "segundos_estimados"])


def costes(salida: Path = SALIDA) -> dict[tuple[str, str], float]:
    p = leer_piloto(salida)
    return {(str(a), str(b)): float(s) for a, b, s in zip(p["pista"], p["id"], p["segundos_estimados"])}


def _log(salida: Path, texto: str) -> None:
    linea = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {texto}"
    print(linea, flush=True)
    Path(salida).mkdir(parents=True, exist_ok=True)
    with open(Path(salida) / "ejecucion.log", "a", encoding="utf-8") as f:
        f.write(linea + "\n")


def _nuevo_pool(jobs: int, threads_per_job: int | None) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=max(1, int(jobs)), mp_context=multiprocessing.get_context("spawn"),
                               initializer=mo._inicializar_hijo, initargs=(threads_per_job,),
                               max_tasks_per_child=1)


def en_procesos(trabajos: Sequence[tuple[str, str]], funcion: Callable[[tuple], dict], extra: tuple, *,
                salida: Path, jobs: int, horas: float | None, coste: Callable[[str, str], float],
                threads_per_job: int | None, al_terminar: Callable[[dict, str, str], None]) -> list[tuple[str, str]]:
    """Lanza ``funcion((pista, id, *extra))`` en procesos, lentos primero, dentro de ``horas``.

    Igual que :func:`regimenes.benchmark.ablaciones.ejecutar`: si un hijo muere el pool se
    recrea y los trabajos en vuelo vuelven a pendientes; si solo había uno, o uno acumula
    ``MAX_ROTURAS`` roturas, se da por fallido. Devuelve los trabajos que no se arrancaron."""
    pendientes = sorted(trabajos, key=lambda t: (-coste(*t), t))
    limite = float("inf") if horas is None else float(horas) * 3600
    roturas: dict[tuple[str, str], int] = {}
    t0 = time.perf_counter()
    if not pendientes:
        return []
    with _thread_limits(threads_per_job):
        pool = _nuevo_pool(min(int(jobs), len(pendientes)), threads_per_job)
        try:
            en_vuelo: dict = {}
            while pendientes or en_vuelo:
                roto: list = []
                while len(en_vuelo) < jobs and pendientes:
                    restante = limite - (time.perf_counter() - t0)
                    siguiente = next((t for t in pendientes if coste(*t) <= restante), None)
                    if siguiente is None:
                        break
                    try:
                        futuro = pool.submit(funcion, (*siguiente, *extra))
                    except BrokenProcessPool:
                        roto.append(None)
                        break
                    pendientes.remove(siguiente)
                    en_vuelo[futuro] = siguiente
                if not en_vuelo and not roto:
                    break
                if not roto:
                    hechos, _ = wait(en_vuelo, return_when=FIRST_COMPLETED)
                    for futuro in hechos:
                        t, d = en_vuelo.pop(futuro)
                        try:
                            estado = futuro.result()
                        except BrokenProcessPool:
                            roto.append((t, d))
                            continue
                        except Exception as exc:  # noqa: BLE001
                            estado = {"trabajo": f"{t}_{d}", "estado": "error",
                                      "detalle": f"{type(exc).__name__}: {exc}"}
                        al_terminar(estado, t, d)
                if roto:
                    afectados = [a for a in roto if a is not None] + list(en_vuelo.values())
                    en_vuelo.clear()
                    pool.shutdown(wait=True, cancel_futures=True)
                    for a in afectados:
                        roturas[a] = roturas.get(a, 0) + 1
                        if len(afectados) == 1 or roturas[a] >= MAX_ROTURAS:
                            al_terminar({"trabajo": "_".join(a), "estado": "error",
                                         "detalle": f"BrokenProcessPool: el proceso murió ({roturas[a]} "
                                                    f"rotura(s), {len(afectados)} en vuelo)"}, *a)
                        else:
                            pendientes.append(a)
                    pendientes = sorted(pendientes, key=lambda t: (-coste(*t), t))
                    _log(salida, f"pool roto con {len(afectados)} trabajos en vuelo; se recrea")
                    pool = _nuevo_pool(min(int(jobs), max(1, len(pendientes))), threads_per_job)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
    return pendientes


def _registrar_error(salida: Path, estado: dict, t: str, d: str, tipo: str) -> None:
    if estado.get("estado") == "error":
        texto = json.dumps({**estado, "fecha": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False, indent=1)
        _write_atomic(ruta_error(salida, t, d, tipo), lambda tmp, x=texto: Path(tmp).write_text(x, encoding="utf-8"))
    _log(salida, str(estado))


# --------------------------------------------------------------------------- #
# Órdenes
# --------------------------------------------------------------------------- #
def _ids(track: str, detectores: Sequence[str] | None) -> list[str]:
    return [s.detector_id for s in seleccionar(todas_las_specs(track), detectores)]


def piloto(tracks: Sequence[str], detectores: Sequence[str] | None = None, *, salida: Path = SALIDA,
           jobs: int = 1, n_reajustes: int = N_PILOTO, threads_per_job: int | None = 1) -> pd.DataFrame:
    """Piloto de coste (ADR-010 §7) de cada variante seleccionada; actualiza ``piloto.csv``."""
    salida = Path(salida)
    trabajos = []
    for track in [t.upper() for t in tracks]:
        comprobar_huella(leer_preparacion(salida, track), track, salida)
        trabajos += [(track, d) for d in _ids(track, detectores)]
    filas: list[dict] = []

    def al_terminar(estado: dict, t: str, d: str) -> None:
        _registrar_error(salida, estado, t, d, "piloto")
        if estado.get("estado") == "ok":
            filas.append({k: v for k, v in estado.items() if k not in ("trabajo", "estado")})

    _log(salida, f"piloto: {len(trabajos)} variantes, {n_reajustes} reajustes al principio y al final, jobs={jobs}")
    en_procesos(trabajos, _piloto_seguro, (str(salida), int(n_reajustes)), salida=salida, jobs=jobs,
                horas=None, coste=lambda t, d: 0.0, threads_per_job=threads_per_job, al_terminar=al_terminar)
    previo = leer_piloto(salida)
    nuevo = pd.DataFrame(filas)
    if len(nuevo):
        claves = set(zip(nuevo["pista"], nuevo["id"]))
        previo = previo[[(a, b) not in claves for a, b in zip(previo["pista"], previo["id"])]]
    tabla = pd.concat([previo, nuevo], ignore_index=True).sort_values(["pista", "id"]).reset_index(drop=True)
    _write_atomic(salida / "piloto.csv", lambda tmp: tabla.to_csv(tmp, index=False))
    return tabla


def ejecutar(tracks: Sequence[str], detectores: Sequence[str] | None = None, *, salida: Path = SALIDA,
             jobs: int = 1, horas: float | None = None, threads_per_job: int | None = 1) -> list[dict]:
    """Lanza las variantes pendientes (sin CSV) en orden LPT dentro del presupuesto ``horas``."""
    salida = Path(salida)
    trabajos = []
    for track in [t.upper() for t in tracks]:
        comprobar_huella(leer_preparacion(salida, track), track, salida)
        trabajos += [(track, d) for d in _ids(track, detectores) if not ruta_trabajo(salida, track, d).exists()]
    c = costes(salida)
    coste = lambda t, d: c.get((t, d), COSTE_DESCONOCIDO)  # noqa: E731
    _log(salida, f"ejecutar: {len(trabajos)} pendientes, jobs={jobs}, horas={horas}, CPU estimada "
                 f"{sum(coste(*t) for t in trabajos) / 3600:.2f} h")
    estados: list[dict] = []

    def al_terminar(estado: dict, t: str, d: str) -> None:
        _registrar_error(salida, estado, t, d, "ejecutar")
        estados.append(estado)

    sin_arrancar = en_procesos(trabajos, _trabajo_seguro, (str(salida),), salida=salida, jobs=jobs, horas=horas,
                               coste=coste, threads_per_job=threads_per_job, al_terminar=al_terminar)
    if sin_arrancar:
        _log(salida, f"presupuesto agotado: {len(sin_arrancar)} trabajos sin arrancar")
    return estados


def estado(tracks: Sequence[str], detectores: Sequence[str] | None = None, *, salida: Path = SALIDA,
           jobs: int = 1, v2: Path = V2) -> pd.DataFrame:
    """Hechos, pendientes, errores, horas restantes y paneles cuyo OOS no es el de v2."""
    salida = Path(salida)
    c = costes(salida)
    filas = []
    for track in [t.upper() for t in tracks]:
        ids = _ids(track, detectores)
        hechos = [d for d in ids if ruta_trabajo(salida, track, d).exists()]
        pend = [d for d in ids if d not in hechos]
        cpu = sum(c.get((track, d), COSTE_DESCONOCIDO) for d in pend)
        try:
            ref = indice_v2(track, v2)
            distintos = [d for d in hechos if not pd.DatetimeIndex(pd.read_parquet(
                ruta_trabajo(salida, track, d).with_suffix(".parquet"), columns=["state"]).index).equals(ref)]
        except FileNotFoundError:
            distintos = ["(sin paneles de v2)"]
        filas.append({"pista": track, "hechos": len(hechos), "pendientes": len(pend),
                      "errores": sum(ruta_error(salida, track, d).exists() for d in pend),
                      "sin_piloto": sum((track, d) not in c for d in pend),
                      "horas_cpu_restantes": round(cpu / 3600, 2),
                      "horas_reloj_restantes": round(cpu / 3600 / max(1, jobs), 2),
                      "pendientes_ids": "|".join(pend), "oos_distinto_de_v2": "|".join(distintos)})
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------- #
# Lectura (ADR-010 §5 y §6)
# --------------------------------------------------------------------------- #
def cargar(salida: Path = SALIDA) -> pd.DataFrame:
    """Tabla larga de los CSV terminados (una fila por pista y variante)."""
    rutas_csv = [p for p in sorted(Path(salida).glob("[AB]_D*.csv"))]
    if not rutas_csv:
        return pd.DataFrame(columns=["pista", "id", "base", "menu", "score_deteccion"])
    return pd.concat([pd.read_csv(p) for p in rutas_csv], ignore_index=True)


def comparacion_pares(tabla: pd.DataFrame, v2: Path = V2) -> pd.DataFrame:
    """Criterio (a) del §5: cada variante no supervisada de P/R frente a su par de N en v2
    (``score_deteccion`` estrictamente mayor). Añade, sin peso, las comparaciones informativas
    (D15 frente a D09)."""
    filas = []
    for fila in tabla.itertuples(index=False):
        par = PARES.get(fila.id) or PARES_INFORMATIVOS.get(str(fila.base))
        if par is None:
            continue
        s_par = float(mo.fila_n_v2(fila.pista, par, v2).loc[0, "score_deteccion"])
        s = float(fila.score_deteccion)
        filas.append({"pista": fila.pista, "id": fila.id, "menu": fila.menu, "par_v2": par,
                      "score": s, "score_par": s_par, "delta_score": s - s_par,
                      "a_mejora_par": bool(s > s_par), "cuenta_para_veredicto": fila.id in PARES})
    return pd.DataFrame(filas)


def _senales_pool(track: str, ids: Sequence[str], salida: Path) -> pd.DataFrame:
    from regimenes.evaluacion import mcs

    if not ids:
        return pd.DataFrame()
    pan = mcs.cargar_paneles(track, Path(salida), list(ids))
    return mcs.senales_crisis(pan, n_states=mcs.n_states_metricas(track, Path(salida), list(ids)))


def mcs_conjunto(track: str, ids_pool: Sequence[str], *, salida: Path = SALIDA, v2: Path = V2,
                 **kwargs) -> pd.DataFrame:
    """MCS de los 12 de v2 (N, con los univariantes de U) + las variantes ``ids_pool`` + los 5
    baselines de la vara, con la vara definitiva (``ablaciones_lectura.MCS_KW``: λ = 1, α = 0,10,
    ``T_max``, bootstrap estacionario de bloque = mediana de las crisis, 66 en A y 50 en B,
    5.000 réplicas de razón, semilla 0)."""
    from regimenes.benchmark import ablaciones_lectura as al
    from regimenes.evaluacion import mcs
    from regimenes.evaluacion import ranking as rk

    track = track.upper()
    base = al._senales_v2(track, v2)
    pool = _senales_pool(track, ids_pool, salida)
    if len(pool) and not pd.DatetimeIndex(pool.index).equals(pd.DatetimeIndex(base.index)):
        raise ValueError(f"{track}: los paneles del pool no tienen el índice OOS de v2")
    sen = pd.concat([base, pool, al.senales_baselines_vara(track, v2=v2)], axis=1)
    win = rk.track_crisis_windows(track)
    bloque = mcs.longitud_bloque_crisis(pd.DatetimeIndex(sen.index), win)
    _, res = mcs.mcs_senales(sen, win, block_len=bloque, **{**al.MCS_KW, **kwargs})
    t = res.tabla.copy()
    t.index.name = "modelo"
    return t.assign(pista=track, bloque=bloque)


def criterio(comp: pd.DataFrame, tablas_mcs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Veredicto del §5: en cada pista, ¿alguna variante de P/R cumple (a) y (b) a la vez?"""
    filas = []
    for track, tabla in tablas_mcs.items():
        en_mcs = [str(i) for i in tabla.index[tabla["incluido"].astype(bool)]]
        c = comp[(comp["pista"] == track) & comp["cuenta_para_veredicto"]] if len(comp) else comp
        cumplen = [r.id for r in c.itertuples() if r.a_mejora_par and r.id in en_mcs]
        filas.append({"pista": track, "candidatas": "|".join(c["id"]) if len(c) else "",
                      "a_mejoran_par": "|".join(c.loc[c["a_mejora_par"], "id"]) if len(c) else "",
                      "b_en_mcs": "|".join(i for i in en_mcs if i in set(c["id"])) if len(c) else "",
                      "cumplen_a_y_b": "|".join(cumplen), "cumple_pista": bool(cumplen),
                      "mcs_miembros": "|".join(en_mcs), "vol_p90_fuera_del_mcs": "VOL_P90" not in en_mcs})
    out = pd.DataFrame(filas)
    n = int(out["cumple_pista"].sum()) if len(out) else 0
    out["veredicto"] = ("aporta" if n == 2 else "indicio (una sola pista)" if n == 1
                        else "no aporta: ni comprimido ni en bruto, lo simple sigue ganando")
    return out


def informe_d18(track: str, salida: Path = SALIDA) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frecuencia con la que D18 da peso > 0 a cada columna y a cada bloque, por reajuste."""
    from regimenes.detectores.pool.pool_crudo import bloque_de_columna

    ruta = ruta_reajustes(salida, track, "D18-P")
    if not ruta.exists():
        return pd.DataFrame(), pd.DataFrame()
    r = pd.read_csv(ruta)
    cols = r["d18_columnas_peso_pos"].fillna("").map(lambda s: [c for c in str(s).split("|") if c])
    bloque = bloque_de_columna(track)
    n = len(r)
    cuenta = pd.Series([c for fila in cols for c in fila]).value_counts()
    por_col = pd.DataFrame({"columna": cuenta.index, "n_reajustes": cuenta.values,
                            "frecuencia": cuenta.values / n}).assign(
        bloque=lambda d: d["columna"].map(bloque), pista=track)
    por_bloque = pd.Series([bloque.get(c, "?") for fila in cols for c in set(fila)]).value_counts()
    con_bloque = pd.Series([b for fila in cols for b in {bloque.get(c, "?") for c in fila}]).value_counts()
    tb = pd.DataFrame({"bloque": por_bloque.index, "n_columnas_reajuste": por_bloque.values}).assign(
        n_reajustes_con_alguna=lambda d: d["bloque"].map(con_bloque).fillna(0).astype(int),
        frecuencia_reajustes=lambda d: d["n_reajustes_con_alguna"] / n, pista=track)
    return por_col, tb


def orden_canonico(tabla: pd.DataFrame) -> pd.DataFrame:
    cols = ["pista", "id", "n_reajustes", "n_reajustes_orden_invertido"]
    return tabla[[c for c in cols if c in tabla]].copy() if len(tabla) else pd.DataFrame(columns=cols)


def leer(salida: Path = SALIDA, *, v2: Path = V2, mcs: bool = True) -> list[Path]:
    """Escribe en ``salida`` (atómico) las tablas del §5 y el §6 con lo que haya terminado."""
    salida = Path(salida)
    tabla = cargar(salida)
    hechas: list[Path] = []

    def csv(nombre: str, df: pd.DataFrame, index: bool = False) -> None:
        _write_atomic(salida / nombre, lambda tmp: df.to_csv(tmp, index=index))
        hechas.append(salida / nombre)

    comp = comparacion_pares(tabla, v2) if len(tabla) else pd.DataFrame(
        columns=["pista", "id", "a_mejora_par", "cuenta_para_veredicto"])
    csv("comparacion_pares.csv", comp)
    csv("orden_canonico.csv", orden_canonico(tabla))
    for track in PISTAS:
        por_col, por_bloque = informe_d18(track, salida)
        if len(por_col):
            csv(f"d18_columnas_{track}.csv", por_col)
            csv(f"d18_bloques_{track}.csv", por_bloque)
    if not mcs or not len(tabla):
        return hechas
    tablas = {}
    techo = []
    for track in PISTAS:
        ids = tabla.loc[tabla["pista"] == track, "id"].astype(str).tolist()
        no_sup = [i for i in ids if not es_supervisado(i)]
        sup = [i for i in ids if es_supervisado(i)]
        tablas[track] = mcs_conjunto(track, no_sup, salida=salida, v2=v2)
        csv(f"mcs_NUPR_{track}.csv", tablas[track], index=True)
        if sup:
            t_sup = mcs_conjunto(track, no_sup + sup, salida=salida, v2=v2)
            csv(f"mcs_supervisados_{track}.csv", t_sup, index=True)
            scores_v2 = {d: float(mo.fila_n_v2(track, d, v2).loc[0, "score_deteccion"])
                         for d in tablas[track].index if str(d).startswith("D") and "-" not in str(d)}
            scores_pool = dict(zip(tabla.loc[tabla["pista"] == track, "id"], tabla.loc[tabla["pista"] == track,
                                                                                     "score_deteccion"]))
            no_sup_scores = {**scores_v2, **{i: float(scores_pool[i]) for i in no_sup}}
            mejor = max(no_sup_scores, key=no_sup_scores.get)
            en = set(t_sup.index[t_sup["incluido"].astype(bool)].astype(str))
            for i in sup:
                techo.append({"pista": track, "id": i, "score": float(scores_pool[i]),
                              "mejor_no_supervisado": mejor, "score_mejor": no_sup_scores[mejor],
                              "mejora_al_mejor": bool(float(scores_pool[i]) > no_sup_scores[mejor]),
                              "en_mcs_conjunto": i in en})
    csv("criterio_adr010.csv", criterio(comp, tablas))
    if techo:
        csv("techo_supervisados.csv", pd.DataFrame(techo))
    return hechas


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("orden", choices=["preparar", "piloto", "ejecutar", "estado", "leer"])
    parser.add_argument("--pista", nargs="+", default=list(PISTAS), choices=["A", "B", "a", "b"])
    parser.add_argument("--detector", nargs="+", default=None, help="ids o comodines (D10-P, 'D1*'...)")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--horas", type=float, default=7.5, help="presupuesto de reloj de ejecutar (0 = sin límite)")
    parser.add_argument("--reajustes", type=int, default=N_PILOTO, help="reajustes del piloto en cada extremo")
    parser.add_argument("--threads-per-job", type=int, default=1)
    parser.add_argument("--salida", type=Path, default=SALIDA)
    parser.add_argument("--sin-mcs", action="store_true", help="leer sin recalcular los MCS")
    args = parser.parse_args(argv)
    pistas = [p.upper() for p in args.pista]
    hilos = args.threads_per_job or None
    if args.orden == "preparar":
        preparar(pistas, salida=args.salida)
    elif args.orden == "piloto":
        tabla = piloto(pistas, args.detector, salida=args.salida, jobs=args.jobs, n_reajustes=args.reajustes,
                       threads_per_job=hilos)
        print(tabla.to_string(index=False), flush=True)
    elif args.orden == "ejecutar":
        estados = ejecutar(pistas, args.detector, salida=args.salida, jobs=args.jobs, horas=args.horas or None,
                           threads_per_job=hilos)
        print(estado(pistas, args.detector, salida=args.salida, jobs=args.jobs).to_string(index=False), flush=True)
        errores = [e for e in estados if e["estado"] == "error"]
        if errores:
            raise SystemExit(f"{len(errores)} trabajos con error (ver _errores/)")
    elif args.orden == "estado":
        print(estado(pistas, args.detector, salida=args.salida, jobs=args.jobs).to_string(index=False), flush=True)
    else:
        for ruta in leer(args.salida, mcs=not args.sin_mcs):
            print(ruta)


if __name__ == "__main__":
    main()
