"""Trabajos del aumento, cache por huella, piloto de coste y ejecucion con presupuesto de horas.

Un trabajo = (pista, detector, fuente, brazo, proporcion, repeticion) o, para el
suelo de ruido, (pista, detector, semilla alternativa) con proporcion 0. Cada
trabajo ejecuta el ``walk_forward`` del benchmark (mismos ``train_size``,
``step``, ``expanding`` y contexto) con una fabrica de ``DetectorAumentado`` y
guarda:

- ``data/sinteticos/aumento/paneles/<P>/<clave>.parquet`` (gitignored): estados OOS.
- ``data/sinteticos/aumento/pliegues/<P>/<clave>.csv``: ancla, base y filas
  sinteticas por pliegue, y la ocupacion de estados.
- ``results/sinteticos/aumento/filas/<P>/<clave>.json``: resumen de deteccion ADR-003
  y diagnosticos, con la huella (lo que da el trabajo por terminado).

Huella = config pre-registrada relevante + trabajo + spec + codigo de
``anclas.py``/``muestras.py``/``envoltorio.py`` + huella del benchmark de ese
detector + anclas validas de la fuente. NO entra ``ejecucion.py``.

``ejecutar(horas=...)`` solo arranca un trabajo si, segun su estimacion de
tiempo, termina antes de agotar el presupuesto; el resto queda para la sesion
siguiente (lo terminado esta en cache).
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, wait
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import functools
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd

from regimenes.sinteticos.aumento import anclas as an
from regimenes.sinteticos.aumento import muestras as mu
from regimenes.sinteticos.aumento.config import (
    DIR_DATOS,
    DIR_RESULTADOS,
    config_aumento,
    detectores_pista,
    fuentes,
    hash_ficheros,
)
from regimenes.sinteticos.aumento.config import un_hilo
from regimenes.sinteticos.aumento.envoltorio import fabrica

SIN_FUENTE = "ninguna"
# Version del protocolo de ejecucion (correr_walk_forward, definicion de crisis, diagnosticos): subirla
# invalida la cache de todos los trabajos.
VERSION_EJECUCION = 1
CADA_OCUPACION = 12            # pliegues entre diagnosticos de ocupacion
FACTOR_SEGURIDAD = 1.15
TOLERANCIA_P = 1e-9            # ruido numerico de BLAS en p_crisis (D08 varia ~1e-13 entre ajustes)        # margen sobre la estimacion de tiempo de cada trabajo
MULT_POR_DEFECTO = 2.0         # multiplicador de coste a p = 1 si aun no hay piloto
_CLAVES_HUELLA = ("semilla", "anclas", "brazos", "trayectoria_largo", "proporciones", "min_run")


# --------------------------------------------------------------------------- trabajos
@dataclass(frozen=True)
class Trabajo:
    pista: str
    detector: str
    fuente: str
    brazo: str
    proporcion: float
    rep: int = 0
    semilla_det: int | None = None

    @property
    def clave(self) -> str:
        base = f"{self.detector}__{self.fuente}__{self.brazo}__p{self.proporcion:.2f}__r{self.rep}"
        return base if self.semilla_det is None else f"{base}__s{self.semilla_det}"

    @property
    def es_suelo(self) -> bool:
        return self.semilla_det is not None


def trabajos(pistas: Iterable[str] | None = None, cfg: dict | None = None) -> list[Trabajo]:
    """Todos los trabajos pre-registrados (aumento + suelo de ruido)."""
    cfg = config_aumento() if cfg is None else cfg
    lista = []
    for pista in (pistas or cfg["pistas"]):
        pista = pista.upper()
        for det in detectores_pista(pista, cfg):
            for rep in range(int(cfg["repeticiones"][pista])):
                for fuente in fuentes(cfg):
                    for brazo in mu.BRAZOS:
                        for p in cfg["proporciones"]:
                            lista.append(Trabajo(pista, det, fuente, brazo, float(p), rep))
        for det in cfg["suelo_ruido"]["detectores"].get(pista, []):
            for s in cfg["suelo_ruido"]["semillas"]:
                lista.append(Trabajo(pista, det, SIN_FUENTE, "-", 0.0, 0, int(s)))
    return lista


def lote(tr: Trabajo, cfg: dict | None = None) -> int:
    """Indice del primer lote de ``ejecucion.prioridad`` que contiene el trabajo."""
    cfg = config_aumento() if cfg is None else cfg
    prioridad = cfg["ejecucion"]["prioridad"]
    for k, regla in enumerate(prioridad):
        if regla.get("pista", tr.pista).upper() != tr.pista:
            continue
        if "detectores" in regla and tr.detector not in regla["detectores"]:
            continue
        if "brazo" in regla and not tr.es_suelo and regla["brazo"] != tr.brazo:
            continue
        if "repeticion" in regla and int(regla["repeticion"]) != tr.rep:
            continue
        return k
    return len(prioridad)


# --------------------------------------------------------------------------- rutas
def ruta_fila(tr: Trabajo, base_res: Path | None = None) -> Path:
    base_res = DIR_RESULTADOS if base_res is None else Path(base_res)
    return base_res / "filas" / tr.pista / f"{tr.clave}.json"


def ruta_panel(tr: Trabajo, base: Path | None = None) -> Path:
    base = DIR_DATOS if base is None else Path(base)
    return base / "paneles" / tr.pista / f"{tr.clave}.parquet"


def ruta_pliegues(tr: Trabajo, base: Path | None = None) -> Path:
    base = DIR_DATOS if base is None else Path(base)
    return base / "pliegues" / tr.pista / f"{tr.clave}.csv"


# --------------------------------------------------------------------------- datos de la pista
@functools.lru_cache(maxsize=2)
def _panel_pista(pista: str) -> pd.DataFrame:
    from regimenes.benchmark.ejecucion import load_track_panel

    return load_track_panel(pista)


@functools.lru_cache(maxsize=2)
def _indice_pista(pista: str) -> pd.DatetimeIndex:
    from regimenes.benchmark.ejecucion import common_track_index

    return common_track_index(pista, _panel_pista(pista))


def spec_de(pista: str, detector: str, semilla_det: int | None = None):
    """Spec del benchmark (con ``random_state`` cambiado para el suelo de ruido)."""
    from regimenes.detectores.registry import detector_specs

    for spec in detector_specs(pista):
        if spec.detector_id == detector.upper():
            if semilla_det is None:
                return spec
            if "random_state" not in spec.params:
                raise ValueError(f"{detector} no tiene random_state: no entra en el suelo de ruido.")
            return replace(spec, params={**spec.params, "random_state": int(semilla_det)})
    raise KeyError(f"No existe el detector {detector} en la pista {pista}.")


def datos_trabajo(pista: str, spec) -> tuple[pd.DataFrame, pd.Series, int]:
    """``X``, ``market_returns`` y train inicial, exactamente como ``benchmark.run_one``."""
    from regimenes.benchmark.ejecucion import DEFAULT_TRAIN_DAYS, configure_evaluation

    configure_evaluation(pista)
    panel = _panel_pista(pista)
    X = panel.loc[_indice_pista(pista), list(spec.features)]
    return X, panel["SP500_ret"].reindex(X.index), int(DEFAULT_TRAIN_DAYS[pista])


@functools.lru_cache(maxsize=64)
def huella_benchmark(pista: str, detector: str) -> str:
    from regimenes.benchmark.cache import _cache_fingerprint
    from regimenes.benchmark.ejecucion import DEFAULT_TRAIN_DAYS

    return _cache_fingerprint(pista, spec_de(pista, detector), DEFAULT_TRAIN_DAYS[pista])


def panel_benchmark(pista: str, detector: str) -> pd.DataFrame:
    """Panel OOS del benchmark (proporcion 0); exige que su cache siga vigente."""
    from regimenes.benchmark.cache import DEFAULT_OUTPUT, _cache_matches, _metric_path, _panel_path

    metric, panel = _metric_path(DEFAULT_OUTPUT, pista, detector), _panel_path(DEFAULT_OUTPUT, pista, detector)
    if not _cache_matches(metric, panel, huella_benchmark(pista, detector)):
        raise RuntimeError(
            f"La cache del benchmark {pista}/{detector} no esta vigente: ejecuta "
            f"`python -m regimenes.benchmark --track {pista} --detector {detector}`."
        )
    return pd.read_parquet(panel)


def huella(tr: Trabajo, cfg: dict | None = None, registro: pd.DataFrame | None = None) -> str:
    cfg = config_aumento() if cfg is None else cfg
    spec = spec_de(tr.pista, tr.detector, tr.semilla_det)
    validas: list[tuple[str, str]] = []
    if tr.fuente != SIN_FUENTE and tr.proporcion > 0:
        registro = an.leer_registro(tr.pista) if registro is None else registro
        parte = registro[(registro["fuente"] == tr.fuente) & registro["valida"]].sort_values("tau")
        # huella de cada generador por ancla: cubre su ajuste, su validacion y su reservorio
        validas = [(pd.Timestamp(r.tau).date().isoformat(), str(r.huella)) for r in parte.itertuples()]
    contenido = {
        "cfg": {k: cfg[k] for k in _CLAVES_HUELLA},
        "trabajo": asdict(tr),
        "spec": {**spec.to_row(), "params": spec.params, "features": list(spec.features)},
        "codigo": hash_ficheros("anclas.py", "muestras.py", "envoltorio.py"),
        "benchmark": huella_benchmark(tr.pista, tr.detector),
        "anclas_validas": validas,
        "version_ejecucion": VERSION_EJECUCION,
        "cada_ocupacion": CADA_OCUPACION,
    }
    return hashlib.sha256(json.dumps(contenido, sort_keys=True, default=str).encode()).hexdigest()


def en_cache(tr: Trabajo, h: str | None = None, base_res: Path | None = None, base: Path | None = None) -> bool:
    """Fila JSON con la huella vigente Y panel local presente."""
    ruta = ruta_fila(tr, base_res)
    if not ruta.exists() or not ruta_panel(tr, base).exists():
        return False
    try:
        fila = json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return fila.get("huella") == (huella(tr) if h is None else h)


# --------------------------------------------------------------------------- un trabajo
def correr_walk_forward(spec, X: pd.DataFrame, mr: pd.Series, train_size: int, proveedor=None,
                        ocupacion: list | None = None) -> pd.DataFrame:
    """``walk_forward`` del benchmark con la fabrica del envoltorio."""
    from regimenes import evaluacion as ev

    fab = fabrica(spec, proveedor, ocupacion=ocupacion,
                  cada_ocupacion=CADA_OCUPACION if proveedor is not None else 0)
    return ev.walk_forward(fab, X, market_returns=mr, train_size=train_size, min_train=train_size,
                           step=spec.step, expanding=spec.expanding)


def _escribir(ruta: Path, escritor) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_name(f".{ruta.stem}.{os.getpid()}.tmp{ruta.suffix}")
    escritor(tmp)
    tmp.replace(ruta)


def ejecutar_trabajo(tr: Trabajo, *, forzar: bool = False, base: Path | None = None,
                     base_res: Path | None = None) -> dict[str, Any]:
    """Ejecuta un trabajo (o lo lee de cache) y devuelve su estado."""
    from regimenes import evaluacion as ev
    from regimenes.evaluacion.metricas import detection_summary, mean_regime_duration, switching_rate

    cfg = config_aumento()
    registro = an.leer_registro(tr.pista) if tr.proporcion > 0 else None
    h = huella(tr, cfg, registro)
    if not forzar and en_cache(tr, h, base_res):
        return {"clave": tr.clave, "pista": tr.pista, "estado": "cache", "segundos": 0.0}
    spec = spec_de(tr.pista, tr.detector, tr.semilla_det)
    X, mr, train_size = datos_trabajo(tr.pista, spec)
    proveedor = None
    if tr.proporcion > 0:
        proveedor = mu.proveedor_para(tr.pista, tr.fuente, tr.brazo, tr.proporcion, tr.rep,
                                      registro=registro, cfg=cfg, base=base)
    ocupacion: list = []
    t0 = time.perf_counter()
    wf = correr_walk_forward(spec, X, mr, train_size, proveedor, ocupacion)
    segundos = time.perf_counter() - t0
    crisis = spec.factory().n_states - 1
    flags = wf["state"].eq(crisis)
    resumen = detection_summary(flags, dict(ev.CRISIS_WINDOWS), min_run=int(cfg["min_run"]))
    pliegues = pd.DataFrame(proveedor.registro_pliegues if proveedor else [])
    ocup = pd.DataFrame(ocupacion)
    diag: dict[str, Any] = {"n_pliegues": int(wf["fold"].nunique())}
    if not pliegues.empty:
        con = pliegues["n_sint"] > 0
        diag.update({
            "pliegues_sin_aumento": int((~con).sum()),
            "n_sint_medio": float(pliegues.loc[con, "n_sint"].mean()) if con.any() else 0.0,
            "n_sint_max": int(pliegues["n_sint"].max()),
            "ratio_sint_train_medio": float((pliegues["n_sint"] / pliegues["n_train"]).mean()),
            "prevalencia_bloque": (float(pliegues["dias_crisis_bloque"].sum() / pliegues["n_sint"].sum())
                                   if pliegues["n_sint"].sum() else np.nan),
            "n_anclas_usadas": int(pliegues["tau"].dropna().nunique()),
        })
    if not ocup.empty and f"sint_{crisis}" in ocup:
        diag.update({
            "ocup_crisis_sint_media": float(ocup[f"sint_{crisis}"].mean()),
            "ocup_crisis_real_media": float(ocup[f"real_{crisis}"].mean()),
            # pliegues en que algun estado ocupa el sintetico y nunca el real
            "pliegues_estado_solo_sint": int(sum(
                any(r.get(f"sint_{s}", 0) > 0 and r.get(f"real_{s}", 0) == 0 for s in range(crisis + 1))
                for r in ocup.to_dict("records"))),
            "pliegues_ocupacion": int(len(ocup)),
        })
    fila = {
        **asdict(tr), "clave": tr.clave, "huella": h, "segundos": segundos,
        "version_ejecucion": VERSION_EJECUCION,
        "n_states": crisis + 1, "switching_rate": switching_rate(wf["state"]),
        "mean_regime_duration": mean_regime_duration(wf["state"]),
        **resumen,
        **diag,
        "fin_utc": datetime.now(timezone.utc).isoformat(),
    }
    fila = {k: (None if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v) for k, v in fila.items()}
    panel = wf.copy()
    panel.attrs = {}
    _escribir(ruta_panel(tr, base), lambda tmp: panel.to_parquet(tmp))
    if not pliegues.empty or not ocup.empty:
        tabla = pliegues.merge(ocup, on="fin_train", how="left") if not ocup.empty and not pliegues.empty \
            else (pliegues if not pliegues.empty else ocup)
        _escribir(ruta_pliegues(tr, base), lambda tmp: tabla.to_csv(tmp, index=False))
    texto = json.dumps(fila, ensure_ascii=False, indent=1, default=str)
    _escribir(ruta_fila(tr, base_res), lambda tmp: Path(tmp).write_text(texto, encoding="utf-8"))
    return {"clave": tr.clave, "pista": tr.pista, "estado": "ok", "segundos": segundos}


def _trabajo_seguro(tr: Trabajo) -> dict[str, Any]:
    inicio = datetime.now(timezone.utc).isoformat()
    try:
        with un_hilo():
            estado = ejecutar_trabajo(tr)
    except Exception as exc:  # noqa: BLE001 - el estado debe persistir
        estado = {"clave": tr.clave, "pista": tr.pista, "estado": f"error: {type(exc).__name__}: {exc}",
                  "segundos": np.nan}
    return {**estado, "inicio_utc": inicio, "fin_utc": datetime.now(timezone.utc).isoformat()}


# --------------------------------------------------------------------------- piloto de coste
def ultimos_pliegues(spec, X: pd.DataFrame, mr: pd.Series, train_size: int, n: int):
    """``(X, mr, train_size)`` que reproducen exactamente los ``n`` ultimos pliegues del benchmark."""
    folds = mu.pliegues(len(X), train_size, spec.step, spec.expanding)
    t0 = int(folds["t"].iloc[max(0, len(folds) - int(n))])
    if spec.expanding:
        return X, mr, t0                                  # train [0, t) identico
    desde = t0 - train_size                               # ventana movil identica
    return X.iloc[desde:], mr.iloc[desde:], train_size


def piloto(pista: str, detector: str, *, fuente: str | None = None, n_pliegues: int | None = None) -> dict[str, Any]:
    """Tiempos de los ultimos pliegues con p = 0 y p = 1 (completas) y prueba bit a bit con p = 0."""
    cfg = config_aumento()
    fuente = fuente or cfg["generadores"][0]
    n = int(n_pliegues or cfg["ejecucion"]["piloto_pliegues"])
    spec = spec_de(pista, detector)
    X, mr, train_size = datos_trabajo(pista, spec)
    Xp, mrp, ts = ultimos_pliegues(spec, X, mr, train_size, n)
    t = time.perf_counter()
    wf0 = correr_walk_forward(spec, Xp, mrp, ts)
    seg0 = time.perf_counter() - t
    ref = panel_benchmark(pista, detector).reindex(wf0.index)
    estados_iguales = bool(np.array_equal(wf0["state"].to_numpy(), ref["state"].to_numpy()))
    dif_p = float(np.nanmax(np.abs(wf0["p_crisis"].to_numpy() - ref["p_crisis"].to_numpy())))
    prov = mu.proveedor_para(pista, fuente, "completas", 1.0, 0, cfg=cfg)
    t = time.perf_counter()
    correr_walk_forward(spec, Xp, mrp, ts, prov)
    seg1 = time.perf_counter() - t
    return {"pista": pista, "detector": detector, "fuente": fuente, "n_pliegues": n,
            "seg_p0": seg0, "seg_p1": seg1, "mult_p1": seg1 / seg0 if seg0 > 0 else np.nan,
            "estados_iguales": estados_iguales, "max_dif_p_crisis": dif_p,
            "igual_benchmark": estados_iguales and dif_p <= TOLERANCIA_P, "n_sint_medio": float(pd.DataFrame(prov.registro_pliegues)["n_sint"].mean())}


def _piloto_seguro(args) -> dict[str, Any]:
    pista, detector = args
    try:
        with un_hilo():
            return piloto(pista, detector)
    except Exception as exc:  # noqa: BLE001
        return {"pista": pista, "detector": detector, "error": f"{type(exc).__name__}: {exc}"}


def ejecutar_piloto(pistas: Iterable[str], n_jobs: int = 1) -> pd.DataFrame:
    from regimenes.sinteticos.aumento import _pool

    lista = [(p.upper(), d) for p in pistas for d in detectores_pista(p)]
    filas: list[dict] = []
    _pool.mapa(_piloto_seguro, lista, n_jobs,
               lambda a, r: filas.append({"pista": a[0], "detector": a[1], **r}))
    tabla = pd.DataFrame(filas).sort_values(["pista", "detector"])
    ruta = DIR_RESULTADOS / "piloto_coste.csv"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(ruta, index=False)
    return tabla


# --------------------------------------------------------------------------- estimacion y presupuesto
@functools.lru_cache(maxsize=1)
def _segundos_benchmark() -> dict[tuple[str, str], float]:
    from regimenes import rutas

    tabla = pd.read_csv(rutas.RESULTS_BENCHMARK / "run_status.csv")
    # un benchmark reanudado reescribe run_status con segundos = 0 en los aciertos de cache
    tabla = tabla[(tabla["estado"] == "ok") & (pd.to_numeric(tabla["segundos"], errors="coerce") > 0)]
    return {(r.pista, r.id): float(r.segundos) for r in tabla.itertuples()}


@functools.lru_cache(maxsize=1)
def _multiplicadores() -> dict[tuple[str, str], float]:
    ruta = DIR_RESULTADOS / "piloto_coste.csv"
    if not ruta.exists():
        return {}
    tabla = pd.read_csv(ruta)
    if "mult_p1" not in tabla or "igual_benchmark" not in tabla:
        return {}
    tabla = tabla[tabla["igual_benchmark"].astype(bool)]
    return {(r.pista, r.detector): float(r.mult_p1) for r in tabla.itertuples() if np.isfinite(r.mult_p1)}


@functools.lru_cache(maxsize=2)
def _prevalencia(pista: str) -> float:
    from regimenes.sinteticos.datos import regimen_referencia

    return float(regimen_referencia(_indice_pista(pista), pista).mean())


def estimacion(tr: Trabajo) -> float:
    """Segundos estimados: benchmark x (1 + (mult - 1) x p efectiva) x margen."""
    base = _segundos_benchmark().get((tr.pista, tr.detector), 3600.0)
    mult = _multiplicadores().get((tr.pista, tr.detector), MULT_POR_DEFECTO)
    p_eff = tr.proporcion * (1.0 if tr.brazo == "completas" else _prevalencia(tr.pista))
    return base * (1.0 + (mult - 1.0) * p_eff) * FACTOR_SEGURIDAD


def pendientes(pistas: Iterable[str] | None = None) -> pd.DataFrame:
    """Trabajos sin cache valida, con lote y estimacion (en el orden en que se lanzarian)."""
    cfg = config_aumento()
    filas = []
    registros = {}
    for tr in trabajos(pistas, cfg):
        if tr.proporcion > 0 and tr.pista not in registros:
            registros[tr.pista] = an.leer_registro(tr.pista)
        if en_cache(tr, huella(tr, cfg, registros.get(tr.pista))):
            continue
        filas.append({"trabajo": tr, "clave": tr.clave, "pista": tr.pista, "lote": lote(tr, cfg),
                      "estimacion_s": estimacion(tr)})
    tabla = pd.DataFrame(filas, columns=["trabajo", "clave", "pista", "lote", "estimacion_s"])
    return tabla.sort_values(["lote", "estimacion_s"], ascending=[True, False]).reset_index(drop=True)


def comprobar_benchmark(pistas: Iterable[str]) -> None:
    """Falla antes de calcular nada si la cache del benchmark de algun detector no esta vigente."""
    for pista in pistas:
        for det in detectores_pista(pista):
            panel_benchmark(pista.upper(), det)


class Cerrojo:
    """Fichero de cerrojo: impide dos ``ejecutar`` a la vez sobre los mismos resultados."""

    def __init__(self, ruta: Path) -> None:
        self.ruta = ruta

    def __enter__(self):
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.ruta, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError(
                f"Ya hay una ejecucion en curso ({self.ruta}). Si no es asi (el PC se apago), borra ese fichero."
            ) from exc
        os.write(fd, f"{os.getpid()} {datetime.now(timezone.utc).isoformat()}".encode())
        os.close(fd)
        return self

    def __exit__(self, *exc):
        self.ruta.unlink(missing_ok=True)


def ejecutar(horas: float | None = None, n_jobs: int | None = None, pistas: Iterable[str] | None = None,
             progreso: bool = True) -> pd.DataFrame:
    """Lanza los trabajos pendientes que caben en ``horas`` (reanudable).

    Solo arranca un trabajo si su estimacion termina antes del limite; los que no
    caben quedan para la sesion siguiente. Si un hijo muere (pool roto), los
    trabajos en curso se anotan como error y la sesion sigue con un pool nuevo.
    """
    from concurrent.futures.process import BrokenProcessPool

    from regimenes.sinteticos.aumento import _pool

    cfg = config_aumento()
    pistas = [p.upper() for p in (pistas or cfg["pistas"])]
    horas = float(cfg["ejecucion"]["horas"] if horas is None else horas)
    n_jobs = int(cfg["ejecucion"]["n_jobs"] if n_jobs is None else n_jobs)
    comprobar_benchmark(pistas)
    with Cerrojo(DIR_RESULTADOS / ".ejecutando"):
        cola = pendientes(pistas)[["trabajo", "estimacion_s"]].to_records(index=False).tolist()
        limite = time.time() + horas * 3600.0
        nunca = [tr.clave for tr, est in cola if est > horas * 3600.0]
        estados: list[dict] = []
        log = DIR_RESULTADOS / "estado_ejecucion.csv"
        if progreso:
            print(f"[aumento] {len(cola)} trabajos pendientes; presupuesto {horas:.1f} h con {n_jobs} procesos",
                  flush=True)
            if nunca:
                print(f"[aviso] {len(nunca)} trabajos estiman mas que el presupuesto y no se lanzaran: "
                      f"{nunca[:5]}...", flush=True)

        def anotar(tr, est, estado):
            estado = {"clave": tr.clave, "pista": tr.pista, "segundos": np.nan, **estado,
                      "lote": lote(tr, cfg), "estimacion_s": est}
            estados.append(estado)
            columnas = ["clave", "pista", "estado", "segundos", "lote", "estimacion_s", "inicio_utc", "fin_utc"]
            pd.DataFrame([estado]).reindex(columns=columnas).to_csv(log, mode="a", header=not log.exists(),
                                                                    index=False)
            if progreso:
                print(f"[aumento] {str(estado['estado'])[:40]:>5} {tr.pista} {tr.clave} "
                      f"({estado['segundos']:.0f} s; estimado {est:.0f} s); quedan {len(cola)}", flush=True)

        while True:
            pool = _pool.nuevo_pool(n_jobs)
            en_curso: dict = {}
            roto = False
            try:
                while True:
                    while len(en_curso) < n_jobs:
                        ahora = time.time()
                        k = next((i for i, (_, est) in enumerate(cola) if ahora + est <= limite), None)
                        if k is None:
                            break
                        tr, est = cola.pop(k)
                        en_curso[pool.submit(_trabajo_seguro, tr)] = (tr, est)
                    if not en_curso:
                        break
                    hechos, _ = wait(en_curso, return_when=FIRST_COMPLETED)
                    for fut in hechos:
                        tr, est = en_curso.pop(fut)
                        try:
                            anotar(tr, est, fut.result())
                        except BrokenProcessPool as exc:
                            roto = True
                            anotar(tr, est, {"estado": f"error: proceso hijo caido ({exc})"})
                    if roto:
                        for fut, (tr, est) in en_curso.items():
                            anotar(tr, est, {"estado": "error: pool roto por otro trabajo"})
                        break
            finally:
                pool.shutdown(wait=not roto, cancel_futures=True)
            if not roto:
                break
        if progreso:
            print(f"[aumento] fin de la sesion: {len(estados)} trabajos; {len(cola)} quedan para la siguiente",
                  flush=True)
    return pd.DataFrame(estados)


def consolidar_filas(pistas: Iterable[str] | None = None, base_res: Path | None = None) -> pd.DataFrame:
    """Une en ``filas_<P>.csv`` las filas de los trabajos con cache VIGENTE (huella y panel)."""
    cfg = config_aumento()
    base_res = DIR_RESULTADOS if base_res is None else Path(base_res)
    tablas = []
    for pista in (pistas or cfg["pistas"]):
        pista = pista.upper()
        registro = an.leer_registro(pista, base_res) if an.ruta_registro(pista, base_res).exists() else None
        filas, faltan = [], 0
        for tr in trabajos([pista], cfg):
            if registro is None and tr.proporcion > 0:
                faltan += 1
                continue
            if en_cache(tr, huella(tr, cfg, registro), base_res):
                filas.append(json.loads(ruta_fila(tr, base_res).read_text(encoding="utf-8")))
            else:
                faltan += 1
        tabla = pd.DataFrame(filas)
        if not tabla.empty:
            tabla.to_csv(base_res / f"filas_{pista}.csv", index=False)
        tabla.attrs["faltan"] = faltan
        tablas.append(tabla)
    return pd.concat(tablas, ignore_index=True) if tablas else pd.DataFrame()
