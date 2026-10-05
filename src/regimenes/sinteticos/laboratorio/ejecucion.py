"""Generacion cacheada de trayectorias y ejecucion paralela del laboratorio.

Unidad de trabajo: (pista, celda, trayectoria, detector). Cada una:

1. lee su trayectoria (generada una vez por (celda, trayectoria) con semilla propia,
   asi que anadir trayectorias no cambia las ya generadas);
2. ejecuta ``evaluacion.walk_forward`` con la spec del benchmark (mismas features,
   ``step``, ventana expanding/rolling y train inicial = calentamiento);
3. puntua la senal OOS ``state == crisis_state`` contra el regimen verdadero
   (``metricas.metricas_trayectoria``) y escribe una fila JSON con su huella.

Reanudable: una fila cuya huella coincide no se recalcula. El panel OOS de la
trayectoria 0 de cada celda se guarda (parquet, no versionado) para las figuras.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from regimenes import rutas
from regimenes.sinteticos.laboratorio import escenarios as esc
from regimenes.sinteticos.laboratorio.metricas import metricas_trayectoria

DIR_DATOS = rutas.DATA_SINTETICOS / "laboratorio"
DIR_RESULTADOS = rutas.RESULTS_SINTETICOS / "laboratorio"
_HILOS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
_CLAVES_HUELLA = ("semilla", "largo_puntuado", "crisis_calentamiento", "crisis_puntuadas",
                  "margen_min", "min_run")


# --------------------------------------------------------------------------- rutas
def ruta_trayectoria(celda: esc.Celda, path_id: int, base: Path | None = None) -> Path:
    return Path(base or DIR_DATOS) / celda.pista / celda.clave / f"p{path_id:03d}.parquet"


def ruta_fila(celda: esc.Celda, path_id: int, detector_id: str, base: Path | None = None) -> Path:
    return Path(base or DIR_RESULTADOS) / "filas" / celda.pista / celda.clave / f"p{path_id:03d}_{detector_id}.json"


def ruta_panel(celda: esc.Celda, path_id: int, detector_id: str, base: Path | None = None) -> Path:
    return Path(base or DIR_RESULTADOS) / "paneles" / celda.pista / celda.clave / f"p{path_id:03d}_{detector_id}.parquet"


# --------------------------------------------------------------------------- semillas y huellas
def semilla_trayectoria(cfg: dict, celda: esc.Celda, path_id: int) -> int:
    """Semilla estable por (semilla global, pista, generador, duracion, trayectoria).

    No depende de la intensidad: las celdas que solo difieren en ella comparten la
    trayectoria bruta (diseno pareado; solo cambia la atenuacion).
    """
    texto = f"{cfg['semilla']}|{celda.pista}|{celda.generador}|{celda.duracion}|{path_id}"
    return int(hashlib.sha256(texto.encode()).hexdigest()[:8], 16)


def _hash_codigo() -> str:
    h = hashlib.sha256()
    for nombre in ("escenarios.py", "metricas.py"):
        h.update((Path(__file__).parent / nombre).read_bytes())
    return h.hexdigest()[:16]


def huella(cfg: dict, celda: esc.Celda, path_id: int, spec) -> str:
    """Huella de una fila: config pre-registrada, celda, trayectoria, spec y codigo.

    El codigo hasheado es ``escenarios.py`` + ``metricas.py`` (lo que define la trayectoria y la
    puntuacion). NO entran ``ejecucion.py``, el ``.pkl`` del generador ni el codigo de los detectores
    (sus parametros si, via la spec): si cambian, hay que forzar la re-ejecucion.
    """
    contenido = {
        "cfg": {k: cfg[k] for k in _CLAVES_HUELLA},
        "celda": celda.to_row(),
        "path_id": int(path_id),
        "spec": {**spec.to_row(), "params": spec.params, "features": list(spec.features)},
        "codigo": _hash_codigo(),
    }
    return hashlib.sha256(json.dumps(contenido, sort_keys=True, default=str).encode()).hexdigest()


# --------------------------------------------------------------------------- trayectorias
def cargar_generador(nombre: str, pista: str):
    from regimenes.sinteticos import persistencia
    from regimenes.sinteticos.comun import GeneradorBase

    return GeneradorBase.cargar(persistencia.dir_trayectorias(nombre, pista))


def generar_trayectoria(celda: esc.Celda, path_id: int, cfg: dict | None = None,
                        generador=None, base: Path | None = None) -> pd.DataFrame:
    """Genera (o lee de cache) la trayectoria ``path_id`` de la celda."""
    cfg = esc.config_laboratorio() if cfg is None else cfg
    destino = ruta_trayectoria(celda, path_id, base)
    if destino.exists():
        tray = pd.read_parquet(destino)
        tray.index = pd.DatetimeIndex(tray.index).astype("datetime64[ns]")
        return tray
    generador = cargar_generador(celda.generador, celda.pista) if generador is None else generador
    rng = np.random.default_rng(semilla_trayectoria(cfg, celda, path_id))
    reg = esc.secuencias(celda.pista, celda.duracion, 1, rng, cfg)
    tray = generador.sample(1, reg.shape[1], reg, random_state=int(rng.integers(2**31 - 1)))[0]
    tray = esc.atenuar(tray, generador, celda.intensidad)
    destino.parent.mkdir(parents=True, exist_ok=True)
    tray.to_parquet(destino)
    return tray


def preparar_trayectorias(pistas: Iterable[str], n_paths: int, cfg: dict | None = None) -> int:
    """Genera las trayectorias que falten (``n_paths`` por celda). Devuelve cuantas genero."""
    cfg = esc.config_laboratorio() if cfg is None else cfg
    nuevas = 0
    for pista in pistas:
        cache_gen: dict[str, Any] = {}
        for celda in esc.celdas(pista, cfg):
            for k in range(int(n_paths)):
                if ruta_trayectoria(celda, k).exists():
                    continue
                if celda.generador not in cache_gen:
                    cache_gen[celda.generador] = cargar_generador(celda.generador, pista)
                generar_trayectoria(celda, k, cfg, cache_gen[celda.generador])
                nuevas += 1
    return nuevas


# --------------------------------------------------------------------------- un trabajo
def _spec(pista: str, detector_id: str):
    from regimenes.detectores.registry import detector_specs

    return next(s for s in detector_specs(pista) if s.detector_id == detector_id)


def ejecutar_trabajo(celda: esc.Celda, path_id: int, detector_id: str, cfg: dict | None = None,
                     forzar: bool = False) -> dict[str, Any]:
    """Ejecuta un detector sobre una trayectoria y escribe su fila. Devuelve la fila."""
    from regimenes import evaluacion as ev

    cfg = esc.config_laboratorio() if cfg is None else cfg
    spec = _spec(celda.pista, detector_id)
    firma = huella(cfg, celda, path_id, spec)
    destino = ruta_fila(celda, path_id, detector_id)
    if not forzar and destino.exists():
        fila = json.loads(destino.read_text(encoding="utf-8"))
        if fila.get("huella") == firma:
            return {**fila, "estado": "cache"}

    tray = generar_trayectoria(celda, path_id, cfg)
    calent = esc.CALENTAMIENTO[celda.pista]
    X = tray[list(spec.features)]
    inicio = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        panel = ev.walk_forward(spec.factory, X, market_returns=tray[esc.COL_RET],
                                train_size=calent, min_train=calent,
                                step=spec.step, expanding=spec.expanding)
    segundos = time.perf_counter() - inicio
    flags = panel["state"].eq(spec.factory().crisis_state)
    fila = {
        **celda.to_row(), "path_id": int(path_id), "id": detector_id, "familia": spec.family,
        "n_oos": int(len(panel)), "segundos": segundos, "huella": firma,
        **metricas_trayectoria(flags, tray[esc.COL_REGIMEN], min_run=int(cfg["min_run"])),
    }
    fila = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in fila.items()}
    if path_id == 0:
        p = ruta_panel(celda, path_id, detector_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        salida = panel[["state", "p_crisis"]].copy()
        salida.attrs = {}
        salida["regime"] = tray[esc.COL_REGIMEN].reindex(salida.index).to_numpy()
        salida.to_parquet(p)
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(".tmp")
    tmp.write_text(json.dumps(fila, default=float), encoding="utf-8")
    os.replace(tmp, destino)
    return {**fila, "estado": "ok"}


def _trabajo_seguro(args) -> dict[str, Any]:
    celda, path_id, detector_id, cfg = args
    try:
        return ejecutar_trabajo(celda, path_id, detector_id, cfg)
    except Exception as exc:  # noqa: BLE001 — se registra y el resto sigue
        return {**celda.to_row(), "path_id": path_id, "id": detector_id,
                "estado": "error", "detalle": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------- en lote
def trabajos(pistas: Iterable[str], n_paths: int, cfg: dict | None = None,
             detectores: Iterable[str] | None = None, celdas_por_pista: int | None = None):
    """Lista de (celda, path_id, detector_id, cfg); detectores lentos primero."""
    from regimenes.detectores.registry import detector_specs

    cfg = esc.config_laboratorio() if cfg is None else cfg
    salida = []
    for pista in pistas:
        ids = esc.detectores_pista(pista, cfg)
        if detectores is not None:
            ids = [d for d in ids if d in set(detectores)]
        lentos = {s.detector_id for s in detector_specs(pista) if s.slow}
        ids = sorted(ids, key=lambda d: (d not in lentos, d))
        cs = esc.celdas(pista, cfg)[: celdas_por_pista or None]
        for d in ids:
            for celda in cs:
                for k in range(int(n_paths)):
                    salida.append((celda, k, d, cfg))
    return salida


def ejecutar_lote(lista, n_jobs: int = 1, progreso: bool = True) -> pd.DataFrame:
    """Ejecuta los trabajos (en paralelo si ``n_jobs > 1``) y devuelve sus estados."""
    previos = {n: os.environ.get(n) for n in _HILOS}
    for n in _HILOS:
        os.environ[n] = "1"
    estados: list[dict] = []
    t0 = time.perf_counter()
    try:
        if n_jobs <= 1:
            for i, args in enumerate(lista, 1):
                estados.append(_trabajo_seguro(args))
                if progreso and i % 25 == 0:
                    print(f"[lab] {i}/{len(lista)} ({time.perf_counter() - t0:.0f}s)", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                pendientes = {pool.submit(_trabajo_seguro, a) for a in lista}
                while pendientes:
                    hechos, pendientes = wait(pendientes, return_when=FIRST_COMPLETED)
                    for f in hechos:
                        estados.append(f.result())
                    if progreso and len(estados) % 25 < len(hechos):
                        print(f"[lab] {len(estados)}/{len(lista)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    finally:
        for n, v in previos.items():
            if v is None:
                os.environ.pop(n, None)
            else:
                os.environ[n] = v
    return pd.DataFrame(estados)


def consolidar(base: Path | None = None) -> pd.DataFrame:
    """Todas las filas JSON del laboratorio en un DataFrame (y CSV ``filas.csv``).

    Las filas JSON no se versionan (miles de ficheros); ``filas.csv`` si. Sin JSON
    (p. ej. en un clon nuevo) se lee ``filas.csv``.
    """
    raiz = Path(base or DIR_RESULTADOS)
    filas = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((raiz / "filas").rglob("*.json"))]
    if not filas and (raiz / "filas.csv").exists():
        return pd.read_csv(raiz / "filas.csv")
    tabla = pd.DataFrame(filas)
    if len(tabla):
        tabla = tabla.drop(columns=["huella"], errors="ignore")
        tabla.to_csv(raiz / "filas.csv", index=False)
    return tabla


__all__ = ["DIR_DATOS", "DIR_RESULTADOS", "ruta_trayectoria", "ruta_fila", "ruta_panel",
           "semilla_trayectoria", "huella", "cargar_generador", "generar_trayectoria",
           "preparar_trayectorias", "ejecutar_trabajo", "trabajos", "ejecutar_lote", "consolidar"]
