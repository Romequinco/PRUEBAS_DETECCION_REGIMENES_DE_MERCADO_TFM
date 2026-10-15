"""Reservorios de trayectorias por ancla y bloque sintetico de cada pliegue.

Pre-registro (``aumento.brazos``, ``trayectoria_largo``, ``proporciones``):

- **completas**: trayectorias independientes de ``trayectoria_largo`` sesiones con la
  cadena simulada (arranque estacionario, duraciones empiricas), cada una
  re-derivada desde la historia real del ancla; base de la proporcion = dias del
  train real del pliegue.
- **crisis**: episodios independientes con ``regime = 1`` y duracion remuestreada
  de las crisis reales del ajuste del ancla; base = dias de crisis reconocidos en
  el ancla que caen dentro de la ventana de train del pliegue.

Las trayectorias o episodios ``i = 0, 1, 2, ...`` de un reservorio
``(pista, fuente, tau, brazo, repeticion)`` tienen semilla propia, asi que un
reservorio se puede alargar sin cambiar su principio, y el bloque de proporcion
``p`` es el PRINCIPIO del reservorio (proporciones anidadas).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from regimenes.sinteticos.aumento import anclas as an
from regimenes.sinteticos.aumento.config import DIR_DATOS, config_aumento, fuentes, semilla
from regimenes.sinteticos.datos import COL_REGIMEN, ventanas_crisis

COL_TRAY = "tray"
BRAZOS = ("completas", "crisis")


# --------------------------------------------------------------------------- pliegues y bases
def pliegues(n: int, train_size: int, step: int, expanding: bool) -> pd.DataFrame:
    """Pliegues de ``walk_forward`` (misma aritmetica): ``t`` (inicio del bloque) y ``lo``."""
    filas, t = [], int(train_size)
    while t < n:
        filas.append({"t": t, "lo": 0 if expanding else max(0, t - int(train_size))})
        t += int(step)
    return pd.DataFrame(filas)


def mascara_reconocidas(indice: pd.DatetimeIndex, pista: str, tau) -> np.ndarray:
    """Dias de ``indice`` dentro de crisis reconocidas en ``tau`` (suelo <= tau)."""
    tau = pd.Timestamp(tau)
    indice = pd.DatetimeIndex(indice)
    marca = np.zeros(len(indice), dtype=bool)
    for pico, suelo in ventanas_crisis(pista).values():
        if suelo <= tau:
            marca |= (indice >= pico) & (indice <= suelo)
    return marca


def base_proporcion(brazo: str, indice_train: pd.DatetimeIndex, pista: str, tau) -> int:
    """Dias sobre los que se aplica la proporcion ``p`` en el pliegue."""
    if brazo == "completas":
        return len(indice_train)
    if brazo == "crisis":
        return int(mascara_reconocidas(indice_train, pista, tau).sum())
    raise ValueError(f"brazo desconocido {brazo!r}; validos: {BRAZOS}.")


def n_sintetico(p: float, base: int) -> int:
    return int(round(float(p) * int(base)))


# --------------------------------------------------------------------------- reservorios
def ruta_reservorio(pista: str, fuente: str, tau, brazo: str, rep: int, base: Path | None = None) -> Path:
    base = DIR_DATOS if base is None else Path(base)
    return (base / "reservorios" / pista.upper() / fuente / pd.Timestamp(tau).strftime("%Y-%m-%d")
            / f"{brazo}_r{int(rep)}.parquet")


def _pieza(gen, pista: str, fuente: str, tau, brazo: str, rep: int, i: int, cfg: dict) -> pd.DataFrame:
    """Trayectoria (completas) o episodio (crisis) ``i`` del reservorio."""
    clave = (pista, fuente, pd.Timestamp(tau).date(), brazo, rep, i)
    s = semilla("pieza", *clave, base=int(cfg["semilla"]))
    if brazo == "completas":
        opciones = {k: cfg["brazos"]["completas"][k] for k in ("inicial", "duraciones")}
        tray = gen.sample(1, int(cfg["trayectoria_largo"]), opciones, random_state=s)[0]
    else:
        banco = gen.rachas_.loc[gen.rachas_["regimen"] == 1, "duracion"].to_numpy(dtype=int)
        if banco.size == 0:
            raise ValueError(f"{fuente} en {tau}: el ajuste no tiene episodios de crisis.")
        # semilla propia para la duracion: no comparte flujo con el muestreo del episodio
        d = int(np.random.default_rng(semilla("duracion", *clave, base=int(cfg["semilla"]))).choice(banco))
        tray = gen.sample(1, d, np.ones(d, dtype=int), random_state=s)[0]
    tray = tray.reset_index(drop=True)
    tray[COL_TRAY] = i
    return tray


def asegurar_reservorio(pista: str, fuente: str, tau, brazo: str, rep: int, necesarias: int, *,
                        cfg: dict | None = None, gen=None, base: Path | None = None) -> int:
    """Alarga (o crea) el reservorio hasta ``necesarias`` filas; devuelve sus filas."""
    from regimenes.sinteticos.aumento.config import config_completa, huella_generador

    cfg_total = config_completa()
    cfg = config_aumento(cfg_total) if cfg is None else cfg
    ruta = ruta_reservorio(pista, fuente, tau, brazo, rep, base)
    marca = ruta.with_suffix(".huella")
    h = huella_generador(pista, fuente, tau, cfg_total)
    vigente = ruta.exists() and marca.exists() and marca.read_text(encoding="utf-8").strip() == h
    previo = pd.read_parquet(ruta) if vigente else None     # si la huella cambio, se regenera de cero
    filas = 0 if previo is None else len(previo)
    if filas >= necesarias:
        return filas
    gen = an.cargar_generador(pista, fuente, tau, base) if gen is None else gen
    piezas = [] if previo is None else [previo]
    i = 0 if previo is None else int(previo[COL_TRAY].max()) + 1
    while filas < necesarias:
        pieza = _pieza(gen, pista, fuente, tau, brazo, rep, i, cfg)
        piezas.append(pieza)
        filas += len(pieza)
        i += 1
    tabla = pd.concat(piezas, ignore_index=True)
    columnas = [c for c in tabla.columns if c not in (COL_REGIMEN, COL_TRAY)]
    tabla[columnas] = tabla[columnas].astype("float64")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_suffix(".tmp.parquet")
    tabla.to_parquet(tmp)
    tmp.replace(ruta)
    marca.write_text(h, encoding="utf-8")
    return filas


def necesidades(pista: str, registro: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Filas que necesita cada reservorio ``(fuente, tau, brazo)``: maximo de la base con p = max.

    Recorre los pliegues de todos los detectores de la pista con la aritmetica de
    ``walk_forward`` y el ancla que tocaria a cada uno (con la regla de respaldo).
    """
    from regimenes.benchmark.ejecucion import DEFAULT_TRAIN_DAYS, common_track_index
    from regimenes.detectores.registry import detector_specs

    from regimenes.sinteticos.aumento.config import detectores_pista

    cfg = config_aumento() if cfg is None else cfg
    pista = pista.upper()
    indice = common_track_index(pista)
    margen = int(cfg["anclas"]["margen_reconocimiento"])
    p_max = max(float(p) for p in cfg["proporciones"])
    dets = set(detectores_pista(pista, cfg))
    train_size = DEFAULT_TRAIN_DAYS[pista]
    esquemas = {(s.step, s.expanding) for s in detector_specs(pista) if s.detector_id in dets}
    filas = []
    for fuente in fuentes(cfg):
        validas = an.anclas_validas(registro, fuente)
        for step, expanding in sorted(esquemas):
            for _, pl in pliegues(len(indice), train_size, step, expanding).iterrows():
                idx_train = indice[int(pl["lo"]):int(pl["t"])]
                tau = an.ancla_para(validas, an.limite_reconocimiento(idx_train, margen))
                if tau is None:
                    continue
                for brazo in BRAZOS:
                    filas.append({"fuente": fuente, "tau": tau, "brazo": brazo,
                                  "necesarias": n_sintetico(p_max, base_proporcion(brazo, idx_train, pista, tau))})
    tabla = pd.DataFrame(filas, columns=["fuente", "tau", "brazo", "necesarias"])
    return tabla.groupby(["fuente", "tau", "brazo"], as_index=False)["necesarias"].max()


def _asegurar_seguro(args) -> dict[str, Any]:
    pista, fuente, tau, brazo, rep, necesarias, base = args
    import time

    from regimenes.sinteticos.aumento.config import un_hilo

    t0 = time.perf_counter()
    fila = {"pista": pista, "fuente": fuente, "tau": tau, "brazo": brazo, "rep": rep}
    try:
        with un_hilo():
            filas = asegurar_reservorio(pista, fuente, tau, brazo, rep, int(necesarias), base=base)
        return {**fila, "filas": filas, "estado": "ok", "segundos": time.perf_counter() - t0}
    except Exception as exc:  # noqa: BLE001
        return {**fila, "filas": 0, "estado": f"error: {type(exc).__name__}: {exc}",
                "segundos": time.perf_counter() - t0}


def preparar_reservorios(pistas: Iterable[str], *, n_jobs: int = 1, base: Path | None = None,
                         base_resultados: Path | None = None, progreso: bool = True) -> pd.DataFrame:
    """Genera o alarga los reservorios de todas las repeticiones (reanudable)."""
    from regimenes.sinteticos.aumento import _pool

    cfg = config_aumento()
    estados: list[dict] = []
    for pista in pistas:
        pista = pista.upper()
        nec = necesidades(pista, an.leer_registro(pista, base_resultados), cfg)
        lista = [(pista, r.fuente, r.tau, r.brazo, rep, r.necesarias, base)
                 for r in nec.itertuples() for rep in range(int(cfg["repeticiones"][pista]))]

        def al_terminar(a, estado, pista=pista, total=len(lista)):
            estados.append({"pista": a[0], "fuente": a[1], "tau": a[2], "brazo": a[3], "rep": a[4], **estado})
            if progreso and (len(estados) % 50 == 0 or not str(estado.get("estado", "")).startswith("ok")):
                print(f"[reservorios {pista}] {len(estados)}/{total} {estado.get('estado')}", flush=True)

        _pool.mapa(_asegurar_seguro, lista, n_jobs, al_terminar)
    return pd.DataFrame(estados)


# --------------------------------------------------------------------------- proveedor de bloques
@dataclass
class Proveedor:
    """Da a cada pliegue su bloque sintetico (lo llama ``DetectorAumentado.fit``).

    ``registro_pliegues`` acumula, por pliegue, el ancla usada, la base, las filas
    sinteticas y los dias de crisis realizados del bloque.
    """

    pista: str
    fuente: str
    brazo: str
    proporcion: float
    rep: int
    validas: pd.DatetimeIndex
    margen: int
    base: Path | None = None
    _reservorios: dict = field(default_factory=dict, repr=False)
    registro_pliegues: list = field(default_factory=list, repr=False)

    def reservorio(self, tau) -> pd.DataFrame:
        """Reservorio del ancla (solo se guarda el ultimo: las anclas solo avanzan con los pliegues)."""
        tau = pd.Timestamp(tau)
        if tau not in self._reservorios:
            ruta = ruta_reservorio(self.pista, self.fuente, tau, self.brazo, self.rep, self.base)
            self._reservorios = {tau: pd.read_parquet(ruta)}
        return self._reservorios[tau]

    def __call__(self, indice_train: pd.DatetimeIndex) -> pd.DataFrame | None:
        indice_train = pd.DatetimeIndex(indice_train)
        tau = an.ancla_para(self.validas, an.limite_reconocimiento(indice_train, self.margen))
        fila = {"fin_train": indice_train[-1], "n_train": len(indice_train),
                "tau": tau, "base": 0, "n_sint": 0, "dias_crisis_bloque": 0}
        bloque = None
        if tau is not None and self.proporcion > 0:
            fila["base"] = base_proporcion(self.brazo, indice_train, self.pista, tau)
            n = n_sintetico(self.proporcion, fila["base"])
            if n > 0:
                res = self.reservorio(tau)
                if len(res) < n:
                    raise RuntimeError(
                        f"Reservorio corto ({len(res)} < {n}) en {self.pista}/{self.fuente}/{tau.date()}/"
                        f"{self.brazo}/r{self.rep}: ejecuta la preparacion de reservorios."
                    )
                bloque = res.iloc[:n].drop(columns=[COL_TRAY])
                fila["n_sint"] = n
                fila["dias_crisis_bloque"] = int(bloque[COL_REGIMEN].sum())
        self.registro_pliegues.append(fila)
        return bloque


def proveedor_para(pista: str, fuente: str, brazo: str, proporcion: float, rep: int, *,
                   registro: pd.DataFrame | None = None, cfg: dict | None = None,
                   base: Path | None = None, base_resultados: Path | None = None) -> Proveedor:
    cfg = config_aumento() if cfg is None else cfg
    registro = an.leer_registro(pista, base_resultados) if registro is None else registro
    return Proveedor(pista.upper(), fuente, brazo, float(proporcion), int(rep),
                     an.anclas_validas(registro, fuente), int(cfg["anclas"]["margen_reconocimiento"]), base)
