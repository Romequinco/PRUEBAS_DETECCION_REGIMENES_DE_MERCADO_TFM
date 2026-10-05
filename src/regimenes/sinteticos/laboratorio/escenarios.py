"""Rejilla de escenarios, secuencias de regimen impuestas y verdad-terreno."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import yaml

from regimenes import rutas
from regimenes.sinteticos.datos import COL_RET

COL_REGIMEN = "regime"
# Mismo train inicial que el benchmark (benchmark.ejecucion.DEFAULT_TRAIN_DAYS). Se
# duplica aqui para no importar el benchmark al importar el laboratorio; un test
# comprueba que coinciden.
CALENTAMIENTO = {"A": 252 * 8, "B": 252 * 3}


def config_laboratorio(config: dict | None = None) -> dict[str, Any]:
    """Seccion ``laboratorio`` de ``configs/sinteticos.yaml`` (o de ``config``)."""
    if config is None:
        config = yaml.safe_load(rutas.SINTETICOS_CONFIG.read_text(encoding="utf-8"))
    return dict(config["laboratorio"])


@dataclass(frozen=True)
class Celda:
    """Una celda de la rejilla factorial de una pista."""

    pista: str
    generador: str
    duracion: int
    intensidad: float

    @property
    def clave(self) -> str:
        return f"{self.generador}__d{self.duracion}__i{self.intensidad:.2f}"

    def to_row(self) -> dict[str, Any]:
        return {"pista": self.pista, "generador": self.generador,
                "duracion": self.duracion, "intensidad": self.intensidad, "celda": self.clave}


def celdas(pista: str, cfg: dict | None = None) -> list[Celda]:
    """Celdas generador x duracion x intensidad de la pista, en orden estable."""
    cfg = config_laboratorio() if cfg is None else cfg
    return [Celda(pista.upper(), g, int(d), float(i))
            for g in cfg["generadores"] for d in cfg["duraciones"] for i in cfg["intensidades"]]


def detectores_pista(pista: str, cfg: dict | None = None) -> list[str]:
    """IDs de detector que entran en el laboratorio de la pista (registro menos exclusiones)."""
    from regimenes.detectores.registry import detector_specs

    cfg = config_laboratorio() if cfg is None else cfg
    fuera = set((cfg.get("excluir") or {}).get(pista.upper(), []))
    return [s.detector_id for s in detector_specs(pista) if s.detector_id not in fuera]


def largo_total(pista: str, cfg: dict | None = None) -> int:
    cfg = config_laboratorio() if cfg is None else cfg
    return CALENTAMIENTO[pista.upper()] + int(cfg["largo_puntuado"])


def _colocar(reg: np.ndarray, lo: int, hi: int, n: int, dur: int, margen: int,
             rng: np.random.Generator) -> None:
    """Coloca ``n`` crisis de ``dur`` sesiones en ``[lo, hi)``, una por hueco de igual tamano."""
    if n <= 0:
        return
    hueco = (hi - lo) // n
    holgura = hueco - dur - 2 * margen
    if holgura < 0:
        raise ValueError(f"No caben {n} crisis de {dur} sesiones con margen {margen} en {hi - lo} sesiones.")
    for k in range(n):
        inicio = lo + k * hueco + margen + int(rng.integers(0, holgura + 1))
        reg[inicio:inicio + dur] = 1


def secuencias(pista: str, duracion: int, n_paths: int, rng: np.random.Generator,
               cfg: dict | None = None) -> np.ndarray:
    """Matriz ``(n_paths, largo_total)`` de regimen impuesto (0 calma, 1 crisis).

    Calentamiento con ``crisis_calentamiento[pista]`` crisis y tramo puntuado con
    ``crisis_puntuadas`` crisis, todas de ``duracion`` sesiones; cada una con inicio
    uniforme dentro de su hueco y al menos ``margen_min`` sesiones de calma a cada
    lado (las crisis nunca cruzan el corte calentamiento/puntuado).
    """
    cfg = config_laboratorio() if cfg is None else cfg
    pista = pista.upper()
    calent = CALENTAMIENTO[pista]
    total = largo_total(pista, cfg)
    margen = int(cfg["margen_min"])
    reg = np.zeros((int(n_paths), total), dtype=int)
    for k in range(int(n_paths)):
        _colocar(reg[k], 0, calent, int(cfg["crisis_calentamiento"][pista]), duracion, margen, rng)
        _colocar(reg[k], calent, total, int(cfg["crisis_puntuadas"]), duracion, margen, rng)
    return reg


def atenuar(trayectoria: pd.DataFrame, generador, intensidad: float) -> pd.DataFrame:
    """Acerca la ley de las filas de crisis a la de calma con factor ``intensidad`` (lambda).

    En el espacio de trabajo del generador (columnas modeladas, estandarizadas) y por
    columna, con medias ``m`` y desviaciones ``s`` de calma (0) y crisis (1) de la propia
    trayectoria, cada fila de crisis pasa a
    ``m0 + lambda (m1 - m0) + (z - m1) * (s0 + lambda (s1 - s0)) / s1``:
    la media y la dispersion de crisis se interpolan linealmente entre las de calma
    (lambda = 0) y las nativas (lambda = 1), conservando la forma y la dinamica de las
    innovaciones. Despues se vuelve al espacio publico re-derivando las features del
    S&P 500. ``intensidad = 1`` devuelve la trayectoria tal cual.
    """
    if float(intensidad) == 1.0:
        return trayectoria
    espacio = generador.espacio_
    reg = trayectoria[COL_REGIMEN].to_numpy()
    Z = espacio.a_trabajo(trayectoria)
    crisis = reg == 1
    if crisis.sum() > 1 and (~crisis).sum() > 1:
        lam = float(intensidad)
        m0, m1 = Z[~crisis].mean(axis=0), Z[crisis].mean(axis=0)
        s0, s1 = Z[~crisis].std(axis=0), Z[crisis].std(axis=0)
        escala = np.divide(s0 + lam * (s1 - s0), s1, out=np.ones_like(s1), where=s1 > 0)
        Z[crisis] = m0 + lam * (m1 - m0) + (Z[crisis] - m1) * escala
    salida = espacio.a_publico(Z, pd.DatetimeIndex(trayectoria.index))
    salida[COL_REGIMEN] = reg
    return salida


def ventanas_verdad(regimen: pd.Series) -> dict[str, tuple[str, str]]:
    """Rachas de regimen 1 como ventanas ``{nombre: (inicio, fin)}`` (formato del benchmark)."""
    valores = np.asarray(regimen, dtype=int)
    indice = pd.DatetimeIndex(regimen.index)
    bordes = np.diff(np.concatenate(([0], valores, [0])))
    inicios, fines = np.flatnonzero(bordes == 1), np.flatnonzero(bordes == -1) - 1
    return {f"crisis_{k}": (str(indice[a].date()), str(indice[b].date()))
            for k, (a, b) in enumerate(zip(inicios, fines))}


__all__ = ["COL_REGIMEN", "COL_RET", "CALENTAMIENTO", "Celda", "config_laboratorio", "celdas",
           "detectores_pista", "largo_total", "secuencias", "atenuar", "ventanas_verdad"]
