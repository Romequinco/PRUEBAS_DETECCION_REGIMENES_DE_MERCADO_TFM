"""Metricas de una trayectoria frente a su verdad-terreno exacta.

Reutiliza las de ADR-003 (``evaluacion.metricas.detection_summary`` y
``evaluacion.ranking.detection_score``) y anade las que solo tienen sentido con
verdad exacta: retraso de deteccion, exactitud equilibrada por dia y episodios de
falsa alarma por ano.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regimenes.evaluacion.metricas import detection_summary
from regimenes.evaluacion.ranking import detection_score

SESIONES_ANO = 252


def _retrasos(flags: np.ndarray, verdad: np.ndarray, min_run: int) -> list[float]:
    """Sesiones desde el inicio de cada crisis hasta el inicio de la primera racha
    de ``min_run`` marcas dentro de ella (NaN si no se detecta)."""
    bordes = np.diff(np.concatenate(([0], verdad.astype(int), [0])))
    salida = []
    for a, b in zip(np.flatnonzero(bordes == 1), np.flatnonzero(bordes == -1)):
        run = min(int(min_run), b - a)
        seg = flags[a:b].astype(int)
        racha = np.convolve(seg, np.ones(run, dtype=int), mode="valid") if len(seg) >= run else np.array([])
        hit = np.flatnonzero(racha == run)
        salida.append(float(hit[0]) if len(hit) else np.nan)
    return salida


def metricas_trayectoria(flags: pd.Series, regimen: pd.Series, *, min_run: int = 3) -> dict[str, float]:
    """Metricas de la senal OOS de crisis ``flags`` (bool por fecha) frente a ``regimen``.

    ``regimen`` es la verdad-terreno (0/1) de las MISMAS fechas OOS. Devuelve las
    columnas ``det_*`` del benchmark, ``score_deteccion`` (ADR-003) y ``retraso_medio``,
    ``exactitud_equilibrada`` y ``falsas_alarmas_ano``.
    """
    flags = pd.Series(flags).astype(bool)
    regimen = pd.Series(regimen).reindex(flags.index).astype(int)
    from regimenes.sinteticos.laboratorio.escenarios import ventanas_verdad

    ventanas = ventanas_verdad(regimen)
    out = {k: v for k, v in detection_summary(flags, ventanas, min_run=min_run).items()
           if not k.startswith("det_ev_")}
    # np.float64 (no float de Python): con precision = recall = 0 la division de
    # detection_score da inf/nan bajo np.errstate y se convierte en 0, en vez de ZeroDivisionError
    out["score_deteccion"] = float(detection_score(np.float64(out["det_precision"]),
                                                   np.float64(out["det_event_recall"])))
    f, v = flags.to_numpy(), regimen.to_numpy().astype(bool)
    tpr = (f & v).sum() / v.sum() if v.any() else np.nan
    tnr = (~f & ~v).sum() / (~v).sum() if (~v).any() else np.nan
    out["exactitud_equilibrada"] = float(np.nanmean([tpr, tnr]))
    retrasos = _retrasos(f, v, min_run)
    out["retraso_medio"] = float(np.nanmean(retrasos)) if np.isfinite(retrasos).any() else np.nan
    # episodios de marca que empiezan fuera de toda crisis verdadera, por ano de calma
    inicios = np.flatnonzero(np.diff(np.concatenate(([0], f.astype(int)))) == 1)
    falsos = int(sum(not v[i] for i in inicios))
    anos_calma = (~v).sum() / SESIONES_ANO
    out["falsas_alarmas_ano"] = falsos / anos_calma if anos_calma > 0 else np.nan
    return out


__all__ = ["metricas_trayectoria"]
