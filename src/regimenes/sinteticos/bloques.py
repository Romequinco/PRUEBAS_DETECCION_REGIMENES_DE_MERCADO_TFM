"""Bloques para los generadores neuronales de "siguiente bloque".

Por que existe: una red no puede generar de una vez 2520 sesiones con solo un
tramo de entrenamiento de ~11000; los generadores neuronales aprenden la
distribucion del SIGUIENTE bloque de ``largo_bloque`` sesiones condicionada a
las ``largo_contexto`` anteriores y al regimen del bloque, y una trayectoria
larga se obtiene encadenando bloques de forma autoregresiva. Las dos mitades
(trocear train, encadenar al muestrear) son identicas para flow matching,
difusion, CVAE y CGAN, asi que viven aqui y cada generador solo aporta la red.

- ``construir_bloques``: pares (contexto, bloque, regimen del bloque) con
  ventana deslizante, solo dentro de tramos contiguos de train (una ventana
  nunca salta un hueco de fechas).
- ``encadenar``: llama a ``generar_bloque(contexto, reg_bloque, rng)`` por lotes
  sobre todas las trayectorias a la vez y recorta a ``length``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def tramos_contiguos(fechas: pd.DatetimeIndex | None, n: int, max_hueco_dias: int = 10) -> list[tuple[int, int]]:
    """Tramos ``[inicio, fin)`` sin huecos de mas de ``max_hueco_dias`` naturales.

    Sin ``fechas`` todo el array es un unico tramo. El umbral por defecto (10)
    tolera festivos y el cierre de una semana de septiembre de 2001.
    """
    if fechas is None or n < 2:
        return [(0, n)]
    if len(fechas) != n:
        raise ValueError("`fechas` debe tener la misma longitud que X.")
    dias = np.diff(pd.DatetimeIndex(fechas).asi8) / 86_400e9
    cortes = np.flatnonzero(dias > max_hueco_dias) + 1
    limites = np.r_[0, cortes, n]
    return [(int(a), int(b)) for a, b in zip(limites[:-1], limites[1:])]


def construir_bloques(
    X: np.ndarray,
    reg: np.ndarray,
    largo_bloque: int,
    largo_contexto: int,
    paso: int = 1,
    fechas: pd.DatetimeIndex | None = None,
    max_hueco_dias: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ventanas deslizantes (contexto, bloque siguiente, regimen del bloque).

    Parameters
    ----------
    X : (n, d) matriz de trabajo estandarizada de train.
    reg : (n,) regimen de train.
    largo_bloque, largo_contexto : longitudes ``L_blk`` (>= 1) y ``L_ctx`` (>= 0).
    paso : desplazamiento entre ventanas consecutivas.
    fechas : indice de train; si se pasa, las ventanas no cruzan huecos.

    Returns
    -------
    contextos : (m, L_ctx, d)
    bloques : (m, L_blk, d)
    reg_bloque : (m, L_blk) int
    """
    X = np.asarray(X, dtype=float)
    reg = np.asarray(reg, dtype=int)
    if X.ndim != 2 or reg.shape != (len(X),):
        raise ValueError("X debe ser (n, d) y reg (n,).")
    lb, lc, paso = int(largo_bloque), int(largo_contexto), int(paso)
    if lb < 1 or lc < 0 or paso < 1:
        raise ValueError("largo_bloque >= 1, largo_contexto >= 0 y paso >= 1.")
    total = lb + lc
    inicios: list[np.ndarray] = []
    for a, b in tramos_contiguos(fechas, len(X), max_hueco_dias):
        if b - a >= total:
            inicios.append(np.arange(a, b - total + 1, paso))
    if not inicios:
        raise ValueError(f"Ningun tramo contiguo de train tiene {total} filas (contexto + bloque).")
    ini = np.concatenate(inicios)
    idx_ctx = ini[:, None] + np.arange(lc)[None, :]
    idx_blk = ini[:, None] + lc + np.arange(lb)[None, :]
    return X[idx_ctx], X[idx_blk], reg[idx_blk]


def encadenar(
    generar_bloque: Callable[[np.ndarray, np.ndarray, np.random.Generator], np.ndarray],
    reg: np.ndarray,
    contexto: np.ndarray,
    largo_bloque: int,
    largo_contexto: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Genera ``(n_paths, length, d)`` encadenando bloques autoregresivamente.

    Parameters
    ----------
    generar_bloque:
        ``f(contexto_lote (n_paths, L_ctx, d), reg_bloque_lote (n_paths, L_blk), rng)``
        -> ``(n_paths, L_blk, d)``. Se llama ``ceil(length / L_blk)`` veces.
    reg : (n_paths, length) regimen de cada trayectoria.
    contexto : (>= L_ctx, d) ultimas filas reales de train; arranca todas las
        trayectorias como continuacion de la historia.

    El ultimo bloque se genera entero (su regimen se completa repitiendo el
    ultimo valor) y se recorta a ``length``.
    """
    reg = np.asarray(reg, dtype=int)
    contexto = np.asarray(contexto, dtype=float)
    lb, lc = int(largo_bloque), int(largo_contexto)
    if reg.ndim != 2:
        raise ValueError("reg debe ser (n_paths, length).")
    if contexto.ndim != 2 or len(contexto) < lc:
        raise ValueError(f"contexto debe ser (>= {lc}, d); llego {contexto.shape}.")
    n_paths, length = reg.shape
    d = contexto.shape[1]
    n_bloques = -(-length // lb)
    relleno = n_bloques * lb - length
    if relleno:
        reg = np.concatenate([reg, np.repeat(reg[:, -1:], relleno, axis=1)], axis=1)
    ctx = np.tile(contexto[len(contexto) - lc:][None, :, :], (n_paths, 1, 1))
    salida = np.empty((n_paths, n_bloques * lb, d), dtype=float)
    for b in range(n_bloques):
        tramo = slice(b * lb, (b + 1) * lb)
        bloque = np.asarray(generar_bloque(ctx, reg[:, tramo], rng), dtype=float)
        if bloque.shape != (n_paths, lb, d):
            raise ValueError(f"generar_bloque devolvio {bloque.shape}; se esperaba {(n_paths, lb, d)}.")
        salida[:, tramo] = bloque
        if lc:
            ctx = np.concatenate([ctx, bloque], axis=1)[:, -lc:]
    return salida[:, :length]
