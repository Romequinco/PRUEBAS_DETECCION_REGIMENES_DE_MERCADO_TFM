"""Discriminador real frente a sintetico (classifier two-sample test, Lopez-Paz y Oquab 2017).

Idea: si un clasificador entrenado para separar ventanas reales de sinteticas no
supera el azar en validacion cruzada, las dos muestras son indistinguibles para
esa familia de clasificadores. La metrica es el AUC ROC fuera de muestra
(0,5 = indistinguibles, 1 = trivialmente separables).

Procedimiento
-------------
1. **Ventanas**: bloques NO solapados de ``largo_ventana`` sesiones, desde el
   principio, en el real de train y en cada trayectoria (el resto final que no
   llena una ventana se descarta). Nunca se cruza por fecha.
2. **Resumen por ventana**: las columnas se estandarizan con media y sd del real
   (``_comun.estandarizador``) y cada ventana se resume, por columna, con media,
   sd, minimo, maximo, ultimo - primero y autocorrelacion de orden 1 (dentro de
   la ventana; 0 si la ventana es constante). Para ``SP500_ret`` (si esta) se
   anaden la curtosis (no en exceso) y la media de |r|. Por defecto se usan
   todas las columnas publicas, modeladas y re-derivadas.
3. **Regimen de la ventana** (regla fija): ``'calma'`` si TODOS sus dias son
   calma; ``'crisis'`` si MAS de la mitad son crisis; las ventanas mixtas con
   mayoria de calma solo entran en ``'todos'``. Asi la calma es pura y la crisis
   no pierde los bordes de los episodios, que son cortos.
4. **Equilibrio de clases**: por regimen, las ventanas sinteticas se submuestrean
   (sin reemplazo, con ``semilla``) hasta el numero de ventanas reales, repartiendo
   por turnos entre trayectorias (orden aleatorio de trayectorias y de ventanas
   dentro de cada una) para no concentrar la muestra en pocas trayectorias.
5. **Validacion cruzada agrupada** (``n_pliegues``): el real se parte en
   ``n_pliegues`` bloques cronologicos contiguos de igual numero de ventanas (las
   ventanas vecinas comparten dinamica y no pueden caer a ambos lados); cada
   trayectoria sintetica (``path_id``) va entera a un pliegue (asignacion por
   turnos tras barajar). El pliegue k de prueba = bloque real k + trayectorias
   del pliegue k.
6. **Clasificador**: ``HistGradientBoostingClassifier`` con parametros modestos
   y fijos (``PARAMS_CLASIFICADOR``), sin early stopping, ``random_state=semilla``.

Salida (contrato largo de ``fidelidad``)
----------------------------------------
Por ``regimen`` en ``('todos', 'calma', 'crisis')``, ``columna='—'``, ``tipo='—'``:

- ``metrica='auc'``: ``real`` = 0,5 (referencia de indistinguible), ``sintetico``
  = media del AUC entre pliegues, ``banda_inf`` / ``banda_sup`` = minimo y maximo
  del AUC entre pliegues (dispersion del estimador, NO banda de referencia del
  real), ``cociente`` = NaN, ``en_banda`` = NaN.
- ``metrica='n_ventanas'``: ``real`` = ventanas reales, ``sintetico`` = ventanas
  sinteticas usadas tras equilibrar.

Si un regimen no tiene al menos ``2 * n_pliegues`` ventanas reales el AUC es NaN.
En crisis el n efectivo es el numero de episodios (8 en la pista A, 4 en la B):
del AUC en crisis se publica el orden entre generadores, no el valor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from regimenes.sinteticos.datos import COL_RET
from regimenes.sinteticos.validacion._comun import estandarizador, separar_real, separar_sintetico

REGIMENES = ("todos", "calma", "crisis")

PARAMS_CLASIFICADOR = {
    "max_iter": 200,
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "early_stopping": False,
}


# --------------------------------------------------------------------------- ventanas


def _ventanas(Z: np.ndarray, reg: np.ndarray, largo: int) -> tuple[np.ndarray, np.ndarray]:
    """Ventanas no solapadas ``(n_v, largo, d)`` y su fraccion de dias en crisis."""
    n_v = len(Z) // largo
    W = Z[: n_v * largo].reshape(n_v, largo, Z.shape[1])
    frac = reg[: n_v * largo].reshape(n_v, largo).mean(axis=1)
    return W, frac


def resumen_ventanas(W: np.ndarray, columnas: list[str]) -> np.ndarray:
    """Estadisticos por ventana y columna (ver docstring del modulo): ``(n_v, 6*d [+2])``."""
    media = W.mean(axis=1)
    sd = W.std(axis=1, ddof=1)
    c = W - media[:, None, :]
    num = (c[:, 1:, :] * c[:, :-1, :]).sum(axis=1)
    den = (c**2).sum(axis=1)
    acf1 = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
    partes = [media, sd, W.min(axis=1), W.max(axis=1), W[:, -1, :] - W[:, 0, :], acf1]
    if COL_RET in columnas:
        r = W[:, :, columnas.index(COL_RET)]
        cr = r - r.mean(axis=1, keepdims=True)
        m2 = (cr**2).mean(axis=1)
        m4 = (cr**4).mean(axis=1)
        kurt = np.divide(m4, m2**2, out=np.full_like(m4, 3.0), where=m2 > 1e-12)
        partes += [kurt[:, None], np.abs(r).mean(axis=1)[:, None]]
    return np.hstack(partes)


def _mascara_regimen(frac: np.ndarray, regimen: str) -> np.ndarray:
    if regimen == "todos":
        return np.ones(frac.size, dtype=bool)
    if regimen == "calma":
        return frac == 0.0
    return frac > 0.5


# --------------------------------------------------------------------------- equilibrio y pliegues


def _equilibrar(path_id: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Indices de ``n`` ventanas sinteticas repartidas por turnos entre trayectorias."""
    if path_id.size <= n:
        return np.arange(path_id.size)
    caminos = np.unique(path_id)
    orden_camino = dict(zip(rng.permutation(caminos), range(caminos.size)))
    aleatorio = rng.random(path_id.size)
    # rango de cada ventana dentro de su trayectoria (orden aleatorio)
    rango = np.empty(path_id.size, dtype=int)
    for p in caminos:
        idx = np.flatnonzero(path_id == p)
        rango[idx[np.argsort(aleatorio[idx])]] = np.arange(idx.size)
    clave = np.lexsort((np.vectorize(orden_camino.get)(path_id), rango))
    return np.sort(clave[:n])


def _pliegues(n_real: int, path_id: np.ndarray, k: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Pliegue de cada ventana real (bloques cronologicos) y sintetica (por trayectoria)."""
    pl_real = (np.arange(n_real) * k) // max(n_real, 1)
    caminos = rng.permutation(np.unique(path_id))
    asignacion = {p: i % k for i, p in enumerate(caminos)}
    pl_sint = np.array([asignacion[p] for p in path_id], dtype=int)
    return pl_real, pl_sint


def _auc_cv(X_r: np.ndarray, X_s: np.ndarray, pl_r: np.ndarray, pl_s: np.ndarray, k: int,
            semilla: int) -> list[float]:
    X = np.vstack([X_r, X_s])
    y = np.r_[np.ones(len(X_r)), np.zeros(len(X_s))]
    pl = np.r_[pl_r, pl_s]
    aucs = []
    for f in range(k):
        prueba = pl == f
        if len(np.unique(y[prueba])) < 2 or len(np.unique(y[~prueba])) < 2:
            continue
        clf = HistGradientBoostingClassifier(random_state=semilla, **PARAMS_CLASIFICADOR)
        clf.fit(X[~prueba], y[~prueba])
        aucs.append(float(roc_auc_score(y[prueba], clf.predict_proba(X[prueba])[:, 1])))
    return aucs


# --------------------------------------------------------------------------- funcion publica


def discriminador(real: pd.DataFrame, sintetico: list[pd.DataFrame], *, regimen_real=None,
                  columnas: list[str] | None = None, largo_ventana: int = 21, n_pliegues: int = 5,
                  semilla: int = 42) -> pd.DataFrame:
    """AUC de un clasificador que separa ventanas reales de sinteticas (0,5 = indistinguibles).

    Ver el docstring del modulo para ventanas, regla de regimen, equilibrio,
    pliegues agrupados y filas de salida (``metrica`` 'auc' y 'n_ventanas').
    """
    if largo_ventana < 3:
        raise ValueError("largo_ventana debe ser >= 3 (sd y autocorrelacion por ventana).")
    panel_r, reg_r = separar_real(real, regimen_real)
    trays = separar_sintetico(sintetico)
    columnas = list(panel_r.columns) if columnas is None else list(columnas)
    est = estandarizador(panel_r, columnas)

    W_r, frac_r = _ventanas(est(panel_r), reg_r, largo_ventana)
    X_r = resumen_ventanas(W_r, columnas)
    bloques_s, fracs_s, ids_s = [], [], []
    for i, (panel_s, reg_s) in enumerate(trays):
        W_s, frac_s = _ventanas(est(panel_s), reg_s, largo_ventana)
        bloques_s.append(resumen_ventanas(W_s, columnas))
        fracs_s.append(frac_s)
        ids_s.append(np.full(len(frac_s), i))
    X_s, frac_s, path_s = np.vstack(bloques_s), np.concatenate(fracs_s), np.concatenate(ids_s)

    filas = []
    for j, regimen in enumerate(REGIMENES):
        rng = np.random.default_rng([semilla, j])
        m_r = _mascara_regimen(frac_r, regimen)
        m_s = np.flatnonzero(_mascara_regimen(frac_s, regimen))
        n_r = int(m_r.sum())
        elegidos = m_s[_equilibrar(path_s[m_s], n_r, rng)] if m_s.size else m_s
        aucs: list[float] = []
        if n_r >= 2 * n_pliegues and elegidos.size >= n_pliegues:
            pl_r, pl_s = _pliegues(n_r, path_s[elegidos], n_pliegues, rng)
            aucs = _auc_cv(X_r[m_r], X_s[elegidos], pl_r, pl_s, n_pliegues, semilla)
        media = float(np.mean(aucs)) if aucs else np.nan
        filas.append(dict(regimen=regimen, columna="—", tipo="—", metrica="auc", real=0.5,
                          sintetico=media, banda_inf=float(np.min(aucs)) if aucs else np.nan,
                          banda_sup=float(np.max(aucs)) if aucs else np.nan,
                          cociente=np.nan, en_banda=np.nan))
        filas.append(dict(regimen=regimen, columna="—", tipo="—", metrica="n_ventanas",
                          real=float(n_r), sintetico=float(elegidos.size), banda_inf=np.nan,
                          banda_sup=np.nan, cociente=np.nan, en_banda=np.nan))
    tabla = pd.DataFrame(filas)
    tabla["en_banda"] = pd.array([pd.NA] * len(tabla), dtype="boolean")  # mismo dtype que fidelidad
    return tabla
