"""Memorizacion: distancia al vecino real mas cercano con indice integro.

Pregunta: el generador produce dias (o ventanas) nuevos o devuelve, casi
literalmente, dias del entrenamiento? Leccion del taller previo (F8): con el
indice de vecindad submuestreado el jitter aprobaba (0,912); con el indice
integro suspende (0,046). Por eso:

- El indice (``scipy.spatial.cKDTree``) se construye SIEMPRE sobre el real de
  train INTEGRO. Nunca se submuestrea.
- Solo se pueden submuestrear CONSULTAS: las sinteticas si superan
  ``max_consultas`` (muestra uniforme sin reemplazo con ``semilla``; las filas
  'n_consultas' dicen cuantas se usaron). Las reales no se submuestrean (son
  ~11.000 y la busqueda es barata).

Espacio
-------
Columnas estandarizadas con media y sd del real (``_comun.estandarizador``).
Con ``largo_ventana = L > 1`` cada vector apila L dias consecutivos (dimension
d*L; ventanas deslizantes, paso 1, dentro de cada trayectoria y del real).
Por defecto se evalua en dos espacios (``columna``):

- ``'modeladas'`` (``tipo='modelada'``): las columnas que el generador produce
  de verdad (todas salvo las re-derivadas del S&P 500). Es donde una copia es
  literal: el bootstrap copia dias del real en este espacio.
- ``'todas'`` (``tipo='—'``): todas las columnas publicas. Las re-derivadas
  (drawdown, vol, momentum, z) dependen de la senda sintetica, asi que una copia
  de retornos no es una copia de estas columnas.

El veredicto lee ``'modeladas'`` (``validacion.memorizacion.espacio_veredicto``):
en ``'todas'`` los controles positivos (jitter, bootstrap) aprobarian.

Si ``columnas`` se pasa, solo se evalua ese conjunto (``columna='seleccion'``).
Si no hay columnas re-derivadas, solo sale ``'todas'``.

Distancias
----------
- ``d_sint``: para cada vector sintetico, distancia euclidea al vecino real mas
  cercano en el indice integro.
- ``d_real``: para cada vector real i, distancia al vecino real mas cercano j con
  ``|i - j| >= max(exclusion, L)`` sesiones (se excluyen los vecinos temporales,
  que comparten informacion por autocorrelacion y solape de ventanas). Se piden
  k vecinos y se filtran; las consultas sin vecino valido se repiten con k doble.

Regimen: el de la CONSULTA (columna ``regime`` del sintetico; regimen real del
dia real; con L > 1, crisis si mas de la mitad de la ventana es crisis). El
indice es el real entero, de ambos regimenes: memorizar es devolver un dia real
cualquiera, y condicionar el indice al regimen castigaria a un generador que
pone en crisis un dia de calma por algo que no es copia.

Salida (contrato largo de ``fidelidad``), por ``regimen`` en
``('todos', 'calma', 'crisis')`` y espacio (``columna``):

- ``'cociente_nn'``: ``real`` = mediana(d_real), ``sintetico`` = mediana(d_sint),
  ``cociente`` = mediana(d_sint) / mediana(d_real). Umbral a priori (yaml):
  >= 0,90. Banda = banda jackknife del 95 % de mediana(d_real) suprimiendo una
  racha del regimen cada vez (en crisis: por episodio), ``mediana +- 1,96 se_jk``;
  ``en_banda`` = mediana(d_sint) dentro de ella.
- ``'frac_copias'``: fraccion de d < ``tol_copia`` (copia literal); ``real`` es la
  del real-real (duplicados del propio real), ``sintetico`` la de d_sint.
- ``'dispersion'``: distancia media al centroide del regimen (cada muestra a su
  propio centroide); ``cociente`` sintetico/real. Separa copiar de encoger: una
  copia tiene ``cociente_nn ~ 0`` con ``dispersion ~ 1`` y ``frac_copias`` alta;
  la infradispersion tiene ``cociente_nn < 1`` con ``dispersion < 1`` y
  ``frac_copias ~ 0``. Banda jackknife igual que arriba.
- ``'n_consultas'``: ``real`` / ``sintetico`` = numero de vectores consultados.

Formulas del jackknife (G grupos = rachas del regimen): theta_g es el
estadistico sin la racha g, ``se = sqrt((G-1)/G * sum (theta_g - mean theta)^2)``.
Con G < 2 la banda es NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from regimenes.sinteticos.espacio import DERIVADAS
from regimenes.sinteticos.validacion._comun import (
    estandarizador,
    id_episodio,
    separar_real,
    separar_sintetico,
)

REGIMENES = ("todos", "calma", "crisis")
TOL_COPIA = 1e-8


def _apilar(Z: np.ndarray, reg: np.ndarray, largo: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ventanas deslizantes apiladas ``(n-L+1, d*L)``, su regimen y su posicion final."""
    if largo == 1:
        return Z, reg, np.arange(len(Z))
    n = len(Z) - largo + 1
    if n <= 0:
        return np.empty((0, Z.shape[1] * largo)), np.empty(0, dtype=int), np.empty(0, dtype=int)
    V = np.lib.stride_tricks.sliding_window_view(Z, (largo, Z.shape[1]))[:, 0].reshape(n, -1)
    frac = np.lib.stride_tricks.sliding_window_view(reg, largo).mean(axis=1)
    return np.ascontiguousarray(V), (frac > 0.5).astype(int), np.arange(largo - 1, len(Z))


def _d_real(arbol: cKDTree, V: np.ndarray, pos: np.ndarray, exclusion: int) -> np.ndarray:
    """Distancia de cada vector real a su vecino real mas cercano fuera de la zona de exclusion."""
    d = np.full(len(V), np.inf)
    pendientes = np.arange(len(V))
    k = min(2 * exclusion + 2, len(V))
    while pendientes.size:
        dist, idx = arbol.query(V[pendientes], k=k, workers=-1)
        dist, idx = dist.reshape(len(pendientes), -1), idx.reshape(len(pendientes), -1)
        validos = np.abs(pos[np.minimum(idx, len(pos) - 1)] - pos[pendientes][:, None]) >= exclusion
        validos &= idx < len(pos)
        hay = validos.any(axis=1)
        primero = validos.argmax(axis=1)
        d[pendientes[hay]] = dist[hay, primero[hay]]
        if k >= len(V):
            break
        pendientes = pendientes[~hay]
        k = min(2 * k, len(V))
    return d


def _jackknife(valores: np.ndarray, grupos: np.ndarray, estadistico) -> tuple[float, float]:
    """Banda 95 % ``theta +- 1,96 se_jk`` suprimiendo un grupo cada vez."""
    unicos = np.unique(grupos)
    if unicos.size < 2:
        return np.nan, np.nan
    theta = np.array([estadistico(valores[grupos != g]) for g in unicos])
    se = np.sqrt((unicos.size - 1) / unicos.size * ((theta - theta.mean()) ** 2).sum())
    centro = estadistico(valores)
    return float(centro - 1.96 * se), float(centro + 1.96 * se)


def _dist_centroide(V: np.ndarray) -> np.ndarray:
    return np.linalg.norm(V - V.mean(axis=0), axis=1) if len(V) else np.empty(0)


def _fila(regimen, columna, tipo, metrica, real, sint, banda=(np.nan, np.nan), cociente=np.nan):
    inf, sup = banda
    en_banda = bool(inf <= sint <= sup) if np.isfinite(inf) and np.isfinite(sup) else np.nan
    return dict(regimen=regimen, columna=columna, tipo=tipo, metrica=metrica, real=float(real),
                sintetico=float(sint), banda_inf=inf, banda_sup=sup, cociente=float(cociente),
                en_banda=en_banda)


def _espacio(panel_r, reg_r, trays, cols, nombre, tipo, largo, exclusion, max_consultas, rng):
    est = estandarizador(panel_r, cols)
    V_r, g_r, pos_r = _apilar(est(panel_r), reg_r, largo)
    racha_r = id_episodio(reg_r)[pos_r]
    arbol = cKDTree(V_r)  # indice INTEGRO: nunca se submuestrea
    d_r = _d_real(arbol, V_r, pos_r, max(exclusion, largo))

    trozos = [_apilar(est(p), r, largo) for p, r in trays]
    V_s = np.vstack([t[0] for t in trozos])
    g_s = np.concatenate([t[1] for t in trozos])
    if max_consultas is not None and len(V_s) > max_consultas:
        sel = np.sort(rng.choice(len(V_s), size=max_consultas, replace=False))
        V_s, g_s = V_s[sel], g_s[sel]
    d_s, _ = arbol.query(V_s, k=1, workers=-1)

    filas = []
    for regimen in REGIMENES:
        m_r = np.ones(len(V_r), bool) if regimen == "todos" else g_r == (regimen == "crisis")
        m_s = np.ones(len(V_s), bool) if regimen == "todos" else g_s == (regimen == "crisis")
        if not m_r.any() or not m_s.any():
            continue
        dr, ds = d_r[m_r], d_s[m_s]
        dr = dr[np.isfinite(dr)]
        med_r, med_s = np.median(dr), np.median(ds)
        banda = _jackknife(d_r[m_r], racha_r[m_r], lambda v: np.median(v[np.isfinite(v)]))
        filas.append(_fila(regimen, nombre, tipo, "cociente_nn", med_r, med_s, banda, med_s / med_r))
        filas.append(_fila(regimen, nombre, tipo, "frac_copias", np.mean(dr < TOL_COPIA),
                           np.mean(ds < TOL_COPIA)))
        # dispersion: cada muestra a su propio centroide (del regimen)
        c_r, c_s = _dist_centroide(V_r[m_r]), _dist_centroide(V_s[m_s])
        banda_c = _jackknife(c_r, racha_r[m_r], np.mean)
        filas.append(_fila(regimen, nombre, tipo, "dispersion", c_r.mean(), c_s.mean(), banda_c,
                           c_s.mean() / c_r.mean()))
        filas.append(_fila(regimen, nombre, tipo, "n_consultas", m_r.sum(), m_s.sum()))
    return filas


def memorizacion(real: pd.DataFrame, sintetico: list[pd.DataFrame], *, regimen_real=None,
                 columnas: list[str] | None = None, largo_ventana: int = 1, exclusion: int = 21,
                 semilla: int = 42, max_consultas: int | None = 300_000) -> pd.DataFrame:
    """Cociente d_NN(sintetico -> real) / d_NN(real -> real) por regimen.

    Ver el docstring del modulo (espacios, exclusion temporal, jackknife por
    racha y filas 'cociente_nn', 'frac_copias', 'dispersion', 'n_consultas').
    ``max_consultas`` limita las consultas SINTETICAS (nunca el indice real);
    ``None`` = todas.
    """
    if largo_ventana < 1:
        raise ValueError("largo_ventana debe ser >= 1.")
    panel_r, reg_r = separar_real(real, regimen_real)
    trays = separar_sintetico(sintetico)
    rng = np.random.default_rng(semilla)
    if columnas is not None:
        espacios = [(list(columnas), "seleccion", "—")]
    else:
        todas = list(panel_r.columns)
        modeladas = [c for c in todas if c not in DERIVADAS]
        espacios = [(todas, "todas", "—")]
        if len(modeladas) < len(todas):
            espacios.insert(0, (modeladas, "modeladas", "modelada"))
    filas = []
    for cols, nombre, tipo in espacios:
        filas += _espacio(panel_r, reg_r, trays, cols, nombre, tipo, largo_ventana, exclusion,
                          max_consultas, rng)
    tabla = pd.DataFrame(filas)
    tabla["en_banda"] = tabla["en_banda"].astype("boolean")  # mismo dtype que fidelidad (<NA> sin banda)
    return tabla
