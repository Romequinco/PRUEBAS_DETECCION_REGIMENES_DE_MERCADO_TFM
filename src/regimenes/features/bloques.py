"""bloques.py — Termómetros por bloque: una PCA causal por bloque de features.

Sección 6 de la organización de ``regimenes.features`` (PendientesOscar.md §6.3 y §6.5).
El reparto en bloques y las anclas se pre-registran en ``configs/bloques.yaml`` (ADR-008).

Un termómetro es UNA columna causal por bloque. Convenciones:

- **Reajuste mensual.** Fechas de reajuste = último día del índice de cada mes. En el
  reajuste ``r`` se ajusta con las filas ``<= r`` (expansiva, o las últimas ``ventana``
  filas) y esas cargas se aplican a los días ``(r, r_siguiente]``. El valor de ``t``
  solo depende de datos ``<= t``. Antes del primer reajuste válido: NaN.
- **Escalado.** En la ventana, cada columna se re-estandariza con su media/desv de las
  filas de ajuste (ddof=1) y la PCA es la de la matriz de correlación (``eigh``). Al
  proyectar se usa ese mismo escalado. Pesos finales ``w = a / sqrt(a'Ca)``, con ``a``
  la dirección (PC1 o media): el termómetro tiene media 0 y varianza 1 en la ventana de
  ajuste, así que tramos PCA y tramos media están en la misma escala.
- **Signo.** Tras cada ajuste, si ``carga_ancla * signo < 0`` se invierte el vector:
  el termómetro siempre sube con la ancla orientada (sube con el estrés declarado).
- **Umbral.** Si la PC1 explica menos de ``umbral_var``, el tramo usa la media simple
  de las columnas orientadas: cada una multiplicada por el signo de su correlación (en
  la ventana) con la ancla orientada.
- **NaN.** En cada reajuste son elegibles las columnas con ``>= min_obs`` datos no-NaN y
  desviación > 0 en la ventana; se ajusta con las filas completas de las elegibles (si
  no llegan a ``min_obs``, se descartan las elegibles con menos datos, nunca la ancla).
  Si la ancla no es elegible, el tramo es NaN. Al proyectar, un día con alguna elegible
  en NaN usa las disponibles D y reescala por masa de peso absoluto:
  ``T_t = sum_{i en D} w_i z_it * sum_{i} |w_i| / sum_{i en D} |w_i|`` (NaN si D vacío).
  Esos días se cuentan en ``n_dias_renormalizados``. Las no elegibles se ignoran.
- **Bloque de una columna** declarado así: termómetro = ``signo * columna`` (sin PCA,
  sin reajuste). Un tramo de un bloque mayor con una sola elegible pasa por la PCA
  trivial (la columna re-estandarizada en la ventana y orientada).

Implementación: los momentos de cada ventana salen de sumas acumuladas (``_Acumulados``),
así ~230 reajustes x 8 bloques (B) o ~770 x 5 (A) tardan segundos. Es determinista: el
truncado en ``t`` reproduce exactamente los valores ``<= t``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from regimenes import rutas

BLOQUES_YAML = rutas.CONFIGS / "bloques.yaml"

# Columnas del DataFrame de diagnósticos (una fila por reajuste).
COLUMNAS_DIAGNOSTICO = [
    'fecha_reajuste', 'metodo', 'n_filas_ajuste', 'n_elegibles', 'elegibles', 'var_pc1',
    'cargas', 'peso_max_abs', 'col_peso_max', 'carga_ancla', 'invertido', 'coseno_prev',
    'n_dias', 'n_dias_renormalizados', 'ausentes',
]


# --------------------------------------------------------------------------- #
# 6. Termómetros por bloque
# --------------------------------------------------------------------------- #
def _ancla(ancla: Mapping[str, Any] | Sequence[Any]) -> tuple[str, int]:
    """Normaliza la ancla: ``{'columna': c, 'signo': ±1}`` o ``(c, ±1)``."""
    col, signo = (ancla['columna'], ancla['signo']) if isinstance(ancla, Mapping) else ancla
    if int(signo) not in (1, -1):
        raise ValueError(f"signo de la ancla debe ser +1 o -1, no {signo!r}")
    return str(col), int(signo)


def fechas_reajuste(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Último día del índice de cada mes (fechas de reajuste mensual)."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period('M')).max().to_numpy())


def _coseno(prev: dict[str, float] | None, act: dict[str, float]) -> float:
    """Coseno entre dos vectores de cargas sobre sus columnas comunes (NaN si < 2).
    Coseno y no correlación: detecta inversiones (-1) y vale para bloques pequeños."""
    if not prev:
        return np.nan
    comunes = [c for c in act if c in prev]
    if len(comunes) < 2:
        return np.nan
    a = np.array([prev[c] for c in comunes])
    b = np.array([act[c] for c in comunes])
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den > 0 else np.nan


_TOL_VAR = 1e-10  # varianza (relativa a 1 + media^2) por debajo de la cual una columna es constante


class _Acumulados:
    """Sumas acumuladas (fila 0 = 0) para obtener en O(k^2) los momentos de cualquier
    ventana ``[ini, fin)``: por columna (n no-NaN, suma, suma de cuadrados) y, para un
    conjunto de columnas E, los de las filas completas en E (n, suma, X'X). Las de E se
    calculan la primera vez que se piden y se guarda solo el último E (cambia poco)."""

    def __init__(self, X: np.ndarray):
        self.X, self.valido = X, ~np.isnan(X)
        X0 = np.where(self.valido, X, 0.0)
        self.col = [self._cum(m) for m in (self.valido.astype(float), X0, X0 * X0)]
        self._clave, self._comp = None, None

    @staticmethod
    def _cum(m: np.ndarray) -> np.ndarray:
        return np.concatenate([np.zeros((1,) + m.shape[1:]), np.cumsum(m, axis=0)])

    def por_columna(self, ini: int, fin: int) -> tuple[np.ndarray, np.ndarray]:
        """(n no-NaN, varianza > 0) por columna en la ventana."""
        n, s1, s2 = (c[fin] - c[ini] for c in self.col)
        with np.errstate(invalid='ignore', divide='ignore'):
            media = s1 / n
            var_ok = (s2 / n - media ** 2) > _TOL_VAR * (1 + media ** 2)
        return n, var_ok

    def completas(self, eleg: list[int], ini: int, fin: int
                  ) -> tuple[int, np.ndarray, np.ndarray]:
        """(n, media, covarianza ddof=1) de las filas de la ventana completas en `eleg`."""
        if self._clave != tuple(eleg):
            m = self.valido[:, eleg].all(axis=1)
            Xm = np.where(m[:, None], self.X[:, eleg], 0.0)
            self._clave = tuple(eleg)
            self._comp = [self._cum(m.astype(float)), self._cum(Xm),
                          self._cum(Xm[:, :, None] * Xm[:, None, :])]
        n, s1, s2 = (c[fin] - c[ini] for c in self._comp)
        n = int(round(n))
        if n < 2:
            return n, np.full(len(eleg), np.nan), np.full((len(eleg),) * 2, np.nan)
        mu = s1 / n
        return n, mu, (s2 - n * np.outer(mu, mu)) / (n - 1)


def _ajustar(acu: _Acumulados, ini: int, fin: int, k_anc: int, signo: int, umbral_var: float,
             min_obs: int) -> dict[str, Any] | None:
    """Ajuste de un reajuste sobre la ventana de filas ``[ini, fin)``. None si la ancla
    no es elegible."""
    n_ok, var_ok = acu.por_columna(ini, fin)
    eleg = [i for i in range(len(n_ok)) if n_ok[i] >= min_obs and var_ok[i]]
    if k_anc not in eleg:
        return None
    while True:
        n, mu, cov = acu.completas(eleg, ini, fin)
        var = np.diag(cov)
        constante = ~(var > _TOL_VAR * (1 + mu ** 2))
        if n >= min_obs and not constante.any():
            break
        # descartar la elegible (no ancla) con menos datos; si es la ancla la constante -> None
        malas = [e for e, c in zip(eleg, constante) if c]
        otras = [e for e in eleg if e != k_anc]
        if k_anc in malas or not otras:
            return None
        quitar = malas[0] if malas else min(otras, key=lambda e: n_ok[e])
        eleg.remove(quitar)
    sd = np.sqrt(var)
    C = cov / np.outer(sd, sd)
    vals, vecs = np.linalg.eigh(C)
    v, k = vecs[:, -1], len(eleg)
    var_pc1 = float(vals[-1] / k)
    j_anc = eleg.index(k_anc)
    invertido = bool(v[j_anc] * signo < 0)
    if invertido:
        v = -v
    if var_pc1 >= umbral_var or k == 1:
        metodo, a = 'pca', v
    else:
        s = np.sign(C[:, j_anc] * signo)
        s[s == 0] = 1.0
        metodo, a = 'media', s / np.sqrt(k)
    w = a / np.sqrt(a @ C @ a)
    return {'eleg': eleg, 'mu': mu, 'sd': sd, 'w': w, 'a': a, 'metodo': metodo,
            'var_pc1': var_pc1, 'invertido': invertido, 'n_filas': n}


def _proyectar(X: np.ndarray, aj: dict[str, Any]) -> tuple[np.ndarray, int]:
    """Proyecta filas de ``X`` con el ajuste ``aj``; renormaliza por masa de |w| si faltan
    columnas. Devuelve (valores, nº de días renormalizados)."""
    Z = (X[:, aj['eleg']] - aj['mu']) / aj['sd']
    disp = ~np.isnan(Z)
    aw = np.abs(aj['w'])
    masa = disp @ aw
    num = np.where(disp, Z, 0.0) @ aj['w']
    with np.errstate(invalid='ignore', divide='ignore'):
        val = np.where(masa > 0, num * aw.sum() / masa, np.nan)
    n_disp = disp.sum(axis=1)
    return val, int(((n_disp > 0) & (n_disp < disp.shape[1])).sum())


def termometro_bloque(panel: pd.DataFrame, columnas: Sequence[str],
                      ancla: Mapping[str, Any] | Sequence[Any], *, umbral_var: float = 0.0,
                      min_obs: int = 252, ventana: int | None = None
                      ) -> tuple[pd.Series, pd.DataFrame]:
    """Termómetro causal de un bloque: PC1 (o media) reajustada cada mes con el pasado.

    `panel` con índice de fechas creciente y sin duplicados; `columnas` deben existir en
    él; `ancla` = ``{'columna', 'signo'}`` o ``(columna, signo)`` con la columna en el
    bloque. Devuelve (serie sobre ``panel.index``, diagnósticos con una fila por reajuste,
    columnas ``COLUMNAS_DIAGNOSTICO``). Convenciones en el docstring del módulo.
    """
    col_anc, signo = _ancla(ancla)
    columnas = list(dict.fromkeys(columnas))
    if col_anc not in columnas:
        raise ValueError(f"la ancla {col_anc!r} no pertenece al bloque")
    faltan = [c for c in columnas if c not in panel.columns]
    if faltan:
        raise KeyError(f"columnas ausentes del panel: {faltan}")
    idx = panel.index
    if not (isinstance(idx, pd.DatetimeIndex) and idx.is_monotonic_increasing and idx.is_unique):
        raise ValueError("panel necesita un DatetimeIndex creciente y sin duplicados")
    if ventana is not None and ventana < min_obs:
        raise ValueError("ventana < min_obs: ninguna columna sería elegible")

    if len(columnas) == 1:
        serie = (signo * panel[col_anc].astype(float)).rename(None)
        diag = pd.DataFrame([{
            'fecha_reajuste': pd.NaT, 'metodo': 'columna_unica', 'n_filas_ajuste': int(serie.notna().sum()),
            'n_elegibles': 1, 'elegibles': [col_anc], 'var_pc1': 1.0, 'cargas': {col_anc: float(signo)},
            'peso_max_abs': 1.0, 'col_peso_max': col_anc, 'carga_ancla': float(signo), 'invertido': False,
            'coseno_prev': np.nan, 'n_dias': int(serie.notna().sum()), 'n_dias_renormalizados': 0,
            'ausentes': [],
        }], columns=COLUMNAS_DIAGNOSTICO)
        return serie, diag

    X = panel[columnas].to_numpy(dtype=float)
    acu = _Acumulados(X)
    k_anc = columnas.index(col_anc)
    pos = idx.get_indexer(fechas_reajuste(idx))
    out = np.full(len(idx), np.nan)
    filas, prev = [], None
    for j, p in enumerate(pos):
        ini = 0 if ventana is None else max(0, p + 1 - ventana)
        fin = pos[j + 1] if j + 1 < len(pos) else p          # aplica a (p, fin]
        aj = _ajustar(acu, ini, p + 1, k_anc, signo, umbral_var, min_obs)
        fila = {'fecha_reajuste': idx[p], 'n_dias': int(fin - p), 'ausentes': []}
        if aj is None:
            filas.append({**fila, 'metodo': 'sin_ancla', 'n_elegibles': 0, 'elegibles': [],
                          'n_dias_renormalizados': 0})
            continue
        val, n_ren = _proyectar(X[p + 1:fin + 1], aj)
        out[p + 1:fin + 1] = val
        nombres = [columnas[e] for e in aj['eleg']]
        a = aj['a'] / np.linalg.norm(aj['a'])
        cargas = dict(zip(nombres, a.tolist()))
        cuota = np.abs(a) / np.abs(a).sum()
        filas.append({**fila, 'metodo': aj['metodo'], 'n_filas_ajuste': aj['n_filas'],
                      'n_elegibles': len(nombres), 'elegibles': nombres, 'var_pc1': aj['var_pc1'],
                      'cargas': cargas, 'peso_max_abs': float(cuota.max()),
                      'col_peso_max': nombres[int(cuota.argmax())], 'carga_ancla': cargas[col_anc],
                      'invertido': aj['invertido'], 'coseno_prev': _coseno(prev, cargas),
                      'n_dias_renormalizados': n_ren})
        prev = cargas
    return pd.Series(out, index=idx), pd.DataFrame(filas, columns=COLUMNAS_DIAGNOSTICO)


def construir_termometros(panel: pd.DataFrame, bloques_pista: Mapping[str, Mapping[str, Any]],
                          **kw: Any) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Un termómetro por bloque (``{nombre: {'columnas': [...], 'ancla': {...}}}``, el
    formato de ``cargar_bloques``). `kw` va a `termometro_bloque`.

    Columnas declaradas ausentes del panel se omiten y se listan en la columna
    ``ausentes`` del diagnóstico. Si falta la ancla, el termómetro es NaN (con aviso).
    Devuelve (DataFrame de termómetros sobre ``panel.index``, {nombre: diagnósticos}).
    """
    series, diags = {}, {}
    for nombre, b in bloques_pista.items():
        col_anc, _ = _ancla(b['ancla'])
        presentes = [c for c in b['columnas'] if c in panel.columns]
        ausentes = [c for c in b['columnas'] if c not in panel.columns]
        if col_anc not in presentes:
            warnings.warn(f"{nombre}: la ancla {col_anc!r} no está en el panel; termómetro NaN")
            series[nombre] = pd.Series(np.nan, index=panel.index)
            diag = pd.DataFrame([{'metodo': 'ancla_ausente', 'n_elegibles': 0, 'elegibles': [],
                                  'n_dias_renormalizados': 0}], columns=COLUMNAS_DIAGNOSTICO)
        else:
            series[nombre], diag = termometro_bloque(panel, presentes, b['ancla'], **kw)
        diag['ausentes'] = [ausentes] * len(diag)
        diags[nombre] = diag
    return pd.DataFrame(series, index=panel.index), diags


def resumen_estabilidad(termometros: pd.DataFrame, diagnosticos: Mapping[str, pd.DataFrame]
                        ) -> pd.DataFrame:
    """Resumen por termómetro SIN etiquetas de crisis (§6.5, punto 2), para comparar umbrales.

    - ``frac_pca``, ``n_cambios_metodo``: tramos PCA vs media y alternancias entre ambos.
    - ``var_pc1_mediana``/``_min``; ``peso_max`` y ``n_peso_mayor_50`` (criterio «ninguna
      columna > 50 %», cuota de |carga| sobre la suma de |cargas|).
    - ``coseno_min``/``_mediana`` con el reajuste anterior; ``n_saltos_signo`` (coseno < 0).
    - ``ratio_salto_reajuste``: media de |ΔT| el primer día tras un reajuste / media de
      |ΔT| el resto de días (≈1 si cambiar las cargas no mete escalones).
    """
    filas = []
    for nombre, d in diagnosticos.items():
        ok = d[d['metodo'].isin(['pca', 'media'])]
        t = termometros[nombre]
        dt = t.diff().abs()
        # primer día tras cada reajuste = posición siguiente a la fecha de reajuste
        es_reajuste = t.index.isin(pd.DatetimeIndex(ok['fecha_reajuste']))
        tras = np.r_[False, es_reajuste[:-1]]
        base = dt[~tras].mean()
        ausentes = d['ausentes'].iloc[0] if len(d) else []
        filas.append({
            'termometro': nombre,
            'n_reajustes_validos': len(ok),
            'primer_valor': t.first_valid_index(),
            'frac_pca': float((ok['metodo'] == 'pca').mean()) if len(ok) else np.nan,
            'n_cambios_metodo': int((ok['metodo'] != ok['metodo'].shift()).iloc[1:].sum()),
            'var_pc1_mediana': ok['var_pc1'].median(),
            'var_pc1_min': ok['var_pc1'].min(),
            'peso_max': ok['peso_max_abs'].max(),
            'n_peso_mayor_50': int((ok['peso_max_abs'] > 0.5).sum()),
            'coseno_min': ok['coseno_prev'].min(),
            'coseno_mediana': ok['coseno_prev'].median(),
            'n_saltos_signo': int((ok['coseno_prev'] < 0).sum()),
            'ratio_salto_reajuste': dt[tras].mean() / base if base and base > 0 else np.nan,
            'n_dias_renormalizados': int(d['n_dias_renormalizados'].fillna(0).sum()),
            'ausentes': list(ausentes) if isinstance(ausentes, list) else [],
        })
    return pd.DataFrame(filas)


def comparar_umbrales(panel: pd.DataFrame, bloques_pista: Mapping[str, Mapping[str, Any]],
                      umbrales: Sequence[float], **kw: Any) -> pd.DataFrame:
    """`resumen_estabilidad` para cada ``umbral_var`` de `umbrales` (columna ``umbral_var``)."""
    partes = []
    for u in umbrales:
        t, d = construir_termometros(panel, bloques_pista, umbral_var=u, **kw)
        partes.append(resumen_estabilidad(t, d).assign(umbral_var=u))
    return pd.concat(partes, ignore_index=True)


# --------------------------------------------------------------------------- #
# Pre-registro (configs/bloques.yaml)
# --------------------------------------------------------------------------- #
def _leer_yaml(ruta: Path | None = None) -> dict[str, Any]:
    return yaml.safe_load((ruta or BLOQUES_YAML).read_text(encoding='utf-8'))


def parametros_bloques(ruta: Path | None = None) -> dict[str, Any]:
    """Parámetros globales del pre-registro: reajuste, min_obs, ventana, umbral_var decidido,
    umbrales_var_a_probar e historia_previa (``'toda'``: calcular con la historia anterior a la
    ventana y recortar; ver ADR-008)."""
    cfg = _leer_yaml(ruta)
    claves = ('reajuste', 'min_obs', 'ventana', 'umbral_var', 'umbrales_var_a_probar', 'historia_previa')
    return {k: cfg[k] for k in claves if k in cfg}


def cargar_bloques(pista: str, ruta: Path | None = None) -> dict[str, dict[str, Any]]:
    """Bloques de la pista ``'A'``/``'B'``: ``{nombre: {'columnas': [...], 'ancla':
    {'columna', 'signo'}}}``. Valida que la ancla esté en su bloque y que ninguna columna
    se repita entre bloques."""
    pista = pista.upper()
    if pista not in {'A', 'B'}:
        raise ValueError("pista debe ser 'A' o 'B'")
    bloques = _leer_yaml(ruta)[f'pista_{pista}']
    vistas: set[str] = set()
    out = {}
    for nombre, b in bloques.items():
        col_anc, signo = _ancla(b['ancla'])
        cols = list(b['columnas'])
        if col_anc not in cols:
            raise ValueError(f"{pista}/{nombre}: la ancla {col_anc!r} no está en el bloque")
        repetidas = vistas.intersection(cols) | {c for c in cols if cols.count(c) > 1}
        if repetidas:
            raise ValueError(f"{pista}/{nombre}: columnas repetidas {sorted(repetidas)}")
        vistas.update(cols)
        out[nombre] = {'columnas': cols, 'ancla': {'columna': col_anc, 'signo': signo}}
    return out
