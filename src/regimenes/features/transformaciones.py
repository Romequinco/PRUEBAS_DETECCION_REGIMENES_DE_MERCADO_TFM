"""
transformaciones.py — Features CAUSALES para los detectores de régimen (preprocesado v2).

Este módulo contiene las secciones 1 y 2; la 3 vive
en ``regimenes.features.lags``, la 4 en ``regimenes.features.paneles`` y la 5 en
``regimenes.features.causalidad``. ``regimenes.features`` re-exporta toda la API.

Hallazgo clave de la tarea previa: estandarizar con media/desviación de TODA la
muestra mete información del futuro (look-ahead sutil en el z-score). Aquí TODA
estandarización es causal: en `t` solo se usan estadísticos calculados con datos
`<= t` (expanding o rolling). NINGÚN estadístico de muestra completa.

Organización del módulo
-----------------------
1. **Primitivas causales** (`causal_zscore`, `log_returns`, `realized_vol`,
   `rolling_correlation`, `drawdown`, `momentum`): heredadas de la Capa 1 y
   demostradas en `notebooks/02_diseno_preprocesado.ipynb` §5.
2. **Recetario v2** (`ret_z`, `zscore_raw`, `change_z`, `spread_ret_z`,
   `cross_sectional_std_*`, `macro_z`, `mercado_z`, `mercado_spread_z`,
   `fed_stance`): composiciones de las primitivas usadas por
   `notebooks/03_preprocesado.ipynb`.
3. **Causalidad de calendario** (`LAG_PUBLICACION`, `aplicar_lag_publicacion`,
   ADR-003): tabla ÚNICA de lags de publicación por serie cruda. Los constructores
   la aplican a la entrada antes de calcular nada, de modo que el índice de toda
   serie que entra a una feature es su FECHA DE DISPONIBILIDAD, no su fecha FRED.
4. **Constructores de panel** (`seleccionar_raw`, `construir_diarias`,
   `construir_mensuales`, `alinear_diaria`, `alinear_mensual_con_edad`): implementan
   la tabla `FEAT` de 02 sobre un diccionario `{serie: pd.Series | pd.DataFrame
   cruda}` de una pista. Viven aquí (y no solo en el notebook) para que la
   causalidad del pipeline COMPLETO se pueda verificar con tests
   (`tests/test_preprocesado_causalidad.py`).
5. **Verificación** (`truncation_max_abs_diff`, `assert_causal`): test de
   truncado — recomputar con la entrada cortada en `cut` debe dar exactamente los
   mismos valores `<= cut` que con la muestra completa.

Convención de causalidad
------------------------
`causal_zscore` usa `expanding`/`rolling` `.mean()` y `.std()`, que por
construcción solo agregan observaciones hasta `t` inclusive. Incluir la propia
observación `t` en su normalización NO es look-ahead (ese dato está disponible en
`t`); lo prohibido es usar datos de `t+1..T`. Quien quiera normalización estricta
one-step-ahead puede pasar `lag=1` (usa estadísticos hasta `t-1`).

Alcance del test de truncado (importante)
-----------------------------------------
El truncado demuestra causalidad COMPUTACIONAL: ninguna operación usa filas
posteriores a `t`. NO demuestra que la FECHA con la que se indexa un dato sea su
fecha de disponibilidad real. Un dato mensual fechado el día 1 del mes pero que
es la media de todo ese mes (p. ej. `GS10`, `TB3MS`, `MOODYS_BAA`), o un dato
macro publicado a mitad del mes siguiente, pasa el truncado y aun así contiene
look-ahead de calendario. Desde ADR-003 eso lo resuelve `LAG_PUBLICACION` (sección
3), auditado en 02 §3.2 y cubierto por `tests/test_preprocesado_causalidad.py`.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# 1. Primitivas causales
# --------------------------------------------------------------------------- #
def causal_zscore(
    s: pd.Series,
    method: str = "expanding",
    window: int = 252,
    min_periods: int = 60,
    lag: int = 0,
) -> pd.Series:
    """Z-score CAUSAL: en `t` usa solo media/std de datos `<= t` (o `<= t-lag`).

    Parameters
    ----------
    method : {'expanding', 'rolling'}
        'expanding' usa toda la historia hasta t (estable a largo plazo).
        'rolling' usa una ventana de `window` observaciones (adaptativo).
    window : int
        Ventana para method='rolling'.
    min_periods : int
        Mínimo de observaciones antes de emitir un z-score (antes -> NaN, sin
        imputar). Se cuenta en observaciones NATIVAS de la serie (60 días en una
        diaria, 60 meses en una mensual).
    lag : int
        0 = media/std incluyen t (estándar, causal). 1 = normalización estricta
        one-step-ahead (estadísticos hasta t-1).

    Returns
    -------
    pd.Series alineada con `s`.
    """
    if method == "expanding":
        roll = s.expanding(min_periods=min_periods)
    elif method == "rolling":
        roll = s.rolling(window=window, min_periods=min_periods)
    else:
        raise ValueError(f"method desconocido: {method!r}")
    mu = roll.mean()
    sigma = roll.std(ddof=1)
    if lag > 0:
        mu = mu.shift(lag)
        sigma = sigma.shift(lag)
    z = (s - mu) / sigma.replace(0.0, np.nan)
    return z.rename(f"{s.name}_z" if s.name else "z")


def log_returns(prices: pd.Series) -> pd.Series:
    """Retorno logarítmico entre observaciones consecutivas del índice nativo."""
    return np.log(prices / prices.shift(1))


def realized_vol(returns: pd.Series, window: int = 21, annualize: bool = True) -> pd.Series:
    """Volatilidad realizada rolling causal (anualizada ×√252 si annualize)."""
    vol = returns.rolling(window=window, min_periods=max(5, window // 2)).std(ddof=1)
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS)
    return vol


def rolling_correlation(a: pd.Series, b: pd.Series, window: int = 60) -> pd.Series:
    """Correlación rolling causal entre dos series."""
    return a.rolling(window=window, min_periods=max(20, window // 2)).corr(b)


def drawdown(prices: pd.Series) -> pd.Series:
    """Drawdown corriente respecto al máximo histórico CAUSAL (expanding max)."""
    peak = prices.expanding(min_periods=1).max()
    return prices / peak - 1.0


def momentum(prices: pd.Series, long_w: int = 252, short_w: int = 21) -> pd.Series:
    """Momentum 12M-1M: retorno log a `long_w` días menos retorno a `short_w`."""
    long_ret = np.log(prices / prices.shift(long_w))
    short_ret = np.log(prices / prices.shift(short_w))
    return long_ret - short_ret


# --------------------------------------------------------------------------- #
# 2. Recetario v2 (composiciones de las primitivas; usado por 03_preprocesado)
# --------------------------------------------------------------------------- #
RawDict = Mapping[str, "pd.Series | pd.DataFrame"]

BREADTH_11 = ['NASDAQ_COMP', 'NASDAQ100', 'RUSSELL2000', 'RUSSELL1000', 'RUSSELL3000',
              'SP_MIDCAP400', 'SP100', 'SP_SMALLCAP600', 'WILSHIRE5000', 'NYSE_COMP', 'SP500_TR']
SPDR_9 = ['SPDR_XLB', 'SPDR_XLE', 'SPDR_XLF', 'SPDR_XLI', 'SPDR_XLK',
          'SPDR_XLP', 'SPDR_XLU', 'SPDR_XLV', 'SPDR_XLY']


def _get(raw: RawDict, nombre: str) -> pd.Series | None:
    """Serie cruda si la pista la tiene cargada; si no, None (nunca hace I/O)."""
    return raw.get(nombre)


def logret(raw: RawDict, nombre: str) -> pd.Series | None:
    """log-retorno causal de una serie de PRECIO/INDICE (nunca de un tipo/spread)."""
    s = _get(raw, nombre)
    return log_returns(s) if s is not None else None


def ret_z(raw: RawDict, nombre: str, out_name: str) -> pd.Series | None:
    """z-score causal (expanding) del log-retorno de una serie de precio."""
    r = logret(raw, nombre)
    return causal_zscore(r).rename(out_name) if r is not None else None


def zscore_raw(raw: RawDict, nombre: str, out_name: str) -> pd.Series | None:
    """z-score causal DIRECTO sobre la serie cruda: niveles/tipos/spreads en puntos
    (VIX, DGS10, BAA10Y...) o series que YA son retorno (factores Ken French en %)."""
    s = _get(raw, nombre)
    return causal_zscore(s).rename(out_name) if s is not None else None


def change_z(raw: RawDict, nombre: str, out_name: str) -> pd.Series | None:
    """z-score causal del cambio (diff simple) de un NIVEL de tipo/spread/índice de vol."""
    s = _get(raw, nombre)
    return causal_zscore(s.diff()).rename(out_name) if s is not None else None


def spread_ret_z(raw: RawDict, a: str, b: str, out_stub: str) -> pd.Series | None:
    """z-score causal de ret(a) - ret(b). None si falta alguna pata en la pista."""
    ra, rb = logret(raw, a), logret(raw, b)
    if ra is None or rb is None:
        return None
    spread = (ra - rb).rename(out_stub)  # la resta alinea por fecha (NaN sin solape)
    return causal_zscore(spread).rename(f'{out_stub}_z')


def cross_sectional_std_ret(raw: RawDict, nombres: list[str]) -> pd.Series | None:
    """Dispersión (std, ddof=1) cross-seccional de RETORNOS, fila a fila (causal
    trivialmente: solo usa ese día). None si la pista tiene < 2 series de la familia.
    Nota: el número de series que entra cada día puede variar (inicios distintos)."""
    rets = {n: logret(raw, n) for n in nombres if _get(raw, n) is not None}
    if len(rets) < 2:
        return None
    return pd.concat(rets, axis=1, sort=True).std(axis=1, ddof=1)


def cross_sectional_std_level(raw: RawDict, nombres: list[str]) -> pd.Series | None:
    """Como `cross_sectional_std_ret` pero sobre NIVELES (índices de vol implícita)."""
    lv = {n: _get(raw, n) for n in nombres if _get(raw, n) is not None}
    if len(lv) < 2:
        return None
    return pd.concat(lv, axis=1, sort=True).std(axis=1, ddof=1)


def zscore_or_none(s: pd.Series | None) -> pd.Series | None:
    return causal_zscore(s) if s is not None else None


def lag_publicacion(s: pd.Series | pd.DataFrame, months: int = 1,
                    days: int = 0) -> pd.Series | pd.DataFrame:
    """Desplaza el ÍNDICE `months` meses + `days` días hacia delante: el dato fechado
    en M se considera disponible en M+lag. No toca valores. (Demo en 02 §3.1.)"""
    s2 = s.copy()
    s2.index = s2.index + pd.DateOffset(months=months, days=days)
    return s2


def yoy(s: pd.Series) -> pd.Series:
    """Variación interanual en % de una serie MENSUAL: 100 * s / s.shift(12) - 100."""
    return 100 * s / s.shift(12) - 100


def macro_z(raw: RawDict, nombre: str, out_name: str, transform: str = 'yoy',
            lag_months: int = 0) -> pd.Series | None:
    """Feature MACRO mensual: transformación nativa -> (lag) -> z-score.

    ADR-003: dentro de `construir_mensuales` la entrada YA llega con el lag de
    publicación de `LAG_PUBLICACION` aplicado (índice = fecha de disponibilidad), por
    eso `lag_months` vale 0 por defecto; `lag_months > 0` solo para uso aislado sobre
    una serie con fecha FRED. El orden importa: zscore->lag habría usado el dato antes
    de su fecha de disponibilidad. `transform`: 'yoy', 'diff12' (cambio a 12 meses,
    para tasas/niveles como UNRATE/PAYEMS) o 'nivel' (serie ya calculada).
    """
    s = _get(raw, nombre)
    if s is None:
        return None
    if transform == 'yoy':
        t = yoy(s)
    elif transform == 'diff12':
        t = s.diff(12)
    elif transform == 'nivel':
        t = s
    else:
        raise ValueError(f'transform desconocido: {transform!r}')
    publicado = lag_publicacion(t, lag_months) if lag_months else t
    return causal_zscore(publicado).rename(out_name)


def mercado_z(raw: RawDict, nombre: str, out_name: str) -> pd.Series | None:
    """Feature mensual de 'mercado': z-score causal directo del NIVEL.

    No aplica lag por sí misma: en `construir_mensuales` la entrada ya llega retrasada
    según `LAG_PUBLICACION` (ADR-003: GS10/TB3MS/GS1/GS5/FEDFUNDS/MOODYS_*/BAAFFM/CAPE/
    GW son medias o cierres del mes M fechados el día 1 de M -> +1 mes).
    """
    s = _get(raw, nombre)
    return causal_zscore(s).rename(out_name) if s is not None else None


def mercado_spread_z(raw: RawDict, a: str, b: str, out_stub: str) -> pd.Series | None:
    """z-score causal del spread de NIVELES a - b (misma regla que `mercado_z`)."""
    sa, sb = _get(raw, a), _get(raw, b)
    if sa is None or sb is None:
        return None
    spread = (sa - sb).rename(out_stub)
    return causal_zscore(spread).rename(f'{out_stub}_z')


def fed_stance(raw: RawDict) -> pd.Series | None:
    """Postura monetaria: media de z(FEDFUNDS) y z(TBILL3M_MINUS_FEDFUNDS); exige ambas
    piernas presentes (skipna=False)."""
    ff = _get(raw, 'FEDFUNDS')
    spr = _get(raw, 'TBILL3M_MINUS_FEDFUNDS')
    if ff is None or spr is None:
        return None
    z1 = causal_zscore(ff)
    z2 = causal_zscore(spr)
    combinado = pd.concat({'ff_z': z1, 'spr_z': z2}, axis=1, sort=True).mean(axis=1, skipna=False)
    return combinado.rename('fed_stance_z')
