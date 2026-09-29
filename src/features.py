"""
features.py — Features CAUSALES para los detectores de régimen (preprocesado v2).

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

from typing import Callable, Mapping

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
    return pd.concat(rets, axis=1).std(axis=1, ddof=1)


def cross_sectional_std_level(raw: RawDict, nombres: list[str]) -> pd.Series | None:
    """Como `cross_sectional_std_ret` pero sobre NIVELES (índices de vol implícita)."""
    lv = {n: _get(raw, n) for n in nombres if _get(raw, n) is not None}
    if len(lv) < 2:
        return None
    return pd.concat(lv, axis=1).std(axis=1, ddof=1)


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
    combinado = pd.concat({'ff_z': z1, 'spr_z': z2}, axis=1).mean(axis=1, skipna=False)
    return combinado.rename('fed_stance_z')


# --------------------------------------------------------------------------- #
# 3. Causalidad de calendario: lags de publicación por serie (ADR-003)
# --------------------------------------------------------------------------- #
# FUENTE ÚNICA de verdad. Convención de fecha de entrada = la de la fuente:
#   - mensual FRED/Shiller/GW: día 1 del mes de REFERENCIA M;
#   - trimestral FRED: día 1 del trimestre de referencia;
#   - semanal ICSA: sábado de fin de la semana de referencia.
# `meses`/`dias`: desplazamiento hasta la fecha desde la que el dato es PÚBLICO (se
# redondea hacia el lado conservador: nunca antes de la publicación típica).
# `tipo`: 'media_mes' (media/cierre del mes M -> visible al cerrar M), 'macro'
# (publicación oficial en M+1), 'encuesta_en_mes' (publicada dentro del propio M),
# 'trimestral', 'semanal'. Las series diarias de mercado (cierre en t) NO llevan entrada.
LAG_PUBLICACION: dict[str, dict] = {
    # --- media (o cierre) del mes M, fechada el 1 de M -> visible al cerrar M: +1 mes ---
    'GS10': dict(meses=1, dias=0, tipo='media_mes', fuente='FRED/H.15: media mensual de DGS10 (02 §3.2: MAE 0,003 pp vs media de M)'),
    'GS5': dict(meses=1, dias=0, tipo='media_mes', fuente='FRED/H.15: media mensual de DGS5'),
    'GS1': dict(meses=1, dias=0, tipo='media_mes', fuente='FRED/H.15: media mensual de DGS1'),
    'TB3MS': dict(meses=1, dias=0, tipo='media_mes', fuente='FRED/H.15: media mensual del T-bill 3m (mercado secundario)'),
    'FEDFUNDS': dict(meses=1, dias=0, tipo='media_mes', fuente='FRED/H.15: media mensual de la EFFR diaria'),
    'TBILL3M_MINUS_FEDFUNDS': dict(meses=1, dias=0, tipo='media_mes', fuente='FRED TB3SMFFM = TB3MS - FEDFUNDS (medias mensuales)'),
    'MOODYS_BAA': dict(meses=1, dias=0, tipo='media_mes', fuente="FRED BAA: 'averages of daily data' (Moody's)"),
    'MOODYS_AAA': dict(meses=1, dias=0, tipo='media_mes', fuente="FRED AAA: 'averages of daily data' (Moody's)"),
    'BAAFFM': dict(meses=1, dias=0, tipo='media_mes', fuente='FRED BAAFFM = BAA - FEDFUNDS (medias mensuales)'),
    'SHILLER_SP500_CAPE': dict(meses=1, dias=0, tipo='media_mes', fuente='Shiller ie_data.xls: P = media de cierres diarios de M; E10 con los últimos beneficios estimados/interpolados (peso 1/120)'),
    'GW_PREDICTORS_MONTHLY': dict(meses=1, dias=0, tipo='media_mes', fuente='Goyal-Welch b/m: valor contable del año previo / precio de FIN de M (fechado el 1 de M)'),
    'WTI_SPOT_MONTHLY': dict(meses=1, dias=10, tipo='media_mes', fuente='FRED WTISPLC: media mensual del spot diario; el spot diario de la EIA se publica con ~1 semana de retraso'),
    # --- macro oficial publicada a mitad de M+1 (o con riesgo de retraso) -> día 1 de M+2 ---
    'INDPRO': dict(meses=2, dias=0, tipo='macro', fuente='Fed G.17: ~día 15-17 de M+1'),
    'PPI_ALL_COMMODITIES': dict(meses=2, dias=0, tipo='macro', fuente='BLS PPI: ~día 11-15 de M+1'),
    'PPI_FUELS': dict(meses=2, dias=0, tipo='macro', fuente='BLS PPI: ~día 11-15 de M+1'),
    'PPI_METALS': dict(meses=2, dias=0, tipo='macro', fuente='BLS PPI: ~día 11-15 de M+1'),
    'CPIAUCSL': dict(meses=2, dias=0, tipo='macro', fuente='BLS CPI: ~día 10-15 de M+1'),
    'CPILFESL': dict(meses=2, dias=0, tipo='macro', fuente='BLS CPI: ~día 10-15 de M+1'),
    'UNRATE': dict(meses=2, dias=0, tipo='macro', fuente='BLS Employment Situation: 1er viernes de M+1, a veces día 8-10 o más tarde (cierres de gobierno) -> +1 mes filtraría ~1 semana; se redondea a M+2'),
    'PAYEMS': dict(meses=2, dias=0, tipo='macro', fuente='BLS Employment Situation: idem UNRATE'),
    'MANEMP': dict(meses=2, dias=0, tipo='macro', fuente='BLS Employment Situation: idem UNRATE'),
    'HOUST': dict(meses=2, dias=0, tipo='macro', fuente='Census New Residential Construction: ~día 17-19 de M+1'),
    'CFNAI': dict(meses=2, dias=0, tipo='macro', fuente='Chicago Fed: ~día 20-25 de M+1'),
    'CFNAIMA3': dict(meses=2, dias=0, tipo='macro', fuente='Chicago Fed: ~día 20-25 de M+1'),
    'RNUSBIS': dict(meses=2, dias=0, tipo='macro', fuente='BIS effective exchange rates (media mensual): publicados ~mitad de M+1'),
    'NNUSBIS': dict(meses=2, dias=0, tipo='macro', fuente='BIS effective exchange rates (media mensual): publicados ~mitad de M+1'),
    # --- encuestas publicadas DENTRO del propio mes M -> +1 mes ya es conservador ---
    'UMCSENT': dict(meses=1, dias=0, tipo='encuesta_en_mes', fuente='U. Michigan: preliminar ~2º viernes de M, final ~último viernes de M (FRED guarda el final)'),
    'PMI_PROXY_PHILLY': dict(meses=1, dias=0, tipo='encuesta_en_mes', fuente='Philadelphia Fed Manufacturing Business Outlook Survey: 3er jueves de M'),
    # --- trimestral, fechado al inicio del trimestre -> avance BEA ~fin del mes siguiente al trimestre ---
    'GDP_GROWTH_QOQ': dict(meses=4, dias=0, tipo='trimestral', fuente='BEA avance: ~día 25-30 del mes siguiente al cierre del trimestre = inicio + 1 trimestre + 1 mes. Excepciones por cierre de gobierno: Q3-2013 (7-nov), Q4-2018 (28-feb-2019), Q3-2025 (23-dic)'),
    'GDPC1': dict(meses=4, dias=0, tipo='trimestral', fuente='BEA avance (idem GDP_GROWTH_QOQ); insumo solo-raw'),
    # --- semanal, fechado el sábado de fin de semana de referencia ---
    'ICSA': dict(meses=0, dias=5, tipo='semanal', fuente='DOL Unemployment Insurance Weekly Claims: jueves siguiente al sábado de referencia (8:30 ET, antes de la apertura)'),
    # --- validación (NUNCA feature; se documentan por completitud, no se aplican aquí) ---
    'NFCI': dict(meses=0, dias=5, tipo='semanal', fuente='Chicago Fed NFCI: semana que acaba el viernes, publicado el miércoles siguiente (rol=validation)'),
    'STLFSI4': dict(meses=0, dias=6, tipo='semanal', fuente='St. Louis Fed FSI: semana que acaba el viernes, publicado el jueves siguiente (rol=validation)'),
}


def lag_de(nombre: str) -> pd.DateOffset | None:
    """DateOffset de publicación de una serie cruda (None si es diaria de mercado)."""
    e = LAG_PUBLICACION.get(nombre)
    return None if e is None else pd.DateOffset(months=e['meses'], days=e['dias'])


def aplicar_lag_publicacion(raw: RawDict, lags: Mapping[str, dict] | None = None) -> dict:
    """Copia de `raw` con el ÍNDICE de cada serie de `LAG_PUBLICACION` desplazado a su
    fecha de disponibilidad. Las series sin entrada (diarias de mercado) no cambian.

    Espera series con su fecha de FUENTE (tal como vienen de data/raw). No es
    idempotente: aplicarla dos veces duplica el lag (los constructores la llaman una
    sola vez sobre la entrada cruda). `lags` permite otra tabla (p. ej. reconstruir la
    regla v1 en 03 para medir el impacto de ADR-003); por defecto `LAG_PUBLICACION`."""
    lags = LAG_PUBLICACION if lags is None else lags
    out = {}
    for n, s in raw.items():
        e = lags.get(n)
        out[n] = s if e is None else lag_publicacion(s, e['meses'], e['dias'])
    return out


def tabla_lags_publicacion() -> pd.DataFrame:
    """`LAG_PUBLICACION` como tabla (serie, meses, días, tipo, fuente) para 02/03/_meta."""
    df = pd.DataFrame.from_dict(LAG_PUBLICACION, orient='index')
    df.index.name = 'serie'
    return df.reset_index()


# --------------------------------------------------------------------------- #
# 4. Constructores de panel (tabla FEAT de 02 -> paneles de 03)
# --------------------------------------------------------------------------- #
N_FEAT_DIARIAS = 49
N_FEAT_MENSUALES = 28

# Clasificación de las 28 filas mensuales de FEAT por NATURALEZA del dato. Desde ADR-003
# ambas familias llevan lag de publicación (tabla LAG_PUBLICACION): +1 mes las 'mercado'
# (medias/cierres de mes), +1/+2/+4 meses las 'macro' según su calendario de publicación.
MENSUAL_MERCADO = ['credit_BaaAaa_mensual_z', 'credit_BaaFF_z', 'TB3MS_z', 'GS10_z',
                   'term_spread_hist_z', 'GS1_z', 'GS5_z', 'fed_stance_z', 'CAPE_z', 'GW_valuation_z']
MENSUAL_MACRO = ['INDPRO_yoy_z', 'PPI_yoy_z', 'PPI_fuels_yoy_z', 'PPI_metals_yoy_z', 'UNRATE_chg_z',
                 'CPI_yoy_z', 'CPI_core_yoy_z', 'PAYEMS_chg_z', 'CFNAI_z', 'CFNAIMA3_z', 'PMI_proxy_z',
                 'UMCSENT_z', 'HOUST_yoy_z', 'MANEMP_yoy_z', 'GDP_growth_z', 'WTI_yoy_z',
                 'broad_dollar_real_z', 'broad_dollar_nominal_z']

# Series crudas multi-columna (ADR-003). En disco llevan además una columna alias
# `<nombre_interno>` (= la PRIMERA de esta lista) para no romper `read_parquet(...)[nombre]`.
PANELES_MULTICOLUMNA = {
    'FF_FACTORS_3_DAILY': ['Mkt-RF', 'SMB', 'HML', 'RF'],
    'FF_FACTORS_5_DAILY': ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF'],
    'FF_5_INDUSTRY': ['Cnsmr', 'Manuf', 'HiTec', 'Hlth', 'Other'],
}
GW_COLUMN = 'b/m'  # columna de GW_PREDICTORS_MONTHLY (fijada en 03 §1)


def seleccionar_raw(df: pd.DataFrame, nombre: str) -> pd.Series | pd.DataFrame:
    """Convención de carga ÚNICA (03 y tests): de `pd.read_parquet(data/raw/.../nombre)`
    a la entrada del diccionario crudo de una pista.

    - `GW_PREDICTORS_MONTHLY` -> Series de la columna `b/m` (nombre `..._b/m`).
    - `PANELES_MULTICOLUMNA` -> DataFrame con sus columnas originales (sin el alias); si
      el parquet es del formato antiguo (solo el alias), DataFrame de esa única columna.
    - resto -> Series de la columna homónima.
    """
    if nombre == 'GW_PREDICTORS_MONTHLY':
        return df[GW_COLUMN].sort_index().rename(f'{nombre}_{GW_COLUMN}')
    if nombre in PANELES_MULTICOLUMNA:
        cols = [c for c in PANELES_MULTICOLUMNA[nombre] if c in df.columns]
        return (df[cols] if cols else df[[nombre]]).sort_index()
    return df[nombre].sort_index()


def _col(raw: RawDict, nombre: str, col: str) -> pd.Series | None:
    """Columna `col` de una serie multi-columna de `raw`. Si la pista trae la serie como
    Series (formato antiguo / tests sintéticos) solo se acepta la columna PRIMARIA."""
    x = raw.get(nombre)
    if x is None:
        return None
    primaria = PANELES_MULTICOLUMNA.get(nombre, [None])[0]
    if isinstance(x, pd.Series):
        return x if col == primaria else None
    if col in x.columns:
        return x[col]
    if col == primaria and nombre in x.columns:
        return x[nombre]
    return None


def _zscore_col(raw: RawDict, nombre: str, col: str, out_name: str) -> pd.Series | None:
    s = _col(raw, nombre, col)
    return causal_zscore(s.dropna()).rename(out_name) if s is not None else None


def ff_industry_dispersion(raw: RawDict) -> pd.Series | None:
    """Dispersión cross-seccional (std, ddof=1) de los retornos diarios (%) de las 5
    carteras industriales de Ken French, fila a fila. None si no hay >= 2 carteras."""
    x = raw.get('FF_5_INDUSTRY')
    if not isinstance(x, pd.DataFrame):
        return None
    cols = [c for c in PANELES_MULTICOLUMNA['FF_5_INDUSTRY'] if c in x.columns]
    if len(cols) < 2:
        return None
    return x[cols].std(axis=1, ddof=1).dropna().rename('FF_industry_dispersion')


def _ensamblar(pares: list[tuple[str, pd.Series | None]]) -> tuple[pd.DataFrame, list[str]]:
    out, omitidas = {}, []
    for nombre, serie in pares:
        if serie is None or serie.dropna().empty:
            omitidas.append(nombre)
        else:
            out[nombre] = serie.rename(nombre)
    df = (pd.concat(out.values(), axis=1).sort_index() if out
          else pd.DataFrame(index=pd.DatetimeIndex([])))
    return df, omitidas


def construir_diarias(raw: RawDict, lags: Mapping[str, dict] | None = None
                      ) -> tuple[pd.DataFrame, list[str]]:
    """Las 49 filas `frecuencia=='diaria'` de FEAT, sobre el calendario NATIVO de cada
    fuente (sin alinear). Devuelve (panel, omitidas): una feature se omite si la pista no
    tiene su(s) serie(s) fuente o si sale vacía. `raw` lleva fechas de FUENTE; aquí se
    aplica `LAG_PUBLICACION` (solo afecta a ICSA entre las diarias: +5 días).

    Notas:
    - `SP500_vol_z` se calcula desde SP500 (misma fórmula con la que la ingesta deriva
      `REALIZED_VOL_SP500` desde ADR-003; antes ese parquet contenía el precio).
    - `FF_industry_dispersion_z` = z de la std cross-seccional de las 5 carteras
      industriales (ADR-003; antes era el z de una sola cartera).
    - `FF_RMW_z` (ADR-003) sustituye a `FF_MKT5_z`, que duplicaba `FF_MKT_z` (mismo
      Mkt-RF): factor rentabilidad/calidad RMW del fichero de 5 factores.
    - `ICSA_z`: fechada el jueves de publicación (sábado + 5 días).
    """
    raw = aplicar_lag_publicacion(raw, lags)
    sp500 = _get(raw, 'SP500')
    p: list[tuple[str, pd.Series | None]] = []

    # ESPINA EQUITY
    p.append(('SP500_ret_z', ret_z(raw, 'SP500', 'SP500_ret_z')))
    if sp500 is not None:
        rv = realized_vol(log_returns(sp500), window=21, annualize=True)
        p.append(('SP500_vol_z', causal_zscore(rv)))
    else:
        p.append(('SP500_vol_z', None))
    p.append(('SP500_momentum', momentum(sp500) if sp500 is not None else None))
    p.append(('SP500_drawdown', drawdown(sp500) if sp500 is not None else None))
    p.append(('SP500_megacap_gap_z', spread_ret_z(raw, 'SP500', 'SP500_EW', 'SP500_megacap_gap')))
    p.append(('equity_breadth_dispersion_z', zscore_or_none(cross_sectional_std_ret(raw, BREADTH_11))))
    p.append(('smallcap_largecap_spread_z', spread_ret_z(raw, 'RUSSELL2000', 'SP500', 'smallcap_largecap_spread')))
    p.append(('style_spread_z', spread_ret_z(raw, 'IWF_GROWTH', 'IWD_VALUE', 'style_spread')))

    # FACTORES FAMA-FRENCH (ya vienen como retorno en %)
    p.append(('FF_MKT_z', _zscore_col(raw, 'FF_FACTORS_3_DAILY', 'Mkt-RF', 'FF_MKT_z')))
    p.append(('FF_MOM_z', zscore_raw(raw, 'FF_MOM_DAILY', 'FF_MOM_z')))
    p.append(('FF_industry_dispersion_z', zscore_or_none(ff_industry_dispersion(raw))))
    p.append(('FF_RMW_z', _zscore_col(raw, 'FF_FACTORS_5_DAILY', 'RMW', 'FF_RMW_z')))

    # CREDITO (diarias)
    p.append(('credit_BAA10Y_z', zscore_raw(raw, 'BAA10Y', 'credit_BAA10Y_z')))
    p.append(('credit_BaaAaa_diaria_z', zscore_raw(raw, 'MOODYS_BAA_AAA_SPREAD', 'credit_BaaAaa_diaria_z')))
    p.append(('credit_HYG_IEF_z', spread_ret_z(raw, 'HYG_CREDIT', 'IEF_TREASURY', 'credit_HYG_IEF')))
    p.append(('credit_HYG_LQD_z', spread_ret_z(raw, 'HYG_CREDIT', 'LQD_IGCREDIT', 'credit_HYG_LQD')))

    # TIPOS: NIVEL Y CURVA DIARIA
    p.append(('mid_curve_z', zscore_raw(raw, 'DGS5', 'mid_curve_z')))
    p.append(('DGS10_change_z', change_z(raw, 'DGS10', 'DGS10_change_z')))
    p.append(('curve_vs_policy_10y_z', zscore_raw(raw, 'T10YFF', 'curve_vs_policy_10y_z')))
    p.append(('curve_vs_policy_5y_z', zscore_raw(raw, 'T5YFF', 'curve_vs_policy_5y_z')))
    p.append(('slope_10y2y_z', zscore_raw(raw, 'T10Y2Y', 'slope_10y2y_z')))
    p.append(('slope_10y3m_z', zscore_raw(raw, 'T10Y3M', 'slope_10y3m_z')))
    dgs10, dgs2, dgs30 = _get(raw, 'DGS10'), _get(raw, 'DGS2'), _get(raw, 'DGS30')
    if dgs10 is not None and dgs2 is not None and dgs30 is not None:
        curv = pd.concat({'DGS10': dgs10, 'DGS2': dgs2, 'DGS30': dgs30}, axis=1)
        p.append(('curve_curvature_z', causal_zscore(2 * curv['DGS10'] - curv['DGS2'] - curv['DGS30'])))
    else:
        p.append(('curve_curvature_z', None))
    dtb3, dff = _get(raw, 'DTB3'), _get(raw, 'DFF')
    if dtb3 is not None and dff is not None:
        # Nota: DFF se publica también en fines de semana; en esas fechas la media es
        # solo DFF (skipna). No es look-ahead, pero mezcla composiciones.
        short_rate = pd.concat({'DTB3': dtb3, 'DFF': dff}, axis=1).mean(axis=1)
        p.append(('short_rate_z', causal_zscore(short_rate)))
    else:
        p.append(('short_rate_z', None))
    p.append(('EFFR_z', zscore_raw(raw, 'EFFR', 'EFFR_z')))

    # VELOCIDAD DE TIPOS REALES / BREAKEVENS
    p.append(('DFII10_change_z', change_z(raw, 'DFII10', 'DFII10_change_z')))
    p.append(('breakeven_10y_z', zscore_raw(raw, 'T10YIE', 'breakeven_10y_z')))
    p.append(('breakeven_5y_z', zscore_raw(raw, 'T5YIE', 'breakeven_5y_z')))
    p.append(('breakeven_5y_fwd_z', zscore_raw(raw, 'T5YIFR', 'breakeven_5y_fwd_z')))
    p.append(('DFII5_z', zscore_raw(raw, 'DFII5', 'DFII5_z')))

    # COMPLEJO DE VOLATILIDAD
    p.append(('VIX_level_z', zscore_raw(raw, 'VIX', 'VIX_level_z')))
    p.append(('VIX_change_z', change_z(raw, 'VIX', 'VIX_change_z')))
    p.append(('MOVE_level_z', zscore_raw(raw, 'MOVE', 'MOVE_level_z')))
    p.append(('MOVE_change_z', change_z(raw, 'MOVE', 'MOVE_change_z')))
    p.append(('VVIX_z', zscore_raw(raw, 'VVIX', 'VVIX_z')))
    vix3m, vix = _get(raw, 'VIX3M'), _get(raw, 'VIX')
    if vix3m is not None and vix is not None:
        term = pd.concat({'VIX3M': vix3m, 'VIX': vix}, axis=1)
        p.append(('vix_term_z', causal_zscore(term['VIX3M'] / term['VIX'])))
    else:
        p.append(('vix_term_z', None))
    p.append(('vol_complex_dispersion_z', zscore_or_none(cross_sectional_std_level(raw, ['RVX', 'VXD', 'VXN']))))
    p.append(('SKEW_z', zscore_raw(raw, 'SKEW', 'SKEW_z')))

    # SECTORES
    p.append(('sector_dispersion', cross_sectional_std_ret(raw, SPDR_9)))  # sin z: asi lo fija FEAT
    p.append(('cyc_def_ratio_z', spread_ret_z(raw, 'SPDR_XLY', 'SPDR_XLP', 'cyc_def_ratio')))

    # FX / COMMODITIES / REFUGIO
    p.append(('DXY_change_z', ret_z(raw, 'DXY', 'DXY_change_z')))
    p.append(('JPY_ret_z', ret_z(raw, 'DEXJPUS', 'JPY_ret_z')))
    p.append(('CHF_ret_z', ret_z(raw, 'DEXSZUS', 'CHF_ret_z')))
    p.append(('AUD_ret_z', ret_z(raw, 'DEXUSAL', 'AUD_ret_z')))
    p.append(('GOLD_ret_z', ret_z(raw, 'GOLD_GLD', 'GOLD_ret_z')))
    p.append(('GSCI_ret_z', ret_z(raw, 'SPGSCI', 'GSCI_ret_z')))
    p.append(('BCOM_ret_z', ret_z(raw, 'BCOM', 'BCOM_ret_z')))

    # ACTIVIDAD (semanal en origen; fechada el jueves de publicación por LAG_PUBLICACION)
    p.append(('ICSA_z', zscore_raw(raw, 'ICSA', 'ICSA_z')))

    # TIPIFICACION DE REGIMEN
    spx, tlt = _get(raw, 'SP500'), _get(raw, 'TLT_TREASURY')
    if spx is not None and tlt is not None:
        p.append(('corr_spx_bond', rolling_correlation(log_returns(spx), log_returns(tlt), window=60)))
    else:
        p.append(('corr_spx_bond', None))

    assert len(p) == N_FEAT_DIARIAS
    return _ensamblar(p)


def construir_mensuales(raw: RawDict, lags: Mapping[str, dict] | None = None
                        ) -> tuple[pd.DataFrame, list[str]]:
    """Las 28 filas `frecuencia=='mensual'` de FEAT en su frecuencia NATIVA (sin alinear).

    ADR-003: la entrada `raw` lleva fechas de FUENTE; aquí se aplica
    `aplicar_lag_publicacion` ANTES de cualquier transformación, así que el índice de
    cada feature es la fecha desde la que el dato es público: 'mercado'
    (MENSUAL_MERCADO, medias/cierres del mes) +1 mes; 'macro' (MENSUAL_MACRO) +1/+2/+4
    meses según su calendario (ver `LAG_PUBLICACION`). Después: transformación nativa
    (yoy/diff12/nivel/spread) -> z-score causal.
    Salvedad: son la última vintage de FRED (revisada), no la publicada en tiempo real.
    """
    raw = aplicar_lag_publicacion(raw, lags)
    p: list[tuple[str, pd.Series | None]] = [
        # CREDITO (media del mes, +1 mes)
        ('credit_BaaAaa_mensual_z', mercado_spread_z(raw, 'MOODYS_BAA', 'MOODYS_AAA', 'credit_BaaAaa_mensual')),
        ('credit_BaaFF_z', mercado_z(raw, 'BAAFFM', 'credit_BaaFF_z')),
        # TIPOS HISTORICOS (media del mes, +1 mes)
        ('TB3MS_z', mercado_z(raw, 'TB3MS', 'TB3MS_z')),
        ('GS10_z', mercado_z(raw, 'GS10', 'GS10_z')),
        ('term_spread_hist_z', mercado_spread_z(raw, 'GS10', 'TB3MS', 'term_spread_hist')),
        ('GS1_z', mercado_z(raw, 'GS1', 'GS1_z')),
        ('GS5_z', mercado_z(raw, 'GS5', 'GS5_z')),
        ('fed_stance_z', fed_stance(raw)),
        # MACRO REAL (lag de publicación por serie: LAG_PUBLICACION)
        ('INDPRO_yoy_z', macro_z(raw, 'INDPRO', 'INDPRO_yoy_z', 'yoy')),
        ('PPI_yoy_z', macro_z(raw, 'PPI_ALL_COMMODITIES', 'PPI_yoy_z', 'yoy')),
        ('PPI_fuels_yoy_z', macro_z(raw, 'PPI_FUELS', 'PPI_fuels_yoy_z', 'yoy')),
        ('PPI_metals_yoy_z', macro_z(raw, 'PPI_METALS', 'PPI_metals_yoy_z', 'yoy')),
        ('UNRATE_chg_z', macro_z(raw, 'UNRATE', 'UNRATE_chg_z', 'diff12')),
        ('CPI_yoy_z', macro_z(raw, 'CPIAUCSL', 'CPI_yoy_z', 'yoy')),
        ('CPI_core_yoy_z', macro_z(raw, 'CPILFESL', 'CPI_core_yoy_z', 'yoy')),
        ('PAYEMS_chg_z', macro_z(raw, 'PAYEMS', 'PAYEMS_chg_z', 'diff12')),
        ('CFNAI_z', macro_z(raw, 'CFNAI', 'CFNAI_z', 'nivel')),
        ('CFNAIMA3_z', macro_z(raw, 'CFNAIMA3', 'CFNAIMA3_z', 'nivel')),
        ('PMI_proxy_z', macro_z(raw, 'PMI_PROXY_PHILLY', 'PMI_proxy_z', 'nivel')),
        ('UMCSENT_z', macro_z(raw, 'UMCSENT', 'UMCSENT_z', 'nivel')),
        ('HOUST_yoy_z', macro_z(raw, 'HOUST', 'HOUST_yoy_z', 'yoy')),
        ('MANEMP_yoy_z', macro_z(raw, 'MANEMP', 'MANEMP_yoy_z', 'yoy')),
        ('GDP_growth_z', macro_z(raw, 'GDP_GROWTH_QOQ', 'GDP_growth_z', 'nivel')),
        ('WTI_yoy_z', macro_z(raw, 'WTI_SPOT_MONTHLY', 'WTI_yoy_z', 'yoy')),
        ('broad_dollar_real_z', macro_z(raw, 'RNUSBIS', 'broad_dollar_real_z', 'nivel')),
        ('broad_dollar_nominal_z', macro_z(raw, 'NNUSBIS', 'broad_dollar_nominal_z', 'nivel')),
        # VALORACION (PE10 de Shiller / b/m de Goyal-Welch, +1 mes)
        ('CAPE_z', mercado_z(raw, 'SHILLER_SP500_CAPE', 'CAPE_z')),
        ('GW_valuation_z', mercado_z(raw, 'GW_PREDICTORS_MONTHLY', 'GW_valuation_z')),
    ]
    assert len(p) == N_FEAT_MENSUALES
    assert {n for n, _ in p} == set(MENSUAL_MERCADO) | set(MENSUAL_MACRO)
    return _ensamblar(p)


def _asof(panel: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """As-of backward POR COLUMNA: en cada fecha de `grid`, el último valor no nulo
    <= t de cada columna (aunque su fecha no esté en `grid`). Causal."""
    return panel.sort_index().ffill().reindex(grid, method='ffill')


def alinear_diaria(panel: pd.DataFrame, grid: pd.DatetimeIndex,
                   asof_por_columna: bool = True) -> pd.DataFrame:
    """Alinea un panel de features DIARIAS (calendarios nativos concatenados) a `grid`.

    Por defecto (ADR-003) as-of backward por columna (`_asof`), causal.

    `asof_por_columna=False` reproduce la v1 (`panel.reindex(grid, method='ffill')
    .ffill()`), que tenía un BUG: cuando una fecha de `grid` YA existe como fila de
    `panel` (porque otra columna tiene dato ese día), `reindex` toma esa fila tal cual;
    si una columna solo tiene datos en fechas FUERA de `grid` (p. ej. ICSA, fechada en
    sábado), sus valores nunca llegan a la rejilla y la columna queda entera en NaN (y
    un dato de un festivo NYSE se pierde si la siguiente sesión ya tiene fila).
    """
    if asof_por_columna:
        return _asof(panel, grid)
    return panel.reindex(grid, method='ffill').ffill()


def alinear_mensual_con_edad(panel: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """Alinea un panel MENSUAL (ya con lag) a `grid` con as-of backward por columna y
    añade `<feature>_edad_dias`: días de calendario desde el último dato real (no
    arrastrado) de esa columna visible en cada fecha. Causal: solo fechas <= t.
    (ADR-003: as-of por columna porque, con lags en días, las fechas de disponibilidad ya
    no caen todas el día 1 y un `reindex` por filas podría perder valores.)"""
    alineado = _asof(panel, grid)
    grid_fecha = pd.Series(grid, index=grid)
    edad_cols = {}
    for col in panel.columns:
        real = panel[col].dropna()
        fecha_ultimo_dato = pd.Series(real.index, index=real.index).reindex(grid, method='ffill')
        edad_cols[f'{col}_edad_dias'] = (grid_fecha - fecha_ultimo_dato).dt.days
    edad_df = pd.DataFrame(edad_cols, index=grid)
    return pd.concat([alineado, edad_df], axis=1)


def rejilla_nyse(raw: RawDict, inicio: str, fin: str) -> pd.DatetimeIndex:
    """Rejilla de sesiones NYSE = índice de SP500 recortado a [inicio, fin]."""
    return raw['SP500'].loc[inicio:fin].index


def construir_paneles(raw: RawDict, inicio: str, fin: str, asof_por_columna: bool = True,
                      lags: Mapping[str, dict] | None = None) -> dict[str, pd.DataFrame]:
    """Pipeline completo de una pista: (lags de publicación) -> features nativas ->
    rejilla NYSE -> paneles. `raw` con fechas de FUENTE (tal cual data/raw).

    Devuelve {'diaria', 'mensual', 'omitidas_diaria', 'omitidas_mensual'} con los
    paneles alineados a la rejilla (exactamente lo que 03 escribe a disco, salvo el
    nombre del índice).
    """
    diaria, om_d = construir_diarias(raw, lags)
    mensual, om_m = construir_mensuales(raw, lags)
    grid = rejilla_nyse(raw, inicio, fin)
    return {
        'diaria': alinear_diaria(diaria, grid, asof_por_columna=asof_por_columna),
        'mensual': alinear_mensual_con_edad(mensual, grid),
        'omitidas_diaria': om_d,
        'omitidas_mensual': om_m,
    }


# --------------------------------------------------------------------------- #
# 5. Verificación de causalidad (test de truncado)
# --------------------------------------------------------------------------- #
def truncar(raw: RawDict, cut: str | pd.Timestamp) -> dict[str, pd.Series]:
    """Copia de `raw` con cada serie cortada en `cut` (inclusive, fecha de FUENTE)."""
    return {k: v.loc[:cut] for k, v in raw.items()}


def truncation_max_abs_diff(full: pd.Series | pd.DataFrame,
                            trunc: pd.Series | pd.DataFrame,
                            cut: str | pd.Timestamp) -> pd.Series:
    """max|full - trunc| por columna en las fechas comunes <= cut.

    Además de diferencias numéricas, cuenta como discrepancia (valor inf) que una
    celda sea NaN en una versión y no en la otra (un NaN 'que aparece' también es
    información del futuro alterando el pasado).
    """
    full_df = full.to_frame() if isinstance(full, pd.Series) else full
    trunc_df = trunc.to_frame() if isinstance(trunc, pd.Series) else trunc
    cut = pd.Timestamp(cut)
    idx = full_df.index.intersection(trunc_df.index)
    idx = idx[idx <= cut]
    a = full_df.loc[idx, trunc_df.columns]
    b = trunc_df.loc[idx]
    diff = (a - b).abs().max()
    nan_mismatch = (a.isna() != b.isna()).any()
    return diff.fillna(0.0).where(~nan_mismatch, np.inf)


def assert_causal(builder: Callable[[RawDict], pd.Series | pd.DataFrame | None],
                  raw: RawDict, cut: str = "2015-01-01", tol: float = 1e-9) -> pd.DataFrame:
    """Verifica que `builder(raw)` no usa información futura (test de truncado).

    Computa `builder` sobre la muestra completa y sobre la muestra truncada en `cut`;
    los valores <= cut deben coincidir. Devuelve [feature, max_abs_diff, causal_ok].
    """
    full = builder(raw)
    trunc = builder(truncar(raw, cut))
    if full is None or trunc is None:
        return pd.DataFrame(columns=['feature', 'max_abs_diff', 'causal_ok'])
    mad = truncation_max_abs_diff(full, trunc, cut)
    return pd.DataFrame({'feature': mad.index.astype(str), 'max_abs_diff': mad.values,
                         'causal_ok': (mad.values <= tol)})
