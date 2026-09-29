"""paneles.py — Constructores de panel (tabla FEAT de 02 -> paneles de 03).

Sección 4 del antiguo ``src/features.py`` (ADR-004). Ver el docstring de
``regimenes.features.transformaciones`` para la organización completa.
"""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from regimenes.features.lags import aplicar_lag_publicacion
from regimenes.features.transformaciones import (
    BREADTH_11,
    SPDR_9,
    RawDict,
    _get,
    causal_zscore,
    change_z,
    cross_sectional_std_level,
    cross_sectional_std_ret,
    drawdown,
    fed_stance,
    log_returns,
    macro_z,
    mercado_spread_z,
    mercado_z,
    momentum,
    realized_vol,
    ret_z,
    rolling_correlation,
    spread_ret_z,
    zscore_or_none,
    zscore_raw,
)


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
