"""Features causales y paneles v2 (antes ``src/features.py``).

Submodulos: ``transformaciones`` (primitivas causales y recetario),
``lags`` (lags de publicacion, ADR-003), ``paneles`` (constructores de panel),
``causalidad`` (test de truncado). Este ``__init__`` re-exporta la API publica
para que ``from regimenes import features as ft; ft.causal_zscore(...)`` siga
funcionando. Dependencias en una sola direccion:
``transformaciones`` <- ``lags`` <- ``paneles``; ``causalidad`` solo usa ``RawDict``.
"""

from regimenes.features import causalidad, lags, paneles, transformaciones
from regimenes.features.causalidad import assert_causal, truncar, truncation_max_abs_diff
from regimenes.features.lags import (
    LAG_PUBLICACION,
    aplicar_lag_publicacion,
    lag_de,
    tabla_lags_publicacion,
)
from regimenes.features.paneles import (
    GW_COLUMN,
    MENSUAL_MACRO,
    MENSUAL_MERCADO,
    N_FEAT_DIARIAS,
    N_FEAT_MENSUALES,
    PANELES_MULTICOLUMNA,
    alinear_diaria,
    alinear_mensual_con_edad,
    construir_diarias,
    construir_mensuales,
    construir_paneles,
    ff_industry_dispersion,
    rejilla_nyse,
    seleccionar_raw,
)
from regimenes.features.transformaciones import (
    BREADTH_11,
    SPDR_9,
    TRADING_DAYS,
    RawDict,
    causal_zscore,
    change_z,
    cross_sectional_std_level,
    cross_sectional_std_ret,
    drawdown,
    fed_stance,
    lag_publicacion,
    log_returns,
    logret,
    macro_z,
    mercado_spread_z,
    mercado_z,
    momentum,
    realized_vol,
    ret_z,
    rolling_correlation,
    spread_ret_z,
    yoy,
    zscore_or_none,
    zscore_raw,
)

__all__ = [
    "transformaciones", "lags", "paneles", "causalidad",
    # transformaciones (§1 primitivas causales + §2 recetario v2)
    "TRADING_DAYS", "RawDict", "BREADTH_11", "SPDR_9",
    "causal_zscore", "log_returns", "realized_vol", "rolling_correlation", "drawdown",
    "momentum", "logret", "ret_z", "zscore_raw", "change_z", "spread_ret_z",
    "cross_sectional_std_ret", "cross_sectional_std_level", "zscore_or_none",
    "lag_publicacion", "yoy", "macro_z", "mercado_z", "mercado_spread_z", "fed_stance",
    # lags (§3, ADR-003)
    "LAG_PUBLICACION", "lag_de", "aplicar_lag_publicacion", "tabla_lags_publicacion",
    # paneles (§4)
    "N_FEAT_DIARIAS", "N_FEAT_MENSUALES", "MENSUAL_MERCADO", "MENSUAL_MACRO",
    "PANELES_MULTICOLUMNA", "GW_COLUMN", "seleccionar_raw", "ff_industry_dispersion",
    "construir_diarias", "construir_mensuales", "alinear_diaria", "alinear_mensual_con_edad",
    "rejilla_nyse", "construir_paneles",
    # causalidad (§5)
    "truncar", "truncation_max_abs_diff", "assert_causal",
]
