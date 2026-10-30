"""Features causales y paneles v2.

Submodulos: ``transformaciones`` (primitivas causales y recetario),
``lags`` (lags de publicacion, ADR-003), ``paneles`` (constructores de panel),
``causalidad`` (test de truncado), ``bloques`` (termometros por bloque). Este
``__init__`` re-exporta la API publica para que
``from regimenes import features as ft; ft.causal_zscore(...)`` siga funcionando. Dependencias en una sola direccion:
``transformaciones`` <- ``lags`` <- ``paneles``; ``causalidad`` solo usa ``RawDict``;
``bloques`` no depende de los demas (trabaja sobre paneles ya construidos).
"""

from regimenes.features import bloques, causalidad, lags, paneles, transformaciones
from regimenes.features.bloques import (
    BLOQUES_YAML,
    cargar_bloques,
    comparar_umbrales,
    construir_termometros,
    fechas_reajuste,
    parametros_bloques,
    resumen_estabilidad,
    termometro_bloque,
)
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
    "transformaciones", "lags", "paneles", "causalidad", "bloques",
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
    # bloques (§6, termómetros por bloque)
    "BLOQUES_YAML", "fechas_reajuste", "termometro_bloque", "construir_termometros",
    "resumen_estabilidad", "comparar_umbrales", "cargar_bloques", "parametros_bloques",
]
