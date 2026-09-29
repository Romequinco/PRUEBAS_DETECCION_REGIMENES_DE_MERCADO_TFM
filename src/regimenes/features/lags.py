"""lags.py — Causalidad de calendario: lags de publicación por serie (ADR-003).

Sección 3 de la organización de ``regimenes.features``. Ver el docstring de
``regimenes.features.transformaciones`` para la organización completa.
"""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from regimenes.features.transformaciones import RawDict, lag_publicacion


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
