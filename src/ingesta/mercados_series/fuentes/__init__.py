"""Fuentes del Tema 1: un módulo por fuente (cargadores y extractores de series).

- ``fase1``       ``data/raw`` de la Fase 1 (solo lectura) → tabla larga del contrato de series
                  (``tabla_fase1``, ``resumen_fase1``; ficha ``fase1_raw``).
- ``fred``        API de FRED/ALFRED → sustitutos y componentes que no están en data/raw
                  (``ClienteFred``, ``tabla_fred_sustitutos``; fichas ``fred_alfred`` e ``ice_bofa_oas``).
- ``academicos``  Ken French, Shiller, Goyal-Welch, OFR FSI y JST (``tabla_academicos``, ``tabla_jst``,
                  ``descargar``; fichas ken_french, shiller, goyal_welch, ofr_fsi, jst_macrohistory).

``ingesta.mercados_series.base`` las une en la base de series (``construir_base``).
"""
