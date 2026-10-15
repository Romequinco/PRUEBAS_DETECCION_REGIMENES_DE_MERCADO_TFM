"""Fuentes del Tema 1: un módulo por fuente (cargadores y extractores de series).

- ``fase1``       ``data/raw`` de la Fase 1 (solo lectura) → tabla larga del contrato de series
                  (``tabla_fase1``, ``resumen_fase1``; ficha ``fase1_raw``).
- ``fred``        API de FRED como fuente viva de todas las series de FRED que no van por ALFRED
                  (``ClienteFred``, ``tabla_fred_vivo``, derivadas; fichas ``fred_alfred`` e ``ice_bofa_oas``).
- ``alfred``      vintages exactos de ALFRED de las series ``vintage_exacto`` (``descargar_vintages``,
                  ``tabla_alfred``, revisiones y regla previa; ficha ``alfred_vintages``).
- ``academicos``  Ken French, Shiller y Goyal-Welch con vintage por edición, OFR FSI y JST
                  (``tabla_academicos``, ``tabla_jst``, ``descargar``, ``archivar_ediciones``; fichas
                  ken_french, shiller, goyal_welch, ofr_fsi, jst_macrohistory).

``ingesta.mercados_series.base`` las une en la base de series (``construir_base``).
"""
