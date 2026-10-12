"""Tema 1 · Mercados y series numéricas (get_precios, get_serie).

Mapa de módulos:

- ``contrato``     columnas y dtypes de series (tabla larga con vintages) y de precios (as-traded),
                   ``validar_series`` / ``validar_precios``, ``ErrorContrato`` y la regla del
                   TimeGate (``primera_decision_vec``: decisión a las 16:30 America/New_York).
- ``visibilidad``  cuándo es público un dato: cierres de mercado (``config/mercados.yaml``) y
                   publicaciones (``config/publicaciones.yaml``) → ``disponible_utc`` y
                   ``fecha_publicacion``.
- ``timegate``     ``as_of_series``, ``visibles_precios`` y ``ajustar_a_fecha`` (splits y
                   dividendos conocidos en la fecha); ``des_ajustar_yahoo`` (vista de Yahoo →
                   as-traded).
- ``divisas``      subunidades (GBp, ZAc, ILA) y conversión a USD al leer con el FX del BCE.
- ``almacen``      parquet bajo ``corpus_dir('mercados_series')``; series y precios solo crecen.
- ``tools``        ``get_serie`` y ``get_precios`` para los agentes.
- ``sinteticos``   datos deterministas sin red para tests y notebooks.
- ``catalogo``     catálogo de series (``config/catalogo_series.yaml`` + ``config/lags.yaml``): una
                   entrada por serie, alias de la Fase 1 y ``fecha_publicacion_lag``.
- ``sonda_horas``  sondas de horas de publicación (ALFRED, releases de FRED, páginas oficiales).
- ``base``         base de series (bloque 2): une ``fuentes.fase1``, ``fuentes.fred`` y
                   ``fuentes.academicos``; CLI ``python -m ingesta.mercados_series construir-base``.

Un módulo por fuente (extractores). Alcance, fuentes y entregables: README.md de esta carpeta.
"""
