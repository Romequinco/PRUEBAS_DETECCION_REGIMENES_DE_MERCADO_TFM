"""ingesta: capa de datos de las fases 2-5 del TFM (agentes, RAG y tools).

Un subpaquete por tema, cada uno en su rama y con su responsable:

- ``mercados_series``  precios, curvas, macro y vintages → ``get_precios``, ``get_serie``.
- ``bancos_riesgo``    bancos centrales, organismos, estabilidad financiera, papers → ``buscar_macro``, ``buscar_riesgo``.
- ``empresas``         filings, fundamentales y sentimiento → ``buscar_filings``, ``get_fundamentales``.
- ``noticias_alt``     noticias y datos alternativos → ``buscar_noticias``, ``get_polymarket``…
- ``comun``            lo compartido por todos: esquema ``Documento``, cliente HTTP, rutas y fichas.

Reglas y reparto: ``docs/fase2/README.md``. Este paquete no toca ``regimenes`` (Fase 1) ni su huella de caché.
"""
