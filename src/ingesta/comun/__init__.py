"""Base común de la Fase 2. La usan los cuatro temas; se cambia solo con un PR pequeño a ``main``.

- ``esquema``  ``Documento`` (metadatos de todo documento o trozo, con ``fecha_publicacion``) y ``FichaFuente``.
- ``http``     ``get`` educado (User-Agent, ritmo por host, reintentos, claves enmascaradas) y ``robots_permite``.
- ``rutas``    ``resultados_dir(tema)`` (versionado) y ``corpus_dir(tema)`` (pesado, fuera del repo).
- ``fichas``   ``guardar_ficha`` / ``cargar_fichas`` en ``results/ingesta/<tema>/fichas/``.
"""
