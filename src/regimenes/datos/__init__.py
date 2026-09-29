"""Capa de datos: descarga dirigida por ``configs/catalog.yaml``, sin imputar.

(Antes ``src/ingest``.) Modulos: ``catalogo`` (carga del catalogo),
``descarga`` (``download_all``), ``fuentes`` (fetchers por fuente).
CLI: ``python -m regimenes.datos [--offline|--force --only A,B]``.

Alias de compatibilidad con los nombres antiguos de ``src.ingest``:
``sources`` = ``fuentes`` y ``download`` = ``descarga``.
"""
from regimenes.datos import catalogo, fuentes
from regimenes.datos import descarga
from regimenes.datos.catalogo import load_catalog
from regimenes.datos.descarga import download_all

sources = fuentes
download = descarga

__all__ = ["catalogo", "descarga", "fuentes", "sources", "download",
           "download_all", "load_catalog"]
