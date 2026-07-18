"""Capa de ingesta v2 — descarga dirigida por data/catalog.yaml, sin imputar."""
from . import sources, download
from .download import download_all, load_catalog

__all__ = ["sources", "download", "download_all", "load_catalog"]
