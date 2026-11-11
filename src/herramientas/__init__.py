"""Contrato de tools para los agentes del TFM (ADR-012).

Las fases 3–5 (agentes, cartera, evaluación) importan solo de aquí, nunca de ``regimenes.*`` ni de
``ingesta.*`` directamente. Cada tool es tipada (pydantic), declara qué agentes la usan y recibe la
fecha de la decisión del sistema, no del LLM.

- ``contrato``  modelos comunes: ``Cita``, ``EntradaTool``, ``SalidaTool``.
- ``registro``  ``registrar``, ``catalogo``, ``json_schema``, ``invocar``.
- ``regimen``   ``get_regimen`` (Fase 1).

Pendientes (Fase 2): envolver ``ingesta.mercados_series.tools`` (``get_serie``, ``get_precios``) y,
con la memoria RAG, las tools de búsqueda.
"""

from herramientas import regimen  # noqa: F401  (registra get_regimen en el catálogo)
from herramientas.registro import catalogo, invocar, json_schema

__all__ = ["catalogo", "invocar", "json_schema"]
