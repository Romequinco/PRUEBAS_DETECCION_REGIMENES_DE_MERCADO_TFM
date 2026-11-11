"""Tipos comunes de las tools que consumen los agentes (ADR-012).

Toda tool recibe una ``EntradaTool`` (lo que elige el LLM) y la ``fecha`` de la decisión (la pone el
sistema, nunca el LLM: así ninguna tool puede mirar el futuro) y devuelve una ``SalidaTool`` con sus
citas. Los modelos prohíben campos extra para que un error de esquema falle en vez de colarse.
"""
from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class Modelo(BaseModel):
    """Base de todos los modelos del contrato: sin campos extra y valores inmutables."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Cita(Modelo):
    """De dónde sale un dato: lo que el memo final cita."""

    fuente: str = Field(description="Fichero, serie o documento de origen.")
    id: str | None = Field(default=None, description="Huella o identificador del origen (p. ej. sha256).")
    fecha_publicacion: date | None = Field(default=None, description="Última fecha visible del dato.")
    url: str | None = None
    fragmento: str | None = None


class EntradaTool(Modelo):
    """Argumentos que elige el LLM. Nunca incluye la fecha de la decisión."""


class SalidaTool(Modelo):
    """Respuesta común: qué tool, a qué fecha se consultó y con qué citas."""

    tool: str
    fecha_consulta: date
    citas: list[Cita] = Field(default_factory=list)
    avisos: list[str] = Field(default_factory=list)
