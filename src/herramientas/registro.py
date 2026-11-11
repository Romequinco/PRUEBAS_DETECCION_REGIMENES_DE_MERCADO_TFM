"""Catálogo de tools: registro, JSON schema para el LLM e invocación con fecha impuesta.

Uso::

    @registrar("get_regimen", EntradaRegimen, agentes=("macro", "risk", "portfolio"))
    def _regimen(entrada: EntradaRegimen, fecha: date) -> SalidaRegimen: ...

    esquema = json_schema("get_regimen")          # lo que ve el LLM (sin fecha)
    salida = invocar("get_regimen", {"pista": "B"}, fecha=date(2020, 3, 16))
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

from herramientas.contrato import EntradaTool, SalidaTool


@dataclass(frozen=True)
class Tool:
    nombre: str
    entrada: type[EntradaTool]
    funcion: Callable[[EntradaTool, date], SalidaTool]
    agentes: tuple[str, ...]
    descripcion: str


_CATALOGO: dict[str, Tool] = {}


def registrar(nombre: str, entrada: type[EntradaTool], *, agentes: tuple[str, ...] = ()):
    """Decorador: añade la función al catálogo con su modelo de entrada."""
    if "fecha" in entrada.model_fields:
        raise ValueError(f"{nombre}: la fecha la pone el sistema; no puede ser un argumento del LLM")

    def decorar(funcion: Callable[[EntradaTool, date], SalidaTool]):
        if nombre in _CATALOGO:
            raise ValueError(f"tool repetida: {nombre}")
        doc = (funcion.__doc__ or "").strip().splitlines()
        _CATALOGO[nombre] = Tool(nombre, entrada, funcion, tuple(agentes), doc[0] if doc else "")
        return funcion

    return decorar


def catalogo(agente: str | None = None) -> list[Tool]:
    """Tools registradas (todas, o las de un agente), por nombre."""
    return sorted((t for t in _CATALOGO.values() if agente is None or agente in t.agentes),
                  key=lambda t: t.nombre)


def json_schema(nombre: str) -> dict[str, Any]:
    """Esquema de la tool tal como lo ve el LLM: nombre, descripción y parámetros (sin fecha)."""
    t = _CATALOGO[nombre]
    return {"name": t.nombre, "description": t.descripcion, "parameters": t.entrada.model_json_schema()}


def invocar(nombre: str, argumentos: dict[str, Any], *, fecha: date) -> SalidaTool:
    """Valida los argumentos del LLM e invoca la tool con la ``fecha`` que fija el sistema."""
    t = _CATALOGO[nombre]
    return t.funcion(t.entrada.model_validate(argumentos), fecha)
