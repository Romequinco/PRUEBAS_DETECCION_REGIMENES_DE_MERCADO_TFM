"""Registro de generadores sinteticos (vacio hasta que exista el primero)."""

from __future__ import annotations

from regimenes.sinteticos.base import Generador

GENERADORES: dict[str, type[Generador]] = {}


def registrar(cls: type[Generador]) -> type[Generador]:
    """Decorador para registrar un generador por su ``name`` (sin implementar)."""
    raise NotImplementedError("Registro de generadores sinteticos pendiente.")


def crear(nombre: str, **params) -> Generador:
    """Instancia el generador ``nombre`` con ``params`` (sin implementar)."""
    raise NotImplementedError("Registro de generadores sinteticos pendiente.")
