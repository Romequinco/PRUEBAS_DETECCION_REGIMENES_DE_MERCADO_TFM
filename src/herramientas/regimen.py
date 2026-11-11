"""Tool ``get_regimen`` tipada sobre :func:`regimenes.regimen.get_regimen` (Fase 1).

Solo envuelve: la lógica, el detector por pista y los ficheros que lee están en ``regimenes.regimen``
y ``configs/regimen.yaml``. Cuando la ADR-006 cambie el detector, este contrato no cambia.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import Field

from herramientas.contrato import Cita, EntradaTool, Modelo, SalidaTool
from herramientas.registro import registrar
from regimenes.regimen import get_regimen as _get_regimen


class EntradaRegimen(EntradaTool):
    pista: Literal["A", "B"] | None = Field(
        default=None, description="Pista de datos; None aplica la regla por fecha (B desde su inicio OOS).")


class Contexto(Modelo):
    """Consenso del resto de detectores para la misma sesión (bloque ``contexto`` de la Fase 1)."""

    prob_crisis_consenso: float | None
    estado_consenso: str | None
    detectores_en_crisis: int
    detectores_total: int
    confianza: float | None
    coincide_consenso: bool | None
    lectura: str


class SalidaRegimen(SalidaTool):
    fecha_dato: date = Field(description="Última sesión con dato visible (<= fecha de la decisión).")
    pista: str
    detector: str
    estado: str = Field(description="Estado provisional (hoy calma/crisis; ADR-006 lo fijará).")
    estados_posibles: list[str]
    prob_crisis: float | None
    dias_en_estado: int
    dias_en_estado_censurado: bool
    provisional: bool
    contexto: Contexto | None


def _contexto(bruto: dict[str, Any] | None) -> Contexto | None:
    if not bruto:
        return None
    return Contexto(**{k: bruto.get(k) for k in Contexto.model_fields})


@registrar("get_regimen", EntradaRegimen, agentes=("macro", "risk", "portfolio", "cio"))
def get_regimen(entrada: EntradaRegimen, fecha: date) -> SalidaRegimen:
    """Régimen de mercado publicado hasta la fecha de la decisión, con el consenso de detectores."""
    r = _get_regimen(fecha, pista=entrada.pista)
    f = r["fuente"]
    citas = [Cita(fuente=f["panel"], id=f["sha256"], fecha_publicacion=date.fromisoformat(r["fecha_dato"]))]
    if r.get("contexto"):
        cf = r["contexto"]["fuente"]
        citas += [Cita(fuente=cf["senales_detectores"], id=cf["sha256_senales"]),
                  Cita(fuente=cf["prob_consenso"], id=cf["sha256_prob"])]
    avisos = ["régimen provisional hasta la ADR-006"] if r["provisional"] else []
    return SalidaRegimen(
        tool="get_regimen", fecha_consulta=fecha, citas=citas, avisos=avisos,
        fecha_dato=date.fromisoformat(r["fecha_dato"]), pista=r["pista"], detector=r["detector"],
        estado=r["estado"], estados_posibles=list(r["estados_posibles"]), prob_crisis=r["prob_crisis"],
        dias_en_estado=r["dias_en_estado"], dias_en_estado_censurado=r["dias_en_estado_censurado"],
        provisional=r["provisional"], contexto=_contexto(r.get("contexto")),
    )
