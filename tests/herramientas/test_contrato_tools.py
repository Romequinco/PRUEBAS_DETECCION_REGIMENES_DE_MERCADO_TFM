"""Contrato de tools (``herramientas``): registro, esquema para el LLM y ``get_regimen`` tipado."""
from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from herramientas import catalogo, invocar, json_schema
from herramientas.contrato import EntradaTool, SalidaTool
from herramientas.registro import registrar
from regimenes import rutas


def test_catalogo_registra_get_regimen_para_sus_agentes() -> None:
    assert "get_regimen" in [t.nombre for t in catalogo()]
    assert "get_regimen" in [t.nombre for t in catalogo("macro")]
    assert "get_regimen" not in [t.nombre for t in catalogo("equity")]


def test_el_esquema_que_ve_el_llm_no_tiene_fecha() -> None:
    esquema = json_schema("get_regimen")
    assert esquema["name"] == "get_regimen" and esquema["description"]
    assert "fecha" not in esquema["parameters"].get("properties", {})


def test_registrar_rechaza_una_fecha_elegida_por_el_llm() -> None:
    class ConFecha(EntradaTool):
        fecha: date

    with pytest.raises(ValueError, match="fecha"):
        registrar("tool_tramposa", ConFecha)


def test_invocar_valida_los_argumentos_del_llm() -> None:
    with pytest.raises(ValidationError):
        invocar("get_regimen", {"pista": "Z"}, fecha=date(2020, 3, 16))
    with pytest.raises(ValidationError):
        invocar("get_regimen", {"inventado": 1}, fecha=date(2020, 3, 16))


_PANELES = rutas.RESULTS_BENCHMARK / "panels"


@pytest.mark.resultados
@pytest.mark.skipif(not (_PANELES / "B_D05.parquet").is_file(), reason="falta el panel B_D05 versionado")
def test_get_regimen_tipado_coincide_con_la_fase1() -> None:
    from regimenes.regimen import get_regimen

    salida = invocar("get_regimen", {}, fecha=date(2020, 3, 16))
    bruto = get_regimen("2020-03-16")
    assert isinstance(salida, SalidaTool)
    assert (salida.estado, salida.detector, salida.pista) == (bruto["estado"], bruto["detector"], bruto["pista"])
    assert salida.fecha_dato <= salida.fecha_consulta
    assert salida.citas and salida.citas[0].id == bruto["fuente"]["sha256"]
    assert salida.model_dump_json()  # serializable para el grafo de agentes
