"""Tests de la base común de la Fase 2 (``ingesta.comun``)."""
from datetime import date

import pytest

from ingesta.comun.esquema import Documento, FichaFuente, doc_id
from ingesta.comun.fichas import cargar_fichas, guardar_ficha
from ingesta.comun.http import sin_clave
from ingesta.comun.rutas import TEMAS, corpus_dir, resultados_dir


def _ficha(**kw):
    base = dict(fuente="prueba", nombre="Prueba", tema="empresas", agentes=["equity"], rag="cajon", url="u",
                acceso="api", autenticacion="ninguna", coste="0 €", licencia_y_terminos="dominio público",
                cobertura_inicio="1994-01-01", cobertura_fin=None, frecuencia="por evento",
                retraso_publicacion="0 d", formato="html", volumen_estimado="10 MB", limites="ninguno",
                estado_sonda="ok")
    base.update(kw)
    return FichaFuente(**base)


def test_doc_id_estable_y_por_contenido():
    assert doc_id("hola") == doc_id(b"hola")
    assert doc_id("hola") != doc_id("adios")
    assert len(doc_id("x")) == 24


def test_documento_serializa_fechas():
    d = Documento(doc_id="a", fuente="f", tipo_documento="t", titulo="T",
                  fecha_publicacion=date(2020, 3, 15), url="u", agente="macro", rag="cajon")
    assert d.to_dict()["fecha_publicacion"] == "2020-03-15"


def test_guardar_y_cargar_ficha(tmp_path):
    guardar_ficha(_ficha(), tmp_path)
    (fila,) = cargar_fichas(carpeta=tmp_path)
    assert fila["fuente"] == "prueba" and fila["tema"] == "empresas"


def test_sin_clave_enmascara():
    assert sin_clave("https://x/api?id=A&api_key=abc123&f=json") == "https://x/api?id=A&api_key=***&f=json"


def test_rutas_por_tema(tmp_path, monkeypatch):
    monkeypatch.setenv("INGESTA_CORPUS_DIR", str(tmp_path))
    monkeypatch.setattr("ingesta.comun.rutas.ENV_FILE", tmp_path / "no_existe.env")
    assert corpus_dir("noticias_alt") == tmp_path / "noticias_alt"
    assert resultados_dir("empresas").name == "empresas"
    assert set(TEMAS) == {"mercados_series", "bancos_riesgo", "empresas", "noticias_alt"}
    with pytest.raises(ValueError):
        corpus_dir("tema_inexistente")
