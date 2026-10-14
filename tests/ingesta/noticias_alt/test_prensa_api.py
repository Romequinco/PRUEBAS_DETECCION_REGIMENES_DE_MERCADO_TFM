"""NYT Archive y Guardian: sin clave no hay red; con clave (falsa) se parsea y se aplica el TimeGate."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from ingesta.noticias_alt import base, prensa_api as pa
from ingesta.noticias_alt.muestra import DiaTestigo

FX = Path(__file__).parent / "fixtures" / "prensa_api"


@pytest.fixture
def sin_claves(monkeypatch):
    monkeypatch.delenv("NYT_API_KEY", raising=False)
    monkeypatch.delenv("GUARDIAN_API_KEY", raising=False)


def test_nyt_documento_pub_date_y_sin_url():
    docs = json.loads((FX / "nyt_archivo.json").read_text(encoding="utf-8"))["response"]["docs"]
    doc, fila = pa.nyt_documento(docs[0])
    assert doc["fecha_publicacion"] == "2020-03-16" and doc["extra"]["campo_fecha"] == "pub_date"
    assert doc["extra"]["hora_publicacion"] == "2020-03-16T04:00:09+00:00"
    assert doc["texto"] == "Resumen inventado.\nPrimer párrafo inventado."
    assert doc["entidades"] == ["Bolsa"] and doc["seccion"] == "Business Day" and doc["rag"] == "cajon"
    assert "texto" not in fila
    assert pa.nyt_documento(docs[1])[0]["texto"] == "Igual."             # abstract = lead: una vez
    assert pa.nyt_documento(docs[2]) is None                              # sin URL


def test_guardian_documento_usa_la_primera_publicacion():
    res = json.loads((FX / "guardian_busqueda.json").read_text(encoding="utf-8"))["response"]["results"]
    doc, fila = pa.guardian_documento(res[0])
    assert fila["campo_fecha"] == "firstPublicationDate" and fila["fecha_publicacion"].startswith("2020-03-16T06:30")
    assert doc["texto"] == "Cuerpo inventado de la pieza." and doc["institucion"] == "The Guardian"
    doc2, fila2 = pa.guardian_documento(res[1])
    assert fila2["campo_fecha"] == "webPublicationDate" and doc2["texto"] == ""
    assert pa.guardian_documento({"webUrl": "https://x.example"}) is None


def test_sondear_sin_clave_no_toca_la_red(sin_red, red_falsa, sin_claves):
    red_falsa(pa, {})                                                     # cualquier petición fallaría
    res = pa.sondear(base.ConfigSonda(modo_rapido=True, verbose=False))
    assert res.registro == [] and not res.documentos and not res.tablas
    cob = {c["fuente"]: c for c in res.cobertura}
    assert set(cob) == {"nyt", "guardian"}
    assert all(c["estado"] == "no_probada" and "sin clave" in c["nota"] for c in cob.values())
    assert cob["nyt"]["inicio"] == "1851-09-18" and cob["guardian"]["inicio"] == "1999-01-01"
    assert cob["nyt"]["tipo"] == "titular" and cob["guardian"]["tipo"] == "texto"
    assert all(set(c) == set(base.COLUMNAS_COBERTURA) for c in res.cobertura)
    assert all(f.estado_sonda == "no_probada" and f.autenticacion == "clave gratuita" for f in res.fichas)


def test_sondear_con_clave_falsa(sin_red, red_falsa, monkeypatch):
    monkeypatch.setenv("NYT_API_KEY", "clave-de-prueba")
    monkeypatch.setenv("GUARDIAN_API_KEY", "clave-de-prueba")
    red_falsa(pa, {"api.nytimes.com": "prensa_api/nyt_archivo.json",
                   "content.guardianapis.com": "prensa_api/guardian_busqueda.json"})
    dias = [DiaTestigo("covid", date(2020, 3, 16), "")]
    res = pa.sondear(base.ConfigSonda(dias=dias, modo_rapido=True, verbose=False))
    cob = {c["fuente"]: c for c in res.cobertura}
    assert cob["nyt"]["estado"] == "ok" and cob["guardian"]["estado"] == "ok"
    assert res.documentos == {"nyt_muestra_rapido": 1, "guardian_muestra_rapido": 2}
    assert len(res.tablas["nyt_articulos"]) == 1                          # solo las del día testigo
    assert "texto" not in res.tablas["guardian_articulos"].columns
    assert {t["campo_fecha"] for t in res.timegate} == {"pub_date", "firstPublicationDate", "webPublicationDate"}
    # Segunda pasada: de la caché del corpus, sin red.
    red_falsa(pa, {})
    res2 = pa.sondear(base.ConfigSonda(dias=dias, modo_rapido=True, verbose=False))
    assert res2.documentos == res.documentos


def test_sondear_clave_rechazada_registra_fallo(sin_red, red_falsa, monkeypatch, sin_claves):
    monkeypatch.setenv("GUARDIAN_API_KEY", "mala")
    red_falsa(pa, {"content.guardianapis.com": (401, b'{"message":"Unauthorized"}')})
    res = pa.sondear(base.ConfigSonda(dias=[DiaTestigo("covid", date(2020, 3, 16), "")], verbose=False))
    cob = {c["fuente"]: c for c in res.cobertura}
    assert cob["guardian"]["estado"] == "fallo" and cob["nyt"]["estado"] == "no_probada"


@pytest.mark.live
def test_live_prensa_api_sin_clave_documenta(sin_red):
    res = pa.sondear(base.ConfigSonda(verbose=False))
    assert {c["fuente"] for c in res.cobertura} == {"nyt", "guardian"}
