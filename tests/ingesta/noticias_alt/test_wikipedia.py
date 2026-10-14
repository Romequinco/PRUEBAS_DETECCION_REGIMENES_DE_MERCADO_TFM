"""Tests de ``ingesta.noticias_alt.wikipedia`` (sin red; wikitexto inventado con el formato real)."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from ingesta.noticias_alt import base, wikipedia
from ingesta.noticias_alt.muestra import DiaTestigo

FX = Path(__file__).parent / "fixtures" / "wikipedia"


def _wikitexto(nombre: str) -> str:
    return wikipedia.revision(json.loads((FX / nombre).read_text(encoding="utf-8")))["wikitexto"]


def test_titulo_y_parametros_timegate():
    assert wikipedia.titulo_portal(date(2020, 3, 16)) == "Portal:Current events/2020 March 16"
    assert wikipedia.titulo_portal(date(2008, 9, 5)) == "Portal:Current events/2008 September 5"
    p = wikipedia.params_revision("X", al_cierre=date(2020, 3, 16))
    assert p["rvstart"] == "2020-03-16T23:59:59Z" and p["rvdir"] == "older" and "content" in p["rvprop"]
    p = wikipedia.params_revision("X", primera=True, contenido=False)
    assert p["rvdir"] == "newer" and "rvstart" not in p and "content" not in p["rvprop"]
    assert "rvdir" not in wikipedia.params_revision("X")       # revisión actual


def test_revision_ausente_o_sin_revisiones():
    assert wikipedia.revision(None) is None
    assert wikipedia.revision(json.loads((FX / "ce_ausente.json").read_text(encoding="utf-8"))) is None
    assert wikipedia.revision(json.loads((FX / "ce_sin_revision.json").read_text(encoding="utf-8"))) is None
    r = wikipedia.revision(json.loads((FX / "ce_primera.json").read_text(encoding="utf-8")))
    assert r["wikitexto"] is None and r["timestamp"] == "2008-09-15T01:10:00Z"


def test_parse_current_events_formato_reciente():
    items = wikipedia.parse_current_events(_wikitexto("ce_reciente.json"))
    assert len(items) == 3                                      # solo hojas con fuente
    i0 = items[0]
    assert i0["categoria"] == "Business and economy"
    assert i0["temas"] == ["Economic impact of an example pandemic", "2020 example stock market crash"]
    assert i0["texto"] == "Example index futures fall 5%, triggering a trading curb."
    assert [m for _, m in i0["fuentes"]] == ["Example News", "Example Wire"]
    assert items[1]["texto"] == "The central bank cuts its policy rate to near zero."   # [[a|b]] -> b
    assert items[2]["categoria"] == "Armed conflicts and attacks" and items[2]["temas"] == ["Conflict in Example Region"]


def test_parse_current_events_formato_2008_sin_categorias():
    items = wikipedia.parse_current_events(_wikitexto("ce_2008.json"))
    assert len(items) == 2 and all(i["categoria"] is None for i in items)
    assert items[0]["texto"] == "Example Brothers files for Chapter 11 bankruptcy protection."
    assert items[0]["fuentes"][0] == ("http://example.com/bank", "AP via Example Today")
    assert wikipedia.parse_current_events("") == []


def test_documentos_fecha_publicacion_es_el_dia_siguiente():
    j = json.loads((FX / "ce_reciente.json").read_text(encoding="utf-8"))
    rev = wikipedia.revision(j)
    docs = wikipedia.hechos_a_documentos(wikipedia.parse_current_events(rev["wikitexto"]), date(2020, 3, 16), rev)
    assert len(docs) == 3 and len({d["doc_id"] for d in docs}) == 3
    d = docs[0]
    assert d["fecha_publicacion"] == "2020-03-17" and d["fecha_referencia"] == "2020-03-16" and d["rag"] == "base" and d["agente"] == "macro"
    assert d["licencia"] == "CC BY-SA 4.0" and d["version"] == "945000000"
    assert d["url"].endswith("oldid=945000000") and d["texto"].startswith("Example index futures")


def test_pageviews_serie_d_mas_1_y_picos():
    j = json.loads((FX / "pageviews.json").read_text(encoding="utf-8"))
    s = base.serie_larga(wikipedia.pageviews_a_serie(j, "Recession"))
    assert len(s) == 30 and set(s["serie"]) == {"wiki_pv:Recession"}
    assert ((s["fecha_publicacion"] - s["fecha_referencia"]).dt.days == 1).all()
    p = wikipedia.picos(s)
    assert p["fecha_pico"] == "2020-03-16" and p["visitas_pico"] == 1000 + 150 + 9000
    assert p["inicio"] == "2020-03-01" and p["n_dias"] == 30 and p["ratio_pico_mediana"] > 8


def test_url_pageviews_escapa_articulo():
    u = wikipedia.url_pageviews("S&P 500", date(2015, 7, 1), date(2026, 10, 14))
    assert "/user/S%26P%20500/daily/20150701/20261014" in u


RUTAS = {
    "rvstart=2004-06-01": "wikipedia/ce_ausente.json",
    "rvstart=2007-03-01": "wikipedia/ce_sin_revision.json",
    "rvstart=2008-09-15": "wikipedia/ce_2008.json",
    "rvdir=newer": "wikipedia/ce_primera.json",
    "rvstart=": "wikipedia/ce_reciente.json",
    "titles=Portal": "wikipedia/ce_reciente.json",           # revisión actual (sin rvdir)
    "metrics/pageviews": "wikipedia/pageviews.json",
}
DIAS = [DiaTestigo("antiguo", date(2004, 6, 1), "x"), DiaTestigo("lehman", date(2008, 9, 15), "x"),
        DiaTestigo("covid", date(2020, 3, 16), "x")]


def test_sondear_punta_a_punta_sin_red(sin_red, red_falsa):
    red_falsa(wikipedia, RUTAS)
    res = wikipedia.sondear(base.ConfigSonda(dias=DIAS, modo_rapido=True, verbose=False))
    assert all(r["ok"] for r in res.registro), [r for r in res.registro if not r["ok"]]
    ce = res.tablas["wiki_current_events"].set_index("clave")
    assert ce.loc["antiguo", "nota"].startswith("sin cobertura")
    assert ce.loc["lehman", "n_hechos"] == 2 and ce.loc["covid", "n_hechos"] == 3
    assert ce.loc["covid", "n_categorias"] == 2 and ce.loc["covid", "n_economia"] == 2
    assert ce.loc["lehman", "lag_h"] == pytest.approx(23.73, abs=0.01)
    assert ce.loc["lehman", "lag_creacion_h"] == pytest.approx(1.17, abs=0.01)
    ini = res.tablas["wiki_current_events_inicio"].set_index("fecha")
    assert not ini.loc["2007-03-01", "revision_al_cierre"] and ini.loc["2007-03-01", "existe_hoy"]
    assert ini.loc["2007-06-01", "revision_al_cierre"]
    cob = {c["fuente"]: c for c in res.cobertura}
    assert set(cob) == {wikipedia.F_CE, wikipedia.F_PV}
    assert all(set(c) == set(base.COLUMNAS_COBERTURA) for c in res.cobertura)
    assert cob[wikipedia.F_CE]["inicio"] == "2007-06-01" and cob[wikipedia.F_CE]["fin"] == "2020-03-16"
    assert cob[wikipedia.F_CE]["destino"] == "base" and cob[wikipedia.F_CE]["estado"] == "ok"
    assert cob[wikipedia.F_PV]["inicio"] == "2020-03-01" and cob[wikipedia.F_PV]["fin"] == "2020-03-30"
    # documentos: tope 3 en modo rápido (2 de Lehman + 3 de COVID)
    assert res.documentos["wiki_current_events_rapido"] == 5                 # modo rápido: sufijo, no pisa
    docs = base.leer_documentos("wiki_current_events_rapido")
    assert {d["fecha_publicacion"] for d in docs} == {"2008-09-16", "2020-03-17"}
    # pageviews: 3 artículos (tope) en una serie larga, picos y días testigo
    assert res.series == ["wiki_pageviews_rapido"]
    s = base.leer_serie("wiki_pageviews_rapido")
    assert s["serie"].nunique() == 3 and len(s) == 90
    pk = res.tablas["wiki_pageviews_picos"]
    assert len(pk) == 3 and set(pk["fecha_pico"]) == {"2020-03-16"}
    tt = res.tablas["wiki_pageviews_testigo"]
    assert set(tt["clave"]) == {"covid"} and (tt["ratio"] > 5).all()
    tg = {t["fuente"] for t in res.timegate}
    assert tg == {wikipedia.F_CE, wikipedia.F_PV}
    assert all(set(t) == set(base.COLUMNAS_TIMEGATE) for t in res.timegate)
    fichas = {f.fuente: f for f in res.fichas}
    assert fichas[wikipedia.F_CE].estado_sonda == "ok" and fichas[wikipedia.F_CE].tema == "noticias_alt"
    assert fichas[wikipedia.F_CE].mediciones["paginas_retroactivas"] == ["2007-03-01"]


def test_sondear_fallo_de_red_no_detiene(sin_red, monkeypatch):
    def _falla(url, **kw):
        raise ConnectionError("sin red")

    monkeypatch.setattr(wikipedia, "get", _falla)
    res = wikipedia.sondear(base.ConfigSonda(dias=DIAS[1:2], modo_rapido=True, verbose=False))
    assert base.estado(res.registro, wikipedia.F_CE) == "fallo"
    assert base.estado(res.registro, wikipedia.F_PV) == "fallo"
    assert {c["estado"] for c in res.cobertura} == {"fallo"}
    assert res.series == [] and not res.documentos


@pytest.mark.live
def test_live_wikipedia(sin_red):
    res = wikipedia.sondear(base.ConfigSonda(dias=[DiaTestigo("covid", date(2020, 3, 16), "x")], modo_rapido=True,
                                             verbose=False, max_items=1))
    ce = res.tablas["wiki_current_events"]
    assert ce["n_hechos"].iloc[0] > 5
    assert base.estado(res.registro, wikipedia.F_PV) == "ok"
