"""Tests de ``ingesta.noticias_alt.rss`` (sin red salvo el ``live``). Fixtures inventados."""
from __future__ import annotations

import re
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from ingesta.noticias_alt import base, rss
from ingesta.noticias_alt.muestra import DiaTestigo

FX = Path(__file__).parent / "fixtures" / "rss"
CAPTURA = datetime(2020, 3, 17, 12, 0, tzinfo=UTC)


def _leer(nombre: str) -> bytes:
    return (FX / nombre).read_bytes()


# --------------------------------------------------------------------------- parseo

def test_parsear_bing_desenvuelve_apiclick_y_medio():
    items = rss.parsear(_leer("bing.xml"), bing=True)
    assert len(items) == 2
    assert items[0]["url"] == "https://www.example-news.com/markets/record-123"
    assert items[0]["medio"] == "Example News"
    assert items[1]["url"] == "https://www.example-direct.com/story/9"      # sin envoltorio: igual
    assert items[0]["fecha"] == datetime(2020, 3, 17, 3, 41, tzinfo=UTC)


def test_desenvolver_bing_solo_toca_apiclick():
    assert rss.desenvolver_bing("https://www.bing.com/news/search?q=x&url=https%3a%2f%2fa.com") \
        == "https://www.bing.com/news/search?q=x&url=https%3a%2f%2fa.com"
    assert rss.desenvolver_bing("") == ""


@pytest.mark.parametrize("nombre,n", [("feed_rss2.xml", 4), ("feed_atom.xml", 2), ("feed_rdf.rdf", 1)])
def test_parsear_formatos(nombre, n):
    items = rss.parsear(_leer(nombre))
    assert len(items) == n and all(i["titulo"] for i in items)


def test_parsear_rechaza_html():
    with pytest.raises(ValueError):
        rss.parsear(_leer("no_feed.html"))


def test_fechas_con_zona_pasan_a_utc():
    items = rss.parsear(_leer("feed_rss2.xml"))
    assert items[1]["fecha"] == datetime(2020, 3, 16, 8, 30, tzinfo=UTC)   # +0100
    atom = rss.parsear(_leer("feed_atom.xml"))
    assert atom[0]["fecha"] == datetime(2020, 3, 15, 7, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- TimeGate

def test_regla_fecha_publicacion():
    pub = datetime(2020, 3, 16, 21, 5, tzinfo=UTC)
    assert rss.fecha_publicacion(pub, CAPTURA) == (pub, "pubDate")
    assert rss.fecha_publicacion(None, CAPTURA) == (CAPTURA, "captura (sin pubDate)")
    futuro = datetime(2099, 1, 1, tzinfo=UTC)
    assert rss.fecha_publicacion(futuro, CAPTURA) == (CAPTURA, "captura (pubDate futuro)")


def test_documentos_titular_con_fecha_y_extracto_corto():
    items = rss.parsear(_leer("feed_rss2.xml"))
    docs = rss.documentos(items, fuente=rss.F_RSS, medio="Ejemplo", pais="GB", lengua="en", captura=CAPTURA,
                          feed="ejemplo")
    assert len(docs) == 4
    d0, d2, d3 = docs[0], docs[2], docs[3]
    assert d0["tipo_documento"] == "titular" and d0["rag"] == "cajon" and d0["paises"] == ["GB"]
    assert d0["fecha_publicacion"] == "2020-03-16" and d0["extra"]["campo_fecha"] == "pubDate"
    assert d0["extra"]["fecha_captura"] == CAPTURA.isoformat()
    assert "<p>" not in d0["texto"] and "&amp;" not in d0["texto"]
    assert len(d0["texto"]) <= len(d0["titulo"]) + 2 + rss.MAX_EXTRACTO
    assert d2["fecha_publicacion"] == "2020-03-17" and d2["extra"]["campo_fecha"] == "captura (sin pubDate)"
    assert d3["fecha_publicacion"] == "2020-03-17" and d3["extra"]["campo_fecha"] == "captura (pubDate futuro)"
    # misma URL con y sin parámetros de seguimiento -> mismo doc_id
    otro = rss.documentos([dict(items[0], url="https://news.example.org/business/ruritania-rates")],
                          fuente=rss.F_RSS, medio="E", pais="GB", lengua="en", captura=CAPTURA, feed="x")
    assert otro[0]["doc_id"] == d0["doc_id"]


def test_medir_items_profundidad_y_porcentajes():
    m = rss.medir_items(rss.parsear(_leer("feed_rss2.xml")), CAPTURA)
    assert m["n_items"] == 4 and m["pct_con_fecha"] == 75.0
    # el pubDate futuro no cuenta para la profundidad: 08:30 -> 21:05 del 16-mar
    assert m["profundidad_h"] == pytest.approx(12.6, abs=0.1)
    assert m["edad_min_h"] == pytest.approx(14.92, abs=0.01)
    assert rss.medir_items([], CAPTURA)["profundidad_h"] is None


def test_leer_feed_respeta_robots(monkeypatch):
    monkeypatch.setattr(rss, "robots_permite", lambda url: False)
    monkeypatch.setattr(rss, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no debe descargar")))
    with pytest.raises(PermissionError):
        rss.leer_feed("https://bloqueado.example/rss")


def test_feeds_sin_ids_repetidos_y_15_paises():
    ids = [f[0] for f in rss.FEEDS]
    assert len(ids) == len(set(ids)) >= 36
    assert len({f[1] for f in rss.FEEDS}) >= 15


# --------------------------------------------------------------------------- punta a punta

def _cfg(**kw):
    return base.ConfigSonda(dias=[DiaTestigo("covid", date(2020, 3, 16), "x"), DiaTestigo("hoy", date.today(), "x")],
                            modo_rapido=True, verbose=False, **kw)


def test_sondear_sin_red(sin_red, red_falsa, monkeypatch):
    monkeypatch.setattr(rss, "robots_permite", lambda url: not ("nytimes" in url or "news.google" in url))
    red_falsa(rss, {"feeds.bbci.co.uk": "rss/feed_rss2.xml", "theguardian.com": "rss/feed_atom.xml",
                    "bing.com/news/search": "rss/bing.xml"})
    res = rss.sondear(_cfg())

    feeds = res.tablas["rss_feeds"]
    assert {"medio", "pais", "idioma", "url", "robots_ok", "estado", "n_items", "profundidad_h",
            "pct_con_fecha"} <= set(feeds.columns)
    f = feeds.set_index("feed")
    assert f.loc["bbc_business", "n_items"] == 4 and f.loc["bbc_business", "estado"] == "ok"
    assert not f.loc["nyt_economy", "robots_ok"] and f.loc["nyt_economy", "estado"] == "robots"
    assert not f.loc["google_news_busqueda", "robots_ok"]
    assert (feeds["fuente"] == rss.F_BING).sum() == 3                    # tope = 3 consultas

    items = res.tablas["rss_items"]
    assert {"medio", "pais", "idioma", "titular", "fecha_publicacion", "lag_captura_h"} <= set(items.columns)
    # copyright: ningún extracto en las tablas de results
    for df in res.tablas.values():
        assert "EXTRACTO-SECRETO" not in df.to_csv(index=False)

    cob = pd.DataFrame(res.cobertura).set_index("fuente")
    assert list(pd.DataFrame(res.cobertura).columns) == base.COLUMNAS_COBERTURA
    assert set(cob.index) == {rss.F_RSS, rss.F_BING, rss.F_GOOGLE}
    assert not cob["backtest"].any()
    assert cob.loc[rss.F_GOOGLE, "estado"] == "fallo" and cob.loc[rss.F_RSS, "estado"] == "parcial"
    assert cob.loc[rss.F_BING, "estado"] == "ok" and "covid" in cob.loc[rss.F_RSS, "nota"]
    # inicio/fin como AAAA-MM-DD (igual que las demás fuentes: la tabla común se parsea con to_datetime)
    fechas = cob[["inicio", "fin"]].stack().dropna()
    assert len(fechas) and fechas.map(lambda x: re.fullmatch(r"\d{4}-\d{2}-\d{2}", x) is not None).all()

    reg = pd.DataFrame(res.registro)
    assert (reg.loc[reg["item"] == "nyt_economy", "error"] == "robots.txt").all()
    assert {f.fuente for f in res.fichas} == {rss.F_RSS, rss.F_BING, rss.F_GOOGLE}
    assert all(f.tema == "noticias_alt" for f in res.fichas)
    assert res.timegate and all(set(t) == set(base.COLUMNAS_TIMEGATE) for t in res.timegate)

    docs = base.leer_documentos("rss_titulares_rapido")              # modo rápido: sufijo, no pisa
    assert res.documentos["rss_titulares_rapido"] == len(docs) == 6    # 4 RSS 2.0 + 2 Atom
    assert any("EXTRACTO-SECRETO" in d["texto"] for d in docs)       # el texto sí va al corpus
    assert res.documentos["bing_news_titulares_rapido"] == 2          # 3 consultas, mismos 2 ítems

    # segunda pasada: la captura del día sale de la caché, sin red
    monkeypatch.setattr(rss, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("sin red")))
    res2 = rss.sondear(_cfg())
    assert any(r["item"] == "bbc_business (caché)" and r["ok"] for r in res2.registro)


def test_sondear_feed_que_no_es_rss(sin_red, red_falsa, monkeypatch):
    monkeypatch.setattr(rss, "robots_permite", lambda url: True)
    red_falsa(rss, {"feeds.bbci.co.uk": "rss/no_feed.html", "theguardian.com": (403, b"no"),
                    "nytimes.com": "rss/feed_rdf.rdf", "bing.com": "rss/bing.xml", "news.google": "rss/bing.xml"})
    res = rss.sondear(_cfg(usar_cache=False))
    f = res.tablas["rss_feeds"].set_index("feed")
    assert f.loc["bbc_business", "estado"] == "error" and "ValueError" in f.loc["bbc_business", "error"]
    assert f.loc["guardian_business", "estado"] == "error"
    assert f.loc["nyt_economy", "n_items"] == 1


@pytest.mark.live
def test_live_bing_y_bbc():
    items = rss.buscar_bing("stock market")
    assert items and all("apiclick" not in i["url"] for i in items)
    contenido, captura, _ = rss.leer_feed(rss.FEEDS[0][4], usar_cache=False)
    assert rss.medir_items(rss.parsear(contenido), captura)["n_items"] > 0
