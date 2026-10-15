"""Reglas de limpieza del Tema 4 (funciones puras, sin red). Textos inventados."""
from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from ingesta.noticias_alt import limpieza as L

RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Feed inventado</title>
<item><title>Los mercados caen en Valdoria - Diario Inventado</title><link>https://diario.example/a?utm_source=rss</link>
<pubDate>Mon, 16 Mar 2020 07:00:00 GMT</pubDate><source url="https://diario.example">Diario Inventado</source>
<description>&lt;p&gt;Resumen &lt;b&gt;inventado&lt;/b&gt;&amp;nbsp;de prueba&lt;/p&gt;</description></item>
<item><title>Segunda pieza</title><link>https://otro.example/b</link><pubDate>Mon, 16 Mar 2020 09:30:00 +0100</pubDate></item>
</channel></rss>"""

ATOM = b"""<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>Atom inventado</title>
<entry><title>Entrada Atom</title><link href="https://atom.example/1"/><updated>2023-03-10T15:04:05Z</updated>
<summary>Resumen</summary></entry></feed>"""


@pytest.mark.parametrize("valor, esperado", [
    ("2020-03-16T07:00:00Z", datetime(2020, 3, 16, 7, tzinfo=UTC)),
    ("2020-03-16T04:00:09+0000", datetime(2020, 3, 16, 4, 0, 9, tzinfo=UTC)),
    ("2020-06-05 06:30:54 UTC", datetime(2020, 6, 5, 6, 30, 54, tzinfo=UTC)),
    ("2020-03-16T09:00:00+02:00", datetime(2020, 3, 16, 7, tzinfo=UTC)),
    ("2020-03-16", datetime(2020, 3, 16, tzinfo=UTC)),
    ("20200316120000", datetime(2020, 3, 16, 12, tzinfo=UTC)),
    ("Mon, 16 Mar 2020 07:00:00 GMT", datetime(2020, 3, 16, 7, tzinfo=UTC)),
    (1584342000, datetime(2020, 3, 16, 7, tzinfo=UTC)),
    (1584342000000, datetime(2020, 3, 16, 7, tzinfo=UTC)),
    (date(2020, 3, 16), datetime(2020, 3, 16, tzinfo=UTC)),
])
def test_a_fecha_formatos(valor, esperado):
    assert L.a_fecha(valor) == esperado


@pytest.mark.parametrize("valor", [None, "", "no es una fecha", "2020-13-45", float("nan")])
def test_a_fecha_basura_devuelve_none(valor):
    assert L.a_fecha(valor) is None


def test_fecha_rss_convierte_a_utc():
    assert L.fecha_rss("Mon, 16 Mar 2020 09:30:00 +0100") == datetime(2020, 3, 16, 8, 30, tzinfo=UTC)
    assert L.fecha_rss(None) is None


def test_limpiar_html():
    assert L.limpiar_html("<p>Hola&nbsp;<b>mundo</b> &amp; más</p>") == "Hola mundo & más"
    corto = L.limpiar_html("uno dos tres cuatro cinco", max_chars=12)
    assert corto.endswith("…") and len(corto) <= 12


def test_parse_rss_y_atom():
    items = L.parse_rss(RSS)
    assert len(items) == 2
    assert items[0]["titulo"] == "Los mercados caen en Valdoria"          # quita « - Medio»
    assert items[0]["medio"] == "Diario Inventado"
    assert items[0]["fecha"] == datetime(2020, 3, 16, 7, tzinfo=UTC)
    assert items[0]["resumen"] == "Resumen inventado de prueba"
    assert items[1]["fecha"] == datetime(2020, 3, 16, 8, 30, tzinfo=UTC)
    atom = L.parse_rss(ATOM)
    assert atom[0]["url"] == "https://atom.example/1" and atom[0]["fecha"].year == 2023


def test_parse_rss_rechaza_html():
    with pytest.raises(ValueError):
        L.parse_rss(b"<html><body>Consentimiento de cookies</body></html>")


def test_clave_url_quita_seguimiento_www_y_barra():
    a = L.clave_url("https://www.diario.example/a/b/?utm_source=x&id=3&fbclid=zz#frag")
    b = L.clave_url("http://diario.example/a/b?id=3")
    assert a == b == "//diario.example/a/b?id=3"
    assert L.dominio("https://m.diario.example/x") == "diario.example"


def test_similares_y_simhash():
    assert L.similares("Fed raises interest rates by half a point", "Fed raises interest rates by half point")
    assert not L.similares("Fed raises interest rates", "Valdoria bans imports of cheese")
    base_txt = " ".join(f"palabra{i}" for i in range(80))
    casi = base_txt.replace("palabra7 ", "otra ")
    distinto = " ".join(f"termino{i}" for i in range(80))
    assert L.hamming(L.simhash(base_txt), L.simhash(casi)) <= 3
    assert L.hamming(L.simhash(base_txt), L.simhash(distinto)) > 10


def _doc(url, titulo, fecha, medio, texto=""):
    return dict(url=url, titulo=titulo, fecha_publicacion=fecha, institucion=medio, texto=texto, extra={})


def test_deduplicar_conserva_la_mas_antigua_y_anota_sindicacion():
    cuerpo = " ".join(f"w{i}" for i in range(60))
    docs = [
        _doc("https://b.example/x", "Valdoria cierra sus fronteras comerciales hoy", "2020-03-16T10:00:00Z", "Medio B"),
        _doc("https://a.example/x?utm_source=tw", "Valdoria cierra sus fronteras comerciales hoy", "2020-03-16T07:00:00Z",
             "Medio A"),
        _doc("https://www.a.example/x", "Otro titular", "2020-03-16T11:00:00Z", "Medio A"),         # misma URL
        _doc("https://c.example/z", "Titular totalmente distinto", "2020-03-16T12:00:00Z", "Medio C", cuerpo),
        _doc("https://d.example/z", "Nada que ver con el anterior", "2020-03-16T13:00:00Z", "Medio D", cuerpo),
    ]
    entrada = [dict(d) for d in docs]
    salida = L.deduplicar(docs)
    assert len(salida) == 2
    primero = salida[0]
    assert primero["institucion"] == "Medio A"                     # la más antigua
    assert primero["extra"]["sindicada_en"] == ["Medio B"]          # la URL repetida es del mismo medio
    assert salida[1]["institucion"] == "Medio C" and salida[1]["extra"]["sindicada_en"] == ["Medio D"]
    assert docs == entrada                                          # no modifica la entrada


def test_embudo_duplicados_por_etapas():
    cuerpo = " ".join(f"w{i}" for i in range(60))
    docs = [
        _doc("https://a.example/1", "Titular uno de prueba inventado", "2020-01-01", "A"),
        _doc("https://www.a.example/1/", "Titular uno de prueba inventado", "2020-01-02", "A"),   # URL
        _doc("https://b.example/2", "Titular uno de prueba inventado", "2020-01-02", "B"),       # Jaccard
        _doc("https://c.example/3", "Algo distinto", "2020-01-03", "C", cuerpo),
        _doc("https://d.example/4", "Completamente otro", "2020-01-04", "D", cuerpo),           # simhash
    ]
    emb = L.embudo_duplicados(docs)
    assert list(emb["etapa"]) == ["bruto", "tras_url", "tras_jaccard", "tras_simhash"]
    assert list(emb["n"]) == [5, 4, 3, 2]
    assert list(emb["eliminados"]) == [0, 1, 1, 1]
    assert emb["pct_restante"].iloc[-1] == 40.0


def test_deduplicar_sin_fecha_va_al_final_y_vacio():
    assert L.deduplicar([]) == []
    docs = [_doc("https://a.example/1", "Mismo titular de prueba", None, "A"),
            _doc("https://b.example/1", "Mismo titular de prueba", "2020-01-01", "B")]
    assert L.deduplicar(docs)[0]["institucion"] == "B"


@pytest.mark.parametrize("texto, esperado", [
    ("The market fell sharply as the bank said that it was in trouble with the regulators", "en"),
    ("El mercado cayó con fuerza y los inversores de la bolsa se refugiaron en el oro por la crisis", "es"),
    ("Le marché est en baisse et les investisseurs sont inquiets pour la suite des événements", "fr"),
    ("Der Markt ist eingebrochen und die Anleger sind sich nicht sicher, was mit dem Kurs passiert", "de"),
    ("hola", "und"),
])
def test_idioma(texto, esperado):
    assert L.idioma(texto) == esperado


def test_trocear_y_tokens():
    texto = " ".join(f"Frase inventada número {i} con algunas palabras más." for i in range(60))
    trozos = L.trocear(texto, max_tokens=60, min_tokens=20, titulo="Titular")
    assert len(trozos) > 3
    assert all(t.startswith("Titular. ") for t in trozos)
    assert all(L.tokens_aprox(t) <= 70 for t in trozos)
    assert L.tokens_aprox("") == 0
