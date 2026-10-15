"""CC-NEWS: listados, WARC parcial, extracción, regla de fecha (TimeGate) y sonda sin red.

El WARC de ``fixtures/ccnews`` es sintético (textos inventados con el formato real); lo genera un
script de una vez: 5 respuestas (A anterior al rastreo, B con metadato posterior, C sindicada de A,
D portada sin texto, E error 404) y un miembro gzip cortado como el de una lectura con Range.
"""
from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from ingesta.noticias_alt import base, ccnews
from ingesta.noticias_alt.muestra import DiaTestigo

FX = Path(__file__).parent / "fixtures" / "ccnews"
WARC = FX / "CC-NEWS-20200316160000-00102.warc.gz.part"


def test_parse_paths_gzip_y_texto():
    rutas = ccnews.parse_paths((FX / "warc_paths_202003.gz").read_bytes())
    assert len(rutas) == 4 and rutas[0].endswith("00100.warc.gz")
    assert ccnews.parse_paths((FX / "warc_paths_201608.txt").read_bytes())[0].startswith("crawl-data/CC-NEWS/2016/08")
    assert ccnews.fecha_ruta(rutas[1]) == datetime(2020, 3, 16, 8, tzinfo=UTC)
    assert ccnews.fecha_ruta("otra/cosa.gz") is None


def test_elegir_warc_dia_exacto_y_aproximado():
    rutas = ccnews.parse_paths((FX / "warc_paths_202003.gz").read_bytes())
    ruta, exacto = ccnews.elegir_warc(rutas, date(2020, 3, 16))
    assert exacto and ruta.endswith("00102.warc.gz")               # el central de los del día
    ruta, exacto = ccnews.elegir_warc(rutas, date(2020, 3, 25))
    assert not exacto and ruta.endswith("00103.warc.gz")           # el más cercano
    with pytest.raises(ValueError):
        ccnews.elegir_warc([], date(2020, 3, 16))


def test_miembros_gzip_para_en_el_cortado():
    miembros = ccnews.miembros_gzip(WARC.read_bytes())
    assert len(miembros) == 7                                       # warcinfo, request y 5 respuestas
    assert ccnews.miembros_gzip(b"basura") == []


def test_registros_y_extraccion():
    regs = list(ccnews.registros_warc(ccnews.miembros_gzip(WARC.read_bytes())))
    assert [r["http_estado"] for r in regs] == [200, 200, 200, 200, 404]
    a = ccnews.extraer_articulo(regs[0]["html"])
    assert a["titulo"] == "Valdoria se desploma en bolsa"
    assert a["fecha_meta"] == datetime(2020, 3, 16, 7, tzinfo=UTC) and a["campo_meta"] == "article:published_time"
    assert a["lang_declarado"] == "es" and a["medio"] == "Diario A"
    assert "var x" not in a["texto"] and "menú" not in a["texto"]
    assert len(a["texto"].splitlines()) == 8


def test_fecha_timegate_reglas():
    r = datetime(2020, 3, 16, 16, tzinfo=UTC)
    assert ccnews.fecha_timegate(r - timedelta(hours=3), r) == (r - timedelta(hours=3), "meta")
    assert ccnews.fecha_timegate(r + timedelta(days=1), r) == (r, "rastreo_meta_posterior")
    assert ccnews.fecha_timegate(None, r) == (r, "rastreo_sin_meta")
    assert ccnews.fecha_timegate(None, None) == (None, "meta_sin_rastreo")


def test_articulos_aplica_timegate_y_descarta_errores():
    arts = {a["dominio"]: a for a in ccnews.articulos(WARC.read_bytes())}
    assert set(arts) == {"diario-a.example", "noticias-b.example", "agregador-c.example", "portada-d.example"}
    a, b, d = arts["diario-a.example"], arts["noticias-b.example"], arts["portada-d.example"]
    assert a["regla"] == "meta" and a["fecha_publicacion"] == datetime(2020, 3, 16, 7, tzinfo=UTC)
    assert a["lag_h"] == 9.08
    assert b["regla"] == "rastreo_meta_posterior" and b["fecha_publicacion"] == b["warc_date"]
    assert b["fecha_publicacion"] <= datetime(2020, 3, 16, 16, 6, tzinfo=UTC)   # nunca la fecha futura
    assert b["idioma"] == "en" and d["n_tokens"] < ccnews.MIN_TOKENS and d["regla"] == "rastreo_sin_meta"


def test_documento_lleva_fecha_dia_y_licencia():
    a = next(x for x in ccnews.articulos(WARC.read_bytes()) if x["dominio"] == "noticias-b.example")
    d = ccnews.documento(a, "covid")
    assert d["fecha_publicacion"] == "2020-03-16" and d["rag"] == "cajon" and d["agente"] == "macro"
    assert d["licencia"] == base.LIC_COPYRIGHT and d["extra"]["regla_fecha"] == "rastreo_meta_posterior"
    assert d["extra"]["fecha_rastreo"].startswith("2020-03-16T16:06") and d["texto"]


def _rutas_falsas():
    return {"2016/08/warc.paths.gz": "ccnews/warc_paths_201608.txt",
            "2020/03/warc.paths.gz": "ccnews/warc_paths_202003.gz",
            "CC-NEWS-202003": "ccnews/CC-NEWS-20200316160000-00102.warc.gz.part"}


def test_sondear_sin_red_modo_rapido(sin_red, red_falsa):
    red_falsa(ccnews, _rutas_falsas())
    dias = [DiaTestigo("lehman", date(2008, 9, 15), ""), DiaTestigo("covid", date(2020, 3, 16), ""),
            DiaTestigo("hoy", date(2020, 3, 17), "")]
    res = ccnews.sondear(base.ConfigSonda(dias=dias, modo_rapido=True, verbose=False))

    cob = res.cobertura[0]
    assert set(cob) == set(base.COLUMNAS_COBERTURA)
    assert cob["inicio"] == "2016-08-26" and cob["fin"] == "2020-03-17"
    assert cob["estado"] == "ok" and cob["hora"] and cob["backtest"] and cob["destino"] == "cajon"
    assert "lehman" in cob["nota"] and "sin cobertura" in cob["nota"]

    art = res.tablas["ccnews_articulos"]
    assert len(art) == 6                                            # tope 3 por día x 2 días
    assert "texto" not in art.columns                               # copyright: sin cuerpo en results
    assert set(art["regla"]) == {"meta", "rastreo_meta_posterior"}
    assert art["duplicado"].sum() >= 1                              # C (sindicada de A)
    vol = res.tablas["ccnews_volumen"]
    assert list(vol["cobertura"]) == [False, True, True]

    assert res.documentos == {"ccnews_muestra_rapido": 2}                 # modo rápido: no pisa la muestra
    docs = base.leer_documentos("ccnews_muestra_rapido")
    a = next(d for d in docs if d["institucion"] == "Diario A")
    assert a["extra"]["sindicada_en"] == ["Agregador C"]
    assert res.tablas["ccnews_embudo"]["n"].iloc[-1] == 2

    assert res.timegate and all(t["fecha_publicacion"] for t in res.timegate)
    assert {t["campo_fecha"] for t in res.timegate} >= {"article:published_time (≤ WARC-Date)"}
    f = res.fichas[0]
    assert f.estado_sonda == "ok" and f.cobertura_inicio == "2016-08-26" and f.tema == "noticias_alt"
    json.dumps(f.mediciones, default=str)
    assert all(r["ok"] for r in res.registro) and len(res.registro) == 6


def test_sondear_cachea_y_registra_fallos(sin_red, red_falsa):
    rutas = _rutas_falsas()
    red_falsa(ccnews, rutas)
    dias = [DiaTestigo("covid", date(2020, 3, 16), ""), DiaTestigo("hoy", date(2020, 3, 17), "")]
    ccnews.sondear(base.ConfigSonda(dias=dias, modo_rapido=True, verbose=False))
    # Segunda pasada: todo de la caché del corpus (una URL no prevista haría fallar el falso).
    red_falsa(ccnews, {})
    res = ccnews.sondear(base.ConfigSonda(dias=dias, modo_rapido=True, verbose=False))
    assert all(r["ok"] for r in res.registro) and len(res.tablas["ccnews_articulos"]) == 6
    # Sin caché y con el WARC devolviendo 403: el día falla, la sonda sigue y queda 'parcial'.
    red_falsa(ccnews, dict(rutas, **{"CC-NEWS-202003": (403, b"prohibido")}))
    res = ccnews.sondear(base.ConfigSonda(dias=dias, modo_rapido=True, usar_cache=False, verbose=False))
    assert res.cobertura[0]["estado"] == "parcial"
    assert any(not r["ok"] and "403" in r["error"] for r in res.registro)


@pytest.mark.live
def test_live_ccnews_un_dia(sin_red):
    dias = [DiaTestigo("covid", date(2020, 3, 16), "")]
    res = ccnews.sondear(base.ConfigSonda(dias=dias, modo_rapido=True, verbose=False))
    assert res.cobertura[0]["estado"] in ("ok", "parcial")
    assert len(res.tablas["ccnews_articulos"]) > 0
