"""Tests de ``ingesta.noticias_alt.gdelt`` (sin red; fixtures inventados con el formato real)."""
from __future__ import annotations

import io
import zipfile
from datetime import UTC, date, datetime

import pandas as pd
import pytest

from ingesta.noticias_alt import base, gdelt
from ingesta.noticias_alt.muestra import DiaTestigo

FX = __import__("pathlib").Path(__file__).parent / "fixtures" / "gdelt"


def _zip(nombre: str, fichero: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(nombre, (FX / fichero).read_text(encoding="utf-8"))
    return buf.getvalue()


# --------------------------------------------------------------------------- URLs y muestreo

def test_urls():
    assert gdelt.url_gkg(datetime(2020, 3, 16, 12, 44)).endswith("/gdeltv2/20200316123000.gkg.csv.zip")
    assert gdelt.url_eventos2(datetime(2020, 3, 16, 0, 5)).endswith("/gdeltv2/20200316000000.export.CSV.zip")
    assert gdelt.url_eventos1(date(2020, 3, 16)).endswith("/events/20200316.export.CSV.zip")
    assert gdelt.url_eventos1(date(2008, 9, 15)).endswith("/events/200809.zip")      # mensual
    assert gdelt.url_eventos1(date(2004, 1, 2)).endswith("/events/2004.zip")         # anual


def test_horas_muestra_respeta_cobertura_y_futuro():
    assert gdelt.horas_muestra(date(2008, 9, 15), 6) == []                        # antes de GKG 2.1
    hs = gdelt.horas_muestra(date(2020, 3, 16), 6)
    assert [t.hour for t in hs] == [0, 4, 8, 12, 16, 20]
    ahora = datetime(2026, 10, 14, 9, 0, tzinfo=UTC)
    hs = gdelt.horas_muestra(date(2026, 10, 14), 6, ahora=ahora)
    assert [t.hour for t in hs] == [0, 4, 8] or [t.hour for t in hs] == [0, 4]
    assert all(t <= ahora for t in hs)
    assert gdelt.horas_muestra(date(2015, 2, 18), 2) == []                         # empieza a las 23:00


def test_a_utc():
    assert gdelt.a_utc("20200316T143000Z") == datetime(2020, 3, 16, 14, 30, tzinfo=UTC)
    assert gdelt.a_utc("20130203") == datetime(2013, 2, 3, tzinfo=UTC)
    assert gdelt.a_utc("") is None and gdelt.a_utc("2020") is None


# --------------------------------------------------------------------------- GKG

def test_parse_gkg_y_resumen():
    filas = gdelt.parse_gkg((FX / "gkg_sintetico.csv").read_text(encoding="utf-8"))
    assert len(filas) == 3                                      # la fila rota se descarta
    f0 = filas[0]
    assert f0["fecha"] == datetime(2020, 3, 16, 12, tzinfo=UTC)
    assert f0["tono"] == -4.5 and f0["titulo"] == "Futures slide as markets open lower"
    assert "ECON_STOCKMARKET" in f0["temas"] and f0["pub"] == datetime(2020, 3, 16, 10, tzinfo=UTC)
    assert filas[2]["titulo"] is None and filas[2]["pub"] is None
    r = gdelt.resumen_gkg(filas)
    assert r["n_articulos"] == 3 and r["n_medios"] == 2
    assert r["tono_medio"] == pytest.approx((-4.5 - 1.5 + 2.0) / 3, abs=1e-3)
    assert r["pct_con_titular"] == pytest.approx(66.7)
    # retraso medio -> GDELT: 2 h y 0,25 h -> mediana 1,125 h
    assert r["lag_pub_captura_mediana_h"] == pytest.approx(1.12, abs=0.01)


def test_temas_top_y_tono():
    filas = gdelt.parse_gkg((FX / "gkg_sintetico.csv").read_text(encoding="utf-8"))
    top = gdelt.temas_top(filas, 2, gdelt.PREFIJOS_ECON)
    assert top[0]["tema"] == "ECON_STOCKMARKET" and top[0]["n"] == 2
    assert top[0]["tono_medio"] == pytest.approx(-3.0)
    assert all(t["tema"].startswith(gdelt.PREFIJOS_ECON) for t in top)
    ag = gdelt.tono_agregado(filas, tema="ECON_STOCKMARKET")
    assert ag == {"2020-03-16": {"n": 2, "tono": -3.0}}


def test_titulares_gkg_fecha_es_la_captura():
    filas = gdelt.parse_gkg((FX / "gkg_sintetico.csv").read_text(encoding="utf-8"))
    docs = gdelt.titulares_gkg(filas)
    assert len(docs) == 2                                       # solo filas con titular
    assert docs[0]["fecha_publicacion"] == "2020-03-16" and docs[0]["rag"] == "cajon"
    assert "texto" not in docs[0]                               # nunca cuerpo
    assert docs[0]["extra"]["captura_utc"].startswith("2020-03-16T12:00")


# --------------------------------------------------------------------------- Events

def test_parse_eventos_2_y_timegate():
    filas = gdelt.parse_eventos((FX / "eventos2_sintetico.csv").read_text(encoding="utf-8"))
    assert len(filas) == 4 and {f["version"] for f in filas} == {"2.0"}
    assert all(f["dateadded"] == datetime(2020, 3, 16, 12, tzinfo=UTC) for f in filas)
    r = gdelt.resumen_eventos(filas)
    assert r["n"] == 4 and r["pct_sqldate_anterior"] == 50.0
    assert r["lag_dias_mediana"] == 0.5 and r["lag_dias_max"] == 366 and r["pct_con_url"] == 100.0
    ej = gdelt.ejemplos_timegate_eventos(filas, "covid")
    assert ej[-1]["fecha_hecho"] == "2019-03-16"                # el hecho más antiguo
    assert ej[-1]["fecha_publicacion"].startswith("2020-03-16T12:00")  # publicación = DATEADDED
    assert ej[-1]["lag_h"] == pytest.approx(366 * 24 + 12)


def test_parse_eventos_historico_sin_url():
    filas = gdelt.parse_eventos((FX / "eventos1_historico_sintetico.csv").read_text(encoding="utf-8"))
    assert {f["version"] for f in filas} == {"1.0_historico"}
    lehman = [f for f in filas if f["sqldate"].date() == date(2008, 9, 15)]
    r = gdelt.resumen_eventos(lehman)
    assert r["n"] == 3 and r["pct_sqldate_anterior"] == 100.0 and r["pct_con_url"] == 0.0
    assert r["lag_dias_mediana"] == (date(2013, 2, 3) - date(2008, 9, 15)).days


def test_resumen_eventos_vacio():
    assert gdelt.resumen_eventos([])["n"] == 0


# --------------------------------------------------------------------------- DOC API

def test_articulos_a_documentos():
    import json

    arts = json.loads((FX / "doc_artlist.json").read_text(encoding="utf-8"))["articles"]
    docs = gdelt.articulos_a_documentos(arts, "stock market", "covid")
    assert len(docs) == 3                                       # el artículo sin seendate se descarta
    assert {d["idioma"] for d in docs} == {"en", "es"}
    assert all(d["fecha_publicacion"] == "2020-03-16" and d["rag"] == "cajon" for d in docs)
    assert docs[0]["extra"]["seendate_utc"].startswith("2020-03-16T14:30")


def test_tono_a_serie_publica_d_mas_5():
    import json

    j = json.loads((FX / "doc_timelinetone.json").read_text(encoding="utf-8"))
    s = base.serie_larga(gdelt.tono_a_serie(j, "gdelt_tono:economy", "economy"))
    assert len(s) == 4
    # el índice DOC va ≈4 días por detrás: el tono de D no se conoce hasta D+5
    assert ((s["fecha_publicacion"] - s["fecha_referencia"]).dt.days == 5).all()
    assert s["metodo_publicacion"].str.contains("vintage único").all()
    assert s.loc[s["fecha_referencia"] == "2020-03-16", "valor"].item() == pytest.approx(-2.41)


def test_doc_api_limite_reintenta_y_falla(monkeypatch, red_falsa):
    llamadas = []
    red_falsa(gdelt, {"mode=artlist": (429, (FX / "doc_limite.txt").read_bytes())})
    original = gdelt.get
    monkeypatch.setattr(gdelt, "get", lambda url, **kw: (llamadas.append(1), original(url, **kw))[1])
    monkeypatch.setattr(gdelt.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="429"):
        gdelt.doc_api("x", datetime(2020, 3, 16), datetime(2020, 3, 17), intentos=3, espera_429=0)
    assert len(llamadas) == 3


def test_doc_api_texto_de_limite_con_200(red_falsa, monkeypatch):
    red_falsa(gdelt, {"mode=artlist": (FX / "doc_limite.txt").read_bytes()})
    monkeypatch.setattr(gdelt.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError):
        gdelt.doc_api("x", datetime(2020, 3, 16), datetime(2020, 3, 17), intentos=2, espera_429=0)


def test_doc_api_respuesta_vacia(red_falsa):
    red_falsa(gdelt, {"mode=artlist": b"  "})
    j, crudo = gdelt.doc_api("x", datetime(2020, 3, 16), datetime(2020, 3, 17))
    assert j == {}


# --------------------------------------------------------------------------- punta a punta

def _rutas(doc_ok: bool = True) -> dict:
    return {
        "lastupdate.txt": "gdelt/lastupdate.txt",
        "events/index.html": "gdelt/events_index.html",
        ".gkg.csv.zip": _zip("x.gkg.csv", "gkg_sintetico.csv"),
        ".export.CSV.zip": _zip("x.export.CSV", "eventos2_sintetico.csv"),
        "events/200809.zip": _zip("200809.csv", "eventos1_historico_sintetico.csv"),
        "mode=artlist": "gdelt/doc_artlist.json" if doc_ok else (429, b"Please limit requests"),
        "mode=timelinetone": "gdelt/doc_timelinetone.json",
    }


DIAS = [DiaTestigo("lehman", date(2008, 9, 15), "x"), DiaTestigo("elecciones_2016", date(2016, 11, 9), "x"),
        DiaTestigo("covid", date(2020, 3, 16), "x")]


def test_sondear_punta_a_punta_sin_red(sin_red, red_falsa):
    red_falsa(gdelt, _rutas())
    cfg = base.ConfigSonda(dias=DIAS, max_items=3, verbose=False,
                           extra={"espera_429": 0, "gkg_por_dia": 2, "doc_consultas": 3})
    res = gdelt.sondear(cfg)
    assert all(r["ok"] for r in res.registro), [r for r in res.registro if not r["ok"]]
    assert {c["fuente"] for c in res.cobertura} == {gdelt.F_GKG, gdelt.F_EV, gdelt.F_DOC}
    assert all(set(c) == set(base.COLUMNAS_COBERTURA) for c in res.cobertura)
    cob = {c["fuente"]: c for c in res.cobertura}
    assert cob[gdelt.F_GKG]["inicio"] == "2015-02-18" and cob[gdelt.F_GKG]["fin"] == "2026-10-14"
    assert cob[gdelt.F_EV]["inicio"] == "1979-01-01"
    assert cob[gdelt.F_DOC]["inicio"] == "2017-01-01" and cob[gdelt.F_DOC]["estado"] == "ok"
    g = res.tablas["gdelt_gkg_muestra"]
    assert g.loc[g["clave"] == "lehman", "nota"].str.contains("sin cobertura").all()
    assert (g.dropna(subset=["n_articulos"])["n_articulos"] == 3).all()
    assert len(g.dropna(subset=["n_articulos"])) == 4             # 2 ficheros × 2 días cubiertos
    assert set(res.tablas["gdelt_temas"]["grupo"]) == {"todos", "economia"}
    assert {"dia", "hora_utc", "tono_medio", "n"} <= set(res.tablas["gdelt_tono_intradia"].columns)
    ev = res.tablas["gdelt_eventos_lag"].set_index("clave")
    assert ev.loc["lehman", "version"] == "1.0_historico" and ev.loc["lehman", "n"] == 3
    assert ev.loc["covid", "pct_sqldate_anterior"] == 50.0
    dm = res.tablas["gdelt_doc_muestra"]
    assert dm.loc[dm["clave"] == "elecciones_2016", "nota"].str.contains("sin cobertura").all()
    assert (dm.dropna(subset=["consulta"])["n"] == 4).all() and len(dm.dropna(subset=["consulta"])) == 3
    assert res.series == ["gdelt_tono_economia"]
    s = base.leer_serie("gdelt_tono_economia")
    assert len(s) == 4 and ((s["fecha_publicacion"] - s["fecha_referencia"]).dt.days == 5).all()
    assert res.documentos["gdelt_doc_titulares"] == 3 and res.documentos["gdelt_gkg_titulares"] == 2
    assert {t["fuente"] for t in res.timegate} == {gdelt.F_GKG, gdelt.F_EV, gdelt.F_DOC}
    assert all(set(t) == set(base.COLUMNAS_TIMEGATE) for t in res.timegate)
    fichas = {f.fuente: f for f in res.fichas}
    assert set(fichas) == {gdelt.F_GKG, gdelt.F_EV, gdelt.F_DOC}
    assert fichas[gdelt.F_EV].mediciones["volumen_events1_gb"] == pytest.approx(1.2 + (9.0 + 32.9) / 1024, abs=0.1)
    assert all(f.tema == "noticias_alt" for f in fichas.values())
    # la caché del corpus se reutiliza: segunda pasada sin red
    red_falsa(gdelt, {"lastupdate.txt": "gdelt/lastupdate.txt"})
    res2 = gdelt.sondear(cfg)
    assert all(r["ok"] for r in res2.registro)
    pd.testing.assert_frame_equal(res2.tablas["gdelt_eventos_lag"], res.tablas["gdelt_eventos_lag"])


def test_modo_rapido_es_rapido_y_no_pisa(sin_red, red_falsa, monkeypatch):
    pedidas: list[str] = []
    falso = red_falsa(gdelt, _rutas())
    monkeypatch.setattr(gdelt, "get",
                        lambda url, **kw: (pedidas.append(url + str(kw.get("params") or "")), falso(url, **kw))[1])
    cfg = base.ConfigSonda(dias=DIAS, modo_rapido=True, verbose=False, extra={"espera_429": 0})
    res = gdelt.sondear(cfg)
    assert all(r["ok"] for r in res.registro)
    g = res.tablas["gdelt_gkg_muestra"].dropna(subset=["n_articulos"])
    assert len(g) == 2                                             # 1 fichero GKG por día cubierto
    assert not any("200809.zip" in u for u in pedidas)            # Events 1.0 histórico solo si está en caché
    dm = res.tablas["gdelt_doc_muestra"].dropna(subset=["consulta"])
    assert len(dm) == 1                                            # 1 consulta, 1 día con cobertura
    assert sum("'artlist'" in u for u in pedidas) == 1
    # salidas con sufijo: no pisan las de la pasada completa
    assert res.series == ["gdelt_tono_economia_rapido"]
    assert set(res.documentos) == {"gdelt_doc_titulares_rapido", "gdelt_gkg_titulares_rapido"}
    assert base.leer_serie("gdelt_tono_economia").empty


def test_cache_zip_truncada_se_vuelve_a_bajar(sin_red, red_falsa):
    red_falsa(gdelt, _rutas())
    cfg, res = base.ConfigSonda(verbose=False), base.Resultado("gdelt")
    url = "https://data.gdeltproject.org/gdeltv2/20200316120000.gkg.csv.zip"
    ruta = base.corpus("crudo", "gdelt") / "20200316120000.gkg.csv.zip"
    ruta.write_bytes(_zip("x.gkg.csv", "gkg_sintetico.csv")[:40])     # zip cortado a medias
    datos = gdelt.bajar(res, cfg, gdelt.F_GKG, "gkg", url, cache=ruta.name)
    assert gdelt.parse_gkg(gdelt.leer_zip(datos)) and ruta.read_bytes() == datos
    assert res.registro[-1]["ok"] and res.registro[-1]["bytes"] == len(datos)


def test_sondear_doc_bloqueado_no_detiene(sin_red, red_falsa, monkeypatch):
    red_falsa(gdelt, _rutas(doc_ok=False) | {"mode=timelinetone": (429, b"Please limit requests")})
    monkeypatch.setattr(gdelt.time, "sleep", lambda s: None)
    cfg = base.ConfigSonda(dias=DIAS[2:], max_items=3, verbose=False,
                           extra={"espera_429": 0, "gkg_por_dia": 2, "doc_consultas": 3})
    res = gdelt.sondear(cfg)
    assert base.estado(res.registro, gdelt.F_DOC) == "fallo"
    assert base.estado(res.registro, gdelt.F_GKG) == "ok"
    assert "gdelt_doc_titulares" not in res.documentos and res.series == []
    assert {c["fuente"]: c["estado"] for c in res.cobertura}[gdelt.F_DOC] == "fallo"
    # cortacircuitos: tono + 2 consultas fallan (3 seguidos) y la tercera consulta ya no se pide
    dm = res.tablas["gdelt_doc_muestra"]
    assert dm["nota"].str.startswith("omitida").sum() == 1
    assert sum(1 for r in res.registro if r["fuente"] == gdelt.F_DOC) == 3
    assert {f.fuente: f for f in res.fichas}[gdelt.F_DOC].mediciones["omitidas_por_429"] == 1


@pytest.mark.live
def test_live_gdelt(sin_red):
    cfg = base.ConfigSonda(dias=[DiaTestigo("covid", date(2020, 3, 16), "x")], modo_rapido=True, verbose=False,
                           max_items=1)
    res = gdelt.sondear(cfg)
    assert base.estado(res.registro, gdelt.F_GKG) == "ok"
    assert res.tablas["gdelt_gkg_muestra"]["n_articulos"].max() > 100


def test_titulares_sin_entidades_html():
    linea = "\t".join(["1", "20200316120000", "1", "m.com", "https://m.com/a"] + [""] * 10 + ["-1.0,1,2"]
                      + [""] * 10 + ["<PAGE_TITLE>Markets &#x2013; live &amp; more</PAGE_TITLE>"])
    (f,) = gdelt.parse_gkg(linea)
    assert f["titulo"] == "Markets \u2013 live & more"
