"""Tests de ``ingesta.noticias_alt.social`` (sin red salvo el ``live``). Fixtures inventados."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import requests

from ingesta.noticias_alt import base, social
from ingesta.noticias_alt.muestra import DiaTestigo

FX = Path(__file__).parent / "fixtures" / "social"


def _posts() -> list[dict]:
    return json.loads((FX / "truth_archive.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- Truth Social

def test_posts_por_dia_rellena_ceros_y_serie_valida():
    df = social.tabla_posts(_posts())
    assert len(df) == 5 and int(df["vacio"].sum()) == 1
    n = social.posts_por_dia(df)
    assert list(n.index.date) == [date(2023, 3, d) for d in range(8, 13)]
    assert n.tolist() == [1, 0, 3, 0, 1]                     # el post de las 22:15 UTC cuenta el 10
    serie = base.serie_larga(social.serie_truth(n))
    # el recuento del día UTC se cierra a las 24:00 UTC (tras el cierre de NY) -> D+1
    assert (serie["fecha_publicacion"] == serie["fecha_referencia"] + pd.Timedelta(days=1)).all()
    assert serie["disponible_utc"].str.endswith("T00:00:00Z").all()
    assert set(serie["serie"]) == {"truth_posts_dia"}


def test_serie_truth_quita_huecos_del_archivo():
    # 3 días con posts, 10 días a 0 (hueco del archivo), 2 días a 0 (silencio real) y 1 día con posts
    dias = pd.date_range("2024-01-01", periods=16, freq="D")
    n = pd.Series([5, 4, 6] + [0] * 10 + [0, 0, 7], index=dias)
    assert social.rachas_hueco(n).sum() == 12                       # la racha de 12 ceros (10 + 2 seguidos)
    n2 = pd.Series([5, 4, 6] + [0] * 10 + [3, 0, 0, 7], index=pd.date_range("2024-01-01", periods=17, freq="D"))
    m = social.rachas_hueco(n2)
    assert m.sum() == 10 and not m.iloc[-3:].any()                 # dos ceros sueltos = silencio, se quedan
    serie = base.serie_larga(social.serie_truth(n2))
    assert len(serie) == 7 and (serie["valor"] == 0).sum() == 2
    assert pd.Timestamp("2024-01-05") not in set(serie["fecha_referencia"])


def test_tabla_posts_sin_texto():
    df = social.tabla_posts(_posts())
    assert "content" not in df.columns and "TEXTO-PRIVADO" not in df.to_csv(index=False)
    assert social.posts_por_dia(social.tabla_posts([])).empty


# --------------------------------------------------------------------------- FINRA

def test_parse_finra_y_ratio():
    df = social.parse_finra((FX / "finra_dia.txt").read_text())
    assert len(df) == 3                                       # la fila mal formada y el recuento fuera
    m = social.ratio_mercado(df)
    assert m["n_tickers"] == 3 and m["short_ratio"] == pytest.approx(400.5 / 1000)
    with pytest.raises(ValueError):
        social.parse_finra("<Error><Code>AccessDenied</Code></Error>")


def test_http_status_de_errores():
    assert social._http_status(requests.HTTPError("403 Client Error: Forbidden for url: x")) == 403
    assert social._http_status(requests.HTTPError("503 en https://x")) == 503
    assert social._http_status(requests.ConnectionError("Connection aborted")) is None


def test_lag_finra_horario_de_verano():
    # 16-mar-2020: EDT (UTC-4) -> cierre 20:00 UTC; Last-Modified 21:30 GMT -> 1,5 h
    cierre, lm, lag = social._lag_finra(date(2020, 3, 16), "Mon, 16 Mar 2020 21:30:00 GMT")
    assert cierre.startswith("2020-03-16T20:00") and lm.startswith("2020-03-16T21:30") and lag == 1.5
    cierre, _, lag = social._lag_finra(date(2020, 1, 2), None)          # EST (UTC-5)
    assert cierre.startswith("2020-01-02T21:00") and lag is None


def test_descargar_finra_403_es_sin_fichero_y_se_recuerda(sin_red, red_falsa, monkeypatch):
    red_falsa(social, {"CNMSshvol20180702": (403, b"<Error>AccessDenied</Error>")})
    with pytest.raises(social.SinFichero):
        social.descargar_finra(date(2018, 7, 2))
    monkeypatch.setattr(social, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("sin red")))
    with pytest.raises(social.SinFichero, match="caché"):
        social.descargar_finra(date(2018, 7, 2))


def test_candidatos_son_laborables_del_mes():
    c = social._candidatos(date(2019, 1, 1), date(2026, 1, 1))
    assert c == [date(2019, 1, d) for d in (1, 2, 3, 4)]
    assert social._candidatos(date(2026, 10, 1), date(2026, 10, 1)) == [date(2026, 10, 1)]


# --------------------------------------------------------------------------- punta a punta

def test_sondear_sin_red(sin_red, red_falsa, monkeypatch):
    monkeypatch.setattr(social, "robots_permite", lambda url: "reddit" not in url)
    red_falsa(social, {
        "truth_archive.json": "social/truth_archive.json",
        "CNMSshvol20190101": (403, b"<Error>AccessDenied</Error>"),       # festivo: día sin sesión
        "CNMSshvol20080915": (403, b"<Error>AccessDenied</Error>"),       # lehman: antes del CDN
        "CNMSshvol": "social/finra_dia.txt",
        "aaii.com": (403, b"<html>blocked</html>"),
        "dataviz.cnn.io": (418, b"I'm a teapot. You're a bot."),
    })
    cfg = base.ConfigSonda(dias=[DiaTestigo("lehman", date(2008, 9, 15), "x"), DiaTestigo("svb", date(2023, 3, 10), "x")],
                           modo_rapido=True, verbose=False)
    res = social.sondear(cfg)

    cob = pd.DataFrame(res.cobertura)
    assert list(cob.columns) == base.COLUMNAS_COBERTURA
    est = cob.set_index("fuente")["estado"].to_dict()
    assert est == {social.F_TRUTH: "ok", social.F_FINRA: "ok", social.F_REDDIT: "no_probada", social.F_X: "no_probada",
                   social.F_TRENDS: "no_probada", social.F_AAII: "fallo", social.F_CNN: "fallo"}
    assert {f.fuente for f in res.fichas} == set(est) and all(f.estado_sonda == est[f.fuente] for f in res.fichas)

    # Truth: serie, resumen y días testigo; textos solo al corpus
    # modo rápido: salidas con sufijo _rapido (no pisan la pasada completa)
    assert "truth_posts_dia_rapido" in res.series and len(base.leer_serie("truth_posts_dia_rapido")) == 5
    assert base.leer_serie("truth_posts_dia").empty
    r = res.tablas["truth_resumen"].iloc[0]
    assert r["n_posts"] == 5 and r["inicio"].startswith("2023-03-08") and r["ultimo"].startswith("2023-03-12")
    dt = res.tablas["truth_dias_testigo"].set_index("dia")
    assert not dt.loc["lehman", "cobertura"] and dt.loc["svb", "n_posts"] == 3
    docs = base.leer_documentos("truth_posts_testigo_rapido")
    assert res.documentos["truth_posts_testigo_rapido"] == len(docs) == 2       # el post vacío no entra
    assert all(d["fecha_publicacion"] == "2023-03-10" and "TEXTO-PRIVADO" in d["texto"] for d in docs)
    for df in res.tablas.values():
        assert "TEXTO-PRIVADO" not in df.to_csv(index=False)

    # FINRA: 3 meses en modo rápido; el festivo y lehman son «sin cobertura», no fallo
    m = res.tablas["finra_short_muestra"]
    assert {"fecha", "n_tickers", "short_ratio"} <= set(m.columns)
    assert (m["origen"] == "mensual").sum() == 3 and "2019-01-02" in set(m["fecha"])
    assert not m.loc[m["origen"] == "testigo:lehman", "disponible"].iloc[0]
    reg = pd.DataFrame(res.registro)
    sin = reg[reg["error"].str.startswith(social.SIN_COB)]
    assert {"2019-01-01 (mensual)", "2008-09-15 (testigo:lehman)"} <= set(sin["item"])
    s = base.leer_serie("finra_short_ratio_mercado_rapido")
    assert len(s) == 3 and s["valor"].round(4).eq(0.4005).all()
    # fichero ≈18:00 ET, tras el cierre -> D+1 hábil
    assert (s["fecha_publicacion"] == s["fecha_referencia"] + pd.offsets.BDay(1)).all()
    assert s["disponible_utc"].str.contains("T22:00|T23:00").all()        # 18:00 ET sin Last-Modified
    assert cob.set_index("fuente").loc[social.F_FINRA, "inicio"] == "2019-01-02"
    assert "lehman" in cob.set_index("fuente").loc[social.F_FINRA, "nota"]

    # AAII/CNN: una petición educada registrada con su código
    assert reg.loc[reg["fuente"] == social.F_CNN, "error"].str.contains("418").all()
    assert reg.loc[reg["fuente"] == social.F_REDDIT].empty                  # sin OAuth no se consulta


def test_sondear_fuente_que_falla_no_detiene(sin_red, red_falsa, monkeypatch):
    monkeypatch.setattr(social, "robots_permite", lambda url: True)
    red_falsa(social, {"truth_archive.json": b"{no es json", "CNMSshvol": "social/finra_dia.txt",
                       "aaii.com": (403, b""), "dataviz.cnn.io": (418, b"")})
    cfg = base.ConfigSonda(dias=[DiaTestigo("svb", date(2023, 3, 10), "x")], modo_rapido=True, verbose=False)
    res = social.sondear(cfg)
    cob = pd.DataFrame(res.cobertura).set_index("fuente")
    assert cob.loc[social.F_TRUTH, "estado"] == "fallo" and cob.loc[social.F_FINRA, "estado"] == "ok"


@pytest.mark.live
def test_live_finra_un_dia():
    datos, info = social.descargar_finra(date(2020, 3, 16), usar_cache=False)
    m = social.ratio_mercado(social.parse_finra(datos.decode()))
    assert m["n_tickers"] > 5000 and 0.2 < m["short_ratio"] < 0.8
