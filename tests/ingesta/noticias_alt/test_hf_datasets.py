"""Datasets de Hugging Face: parseo de metadatos, regla de fecha y sonda sin red (JSON inventados)."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ingesta.noticias_alt import base, hf_datasets as hf

FX = Path(__file__).parent / "fixtures" / "hf_datasets"


def _j(nombre):
    return json.loads((FX / nombre).read_text(encoding="utf-8"))


def test_info_hub_suma_bytes_y_licencia():
    inf = hf.info_hub(_j("hub_ccnews.json"))
    assert inf["bytes"] == 5_000_001_000 and inf["ficheros"] == 3 and inf["gated"] is False
    assert inf["configs_tarjeta"] == ["2016", "2017"] and inf["extensiones"] == {"parquet": 2, "md": 1}
    gated = hf.info_hub(_j("hub_multisource.json"))
    assert gated["gated"] == "auto" and gated["licencia"] == "other"
    assert hf.info_hub(_j("hub_fnspid.json"))["licencia"] is None          # cardData nulo


def test_info_size_por_config_y_fallido():
    tam = hf.info_size(_j("size_ccnews.json"))
    assert tam["filas"] == 5000 and tam["por_config"] == {"2016": 1500, "2017": 3500}
    assert hf.info_size(_j("size_fnspid.json")) == dict(filas=None, por_config={}, parquet_bytes=None)


def test_filas_e_idiomas():
    assert len(hf.filas_primeras(_j("first_rows_ccnews.json"))) == 2
    assert hf.idiomas_estadisticas(_j("statistics_ccnews.json")) == {"en": 60, "es": 30, "fr": 10}
    assert hf.idiomas_estadisticas({}) == {}


def test_fecha_timegate_ccnews():
    pub, campo, lag = hf.fecha_timegate_ccnews("2017-03-01", "2017-03-02T10:00:00+00:00")
    assert pub == datetime(2017, 3, 1, tzinfo=UTC) and campo.startswith("published_date") and lag == 34.0
    pub, campo, _ = hf.fecha_timegate_ccnews("2017-03-05", "2017-03-02T10:00:00+00:00")   # posterior: rastreo
    assert pub == datetime(2017, 3, 2, 10, tzinfo=UTC) and "posterior" in campo
    pub, campo, lag = hf.fecha_timegate_ccnews(None, "2017-03-02T10:00:00Z")
    assert "sin published_date" in campo and lag is None


def _rutas():
    return {
        "api/datasets/Zihan1004/FNSPID": "hf_datasets/hub_fnspid.json",
        "size?dataset=Zihan1004/FNSPID": "hf_datasets/size_fnspid.json",
        "first-rows?dataset=Zihan1004/FNSPID": "hf_datasets/first_rows_fnspid.json",
        "api/datasets/stanford-oval/ccnews": "hf_datasets/hub_ccnews.json",
        "size?dataset=stanford-oval/ccnews": "hf_datasets/size_ccnews.json",
        "first-rows?dataset=stanford-oval/ccnews": "hf_datasets/first_rows_ccnews.json",
        "statistics?dataset=stanford-oval/ccnews": "hf_datasets/statistics_ccnews.json",
        "api/datasets/KrossKinetic": "hf_datasets/hub_bloomberg.json",
        "dataset=KrossKinetic": (500, b"error interno"),
        "api/datasets/ashraq": "hf_datasets/hub_reuters.json",
        "size?dataset=ashraq": "hf_datasets/size_reuters.json",
        "first-rows?dataset=ashraq": "hf_datasets/first_rows_reuters.json",
        "api/datasets/Brianferrell787": "hf_datasets/hub_multisource.json",
    }


def test_sondear_sin_red_modo_rapido(sin_red, red_falsa):
    red_falsa(hf, _rutas())
    res = hf.sondear(base.ConfigSonda(modo_rapido=True, verbose=False))
    cob = {c["fuente"]: c for c in res.cobertura}
    assert set(cob) == {d.fuente for d in hf.DATASETS}
    assert all(set(c) == set(base.COLUMNAS_COBERTURA) for c in res.cobertura)
    assert cob["hf_ccnews"]["estado"] == "ok" and cob["hf_ccnews"]["idiomas"].startswith("en,es")
    assert (cob["hf_ccnews"]["inicio"], cob["hf_ccnews"]["fin"]) == ("2016-08-24", "2017-12-31")
    assert cob["hf_fnspid"]["estado"] == "parcial"                       # el viewer no da filas
    assert cob["hf_bloomberg_2006_2013"]["estado"] == "parcial"          # first-rows con 500
    assert cob["hf_financial_news_multisource"]["estado"] == "parcial"   # gated
    assert cob["hf_reuters_cnbc_2018"]["backtest"] is False

    tab = res.tablas["hf_datasets"]
    assert {"dataset", "inicio", "fin", "filas", "gb", "licencia", "gated"} <= set(tab.columns)
    assert not any(c.startswith("_") for c in tab.columns)
    assert tab.set_index("fuente").loc["hf_ccnews", "gb"] == 5.0
    assert tab.set_index("fuente").loc["hf_reuters_cnbc_2018", "dominios_top"] == "reuters.example"
    anios = res.tablas["hf_filas_anio"]
    assert list(anios["anio"]) == [2016, 2017] and anios["filas"].sum() == 5000
    assert res.tablas["hf_ccnews_idiomas"]["pct"].sum() == pytest.approx(100)

    tg = [t for t in res.timegate if t["fuente"] == "hf_ccnews"]
    assert len(tg) == 2 and tg[1]["fecha_publicacion"].startswith("2017-03-02T10")
    assert any(t["fuente"] == "hf_fnspid" and t["campo_fecha"].startswith("Date") for t in res.timegate)
    fichas = {f.fuente: f for f in res.fichas}
    assert fichas["hf_financial_news_multisource"].autenticacion.startswith("cuenta HF")
    assert not any(r["fuente"] == "hf_financial_news_multisource" and "datasets-server" in r["url"]
                   for r in res.registro)                               # gated: no se toca el viewer
    assert not res.documentos                                           # solo metadatos
    # Los crudos (con texto de las primeras filas) van al corpus, no a results.
    assert any((sin_red / "corpus" / "noticias_alt" / "crudo" / "hf_datasets").iterdir())


def test_sondear_un_dataset_que_falla_no_para(sin_red, red_falsa):
    rutas = _rutas()
    rutas["api/datasets/Zihan1004/FNSPID"] = (404, b"no existe")
    red_falsa(hf, rutas)
    res = hf.sondear(base.ConfigSonda(modo_rapido=True, verbose=False))
    cob = {c["fuente"]: c for c in res.cobertura}
    assert cob["hf_fnspid"]["estado"] == "fallo" and "error" in cob["hf_fnspid"]["nota"]
    assert cob["hf_ccnews"]["estado"] == "ok"


@pytest.mark.live
def test_live_hf_metadatos(sin_red):
    res = hf.sondear(base.ConfigSonda(modo_rapido=True, usar_cache=False, verbose=False))
    cob = {c["fuente"]: c for c in res.cobertura}
    assert cob["hf_ccnews"]["estado"] == "ok"
    assert res.tablas["hf_filas_anio"]["filas"].sum() > 1e8
