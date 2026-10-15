"""Tests del módulo ``insiders`` (Form 4 de SEC EDGAR y datasets DERA). Sin red."""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from ingesta.noticias_alt import base, insiders
from ingesta.noticias_alt.muestra import DiaTestigo

FX = Path(__file__).parent / "fixtures" / "insiders"
SIN_XML = (b"<SEC-DOCUMENT>0009000002-20-000003.txt : 20200316\n<ACCEPTANCE-DATETIME>20200316120000\n"
           b"ACCESSION NUMBER:\t\t0009000002-20-000003\nFILED AS OF DATE:\t\t20200316\n<TEXT>formulario en texto libre"
           b"</TEXT>\n</SEC-DOCUMENT>\n")


def test_url_indice_diario():
    assert insiders.url_indice_diario(date(2008, 9, 15)).endswith("daily-index/2008/QTR3/form.20080915.idx")
    assert insiders.url_indice_diario(date(2020, 3, 16)).endswith("daily-index/2020/QTR1/form.20200316.idx")
    assert "/QTR4/" in insiders.url_indice_diario(date(2025, 12, 31))


def test_parse_indice_diario():
    texto = (FX / "form.idx").read_text(encoding="utf-8")
    f4 = insiders.parse_indice_diario(texto)
    assert [f["form"] for f in f4] == ["4", "4", "4/A", "4"]
    assert f4[0]["empresa"] == "Empresa Inventada Uno Inc" and f4[0]["cik"] == "9000001"
    assert f4[0]["fecha"] == date(2020, 3, 16) and f4[0]["ruta"].endswith("0009000001-20-000002.txt")
    assert f4[3]["empresa"] == "Gamma Falsa Holdings, Inc."
    assert len(insiders.parse_indice_diario(texto, None)) == 5


def test_parse_cabecera_y_form4_compra():
    texto = (FX / "form4_compra.txt").read_text(encoding="utf-8")
    cab = insiders.parse_cabecera(texto)
    assert cab["aceptacion"] == datetime(2020, 3, 16, 16, 30, 15)
    assert cab["fecha_presentacion"] == date(2020, 3, 16) and cab["accession"] == "0009000001-20-000002"
    d = insiders.parse_form4(texto)
    assert d["ticker"] == "EIU" and d["emisor"] == "EMPRESA INVENTADA UNO INC"
    assert d["insider"] == "PEREZ FICTICIO JUAN" and d["cargo"] == "director; directivo: CEO & Chair"
    (t,) = d["transacciones"]
    assert (t["codigo"], t["fecha"], t["acciones"], t["precio"], t["derivado"]) == ("P", date(2020, 3, 12), 2500.0, 32.28, False)


def test_parse_form4_venta_derivados_y_bordes():
    d = insiders.parse_form4((FX / "form4_venta.txt").read_text(encoding="utf-8"))
    assert d["n_insiders"] == 2 and d["cargo"] == "director"
    nd, der = d["transacciones"]
    assert nd["codigo"] == "S" and nd["fecha"] == date(2020, 3, 11)  # fecha con zona «-05:00»
    assert nd["precio"] is None                                         # precio solo con nota al pie
    assert der["derivado"] and der["codigo"] == "M"
    assert insiders.parse_form4(SIN_XML.decode()) == {}


# --------------------------------------------------------------------------- punta a punta

@pytest.fixture
def red_sec(red_falsa):
    return red_falsa(insiders, {"daily-index/": "insiders/form.idx",
                                "0009000001-20-000002": "insiders/form4_compra.txt",
                                "0009000002-20-000003": SIN_XML,
                                "0009000003-20-000005": "insiders/form4_venta.txt",
                                "insider-transactions-data-sets": "insiders/dera.html"})


def test_sondear_punta_a_punta_sin_red(sin_red, red_sec):
    dias = [DiaTestigo("lehman", date(2008, 9, 15), ""), DiaTestigo("covid", date(2020, 3, 16), ""),
            DiaTestigo("hoy", date(2026, 10, 14), "")]
    res = insiders.sondear(base.ConfigSonda(modo_rapido=True, verbose=False, dias=dias))

    cob = pd.DataFrame(res.cobertura)
    assert list(cob.columns) == base.COLUMNAS_COBERTURA and cob["fuente"].is_unique
    f4 = cob.set_index("fuente").loc["sec_form4_insiders"]
    assert f4["estado"] == "ok" and f4["hora"] and f4["backtest"]
    dera = cob.set_index("fuente").loc["sec_form345_datasets"]
    assert (dera["inicio"], dera["fin"]) == ("2006-01-01", "2026-06-30")

    idx = res.tablas["insiders_indice"]
    assert len(idx) == 3 and (idx["n_form4"] == 3).all() and (idx["n_form4a"] == 1).all()

    ops = res.tablas["insiders_operaciones"]
    assert set(ops["codigo"]) == {"P", "S", "M"}
    # TimeGate: aceptada a las 16:30 ET (tras el cierre) -> día hábil siguiente; a las 09:01 -> ese día.
    assert (ops["fecha_presentacion"] == date(2020, 3, 16)).all()
    p = ops[ops["codigo"] == "P"].iloc[0]
    assert p["fecha_publicacion"] == date(2020, 3, 17) and p["disponible_utc"] == "2020-03-16T20:30:15+00:00"
    assert (ops.loc[ops["codigo"] == "S", "fecha_publicacion"] == date(2020, 3, 16)).all()
    assert p["lag_dias"] == 5 and p["valor_usd"] == round(2500 * 32.28, 2) and p["ticker"] == "EIU"
    fil = res.tablas["insiders_form4"]
    assert len(fil) == 9 and (~fil["xml"]).sum() == 3   # 3 por día, uno sin XML
    assert res.timegate and all(t["lag_h"] >= 0 for t in res.timegate)
    assert {f.fuente for f in res.fichas} == {"sec_form4_insiders", "sec_form345_datasets"}
    ficha = next(f for f in res.fichas if f.fuente == "sec_form4_insiders")
    assert ficha.mediciones["form4_con_compra_p_y_ticker"] == 3
    assert (base.corpus("crudo", "insiders") / "form4" / "0009000001-20-000002.txt").exists()


def test_publicacion_corte_16_et():
    # antes del cierre -> ese día; tras el cierre -> hábil siguiente (viernes -> lunes); fin de semana -> lunes
    assert insiders.publicacion(datetime(2020, 3, 16, 15, 59), date(2020, 3, 16))[0] == date(2020, 3, 16)
    assert insiders.publicacion(datetime(2020, 3, 16, 16, 0), date(2020, 3, 16))[0] == date(2020, 3, 17)
    assert insiders.publicacion(datetime(2020, 3, 20, 17, 45), date(2020, 3, 23))[0] == date(2020, 3, 23)
    assert insiders.publicacion(datetime(2020, 3, 21, 10, 0), date(2020, 3, 23))[0] == date(2020, 3, 23)
    d, disp = insiders.publicacion(datetime(2020, 1, 6, 10, 0), date(2020, 1, 6))
    assert disp == "2020-01-06T15:00:00+00:00"                          # EST (UTC-5)
    assert insiders.publicacion(None, date(2020, 3, 16)) == (date(2020, 3, 17), None)   # sin hora: prudente


def test_cache_truncada_se_vuelve_a_bajar(sin_red, red_falsa):
    red_falsa(insiders, {"daily-index/": "insiders/form.idx"})
    cfg, res = base.ConfigSonda(verbose=False), base.Resultado("insiders")
    ruta = base.corpus("crudo", "insiders") / "x.idx"
    ruta.write_bytes(b"truncado")
    datos = base.get_cacheado(res, cfg, "f", "i", "https://www.sec.gov/Archives/edgar/daily-index/x", ruta,
                              getter=insiders.get, validar=lambda b: b.startswith(b"Description") or 1 / 0)
    assert datos and datos.startswith(b"Description") and ruta.read_bytes() == datos
    assert not list(ruta.parent.glob("*.tmp"))


def test_dia_sin_indice_es_sin_cobertura(sin_red, red_falsa):
    red_falsa(insiders, {"daily-index/": (404, b"Not Found"), "insider-transactions-data-sets": (403, b"")})
    res = insiders.sondear(base.ConfigSonda(modo_rapido=True, verbose=False,
                                            dias=[DiaTestigo("svb", date(2023, 3, 10), "")]))
    idx = res.tablas["insiders_indice"]
    assert idx.iloc[0]["nota"].startswith("sin cobertura") and idx.iloc[0]["n_form4"] == 0
    cob = {c["fuente"]: c for c in res.cobertura}
    assert cob["sec_form4_insiders"]["estado"] == "fallo" and cob["sec_form345_datasets"]["estado"] == "fallo"


def test_festivo_en_busqueda_no_es_fallo(sin_red, red_falsa):
    red_falsa(insiders, {"daily-index/": (403, b"Forbidden")})
    cfg = base.ConfigSonda(verbose=False)
    res = base.Resultado("insiders")
    m = insiders._Muestreo(cfg, res)
    assert m.indice(date(2016, 11, 11), "elecciones_2016", opcional=True) is None
    assert res.registro == [] and "festivo" in m.filas_idx[0]["nota"]
    assert insiders._dias_habiles_siguientes(date(2023, 3, 10), 3) == [date(2023, 3, 13), date(2023, 3, 14),
                                                                       date(2023, 3, 15)]


@pytest.mark.live
def test_live_indice_y_form4():
    fs = insiders.listar_form4(date(2020, 3, 16))
    assert len([f for f in fs if f["form"] == "4"]) > 1000
    d = insiders.leer_form4(fs[0]["ruta"])
    assert d["fecha_presentacion"] == date(2020, 3, 16) and d["transacciones"] is not None
