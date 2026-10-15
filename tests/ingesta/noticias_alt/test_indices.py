"""Tests de ``noticias_alt.indices`` (GPR, EPU, CBOE): parseo, reglas de publicación y sonda sin red."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from ingesta.noticias_alt import base, indices
from ingesta.noticias_alt.muestra import DiaTestigo


def _hay(modulo: str) -> bool:
    try:
        __import__(modulo)
    except ImportError:
        return False
    return True


# GPR (.xls) y EPU global (.xlsx) necesitan xlrd y openpyxl (requirements/ingesta-noticias_alt.txt); la CI
# solo instala [dev]: sin ellas, estos tests se saltan en vez de fallar.
requiere_xlrd = pytest.mark.skipif(not _hay("xlrd"), reason="falta xlrd (requirements/ingesta-noticias_alt.txt)")
requiere_openpyxl = pytest.mark.skipif(not _hay("openpyxl"), reason="falta openpyxl (requirements/ingesta-noticias_alt.txt)")

FX = Path(__file__).parent / "fixtures" / "indices"

RUTAS = {
    "data_gpr_daily_recent_20230313.xls": "indices/gpr_vintage_diario.xlsx",
    "data_gpr_daily_recent_20230306.xls": "indices/gpr_vintage_diario.xlsx",
    "data_gpr_export_202304.xls": "indices/gpr_vintage_mensual.xlsx",
    "data_gpr_export_202303.xls": "indices/gpr_vintage_mensual.xlsx",
    "data_gpr_daily_recent.xls": "indices/gpr_diario.xlsx",
    "data_gpr_export.xls": "indices/gpr_mensual.xlsx",
    "api.github.com": "indices/gpr_vintages.json",
    "All_Daily_Policy_Data.csv": "indices/epu_diario.csv",
    "Global_Policy_Uncertainty_Data.xlsx": "indices/epu_global.xlsx",
    "SKEW_History.csv": "indices/cboe_skew.csv",
    "VIX_History.csv": "indices/cboe_vix.csv",
    "totalpc.csv": "indices/cboe_totalpc.csv",
    "equitypc.csv": "indices/cboe_equitypc.csv",
    "indexpc.csv": "indices/cboe_equitypc.csv",
    "_daily_options": "indices/cboe_diario.json",
}


def _cfg(**kw) -> base.ConfigSonda:
    dias = [DiaTestigo("lehman", date(2008, 9, 15), ""), DiaTestigo("svb", date(2023, 3, 10), ""),
            DiaTestigo("hoy", date(2023, 3, 20), "")]
    return base.ConfigSonda(dias=dias, verbose=False, **kw)


@pytest.fixture
def sin_robots(monkeypatch):
    monkeypatch.setattr(indices, "robots_permite", lambda url: True)


# --------------------------------------------------------------------------- parseo

@requiere_xlrd
def test_leer_gpr_diario_quita_metadatos():
    df = indices.leer_gpr((FX / "gpr_diario.xlsx").read_bytes())
    assert isinstance(df.index, pd.DatetimeIndex) and df.index.is_monotonic_increasing
    assert {"GPRD", "GPRD_ACT", "GPRD_THREAT", "N10D"} <= set(df.columns)
    assert not {"var_name", "var_label", "event", "date", "DAY"} & set(df.columns)
    assert df.at[pd.Timestamp("2023-03-10"), "GPRD"] == pytest.approx(113.444)


@requiere_xlrd
def test_leer_gpr_mensual():
    df = indices.leer_gpr((FX / "gpr_mensual.xlsx").read_bytes())
    assert df.index.min() == pd.Timestamp("2022-01-01") and "GPRC_ESP" in df.columns


def test_leer_csv_cboe_con_aviso_legal_y_espacios():
    df = indices.leer_csv_cboe((FX / "cboe_totalpc.csv").read_text(encoding="utf-8"))
    assert list(df.columns) == ["CALLS", "PUTS", "TOTAL", "P/C Ratio"]
    assert df.index.max() == pd.Timestamp("2019-10-04")
    assert df.loc["2019-10-04", "P/C Ratio"] == pytest.approx(1.05) and df["CALLS"].dtype.kind in "if"
    sk = indices.leer_csv_cboe((FX / "cboe_skew.csv").read_text(encoding="utf-8"))
    assert sk.index.min() == pd.Timestamp("2023-02-01")


@requiere_openpyxl
def test_leer_epu_diario_y_global_sin_cita():
    s = indices.leer_epu_diario("day,month,year,daily_policy_index\n1,1,1985,103.83\n2,1,1985,296.43\n")
    assert s.index[0] == pd.Timestamp("1985-01-01") and s.iloc[1] == pytest.approx(296.43)
    g = indices.leer_epu_global((FX / "epu_global.xlsx").read_bytes())
    assert list(g.columns) == ["GEPU_current", "GEPU_ppp"]
    assert g.index.max() == pd.Timestamp("2023-01-01") and g.notna().all().all()


def test_leer_cboe_diario_ignora_no_numericos():
    r = indices.leer_cboe_diario(json.loads((FX / "cboe_diario.json").read_text(encoding="utf-8")))
    assert r == {"total": 1.23, "index": 1.11, "equity": 1.10, "vix": 0.42}


def test_listar_vintages_solo_xls():
    v = indices.listar_vintages_gpr(json.loads((FX / "gpr_vintages.json").read_text(encoding="utf-8")))
    assert len(v) == 4 and set(v["tipo"]) == {"diario", "mensual"}
    d = v[v["tipo"] == "diario"]
    assert d["dias_desde_anterior"].dropna().tolist() == [7.0]
    assert v.loc[v["fichero"] == "data_gpr_export_202304.xls", "fecha_vintage"].iloc[0] == pd.Timestamp("2023-04-01")


# --------------------------------------------------------------------------- reglas de publicación (TimeGate)

def test_pub_gpr_diario_lunes_siguiente_y_festivos():
    f = pd.Series(pd.to_datetime(["2026-10-13", "2026-10-09", "2026-10-05", "2026-10-11", "2026-10-19"]))
    p = indices.pub_gpr_diario(f)
    # martes -> lunes siguiente; viernes/lunes/domingo -> lunes 12 es Columbus Day -> martes 13;
    # un lunes nunca se publica a sí mismo (regla estricta: el lunes siguiente).
    assert [x.date().isoformat() for x in p] == ["2026-10-19", "2026-10-13", "2026-10-13", "2026-10-13",
                                                "2026-10-26"]
    assert (p > f).all() and ((p - f).dt.days <= 8).all()


def test_pub_mensual_primer_habil_y_fin_de_mes():
    f = pd.Series(pd.to_datetime(["2026-09-01", "2025-08-01", "2025-08-31"]))
    p = indices.pub_primer_habil_mes_siguiente(f)
    assert [x.date().isoformat() for x in p] == ["2026-10-01", "2025-09-02", "2025-09-02"]   # 1-sep-2025: Labor Day
    q = indices.pub_fin_mes(pd.Series(pd.to_datetime(["2026-07-01"])), 2)
    assert q.iloc[0] == pd.Timestamp("2026-09-30")


def test_disponible_cierre_en_utc_con_horario_de_verano():
    d = indices.disponible_cierre(pd.Series(pd.to_datetime(["2023-03-10", "2023-03-13"])))
    assert d.tolist() == ["2023-03-10T21:15:00Z", "2023-03-13T20:15:00Z"]
    d6 = indices.disponible_cierre(pd.Series(pd.to_datetime(["2023-03-10"])), indices.FICHERO_TRAS_CIERRE_H)
    assert d6.tolist() == ["2023-03-11T03:15:00Z"]


def test_pub_habil_siguiente_cboe():
    # CSV de CBOE ≈5 h tras el cierre -> siguiente hábil (viernes -> lunes; víspera de festivo -> saltarlo)
    f = pd.Series(pd.to_datetime(["2023-03-09", "2023-03-10", "2026-10-09"]))
    assert [x.date().isoformat() for x in indices.pub_habil_siguiente(f)] == ["2023-03-10", "2023-03-13",
                                                                              "2026-10-13"]


def test_serie_desde_vintages_primer_vintage():
    idx = pd.to_datetime(["2023-03-03", "2023-03-10"])
    v1 = pd.DataFrame({"GPRD": [100.0]}, index=idx[:1])
    v2 = pd.DataFrame({"GPRD": [130.0, 98.0]}, index=idx)       # el 3 revisado a 130; el 10 aparece
    s = indices.serie_desde_vintages([(pd.Timestamp("2023-03-13"), "v2", v2), (pd.Timestamp("2023-03-06"), "v1", v1)],
                                     {"GPRD": "gprd"}, unidad="indice")
    s = base.serie_larga(s).set_index("fecha_referencia")
    assert s.at[pd.Timestamp("2023-03-03"), "valor"] == 100.0           # valor visto entonces, no el revisado
    assert s.at[pd.Timestamp("2023-03-03"), "fecha_publicacion"] == pd.Timestamp("2023-03-06")
    assert s.at[pd.Timestamp("2023-03-10"), "fecha_publicacion"] == pd.Timestamp("2023-03-13")
    assert (s["metodo_publicacion"] == "primer_vintage").all()


def test_tabla_testigo_causal_solo_usa_lo_publicado():
    larga = base.serie_larga(pd.DataFrame(dict(
        serie="x", fecha_referencia=pd.to_datetime(["2023-03-01", "2023-03-02", "2023-03-10"]),
        fecha_publicacion=pd.to_datetime(["2023-03-06", "2023-03-06", "2023-03-13"]), valor=[1.0, 2.0, 99.0],
        unidad="u", fuente="f", metodo_publicacion=base.REVISABLE)))
    (f,) = indices.tabla_testigo(larga, [DiaTestigo("svb", date(2023, 3, 10), "")])
    assert f["valor"] == 2.0 and f["fecha_dato"] == "2023-03-02" and f["percentil_causal"] == 100.0
    assert f["valor_ex_post"] == 99.0 and f["fecha_dato_ex_post"] == "2023-03-10" and f["revisable"]
    assert base.a_fecha_publicacion(larga, "2023-03-13", excluir_revisables=True).empty


@requiere_xlrd
def test_tabla_larga_valida_y_vista_timegate():
    df = indices.leer_gpr((FX / "gpr_diario.xlsx").read_bytes())
    larga = indices.a_larga(df, {"GPRD": "gprd"}, unidad="indice", fuente="gpr",
                            pub=indices.pub_gpr_diario(pd.Series(df.index)), metodo="lag_fijo", vintage="2023-03-20")
    larga = base.serie_larga(larga)
    assert (larga["fecha_publicacion"] > larga["fecha_referencia"]).all()
    vista = base.a_fecha_publicacion(larga, "2023-03-10")
    # El viernes 10 solo se conoce lo publicado hasta el lunes 6 (datos hasta el domingo 5).
    assert vista["fecha_referencia"].max() == pd.Timestamp("2023-03-05")


def test_huecos_por_frecuencia():
    f = pd.Series(pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-05"]))
    assert indices.huecos(f, "D") == 1 and indices.huecos(f, "B") == 0
    m = pd.Series(pd.to_datetime(["2020-01-01", "2020-02-01", "2020-05-01"]))
    assert indices.huecos(m, "M") == 1


# --------------------------------------------------------------------------- punta a punta sin red

@requiere_xlrd
@requiere_openpyxl
def test_sondear_sin_red_modo_rapido(sin_red, red_falsa, sin_robots):
    red_falsa(indices, RUTAS)
    res = indices.sondear(_cfg(modo_rapido=True))
    assert set(res.series) == {f"{n}_rapido" for n in (
        "gpr_diario", "gpr_mensual", "epu_diario_usa", "epu_global_mensual", "cboe_skew", "cboe_vix",
        "cboe_putcall_total", "cboe_putcall_equity", "cboe_putcall_index")}
    for nombre in res.series:
        s = base.leer_serie(nombre)
        assert not s.empty and (s["fecha_publicacion"] >= s["fecha_referencia"]).all()
    # CBOE: fichero tras el cierre -> D+1 hábil; GPR/EPU: vintage de hoy marcado como revisable
    vix = base.leer_serie("cboe_vix_rapido")
    assert (vix["fecha_publicacion"] > vix["fecha_referencia"]).all()
    assert (vix["metodo_publicacion"] == "fichero_tras_cierre").all()
    for n in ("gpr_diario_rapido", "gpr_mensual_rapido", "epu_diario_usa_rapido", "epu_global_mensual_rapido"):
        assert (base.leer_serie(n)["metodo_publicacion"] == base.REVISABLE).all()
    tt = res.tablas["indices_testigo"]
    assert {"valor", "percentil_causal", "valor_ex_post", "revisable"} <= set(tt.columns)
    cob = pd.DataFrame(res.cobertura).set_index("fuente")
    assert list(cob.columns) == [c for c in base.COLUMNAS_COBERTURA if c != "fuente"]
    assert set(cob.index) == {"gpr", "epu", "cboe_skew", "cboe_vix", "cboe_putcall"}
    assert cob.loc["gpr", "estado"] == "ok" and cob.loc["cboe_putcall", "estado"] == "parcial"
    assert cob.loc["cboe_putcall", "fin"] == "2019-10-04" and (cob["destino"] == "serie").all()
    # Vintages: el 10-03-2023 aparece en el vintage del lunes 13 (3 días) y se revisó después.
    vt = res.tablas["gpr_vintages_testigo"].set_index(["dia", "tipo"])
    assert vt.loc[("svb", "diario"), "lag_dias"] == 3
    assert vt.loc[("svb", "diario"), "revision_pct"] == pytest.approx(15.7, abs=0.1)
    assert vt.loc[("lehman", "diario"), "estado"] == "sin cobertura de vintages"
    pc = res.tablas["cboe_putcall_diario"].set_index("dia")
    assert pc.loc["svb", "origen"] == "json_diario" and pc.loc["svb", "total"] == pytest.approx(1.23)
    assert pc.loc["lehman", "origen"] is None or pc.loc["lehman", "origen"] == "csv"
    tg = pd.DataFrame(res.timegate)
    assert list(tg.columns) == base.COLUMNAS_TIMEGATE and tg["campo_fecha"].str.contains("vintage").any()
    rs = res.tablas["indices_resumen"]
    assert {"serie", "inicio", "fin", "n", "frecuencia", "huecos", "lag_publicacion", "metodo"} <= set(rs.columns)
    assert {f.fuente for f in res.fichas} == set(cob.index)
    assert all(f.tema == "noticias_alt" for f in res.fichas)
    # Caché: la segunda pasada no pide nada a la red (salvo listado y json diario).
    res2 = indices.sondear(_cfg(modo_rapido=True))
    assert any("(caché)" in r["item"] for r in res2.registro)


@requiere_xlrd
@requiere_openpyxl
def test_fuente_que_falla_no_detiene(sin_red, red_falsa, sin_robots):
    rutas = {**RUTAS, "All_Daily_Policy_Data.csv": (500, b"")}
    red_falsa(indices, rutas)
    res = indices.sondear(_cfg(modo_rapido=True, usar_cache=False))
    cob = pd.DataFrame(res.cobertura).set_index("fuente")
    assert cob.loc["epu", "estado"] in ("fallo", "parcial") and cob.loc["gpr", "estado"] == "ok"
    assert any(not r["ok"] and r["fuente"] == "epu" for r in res.registro)
    assert "epu_diario_usa_rapido" not in res.series and "cboe_skew_rapido" in res.series


def test_json_diario_sin_fichero_no_es_error(sin_red, red_falsa, sin_robots):
    rutas = {**RUTAS, "_daily_options": (403, b"<Error>AccessDenied</Error>")}
    red_falsa(indices, rutas)
    res = indices.sondear(_cfg(modo_rapido=True))
    pc = res.tablas["cboe_putcall_diario"].set_index("dia")
    assert pc.loc["svb", "origen"].startswith("sin fichero")
    assert all(r["ok"] for r in res.registro if r["fuente"] == "cboe_putcall")


@pytest.mark.live
def test_live_indices(sin_red):
    res = indices.sondear(base.ConfigSonda(modo_rapido=True, verbose=False))
    cob = pd.DataFrame(res.cobertura).set_index("fuente")
    assert cob.loc["cboe_vix", "estado"] == "ok" and cob.loc["gpr", "inicio"] <= "1985-01-01"
    assert cob.loc["cboe_putcall", "fin"] == "2019-10-04"
