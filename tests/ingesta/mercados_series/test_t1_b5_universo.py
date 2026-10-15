"""Tests de ``ingesta.mercados_series.universo`` (bloque 5) y de sus mercados en ``mercados.yaml``.

Sin red: el YAML real del universo (solo metadatos) y universos SINTÉTICOS mínimos para los
errores. ``leer_justetf`` con un HTML sintético con el formato de la ficha de justETF.
``@pytest.mark.live``: justETF y Yahoo reales (``INGESTA_LIVE=1``).
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from ingesta.mercados_series import universo as U
from ingesta.mercados_series import visibilidad

PROPUESTA = ("SPY QQQ IWM VGK EWJ EEM FXI EWZ MCHI VPL XLK XLF XLE XLV XLC XLI XLY XLP XLRE XLB XLU TLT IEF SHY "
             "TIP LQD HYG EMB BNDX GLD SLV USO UNG PDBC DBA CPER VNQ VNQI BITO ETHE VIXY SHV").split()


@pytest.fixture(scope="module")
def universo():
    return U.cargar_universo()


def _minimo(**extra) -> dict:
    base = {"version": 1, "fondos_por_que_no_entran": "porque NAV en t+1",
            "grupos": [{"nombre": "g", "defaults": {"clase": "etf", "capa": "tactica", "invertible_minorista_ue": False,
                                                     "mercado": "XNYS", "divisa": "USD", "estado": "activo"},
                        "instrumentos": [{"id": "yf:AAA", "nombre": "A", "bucket": "rv_eeuu"}]}]}
    base["grupos"][0]["instrumentos"][0].update(extra)
    return base


# --- YAML real ---------------------------------------------------------------------------------------
def test_t1_b5_universo_carga_y_tamano(universo):
    ins = universo["instrumentos"]
    assert 180 <= len(ins) <= 260
    ids = {i["id"] for i in ins}
    assert {f"yf:{t}" for t in PROPUESTA} <= ids and len(PROPUESTA) == 42
    assert {"yf:AGG", "yf:RSP", "yf:VBMFX", "yf:IBIT", "yf:ETHA", "coinbase:BTC-USD", "coinbase:ETH-USD"} <= ids
    assert {"yf:^GSPC", "yf:^VIX", "yf:^MOVE", "yf:EURUSD=X", "yf:ES=F", "yf:ZN=F"} <= ids


def test_t1_b5_universo_clases_y_capas(universo):
    df = U.tabla_universo(universo)
    assert set(df["capa"]) == set(U.CAPAS)
    assert (df["clase"] == "etf").sum() >= 120
    assert 30 <= ((df["grupo"] == "ucits")).sum() <= 60
    assert df.loc[df["clase"] == "futuro", "capa"].eq("contexto").all()
    assert not df.loc[df["clase"] == "futuro", "invertible_minorista_ue"].any()


def test_t1_b5_universo_ucits_invertibles_y_ee_uu_no(universo):
    for i in universo["instrumentos"]:
        if i["isin"] and i["isin"].startswith("US"):
            assert not i["invertible_minorista_ue"], i["id"]
        if i["equivalente_ucits"]:
            eq = next(x for x in universo["instrumentos"] if x["id"] == i["equivalente_ucits"])
            assert eq["invertible_minorista_ue"] and eq["proxy_previo"] is not None


def test_t1_b5_universo_ucits_con_lanzamiento_o_nota(universo):
    for i in universo["instrumentos"]:
        if i["grupo"] == "ucits":
            assert i["inicio_esperado"] is not None or i["estado"] == "por_verificar", i["id"]


def test_t1_b5_universo_ticker_yahoo_deducido(universo):
    por_id = {i["id"]: i for i in universo["instrumentos"]}
    assert por_id["yf:SPY"]["tickers"] == {"yahoo": "SPY"}
    assert por_id["yf:USDJPY=X"]["tickers"]["yahoo"] == "JPY=X"
    assert por_id["coinbase:BTC-USD"]["tickers"]["binance"] == "BTCUSDT"


def test_t1_b5_universo_fondos_por_que_no_entran(universo):
    assert "NAV" in universo["fondos_por_que_no_entran"]
    assert next(i for i in universo["instrumentos"] if i["id"] == "yf:VBMFX")["estado"] == "proxy"


def test_t1_b5_mercados_del_universo_en_mercados_yaml():
    mercados = visibilidad.cargar_mercados()
    assert {"ARCX", "NAV_US", "STOXX_IDX", "ICE_MOVE", "FX_YAHOO"} <= set(mercados)
    usados = {i["mercado"] for i in U.cargar_universo(mercados=set(mercados))["instrumentos"] if i.get("mercado")}
    assert usados <= set(mercados)


def test_t1_b5_publicacion_arca_nav_stoxx_fx(tmp_path):
    m = visibilidad.cargar_mercados()
    # cierre 16:00 + 30 min = 16:30: la sesión del martes se ve en la decisión del mismo martes
    assert str(visibilidad.fecha_publicacion_cierre("2026-10-06", "ARCX", m).date()) == "2026-10-06"
    # el NAV (18:00 NY) y el cierre STOXX (17:50 Fráncfort = 11:50 NY) en t+1 y en t
    assert str(visibilidad.fecha_publicacion_cierre("2026-10-06", "NAV_US", m).date()) == "2026-10-07"
    assert str(visibilidad.fecha_publicacion_cierre("2026-10-06", "STOXX_IDX", m).date()) == "2026-10-06"
    # la vela de FX de Yahoo (medianoche de Londres) también en t+1
    assert str(visibilidad.fecha_publicacion_cierre("2026-10-06", "FX_YAHOO", m).date()) == "2026-10-07"


# --- Validación con universos sintéticos ---------------------------------------------------------------
@pytest.mark.parametrize("isin,ok", [("IE00B5BMR087", True), ("US78462F1030", True), ("CH0454664001", True),
                                     ("IE00B5BMR088", False), ("US78462F103", False), ("ie00b5bmr087", False)])
def test_t1_b5_isin_digito_control(isin, ok):
    assert U.isin_valido(isin) is ok


def test_t1_b5_defaults_y_subunidad():
    u = U.validar_universo(_minimo(divisa="GBp", mercado="XLON"), mercados={"XNYS", "XLON"})
    i = u["instrumentos"][0]
    assert i["divisa"] == "GBp" and i["tickers"] == {"yahoo": "AAA"} and i["isin"] is None
    assert U.divisa_base("GBp") == ("GBP", 100) and U.divisa_base("USD") == ("USD", 1)


@pytest.mark.parametrize("extra,mensaje", [
    ({"mercado": "XXXX"}, "mercado"),
    ({"divisa": "usd"}, "divisa"),
    ({"capa": "otra"}, "capa"),
    ({"isin": "IE00B5BMR088"}, "ISIN"),
    ({"inicio_esperado": "2020-13-01"}, "fecha"),
    ({"campo_raro": 1}, "desconocidos"),
    ({"invertible_minorista_ue": True, "isin": "US78462F1030"}, "EE. UU."),
    ({"invertible_minorista_ue": True}, "sin ISIN"),
    ({"proxy_previo": "yf:NOESTA"}, "no está en el universo"),
])
def test_t1_b5_errores(extra, mensaje):
    with pytest.raises(ValueError, match=mensaje):
        U.validar_universo(_minimo(**extra), mercados={"XNYS"})


def test_t1_b5_id_duplicado_y_sin_prefijo():
    c = _minimo()
    c["grupos"][0]["instrumentos"].append(copy.deepcopy(c["grupos"][0]["instrumentos"][0]))
    with pytest.raises(ValueError, match="duplicado"):
        U.validar_universo(c, mercados={"XNYS"})
    with pytest.raises(ValueError, match="prefijo"):
        U.validar_universo(_minimo(id="SPY"), mercados={"XNYS"})


def test_t1_b5_equivalente_ucits_debe_ser_invertible():
    c = _minimo(equivalente_ucits="yf:BBB")
    c["grupos"][0]["instrumentos"].append({"id": "yf:BBB", "nombre": "B", "bucket": "rv_eeuu"})
    with pytest.raises(ValueError, match="no es invertible"):
        U.validar_universo(c, mercados={"XNYS"})


def test_t1_b5_claves_mercados_solo_de_mercados_yaml(tmp_path):
    (tmp_path / "m.yaml").write_text("XNYS: {}\nARCX: {}\n", encoding="utf-8")
    assert U.claves_mercados(tmp_path / "m.yaml") == {"XNYS", "ARCX"}
    assert {"ARCX", "NAV_US", "STOXX_IDX", "FX_YAHOO"} <= U.claves_mercados()


def test_t1_b5_leer_justetf_sintetico():
    html = ('<html><h1 class="x"><span>iShares Core S&amp;P 500 UCITS ETF USD (Acc)</span></h1>'
            '<div>Inception Date <div class="data"><span>info</span><div class="val">19 May 2010</div></div></div></html>')
    assert U.leer_justetf(html) == {"nombre_justetf": "iShares Core S&P 500 UCITS ETF USD (Acc)",
                                    "lanzamiento_justetf": "2010-05-19"}
    assert U.leer_justetf("<html></html>") == {"nombre_justetf": None, "lanzamiento_justetf": None}


def test_t1_b5_ruta_universo_existe():
    assert Path(U.RUTA_UNIVERSO).exists()


# --- Red (INGESTA_LIVE=1) ----------------------------------------------------------------------------------
@pytest.mark.live
def test_t1_b5_live_justetf_y_yahoo(tmp_path):
    u = U.cargar_universo()
    sub = {**u, "instrumentos": [i for i in u["instrumentos"] if i["id"] in ("yf:CSPX.L", "yf:SPY")]}
    jt = U.verificar_ucits_justetf(sub, destino=tmp_path / "j.csv")
    assert jt.loc[0, "lanzamiento_justetf"] == "2010-05-19"
    yh = U.comprobar_yahoo(sub, destino=tmp_path / "y.csv")
    spy = yh.set_index("id").loc["yf:SPY"]
    assert spy["primer_dia"] <= "1993-02-01" and spy["divisa_yahoo"] == "USD"


def test_t1_b5_divisas_coinciden_con_yahoo_medido(universo):
    """Metadatos medidos (tablas/universo_yahoo.csv): la divisa declarada es la de la línea de Yahoo."""
    import pandas as pd

    ruta = U.resultados_dir(U.TEMA) / "tablas" / U.FICHERO_YAHOO
    if not ruta.exists():
        pytest.skip("sin medición de Yahoo")
    y = pd.read_csv(ruta).dropna(subset=["divisa_yahoo"]).set_index("id")["divisa_yahoo"]
    distintas = [(i["id"], i["divisa"], y[i["id"]]) for i in universo["instrumentos"]
                 if i["id"] in y.index and i["divisa"] != y[i["id"]]]
    assert not distintas

