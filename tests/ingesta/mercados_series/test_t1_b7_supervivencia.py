"""Tests del bloque 7 (supervivencia del S&P 500): sin red, con tramos y vistas de Yahoo sintéticos."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from ingesta.mercados_series import almacen, sinteticos
from ingesta.mercados_series import supervivencia as sv


def _tramos() -> pd.DataFrame:
    filas = [
        ("AAA", "1996-01-02", "2000-03-01"), ("BBB", "1996-01-02", "2000-07-03"), ("CCC", "1996-01-02", "2000-09-01"),
        ("DDD", "1998-01-02", "2002-01-02"), ("EEE", "1996-01-02", None), ("FFF", "2001-01-02", None),
        ("GG.B", "1996-01-02", None),
    ]
    df = pd.DataFrame(filas, columns=["ticker", "alta", "baja"])
    df["alta"] = pd.to_datetime(df["alta"])
    df["baja"] = pd.to_datetime(df["baja"])
    df["indice"] = "sp500"
    df["fecha_publicacion"] = df["alta"]
    df["fecha_publicacion_baja"] = df["baja"]
    df["nota"] = None
    return df


def _vista(inicio: str, fin: str, crec: float = 0.0005, div_cada: int = 0) -> pd.DataFrame:
    """Vista con el formato de ``Ticker.history(auto_adjust=False, actions=True)`` (índice con huso)."""
    idx = pd.bdate_range(inicio, fin, tz="America/New_York")
    close = 50.0 * (1 + crec) ** np.arange(len(idx))
    div = np.zeros(len(idx))
    if div_cada:
        div[div_cada::div_cada] = 0.25
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Adj Close": close,
                         "Volume": 1e6, "Dividends": div, "Stock Splits": 0.0}, index=idx)


def test_ticker_yahoo():
    assert sv.ticker_yahoo("BRK.B") == "BRK-B"
    assert sv.ticker_yahoo(" GG.B ") == "GG-B"


def test_muestra_estratificada_por_anio():
    t = _tramos()
    m = sv.muestra_salidos(t, por_anio=2, semilla=1)
    assert set(m["anio_salida"]) == {2000, 2002}
    assert (m.groupby("anio_salida").size() <= 2).all()
    assert len(m[m["anio_salida"] == 2002]) == 1          # solo hay uno ese año
    assert (m["grupo"] == "salido").all()
    pd.testing.assert_frame_equal(m, sv.muestra_salidos(t, por_anio=2, semilla=1))   # reproducible
    a = sv.muestra_actuales(t, n=10)
    assert set(a["ticker"]) == {"EEE", "FFF", "GG.B"} and (a["grupo"] == "actual").all()


def test_rentabilidad_diaria_con_dividendo():
    v = _vista("2020-01-01", "2020-01-10", crec=0.0)
    v.iloc[3, v.columns.get_loc("Dividends")] = 1.0
    r = sv.rentabilidad_diaria(v)
    assert r["sesion"].dt.tz is None
    assert np.isnan(r["rentabilidad"].iloc[0])
    assert r["rentabilidad"].iloc[3] == pytest.approx(1.0 / 50.0)
    assert r["rentabilidad"].iloc[2] == pytest.approx(0.0)
    assert sv.rentabilidad_diaria(pd.DataFrame()).empty


def test_cobertura_clases():
    esp = pd.bdate_range("2000-01-03", "2000-12-29")
    completa = sv.cobertura_tramo(esp, "2000-01-03", "2001-01-01", esperadas=esp)
    assert completa["clase"] == "completa" and completa["fraccion"] == 1.0
    parcial = sv.cobertura_tramo(esp[len(esp) // 2:], "2000-01-03", "2001-01-01", esperadas=esp)
    assert parcial["clase"] == "parcial" and 0.4 < parcial["fraccion"] < 0.6
    fuera = sv.cobertura_tramo(pd.bdate_range("2010-01-01", "2010-02-01"), "2000-01-03", "2001-01-01", esperadas=esp)
    assert fuera["clase"] == "fuera_de_tramo" and fuera["sesiones_con_precio"] == 0
    nada = sv.cobertura_tramo(None, "2000-01-03", "2001-01-01", esperadas=esp)
    assert nada["clase"] == "sin_datos" and nada["primera_yahoo"] is None


def test_sesiones_nyse_excluye_festivos():
    pytest.importorskip("exchange_calendars")
    s = sv.sesiones_nyse("2024-12-23", "2024-12-27")
    assert pd.Timestamp("2024-12-25") not in s and len(s) == 4


def test_descarga_reanudable_y_tabla(tmp_path):
    vistas = {"AAA": _vista("1990-01-01", "2026-01-01"), "BBB": _vista("1999-01-01", "2000-03-01"),
              "CCC": pd.DataFrame(), "DDD": _vista("2010-01-01", "2026-01-01"),
              "EEE": _vista("1990-01-01", "2026-10-09"), "FFF": _vista("2001-01-02", "2026-10-09"),
              "GG-B": _vista("1990-01-01", "2026-10-09")}
    pedidos: list[str] = []

    def obtener(sym):
        pedidos.append(sym)
        if sym == "XXX":
            raise ValueError("fallo de red simulado")
        return vistas[sym]

    syms = [*vistas, "XXX"]
    reg = sv.descargar_muestra(syms, raiz=tmp_path, ritmo=0, obtener=obtener)
    estados = dict(zip(reg["ticker_yahoo"], reg["estado"], strict=True))
    assert estados["CCC"] == "sin_datos" and estados["XXX"] == "error" and estados["AAA"] == "ok"
    pedidos.clear()
    sv.descargar_muestra(syms, raiz=tmp_path, ritmo=0, obtener=obtener)
    assert pedidos == ["XXX"]                             # reanuda: solo reintenta el error

    t = _tramos()
    muestra = pd.concat([sv.muestra_salidos(t, 5), sv.muestra_actuales(t, 5)], ignore_index=True)
    cob = sv.tabla_cobertura(muestra, raiz=tmp_path, hoy="2026-10-09")
    assert list(cob.columns) == list(sv.COLUMNAS_COBERTURA)
    clases = dict(zip(cob["ticker"], cob["clase"], strict=True))
    assert clases["AAA"] == "completa" and clases["CCC"] == "sin_datos" and clases["DDD"] == "fuera_de_tramo"
    assert clases["BBB"] == "parcial"
    assert "reutilizado" in cob.set_index("ticker").loc["DDD", "nota"]
    assert not {"close", "rentabilidad", "Close"} & set(cob.columns)   # solo metadatos
    res = sv.resumen_por_anio(cob)
    assert set(res["anio_salida"]) == {2000, 2002}
    assert res.loc[res["anio_salida"] == 2000, "n"].item() == 3


def test_sesgo_supervivientes_ganan_a_pit(tmp_path):
    t = _tramos()
    # salidos: rentabilidad negativa en su tramo; actuales: positiva
    vistas = {"AAA": _vista("1995-01-02", "2000-02-29", crec=-0.002), "BBB": _vista("1995-01-02", "2000-07-01", crec=-0.002),
              "CCC": _vista("1995-01-02", "2000-08-31", crec=-0.002), "DDD": _vista("1998-01-02", "2001-12-31", crec=-0.002),
              "EEE": _vista("1995-01-02", "2003-12-31", crec=0.001), "FFF": _vista("2001-01-02", "2003-12-31", crec=0.001),
              "GG-B": _vista("1995-01-02", "2003-12-31", crec=0.001)}
    muestra = pd.concat([sv.muestra_salidos(t, 5), sv.muestra_actuales(t, 5)], ignore_index=True)
    sesgo = sv.estimar_sesgo(t, muestra, lambda s: sv.rentabilidad_diaria(vistas[s]))
    assert {"pit", "supervivientes", "diferencia", "n_salidos_pit", "pit_solo_actuales"} <= set(sesgo.columns)
    m = sesgo.set_index("mes")
    feb98 = m.loc[pd.Timestamp("1998-02-28")]
    assert feb98["n_salidos_pit"] == 4 and feb98["n_actuales_pit"] == 2      # FFF entra en 2001
    assert feb98["pit"] < feb98["supervivientes"]
    # PIT pondera: 4/6 de salidos (negativos) y 2/6 de actuales
    n = len(pd.bdate_range("1998-02-01", "1998-02-28"))
    m_sal, m_act = 0.998 ** n - 1, 1.001 ** n - 1
    assert feb98["pit"] == pytest.approx((4 * m_sal + 2 * m_act) / 6)
    assert feb98["supervivientes"] == pytest.approx(m_act)
    res = sv.resumen_sesgo(sesgo)
    assert res["sesgo_anual_pct"] > 0 and res["meses"] > 0
    assert m.loc[pd.Timestamp("2003-06-30"), "pit_solo_actuales"]           # ya no queda ningún salido vivo


def test_mediciones_y_fragmento(tmp_path):
    t = _tramos()
    vistas = {s: _vista("1995-01-02", "2026-10-09") for s in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GG-B")}
    muestra = pd.concat([sv.muestra_salidos(t, 5), sv.muestra_actuales(t, 5)], ignore_index=True)
    cob = sv.tabla_cobertura(muestra, hoy="2026-10-09", lector=lambda s: sv.rentabilidad_diaria(vistas[s]))
    sesgo = sv.estimar_sesgo(t, muestra, lambda s: sv.rentabilidad_diaria(vistas[s]))
    reg = pd.DataFrame({"ticker_yahoo": ["AAA"], "estado": ["ok"], "segundos": [1.0]})
    med = sv.mediciones(cob, sesgo, reg)
    assert med["salidos_muestra"] == 4 and med["salidos_por_clase"]["completa"] == 4
    assert "limitacion" in med["sesgo"]


# --- Vía almacén (parquet as-traded del bloque 6) ------------------------------------------------
def _rent(inicio: str, fin: str, cierre: float = 50.0, volumen: float = 1e6) -> pd.DataFrame:
    """Salida de ``lector_almacen`` sintética: rentabilidad nula, cierre y efectivo constantes."""
    idx = pd.bdate_range(inicio, fin)
    return pd.DataFrame({"sesion": idx, "rentabilidad": 0.0, "cierre": cierre, "efectivo": cierre * volumen})


def _tramo(ticker: str, alta: str, baja: str | None, nota: str | None = None) -> dict:
    return {"indice": "sp500", "ticker": ticker, "alta": pd.Timestamp(alta),
            "baja": pd.Timestamp(baja) if baja else pd.NaT, "nota": nota}


def test_rentabilidad_total_con_split_y_dividendo():
    p = sinteticos.accion_con_eventos()
    r = sv.rentabilidad_total_precios(p).set_index("sesion")
    c = p.set_index("sesion")
    assert {"rentabilidad", "cierre", "efectivo"} <= set(r.columns)
    # Split 2:1: el cierre as-traded cae a la mitad, la rentabilidad total no
    s = sinteticos.SESION_SPLIT
    previa = c.index[c.index.get_loc(s) - 1]
    esperado = c.loc[s, "close"] * sinteticos.RATIO_SPLIT / c.loc[previa, "close"] - 1
    assert r.loc[s, "rentabilidad"] == pytest.approx(esperado)
    assert abs(r.loc[s, "rentabilidad"]) < 0.2
    # Dividendo en la fecha ex: close_t / (close_{t-1} - D_t) - 1 (método multiplicativo)
    ex = c.index[(c["dividendo"] > 0) & (c.index < s)][0]
    previa = c.index[c.index.get_loc(ex) - 1]
    esperado = c.loc[ex, "close"] / (c.loc[previa, "close"] - c.loc[ex, "dividendo"]) - 1
    assert r.loc[ex, "rentabilidad"] == pytest.approx(esperado)
    # Coherente con timegate.ajustar_a_fecha a la última fecha (= Adj Close de la vista de hoy)
    adj = sinteticos.vista_yahoo_hoy(p)["Adj Close"].pct_change()
    np.testing.assert_allclose(r["rentabilidad"].to_numpy()[1:], adj.to_numpy()[1:], rtol=1e-9)
    assert np.isnan(r["rentabilidad"].iloc[0])
    assert sv.rentabilidad_total_precios(pd.DataFrame()).empty


def test_lector_almacen_solo_lectura(tmp_path):
    p = sinteticos.accion_con_eventos()
    almacen.escribir_precios(p, "yahoo", ruta=tmp_path)
    antes = {f.name: f.stat().st_mtime_ns for f in tmp_path.iterdir()}
    leer = sv.lector_almacen(tmp_path)
    r = leer(sinteticos.TICKER_ACCION)
    assert len(r) == len(p) and r["sesion"].is_monotonic_increasing
    assert leer(sinteticos.TICKER_ACCION) is r                     # caché
    assert leer("NO-EXISTE") is None
    assert {f.name: f.stat().st_mtime_ns for f in tmp_path.iterdir()} == antes


def test_cobertura_dentro_del_tramo_y_sospechas():
    reuso = "ticker con 2 tramos: reentrada o ticker reutilizado por otra empresa (sin CIK/PERMNO)"
    tramos = pd.DataFrame([
        _tramo("OK", "2000-01-03", "2005-01-03"),                       # completo; el fichero sigue después
        _tramo("TARDE", "2000-01-03", "2012-01-03"),                    # precio solo desde 2010
        _tramo("FUERA", "2000-01-03", "2004-01-02"),                    # precio solo tras la baja
        _tramo("REENT", "2000-01-03", "2005-01-03", reuso),             # sigue años tras la baja, con nota
        _tramo("CENT", "2000-01-03", "2005-01-03"),                     # céntimos: otra empresa
        _tramo("ILIQ", "2000-01-03", "2005-01-03"),                     # efectivo mínimo
        _tramo("BKQ", "2000-01-03", "2005-01-03", "ticker con sufijo Q/QQ de quiebra"),
        _tramo("ACT", "2001-01-02", None),
    ])
    datos = {"OK": _rent("1995-01-02", "2006-01-02"), "TARDE": _rent("2010-01-04", "2026-01-02"),
             "FUERA": _rent("2015-01-02", "2026-01-02"), "REENT": _rent("1995-01-02", "2026-01-02"),
             "CENT": _rent("1995-01-02", "2026-01-02", cierre=0.2, volumen=1e8),
             "ILIQ": _rent("1995-01-02", "2026-01-02", cierre=20.0, volumen=1e3),
             "BKQ": _rent("1995-01-02", "2005-01-01"), "ACT": _rent("1990-01-02", "2026-10-09")}
    tabla = sv.tabla_cobertura(sv.todos_los_tramos(tramos), hoy="2026-10-09", lector=datos.get)
    assert list(tabla.columns) == list(sv.COLUMNAS_COBERTURA)
    cob = tabla.set_index("ticker")
    # La cobertura se mide DENTRO del tramo, no por existir el fichero
    assert cob.loc["OK", "clase"] == "completa" and cob.loc["OK", "sospecha"] is None and cob.loc["OK", "en_estimacion"]
    esperadas = len(sv.sesiones_nyse("2000-01-03", "2012-01-02"))
    con = len(sv.sesiones_nyse("2010-01-04", "2012-01-02"))
    assert cob.loc["TARDE", "clase"] == "parcial"
    assert cob.loc["TARDE", "fraccion"] == pytest.approx(con / esperadas, abs=1e-4)
    assert cob.loc["TARDE", "sospecha"] == "empieza_tarde" and not cob.loc["TARDE", "en_estimacion"]
    assert cob.loc["FUERA", "clase"] == "fuera_de_tramo" and cob.loc["FUERA", "sesiones_con_precio"] == 0
    assert cob.loc["FUERA", "sospecha"] == "reutilizado_despues" and not cob.loc["FUERA", "en_estimacion"]
    assert "reutilizado" in cob.loc["FUERA", "nota"]
    # Reentrada: se marca, pero solo la variante estricta la excluye
    assert cob.loc["REENT", "sospecha"] == "termina_tarde" and cob.loc["REENT", "en_estimacion"]
    assert cob.loc["CENT", "sospecha"] == "impropio_sp500" and not cob.loc["CENT", "en_estimacion"]
    assert cob.loc["ILIQ", "sospecha"] == "impropio_sp500" and not cob.loc["ILIQ", "en_estimacion"]
    assert cob.loc["BKQ", "sospecha"] == "ticker_q" and not cob.loc["BKQ", "en_estimacion"]
    assert cob.loc["ACT", "grupo"] == "actual" and cob.loc["ACT", "en_estimacion"]
    assert not {"cierre", "efectivo", "rentabilidad", "close"} & set(cob.columns)   # solo metadatos
    dec = sv.resumen_por_decada(tabla)
    assert dec["2000-2009"]["n"] == 6 and dec["2000-2009"]["en_estimacion"] == 2        # OK y REENT
    assert dec["2000-2009"]["sospechosos_excluidos"] == 3                              # CENT, ILIQ y BKQ
    assert dec["2010-2019"] == {**dec["2010-2019"], "n": 1, "con_precio_en_tramo": 1, "sospechosos_excluidos": 1,
                                "en_estimacion": 0, "fraccion_media_util": 0.0}       # TARDE


def test_sospechas_tramo_sin_medianas_ni_precio():
    vacio = sv.cobertura_tramo(None, "2000-01-03", "2001-01-02", esperadas=pd.bdate_range("2000-01-03", "2000-12-29"))
    assert sv.sospechas_tramo(vacio, "2000-01-03", "2001-01-02") == []
    cob = {"clase": "completa", "primera_yahoo": "2000-01-03", "ultima_yahoo": "2000-12-29"}
    assert sv.sospechas_tramo(cob, "2000-01-03", "2001-01-02", mediana_cierre=None, mediana_efectivo=None) == []
    assert sv.sospechas_tramo(cob, "2000-01-03", "2001-01-02", mediana_efectivo=5e5) == ["impropio_sp500"]


def test_medir_almacen_extremo_a_extremo(tmp_path):
    base = sinteticos.accion_con_eventos()
    for tk in ("AAA", "CCC", "DDD"):
        p = base.copy()
        p["ticker"] = tk
        if tk == "AAA":                                   # el salido rinde menos (pierde tres cuartos)
            caida = np.linspace(1.0, 0.25, len(p))
            for c in ("open", "high", "low", "close"):
                p[c] = p[c] * caida
            p["dividendo"] = 0.0
        almacen.escribir_precios(p, "yahoo", ruta=tmp_path)
    tramos = pd.DataFrame([_tramo("AAA", "2022-01-03", "2023-09-01"), _tramo("BBB", "2022-01-03", "2023-01-03"),
                           _tramo("CCC", "2022-06-01", None), _tramo("DDD", "2022-01-03", None)])
    res = sv.medir_almacen(tramos, ruta=tmp_path, hoy="2024-12-31", con_muestra_antigua=False)
    cob = res["cobertura"].set_index("ticker")
    assert len(cob) == 4 and cob.loc["BBB", "clase"] == "sin_datos" and cob.loc["AAA", "clase"] == "completa"
    assert set(res["variantes"]) == {"sin_sospechas", "estricta", "con_sospechas"}
    med = res["mediciones"]
    assert med["via"] == "almacen" and med["tramos_en_estimacion"] == {"salidos": 1, "actuales": 2}
    assert med["almacen"]["tramos_con_fichero"] == 3
    s = med["sesgo"]
    assert s["meses"] > 0 and s["sesgo_anual_pct"] > 0
    lo, hi = s["sesgo_ic95_anual_pct"]
    assert lo <= s["sesgo_anual_pct"] <= hi and 0 < s["cobertura_pit_media"] <= 1
    ses = res["sesgo"].set_index("mes")
    assert ses.loc[pd.Timestamp("2022-03-31"), "n_salidos_pit"] == 2           # AAA y BBB vivos; BBB sin precio
    assert ses.loc[pd.Timestamp("2022-03-31"), "cobertura_pit"] == pytest.approx(2 / 3)


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("INGESTA_LIVE"), reason="INGESTA_LIVE=1 para pruebas con red")
def test_live_yahoo_salido_y_actual(tmp_path):
    pytest.importorskip("yfinance")
    reg = sv.descargar_muestra(["AAPL", "LEHMQ_NO_EXISTE"], raiz=tmp_path, ritmo=0.5)
    estados = dict(zip(reg["ticker_yahoo"], reg["estado"], strict=True))
    assert estados["AAPL"] == "ok"
    assert estados["LEHMQ_NO_EXISTE"] in ("sin_datos", "error")
