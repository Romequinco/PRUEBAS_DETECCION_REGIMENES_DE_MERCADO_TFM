"""Tests de A5 · mercados de predicción (Polymarket, Kalshi, PredictIt). Sin red salvo el ``live``."""
from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from ingesta.noticias_alt import base
from ingesta.noticias_alt import prediccion as pr
from ingesta.noticias_alt.muestra import DiaTestigo

FX = Path(__file__).parent / "fixtures" / "prediccion"

RUTAS = {
    # Kalshi (lo más específico primero)
    "historical/cutoff": "prediccion/kalshi_cutoff.json",
    "candlesticks": "prediccion/kalshi_velas.json",
    "trade-api/v2/markets/trades": "prediccion/kalshi_trades.json",
    "trade-api/v2/historical/markets?": "prediccion/kalshi_hist_markets.json",
    "status=open": "prediccion/kalshi_abiertos.json",
    "status=settled": "prediccion/kalshi_vivo_vacio.json",
    # Polymarket
    "gamma-api.polymarket.com/markets/slug/": "prediccion/gamma_market_slug.json",
    "order=createdAt": "prediccion/gamma_primer.json",
    "gamma-api.polymarket.com/markets?": "prediccion/gamma_markets_resueltos.json",
    "gamma-api.polymarket.com/events": "prediccion/gamma_events.json",
    "clob.polymarket.com/prices-history": "prediccion/clob_historial.json",
    "data-api.polymarket.com/trades": "prediccion/data_trades.json",
    # PredictIt
    "predictit.org/api/marketdata": "prediccion/predictit.json",
}


def _ts(*a: int) -> float:
    return datetime(*a, tzinfo=UTC).timestamp()


def _mercados() -> list[dict]:
    return json.loads((FX / "gamma_markets_resueltos.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- TimeGate

def test_corte_ny_respeta_horario_de_verano():
    assert pr.corte_ny(date(2025, 1, 7)) == datetime(2025, 1, 7, 21, 30, tzinfo=UTC)   # EST (UTC-5)
    assert pr.corte_ny("2025-04-07") == datetime(2025, 4, 7, 20, 30, tzinfo=UTC)       # EDT (UTC-4)


def test_precio_a_fecha_usa_el_ultimo_punto_antes_de_las_1630_ny():
    h = [{"t": _ts(2025, 4, 7, 12), "p": 0.10}, {"t": _ts(2025, 4, 7, 20), "p": 0.30},
         {"t": _ts(2025, 4, 7, 21), "p": 0.90}]          # 21:00 UTC = 17:00 NY: aún no publicado a las 16:30
    assert pr.precio_a_fecha(h, date(2025, 4, 7)) == 0.30
    assert pr.precio_a_fecha(h, date(2025, 4, 8)) == 0.90
    assert pr.precio_a_fecha(h, date(2025, 4, 6)) is None
    assert pr.precio_a_fecha(h, datetime(2025, 4, 7, 12, 30, tzinfo=UTC)) == 0.10   # instante: tal cual


def test_serie_diaria_no_mira_el_futuro():
    hist = json.loads((FX / "clob_historial.json").read_text())["history"]
    corta = pr.serie_diaria([h for h in hist if h["t"] <= _ts(2025, 3, 1)])
    larga = pr.serie_diaria(hist)
    comun = larga[larga["fecha"].isin(corta["fecha"])]
    assert list(comun["valor"]) == list(corta["valor"])[: len(comun)]
    assert larga["fecha"].is_unique
    # cada valor se publicó antes del corte de su día
    assert all(t <= pr.corte_ny(f).timestamp() for f, t in zip(larga["fecha"], larga["t"]))


def test_saltos_umbral_15pp():
    s = pd.DataFrame({"fecha": pd.date_range("2025-01-01", periods=4).date, "valor": [0.20, 0.36, 0.30, 0.60]})
    out = pr.saltos(s, 0.15, nombre="x")
    assert list(out["cambio_pp"]) == [16.0, 30.0]
    assert list(out.columns) == ["serie", "fecha", "de", "a", "cambio_pp"]
    exacto = pd.DataFrame({"fecha": pd.date_range("2025-01-01", periods=2).date, "valor": [0.30, 0.15]})
    assert len(pr.saltos(exacto, 0.15)) == 1                      # 15 pp justos cuentan
    assert pr.saltos([]).empty


# --------------------------------------------------------------------------- parseo Polymarket

def test_parseo_mercados_gamma():
    m = {x["id"]: x for x in _mercados()}
    assert pr.token_yes(m["1"]) == "111"
    assert pr.token_yes(m["4"]) is None                            # no es Yes/No
    assert pr.resultado_binario(m["1"]) == 1 and pr.resultado_binario(m["2"]) == 0
    assert pr.resultado_binario(m["3"]) == 1                       # precios finales no exactos (AMM)
    assert pr.resultado_binario(m["5"]) is None                    # reparto 50/50
    assert pr.fecha_cierre(m["1"]) == datetime(2025, 3, 20, tzinfo=UTC)   # min(endDate, closedTime)
    assert pr.volumen({"volume": "150000"}) == 150000.0


def test_categorias():
    assert pr.categoria("Fed decreases interest rates by 25 bps after March?") == "fed"
    assert pr.categoria("US recession in 2025?") == "economia"
    assert pr.categoria("Will Pierre Poilievre be the next Canadian Prime Minister?") == "politica"  # no «oil»
    assert pr.categoria("Will Bitcoin hit $100k?") == "cripto"
    assert pr.categoria("Lakers vs. Celtics") == "deportes"
    assert pr.categoria("Is the superconductor real?") == "otros"


def test_fila_calibracion_y_resumen():
    hist = [{"t": _ts(2025, 2, 1), "p": 0.2}, {"t": _ts(2025, 3, 10), "p": 0.6}, {"t": _ts(2025, 3, 19, 12), "p": 0.9}]
    cierre = datetime(2025, 3, 20, tzinfo=UTC)
    f = pr.fila_calibracion("polymarket", "1", "¿Pregunta ficticia?", "fed", cierre, 2e5, hist, 1)
    assert (f["p_1d"], f["p_7d"], f["p_30d"]) == (0.6, 0.6, 0.2)   # el punto de las 12:00 del 19 es posterior a −1 día
    f2 = pr.fila_calibracion("polymarket", "2", "x", "fed", cierre, 2e5, hist[1:], 0)
    assert f2["p_30d"] is None and f2["inicio_historial"] == "2025-03-10"
    r = pr.resumen_calibracion(pd.DataFrame([f, f2]))
    assert set(r["horizonte"]) == {"1d", "7d", "30d"}
    fila = r[(r["horizonte"] == "1d")].iloc[0]
    assert fila["n_total"] == 2 and fila["brier"] == pytest.approx(((0.6 - 1) ** 2 + 0.6 ** 2) / 2, abs=1e-4)


def test_repartir_alterna_grupos():
    cand = [dict(g="a", v=v) for v in (5, 4, 3)] + [dict(g="b", v=1)]
    out = pr.repartir(cand, 3, grupo=lambda c: c["g"], peso=lambda c: c["v"])
    assert [(c["g"], c["v"]) for c in out] == [("a", 5), ("b", 1), ("a", 4)]


# --------------------------------------------------------------------------- Kalshi y PredictIt

def test_precio_vela_alternativas():
    assert pr.precio_vela({"price": {"close": "0.4200"}}) == 0.42
    assert pr.precio_vela({"price": {"close": None}, "yes_bid": {"close": "0.40"}, "yes_ask": {"close": "0.50"}}) == 0.45
    assert pr.precio_vela({"price": {"close": None, "previous": "0.3"}, "yes_bid": {"close": None}}) == 0.3
    assert pr.precio_vela({"price": {"close": None, "previous": "0.3"}, "yes_bid": {"close": "0"},
                           "yes_ask": {"close": "1"}}) == 0.3                  # horquilla 0-1: no es un precio
    assert pr.precio_vela({"price": {"close": 37}}) == 0.37        # formato antiguo en centavos
    velas = json.loads((FX / "kalshi_velas.json").read_text())["candlesticks"]
    h = pr.velas_a_historial(velas)
    assert len(h) == 32 and h[-7]["p"] == 0.60                       # vela sin operaciones -> punto medio


def test_resumen_predictit_hora_ny_a_utc():
    j = json.loads((FX / "predictit.json").read_text(encoding="utf-8"))
    r = pr.resumen_predictit(j, ahora=datetime(2026, 10, 14, 11, 30, tzinfo=UTC))
    assert r["n_mercados"] == 2 and r["n_contratos"] == 3
    assert r["ts_fuente"] == "2026-10-14T11:21:59+00:00"          # 07:21 NY (EDT) = 11:21 UTC
    assert r["lag_h"] == pytest.approx(0.13, abs=0.01)


# --------------------------------------------------------------------------- cliente y caché

def test_cliente_cachea_y_registra(sin_red, red_falsa, monkeypatch):
    llamadas = []
    falso = red_falsa(pr, RUTAS)
    monkeypatch.setattr(pr, "get", lambda url, **kw: (llamadas.append(url), falso(url, **kw))[1])
    res = base.Resultado("prediccion")
    cli = pr.Cliente(res)
    h1 = pr.historial_precio("111", cli)
    h2 = pr.historial_precio("111", cli)
    assert h1 == h2 and len(llamadas) == 1
    assert [r["ok"] for r in res.registro] == [True, True] and res.registro[1]["segundos"] == 0.0
    assert list((sin_red / "corpus" / "noticias_alt" / "crudo" / "prediccion" / "polymarket").glob("*.json"))


def test_cliente_registra_fallos(sin_red, red_falsa):
    red_falsa(pr, {"clob.polymarket.com": (400, b'{"error":"bad"}')})
    res = base.Resultado("prediccion")
    with pytest.raises(Exception):
        pr.historial_precio("111", pr.Cliente(res))
    assert res.registro[0]["ok"] is False and "400" in res.registro[0]["error"]


# --------------------------------------------------------------------------- punta a punta

def _cfg(**kw) -> base.ConfigSonda:
    dias = [DiaTestigo("lehman", date(2008, 9, 15), "x"), DiaTestigo("svb", date(2023, 3, 10), "x"),
            DiaTestigo("aranceles", date(2025, 4, 7), "x")]
    return base.ConfigSonda(dias=dias, modo_rapido=True, verbose=False, **kw)


def test_sondear_punta_a_punta_sin_red(sin_red, red_falsa):
    red_falsa(pr, RUTAS)
    res = pr.sondear(_cfg())
    fallos = [r for r in res.registro if not r["ok"]]
    assert not fallos, fallos
    assert {f.fuente: f.estado_sonda for f in res.fichas} == {"polymarket": "ok", "kalshi": "ok", "predictit": "ok"}
    assert all(f.tema == "noticias_alt" for f in res.fichas)

    cob = pd.DataFrame(res.cobertura)
    assert list(cob.columns) == base.COLUMNAS_COBERTURA and len(cob) == 3
    pm = cob.set_index("fuente").loc["polymarket"]
    assert pm["inicio"] == "2020-10-02" and pm["destino"] == "serie" and pm["familia"] == "prediccion"
    assert cob.set_index("fuente").loc["predictit", "backtest"] == False  # noqa: E712

    t = res.tablas
    assert {"polymarket_liquidez_anio", "kalshi_liquidez_anio", "prediccion_calibracion", "prediccion_calibracion_resumen",
            "prediccion_dias_testigo", "polymarket_emblematicos", "polymarket_saltos", "predictit_instantanea"} <= set(t)
    liq = t["polymarket_liquidez_anio"]
    assert list(liq["anio"]) == list(range(2020, date.today().year + 1))
    assert liq["n_mayor_1m"].iloc[0] == 1 and liq["volumen_usd"].iloc[0] == pytest.approx(3_050_000.5)

    cal = t["prediccion_calibracion"]
    assert {"plataforma", "mercado_id", "pregunta", "categoria", "fecha_cierre", "volumen_usd", "p_1d", "p_7d", "p_30d",
            "resultado"} <= set(cal.columns)
    pmc = cal[cal["plataforma"] == "polymarket"]
    assert set(pmc["mercado_id"]) == {"1", "2"}          # sin AMM, sin no binarios ni repartos; sin duplicados
    assert pmc["p_1d"].notna().all()
    kac = cal[cal["plataforma"] == "kalshi"]
    assert set(kac["mercado_id"]) == {"FED-23MAY-T5.00", "FED-23MAY-T5.25"}   # el de 500 contratos queda fuera
    assert kac["p_1d"].iloc[0] == pytest.approx(0.91)

    dias = t["prediccion_dias_testigo"].set_index(["plataforma", "dia"])
    assert dias.loc[("polymarket", "lehman"), "cobertura"] == "sin cobertura"
    assert dias.loc[("kalshi", "svb"), "n_activos"] == 2

    assert res.series == ["polymarket_recesion_eeuu_2025_rapido", "polymarket_fed_recorte25_sep2025_rapido",
                          "polymarket_recesion_eeuu_2026_rapido"]                # modo rápido: no pisa
    s = base.leer_serie(res.series[0])
    assert list(s.columns) == base.COLUMNAS_SERIE and (s["fecha_publicacion"] == s["fecha_referencia"]).all()
    assert set(t["polymarket_saltos"]["fecha"].astype(str)) == {"2025-04-03"}   # el salto de +20 pp del fixture

    tg = pd.DataFrame(res.timegate)
    assert list(tg.columns) == base.COLUMNAS_TIMEGATE
    assert {"polymarket", "kalshi", "predictit"} <= set(tg["fuente"])
    assert (tg["lag_h"].dropna() >= 0).all()
    assert t["predictit_instantanea"]["n_mercados"].iloc[0] == 2
    assert res.documentos == {}                           # sin texto de noticias


def test_sondear_sigue_si_una_fuente_falla(sin_red, red_falsa):
    rutas = {"kalshi.com": (503, b"caido"), **RUTAS}
    red_falsa(pr, rutas)
    res = pr.sondear(_cfg(usar_cache=False))
    est = {f.fuente: f.estado_sonda for f in res.fichas}
    assert est["kalshi"] == "fallo" and est["polymarket"] == "ok" and est["predictit"] == "ok"
    assert "kalshi_liquidez_anio" not in res.tablas and "polymarket_liquidez_anio" in res.tablas


@pytest.mark.live
def test_live_prediccion(sin_red):
    m = pr.mercado_por_slug(pr.EMBLEMATICOS["recesion_eeuu_2025"])
    p = pr.precio_a_fecha(pr.token_yes(m), date(2025, 4, 7))
    assert p is not None and 0 < p < 1
    hist = pr.historial_precio(pr.token_yes(m))
    assert datetime.fromtimestamp(hist[0]["t"], tz=UTC) < datetime(2025, 2, 1, tzinfo=UTC) + timedelta(days=1)
