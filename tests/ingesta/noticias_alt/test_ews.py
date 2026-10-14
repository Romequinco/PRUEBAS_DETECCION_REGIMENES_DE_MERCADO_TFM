"""Tests de ``noticias_alt.ews``: tramos del archivo, z causal sin mirar el futuro, revisión y sonda sin red."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ingesta.noticias_alt import base, ews
from ingesta.noticias_alt.muestra import DiaTestigo

FX = Path(__file__).parent / "fixtures" / "ews"
RUTAS = {"dashboard.json": "ews/dashboard.json", "rss.xml": "ews/rss.xml"}


def _panel(n: int, pico: bool = True, ruido: float = 5.0) -> dict:
    """Panel sintético: conteo con ciclo diario, semihorario regular, pico opcional al final."""
    rng = np.random.default_rng(0)
    c = [int(300 + 100 * np.sin(2 * np.pi * (i % 48) / 48) + rng.normal(0, ruido)) for i in range(n)]
    if pico:
        c[-1] += 200
    return {"current": {"asOf": "2026-01-01T00:00:00+00:00"},
            "trends": {"archive": {"t0": "2025-10-06T00:00:00+00:00", "tr": [[1_800_000, n - 1]], "c": c,
                                   "p": [float(x) for x in c], "s": [1.0] * n, "z": [0.0] * n}}}


def _cfg(**kw) -> base.ConfigSonda:
    return base.ConfigSonda(dias=[DiaTestigo("hoy", date(2026, 2, 9), "")], verbose=False, **kw)


@pytest.fixture
def sin_robots(monkeypatch):
    monkeypatch.setattr(ews, "robots_permite", lambda url: True)


# --------------------------------------------------------------------------- parseo

def test_tiempos_archivo_por_tramos():
    a = {"t0": "2026-01-01T00:00:00Z", "tr": [[1_800_000, 2], [3_600_000, 1]], "c": [1, 2, 3, 4]}
    t = ews.tiempos_archivo(a)
    assert [x.strftime("%H:%M") for x in t] == ["00:00", "00:30", "01:00", "02:00"]
    with pytest.raises(ValueError):
        ews.tiempos_archivo({**a, "c": [1, 2]})
    sin_tr = ews.tiempos_archivo({"t0": "2026-01-01T00:00:00+00:00", "c": [1, 2, 3]})
    assert (sin_tr[1] - sin_tr[0]) == pd.Timedelta(minutes=30)


def test_serie_archivo_fixture_irregular():
    d = json.loads((FX / "dashboard.json").read_text(encoding="utf-8"))
    s = ews.serie_archivo(d)
    assert len(s) == len(d["trends"]["archive"]["c"])
    assert s["ts"].iloc[-1] == pd.Timestamp(d["current"]["asOf"])          # el último instante es asOf
    assert (s["ts"].diff().dt.total_seconds().div(60).dropna() == 60).sum() == 2


def test_leer_rss():
    items = ews.leer_rss((FX / "rss.xml").read_text(encoding="utf-8"))
    assert len(items) == 1 and items[0]["fecha"].startswith("Mon, 09 Feb 2026")
    vacio = '<?xml version="1.0"?><rss version="2.0"><channel><title>x</title></channel></rss>'
    assert ews.leer_rss(vacio) == []


# --------------------------------------------------------------------------- z causal (TimeGate)

def test_z_causal_no_mira_el_futuro():
    corta = ews.serie_archivo(_panel(48 * 7 * 6, pico=False))
    larga = ews.serie_archivo(_panel(48 * 7 * 8))
    z_c, z_l = ews.z_causal(corta), ews.z_causal(larga)
    n = len(corta)
    # Añadir futuro no cambia ningún z pasado.
    pd.testing.assert_series_equal(z_c, z_l.iloc[:n], check_names=False)
    # Las primeras 4 semanas no tienen línea base (mínimo 4 observaciones del mismo hueco). En UTC, sin el
    # cambio de hora de noviembre (que en Los Ángeles repite una hora y adelanta dos huecos).
    z_utc = ews.z_causal(larga, tz="UTC")
    assert z_utc.iloc[: 48 * 7 * 4].isna().all() and z_utc.iloc[48 * 7 * 4:].notna().all()
    assert z_l.iloc[-48 * 7:].notna().all()
    assert z_l.iloc[-1] > ews.ALARMA and ews.nivel(z_l.iloc[-1]) == 5


def test_z_causal_piso_reduce_alarmas_por_varianza_baja():
    s = ews.serie_archivo(_panel(48 * 7 * 6, ruido=0.5))
    z0 = ews.z_causal(s)
    z1 = ews.z_causal(s, piso_rel=0.12)
    m = z0.notna()
    assert m.sum() > 0 and (z0[m].abs() >= z1[m].abs() - 1e-9).all() and z1[m].abs().max() < z0[m].abs().max()


def test_nivel():
    assert ews.nivel(None) == 0 and ews.nivel(float("nan")) == 0
    assert ews.nivel(0.1) == 1 and ews.nivel(1.5) == 2 and ews.nivel(2.5) == 3
    assert ews.nivel(4.0) == 4 and ews.nivel(4.7) == 5


def test_revision_detecta_cambios_ex_post():
    a = ews.serie_archivo(_panel(100))
    b = a.copy()
    b.loc[10:19, "esperado"] += 1.0
    b.loc[10:14, "z_publicado"] += 0.04
    r = ews.revision(a, b.iloc[5:])
    assert r["n_comunes"] == 95 and r["pct_conteo_cambia"] == 0.0
    assert r["pct_esperado_cambia"] == pytest.approx(10 / 95 * 100, abs=0.01)
    assert r["max_dif_z_publicado"] == pytest.approx(0.04)


def test_series_ews_fechas_de_publicacion():
    s = ews.serie_archivo(_panel(48 * 7 * 5))
    s["z_causal"] = ews.z_causal(s)
    series = ews.series_ews(s, "2026-01-01")
    for nombre, larga in series.items():
        v = base.serie_larga(larga)
        assert (v["fecha_publicacion"] >= v["fecha_referencia"]).all(), nombre
    zp = base.serie_larga(series["ews_z_publicado"])
    assert (zp["fecha_publicacion"] == pd.Timestamp("2026-01-01")).all()     # revisado: solo al vintage
    act = base.serie_larga(series["ews_actividad"])
    diario = act[act["serie"] == "ews_conteo_diario_medio"]
    assert ((diario["fecha_publicacion"] - diario["fecha_referencia"]).dt.days == 1).all()


# --------------------------------------------------------------------------- punta a punta sin red

def test_sondear_sin_red_modo_rapido(sin_red, red_falsa, sin_robots):
    red_falsa(ews, RUTAS)
    # Instantánea anterior archivada con el 'esperado' distinto en una franja -> revisión medida.
    d = json.loads((FX / "dashboard.json").read_text(encoding="utf-8"))
    d["trends"]["archive"]["p"] = [x + (1.0 if 100 <= i < 200 else 0) for i, x in enumerate(d["trends"]["archive"]["p"])]
    d["snapshotGeneratedAt"] = "2026-02-01T00:00:00.000Z"
    (base.corpus("crudo", "ews") / "dashboard_20260201T000000Z.json").write_text(json.dumps(d), encoding="utf-8")

    res = ews.sondear(_cfg(modo_rapido=True))
    assert set(res.series) == {"ews_actividad_rapido", "ews_z_publicado_rapido", "ews_z_causal_rapido"}
    for nombre in res.series:
        assert not base.leer_serie(nombre).empty
    cob = res.cobertura[0]
    assert set(cob) == set(base.COLUMNAS_COBERTURA)
    assert cob["backtest"] is False and cob["destino"] == "vivo" and cob["estado"] == "ok" and cob["hora"] is True
    assert cob["lag_mediana_h"] == pytest.approx(4 / 60, abs=0.01)
    tz = res.tablas["ews_z"]
    assert {"ts", "valor", "z_publicado", "z_causal", "nivel_publicado"} <= set(tz.columns)
    al = res.tablas["ews_alarmas"]
    assert {"umbral", "n_alarmas_publicado", "n_alarmas_causal"} <= set(al.columns)
    assert al.loc[al["umbral"] == ews.ALARMA, "n_alarmas_causal"].iloc[0] >= 1     # el pico final
    rev = res.tablas["ews_revision"].iloc[0]
    assert rev["pct_esperado_cambia"] > 0 and rev["pct_conteo_cambia"] == 0
    f = res.fichas[0]
    assert f.estado_sonda == "ok" and f.mediciones["rss_items"] == 1 and f.mediciones["puntos_archivo"] == len(tz)
    assert "otros_paneles" not in f.mediciones        # modo rápido: sin paneles secundarios
    # la serie guardada es el z causal CON suelo de desviación (sin los z absurdos de sd≈0)
    zc = base.leer_serie("ews_z_causal_rapido")
    assert len(zc) == tz["z_causal_piso"].notna().sum()
    assert zc["valor"].round(2).tolist() == tz["z_causal_piso"].dropna().tolist()
    assert base.leer_serie("ews_z_causal").empty                           # el modo rápido no pisa
    # Nada de matrículas en las tablas de results.
    assert not any("registration" in c or "SINTETICO" in str(t.values) for t in res.tablas.values() for c in t.columns)


def test_sondear_panel_caido_usa_ultima_instantanea(sin_red, red_falsa, sin_robots):
    (base.corpus("crudo", "ews") / "dashboard_20260209T020400Z.json").write_bytes((FX / "dashboard.json").read_bytes())
    red_falsa(ews, {"dashboard.json": (503, b""), "rss.xml": "ews/rss.xml"})
    res = ews.sondear(_cfg(modo_rapido=True))
    assert res.cobertura[0]["estado"] == "parcial" and res.fichas[0].mediciones["desde_cache"] is True
    assert "ews_z_causal_rapido" in res.series


@pytest.mark.live
def test_live_ews(sin_red):
    res = ews.sondear(base.ConfigSonda(modo_rapido=True, verbose=False))
    m = res.fichas[0].mediciones
    assert res.cobertura[0]["backtest"] is False and m["puntos_archivo"] > 1000
    assert m["minutos_desde_ultima_muestra"] < 24 * 60
