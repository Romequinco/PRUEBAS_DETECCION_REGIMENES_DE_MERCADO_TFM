"""La señal de un baseline no puede usar el cierre con el que se ejecuta (sin red, precios sintéticos).

Convención (``cartera.baselines``): se opera al cierre de la fecha de control (el último publicado a
esa fecha); ``estrategia.pesos`` recibe las vistas sin ese cierre ni nada posterior. Una estrategia espía registra lo que ve: si el motor
le pasara la sesión de la fecha de control (o una posterior), el test falla. Otra estrategia que
«mira» el cierre de ejecución (compra si sube ese día) daría otra curva: se comprueba que no puede.
"""
from __future__ import annotations

import pandas as pd

from ingesta.mercados_series.cartera import baselines as b
from test_t1_b8_cartera import precios_sinteticos  # fixture sintética compartida (misma carpeta)


def _precios() -> dict[str, pd.DataFrame]:
    return {"AAA": precios_sinteticos("AAA", "2010-01-04", "2011-12-30", semilla=5),
            "BBB": precios_sinteticos("BBB", "2010-01-04", "2011-12-30", semilla=6, mercado="XLON")}


def test_t1_a_baselines_senal_no_usa_cierre_de_ejecucion():
    vistos: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def pesos(fecha, vistas, actuales):
        for v in vistas.values():
            if v is not None and len(v):
                vistos.append((fecha, v["sesion"].max()))
        return {"AAA": 0.5, "BBB": 0.5}

    est = b.Estrategia(nombre="espia", universo=lambda f: ["AAA", "BBB"], pesos=pesos)
    res = b.simular(est, _precios(), "2010-03-01", "2011-12-30")
    assert len(vistos) > 20 and len(res.rebalanceos) > 20
    assert all(ultima < fecha.normalize() for fecha, ultima in vistos), "la señal vio el cierre de ejecución"


def test_t1_a_baselines_estrategia_que_mira_el_cierre_no_lo_ve():
    """Una regla que dependiera del cierre de ``t`` (sube ese día → AAA) no puede activarse con él."""
    usados: list[bool] = []

    def pesos(fecha, vistas, actuales):
        v = vistas["AAA"]
        usados.append(bool(len(v)) and v["sesion"].iloc[-1] == fecha.normalize())
        return {"AAA": 1.0}

    b.simular(b.Estrategia(nombre="mira", universo=lambda f: ["AAA"], pesos=pesos), _precios(),
              "2010-03-01", "2011-06-30")
    assert usados and not any(usados)


def test_t1_a_vista_senal_recorta_la_sesion_de_la_fecha():
    p = precios_sinteticos("AAA", "2010-01-04", "2010-01-29")
    v = b.vista_senal(p, pd.Timestamp("2010-01-15"))
    assert v["sesion"].max() == pd.Timestamp("2010-01-14")
    assert b.vista_senal(None, pd.Timestamp("2010-01-15")) is None


def _espia_ejecucion(precios, inicio, fin):
    """(cierre de ejecución, último cierre que ve la señal) en cada control."""
    pares: list[tuple[float, float]] = []

    def pesos(fecha, vistas, actuales):
        tk = next(iter(precios))
        completa = b._vista(tk, fecha, precios[tk], fecha - pd.Timedelta(days=30), None, [])
        ejec = b.precio_en(completa, [fecha])[0]
        v = vistas[tk]
        if v is not None and len(v):
            pares.append((ejec, float(v["close"].iloc[-1])))
        return {tk: 1.0}

    b.simular(b.Estrategia(nombre="espia", universo=lambda f: list(precios), pesos=pesos), precios, inicio, fin)
    return pares


def test_t1_a_baselines_senal_d1_no_ve_el_cierre_de_ejecucion():
    """Regresión (revisión): con un NAV D+1 la sesión de ``t − 1`` se publica en ``t`` y es justo el cierre
    con el que se ejecuta; recortar por ``sesion < fecha`` la dejaba en la señal."""
    nav = {"NAV": precios_sinteticos("NAV", "2010-01-04", "2011-06-30", semilla=7, retraso_pub=1, mercado="NAV_US")}
    pares = _espia_ejecucion(nav, "2010-03-01", "2011-06-30")
    assert len(pares) > 10 and all(ejec != visto for ejec, visto in pares)


def test_t1_a_vista_senal_control_en_festivo():
    """Regresión (revisión): si el control cae en festivo del mercado (2021-05-31, Memorial Day), se ejecuta
    al cierre del 2021-05-28 y la señal no puede verlo."""
    p = precios_sinteticos("AAA", "2021-05-03", "2021-06-30")
    p = p[p["sesion"] != pd.Timestamp("2021-05-31")].reset_index(drop=True)
    fecha = pd.Timestamp("2021-05-31")
    v = b.vista_senal(p, fecha)
    assert b.precio_en(p, [fecha])[0] == p.loc[p["sesion"] == "2021-05-28", "close"].iloc[0]
    assert v["sesion"].max() == pd.Timestamp("2021-05-27")
    d1 = precios_sinteticos("NAV", "2021-05-03", "2021-06-30", retraso_pub=1, mercado="NAV_US")
    assert b.vista_senal(d1, pd.Timestamp("2021-06-15"))["sesion"].max() == pd.Timestamp("2021-06-11")
