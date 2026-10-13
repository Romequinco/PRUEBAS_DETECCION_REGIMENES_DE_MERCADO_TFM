"""La señal de un baseline no puede usar el cierre con el que se ejecuta (sin red, precios sintéticos).

Convención (``cartera.baselines``): se opera al cierre de la fecha de control; ``estrategia.pesos``
recibe las vistas recortadas a ``sesion < fecha``. Una estrategia espía registra lo que ve: si el motor
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
