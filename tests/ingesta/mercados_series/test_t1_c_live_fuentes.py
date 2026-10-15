"""Cierre del Tema 1 · pruebas con red de las fuentes que no tenían test ``live``.

Cada prueba hace una petición pequeña con ``ingesta.comun.http`` (robots y ritmo por host), escribe los
crudos en ``tmp_path`` (nunca en el corpus) y comprueba que el lector del extractor sigue entendiendo el
formato: tabla no vacía y con datos recientes. Un bloqueo (403, reto) se salta: no se esquiva.
"""
from __future__ import annotations

import pandas as pd
import pytest

from ingesta.comun import http

pytestmark = pytest.mark.live
HOY = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()


def _get(url: str, **kw):
    if not http.robots_permite(url):
        pytest.skip(f"robots.txt no permite {url}")
    try:
        return http.get(url, reintentos=2, timeout=90, **kw)
    except Exception as e:  # noqa: BLE001 — bloqueo o caída: se salta, no se esquiva
        pytest.skip(f"sin respuesta de {url}: {str(e)[:120]}")


@pytest.mark.parametrize("sid", ["boj:FM01.STRDCLUCON",
                                 "bbk:BBSIS.D.I.ZST.ZI.EUR.S1311.B.A604.R10XX.R.A.A._Z._Z.A",
                                 "rba:FIRMMCRTD"])
def test_t1_c_live_bancos_centrales(sid):
    from ingesta.mercados_series.fuentes import bancos_centrales as bc

    info = bc.SERIES[sid]
    r = _get(info["url"], params=info.get("params"))
    s = bc.leer(sid, r.content)
    assert len(s) > 100
    assert s.index.max() >= HOY - pd.Timedelta(days=120)


def test_t1_c_live_worldbank():
    from ingesta.mercados_series.fuentes import worldbank as wb

    r = _get(wb.URL_INDICADOR.format(indicador="NY.GDP.MKTP.KD.ZG"),
             params={"format": "json", "per_page": 20000, "date": "2015:2025"})
    datos, info = wb.leer_indicador(r.content)
    assert len(datos) > 100
    assert pd.to_numeric(datos.filter(like="anio").iloc[:, 0], errors="coerce").max() >= HOY.year - 3 \
        if datos.filter(like="anio").shape[1] else len(datos) > 100


@pytest.mark.parametrize("fuente", ["ust_curva", "boc_curva"])  # bce_curvas: histórico entero (lento); mismo host que la prueba live del BCE
def test_t1_c_live_curvas(fuente, tmp_path):
    from ingesta.mercados_series.fuentes import curvas

    metas = curvas.descargar(fuente, desde_anio=HOY.year, forzar=True, raiz=tmp_path)
    if any(m.get("error") for m in metas):
        pytest.skip(f"{fuente}: {[m.get('error') for m in metas][:2]}")
    tabla = curvas.ancho_desde_crudos(fuente, tmp_path)
    assert len(tabla) > 0
    fechas = pd.to_datetime(tabla["fecha"] if "fecha" in tabla else tabla.index)
    assert fechas.max() >= HOY - pd.Timedelta(days=15)


@pytest.mark.parametrize("fuente", ["shiller", "goyal_welch"])
def test_t1_c_live_academicos(fuente, tmp_path):
    from ingesta.mercados_series.fuentes import academicos

    metas = academicos.descargar(fuente, forzar=True, raiz=tmp_path)
    if any(m.get("error") for m in metas):
        pytest.skip(f"{fuente}: {[m.get('error') for m in metas][:2]}")
    assert metas and all(m.get("bytes", 0) > 10_000 for m in metas)


def test_t1_c_live_imf_sdmx(tmp_path):
    from ingesta.mercados_series.fuentes import imf

    contenido, meta = imf.descargar_sdmx(imf.VERSIONES_SDMX[0], forzar=True, raiz=tmp_path)
    if contenido is None:
        pytest.skip(f"FMI SDMX: {meta.get('error')}")
    datos, edicion = imf.leer_sdmx(contenido)
    assert len(datos) > 100 and edicion


def test_t1_c_live_philly_y_oecd_robots():
    from ingesta.mercados_series.fuentes import oecd, philly_rtdsm

    r = _get(philly_rtdsm.URL_PAGINA.format(var="routput"))
    assert r.status_code == 200
    assert http.robots_permite(oecd.URL_SDMX.split("?")[0])
    assert not http.robots_permite(oecd.URL_DBNOMICS)  # DBnomics sigue vetado (Disallow: /)
