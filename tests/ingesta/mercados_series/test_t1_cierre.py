"""Cierre del Tema 1 · truncado real de ``get_serie`` y ``get_precios`` sobre la base y el almacén guardados.

Propiedad que se comprueba: lo que las tools devuelven en ``t`` no cambia al quitar de la base (o del
almacén de precios) todo lo publicado después de ``t``. Se compara la consulta por defecto (leída del
disco), la misma con la tabla completa en memoria y la misma con la tabla truncada en ``t``, en 5 series
de métodos distintos, 5 tickers de mercados y divisas distintos y 3 fechas.

Necesita ``<corpus>/mercados_series/series/base_series.parquet`` y los parquet de
``<corpus>/mercados_series/precios/yahoo/``: lleva ``@pytest.mark.datos`` y se salta si faltan
(``python -m ingesta.mercados_series construir-base --sin-red`` y la descarga del núcleo de Yahoo).
"""
from __future__ import annotations

import pandas as pd
import pytest

from ingesta.mercados_series import almacen, divisas, tools

# Una serie por método de publicación: vintage_exacto (ALFRED), calendario (H.15 y BCE), fichero (Ken French
# por edición) y vintage_exacto por edición de la OCDE.
SERIES = ("fred:PAYEMS", "fred:DGS10", "ecb:EXR.D.USD.EUR", "ff:F3M.Mkt-RF", "oecd:CP.DEU")
# Mercados y divisas distintos: NYSE Arca, Xetra (EUR), LSE (GBp), Tokio (JPY) e índice de NYSE.
TICKERS = ("SPY", "SAP.DE", "VOD.L", "7203.T", "^GSPC")
FECHAS = ("2008-09-15", "2020-03-16", "2024-06-28")


def _iguales(a: pd.DataFrame, b: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_dtype=True)


@pytest.fixture(scope="module")
def base_real() -> pd.DataFrame:
    if not almacen.ruta_series().exists():
        pytest.skip(f"falta la base real {almacen.ruta_series()} (construir-base --sin-red)")
    tabla = almacen.leer_series(list(SERIES), validar=False)
    faltan = set(SERIES) - set(tabla["serie"])
    if faltan:
        pytest.skip(f"base guardada sin {sorted(faltan)}: reconstruir con todos los cargadores")
    return tabla


@pytest.fixture(scope="module")
def precios_reales() -> pd.DataFrame:
    guardados = set(almacen.tickers_guardados("yahoo"))
    faltan = set(TICKERS) - guardados
    if faltan:
        pytest.skip(f"almacén de Yahoo sin {sorted(faltan)} (descarga del núcleo)")
    return pd.concat([almacen.leer_precios(t, "yahoo") for t in TICKERS], ignore_index=True)


@pytest.fixture(scope="module")
def fx_real() -> pd.DataFrame:
    ids = [divisas.SERIE_USD_EUR] + [divisas.SERIE_FX.format(divisa=d) for d in ("GBP", "JPY")]
    return almacen.leer_series(sorted(set(ids)), validar=False)


@pytest.mark.datos
@pytest.mark.parametrize("fecha", FECHAS)
@pytest.mark.parametrize("serie", SERIES)
def test_t1_cierre_truncado_get_serie(base_real, serie, fecha):
    t = pd.Timestamp(fecha)
    completa = base_real[base_real["serie"] == serie]
    truncada = completa[completa["fecha_publicacion"] <= t]
    try:
        disco = tools.get_serie(serie, fecha)
    except KeyError:
        # Nada publicado hasta t: la tabla truncada responde igual (mismo error neutro).
        assert truncada.empty
        with pytest.raises(KeyError):
            tools.get_serie(serie, fecha, base=truncada)
        return
    memoria = tools.get_serie(serie, fecha, base=completa)
    corta = tools.get_serie(serie, fecha, base=truncada)
    _iguales(disco.datos, memoria.datos)
    _iguales(memoria.datos, corta.datos)
    assert (disco.unidad, disco.divisa, disco.fuente, disco.metodo_publicacion) == \
           (corta.unidad, corta.divisa, corta.fuente, corta.metodo_publicacion)
    assert disco.avisos == corta.avisos
    assert (disco.datos["fecha_publicacion"] <= t).all()


@pytest.mark.datos
@pytest.mark.parametrize("fecha", FECHAS)
def test_t1_cierre_truncado_get_precios(precios_reales, fx_real, fecha):
    t = pd.Timestamp(fecha)
    truncados = precios_reales[precios_reales["fecha_publicacion"] <= t]
    fx_truncado = fx_real[fx_real["fecha_publicacion"] <= t]
    for ajuste in ("ninguno", "total"):
        disco = tools.get_precios(list(TICKERS), fecha, ajuste=ajuste, divisa="original")
        memoria = tools.get_precios(list(TICKERS), fecha, ajuste=ajuste, divisa="original", base=precios_reales)
        corta = tools.get_precios(list(TICKERS), fecha, ajuste=ajuste, divisa="original", base=truncados)
        _iguales(disco.datos, memoria.datos)
        _iguales(memoria.datos, corta.datos)
        assert disco.avisos == corta.avisos
        assert (disco.datos["fecha_publicacion"] <= t).all()
    # En USD el FX del BCE también se trunca: la conversión no ve fixings posteriores a t.
    usd = tools.get_precios(list(TICKERS), fecha, base=precios_reales, base_fx=fx_real)
    usd_corta = tools.get_precios(list(TICKERS), fecha, base=truncados, base_fx=fx_truncado)
    _iguales(usd.datos, usd_corta.datos)
    assert usd.avisos == usd_corta.avisos
