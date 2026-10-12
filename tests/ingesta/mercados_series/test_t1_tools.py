"""Tests de las tools del Tema 1 (``get_serie``, ``get_precios``), de extremo a extremo.

Los datos sintéticos se escriben como parquet en ``tmp_path`` (``almacen`` redirigido) y las
tools los leen del almacén, sin ``base=``. Sin red.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ingesta.mercados_series import almacen, divisas, timegate, tools
from ingesta.mercados_series import sinteticos as s
from ingesta.mercados_series.almacen import escribir_precios, escribir_series
from ingesta.mercados_series.contrato import validar_series
from ingesta.mercados_series.tools import get_precios, get_serie

ALIAS = {"PARO": s.SERIE_VINTAGES}


@pytest.fixture(autouse=True)
def corpus_en_tmp(tmp_path, monkeypatch):
    """Las rutas por defecto del almacén apuntan a ``tmp_path`` (nunca al corpus real)."""
    def corpus_dir(tema=None):
        p = tmp_path / "corpus" / tema if tema else tmp_path / "corpus"
        p.mkdir(parents=True, exist_ok=True)
        return p
    monkeypatch.setattr(almacen, "corpus_dir", corpus_dir)
    return tmp_path / "corpus"


@pytest.fixture(scope="module")
def vintages():
    return s.serie_con_vintages()


@pytest.fixture(scope="module")
def fx():
    return s.fx_ecb_sintetico(desde="2022-01-03", hasta="2024-06-28")


@pytest.fixture(scope="module")
def accion():
    return s.accion_con_eventos()


@pytest.fixture(scope="module")
def multi():
    return s.precios_multimercado()


@pytest.fixture
def almacen_lleno(vintages, fx, accion, multi):
    """Series (paro con vintages + FX del BCE) y precios (ZZAC + multimercado) en el almacén."""
    escribir_series(pd.concat([vintages, fx], ignore_index=True))
    escribir_precios(pd.concat([accion, multi], ignore_index=True), "yahoo")


def _as_of_ingenuo(tabla: pd.DataFrame, serie: str, fecha: str) -> pd.Series:
    """Valor visible por periodo: fila a fila, mayor fecha_publicacion <= fecha (y mayor vintage)."""
    f = pd.Timestamp(fecha)
    out = {}
    for ref, grupo in tabla[tabla["serie"] == serie].groupby("fecha_referencia"):
        vis = grupo[grupo["fecha_publicacion"] <= f]
        if len(vis):
            out[ref] = vis.sort_values(["fecha_publicacion", "vintage"]).iloc[-1]["valor"]
    return pd.Series(out, dtype=float)


# --- get_serie ----------------------------------------------------------------------------------
@pytest.mark.parametrize("fecha", ["2023-03-01", "2024-02-02", "2025-06-15", "2025-11-21", "2026-03-31"])
def test_t1_get_serie_coincide_con_as_of_ingenuo(almacen_lleno, vintages, fecha):
    r = get_serie(s.SERIE_VINTAGES, fecha)
    esperado = _as_of_ingenuo(vintages, s.SERIE_VINTAGES, fecha)
    obtenido = pd.Series(r.datos["valor"].to_numpy(), index=pd.DatetimeIndex(r.datos["fecha_referencia"]))
    pd.testing.assert_index_equal(obtenido.index, pd.DatetimeIndex(esperado.index), check_names=False)
    np.testing.assert_array_equal(obtenido.to_numpy(), esperado.to_numpy())
    assert (r.datos["fecha_publicacion"] <= pd.Timestamp(fecha)).all()
    assert r.unidad == "porcentaje" and r.fuente == s.FUENTE and r.divisa is None


def test_t1_get_serie_vintage_cambia_con_la_fecha(almacen_lleno, vintages):
    """Un mes con revisiones: cada fecha de decisión ve la vintage publicada hasta entonces."""
    mes = pd.Timestamp("2024-03-01")
    filas = vintages[(vintages["fecha_referencia"] == mes)].sort_values("fecha_publicacion")
    assert len(filas) >= 2
    for _, fila in filas.iterrows():
        r = get_serie(s.SERIE_VINTAGES, fila["fecha_publicacion"])
        ultimo = r.datos.set_index("fecha_referencia").loc[mes]
        assert ultimo["vintage"] == fila["vintage"]
        assert ultimo["valor"] == fila["valor"]
    antes = filas["fecha_publicacion"].iloc[0] - pd.Timedelta(days=1)
    assert mes not in set(get_serie(s.SERIE_VINTAGES, antes).datos["fecha_referencia"])


def test_t1_get_serie_truncado_almacen(almacen_lleno, vintages, tmp_path):
    """Añadir al almacén vintages publicadas después de ``fecha`` no cambia lo visto en ``fecha``."""
    fecha = pd.Timestamp("2025-02-10")
    pasado = vintages[vintages["fecha_publicacion"] <= fecha]
    ruta = tmp_path / "solo_pasado.parquet"
    escribir_series(pasado, ruta=ruta)
    r_pasado = get_serie(s.SERIE_VINTAGES, fecha, base=almacen.leer_series(ruta=ruta))
    r_todo = get_serie(s.SERIE_VINTAGES, fecha)
    pd.testing.assert_frame_equal(r_pasado.datos, r_todo.datos)
    assert r_pasado.ultimo() == r_todo.ultimo()


def test_t1_get_serie_retraso_y_hueco(almacen_lleno):
    r = get_serie(s.SERIE_VINTAGES, s.FECHA_PREVISTA_RETRASADO + pd.Timedelta(days=7))
    assert s.MES_RETRASADO not in set(r.datos["fecha_referencia"])
    r = get_serie(s.SERIE_VINTAGES, s.INFORME_HUECO)
    fila = r.datos.set_index("fecha_referencia").loc[s.MES_HUECO]
    assert np.isnan(fila["valor"]) and fila["nota"] == s.NOTA_HUECO
    assert any("hueco" in a and s.MES_HUECO.date().isoformat() in a for a in r.avisos)
    retrasado = r.datos.set_index("fecha_referencia").loc[s.MES_RETRASADO]
    assert retrasado["valor"] == retrasado["valor"]   # no nulo


def test_t1_get_serie_ultimo_citable_con_hueco(almacen_lleno):
    r = get_serie(s.SERIE_VINTAGES, s.INFORME_HUECO, desde="2025-08-01")
    assert list(r.datos["fecha_referencia"].dt.strftime("%Y-%m")) == ["2025-08", "2025-09", "2025-10",
                                                                        "2025-11"]
    u = r.ultimo()
    assert u["fecha_referencia"] == "2025-11-01" and u["fecha_publicacion"] == "2025-12-16"
    assert isinstance(u["valor"], float) and u["fuente"] == s.FUENTE
    assert u["metodo_publicacion"] == "vintage_exacto" and u["fecha_consulta"] == "2025-12-16"
    solo_hueco = get_serie(s.SERIE_VINTAGES, s.INFORME_HUECO, desde="2025-10-01").datos.iloc[:1]
    assert solo_hueco["valor"].isna().all()


def test_t1_get_serie_antes_del_primer_dato_igual_que_desconocida(almacen_lleno, vintages):
    """Regresión: antes del primer dato, el aviso revelaba la fecha FUTURA de la primera publicación.

    Ahora una serie sin nada publicado hasta ``fecha`` se trata igual que una que no existe.
    """
    primera = vintages["fecha_publicacion"].min()
    for fecha in ("2022-06-01", (primera - pd.Timedelta(days=1)).date().isoformat()):
        with pytest.raises(KeyError) as futura:
            get_serie(s.SERIE_VINTAGES, fecha)
        with pytest.raises(KeyError) as inexistente:
            get_serie(s.SERIE_VINTAGES, fecha, base=vintages.iloc[:0])
        assert str(futura.value) == str(inexistente.value)
        assert primera.date().isoformat() not in str(futura.value)
    assert get_serie(s.SERIE_VINTAGES, primera).ultimo()["fecha_publicacion"] == primera.date().isoformat()


def test_t1_get_serie_id_desconocido_keyerror(almacen_lleno):
    with pytest.raises(KeyError, match="desconocida"):
        get_serie("fred:NO_EXISTE", "2025-01-01")
    with pytest.raises(KeyError, match="alias"):
        get_serie("PARO", "2025-01-01", alias={"PARO": "sint:OTRA"})


def test_t1_get_serie_desde_sin_periodos_avisa(almacen_lleno):
    r = get_serie(s.SERIE_VINTAGES, "2024-03-15", desde="2025-01-01")
    assert r.datos.empty and r.ultimo() is None and any("desde 2025-01-01" in a for a in r.avisos)
    assert r.unidad == "porcentaje"


def test_t1_get_serie_alias(almacen_lleno):
    a = get_serie("PARO", "2025-01-15", alias=ALIAS)
    b = get_serie(s.SERIE_VINTAGES, "2025-01-15")
    assert a.serie == s.SERIE_VINTAGES
    pd.testing.assert_frame_equal(a.datos, b.datos)


def test_t1_get_serie_alias_por_defecto_de_publicaciones():
    assert tools.resolver_id("VIX") == "fred:VIXCLS"
    assert tools.resolver_id("fred:DGS10") == "fred:DGS10"


def test_t1_get_serie_fecha_y_divisa_invalidas(vintages):
    with pytest.raises(ValueError):
        get_serie(s.SERIE_VINTAGES, "no-es-fecha", base=vintages)
    with pytest.raises(ValueError):
        get_serie(s.SERIE_VINTAGES, pd.Timestamp("2025-01-01", tz="UTC"), base=vintages)
    with pytest.raises(ValueError, match="divisa"):
        get_serie(s.SERIE_VINTAGES, "2025-01-01", divisa="EUR", base=vintages)


def _serie_lag_fijo() -> pd.DataFrame:
    meses = pd.date_range("2024-01-01", "2024-06-01", freq="MS")
    return validar_series(pd.DataFrame({
        "serie": "sint:PIB_EUR", "fecha_referencia": meses,
        "fecha_publicacion": meses + pd.offsets.MonthEnd(0) + pd.Timedelta(days=45),
        "disponible_utc": pd.NaT, "vintage": "unico", "valor": np.linspace(100.0, 105.0, len(meses)),
        "unidad": "millones de EUR", "divisa": "EUR", "fuente": "sintetico",
        "metodo_publicacion": "lag_fijo", "nota": None,
    }))


def test_t1_get_serie_lag_fijo_avisa(fx):
    base = pd.concat([_serie_lag_fijo(), fx], ignore_index=True)
    r = get_serie("sint:PIB_EUR", "2024-12-31", base=base)
    assert r.metodo_publicacion == "lag_fijo" and len(r.datos) == 6
    assert any("por regla" in a and "lag_fijo" in a for a in r.avisos)
    assert not any("conservador" in a for a in r.avisos)   # regresión: no lo es en los cierres del Gobierno


def test_t1_get_serie_a_usd(fx):
    base = pd.concat([_serie_lag_fijo(), fx], ignore_index=True)
    fecha = "2024-12-31"
    orig = get_serie("sint:PIB_EUR", fecha, base=base)
    usd = get_serie("sint:PIB_EUR", fecha, base=base, divisa="USD")
    assert orig.divisa == "EUR" and usd.divisa == "USD"
    usd_eur = divisas.fx_usd_por_unidad(fx, "EUR", fecha)
    for _, fila in usd.datos.iterrows():
        # serie mensual: el fixing del ÚLTIMO día del mes (o el último anterior), no el del día 1
        fin_mes = fila["fecha_referencia"] + pd.offsets.MonthEnd(0)
        fixing = usd_eur[usd_eur.index <= fin_mes]
        assert fila["fx_usd"] == pytest.approx(fixing.iloc[-1])
        assert fila["fx_fecha"] == fixing.index[-1] and fila["fx_fecha"].month == fin_mes.month
    np.testing.assert_allclose(usd.datos["valor"], orig.datos["valor"] * usd.datos["fx_usd"])
    assert usd.ultimo()["divisa"] == "USD"
    # Serie no monetaria: se devuelve sin convertir, con aviso y las mismas columnas FX (nulas).
    paro = get_serie(s.SERIE_VINTAGES, fecha, base=s.serie_con_vintages(), divisa="USD")
    assert paro.divisa is None and any("no es monetaria" in a for a in paro.avisos)
    assert list(paro.datos.columns) == list(usd.datos.columns) and paro.datos["fx_usd"].isna().all()


def test_t1_get_serie_fx_fin_de_trimestre_y_serie_en_usd(fx):
    """Trimestral: fixing del fin de trimestre. Ya en USD: fx_usd = 1 y las mismas columnas que get_precios."""
    trim = _serie_lag_fijo().iloc[[0, 3]].assign(serie="sint:PIB_T")
    r = get_serie("sint:PIB_T", "2024-12-31", base=pd.concat([trim, fx], ignore_index=True), divisa="USD")
    assert r.datos["fx_fecha"].dt.month.tolist() == [3, 6]
    en_usd = validar_series(trim.assign(serie="sint:PIB_USD", divisa="USD", unidad="millones de USD"))
    r = get_serie("sint:PIB_USD", "2024-12-31", base=en_usd, divisa="USD")
    assert (r.datos["fx_usd"] == 1.0).all() and (r.datos["divisa_original"] == "USD").all()
    assert list(r.datos.columns) == [*tools.COLUMNAS_DATOS_SERIE, "divisa_original", "fx_usd", "fx_fecha"]
    np.testing.assert_array_equal(r.datos["valor"], en_usd["valor"])


# --- get_precios --------------------------------------------------------------------------------
def test_t1_get_precios_split_futuro_no_se_filtra(almacen_lleno, accion):
    """Antes del split, el precio 'ajustado' es el as-traded: el split futuro no existe aún."""
    antes = s.SESION_SPLIT - pd.Timedelta(days=3)
    r = get_precios(s.TICKER_ACCION, antes, ajuste="splits", divisa="original")
    crudo = accion[accion["fecha_publicacion"] <= antes]
    np.testing.assert_array_equal(r.datos["close"].to_numpy(), crudo["close"].to_numpy())
    np.testing.assert_array_equal(r.datos["volume"].to_numpy(), crudo["volume"].to_numpy())
    # Después del split, las sesiones previas se expresan en acciones nuevas (la mitad).
    despues = get_precios(s.TICKER_ACCION, s.SESION_SPLIT, ajuste="splits", divisa="original")
    previas = despues.datos[despues.datos["sesion"] < s.SESION_SPLIT]
    np.testing.assert_allclose(previas["close"].to_numpy(),
                               crudo["close"].to_numpy()[: len(previas)] / s.RATIO_SPLIT)


@pytest.mark.parametrize("ajuste", tools.AJUSTES)
@pytest.mark.parametrize("fecha", ["2022-11-14", "2023-06-09", "2023-06-12", "2023-08-16", "2024-12-31"])
def test_t1_get_precios_truncado_almacen(almacen_lleno, accion, tmp_path, ajuste, fecha):
    """Almacén con todo vs. almacén solo con lo publicado hasta ``fecha``: resultado idéntico."""
    carpeta = tmp_path / "solo_pasado"
    escribir_precios(accion[accion["fecha_publicacion"] <= pd.Timestamp(fecha)], "yahoo", ruta=carpeta)
    pasado = almacen.leer_precios(s.TICKER_ACCION, ruta=carpeta)
    a = get_precios(s.TICKER_ACCION, fecha, ajuste=ajuste, base=pasado)
    b = get_precios(s.TICKER_ACCION, fecha, ajuste=ajuste)
    pd.testing.assert_frame_equal(a.datos, b.datos)
    assert a.avisos == b.avisos and a.ultimo() == b.ultimo()


def test_t1_get_precios_total_coincide_con_timegate(almacen_lleno, accion):
    fecha = "2023-12-29"
    r = get_precios(s.TICKER_ACCION, fecha, ajuste="total")
    ref = timegate.ajustar_a_fecha(accion, fecha)
    np.testing.assert_array_equal(r.datos["close"].to_numpy(), ref["close_aj_total"].to_numpy())
    np.testing.assert_array_equal(r.datos["dividendo"].to_numpy(), ref["dividendo_aj_splits"].to_numpy())
    np.testing.assert_array_equal(r.datos["volume"].to_numpy(), ref["volume_aj_splits"].to_numpy())
    assert (r.datos["fx_usd"] == 1.0).all() and (r.datos["divisa"] == "USD").all()


def test_t1_get_precios_gbp_a_usd_con_fx_visible(almacen_lleno, multi, fx):
    tk, fecha = "ZZLN.L", "2024-03-08"
    orig = get_precios(tk, fecha, ajuste="ninguno", divisa="original")
    usd = get_precios(tk, fecha, ajuste="ninguno")
    assert (orig.datos["divisa"] == "GBp").all()
    assert (usd.datos["divisa"] == "USD").all() and (usd.datos["divisa_original"] == "GBp").all()
    gbp = fx[fx["serie"] == "ecb:EXR.D.GBP.EUR"].set_index("fecha_referencia")["valor"]
    usd_eur = fx[fx["serie"] == "ecb:EXR.D.USD.EUR"].set_index("fecha_referencia")["valor"]
    esperado = (usd_eur / gbp / 100.0).reindex(pd.DatetimeIndex(usd.datos["sesion"])).to_numpy()
    np.testing.assert_allclose(usd.datos["fx_usd"].to_numpy(), esperado)
    np.testing.assert_allclose(usd.datos["close"].to_numpy(), orig.datos["close"].to_numpy() * esperado)
    np.testing.assert_array_equal(usd.datos["volume"].to_numpy(), orig.datos["volume"].to_numpy())
    u = usd.ultimo()
    assert u["sesion"] == fecha and u["divisa"] == "USD" and u["valor"] == pytest.approx(
        orig.ultimo()["valor"] * esperado[-1])


def test_t1_get_precios_fx_publicado_tarde_no_se_usa(multi, fx):
    """Si el fixing del día aún no era visible en ``fecha``, se usa el anterior (y se avisa)."""
    tarde = fx.copy()
    dia = pd.Timestamp("2024-03-08")
    m = tarde["fecha_referencia"] == dia
    tarde.loc[m, "fecha_publicacion"] = pd.Timestamp("2024-03-11")
    tarde.loc[m, "disponible_utc"] = pd.NaT
    r = get_precios("ZZLN.L", dia, ajuste="ninguno", base=multi, base_fx=validar_series(tarde))
    assert r.datos["fx_fecha"].iloc[-1] == pd.Timestamp("2024-03-07")
    assert any("último fixing visible" in a for a in r.avisos)
    r2 = get_precios("ZZLN.L", "2024-03-11", ajuste="ninguno", base=multi, base_fx=validar_series(tarde),
                     desde=dia)
    assert r2.datos.set_index("sesion").loc[dia, "fx_fecha"] == dia


def test_t1_get_precios_varios_tickers_y_desconocido(almacen_lleno):
    r = get_precios(["ZZNY", "9999.T", "ZZNY", "NO.EXISTE"], "2024-03-15", ajuste="ninguno")
    assert r.tickers == ["ZZNY", "9999.T", "NO.EXISTE"]
    assert set(r.datos["ticker"]) == {"ZZNY", "9999.T"}
    assert any("NO.EXISTE" in a for a in r.avisos)
    ult = r.ultimos()
    assert ult["NO.EXISTE"] is None and ult["ZZNY"]["sesion"] == "2024-03-15"
    with pytest.raises(ValueError, match="varios"):
        r.ultimo()
    # CME cierra a las 17:00 NY: la sesión del día aún no es visible en la decisión de ese día.
    cme = get_precios("ZZ=F", "2024-03-15", ajuste="ninguno")
    assert cme.ultimo()["sesion"] == "2024-03-14"


def test_t1_get_precios_desde_no_cambia_el_ajuste(almacen_lleno):
    fecha, desde = "2024-06-28", "2023-09-01"
    todo = get_precios(s.TICKER_ACCION, fecha)
    recorte = get_precios(s.TICKER_ACCION, fecha, desde=desde)
    esperado = todo.datos[todo.datos["sesion"] >= pd.Timestamp(desde)].reset_index(drop=True)
    pd.testing.assert_frame_equal(recorte.datos, esperado)


def test_t1_get_precios_nada_visible_igual_que_desconocido_y_errores_de_uso(almacen_lleno):
    """Regresión: el aviso revelaba la fecha de la primera sesión futura y si el ticker existiría."""
    r = get_precios(s.TICKER_ACCION, "2021-12-31")
    otro = get_precios(s.TICKER_ACCION, "2021-12-31", base=s.accion_con_eventos().iloc[:0])
    assert r.datos.empty and r.ultimo() is None and r.avisos == otro.avisos
    assert "2022-01-03" not in " ".join(r.avisos) and any("sin sesiones publicadas" in a for a in r.avisos)
    assert list(r.datos.columns) == [*tools.COLUMNAS_DATOS_PRECIOS, "divisa_original", "fx_usd", "fx_fecha"]
    vacio = get_precios("ZZNY", "2024-03-15", desde="2025-01-01")
    assert vacio.datos.empty and any("desde 2025-01-01" in a for a in vacio.avisos)
    with pytest.raises(ValueError, match="ajuste"):
        get_precios(s.TICKER_ACCION, "2024-01-02", ajuste="adj")
    with pytest.raises(ValueError, match="divisa"):
        get_precios(s.TICKER_ACCION, "2024-01-02", divisa="EUR")
    with pytest.raises(ValueError):
        get_precios([], "2024-01-02")


def test_t1_get_precios_sin_columnas_adj(almacen_lleno):
    r = get_precios(s.TICKER_ACCION, "2024-12-31")
    assert not [c for c in r.datos.columns if "adj" in c.lower()]


# --- Truncado de las tools: datos, avisos, metadatos y errores ------------------------------------
def _respuesta(funcion, *args, **kwargs):
    """Respuesta comparable de una tool, o el tipo y el texto de su excepción."""
    try:
        r = funcion(*args, **kwargs)
    except (KeyError, ValueError) as e:
        return ("error", type(e).__name__, str(e))
    meta = {k: v for k, v in vars(r).items() if k != "datos"}
    ultimo = r.ultimo() if isinstance(r, tools.RespuestaSerie) else r.ultimos()
    return r.datos, meta, ultimo


def _iguales(a, b) -> None:
    error_a, error_b = isinstance(a[0], str), isinstance(b[0], str)
    assert error_a == error_b, (a, b)
    if error_a:
        assert a == b
        return
    pd.testing.assert_frame_equal(a[0], b[0])
    assert a[1] == b[1] and a[2] == b[2]


def _truncar(df: pd.DataFrame, t: str) -> pd.DataFrame:
    return df[df["fecha_publicacion"] <= pd.Timestamp(t)].reset_index(drop=True)


# Fechas antes del primer dato, en el borde del split (12-jun-2023) y de un dividendo (15-nov-2022),
# en el retraso y el hueco del paro, y después de todo.
FECHAS_TRUNCADO = ["2021-06-01", "2022-01-02", "2022-11-14", "2022-11-15", "2023-02-02", "2023-02-03",
                   "2023-06-09", "2023-06-12", "2025-10-03", "2025-12-16", "2026-03-31"]


@pytest.mark.parametrize("fecha", FECHAS_TRUNCADO)
def test_t1_tools_truncado_generico(vintages, fx, accion, fecha):
    """get_*(base completa, t) == get_*(base truncada a t, t): datos, avisos, metadatos, ultimo() y errores."""
    series = pd.concat([vintages, fx, _serie_lag_fijo()], ignore_index=True)
    for id_, divisa in ((s.SERIE_VINTAGES, "original"), ("sint:PIB_EUR", "USD"), ("ecb:EXR.D.GBP.EUR", "USD")):
        _iguales(_respuesta(get_serie, id_, fecha, base=series, divisa=divisa),
                 _respuesta(get_serie, id_, fecha, base=_truncar(series, fecha), divisa=divisa))
    for ajuste in tools.AJUSTES:
        for divisa in tools.DIVISAS_SALIDA:
            _iguales(_respuesta(get_precios, [s.TICKER_ACCION, "NO.EXISTE"], fecha, ajuste=ajuste,
                                divisa=divisa, base=accion, base_fx=fx),
                     _respuesta(get_precios, [s.TICKER_ACCION, "NO.EXISTE"], fecha, ajuste=ajuste,
                                divisa=divisa, base=_truncar(accion, fecha), base_fx=_truncar(fx, fecha)))


def test_t1_get_precios_sin_fx_avisa(multi):
    r = get_precios("9999.T", "2024-03-15", base=multi, base_fx=s.fx_ecb_sintetico().iloc[:0])
    assert r.datos["close"].isna().all() and any("sin FX" in a for a in r.avisos)


def test_t1_get_precios_aviso_de_dividendo_sin_warnings_globales(accion):
    """Los avisos de ajustar_a_fecha llegan a la respuesta sin catch_warnings (seguro entre hilos)."""
    import warnings

    p = accion.copy()
    i = p.index[p["sesion"] == pd.Timestamp("2022-03-01")][0]
    p.loc[i, "dividendo"] = p.loc[i - 1, "close"] * 2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = get_precios(s.TICKER_ACCION, "2022-06-30", base=p, divisa="original")
    assert any("2022-03-01" in a and "no se pueden ajustar" in a for a in r.avisos)
    assert r.datos.attrs == {}
