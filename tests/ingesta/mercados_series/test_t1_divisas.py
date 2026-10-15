"""Tests de divisas del Tema 1 (``ingesta.mercados_series.divisas``): FX del BCE a fecha, solo pandas."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ingesta.mercados_series import sinteticos as s
from ingesta.mercados_series.contrato import primera_decision_vec, vacio_series, validar_series
from ingesta.mercados_series.divisas import (
    SUBUNIDADES,
    a_usd,
    fx_usd_por_unidad,
    normalizar_subunidad,
)

# Fixings a mano alrededor de la Semana Santa de 2024 (TARGET cerrado el 29-mar y el 1-abr).
DIAS = pd.to_datetime(["2024-03-26", "2024-03-27", "2024-03-28", "2024-04-02", "2024-04-03"])
# Fixings SINTÉTICOS con la estructura del BCE (unidades de X por 1 EUR; Semana Santa sin fixing).
USD_EUR = [1.1000, 1.1200, 1.0900, 1.1100, 1.0800]
GBP_EUR = [0.86000, 0.85000, 0.87000, 0.84000, 0.88000]
JPY_EUR = [160.00, 161.50, 159.00, 162.00, 158.50]


def _filas_fx(serie: str, dias, valores, publicado=None) -> pd.DataFrame:
    """Fixings visibles a las 16:00 Europe/Berlin de ``publicado`` (por defecto, su propio día)."""
    pub = pd.DatetimeIndex(publicado if publicado is not None else dias)
    disp = (pub + pd.Timedelta(hours=16)).tz_localize("Europe/Berlin").tz_convert("UTC")
    divisa = serie.split(".")[2]
    return pd.DataFrame({
        "serie": serie, "fecha_referencia": pd.DatetimeIndex(dias),
        "fecha_publicacion": primera_decision_vec(pd.Series(disp)).to_numpy(),
        "disponible_utc": disp, "vintage": pub.strftime("%Y-%m-%d"), "valor": valores,
        "unidad": f"{divisa} por 1 EUR", "divisa": divisa, "fuente": "prueba",
        "metodo_publicacion": "calendario", "nota": None,
    })


@pytest.fixture(scope="module")
def base():
    return validar_series(pd.concat([
        _filas_fx("ecb:EXR.D.USD.EUR", DIAS, USD_EUR),
        _filas_fx("ecb:EXR.D.GBP.EUR", DIAS, GBP_EUR),
        _filas_fx("ecb:EXR.D.JPY.EUR", DIAS, JPY_EUR),
    ], ignore_index=True))


def _usd_por(divisa_eur: list[float], i: int) -> float:
    return USD_EUR[i] / divisa_eur[i]


def _tabla(fechas, divisa, valores) -> pd.DataFrame:
    return pd.DataFrame({"sesion": pd.to_datetime(fechas), "close": valores, "divisa": divisa})


# --- Subunidades --------------------------------------------------------------------------------
def test_t1_normalizar_subunidad():
    v = pd.Series([2500.0, 100.0])
    out, iso = normalizar_subunidad(v, "GBp")
    assert iso == "GBP" and out.tolist() == [25.0, 1.0]
    assert v.tolist() == [2500.0, 100.0]    # no modifica la entrada
    assert normalizar_subunidad(1234.0, "ZAc") == (12.34, "ZAR")
    assert normalizar_subunidad(np.array([500.0]), "ILA")[1] == "ILS"
    assert normalizar_subunidad(250.0, "GBX") == (2.5, "GBP")
    assert normalizar_subunidad(250.0, "GBP") == (250.0, "GBP")   # distingue mayúsculas
    assert normalizar_subunidad(250.0, "USD") == (250.0, "USD")
    assert set(SUBUNIDADES) >= {"GBp", "ZAc", "ILA"}


# --- fx_usd_por_unidad --------------------------------------------------------------------------
def test_t1_triangulacion_gbp_exacta(base):
    fx = fx_usd_por_unidad(base, "GBP", "2024-04-03")
    assert fx.name == "USD/GBP" and fx.index.name == "fecha_referencia"
    assert list(fx.index) == list(DIAS)
    for i, d in enumerate(DIAS):
        assert fx.loc[d] == USD_EUR[i] / GBP_EUR[i]   # exacto: (USD/EUR) / (GBP/EUR)


def test_t1_triangulacion_con_sinteticos():
    fxs = s.fx_ecb_sintetico(desde="2024-01-02", hasta="2024-03-28")
    fx = fx_usd_por_unidad(fxs, "JPY", "2024-03-28")
    usd = fxs[fxs["serie"] == "ecb:EXR.D.USD.EUR"].set_index("fecha_referencia")["valor"]
    jpy = fxs[fxs["serie"] == "ecb:EXR.D.JPY.EUR"].set_index("fecha_referencia")["valor"]
    pd.testing.assert_series_equal(fx, (usd / jpy).rename("USD/JPY"), check_freq=False)


def test_t1_fx_eur_y_usd(base):
    eur = fx_usd_por_unidad(base, "EUR", "2024-04-03")
    assert eur.name == "USD/EUR" and eur.tolist() == USD_EUR
    usd = fx_usd_por_unidad(base, "USD", "2024-04-03")
    assert usd.name == "USD/USD" and (usd == 1.0).all() and list(usd.index) == list(DIAS)
    assert fx_usd_por_unidad(vacio_series(), "USD", "2024-04-03").empty


def test_t1_fx_errores_de_uso(base):
    with pytest.raises(ValueError, match="subunidad"):
        fx_usd_por_unidad(base, "GBp", "2024-04-03")
    with pytest.raises(KeyError):
        fx_usd_por_unidad(base, "CHF", "2024-04-03")
    with pytest.raises(KeyError):
        fx_usd_por_unidad(vacio_series(), "GBP", "2024-04-03")


def test_t1_fx_solo_fixings_visibles(base):
    # El fixing del 27-mar es visible a las 16:00 de Fráncfort = 11:00 en Nueva York: decisión del 27.
    fx = fx_usd_por_unidad(base, "GBP", "2024-03-27")
    assert list(fx.index) == list(DIAS[:2])
    assert fx_usd_por_unidad(base, "GBP", "2024-03-25").empty


# --- a_usd --------------------------------------------------------------------------------------
def test_t1_a_usd_gbp_en_peniques(base):
    df = _tabla(["2024-03-27"], "GBp", [2500.0])
    out = a_usd(df, ["close"], "divisa", base, "2024-04-03", "sesion")
    esperado = 2500.0 / 100.0 * _usd_por(GBP_EUR, 1)
    assert out["close"].iloc[0] == pytest.approx(esperado, rel=1e-15)
    assert out["fx_usd"].iloc[0] == pytest.approx(_usd_por(GBP_EUR, 1) / 100.0, rel=1e-15)
    assert out["divisa"].iloc[0] == "USD" and out["divisa_original"].iloc[0] == "GBp"
    assert out["fx_fecha"].iloc[0] == pd.Timestamp("2024-03-27")
    assert out.attrs["avisos"] == []
    assert df["close"].iloc[0] == 2500.0 and df["divisa"].iloc[0] == "GBp"   # no modifica df


def test_t1_a_usd_festivo_usa_fixing_anterior(base):
    df = _tabla(["2024-03-28", "2024-03-29", "2024-03-30", "2024-04-01", "2024-04-02"], "GBP", 10.0)
    out = a_usd(df, ["close"], "divisa", base, "2024-04-03", "sesion")
    usadas = pd.to_datetime(["2024-03-28"] * 4 + ["2024-04-02"])
    assert list(out["fx_fecha"]) == list(usadas)
    assert out["close"].tolist() == [10.0 * _usd_por(GBP_EUR, 2)] * 4 + [10.0 * _usd_por(GBP_EUR, 3)]
    assert out.attrs["avisos"] == []


def test_t1_a_usd_no_usa_fixings_publicados_despues(base):
    df = _tabla(["2024-03-27", "2024-03-28"], "GBP", 10.0)
    # En la decisión del 27 el fixing del 28 no existe aún: el 28 usa el del 27 y se avisa.
    out = a_usd(df, ["close"], "divisa", base, "2024-03-27", "sesion")
    assert out["close"].tolist() == [10.0 * _usd_por(GBP_EUR, 1)] * 2
    assert list(out["fx_fecha"]) == [pd.Timestamp("2024-03-27")] * 2
    assert len(out.attrs["avisos"]) == 1 and "posteriores al último fixing" in out.attrs["avisos"][0]
    # Una revisión del fixing del 26 publicada el 28 no se usa en la decisión del 27...
    revision = _filas_fx("ecb:EXR.D.GBP.EUR", ["2024-03-26"], [0.9], publicado=["2024-03-28"])
    con_revision = validar_series(pd.concat([base, revision], ignore_index=True))
    dia26 = _tabla(["2024-03-26"], "GBP", 10.0)
    antes = a_usd(dia26, ["close"], "divisa", con_revision, "2024-03-27", "sesion")
    assert antes["close"].iloc[0] == 10.0 * _usd_por(GBP_EUR, 0)
    # ...y sí en la del 28.
    despues = a_usd(dia26, ["close"], "divisa", con_revision, "2024-03-28", "sesion")
    assert despues["close"].iloc[0] == 10.0 * (USD_EUR[0] / 0.9)


def test_t1_a_usd_truncado(base):
    """Añadir fixings publicados después de ``fecha`` no cambia la conversión."""
    df = _tabla(["2024-03-26", "2024-03-27", "2024-03-28"], "JPY", [1000.0, 1100.0, 1200.0])
    antes = a_usd(df, ["close"], "divisa", base, "2024-03-27", "sesion")
    futuro = pd.concat([
        _filas_fx("ecb:EXR.D.JPY.EUR", ["2024-03-26", "2024-04-04"], [1.0, 2.0],
                  publicado=["2024-04-04", "2024-04-04"]),
        _filas_fx("ecb:EXR.D.USD.EUR", ["2024-04-04"], [1.5]),
    ], ignore_index=True)
    despues = a_usd(df, ["close"], "divisa", validar_series(pd.concat([base, futuro], ignore_index=True)),
                    "2024-03-27", "sesion")
    pd.testing.assert_frame_equal(antes, despues)
    assert antes.attrs == despues.attrs


def test_t1_a_usd_sin_fixing_previo(base):
    df = _tabla(["2024-03-20", "2024-03-26"], "GBP", 10.0)
    out = a_usd(df, ["close"], "divisa", base, "2024-04-03", "sesion")
    assert np.isnan(out["close"].iloc[0]) and pd.isna(out["fx_fecha"].iloc[0])
    assert out["close"].iloc[1] == 10.0 * _usd_por(GBP_EUR, 0)
    assert len(out.attrs["avisos"]) == 1 and out.attrs["avisos"][0].startswith("1 filas en GBP")


def test_t1_a_usd_identidad_usd():
    df = _tabla(["2024-03-26", "2024-03-30"], "USD", [10.0, 11.0])
    out = a_usd(df, ["close"], "divisa", vacio_series(), "2024-04-03", "sesion")
    assert out["close"].tolist() == [10.0, 11.0]
    assert out["fx_usd"].tolist() == [1.0, 1.0]
    assert out["divisa"].tolist() == ["USD", "USD"] and out.attrs["avisos"] == []


def test_t1_a_usd_divisa_comun_y_eur(base):
    df = pd.DataFrame({"fecha_referencia": pd.to_datetime(["2024-03-28", "2024-04-01"]), "valor": [2.0, 3.0]})
    out = a_usd(df, ["valor"], "EUR", base, "2024-04-03", "fecha_referencia")
    assert out["valor"].tolist() == [2.0 * USD_EUR[2], 3.0 * USD_EUR[2]]
    assert out["divisa"].tolist() == ["USD", "USD"] and out["divisa_original"].tolist() == ["EUR", "EUR"]


def test_t1_a_usd_divisa_sin_serie_del_bce(base):
    df = _tabla(["2024-03-27"], "CHF", [10.0])
    out = a_usd(df, ["close"], "divisa", base, "2024-04-03", "sesion")
    assert np.isnan(out["close"].iloc[0])
    assert any("CHF" in a and "sin FX" in a for a in out.attrs["avisos"])


def test_t1_a_usd_multimercado():
    """Cada fila con su divisa y el fixing de su sesión; volumen intacto; varias columnas."""
    precios = s.precios_multimercado()
    fxs = s.fx_ecb_sintetico(desde="2024-01-02", hasta="2024-03-29")
    copia = precios.copy()
    fecha = "2024-03-18"
    out = a_usd(precios, ["open", "close", "dividendo"], "divisa", fxs, fecha, "sesion")
    pd.testing.assert_frame_equal(precios, copia)
    usd = fxs[fxs["serie"] == "ecb:EXR.D.USD.EUR"].set_index("fecha_referencia")["valor"]
    for _, fila in out.sample(25, random_state=0).iterrows():
        misma = (precios["ticker"] == fila["ticker"]) & (precios["sesion"] == fila["sesion"])
        original = precios.loc[misma].iloc[0]
        cotizada = original["divisa"]
        valor, iso = normalizar_subunidad(original["close"], cotizada)
        dia = usd.index[usd.index <= fila["sesion"]].max()       # fines de semana: viernes
        if iso == "USD":
            factor = 1.0
        elif iso == "EUR":
            factor = usd.loc[dia]
        else:
            fila_x = (fxs["serie"] == f"ecb:EXR.D.{iso}.EUR") & (fxs["fecha_referencia"] == dia)
            x = fxs.loc[fila_x, "valor"].iloc[0]
            factor = usd.loc[dia] / x
        assert fila["close"] == pytest.approx(valor * factor, rel=1e-12)
        assert fila["divisa_original"] == cotizada and fila["divisa"] == "USD"
    assert out["volume"].equals(precios["volume"])
    assert out.attrs["avisos"] == []


def test_t1_a_usd_errores_de_uso(base):
    df = _tabla(["2024-03-27"], None, [10.0])
    with pytest.raises(ValueError, match="sin divisa"):
        a_usd(df, ["close"], "divisa", base, "2024-04-03", "sesion")
    with pytest.raises(KeyError):
        a_usd(_tabla(["2024-03-27"], "GBP", [1.0]), ["no_existe"], "divisa", base, "2024-04-03", "sesion")
    with pytest.raises(ValueError):
        a_usd(_tabla(["2024-03-27"], "GBP", [1.0]), ["close"], "divisa", base,
              pd.Timestamp("2024-04-03", tz="UTC"), "sesion")


def test_t1_a_usd_conserva_avisos_previos(base):
    df = _tabla(["2024-03-20"], "GBP", [1.0])
    df.attrs["avisos"] = ["previo"]
    out = a_usd(df, ["close"], "divisa", base, "2024-04-03", "sesion")
    assert out.attrs["avisos"][0] == "previo" and len(out.attrs["avisos"]) == 2
    assert df.attrs["avisos"] == ["previo"]


def test_t1_a_usd_fecha_nula_sin_fx(base):
    """Regresión: una fila con fecha nula se convertía en silencio con el ÚLTIMO fixing."""
    df = _tabla(["2024-03-27", None], "EUR", [10.0, 10.0])
    out = a_usd(df, ["close"], "divisa", base, "2024-04-03", "sesion")
    assert out["close"].iloc[0] == pytest.approx(10.0 * USD_EUR[1])
    assert np.isnan(out["close"].iloc[1]) and pd.isna(out["fx_fecha"].iloc[1])
    assert any("nula" in a for a in out.attrs["avisos"])


def test_t1_a_usd_fuera_de_la_cobertura_del_bce():
    """Límite conocido (documentado): antes de 1999-01-04 y sin serie del BCE (TWD) → NaN y aviso."""
    fx = s.fx_ecb_sintetico(desde="1999-01-04", hasta="1999-03-31")
    antes = a_usd(_tabla(["1998-06-01", "1999-02-01"], "GBP", [10.0, 10.0]), ["close"], "divisa", fx,
                  "1999-03-31", "sesion")
    assert np.isnan(antes["close"].iloc[0]) and np.isfinite(antes["close"].iloc[1])
    assert any("sin fixing" in a for a in antes.attrs["avisos"])
    twd = a_usd(_tabla(["1999-02-01"], "TWD", [10.0]), ["close"], "divisa", fx, "1999-03-31", "sesion")
    assert twd["close"].isna().all() and any("TWD" in a for a in twd.attrs["avisos"])
    with pytest.raises(KeyError):
        fx_usd_por_unidad(fx, "TWD", "1999-03-31")
