"""Tests del TimeGate de precios (``visibles_precios`` y ``ajustar_a_fecha``): truncado y ajuste."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ingesta.mercados_series import sinteticos as s
from ingesta.mercados_series.contrato import COLUMNAS_PRECIOS, vacio_precios, validar_precios
from ingesta.mercados_series.timegate import (
    COLUMNAS_AJUSTE,
    ajustar_a_fecha,
    des_ajustar_yahoo,
    visibles_precios,
)

FINAL = pd.Timestamp("2030-01-01")      # posterior a todo: equivale a la vista de hoy
OHLC = ("open", "high", "low", "close")


@pytest.fixture(scope="module")
def accion():
    return s.accion_con_eventos()


@pytest.fixture(scope="module")
def multi():
    return s.precios_multimercado()


def _ajuste_ingenuo(p: pd.DataFrame, fecha) -> pd.DataFrame:
    """Implementación independiente, fila a fila y sin vectorizar: (close_aj_splits, close_aj_total).

    Para cada fila r: factor de splits = 1 / prod(split_j, j visible y posterior a r); factor de
    dividendos = prod(1 - D_j / C_{j-1}, j visible, posterior a r, con dividendo), con D y C en
    acciones de hoy-en-fecha.
    """
    v = p[p["fecha_publicacion"] <= pd.Timestamp(fecha)].sort_values("sesion").reset_index(drop=True)
    n = len(v)
    splits = list(v["split"])
    cierres = list(v["close"])
    divs = list(v["dividendo"])

    def f_split(r):
        prod = 1.0
        for j in range(r + 1, n):
            prod *= splits[j]
        return 1.0 / prod

    c_aj = [cierres[r] * f_split(r) for r in range(n)]
    d_aj = [divs[r] * f_split(r) for r in range(n)]
    total = []
    for r in range(n):
        fd = 1.0
        for j in range(r + 1, n):
            if d_aj[j] > 0:
                fd *= 1.0 - d_aj[j] / c_aj[j - 1]
        total.append(c_aj[r] * fd)
    return pd.DataFrame({"sesion": v["sesion"], "close_aj_splits": c_aj, "close_aj_total": total})


def _sin_split(p: pd.DataFrame) -> pd.DataFrame:
    """La misma acción si nunca hubiera hecho el split: todo en acciones de hoy, ``split = 1``."""
    q = p.copy()
    antes = q["sesion"] < s.SESION_SPLIT
    for c in (*OHLC, "dividendo"):
        q.loc[antes, c] = q.loc[antes, c] / s.RATIO_SPLIT
    q.loc[antes, "volume"] = q.loc[antes, "volume"] * s.RATIO_SPLIT
    q["split"] = 1.0
    return q


def _futuro(p: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """Sesiones nuevas publicadas tras ``t`` con un split 2:1 y un dividendo (eventos FUTUROS)."""
    ultima = p["sesion"].max()
    sesiones = pd.bdate_range(max(ultima, t) + pd.Timedelta(days=1), periods=6)
    base = p.iloc[-1]
    filas = []
    for k, ses in enumerate(sesiones):
        filas.append({**base.to_dict(), "sesion": ses, "close": 30.0 + k, "open": 30.0 + k,
                      "high": 31.0 + k, "low": 29.0 + k,
                      "split": 2.0 if k == 2 else 1.0, "dividendo": 0.5 if k == 4 else 0.0})
    fut = pd.DataFrame(filas)
    disponible = fut["sesion"].dt.tz_localize("America/New_York") + pd.Timedelta(hours=16, minutes=30)
    fut["disponible_utc"] = disponible.dt.tz_convert("UTC")
    fut["fecha_publicacion"] = fut["sesion"]
    fut["descargado"] = fut["disponible_utc"].max() + pd.Timedelta(hours=1)
    return fut


# --- (a) Truncado --------------------------------------------------------------------------------
@pytest.mark.parametrize("t", ["2022-06-30", "2022-11-14", "2022-11-15", "2023-06-09", "2023-06-12",
                               "2023-12-31"])
def test_t1_ajuste_truncado_con_split_y_dividendo_futuros(accion, t):
    t = pd.Timestamp(t)
    hasta_t = accion[accion["fecha_publicacion"] <= t]
    esperado = ajustar_a_fecha(hasta_t, t)
    # con todo el histórico (incluye el split 2:1 de 2023-06-12 y dividendos aún no conocidos en t)
    pd.testing.assert_frame_equal(ajustar_a_fecha(accion, t), esperado)
    # y con sesiones aún más futuras con otro split y otro dividendo
    ampliada = validar_precios(pd.concat([accion, _futuro(accion, FINAL)], ignore_index=True))
    pd.testing.assert_frame_equal(ajustar_a_fecha(ampliada, t), esperado)
    assert (esperado["fecha_publicacion"] <= t).all()


def test_t1_ajuste_antes_del_split_no_lo_conoce(accion):
    t = s.SESION_SPLIT - pd.Timedelta(days=1)
    r = ajustar_a_fecha(accion, t)
    assert (r["factor_splits"] == 1.0).all()
    pd.testing.assert_series_equal(r["close_aj_splits"], r["close"], check_names=False)
    # el día del split ya se conoce: todo lo anterior se parte por la mitad
    r = ajustar_a_fecha(accion, s.SESION_SPLIT)
    antes = r["sesion"] < s.SESION_SPLIT
    assert np.allclose(r.loc[antes, "factor_splits"], 1.0 / s.RATIO_SPLIT)
    assert (r.loc[~antes, "factor_splits"] == 1.0).all()


def test_t1_ajuste_dividendo_solo_desde_su_fecha_ex(accion):
    ex = pd.Timestamp("2022-11-15")
    r = ajustar_a_fecha(accion, ex - pd.Timedelta(days=1))
    assert (r["factor_dividendos"] == 1.0).all()
    r = ajustar_a_fecha(accion, ex)
    antes = r["sesion"] < ex
    assert (r.loc[antes, "factor_dividendos"] < 1.0).all()
    assert (r.loc[~antes, "factor_dividendos"] == 1.0).all()


# --- (b) Coincide con la vista de hoy y con una implementación independiente --------------------
def test_t1_ajuste_final_coincide_con_vista_yahoo(accion):
    r = ajustar_a_fecha(accion, FINAL)
    y = s.vista_yahoo_hoy(accion)
    assert len(r) == len(y)
    np.testing.assert_array_equal(r["sesion"].to_numpy(), y.index.to_numpy())
    np.testing.assert_allclose(r["close_aj_total"], y["Adj Close"], rtol=1e-12)
    np.testing.assert_allclose(r["close_aj_splits"], y["Close"], rtol=1e-12)
    for c, cy in (("open", "Open"), ("high", "High"), ("low", "Low")):
        np.testing.assert_allclose(r[f"{c}_aj_splits"], y[cy], rtol=1e-12)
    np.testing.assert_allclose(r["volume_aj_splits"], y["Volume"], rtol=1e-12)
    np.testing.assert_allclose(r["dividendo_aj_splits"], y["Dividends"], rtol=1e-12)
    # dividendos en acciones de hoy: 0,40 (as-traded 0,80 antes del split)
    con_div = r[r["dividendo_aj_splits"] > 0]
    assert list(con_div["sesion"]) == sorted(s.DIVIDENDOS_ECONOMICOS)
    np.testing.assert_allclose(con_div["dividendo_aj_splits"], 0.40, atol=0.005)


@pytest.mark.parametrize("t", ["2022-11-15", "2023-03-01", "2023-06-12", "2023-08-20", FINAL])
def test_t1_ajuste_coincide_con_implementacion_ingenua(accion, t):
    r = ajustar_a_fecha(accion, t)
    ing = _ajuste_ingenuo(accion, t)
    np.testing.assert_array_equal(r["sesion"].to_numpy(), ing["sesion"].to_numpy())
    np.testing.assert_allclose(r["close_aj_splits"], ing["close_aj_splits"], rtol=1e-12)
    np.testing.assert_allclose(r["close_aj_total"], ing["close_aj_total"], rtol=1e-12)


# --- (c) Retorno total invariante a un split ----------------------------------------------------
@pytest.mark.parametrize("t", [s.SESION_SPLIT, pd.Timestamp("2023-09-01"), FINAL])
def test_t1_retorno_total_invariante_al_split(accion, t):
    con = ajustar_a_fecha(accion, t)
    sin = ajustar_a_fecha(_sin_split(accion), t)
    ret_con = con["close_aj_total"].pct_change()
    ret_sin = sin["close_aj_total"].pct_change()
    np.testing.assert_allclose(ret_con.to_numpy()[1:], ret_sin.to_numpy()[1:], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(con["close_aj_total"], sin["close_aj_total"], rtol=1e-12)
    # el día del split, el crudo cae ~50 % y el ajustado no
    dia = con.index[con["sesion"] == s.SESION_SPLIT][0]
    assert con["close"].pct_change().iloc[dia] < -0.4
    assert abs(ret_con.iloc[dia]) < 0.1


def test_t1_retorno_total_identidad_multiplicativa(accion):
    r = ajustar_a_fecha(accion, FINAL)
    c, d = r["close_aj_splits"].to_numpy(), r["dividendo_aj_splits"].to_numpy()
    ret_total = r["close_aj_total"].to_numpy()[1:] / r["close_aj_total"].to_numpy()[:-1]
    np.testing.assert_allclose(ret_total, c[1:] / (c[:-1] - d[1:]), rtol=1e-12)


# --- (d) Coherencia OHLC y volumen ----------------------------------------------------------------
@pytest.mark.parametrize("t", [s.SESION_SPLIT, FINAL])
def test_t1_ajuste_ohlc_coherente_y_volumen_inverso(accion, t):
    r = ajustar_a_fecha(accion, t)
    for suf in ("aj_splits", "aj_total"):
        o, h, lo, c = (r[f"{x}_{suf}"] for x in OHLC)
        assert (lo <= o + 1e-12).all() and (lo <= c + 1e-12).all()
        assert (o <= h + 1e-12).all() and (c <= h + 1e-12).all()
    np.testing.assert_allclose(r["volume_aj_splits"], r["volume"] / r["factor_splits"], rtol=1e-12)
    # el importe negociado (precio × volumen) no cambia con el ajuste de splits
    np.testing.assert_allclose(r["volume_aj_splits"] * r["close_aj_splits"], r["volume"] * r["close"],
                               rtol=1e-12)
    assert ((r["factor_dividendos"] > 0) & (r["factor_dividendos"] <= 1)).all()
    assert r["factor_dividendos"].is_monotonic_increasing


def test_t1_ajuste_columnas_y_no_modifica_entrada(accion):
    copia = accion.copy()
    r = ajustar_a_fecha(accion, FINAL)
    pd.testing.assert_frame_equal(accion, copia)
    assert list(r.columns) == list(COLUMNAS_PRECIOS) + list(COLUMNAS_AJUSTE)
    assert not any("adj" in c.lower() for c in r.columns)
    assert r["sesion"].is_monotonic_increasing
    assert all(r[c].dtype == "float64" for c in COLUMNAS_AJUSTE)
    pd.testing.assert_frame_equal(r[list(COLUMNAS_PRECIOS)], validar_precios(accion))


def test_t1_ajuste_vacio_y_varios_tickers(accion, multi):
    r = ajustar_a_fecha(accion, "2000-01-01")
    assert r.empty and list(r.columns) == list(COLUMNAS_PRECIOS) + list(COLUMNAS_AJUSTE)
    assert ajustar_a_fecha(vacio_precios(), FINAL).empty
    with pytest.raises(ValueError, match="único ticker"):
        ajustar_a_fecha(multi, FINAL)


def test_t1_ajuste_dividendo_imposible_avisa(accion):
    p = accion.copy()
    i = p.index[p["sesion"] == pd.Timestamp("2022-03-01")][0]
    p.loc[i, "dividendo"] = p.loc[i - 1, "close"] * 2
    with pytest.warns(RuntimeWarning, match="2022-03-01"):
        r = ajustar_a_fecha(p, "2022-06-30")
    assert np.isfinite(r["close_aj_total"]).all() and (r["factor_dividendos"] == 1.0).all()


# --- (e) visibles_precios -------------------------------------------------------------------------
@pytest.mark.parametrize("t", ["2024-03-03", "2024-03-08", "2024-03-11", "2024-03-15", "2024-03-16"])
def test_t1_visibles_precios_descarta_el_futuro(multi, t):
    t = pd.Timestamp(t)
    copia = multi.copy()
    barajada = multi.sample(frac=1.0, random_state=1)
    r = visibles_precios(barajada, t)
    pd.testing.assert_frame_equal(multi, copia)
    assert (r["fecha_publicacion"] <= t).all()
    esperado = multi[multi["fecha_publicacion"] <= t].sort_values(["ticker", "sesion"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(r, esperado)
    assert list(r.columns) == list(COLUMNAS_PRECIOS)
    # CME y CRYPTO cierran tras la decisión de NY: su sesión de t no es visible en t
    hoy = r[r["sesion"] == t]
    assert not set(hoy["mercado"]) & {"CME", "CRYPTO"}
    if t.weekday() < 5:
        assert {"XNYS", "XLON", "XETR", "XTKS"} <= set(hoy["mercado"])


def test_t1_visibles_precios_vacio(multi):
    r = visibles_precios(multi, "2024-03-01")
    assert r.empty and list(r.columns) == list(COLUMNAS_PRECIOS)
    pd.testing.assert_series_equal(r.dtypes, vacio_precios().dtypes)


# --- (f) Sin eventos: ajustados == crudos -------------------------------------------------------
@pytest.mark.parametrize("ticker", ["ZZNY", "ZZLN.L", "ZZ-USD"])
def test_t1_ajuste_sin_eventos_es_identidad(multi, ticker):
    p = multi[multi["ticker"] == ticker]
    r = ajustar_a_fecha(p, FINAL)
    assert len(r) == len(p)
    for c in OHLC:
        pd.testing.assert_series_equal(r[f"{c}_aj_splits"], r[c], check_names=False)
        pd.testing.assert_series_equal(r[f"{c}_aj_total"], r[c], check_names=False)
    pd.testing.assert_series_equal(r["volume_aj_splits"], r["volume"], check_names=False)
    assert (r["factor_splits"] == 1.0).all() and (r["factor_dividendos"] == 1.0).all()


# --- Propiedad con hypothesis: truncado con eventos aleatorios ---------------------------------
def test_t1_ajuste_truncado_propiedad_aleatoria():
    hyp = pytest.importorskip("hypothesis")
    st = hyp.strategies
    origen = pd.Timestamp("2024-01-01")
    fila = st.tuples(
        st.integers(0, 3),                               # retraso de publicación (días)
        st.floats(5.0, 200.0),                           # cierre
        st.sampled_from([1.0, 1.0, 1.0, 2.0, 0.5, 3.0]), # split
        st.sampled_from([0.0, 0.0, 0.0, 0.1, 0.7]),      # dividendo
    )

    @hyp.settings(max_examples=150, deadline=None, database=None)
    @hyp.given(filas=st.lists(fila, min_size=1, max_size=25), corte=st.integers(-2, 40))
    def propiedad(filas, corte):
        sesiones = pd.bdate_range(origen, periods=len(filas))
        cierre = np.array([f[1] for f in filas])
        p = pd.DataFrame({
            "ticker": "ZZH", "sesion": sesiones, "mercado": "XNYS",
            "open": cierre, "high": cierre * 1.01, "low": cierre * 0.99, "close": cierre,
            "volume": 1000.0, "dividendo": [f[3] for f in filas], "split": [f[2] for f in filas],
            "divisa": "USD", "fecha_publicacion": sesiones + pd.to_timedelta([f[0] for f in filas], unit="D"),
            "disponible_utc": pd.NaT, "fuente": "test", "descargado": pd.Timestamp("2030-01-01", tz="UTC"),
        })
        t = origen + pd.Timedelta(days=corte)
        r = ajustar_a_fecha(p, t)
        assert (r["fecha_publicacion"] <= t).all()
        pd.testing.assert_frame_equal(ajustar_a_fecha(p[p["fecha_publicacion"] <= t], t), r)
        ing = _ajuste_ingenuo(p, t)
        np.testing.assert_allclose(r["close_aj_total"], ing["close_aj_total"], rtol=1e-9)

    propiedad()


# --- (g) Casos difíciles sin hypothesis: contrasplit, split 3:2 y split + dividendo el mismo día -
@pytest.fixture(scope="module")
def dificil():
    return s.accion_eventos_dificiles()


@pytest.mark.parametrize("t", ["2024-02-14", "2024-02-15", "2024-03-14", "2024-03-15", "2024-04-12",
                               "2024-04-15", "2024-05-15", FINAL])
def test_t1_ajuste_eventos_dificiles_coincide_con_ingenuo_y_trunca(dificil, t):
    r = ajustar_a_fecha(dificil, t)
    ing = _ajuste_ingenuo(dificil, t)
    np.testing.assert_allclose(r["close_aj_splits"], ing["close_aj_splits"], rtol=1e-12)
    np.testing.assert_allclose(r["close_aj_total"], ing["close_aj_total"], rtol=1e-12)
    pd.testing.assert_frame_equal(ajustar_a_fecha(dificil[dificil["fecha_publicacion"] <= pd.Timestamp(t)], t), r)


def test_t1_ajuste_split_y_dividendo_el_mismo_dia(dificil):
    """El dividendo del día del split va en acciones nuevas; el retorno total no salta por el split."""
    r = ajustar_a_fecha(dificil, FINAL).set_index("sesion")
    for sesion, ratio in s.SPLITS_DIFICILES.items():
        i = r.index.get_loc(sesion)
        assert abs(r["close"].iloc[i] / r["close"].iloc[i - 1] - 1) > 0.25          # el crudo salta
        assert abs(r["close_aj_total"].iloc[i] / r["close_aj_total"].iloc[i - 1] - 1) < 0.1
    dia = pd.Timestamp("2024-04-15")
    assert r.loc[dia, "split"] == 1.5 and r.loc[dia, "dividendo"] == pytest.approx(0.5)
    # Identidad multiplicativa también ese día: C_aj_total(i)/C_aj_total(i-1) = C(i)/(C(i-1) - D(i)).
    c, d = r["close_aj_splits"].to_numpy(), r["dividendo_aj_splits"].to_numpy()
    tot = r["close_aj_total"].to_numpy()
    np.testing.assert_allclose(tot[1:] / tot[:-1], c[1:] / (c[:-1] - d[1:]), rtol=1e-12)


@pytest.mark.parametrize("generador", [s.accion_con_eventos, s.accion_eventos_dificiles])
def test_t1_des_ajustar_yahoo_ida_y_vuelta(generador):
    """as-traded → vista de Yahoo de hoy → des_ajustar_yahoo → as-traded (lo que hará el extractor)."""
    p = generador()
    vuelta = des_ajustar_yahoo(s.vista_yahoo_hoy(p))
    for c in ("open", "high", "low", "close", "volume", "dividendo", "split"):
        np.testing.assert_allclose(vuelta[c].to_numpy(), p[c].to_numpy(), rtol=1e-12, err_msg=c)
    np.testing.assert_array_equal(vuelta["sesion"].to_numpy(), p["sesion"].to_numpy())
    with pytest.raises(ValueError, match="faltan"):
        des_ajustar_yahoo(s.vista_yahoo_hoy(p).drop(columns="Stock Splits"))


def test_t1_ajuste_avisos_en_attrs_sin_warnings(accion):
    """Con avisar=False no se toca el estado global de warnings: los avisos vuelven en attrs."""
    import warnings

    p = accion.copy()
    i = p.index[p["sesion"] == pd.Timestamp("2022-03-01")][0]
    p.loc[i, "dividendo"] = p.loc[i - 1, "close"] * 2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = ajustar_a_fecha(p, "2022-06-30", avisar=False)
    assert len(r.attrs["avisos"]) == 1 and "2022-03-01" in r.attrs["avisos"][0]
    assert ajustar_a_fecha(accion, "2022-06-30").attrs["avisos"] == []


def test_t1_ajuste_rechaza_tablas_sin_validar(accion):
    with pytest.raises(ValueError, match="datetime64"):
        ajustar_a_fecha(accion.assign(fecha_publicacion=accion["fecha_publicacion"].dt.strftime("%Y-%m-%d")),
                        FINAL)
    with pytest.raises(ValueError, match="ticker nulo"):
        ajustar_a_fecha(accion.assign(ticker=[None] + list(accion["ticker"].iloc[1:])), FINAL)
    with pytest.raises(ValueError, match="datetime64"):
        visibles_precios(accion.assign(sesion=accion["sesion"].astype(str)), FINAL)
