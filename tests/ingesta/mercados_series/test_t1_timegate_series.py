"""Tests del TimeGate de series (``timegate.as_of_series``): no-futuro, truncado y vintages."""
from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from ingesta.mercados_series import sinteticos as s
from ingesta.mercados_series.contrato import COLUMNAS_SERIES, vacio_series, validar_series
from ingesta.mercados_series.timegate import as_of_series, fecha_decision

SERIE = s.SERIE_VINTAGES


@pytest.fixture(scope="module")
def vintages():
    return s.serie_con_vintages()


def _as_of_ingenuo(base: pd.DataFrame, serie: str, fecha) -> dict[pd.Timestamp, tuple]:
    """Referencia independiente: bucle por periodo, máx. (fecha_publicacion, vintage) visible."""
    f = pd.Timestamp(fecha)
    mejor: dict[pd.Timestamp, tuple] = {}
    for fila in base.itertuples(index=False):
        if fila.serie != serie or fila.fecha_publicacion > f:
            continue
        clave = (fila.fecha_publicacion, fila.vintage)
        actual = mejor.get(fila.fecha_referencia)
        if actual is None or clave > actual[0]:
            mejor[fila.fecha_referencia] = (clave, fila.valor, fila.nota)
    return mejor


def _tabla(filas: list[dict], serie: str = "sint:X") -> pd.DataFrame:
    """Tabla del contrato a partir de (fecha_referencia, fecha_publicacion, vintage, valor, nota)."""
    df = pd.DataFrame(filas)
    df = df.assign(serie=serie, disponible_utc=pd.NaT, unidad="u", divisa=None, fuente="test",
                   metodo_publicacion="vintage_exacto")
    if "nota" not in df:
        df["nota"] = None
    return validar_series(df[list(COLUMNAS_SERIES)], comprobar_disponible=False)


def _futuras(base: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """Filas nuevas publicadas después de ``t``: revisiones de periodos ya vistos y periodos nuevos."""
    ultimo = base["fecha_referencia"].max()
    refs = list(base["fecha_referencia"].drop_duplicates().iloc[-3:]) + [ultimo + pd.offsets.MonthBegin(k)
                                                                         for k in (1, 2)]
    filas = []
    for k, ref in enumerate(refs):
        # posterior a t y nunca antes del fin de su periodo (contrato)
        pub = max(t + pd.Timedelta(days=1 + k), ref + pd.offsets.MonthEnd(0) + pd.Timedelta(days=5))
        filas.append({"serie": SERIE, "fecha_referencia": ref, "fecha_publicacion": pub,
                      "disponible_utc": pd.NaT, "vintage": "2099-01-" + f"{k + 1:02d}", "valor": 99.0 + k,
                      "unidad": base["unidad"].iloc[0], "divisa": None, "fuente": "test",
                      "metodo_publicacion": "vintage_exacto", "nota": None})
    return pd.DataFrame(filas)


# --- (a) Nunca sale el futuro ------------------------------------------------------------------
def test_t1_as_of_nunca_devuelve_filas_publicadas_despues(vintages):
    fechas = pd.date_range("2022-12-01", "2026-04-30", freq="3D")
    for f in fechas:
        r = as_of_series(vintages, SERIE, f)
        assert (r["fecha_publicacion"] <= f).all()
        assert r["fecha_referencia"].is_unique and r["fecha_referencia"].is_monotonic_increasing
        ingenuo = _as_of_ingenuo(vintages, SERIE, f)
        assert list(r["fecha_referencia"]) == sorted(ingenuo)
        esperado = [ingenuo[ref][1] for ref in r["fecha_referencia"]]
        np.testing.assert_array_equal(r["valor"].to_numpy(), np.array(esperado, dtype=float))


def test_t1_as_of_conserva_contrato_y_no_modifica_base(vintages):
    copia = vintages.copy()
    r = as_of_series(vintages, SERIE, "2025-06-30")
    pd.testing.assert_frame_equal(vintages, copia)
    assert list(r.columns) == list(COLUMNAS_SERIES)
    pd.testing.assert_series_equal(r.dtypes, vacio_series().dtypes)
    validar_series(r)   # sigue cumpliendo el contrato (clave única, coherencia de disponible_utc)


def test_t1_as_of_formas_de_fecha_equivalentes(vintages):
    base = as_of_series(vintages, SERIE, "2024-07-05")
    for f in (date(2024, 7, 5), pd.Timestamp("2024-07-05"), datetime(2024, 7, 5),
              np.datetime64("2024-07-05")):
        pd.testing.assert_frame_equal(as_of_series(vintages, SERIE, f), base)
    with pytest.raises(ValueError):
        as_of_series(vintages, SERIE, pd.Timestamp("2024-07-05", tz="UTC"))
    with pytest.raises(ValueError):
        fecha_decision(None)
    with pytest.raises(ValueError):
        fecha_decision("no es una fecha")


@pytest.mark.parametrize("fecha", [
    "02/01/2024",                     # 2 de enero en España, 1 de febrero en EE. UU.: ambigua
    "13/01/2024", "2024/01/13", "Jan 2024", "2024-1-5", "2024-01-05 ", "2024-01-05T00:00",
    "now", "today", "", "2024-02-30",
    20240115, 1.5,                    # números: pandas los leería como nanosegundos desde 1970
    pd.NaT,
])
def test_t1_fecha_decision_rechaza_formatos_libres(fecha):
    """Regresión: '02/01/2024' se leía como 1-feb (un mes de fuga) y 20240115 como 1970-01-01."""
    with pytest.raises(ValueError):
        fecha_decision(fecha)


@pytest.mark.parametrize("fecha", [
    pd.Timestamp("2024-03-15 02:00"),   # utcnow() naive: 22:00 NY del 14 tratado como el 15
    datetime(2024, 3, 15, 16, 31),
    np.datetime64("2024-03-15T00:00:01"),
])
def test_t1_fecha_decision_rechaza_horas(fecha):
    """Regresión: una fecha naive con hora se truncaba en silencio y podía adelantar casi un día."""
    with pytest.raises(ValueError, match="sin hora"):
        fecha_decision(fecha)


def test_t1_fecha_decision_formatos_admitidos():
    esperado = pd.Timestamp("2024-03-15")
    for f in ("2024-03-15", date(2024, 3, 15), datetime(2024, 3, 15), pd.Timestamp("2024-03-15"),
              np.datetime64("2024-03-15"), pd.Timestamp("2024-03-15").to_pydatetime()):
        assert fecha_decision(f) == esperado


# --- (b) Truncado ------------------------------------------------------------------------------
@pytest.mark.parametrize("t", ["2023-03-10", "2024-01-31", "2025-10-03", "2025-11-20", "2025-12-15",
                               "2025-12-16", "2026-02-01"])
def test_t1_as_of_truncado(vintages, t):
    t = pd.Timestamp(t)
    esperado = as_of_series(vintages, SERIE, t)
    # quitar lo publicado después de t no cambia nada
    truncada = vintages[vintages["fecha_publicacion"] <= t]
    pd.testing.assert_frame_equal(as_of_series(truncada, SERIE, t), esperado)
    # añadir filas publicadas después de t (revisiones y periodos nuevos) tampoco
    ampliada = pd.concat([vintages, _futuras(vintages, t)], ignore_index=True)
    ampliada = validar_series(ampliada, comprobar_disponible=False)
    pd.testing.assert_frame_equal(as_of_series(ampliada, SERIE, t), esperado)
    # ni otras series en la base
    otra = s.serie_mercado_diaria()
    pd.testing.assert_frame_equal(as_of_series(pd.concat([otra, ampliada], ignore_index=True), SERIE, t),
                                  esperado)


# --- (c) Revisiones -----------------------------------------------------------------------------
def test_t1_as_of_revision_primera_y_revisada(vintages):
    ref = pd.Timestamp("2024-03-01")
    g = vintages[vintages["fecha_referencia"] == ref].sort_values("fecha_publicacion")
    assert len(g) >= 2 and g["valor"].nunique() >= 2
    p1, p2 = g["fecha_publicacion"].iloc[0], g["fecha_publicacion"].iloc[1]

    def valor(f):
        r = as_of_series(vintages, SERIE, f)
        fila = r[r["fecha_referencia"] == ref]
        return None if fila.empty else (fila["valor"].iloc[0], fila["vintage"].iloc[0])

    assert valor(p1 - pd.Timedelta(days=1)) is None
    assert valor(p1) == (g["valor"].iloc[0], g["vintage"].iloc[0])
    assert valor(p2 - pd.Timedelta(days=1)) == (g["valor"].iloc[0], g["vintage"].iloc[0])
    assert valor(p2) == (g["valor"].iloc[1], g["vintage"].iloc[1])
    assert valor("2030-01-01") == (g["valor"].iloc[-1], g["vintage"].iloc[-1])


def test_t1_as_of_publicacion_retrasada(vintages):
    antes = as_of_series(vintages, SERIE, s.FECHA_PREVISTA_RETRASADO)
    assert s.MES_RETRASADO not in set(antes["fecha_referencia"])
    despues = as_of_series(vintages, SERIE, "2025-11-20")
    fila = despues[despues["fecha_referencia"] == s.MES_RETRASADO].iloc[0]
    assert fila["nota"] == s.NOTA_RETRASO and pd.notna(fila["valor"])
    assert despues["fecha_referencia"].max() == s.MES_RETRASADO


# --- (d) Hueco permanente -----------------------------------------------------------------------
def test_t1_as_of_hueco_permanente_visible_con_nota(vintages):
    antes = as_of_series(vintages, SERIE, s.INFORME_HUECO - pd.Timedelta(days=1))
    assert s.MES_HUECO not in set(antes["fecha_referencia"])
    for f in (s.INFORME_HUECO, "2026-06-30"):
        r = as_of_series(vintages, SERIE, f)
        fila = r[r["fecha_referencia"] == s.MES_HUECO]
        assert len(fila) == 1
        assert pd.isna(fila["valor"].iloc[0]) and fila["nota"].iloc[0] == s.NOTA_HUECO
        # los demás periodos no quedan contaminados
        assert r.loc[r["fecha_referencia"] != s.MES_HUECO, "valor"].notna().all()


# --- (e) Nada visible ---------------------------------------------------------------------------
@pytest.mark.parametrize("serie, fecha", [(SERIE, "2000-01-01"), (SERIE, "2023-02-02"),
                                          ("sint:NO_EXISTE", "2026-01-01")])
def test_t1_as_of_vacio_con_columnas_del_contrato(vintages, serie, fecha):
    r = as_of_series(vintages, serie, fecha)
    assert r.empty
    pd.testing.assert_frame_equal(r, vacio_series())


def test_t1_as_of_base_vacia():
    pd.testing.assert_frame_equal(as_of_series(vacio_series(), SERIE, "2025-01-01"), vacio_series())


def test_t1_as_of_base_sin_columnas_del_contrato():
    with pytest.raises(ValueError, match="faltan columnas"):
        as_of_series(pd.DataFrame({"serie": [SERIE]}), SERIE, "2025-01-01")


# --- (f) Empates ---------------------------------------------------------------------------------
def test_t1_as_of_empate_de_publicacion_lo_resuelve_el_vintage():
    ref = pd.Timestamp("2024-01-01")
    pub = pd.Timestamp("2024-02-05")
    base = _tabla([
        {"fecha_referencia": ref, "fecha_publicacion": pub, "vintage": "2024-02-03", "valor": 1.0},
        {"fecha_referencia": ref, "fecha_publicacion": pub, "vintage": "2024-02-05", "valor": 3.0},
        {"fecha_referencia": ref, "fecha_publicacion": pub, "vintage": "2024-02-04", "valor": 2.0},
        {"fecha_referencia": ref, "fecha_publicacion": pub - pd.Timedelta(days=10), "vintage": "2024-02-06",
         "valor": 0.5},   # vintage mayor pero publicada antes: pierde frente a la publicación más reciente
    ])
    for semilla in range(5):   # el orden de las filas de la base no importa
        barajada = base.sample(frac=1.0, random_state=semilla)
        r = as_of_series(barajada, "sint:X", pub)
        assert len(r) == 1 and r["valor"].iloc[0] == 3.0 and r["vintage"].iloc[0] == "2024-02-05"
    r = as_of_series(base, "sint:X", pub - pd.Timedelta(days=1))
    assert r["valor"].iloc[0] == 0.5


# --- Propiedades con hypothesis -----------------------------------------------------------------
def _hypothesis():
    return pytest.importorskip("hypothesis")


def test_t1_as_of_propiedades_aleatorias():
    hyp = _hypothesis()
    st = hyp.strategies
    origen = pd.Timestamp("2024-01-01")

    fila = st.tuples(
        st.integers(0, 5),                       # periodo (mes)
        st.integers(0, 120),                     # días de publicación desde el origen
        st.integers(0, 400),                     # días del vintage desde el origen
        st.one_of(st.none(), st.floats(-1e6, 1e6, allow_nan=False)),
    )

    @hyp.settings(max_examples=150, deadline=None, database=None)
    @hyp.given(filas=st.lists(fila, min_size=0, max_size=30), corte=st.integers(-5, 130),
               extra=st.lists(fila, min_size=0, max_size=10))
    def propiedad(filas, corte, extra):
        t = origen + pd.Timedelta(days=corte)

        def registros(tuplas, solo_futuras=False, desplaza=0):
            out = {}
            for mes, pub, vin, valor in tuplas:
                ref = origen + pd.offsets.MonthBegin(mes) if mes else origen
                fp = origen + pd.Timedelta(days=pub)
                if solo_futuras:
                    fp = t + pd.Timedelta(days=1 + pub)
                fp = max(fp, ref)   # contrato: nada se publica antes de su periodo
                vintage = (origen + pd.Timedelta(days=vin + desplaza)).strftime("%Y-%m-%d")
                clave = (mes, vintage)
                out[clave] = {"fecha_referencia": ref,
                              "fecha_publicacion": fp, "vintage": vintage,
                              "valor": np.nan if valor is None else valor,
                              "nota": "hueco" if valor is None else None}
            return out

        base_reg = registros(filas)
        if not base_reg:
            return
        base = _tabla(list(base_reg.values()))
        r = as_of_series(base, "sint:X", t)

        # no-futuro y una fila por periodo, igual que la referencia ingenua
        assert (r["fecha_publicacion"] <= t).all()
        assert r["fecha_referencia"].is_unique and r["fecha_referencia"].is_monotonic_increasing
        ingenuo = _as_of_ingenuo(base, "sint:X", t)
        assert list(r["fecha_referencia"]) == sorted(ingenuo)
        for ref, valor, nota in zip(r["fecha_referencia"], r["valor"], r["nota"], strict=True):
            _, v_ing, n_ing = ingenuo[ref]
            assert (pd.isna(valor) and pd.isna(v_ing)) or valor == v_ing
            assert nota == n_ing

        # truncado: quitar lo posterior a t o añadir filas futuras (vintages nuevos) no cambia nada
        pd.testing.assert_frame_equal(as_of_series(base[base["fecha_publicacion"] <= t], "sint:X", t), r)
        futuras = registros(extra, solo_futuras=True, desplaza=1000)
        if futuras:
            ampliada = pd.concat([base, _tabla(list(futuras.values()))], ignore_index=True)
            pd.testing.assert_frame_equal(as_of_series(ampliada, "sint:X", t), r)

        # lo que se sabe solo crece: los periodos visibles en t siguen visibles después
        r2 = as_of_series(base, "sint:X", t + pd.Timedelta(days=7))
        assert set(r["fecha_referencia"]) <= set(r2["fecha_referencia"])

    propiedad()


def test_t1_as_of_rechaza_fechas_en_texto(vintages):
    """Una base sin validar (fechas en texto) da un ValueError claro, no un TypeError críptico."""
    texto = vintages.assign(fecha_publicacion=vintages["fecha_publicacion"].dt.strftime("%Y-%m-%d"))
    with pytest.raises(ValueError, match="datetime64"):
        as_of_series(texto, SERIE, "2025-01-15")
