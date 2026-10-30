"""Tests sintéticos (sin datos en disco) de los termómetros por bloque (regimenes.features.bloques).

Validación 1 de PendientesOscar.md §6.5: recuperar un factor plantado, prueba de prefijo
(truncado), signo estable entre reajustes, NaN que no se propagan, umbral de varianza.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regimenes import features as ft
from regimenes.features.bloques import cargar_bloques, parametros_bloques

MIN_OBS = 126


def _panel_factor(n: int = 1000, seed: int = 0, cargas=(0.9, 0.7, -0.6, 0.5), ruido: float = 0.5
                  ) -> tuple[pd.DataFrame, pd.Series]:
    """Columnas c_i = carga_i * f + ruido, con f AR(1) persistente."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n)
    f = np.zeros(n)
    e = rng.normal(size=n)
    for t in range(1, n):
        f[t] = 0.97 * f[t - 1] + 0.25 * e[t]
    f /= f.std()
    cols = {f"c{i}": l * f + ruido * rng.normal(size=n) for i, l in enumerate(cargas)}
    return pd.DataFrame(cols, index=idx), pd.Series(f, index=idx, name="f")


def _con_nans(panel: pd.DataFrame) -> pd.DataFrame:
    """c3 empieza tarde (fila 500) y c1 tiene un NaN puntual (fila 800)."""
    p = panel.copy()
    p.iloc[:500, p.columns.get_loc("c3")] = np.nan
    p.iloc[800, p.columns.get_loc("c1")] = np.nan
    return p


ANCLA = {"columna": "c0", "signo": 1}


def test_fechas_reajuste_ultimo_dia_de_cada_mes():
    idx = pd.bdate_range("2001-01-01", "2001-03-31")
    r = ft.fechas_reajuste(idx)
    assert list(r) == [pd.Timestamp("2001-01-31"), pd.Timestamp("2001-02-28"), pd.Timestamp("2001-03-30")]


def test_recupera_factor_plantado():
    panel, f = _panel_factor()
    t, d = ft.termometro_bloque(panel, list(panel.columns), ANCLA, min_obs=MIN_OBS)
    ok = t.notna()
    assert ok.sum() > 600
    assert abs(np.corrcoef(t[ok], f[ok])[0, 1]) > 0.9
    assert (d.dropna(subset=["var_pc1"])["metodo"] == "pca").all()


def test_antes_del_primer_reajuste_valido_es_nan():
    panel, _ = _panel_factor()
    t, d = ft.termometro_bloque(panel, list(panel.columns), ANCLA, min_obs=MIN_OBS)
    primer = d.loc[d["metodo"] == "pca", "fecha_reajuste"].iloc[0]
    assert t.loc[:primer].isna().all()
    assert t.loc[primer:].iloc[1:].notna().all()
    assert (panel.index <= primer).sum() >= MIN_OBS


@pytest.mark.parametrize("kw", [
    {},
    {"umbral_var": 0.95},                    # fuerza tramos 'media'
    {"ventana": 300},                        # ventana rolling
])
@pytest.mark.parametrize("cut", ["2001-06-29", "2001-11-14", "2002-03-16", "2002-09-30", "2003-02-05", "2000-03-15"])
def test_prefijo_truncado_identico(kw, cut):
    panel, _ = _panel_factor()
    panel = _con_nans(panel)
    cols = list(panel.columns)
    raw = {c: panel[c] for c in cols}

    def builder(r):
        return ft.termometro_bloque(pd.concat(r, axis=1), cols, ANCLA, min_obs=MIN_OBS, **kw)[0].rename("T")

    res = ft.assert_causal(builder, raw, cut=cut, tol=1e-10)
    assert res["causal_ok"].all(), res


def test_signo_no_salta_entre_reajustes():
    panel, f = _panel_factor()
    for signo in (1, -1):
        t, d = ft.termometro_bloque(panel, list(panel.columns), {"columna": "c0", "signo": signo},
                                    min_obs=MIN_OBS)
        v = d.dropna(subset=["var_pc1"])
        assert (v["carga_ancla"] * signo > 0).all()
        assert (v["coseno_prev"].dropna() > 0.9).all()
        ok = t.notna()
        assert signo * np.corrcoef(t[ok], panel.loc[ok, "c0"])[0, 1] > 0
    # ancla con carga negativa en el factor: el termómetro sube con la ancla orientada
    t, d = ft.termometro_bloque(panel, list(panel.columns), {"columna": "c2", "signo": 1}, min_obs=MIN_OBS)
    ok = t.notna()
    assert np.corrcoef(t[ok], panel.loc[ok, "c2"])[0, 1] > 0
    assert np.corrcoef(t[ok], f[ok])[0, 1] < -0.9


def test_columna_tardia_y_nan_puntual_no_propagan():
    panel, f = _panel_factor()
    panel = _con_nans(panel)
    t, d = ft.termometro_bloque(panel, list(panel.columns), ANCLA, min_obs=MIN_OBS)
    primer = d.loc[d["metodo"] == "pca", "fecha_reajuste"].iloc[0]
    tras = t.loc[primer:].iloc[1:]
    assert tras.notna().all()                                    # ni antes ni después de la entrada de c3
    assert pd.notna(t.iloc[800])                                 # NaN puntual de c1
    v = d.dropna(subset=["var_pc1"])
    assert "c3" not in v["elegibles"].iloc[0]
    assert "c3" in v["elegibles"].iloc[-1]
    assert int(d["n_dias_renormalizados"].sum()) == 1            # solo el día del NaN puntual
    ok = t.notna()
    assert abs(np.corrcoef(t[ok], f[ok])[0, 1]) > 0.9


def test_umbral_alto_con_columnas_independientes_cae_en_media():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2000-01-03", periods=800)
    panel = pd.DataFrame(rng.normal(size=(800, 5)), index=idx, columns=[f"x{i}" for i in range(5)])
    anc = {"columna": "x0", "signo": 1}
    _, d_alto = ft.termometro_bloque(panel, list(panel.columns), anc, umbral_var=0.5, min_obs=MIN_OBS)
    v = d_alto.dropna(subset=["var_pc1"])
    assert (v["metodo"] == "media").all()
    assert (v["var_pc1"] < 0.5).all()
    assert np.allclose(v["peso_max_abs"], 1 / 5)
    assert all(np.allclose(np.abs(list(c.values())), 1 / np.sqrt(5)) for c in v["cargas"])
    assert (np.array([c["x0"] for c in v["cargas"]]) > 0).all()
    _, d_cero = ft.termometro_bloque(panel, list(panel.columns), anc, umbral_var=0.0, min_obs=MIN_OBS)
    assert (d_cero.dropna(subset=["var_pc1"])["metodo"] == "pca").all()


def test_bloque_de_una_columna():
    panel, _ = _panel_factor()
    t, d = ft.termometro_bloque(panel, ["c2"], {"columna": "c2", "signo": -1}, min_obs=MIN_OBS)
    pd.testing.assert_series_equal(t, -panel["c2"], check_names=False)
    assert d["metodo"].tolist() == ["columna_unica"]


def test_construir_termometros_con_columna_ausente():
    panel, _ = _panel_factor()
    bloques = {
        "T1_juguete": {"columnas": ["c0", "c1", "CAPE_z"], "ancla": ANCLA},
        "T2_juguete": {"columnas": ["c2", "c3"], "ancla": {"columna": "c2", "signo": -1}},
        "T3_unica": {"columnas": ["c3", "NO_EXISTE"], "ancla": {"columna": "c3", "signo": 1}},
    }
    t, diags = ft.construir_termometros(panel, bloques, min_obs=MIN_OBS)
    assert list(t.columns) == list(bloques)
    assert diags["T1_juguete"]["ausentes"].iloc[0] == ["CAPE_z"]
    assert diags["T2_juguete"]["ausentes"].iloc[0] == []
    assert diags["T3_unica"]["metodo"].tolist() == ["columna_unica"]
    pd.testing.assert_series_equal(t["T3_unica"], panel["c3"], check_names=False)
    with pytest.warns(UserWarning):
        t2, d2 = ft.construir_termometros(panel, {"Tx": {"columnas": ["ZZ", "c0"],
                                                         "ancla": {"columna": "ZZ", "signo": 1}}})
    assert t2["Tx"].isna().all() and d2["Tx"]["metodo"].iloc[0] == "ancla_ausente"


def test_resumen_y_comparar_umbrales_sin_etiquetas():
    panel, _ = _panel_factor()
    bloques = {"T1": {"columnas": list(panel.columns), "ancla": ANCLA}}
    t, d = ft.construir_termometros(panel, bloques, min_obs=MIN_OBS)
    r = ft.resumen_estabilidad(t, d)
    assert r.loc[0, "n_saltos_signo"] == 0 and r.loc[0, "frac_pca"] == 1.0
    comp = ft.comparar_umbrales(panel, bloques, [0.3, 0.99], min_obs=MIN_OBS)
    assert len(comp) == 2 and comp["umbral_var"].tolist() == [0.3, 0.99]
    assert comp.loc[comp["umbral_var"] == 0.99, "frac_pca"].iloc[0] == 0.0


@pytest.mark.parametrize("pista, n_bloques, n_cols", [("A", 5, 35), ("B", 8, 77)])
def test_cargar_bloques(pista, n_bloques, n_cols):
    b = cargar_bloques(pista)
    assert len(b) == n_bloques
    cols = [c for x in b.values() for c in x["columnas"]]
    assert len(cols) == len(set(cols)) == n_cols
    for nombre, x in b.items():
        assert nombre.startswith("T")
        assert x["ancla"]["columna"] in x["columnas"]
        assert x["ancla"]["signo"] in (1, -1)
    p = parametros_bloques()
    assert p["umbrales_var_a_probar"] == [0.30, 0.40, 0.50]
    assert p["reajuste"] == "mensual"
