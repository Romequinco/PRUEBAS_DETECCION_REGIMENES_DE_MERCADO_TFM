"""Tests de ``regimenes.sinteticos.validacion.fidelidad`` (notebook 16).

Casos con respuesta conocida sobre paneles de juguete: el real contra si mismo
cae en banda, una escala doble da cociente ~2 fuera de banda, el ruido i.i.d.
tiene ACF ~0, la ACF dentro de regimen no mezcla rachas y el contrato de
columnas de salida se cumple en las cinco funciones. El test marcado ``datos``
corre marginal y dependencia sobre la pista A real con ``bootstrap_regimen``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from generador_referencia import panel_con_sp500, panel_juguete, regimen_juguete

from regimenes import rutas
from regimenes.sinteticos.validacion import fidelidad as F

N_BOOT = 40


def _con_regimen(panel: pd.DataFrame, reg) -> pd.DataFrame:
    salida = panel.copy()
    salida["regime"] = np.asarray(reg, dtype=int)
    return salida


def _trayectorias_juguete(n_tray: int = 6, n: int = 2100, semilla: int = 100) -> list[pd.DataFrame]:
    """Trayectorias de la misma ley que ``panel_juguete`` (volatilidad x3 en crisis)."""
    return [_con_regimen(*panel_juguete(n, semilla + i)) for i in range(n_tray)]


def _comprobar_contrato(tabla: pd.DataFrame) -> None:
    assert list(tabla.columns) == F.COLUMNAS_SALIDA
    assert set(tabla["regimen"]) <= {"calma", "crisis", "todos"}
    assert set(tabla["tipo"]) <= {"modelada", "re-derivada", F.SIN_COLUMNA}
    assert str(tabla["en_banda"].dtype) == "boolean"
    sin_banda = tabla["banda_inf"].isna() & tabla["banda_sup"].isna()
    assert tabla.loc[sin_banda, "en_banda"].isna().all()
    assert tabla.loc[~sin_banda, "en_banda"].notna().all()
    assert not tabla.duplicated(["regimen", "columna", "metrica"]).any()


def _valor(tabla: pd.DataFrame, regimen: str, columna: str, metrica: str) -> pd.Series:
    fila = tabla[(tabla.regimen == regimen) & (tabla.columna == columna) & (tabla.metrica == metrica)]
    assert len(fila) == 1, (regimen, columna, metrica)
    return fila.iloc[0]


# --------------------------------------------------------------------------- contrato


def test_contrato_de_salida_de_las_cinco_funciones() -> None:
    panel, reg = panel_juguete(2100, 0)
    panel = panel.rename(columns={"ret": "SP500_ret"})
    trays = [t.rename(columns={"ret": "SP500_ret"}) for t in _trayectorias_juguete(3)]
    tablas = {
        "marginal": F.fidelidad_marginal(panel, trays, regimen_real=reg, n_boot=N_BOOT),
        "dependencia": F.fidelidad_dependencia(panel, trays, regimen_real=reg, n_boot=N_BOOT),
        "correlaciones": F.distancia_correlaciones(panel, trays, regimen_real=reg, n_boot=N_BOOT),
        "regimenes": F.fidelidad_regimenes(reg, trays),
        "condicionamiento": F.condicionamiento(panel, trays, regimen_real=reg, n_boot=N_BOOT),
    }
    for tabla in tablas.values():
        _comprobar_contrato(tabla)
    marg = tablas["marginal"]
    assert set(marg["metrica"]) == {"media", "desviacion", "asimetria", "curtosis", "q01", "q05", "q95", "q99",
                                    "wasserstein", "ks"}
    assert len(marg) == 2 * 4 * 10
    dep = tablas["dependencia"]
    for k in (1, 5, 21):
        assert {f"acf_r_{k}", f"acf_abs_{k}", f"acf_sq_{k}", f"apalancamiento_{k}"} <= set(dep["metrica"])
    assert {"curtosis_condicional", "pendiente_loglog_abs"} <= set(dep["metrica"])
    assert set(dep.loc[dep.regimen == "todos", "metrica"]) == {"acf_r_1", "acf_abs_1"}
    assert set(tablas["correlaciones"]["metrica"]) == {"frobenius", "salto_correlacion"}
    assert {"fraccion_dias", "duracion_media", "duracion_mediana", "duracion_p90", "P_kk",
            "ks_duraciones"} <= set(tablas["regimenes"]["metrica"])
    assert set(tablas["condicionamiento"]["metrica"]) == {"separacion_vol", "media_ret_crisis_pb",
                                                          "auc_regimen", "auc_regimen_ajuste"}
    # determinismo con la semilla
    otra = F.fidelidad_marginal(panel, trays, regimen_real=reg, n_boot=N_BOOT)
    pd.testing.assert_frame_equal(marg, otra)


def test_tipo_distingue_columnas_rederivadas() -> None:
    panel, reg, _ = panel_con_sp500(900, 1)
    tabla = F.fidelidad_marginal(panel, [_con_regimen(panel, reg)], regimen_real=reg, n_boot=10)
    tipos = tabla.groupby("columna")["tipo"].first()
    assert tipos["SP500_vol_z"] == tipos["SP500_drawdown"] == "re-derivada"
    assert tipos["SP500_ret"] == tipos["FF_MKT_z"] == tipos["macro"] == "modelada"


# --------------------------------------------------------------------------- bootstrap


def test_remuestreo_calma_por_bloques_y_crisis_por_episodio() -> None:
    reg = regimen_juguete(2100)
    boot = F._remuestreos(reg, 20, semilla=3, largo_bloque=63)
    episodios = [(a, b) for a, b, k in F.tramos(reg) if k == 1]
    for idx, seg in boot[0]:
        assert idx.size == int((reg == 0).sum())   # mismo numero de dias de calma
        assert (reg[idx] == 0).all()               # ningun bloque cruza a crisis
        for s in np.unique(seg):                   # cada bloque es contiguo
            assert (np.diff(idx[seg == s]) == 1).all()
    for idx, seg in boot[1]:
        assert np.unique(seg).size == len(episodios)  # tantos episodios como el real
        for s in np.unique(seg):
            bloque = idx[seg == s]
            assert (bloque[0], bloque[-1] + 1) in episodios  # episodios completos


# --------------------------------------------------------------------------- marginal


def test_real_contra_si_mismo_queda_en_banda() -> None:
    panel, reg = panel_juguete(2100, 0)
    tabla = F.fidelidad_marginal(panel, [_con_regimen(panel, reg)], regimen_real=reg, n_boot=N_BOOT)
    distancias = tabla[tabla.metrica.isin(["wasserstein", "ks"])]
    assert np.allclose(distancias["sintetico"], 0.0) and distancias["en_banda"].all()
    resto = tabla[~tabla.metrica.isin(["wasserstein", "ks"])]
    assert resto["en_banda"].mean() >= 0.9
    con_cociente = resto.dropna(subset=["cociente"])
    assert set(con_cociente["metrica"]) == {"desviacion", "q01", "q05", "q95", "q99"}
    assert np.allclose(con_cociente["cociente"], 1.0)


def test_escala_doble_da_cociente_dos_fuera_de_banda() -> None:
    panel, reg = panel_juguete(2100, 0)
    doble = [_con_regimen(2.0 * panel, reg)]
    tabla = F.fidelidad_marginal(panel, doble, regimen_real=reg, n_boot=N_BOOT)
    for regimen in ("calma", "crisis"):
        for col in ("ret", "spread"):
            fila = _valor(tabla, regimen, col, "desviacion")
            assert fila["cociente"] == pytest.approx(2.0)
            assert not fila["en_banda"]
            # la amplitud de cola tambien se duplica (es invariante a la posicion)
            assert _valor(tabla, regimen, col, "q99")["cociente"] == pytest.approx(2.0)
            assert not _valor(tabla, regimen, col, "wasserstein")["en_banda"]


def test_misma_ley_queda_cerca_y_curtosis_es_de_pearson() -> None:
    panel, reg = panel_juguete(2100, 0)
    tabla = F.fidelidad_marginal(panel, _trayectorias_juguete(6), regimen_real=reg, n_boot=N_BOOT)
    sd = tabla[tabla.metrica == "desviacion"]
    assert sd["cociente"].between(0.85, 1.15).all()
    curt = tabla[tabla.metrica == "curtosis"]
    assert curt["sintetico"].between(2.7, 3.3).all()  # normal -> 3 (no exceso)


# --------------------------------------------------------------------------- dependencia


def test_ruido_iid_acf_cero_dentro_de_regimen() -> None:
    panel, reg = panel_juguete(3000, 0)
    panel = panel.rename(columns={"ret": "SP500_ret"})
    trays = [t.rename(columns={"ret": "SP500_ret"}) for t in _trayectorias_juguete(8, 3000)]
    tabla = F.fidelidad_dependencia(panel, trays, regimen_real=reg, n_boot=N_BOOT)
    en_banda = []
    for regimen in ("calma", "crisis"):
        for m in ("acf_r_1", "acf_abs_1", "acf_sq_1", "acf_abs_5"):
            fila = _valor(tabla, regimen, "SP500_ret", m)
            assert abs(fila["sintetico"]) < 0.05 and abs(fila["real"]) < 0.1
            en_banda.append(bool(fila["en_banda"]))
        # en crisis la EWMA tarda en recoger el salto x3 de volatilidad al entrar en el
        # regimen (mezcla de escalas): la curtosis condicional sube por el estimador causal
        cc = _valor(tabla, regimen, "SP500_ret", "curtosis_condicional")
        assert 2.9 < cc["sintetico"] < (4.0 if regimen == "calma" else 8.0)
    # una banda al 95 % deja fuera al valor poblacional ~5 % de las veces por metrica
    # (con esta semilla el real de crisis tiene acf_r_1 = -0,09, a ~2,7 errores estandar)
    assert sum(en_banda) >= 6
    # con la mezcla de regimenes la ACF de |r| de la serie entera ya no es ~0
    todos = _valor(tabla, "todos", "SP500_ret", "acf_abs_1")
    assert todos["real"] > 0.1 and todos["sintetico"] > 0.1


def test_acf_dentro_de_regimen_no_mezcla_rachas() -> None:
    # crisis en rachas de longitud IMPAR con r alternando +-: dentro de cada racha la
    # acf1 es -1 exacta; si se juntaran rachas, el par (fin de una, inicio de la
    # siguiente) seria (+, +) y la acf1 dejaria de ser -1.
    n = 1500
    reg = np.zeros(n, dtype=int)
    for ini in range(100, n - 100, 300):
        reg[ini:ini + 61] = 1
    rng = np.random.default_rng(0)
    r = rng.standard_normal(n) * 0.01
    for a, b, k in F.tramos(reg):
        if k == 1:
            r[a:b] = 0.01 * np.where(np.arange(b - a) % 2 == 0, 1.0, -1.0)
    idx = pd.bdate_range("2000-01-03", periods=n)
    panel = pd.DataFrame({"SP500_ret": r, "otra": rng.standard_normal(n)}, index=idx)
    tabla = F.fidelidad_dependencia(panel, [_con_regimen(panel, reg)], regimen_real=reg, n_boot=10)
    fila = _valor(tabla, "crisis", "SP500_ret", "acf_r_1")
    assert fila["real"] == pytest.approx(-1.0) and fila["sintetico"] == pytest.approx(-1.0)
    # el bootstrap por episodio tampoco mezcla episodios
    assert fila["banda_inf"] == pytest.approx(-1.0) and fila["banda_sup"] == pytest.approx(-1.0)


def test_persistentes_y_escalones() -> None:
    n = 2100
    panel, reg = panel_juguete(n, 0)
    panel = panel.rename(columns={"ret": "SP500_ret"})
    panel["mensual"] = np.repeat(np.random.default_rng(1).standard_normal(n // 21 + 1).cumsum(), 21)[:n]
    tabla = F.fidelidad_dependencia(panel, [_con_regimen(panel, reg)], regimen_real=reg, n_boot=10)
    assert tabla.attrs["persistentes"] == ["mensual"] and tabla.attrs["mensuales"] == ["mensual"]
    esc = _valor(tabla, "calma", "mensuales", "escalones")
    assert esc["real"] == pytest.approx(20 / 21, abs=0.01) and esc["cociente"] == pytest.approx(1.0)
    # un sintetico sin escalones (ruido) da fraccion 0 y sale de banda
    ruido = panel.copy()
    ruido["mensual"] = np.random.default_rng(2).standard_normal(n)
    tabla = F.fidelidad_dependencia(panel, [_con_regimen(ruido, reg)], regimen_real=reg, n_boot=10)
    for m, col in (("escalones", "mensuales"), ("acf1_persistentes", "persistentes")):
        fila = _valor(tabla, "calma", col, m)
        assert fila["sintetico"] < 0.1 and not fila["en_banda"]


def test_ewma_causal_y_curtosis_condicional_de_la_normal() -> None:
    # ruido normal de varianza constante: z = r / sigma_EWMA tiene curtosis ~3,2 (algo
    # mas que 3 por el error de estimar sigma con ~30 dias efectivos: referencia de
    # "sin cola condicional" en la lectura del notebook 16)
    largo = np.random.default_rng(1).standard_normal(100_000)
    assert 3.0 < F._curtosis(F._z_ewma(largo)) < 3.4
    r = np.random.default_rng(0).standard_normal(300)
    z = F._z_ewma(r)
    assert np.isnan(z[:21]).all() and np.isfinite(z[21:]).all()
    # cambiar el futuro no cambia el pasado
    r2 = r.copy()
    r2[200:] *= 10
    np.testing.assert_allclose(F._z_ewma(r2)[:200], z[:200])
    assert F._z_ewma(r2)[201] != z[201]


# --------------------------------------------------------------------------- correlaciones


def test_distancia_correlaciones() -> None:
    panel, reg = panel_juguete(2100, 0)
    tabla = F.distancia_correlaciones(panel, [_con_regimen(panel, reg)], regimen_real=reg, n_boot=N_BOOT)
    assert np.allclose(tabla["sintetico"], 0.0) and tabla["en_banda"].all()
    # sintetico con todas las columnas muy correladas: lejos de las ~0 reales
    rng = np.random.default_rng(5)
    comun = rng.standard_normal((len(panel), 1))
    correlado = pd.DataFrame(comun + 0.2 * rng.standard_normal(panel.shape), index=panel.index,
                             columns=panel.columns)
    tabla = F.distancia_correlaciones(panel, [_con_regimen(correlado, reg)], regimen_real=reg, n_boot=N_BOOT)
    frob = tabla[tabla.metrica == "frobenius"]
    assert (frob["sintetico"] > 0.8).all() and not frob["en_banda"].any()


# --------------------------------------------------------------------------- regimenes y condicionamiento


def test_fidelidad_regimenes_misma_cadena() -> None:
    reg = pd.Series(regimen_juguete(2100))
    tabla = F.fidelidad_regimenes(reg, [_con_regimen(panel_juguete(2100, 0)[0], reg)])
    con_cociente = tabla.dropna(subset=["cociente"])
    assert np.allclose(con_cociente["cociente"], 1.0)
    assert np.allclose(tabla.loc[tabla.metrica == "ks_duraciones", "sintetico"], 0.0)
    assert _valor(tabla, "crisis", F.SIN_COLUMNA, "duracion_media")["real"] == 60


def test_condicionamiento_detecta_regimen_sin_senal() -> None:
    def con_nivel(panel, reg, semilla):
        salida = panel.rename(columns={"ret": "SP500_ret"})
        salida["nivel"] = 1.5 * np.asarray(reg) + np.random.default_rng(semilla).standard_normal(len(panel))
        return salida

    panel, reg = panel_juguete(2100, 0)
    real = con_nivel(panel, reg, 0)
    fieles = [_con_regimen(con_nivel(*panel_juguete(2100, 10 + i), 20 + i), reg) for i in range(4)]
    tabla = F.condicionamiento(real, fieles, regimen_real=reg, n_boot=N_BOOT)
    sep = _valor(tabla, "todos", "SP500_ret", "separacion_vol")
    assert sep["real"] == pytest.approx(3.0, rel=0.15)
    assert sep["cociente"] == pytest.approx(1.0, abs=0.15)
    aj = _valor(tabla, "todos", F.SIN_COLUMNA, "auc_regimen_ajuste")
    assert aj["real"] > 0.8 and aj["cociente"] == pytest.approx(1.0, abs=0.05)
    assert _valor(tabla, "todos", F.SIN_COLUMNA, "auc_regimen")["sintetico"] > 0.8
    # regimen barajado: las features ya no llevan senal de regimen
    rng = np.random.default_rng(7)
    sin_senal = [t.assign(regime=rng.permutation(t["regime"].to_numpy())) for t in fieles]
    tabla = F.condicionamiento(real, sin_senal, regimen_real=reg, n_boot=N_BOOT)
    sep = _valor(tabla, "todos", "SP500_ret", "separacion_vol")
    assert sep["sintetico"] == pytest.approx(1.0, abs=0.1) and not sep["en_banda"]
    assert abs(_valor(tabla, "todos", F.SIN_COLUMNA, "auc_regimen")["sintetico"] - 0.5) < 0.05
    assert _valor(tabla, "todos", F.SIN_COLUMNA, "auc_regimen_ajuste")["cociente"] < 0.7


# --------------------------------------------------------------------------- datos reales


@pytest.mark.datos
def test_pista_a_real_bootstrap_regimen() -> None:
    import yaml

    from regimenes.benchmark.cache import processed_available
    from regimenes.sinteticos import datos, persistencia

    if not processed_available("A") or not any(rutas.DATA_RAW.glob("*/SP500.parquet")):
        pytest.skip("faltan data/processed (pista A) o data/raw/<fuente>/SP500.parquet")
    cfg = yaml.safe_load(rutas.SINTETICOS_CONFIG.read_text(encoding="utf-8"))["entrenamiento"]
    try:
        trays = persistencia.cargar_trayectorias("bootstrap_regimen", "A")
    except FileNotFoundError:
        pytest.skip("faltan las trayectorias de bootstrap_regimen (notebook 15)")
    panel, reg = datos.cargar_entrenamiento("A", cfg["fin_train"], cfg["features"])
    trays = trays[:30]

    marg = F.fidelidad_marginal(panel, trays, regimen_real=reg, n_boot=50)
    _comprobar_contrato(marg)
    assert set(marg["columna"]) == set(panel.columns)
    sd = marg[(marg.metrica == "desviacion") & (marg.columna == "SP500_ret")]
    assert sd["cociente"].between(0.8, 1.25).all()  # el remuestreo conserva la escala por regimen

    dep = F.fidelidad_dependencia(panel, trays, regimen_real=reg, n_boot=50)
    _comprobar_contrato(dep)
    assert "SP500_vol_z" in dep.attrs["persistentes"]
    assert set(dep.attrs["mensuales"]) == {"credit_BaaAaa_mensual_z", "term_spread_hist_z", "INDPRO_yoy_z"}
    calma = _valor(dep, "calma", "SP500_ret", "acf_abs_1")
    assert calma["real"] > 0.1 and calma["sintetico"] > 0.05  # agrupamiento de volatilidad
    cc = _valor(dep, "calma", "SP500_ret", "curtosis_condicional")
    assert cc["real"] > 4.0


# --------------------------------------------------------------------------- ampliacion: vida media y curtosis por trayectoria


def _panel_ar1(rho: float, n: int, semilla: int, reg: np.ndarray) -> pd.DataFrame:
    """``SP500_ret`` normal y una columna AR(1) estacionaria de acf1 ``rho``."""
    rng = np.random.default_rng(semilla)
    eps = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = eps[0] / np.sqrt(1 - rho ** 2)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + eps[t]
    idx = pd.bdate_range("2000-01-03", periods=n)
    return pd.DataFrame({"SP500_ret": 0.01 * rng.standard_normal(n), "ar1": x}, index=idx).assign(regime=reg)


def test_vida_media_ar1_teorica_y_cociente() -> None:
    n, rho = 6000, 0.97
    reg = regimen_juguete(n)
    real = _panel_ar1(rho, n, 0, reg)
    teorica = np.log(0.5) / np.log(rho)  # ~22,8 sesiones
    tabla = F.fidelidad_dependencia(real, [real], n_boot=N_BOOT)
    assert tabla.attrs["persistentes"] == ["ar1"]
    fila = _valor(tabla, "calma", "persistentes", "vida_media_persistentes")
    assert fila["real"] == pytest.approx(teorica, rel=0.2)
    assert fila["cociente"] == pytest.approx(1.0) and fila["en_banda"]
    # misma ley: cociente ~1; persistencia distinta (rho 0,90, vida media ~6,6): fuera de banda
    misma = F.fidelidad_dependencia(real, [_panel_ar1(rho, n, 10 + i, reg) for i in range(4)], n_boot=N_BOOT)
    assert _valor(misma, "calma", "persistentes", "vida_media_persistentes")["cociente"] == pytest.approx(1.0, abs=0.25)
    otra = F.fidelidad_dependencia(real, [_panel_ar1(0.90, n, 10 + i, reg) for i in range(4)], n_boot=N_BOOT)
    for regimen in ("calma", "crisis"):
        fila = _valor(otra, regimen, "persistentes", "vida_media_persistentes")
        assert fila["cociente"] < 0.5 and not fila["en_banda"]
    # el acf1 de la misma comparacion apenas se mueve (motivo de la metrica nueva)
    assert _valor(otra, "calma", "persistentes", "acf1_persistentes")["cociente"] > 0.9


def test_vida_media_limites() -> None:
    h = F._vida_media([-0.2, 0.0, 0.5, 1.0, 0.99999999, np.nan])
    assert h[0] == h[1] == 0.0 and h[2] == pytest.approx(1.0)
    assert h[3] == h[4] == F.TOPE_VIDA_MEDIA and np.isnan(h[5])


def _panel_colas(n: int, semilla: int, reg: np.ndarray, gl: float | None) -> pd.DataFrame:
    """Retornos i.i.d. de varianza constante: t de Student con ``gl`` grados o normal (None)."""
    rng = np.random.default_rng(semilla)
    r = rng.standard_normal(n) if gl is None else rng.standard_t(gl, n) / np.sqrt(gl / (gl - 2))
    idx = pd.bdate_range("2000-01-03", periods=n)
    return pd.DataFrame({"SP500_ret": 0.01 * r, "otra": rng.standard_normal(n)}, index=idx).assign(regime=reg)


def test_curtosis_condicional_por_trayectoria() -> None:
    n = 2520
    reg = regimen_juguete(n)
    real = _panel_colas(n, 0, reg, gl=4.0)
    misma = [_panel_colas(n, 100 + i, reg, gl=4.0) for i in range(40)]
    tabla = F.fidelidad_dependencia(real, misma, n_boot=10)
    for regimen in ("calma", "crisis"):
        fila = _valor(tabla, regimen, "SP500_ret", "curtosis_condicional_tray")
        assert fila["real"] == _valor(tabla, regimen, "SP500_ret", "curtosis_condicional")["real"]
        assert np.isnan(fila["cociente"]) and fila["banda_inf"] < fila["real"] < fila["banda_sup"]
        assert fila["en_banda"]
    gauss = [_panel_colas(n, 100 + i, reg, gl=None) for i in range(40)]
    tabla = F.fidelidad_dependencia(real, gauss, n_boot=10)
    fila = _valor(tabla, "calma", "SP500_ret", "curtosis_condicional_tray")
    assert fila["real"] > fila["banda_sup"] and not fila["en_banda"]
    # con menos de 10 trayectorias validas no hay banda
    tabla = F.fidelidad_dependencia(real, gauss[:5], n_boot=10)
    fila = _valor(tabla, "calma", "SP500_ret", "curtosis_condicional_tray")
    assert np.isnan(fila["banda_inf"]) and pd.isna(fila["en_banda"])


# --------------------------------------------------------------------------- ampliacion: referencia por ventanas


def test_referencia_por_ventanas_con_largo_del_real_coincide_con_el_real() -> None:
    n = 3000
    reg = regimen_juguete(n)
    real = _panel_ar1(0.97, n, 0, reg)
    trays = [_panel_ar1(0.97, 1000, 10 + i, regimen_juguete(1000)) for i in range(3)]
    base = F.fidelidad_dependencia(real, trays, n_boot=10)
    vent = F.fidelidad_dependencia(real, trays, n_boot=10, largo_referencia=n, min_ventanas_no_solapadas=1)
    assert vent.attrs["n_ventanas_reales"] == 1 and not vent.attrs["ventanas_solapadas"]
    assert "vida_media_persistentes" in vent.attrs["referencia_ventanas"]
    assert {"acf_abs_1", "acf_r_5", "acf_sq_21", "apalancamiento_1"} <= set(vent.attrs["referencia_ventanas"])
    assert base.attrs["referencia_ventanas"] == [] and base.attrs["n_ventanas_reales"] == 0
    pd.testing.assert_series_equal(base["real"], vent["real"])  # una ventana = el real entero
    afectadas = vent.metrica.isin(vent.attrs["referencia_ventanas"]) & (vent.regimen != "todos")
    # banda de una sola ventana: degenerada en el propio valor
    assert np.allclose(vent.loc[afectadas, "banda_inf"], vent.loc[afectadas, "real"], equal_nan=True)
    # el resto de filas no cambia
    pd.testing.assert_frame_equal(base[~afectadas], vent[~afectadas])


def test_referencia_por_ventanas_corrige_el_sesgo_de_muestra_corta() -> None:
    # AR(1) muy persistente: la acf1 estimada en 600 sesiones esta sesgada a la baja
    # frente a la de 12.000; la referencia por ventanas de 600 lo compensa.
    rho, n, largo = 0.995, 12_000, 600
    real = _panel_ar1(rho, n, 0, np.zeros(n, dtype=int))
    trays = [_panel_ar1(rho, largo, 100 + i, np.zeros(largo, dtype=int)) for i in range(60)]
    sin = _valor(F.fidelidad_dependencia(real, trays, n_boot=20), "calma", "persistentes",
                 "vida_media_persistentes")
    con = F.fidelidad_dependencia(real, trays, n_boot=20, largo_referencia=largo, paso_ventanas=300)
    assert con.attrs["ventanas_solapadas"] and con.attrs["n_ventanas_reales"] == 39
    fila = _valor(con, "calma", "persistentes", "vida_media_persistentes")
    assert sin["cociente"] < 0.6 and not sin["en_banda"]
    # la mediana entre ventanas reales es ruidosa (ventanas correladas de una sola senda):
    # con 4 semillas el cociente quedo entre 0,78 y 1,08, frente a 0,36-0,54 sin ventanas
    assert 0.75 < fila["cociente"] < 1.33 and fila["en_banda"]
    assert fila["cociente"] > sin["cociente"] + 0.2


def test_referencia_por_ventanas_no_se_aplica_con_menos_de_tres_ventanas() -> None:
    n = 3000
    reg = regimen_juguete(n)
    real = _panel_ar1(0.97, n, 0, reg)
    trays = [_panel_ar1(0.97, 1000, 10 + i, regimen_juguete(1000)) for i in range(3)]
    base = F.fidelidad_dependencia(real, trays, n_boot=10)
    corta = F.fidelidad_dependencia(real, trays, n_boot=10, largo_referencia=1001)  # 3000 // 1001 = 2
    pd.testing.assert_frame_equal(base, corta)
    assert not corta.attrs["referencia_ventanas_aplicada"] and corta.attrs["referencia_ventanas"] == []
    assert corta.attrs["motivo_referencia"].startswith("no aplicada")
    larga = F.fidelidad_dependencia(real, trays, n_boot=10, largo_referencia=1000)  # 3 ventanas: si
    assert larga.attrs["referencia_ventanas_aplicada"] and larga.attrs["referencia_ventanas"]
