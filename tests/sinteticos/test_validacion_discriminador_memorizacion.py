"""``validacion.discriminador`` y ``validacion.memorizacion``: contrato y controles.

Controles con paneles de juguete (``generador_referencia``):

- discriminador: otra realizacion del mismo proceso -> AUC ~ 0,5; escala doble
  -> AUC ~ 1; el propio real remuestreado por bloques (copias) no sube de 0,5.
- memorizacion: copia literal (bootstrap por bloques) -> cociente ~ 0 y
  frac_copias ~ 1; jitter pequeno -> cociente << 0,9; gaussiano independiente
  ajustado -> cociente >= ~1.

El test ``datos`` usa la pista A real con las trayectorias de bootstrap_regimen
y jitter del notebook 15: ambos DEBEN suspender memorizacion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from generador_referencia import GaussianoReferencia, panel_juguete

from regimenes import rutas
from regimenes.sinteticos.validacion import discriminador, memorizacion

COLUMNAS_CONTRATO = ["regimen", "columna", "tipo", "metrica", "real", "sintetico",
                     "banda_inf", "banda_sup", "cociente", "en_banda"]
N = 4200


@pytest.fixture(scope="module")
def real():
    panel, reg = panel_juguete(N, semilla=0)
    return panel, reg


def _con_regimen(panel, reg):
    return panel.assign(regime=np.asarray(reg))


def _otras_realizaciones(n_tray=12, largo=1260):
    out = []
    for s in range(1, n_tray + 1):
        p, r = panel_juguete(largo, semilla=100 + s)
        out.append(_con_regimen(p, r))
    return out


def _bootstrap_bloques(panel, reg, n_tray=12, largo=1260, bloque=63, semilla=0):
    """Copias literales de bloques contiguos del real (regimen incluido)."""
    rng = np.random.default_rng(semilla)
    full = _con_regimen(panel, reg)
    out = []
    for _ in range(n_tray):
        inicios = rng.integers(0, len(full) - bloque, size=largo // bloque)
        out.append(pd.concat([full.iloc[a:a + bloque] for a in inicios]).reset_index(drop=True))
    return out


def _auc(tabla, regimen="todos"):
    return float(tabla.query("metrica == 'auc' and regimen == @regimen")["sintetico"].iloc[0])


def _mem(tabla, metrica, regimen="todos", campo="cociente"):
    return float(tabla.query("metrica == @metrica and regimen == @regimen")[campo].iloc[0])


# --------------------------------------------------------------------------- contrato


def test_contrato_discriminador(real):
    panel, reg = real
    t = discriminador(panel, _otras_realizaciones(), regimen_real=reg)
    assert list(t.columns) == COLUMNAS_CONTRATO
    assert set(t["regimen"]) == {"todos", "calma", "crisis"}
    assert set(t["metrica"]) == {"auc", "n_ventanas"}
    auc = t[t["metrica"] == "auc"]
    assert (auc["real"] == 0.5).all()
    assert (auc["banda_inf"] <= auc["sintetico"]).all() and (auc["sintetico"] <= auc["banda_sup"]).all()
    nv = t[t["metrica"] == "n_ventanas"].set_index("regimen")
    assert nv.loc["todos", "real"] == N // 21
    assert (nv["sintetico"] <= nv["real"]).all()  # equilibrado
    # determinista con semilla y regimen como columna equivalente a regimen_real
    t2 = discriminador(_con_regimen(panel, reg), _otras_realizaciones())
    pd.testing.assert_frame_equal(t, t2)


def test_contrato_memorizacion(real):
    panel, reg = real
    t = memorizacion(panel, _otras_realizaciones(), regimen_real=reg)
    assert list(t.columns) == COLUMNAS_CONTRATO
    assert set(t["metrica"]) == {"cociente_nn", "frac_copias", "dispersion", "n_consultas"}
    assert set(t["regimen"]) == {"todos", "calma", "crisis"}
    assert set(t["columna"]) == {"todas"}  # sin columnas re-derivadas solo hay un espacio
    c = t[t["metrica"] == "cociente_nn"]
    np.testing.assert_allclose(c["cociente"], c["sintetico"] / c["real"])
    assert (c["banda_inf"] < c["real"]).all() and (c["real"] < c["banda_sup"]).all()
    t2 = memorizacion(_con_regimen(panel, reg), _otras_realizaciones())
    pd.testing.assert_frame_equal(t, t2)


# --------------------------------------------------------------------------- controles discriminador


def test_discriminador_mismo_proceso_cerca_de_medio(real):
    panel, reg = real
    t = discriminador(panel, _otras_realizaciones(), regimen_real=reg)
    assert abs(_auc(t) - 0.5) < 0.12


def test_discriminador_escala_doble_separable(real):
    panel, reg = real
    media = panel.mean()
    dobles = [_con_regimen((p.drop(columns="regime") - media) * 2 + media, p["regime"])
              for p in _otras_realizaciones()]
    t = discriminador(panel, dobles, regimen_real=reg)
    assert _auc(t) > 0.95 and _auc(t, "calma") > 0.95


def test_discriminador_copias_no_suben_de_medio(real):
    """Artefacto documentado: las copias de ventanas reales del pliegue de prueba caen
    en entrenamiento con etiqueta sintetica, asi que una copia da AUC <= 0,5 (no ~1).
    El discriminador no detecta copias: eso es la memorizacion."""
    panel, reg = real
    t = discriminador(panel, _bootstrap_bloques(panel, reg), regimen_real=reg)
    assert _auc(t) < 0.6


# --------------------------------------------------------------------------- controles memorizacion


def test_memorizacion_copia_literal(real):
    panel, reg = real
    t = memorizacion(panel, _bootstrap_bloques(panel, reg), regimen_real=reg)
    for regimen in ("todos", "calma", "crisis"):
        assert _mem(t, "cociente_nn", regimen) < 0.05
        assert _mem(t, "frac_copias", regimen, "sintetico") > 0.95
        assert 0.8 < _mem(t, "dispersion", regimen) < 1.25  # copia: no encoge
    assert _mem(t, "frac_copias", "todos", "real") == 0.0


def test_memorizacion_jitter_suspende(real):
    panel, reg = real
    sd = panel.std()
    rng = np.random.default_rng(3)
    jit = []
    for p in _bootstrap_bloques(panel, reg, semilla=1):
        x = p.drop(columns="regime")
        jit.append(_con_regimen(x + rng.standard_normal(x.shape) * 0.05 * sd.to_numpy(), p["regime"]))
    t = memorizacion(panel, jit, regimen_real=reg)
    assert _mem(t, "cociente_nn") < 0.6 and _mem(t, "cociente_nn", "crisis") < 0.6
    assert _mem(t, "frac_copias", "todos", "sintetico") == 0.0


def test_memorizacion_gaussiano_independiente_aprueba(real):
    panel, reg = real
    gen = GaussianoReferencia().fit(panel, reg)
    trays = gen.sample(10, 1260, regimes=np.asarray(reg)[:1260], random_state=7)
    t = memorizacion(panel, trays, regimen_real=reg)
    assert _mem(t, "cociente_nn") > 0.9 and _mem(t, "cociente_nn", "crisis") > 0.9
    assert 0.85 < _mem(t, "dispersion") < 1.15


def test_memorizacion_ventanas_apiladas_y_submuestreo(real):
    panel, reg = real
    copias = _bootstrap_bloques(panel, reg)
    t = memorizacion(panel, copias, regimen_real=reg, largo_ventana=5, max_consultas=2000)
    nc = t.query("metrica == 'n_consultas' and regimen == 'todos'")
    assert nc["real"].iloc[0] == N - 4 and nc["sintetico"].iloc[0] == 2000
    # las ventanas que no cruzan una costura entre bloques son copias exactas
    assert _mem(t, "frac_copias", "todos", "sintetico") > 0.85


# --------------------------------------------------------------------------- datos reales


@pytest.mark.datos
def test_pista_a_real_jitter_y_bootstrap_suspenden_memorizacion() -> None:
    import yaml

    from regimenes.benchmark.cache import processed_available
    from regimenes.sinteticos import datos, persistencia

    if not processed_available("A"):
        pytest.skip("faltan data/processed (pista A)")
    if not all(persistencia.dir_trayectorias(g, "A").joinpath(persistencia.FICHERO_TRAYECTORIAS).exists()
               for g in ("jitter", "bootstrap_regimen")):
        pytest.skip("faltan las trayectorias del notebook 15 (data/sinteticos)")
    cfg = yaml.safe_load(rutas.SINTETICOS_CONFIG.read_text(encoding="utf-8"))
    panel, reg = datos.cargar_entrenamiento("A", cfg["entrenamiento"]["fin_train"])
    par = cfg["validacion"]["memorizacion"]
    suelo = cfg["validacion"]["umbrales"]["memorizacion_cociente_min"]
    for nombre in ("jitter", "bootstrap_regimen"):
        t = memorizacion(panel, persistencia.cargar_trayectorias(nombre, "A"), regimen_real=reg,
                         largo_ventana=par["largo_ventana"], exclusion=par["exclusion"])
        mod = t[t["columna"] == "modeladas"]
        for regimen in ("todos", "crisis"):
            assert _mem(mod, "cociente_nn", regimen) < suelo, (nombre, regimen)
    assert _mem(mod, "frac_copias", "todos", "sintetico") > 0.95  # bootstrap: copia literal
