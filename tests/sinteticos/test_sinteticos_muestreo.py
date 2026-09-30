"""Cadena de regimenes, diagnostico del muestreo y unitarios baratos de los generadores.

Cubre lo que anade la revision del notebook 15 (sin datos locales, rapido):

- ``datos.simular_regimenes``: arranque ``ultimo`` / ``estacionaria`` y duraciones
  ``geometricas`` / ``empiricas`` (fraccion esperada de crisis, duraciones dentro
  del conjunto empirico, reproducibilidad, el defecto no cambia);
- ``GeneradorBase.sample`` con el dict de opciones y ``diagnostico_muestreo()``
  (sin mutar nada que cambie muestras posteriores);
- ``datos.momentos_por_regimen``;
- no se retiene historia del S&P 500 posterior a ``fin_train`` (tampoco en el pickle);
- VAR: recupera coeficientes conocidos y ``anclar_media`` fija el punto fijo;
- bootstrap: toda fila copiada es del regimen pedido;
- GARCH: contadores de recorte;
- torch: los hilos del proceso se restauran tras ``fit`` y ``sample``; CVAE con
  ``val_elbo`` siempre NaN (se saltan sin el extra ``[deep]``).
"""
from __future__ import annotations

import pickle
import struct

import numpy as np
import pandas as pd
import pytest
from generador_referencia import GaussianoReferencia, panel_con_sp500, panel_juguete

from regimenes.sinteticos import datos, registry

# --------------------------------------------------------------------------- cadena


def _regimen_con_rachas() -> np.ndarray:
    """Calma de 100, 300 y 50 sesiones; crisis de 20 y 40 (termina en calma)."""
    return np.r_[
        np.zeros(100, dtype=int), np.ones(20, dtype=int), np.zeros(300, dtype=int),
        np.ones(40, dtype=int), np.zeros(50, dtype=int),
    ]


def test_distribucion_estacionaria() -> None:
    P = np.array([[0.95, 0.05], [0.20, 0.80]])
    pi = datos.distribucion_estacionaria(P)
    np.testing.assert_allclose(pi, [0.8, 0.2], atol=1e-12)
    np.testing.assert_allclose(pi @ P, pi, atol=1e-12)
    # cadena reducible: decide el respaldo (un regimen nunca visto recibe 0)
    np.testing.assert_allclose(datos.distribucion_estacionaria(np.eye(2), np.array([1.0, 0.0])), [1.0, 0.0])
    np.testing.assert_allclose(datos.distribucion_estacionaria(np.eye(2)), [0.5, 0.5])


def test_simular_regimenes_por_defecto_es_la_cadena_de_markov_de_siempre() -> None:
    P = np.array([[0.95, 0.05], [0.20, 0.80]])
    nuevo = datos.simular_regimenes(7, 300, np.random.default_rng(3), P, ultimo=1)
    viejo = datos.simular_cadena(P, 300, np.random.default_rng(3), inicial=1, n_paths=7)
    np.testing.assert_array_equal(nuevo, viejo)
    with pytest.raises(ValueError, match="inicial"):
        datos.simular_regimenes(2, 10, np.random.default_rng(0), P, inicial="otro")
    with pytest.raises(ValueError, match="duraciones"):
        datos.simular_regimenes(2, 10, np.random.default_rng(0), P, duraciones="otras")
    with pytest.raises(ValueError, match="rachas"):
        datos.simular_regimenes(2, 10, np.random.default_rng(0), P, duraciones="empiricas")


def test_arranque_estacionario_da_la_fraccion_estacionaria_desde_el_primer_dia() -> None:
    P = np.array([[0.998, 0.002], [0.008, 0.992]])  # estacionaria: 20 % de crisis
    kw = {"P": P, "ultimo": 0}
    ultimo = datos.simular_regimenes(4000, 200, np.random.default_rng(0), **kw)
    estacionaria = datos.simular_regimenes(4000, 200, np.random.default_rng(0), inicial="estacionaria", **kw)
    assert estacionaria.shape == (4000, 200) and estacionaria.dtype.kind == "i"
    # arrancando en calma el horizonte corto infra-representa la crisis; el estacionario no
    assert ultimo[:, 0].mean() < 0.01 and ultimo.mean() < 0.15
    assert estacionaria[:, 0].mean() == pytest.approx(0.20, abs=0.02)
    assert estacionaria.mean() == pytest.approx(0.20, abs=0.02)
    # mas trayectorias de una sola clase (sin crisis) con el arranque en calma
    assert (ultimo == 0).all(axis=1).sum() > (estacionaria == 0).all(axis=1).sum()
    np.testing.assert_array_equal(
        estacionaria, datos.simular_regimenes(4000, 200, np.random.default_rng(0), inicial="estacionaria", **kw)
    )


def test_duraciones_empiricas_remuestrean_las_rachas_de_train() -> None:
    reg = _regimen_con_rachas()
    P, tabla = datos.matriz_transicion(reg), datos.rachas(reg)
    admitidas = {0: {100, 300, 50}, 1: {20, 40}}
    sim = datos.simular_regimenes(
        600, 1500, np.random.default_rng(1), P, tabla, inicial="estacionaria", duraciones="empiricas"
    )
    assert sim.shape == (600, 1500) and set(np.unique(sim)) == {0, 1}
    # fraccion esperada de crisis del proceso de renovacion: media crisis / (media calma + media crisis)
    assert sim.mean() == pytest.approx(30.0 / (150.0 + 30.0), abs=0.015)
    assert sim[:, 0].mean() == pytest.approx(30.0 / 180.0, abs=0.04)
    for fila in sim[:200]:
        r = datos.rachas(fila)
        interiores = r.iloc[1:-1]  # la primera es un tiempo residual y la ultima esta cortada
        for k, duraciones in admitidas.items():
            assert set(interiores.loc[interiores["regimen"] == k, "duracion"]) <= duraciones
            # ninguna racha (tampoco la primera ni la ultima) supera la mayor racha real
            assert r.loc[r["regimen"] == k, "duracion"].max() <= max(duraciones)
    # arranque en el ultimo regimen de train: todas empiezan en calma, dentro de una racha en curso
    ultimo = datos.simular_regimenes(
        300, 400, np.random.default_rng(2), P, tabla, duraciones="empiricas", ultimo=int(reg[-1])
    )
    assert (ultimo[:, 0] == 0).all()
    primera = np.array([datos.rachas(f)["duracion"].iloc[0] for f in ultimo])
    assert primera.max() <= 300 and len(np.unique(primera)) > 20  # residual uniforme, no una duracion fija
    # reproducible
    otra = datos.simular_regimenes(
        600, 1500, np.random.default_rng(1), P, tabla, inicial="estacionaria", duraciones="empiricas"
    )
    np.testing.assert_array_equal(sim, otra)


def test_duraciones_empiricas_con_un_solo_regimen_no_salen_de_el() -> None:
    reg = np.zeros(50, dtype=int)
    sim = datos.simular_regimenes(
        3, 200, np.random.default_rng(0), datos.matriz_transicion(reg), datos.rachas(reg),
        inicial="estacionaria", duraciones="empiricas", frecuencia=np.array([1.0, 0.0]),
    )
    assert (sim == 0).all()


# --------------------------------------------------------------------------- sample y diagnostico


def test_sample_acepta_opciones_de_cadena_y_no_cambia_el_defecto() -> None:
    panel, reg = panel_juguete(600)
    gen = GaussianoReferencia().fit(panel, reg)
    base = gen.sample(4, 200, random_state=5)
    for a, b in zip(base, gen.sample(4, 200, {}, random_state=5)):
        pd.testing.assert_frame_equal(a, b)
    explicito = {"inicial": "ultimo", "duraciones": "geometricas"}
    for a, b in zip(base, gen.sample(4, 200, explicito, random_state=5)):
        pd.testing.assert_frame_equal(a, b)
    # el regimen del dict es el de datos.simular_regimenes con el mismo rng
    opciones = {"inicial": "estacionaria", "duraciones": "empiricas"}
    paths = gen.sample(5, 300, opciones, random_state=9)
    esperado = datos.simular_regimenes(
        5, 300, np.random.default_rng(9), gen.P_, gen.rachas_, ultimo=gen.regimen_final_,
        frecuencia=gen.frecuencia_regimen_, **opciones,
    )
    np.testing.assert_array_equal(np.vstack([p["regime"].to_numpy() for p in paths]), esperado)
    # y la matriz ya simulada se puede imponer tal cual
    for p, fila in zip(gen.sample(5, 300, esperado, random_state=9), esperado):
        np.testing.assert_array_equal(p["regime"].to_numpy(), fila)
    with pytest.raises(ValueError, match="desconocidas"):
        gen.sample(2, 50, {"inicio": "estacionaria"})
    with pytest.raises(ValueError, match="inicial"):
        gen.sample(2, 50, {"inicial": "aleatorio"})
    with pytest.raises(TypeError, match="dict"):
        gen.sample(2, 50, "estacionaria")


def test_arranque_estacionario_en_sample_acerca_la_crisis_a_la_de_train() -> None:
    panel, reg = panel_juguete(560)  # termina en calma
    gen = GaussianoReferencia().fit(panel, reg)
    assert gen.regimen_final_ == 0
    gen.sample(800, 30, random_state=0)
    ultimo = gen.diagnostico_muestreo()
    gen.sample(800, 30, {"inicial": "estacionaria"}, random_state=0)
    estacionaria = gen.diagnostico_muestreo()
    pi = datos.distribucion_estacionaria(gen.P_)[1]
    assert ultimo["fraccion_regimen_1"] < 0.5 * pi
    assert estacionaria["fraccion_regimen_1"] == pytest.approx(pi, abs=0.05)
    assert estacionaria["inicial"] == "estacionaria" and ultimo["inicial"] == "ultimo"


def test_diagnostico_de_muestreo_campos_y_sin_efectos_en_muestras_posteriores() -> None:
    panel, reg = panel_juguete(600)
    gen = GaussianoReferencia().fit(panel, reg)
    assert gen.diagnostico_muestreo() == {} and gen.diagnostico_muestreo_ == {}
    primera = gen.sample(6, 120, random_state=11)
    diag = gen.diagnostico_muestreo()
    regs = np.vstack([p["regime"].to_numpy() for p in primera])
    assert diag["n_paths"] == 6 and diag["length"] == 120
    assert diag["fraccion_regimen_1"] == pytest.approx((regs == 1).mean())
    assert diag["fraccion_por_regimen"] == pytest.approx([(regs == 0).mean(), (regs == 1).mean()])
    assert diag["n_trayectorias_una_clase"] == int((regs == regs[:, :1]).all(axis=1).sum())
    assert diag["n_trayectorias_sin_crisis"] == int((regs == 0).all(axis=1).sum())
    assert diag["regimen_impuesto"] is False and diag["random_state"] == 11
    assert diag["inicial"] == "ultimo" and diag["duraciones"] == "geometricas"
    # es una copia: tocarla no altera el generador
    diag["n_paths"] = -1
    assert gen.diagnostico_muestreo()["n_paths"] == 6
    # se reinicia en cada llamada y refleja la ultima
    gen.sample(2, 30, regimes=np.ones(30, dtype=int), random_state=12)
    impuesto = gen.diagnostico_muestreo()
    assert impuesto["n_paths"] == 2 and impuesto["fraccion_regimen_1"] == 1.0
    assert impuesto["n_trayectorias_una_clase"] == 2 and impuesto["n_trayectorias_sin_crisis"] == 0
    assert impuesto["regimen_impuesto"] is True and impuesto["inicial"] is None
    # sample no muta nada que cambie muestras posteriores
    for a, b in zip(primera, gen.sample(6, 120, random_state=11)):
        pd.testing.assert_frame_equal(a, b)
    assert gen.diagnostico_muestreo() == {**diag, "n_paths": 6}
    # un nuevo fit lo vacia
    assert gen.fit(panel, reg).diagnostico_muestreo() == {}


def test_momentos_por_regimen_con_panel_y_con_trayectorias() -> None:
    panel, reg = panel_juguete(600)
    real = datos.momentos_por_regimen(panel, reg)
    assert list(real.columns) == ["n", "media", "desviacion"]
    assert real.index.names == ["regimen", "columna"] and len(real) == 2 * panel.shape[1]
    for k in (0, 1):
        np.testing.assert_allclose(real.loc[k, "media"].to_numpy(), panel[reg == k].mean().to_numpy())
        np.testing.assert_allclose(real.loc[k, "desviacion"].to_numpy(), panel[reg == k].std().to_numpy())
        assert (real.loc[k, "n"] == int((reg == k).sum())).all()
    paths = GaussianoReferencia().fit(panel, reg).sample(8, 400, {"inicial": "estacionaria"}, random_state=0)
    sint = datos.momentos_por_regimen(paths)
    assert sint.index.equals(real.index)
    apilado = pd.concat(paths, ignore_index=True)
    assert sint.loc[(1, "spread"), "media"] == pytest.approx(apilado.loc[apilado["regime"] == 1, "spread"].mean())
    # el gaussiano por regimen reproduce la dispersion de cada regimen
    cociente = sint["desviacion"] / real["desviacion"]
    assert cociente.between(0.8, 1.25).all(), cociente
    solo = datos.momentos_por_regimen(paths, columnas=["ret"])
    assert solo.index.get_level_values("columna").unique().tolist() == ["ret"]
    with pytest.raises(ValueError, match="regimen"):
        datos.momentos_por_regimen(panel)
    with pytest.raises(ValueError, match="longitud"):
        datos.momentos_por_regimen(panel, reg.iloc[:-1])


# --------------------------------------------------------------------------- historia futura


def test_no_se_retiene_historia_posterior_a_fin_train_ni_en_el_pickle(tmp_path) -> None:
    panel, reg, precios = panel_con_sp500()
    train, reg_train = panel.iloc[:600], reg.iloc[:600]
    corte = train.index[-1]
    # precios futuros con un valor delator que no aparece en ningun otro sitio
    delator = 987654.321
    precios = precios.copy()
    precios[precios.index > corte] = delator
    assert (precios.index > corte).sum() > 40
    gen = GaussianoReferencia(historia_sp500=precios).fit(train, reg_train)
    assert gen._historia_sp500.index.max() == corte
    assert gen.espacio_.historia_.index.max() == corte
    assert gen.espacio_._historia_entrada is None
    bruto = pickle.dumps(gen)
    assert struct.pack("<d", delator) not in bruto
    assert struct.pack("<d", float(precios.loc[corte])) in bruto  # control: la historia de train si esta
    gen.guardar(tmp_path)
    assert struct.pack("<d", delator) not in (tmp_path / "generador.pkl").read_bytes()
    copia = GaussianoReferencia.cargar(tmp_path)
    assert copia._historia_sp500.index.max() == corte
    for a, b in zip(gen.sample(2, 80, random_state=3), copia.sample(2, 80, random_state=3)):
        pd.testing.assert_frame_equal(a, b)
    # reajustar el mismo objeto con el mismo corte sigue funcionando
    for a, b in zip(gen.sample(2, 80, random_state=3), gen.fit(train, reg_train).sample(2, 80, random_state=3)):
        pd.testing.assert_frame_equal(a, b)


# --------------------------------------------------------------------------- VAR


def _var_simulado(n: int = 24000, semilla: int = 0):
    """VAR(1) con dos regimenes por bloques largos y coeficientes conocidos."""
    rng = np.random.default_rng(semilla)
    A = np.array([
        [[0.6, 0.1, 0.0], [0.0, 0.5, 0.2], [0.1, 0.0, 0.4]],
        [[0.2, -0.3, 0.1], [0.3, 0.1, 0.0], [0.0, 0.2, -0.4]],
    ])
    c = np.array([[0.0, 0.5, -0.2], [1.0, -0.5, 0.3]])
    sigma = np.array([1.0, 2.0])
    reg = (np.arange(n) // 3000) % 2
    x = np.zeros((n, 3))
    for t in range(1, n):
        k = reg[t]
        x[t] = c[k] + A[k] @ x[t - 1] + sigma[k] * rng.standard_normal(3)
    idx = pd.bdate_range("1950-01-02", periods=n)
    return pd.DataFrame(x, index=idx, columns=["a", "b", "c"]), pd.Series(reg, index=idx), A, c


def test_var_recupera_coeficientes_conocidos_y_ancla_la_media() -> None:
    panel, reg, A, c = _var_simulado()
    libre = registry.crear("var_regimen", ridge=0.0, anclar_media=False).fit(panel, reg)
    s, m = libre.espacio_.escala_, libre.espacio_.media_
    assert not libre.historial["contraido"].any() and not libre.historial["respaldo"].any()
    for k in (0, 1):
        # el generador trabaja en columnas estandarizadas: A_std = D^-1 A D, c_std = D^-1 (c + A m - m)
        A_original = libre.A_[k] * s[:, None] / s[None, :]
        np.testing.assert_allclose(A_original, A[k], atol=0.03)
        c_original = s * libre.c_[k] + m - A_original @ m
        np.testing.assert_allclose(c_original, c[k], atol=0.08)
    anclado = registry.crear("var_regimen", ridge=0.0).fit(panel, reg)
    assert anclado.params["anclar_media"] is True
    np.testing.assert_allclose(anclado.A_, libre.A_)  # anclar solo cambia la constante
    X = anclado.espacio_.a_trabajo(panel)
    for k in (0, 1):
        punto_fijo = np.linalg.solve(np.eye(3) - anclado.A_[k], anclado.c_[k])
        np.testing.assert_allclose(punto_fijo, X[reg.to_numpy() == k].mean(axis=0), atol=1e-10)
        np.testing.assert_allclose(anclado.residuos_[k].mean(axis=0), 0.0, atol=1e-10)
    assert anclado.historial["desvio_punto_fijo_max"].max() < 1e-10
    assert (anclado.historial["deriva_constante_max"] < 0.05).all()
    # con un VAR estacionario y bien identificado ambas constantes generan la misma media por regimen
    real = datos.momentos_por_regimen(panel, reg)
    vector = np.r_[np.zeros(1500, dtype=int), np.ones(1500, dtype=int)]
    for gen in (libre, anclado):
        sint = datos.momentos_por_regimen([p.iloc[50:1500] for p in gen.sample(20, 3000, vector, random_state=1)]
                                          + [p.iloc[1550:] for p in gen.sample(20, 3000, vector, random_state=1)])
        np.testing.assert_allclose(sint["media"], real["media"], atol=0.15)
        np.testing.assert_allclose(sint["desviacion"] / real["desviacion"], 1.0, atol=0.1)


# --------------------------------------------------------------------------- bootstrap


@pytest.mark.parametrize("nombre", ["bootstrap_regimen", "jitter"])
def test_bootstrap_toda_fila_copiada_es_del_regimen_pedido(nombre: str) -> None:
    panel, reg = panel_juguete(600)
    gen = registry.crear(nombre, longitud_media=10).fit(panel, reg)
    rng = np.random.default_rng(0)
    pedido = datos.simular_regimenes(30, 500, rng, gen.P_, inicial="estacionaria", frecuencia=gen.frecuencia_regimen_)
    assert (pedido == 1).any() and (pedido == 0).any()
    indices = gen.muestrear_indices(pedido, rng)
    assert indices.shape == pedido.shape and indices.min() >= 0 and indices.max() < len(panel)
    np.testing.assert_array_equal(gen.reg_train_[indices], pedido)
    # dentro de un bloque las filas son consecutivas en train y no cruzan el final de una racha real
    seguidas = indices[:, 1:] == indices[:, :-1] + 1
    assert 0.5 < seguidas.mean() < 1.0
    assert not gen.fin_racha_[indices[:, :-1]][seguidas].any()
    assert (pedido[:, 1:] == pedido[:, :-1])[seguidas].all()
    if nombre == "bootstrap_regimen":  # sin ruido, la salida es literalmente la fila de train
        salida = gen._sample(pedido, np.random.default_rng(0), gen.contexto_)
        np.testing.assert_array_equal(salida, gen.X_[gen.muestrear_indices(pedido, np.random.default_rng(0))])
    with pytest.raises(ValueError, match="regimen"):
        registry.crear(nombre).fit(panel).muestrear_indices(np.ones((1, 5), dtype=int), rng)


# --------------------------------------------------------------------------- GARCH


def test_garch_cuenta_los_recortes_en_el_diagnostico() -> None:
    panel, reg = panel_juguete(600)
    gen = registry.crear("garch_regimen").fit(panel, reg)
    assert gen.h_max_.shape == (2,) and (gen.h_max_ > 0).all()
    assert gen.params["cuantil_techo"] == 0.99 and gen.params["techo_varianza"] == 1.0
    # el techo es el cuantil de la varianza filtrada del regimen: deja fuera ~1 % de los dias de train
    assert (gen.historial["frac_train_sobre_techo"] <= 0.02).all()
    assert (gen.historial["techo_h"] <= gen.historial["h_filtrada_max"]).all()
    gen.sample(40, 300, {"inicial": "estacionaria"}, random_state=0)
    diag = gen.diagnostico_muestreo()
    claves = ("fraccion_recorte_varianza", "fraccion_recorte_mercado", "fraccion_recorte_nivel")
    for clave in claves:
        assert 0.0 <= diag[clave] <= 1.0, clave
    assert len(diag["fraccion_recorte_varianza_regimen"]) == 2 and diag["n_paths"] == 40
    assert diag["techo_h"] == pytest.approx(list(gen.h_max_))
    assert diag["fraccion_recorte_varianza"] < 0.2
    # con el maximo como referencia el techo actua menos; con un techo minusculo, casi siempre
    maximo = registry.crear("garch_regimen", cuantil_techo=1.0).fit(panel, reg)
    assert (maximo.h_max_ >= gen.h_max_).all()
    maximo.sample(40, 300, {"inicial": "estacionaria"}, random_state=0)
    assert maximo.diagnostico_muestreo()["fraccion_recorte_varianza"] <= diag["fraccion_recorte_varianza"]
    bajo = registry.crear("garch_regimen", techo_varianza=1e-3).fit(panel, reg)
    paths = bajo.sample(10, 200, {"inicial": "estacionaria"}, random_state=0)
    assert bajo.diagnostico_muestreo()["fraccion_recorte_varianza"] > 0.9
    # y el techo es efectivo: la volatilidad generada se hunde
    assert pd.concat(paths)["ret"].std() < 0.2 * panel["ret"].std()
    with pytest.raises(ValueError, match="cuantil_techo"):
        registry.crear("garch_regimen", cuantil_techo=0.0).fit(panel, reg)


# --------------------------------------------------------------------------- torch


NEURONALES = ["flow_matching", "difusion", "cvae", "cgan"]


def test_hilos_torch_restaura_y_sembrar_no_toca_los_hilos() -> None:
    torch = pytest.importorskip("torch", reason="extra [deep] no instalado (generadores neuronales)")
    from regimenes.sinteticos.neuronales import _torch as nt

    previos = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        if torch.get_num_threads() != 2:
            pytest.skip("este torch no permite cambiar los hilos")
        nt.sembrar(0)
        assert torch.get_num_threads() == 2
        with nt.hilos_torch(1):
            assert torch.get_num_threads() == 1
        assert torch.get_num_threads() == 2
        with pytest.raises(RuntimeError, match="dentro"):
            with nt.hilos_torch(1):
                raise RuntimeError("dentro")
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(previos)


@pytest.mark.parametrize("nombre", NEURONALES)
def test_neuronales_un_hilo_durante_fit_y_sample_y_restauran(nombre: str, monkeypatch) -> None:
    torch = pytest.importorskip("torch", reason="extra [deep] no instalado (generadores neuronales)")
    from regimenes.sinteticos.neuronales._torch import GeneradorNeuronal

    cls = registry.clase(nombre)
    assert issubclass(cls, GeneradorNeuronal)
    vistos: dict[str, int] = {}
    fit_original, sample_original = cls._fit, cls._sample

    def fit_espia(self, X, reg, fechas):
        vistos["fit"] = torch.get_num_threads()
        return fit_original(self, X, reg, fechas)

    def sample_espia(self, reg, rng, contexto):
        vistos["sample"] = torch.get_num_threads()
        return sample_original(self, reg, rng, contexto)

    monkeypatch.setattr(cls, "_fit", fit_espia)
    monkeypatch.setattr(cls, "_sample", sample_espia)
    panel, reg = panel_juguete(400)
    previos = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        if torch.get_num_threads() != 2:
            pytest.skip("este torch no permite cambiar los hilos")
        gen = cls(random_state=42, **{**cls.PARAMS_RAPIDOS, "epocas": 2}).fit(panel, reg)
        assert torch.get_num_threads() == 2
        a = gen.sample(2, 50, random_state=1)
        assert torch.get_num_threads() == 2
        assert vistos == {"fit": 1, "sample": 1}
        # con otro numero de hilos en el proceso la muestra es la misma bit a bit
        torch.set_num_threads(1)
        for x, y in zip(a, gen.sample(2, 50, random_state=1)):
            pd.testing.assert_frame_equal(x, y, check_exact=True)
    finally:
        torch.set_num_threads(previos)


def test_cvae_con_val_elbo_siempre_nan_usa_la_ultima_epoca(monkeypatch) -> None:
    pytest.importorskip("torch", reason="extra [deep] no instalado (generadores neuronales)")
    from regimenes.sinteticos.neuronales import cvae

    original = cvae.RedCVAE.elbo

    def elbo_nan_en_validacion(self, x, c, ruido):
        nll, kl, lv = original(self, x, c, ruido)
        return (nll if self.training else nll * float("nan")), kl, lv

    monkeypatch.setattr(cvae.RedCVAE, "elbo", elbo_nan_en_validacion)
    panel, reg = panel_juguete(600)
    params = {**cvae.CVAE.PARAMS_RAPIDOS, "epocas": 6, "paciencia": 3, "epocas_rampa_beta": 2}
    gen = cvae.CVAE(random_state=0, **params)
    with pytest.warns(RuntimeWarning, match="val_elbo"):
        gen.fit(panel, reg)
    fase1 = gen.historial[gen.historial["fase"] == 1]
    assert fase1["val_elbo"].isna().all() and len(fase1) >= 1
    assert gen.val_elbo_invalido_ is True
    assert gen.mejor_epoca_ == int(fase1["epoca"].iloc[-1])
    ficha = gen.resumen()
    assert ficha["val_elbo_invalido"] is True and ficha["mejor_epoca"] == gen.mejor_epoca_
    # la fase 2 reentrena ese numero de epocas y el generador muestrea con normalidad
    assert len(gen.historial[gen.historial["fase"] == 2]) == gen.mejor_epoca_ + 1
    (p,) = gen.sample(1, 40, random_state=0)
    assert np.isfinite(p.to_numpy(dtype=float)).all()
    # sin el fallo, la ficha dice que la validacion fue valida
    monkeypatch.setattr(cvae.RedCVAE, "elbo", original)
    sano = cvae.CVAE(random_state=0, **params).fit(panel, reg)
    assert sano.val_elbo_invalido_ is False and sano.resumen()["val_elbo_invalido"] is False
