"""Cimientos de ``regimenes.sinteticos``: datos, espacio, bloques, persistencia y registro.

Cubre lo que no depende de ningun generador concreto:

- aritmetica del regimen (``rachas``, ``matriz_transicion``, ``simular_cadena``);
- ``EspacioGeneracion``: identidad en paneles de juguete y re-derivacion exacta
  de las columnas deterministas del S&P 500 (sin datos locales, con una
  historia de precios sintetica);
- ``construir_bloques`` / ``encadenar``;
- parquet largo de trayectorias y ficha de resumen;
- registro perezoso (catalogo, errores claros);
- ``neuronales._torch`` (se salta sin el extra ``[deep]``).

Los tests marcados ``datos`` cargan la pista real y comprueban las cifras del
panel de ajuste y el error de re-derivacion sobre la historia REAL.
"""
from __future__ import annotations

import importlib.util
import json

import numpy as np
import pandas as pd
import pytest
import yaml
from generador_referencia import GaussianoReferencia, panel_con_sp500, panel_juguete

from regimenes import rutas
from regimenes.sinteticos import bloques, datos, persistencia, registry
from regimenes.sinteticos.comun import GeneradorBase
from regimenes.sinteticos.espacio import DERIVADAS, EspacioGeneracion, derivar_sp500

# --------------------------------------------------------------------------- regimen


def test_rachas_y_matriz_de_transicion() -> None:
    reg = np.array([0, 0, 0, 1, 1, 0, 0, 1])
    tabla = datos.rachas(reg)
    assert list(tabla.columns) == ["inicio", "fin", "regimen", "duracion"]
    assert tabla["duracion"].tolist() == [3, 2, 2, 1]
    assert tabla["regimen"].tolist() == [0, 1, 0, 1]
    assert tabla["inicio"].tolist() == [0, 3, 5, 7] and tabla["fin"].tolist() == [2, 4, 6, 7]
    P = datos.matriz_transicion(reg)
    np.testing.assert_allclose(P, [[3 / 5, 2 / 5], [1 / 2, 1 / 2]])
    np.testing.assert_allclose(P.sum(axis=1), 1.0)


def test_rachas_con_serie_devuelve_fechas() -> None:
    idx = pd.bdate_range("2020-01-01", periods=5)
    tabla = datos.rachas(pd.Series([0, 1, 1, 0, 0], index=idx))
    assert tabla["inicio"].tolist() == [idx[0], idx[1], idx[3]]
    assert tabla["fin"].tolist() == [idx[0], idx[2], idx[4]]


def test_matriz_de_transicion_fila_sin_datos_es_absorbente() -> None:
    P = datos.matriz_transicion(np.zeros(10, dtype=int), n=2)
    np.testing.assert_allclose(P, np.eye(2))
    with pytest.raises(ValueError):
        datos.matriz_transicion(np.array([0, 2]), n=2)


def test_simular_cadena_reproduce_la_matriz() -> None:
    P = np.array([[0.95, 0.05], [0.20, 0.80]])
    cadenas = datos.simular_cadena(P, 4000, np.random.default_rng(0), inicial=0, n_paths=8)
    assert cadenas.shape == (8, 4000) and cadenas.dtype.kind == "i"
    estimada = np.mean([datos.matriz_transicion(c) for c in cadenas], axis=0)
    np.testing.assert_allclose(estimada, P, atol=0.02)
    una = datos.simular_cadena(P, 50, np.random.default_rng(1), inicial=1)
    assert una.shape == (50,)
    np.testing.assert_array_equal(una, datos.simular_cadena(P, 50, np.random.default_rng(1), inicial=1))
    # continuacion: desde un estado absorbente no se sale
    assert datos.simular_cadena(np.eye(2), 30, np.random.default_rng(2), inicial=1).tolist() == [1] * 30
    with pytest.raises(ValueError):
        datos.simular_cadena(np.array([[0.5, 0.4], [0.5, 0.5]]), 5, np.random.default_rng(0))


# --------------------------------------------------------------------------- espacio


def test_espacio_identidad_en_panel_sin_columnas_del_sp500() -> None:
    panel, _ = panel_juguete(300)
    espacio = EspacioGeneracion()
    X = espacio.ajustar(panel)
    assert espacio.derivadas_ == [] and espacio.columnas_trabajo_ == list(panel.columns)
    np.testing.assert_allclose(X.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(X.std(axis=0), 1, atol=1e-12)
    vuelta = espacio.a_publico(X, panel.index)
    pd.testing.assert_frame_equal(vuelta, panel, check_exact=False, rtol=1e-12, atol=1e-12)
    indice = espacio.indice_sintetico(5)
    assert indice[0] == panel.index[-1] + pd.offsets.BDay(1) and indice.dtype == "datetime64[ns]"


def test_espacio_rederiva_las_columnas_del_sp500_sobre_la_historia() -> None:
    panel, _, precios = panel_con_sp500()
    corte = panel.index[600]
    train = panel.loc[:corte]
    espacio = EspacioGeneracion(precios)  # la historia incluye futuro: debe recortarse
    X = espacio.ajustar(train)
    assert espacio.derivadas_ == [c for c in panel.columns if c in DERIVADAS]
    assert espacio.columnas_trabajo_ == ["FF_MKT_z", "SP500_ret", "macro"]
    assert X.shape == (len(train), 3)
    assert espacio.historia_.index[-1] == corte
    assert max(espacio.error_rederivacion_.values()) == 0.0
    # ida y vuelta del tramo real posterior al corte: pasar los retornos REALES
    # como si fueran sinteticos debe reproducir el panel real posterior
    futuro = panel.loc[panel.index > corte].iloc[:120]
    vuelta = espacio.a_publico(espacio.a_trabajo(futuro), futuro.index)
    pd.testing.assert_frame_equal(vuelta, futuro, check_exact=False, rtol=1e-9, atol=1e-9)


def test_espacio_exige_retorno_crudo_y_historia_hasta_el_corte() -> None:
    panel, _, precios = panel_con_sp500()
    with pytest.raises(ValueError, match="SP500_ret"):
        EspacioGeneracion(precios).ajustar(panel.drop(columns="SP500_ret"))
    with pytest.raises(ValueError, match="fin_train"):
        EspacioGeneracion(precios.iloc[:500]).ajustar(panel)


def test_generador_con_columnas_derivadas_es_coherente_con_su_retorno() -> None:
    panel, reg, precios = panel_con_sp500()
    train, reg_train = panel.iloc[:600], reg.iloc[:600]
    gen = GaussianoReferencia(historia_sp500=precios).fit(train, reg_train)
    assert gen.columnas_trabajo_ == ["FF_MKT_z", "SP500_ret", "macro"] and gen.d_ == 3
    paths = gen.sample(3, 300, random_state=0)
    historia = precios.loc[:train.index[-1]]
    for p in paths:
        assert list(p.columns) == list(train.columns) + ["regime"]
        assert np.isfinite(p.to_numpy(dtype=float)).all()
        senda = pd.concat([historia, historia.iloc[-1] * np.exp(p["SP500_ret"].cumsum())])
        esperado = derivar_sp500(senda).loc[p.index]
        pd.testing.assert_frame_equal(p[esperado.columns], esperado, check_exact=False, rtol=1e-12)
        assert (p["SP500_drawdown"] <= 0).all()
    # el futuro de la historia no influye: misma salida con la historia recortada
    recortado = GaussianoReferencia(historia_sp500=historia).fit(train, reg_train)
    for a, b in zip(paths, recortado.sample(3, 300, random_state=0)):
        pd.testing.assert_frame_equal(a, b)


# --------------------------------------------------------------------------- base


def test_generador_base_valida_entradas() -> None:
    panel, reg = panel_juguete(200)
    gen = GaussianoReferencia()
    with pytest.raises(ValueError, match="NaN"):
        gen.fit(panel.mask(panel > 100).assign(ret=np.nan), reg)
    with pytest.raises(ValueError, match="indice"):
        gen.fit(panel, reg.iloc[:-1])
    with pytest.raises(ValueError, match="regime"):
        gen.fit(panel.assign(regime=0), reg)
    with pytest.raises(ValueError, match="DatetimeIndex"):
        gen.fit(panel.reset_index(drop=True), reg.to_numpy())
    with pytest.raises(NotImplementedError):
        GeneradorBase().name
    sin_regimen = GaussianoReferencia().fit(panel)  # regimes=None -> un solo regimen
    (p,) = sin_regimen.sample(1, 40, random_state=0)
    assert (p["regime"] == 0).all()


def test_generador_base_estado_tras_fit() -> None:
    panel, reg = panel_juguete(600)
    gen = GaussianoReferencia(largo_contexto=5).fit(panel, reg)
    assert gen.params == {"regularizacion": 1e-6, "largo_contexto": 5} and gen.largo_contexto == 5
    assert gen.contexto_.shape == (5, 4) and gen.reg_contexto_.shape == (5,)
    np.testing.assert_allclose(gen.contexto_, gen.espacio_.a_trabajo(panel.iloc[-5:]))
    np.testing.assert_allclose(gen.P_, datos.matriz_transicion(reg))
    assert gen.regimen_final_ == int(reg.iloc[-1]) and gen.n_regimenes_ == 2
    assert int((gen.rachas_["regimen"] == 1).sum()) == 3
    assert list(gen.historial.columns) == ["regimen", "n_obs"] and len(gen.historial) == 2
    ficha = gen.resumen()
    assert ficha["final_n_obs"] == int(reg.sum()) and ficha["final_regimen"] == 1
    assert ficha["pct_crisis_train"] == pytest.approx(100 * reg.mean())
    json.dumps(persistencia._serializable(ficha))


# --------------------------------------------------------------------------- bloques


def test_construir_bloques_formas_y_contenido() -> None:
    X = np.arange(40, dtype=float).reshape(20, 2)
    reg = np.arange(20) % 2
    ctx, blk, rb = bloques.construir_bloques(X, reg, largo_bloque=3, largo_contexto=4)
    assert ctx.shape == (14, 4, 2) and blk.shape == (14, 3, 2) and rb.shape == (14, 3)
    np.testing.assert_array_equal(ctx[0], X[0:4])
    np.testing.assert_array_equal(blk[0], X[4:7])
    np.testing.assert_array_equal(rb[0], reg[4:7])
    np.testing.assert_array_equal(blk[-1], X[17:20])
    ctx2, blk2, _ = bloques.construir_bloques(X, reg, 3, 4, paso=5)
    assert len(blk2) == 3
    ctx0, blk0, _ = bloques.construir_bloques(X, reg, 5, 0)
    assert ctx0.shape == (16, 0, 2) and blk0.shape == (16, 5, 2)
    with pytest.raises(ValueError):
        bloques.construir_bloques(X, reg, 30, 4)


def test_construir_bloques_no_cruza_huecos_de_fechas() -> None:
    fechas = pd.bdate_range("2020-01-01", periods=10).append(pd.bdate_range("2021-01-01", periods=10))
    X = np.arange(20, dtype=float)[:, None]
    ctx, blk, _ = bloques.construir_bloques(X, np.zeros(20, dtype=int), 2, 2, fechas=fechas)
    assert len(blk) == 14  # 7 ventanas por tramo de 10
    juntos = np.concatenate([ctx, blk], axis=1)[:, :, 0]
    assert not ((juntos.min(axis=1) < 10) & (juntos.max(axis=1) >= 10)).any()
    assert bloques.tramos_contiguos(fechas, 20) == [(0, 10), (10, 20)]


def test_encadenar_es_autoregresivo_y_recorta() -> None:
    llamadas: list[tuple] = []

    def siguiente(ctx, reg_blk, rng):
        llamadas.append((ctx.shape, reg_blk.copy()))
        base = ctx[:, -1:, :] if ctx.shape[1] else np.zeros((len(reg_blk), 1, 2))
        return base + np.arange(1, reg_blk.shape[1] + 1)[None, :, None] + 0 * reg_blk[:, :, None]

    reg = np.tile(np.r_[np.zeros(5, dtype=int), np.ones(5, dtype=int)], (3, 1))
    contexto = np.full((6, 2), 100.0)
    out = bloques.encadenar(siguiente, reg, contexto, largo_bloque=4, largo_contexto=2, rng=np.random.default_rng(0))
    assert out.shape == (3, 10, 2)
    np.testing.assert_allclose(out[0, :, 0], 100 + np.arange(1, 11))  # continua el contexto
    assert len(llamadas) == 3 and all(forma == (3, 2, 2) for forma, _ in llamadas)
    np.testing.assert_array_equal(llamadas[-1][1][0], [1, 1, 1, 1])  # ultimo bloque relleno con el ultimo regimen
    with pytest.raises(ValueError):
        bloques.encadenar(siguiente, reg, contexto[:1], 4, 2, np.random.default_rng(0))
    with pytest.raises(ValueError):
        bloques.encadenar(lambda c, r, g: np.zeros((1, 1, 1)), reg, contexto, 4, 2, np.random.default_rng(0))


# --------------------------------------------------------------------------- persistencia


def test_trayectorias_ida_y_vuelta_en_parquet(tmp_path) -> None:
    panel, reg = panel_juguete(300)
    gen = GaussianoReferencia().fit(panel, reg)
    paths = gen.sample(4, 30, random_state=0)
    destino = persistencia.guardar_trayectorias(paths, gen.name, "a", base=tmp_path)
    assert destino == tmp_path / "_referencia" / "pistaA" / "trayectorias.parquet"
    largo = pd.read_parquet(destino)
    assert list(largo.columns) == ["path_id", "date", *panel.columns, "regime"]
    assert len(largo) == 4 * 30 and sorted(largo["path_id"].unique()) == [0, 1, 2, 3]
    for a, b in zip(paths, persistencia.cargar_trayectorias(gen.name, "A", base=tmp_path)):
        pd.testing.assert_frame_equal(a, b, check_freq=False)
    with pytest.raises(FileNotFoundError):
        persistencia.cargar_trayectorias("no_existe", "A", base=tmp_path)


def test_guardar_resumen_escribe_json_e_historial(tmp_path) -> None:
    panel, reg = panel_juguete(300)
    gen = GaussianoReferencia().fit(panel, reg)
    destino = persistencia.guardar_resumen(gen, "A", base=tmp_path, n_paths=4)
    assert destino == tmp_path / "generadores" / "_referencia_pistaA_resumen.json"
    ficha = json.loads(destino.read_text(encoding="utf-8"))
    assert ficha["nombre"] == "_referencia" and ficha["pista"] == "A" and ficha["n_paths"] == 4
    historial = pd.read_csv(destino.with_name("_referencia_pistaA_historial.csv"))
    assert list(historial.columns) == ["regimen", "n_obs"]
    tabla = persistencia.cargar_resumenes(base=tmp_path)
    assert len(tabla) == 1 and tabla.loc[0, "nombre"] == "_referencia"


def test_rutas_por_defecto_cuelgan_de_regimenes_rutas() -> None:
    assert persistencia.dir_trayectorias("jitter", "b") == rutas.DATA_SINTETICOS / "jitter" / "pistaB"
    assert persistencia.dir_resumenes() == rutas.RESULTS_SINTETICOS / "generadores"


# --------------------------------------------------------------------------- registro


def test_catalogo_del_registro() -> None:
    assert list(registry.CATALOGO) == [
        "jitter", "bootstrap_regimen", "gaussiano_regimen", "var_regimen", "garch_regimen", "rbig",
        "flow_matching", "difusion", "cvae", "cgan",
    ]
    for nombre, (modulo, familia) in registry.CATALOGO.items():
        assert modulo == f"{familia}.{nombre}" and familia in {"parametricos", "neuronales"}
        assert registry.modulo_de(nombre) == f"regimenes.sinteticos.{familia}.{nombre}"
    assert set(registry.disponibles()) <= set(registry.CATALOGO) | set(registry.GENERADORES)


def test_crear_da_errores_claros() -> None:
    with pytest.raises(KeyError, match="desconocido"):
        registry.crear("no_existe")
    for nombre in registry.CATALOGO:
        if not registry.existe_modulo(nombre):
            with pytest.raises(ModuleNotFoundError, match="aun no esta implementado"):
                registry.crear(nombre)
            break


def test_config_declara_los_diez_generadores_y_cortes_validos() -> None:
    cfg = yaml.safe_load(rutas.SINTETICOS_CONFIG.read_text(encoding="utf-8"))
    assert cfg["version"] == 1 and cfg["semilla"] == 42
    assert set(cfg["generadores"]) == set(registry.CATALOGO)
    for nombre, bloque in cfg["generadores"].items():
        assert bloque["familia"] == registry.familia_de(nombre)
    assert cfg["muestreo"] == {
        "n_paths": 100, "length": 2520, "regimenes": None,
        "inicial": "estacionaria", "duraciones": "geometricas",
    }
    assert cfg["muestreo"]["inicial"] in datos.INICIALES and cfg["muestreo"]["duraciones"] in datos.DURACIONES
    # los parametros relevantes quedan explicitos y coinciden con los valores por defecto de la clase
    for nombre, bloque in cfg["generadores"].items():
        assert bloque["params"], nombre
        if registry.existe_modulo(nombre) and (
            registry.familia_de(nombre) != "neuronales" or importlib.util.find_spec("torch") is not None
        ):
            por_defecto = registry.clase(nombre).PARAMS
            for clave, valor in bloque["params"].items():
                assert por_defecto[clave] == valor, (nombre, clave)
    assert cfg["entrenamiento"]["pista"] == "A" and cfg["entrenamiento_secundario"]["pista"] == "B"
    for clave in ("entrenamiento", "entrenamiento_secundario"):
        bloque = cfg[clave]
        assert datos.episodio_cruzado(bloque["pista"], bloque["fin_train"]) is None


def test_corte_que_parte_un_episodio_se_detecta_y_se_propone_otro() -> None:
    assert datos.episodio_cruzado("A", "2008-06-30") == "gfc_2007_09"
    assert datos.corte_sin_cruce("A", "2007-10-20") == pd.Timestamp("2007-10-08")
    assert datos.corte_sin_cruce("A", "2009-03-01") == pd.Timestamp("2009-03-09")
    assert datos.corte_sin_cruce("A", "2006-12-31") == pd.Timestamp("2006-12-31")
    idx = pd.to_datetime(["2007-10-08", "2007-10-09", "2009-03-09", "2009-03-10"])
    assert datos.regimen_referencia(idx, "A").tolist() == [0, 1, 1, 0]


# --------------------------------------------------------------------------- torch


def test_ayudas_de_torch() -> None:
    torch = pytest.importorskip("torch", reason="extra [deep] no instalado (generadores neuronales)")
    from regimenes.sinteticos.neuronales import _torch as nt

    assert nt.importar_torch() is torch
    nt.sembrar(42)
    a = torch.randn(3)
    nt.sembrar(42)
    assert torch.equal(a, torch.randn(3))
    g1 = nt.generador_torch(np.random.default_rng(5))
    g2 = nt.generador_torch(np.random.default_rng(5))
    assert torch.equal(torch.randn(4, generator=g1), torch.randn(4, generator=g2))
    assert nt.a_tensor(np.ones((2, 3))).dtype == torch.float32
    assert nt.a_numpy(torch.ones(2)).dtype == np.float64
    trozos = list(nt.lotes(10, 4, np.random.default_rng(0)))
    assert [len(t) for t in trozos] == [4, 4, 2] and sorted(np.concatenate(trozos)) == list(range(10))

    # regresion lineal: la perdida baja y el historial se rellena
    nt.sembrar(0)
    x = nt.a_tensor(np.random.default_rng(0).standard_normal((256, 3)))
    y = x @ torch.tensor([1.0, -2.0, 0.5])
    red = torch.nn.Linear(3, 1)
    filas: list[dict] = []
    curva = nt.entrenar(
        lambda idx: ((red(x[idx]).squeeze(-1) - y[idx]) ** 2).mean(), red.parameters(), len(x),
        epocas=30, tam_lote=64, lr=0.05, rng=np.random.default_rng(0),
        registrar=lambda **fila: filas.append(fila),
    )
    assert len(curva) == 30 == len(filas) and curva[-1] < 0.1 * curva[0]
    assert set(filas[0]) == {"epoca", "perdida"}


# --------------------------------------------------------------------------- datos reales


def _necesita_pista(pista: str) -> None:
    from regimenes.benchmark.cache import processed_available

    if not processed_available(pista) or not any(rutas.DATA_RAW.glob("*/SP500.parquet")):
        pytest.skip(f"faltan data/processed (pista {pista}) o data/raw/<fuente>/SP500.parquet")


@pytest.mark.datos
def test_pista_a_real_panel_regimen_y_rederivacion_exacta() -> None:
    _necesita_pista("A")
    cfg = yaml.safe_load(rutas.SINTETICOS_CONFIG.read_text(encoding="utf-8"))["entrenamiento"]
    panel, reg = datos.cargar_entrenamiento(cfg["pista"], cfg["fin_train"], cfg["features"])
    assert panel.shape[1] == 10 and list(panel.columns)[-1] == "SP500_ret"
    assert list(panel.columns)[:9] == datos.nucleo_pista("A")
    assert panel.index.max() <= pd.Timestamp(cfg["fin_train"]) and panel.index.dtype == "datetime64[ns]"
    assert not panel.isna().any().any() and reg.index.equals(panel.index)
    tabla = datos.rachas(reg)
    assert int((tabla["regimen"] == 1).sum()) == 8  # 1966 ... dotcom
    assert 0.15 < reg.mean() < 0.25

    gen = GaussianoReferencia().fit(panel, reg)
    assert gen.columnas_trabajo_ == [
        "FF_MKT_z", "DGS10_change_z", "credit_BaaAaa_mensual_z", "term_spread_hist_z",
        "INDPRO_yoy_z", "SP500_ret",
    ]
    # La re-derivacion sobre la historia REAL reproduce el panel real: misma serie
    # cruda, mismas primitivas y mismo calendario -> coincidencia exacta (tolerancia
    # 1e-12 por si cambia la version de pandas).
    errores = gen.espacio_.error_rederivacion_
    assert set(errores) == {*DERIVADAS, "SP500_ret"}
    assert max(errores.values()) <= 1e-12, errores
    np.testing.assert_allclose(gen.P_, datos.matriz_transicion(reg))
    assert gen.P_[0, 0] > 0.99 and gen.P_[1, 1] > 0.99

    paths = gen.sample(3, 252, random_state=42)
    for p in paths:
        assert list(p.columns) == list(panel.columns) + ["regime"]
        assert np.isfinite(p.to_numpy(dtype=float)).all()
        assert p.index[0] > panel.index[-1]
        assert (p["SP500_drawdown"] <= 0).all()
    # continuidad con la historia real: el primer drawdown sintetico es el real
    # del ultimo dia de train compuesto con el primer retorno sintetico (o 0 si hay nuevo maximo)
    p = paths[0]
    esperado = min((1 + panel["SP500_drawdown"].iloc[-1]) * np.exp(p["SP500_ret"].iloc[0]) - 1, 0.0)
    assert p["SP500_drawdown"].iloc[0] == pytest.approx(esperado, abs=1e-12)
    # crisis mas volatil que calma tambien en el retorno crudo sintetico
    largo = pd.concat(gen.sample(4, 1000, regimes=np.r_[np.zeros(500, int), np.ones(500, int)], random_state=1))
    assert largo.loc[largo["regime"] == 1, "SP500_ret"].std() > largo.loc[largo["regime"] == 0, "SP500_ret"].std()


@pytest.mark.datos
def test_pista_b_real_corte_y_rederivacion() -> None:
    _necesita_pista("B")
    cfg = yaml.safe_load(rutas.SINTETICOS_CONFIG.read_text(encoding="utf-8"))["entrenamiento_secundario"]
    panel, reg = datos.cargar_entrenamiento(cfg["pista"], cfg["fin_train"], cfg["features"])
    assert panel.shape[1] == 15 and list(panel.columns)[:14] == datos.nucleo_pista("B")
    assert int((datos.rachas(reg)["regimen"] == 1).sum()) == 4
    gen = GaussianoReferencia().fit(panel, reg)
    assert gen.d_ == 11 and max(gen.espacio_.error_rederivacion_.values()) <= 1e-12
    with pytest.raises(ValueError, match="parte el episodio"):
        datos.cargar_entrenamiento("B", "2018-02-01")
