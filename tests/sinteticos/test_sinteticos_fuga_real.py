"""Fuga temporal con los datos REALES: nada posterior a ``fin_train`` influye en ``fit + sample``.

Marcador ``datos`` (necesita ``data/processed`` y ``data/raw/<fuente>/SP500.parquet``;
se salta si faltan). Dos comprobaciones de extremo a extremo:

1. **Futuro perturbado**. Se ajusta cada generador con el panel real y con una
   copia en la que TODO lo posterior al corte esta perturbado: el panel de la
   pista (sustituyendo la carga ``load_track_panel``) y la historia de precios
   del S&P 500. ``fit + sample`` debe ser identico bit a bit. Los parametricos
   van con sus parametros por defecto; ``rbig`` y los neuronales con
   ``PARAMS_RAPIDOS`` y pocas epocas (aqui importa la fuga, no la calidad).
2. **Re-derivacion**. Pasar los retornos REALES posteriores al corte como si
   fueran sinteticos reproduce las columnas reales del panel (tolerancia
   1e-10): la vuelta a publico usa la misma receta que el preprocesado y solo
   la historia hasta el corte.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import pytest
import yaml

from regimenes import rutas
from regimenes.sinteticos import datos, registry
from regimenes.sinteticos.espacio import DERIVADAS, EspacioGeneracion, cargar_historia_sp500

pytestmark = pytest.mark.datos

PARAMETRICOS = ["jitter", "bootstrap_regimen", "gaussiano_regimen", "var_regimen", "garch_regimen", "rbig"]
NEURONALES = ["flow_matching", "difusion", "cvae", "cgan"]
# Sobrescrituras sobre PARAMS_RAPIDOS para que el ajuste con ~11000 filas tarde segundos.
EPOCAS_FUGA = {
    "flow_matching": {"epocas": 2},
    "difusion": {"epocas": 2},
    "cvae": {"epocas": 3, "paciencia": 2, "epocas_rampa_beta": 1},
    "cgan": {"epocas": 2},
}


def _necesita_pista(pista: str) -> None:
    from regimenes.benchmark.cache import processed_available

    if not processed_available(pista) or not any(rutas.DATA_RAW.glob("*/SP500.parquet")):
        pytest.skip(f"faltan data/processed (pista {pista}) o data/raw/<fuente>/SP500.parquet")


def _config(clave: str = "entrenamiento") -> dict:
    return yaml.safe_load(rutas.SINTETICOS_CONFIG.read_text(encoding="utf-8"))[clave]


@lru_cache(maxsize=None)
def _panel_completo(pista: str) -> pd.DataFrame:
    from regimenes.benchmark.ejecucion import load_track_panel

    return load_track_panel(pista)


def _panel_perturbado(pista: str, corte: pd.Timestamp) -> pd.DataFrame:
    """Copia del panel de la pista con todo lo posterior a ``corte`` cambiado (escala x5 + ruido)."""
    panel = _panel_completo(pista).copy()
    futuro = panel.index > corte
    assert futuro.sum() > 1000
    ruido = np.random.default_rng(1).standard_normal((int(futuro.sum()), panel.shape[1]))
    panel.loc[futuro] = panel.loc[futuro].to_numpy() * 5.0 + 3.0 * ruido
    return panel


def _historia_perturbada(historia: pd.Series, corte: pd.Timestamp) -> pd.Series:
    """Copia de los precios con un paseo aleatorio multiplicativo tras ``corte``."""
    perturbada = historia.copy()
    futuro = perturbada.index > corte
    assert futuro.sum() > 1000
    paseo = np.random.default_rng(2).standard_normal(int(futuro.sum())).cumsum() * 0.05
    perturbada[futuro] = perturbada[futuro].to_numpy() * np.exp(paseo)
    return perturbada


@lru_cache(maxsize=None)
def _escenarios() -> tuple:
    """(panel, regimen, historia) reales y sus gemelos con el futuro perturbado (pista principal)."""
    import regimenes.benchmark.ejecucion as ejecucion

    cfg = _config()
    pista, corte = cfg["pista"], pd.Timestamp(cfg["fin_train"])
    panel, reg = datos.cargar_entrenamiento(pista, corte, cfg["features"])
    historia = cargar_historia_sp500()  # completa: llega hasta hoy
    assert historia.index.max() > corte
    original = ejecucion.load_track_panel
    perturbado = _panel_perturbado(pista, corte)
    try:
        ejecucion.load_track_panel = lambda _pista: perturbado
        panel2, reg2 = datos.cargar_entrenamiento(pista, corte, cfg["features"])
    finally:
        ejecucion.load_track_panel = original
    return panel, reg, historia, panel2, reg2, _historia_perturbada(historia, corte), corte


def test_la_carga_de_entrenamiento_ignora_el_futuro_del_panel(monkeypatch) -> None:
    cfg = _config()
    _necesita_pista(cfg["pista"])
    import regimenes.benchmark.ejecucion as ejecucion

    corte = pd.Timestamp(cfg["fin_train"])
    panel, reg = datos.cargar_entrenamiento(cfg["pista"], corte, cfg["features"])
    perturbado = _panel_perturbado(cfg["pista"], corte)
    assert not perturbado.loc[perturbado.index > corte].equals(
        _panel_completo(cfg["pista"]).loc[perturbado.index > corte]
    )
    monkeypatch.setattr(ejecucion, "load_track_panel", lambda _pista: perturbado)
    panel2, reg2 = datos.cargar_entrenamiento(cfg["pista"], corte, cfg["features"])
    pd.testing.assert_frame_equal(panel, panel2, check_exact=True)
    pd.testing.assert_series_equal(reg, reg2, check_exact=True)
    assert panel.index.max() <= corte


@pytest.mark.parametrize("nombre", PARAMETRICOS + NEURONALES)
def test_fit_y_sample_identicos_con_el_futuro_perturbado(nombre: str) -> None:
    _necesita_pista(_config()["pista"])
    if nombre in NEURONALES:
        pytest.importorskip("torch", reason=f"extra [deep] no instalado ({nombre})")
    cls = registry.clase(nombre)
    params = {} if nombre in PARAMETRICOS and nombre != "rbig" else dict(cls.PARAMS_RAPIDOS)
    params.update(EPOCAS_FUGA.get(nombre, {}))
    panel, reg, historia, panel2, reg2, historia2, corte = _escenarios()
    assert not historia.equals(historia2) and historia.loc[:corte].equals(historia2.loc[:corte])

    real = cls(random_state=42, historia_sp500=historia, **params).fit(panel, reg)
    gemelo = cls(random_state=42, historia_sp500=historia2, **params).fit(panel2, reg2)
    assert real.fin_train_ == gemelo.fin_train_ <= corte
    # ninguno de los dos conserva precios posteriores al corte
    for gen in (real, gemelo):
        assert gen._historia_sp500.index.max() <= corte
        assert gen.espacio_.historia_.index.max() <= corte
    for regimenes in (None, {"inicial": "estacionaria", "duraciones": "empiricas"}):
        a = real.sample(4, 300, regimenes, random_state=1)
        b = gemelo.sample(4, 300, regimenes, random_state=1)
        for x, y in zip(a, b):
            pd.testing.assert_frame_equal(x, y, check_exact=True)
            assert x.index.min() > real.fin_train_
    assert real.diagnostico_muestreo() == gemelo.diagnostico_muestreo()
    pd.testing.assert_frame_equal(real.historial, gemelo.historial, check_exact=True)


@pytest.mark.parametrize("clave", ["entrenamiento", "entrenamiento_secundario"])
def test_rederivar_con_los_retornos_reales_posteriores_reproduce_el_panel(clave: str) -> None:
    cfg = _config(clave)
    pista = cfg["pista"]
    _necesita_pista(pista)
    panel, _ = datos.cargar_entrenamiento(pista, cfg["fin_train"], cfg["features"])
    corte = panel.index[-1]
    # el espacio solo conoce la historia hasta el corte (la carga de data/raw ya la recorta)
    espacio = EspacioGeneracion()
    espacio.ajustar(panel)
    assert espacio.historia_.index.max() == corte
    assert set(espacio.derivadas_) == set(DERIVADAS) & set(panel.columns) and espacio.derivadas_

    completo = _panel_completo(pista)
    posterior = completo.loc[completo.index > corte, panel.columns].astype(float)
    posterior.index = pd.DatetimeIndex(posterior.index).astype("datetime64[ns]")
    # tramo contiguo inmediatamente posterior al corte (los retornos "sinteticos" deben ser consecutivos)
    completas = posterior.notna().all(axis=1).to_numpy()
    n = len(completas) if completas.all() else int(np.argmin(completas))
    futuro = posterior.iloc[:min(n, 2520)]
    assert len(futuro) >= 1000
    # y sobre el calendario real de precios, sin sesiones intermedias que falten en el panel
    precios = cargar_historia_sp500()
    sesiones = precios.index[(precios.index > corte) & (precios.index <= futuro.index[-1])]
    assert sesiones.equals(futuro.index)

    vuelta = espacio.a_publico(espacio.a_trabajo(futuro), futuro.index)
    assert list(vuelta.columns) == list(panel.columns)
    for col in panel.columns:
        error = float((vuelta[col] - futuro[col]).abs().max())
        assert error <= 1e-10, (col, error)
