"""Contrato del esqueleto ``regimenes.sinteticos``.

Fija dos cosas mientras no exista ningun generador real:

1. La interfaz abstracta ``Generador`` (``name``/``fit``/``sample``): no se puede
   instanciar, una subclase incompleta tampoco, y una subclase minima cumple el
   contrato documentado (``fit`` devuelve ``self``; ``sample`` devuelve
   ``n_paths`` trayectorias de ``length`` pasos con columna ``regime``).
2. Los stubs (registro y validacion) levantan ``NotImplementedError`` en lugar
   de devolver resultados vacios silenciosos. Cuando se implemente uno, este
   test debe actualizarse a la vez (es deliberado: obliga a cubrirlo).

No necesita datos: corre igual en local, en ``make test-rapido`` y en CI.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
import yaml

from regimenes import rutas, sinteticos
from regimenes.sinteticos import base, neuronales, parametricos, registry, validacion
from regimenes.sinteticos.base import Generador


class _GeneradorGaussiano(Generador):
    """Generador de juguete: ruido normal por regimen (solo para el contrato)."""

    @property
    def name(self) -> str:
        return "juguete_gauss"

    def fit(self, train: pd.DataFrame, regimes: pd.Series | None = None) -> "_GeneradorGaussiano":
        self.columns_ = list(train.columns)
        self.mu_ = train.mean().to_numpy()
        self.sd_ = train.std(ddof=0).to_numpy()
        return self

    def sample(self, n_paths, length, regimes=None, *, random_state=None):
        rng = np.random.default_rng(random_state)
        reg = np.zeros(length, dtype=int) if regimes is None else np.asarray(regimes, dtype=int)
        idx = pd.bdate_range("2000-01-03", periods=length)
        out = []
        for _ in range(n_paths):
            x = self.mu_ + self.sd_ * rng.standard_normal((length, len(self.columns_)))
            df = pd.DataFrame(x, index=idx, columns=self.columns_)
            df["regime"] = reg
            out.append(df)
        return out


@pytest.fixture
def panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2010-01-04", periods=120)
    return pd.DataFrame(rng.standard_normal((120, 3)), index=idx, columns=["r", "vol", "vix"])


# --------------------------------------------------------------------------- interfaz

def test_generador_exportado_en_el_paquete() -> None:
    assert sinteticos.Generador is base.Generador
    assert "Generador" in sinteticos.__all__


def test_generador_es_abstracto_con_name_fit_sample() -> None:
    assert inspect.isabstract(Generador)
    assert Generador.__abstractmethods__ == frozenset({"name", "fit", "sample"})
    with pytest.raises(TypeError):
        Generador()  # type: ignore[abstract]


def test_subclase_incompleta_no_se_instancia() -> None:
    class SinSample(Generador):
        @property
        def name(self) -> str:
            return "sin_sample"

        def fit(self, train, regimes=None):
            return self

    with pytest.raises(TypeError):
        SinSample()  # type: ignore[abstract]


def test_firma_de_sample_tiene_random_state_keyword_only() -> None:
    params = inspect.signature(Generador.sample).parameters
    assert list(params)[:4] == ["self", "n_paths", "length", "regimes"]
    assert params["regimes"].default is None
    assert params["random_state"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["random_state"].default is None
    fit_params = inspect.signature(Generador.fit).parameters
    assert list(fit_params) == ["self", "train", "regimes"]
    assert fit_params["regimes"].default is None


def test_subclase_minima_cumple_el_contrato(panel: pd.DataFrame) -> None:
    gen = _GeneradorGaussiano()
    assert gen.fit(panel) is gen
    paths = gen.sample(3, 50, random_state=7)
    assert isinstance(paths, list) and len(paths) == 3
    for p in paths:
        assert isinstance(p, pd.DataFrame)
        assert len(p) == 50
        assert list(p.columns) == list(panel.columns) + ["regime"]
    # reproducibilidad con semilla
    again = gen.sample(3, 50, random_state=7)
    for a, b in zip(paths, again):
        pd.testing.assert_frame_equal(a, b)


def test_sample_respeta_la_secuencia_de_regimenes_impuesta(panel: pd.DataFrame) -> None:
    reg = np.r_[np.zeros(20, dtype=int), np.ones(10, dtype=int)]
    (path,) = _GeneradorGaussiano().fit(panel).sample(1, len(reg), regimes=reg, random_state=1)
    np.testing.assert_array_equal(path["regime"].to_numpy(), reg)


def test_metodos_abstractos_de_la_base_levantan_not_implemented(panel: pd.DataFrame) -> None:
    gen = _GeneradorGaussiano()
    with pytest.raises(NotImplementedError):
        Generador.name.fget(gen)  # type: ignore[attr-defined]
    with pytest.raises(NotImplementedError):
        Generador.fit(gen, panel)
    with pytest.raises(NotImplementedError):
        Generador.sample(gen, 1, 10)


# --------------------------------------------------------------------------- stubs

def test_registro_vacio_y_sin_implementar() -> None:
    assert registry.GENERADORES == {}
    with pytest.raises(NotImplementedError):
        registry.registrar(_GeneradorGaussiano)
    with pytest.raises(NotImplementedError):
        registry.crear("msvar")
    assert registry.GENERADORES == {}


def test_subpaquetes_de_familias_vacios() -> None:
    assert parametricos.__all__ == []
    assert neuronales.__all__ == []


def _metricas_configuradas() -> list[str]:
    cfg = yaml.safe_load(rutas.SINTETICOS_CONFIG.read_text(encoding="utf-8"))
    return list(cfg["validacion"]["metricas"])


def test_config_lista_las_metricas_de_validacion_existentes() -> None:
    metricas = _metricas_configuradas()
    assert metricas, "configs/sinteticos.yaml sin metricas de validacion"
    for nombre in metricas:
        assert callable(getattr(validacion, nombre, None)), nombre


@pytest.mark.parametrize("nombre", [
    "fidelidad_marginal", "fidelidad_dependencia", "fidelidad_regimenes",
    "utilidad_tstr", "memorizacion",
])
def test_metricas_de_validacion_son_stubs(nombre: str, panel: pd.DataFrame) -> None:
    fn = getattr(validacion, nombre)
    args = {
        "real": panel, "sintetico": [panel], "real_regimes": pd.Series(0, index=panel.index),
        "detector_id": "D04",
    }
    kwargs = {k: args[k] for k in inspect.signature(fn).parameters}
    with pytest.raises(NotImplementedError):
        fn(**kwargs)


def test_rutas_de_sinteticos_definidas() -> None:
    assert rutas.DATA_SINTETICOS == rutas.DATA / "sinteticos"
    assert rutas.RESULTS_SINTETICOS == rutas.RESULTS / "sinteticos"
    assert rutas.SINTETICOS_CONFIG.exists()
