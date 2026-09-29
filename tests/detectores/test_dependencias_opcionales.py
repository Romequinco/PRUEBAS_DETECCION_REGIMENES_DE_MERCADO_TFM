"""Detectores con dependencias opcionales (extras ``[deep]`` y ``[jump]``).

En CI no se instalan ``torch`` ni ``jumpmodels``: estos tests se saltan con
``pytest.importorskip``. En local (entorno completo) comprueban que la factoria
del registro construye el detector y que la dependencia se resuelve.
"""
from __future__ import annotations

import pytest

from regimenes.detectores.registry import detector_specs


def _spec(track: str, detector_id: str):
    (spec,) = [s for s in detector_specs(track) if s.detector_id == detector_id]
    return spec


@pytest.mark.parametrize("track", ["A", "B"])
def test_d12_deep_ae_requiere_torch(track: str) -> None:
    pytest.importorskip("torch", reason="extra [deep] no instalado (D12)")
    det = _spec(track, "D12").factory()
    assert det.name.startswith("deep_ae_regime")


@pytest.mark.parametrize("track", ["A", "B"])
def test_d09_jump_model_requiere_jumpmodels(track: str) -> None:
    pytest.importorskip("jumpmodels", reason="extra [jump] no instalado (D09)")
    from jumpmodels.jump import JumpModel  # noqa: F401  import perezoso de fit()

    det = _spec(track, "D09").factory()
    assert det.name.startswith("jump_model")
