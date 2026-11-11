"""Figura de portada del README (``regimenes.viz.portada``): se genera sin errores con los datos locales."""
from __future__ import annotations

import pytest

from regimenes import rutas


@pytest.mark.datos
@pytest.mark.skipif(not list(rutas.DATA_RAW.glob("*/SP500.parquet")), reason="falta data/raw/<fuente>/SP500.parquet")
@pytest.mark.skipif(not (rutas.RESULTS_BENCHMARK / "panels" / "B_D05.parquet").is_file(), reason="falta el panel B_D05")
def test_portada_se_genera(tmp_path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from regimenes.viz.portada import dibujar

    salida = dibujar(tmp_path / "portada.png")
    assert salida.is_file() and salida.stat().st_size > 50_000
