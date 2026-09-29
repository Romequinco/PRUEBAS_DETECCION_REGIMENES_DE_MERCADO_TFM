"""La caché versionada de ``results/benchmark`` debe seguir vigente.

Regresión: un cambio de código en un archivo hasheado por la huella de caché
(p. ej. ``evaluacion/walk_forward.py``) invalida en bloque las 24 combinaciones y
los notebooks 05-20, que solo leen la caché, dejan de ejecutarse. Si este test
falla tras tocar código del benchmark hay dos salidas: deshacer el cambio o
re-ejecutar el benchmark completo (``python -m regimenes.benchmark``) y
regenerar los notebooks.

Se omite si no hay datos procesados/caché, si falta ``data/raw/<fuente>/SP500.parquet``
(entra en la huella) o si el entorno (versiones) no es el que generó la caché: con
otras versiones la huella difiere por diseño. Lleva los marcadores ``datos`` y
``resultados`` (lee data/raw, data/processed y results/benchmark), de modo que
``make test-rapido`` lo deselecciona.
"""

from __future__ import annotations

import json
import unittest

import pytest

from regimenes import rutas
from regimenes.benchmark import cache as bm_cache


@pytest.mark.datos
@pytest.mark.resultados
class CacheVersionadaVigenteTests(unittest.TestCase):
    def test_todas_las_combinaciones_de_la_cache_estan_vigentes(self) -> None:
        manifest_path = rutas.RESULTS_BENCHMARK / "manifest.json"
        if not manifest_path.is_file() or not bm_cache.processed_available():
            self.skipTest("sin caché versionada o sin data/processed")
        try:
            bm_cache._sp500_path()
        except FileNotFoundError:
            self.skipTest("sin data/raw/<fuente>/SP500.parquet unívoco (entra en la huella)")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("runtime_versions") != bm_cache._runtime_versions():
            self.skipTest("entorno distinto del que generó la caché (la huella difiere por diseño)")

        from regimenes.benchmark.ejecucion import run_benchmark

        _, estado = run_benchmark(
            tracks=["A", "B"], output_dir=rutas.RESULTS_BENCHMARK, cache_only=True
        )
        malas = estado[estado["estado"] != "cache"]
        self.assertTrue(
            malas.empty,
            "Caché del benchmark NO vigente para: "
            + ", ".join(f"{r.pista}/{r.id}" for r in malas.itertuples()),
        )


if __name__ == "__main__":
    unittest.main()
