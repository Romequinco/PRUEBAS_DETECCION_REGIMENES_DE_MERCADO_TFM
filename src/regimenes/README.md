# src/regimenes/ — paquete único (ADR-004)

Paquete instalable `regimenes` (`pip install -e .`) que unifica la Capa 1 congelada
(`capa1_exploracion/`, detectores) y el código v2 (datos, features, juez, benchmark,
fusión). Imports internos absolutos (`from regimenes.x import y`), sin tocar `sys.path`.
Todas las rutas de disco salen de `regimenes.rutas`.

| Módulo | Origen | Rol |
|---|---|---|
| `rutas.py` | nuevo (ADR-004) | rutas centralizadas: `configs/`, `data/{raw,processed,sinteticos}`, `results/{benchmark,fusion,detectores,sinteticos,pseudolive}`. |
| `datos/` | `src/ingest/` | `catalogo.py` (`load_catalog`, lee `configs/catalog.yaml`), `descarga.py` (`download_all`), `fuentes.py` (FRED, yfinance, Stooq, GitHub…). CLI: `python -m regimenes.datos [--offline\|--force --only A,B]`. |
| `features/` | `src/features.py` | `transformaciones.py` (primitivas causales + recetario v2), `lags.py` (lags de publicación), `paneles.py` (`construir_paneles`, `alinear_*`), `causalidad.py` (test de truncado `assert_causal`). |
| `detectores/` | `src/detector_base.py`, `src/detectors/registry.py`, `capa1_exploracion/detectors/` | `base.py` (única `RegimeDetector`), `registry.py` (`DetectorSpec`, `detector_specs`) y las implementaciones por familia `f1_reglas` … `f7_deep`. |
| `evaluacion/` | `src/evaluation.py` + parte de `src/benchmark.py` | **EL JUEZ**. `walk_forward.py` (`walk_forward`, `evaluate`, `EvaluationResult`, `results_table`), `metricas.py` (ventanas de eventos configurables por pista, métricas causales y detección por evento `det_*`), `ranking.py` (criterio de ranking de detección ADR-003, ranking descriptivo heredado, líneas base y nulo de azar). `from regimenes import evaluacion as ev` expone la API del antiguo `src.evaluation`; `ranking` se importa aparte. |
| `benchmark/` | `src/benchmark.py` | `ejecucion.py` (paneles, gate, `run_one`/`run_job`/`run_benchmark` secuencial o paralelo `spawn`, `consolidate_run`), `cache.py` (huella por contenido, verificación de caché, `load_metrics`, `metrics_provenance`), `cli.py`. `run_benchmark(cache_only=True)` no recalcula nada. CLI: `python -m regimenes.benchmark --track A B --jobs 4`. Salida por defecto: `results/benchmark/`. |
| `fusion/` | `src/fusion.py` | `maquina.py`: máquina causal normal/vigilancia/confirmado; emparejamiento uno-a-uno, atribución a crisis reales, ablación, sensibilidad, invariancia al truncado, `operational_utility`, `fusion_verdict`, `period_scorecard`, `select_then_evaluate`. `__init__` re-exporta la API pública. |
| `viz/` | `src/viz.py` | `figuras.py`: estilo de casa y figuras estándar; `__init__` re-exporta la API y las ventanas de eventos. |
| `informes.py` | (código repetido en los notebooks 05–11) | Utilidades de los notebooks de familia: carga verificada de caché + paneles OOS + ranking ADR-003 (`cargar_resultados_familia`, falla con el comando a ejecutar), contexto de pistas, tablas de cobertura/trampas/Jaccard y figuras comunes (estados OOS sobre el S&P 500, cobertura por crisis, plano del ranking). No entra en la huella de caché. |
| `sinteticos/` | nuevo (esqueleto) | interfaz `Generador` (`fit`/`sample`/`name`), registro vacío y firmas de validación (`NotImplementedError`). |

Grafo de imports sin ciclos: `evaluacion.metricas` ← `evaluacion.walk_forward` ←
`benchmark.cache` ← `benchmark.ejecucion` ← `benchmark.cli`; `evaluacion.ranking` →
(`benchmark.cache`, `evaluacion.metricas`). Nada en `benchmark/` importa `evaluacion.ranking`.

> La Capa 1 congelada sigue disponible en el tag `capa1-final` y el código v2 previo en
> `v2-pre-unificacion`. El *contrato* del juez (firmas de `evaluate`/`walk_forward` y de
> `RegimeDetector`) no cambia; ver `../../docs/decisions/ADR-001-rebase-datos.md` y
> `../../docs/revisiones/unificacion_contrato.md`.
