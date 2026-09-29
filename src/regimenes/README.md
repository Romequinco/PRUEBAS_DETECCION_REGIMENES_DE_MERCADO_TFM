# src/regimenes/ — paquete único

Paquete instalable `regimenes` (`pip install -e .`) con todo el código del TFM: datos, features,
detectores (los 12 de la Capa 1 más la ablación D13, por familia), juez, benchmark, fusión y
sintéticos ([ADR-004](../../docs/decisions/ADR-004-unificacion.md)). Imports internos absolutos (`from regimenes.x import y`), sin tocar `sys.path`.
Todas las rutas de disco salen de `regimenes.rutas`.

| Módulo | Rol |
|---|---|
| `rutas.py` | rutas centralizadas: `configs/`, `data/{raw,processed,sinteticos}`, `results/{benchmark,fusion,detectores,sinteticos,pseudolive}`. |
| `datos/` | `catalogo.py` (`load_catalog`, lee `configs/catalog.yaml`), `descarga.py` (`download_all`), `fuentes.py` (FRED, yfinance, Stooq, GitHub…). CLI: `python -m regimenes.datos [--offline\|--force --only A,B]`. |
| `features/` | `transformaciones.py` (primitivas causales + recetario v2), `lags.py` (lags de publicación), `paneles.py` (`construir_paneles`, `alinear_*`), `causalidad.py` (test de truncado `assert_causal`). |
| `detectores/` | `base.py` (única `RegimeDetector`), `registry.py` (`DetectorSpec`, `detector_specs`) y las implementaciones por familia `f1_reglas` … `f7_deep`. |
| `evaluacion/` | **EL JUEZ**. `walk_forward.py` (`walk_forward`, `evaluate`, `EvaluationResult`, `results_table`), `metricas.py` (ventanas de eventos configurables por pista, métricas causales y detección por evento `det_*`), `ranking.py` (criterio de ranking de detección ADR-003, ranking descriptivo heredado, líneas base y nulo de azar). `from regimenes import evaluacion as ev` expone la API del juez; `ranking` se importa aparte. |
| `benchmark/` | `ejecucion.py` (paneles, gate, `run_one`/`run_job`/`run_benchmark` secuencial o paralelo `spawn`, `consolidate_run`), `cache.py` (huella por contenido, verificación de caché, `load_metrics`, `metrics_provenance`), `cli.py`. `run_benchmark(cache_only=True)` no recalcula nada. CLI: `python -m regimenes.benchmark --track A B --jobs 4`. Salida por defecto: `results/benchmark/`. |
| `fusion/` | `maquina.py`: máquina causal normal/vigilancia/confirmado; emparejamiento uno-a-uno, atribución a crisis reales, ablación, sensibilidad, invariancia al truncado, `operational_utility`, `fusion_verdict`, `period_scorecard`, `select_then_evaluate`. `__init__` re-exporta la API pública. |
| `viz/` | `figuras.py`: estilo de casa y figuras estándar; `__init__` re-exporta la API y las ventanas de eventos. |
| `informes.py` | Utilidades de los notebooks de familia: carga verificada de caché + paneles OOS + ranking ADR-003 (`cargar_resultados_familia`, falla con el comando a ejecutar), contexto de pistas, tablas de cobertura/trampas/Jaccard y figuras comunes (estados OOS sobre el S&P 500, cobertura por crisis, plano del ranking). No entra en la huella de caché. |
| `sinteticos/` | interfaz `Generador` (`fit`/`sample`/`name`), registro vacío y firmas de validación (`NotImplementedError`). |

Grafo de imports sin ciclos: `evaluacion.metricas` ← `evaluacion.walk_forward` ←
`benchmark.cache` ← `benchmark.ejecucion` ← `benchmark.cli`; `evaluacion.ranking` →
(`benchmark.cache`, `evaluacion.metricas`). Nada en `benchmark/` importa `evaluacion.ranking`.

> El *contrato* del juez (firmas de `evaluate`/`walk_forward` y de `RegimeDetector`) es el de
> [ADR-001](../../docs/decisions/ADR-001-rebase-datos.md) §4. El código original de la Capa 1 está en
> el tag `capa1-final`.
