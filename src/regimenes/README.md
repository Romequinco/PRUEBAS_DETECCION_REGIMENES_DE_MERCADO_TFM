# src/regimenes/ — código de la Fase 1

Paquete `regimenes` (distribución `regime-aware-macro-equity`; instalación en el [README raíz](../../README.md#instalación))
con el código de la Fase 1 del TFM, el régimen de mercado ([ADR-005](../../docs/decisions/ADR-005-reencuadre-tfm-multiagente.md)).
Es el **único mapa de módulos** de la Fase 1; el pipeline de notebooks, el orden de ejecución y las rutas en disco
están en [`docs/fase1/README.md`](../../docs/fase1/README.md). Imports internos absolutos (`from regimenes.x import y`),
sin tocar `sys.path`; todas las rutas de disco salen de `regimenes.rutas`.

Los agentes de las fases 3–5 no importan de aquí: usan [`herramientas`](../herramientas/README.md)
([ADR-012](../../docs/decisions/ADR-012-estructura-repo.md) §2.1).

## Huellas de caché

Tres cálculos guardan sus resultados ligados a una huella (sha256) del código y la configuración que los producen.
Cambiar un byte de un fichero marcado invalida esa caché y obliga a re-ejecutar el cálculo (horas). La huella usa la
**ruta relativa** de cada fichero, así que moverlo o renombrarlo también la invalida.

| Marca | Huella | Qué entra |
|---|---|---|
| 🔒B | benchmark v2 (`benchmark.cache._cache_fingerprint`) | por bytes: `evaluacion/walk_forward.py`, `evaluacion/metricas.py`, `detectores/__init__.py`, `detectores/registry.py`, `detectores/base.py` y `detectores/f*/*.py` salvo `hsmm_tstudent.py`; por AST (sin comentarios ni docstrings), las funciones que fijan predicciones y métricas en `benchmark/ejecucion.py`, `benchmark/cache.py` y `evaluacion/metricas.py`; además `configs/benchmark_spec.yaml` (parseado), los paneles de `data/processed/`, el S&P 500 crudo y las versiones de las librerías numéricas |
| 🔒P | pool sin comprimir (`benchmark.pool.huella_codigo`, ADR-010) | `detectores/pool/*.py`, `benchmark/pool.py`, `features/rentabilidades.py` y `configs/bloques.yaml` |
| 🔒A | ablaciones del menú T (`benchmark.ablaciones.huella_codigo`, ADR-009) | `benchmark/ablaciones.py`, `features/bloques.py` y `configs/bloques.yaml` |

Los módulos de lectura (`*_lectura.py`, `panorama.py`, `informes.py`, `fusion/figuras_*.py`) viven aparte a
propósito: se pueden tocar sin invalidar nada.

## Módulos

| Módulo | Qué hace |
|---|---|
| `__init__.py` | Cabecera del paquete: lista de subpaquetes y `__version__`; no importa ninguno. |
| `rutas.py` | Rutas centralizadas: `configs/`, `data/{raw,processed}`, `results/{benchmark,fusion,detectores,pseudolive}` y sus subcarpetas. Ningún otro módulo calcula rutas. |
| **`datos/`** | |
| `datos/catalogo.py` | `load_catalog`: lee el universo de series y el catálogo de crisis de `configs/catalog.yaml`. |
| `datos/descarga.py` | `download_all`: descarga cada serie del catálogo por su fuente, sin imputar, a `data/raw/` con su procedencia. |
| `datos/fuentes.py` | Descargadores por fuente (FRED, yfinance, OFR, GitHub, académicos…); cada uno devuelve una `pd.Series`. |
| `datos/__main__.py` | CLI `python -m regimenes.datos [--offline\|--force --only A,B]`. |
| **`features/`** | |
| `features/transformaciones.py` | Primitivas causales (z-scores *expanding*/*rolling*) y recetario de features v2. |
| `features/lags.py` | Lags de publicación por serie (`LAG_PUBLICACION`, ADR-003). |
| `features/paneles.py` | `construir_paneles`, `alinear_*`: paneles por pista que escribe el notebook 03. |
| `features/causalidad.py` | Test de truncado `assert_causal`. |
| `features/bloques.py` 🔒A | Termómetros por bloque: una PCA causal por bloque de `configs/bloques.yaml` (ADR-008). |
| `features/rentabilidades.py` 🔒P | Menú R: rentabilidades log diarias de 22 activos de la pista B, sin z-score (ADR-010 §2.2). |
| **`detectores/`** | |
| `detectores/__init__.py` 🔒B | Re-exporta la interfaz y el registro. |
| `detectores/base.py` 🔒B | Interfaz única `RegimeDetector` y canonicalización de estados. |
| `detectores/registry.py` 🔒B | `DetectorSpec`, `detector_specs`, `specs_table`: configuración a priori de los 12 detectores por pista (menú N). |
| `detectores/f1_reglas/` 🔒B | D01 `rule_vix_threshold`, D02 `rule_composite_riskoff`, D10 `turbulence_mahalanobis`. |
| `detectores/f2_clustering/` 🔒B | D03 `clustering_gmm`, D09 `jump_model`. |
| `detectores/f3_hmm/` 🔒B | D04 `hmm_gaussian_2s`, D08 `hmm_tstudent` y sus utilidades de filtrado causal (`_hmm_utils`, `_hmm_t_utils`); D13 `hsmm_tstudent` (ablación de D08) está fuera de la huella. |
| `detectores/f4_switching/` 🔒B | D05 `markov_switching_var` (media y varianza conmutantes; no es un VAR). |
| `detectores/f5_garch/` 🔒B | D06 `garch_t_vol` (GJR-GARCH-t), D11 `msgarch_regime`. |
| `detectores/f6_changepoint/` 🔒B | D07 `changepoint_online` (CUSUM online). |
| `detectores/f7_deep/` 🔒B | D12 `deep_ae_regime` (autoencoder + GMM; extra `[deep]`). |
| `detectores/menus.py` | Variantes detector × menú de datos U, N y T (ADR-008 §3); fuera de la huella. |
| `detectores/pool/` 🔒P | Detectores sobre el pool sin comprimir (ADR-010). No empieza por `f` para quedar fuera de la huella del benchmark. |
| `detectores/pool/especificaciones.py` 🔒P | `SpecPool` y `specs_pool`: catálogo de variantes de la Fase 2 sin tocar `registry.py`. |
| `detectores/pool/disponibilidad.py` 🔒P | Regla causal de qué columnas de los menús P y R entran en cada reajuste. |
| `detectores/pool/pool_crudo.py` 🔒P | Variantes no supervisadas sobre los menús P y R (D10-P, D10-R, D14-R, D18-P). |
| `detectores/pool/absorption_ratio.py` 🔒P | D14: absorption ratio de Kritzman et al. (2011), menú R. |
| `detectores/pool/sparse_jump.py` 🔒P | D18: sparse jump model de Nystrup, Kolm y Lindström (2021), menú P. |
| `detectores/pool/shu.py` 🔒P | D15: jump model de Shu, Yu y Mulvey (2024) con λ elegida sin etiquetas. |
| `detectores/pool/supervisados.py` 🔒P | D16 (logit L1) y D17 (random forest) con etiquetas causales (`etiquetas_causales`). |
| `detectores/pool/shu_supervisados.py` 🔒P | Especificaciones de D15, D16 y D17. |
| **`evaluacion/`** | **El juez.** `from regimenes import evaluacion as ev` expone su API; `ranking`, `mcs` y `curvas` se importan aparte. |
| `evaluacion/walk_forward.py` 🔒B | Protocolo walk-forward: `walk_forward`, `evaluate`, `EvaluationResult`, `results_table`. |
| `evaluacion/metricas.py` 🔒B | Ventanas de eventos por pista, métricas causales y detección por evento `det_*`. |
| `evaluacion/ranking.py` | Ranking de detección ADR-003 (`rank_detection`), ranking heredado, líneas base y nulo de azar. |
| `evaluacion/mcs.py` | Model Confidence Set (Hansen, Lunde y Nason, 2011) con pérdida diaria y bootstrap por bloques (ADR-008 §4). |
| `evaluacion/curvas.py` | Detección, retraso y ruido-señal a igual tasa de falsas alarmas al año (0,5, 1 y 2). |
| **`benchmark/`** | |
| `benchmark/ejecucion.py` 🔒B | Paneles, gate, `run_one`/`run_job`/`run_benchmark` (secuencial o en paralelo `spawn`), `consolidate_run`. |
| `benchmark/cache.py` 🔒B | Rutas de artefactos, huella de caché, verificación, `load_metrics` y `metrics_provenance`. |
| `benchmark/cli.py` · `__main__.py` | CLI `python -m regimenes.benchmark --track A B --jobs 4 [--cache-only]`. |
| `benchmark/variantes.py` | Evaluación de variantes detector × menú sobre un panel dado, sin caché (prueba de humo de ADR-008). |
| `benchmark/menus_oficial.py` | Ejecución oficial del menú T frente al N sobre los datos verificados, reanudable (ADR-008 §4–§6) → `results/detectores/menus/`. |
| `benchmark/ablaciones.py` 🔒A | Ablaciones y controles del menú T con reajuste anual (ADR-009) → un CSV por trabajo en `results/detectores/ablaciones/` (solo local). |
| `benchmark/ablaciones_lectura.py` | Lectura de las ablaciones: `cargar()` lee los CSV sueltos o, si no están, `trabajos.csv`; `escribir_tablas()` escribe el consolidado y las tablas derivadas. |
| `benchmark/pool.py` 🔒P | Pool sin comprimir (ADR-010): preparar, piloto de coste, ejecutar, estado y lectura → `results/detectores/pool/`. |
| `benchmark/pool_lectura.py` | Lectura del pool para los notebooks (sin ejecutar detectores ni el MCS). |
| `benchmark/panorama.py` | Tablas y figuras de todos los detectores para el notebook 14; solo lee y dibuja. |
| **`fusion/`** | `__init__` re-exporta la API de la máquina. |
| `fusion/maquina.py` | Máquina causal normal / vigilancia / confirmado (alerta + confirmación) y sus diagnósticos. |
| `fusion/vara.py` | Vara común de las fusiones: candidatos, señales, mitades y MCS de los individuales (ADR-011 §2 y §5). |
| `fusion/fijas.py` | Fusiones fijas (parejas alerta→confirmación, votación k-de-n, OR y AND) → `results/fusion/fijas/` (notebook 15). |
| `fusion/online.py` | Fusión online C1: media de probabilidades con pesos mensuales según el acierto pasado → `results/fusion/online/` (notebook 16). |
| `fusion/figuras_fijas.py` | Tablas y figuras del notebook 15; solo lee. |
| `fusion/figuras_online.py` | Tablas y figuras del notebook 16; solo lee. |
| **`regimen/`** | |
| `regimen/tool.py` | `get_regimen(fecha, *, pista=None, detector=None, contexto=True)` **provisional**: lee el panel OOS versionado (`results/benchmark/panels/{A_D06,B_D05}.parquet`, detector fijado en `configs/regimen.yaml`) y devuelve estado {calma, crisis}, `prob_crisis`, `dias_en_estado`, la fuente con su sha256 y el contexto (consenso de la fusión online y acuerdo del resto). No ajusta nada. |
| **`viz/`** | |
| `viz/figuras.py` | Estilo de casa y figuras estándar (franjas de crisis, paneles de estados); `__init__` re-exporta la API y las ventanas de eventos. |
| `viz/portada.py` | Figura de portada del README (`docs/img/portada_regimen.png`): `get_regimen` en la pista B con la probabilidad de crisis y el consenso; `python -m regimenes.viz.portada`. |
| **`informes.py`** | Utilidades de los notebooks de familia 06–12: carga verificada de caché y ranking (`cargar_resultados_familia`), tablas y figuras comunes. |
| **`sinteticos/`** | Generadores de trayectorias con régimen conocido ([generadores sintéticos](../../docs/fase1/sinteticos.md), `configs/sinteticos.yaml`). `base.py` (interfaz `Generador`: `fit`/`sample`/`name`), `comun.py` (`GeneradorBase`: validación de entradas, semillas, cadena de regímenes, diagnóstico de muestreo; cada generador solo implementa `_fit`/`_sample`), `datos.py` (panel de ajuste, régimen de referencia desde `crisis_windows`, rachas, matriz de transición, `simular_regimenes`), `espacio.py` (espacio de generación y re-derivación causal de las features del S&P 500), `bloques.py` (troceo y encadenado de «siguiente bloque»), `persistencia.py` (trayectorias en parquet largo —`trayectorias.parquet`; el notebook 17 guarda además `trayectorias_impuesto.parquet` con la secuencia impuesta—, fichas e historial), `registry.py` (catálogo perezoso: `crear`, `disponibles`). `parametricos/`: `jitter`, `bootstrap_regimen`, `gaussiano_regimen`, `var_regimen`, `garch_regimen`, `rbig`. `neuronales/` (extra `[deep]`): `flow_matching`, `difusion`, `cvae`, `cgan` (+ `_torch`, `_redes_flujo`, `_redes_latentes`). `validacion/` (notebook 18): subpaquete con un módulo por dimensión —`fidelidad` (marginales, hechos estilizados dentro de racha, correlaciones, regímenes y condicionamiento; define el contrato de tabla larga), `discriminador` (C2ST con ventanas resumidas y `HistGradientBoostingClassifier`), `memorizacion` (cociente NN con índice real íntegro), `utilidad` (TSTR frente a TRTR walk-forward y fijo), `veredicto` (umbrales, reglas y niveles de `configs/sinteticos.yaml`) y `_comun` (rachas, episodios, estandarización)—. Importar el subpaquete no importa torch ni el benchmark. `laboratorio/` (notebook 19, simulación controlada): `escenarios` (sección `laboratorio` del yaml, rejilla generador × duración × intensidad, secuencias de régimen impuestas, atenuación λ y ventanas de verdad-terreno), `metricas` (ADR-003 por trayectoria + retraso, exactitud equilibrada y falsas alarmas/año), `ejecucion` (trayectorias cacheadas con semilla por trayectoria, `walk_forward` con la spec del benchmark, filas JSON con huella, paralelo y reanudable; CLI `python -m regimenes.sinteticos.laboratorio [--piloto]`) y `analisis` (IC bootstrap e hipótesis H1–H4). Solo importa `evaluacion` y `detectores`; no los modifica. |

Grafo de imports sin ciclos: `evaluacion.metricas` ← `evaluacion.walk_forward` ← `benchmark.cache` ←
`benchmark.ejecucion` ← `benchmark.cli`; `evaluacion.ranking` → (`benchmark.cache`, `evaluacion.metricas`). Nada en
`benchmark/ejecucion.py` ni en `benchmark/cache.py` importa `evaluacion.ranking`.

> El *contrato* del juez (firmas de `evaluate`/`walk_forward` y de `RegimeDetector`) es el de
> [ADR-001](../../docs/decisions/ADR-001-rebase-datos.md) §4. El código original de la Capa 1 está en la etiqueta
> `capa1-final`.
