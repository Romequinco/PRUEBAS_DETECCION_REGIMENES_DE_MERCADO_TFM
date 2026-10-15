# Fase 1 — Régimen de mercado

La Fase 1 del TFM detecta en qué régimen está el mercado y entrega esa señal a los agentes como la tool
`get_regimen(fecha)` ([ADR-005](../decisions/ADR-005-reencuadre-tfm-multiagente.md) §2.1). Visión del TFM y estado de
las fases: [`tfm/README.md`](../tfm/README.md). Índice de toda la documentación: [`docs/README.md`](../README.md).
Definiciones de los términos (pistas, roles, métricas, sintéticos): [GLOSARIO](../GLOSARIO.md).

Este README reúne lo operativo de la fase: el pipeline de notebooks `00`–`20`, el orden en que se ejecutan, dónde está
cada cosa en el código y en disco, y el índice de detectores por familia.

## Documentos de esta carpeta

| Documento | Contenido |
|---|---|
| [`eda.md`](eda.md) + [`figs/`](figs/) | Informe EDA: colas gordas, clustering de volatilidad, correlación acción-bono, complejo de volatilidad, crédito y curva, el punto ciego de 2013, profundidad frente a potencia, ranking causal de features |
| [`F1_reglas.md`](F1_reglas.md) … [`F7_redes_neuronales.md`](F7_redes_neuronales.md) | Teoría de cada familia y sus detectores v2 (ver el índice de detectores más abajo) |
| [`sinteticos.md`](sinteticos.md) | Generadores sintéticos, espacio de generación, criterios de validación, niveles del veredicto y laboratorio (notebooks `15`–`18`) |

El estado del arte de los detectores y de las fuentes de datos está en [`docs/estado_del_arte/`](../estado_del_arte/README.md);
la primera vuelta de detectores (Capa 1), en [`docs/historia/`](../historia/capa1/README.md).

## Pipeline 00–20

Los notebooks `00`–`20` son el pipeline de la Fase 1; las «fases» de la tabla son sus **sub-fases** (no confundir con
las fases 1–5 del TFM).

| Sub-fase | Notebooks | Qué produce |
|---|---|---|
| **Datos** | `00_descarga` → `01_eda` | `data/raw/` + informe EDA |
| **Features** | `02_diseno_preprocesado` → `03_preprocesado` | `data/processed/pista{A,B}_{diaria,mensual}.parquet` + labels |
| **D — detectores** | `04_protocolo_evaluacion` → `12_comparativa` → `05`–`11` (una familia cada uno) | `results/benchmark/` (métricas, ranking, paneles OOS) + `results/detectores/` |
| **E — fusión** | `13_fusion_d07_d08` · `14_fusion_d02_d06` | `results/fusion/` |
| **S — sintéticos** | `15_sinteticos_generadores` → `16_sinteticos_validacion` → `17_sinteticos_laboratorio` → `18_sinteticos_aumento` | `15`: ajusta los 10 generadores por pista y guarda trayectorias con régimen conocido en `data/sinteticos/<generador>/pista<X>/` (`trayectorias.parquet` con la cadena simulada y `trayectorias_impuesto.parquet` con la secuencia impuesta común) y fichas de ajuste, historial de convergencia y tablas de sanidad (`sanidad_*.csv`) en `results/sinteticos/generadores/`. Mide sanidad, no admite generadores (eso es `16`). `16`: valida cada generador contra el tramo real de entrenamiento (fidelidad, condicionamiento, discriminador, memorización, utilidad TSTR) y emite el veredicto por niveles en `results/sinteticos/validacion/`. `17`: laboratorio de detectores con régimen conocido (simulación controlada con `gaussiano_regimen`, `var_regimen` y `garch_regimen`; CLI `python -m regimenes.sinteticos.laboratorio`) en `results/sinteticos/laboratorio/`. `18`: esqueleto (aumento) |
| **F — cierre** | `19_decision_final` → `20_pseudolive` | `19`: decisión final del detector de régimen y contrato de `get_regimen` (futura ADR-006); `20`: validación de la tool sobre datos no usados en `results/pseudolive/` (esqueletos) |

### Notebook a notebook

| Nº | Notebook | Fase | Qué hace |
|---|---|---|---|
| 00 | [`00_descarga`](../../notebooks/00_descarga.ipynb) | datos | panorámica de las 166 series, cobertura y descarga |
| 01 | [`01_eda`](../../notebooks/01_eda.ipynb) | datos | EDA maestro (12 secciones) |
| 02 | [`02_diseno_preprocesado`](../../notebooks/02_diseno_preprocesado.ipynb) | features | decisiones del preprocesado (features, frecuencias, alineación causal) |
| 03 | [`03_preprocesado`](../../notebooks/03_preprocesado.ipynb) | features | paneles causales `pista{A,B}_{diaria,mensual}` + labels, con gates |
| 04 | [`04_protocolo_evaluacion`](../../notebooks/04_protocolo_evaluacion.ipynb) | D | protocolo walk-forward, gate/preflight, caché por huella, CLI paralela |
| 05 | [`05_familia_F1_reglas`](../../notebooks/05_familia_F1_reglas.ipynb) | D | D01 umbral de volatilidad (VIX solo en B) · D02 riesgo compuesto · D10 turbulencia Mahalanobis |
| 06 | [`06_familia_F2_clustering`](../../notebooks/06_familia_F2_clustering.ipynb) | D | D03 GMM · D09 Statistical Jump Model |
| 07 | [`07_familia_F3_hmm`](../../notebooks/07_familia_F3_hmm.ipynb) | D | D04 HMM gaussiano · D08 HMM t-Student · D13 HSMM (ablación) |
| 08 | [`08_familia_F4_markov_switching`](../../notebooks/08_familia_F4_markov_switching.ipynb) | D | D05 Markov-Switching de media y varianza (univariante; pese al nombre `markov_switching_var`, no es un VAR) |
| 09 | [`09_familia_F5_garch`](../../notebooks/09_familia_F5_garch.ipynb) | D | D06 GJR-GARCH-t · D11 MS-GARCH |
| 10 | [`10_familia_F6_changepoint`](../../notebooks/10_familia_F6_changepoint.ipynb) | D | D07 CUSUM online robusto |
| 11 | [`11_familia_F7_deep`](../../notebooks/11_familia_F7_deep.ipynb) | D | D12 autoencoder + GMM |
| 12 | [`12_comparativa`](../../notebooks/12_comparativa.ipynb) | D | scorecards, ranking ADR-003 y comparación entre familias |
| 13 | [`13_fusion_d07_d08`](../../notebooks/13_fusion_d07_d08.ipynb) | E | D7 alerta + D8 confirma |
| 14 | [`14_fusion_d02_d06`](../../notebooks/14_fusion_d02_d06.ipynb) | E | selección auditable: D2 alerta + D6 confirma |
| 15 | [`15_sinteticos_generadores`](../../notebooks/15_sinteticos_generadores.ipynb) | S | ajuste y muestreo de los 10 generadores por pista (régimen de referencia desde las ventanas de crisis, espacio de generación y re-derivación), curvas de convergencia, trayectorias con régimen conocido (cadena simulada y secuencia impuesta) y tablas de sanidad; no valida ni admite generadores (eso es el 16) |
| 16 | [`16_sinteticos_validacion`](../../notebooks/16_sinteticos_validacion.ipynb) | S | validación de cada generador contra el tramo real de entrenamiento (hechos estilizados dentro de racha, condicionamiento al régimen, discriminador real/sintético y memorización con índice íntegro; correlaciones y utilidad TSTR frente a TRTR, informativas) y veredicto por niveles (`apto_laboratorio`, `laboratorio_condicionado`, `apto_aumento`) con umbrales fijados a priori; los cambios posteriores de reglas y niveles están declarados y fechados en `configs/sinteticos.yaml` y su efecto se recalcula en el propio notebook |
| 17 | [`17_sinteticos_laboratorio`](../../notebooks/17_sinteticos_laboratorio.ipynb) | S | laboratorio de detectores con régimen conocido como **simulación controlada** (ningún generador salió apto en el 16): los 3 paramétricos (`gaussiano_regimen`, `var_regimen`, `garch_regimen`) como mundos de juguete, rejilla pre-registrada generador × duración de crisis × intensidad, mismos walk-forward y specs del benchmark, verdad-terreno exacta; hipótesis H1 recuperación, H2 degradación, H3 concordancia con ADR-003 y H4 circularidad |
| 18 | [`18_sinteticos_aumento`](../../notebooks/18_sinteticos_aumento.ipynb) | S | aumento de datos real + sintético (esqueleto; se mantiene, ADR-005) |
| 19 | [`19_decision_final`](../../notebooks/19_decision_final.ipynb) | F | decisión final del detector de régimen y contrato de `get_regimen`, futura ADR-006 (esqueleto) |
| 20 | [`20_pseudolive`](../../notebooks/20_pseudolive.ipynb) | F | validación de la tool `get_regimen` con la regla congelada sobre datos no usados (esqueleto) |

Los notebooks de familia (05–11) siguen una plantilla común: teoría de la familia, configuración
desde el registro, ejecución con `EJECUTAR = False` (lee la caché de `results/benchmark`), estados OOS
por pista con franjas de crisis, diagnóstico propio, métricas y puesto en el ranking, hallazgos de la
Capa 1 y conclusión. Sus figuras van a `results/detectores/<fk_nombre>/`.

Los notebooks v1 de la Capa 1 (uno por detector) se resumen en los notebooks de familia `05`–`11` y quedan íntegros en
el tag `capa1-final`.

### Orden de ejecución

La numeración es de *lectura* (familias antes de la comparativa), pero tras un benchmark nuevo hay que ejecutar
`12_comparativa` **antes** que `05`–`11`: es `12` quien escribe `results/benchmark/ranking*.csv`, y los notebooks de
familia lo leen (`regimenes.informes.cargar_resultados_familia`). Los comandos (`make benchmark`, `make notebooks`,
`make test`) están en [«Cómo reproducir»](../../README.md#cómo-reproducir) del README raíz.

**Qué comprueba cada notebook sobre la caché** (la huella cubre código, datos, especificación y
versiones):

- `04_protocolo_evaluacion` (por defecto solo caché) **no falla**: lista las combinaciones sin caché
  vigente e imprime el comando `python -m regimenes.benchmark ... --detector ...` para regenerarlas.
- `12_comparativa` verifica la huella; si no es vigente **no falla**: lo avisa y usa las métricas
  versionadas, comprobando su procedencia contra `manifest.json`.
- `05`–`11` verifican la huella (`run_benchmark(cache_only=True)`) **y** que el `ranking*.csv` que
  escribe `12_comparativa` corresponde a esas métricas; si no, fallan pidiendo re-ejecutar 12. Por
  eso, tras un benchmark nuevo, el orden es `04` → `12` → `05`–`11` → `13`–`14`.
- `13_fusion_d07_d08` lee los paneles OOS directamente (`load_panel`) **sin** verificar la huella, y
  `14_fusion_d02_d06` recurre al cribado versionado si faltan paneles: con una caché obsoleta no
  fallan, así que hay que asegurarse antes de que el benchmark está vigente.

`make notebooks` ejecuta 00–14 en orden numérico: sirve cuando caché y ranking ya son coherentes;
tras un benchmark nuevo, ejecuta antes `12_comparativa` a mano. Los paneles OOS
(`results/benchmark/panels/*.parquet`) no se versionan: se regeneran con el benchmark.
Los notebooks 15–17 (sintéticos) no entran en `make notebooks`: necesitan `data/sinteticos/`
(~1 GB, no versionado), que genera `15_sinteticos_generadores` con `EJECUTAR = True`; después, 15 y
16 se ejecutan a mano con `EJECUTAR = False` (16 lee además sus tablas de `results/sinteticos/validacion/`).
El laboratorio (17) se ejecuta con su CLI, reanudable y con caché por huella:
`python -m regimenes.sinteticos.laboratorio` (el notebook 17 lee `results/sinteticos/laboratorio/`).

**Huella de caché.** El benchmark (`python -m regimenes.benchmark`, módulo `regimenes.benchmark.cache`) guarda sus
resultados por huella de contenido. Entran:

- por bytes, los ficheros de los detectores oficiales (`regimenes.detectores`: `__init__`, `registry`, la base y los
  subpaquetes `f*/`, salvo D13) y del juez (`evaluacion/walk_forward.py`, `evaluacion/metricas.py`);
- por AST (sin comentarios ni docstrings), las funciones de `regimenes.benchmark` (`ejecucion`, `cache`) y de
  `evaluacion.metricas` que pueden cambiar predicciones o métricas (`run_one`, carga de paneles, detección por evento…);
- el banco congelado `configs/benchmark_spec.yaml`, ya parseado (sus comentarios no cuentan; `catalog.yaml` no entra);
- los paneles procesados y el S&P 500 crudo, por contenido;
- las versiones de Python y de las librerías numéricas (numpy, pandas, scipy, scikit-learn, hmmlearn, arch, torch,
  jumpmodels).

Cambiar cualquiera de ellos invalida la caché y obliga a re-ejecutar el benchmark completo (varias horas de cómputo).
Por eso la Fase 2 no toca esas rutas ni el entorno principal ([ADR-007](../decisions/ADR-007-fase2-ingesta-y-docs.md)).

## Estado de las sub-fases

Estado de la Fase 1 por sub-fase; el de las cinco fases del TFM está en
[`tfm/README.md`](../tfm/README.md#3-plan-de-trabajo-cinco-fases-y-su-estado). Las sub-fases 1–4 son las del re-base de
datos ([ADR-001](../decisions/ADR-001-rebase-datos.md)); las posteriores se nombran por letra (ver el
[GLOSARIO](../GLOSARIO.md#fase-no-es-siempre-la-fase-del-tfm)).

| Sub-fase | Qué | Estado |
|---|---|---|
| Capa 1 | 12 detectores v1 sobre 9 series (exploración) | ✅ congelada 2026-07-18 · [historia](../historia/capa1/README.md) |
| 1–3 | Reorganización · datos (166/174 series) · EDA + banco congelado | ✅ [ADR-001](../decisions/ADR-001-rebase-datos.md) · [ADR-002](../decisions/ADR-002-ajuste-ventanas.md) |
| 4 | Diseño + preprocesado causal v2 | ✅ notebooks 02–03 |
| D | Re-evaluar D01–D12 sobre `benchmark_spec.yaml` con lags de publicación, propagación de estado y ranking por detección | ✅ benchmark completo (24/24, caché verificada) · [ADR-003](../decisions/ADR-003-causalidad-calendario-estado-ranking.md) · notebooks 04–12 |
| — | Paquete único `regimenes` y un notebook por familia | ✅ [ADR-004](../decisions/ADR-004-unificacion.md) |
| E | Fusión alerta + confirmación | 🟡 D2+D6 seleccionado; regla final pendiente (notebooks 13–14) |
| S | Datos sintéticos: generadores, validación, laboratorio, aumento | 🟡 generadores, validación y laboratorio hechos (notebooks 15–17); aumento pendiente (18, esqueleto) · `regimenes.sinteticos` · [generadores sintéticos](sinteticos.md) |
| F | Decisión final del detector y contrato de `get_regimen` (futura ADR-006) + validación pseudolive de la tool | 🔜 notebooks 19–20 (esqueletos) |

## Índice de detectores por familia

`Fk` = familia (subpaquete `fk_*` de `src/regimenes/detectores/` y notebook de familia); `Dnn` = detector. D01–D12
forman el benchmark; D13 (`hsmm_tstudent`) es solo una ablación de D08, fuera del ranking y de la huella de caché. El
puesto y el `score_deteccion` de cada detector por pista están en `results/benchmark/ranking_v2.csv` y en el notebook
[`12_comparativa`](../../notebooks/12_comparativa.ipynb); su comentario, en la sección de resultados de cada ficha de
familia.

| Familia (teoría y detectores) | Detector | Código (`src/regimenes/detectores/`) | Notebook |
|---|---|---|---|
| [F1 reglas y umbrales](F1_reglas.md) | D01 `rule_vix_threshold` | `f1_reglas/rule_vix_threshold.py` | [`05_familia_F1_reglas`](../../notebooks/05_familia_F1_reglas.ipynb) |
| | D02 `rule_composite_riskoff` | `f1_reglas/rule_composite_riskoff.py` | ídem |
| | D10 `turbulence_mahalanobis` (multivariante) | `f1_reglas/turbulence_mahalanobis.py` | ídem |
| [F2 clustering](F2_clustering.md) | D03 `clustering_gmm` | `f2_clustering/clustering_gmm.py` | [`06_familia_F2_clustering`](../../notebooks/06_familia_F2_clustering.ipynb) |
| | D09 `jump_model` | `f2_clustering/jump_model.py` | ídem |
| [F3 HMM](F3_hmm.md) | D04 `hmm_gaussian_2s` | `f3_hmm/hmm_gaussian_2s.py` | [`07_familia_F3_hmm`](../../notebooks/07_familia_F3_hmm.ipynb) |
| | D08 `hmm_tstudent` | `f3_hmm/hmm_tstudent.py` | ídem |
| | D13 `hsmm_tstudent` (ablación) | `f3_hmm/hsmm_tstudent.py` | ídem |
| [F4 Markov-Switching](F4_markov_switching.md) | D05 `markov_switching_var` (media y varianza conmutantes; no es un VAR) | `f4_switching/markov_switching_var.py` | [`08_familia_F4_markov_switching`](../../notebooks/08_familia_F4_markov_switching.ipynb) |
| [F5 volatilidad GARCH](F5_volatilidad_garch.md) | D06 `garch_t_vol` | `f5_garch/garch_t_vol.py` | [`09_familia_F5_garch`](../../notebooks/09_familia_F5_garch.ipynb) |
| | D11 `msgarch_regime` | `f5_garch/msgarch_regime.py` | ídem |
| [F6 change-point](F6_change_point.md) | D07 `changepoint_online` | `f6_changepoint/changepoint_online.py` | [`10_familia_F6_changepoint`](../../notebooks/10_familia_F6_changepoint.ipynb) |
| [F7 redes neuronales](F7_redes_neuronales.md) | D12 `deep_ae_regime` | `f7_deep/deep_ae_regime.py` | [`11_familia_F7_deep`](../../notebooks/11_familia_F7_deep.ipynb) |

Los nombres heredados que confunden (D01 en la pista A no usa el VIX; D05 no es un VAR) están aclarados en el
[GLOSARIO](../GLOSARIO.md#nombres-de-detectores-que-confunden). Qué features recibe cada detector:
[GLOSARIO](../GLOSARIO.md#núcleo-y-features-por-detector). La narrativa original v1 de D01–D12 (lo implementado, lo
descubierto y las hipótesis de la Capa 1) está en [`historia/fichas_v1.md`](../historia/fichas_v1.md).

## Dónde está cada cosa (rutas y módulos)

Todo el código de la Fase 1 es el paquete **`regimenes`** (`src/regimenes/`, instalado con `pip install -e .`;
[ADR-004](../decisions/ADR-004-unificacion.md)). Las rutas en disco salen **siempre** de `regimenes.rutas` (ningún
módulo ni notebook calcula rutas con `Path(__file__)`):

| Constante (`regimenes.rutas`) | Ruta | Contenido |
|---|---|---|
| `ROOT` | raíz del checkout | — |
| `CONFIGS` · `CATALOG` · `BENCHMARK_SPEC` · `SINTETICOS_CONFIG` | `configs/` | `catalog.yaml`, `benchmark_spec.yaml`, `sinteticos.yaml` |
| `DATA_RAW` · `DATA_PROCESSED` · `DATA_SINTETICOS` | `data/raw/` · `data/processed/` · `data/sinteticos/` | series crudas · paneles por pista · trayectorias sintéticas |
| `RESULTS_BENCHMARK` | `results/benchmark/` | `metrics/`, `status/`, `panels/`, `manifest.json`, `ranking_v2.csv`, `metrics_master_v2.csv` |
| `RESULTS_FUSION` · `RESULTS_FUSION_D07_D08` · `RESULTS_FUSION_D02_D06` | `results/fusion/{d07_d08,d02_d06}/` | tablas de las fusiones |
| `RESULTS_DETECTORES` | `results/detectores/<fk_nombre>/` | figuras de los notebooks de familia |
| `RESULTS_SINTETICOS` · `RESULTS_PSEUDOLIVE` | `results/sinteticos/` · `results/pseudolive/` | sub-fases S y F; `results/sinteticos/generadores/` (notebook 15), `results/sinteticos/validacion/` (notebook 16: tablas largas por dimensión y pista, veredicto, ficha JSON; las PNG no se versionan) y `results/sinteticos/laboratorio/` (notebook 17: filas por trabajo, tablas H1–H4, figuras) |
| `DOCS` · `ENV_FILE` | `docs/` · `.env` | — |

| Módulo | Qué contiene |
|---|---|
| `regimenes.datos` (`catalogo`, `descarga`, `fuentes`) | descarga dirigida por catálogo · `python -m regimenes.datos` |
| `regimenes.features` (`transformaciones`, `lags`, `paneles`, `causalidad`) | primitivas causales, `LAG_PUBLICACION`, `construir_paneles`, `assert_causal` |
| `regimenes.detectores` (`base`, `registry`, `f1_reglas` … `f7_deep`) | `RegimeDetector`, `detector_specs`/`specs_table`, los 12 detectores + D13 |
| `regimenes.evaluacion` (`walk_forward`, `metricas`, `ranking`) | el juez: walk-forward, métricas, ventanas de crisis, ranking ADR-003 |
| `regimenes.benchmark` (`ejecucion`, `cache`, `cli`) | ejecución y caché por huella · `python -m regimenes.benchmark` |
| `regimenes.fusion` (`maquina`) | máquina normal / vigilancia / confirmado |
| `regimenes.viz` (`figuras`) | estilo de casa de figuras |
| `regimenes.informes` | utilidades comunes de los notebooks de familia 05–11: carga verificada de resultados (`cargar_resultados_familia`), tablas y figuras |
| `regimenes.sinteticos` (`base`, `comun`, `datos`, `espacio`, `bloques`, `persistencia`, `registry`, `parametricos/`, `neuronales/`, `validacion`, `laboratorio`) | interfaz `Generador` y base común `GeneradorBase`, régimen de referencia y cadena de regímenes, espacio de generación y re-derivación, bloques y encadenado, registro perezoso (`registry.crear`) y 10 generadores (6 paramétricos; 4 neuronales con el extra `[deep]`). `validacion/` (notebook 16) es un subpaquete con un módulo por dimensión: `fidelidad`, `discriminador`, `memorizacion`, `utilidad` y `veredicto`. `laboratorio/` (notebook 17) contiene `escenarios`, `metricas`, `ejecucion` (CLI `python -m regimenes.sinteticos.laboratorio`) y `analisis` (H1–H4) |

La historia de la Capa 1 (decisiones, hallazgos y métricas v1) está en [`historia/capa1/`](../historia/capa1/README.md);
su código, notebooks, memoria e informe originales, en el tag `capa1-final`.
