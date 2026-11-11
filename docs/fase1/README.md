# Fase 1 — Régimen de mercado

La Fase 1 del TFM detecta en qué régimen está el mercado y entrega esa señal a los agentes como la tool
`get_regimen(fecha)` ([ADR-005](../decisions/ADR-005-reencuadre-tfm-multiagente.md) §2.1). Visión del TFM y estado de
las fases: [`tfm/README.md`](../tfm/README.md). Índice de toda la documentación: [`docs/README.md`](../README.md).
Definiciones de los términos (pistas, roles, métricas, sintéticos): [GLOSARIO](../GLOSARIO.md).

Este README reúne lo operativo de la fase: el pipeline de notebooks `00`–`22` (en `notebooks/fase1/`), el orden en que se ejecutan, la huella
de caché, las rutas en disco y el índice de detectores por familia. El mapa de módulos del código está en
[`src/regimenes/README.md`](../../src/regimenes/README.md); la instalación, en el [README raíz](../../README.md#instalación).

## Documentos de esta carpeta

| Documento | Contenido |
|---|---|
| [`eda.md`](eda.md) + [`figs/`](figs/) | Informe EDA: colas gordas, clustering de volatilidad, correlación acción-bono, complejo de volatilidad, crédito y curva, el punto ciego de 2013, profundidad frente a potencia, ranking causal de features |
| [`F1_reglas.md`](F1_reglas.md) … [`F7_redes_neuronales.md`](F7_redes_neuronales.md) | Teoría de cada familia y sus detectores v2 (ver el índice de detectores más abajo) |
| [`sinteticos.md`](sinteticos.md) | Generadores sintéticos, espacio de generación, criterios de validación, niveles del veredicto y laboratorio (notebooks `17`–`20`). |

El estado del arte de los detectores y de las fuentes de datos está en [`docs/estado_del_arte/`](../estado_del_arte/README.md);
la primera vuelta de detectores (Capa 1), en [`docs/historia/`](../historia/capa1/README.md).

## Pipeline 00–22

Los notebooks `00`–`22` son el pipeline de la Fase 1; las «fases» de la tabla son sus **sub-fases** (no confundir con
las fases 1–5 del TFM).

| Sub-fase | Notebooks | Qué produce |
|---|---|---|
| **Datos** | `00_descarga` → `01_eda` | `data/raw/` + informe EDA |
| **Features** | `02_diseno_preprocesado` → `03_preprocesado` → `04_termometros` | `data/processed/pista{A,B}_{diaria,mensual}.parquet` + labels; `pista{A,B}_termometros.parquet` (menú T, ADR-008) |
| **D — detectores** | `05_protocolo_evaluacion` → `14_comparativa` → `06`–`12` (una familia cada uno) · `13_familia_F8_supervisado` (esqueleto) | `results/benchmark/` (métricas, ranking, paneles OOS) + `results/benchmark/vara/` + `results/detectores/` (menús N y T en `menus/`, ablaciones en `ablaciones/`, pool sin comprimir en `pool/`) |
| **E — fusión** | `15_fusiones_fijas` · `16_fusion_online` | `results/fusion/fijas/` · `results/fusion/online/` (antecedentes en `d07_d08/` y `d02_d06/`) |
| **S — sintéticos** | `17_sinteticos_generadores` → `18_sinteticos_validacion` → `19_sinteticos_laboratorio` → `20_sinteticos_aumento` | Generadores, validación por niveles, laboratorio con régimen conocido y aumento real + sintético (no se adopta). Método en [`sinteticos.md`](sinteticos.md) |
| **F — cierre** | `21_decision_final` → `22_pseudolive` | `21`: decisión final del detector de régimen y contrato de `get_regimen` (futura ADR-006); `22`: validación de la tool sobre datos no usados en `results/pseudolive/` (esqueletos) |

### Notebook a notebook

| Nº | Notebook | Fase | Qué hace |
|---|---|---|---|
| 00 | [`00_descarga`](../../notebooks/fase1/00_descarga.ipynb) | datos | panorámica de las 166 series, cobertura y descarga |
| 01 | [`01_eda`](../../notebooks/fase1/01_eda.ipynb) | datos | EDA maestro (12 secciones) |
| 02 | [`02_diseno_preprocesado`](../../notebooks/fase1/02_diseno_preprocesado.ipynb) | features | decisiones del preprocesado (features, frecuencias, alineación causal) |
| 03 | [`03_preprocesado`](../../notebooks/fase1/03_preprocesado.ipynb) | features | paneles causales `pista{A,B}_{diaria,mensual}` + labels, con gates |
| 04 | [`04_termometros`](../../notebooks/fase1/04_termometros.ipynb) | features | termómetros por bloque (PCA causal por bloque, ADR-008): construcción sobre los datos verificados, validación 1–3 sin etiquetas de crisis, relevancia frente a crisis y trampas (4) y menú N frente a T |
| 05 | [`05_protocolo_evaluacion`](../../notebooks/fase1/05_protocolo_evaluacion.ipynb) | D | protocolo walk-forward, gate/preflight, caché por huella, CLI paralela · vara de medir ADR-008 (§9–12): `VOL_P90`, MCS diario, curvas a igual FA/año y tabla CP1 → `results/benchmark/vara/` |
| 06 | [`06_familia_F1_reglas`](../../notebooks/fase1/06_familia_F1_reglas.ipynb) | D | D01 umbral de volatilidad (VIX solo en B) · D02 riesgo compuesto · D10 turbulencia Mahalanobis |
| 07 | [`07_familia_F2_clustering`](../../notebooks/fase1/07_familia_F2_clustering.ipynb) | D | D03 GMM · D09 Statistical Jump Model |
| 08 | [`08_familia_F3_hmm`](../../notebooks/fase1/08_familia_F3_hmm.ipynb) | D | D04 HMM gaussiano · D08 HMM t-Student · D13 HSMM (ablación) |
| 09 | [`09_familia_F4_markov_switching`](../../notebooks/fase1/09_familia_F4_markov_switching.ipynb) | D | D05 Markov-Switching de media y varianza (univariante; pese al nombre `markov_switching_var`, no es un VAR) |
| 10 | [`10_familia_F5_garch`](../../notebooks/fase1/10_familia_F5_garch.ipynb) | D | D06 GJR-GARCH-t · D11 MS-GARCH |
| 11 | [`11_familia_F6_changepoint`](../../notebooks/fase1/11_familia_F6_changepoint.ipynb) | D | D07 CUSUM online robusto |
| 12 | [`12_familia_F7_deep`](../../notebooks/fase1/12_familia_F7_deep.ipynb) | D | D12 autoencoder + GMM |
| 13 | [`13_familia_F8_supervisado`](../../notebooks/fase1/13_familia_F8_supervisado.ipynb) | D | D16 logit L1 · D17 random forest, entrenados solo con crisis terminadas; categoría supervisada aparte del ranking no supervisado (esqueleto) |
| 14 | [`14_comparativa`](../../notebooks/fase1/14_comparativa.ipynb) | D | scorecards, ranking ADR-003 y comparación entre familias |
| 15 | [`15_fusiones_fijas`](../../notebooks/fase1/15_fusiones_fijas.ipynb) | E | fusiones fijas del [ADR-011](../decisions/ADR-011-preregistro-fusiones.md): antecedentes D7+D8 y D2+D6 y las tres familias (alerta→confirmación, votación k-de-n, OR/AND), elegidas en H1 y evaluadas en H2 con la vara de los individuales |
| 16 | [`16_fusion_online`](../../notebooks/fase1/16_fusion_online.ipynb) | E | C1: agregación online de los detectores con pesos mensuales según acierto pasado, frente a la mejor fusión fija F\*, la media simple, la votación por mayoría y el mejor individual (ADR-011) |
| 17 | [`17_sinteticos_generadores`](../../notebooks/fase1/17_sinteticos_generadores.ipynb) | S | ajuste y muestreo de los 10 generadores por pista (régimen de referencia desde las ventanas de crisis, espacio de generación y re-derivación), curvas de convergencia, trayectorias con régimen conocido (cadena simulada y secuencia impuesta) y tablas de sanidad; no valida ni admite generadores (eso es el 18) |
| 18 | [`18_sinteticos_validacion`](../../notebooks/fase1/18_sinteticos_validacion.ipynb) | S | validación de cada generador contra el tramo real de entrenamiento (hechos estilizados dentro de racha, condicionamiento al régimen, discriminador real/sintético y memorización con índice íntegro; correlaciones y utilidad TSTR frente a TRTR, informativas) y veredicto por niveles (`apto_laboratorio`, `laboratorio_condicionado`, `apto_aumento`) con umbrales fijados a priori; los cambios posteriores de reglas y niveles están declarados y fechados en `configs/sinteticos.yaml` y su efecto se recalcula en el propio notebook |
| 19 | [`19_sinteticos_laboratorio`](../../notebooks/fase1/19_sinteticos_laboratorio.ipynb) | S | laboratorio de detectores con régimen conocido como **simulación controlada** (ningún generador salió apto en el 18): los 3 paramétricos (`gaussiano_regimen`, `var_regimen`, `garch_regimen`) como mundos de juguete, rejilla pre-registrada generador × duración de crisis × intensidad, mismos walk-forward y specs del benchmark, verdad-terreno exacta; hipótesis H1 recuperación, H2 degradación, H3 concordancia con ADR-003 y H4 circularidad |
| 20 | [`20_sinteticos_aumento`](../../notebooks/fase1/20_sinteticos_aumento.ipynb) | S | aumento de datos real + sintético (se mantiene, ADR-005): pre-registro, `regimenes.sinteticos.aumento`, cálculo completo (1.166 trabajos, 12 sesiones) y lectura: el aumento no se adopta |
| 21 | [`21_decision_final`](../../notebooks/fase1/21_decision_final.ipynb) | F | decisión final del detector de régimen y contrato de `get_regimen`, futura ADR-006 (esqueleto) |
| 22 | [`22_pseudolive`](../../notebooks/fase1/22_pseudolive.ipynb) | F | validación de la tool `get_regimen` con la regla congelada sobre datos no usados (esqueleto) |

Los notebooks de familia (06–12) siguen una plantilla común: teoría de la familia, configuración
desde el registro, ejecución con `EJECUTAR = False` (lee la caché de `results/benchmark`), estados OOS
por pista con franjas de crisis, diagnóstico propio, métricas y puesto en el ranking, hallazgos de la
Capa 1 y conclusión. Sus figuras van a `results/detectores/<fk_nombre>/`.

Los notebooks v1 de la Capa 1 (uno por detector) se resumen en los notebooks de familia `06`–`12` y quedan íntegros en
el tag `capa1-final`.

### Orden de ejecución

La numeración es de *lectura* (familias antes de la comparativa), pero tras un benchmark nuevo hay que ejecutar
`14_comparativa` **antes** que `06`–`12`: es `14` quien escribe `results/benchmark/ranking*.csv`, y los notebooks de
familia lo leen (`regimenes.informes.cargar_resultados_familia`). Los comandos (`make benchmark`, `make notebooks`,
`make test`) están en [«Cómo reproducir»](../../README.md#cómo-reproducir) del README raíz.

**Qué comprueba cada notebook sobre la caché** (la huella cubre código, datos, especificación y
versiones):

- `05_protocolo_evaluacion` (por defecto solo caché) **no falla**: lista las combinaciones sin caché
  vigente e imprime el comando `python -m regimenes.benchmark ... --detector ...` para regenerarlas.
- `14_comparativa` verifica la huella; si no es vigente **no falla**: lo avisa y usa las métricas
  versionadas, comprobando su procedencia contra `manifest.json`.
- `06`–`12` verifican la huella (`run_benchmark(cache_only=True)`) **y** que el `ranking*.csv` que
  escribe `14_comparativa` corresponde a esas métricas; si no, fallan pidiendo re-ejecutar 14. Por
  eso, tras un benchmark nuevo, el orden es `05` → `14` → `06`–`12` → `15`.
- `15_fusiones_fijas` solo lee `results/fusion/fijas/`, que escribe `python -m regimenes.fusion.fijas ejecutar`
  a partir de los paneles OOS **sin** verificar la huella: con una caché obsoleta no falla, así que hay que
  asegurarse antes de que el benchmark está vigente.

`make notebooks` ejecuta 00–16 en orden numérico: sirve cuando caché y ranking ya son coherentes;
tras un benchmark nuevo, ejecuta antes `14_comparativa` a mano. Los paneles OOS
(`results/benchmark/panels/*.parquet`) no se versionan: se regeneran con el benchmark.
Excepción: los dos paneles que lee `get_regimen` (`results/benchmark/panels/A_D06.parquet` y `B_D05.parquet`) sí
se versionan, para que la tool funcione en un clon limpio.

Los notebooks 17 y 18 necesitan `data/sinteticos` (~3 GB, fuera de git), que genera el 17 con `EJECUTAR=True`; el 19
y el 20 se calculan con sus CLI (`python -m regimenes.sinteticos.laboratorio` y `python -m regimenes.sinteticos.aumento`)
y los notebooks solo leen sus resultados.

**Huella de caché.** Tres cálculos guardan sus resultados ligados a una huella de contenido: el benchmark v2, el
pool sin comprimir y las ablaciones del menú T. Los ficheros de cada una están marcados con 🔒 en
[`src/regimenes/README.md`](../../src/regimenes/README.md#huellas-de-caché). En la del **benchmark**
(`python -m regimenes.benchmark`, módulo `regimenes.benchmark.cache`) entran:

- por bytes, los ficheros de los detectores oficiales (`regimenes.detectores`: `__init__`, `registry`, la base y los
  subpaquetes `f*/`, salvo D13) y del juez (`evaluacion/walk_forward.py`, `evaluacion/metricas.py`);
- por AST (sin comentarios ni docstrings), las funciones de `regimenes.benchmark` (`ejecucion`, `cache`) y de
  `evaluacion.metricas` que pueden cambiar predicciones o métricas (`run_one`, carga de paneles, detección por evento…);
- el banco congelado `configs/benchmark_spec.yaml`, ya parseado (sus comentarios no cuentan; `catalog.yaml` no entra);
- los paneles procesados y el S&P 500 crudo, por contenido;
- las versiones de Python y de las librerías numéricas (numpy, pandas, scipy, scikit-learn, hmmlearn, arch, torch,
  jumpmodels).

Las otras dos huellas:

- **pool** (`regimenes.benchmark.pool.huella_codigo`, [ADR-010](../decisions/ADR-010-preregistro-pool-crudo.md)):
  `detectores/pool/*.py`, `benchmark/pool.py`, `features/rentabilidades.py` y `configs/bloques.yaml`;
- **ablaciones** (`regimenes.benchmark.ablaciones.huella_codigo`, [ADR-009](../decisions/ADR-009-resultados-pool.md)):
  `benchmark/ablaciones.py`, `features/bloques.py` y `configs/bloques.yaml`.

Cambiar cualquiera de estos ficheros invalida su caché y obliga a re-ejecutar el cálculo (varias horas). Las claves
son **rutas relativas**: mover o renombrar un fichero también la invalida. Por eso nadie toca esas rutas, y las
versiones de las librerías de la huella están fijadas en `constraints.txt` ([ADR-012](../decisions/ADR-012-estructura-repo.md) §2.2).

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
| D | Re-evaluar D01–D12 sobre `benchmark_spec.yaml` con lags de publicación, propagación de estado y ranking por detección | ✅ benchmark completo (24/24, caché verificada) · [ADR-003](../decisions/ADR-003-causalidad-calendario-estado-ranking.md) · notebooks 05–14 |
| — | Paquete único `regimenes` y un notebook por familia | ✅ [ADR-004](../decisions/ADR-004-unificacion.md) |
| E | Fusión alerta + confirmación | 🟡 fusiones fijas: no aportan frente al mejor detector solo (notebook 15, ADR-011); fusión online C1 en el 16 |
| S | Datos sintéticos: generadores, validación, laboratorio, aumento | ✅ hecha (notebooks 17–20; el aumento no se adopta) · [generadores sintéticos](sinteticos.md) |
| F | Decisión final del detector y contrato de `get_regimen` (futura ADR-006) + validación pseudolive de la tool | 🔜 notebooks 21–22 (esqueletos) |

## Índice de detectores por familia

`Fk` = familia (subpaquete `fk_*` de `src/regimenes/detectores/` y notebook de familia); `Dnn` = detector. D01–D12
forman el benchmark; D13 (`hsmm_tstudent`) es solo una ablación de D08, fuera del ranking y de la huella de caché. El
puesto y el `score_deteccion` de cada detector por pista están en `results/benchmark/ranking_v2.csv` y en el notebook
[`14_comparativa`](../../notebooks/fase1/14_comparativa.ipynb); su comentario, en la sección de resultados de cada ficha de
familia.

| Familia (teoría y detectores) | Detector | Código (`src/regimenes/detectores/`) | Notebook |
|---|---|---|---|
| [F1 reglas y umbrales](F1_reglas.md) | D01 `rule_vix_threshold` | `f1_reglas/rule_vix_threshold.py` | [`06_familia_F1_reglas`](../../notebooks/fase1/06_familia_F1_reglas.ipynb) |
| | D02 `rule_composite_riskoff` | `f1_reglas/rule_composite_riskoff.py` | ídem |
| | D10 `turbulence_mahalanobis` (multivariante) | `f1_reglas/turbulence_mahalanobis.py` | ídem |
| [F2 clustering](F2_clustering.md) | D03 `clustering_gmm` | `f2_clustering/clustering_gmm.py` | [`07_familia_F2_clustering`](../../notebooks/fase1/07_familia_F2_clustering.ipynb) |
| | D09 `jump_model` | `f2_clustering/jump_model.py` | ídem |
| [F3 HMM](F3_hmm.md) | D04 `hmm_gaussian_2s` | `f3_hmm/hmm_gaussian_2s.py` | [`08_familia_F3_hmm`](../../notebooks/fase1/08_familia_F3_hmm.ipynb) |
| | D08 `hmm_tstudent` | `f3_hmm/hmm_tstudent.py` | ídem |
| | D13 `hsmm_tstudent` (ablación) | `f3_hmm/hsmm_tstudent.py` | ídem |
| [F4 Markov-Switching](F4_markov_switching.md) | D05 `markov_switching_var` (media y varianza conmutantes; no es un VAR) | `f4_switching/markov_switching_var.py` | [`09_familia_F4_markov_switching`](../../notebooks/fase1/09_familia_F4_markov_switching.ipynb) |
| [F5 volatilidad GARCH](F5_volatilidad_garch.md) | D06 `garch_t_vol` | `f5_garch/garch_t_vol.py` | [`10_familia_F5_garch`](../../notebooks/fase1/10_familia_F5_garch.ipynb) |
| | D11 `msgarch_regime` | `f5_garch/msgarch_regime.py` | ídem |
| [F6 change-point](F6_change_point.md) | D07 `changepoint_online` | `f6_changepoint/changepoint_online.py` | [`11_familia_F6_changepoint`](../../notebooks/fase1/11_familia_F6_changepoint.ipynb) |
| [F7 redes neuronales](F7_redes_neuronales.md) | D12 `deep_ae_regime` | `f7_deep/deep_ae_regime.py` | [`12_familia_F7_deep`](../../notebooks/fase1/12_familia_F7_deep.ipynb) |

Los nombres heredados que confunden (D01 en la pista A no usa el VIX; D05 no es un VAR) están aclarados en el
[GLOSARIO](../GLOSARIO.md#nombres-de-detectores-que-confunden). Qué features recibe cada detector:
[GLOSARIO](../GLOSARIO.md#núcleo-y-features-por-detector). La narrativa original v1 de D01–D12 (lo implementado, lo
descubierto y las hipótesis de la Capa 1) está en [`historia/fichas_v1.md`](../historia/fichas_v1.md).

## Dónde está cada cosa (rutas)

Todo el código de la Fase 1 es el paquete **`regimenes`** (`src/regimenes/`); el mapa de sus módulos, con lo que entra
en cada huella, está en [`src/regimenes/README.md`](../../src/regimenes/README.md). Las rutas en disco salen
**siempre** de `regimenes.rutas` (ningún módulo ni notebook calcula rutas con `Path(__file__)`):

| Constante (`regimenes.rutas`) | Ruta | Contenido |
|---|---|---|
| `ROOT` | raíz del checkout | — |
| `CONFIGS` · `CATALOG` · `BENCHMARK_SPEC` | `configs/` | `catalog.yaml` (universo y crisis), `benchmark_spec.yaml` (banco congelado), `bloques.yaml` (bloques de los termómetros y menú P), `regimen.yaml` (detector de `get_regimen`) |
| `DATA_RAW` · `DATA_PROCESSED` | `data/raw/` · `data/processed/` | series crudas · paneles por pista y termómetros (fuera de git) |
| `RESULTS_BENCHMARK` | `results/benchmark/` | `metrics/`, `status/`, `manifest.json`, `ranking_v2.csv`, `metrics_master_v2.csv`, `vara/` (MCS y curvas de la vara de medir) y `panels/` (ignorado salvo `A_D06.parquet` y `B_D05.parquet`, los de `get_regimen`) |
| `RESULTS_DETECTORES` | `results/detectores/` | figuras de los notebooks de familia (`<fk_nombre>/`), `comparativa/`, `menus/` y `menus_humo/` (menú T frente al N), `ablaciones/` (`trabajos.csv` consolidado y tablas derivadas; los CSV por trabajo quedan solo en local) y `pool/` (ADR-010) |
| `RESULTS_FUSION` · `RESULTS_FUSION_FIJAS` · `RESULTS_FUSION_ONLINE` | `results/fusion/{fijas,online}/` | fusiones del ADR-011 (notebooks 15 y 16); `d07_d08/` y `d02_d06/` (`RESULTS_FUSION_D07_D08`, `RESULTS_FUSION_D02_D06`) son los antecedentes |
| `RESULTS_PSEUDOLIVE` | `results/pseudolive/` | sub-fase F (vacío hasta el notebook 22) |
| `SINTETICOS_CONFIG` · `DATA_SINTETICOS` · `RESULTS_SINTETICOS` | `configs/sinteticos.yaml` · `data/sinteticos/` · `results/sinteticos/` | configuración, trayectorias generadas (fuera de git) y resultados de la sub-fase S (notebooks 17–20) |
| `DOCS` · `ENV_FILE` | `docs/` · `.env` | — |

La historia de la Capa 1 (decisiones, hallazgos y métricas v1) está en [`historia/capa1/`](../historia/capa1/README.md);
su código, notebooks, memoria e informe originales, en la etiqueta `capa1-final`.
