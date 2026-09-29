# Detección de regímenes de mercado — TFM (MIAX)

Banco de pruebas para **detección de regímenes de mercado**. El objetivo **no es un detector
concreto**, sino el **marco de evaluación causal y comparable** que juzga a muchos detectores, la
**base de datos sólida** sobre la que hacerlo y, a partir de ahí, un sistema de alerta + confirmación
validado en pseudolive.

> **La historia de este repo, en una línea:**
> *hicimos 12 detectores (Capa 1) → vimos que no eran comparables (cada uno con sus datos y su
> periodo) → re-basamos los datos desde cero (dos pistas congeladas) → re-evaluamos los 12 con un
> ranking por detección → unificamos todo en un solo paquete, con un notebook por familia.*
> Decisiones: [ADR-000](docs/decisions/ADR-000-capa1-retrospectiva.md) →
> [ADR-001](docs/decisions/ADR-001-rebase-datos.md) → [ADR-002](docs/decisions/ADR-002-ajuste-ventanas.md) →
> [ADR-003](docs/decisions/ADR-003-causalidad-calendario-estado-ranking.md) →
> [ADR-004](docs/decisions/ADR-004-unificacion.md).

**Si acabas de llegar:** el índice de toda la documentación y el **orden de lectura** están en
[`docs/README.md`](docs/README.md#orden-de-lectura). Visita rápida (el comienzo de ese mismo orden):
[`docs/GLOSARIO.md`](docs/GLOSARIO.md) (pistas, métricas, rutas) →
[ADR-000](docs/decisions/ADR-000-capa1-retrospectiva.md) (de dónde venimos) → ADR-001 … ADR-003 →
[ADR-004](docs/decisions/ADR-004-unificacion.md) (estructura actual).

---

## Principio rector (no negociable)

- **Features causales**: toda estandarización es z-score *expanding/rolling* (en `t`, solo
  estadísticos de datos `≤ t`), y cada dato entra con su **lag de publicación** (ADR-003). Nunca de
  muestra completa: ese *look-ahead* fue el error de la tarea previa.
- **Evaluación walk-forward / out-of-sample**: nada se juzga in-sample.
- **Misma interfaz, mismas métricas**: cada detector implementa `RegimeDetector`
  (`regimenes.detectores.base`) y lo juzga el mismo `regimenes.evaluacion`.

## Arquitectura del paquete `regimenes`

Todo el código vive en un paquete instalable (`src/regimenes`, `pip install -e .`). Los notebooks,
los tests y la CLI lo importan igual: `from regimenes import evaluacion as ev`.

| Módulo | Qué hace |
|---|---|
| `regimenes.rutas` | rutas centralizadas (`CONFIGS`, `DATA_RAW`, `DATA_PROCESSED`, `RESULTS_BENCHMARK`, `RESULTS_DETECTORES`, …); ningún otro módulo calcula rutas |
| `regimenes.datos` | descarga dirigida por `configs/catalog.yaml` (FRED, yfinance, OFR, GitHub, académicos) · `python -m regimenes.datos` |
| `regimenes.features` | primitivas causales (`transformaciones`), lags de publicación (`lags`), paneles por pista (`paneles`), prueba de truncado (`causalidad`) |
| `regimenes.detectores` | interfaz `RegimeDetector` (`base`), registro de especificaciones por pista (`registry`) y los detectores por familia `f1_reglas` … `f7_deep` |
| `regimenes.evaluacion` | **el juez**: walk-forward (`walk_forward`), métricas y detección por evento (`metricas`), ranking ADR-003 y líneas base (`ranking`) |
| `regimenes.benchmark` | ejecución reanudable y paralela con caché validada por huella (`ejecucion`, `cache`, `cli`) · `python -m regimenes.benchmark` |
| `regimenes.fusion` | máquina causal normal / vigilancia / confirmado (alerta + confirmador) |
| `regimenes.viz` | estilo de casa para figuras (franjas de crisis, paneles de estados) |
| `regimenes.informes` | utilidades comunes de los notebooks de familia 05–11: carga verificada de resultados (`cargar_resultados_familia`), tablas de configuración/cobertura y figuras de estados OOS |
| `regimenes.sinteticos` | interfaz `Generador`, registro y validación de datos sintéticos (esqueleto de la próxima fase) |

Flujo: `datos` → `features` → `detectores` → `evaluacion` ← `benchmark` → `fusion` → (`sinteticos`, pseudolive);
`informes` solo lee resultados para los notebooks 05–11.

## Mapa del repo

```
/
├── README.md · pyproject.toml · requirements.txt · Makefile · .env.example
├── configs/            catalog.yaml (universo + crisis) · benchmark_spec.yaml (banco congelado) · sinteticos.yaml
├── src/regimenes/      el paquete (tabla de arriba); detalle en src/regimenes/README.md
├── tests/              pytest por subpaquete (datos, features, detectores, evaluacion, benchmark, fusion, informes, sinteticos)
├── notebooks/          00–20, planos y ordenados (tabla abajo)
├── data/               raw/ (166 series, gitignored + procedencia) · processed/ (paneles por pista) · sinteticos/
├── results/
│   ├── benchmark/      métricas por detector/pista, ranking_v2.csv, metrics_master_v2.csv, manifest.json, panels/ (gitignored)
│   ├── fusion/         d07_d08/ · d02_d06/
│   └── detectores/ · sinteticos/ · pseudolive/
└── docs/               TODO el conocimiento: decisiones, datos, teoría, fichas, historia, revisiones (docs/README.md)
```

## Notebooks 00–20

| Nº | Notebook | Fase | Qué hace |
|---|---|---|---|
| 00 | [`00_descarga`](notebooks/00_descarga.ipynb) | datos | panorámica de las 166 series, cobertura y descarga |
| 01 | [`01_eda`](notebooks/01_eda.ipynb) | datos | EDA maestro (12 secciones) |
| 02 | [`02_diseno_preprocesado`](notebooks/02_diseno_preprocesado.ipynb) | features | decisiones del preprocesado (features, frecuencias, alineación causal) |
| 03 | [`03_preprocesado`](notebooks/03_preprocesado.ipynb) | features | paneles causales `pista{A,B}_{diaria,mensual}` + labels, con gates |
| 04 | [`04_protocolo_evaluacion`](notebooks/04_protocolo_evaluacion.ipynb) | D | protocolo walk-forward, gate/preflight, caché por huella, CLI paralela |
| 05 | [`05_familia_F1_reglas`](notebooks/05_familia_F1_reglas.ipynb) | D | D01 umbral de volatilidad (VIX solo en B) · D02 riesgo compuesto · D10 turbulencia Mahalanobis |
| 06 | [`06_familia_F2_clustering`](notebooks/06_familia_F2_clustering.ipynb) | D | D03 GMM · D09 Statistical Jump Model |
| 07 | [`07_familia_F3_hmm`](notebooks/07_familia_F3_hmm.ipynb) | D | D04 HMM gaussiano · D08 HMM t-Student · D13 HSMM (ablación) |
| 08 | [`08_familia_F4_markov_switching`](notebooks/08_familia_F4_markov_switching.ipynb) | D | D05 Markov-Switching de media y varianza (univariante; pese al nombre `markov_switching_var`, no es un VAR) |
| 09 | [`09_familia_F5_garch`](notebooks/09_familia_F5_garch.ipynb) | D | D06 GJR-GARCH-t · D11 MS-GARCH |
| 10 | [`10_familia_F6_changepoint`](notebooks/10_familia_F6_changepoint.ipynb) | D | D07 CUSUM online robusto |
| 11 | [`11_familia_F7_deep`](notebooks/11_familia_F7_deep.ipynb) | D | D12 autoencoder + GMM |
| 12 | [`12_comparativa`](notebooks/12_comparativa.ipynb) | D | scorecards, ranking ADR-003 y comparación entre familias |
| 13 | [`13_fusion_d07_d08`](notebooks/13_fusion_d07_d08.ipynb) | E | D7 alerta + D8 confirma |
| 14 | [`14_fusion_d02_d06`](notebooks/14_fusion_d02_d06.ipynb) | E | selección auditable: D2 alerta + D6 confirma |
| 15 | [`15_sinteticos_generadores`](notebooks/15_sinteticos_generadores.ipynb) | S | generadores de escenarios con régimen conocido (esqueleto) |
| 16 | [`16_sinteticos_validacion`](notebooks/16_sinteticos_validacion.ipynb) | S | validación de fidelidad, utilidad (TSTR) y memorización (esqueleto) |
| 17 | [`17_sinteticos_laboratorio`](notebooks/17_sinteticos_laboratorio.ipynb) | S | laboratorio de detectores con régimen conocido (esqueleto) |
| 18 | [`18_sinteticos_aumento`](notebooks/18_sinteticos_aumento.ipynb) | S | aumento de datos real + sintético (esqueleto) |
| 19 | [`19_decision_final`](notebooks/19_decision_final.ipynb) | F | decisión final del sistema a congelar, futura ADR-005 (esqueleto) |
| 20 | [`20_pseudolive`](notebooks/20_pseudolive.ipynb) | F | regla congelada sobre datos no usados (esqueleto) |

Los notebooks de familia (05–11) siguen una plantilla común: teoría de la familia, configuración
desde el registro, ejecución con `EJECUTAR = False` (lee la caché de `results/benchmark`), estados OOS
por pista con franjas de crisis, diagnóstico propio, métricas y puesto en el ranking, hallazgos de la
Capa 1 y conclusión. Sus figuras van a `results/detectores/<fk_nombre>/`. La numeración es de
lectura: tras un benchmark nuevo, `12_comparativa` se ejecuta **antes** que 05–11, porque escribe el
`ranking*.csv` que estos leen (ver «Cómo reproducir»).

## Las dos pistas de datos (ADR-001, ventanas reajustadas por ADR-002)

| Pista | Qué | Ventana | Features | Crisis | Ataca |
|---|---|---|:---:|:---:|---|
| **A — Espina profunda + curva** | S&P 500 + vol + factores + crédito/macro profundos + curva de tipos completa | 1962-01-02 → 2026-05-29 | **41** | **18** | el n≈4 (potencia); sacrifica conscientemente 4 crisis pre-1962 (incluida la Gran Depresión) a cambio de la curva de tipos |
| **B — Panel rico** | espina de A + crédito HY/IG, curva/breakevens, vol complex, 9 sectores, 11 índices de amplitud, FX | 2007-04-11 → 2026-05-29 | **106** | 10 | el punto ciego de 2013; mismo fin que A a propósito, cero coste en crisis frente a 2003 |

`A ⊆ B` por construcción (toda serie viva en 1962 también lo está en 2007). Cada pista congela su
ventana + features en [`configs/benchmark_spec.yaml`](configs/benchmark_spec.yaml) → *un leaderboard
justo 1:1 dentro de cada pista*.

41 y 106 son el **pool** de features candidatas de cada pista, no lo que ve cada detector: cada uno usa
de 1 a 14, fijadas en `regimenes.detectores.registry`. Los multivariantes (D03, D04, D08, D09, D12)
usan el **núcleo** (9 features en A, 14 en B); D05, D06, D07 y D11 solo el retorno del S&P 500; D01
una señal (volatilidad realizada en A, VIX en B) y D02/D10 cuatro. Detalle en el
[GLOSARIO](docs/GLOSARIO.md#núcleo-y-features-por-detector).

## Resultados clave (ADR-003)

Ranking de **detección** por pista: `score_deteccion` = F1 entre precisión diaria y recall por evento
(crisis detectada = ≥ 3 sesiones consecutivas marcadas dentro de su ventana), tras filtrar por niveles
(precisión que no supera al azar, parpadeo, salida degenerada). *Lift sobre azar* = precisión / tasa
base (tasa base = precisión de marcar siempre crisis); definiciones de todas las columnas en el
[GLOSARIO](docs/GLOSARIO.md#términos-del-ranking-columnas-de-ranking_v2csv--metrics_master_v2csv). Valores de
`results/benchmark/ranking_v2.csv` versionado en git (commit `1f95a9b`); el ranking vivo lo
recalcula [`12_comparativa`](notebooks/12_comparativa.ipynb), que escribe el `ranking*.csv` que leen los
notebooks de familia.

**Pista A** (17 crisis evaluadas OOS, tasa base de días de crisis 0.194)

| Puesto | Detector | Score | Crisis detectadas | Precisión | Lift sobre azar |
|:---:|---|:---:|:---:|:---:|:---:|
| 1 | D10 turbulencia Mahalanobis | 0.614 | 16/17 | 0.455 | 2.35 |
| 2 | D01 umbral de volatilidad realizada¹ | 0.587 | 16/17 | 0.427 | 2.20 |
| 3 | D03 GMM (K=3) | 0.565 | 15/17 | 0.416 | 2.14 |
| 4 | D02 regla compuesta risk-off | 0.559 | 12/17 | 0.462 | 2.38 |
| 5 | D05 Markov-Switching | 0.546 | 17/17 | 0.375 | 1.94 |

¹ En la pista A, D01 (`rule_vix_threshold`) no usa el VIX, que no existe antes de 1990, sino un umbral
sobre la volatilidad realizada del S&P 500 (`SP500_vol_z`); en B sí es la regla VIX original.

**Pista B** (9 crisis evaluadas OOS, tasa base 0.173)

| Puesto | Detector | Score | Crisis detectadas | Precisión | Lift sobre azar |
|:---:|---|:---:|:---:|:---:|:---:|
| 1 | D05 Markov-Switching | 0.647 | 9/9 | 0.478 | 2.77 |
| 2 | D06 GJR-GARCH-t | 0.597 | 8/9 | 0.449 | 2.60 |
| 3 | D01 regla VIX | 0.505 | 6/9 | 0.406 | 2.35 |
| 4 | D02 regla compuesta risk-off | 0.501 | 6/9 | 0.402 | 2.33 |
| 5 | D04 HMM gaussiano | 0.497 | 7/9 | 0.366 | 2.12 |

En B, **D09** (jump model) no supera la precisión del azar y **D12** (autoencoder) parpadea: quedan
fuera del nivel elegible. Las reglas simples y los modelos de volatilidad dominan en ambas pistas;
la complejidad (D11, D12) sigue sin pagarse, como ya indicaba la Capa 1.

**Fusión (Fase E).** D2 alerta + D6 confirma mejora a D2 y a D6 aislados en ambas pistas, pero **no
anticipa** el inicio de las crisis; D2 y D7 quedan prácticamente empatados como alerta y el control por
mitades elige otras alertas. La capa de alerta no añade anticipación: la regla a congelar se decide
en `19_decision_final` antes del pseudolive (detalle en [`14_fusion_d02_d06`](notebooks/14_fusion_d02_d06.ipynb)).

## Cómo reproducir

```bash
cp .env.example .env              # pega tu FRED_API_KEY (gratis: fred.stlouisfed.org)
python -m pip install -e ".[dev]"  # paquete + pytest/jupyter; `make install` añade [deep] torch (D12) y [jump] jumpmodels (D09)
make help                          # lista los objetivos del Makefile (GNU Make; en Windows bajo Git Bash)
```

| Paso | `make` | Equivalente directo |
|---|---|---|
| Entorno completo | `make install` | `python -m pip install -e ".[deep,jump,dev]"` |
| Datos crudos → `data/raw/` | `make datos` (verifica sin red) · `make datos DESCARGAR=1` (descarga lo que falte) | `python -m regimenes.datos --offline` · `python -m regimenes.datos` |
| Paneles causales → `data/processed/` | `make features` | ejecuta `notebooks/03_preprocesado.ipynb` (usa `regimenes.features.construir_paneles`) |
| Benchmark (12 detectores × 2 pistas) → `results/benchmark/` | `make benchmark JOBS=4` | `python -m regimenes.benchmark --track A B --jobs 4` (≈7,6 h de cómputo en serie²) |
| Subconjunto de detectores | `make benchmark DETECTOR="D04 D08"` | `python -m regimenes.benchmark --track A B --detector D04 D08` (con `--jobs 1`, cerrar con `--consolidate`) |
| Verificar caché sin reajustar | — | `python -m regimenes.benchmark --track A B --cache-only` |
| Ejecutar los notebooks 00–14 en orden | `make notebooks` | `python -m jupyter nbconvert --to notebook --execute --inplace notebooks/NN_*.ipynb` |
| Tests | `make test` · `make test-rapido` (sin datos ni resultados locales) | `python -m pytest` |

² Suma de `elapsed_seconds` de `ranking_v2.csv` versionado (commit `1f95a9b`): ≈6,9 h en A y ≈0,7 h
en B; solo D04-A tarda ≈1,9 h, que es el mínimo con cualquier número de procesos (ADR-003 midió
~1 h 55 min con 9).

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

## Estado del proyecto

Fases: **1–4** son las del re-base de datos ([ADR-001](docs/decisions/ADR-001-rebase-datos.md)):
1 reorganización, 2 datos, 3 EDA + banco congelado, 4 diseño y preprocesado causal. Las posteriores se
nombran por letra: **D** detectores (benchmark), **E** fusión *early warning* (alerta + confirmación),
**U** unificación, **S** sintéticos, **F** final (decisión + pseudolive).

| Fase | Qué | Estado |
|---|---|---|
| Capa 1 | 12 detectores v1 sobre 9 series (exploración) | ✅ congelada 2026-07-18 · [historia](docs/historia/capa1/README.md) · [ADR-000](docs/decisions/ADR-000-capa1-retrospectiva.md) |
| 1–3 | Reorganización · datos (166/174 series) · EDA + banco congelado | ✅ [ADR-001](docs/decisions/ADR-001-rebase-datos.md) · [ADR-002](docs/decisions/ADR-002-ajuste-ventanas.md) |
| 4 | Diseño + preprocesado causal v2 | ✅ notebooks 02–03 |
| D | Re-evaluar D01–D12 sobre `benchmark_spec.yaml` | ✅ benchmark completo (24/24; cifras versionadas en `1f95a9b`) · ⚠️ la caché local de `results/benchmark` está **obsoleta** tras la fase 1 de la unificación (cambian las huellas, [ADR-004](docs/decisions/ADR-004-unificacion.md) §5) hasta que termine su re-ejecución completa: mientras tanto 05–11 fallan al cargar con un `RuntimeError` que indica el comando a ejecutar, y 04/12 lo avisan y trabajan con las métricas versionadas |
| — | Revisión completa (2026-09-29) + ADR-003 (lags de publicación, propagación de estado, ranking por detección) | ✅ [revisión](docs/revisiones/REVISION_2026-09-29.md) · [ADR-003](docs/decisions/ADR-003-causalidad-calendario-estado-ranking.md) |
| E | Fusión alerta + confirmación | 🟡 D2+D6 seleccionado; regla final pendiente (notebooks 13–14) |
| U | **Unificación** Capa 1 + v2: paquete `regimenes`, notebooks por familia, docs | 🟡 en curso en la rama `unificacion` · [ADR-004](docs/decisions/ADR-004-unificacion.md) |
| S | Datos sintéticos: generadores, validación, laboratorio, aumento | 🔜 notebooks 15–18 (esqueletos) + `regimenes.sinteticos` |
| F | Decisión final + pseudolive independiente | 🔜 notebooks 19–20 |

Material histórico: el código y los notebooks v1 originales se recuperan con el tag git
`capa1-final`; el código v2 previo a la unificación, con `v2-pre-unificacion`.
