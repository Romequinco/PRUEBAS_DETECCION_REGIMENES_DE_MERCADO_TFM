# Multi-Agent RAG System for Regime-Aware Macro-Equity Intelligence — TFM (MIAX)

El TFM es un **sistema multi-agente con RAG** (agentes Macro, Equity, Risk, Portfolio y CIO en LangGraph) que lee
información financiera con fecha de publicación, sabe en qué **régimen** está el mercado y con ello construye carteras
explicadas en un *investment memo* con citas. La visión, las cinco fases y su estado real están en
[`docs/tfm/README.md`](docs/tfm/README.md); la decisión de alcance, en [ADR-005](docs/decisions/ADR-005-reencuadre-tfm-multiagente.md).

Requisitos añadidos en la tutoría, todos con el diseño abierto: cliente minorista con formulario de perfil
que fija los límites del CIO, dos carteras (con límites y «en bruto»), memoria de la cartera anterior, datos
alternativos (operaciones del Congreso, divergencias en Polymarket…), un «botón de cataclismo» automático
(EWS, modo broma) y los *System One decision agents* como línea de investigación. Detalle en
[`docs/tfm/cliente_y_carteras.md`](docs/tfm/cliente_y_carteras.md) y
[`docs/tfm/lineas_abiertas.md`](docs/tfm/lineas_abiertas.md).

**Este repositorio contiene la Fase 1** (el régimen, paquete `regimenes`) y el arranque de la **Fase 2**
(ingesta de datos de los agentes, paquete `src/ingesta`; ver [ADR-007](docs/decisions/ADR-007-fase2-ingesta-y-docs.md)).
La Fase 1 no busca un detector concreto, sino el **marco
de evaluación causal y comparable** que juzga a muchos detectores, la **base de datos sólida** sobre la que
hacerlo y, a partir de ahí, un sistema de alerta + confirmación validado con datos sintéticos y en
pseudolive. Su salida es la tool **`get_regimen(fecha)`**: estado, probabilidad y días en el estado, con
solo información publicada hasta `fecha`.

> **En una línea:** *12 detectores de 7 familias (Capa 1) → no eran comparables (cada uno con sus
> datos y su periodo) → datos re-basados en dos pistas congeladas → los 12 re-evaluados con un ranking
> por detección → un solo paquete, con un notebook por familia → datos sintéticos y laboratorio →
> decisión final y pseudolive → `get_regimen` → agentes.*

> **Ojo con «Capa 1»:** en este repositorio es la primera vuelta exploratoria de 12 detectores v1 (tag
> `capa1-final`); **no** es la «Capa 1/Capa 2» de la propuesta (42 ETFs / ~1.300 acciones). Ver el
> [GLOSARIO](docs/GLOSARIO.md#capa-1-del-repositorio-no-es-la-capa-1-de-la-propuesta).

**Si acabas de llegar:** el índice de la documentación y el **único orden de lectura** (por público: equipo, tutor
y memoria, más una visita rápida) están en [`docs/README.md`](docs/README.md#orden-de-lectura).

---

## Principio rector (no negociable)

- **Features causales**: toda estandarización es z-score *expanding/rolling* (en `t`, solo
  estadísticos de datos `≤ t`), y cada dato entra con su **lag de publicación** (ADR-003). Nunca de
  muestra completa: ese *look-ahead* fue el error de la tarea previa.
- **Evaluación walk-forward / out-of-sample**: nada se juzga in-sample.
- **Misma interfaz, mismas métricas**: cada detector implementa `RegimeDetector`
  (`regimenes.detectores.base`) y lo juzga el mismo `regimenes.evaluacion`.

## Arquitectura del paquete `regimenes`

Todo el código de la Fase 1 vive en un paquete instalable (`src/regimenes`, `pip install -e .`). Los notebooks,
los tests y la CLI lo importan igual: `from regimenes import evaluacion as ev`. Detalle por módulo en
[`src/regimenes/README.md`](src/regimenes/README.md).

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
| `regimenes.sinteticos` | generadores de trayectorias con régimen conocido: base común (`comun`, `datos`, `espacio`, `bloques`, `persistencia`, `registry`), 6 paramétricos (`jitter`, `bootstrap_regimen`, `gaussiano_regimen`, `var_regimen`, `garch_regimen`, `rbig`) y 4 neuronales (`flow_matching`, `difusion`, `cvae`, `cgan`; extra `[deep]`); `validacion/` (notebook 16): un módulo por dimensión (`fidelidad`, `discriminador`, `memorizacion`, `utilidad` TSTR/TRTR y `veredicto` por niveles) con umbrales fijados a priori y cambios de reglas y niveles declarados y fechados en `configs/sinteticos.yaml`; `laboratorio/` (notebook 17): `escenarios` (rejilla, secuencias impuestas, intensidad), `metricas` (ADR-003 + retraso), `ejecucion` (paralela, reanudable, caché por huella; CLI `python -m regimenes.sinteticos.laboratorio`) y `analisis` (H1–H4) |

Flujo: `datos` → `features` → `detectores` → `evaluacion` ← `benchmark` → `fusion` → (`sinteticos`, pseudolive);
`informes` solo lee resultados para los notebooks 05–11.

## Mapa del repo

```
/
├── README.md · pyproject.toml · requirements.txt · Makefile · .env.example
├── configs/            catalog.yaml (universo + crisis) · benchmark_spec.yaml (banco congelado) · sinteticos.yaml
├── src/regimenes/      el paquete (tabla de arriba)
├── src/ingesta/        Fase 2: datos de los agentes, un subpaquete por tema (ver docs/fase2)
├── tests/              pytest por subpaquete (datos, features, detectores, evaluacion, benchmark, fusion, informes, sinteticos) · ingesta/ (Fase 2)
├── notebooks/          00–20, planos y ordenados (ver docs/fase1) · fase2/ (notebooks de la Fase 2: <tema>/ o tN_*.ipynb)
├── data/               raw/ (166 series, gitignored + procedencia) · processed/ (paneles por pista) · sinteticos/
├── results/
│   ├── benchmark/      métricas por detector/pista, ranking_v2.csv, metrics_master_v2.csv, manifest.json, panels/ (gitignored)
│   ├── fusion/         d07_d08/ · d02_d06/
│   └── detectores/ · sinteticos/ (generadores/ · validacion/ · laboratorio/) · pseudolive/
└── docs/               README (índice) · GLOSARIO · references.bib · decisions/ (ADR-001…005, 007)
                    tfm/ (visión del TFM, fuentes/) · estado_del_arte/ · fase1/ (pipeline, EDA, F1–F7 con sus detectores, sintéticos)
                    fase2/ (reglas, plan de datos) · historia/ (Capa 1, fichas v1, tarea previa)
```

## Notebooks 00–20

Son el pipeline de la Fase 1, en `notebooks/` (planos y numerados). La tabla notebook a notebook, la plantilla
común de los notebooks de familia y el **orden de ejecución** están en
[`docs/fase1/README.md`](docs/fase1/README.md#pipeline-0020).

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
[`results/benchmark/ranking_v2.csv`](results/benchmark/ranking_v2.csv) (versionado; benchmark completo
24/24 con caché verificada); el ranking lo recalcula [`12_comparativa`](notebooks/12_comparativa.ipynb),
que escribe el `ranking*.csv` que leen los notebooks de familia.

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
en `19_decision_final` (futura ADR-006) antes del pseudolive (detalle en [`14_fusion_d02_d06`](notebooks/14_fusion_d02_d06.ipynb)).

**Laboratorio con régimen conocido (notebook 17).** Simulación controlada con `gaussiano_regimen`,
`var_regimen` y `garch_regimen` (18 celdas por pista, 4356 trabajos), resultados en
`results/sinteticos/laboratorio/`. H1 (recuperación) se cumple: 92 % en A y 100 % en B. H2: el score baja
al acortar y atenuar la crisis (en parte por prevalencia; el recall por evento cae de 0,86 a 0,54 en A).
H3: concordancia con el ranking de ADR-003 de ρ = 0,87 (p = 0,001) en A y 0,48 (p = 0,16) en B. H4: hay
indicio de circularidad para D03 (+0,23 relativa en B) y ninguno para GARCH.

## Cómo reproducir

```bash
cp .env.example .env               # pega tu FRED_API_KEY (gratis: fred.stlouisfed.org)
python -m pip install -e ".[dev]"  # paquete + pytest/jupyter; `make install` añade [deep] torch (D12 y generadores neuronales) y [jump] jumpmodels (D09)
make help                          # lista los objetivos del Makefile (GNU Make; en Windows bajo Git Bash)
```

| Paso | `make` | Equivalente directo |
|---|---|---|
| Entorno completo | `make install` | `python -m pip install -e ".[deep,jump,dev]"` |
| Datos crudos → `data/raw/` | `make datos` (verifica sin red) · `make datos DESCARGAR=1` (descarga lo que falte) | `python -m regimenes.datos --offline` · `python -m regimenes.datos` |
| Paneles causales → `data/processed/` | `make features` | ejecuta `notebooks/03_preprocesado.ipynb` (usa `regimenes.features.construir_paneles`) |
| Benchmark (12 detectores × 2 pistas) → `results/benchmark/` | `make benchmark JOBS=4` | `python -m regimenes.benchmark --track A B --jobs 4` (≈7,4 h de cómputo en serie²) |
| Subconjunto de detectores | `make benchmark DETECTOR="D04 D08"` | `python -m regimenes.benchmark --track A B --detector D04 D08` (con `--jobs 1`, cerrar con `--consolidate`) |
| Verificar caché sin reajustar | — | `python -m regimenes.benchmark --track A B --cache-only` |
| Ejecutar los notebooks 00–14 en orden | `make notebooks` | `python -m jupyter nbconvert --to notebook --execute --inplace notebooks/NN_*.ipynb` |
| Tests | `make test` · `make test-rapido` (sin datos ni resultados locales) | `python -m pytest` |

² Suma de `elapsed_seconds` de `results/benchmark/ranking_v2.csv`: ≈6,6 h en A y ≈0,8 h en B; solo
D04-A tarda ≈1,8 h, que es el mínimo con cualquier número de procesos (con 9 procesos, ≈2 h).

Qué comprueba cada notebook sobre la caché, en qué orden se ejecutan tras un benchmark nuevo y cómo se
ejecutan los de sintéticos (15–17): [`docs/fase1/README.md`](docs/fase1/README.md#orden-de-ejecución).

## Estado del proyecto

El estado de las cinco fases del TFM está en [`docs/tfm/README.md`](docs/tfm/README.md#3-plan-de-trabajo-cinco-fases-y-su-estado)
y el de las sub-fases internas de la Fase 1 (Capa 1, 1–4, D, E, S, F), en
[`docs/fase1/README.md`](docs/fase1/README.md#estado-de-las-sub-fases). En una línea: la Fase 1 tiene hechos
datos, preprocesado, benchmark, fusiones y sintéticos (notebooks 00–17) y le faltan el aumento (18), la decisión
final (19) y el pseudolive (20); la Fase 2 está en marcha en `src/ingesta` ([ADR-007](docs/decisions/ADR-007-fase2-ingesta-y-docs.md)).

Material histórico: el código y los notebooks v1 originales de la Capa 1 se recuperan con el tag git
`capa1-final` (ver [`docs/historia/capa1/README.md`](docs/historia/capa1/README.md) §8).
