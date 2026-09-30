# Glosario — conceptos del proyecto (fuente canónica)

> Definiciones **echadas a tierra** de los términos que gobiernan todo el repo. Es la **fuente
> única**: los notebooks (`00`–`20`) y el resto de docs enlazan aquí en vez de redefinir.
> Incluye también **dónde está cada cosa** (rutas y módulos del paquete `regimenes`, [ADR-004](decisions/ADR-004-unificacion.md))
> y el **pipeline 00–20** (al final).

---

## Las dos pistas (y `ambas` / `validacion`)

Cada serie/feature se asigna a una **pista** = *para qué banco de pruebas sirve*. Nacen de la
decisión [`ADR-001`](decisions/ADR-001-rebase-datos.md): **no se puede tener a la vez máxima historia
y máxima riqueza de features**, así que se construyen dos bancos en paralelo. Las ventanas y recuentos
de esta tabla fueron **reajustados por [`ADR-002`](decisions/ADR-002-ajuste-ventanas.md)** (2026-07-20):
el fin de ventana pasó a estar gobernado por la serie diaria más fresca (nunca una mensual) y el pool
de features candidatas creció de 35 a 106 series únicas (64% de las 166 descargadas).

| Pista | Qué es | Ventana del banco | Nº features | Nº crisis | Objetivo |
|---|---|---|:---:|:---:|---|
| **A — espina histórica + curva de tipos** | equity + vol realizada + factores + crédito/macro profundos + curva de tipos completa (DGS5/10, T10YFF) | **1962-01-02 → 2026-05-29** (gobiernan DGS10/DGS5/T10YFF al inicio; FF_FACTORS_3_DAILY al fin) | **41** | **18** | **potencia estadística** — se sacrifican conscientemente las 4 crisis con pico < 1962 (incluida la Gran Depresión) a cambio del bloque de curva completo |
| **B — panel rico multi-activo** | espina de A **+** crédito HY/IG, curva/breakevens/velocidad, complejo de vol, 9 sectores, 11 índices de amplitud (breadth), FX/commodities | **2007-04-11 → 2026-05-29** (gobierna HYG_CREDIT al inicio; FF_FACTORS_3_DAILY al fin — **el mismo fin que A, a propósito**) | **106** | 10 | **riqueza / discriminación** — separar mejor y **atacar el punto ciego de 2013**; mover el inicio de 2003 a 2007 no cuesta ninguna crisis |
| **ambas** | serie que existe en las dos ventanas (p. ej. VIX, pendientes de curva); por construcción **A ⊆ B** (toda serie viva en 1962 lo está en 2007) | — | — | — | se **cuenta una sola vez** (dedup) pero alimenta los dos bancos |
| **validacion** | índices de estrés ya hechos y labels (OFR FSI, NFCI, NBER) | — | — | — | **ground truth laxo** para *juzgar* regímenes — **nunca** entra como feature |

> **`pista` y `rol` son ejes independientes.** `pista` = a qué banco sirve; `rol` = qué papel juega la
> serie (abajo). Por eso el recuento de "validacion" difiere según el eje (25 series con
> `pista=validacion` vs 21 con `rol=validation`).
>
> ⚠️ **El campo `pista` de `configs/catalog.yaml` (por serie) NO se reescribió con ADR-002** — sigue
> reflejando la clasificación manual original (A≈1927+, B≈2003+) y es solo descriptivo del universo
> declarado. La fuente de verdad **operativa** (qué serie entra en qué ventana congelada) es
> exclusivamente [`configs/benchmark_spec.yaml`](../configs/benchmark_spec.yaml).

---

## Los cinco roles

El campo `rol` (en `configs/catalog.yaml`) dice **qué papel** juega cada serie. No es un ranking de
calidad: es una función.

| Rol | Qué es (echado a tierra) | Ejemplo |
|---|---|---|
| **`spine`** | **columna vertebral** de la pista: la serie imprescindible que la define y le da su historia | `SP500` (Pista A), `DGS10` (curva) |
| **`core`** | **feature principal** del pool: entra al detector con peso propio | `VIX`, `MOVE`, `MOODYS_BAA_AAA_SPREAD` |
| **`enricher`** | **enriquecedor opcional**: añade matiz, no es imprescindible | sectores SPDR, breakevens, VVIX |
| **`fallback`** | **sustituto redundante** de una serie ya presente: solo se usa si falla la fuente primaria (regla de dedup del catálogo) | `GOLD_FUT` (fallback de `GOLD_GLD`) |
| **`validation`** | **ground truth** para evaluar/etiquetar — **jamás feature** | `OFR_FSI`, `NFCI`, `NBER_RECESSION_DAILY` |

Recuento sobre las 174 **declaradas**: spine 23 · core 44 · enricher 62 · fallback 24 · validation 21.
Sobre las 166 **descargadas**: enricher 58 · fallback 21 · validation 20 (spine/core sin cambio).

---

## Regla de oro anti-fuga (la más importante)

Una serie de **`rol=validation`** (índices de estrés, recesión NBER) es *ground truth*: sabe "esto
**fue** una crisis". **Nunca** puede entrar a la vez como **feature** y como **etiqueta** — sería
**fuga de información** (el detector "adivinaría" mirando la respuesta). Por eso `validation` solo se
usa para **juzgar** los detectores en la Fase D, jamás en la matriz de features.

## Causalidad (no look-ahead)

Toda feature es **causal**: en el instante `t` solo usa estadísticos de datos `≤ t` (z-score
*expanding*/*rolling*, nunca de muestra completa — ese fue el error de la tarea previa). Se verifica
con `assert_causal(builder, raw, cut)` de `regimenes.features.causalidad` (truncar el futuro y recomputar debe dar
`max|Δ| = 0`). Se demuestra en [`02_diseno_preprocesado.ipynb`](../notebooks/02_diseno_preprocesado.ipynb)
§4-5 y es el gate 1 de `03_preprocesado`.

Hay que distinguir dos niveles:
- **Causalidad computacional**: ningún cálculo en `t` usa filas posteriores a `t`. Es lo que prueba el
  truncado, y se cumple.
- **Causalidad de calendario**: el dato fechado en `t` ya estaba *publicado* en `t`. El truncado **no**
  lo detecta. Hoy hay fugas conocidas de este tipo (series mensuales de FRED que son media del mes
  fechada el día 1; lag macro corto) que **ADR-003 corrigió** con una tabla única de lags
  (`regimenes.features.lags.LAG_PUBLICACION`) y un test de truncado por fecha de publicación.

## Métricas del benchmark (lectura correcta)

- **`false_alarm_rate`** es en realidad **1 − precisión** (fracción de días marcados como crisis que
  caen fuera de ventana de crisis), no FP/(FP+TN). La tasa base de marcar siempre crisis es ≈0,81 (A)
  y ≈0,83 (B).
- **Ranking principal (ADR-003):** `score_deteccion` = F1 entre precisión diaria y **recall por
  evento** (crisis detectada = ≥ 3 sesiones consecutivas marcadas dentro de su ventana), tras filtrar
  por niveles (precisión que no supera al azar, parpadeo, salida degenerada).
- **`rank_medio`** (legacy) estaba sesgado hacia la inactividad: la línea base "siempre crisis"
  quedaba 4ª en ambas pistas con el banco previo a ADR-003
  y 5ª en ambas con el banco re-ejecutado (`12_comparativa` §8). Se conserva solo como columna de
  comparación (`rank_medio_legacy`, `puesto_legacy`).
- **Crisis "en ventana" ≠ "evaluadas OOS"**: 18/10 en ventana; 17 (una parcial) / 9 evaluadas OOS.
- **Utilidad operativa** (fusión): solo sirve para **ordenar** alternativas; su signo no se interpreta.
- **Precisión de avisos** (acuerdo con el confirmador) ≠ **atribución a crisis reales**.

### Términos del ranking (columnas de `ranking_v2.csv` / `metrics_master_v2.csv`)

| Término | Columna | Definición |
|---|---|---|
| **precisión diaria** | `det_precision` | fracción de días marcados como crisis que caen dentro de alguna ventana de crisis `[pico, suelo]`; = 1 − `false_alarm_rate` |
| **tasa base** | `det_base_rate` | fracción de días OOS que están dentro de una ventana de crisis = precisión de marcar *siempre* crisis |
| **lift sobre azar** | `det_lift_precision` | precisión / tasa base. Lift ≤ 1 = no mejora a marcar al azar → nivel 1 (no elegible) |
| **recall por evento** | `det_event_recall` (`det_n_detectados`/`det_n_eventos`) | fracción de crisis evaluadas OOS con ≥ 3 sesiones consecutivas marcadas dentro de su ventana |
| **`score_deteccion`** | `score_deteccion` | F1 (media **armónica**, no producto) de precisión diaria y recall por evento: 2·P·R/(P+R) |
| **nivel** | `nivel_etiqueta` | filtro previo al score: `elegible` · `precision_no_supera_azar` (lift ≤ 1) · `parpadeo` · `degenerado` (salida constante); un nivel peor nunca supera a uno mejor |
| **cobertura media** | `mean_crisis_coverage` | media, sobre las crisis evaluadas, de la fracción de días de cada ventana `[pico, suelo]` marcados como crisis (sensibilidad diaria, no por evento) |
| **trampas** | `mean_trap_activation`, `fa_*` | episodios de estrés que **no** son crisis del catálogo (`false_positive_windows` de `benchmark_spec.yaml`, p. ej. el *taper* de 2013); la activación es la fracción de sus días marcados como crisis (ideal: 0) |
| **switching rate** | `switching_rate` | nº de cambios de estado / nº de días OOS; alto = parpadeo (*flickering*). Primer desempate del ranking (menor es mejor) |
| **duración media de régimen** | `mean_regime_duration` | días OOS / nº de episodios; < 5 sesiones = parpadeo (nivel 2) |
| **estabilidad de etiqueta** | `label_stability` | para cada fecha re-etiquetada en varios *refits* del walk-forward, fracción de acuerdo con la etiqueta más frecuente; media en [0, 1] (1 = la etiqueta no "baila" al añadir datos). Segundo desempate |
| **puesto legacy** | `rank_medio_legacy`, `puesto_legacy` | puesto por el criterio anterior a ADR-003 (media de ranks de 5 ejes), recalculado sobre los resultados actuales solo para trazar el cambio |
| **Pareto** | — | frontera de Pareto recall por evento / precisión: detectores a los que ningún otro supera en ambos ejes a la vez. Vista complementaria del ranking (`12_comparativa` §4), sin ganador único |

**Gate / preflight.** *Gate* = comprobación bloqueante: si falla, el notebook se detiene en vez de
seguir con datos parciales (gates de causalidad y de paneles en `03_preprocesado`; gate de
comparabilidad de `12_comparativa` §2: mismo intervalo OOS por pista, sin duplicados, trazable a la
ejecución). *Preflight* = el gate previo al benchmark (`regimenes.benchmark.preflight`,
`04_protocolo_evaluacion` §6): comprueba paneles, columnas y dependencias opcionales **sin ajustar
ningún modelo**.

## Núcleo y features por detector

Cada pista tiene un **pool** de 41 (A) o 106 (B) features candidatas, pero **ningún detector las usa
todas**: cada uno recibe de 1 a 14, fijadas por pista en `regimenes.detectores.registry` (tabla §2 de
cada notebook de familia):

- **Núcleo** (`CORE_A`, 9 features; `CORE_B` = núcleo de A + 5 de la pista rica, 14 features): lo usan
  los detectores multivariantes D03, D04, D08, D09 y D12. El núcleo de A son retorno, volatilidad,
  momentum y drawdown del S&P 500, factor de mercado, cambio del 10 años, crédito Baa-Aaa mensual,
  pendiente histórica e INDPRO interanual.
- **Univariantes** sobre el retorno del S&P 500 (`SP500_ret`): D05, D06, D07, D11.
- **Reglas**: D01 usa una sola señal (en A `SP500_vol_z`, volatilidad realizada, porque el VIX no
  existe antes de 1990; en B `VIX_level_z`); D02 y D10 usan 4 features distintas por pista.

## Nombres de detectores que confunden

- **D01 `rule_vix_threshold`**: en la pista A **no usa el VIX** (no existe antes de 1990) sino un umbral
  sobre la volatilidad realizada del S&P 500 (`variante_v2` de `metrics_master_v2.csv`); solo en B es la
  regla VIX original. Por eso se cita como "regla de volatilidad (VIX en B)".
- **D05 `markov_switching_var` / "MS-VAR"**: el nombre heredado de la Capa 1 sugiere un VAR
  (autorregresión vectorial), pero **no lo es**: es un `MarkovRegression` univariante de Hamilton sobre
  el retorno del S&P 500 con **media y varianza conmutantes** (`trend="c"`, `switching_variance=True`),
  sin términos autorregresivos. "Markov-Switching de media y varianza" es la descripción correcta.

## El banco congelado (benchmark)

[`configs/benchmark_spec.yaml`](../configs/benchmark_spec.yaml) **congela**, por pista, la ventana + las
features + las etiquetas (crisis, falsos positivos, troughs). Es la **variable controlada** de la
Fase D: un detector puede cambiar su algoritmo, pero **no** estas ventanas/etiquetas.

## Generadores sintéticos (fase S)

Conceptos de `regimenes.sinteticos`; teoría y límites de cada generador en
[`teoria/F8_generadores_sinteticos.md`](teoria/F8_generadores_sinteticos.md).

| Término | Definición |
|---|---|
| **régimen de referencia sintético** | etiqueta binaria con la que se ajustan **todos** los generadores: 1 = el día cae dentro de alguna ventana `[pico, suelo]` de `crisis_windows` de la pista (`configs/benchmark_spec.yaml`), 0 = calma (`sinteticos.datos.regimen_referencia`). **Nunca sale de un detector** (sería circular: ese detector jugaría en casa sobre lo generado) |
| **semántica del régimen sintético** | en el histórico `regime = 1` es «tramo pico→suelo de una crisis del catálogo»; en una trayectoria sintética es **«día extraído de la ley condicional de crisis»**: el tramo no tiene por qué ir de un máximo a un mínimo del precio generado. El laboratorio mide si un detector reconoce ese cambio de ley, no el recall por evento del benchmark |
| **corte de entrenamiento** (`fin_train`) | última fecha que ve un generador: pista A 2006-12-31, pista B 2017-12-31 (`configs/sinteticos.yaml`). No puede partir un episodio de crisis |
| **cadena de régimen** | el régimen es exógeno a los generadores: o se **impone** la secuencia al muestrear, o se **simula** con la cadena estimada en train (Markov con duraciones geométricas, o semi-Markov con las duraciones empíricas de las rachas), arrancando del último estado de train o de la distribución estacionaria |
| **espacio de generación** (de trabajo) | lo que modelan los generadores: el log-retorno crudo del S&P 500 (`SP500_ret`) más las columnas **no derivables** del núcleo, estandarizadas solo con train. Las cuatro features deterministas de la senda del S&P 500 (`SP500_ret_z`, `SP500_vol_z`, `SP500_momentum`, `SP500_drawdown`) **no** se modelan |
| **re-derivación** | vuelta al espacio público: esas cuatro features se recalculan con las mismas primitivas causales de `regimenes.features` sobre [historia real del S&P 500 hasta `fin_train`] + [retornos sintéticos]. Sobre la historia real reproduce el panel con error 0 (`error_rederivacion_`). Cada trayectoria es una continuación hipotética del mercado tras el corte |
| **siguiente bloque** | formulación de RBIG y de los generadores neuronales: se modela la ley del bloque de 21 sesiones siguiente condicionada al contexto previo (21 sesiones en los neuronales, 5 en RBIG; `configs/sinteticos.yaml`) y al régimen de cada día del bloque, y una trayectoria larga se obtiene **encadenando** bloques (el bloque generado pasa a ser contexto del siguiente; el primer contexto son las últimas filas reales de train) |
| **fechas sintéticas** | días hábiles de lunes a viernes posteriores al corte; son una etiqueta ordenada, **no** el calendario de la NYSE: no se cruzan por fecha con datos reales |

---

## Dónde está cada cosa (rutas y módulos)

Todo el código es el paquete **`regimenes`** (`src/regimenes/`, instalado con `pip install -e .`).
Las rutas en disco salen **siempre** de `regimenes.rutas` (ningún módulo ni notebook calcula rutas
con `Path(__file__)`):

| Constante (`regimenes.rutas`) | Ruta | Contenido |
|---|---|---|
| `ROOT` | raíz del checkout | — |
| `CONFIGS` · `CATALOG` · `BENCHMARK_SPEC` · `SINTETICOS_CONFIG` | `configs/` | `catalog.yaml`, `benchmark_spec.yaml`, `sinteticos.yaml` |
| `DATA_RAW` · `DATA_PROCESSED` · `DATA_SINTETICOS` | `data/raw/` · `data/processed/` · `data/sinteticos/` | series crudas · paneles por pista · trayectorias sintéticas |
| `RESULTS_BENCHMARK` | `results/benchmark/` | `metrics/`, `status/`, `panels/`, `manifest.json`, `ranking_v2.csv`, `metrics_master_v2.csv` |
| `RESULTS_FUSION` · `RESULTS_FUSION_D07_D08` · `RESULTS_FUSION_D02_D06` | `results/fusion/{d07_d08,d02_d06}/` | tablas de las fusiones |
| `RESULTS_DETECTORES` | `results/detectores/<fk_nombre>/` | figuras de los notebooks de familia |
| `RESULTS_SINTETICOS` · `RESULTS_PSEUDOLIVE` | `results/sinteticos/` · `results/pseudolive/` | fases S y F |
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
| `regimenes.sinteticos` (`base`, `comun`, `datos`, `espacio`, `bloques`, `persistencia`, `registry`, `parametricos/`, `neuronales/`, `validacion`) | interfaz `Generador` y base común `GeneradorBase`, régimen de referencia y cadena de regímenes, espacio de generación y re-derivación, bloques y encadenado, registro perezoso (`registry.crear`) y 10 generadores (6 paramétricos; 4 neuronales con el extra `[deep]`). `validacion` sigue siendo solo firmas (notebook 16) |

La historia de la Capa 1 (decisiones, hallazgos, memoria, informe y métricas v1) está en
[`historia/capa1/`](historia/capa1/README.md); su código y notebooks originales, en el tag `capa1-final`.

**Familias e IDs.** `Fk` = familia de la teoría (F1 reglas/umbrales, F2 clustering, F3 HMM, F4
Markov-Switching, F5 GARCH, F6 change-point, F7 redes; F8 no es una familia de detectores sino la
de los generadores sintéticos). `Dnn` = detector (D01–D12 en el benchmark;
D13 `hsmm_tstudent` solo como ablación de D08, fuera del ranking y de la huella de caché). El reparto
por familia es el de los subpaquetes `f1_reglas` … `f7_deep` y de los notebooks 05–11.

---

## Pipeline 00–20

| Fase | Notebooks | Qué produce |
|---|---|---|
| **Datos** | `00_descarga` → `01_eda` | `data/raw/` + informe EDA |
| **Features** | `02_diseno_preprocesado` → `03_preprocesado` | `data/processed/pista{A,B}_{diaria,mensual}.parquet` + labels |
| **D — detectores** | `04_protocolo_evaluacion` → `12_comparativa` → `05`–`11` (una familia cada uno) | `results/benchmark/` (métricas, ranking, paneles OOS) + `results/detectores/` |
| **E — fusión** | `13_fusion_d07_d08` · `14_fusion_d02_d06` | `results/fusion/` |
| **S — sintéticos** | `15_sinteticos_generadores` → `16_sinteticos_validacion` → `17_sinteticos_laboratorio` → `18_sinteticos_aumento` | `15`: ajusta los 10 generadores por pista y guarda trayectorias con régimen conocido en `data/sinteticos/<generador>/pista<X>/` (`trayectorias.parquet` con la cadena simulada y `trayectorias_impuesto.parquet` con la secuencia impuesta común) y fichas de ajuste + historial de convergencia + tablas de sanidad (`sanidad_*.csv`) en `results/sinteticos/generadores/`. Mide sanidad, no admite generadores (eso es `16`). `16`–`18`: esqueletos (validación, laboratorio, aumento) |
| **F — cierre** | `19_decision_final` → `20_pseudolive` | sistema congelado + `results/pseudolive/` (esqueletos) |

**Orden de ejecución en la fase D.** La numeración es de *lectura* (familias antes de la comparativa),
pero tras un benchmark nuevo hay que ejecutar `12_comparativa` **antes** que `05`–`11`: es 12 quien
escribe `results/benchmark/ranking*.csv`, y los notebooks de familia lo leen
(`regimenes.informes.cargar_resultados_familia`) y fallan pidiendo re-ejecutar 12 si no corresponde a
las métricas en caché. Los notebooks de fusión no verifican la huella: `13_fusion_d07_d08` lee
directamente los paneles OOS (`load_panel`) y `14_fusion_d02_d06` usa el cribado versionado si faltan
paneles.

Los notebooks v1 de la Capa 1 (un notebook por detector) se resumen en los notebooks de familia
05–11 y quedan íntegros en el tag `capa1-final`.
