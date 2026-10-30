# Glosario — términos del proyecto

> Definiciones **echadas a tierra** de los términos que gobiernan todo el repositorio. Es la **fuente
> única** de definiciones: los notebooks y el resto de documentos enlazan aquí en vez de redefinir.
> Solo contiene términos. El pipeline de notebooks `00`–`23`, el orden de ejecución y el mapa del código y de
> las rutas están en [`fase1/README.md`](fase1/README.md); la visión del TFM, en [`tfm/README.md`](tfm/README.md);
> el índice de la documentación, en [`README.md`](README.md).

---

## Términos del TFM

Visión del sistema y estado de cada fase: [`tfm/README.md`](tfm/README.md). Encuadre: [ADR-005](decisions/ADR-005-reencuadre-tfm-multiagente.md).

| Término | Definición |
|---|---|
| **TFM · fases 1–5** | 1 régimen de mercado (`src/regimenes`) · 2 ingesta, RAG y tools (`src/ingesta`) · 3 agentes Macro, Equity y Risk · 4 cartera y memo (Portfolio y CIO) · 5 evaluación (backtest con TimeGate frente a *baselines*). Las etapas internas de la Fase 1 (1–4 de datos, D, E, S, F) son **sub-fases** de la Fase 1; ver [abajo](#fase-no-es-siempre-la-fase-del-tfm) los otros usos de «fase» |
| **agentes** | LLM orquestados con LangGraph: **Macro** (contexto macroeconómico), **Equity** (acciones y sectores), **Risk** (riesgo, usa el régimen), **Portfolio** (propone la cartera) y **CIO** (decide y redacta el memo, aplicando los límites del cliente). Detalle en [`tfm/arquitectura.md`](tfm/arquitectura.md) |
| **RAG** | *retrieval-augmented generation*: el agente recupera pasajes de un corpus con fecha de publicación y los cita en lugar de responder de memoria |
| **tool** | función que un agente puede invocar (p. ej. `get_regimen(fecha)` o una búsqueda sobre el corpus) |
| **skill** | paquete reutilizable de instrucciones y procedimientos que un agente carga para una tarea concreta |
| **TimeGate** | barrera temporal dentro de cada tool: solo devuelve información **publicada** hasta la fecha consultada. Es el mismo principio que el **lag de publicación** (`LAG_PUBLICACION`, [ADR-003](decisions/ADR-003-causalidad-calendario-estado-ranking.md)) pero aplicado al corpus y a las series de los agentes y no a los paneles de los detectores. Contrato común en [`fase2/README.md`](fase2/README.md) §4 |
| **`fecha_publicacion`** | día en que un dato o documento se hizo público (no el del hecho); el campo que filtra el TimeGate. Su contrapartida es `fecha_referencia`, el periodo al que se refiere |
| **vintage** | versión de un dato tal como se publicó en una fecha (p. ej. ALFRED guarda cada revisión del empleo de EE. UU.); permite leer una serie revisable como se conocía en el pasado |
| **RAG base / cajón** | los dos índices de cada agente: la **base** es fija y se prepara antes del backtest; el **cajón** se llena con lo que traen sus tools y persiste entre ejecuciones ([`fase2/plan_datos.md`](fase2/plan_datos.md)) |
| **`get_regimen(fecha)`** | tool de salida de la Fase 1: estado, probabilidad y días en el estado, con solo información publicada hasta `fecha`. Los estados los fija el detector final (notebook 22, futura ADR-006); **equivalencia provisional**: calma / alerta / crisis ↔ normal / vigilancia / confirmado de `regimenes.fusion` |
| **investment memo** | informe final del CIO Agent con la decisión de cartera, su justificación y citas a las fuentes |
| **perfil de cliente** | formulario (idea: cuestionario de idoneidad tipo MiFID II, diseño abierto) cuyas respuestas son los **límites** del CIO Agent. Ver [`tfm/cliente_y_carteras.md`](tfm/cliente_y_carteras.md) |
| **cartera con límites / en bruto** | las dos carteras por decisión: la que respeta los límites del perfil del cliente y la que recomiendan los agentes sin ellos (qué restricciones conserva, por decidir) |
| **memoria de cartera** | estado persistente: cada cartera conoce la anterior como entrada de los agentes (formato por decidir) |
| **botón de cataclismo** | alerta automática (modo broma) a partir de [ews.kylemcdonald.net](https://ews.kylemcdonald.net), que vigila vuelos de jets privados: si «los millonarios huyen», se desinvierte todo. Histórico desde octubre de 2025: solo en vivo o demostración |
| **System One** | línea de investigación: modelos de decisión rápidos y tipados que resuelven decisiones simples antes de escalar al LLM («System Two»). Ver [`tfm/lineas_abiertas.md`](tfm/lineas_abiertas.md) |

Datos alternativos candidatos: [`fase2/datos_alternativos.md`](fase2/datos_alternativos.md).

### Capa 1 del repositorio (no es la «Capa 1» de la propuesta)

«**Capa 1**» en este repositorio es la **primera vuelta exploratoria de los 12 detectores v1**
([`historia/capa1/`](historia/capa1/README.md), tag `capa1-final`). **No** es la «Capa 1» de la propuesta
(universo de 42 ETFs) ni guarda relación con su «Capa 2» (~1.300 acciones). Se mantiene el nombre por
coherencia con el tag y la historia ([ADR-005](decisions/ADR-005-reencuadre-tfm-multiagente.md) §2.4).

### «Fase» no es siempre la fase del TFM

En el repositorio conviven cinco numeraciones de fases. Sin más contexto, «Fase 2» es la del TFM. **P1–P5 solo
designa fases del TFM**; las demás listas numeradas usan otra letra (última tabla de esta sección).

| Uso | Dónde aparece | Qué significa |
|---|---|---|
| **Fase 1–5 del TFM** | `tfm/`, `fase1/`, `fase2/`, ADR-005 y ADR-007 | las cinco fases del sistema multi-agente; **Fase 2** = ingesta, RAG y tools (`src/ingesta`) |
| **FASE 0–5 de la Capa 1** (en mayúsculas) | documentos heredados de la Capa 1: fichas de familia, narrativa v1 de los detectores, estado del arte v1 | etapas de la primera vuelta ([`historia/capa1`](historia/capa1/README.md) §1): 0 estructura y evaluador · 1 datos y EDA · **2 estado del arte** · 3 implementación de los 12 detectores · 4 síntesis comparativa · 5 pulido |
| **fases 1–4 del re-base de datos** | [ADR-001](decisions/ADR-001-rebase-datos.md) §6 | 1 reorganización · 2 datos (estado del arte y descarga) · 3 EDA y banco congelado · 4 preprocesado |
| **sub-fases D, E, S, F** | notebooks `04`–`23`, [`fase1/README.md`](fase1/README.md) | detectores, fusión, sintéticos y cierre de la Fase 1 |
| **P1–P5** | [`estado_del_arte/`](estado_del_arte/README.md): dossier, `obras.csv` (`relevancia_fase`), `fichas.json` y notas de `references.bib` | alias de las Fases 1–5 del TFM, para no chocar con las familias F1–F7 ni con la sub-fase F: P1 régimen · P2 ingesta, RAG y tools · P3 agentes · P4 cartera y memo · P5 evaluación y backtest |

Otras listas numeradas que no son fases:

| Uso | Dónde aparece | Qué significa |
|---|---|---|
| **L1–L5** | [`fase2/plan_datos.md`](fase2/plan_datos.md) §7.1 | pruebas del protocolo de look-ahead del LLM: L1 sondas · L2 datación · L3 ablación · L4 fuera del corte · L5 modelo con corte |
| **Q1–Q11** | [`fase2/plan_datos.md`](fase2/plan_datos.md) §8 | preguntas abiertas del plan de datos (en el LaTeX local del plan se numeraban P1–P11) |
| **P1–P6 de *The Alpha Illusion*** | [`estado_del_arte/estado_del_arte.md`](estado_del_arte/estado_del_arte.md) | protocolo de una obra citada; se nombra siempre con la obra, nunca solo |

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
  caen fuera de ventana de crisis), no FP/(FP+TN). La línea base que marca *siempre* crisis tiene un
  `false_alarm_rate` ≈ 0,81 (A) y ≈ 0,83 (B), porque su precisión es la **tasa base** (`det_base_rate`
  de `ranking_v2.csv`): ≈ 0,19 en A y ≈ 0,17 en B.
- **Ranking principal (ADR-003):** `score_deteccion` = F1 entre precisión diaria y **recall por
  evento** (crisis detectada = ≥ 3 sesiones consecutivas marcadas dentro de su ventana), tras filtrar
  por niveles (precisión que no supera al azar, parpadeo, salida degenerada).
- **`rank_medio`** (legacy) estaba sesgado hacia la inactividad: la línea base "siempre crisis"
  quedaba 4ª en ambas pistas con el banco previo a ADR-003
  y 5ª en ambas con el banco re-ejecutado (`14_comparativa` §8). Se conserva solo como columna de
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
| **Pareto** | — | frontera de Pareto recall por evento / precisión: detectores a los que ningún otro supera en ambos ejes a la vez. Vista complementaria del ranking (`14_comparativa` §4), sin ganador único |

**Gate / preflight.** *Gate* = comprobación bloqueante: si falla, el notebook se detiene en vez de
seguir con datos parciales (gates de causalidad y de paneles en `03_preprocesado`; gate de
comparabilidad de `14_comparativa` §2: mismo intervalo OOS por pista, sin duplicados, trazable a la
ejecución). *Preflight* = el gate previo al benchmark (`regimenes.benchmark.preflight`,
`05_protocolo_evaluacion` §6): comprueba paneles, columnas y dependencias opcionales **sin ajustar
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
[`fase1/sinteticos.md`](fase1/sinteticos.md).

| Término | Definición |
|---|---|
| **régimen de referencia sintético** | etiqueta binaria con la que se ajustan **todos** los generadores: 1 = el día cae dentro de alguna ventana `[pico, suelo]` de `crisis_windows` de la pista (`configs/benchmark_spec.yaml`), 0 = calma (`sinteticos.datos.regimen_referencia`). **Nunca sale de un detector** (sería circular: ese detector jugaría en casa sobre lo generado) |
| **semántica del régimen sintético** | en el histórico `regime = 1` es «tramo pico→suelo de una crisis del catálogo»; en una trayectoria sintética es **«día extraído de la ley condicional de crisis»**: el tramo no tiene por qué ir de un máximo a un mínimo del precio generado. El laboratorio mide si un detector reconoce ese cambio de ley, no el recall por evento del benchmark |
| **corte de entrenamiento** (`fin_train`) | última fecha que ve un generador: pista A 2006-12-31, pista B 2017-12-31 (`configs/sinteticos.yaml`). No puede partir un episodio de crisis |
| **cadena de régimen** | el régimen es exógeno a los generadores: o se **impone** la secuencia al muestrear, o se **simula** con la cadena estimada en train (Markov con duraciones geométricas, o semi-Markov con las duraciones empíricas de las rachas), arrancando del último estado de train o de la distribución estacionaria |
| **espacio de generación** (de trabajo) | lo que modelan los generadores: el log-retorno crudo del S&P 500 (`SP500_ret`) más las columnas **no derivables** del núcleo, estandarizadas solo con train. Las cuatro features deterministas de la senda del S&P 500 (`SP500_ret_z`, `SP500_vol_z`, `SP500_momentum`, `SP500_drawdown`) **no** se modelan |
| **re-derivación** | vuelta al espacio público: esas cuatro features se recalculan con las mismas primitivas causales de `regimenes.features` sobre [historia real del S&P 500 hasta `fin_train`] + [retornos sintéticos]. Sobre la historia real reproduce el panel con error 0 (`error_rederivacion_`). Cada trayectoria es una continuación hipotética del mercado tras el corte |
| **siguiente bloque** | formulación de RBIG y de los generadores neuronales: se modela la ley del bloque de 21 sesiones siguiente condicionada al contexto previo (21 sesiones en los neuronales, 5 en RBIG; `configs/sinteticos.yaml`) y al régimen de cada día del bloque, y una trayectoria larga se obtiene **encadenando** bloques (el bloque generado pasa a ser contexto del siguiente; el primer contexto son las últimas filas reales de train) |
| **TSTR / TRTR** | *train on synthetic, test on real* / *train on real, test on real*: un detector ajustado en una trayectoria sintética (TSTR) o en el real (TRTR) y evaluado en los mismos días fuera de muestra del tramo real de entrenamiento con `score_deteccion` (ADR-003). La referencia es el TRTR walk-forward; el **TRTR fijo** (un solo ajuste con todo el real) separa el efecto de los datos del del protocolo. En el notebook 16 es **informativo**: los detectores son no supervisados (mide si la ley marginal permite calibrarlos, no si el sintético transmite la señal de crisis) y un control negativo de filas reales i.i.d. aprueba la regla en la pista A (en la B la suspende, pero con 3 crisis evaluables el TSTR de B tampoco tiene potencia) |
| **discriminador (C2ST)** | *classifier two-sample test*: AUC fuera de pliegue de un clasificador que separa ventanas reales de sintéticas (0,5 = indistinguibles). En el veredicto, AUC global con todas las columnas; en crisis solo se lee el orden. Un AUC por debajo de 0,5 es un **indicio de copia**, no de fidelidad (`validacion.discriminador`) |
| **memorización (cociente NN)** | mediana de la distancia de cada día sintético a su vecino real más cercano dividida por la mediana de la distancia real–real (excluidos los vecinos a menos de 21 sesiones), con el índice de vecindad real **íntegro** y en el espacio de las columnas **modeladas**. Cerca de 1 = días nuevos; bajo = copia o infradispersión (`validacion.memorizacion`) |
| **apto para laboratorio** (`apto_laboratorio`) | nivel del veredicto del notebook 19: pasa fidelidad marginal, de dependencia, condicionamiento y discriminador; el generador sirve para el laboratorio con verdad de régimen (notebook 20) |
| **laboratorio condicionado** (`laboratorio_condicionado`) | nivel declarado después de ver el veredicto: lo mismo que `apto_laboratorio` sin el discriminador. El 17 puede usar esos generadores **advirtiendo** que un clasificador los distingue del real |
| **apto para aumento** (`apto_aumento`) | nivel del veredicto: pasa memorización y condicionamiento (igual en las dos pistas); el generador no copia el entrenamiento y lleva la señal de régimen (notebook 21) |
| **simulación controlada** | uso del notebook 20: como ningún generador salió apto para laboratorio (19), los tres paramétricos (`gaussiano_regimen`, `var_regimen`, `garch_regimen`) se usan como **mundos de juguete** con verdad-terreno exacta, no como réplica del mercado |
| **intensidad λ** | factor con el que el laboratorio acerca la ley de las filas de crisis a la de calma (`escenarios.atenuar`): λ = 1 deja la trayectoria tal cual, λ = 0 elimina la diferencia entre regímenes |
| **H1–H4** | hipótesis pre-registradas del laboratorio: H1 recuperación (los detectores reconocen la crisis en la celda fácil), H2 degradación (el score baja al acortar y atenuar la crisis), H3 concordancia con el ranking de ADR-003 sobre datos reales, H4 circularidad (un detector no gana más en el mundo generado por su propia familia) |
| **circularidad** | ventaja de un detector «de casa» sobre trayectorias generadas por un modelo de su misma familia (p. ej. D03 GMM con `gaussiano_regimen`); es la razón de que el régimen de referencia nunca salga de un detector |
| **retraso** | en el laboratorio, sesiones entre el inicio de un episodio de crisis impuesto y la primera detección (`retraso_medio`, `laboratorio.metricas`) |
| **fechas sintéticas** | días hábiles de lunes a viernes posteriores al corte; son una etiqueta ordenada, **no** el calendario de la NYSE: no se cruzan por fecha con datos reales |

## Familias e IDs de detectores

`Fk` = familia de la teoría (F1 reglas/umbrales, F2 clustering, F3 HMM, F4 Markov-Switching, F5 GARCH, F6
change-point, F7 redes). `Dnn` = detector: D01–D12 en el benchmark; D13 `hsmm_tstudent` solo como ablación de D08,
fuera del ranking y de la huella de caché. Los generadores sintéticos no son una familia de detectores (antes se
numeraban «F8»; hoy están en [`fase1/sinteticos.md`](fase1/sinteticos.md)). Qué detector pertenece a cada familia,
con su código y su notebook: [`fase1/README.md`](fase1/README.md#índice-de-detectores-por-familia).
