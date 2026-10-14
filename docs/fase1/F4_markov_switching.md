# F4 — Markov-Switching econométrico

> **Cómo leer este documento.** La sección 1 es la teoría de la familia (redactada en la Capa 1, original en el tag `capa1-final`); la 2, los detectores implementados con su configuración v2 y sus resultados del benchmark (ADR-003): [D05 `markov_switching_var`](#d05). Notebook de familia: [`notebooks/08_familia_F4_markov_switching.ipynb`](../../notebooks/08_familia_F4_markov_switching.ipynb). Bibliografía: claves `ms_*` y compartidas de [`docs/references.bib`](../references.bib).
>
> **Datos v1 en la teoría.** Las cifras del EDA que cita la sección 1 son de la Capa 1 (15 features, ventana 2007-04 → 2026-06, 4 crisis: 2008, 2011, 2020 y 2022), igual que los nombres de feature `credit_spread_z` o `yield_slope_z` y las referencias a la «tarea previa»; donde el dato importa se indica junto a él. El panel v2 tiene dos pistas (A: 1962-01-02 → 2026-05-29; B: 2007-04-11 → 2026-05-29) y el EDA vigente está en [`eda.md`](eda.md) (curtosis de exceso del S&P 500 = 18,7 en 1928-2026).

## 1. Teoría

> Familia: tradición ECONOMÉTRICA de series temporales con cambio de régimen
> gobernado por una cadena de Markov oculta — Hamilton (1989), MS-AR, Markov-
> Switching de medias y/o varianzas, filtro de Kim, e implementación moderna en
> `statsmodels` (`MarkovRegression` / `MarkovAutoregression`). El HMM con
> emisiones (hmmlearn) y el RS-GARCH se tratan en otras fichas; aquí se señala el
> solape con ellos.

### Definición y supuestos

Un modelo Markov-Switching (MS) describe una serie temporal cuyos parámetros
(media, términos autorregresivos y/o varianza del error) **conmutan entre un
número finito de regímenes** `K`, donde el régimen activo en cada instante `S_t`
es una variable latente discreta que sigue una **cadena de Markov de primer
orden** con matriz de transición `P` fija (homogénea en el tiempo). La
formulación canónica es la de Hamilton (1989): el régimen no es observable y se
infiere por máxima verosimilitud mediante un **filtro recursivo** (el "filtro de
Hamilton") que entrega las probabilidades de cada estado condicionadas a la
información disponible.

Supuestos centrales:

1. **Markov de primer orden**: `P(S_t = j | S_{t-1} = i, S_{t-2}, ...) =
   P(S_t = j | S_{t-1} = i) = p_{ij}`. La probabilidad de estar mañana en un
   régimen depende solo del régimen de hoy → induce **persistencia** vía la
   diagonal de `P` y **duraciones esperadas** geométricas `1/(1-p_{ii})`.
2. **Distribución condicional paramétrica por régimen**, típicamente
   **gaussiana** (`y_t | S_t=k ~ N(μ_k + términos AR, σ_k^2)`). Es el supuesto
   más cuestionado para datos financieros (ver §Fortalezas y debilidades).
3. **Número de regímenes `K` fijado a priori** (no estimado dentro del modelo;
   se elige por criterios de información o teoría económica).
4. **Transiciones exógenas** en la versión clásica (`P` constante); existen
   extensiones con probabilidades de transición dependientes del tiempo o de
   covariables (time-varying transition probabilities, TVTP).

Diferencia clave con un AR/ARMA estándar: el MS captura **no linealidad por
tramos** (la dinámica es lineal *dentro* de cada régimen pero el proceso global
es no lineal) y permite cambios bruscos de comportamiento sin imponer una rotura
estructural en una fecha fija conocida.

### Variantes principales

**1. Hamilton MS / Markov-Switching de la media (MSM).** Versión original:
conmutan la media (y opcionalmente la varianza) de un proceso. Hamilton (1989)
la aplicó al crecimiento del PIB de EE. UU. modelando expansiones y recesiones
como dos estados de media alta/baja. En finanzas, la versión más simple es un
MS sobre los retornos: régimen "calma" (media positiva, varianza baja) vs
"crisis/bear" (media negativa o nula, varianza alta).

**2. Markov-Switching Autoregressive (MS-AR).** Hamilton (1989) en su forma
completa: un proceso AR(p) cuyos coeficientes AR, la media y/o la varianza
conmutan con el régimen. En `statsmodels` esto es `MarkovAutoregression`. La
presencia de términos AR hace el filtro más costoso (hay que integrar sobre las
combinaciones de estados de los últimos `p` periodos).

**3. MS de medias vs de varianzas vs ambos.** Tres configuraciones distintas:
   - **Switching solo en media** (`μ_k`): capta cambios de tendencia/drift
     (bull vs bear) pero asume una única volatilidad.
   - **Switching solo en varianza** (`σ_k^2`): capta clústeres de volatilidad
     (calma vs estrés) manteniendo media común. Muy relevante para retornos
     financieros, donde el cambio de *régimen de volatilidad* es más marcado y
     persistente que el cambio de media.
   - **Switching en ambos**: el más usado en regímenes de mercado (bull de baja
     vol vs bear de alta vol). Es la configuración natural para este proyecto.
   En `statsmodels`, `switching_variance=True` activa la conmutación de varianza.

**4. Filtro de Hamilton y filtro de Kim.** El **filtro de Hamilton** es la
recursión hacia delante que produce las probabilidades **filtradas**
`P(S_t=k | y_1,...,y_t)` (solo pasado → causales) y, como subproducto, la
verosimilitud para estimar los parámetros por ML. El **filtro/suavizador de Kim
(Kim, 1994)** añade un paso de **suavizado** hacia atrás que produce las
probabilidades **suavizadas** `P(S_t=k | y_1,...,y_T)` usando **toda** la muestra
(pasado y futuro → anti-causales); Kim (1994) además generaliza el esquema a
modelos en espacio de estados con Markov-switching (combina filtro de Kalman y
filtro de Hamilton mediante una aproximación de colapso). Kim & Nelson (1999) es
el tratamiento de referencia (enfoques clásico por ML y bayesiano por Gibbs).

**5. Número de regímenes `K`.** No se estima dentro del modelo; se selecciona
por **criterios de información (AIC/BIC)**, verosimilitud penalizada o por
teoría económica. Los tests de hipótesis sobre `K` (p. ej. 1 vs 2 regímenes)
sufren un **problema de parámetros de molestia no identificados bajo la
hipótesis nula** (las probabilidades de transición no están identificadas si el
segundo régimen no existe), lo que invalida el test de razón de verosimilitud
estándar; se usan tests específicos (Hansen, Garcia) o BIC. En la práctica
financiera lo habitual es `K=2` (calma/crisis) o `K=3-4` (bull · normal · bear ·
estanflación).

### Fortalezas y debilidades

**¿Capta crisis rápidas?** Parcialmente. El MS hereda el comportamiento del HMM:
la **persistencia** impuesta por la diagonal alta de `P` favorece episodios
largos y de alta varianza y tiende a **alisar / ignorar correcciones cortas**.
Esto es exactamente lo observado en la tarea previa (HMM gaussiano acertaba 2008
y 2020 pero se perdía el taper tantrum 2013 y el Q4-2018). El switching de
varianza ayuda a reaccionar a saltos de volatilidad, pero la entrada en el
régimen de estrés sigue siendo gradual: la probabilidad filtrada necesita varias
observaciones de cola para superar el umbral. Las probabilidades filtradas
reaccionan **con retardo** frente a las suavizadas.

**¿Flickering (conmutación espuria)?** El MS lo controla **mejor que un
clustering sin memoria** precisamente por la matriz de transición: una `p_{ii}`
alta penaliza los saltos de un día y produce regímenes pegajosos. El control del
flickering está pues **parametrizado** (es endógeno, vía `P`), a diferencia de
k-means/GMM que clasifican cada día independientemente. El riesgo opuesto es la
**inercia**: regímenes demasiado persistentes que tardan en salir.

**¿Asume normalidad y cómo se extiende a colas?** La versión estándar asume
**normalidad condicional dentro de cada régimen**. El propio mecanismo de mezcla
de gaussianas con varianzas distintas genera una distribución incondicional
**leptocúrtica y con colas más gordas** que una normal simple — es una de las
razones por las que el MS ajusta retornos mejor que un AR homocedástico. Aun
así, con curtosis de exceso ≈ 25.6 (S&P 500) y ≈ 39.6 (HYG) en el EDA v1 (2007-2026; 18,7
para el S&P 500 en 1928-2026 según el EDA v2), la mezcla
de 2-3 gaussianas **no basta** para reproducir las colas; los saltos extremos
intra-régimen se infra-modelan. Extensiones documentadas: (i) **emisiones
t-Student por régimen** (colas pesadas condicionales), (ii) **switching de
varianza** combinado con más regímenes, y (iii) **MS-GARCH / RS-GARCH** que
añade heterocedasticidad dentro del régimen (esto último es competencia de la
ficha de volatilidad). `statsmodels` solo ofrece emisiones gaussianas → para
colas t hay que salir de la librería estándar (criterio del equipo).

**¿Causal / online: filtradas vs suavizadas?** Distinción crítica para este
proyecto:
   - **Probabilidades FILTRADAS** `P(S_t | y_{1:t})`: usan **solo el pasado** →
     **causales**, aptas para detección online y evaluación walk-forward. Es la
     salida que debe usarse para una señal de régimen en tiempo real.
   - **Probabilidades SUAVIZADAS** `P(S_t | y_{1:T})`: usan **toda la muestra
     (incluido el futuro)** → **anti-causales**. Producen una datación de
     regímenes históricamente más nítida y son las que aparecen en la mayoría de
     gráficos de papers, **pero introducen look-ahead** si se usan como señal.
   Un error común es entrenar/decodificar in-sample con probabilidades suavizadas
   y reportar lo bien que "detecta" crisis — eso es mirar el futuro. El marco
   causal de este TFM exige **filtradas** (y, además, reestimar los parámetros en
   esquema expanding/rolling para no filtrar parámetros del futuro). `statsmodels`
   expone ambas: `filtered_marginal_probabilities` y
   `smoothed_marginal_probabilities`.

**¿Univariante vs multivariante?** La tradición MS-AR econométrica es
fundamentalmente **univariante**: modela UNA serie (un retorno, el crecimiento
del PIB). Existe el **MS-VAR** (Krolzig) para vectores, pero el número de
parámetros crece rápidamente y la estimación se vuelve frágil con muchas series.
Frente al panel multivariante de features causales (15 en la Capa 1; 9/14 por pista en v2), un MS-AR directo no es el
vehículo natural: o se aplica a la serie de retornos del S&P 500 (univariante,
interpretable) o se reduce el panel a pocas series. Aquí el **HMM multivariante
gaussiano** (tarea previa, 7 features) tiene ventaja dimensional; el MS aporta
**interpretabilidad econométrica** y un marco de inferencia formal sobre una
serie focal. (Solape con HMM: ver §Idoneidad.)

Otras debilidades: superficie de verosimilitud **multimodal** (muchos óptimos
locales → requiere múltiples inicializaciones); identificación de `K` mal
condicionada; sensibilidad a la especificación (media vs varianza switching);
las etiquetas de régimen son intercambiables (label switching) y requieren
ordenación post-hoc por criterio económico (igual que en HMM).

### Idoneidad para este proyecto

**Encaje con el EDA.** El switching de **varianza** ataca directamente el
hallazgo de fat tails y clústeres de volatilidad: un régimen de baja varianza y
otro de alta varianza reproducen la amplificación 2-4× de la vol del S&P 500 en
crisis observada en la tarea previa. La distinción **filtradas vs suavizadas**
resuelve de forma limpia uno de los problemas señalados en el resumen de la
tarea (look-ahead, in-sample, Viterbi duro sin probabilidades): el MS entrega
**probabilidades filtradas continuas** (una "probabilidad de crisis" causal en
[0,1]), no una etiqueta dura. La correlación S&P/bonos que cambia de signo
(Gulko 2002) no es algo que un MS univariante sobre una serie capture
directamente — sería un argumento a favor de un detector multivariante o de
incluir esa correlación como serie/feature.

**Carácter interpretable.** Es la gran baza frente a métodos de caja negra: cada
régimen tiene parámetros económicamente legibles (media, varianza, persistencia,
duración esperada), la matriz de transición es directamente interpretable, y la
inferencia es por máxima verosimilitud con tests y criterios de información
estándar. Sirve como **baseline econométrico interpretable** contra el cual
contrastar el HMM multivariante y métodos avanzados.

**Relación / solape con HMM.** El MS gaussiano y el HMM
gaussiano son **matemáticamente parientes**: ambos tienen un estado latente
discreto con dinámica de Markov y emisiones paramétricas, y ambos se estiman con
el mismo aparato (forward filtering + EM/Baum-Welch ≡ filtro de Hamilton + ML).
Diferencias prácticas de tradición, no de fondo:
   - **MS-AR (econometría / statsmodels)**: típicamente **univariante**, énfasis
     en **estructura AR** y en switching de coeficientes de regresión, marco de
     inferencia ML con tests; probabilidades filtradas/suavizadas explícitas.
   - **HMM (ML / hmmlearn)**: típicamente **multivariante** (emisiones
     gaussianas/t/GMM sobre un vector de features), énfasis en decodificación
     (Viterbi) y en `predict_proba`; sin componente AR por defecto.
   Conviene tratarlos como **dos puntos del mismo continuo** y
   evitar contarlos como familias independientes: la decisión real es
   *univariante-AR-interpretable* (MS) vs *multivariante-emisiones-flexibles*
   (HMM). Para el panel multivariante, HMM; para una señal interpretable sobre
   retornos del S&P 500, MS.

### Aplicaciones documentadas a regímenes de mercado financiero

- **Hamilton (1989)**: paper seminal; aunque aplicado al PIB, define el aparato
  que toda la literatura financiera reutiliza.
- **Ang & Bekaert (2002)** (clave ya existente `angbekaert2002`): regímenes en
  correlaciones y asignación internacional de activos con MS.
- **Guidolin & Timmermann (2007)** (clave existente `guidolintimmermann2007`):
  asignación de activos con cuatro regímenes (crash, slow growth, bull, recovery)
  en retornos conjuntos de acciones y bonos — ejemplo paradigmático de MS
  multivariante con `K>2`.
- **Guidolin (2011)**, *Markov Switching Models in Empirical Finance*: survey de
  referencia sobre MS en retornos de acciones, estructura temporal de tipos,
  tipos de cambio y proceso conjunto acciones-bonos; cubre capacidad de ajuste,
  filtrado de regímenes, contraste de hipótesis y desempeño predictivo.
- **Ang & Timmermann (2012)**, *Regime Changes and Financial Markets* (Annual
  Review of Financial Economics): survey accesible que justifica por qué los
  mercados cambian de comportamiento de forma abrupta y persistente, y cómo el MS
  lo modela; conecta regímenes econométricos con cambios de política/regulación.
- **Hardy (2001)**, *A Regime-Switching Model of Long-Term Stock Returns*: MS
  lognormal de 2 regímenes ajustado por ML a S&P 500 y TSX 300; compara con GARCH
  y lo aplica a garantías de productos equity-linked. Ejemplo claro de MS de
  media+varianza en retornos de equity.
- **Kim & Nelson (1999)**: aplicaciones a ciclo económico, Great Moderation y
  series financieras con los filtros clásico y bayesiano.

### Coste de implementación y librería Python recomendada

**Librería recomendada: `statsmodels.tsa.regime_switching`.**
   - `MarkovRegression`: MS sin componente AR (regresión dinámica con cambio de
     régimen; permite switching en constante/tendencia, en coeficientes de
     regresores exógenos y en varianza vía `switching_variance=True`).
   - `MarkovAutoregression`: MS-AR completo (Hamilton 1989) con `order=p` y
     coeficientes AR conmutables.
   - Salidas relevantes: `filtered_marginal_probabilities` (causales, para online
     y walk-forward), `smoothed_marginal_probabilities` (full-sample, solo para
     análisis histórico), `expected_durations`, matriz de transición estimada,
     AIC/BIC para selección de `K`.

**Coste y limitaciones prácticas:**
   - **Bajo coste de cómputo** para `K=2-3` y una serie univariante: la
     estimación es de segundos a minutos. El MS-AR con `order` alto y `K` grande
     encarece el filtro (integra sobre estados de los últimos `p` periodos).
   - **Multimodalidad**: la verosimilitud tiene muchos máximos locales →
     **imprescindible** lanzar varias inicializaciones (`fit` admite
     `search_reps` / arranque desde múltiples `start_params`) y quedarse con la de
     mayor log-verosimilitud (igual práctica que las 10 seeds del HMM previo).
   - **Limitaciones de `statsmodels`**: (i) **solo emisiones gaussianas** — no hay
     t-Student ni mixturas nativas, lo que es un problema con las colas del EDA;
     (ii) orientado a **series univariantes** (no MS-VAR; para multivariante hay
     que recurrir a otras herramientas o a HMM); (iii) sin probabilidades de
     transición dependientes de covariables (TVTP) listas para usar; (iv) la
     estimación es **batch**: para walk-forward causal hay que **reestimar** el
     modelo en cada ventana (expanding/rolling) y tomar la última probabilidad
     filtrada, lo que multiplica el coste por el número de pasos.
   - Para regímenes con colas pesadas o RS-GARCH habría que salir de
     `statsmodels` (paquetes especializados o implementación propia) — fuera del
     alcance de esta familia.

**Veredicto de implementación**: barato e interpretable como **baseline
econométrico univariante** (p. ej. MS de media+varianza sobre el retorno del
S&P 500, `K=2-3`, usando probabilidades **filtradas** y reestimación
walk-forward). No sustituye al HMM multivariante para explotar el panel de features,
sino que lo complementa con interpretabilidad y un marco de inferencia formal.

## 2. Detectores implementados

Las tablas de resultados salen de `results/benchmark/ranking_v2.csv` y `results/benchmark/metrics_master_v2.csv` (benchmark completo 24/24, `results/benchmark/manifest.json` del 2026-09-29T12:41 UTC; cifras de [ADR-003](../decisions/ADR-003-causalidad-calendario-estado-ranking.md) §4). Walk-forward causal y evaluación OOS por pista (17 crisis evaluables en la A y 9 en la B, según `configs/benchmark_spec.yaml`). Ranking de **detección**: un evento se detecta con ≥ 3 sesiones OOS consecutivas en crisis dentro de [pico, suelo]; `score_deteccion` = F1 entre la precisión diaria (1 − FAR) y el recall por evento, con niveles previos (elegible › precisión ≤ azar › parpadeo › degenerado). Cada tabla va entre marcadores `resultados_v2`: tras un benchmark nuevo se regenera desde esos dos CSV, no se edita a mano.

<a id="d05"></a>

### D05 — `markov_switching_var` · Markov-Switching de media y varianza

Modelo de Hamilton [`hamilton1989`] sobre el retorno del S&P 500: una cadena de Markov de dos estados gobierna a la vez la media y la varianza (`y_t = μ_{S_t} + ε_t`, `ε_t ~ N(0, σ²_{S_t})`). Crisis = régimen de alta varianza. Es el baseline econométrico interpretable de la familia y es **univariante** (el nombre «var» se refiere a la varianza, no a un VAR).

**Configuración v2**

- Código: `src/regimenes/detectores/f4_switching/markov_switching_var.py` (`MarkovSwitchingVar`); registro, id `D05`.
- Modelo: `statsmodels` `MarkovRegression` con `k_regimes=2`, `trend='c'`, `switching_variance=True`, retorno ×100, `search_reps=0`. Ajuste en train y parámetros congelados; la predicción usa un **filtro forward gaussiano univariante propio** sobre burn-in del train + bloque (probabilidades filtradas, nunca suavizadas).
- Feature: `SP500_ret` en ambas pistas.
- Walk-forward: refit cada **63** sesiones (detector lento), ventana expanding.

<!-- BEGIN resultados_v2 D05 -->
**Resultados v2 (ADR-003)**

| Métrica | Pista A | Pista B |
|---|---:|---:|
| Variante evaluada | `markov_switching_var_2s` | `markov_switching_var_2s` |
| Puesto de detección | **5** / 12 | **1** / 12 |
| Nivel de etiqueta | `elegible` | `elegible` |
| `score_deteccion` | 0.546 | 0.647 |
| Recall por evento | 1.000 (17/17) | 1.000 (9/9) |
| Precisión diaria (lift vs base) | 0.375 (×1.94) | 0.478 (×2.77) |
| Cobertura media de crisis | 0.731 | 0.499 |
| Tasa de falsas alarmas (FAR) | 0.625 | 0.522 |
| Activación media en trampas | 0.032 | 0.000 |
| Switching rate | 0.0683 | 0.0434 |
| Duración media de régimen (sesiones) | 14.6 | 22.9 |
| Estabilidad de etiqueta | 0.995 | 0.997 |
| Puesto legacy (`rank_medio`) | 6 (7.0) | 4 (4.6) |

**Ventana OOS y eventos por pista**

- **Pista A** · OOS 1970-05-12 → 2026-05-29 (n = 14133) · train inicial 2016 sesiones, refit cada 63 (expanding) · 1 feature(s): `SP500_ret`.
  - Detectados (17): `bear_1969_70`, `oil_stagflation_1973`, `volcker_1980_82`, `black_monday_1987`, `gulf_war_sl_1990`, `ltcm_russia_1998`, `dotcom_2000_02`, `gfc_2007_09`, `flash_crash_euro1_2010`, `us_downgrade_euro2_2011`, `china_oil_2015_16`, `volmageddon_2018q1`, `fed_tightening_2018q4`, `covid_2020`, `inflation_bear_2022`, `svb_banking_2023`, `tariff_selloff_2025`.
  - No detectados (0): ninguno.
- **Pista B** · OOS 2010-04-12 → 2026-05-29 (n = 4059) · train inicial 756 sesiones, refit cada 63 (expanding) · 1 feature(s): `SP500_ret`.
  - Detectados (9): `flash_crash_euro1_2010`, `us_downgrade_euro2_2011`, `china_oil_2015_16`, `volmageddon_2018q1`, `fed_tightening_2018q4`, `covid_2020`, `inflation_bear_2022`, `svb_banking_2023`, `tariff_selloff_2025`.
  - No detectados (0): ninguno.
<!-- END resultados_v2 D05 -->

Narrativa de la Capa 1 (implementación v1, hallazgos e hipótesis del CHECKPOINT 2): [`historia/fichas_v1.md`](../historia/fichas_v1.md#d05).

## 3. Variantes no implementadas

- **MS-VAR (Krolzig):** versión multivariante del MS-AR; coste en parámetros alto. (Ojo: el «MS-VAR» D05 del repositorio es univariante, ver GLOSARIO.)
- **TVTP** (Filardo; Diebold-Lee-Weinbach): probabilidades de transición dependientes de covariables como VIX o pendiente de la curva; no está en `statsmodels` nativo.
- **Emisiones t-Student en MS** (colas del EDA) y **tests del número de regímenes** (Hansen 1992; Garcia 1998), útiles para justificar `K` más allá del BIC; ambos exigen implementación propia.

## 4. Referencias

- Hamilton, J. D. (1989). *A New Approach to the Economic Analysis of
  Nonstationary Time Series and the Business Cycle*. Econometrica 57(2),
  357–384. DOI 10.2307/1912559. [`hamilton1989`]
- Kim, C.-J. (1994). *Dynamic Linear Models with Markov-Switching*. Journal of
  Econometrics 60(1-2), 1–22. DOI 10.1016/0304-4076(94)90036-1. (Filtro/suavizador
  de Kim.) [`ms_kim1994`]
- Kim, C.-J. & Nelson, C. R. (1999). *State-Space Models with Regime Switching:
  Classical and Gibbs-Sampling Approaches with Applications*. MIT Press. ISBN
  978-0-262-11238-2. [`ms_kimnelson1999`]
- Guidolin, M. (2011). *Markov Switching Models in Empirical Finance*. En
  Advances in Econometrics, vol. 27B (Missing Data Methods), Emerald, 1–86.
  DOI 10.1108/S0731-9053(2011)000027B004. [`ms_guidolin2011`]
- Ang, A. & Timmermann, A. (2012). *Regime Changes and Financial Markets*.
  Annual Review of Financial Economics 4, 313–337.
  DOI 10.1146/annurev-financial-110311-101808. [`hmm_angtimmermann2012`]
- Hardy, M. R. (2001). *A Regime-Switching Model of Long-Term Stock Returns*.
  North American Actuarial Journal 5(2), 41–53.
  DOI 10.1080/10920277.2001.10595984. [`hmm_hardy2001`]
- statsmodels developers. *Markov switching dynamic regression /
  autoregression models* (`statsmodels.tsa.regime_switching`). Documentación
  oficial. https://www.statsmodels.org/stable/generated/statsmodels.tsa.regime_switching.markov_autoregression.MarkovAutoregression.html [`ms_statsmodels`]
- Ang, A. & Bekaert, G. (2002). [`angbekaert2002`]
- Guidolin, M. & Timmermann, A. (2007). [`guidolintimmermann2007`]
