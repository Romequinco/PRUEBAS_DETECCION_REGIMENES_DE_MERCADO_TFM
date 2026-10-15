# F3 — Modelos ocultos de Markov (HMM)

> **Cómo leer este documento.** La sección 1 es la teoría de la familia (redactada en la Capa 1, original en el tag `capa1-final`); la 2, los detectores implementados con su configuración v2 y sus resultados del benchmark (ADR-003): [D04 `hmm_gaussian_2s`](#d04), [D08 `hmm_tstudent`](#d08), [D13 `hsmm_tstudent`](#d13) (ablación). Notebook de familia: [`notebooks/07_familia_F3_hmm.ipynb`](../../notebooks/07_familia_F3_hmm.ipynb). Bibliografía: claves `hmm_*` y compartidas de [`docs/references.bib`](../references.bib).
>
> **Datos v1 en la teoría.** Las cifras del EDA que cita la sección 1 son de la Capa 1 (15 features, ventana 2007-04 → 2026-06, 4 crisis: 2008, 2011, 2020 y 2022), igual que los nombres de feature `credit_spread_z` o `yield_slope_z` y las referencias a la «tarea previa»; donde el dato importa se indica junto a él. El panel v2 tiene dos pistas (A: 1962-01-02 → 2026-05-29; B: 2007-04-11 → 2026-05-29) y el EDA vigente está en [`eda.md`](eda.md) (curtosis de exceso del S&P 500 = 18,7 en 1928-2026).

## 1. Teoría

> Estado del arte (FASE 2) de la familia **HMM como modelo de espacio de estados
> latente con emisiones** (estilo `hmmlearn`/`pomegranate`). Frontera con el
> **Markov-Switching econométrico** (Hamilton MS-AR, `statsmodels`): se trata en la
> ficha [F4](F4_markov_switching.md) (mismo motor de espacio de estados, distinto énfasis). Referencias: claves `hmm_*` y compartidas de
> [`docs/references.bib`](../references.bib).

---

### Definición y supuestos

Un **HMM** modela una serie observada `y_1,...,y_T` (aquí: vector de features
causales diarias) como generada por una **cadena de Markov latente** de `K`
estados discretos `s_t ∈ {1,...,K}` que no se observa directamente
[hmm_rabiner1989; hmm_zucchini2016]. Tres bloques de parámetros:

1. **Probabilidades iniciales** `π_k = P(s_1 = k)` (`startprob_` en hmmlearn).
2. **Matriz de transición** `A`, con `A_ij = P(s_t = j | s_{t-1} = i)` — homogénea
   en el tiempo (`transmat_`). La **diagonal** gobierna la persistencia/duración
   esperada de cada régimen (`E[duración_i] = 1/(1 - A_ii)`).
3. **Distribución de emisión** `b_k(y_t) = P(y_t | s_t = k)` — en finanzas suele
   ser una densidad continua multivariante (gaussiana, t-Student o mixtura).

**Supuestos clave (y sus grietas para finanzas):**

- **Markov de primer orden**: `s_t` depende solo de `s_{t-1}`. Implica que el
  tiempo de permanencia en un estado es **geométrico** → puede infra-modelar
  regímenes con duraciones no geométricas (lo corrigen los **hidden
  *semi*-Markov**, HSMM [hmm_bullabulla2006]).
- **Independencia condicional de las emisiones**: dado `s_t`, `y_t` es
  independiente del pasado. Los retornos diarios reales tienen
  **autocorrelación en volatilidad (clustering)** que el HMM básico solo captura
  vía cambios de estado, no dentro del estado.
- **Estacionariedad de los parámetros**: `A` y `b_k` fijos en toda la muestra.
  Discutible en mercados que mutan; lo flexibilizan los HMM de parámetros
  variables en el tiempo [hmm_nystrup2017].
- **Forma de las emisiones**: el HMM **gaussiano** asume normalidad condicional.
  Con el EDA de este proyecto (curtosis de exceso S&P 500 ≈ 25.6 y HYG ≈ 39.6 en la ventana v1 2007-2026; el EDA v2 da 18,7 para el S&P 500 en 1928-2026) este
  supuesto **subestima las colas** dentro de cada régimen (ver §Idoneidad).

**Estimación**: máxima verosimilitud por **Baum-Welch** (caso particular de EM)
[hmm_baumwelch1970], que alterna el paso E (forward-backward: probabilidades
posteriores de estado `γ_t(k) = P(s_t = k | y_{1:T})`) con el paso M (re-estima
`π, A, b_k`). Es no convexo → sensible a la **inicialización** y a óptimos
locales; práctica habitual: varias semillas (k-means o aleatorias) y quedarse con
la de mayor log-verosimilitud (en la tarea previa: 10 seeds 42–51).

---

### Variantes principales

#### 1. HMM gaussiano
Emisión `b_k = N(μ_k, Σ_k)`. Es el **baseline** y el modelo de la tarea previa
(`GaussianHMM`, 2 estados, `covariance_type='full'`).

- **`covariance_type`** (hmmlearn): `spherical` (1 varianza), `diag` (varianzas
  por feature, sin correlación), `full` (matriz completa por estado, capta
  **correlaciones entre features dentro del régimen**), `tied` (Σ compartida).
  Para este proyecto `full` es relevante porque la **correlación S&P/bonos cambia
  de signo** (Gulko 2002 [gulko2002]) y una covarianza completa por estado captura
  esos co-movimientos condicionales (p. ej. un estado "risk-off" con correlación
  equity-bono negativa frente a uno "inflación/tightening" con correlación
  positiva). El coste es más parámetros (`K · d(d+1)/2`) → riesgo de
  sobreajuste con pocas observaciones por estado.

#### 2. HMM con emisiones t-Student
Sustituye la normal por una **t multivariante** por estado, con grados de libertad
`ν_k` que controlan el grosor de cola (`ν → ∞` recupera la gaussiana). Es la
respuesta directa a las **fat tails**: Bulla (2011) muestra que las componentes t
**reproducen mejor los hechos estilizados** de retornos diarios, son **robustas a
outliers** y, de forma notable, **aumentan la persistencia** de los estados (menos
*flickering*), porque la cola pesada absorbe los retornos extremos sin forzar un
salto de régimen [hmm_bulla2011]. Hardy (2001) ya documentaba la ventaja de
regímenes para capturar colas y heterocedasticidad en retornos de largo plazo
[hmm_hardy2001].

#### 3. GMM-HMM (emisiones de mixtura)
Cada estado emite según una **mixtura de M gaussianas** (`GMMHMM` en hmmlearn).
Aproxima densidades no gaussianas (asimetría, colas, multimodalidad) sin asumir
una forma paramétrica de cola concreta. Más flexible que el gaussiano simple, pero
**más parámetros** (`K · M` componentes) y **más fronteras de identificabilidad**
(un GMM-HMM puede "imitar" estados extra dentro de un estado). Alternativa práctica
a la t cuando la librería no ofrece emisiones t (ver §Coste).

#### 4. Selección del número de estados `K`
No hay test de razón de verosimilitudes estándar válido (problema de parámetros en
la frontera / regularidad). Criterios habituales [hmm_zucchini2016]:

- **BIC** `= -2·logL + p·log(N)` (penaliza más → modelos más parsimoniosos; suele
  preferirse en finanzas para evitar estados espurios).
- **AIC** `= -2·logL + 2p` (penaliza menos → tiende a más estados).
- **Log-verosimilitud penalizada / validación out-of-sample** y criterio
  económico (¿los estados son interpretables: calma / corrección / crisis /
  estanflación?). hmmlearn ≥0.3 expone `.aic()` y `.bic()`.
- Cuidado: `p` (nº de parámetros) crece rápido con `covariance_type='full'` y con
  `d=15` features → BIC penaliza fuerte; conviene comparar `full` vs `diag` y
  considerar reducción de dimensionalidad o un subconjunto de features.

#### 5. Decodificación: Viterbi (duro) vs posteriores suaves
- **Viterbi** [hmm_viterbi1967] (`model.predict`): secuencia de estados **más
  probable globalmente** dada **toda** la muestra `y_{1:T}`. Asignación **dura**
  (0/1). **Usa el futuro** para decidir el estado de `t` → **anti-causal/suavizado**
  (ver §Causalidad).
- **Forward-backward / posteriores suaves** (`model.predict_proba`): devuelve
  `γ_t(k) = P(s_t = k | y_{1:T})` — una **probabilidad continua** de cada régimen,
  con incertidumbre. También usa toda la muestra (suavizado).
- **Filtrado (forward solo)**: `α_t(k) ∝ P(s_t = k | y_{1:t})` — usa **solo el
  pasado** → es la versión **causal/online**. hmmlearn no expone `predict_proba`
  filtrado directamente, pero se obtiene con `model.score_samples` / el paso
  forward, o re-estimando en ventana (ver §Coste y §Causalidad).

---

### Fortalezas y debilidades

**Fortalezas**
- **Persistencia explícita**: la matriz `A` modela directamente la inercia de los
  regímenes y la duración esperada → reduce el *flickering* sin reglas ad-hoc. La
  diagonal alta de la tarea previa (P(Calma→Calma)=0.979, P(Crisis→Crisis)=0.940)
  ilustra esto.
- **Probabilidad de régimen continua y con incertidumbre** (`predict_proba`),
  superior al Viterbi duro: permite umbrales calibrados y señales graduadas.
- **Interpretabilidad económica**: `μ_k, Σ_k` por estado describen el régimen
  (retorno, vol, correlaciones) — útil aguas abajo (riesgo condicional, cópulas).
- **Capta crisis sistémicas grandes y rápidas en magnitud** (2008, 2020): un salto
  brusco a alta varianza dispara la transición — la tarea previa acertó 2008/2020
  (98.6% / 92.3% de días en Crisis).
- **Multivariante de forma natural** (15 features en la Capa 1; 9 en la pista A y 14 en la B en v2).

**Debilidades**
- **Crisis rápidas/medianas mal detectadas**: el gaussiano de 2 estados prioriza
  episodios largos y de alta varianza; se perdió **Taper Tantrum 2013 (10.9%)** y
  **sell-off Q4 2018 (20.6%)**. Más estados (corrección vs crisis) y/o emisiones de
  cola pesada ayudan, pero hay límite intrínseco.
- **Supuesto gaussiano subestima colas**: con curtosis 25–40 (ventana v1 2007-2026), el HMM gaussiano
  asigna probabilidad ínfima a retornos extremos → o bien crea un "estado crisis"
  solo para absorber outliers, o decodifica mal. **t-Student/GMM-HMM** lo
  resuelven absorbiendo las colas dentro del estado [hmm_bulla2011].
- **Anti-causalidad del Viterbi/suavizado**: la decodificación de toda la muestra
  ve el futuro → **inflado optimista** in-sample; inservible tal cual para
  evaluación walk-forward (ver §Causalidad). Es la grieta más importante para
  este TFM.
- **Sensibilidad a la inicialización y a óptimos locales** (Baum-Welch no convexo);
  `startprob_` degenerado (`P(estado0)=1` en la tarea previa) sesga el arranque.
- **Etiquetado post-hoc** de estados (qué estado es "crisis") por umbral
  vol+VIX → frágil si surge un estado intermedio o si vol y VIX divergen.
- **Estacionariedad de `A` y `b_k`**: regímenes que cambian de naturaleza con los
  años no se capturan sin re-fit o parámetros variables [hmm_nystrup2017].
- **Permanencia geométrica** (Markov-1): si las duraciones reales no son
  geométricas, sesga; los HSMM lo relajan [hmm_bullabulla2006].

---

### Idoneidad para este proyecto

El HMM encaja de lleno con los hallazgos del EDA de la Capa 1 (memoria v1, tag `capa1-final`;
EDA vigente en [`eda.md`](eda.md)) y con el diagnóstico de la tarea previa
([`historia/tarea_previa.md`](../historia/tarea_previa.md)):

- **Fat tails (curtosis 25.6 / 39.6 en v1; 18,7 en el S&P 500 1928-2026) → emisiones t-Student o GMM-HMM.** Es la
  mejora más directa sobre el baseline gaussiano. La t reduce además el flickering
  y la robustez a outliers [hmm_bulla2011] — atractivo dado el ruido diario.
- **Cambio de signo de la correlación S&P/bonos → `covariance_type='full'`.** Una
  covarianza completa por estado captura las correlaciones condicionales que
  cambian entre risk-on/risk-off (Gulko 2002 [gulko2002]; la feature
  `corr_spx_bond` ya lo recoge a nivel observado).
- **HMM gaussiano de 2 estados como BASELINE puente** con la tarea previa (diseño de la Capa 1; en v2 D04 usa las features `core` de cada
  pista): misma ventana (2007-04-11, gobernada por HYG), mismo modelo (`GaussianHMM full`), para
  reproducir su comportamiento (acierta 2008/2020, falla 2013/2018) y medir
  honestamente cuánto aportan las mejoras.
- **Mejoras documentadas a implementar sobre el baseline** (todas alineadas con
  §7-8 del resumen de la tarea previa):
  1. **z-scores causales** (expanding/rolling, `min_periods=60`) en vez de
     estandarizar con media/desv de toda la muestra → elimina un look-ahead sutil.
     Las features de `src/regimenes/features/` son causales (`max_abs_diff = 0.0`; en la
     Capa 1, las 15 de `src/features.py`).
  2. **Probabilidades suaves filtradas** (forward) en vez de Viterbi duro → señal
     de crisis continua con incertidumbre y umbral calibrado.
  3. **Más estados (K=3–4) por BIC/AIC** → separar "corrección normal" de "crisis
     sistémica" y, potencialmente, capturar 2013/2018.
  4. **Walk-forward / re-fit expanding** con predicción causal (ver §Causalidad)
     → evaluación sin look-ahead, midiendo estabilidad de etiquetas.
  5. **t-Student / GMM-HMM** para las colas.
- **Tensión de cobertura para walk-forward** (declarada en el EDA §3): la GFC 2008
  está muy cerca del inicio de datos (2007-04; situación v1, hoy solo de la pista B). Un train expanding largo deja
  2008–2011 dentro del train (no evaluable OOS). A decidir en FASE 3 (ventanas más
  cortas, o detector con subconjunto SP500+VIX desde 1990 para evaluar 2008 OOS).

#### Causalidad: Viterbi anti-causal vs filtrado online (punto crítico)
El marco del TFM es **causal, sin look-ahead, walk-forward**. Para el HMM esto es
**determinante**:

- **Viterbi y `predict_proba` (forward-backward) usan TODA la muestra** → al
  decodificar el estado de `t` incorporan información de `t+1,...,T`. Es
  **suavizado anti-causal**: válido para *describir* la historia ex-post, **NO**
  para una señal operable ni para evaluación honesta. Reportar aciertos in-sample
  con Viterbi sobre toda la muestra **infla** los resultados.
- **Opciones causales correctas:**
  1. **Filtrado (forward solo)**: `P(s_t | y_{1:t})` — el estado/probabilidad de
     `t` depende solo de datos `≤ t`. Es la analogía online natural; con parámetros
     fijos requiere estimar el modelo en un train inicial y luego **solo filtrar**
     hacia delante (sin re-ver). Atención: incluso el *fit* del modelo no debe usar
     datos posteriores al punto de evaluación.
  2. **Re-fit en ventana *expanding* (o rolling)**: en cada `t` (o cada cierto
     paso) se re-estima el HMM con `y_{1:t}` y se predice **solo** el estado de `t`
     (filtrado al final de la ventana). Es el esquema walk-forward riguroso; coste
     computacional alto (un Baum-Welch por paso) pero defendible. Cuidado con el
     **label switching** entre re-fits (los estados 0/1 pueden permutar) → anclar
     etiquetas por criterio económico estable (p. ej. estado de mayor vol = crisis)
     o por continuidad con el ajuste anterior.
  3. **Estimación adaptativa / parámetros variables** (Nystrup et al.): HMM con
     parámetros que se actualizan online, pensados precisamente para regímenes
     cambiantes y uso causal [hmm_nystrup2017; hmm_nystrup2015].

> **Recomendación documental para FASE 3:** baseline gaussiano in-sample SOLO como
> puente reproductor de la tarea previa (etiquetado claramente como NO causal);
> resultados evaluables siempre con filtrado forward o re-fit expanding. Señalar
> explícitamente en la memoria que el Viterbi global no es admisible como señal.

---

### Aplicaciones documentadas a regímenes de mercado financiero

- **Guidolin & Timmermann (2007)** [guidolintimmermann2007]: regímenes
  multivariantes (4 estados) en asignación de activos; momentos condicionales
  (media, vol, correlaciones) cambian por régimen. Referencia central del solape
  HMM ↔ Markov-Switching aplicado a carteras.
- **Ang & Bekaert (2002)** [angbekaert2002]: correlaciones internacionales y
  asignación bajo regime-switching; las correlaciones suben en el régimen bajista.
- **Ang & Timmermann (2012)** [hmm_angtimmermann2012]: *survey* de cambios de
  régimen en mercados; explica cómo el regime-switching reproduce fat tails,
  heterocedasticidad, asimetría y correlaciones variables en el tiempo.
- **Bulla (2011)** [hmm_bulla2011]: HMM con componentes t sobre retornos diarios;
  mejora hechos estilizados, robustez y persistencia frente al gaussiano.
- **Hardy (2001)** [hmm_hardy2001]: modelo regime-switching lognormal sobre S&P 500
  / TSX 300 mensual (2 estados), aplicado a garantías de seguros equity-linked;
  comparado con GARCH.
- **Nystrup et al. (2015, 2017, 2018, 2020)**: HMM en tiempo continuo y de
  parámetros variables para hechos estilizados y memoria larga
  [hmm_nystrup2015; hmm_nystrup2017]; **asignación dinámica basada en regímenes**
  [hmm_nystrup2018]; y aprendizaje de HMM con **estados persistentes penalizando
  saltos** (jump models, reduce flickering y mejora detección online)
  [hmm_nystrup2020]. Este último es muy relevante para el objetivo causal/estable.
- **Kritzman, Page & Turkington (2012)** [kritzman2012]: regímenes (turbulencia,
  aversión al riesgo, liquidez) para gestión dinámica — contexto de aplicación.

---

### Coste de implementación y librería Python recomendada

- **`hmmlearn`** (recomendada como base): API tipo scikit-learn. `GaussianHMM`
  (con `covariance_type` ∈ {spherical, diag, full, tied}), `GMMHMM` (mixturas),
  `CategoricalHMM`/`MultinomialHMM`. Métodos `.fit`, `.predict` (Viterbi),
  `.predict_proba` (posteriores suavizados), `.score`/`.score_samples`, y en
  versiones recientes `.aic()`/`.bic()` para selección de `K` [hmm_hmmlearn].
  **Limitaciones**: **no** trae emisiones **t-Student** nativas; el filtrado
  causal (forward puro) no está expuesto como método de una línea (hay que usar el
  paso forward o re-fit en ventana); el re-fit walk-forward es responsabilidad del
  usuario; sensible a inicialización (usar `init_params`, varias semillas, k-means).
  Coste de un `fit`: barato (segundos) para `K` pequeño y `d=15`; el coste real
  está en el **walk-forward** (cientos/miles de re-fits).
- **`pomegranate`** (para emisiones t-Student / no gaussianas): framework
  probabilístico flexible que permite **distribuciones de emisión arbitrarias**
  (incluida la t-Student y mixturas) y construir HMM a medida [hmm_schreiber2018].
  **Limitaciones**: API reescrita sobre PyTorch en v1.x (curva de aprendizaje;
  cambios de interfaz entre versiones), menos "plug-and-play" para regímenes que
  hmmlearn; conviene fijar versión.
- **Alternativas/menciones**: `statsmodels` cubre el **Markov-Switching
  econométrico** (MS-AR/Hamilton) — tratado en la ficha [F4](F4_markov_switching.md). En R, `depmixS4`, `HiddenMarkov` y el `fHMM` (HMM para series financieras)
  ofrecen emisiones t/gamma y selección por AIC/BIC, útiles como referencia
  metodológica [hmm_zucchini2016].
- **t-Student "a mano" sobre hmmlearn**: si se quiere quedarse en hmmlearn, una
  aproximación pragmática es **GMM-HMM** (la mixtura aproxima la cola pesada) o
  estandarizar/recortar colas — peor fundamentado que la t real (criterio del
  equipo).

**Estimación de esfuerzo (criterio del equipo):** baseline gaussiano ≈ bajo
(reusa la tarea previa); añadir BIC/`K` y `predict_proba` ≈ bajo-medio;
walk-forward con re-fit + anclaje de etiquetas ≈ medio (la parte cara); emisiones
t-Student vía pomegranate ≈ medio-alto (nueva librería).

## 2. Detectores implementados

Las tablas de resultados salen de `results/benchmark/ranking_v2.csv` y `results/benchmark/metrics_master_v2.csv` (benchmark completo 24/24, `results/benchmark/manifest.json` del 2026-09-29T12:41 UTC; cifras de [ADR-003](../decisions/ADR-003-causalidad-calendario-estado-ranking.md) §4). Walk-forward causal y evaluación OOS por pista (17 crisis evaluables en la A y 9 en la B, según `configs/benchmark_spec.yaml`). Ranking de **detección**: un evento se detecta con ≥ 3 sesiones OOS consecutivas en crisis dentro de [pico, suelo]; `score_deteccion` = F1 entre la precisión diaria (1 − FAR) y el recall por evento, con niveles previos (elegible › precisión ≤ azar › parpadeo › degenerado). Cada tabla va entre marcadores `resultados_v2`: tras un benchmark nuevo se regenera desde esos dos CSV, no se edita a mano.

<a id="d04"></a>

### D04 — `hmm_gaussian_2s` · HMM gaussiano de dos estados

HMM gaussiano con covarianza completa: el baseline de la familia y el puente con la tarea previa ([`historia/tarea_previa.md`](../historia/tarea_previa.md)). La evaluación usa **filtrado forward causal** (`P(s_t | y_{1:t})`) con burn-in del train; el Viterbi global solo se usa como diagnóstico in-sample marcado no causal.

**Configuración v2**

- Código: `src/regimenes/detectores/f3_hmm/hmm_gaussian_2s.py` (`HMMGaussian2S`; filtrado en `_hmm_utils.py`); registro, id `D04`.
- Modelo: `GaussianHMM` de `hmmlearn`, **K = 2**, `covariance_type='full'`, `n_iter=200`, `n_init=5` (se queda el ajuste de mayor log-verosimilitud), semilla 42.
- Features `core` de `regimenes.detectores.registry`: **9 en la pista A** (`CORE_A`: `SP500_ret_z`, `SP500_vol_z`, `SP500_momentum`, `SP500_drawdown`, `FF_MKT_z`, `DGS10_change_z`, `credit_BaaAaa_mensual_z`, `term_spread_hist_z`, `INDPRO_yoy_z`) y **14 en la B** (`CORE_B` = las 9 de A + `VIX_level_z`, `MOVE_level_z`, `credit_BAA10Y_z`, `DFII10_change_z`, `corr_spx_bond`). En la Capa 1 usaba las 7 features puente de la tarea previa (`SP500_ret_z`, `TLT_ret_z`, `IEF_ret_z`, `HYG_ret_z`, `SP500_vol_z`, `credit_spread_z`, `VIX_level_z`).
- Walk-forward: refit cada 21 sesiones, ventana expanding.

<!-- BEGIN resultados_v2 D04 -->
**Resultados v2 (ADR-003)**

| Métrica | Pista A | Pista B |
|---|---:|---:|
| Variante evaluada | `hmm_gaussian_2s` | `hmm_gaussian_2s` |
| Puesto de detección | **10** / 12 | **5** / 12 |
| Nivel de etiqueta | `elegible` | `elegible` |
| `score_deteccion` | 0.455 | 0.497 |
| Recall por evento | 1.000 (17/17) | 0.778 (7/9) |
| Precisión diaria (lift vs base) | 0.294 (×1.52) | 0.366 (×2.12) |
| Cobertura media de crisis | 0.788 | 0.370 |
| Tasa de falsas alarmas (FAR) | 0.706 | 0.634 |
| Activación media en trampas | 0.037 | 0.000 |
| Switching rate | 0.0162 | 0.0123 |
| Duración media de régimen (sesiones) | 61.4 | 79.6 |
| Estabilidad de etiqueta | 0.979 | 0.966 |
| Puesto legacy (`rank_medio`) | 10 (8.0) | 5 (5.8) |

**Ventana OOS y eventos por pista**

- **Pista A** · OOS 1970-05-12 → 2026-05-29 (n = 14133) · train inicial 2016 sesiones, refit cada 21 (expanding) · 9 feature(s): `SP500_ret_z|SP500_vol_z|SP500_momentum|SP500_drawdown|FF_MKT_z|DGS10_change_z|credit_BaaAaa_mensual_z|term_spread_hist_z|INDPRO_yoy_z`.
  - Detectados (17): `bear_1969_70`, `oil_stagflation_1973`, `volcker_1980_82`, `black_monday_1987`, `gulf_war_sl_1990`, `ltcm_russia_1998`, `dotcom_2000_02`, `gfc_2007_09`, `flash_crash_euro1_2010`, `us_downgrade_euro2_2011`, `china_oil_2015_16`, `volmageddon_2018q1`, `fed_tightening_2018q4`, `covid_2020`, `inflation_bear_2022`, `svb_banking_2023`, `tariff_selloff_2025`.
  - No detectados (0): ninguno.
- **Pista B** · OOS 2010-04-12 → 2026-05-29 (n = 4059) · train inicial 756 sesiones, refit cada 21 (expanding) · 14 feature(s): `SP500_ret_z|SP500_vol_z|SP500_momentum|SP500_drawdown|FF_MKT_z|DGS10_change_z|credit_BaaAaa_mensual_z|term_spread_hist_z|INDPRO_yoy_z|VIX_level_z|MOVE_level_z|credit_BAA10Y_z|DFII10_change_z|corr_spx_bond`.
  - Detectados (7): `us_downgrade_euro2_2011`, `china_oil_2015_16`, `volmageddon_2018q1`, `covid_2020`, `inflation_bear_2022`, `svb_banking_2023`, `tariff_selloff_2025`.
  - No detectados (2): `flash_crash_euro1_2010`, `fed_tightening_2018q4`.
<!-- END resultados_v2 D04 -->

Narrativa de la Capa 1 (implementación v1, hallazgos e hipótesis del CHECKPOINT 2): [`historia/fichas_v1.md`](../historia/fichas_v1.md#d04).

<a id="d08"></a>

### D08 — `hmm_tstudent` · HMM con emisiones t-Student

HMM con **emisiones t-Student multivariantes** (localización, matriz de escala y grados de libertad ν por estado), estimado por EM con variable de escala latente [`hmm_bulla2011`]. Añade solo un parámetro por estado sobre el gaussiano y absorbe las colas dentro de cada régimen. Predicción por filtrado forward causal con la emisión t.

**Configuración v2**

- Código: `src/regimenes/detectores/f3_hmm/hmm_tstudent.py` (`HMMTStudent`; EM y filtrado en `_hmm_t_utils.py`); registro, id `D08`.
- Parámetros: **K = 4** (calma · leve · corrección · crisis), `n_init=3`, arranque gaussiano con `gauss_n_iter=100`, `t_n_iter=25` iteraciones del EM-t, semilla 42.
- Features `core` de `regimenes.detectores.registry`: **9 en la pista A** (`CORE_A`: `SP500_ret_z`, `SP500_vol_z`, `SP500_momentum`, `SP500_drawdown`, `FF_MKT_z`, `DGS10_change_z`, `credit_BaaAaa_mensual_z`, `term_spread_hist_z`, `INDPRO_yoy_z`) y **14 en la B** (`CORE_B` = las 9 de A + `VIX_level_z`, `MOVE_level_z`, `credit_BAA10Y_z`, `DFII10_change_z`, `corr_spx_bond`).
- Walk-forward: refit cada **126** sesiones (detector lento), ventana expanding.

> **Lectura del estado de crisis.** Con K = 4, el «estado de crisis» es el más severo de los cuatro. En la pista A es el estado OOS más poblado, no una cola estrecha; en la B sí es estrecho y el estrés más amplio cae en «corrección». Ver `notebooks/07_familia_F3_hmm.ipynb` §1.3 y §4.6.

<!-- BEGIN resultados_v2 D08 -->
**Resultados v2 (ADR-003)**

| Métrica | Pista A | Pista B |
|---|---:|---:|
| Variante evaluada | `hmm_tstudent_4s` | `hmm_tstudent_4s` |
| Puesto de detección | **9** / 12 | **9** / 12 |
| Nivel de etiqueta | `elegible` | `elegible` |
| `score_deteccion` | 0.467 | 0.274 |
| Recall por evento | 0.824 (14/17) | 0.333 (3/9) |
| Precisión diaria (lift vs base) | 0.326 (×1.68) | 0.233 (×1.35) |
| Cobertura media de crisis | 0.554 | 0.081 |
| Tasa de falsas alarmas (FAR) | 0.674 | 0.767 |
| Activación media en trampas | 0.000 | 0.000 |
| Switching rate | 0.0157 | 0.0177 |
| Duración media de régimen (sesiones) | 63.4 | 55.6 |
| Estabilidad de etiqueta | 0.773 | 0.664 |
| Puesto legacy (`rank_medio`) | 8 (7.6) | 10 (9.0) |

**Ventana OOS y eventos por pista**

- **Pista A** · OOS 1970-05-12 → 2026-05-29 (n = 14133) · train inicial 2016 sesiones, refit cada 126 (expanding) · 9 feature(s): `SP500_ret_z|SP500_vol_z|SP500_momentum|SP500_drawdown|FF_MKT_z|DGS10_change_z|credit_BaaAaa_mensual_z|term_spread_hist_z|INDPRO_yoy_z`.
  - Detectados (14): `oil_stagflation_1973`, `volcker_1980_82`, `black_monday_1987`, `gulf_war_sl_1990`, `ltcm_russia_1998`, `dotcom_2000_02`, `gfc_2007_09`, `flash_crash_euro1_2010`, `us_downgrade_euro2_2011`, `china_oil_2015_16`, `covid_2020`, `inflation_bear_2022`, `svb_banking_2023`, `tariff_selloff_2025`.
  - No detectados (3): `bear_1969_70`, `volmageddon_2018q1`, `fed_tightening_2018q4`.
- **Pista B** · OOS 2010-04-12 → 2026-05-29 (n = 4059) · train inicial 756 sesiones, refit cada 126 (expanding) · 14 feature(s): `SP500_ret_z|SP500_vol_z|SP500_momentum|SP500_drawdown|FF_MKT_z|DGS10_change_z|credit_BaaAaa_mensual_z|term_spread_hist_z|INDPRO_yoy_z|VIX_level_z|MOVE_level_z|credit_BAA10Y_z|DFII10_change_z|corr_spx_bond`.
  - Detectados (3): `china_oil_2015_16`, `covid_2020`, `inflation_bear_2022`.
  - No detectados (6): `flash_crash_euro1_2010`, `us_downgrade_euro2_2011`, `volmageddon_2018q1`, `fed_tightening_2018q4`, `svb_banking_2023`, `tariff_selloff_2025`.
<!-- END resultados_v2 D08 -->

Narrativa de la Capa 1 (implementación v1, hallazgos e hipótesis del CHECKPOINT 2): [`historia/fichas_v1.md`](../historia/fichas_v1.md#d08).

<a id="d13"></a>

### D13 — `hsmm_tstudent` · HSMM t-Student (ablación, fuera del benchmark)

**Ablación A1 de la Capa 1, no un detector del benchmark v2**: no tiene `DetectorSpec` en `regimenes.detectores.registry` (solo figura en `_FAMILIA_POR_MODULO` para poder importarse) ni fila en `ranking_v2.csv`, y `hsmm_tstudent.py` está excluido de la huella de caché del benchmark.

**Ubicación**

- Código: `src/regimenes/detectores/f3_hmm/hsmm_tstudent.py` (`HSMMTStudent`, hereda de `HMMTStudent` de D08).
- Parámetros del constructor: `max_duration = 126`, `duration_prior_strength = 12.0`, `transition_prior_strength = 2.0`; el resto, como D08.
- Notebook de la ablación: `capa1_exploracion/notebooks/A1_hsmm_ablation.ipynb` (tag `capa1-final`); en v2, sección de ablación de [`notebooks/07_familia_F3_hmm.ipynb`](../../notebooks/07_familia_F3_hmm.ipynb).
- Métricas: [`historia/capa1/resultados/ablation_hsmm/`](../historia/capa1/resultados/ablation_hsmm/) (`metrics_d13_hsmm.csv`, `metrics_d8_reference.csv` y su README).

> Un *Hidden Semi-Markov Model* (HSMM) relaja el supuesto temporal más restrictivo del HMM
> de D8: en un HMM la duración de cada régimen es **geométrica y sin memoria** (hazard de
> salida constante `h(d) = 1 − A_ss`). D13 modela explícitamente cuánto dura cada estado, de
> modo que la probabilidad de salir depende de la **edad del episodio**
> (`h_s(d) = P(D = d | D ≥ d)`). Bibliografía: `hmm_bullabulla2006`, `hmm_bulla2011`,
> `hmm_rabiner1989`, `guidolintimmermann2007`.

#### Implementado

**Experimento controlado frente a D8.** Mismas 7 features puente de la Capa 1, misma
ventana (2007-07-06 → 2026-08-26), mismo rango de K (`{3, 4}`), mismas emisiones t-Student
multivariantes, mismo `train_size = 252·5` y misma frecuencia de refit (`step = 126`). Solo
cambia la dinámica temporal:

- **D8 — t-HMM:** permanencia geométrica inducida por la diagonal de la matriz de transición.
- **D13 — t-HSMM:** duración **Gamma desplazada** por estado, truncada en `D_MAX = 126`
  sesiones (la última celda acumula la cola, sin perder masa) y **regularizada** hacia la
  geometría del HMM de arranque (`duration_prior_strength = 12` episodios equivalentes;
  `transition_prior_strength = 2` sobre la cadena de saltos).

**Estimación en dos etapas (no EM conjunto).** (1) Se ajusta el HMM-t de D8 sobre el train;
(2) se decodifica el train y se estiman las duraciones Gamma y la cadena **embebida** (diagonal
cero: la permanencia la da la PMF de duración, no las auto-transiciones). Consecuencia: el
BIC de D13 es **aproximado** y sirve para elegir K dentro de D13, no como test frente a D8.

**Causalidad.** En walk-forward se filtra el estado ampliado
`P(S_t, edad_t | observaciones ≤ t)` con parámetros ajustados solo en el pasado; `p_crisis`
marginaliza la edad. La ruta segmental completa (Viterbi HSMM) solo se usa como diagnóstico
**in-sample, NO causal**.

**Coste.** Mayor que D8 (filtro sobre K·D_MAX estados ampliados). Por eso, y por el
resultado, D13 **no entra en el benchmark v2**: no tiene `DetectorSpec` en
`regimenes.detectores.registry` (solo aparece en `_FAMILIA_POR_MODULO` para poder importarse) y
`hsmm_tstudent.py` está excluido de la huella de caché del benchmark
(`regimenes.benchmark.cache`).

#### Descubierto (notebook A1, Capa 1)

Cifras copiadas de las salidas del notebook A1 y de los dos CSV de
`docs/historia/capa1/resultados/ablation_hsmm/`.

**Selección de K (BIC aproximado).** K = 3: BIC ≈ 28 205; **K = 4: BIC ≈ 24 296** → se
despliega K = 4 (calma · leve · corrección · crisis). ν por estado canónico decreciente
[10.55, 8.06, 3.94, 2.42] y duración esperada explícita E[D] = [33.0, 21.1, 43.7, 71.5]
sesiones (la geometría del HMM de arranque daba [31.9, 19.6, 48.1, 123.4]). Monotonía de
volatilidad verificada in-sample y en walk-forward.

**Walk-forward causal** (OOS 2012-07-31 → 2026-08-26, n = 3432, 28 folds; 2008 y 2011 caen en
el train, como en D4/D8).

| Métrica (OOS, misma ventana) | D8 t-HMM | D13 t-HSMM | Lectura |
|---|---:|---:|---|
| *Switching rate* | 0.0475 | 0.0495 | empeora |
| Duración media de régimen | 20.93 | 20.07 | empeora |
| Tasa de falsas alarmas | 0.411 | 0.414 | sin cambio material |
| Estabilidad de etiqueta | 0.896 | 0.898 | mejora insignificante |
| Cobertura COVID-2020 | 0.660 | 0.680 | +2 pp |
| Cobertura Inflación-2022 | 0.643 | 0.623 | −1.9 pp |
| Activación en trampa 2013 / Q4-2018 | 0.000 / 0.017 | 0.000 / 0.017 | idéntico |
| BIC (D13 aproximado) | 24 129 | 24 296 | D13 peor (con cautela) |
| *Lead/lag* COVID / Inflación | −149 / −171 | −149 / −171 | idéntico |

**Criterios declarados antes de mirar los resultados:** (1) menos *switching* → **falla**;
(2) mayor duración media → **falla**; (3) la cobertura media de crisis OOS no cae más de 10 pp →
OK (65.2 % vs 65.1 %); (4) la tasa de falsas alarmas no sube más de 10 pp → OK.
**Veredicto: hipótesis no apoyada.** La duración explícita produce una segmentación
prácticamente idéntica a D8 y no mejora las dos variables que motivaban el modelo.

#### Decisión y papel en v2

- **D13 no se selecciona**: resultado negativo informativo. Por parsimonia, rendimiento
  equivalente y menor complejidad, **D8 `hmm_tstudent_4s` sigue siendo el representante de
  F3** (ver [D08](#d08)).
- En v2 se conserva como **ablación** dentro del notebook de familia F3; no tiene fila en
  `results/benchmark/ranking_v2.csv` ni en `metrics_master_v2.csv`, por lo que esta ficha no
  lleva la tabla "Resultados v2 (ADR-003)" del resto.
- Trabajo futuro solo si la duración de régimen pasa a ser pregunta central: sensibilidad a
  `D_MAX` y a la fuerza del prior, y un EM HSMM conjunto.

#### Fricción con el núcleo

Ninguna. `HSMMTStudent` reutiliza `t_log_emission` de `_hmm_t_utils.py` y la API de
`HMMTStudent`; el orden económico de estados lo fija el núcleo común (vol-primario).

## 3. Variantes no implementadas

- **Sticky / HDP-HMM bayesiano no paramétrico:** elige `K` solo y favorece estados persistentes; coste alto y fuera del stack `hmmlearn`.
- **HMM de parámetros variables en el tiempo / tiempo continuo** [hmm_nystrup2017; hmm_nystrup2015]: relajan la estacionariedad de `A` y de las emisiones; implementación no trivial.
- El HSMM [hmm_bullabulla2006] sí está implementado como ablación ([D13](#d13)); los *jump models* [hmm_nystrup2020] como [D09](F2_clustering.md#d09); HMM y Markov-Switching comparten motor y se distinguen en [F4](F4_markov_switching.md).

## 4. Referencias

Claves de la familia:
- [hmm_rabiner1989] Rabiner (1989), *A Tutorial on HMM...*, Proc. IEEE 77(2):257–286. doi:10.1109/5.18626
- [hmm_baumpetrie1966] Baum & Petrie (1966), Ann. Math. Stat. 37(6):1554–1563. doi:10.1214/aoms/1177699147
- [hmm_baumwelch1970] Baum, Petrie, Soules & Weiss (1970), Ann. Math. Stat. 41(1):164–171. doi:10.1214/aoms/1177697196
- [hmm_viterbi1967] Viterbi (1967), IEEE Trans. Inf. Theory 13(2):260–269. doi:10.1109/TIT.1967.1054010
- [hmm_zucchini2016] Zucchini, MacDonald & Langrock (2016), *HMM for Time Series* (2ª ed.), CRC Press.
- [hmm_bulla2011] Bulla (2011), *HMM with t components*, Quant. Finance 11(3):459–475. doi:10.1080/14697681003685563
- [hmm_bullabulla2006] Bulla & Bulla (2006), Comput. Stat. Data Anal. 51(4):2192–2209. doi:10.1016/j.csda.2006.07.021
- [hmm_hardy2001] Hardy (2001), NAAJ 5(2):41–53. doi:10.1080/10920277.2001.10595984
- [hmm_nystrup2015] Nystrup, Madsen & Lindström (2015), Quant. Finance 15(9):1531–1541. doi:10.1080/14697688.2015.1004801
- [hmm_nystrup2017] Nystrup, Madsen & Lindström (2017), J. Forecasting 36(8):989–1002. doi:10.1002/for.2447
- [hmm_nystrup2020] Nystrup, Lindström & Madsen (2020), Expert Syst. Appl. 150:113307. doi:10.1016/j.eswa.2020.113307
- [hmm_nystrup2018] Nystrup et al. (2018), J. Portfolio Manag. 44(2):62–73. doi:10.3905/jpm.2018.44.2.062
- [hmm_angtimmermann2012] Ang & Timmermann (2012), Annu. Rev. Financ. Econ. 4:313–337. doi:10.1146/annurev-financial-110311-101808
- [hmm_schreiber2018] Schreiber (2018), *pomegranate*, JMLR 18(164):1–6.
- [hmm_hmmlearn] hmmlearn developers, docs (https://hmmlearn.readthedocs.io).

Claves compartidas con otras familias:
- [guidolintimmermann2007] Guidolin & Timmermann (2007).
- [angbekaert2002] Ang & Bekaert (2002).
- [gulko2002] Gulko (2002).
- [kritzman2012] Kritzman, Page & Turkington (2012).
- [hamilton1989] Hamilton (1989) — base del Markov-Switching (ver [F4](F4_markov_switching.md)).
