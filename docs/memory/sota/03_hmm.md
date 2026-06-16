# 03 — Hidden Markov Models (HMM)

> Estado del arte (FASE 2) de la familia **HMM como modelo de espacio de estados
> latente con emisiones** (estilo `hmmlearn`/`pomegranate`). Frontera con el
> **Markov-Switching econométrico** (Hamilton MS-AR, `statsmodels`): lo cubre OTRO
> subagente. Ver nota de solape en "Candidatas adicionales". Referencias en
> `03_hmm.bib` (claves nuevas `hmm_*`) y en `docs/references.bib` (claves ya
> existentes citadas sin redefinir).

---

## Definición y supuestos

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
  Con el EDA de este proyecto (kurtosis exceso S&P 500 ≈ 25.6, HYG ≈ 39.6) este
  supuesto **subestima las colas** dentro de cada régimen (ver §Idoneidad).

**Estimación**: máxima verosimilitud por **Baum-Welch** (caso particular de EM)
[hmm_baumwelch1970], que alterna el paso E (forward-backward: probabilidades
posteriores de estado `γ_t(k) = P(s_t = k | y_{1:T})`) con el paso M (re-estima
`π, A, b_k`). Es no convexo → sensible a la **inicialización** y a óptimos
locales; práctica habitual: varias semillas (k-means o aleatorias) y quedarse con
la de mayor log-verosimilitud (en la tarea previa: 10 seeds 42–51).

---

## Variantes principales

### 1. HMM gaussiano
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

### 2. HMM con emisiones t-Student
Sustituye la normal por una **t multivariante** por estado, con grados de libertad
`ν_k` que controlan el grosor de cola (`ν → ∞` recupera la gaussiana). Es la
respuesta directa a las **fat tails**: Bulla (2011) muestra que las componentes t
**reproducen mejor los hechos estilizados** de retornos diarios, son **robustas a
outliers** y, de forma notable, **aumentan la persistencia** de los estados (menos
*flickering*), porque la cola pesada absorbe los retornos extremos sin forzar un
salto de régimen [hmm_bulla2011]. Hardy (2001) ya documentaba la ventaja de
regímenes para capturar colas y heterocedasticidad en retornos de largo plazo
[hmm_hardy2001].

### 3. GMM-HMM (emisiones de mixtura)
Cada estado emite según una **mixtura de M gaussianas** (`GMMHMM` en hmmlearn).
Aproxima densidades no gaussianas (asimetría, colas, multimodalidad) sin asumir
una forma paramétrica de cola concreta. Más flexible que el gaussiano simple, pero
**más parámetros** (`K · M` componentes) y **más fronteras de identificabilidad**
(un GMM-HMM puede "imitar" estados extra dentro de un estado). Alternativa práctica
a la t cuando la librería no ofrece emisiones t (ver §Coste).

### 4. Selección del número de estados `K`
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

### 5. Decodificación: Viterbi (duro) vs posteriores suaves
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

## Fortalezas y debilidades

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
- **Multivariante de forma natural** (las 15 features causales del proyecto).

**Debilidades**
- **Crisis rápidas/medianas mal detectadas**: el gaussiano de 2 estados prioriza
  episodios largos y de alta varianza; se perdió **Taper Tantrum 2013 (10.9%)** y
  **sell-off Q4 2018 (20.6%)**. Más estados (corrección vs crisis) y/o emisiones de
  cola pesada ayudan, pero hay límite intrínseco.
- **Supuesto gaussiano subestima colas**: con kurtosis 25–40, el HMM gaussiano
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

## Idoneidad para este proyecto

El HMM encaja de lleno con los hallazgos del EDA (`docs/memory/01_data_and_eda.md`)
y con el diagnóstico de la tarea previa (`docs/context/RESUMEN_...md`):

- **Fat tails (kurtosis 25.6 / 39.6) → emisiones t-Student o GMM-HMM.** Es la
  mejora más directa sobre el baseline gaussiano. La t reduce además el flickering
  y la robustez a outliers [hmm_bulla2011] — atractivo dado el ruido diario.
- **Cambio de signo de la correlación S&P/bonos → `covariance_type='full'`.** Una
  covarianza completa por estado captura las correlaciones condicionales que
  cambian entre risk-on/risk-off (Gulko 2002 [gulko2002]; la feature
  `corr_spx_bond` ya lo recoge a nivel observado).
- **HMM gaussiano de 2 estados como BASELINE puente** con la tarea previa: misma
  ventana (2007-04-11, gobernada por HYG), mismo modelo (`GaussianHMM full`), para
  reproducir su comportamiento (acierta 2008/2020, falla 2013/2018) y medir
  honestamente cuánto aportan las mejoras.
- **Mejoras documentadas a implementar sobre el baseline** (todas alineadas con
  §7-8 del resumen de la tarea previa):
  1. **z-scores causales** (expanding/rolling, `min_periods=60`) en vez de
     estandarizar con media/desv de toda la muestra → elimina un look-ahead sutil.
     Las 15 features de `src/features.py` ya son causales (`max_abs_diff = 0.0`).
  2. **Probabilidades suaves filtradas** (forward) en vez de Viterbi duro → señal
     de crisis continua con incertidumbre y umbral calibrado.
  3. **Más estados (K=3–4) por BIC/AIC** → separar "corrección normal" de "crisis
     sistémica" y, potencialmente, capturar 2013/2018.
  4. **Walk-forward / re-fit expanding** con predicción causal (ver §Causalidad)
     → evaluación sin look-ahead, midiendo estabilidad de etiquetas.
  5. **t-Student / GMM-HMM** para las colas.
- **Tensión de cobertura para walk-forward** (declarada en el EDA §3): la GFC 2008
  está muy cerca del inicio de datos (2007-04). Un train expanding largo deja
  2008–2011 dentro del train (no evaluable OOS). A decidir en FASE 3 (ventanas más
  cortas, o detector con subconjunto SP500+VIX desde 1990 para evaluar 2008 OOS).

### Causalidad: Viterbi anti-causal vs filtrado online (punto crítico)
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

## Aplicaciones documentadas a regímenes de mercado financiero

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

## Coste de implementación y librería Python recomendada

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
  econométrico** (MS-AR/Hamilton) — frontera de OTRO subagente, no desarrollar
  aquí. En R, `depmixS4`, `HiddenMarkov` y el `fHMM` (HMM para series financieras)
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

---

## Referencias

Nuevas (en `03_hmm.bib`):
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

Ya existentes (en `docs/references.bib`, citadas sin redefinir):
- [guidolintimmermann2007] Guidolin & Timmermann (2007).
- [angbekaert2002] Ang & Bekaert (2002).
- [gulko2002] Gulko (2002).
- [kritzman2012] Kritzman, Page & Turkington (2012).
- [hamilton1989] Hamilton (1989) — base del Markov-Switching (frontera del otro subagente).

---

## Candidatas adicionales (para el sintetizador)

- **SOLAPE HMM ↔ Markov-Switching econométrico (Hamilton, `statsmodels`).** El
  HMM gaussiano de emisiones y el MS-AR de Hamilton son el **mismo motor de
  espacio de estados con cadena latente**; difieren en énfasis: aquí
  emisiones/features multivariantes latentes (hmmlearn), allí dinámica
  autorregresiva univariante con cambio de régimen (statsmodels). **A resolver por
  el sintetizador**: evitar doble conteo, presentar HMM y MS como dos caras del
  regime-switching, y decidir si el baseline va por hmmlearn (este doc) o por
  statsmodels (otro doc). [hamilton1989; hmm_angtimmermann2012]
- **Hidden semi-Markov (HSMM)**: relajan la permanencia geométrica modelando
  explícitamente la duración del régimen [hmm_bullabulla2006]. Útil si las
  duraciones reales no son geométricas; podría mejorar 2013/2018. Mencionar como
  extensión, no desarrollar aquí.
- **Jump models / HMM con penalización de saltos** [hmm_nystrup2020]: a caballo
  entre HMM y clustering; estados persistentes, buenos para detección online y
  contra el flickering. Posible candidato propio si el sintetizador abre una
  familia "regímenes robustos/online".
- **Sticky / Bayesian nonparametric HMM (HDP-HMM)**: elige `K` automáticamente y
  favorece estados persistentes; mayor coste, fuera del stack hmmlearn (criterio
  del equipo).
- **HMM de parámetros variables en el tiempo / tiempo continuo** [hmm_nystrup2017;
  hmm_nystrup2015]: relajan la estacionariedad de `A` y `b_k`; relevantes para el
  marco causal, pero implementación no trivial.
- **Frontera con clustering (GMM/k-means sobre features)** y con **change-point
  (CUSUM, bayesiano, `ruptures`)**: familias vecinas que cubren otros subagentes;
  el GMM-HMM es el puente natural entre clustering y HMM.
