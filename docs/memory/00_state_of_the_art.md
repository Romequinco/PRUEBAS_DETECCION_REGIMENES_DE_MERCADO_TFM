# 00 — Estado del arte: detección de regímenes de mercado (FASE 2)

> Documento maestro de la FASE 2. Sintetiza las 7 fichas de familia de
> `docs/memory/sota/` en una tabla transversal, resuelve solapes entre familias
> y propone la **lista definitiva de detectores para la FASE 3**. Es
> autosuficiente: permite aprobar la lista de detectores (CHECKPOINT 2) sin abrir
> las 7 fichas. Toda afirmación se cita por clave BibTeX de `docs/references.bib`;
> lo no respaldado se marca "(síntesis del equipo)".
>
> Fichas fuente (solo lectura): `sota/01_reglas_umbrales.md`,
> `sota/02_clustering.md`, `sota/03_hmm.md`, `sota/04_markov_switching.md`,
> `sota/05_volatilidad_garch.md`, `sota/06_change_point.md`,
> `sota/07_redes_neuronales.md`.

---

## 1. Introducción: objetivo, marco y conexión con tarea previa + EDA

El banco de pruebas compara detectores de régimen bajo **una sola interfaz**
(`RegimeDetector`, etiquetas canónicas `0=calma … n-1=crisis`) y un **protocolo
de evaluación causal** común: walk-forward / out-of-sample, sin look-ahead, con
las mismas métricas para todos — cobertura de las crisis 2008/2011/2020/2022, NO
disparo en falsos positivos (taper 2013, Q4 2018), lead/lag respecto al suelo del
drawdown (`DRAWDOWN_TROUGHS`: 2009-03-09, 2011-10-03, 2020-03-23, 2022-10-12),
tasa de falsas alarmas, persistencia/flickering, AIC/BIC/log-lik y estabilidad de
etiquetas entre folds.

**Conexión con la tarea previa.** El punto de partida es un HMM gaussiano de 2
estados, in-sample y con look-ahead, que acertaba las crisis grandes (2008:
98.6%, 2020: 92.3%) pero se perdía las correcciones rápidas (taper 2013: 10.9%,
Q4 2018: 20.6%) [hamilton1989; hmm_angtimmermann2012]. Ese modelo entra aquí como
**BASELINE puente** (reproductor honesto, etiquetado como NO causal), y el resto
del banco se diseña para atacar sus cuatro debilidades documentadas: look-ahead
en z-scores de muestra completa, ausencia de walk-forward, Viterbi duro sin
probabilidades y supuesto gaussiano frente a las colas.

**Anclaje en el EDA** (`docs/memory/01_data_and_eda.md`):
- **Fat tails fuertes**: kurtosis de exceso S&P 500 ≈ 25.6, HYG ≈ 39.6 → el
  supuesto gaussiano es frágil; motiva emisiones t-Student / GED y métodos no
  paramétricos [hmm_bulla2011; vol_bollerslev1987].
- **Correlación S&P/bonos que cambia de signo** [gulko2002] → favorece métodos
  multivariantes con covarianza por estado (HMM `full`, GMM) frente a reglas
  univariantes.
- **Ventana común 2007-04-11 → 2026-06-12** (gobernada por HYG), con pocas crisis
  y la GFC pegada al inicio → tensiona el walk-forward (2008 difícil de evaluar
  OOS) y, sobre todo, **descarta el deep learning como vía principal** por
  escasez de datos [lopezdeprado2018].
- **15 features causales** verificadas (`max_abs_diff=0.0`) disponibles para
  todos los detectores.

---

## 2. Tabla comparativa transversal de las 7 familias

Criterio de cada celda fundado en la ficha correspondiente. "Causal/online"
indica si la familia es causal de forma nativa o cómo se causaliza.

| Familia | Modela dinámica temporal | Asume normalidad | Robustez a fat tails | ¿Capta crisis rápidas? | Riesgo de flickering | Causal/online (cómo causalizar) | Nº estados/segmentos | Probabilidad blanda | Coste implementación | Librería Python | Refs clave |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **F1 Reglas/Umbrales** | No (memoria solo si se añade histéresis/dwell) | No (sin supuesto distribucional) | **Alta** (el outlier ES la señal) | **Sí**, reacciona el mismo día | **Alto** sin histéresis; bajo con banda muerta/dwell | **Causal nativo** si z-score expanding/rolling; único riesgo: fijar τ mirando todo el histórico | Fijado por diseño (2–4 bandas) | No (0/1 duro; salvo construir score) | **Muy bajo** | pandas, NumPy | reglas_bloom2009, reglas_faber2007, reglas_gilchristzakrajsek2012, estrellamishkin1998, kritzman2012 |
| **F2 Clustering (estático)** | **No** (asignación independiente por día) | GMM sí (componentes gaussianos); k-means asume distancia euclídea | Media (GMM cov. plena/k-medoids) a baja (k-means euclídeo) | Parcial (reacciona pero sin memoria) | **Muy alto** (sin término de persistencia) | **No nativo**: re-fit expanding + asignar nuevo punto + alinear etiquetas (húngaro) | Fijar k (silhouette, gap, BIC, ONC) | GMM sí (pertenencia); k-means no | Bajo (algoritmo) / medio (wrapper causal) | scikit-learn, hdbscan, scipy | clust_munnix2012, clust_twosigma2021regime, clust_horvath2021wasserstein, clust_lopezdeprado2020 |
| **F3 HMM (emisiones latentes)** | **Sí** (matriz de transición A) | Gaussiano sí; t-Student/GMM-HMM no | Baja (gaussiano) → **alta** con t-Student/GMM-HMM | Parcial (2 estados gaussiano falla 2013/2018; mejora con K=3–4 y t) | **Bajo** (persistencia explícita vía diagonal de A) | Viterbi/`predict_proba` son **anti-causales** (usan todo); causalizar con **filtrado forward** o re-fit expanding | K fijado a priori (BIC/AIC) | **Sí** (`predict_proba`, posteriores) | Bajo (baseline) → medio (walk-forward) → alto (t-Student) | hmmlearn, pomegranate | hmm_rabiner1989, hmm_bulla2011, hmm_nystrup2017, guidolintimmermann2007 |
| **F4 Markov-Switching (econométrico)** | **Sí** (cadena de Markov sobre parámetros) | Sí (gaussiano condicional por régimen) | Media (mezcla de gaussianas es leptocúrtica, pero insuficiente con kurt 25–40) | Parcial (igual punto ciego que HMM; switching de varianza ayuda) | **Bajo** (controlado por la diagonal de P) | **Filtradas** `P(S_t\|y_{1:t})` causales; **suavizadas** anti-causales; re-fit expanding | K fijado a priori (AIC/BIC, tests Hansen/Garcia) | **Sí** (prob. filtradas/suavizadas) | **Bajo** (univariante K=2–3) | statsmodels (`regime_switching`) | hamilton1989, ms_kim1994, ms_guidolin2011, ms_kimnelson1999 |
| **F5 Volatilidad/GARCH (RS-GARCH)** | GARCH: no (solo varianza); RS-GARCH/SWARCH: **sí** | Gaussiano por defecto; **GARCH-t/GED** para colas | **Alta** con GARCH-t/GED (ν por régimen) | **Sí** (la varianza salta el mismo día con `ε²`); asimetría (EGARCH/GJR) refuerza | Bajo (señal continua; RS-GARCH usa persistencia de P) | **Causal nativo** (`σ_t` usa solo pasado); estimar por ventana o actualizar recursivo | GARCH: ninguno (capa de umbral); RS-GARCH: K=2–3 | RS-GARCH: **sí** (prob. filtrada de régimen) | **Bajo** (GARCH-t) / **alto** (RS-GARCH: path dependence, no maduro en Python) | arch (Sheppard); RS-GARCH: R MSGARCH o propio | vol_engle1982, vol_bollerslev1987, vol_gray1996, vol_haasmittnikpaolella2004, vol_marcucci2005 |
| **F6 Change-point** | Segmenta (no etiqueta estados recurrentes) | CUSUM/ICSS sí; kernel CPD **no** | Baja (gaussiano) → **alta** (kernel/robusto) | **Sí, su punto fuerte** (detección temprana, métrica lead/lag) | Alto si β/umbral sensible; se controla con β, min_size, histéresis | Online (**causal**): CUSUM, ICSS, BOCPD, Fearnhead-Liu; offline (PELT/BinSeg) anti-causal salvo re-aplicar en ventana | Nº de segmentos emerge de la penalización (no se fija K) | BOCPD **sí** (prob. de cambio) | Bajo-medio (caro: calibrar β y post-etiquetar) | ruptures, bayesian_changepoint_detection | cp_page1954, cp_adamsmackay2007, cp_killick2012, cp_truong2020 |
| **F7 Redes neuronales / no superv. moderno** | LSTM/AE-secuencial sí; AE+clustering no | **No** (capturan no linealidad y colas) | Alta en teoría, **pero necesita datos que aquí no hay** | AE-anomalía sí en principio (reacción a "rareza") | Alto (clustering del latente parpadea como GMM) | Frágil: doble look-ahead (normalización + entrenamiento); re-fit por fold caro | k (clustering del latente) o score continuo (anomalía) | VAE/anomalía sí (score); clustering latente según método | **Alto** (overfitting con ~4 crisis; poco reproducible) | PyTorch/Keras, MiniSom, PyOD | nn_hochreiter1997, nn_kingma2014, nn_ancho2015, nn_bucci2021, lopezdeprado2018 |

---

## 3. Síntesis por familia (2–4 frases)

- **F1 — Reglas / Umbrales** (`sota/01_reglas_umbrales.md`). El régimen se decide
  comparando observables (VIX, drawdown, spread de crédito, pendiente de curva)
  contra umbrales fijos o adaptativos; es el caso degenerado observable del
  cambio de régimen (SETAR/TAR, [reglas_tong1990]). Su ventaja es la reactividad
  inmediata y la robustez a colas (el outlier es la señal, no ruido), y encaja
  1:1 con las 15 features ya causales; su talón es el flickering, mitigable con
  histéresis/dwell-time [reglas_faber2007; reglas_sahm2019]. **Baseline
  imprescindible.**

- **F2 — Clustering estático** (`sota/02_clustering.md`). Particiona el espacio de
  features sin matriz de transición (k-means, **GMM**, jerárquico, DBSCAN);
  identifica "estados de mercado" descriptivos [clust_munnix2012;
  clust_twosigma2021regime]. Es el baseline **no temporal** ideal para medir
  cuánto aporta la dinámica del HMM sobre la misma información instantánea; sufre
  flickering severo y no es causal de forma nativa (exige re-fit expanding +
  alineado de etiquetas).

- **F3 — HMM (emisiones latentes)** (`sota/03_hmm.md`). Cadena de Markov latente
  con emisiones continuas multivariantes (hmmlearn/pomegranate); añade
  persistencia explícita y probabilidad blanda de régimen [hmm_rabiner1989]. El
  gaussiano subestima las colas; las emisiones **t-Student** las absorben y
  además aumentan la persistencia (menos flickering) [hmm_bulla2011]. Punto
  crítico: Viterbi/suavizado son anti-causales → usar **filtrado forward** o
  re-fit expanding.

- **F4 — Markov-Switching econométrico** (`sota/04_markov_switching.md`).
  Tradición Hamilton: parámetros (media/varianza/AR) que conmutan con un estado
  latente markoviano, estimado por ML con filtro de Hamilton [hamilton1989;
  ms_kim1994]. Es **pariente matemático del HMM** pero univariante e
  interpretable, con probabilidades **filtradas** causales nativas en statsmodels;
  el switching de varianza ataca los clústeres de volatilidad [ms_guidolin2011].

- **F5 — Volatilidad / GARCH / RS-GARCH** (`sota/05_volatilidad_garch.md`).
  Modela la varianza condicional heteroscedástica; **causal por construcción** y
  reacciona el mismo día a un shock [vol_engle1982; vol_bollerslev1986]. GARCH-t
  resuelve las colas [vol_bollerslev1987]; SWARCH/RS-GARCH añaden cambio de
  régimen sobre la varianza y entregan probabilidad filtrada de régimen
  [vol_hamiltonsusmel1994; vol_gray1996; vol_haasmittnikpaolella2004]. GARCH-t es
  barato; RS-GARCH es caro y poco maduro en Python.

- **F6 — Change-point detection** (`sota/06_change_point.md`). Detecta instantes
  de cambio estructural (segmenta, no etiqueta estados recurrentes); CUSUM/ICSS y
  **BOCPD** son online y causales, PELT/kernel son offline [cp_page1954;
  cp_adamsmackay2007; cp_killick2012]. Es la familia natural para la **métrica
  lead/lag** (detección temprana del giro), con la cautela de fat tails (preferir
  kernel/robusto) y el coste de post-etiquetar cada segmento por criterio
  económico causal.

- **F7 — Redes neuronales / no supervisado moderno**
  (`sota/07_redes_neuronales.md`). Autoencoder/VAE + clustering latente,
  LSTM/GRU, AE-anomalía, SOM, híbridos deep+HMM [nn_kingma2014; nn_ancho2015].
  No asumen normalidad, pero **exigen datos abundantes** que la ventana (4
  crisis desde 2007) no ofrece → riesgo de overfitting alto y baja
  interpretabilidad [lopezdeprado2018; nn_bucci2021]. Solo defendible un
  representante **ligero, no supervisado y exploratorio**, con resultado negativo
  aceptable.

---

## 4. Resolución de solapes y colocación de candidatas adicionales

### 4.1 Solapes entre familias

- **HMM ↔ Markov-Switching** (el solape principal). Son **dos puntos del mismo
  continuo** (estado latente discreto markoviano + emisiones paramétricas, mismo
  aparato forward-filtering/EM ≡ filtro de Hamilton) [hamilton1989;
  hmm_angtimmermann2012]. No se cuentan como familias independientes: se reparten
  por **vehículo y rol**. → **HMM (hmmlearn)** se queda con el caso
  **multivariante** sobre las 15 features (explota la covarianza por estado y el
  cambio de signo de correlación). → **Markov-Switching (statsmodels)** se queda
  con el caso **univariante interpretable** sobre el retorno del S&P 500
  (switching de varianza, probabilidades filtradas, marco de inferencia ML). El
  **baseline puente** de la tarea previa va por **hmmlearn** (era un GaussianHMM).

- **GMM estático ↔ GMM-HMM.** El **GMM estático** (sin cadena de Markov) pertenece
  a F2 y entra como baseline no temporal. El **GMM-HMM** (mixtura como emisión con
  matriz de transición) pertenece a F3 y se trata como **variante avanzada del
  HMM** (alternativa a la t-Student cuando la librería no ofrece emisiones t). No
  se duplican: el GMM "se gana" la matriz de transición solo al pasar a F3.

- **RS-GARCH ↔ Markov-Switching.** RS-GARCH = MS + dinámica GARCH **dentro** de
  cada régimen (la varianza evoluciona, no es constante por estado)
  [vol_gray1996; vol_haasmittnikpaolella2004]. Reparto: el **MS gaussiano de
  varianza conmutada** (statsmodels) es el baseline econométrico; el **RS-GARCH /
  MS-GARCH** es la extensión avanzada que ataca mejor las colas, asumiendo su
  coste/fragilidad. Entre ambos, **GARCH-t sin régimen** (F5) es el sensor de
  volatilidad barato y fuerte.

- **Clustering latente ↔ redes.** El verdadero punto de comparación no es el
  clustering en sí (ya en F2) sino el **reductor**: PCA (lineal) vs autoencoder
  (no lineal) antes de clusterizar [nn_akioyamen2021]. Se trata como **una sola
  línea comparable** y se evita duplicar el clustering. El deep entra como **un
  único** representante exploratorio.

> **Nota de fusión bibliográfica.** Algunos papers fueron citados por dos
> subagentes bajo claves distintas: Ang & Timmermann 2012
> (`hmm_angtimmermann2012` ≡ `ms_angtimmermann2012`), Hardy 2001
> (`hmm_hardy2001` ≡ `ms_hardy2001`) y Bucci & Ciciretti
> (`clust_bucci2022realized`, versión revista con DOI ≡ `nn_bucci2021`,
> preprint). No son colisiones de clave (BibTeX las trata como entradas
> separadas); en la redacción final conviene unificar por la versión con DOI.
> Cuidado: `hmm_nystrup2018` y `nn_nystrup2018` son **papers distintos** de los
> mismos autores (asignación basada en regímenes vs. optimización de cartera
> across hidden regimes), no duplicados.

### 4.2 Candidatas adicionales (decisión explícita)

| Candidata | Decisión | Familia / ubicación | Justificación |
|---|---|---|---|
| **Statistical Jump Models** (clustering con penalización de saltos) | **ENTRA** | Puente F2↔F3 (clustering con persistencia) | Versión "aprendida" de la histéresis: estados persistentes, online, menos flickering, drawdowns suaves [hmm_nystrup2020]. Rival directo del AE+clustering y candidato sólido (síntesis del equipo). → **D9** |
| **Turbulencia de Mahalanobis** (Kritzman et al.) | **ENTRA** | F1 multivariante (regla sobre distancia) | Índice multivariante umbralizable, causal con covarianza expanding; capta el **colapso de correlaciones** que una regla univariante no ve [kritzman2012; gulko2002]. → **D10** |
| **HMM t-Student / GMM-HMM** | **ENTRA** | F3 avanzado | Ataca de frente las fat tails (kurt 25–40) y aumenta persistencia [hmm_bulla2011]. → **D8** |
| **MS-GARCH / RS-GARCH** | **ENTRA (exploratorio)** | F5 avanzado | Heteroscedasticidad intra-régimen + prob. filtrada [vol_marcucci2005; vol_ardia2019]; coste alto y dependencia de R → opcional. → **D11** |
| **DCC-GARCH** (correlación condicional dinámica) | **DESCARTA** (core) | — | GARCH multivariante para co-movimiento; no etiqueta régimen de forma nativa y es pesado. Anotar como línea futura para `corr_spx_bond` multivariante (síntesis del equipo). |
| **HSMM** (hidden semi-Markov) | **DESCARTA** (core) | Extensión de F3 | Relaja la permanencia geométrica [hmm_bullabulla2006]; solo si se observa que las duraciones no son geométricas. Mención, no detector. |
| **TVTP** (probabilidades de transición variables) | **DESCARTA** (core) | Extensión de F4 | Transiciones dependientes de covariables (VIX, slope); no nativo en statsmodels (síntesis del equipo). Extensión futura. |
| **MS-VAR** (Krolzig) | **DESCARTA** | — | MS multivariante con explosión de parámetros; el HMM multivariante cubre mejor el panel de 15 features. |
| **DBSCAN / HDBSCAN** | **DESCARTA** como particionador | F2 (uso como anomalía) | Marcan los días de crisis (raros) como "ruido", justo los que interesan [clust_campello2013hdbscan]. Útiles solo como detector de anomalías. |
| **SOM, DEC, LSTM supervisado** | **DESCARTA** | F7 | SOM = solo EDA visual; DEC = código a medida y datos grandes; LSTM supervisado = sin ground-truth de régimen [nn_xie2016; nn_kohonen1990]. |
| **Wasserstein / Sliced-Wasserstein k-means** | **DESCARTA** (core) | F2 avanzado | Clustering robusto sin supuesto de modelo [clust_horvath2021wasserstein; clust_luan2023swkmeans]; línea metodológica interesante pero secundaria frente al GMM como baseline. |
| **TDA / spectral-residual CPD** | **DESCARTA** | F6 emergente | Línea no madura como baseline (síntesis del equipo). |

---

## 5. Propuesta de lista definitiva de detectores para la FASE 3

Ordenada de **baseline a avanzado**. Cada detector lleva: nombre propuesto (estilo
`detectors/`), familia, por qué entra (literatura + EDA), qué se espera que
**capte** y qué se espera que **falle**, librería y **prioridad**
(imprescindible / recomendado / opcional-exploratorio). Numerados para
aprobar/ajustar por número en el CHECKPOINT 2.

> **BASELINES IMPRESCINDIBLES: D1 (regla sobre VIX) y D4 (HMM gaussiano puente).**

---

**D1 — `rule_vix_threshold`** · Familia F1 (Reglas/Umbrales) · **IMPRESCINDIBLE
(baseline)**
- *Por qué entra*: baseline reactivo, robusto a colas y causal nativo; contrasta
  cuánto pierde realmente el HMM por la mala especificación gaussiana
  [reglas_bloom2009]. Encaja 1:1 con `VIX_level_z` (causal, `max_abs_diff=0.0`).
- *Captará*: las 4 crisis y, por reactividad, probablemente **2013 y 2018** que el
  HMM falló.
- *Fallará*: falsos positivos ante picos efímeros si no lleva histéresis; ciego al
  cambio de signo de correlación (univariante).
- *Detalle*: umbral sobre `VIX_level_z` con **histéresis** (banda muerta τ_in/τ_out)
  + dwell-time mínimo. Librería: **pandas/NumPy**.

**D2 — `rule_composite_riskoff`** · Familia F1 · **RECOMENDADO**
- *Por qué entra*: una regla compuesta (voto de varios umbrales) capta interacciones
  que un umbral simple no ve, incluido el crédito [reglas_gilchristzakrajsek2012] y
  la pendiente de curva como señal líder [estrellamishkin1998].
- *Captará*: estrés multivariante (equity + crédito + curva); 2008/2011/2020/2022.
- *Fallará*: calibración de pesos/umbrales sensible; sigue siendo dura (0/1) salvo
  construir score.
- *Detalle*: voto/score sobre `VIX_level_z` + `credit_spread_z` + `yield_slope_z` +
  `SP500_drawdown`. Librería: **pandas/NumPy**.

**D3 — `clustering_gmm`** · Familia F2 (Clustering) · **RECOMENDADO (baseline no
temporal)**
- *Por qué entra*: aísla limpiamente el valor de la dinámica temporal — mismo input
  instantáneo que el HMM, sin matriz de transición [clust_twosigma2021regime]. GMM
  con covarianza plena capta el régimen de correlación que cambia de signo
  [gulko2002].
- *Captará*: regímenes con estructura de correlación distinta (diversificación vs
  "todo cae junto").
- *Fallará*: **flickering severo** (sin persistencia); no causal nativo.
- *Detalle*: `GaussianMixture` cov. plena, k=2–3 por BIC/silhouette, **wrapper
  causal** (re-fit expanding + asignación por probabilidad + alineado de etiquetas
  húngaro). Librería: **scikit-learn**.

**D4 — `hmm_gaussian_2s`** · Familia F3 (HMM) · **IMPRESCINDIBLE (baseline
puente)**
- *Por qué entra*: reproduce la tarea previa (misma ventana, `GaussianHMM full`,
  2 estados) para medir honestamente cuánto aportan las mejoras [hamilton1989].
- *Captará*: 2008 y 2020 (crisis grandes y persistentes).
- *Fallará*: **2013 y 2018** (correcciones rápidas); el gaussiano subestima las
  colas (kurt 25–40). Etiquetado claramente como **NO causal** en su versión
  in-sample; resultados evaluables solo con **filtrado forward / re-fit expanding**.
- *Detalle*: `GaussianHMM(n_components=2, covariance_type='full')`. Librería:
  **hmmlearn**.

**D5 — `markov_switching_var`** · Familia F4 (Markov-Switching) · **RECOMENDADO**
- *Por qué entra*: baseline econométrico **interpretable** con probabilidades
  filtradas causales nativas; switching de varianza ataca los clústeres de
  volatilidad [hamilton1989; ms_guidolin2011; ms_kim1994].
- *Captará*: regímenes calma/estrés sobre el retorno del S&P 500 con duración y
  matriz de transición legibles.
- *Fallará*: mismo punto ciego que el HMM en crisis rápidas (entrada gradual);
  univariante → no ve la correlación cross-asset; gaussiano insuficiente para las
  colas.
- *Detalle*: `MarkovRegression`/`MarkovAutoregression` sobre `SP500_ret`,
  `switching_variance=True`, K=2–3, **probabilidades filtradas**, re-fit expanding.
  Librería: **statsmodels**.

**D6 — `garch_t_vol`** · Familia F5 (Volatilidad) · **RECOMENDADO (baseline fuerte)**
- *Por qué entra*: **causal por construcción**, reacciona el mismo día a un shock y
  GARCH-t resuelve las colas residuales [vol_bollerslev1986; vol_bollerslev1987];
  la asimetría (GJR/EGARCH) replica el efecto apalancamiento de los drawdowns de
  equity [vol_glosten1993].
- *Captará*: las 4 crisis y, por reactividad de la varianza, **probablemente 2013 y
  2018**.
- *Fallará*: no da estados "tipados" (solo intensidad de vol → calma/estrés);
  univariante sobre equity.
- *Detalle*: `GJR-GARCH-t(1,1)` sobre `SP500_ret`, umbral sobre `σ_t` (percentil
  expanding) con histéresis. Librería: **arch** (Sheppard).

**D7 — `changepoint_online`** · Familia F6 (Change-point) · **RECOMENDADO**
- *Por qué entra*: detección temprana — la familia natural para la **métrica
  lead/lag** respecto al suelo del drawdown [cp_page1954; cp_adamsmackay2007].
- *Captará*: el **giro** de entrada en crisis con poco retardo (CUSUM/BOCPD).
- *Fallará*: falsas alarmas ante outliers leptocúrticos si se usa coste gaussiano
  (preferir **kernel/robusto**); requiere post-etiquetado económico causal de cada
  segmento.
- *Detalle*: CUSUM/ICSS + **BOCPD** online sobre features z-scoreadas; PELT (kernel
  `rbf`) de `ruptures` solo como oráculo offline re-aplicado en ventana
  [cp_killick2012; cp_truong2020]. Librería: **ruptures** +
  **bayesian_changepoint_detection**.

**D8 — `hmm_tstudent`** · Familia F3 (HMM avanzado) · **RECOMENDADO**
- *Por qué entra*: mejora más directa sobre el baseline gaussiano — las emisiones t
  absorben las colas y **aumentan la persistencia** (menos flickering)
  [hmm_bulla2011]; K=3–4 separa "corrección" de "crisis sistémica"
  [guidolintimmermann2007].
- *Captará*: 2008/2020 y, gracias a más estados + colas pesadas, **potencialmente
  2013/2018**.
- *Fallará*: más parámetros → riesgo de sobreajuste con pocas observaciones por
  estado; coste de walk-forward + anclaje de etiquetas.
- *Detalle*: HMM con emisiones **t-Student** (o GMM-HMM como fallback), cov. plena,
  K por BIC/AIC, **filtrado forward** causal. Librería: **pomegranate** (t) o
  **hmmlearn** (GMM-HMM).

**D9 — `jump_model`** · Familia F2↔F3 (clustering con persistencia / jump models) ·
**RECOMENDADO**
- *Por qué entra*: puente exacto entre clustering estático y dinámica temporal —
  impone persistencia penalizando saltos (histéresis "aprendida"), pensado para
  detección **online** y contra el flickering [hmm_nystrup2020].
- *Captará*: regímenes persistentes con drawdowns más suaves; rival honesto del
  AE+clustering en muestra pequeña.
- *Fallará*: depende de la penalización de salto (otro hiperparámetro a calibrar);
  madurez de librería en Python.
- *Detalle*: Statistical Jump Model (clustering + jump penalty). Librería:
  **jumpmodels** (o implementación propia según [hmm_nystrup2020]).

**D10 — `turbulence_mahalanobis`** · Familia F1 multivariante (Kritzman) ·
**RECOMENDADO (opcional)**
- *Por qué entra*: índice de turbulencia multivariante umbralizable; capta el
  **colapso de correlaciones** que las reglas univariantes no ven, manteniéndose
  barato y causal [kritzman2012; gulko2002].
- *Captará*: episodios de estrés sistémico multivariante (ruptura de la estructura
  de covarianza).
- *Fallará*: sensible a la ventana de la covarianza; un único umbral sigue siendo
  duro.
- *Detalle*: distancia de Mahalanobis sobre features con **covarianza expanding**
  (causal), umbral por percentil + histéresis. Librería: **NumPy/SciPy**.

**D11 — `msgarch_regime`** · Familia F5 (RS-GARCH avanzado) · **OPCIONAL-EXPLORATORIO**
- *Por qué entra*: la "extensión natural" — heteroscedasticidad GARCH **dentro** de
  cada régimen + probabilidad filtrada de régimen, mejor para las colas que el MS
  gaussiano [vol_gray1996; vol_haasmittnikpaolella2004; vol_marcucci2005].
- *Captará*: regímenes de volatilidad con colas dependientes del estado (ν por
  régimen).
- *Fallará*: **coste alto y fragilidad** (path dependence, óptimos locales,
  label-switching); sin equivalente maduro en Python puro (R MSGARCH o
  implementación propia) [vol_ardia2019].
- *Detalle*: MS-GARCH 2–3 estados, prob. filtrada. Librería: **rpy2 + MSGARCH (R)**
  o implementación propia (formulación Haas–Mittnik–Paolella).

**D12 — `deep_ae_regime`** · Familia F7 (Redes neuronales) · **OPCIONAL-EXPLORATORIO**
- *Por qué entra*: contraste ablativo honesto — comprobar si la no linealidad aporta
  algo sobre PCA+GMM; resultado **negativo aceptable** como contribución
  [nn_akioyamen2021; lopezdeprado2018].
- *Captará* (en teoría): no linealidades y, vía AE-anomalía, "rareza" instantánea
  (COVID-2020).
- *Fallará* (lo esperable): **overfitting** con ~4 crisis y ventana corta;
  flickering del latente; baja interpretabilidad; poca reproducibilidad. Con
  disclaimer explícito de muestra pequeña [nn_bucci2021].
- *Detalle*: **un solo** representante ligero — AE pequeño → GMM híbrido, **o**
  AE-anomalía como stress index — regularización fuerte, normalización ajustada
  por fold. Librería: **PyTorch + scikit-learn** (o PyOD para AE-anomalía).

---

### Resumen de prioridades

| Prioridad | Detectores |
|---|---|
| **Imprescindible (baseline)** | D1 `rule_vix_threshold`, D4 `hmm_gaussian_2s` |
| **Recomendado** | D2 `rule_composite_riskoff`, D3 `clustering_gmm`, D5 `markov_switching_var`, D6 `garch_t_vol`, D7 `changepoint_online`, D8 `hmm_tstudent`, D9 `jump_model`, D10 `turbulence_mahalanobis` |
| **Opcional-exploratorio** | D11 `msgarch_regime`, D12 `deep_ae_regime` |

Cobertura de familias: **las 7 familias quedan representadas** (F1: D1, D2, D10;
F2: D3, D9; F3: D4, D8; F4: D5; F5: D6, D11; F6: D7; F7: D12). Ninguna familia se
excluye; el deep (F7) entra solo como exploratorio por escasez de datos, como
exige la honestidad metodológica del EDA.

---

## 6. Cierre — CHECKPOINT 2

Esta lista de **12 detectores** es una **propuesta** y requiere **aprobación del
usuario (CHECKPOINT 2)** antes de implementar nada en `detectors/` (FASE 3). El
usuario puede aprobar/ajustar por número: añadir, quitar, fusionar o cambiar la
prioridad de cualquier `Dn`. Decisiones que quedan abiertas para la FASE 3 y que
condicionan algunos detectores: la estrategia de walk-forward ante el inicio
tardío de datos (2008 OOS, posiblemente con subconjunto SP500+VIX desde 1990) y
el set definitivo de features. Nada se implementa hasta que esta lista esté
firmada.
