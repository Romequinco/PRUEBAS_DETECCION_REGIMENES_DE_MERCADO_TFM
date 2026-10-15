# Estado del arte de los detectores de régimen

> Síntesis de las siete familias de detectores de régimen (F1–F7) de la Fase 1: tabla
> comparativa, síntesis por familia, solapes y candidatas, y qué se implementó. Nació como
> documento maestro de la Capa 1 · etapa 2 (v1; original en el tag `capa1-final`) y se ha
> actualizado con lo construido. La lista de detectores que se **propuso** entonces (con sus
> expectativas por detector) está en
> [`../historia/estado_del_arte_v1.md`](../historia/estado_del_arte_v1.md).
>
> Teoría y detectores de cada familia: [`../fase1/F1_reglas.md`](../fase1/F1_reglas.md) …
> [`../fase1/F7_redes_neuronales.md`](../fase1/F7_redes_neuronales.md); índice de detectores en
> [`../fase1/README.md`](../fase1/README.md). Toda afirmación se cita por clave BibTeX de
> [`../references.bib`](../references.bib); lo no respaldado se marca «(síntesis del equipo)».
> El estado del arte general del TFM (agentes, RAG, look-ahead, cartera) está en
> [`estado_del_arte.md`](estado_del_arte.md) y el de fuentes de datos en
> [`fuentes_datos.md`](fuentes_datos.md).

---

## 1. Introducción: objetivo, marco y conexión con la tarea previa y el EDA

El banco de pruebas compara detectores de régimen bajo **una sola interfaz**
(`RegimeDetector`, etiquetas canónicas `0=calma … n-1=crisis`) y un **protocolo
de evaluación causal** común: walk-forward / out-of-sample, sin look-ahead, con
las mismas métricas para todos — cobertura de las crisis 2008/2011/2020/2022, NO
disparo en falsos positivos (taper 2013, Q4 2018), lead/lag respecto al suelo del
drawdown (`DRAWDOWN_TROUGHS`: 2009-03-09, 2011-10-03, 2020-03-23, 2022-10-12),
tasa de falsas alarmas, persistencia/flickering, AIC/BIC/log-lik y estabilidad de
etiquetas entre folds. Ese era el protocolo de la Capa 1; el vigente (lags de publicación,
ranking por detección de crisis con criterio de azar) es el de
[ADR-003](../decisions/ADR-003-causalidad-calendario-estado-ranking.md).

**Conexión con la [tarea previa](../historia/tarea_previa.md).** El punto de partida es un HMM gaussiano de 2
estados, in-sample y con look-ahead, que acertaba las crisis grandes (2008:
98.6%, 2020: 92.3%) pero se perdía las correcciones rápidas (taper 2013: 10.9%,
Q4 2018: 20.6%) [hamilton1989; hmm_angtimmermann2012]. Ese modelo entra aquí como
**BASELINE puente** (reproductor honesto, etiquetado como NO causal), y el resto
del banco se diseña para atacar sus cuatro debilidades documentadas: look-ahead
en z-scores de muestra completa, ausencia de walk-forward, Viterbi duro sin
probabilidades y supuesto gaussiano frente a las colas.

**Anclaje en el EDA.** Las cifras de esta sección son del EDA v1 de la Capa 1 (tag
`capa1-final`), sobre la ventana común de entonces; el EDA vigente, con la historia larga, es
[`../fase1/eda.md`](../fase1/eda.md).
- **Colas gruesas**: exceso de curtosis de los retornos diarios ≈ 25,6 (S&P 500) y
  ≈ 39,6 (HYG) en la ventana común v1, 2007-04-11 → 2026-06-12. Con la historia larga
  del EDA vigente (S&P 500, 1928–2026) el exceso de curtosis es 18,7 (13,7 sin la
  semana de Black Monday) y la t-Student ajustada da ν ≈ 2,5. En ambos casos el
  supuesto gaussiano es frágil; motiva emisiones t-Student / GED y métodos no
  paramétricos [hmm_bulla2011; vol_bollerslev1987].
- **Correlación S&P/bonos que cambia de signo** [gulko2002] → favorece métodos
  multivariantes con covarianza por estado (HMM `full`, GMM) frente a reglas
  univariantes.
- **Ventana común v1 2007-04-11 → 2026-06-12** (gobernada por HYG; la revisaron
  [ADR-001](../decisions/ADR-001-rebase-datos.md) y [ADR-002](../decisions/ADR-002-ajuste-ventanas.md)), con pocas crisis
  y la GFC pegada al inicio → tensiona el walk-forward (2008 difícil de evaluar
  OOS) y, sobre todo, **descarta el deep learning como vía principal** por
  escasez de datos [lopezdeprado2018].
- **15 features causales** verificadas (`max_abs_diff=0.0`) disponibles para
  todos los detectores (set v1; ADR-002 amplió después el pool de features).

---

## 2. Tabla comparativa transversal de las 7 familias

Criterio de cada celda fundado en la ficha correspondiente. "Causal/online"
indica si la familia es causal de forma nativa o cómo se causaliza.

| Familia | Modela dinámica temporal | Asume normalidad | Robustez a fat tails | ¿Capta crisis rápidas? | Riesgo de flickering | Causal/online (cómo causalizar) | Nº estados/segmentos | Probabilidad blanda | Coste implementación | Librería Python | Refs clave |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **F1 Reglas/Umbrales** | No (memoria solo si se añade histéresis/dwell) | No (sin supuesto distribucional) | **Alta** (el outlier ES la señal) | **Sí**, reacciona el mismo día | **Alto** sin histéresis; bajo con banda muerta/dwell | **Causal nativo** si z-score expanding/rolling; único riesgo: fijar τ mirando todo el histórico | Fijado por diseño (2–4 bandas) | No (0/1 duro; salvo construir score) | **Muy bajo** | pandas, NumPy | reglas_bloom2009, reglas_faber2007, reglas_gilchristzakrajsek2012, estrellamishkin1998, kritzman2012 |
| **F2 Clustering (estático)** | **No** (asignación independiente por día) | GMM sí (componentes gaussianos); k-means asume distancia euclídea | Media (GMM cov. plena/k-medoids) a baja (k-means euclídeo) | Parcial (reacciona pero sin memoria) | **Muy alto** (sin término de persistencia) | **No nativo**: re-fit expanding + asignar nuevo punto + alinear etiquetas (húngaro) | Fijar k (silhouette, gap, BIC, ONC) | GMM sí (pertenencia); k-means no | Bajo (algoritmo) / medio (wrapper causal) | scikit-learn, hdbscan, scipy | clust_munnix2012, clust_twosigma2021regime, clust_horvath2021wasserstein, clust_lopezdeprado2020 |
| **F3 HMM (emisiones latentes)** | **Sí** (matriz de transición A) | Gaussiano sí; t-Student/GMM-HMM no | Baja (gaussiano) → **alta** con t-Student/GMM-HMM | Parcial (2 estados gaussiano falla 2013/2018; mejora con K=3–4 y t) | **Bajo** (persistencia explícita vía diagonal de A) | Viterbi/`predict_proba` son **anti-causales** (usan todo); causalizar con **filtrado forward** o re-fit expanding | K fijado a priori (BIC/AIC) | **Sí** (`predict_proba`, posteriores) | Bajo (baseline) → medio (walk-forward) → alto (t-Student) | hmmlearn (t-Student: EM propia en el repo) | hmm_rabiner1989, hmm_bulla2011, hmm_nystrup2017, guidolintimmermann2007 |
| **F4 Markov-Switching (econométrico)** | **Sí** (cadena de Markov sobre parámetros) | Sí (gaussiano condicional por régimen) | Media (mezcla de gaussianas es leptocúrtica, pero insuficiente con curtosis 25–40) | Parcial (igual punto ciego que HMM; switching de varianza ayuda) | **Bajo** (controlado por la diagonal de P) | **Filtradas** `P(S_t\|y_{1:t})` causales; **suavizadas** anti-causales; re-fit expanding | K fijado a priori (AIC/BIC, tests Hansen/Garcia) | **Sí** (prob. filtradas/suavizadas) | **Bajo** (univariante K=2–3) | statsmodels (`regime_switching`) | hamilton1989, ms_kim1994, ms_guidolin2011, ms_kimnelson1999 |
| **F5 Volatilidad/GARCH (RS-GARCH)** | GARCH: no (solo varianza); RS-GARCH/SWARCH: **sí** | Gaussiano por defecto; **GARCH-t/GED** para colas | **Alta** con GARCH-t/GED (ν por régimen) | **Sí** (la varianza salta el mismo día con `ε²`); asimetría (EGARCH/GJR) refuerza | Bajo (señal continua; RS-GARCH usa persistencia de P) | **Causal nativo** (`σ_t` usa solo pasado); estimar por ventana o actualizar recursivo | GARCH: ninguno (capa de umbral); RS-GARCH: K=2–3 | RS-GARCH: **sí** (prob. filtrada de régimen) | **Bajo** (GARCH-t) / **alto** (RS-GARCH: path dependence, no maduro en Python) | arch (Sheppard); RS-GARCH: R MSGARCH o implementación propia (la del repo) | vol_engle1982, vol_bollerslev1987, vol_gray1996, vol_haasmittnikpaolella2004, vol_marcucci2005 |
| **F6 Change-point** | Segmenta (no etiqueta estados recurrentes) | CUSUM/ICSS sí; kernel CPD **no** | Baja (gaussiano) → **alta** (kernel/robusto) | **Sí, su punto fuerte** (detección temprana, métrica lead/lag) | Alto si β/umbral sensible; se controla con β, min_size, histéresis | Online (**causal**): CUSUM, ICSS, BOCPD, Fearnhead-Liu; offline (PELT/BinSeg) anti-causal salvo re-aplicar en ventana | Nº de segmentos emerge de la penalización (no se fija K) | BOCPD **sí** (prob. de cambio) | Bajo-medio (caro: calibrar β y post-etiquetar) | ruptures (offline); online: CUSUM propio en el repo | cp_page1954, cp_adamsmackay2007, cp_killick2012, cp_truong2020 |
| **F7 Redes neuronales / no superv. moderno** | LSTM/AE-secuencial sí; AE+clustering no | **No** (capturan no linealidad y colas) | Alta en teoría, **pero necesita datos que aquí no hay** | AE-anomalía sí en principio (reacción a "rareza") | Alto (clustering del latente parpadea como GMM) | Frágil: doble look-ahead (normalización + entrenamiento); re-fit por fold caro | k (clustering del latente) o score continuo (anomalía) | VAE/anomalía sí (score); clustering latente según método | **Alto** (overfitting con ~4 crisis; poco reproducible) | PyTorch/Keras, MiniSom, PyOD | nn_hochreiter1997, nn_kingma2014, nn_ancho2015, nn_bucci2021, lopezdeprado2018 |

---

## 3. Síntesis por familia (2–4 frases)

- **F1 — Reglas / Umbrales** ([`F1_reglas.md`](../fase1/F1_reglas.md)). El régimen se decide
  comparando observables (VIX, drawdown, spread de crédito, pendiente de curva)
  contra umbrales fijos o adaptativos; es el caso degenerado observable del
  cambio de régimen (SETAR/TAR, [reglas_tong1990]). Su ventaja es la reactividad
  inmediata y la robustez a colas (el outlier es la señal, no ruido), y encaja
  1:1 con las 15 features ya causales; su talón es el flickering, mitigable con
  histéresis/dwell-time [reglas_faber2007; reglas_sahm2019]. **Baseline
  imprescindible.**

- **F2 — Clustering estático** ([`F2_clustering.md`](../fase1/F2_clustering.md)). Particiona el espacio de
  features sin matriz de transición (k-means, **GMM**, jerárquico, DBSCAN);
  identifica "estados de mercado" descriptivos [clust_munnix2012;
  clust_twosigma2021regime]. Es el baseline **no temporal** ideal para medir
  cuánto aporta la dinámica del HMM sobre la misma información instantánea; sufre
  flickering severo y no es causal de forma nativa (exige re-fit expanding +
  alineado de etiquetas).

- **F3 — HMM (emisiones latentes)** ([`F3_hmm.md`](../fase1/F3_hmm.md)). Cadena de Markov latente
  con emisiones continuas multivariantes (hmmlearn); añade
  persistencia explícita y probabilidad blanda de régimen [hmm_rabiner1989]. El
  gaussiano subestima las colas; las emisiones **t-Student** las absorben y
  además aumentan la persistencia (menos flickering) [hmm_bulla2011]. Punto
  crítico: Viterbi/suavizado son anti-causales → usar **filtrado forward** o
  re-fit expanding.

- **F4 — Markov-Switching econométrico** ([`F4_markov_switching.md`](../fase1/F4_markov_switching.md)).
  Tradición Hamilton: parámetros (media/varianza/AR) que conmutan con un estado
  latente markoviano, estimado por ML con filtro de Hamilton [hamilton1989;
  ms_kim1994]. Es **pariente matemático del HMM** pero univariante e
  interpretable, con probabilidades **filtradas** causales nativas en statsmodels;
  el switching de varianza ataca los clústeres de volatilidad [ms_guidolin2011].

- **F5 — Volatilidad / GARCH / RS-GARCH** ([`F5_volatilidad_garch.md`](../fase1/F5_volatilidad_garch.md)).
  Modela la varianza condicional heteroscedástica; **causal por construcción** y
  reacciona el mismo día a un shock [vol_engle1982; vol_bollerslev1986]. GARCH-t
  resuelve las colas [vol_bollerslev1987]; SWARCH/RS-GARCH añaden cambio de
  régimen sobre la varianza y entregan probabilidad filtrada de régimen
  [vol_hamiltonsusmel1994; vol_gray1996; vol_haasmittnikpaolella2004]. GARCH-t es
  barato; RS-GARCH es caro y poco maduro en Python.

- **F6 — Change-point detection** ([`F6_change_point.md`](../fase1/F6_change_point.md)). Detecta instantes
  de cambio estructural (segmenta, no etiqueta estados recurrentes); CUSUM/ICSS y
  BOCPD son online y causales (el repo usa CUSUM en D07), PELT/kernel son offline [cp_page1954;
  cp_adamsmackay2007; cp_killick2012]. Es la familia natural para la **métrica
  lead/lag** (detección temprana del giro), con la cautela de fat tails (preferir
  kernel/robusto) y el coste de post-etiquetar cada segmento por criterio
  económico causal.

- **F7 — Redes neuronales / no supervisado moderno**
  ([`F7_redes_neuronales.md`](../fase1/F7_redes_neuronales.md)). Autoencoder/VAE + clustering latente,
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

> **Nota bibliográfica.** Algunas obras aparecen bajo dos claves: Ang & Timmermann 2012
> (`hmm_angtimmermann2012` y `ms_angtimmermann2012`), Hardy 2001 (`hmm_hardy2001` y
> `ms_hardy2001`) y Bucci & Ciciretti (`clust_bucci2022realized`, versión de revista con DOI,
> y `nn_bucci2021`, preprint). Las seis claves resuelven en
> [`../references.bib`](../references.bib); en la redacción final conviene citar la versión
> con DOI. Cuidado: `hmm_nystrup2018` y `nn_nystrup2018` son **obras distintas** de los
> mismos autores (asignación basada en regímenes frente a optimización de cartera entre
> regímenes ocultos), no duplicados.

### 4.2 Candidatas adicionales (decisión explícita)

| Candidata | Decisión | Familia / ubicación | Justificación |
|---|---|---|---|
| **Statistical Jump Models** (clustering con penalización de saltos) | **ENTRA** | Puente F2↔F3 (clustering con persistencia) | Versión "aprendida" de la histéresis: estados persistentes, online, menos flickering, drawdowns suaves [hmm_nystrup2020]. Rival directo del AE+clustering y candidato sólido (síntesis del equipo). → **D9** |
| **Turbulencia de Mahalanobis** (Kritzman et al.) | **ENTRA** | F1 multivariante (regla sobre distancia) | Índice multivariante umbralizable, causal con covarianza expanding; capta el **colapso de correlaciones** que una regla univariante no ve [kritzman2012; gulko2002]. → **D10** |
| **HMM t-Student / GMM-HMM** | **ENTRA** | F3 avanzado | Ataca de frente las colas gruesas (curtosis 25–40 en la ventana v1) y aumenta la persistencia [hmm_bulla2011]. → **D8** |
| **MS-GARCH / RS-GARCH** | **ENTRA (exploratorio)** | F5 avanzado | Heteroscedasticidad intra-régimen + prob. filtrada [vol_marcucci2005; vol_ardia2019]; coste alto y dependencia de R → opcional. → **D11** (al final, sin R: Haas-Mittnik-Paolella propio) |
| **DCC-GARCH** (correlación condicional dinámica) | **DESCARTA** (core) | — | GARCH multivariante para co-movimiento; no etiqueta régimen de forma nativa y es pesado. Anotar como línea futura para `corr_spx_bond` multivariante (síntesis del equipo). |
| **HSMM** (hidden semi-Markov) | **DESCARTA** (core) → **ablación** | Extensión de F3 | Relaja la permanencia geométrica [hmm_bullabulla2006]; solo si se observa que las duraciones no son geométricas. Se dejó fuera del banco principal y después se implementó como ablación frente a D8: → **D13** (fuera del benchmark v2). |
| **TVTP** (probabilidades de transición variables) | **DESCARTA** (core) | Extensión de F4 | Transiciones dependientes de covariables (VIX, slope); no nativo en statsmodels (síntesis del equipo). Extensión futura. |
| **MS-VAR** (Krolzig) | **DESCARTA** | — | MS multivariante con explosión de parámetros; el HMM multivariante cubre mejor el panel de 15 features. |
| **DBSCAN / HDBSCAN** | **DESCARTA** como particionador | F2 (uso como anomalía) | Marcan los días de crisis (raros) como "ruido", justo los que interesan [clust_campello2013hdbscan]. Útiles solo como detector de anomalías. |
| **SOM, DEC, LSTM supervisado** | **DESCARTA** | F7 | SOM = solo EDA visual; DEC = código a medida y datos grandes; LSTM supervisado = sin ground-truth de régimen [nn_xie2016; nn_kohonen1990]. |
| **Wasserstein / Sliced-Wasserstein k-means** | **DESCARTA** (core) | F2 avanzado | Clustering robusto sin supuesto de modelo [clust_horvath2021wasserstein; clust_luan2023swkmeans]; línea metodológica interesante pero secundaria frente al GMM como baseline. |
| **TDA / spectral-residual CPD** | **DESCARTA** | F6 emergente | Línea no madura como baseline (síntesis del equipo). |

---

## 5. Qué se implementó

Los 12 detectores propuestos se implementaron en `src/regimenes/detectores/` (uno por
fichero, carpeta por familia) y entran en el benchmark v2; D13 se añadió después como
ablación. Donde la implementación se apartó de la propuesta v1, se indica.

| Detector | Familia | Implementación | Desvío respecto a la propuesta v1 |
|---|---|---|---|
| D01 `rule_vix_threshold` | F1 | Umbral con histéresis y dwell-time sobre `VIX_level_z` (pista B) o `SP500_vol_z` (pista A, variante declarada) | En la pista A no hay VIX antes de 1990: se usa la volatilidad realizada y el registro la marca como variante |
| D02 `rule_composite_riskoff` | F1 | Score risk-off de 4 señales con histéresis y dwell-time | — |
| D03 `clustering_gmm` | F2 | GMM de covarianza plena con envoltorio causal (re-fit y alineado de etiquetas) | — |
| D04 `hmm_gaussian_2s` | F3 | GaussianHMM de 2 estados, versión in-sample (no causal) y walk-forward | — |
| D05 `markov_switching_var` | F4 | `MarkovRegression` de statsmodels sobre el retorno del S&P 500, probabilidades filtradas | — |
| D06 `garch_t_vol` | F5 | GJR-GARCH(1,1)-t (arch) y umbral sobre σ con histéresis | — |
| D07 `changepoint_online` | F6 | Autómata de dos CUSUM de Page sobre la volatilidad | **Sin BOCPD** ni `bayesian_changepoint_detection`; PELT (`ruptures`) solo como oráculo offline en el notebook |
| D08 `hmm_tstudent` | F3 | HMM con emisiones t-Student multivariantes por EM propia; K=4 fijado en v2 (elegido por BIC sobre {3, 4} en la Capa 1) | **Sin pomegranate** ni GMM-HMM |
| D09 `jump_model` | F2↔F3 | Statistical Jump Model con la librería `jumpmodels` | — |
| D10 `turbulence_mahalanobis` | F1 multivariante | Distancia de Mahalanobis con media y covarianza expanding, umbral con histéresis | — |
| D11 `msgarch_regime` | F5 | MS-GARCH de Haas-Mittnik-Paolella (2004) con filtro de Hamilton propio (numpy/scipy) | **Sin R** (`rpy2` + MSGARCH descartado) |
| D12 `deep_ae_regime` | F7 | Autoencoder ligero (PyTorch) → GMM sobre el latente, frente a PCA + GMM | — |
| D13 `hsmm_tstudent` | F3 (ablación) | HSMM t-Student: emisiones de D08 y duración Gamma desplazada por estado, filtrado forward | **No estaba en la lista v1** (HSMM se descartó del core); ablación frente a D08, fuera del benchmark v2 |

Fichas, resultados por pista y ranking de detección (ADR-003): índice de detectores en
[`../fase1/README.md`](../fase1/README.md) y sección de detectores de cada ficha de familia.
