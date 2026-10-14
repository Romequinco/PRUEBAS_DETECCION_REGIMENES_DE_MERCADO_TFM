# F7 — Redes neuronales

> **Cómo leer este documento.** La sección 1 es la teoría de la familia (redactada en la Capa 1, original en el tag `capa1-final`); la 2, los detectores implementados con su configuración v2 y sus resultados del benchmark (ADR-003): [D12 `deep_ae_regime`](#d12). Notebook de familia: [`notebooks/11_familia_F7_deep.ipynb`](../../notebooks/11_familia_F7_deep.ipynb). Bibliografía: claves `nn_*` y compartidas de [`docs/references.bib`](../references.bib).
>
> **Datos v1 en la teoría.** Las cifras del EDA que cita la sección 1 son de la Capa 1 (15 features, ventana 2007-04 → 2026-06, 4 crisis: 2008, 2011, 2020 y 2022), igual que los nombres de feature `credit_spread_z` o `yield_slope_z` y las referencias a la «tarea previa»; donde el dato importa se indica junto a él. El panel v2 tiene dos pistas (A: 1962-01-02 → 2026-05-29; B: 2007-04-11 → 2026-05-29) y el EDA vigente está en [`eda.md`](eda.md) (curtosis de exceso del S&P 500 = 18,7 en 1928-2026).

## 1. Teoría

> Familia: métodos deep / no supervisado moderno para detección de regímenes.
> Núcleo: autoencoder (incl. VAE) + clustering del espacio latente; LSTM/GRU;
> detección de anomalías por error de reconstrucción; SOM; e híbridos deep+HMM.
> Aviso de honestidad: gran parte de esta literatura es reciente, heterogénea y
> con fuerte sesgo de publicación (se reportan los aciertos). Distinguimos abajo
> lo consolidado (LSTM, autoencoder, VAE como herramientas) de lo especulativo
> (su ventaja real sobre HMM/clustering en muestras pequeñas de régimen).

### Definición y supuestos

Bajo este paraguas caen modelos de redes neuronales que aprenden una
**representación** de la dinámica de mercado y, sobre ella, separan estados. Dos
lógicas distintas conviven:

1. **No supervisado / representación → clustering.** Un autoencoder (AE) o un
   variational autoencoder (VAE) comprime las features causales (15 en la Capa 1; 9/14 por pista en
   v2) a un espacio
   latente de baja dimensión; luego se agrupa ese latente con k-means/GMM, o se
   entrena clustering y representación a la vez (Deep Embedded Clustering, DEC,
   Xie et al. 2016). Supuesto central: existe una variedad (manifold) de baja
   dimensión donde los regímenes son más separables que en el espacio original.
   **No asume gaussianidad de los retornos** —es su ventaja frente al HMM
   gaussiano dado el EDA (curtosis de exceso S&P 500 ≈ 25.6 en v1; 18,7 en 1928-2026)— pero **sí asume que
   hay suficientes datos para estimar la variedad sin memorizarla**.

2. **Detección de anomalías por reconstrucción.** Se entrena el AE/VAE solo en
   periodos "normales" (p. ej. VIX bajo el percentil 75); en producción, un
   **error de reconstrucción** alto señala que el mercado salió de la
   distribución aprendida → proxy de crisis (An & Cho 2015; Hinton &
   Salakhutdinov 2006 para AE como reductor no lineal). Supuesto: las crisis son
   *outliers* respecto al régimen de calma. Esto da una señal continua de
   "rareza", no una partición en k estados.

3. **Secuencial supervisado/auto-supervisado (LSTM/GRU).** Las redes recurrentes
   con memoria (Hochreiter & Schmidhuber 1997) modelan dependencia temporal
   explícita. Para regímenes su uso es problemático: **no hay ground-truth de
   régimen** (ver más abajo), así que el LSTM "supervisado" requiere etiquetas
   sintéticas (umbral de drawdown, salida de un HMM…), heredando la fragilidad de
   esas etiquetas. Su nicho honesto aquí es como *encoder secuencial* dentro de
   un AE (LSTM-AE) o para forecasting de volatilidad, no como detector directo.

**Supuesto transversal y crítico para este TFM:** el aprendizaje profundo asume
régimen de datos abundante. En la Capa 1 la ventana común empezaba en 2007-04-11 (~4 700
observaciones diarias) con **solo 4 episodios de crisis** (2008, 2011, 2020,
2022); en v2 hay 17 crisis evaluables OOS en la pista A y 9 en la B. El número efectivo de ejemplos de la clase "crisis" es de orden decenas de
eventos, no miles. Eso choca de frente con el supuesto.

### Variantes principales

- **Autoencoder + clustering latente (AE→k-means / VAE→GMM).** Encoder-decoder
  que minimiza error de reconstrucción; el cuello de botella (latente) se
  clusteriza. El VAE (Kingma & Welling 2014) añade un latente probabilístico
  (regularizado a una prior gaussiana vía KL + truco de reparametrización), lo
  que da un espacio latente más suave y muestreable, a costa de posibles
  "posterior collapse" con pocos datos. **DEC** (Xie et al. 2016) une ambos pasos
  optimizando conjuntamente embedding y asignación de cluster con una pérdida
  auto-entrenada — atractivo en teoría, pero diseñado y validado sobre datasets
  grandes (MNIST, Reuters), no sobre ~4 700 puntos correlacionados.

- **LSTM / GRU.** Recurrentes con puertas que mitigan el gradiente evanescente
  (Hochreiter & Schmidhuber 1997). Variantes para regímenes: LSTM-autoencoder
  (reconstruir secuencias y usar el error/latente), o stacked-AE + LSTM para
  forecasting (Bao, Yue & Rao 2017, sobre precios, no regímenes). GRU es una
  variante más liviana con menos parámetros — preferible con datos escasos.

- **Detección de anomalías por error de reconstrucción.** AE/VAE entrenado en
  calma; score = error de reconstrucción (AE) o "reconstruction probability"
  (VAE, An & Cho 2015). Produce una señal continua tipo "stress index" que luego
  se umbrala. Conceptualmente cercano a un indicador de estrés financiero, no a
  una partición markoviana.

- **Self-Organizing Maps (SOM).** Red no supervisada (Kohonen 1990) que proyecta
  el espacio de features a una rejilla 2D preservando topología; cada neurona
  ganadora define un "estado" y las trayectorias sobre el mapa visualizan
  transiciones de régimen. Ventaja: interpretabilidad visual y robustez con
  pocos datos (pocos parámetros frente a un deep net). Es de la familia
  "shallow" neuronal; útil más como herramienta exploratoria/EDA que como
  detector causal de producción.

- **Híbridos deep + HMM / clustering.** Patrón habitual y más defendible: usar la
  red **solo** para representación (AE/VAE/PCA no lineal) y delegar la dinámica
  de estados a un HMM o a clustering sobre el latente. Ejemplos documentados:
  Akioyamen et al. (2021) hacen PCA + k-means (no deep, pero el mismo esqueleto
  "reducir → clusterizar"); Bucci & Ciciretti (2021) comparan clustering
  jerárquico no supervisado contra un modelo no lineal (VLSTAR) sobre covarianzas
  realizadas y concluyen que **el modelo no lineal econométrico etiqueta mejor
  los regímenes que el clustering** — dato sobrio que conviene retener.

### Fortalezas y debilidades

**Fortalezas (potenciales, condicionadas a datos):**
- **No asumen normalidad.** Frente al HMM gaussiano, capturan no linealidades y
  colas gordas implícitamente — relevante dado el EDA (curtosis 25.6 en S&P,
  39.6 en HYG en la ventana v1).
- **Multivariante y de alta dimensión.** Escalan a muchas features y a relaciones
  no lineales (p. ej. el cambio de signo de la correlación S&P/bonos, Gulko 2002)
  sin especificación paramétrica.
- **Anomaly-AE capta crisis rápidas en principio.** Como reacciona a la "rareza"
  instantánea, una caída brusca (COVID-2020) dispara el error de reconstrucción
  sin esperar a acumular evidencia — ventaja frente a un HMM que puede tardar en
  conmutar. (Conjetura razonable, no garantizada con esta muestra.)

**Debilidades (las decisivas para este proyecto):**
- **Overfitting con muestra pequeña y pocas crisis.** Es *la* objeción. Un deep
  net tiene de miles a millones de parámetros; con ~4 700 días autocorrelacionados (v1)
  y ~4 crisis, el riesgo de memorizar episodios concretos es altísimo. El número
  efectivo de observaciones independientes es mucho menor que 4 700 por la fuerte
  autocorrelación. López de Prado (2018) advierte explícitamente del
  sobreajuste y del backtest overfitting en finanzas; aquí aplica con dureza.
- **Flickering.** El clustering del latente, igual que k-means/GMM puro, no impone
  persistencia temporal → asignaciones que parpadean día a día. Hay que añadir
  suavizado ex-post (lo que reintroduce decisiones ad hoc) o un HMM encima.
- **Causalidad/online frágil.** Riesgo doble de look-ahead: (i) **normalización**
  (un StandardScaler ajustado sobre todo el histórico filtra el futuro — debe
  ajustarse solo con el train de cada fold); (ii) **entrenamiento** del AE con
  datos posteriores al punto evaluado. El walk-forward encarece esto: reentrenar
  una red en cada fold es caro y los primeros folds tendrán poquísimos datos.
- **Falta de interpretabilidad.** Un estado "latente 3" no se explica solo; frente
  a un HMM (medias/varianzas por estado) o reglas (VIX>umbral), perdemos la
  narrativa económica que un TFM necesita defender.
- **Coste y reproducibilidad.** Entrenamiento estocástico (semillas, inicialización,
  early stopping) → resultados menos reproducibles; coste computacional alto en
  walk-forward; muchos hiperparámetros que, al tunearse sobre la misma muestra,
  agravan el overfitting.
- **¿Asume normalidad? No** —y eso es bueno— pero la contrapartida es que necesita
  datos para aprender la forma real de la distribución, datos que aquí no hay.

### Idoneidad para este proyecto

**Veredicto honesto: baseline avanzado OPCIONAL, no prioritario; y como detector
deep "puro" supervisado, injustificable con los datos actuales.**

Razonamiento atado al EDA:
- La escasez (ventana v1 desde 2007, 4 crisis; en v2, 17 y 9 crisis OOS por pista) es **incompatible** con entrenar de
  forma fiable un LSTM/VAE profundo por fold en walk-forward. La tensión de
  cobertura ya señalada en el EDA de la Capa 1 (la GFC pegada al inicio de
  datos) empeora: los primeros folds no tendrían ni una crisis para "ver".
- Lo **único defendible** dentro de la familia es la línea **no supervisada y
  ligera**: (a) un **autoencoder pequeño** (1–2 capas, latente 2–4) como
  reductor no lineal seguido de GMM/HMM —híbrido deep+clustering—, comparándolo
  honestamente contra PCA+GMM para ver si la no linealidad aporta algo; o (b) un
  **AE de anomalías** entrenado en calma, cuyo error de reconstrucción se evalúa
  como índice de estrés frente al VIX y al spread de crédito. Ambos con
  regularización fuerte, pocos parámetros y normalización ajustada por fold.
- **SOM** es la opción neuronal más barata y robusta a pocos datos; vale como
  herramienta de visualización/EDA del espacio de regímenes, no como detector de
  producción.
- **Cuándo SÍ se justificaría escalar a deep:** con más features (microestructura,
  opciones, cross-asset) y **datos intradía** (que multiplican las observaciones
  por dos-tres órdenes de magnitud), o ensanchando la ventana usando un subconjunto
  de features con histórico largo (S&P+VIX desde 1990) para tener más crisis. Con
  el panel diario actual, **HMM (incl. t-Student) y clustering son baselines más
  honestos y más interpretables**; el deep entra, si acaso, como contraste
  ablativo para demostrar que *no* mejora —resultado negativo que también es
  ciencia y refuerza la tesis.

Recomendación operativa: incluir como mucho **un** representante ligero
(AE→GMM híbrido **o** AE-anomalía) marcado explícitamente como "avanzado/
exploratorio", con disclaimer de muestra pequeña, y no construir LSTM supervisado.

### Aplicaciones documentadas a regímenes de mercado financiero

- **Akioyamen, Tang & Hussien (2021)** — PCA (reducción) + k-means (clustering)
  sobre datos macro/financieros públicos para identificar regímenes en EE. UU. y
  validar estrategias condicionadas al régimen. Es el esqueleto "reducir→
  clusterizar" en versión clásica (no deep), útil como referencia metodológica.
- **Bucci & Ciciretti (2021)** — comparan clustering jerárquico no supervisado vs.
  modelo no lineal (VLSTAR) sobre covarianzas realizadas mensuales; concluyen que
  el modelo no lineal econométrico etiqueta mejor los regímenes. Evidencia
  templada de que "más deep/no supervisado" no implica mejor.
- **Bao, Yue & Rao (2017)** — framework wavelet + stacked-autoencoder + LSTM para
  *forecasting* de precios (no detección de régimen). Se cita por el patrón
  AE+LSTM y porque ilustra el riesgo: resultados muy optimistas que estudios
  posteriores atribuyen en parte a fugas de información por el preprocesado
  wavelet — recordatorio de causalidad.
- **An & Cho (2015)** — VAE para detección de anomalías vía "reconstruction
  probability"; base conceptual del AE-anomalía como sensor de crisis (origen no
  financiero, pero directamente transplantable).
- Literatura reciente arXiv (representation learning para regímenes,
  autoencoders/CNN + clustering, Wasserstein k-means de regímenes): abundante a
  partir de 2020–2025 pero **desigual**, a menudo sin walk-forward estricto, sin
  control de look-ahead en la normalización, y con muestras igual de pequeñas →
  tratar como inspiración metodológica, no como evidencia sólida de superioridad.
- **Marketing vs. ciencia:** firmas (Two Sigma, Man-AHL, etc.) publican blogs
  sobre "regime detection con ML"; rara vez aportan validación reproducible. No
  se citan como evidencia; (criterio del equipo).

### Coste de implementación y librería Python recomendada

- **Clustering del latente:** `scikit-learn` (`KMeans`, `GaussianMixture`),
  `scipy`/`scikit-learn` para jerárquico. Para la parte deep:
- **Autoencoder / VAE / LSTM-AE:** `PyTorch` (control fino, recomendado) o
  `Keras`/`TensorFlow` (prototipado rápido). AE pequeño = decenas de líneas.
- **DEC:** no hay implementación canónica en sklearn; requiere código a medida en
  PyTorch → coste y riesgo de bug altos para la ganancia esperada aquí. Evitar
  salvo motivación fuerte.
- **SOM:** `MiniSom` (ligera, pura Python) o `sklearn-som`.
- **Anomaly-AE:** PyTorch + umbral sobre error de reconstrucción; `PyOD` ofrece
  AE/VAE empaquetados para anomalía (útil para baseline rápido).
- **Limitaciones prácticas:** coste de reentrenar por fold en walk-forward;
  necesidad de fijar semillas y early stopping para reproducibilidad; cuidado
  extremo con que el `scaler`/normalización se ajuste **solo** con el train de
  cada fold (causalidad). Recomendación: si se incluye, mantener arquitectura
  minúscula, regularización (dropout, weight decay) agresiva, y reportar
  sensibilidad a la semilla.

## 2. Detectores implementados

Las tablas de resultados salen de `results/benchmark/ranking_v2.csv` y `results/benchmark/metrics_master_v2.csv` (benchmark completo 24/24, `results/benchmark/manifest.json` del 2026-09-29T12:41 UTC; cifras de [ADR-003](../decisions/ADR-003-causalidad-calendario-estado-ranking.md) §4). Walk-forward causal y evaluación OOS por pista (17 crisis evaluables en la A y 9 en la B, según `configs/benchmark_spec.yaml`). Ranking de **detección**: un evento se detecta con ≥ 3 sesiones OOS consecutivas en crisis dentro de [pico, suelo]; `score_deteccion` = F1 entre la precisión diaria (1 − FAR) y el recall por evento, con niveles previos (elegible › precisión ≤ azar › parpadeo › degenerado). Cada tabla va entre marcadores `resultados_v2`: tras un benchmark nuevo se regenera desde esos dos CSV, no se edita a mano.

<a id="d12"></a>

### D12 — `deep_ae_regime` · autoencoder + GMM

Contraste **ablativo** de la familia: un autoencoder denso pequeño comprime las features a un latente de dos dimensiones y una mixtura gaussiana agrupa ese latente. Pregunta: ¿aporta la reducción no lineal algo frente a la lineal (PCA → GMM, clase `PCAGMMBaseline` del mismo módulo, usada solo en el notebook)? Un resultado negativo es aceptable.

**Configuración v2**

- Código: `src/regimenes/detectores/f7_deep/deep_ae_regime.py` (`DeepAERegime`); registro, id `D12`.
- Modelo: AE PyTorch `d → 8 → 2 → 8 → d` (ReLU, dropout 0.10, Adam con `lr = 1e-2` y `weight_decay = 1e-3`, 40 épocas) + `GaussianMixture` full con **K = 2** (`gmm_n_init = 3`) sobre el latente; estandarización con estadísticos del train del fold; semilla 42. En la Capa 1 usaba K = 3 sobre las 15 features.
- Features `core` de `regimenes.detectores.registry`: **9 en la pista A** (`CORE_A`: `SP500_ret_z`, `SP500_vol_z`, `SP500_momentum`, `SP500_drawdown`, `FF_MKT_z`, `DGS10_change_z`, `credit_BaaAaa_mensual_z`, `term_spread_hist_z`, `INDPRO_yoy_z`) y **14 en la B** (`CORE_B` = las 9 de A + `VIX_level_z`, `MOVE_level_z`, `credit_BAA10Y_z`, `DFII10_change_z`, `corr_spx_bond`).
- Walk-forward: refit cada 21 sesiones, ventana expanding (detector lento).

<!-- BEGIN resultados_v2 D12 -->
**Resultados v2 (ADR-003)**

| Métrica | Pista A | Pista B |
|---|---:|---:|
| Variante evaluada | `deep_ae_regime_k2` | `deep_ae_regime_k2` |
| Puesto de detección | **6** / 12 | **12** / 12 |
| Nivel de etiqueta | `elegible` | `parpadeo` |
| `score_deteccion` | 0.545 | 0.274 |
| Recall por evento | 0.882 (15/17) | 0.222 (2/9) |
| Precisión diaria (lift vs base) | 0.395 (×2.03) | 0.358 (×2.07) |
| Cobertura media de crisis | 0.568 | 0.159 |
| Tasa de falsas alarmas (FAR) | 0.605 | 0.642 |
| Activación media en trampas | 0.005 | 0.021 |
| Switching rate | 0.1911 | 0.0552 |
| Duración media de régimen (sesiones) | 5.2 | 18.0 |
| Estabilidad de etiqueta | 0.995 | 0.998 |
| Puesto legacy (`rank_medio`) | 9 (7.7) | 9 (8.0) |

**Ventana OOS y eventos por pista**

- **Pista A** · OOS 1970-05-12 → 2026-05-29 (n = 14133) · train inicial 2016 sesiones, refit cada 21 (expanding) · 9 feature(s): `SP500_ret_z|SP500_vol_z|SP500_momentum|SP500_drawdown|FF_MKT_z|DGS10_change_z|credit_BaaAaa_mensual_z|term_spread_hist_z|INDPRO_yoy_z`.
  - Detectados (15): `bear_1969_70`, `oil_stagflation_1973`, `volcker_1980_82`, `black_monday_1987`, `gulf_war_sl_1990`, `ltcm_russia_1998`, `dotcom_2000_02`, `gfc_2007_09`, `flash_crash_euro1_2010`, `us_downgrade_euro2_2011`, `china_oil_2015_16`, `fed_tightening_2018q4`, `covid_2020`, `inflation_bear_2022`, `svb_banking_2023`.
  - No detectados (2): `volmageddon_2018q1`, `tariff_selloff_2025`.
- **Pista B** · OOS 2010-04-12 → 2026-05-29 (n = 4059) · train inicial 756 sesiones, refit cada 21 (expanding) · 14 feature(s): `SP500_ret_z|SP500_vol_z|SP500_momentum|SP500_drawdown|FF_MKT_z|DGS10_change_z|credit_BaaAaa_mensual_z|term_spread_hist_z|INDPRO_yoy_z|VIX_level_z|MOVE_level_z|credit_BAA10Y_z|DFII10_change_z|corr_spx_bond`.
  - Detectados (2): `covid_2020`, `inflation_bear_2022`.
  - No detectados (7): `flash_crash_euro1_2010`, `us_downgrade_euro2_2011`, `china_oil_2015_16`, `volmageddon_2018q1`, `fed_tightening_2018q4`, `svb_banking_2023`, `tariff_selloff_2025`.
<!-- END resultados_v2 D12 -->

Narrativa de la Capa 1 (implementación v1, hallazgos e hipótesis del CHECKPOINT 2): [`historia/fichas_v1.md`](../historia/fichas_v1.md#d12).

## 3. Variantes no implementadas

- **PCA/AE + HMM** como una línea comparable (lineal frente a no lineal en la reducción) en lugar de mundos separados; no implementado más allá de D12.
- **Wasserstein k-means y *jump models*** (Horvath et al.; Nystrup et al.): rivales no-deep del AE+clustering en muestras pequeñas ([F2](F2_clustering.md); el *jump model* es [D09](F2_clustering.md#d09)).
- El error de reconstrucción del AE es otro índice de estrés: conviene contrastarlo con la regla simple sobre VIX ([F1](F1_reglas.md#d01)). Criterio del equipo: un único detector deep, ligero y no supervisado, etiquetado como exploratorio; un resultado negativo es una contribución aceptable.

## 4. Referencias

1. Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural
   Computation 9(8): 1735–1780. DOI 10.1162/neco.1997.9.8.1735. [`nn_hochreiter1997`]
2. Hinton, G. E. & Salakhutdinov, R. R. (2006). *Reducing the Dimensionality of
   Data with Neural Networks*. Science 313(5786): 504–507. DOI 10.1126/science.1127647. [`nn_hinton2006`]
3. Kingma, D. P. & Welling, M. (2014). *Auto-Encoding Variational Bayes*. ICLR
   2014. arXiv:1312.6114. [`nn_kingma2014`]
4. Xie, J., Girshick, R. & Farhadi, A. (2016). *Unsupervised Deep Embedding for
   Clustering Analysis* (DEC). ICML 2016, PMLR 48: 478–487. arXiv:1511.06335. [`nn_xie2016`]
5. An, J. & Cho, S. (2015). *Variational Autoencoder based Anomaly Detection
   using Reconstruction Probability*. SNU Data Mining Center, Tech. Report. [`nn_ancho2015`]
6. Kohonen, T. (1990). *The Self-Organizing Map*. Proceedings of the IEEE 78(9):
   1464–1480. DOI 10.1109/5.58325. [`nn_kohonen1990`]
7. Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*. MIT Press. [`nn_goodfellow2016`]
8. Bao, W., Yue, J. & Rao, Y. (2017). *A Deep Learning Framework for Financial
   Time Series Using Stacked Autoencoders and Long-Short Term Memory*. PLoS ONE
   12(7): e0180944. DOI 10.1371/journal.pone.0180944. [`nn_bao2017`]
9. Akioyamen, P., Tang, Y. Z. & Hussien, H. (2021). *A Hybrid Learning Approach
   to Detecting Regime Switches in Financial Markets*. arXiv:2108.05801. [`nn_akioyamen2021`]
10. Bucci, A. & Ciciretti, V. (2021). *Market Regime Detection via Realized
    Covariances: A Comparison between Unsupervised Learning and Nonlinear Models*.
    arXiv:2104.03667. [`clust_bucci2022realized`]

(López de Prado 2018, ya en `references.bib`, se invoca en prosa por su crítica
al backtest overfitting; no se re-declara para no duplicar la clave.)
