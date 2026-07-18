# 07 — Redes neuronales / no supervisado moderno

> Familia: métodos deep / no supervisado moderno para detección de regímenes.
> Núcleo: autoencoder (incl. VAE) + clustering del espacio latente; LSTM/GRU;
> detección de anomalías por error de reconstrucción; SOM; e híbridos deep+HMM.
> Aviso de honestidad: gran parte de esta literatura es reciente, heterogénea y
> con fuerte sesgo de publicación (se reportan los aciertos). Distinguimos abajo
> lo consolidado (LSTM, autoencoder, VAE como herramientas) de lo especulativo
> (su ventaja real sobre HMM/clustering en muestras pequeñas de régimen).

## Definición y supuestos

Bajo este paraguas caen modelos de redes neuronales que aprenden una
**representación** de la dinámica de mercado y, sobre ella, separan estados. Dos
lógicas distintas conviven:

1. **No supervisado / representación → clustering.** Un autoencoder (AE) o un
   variational autoencoder (VAE) comprime las 15 features causales a un espacio
   latente de baja dimensión; luego se agrupa ese latente con k-means/GMM, o se
   entrena clustering y representación a la vez (Deep Embedded Clustering, DEC,
   Xie et al. 2016). Supuesto central: existe una variedad (manifold) de baja
   dimensión donde los regímenes son más separables que en el espacio original.
   **No asume gaussianidad de los retornos** —es su ventaja frente al HMM
   gaussiano dado el EDA (kurtosis exceso S&P 500 ≈ 25.6)— pero **sí asume que
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
régimen de datos abundante. Aquí la ventana común empieza en 2007-04-11 (~4 700
observaciones diarias) con **solo 4 episodios de crisis** (2008, 2011, 2020,
2022). El número efectivo de ejemplos de la clase "crisis" es de orden decenas de
eventos, no miles. Eso choca de frente con el supuesto.

## Variantes principales

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

## Fortalezas y debilidades

**Fortalezas (potenciales, condicionadas a datos):**
- **No asumen normalidad.** Frente al HMM gaussiano, capturan no linealidades y
  colas gordas implícitamente — relevante dado el EDA (kurtosis 25.6 en S&P,
  39.6 en HYG).
- **Multivariante y de alta dimensión.** Escalan a muchas features y a relaciones
  no lineales (p. ej. el cambio de signo de la correlación S&P/bonos, Gulko 2002)
  sin especificación paramétrica.
- **Anomaly-AE capta crisis rápidas en principio.** Como reacciona a la "rareza"
  instantánea, una caída brusca (COVID-2020) dispara el error de reconstrucción
  sin esperar a acumular evidencia — ventaja frente a un HMM que puede tardar en
  conmutar. (Conjetura razonable, no garantizada con esta muestra.)

**Debilidades (las decisivas para este proyecto):**
- **Overfitting con muestra pequeña y pocas crisis.** Es *la* objeción. Un deep
  net tiene de miles a millones de parámetros; con ~4 700 días autocorrelacionados
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

## Idoneidad para este proyecto

**Veredicto honesto: baseline avanzado OPCIONAL, no prioritario; y como detector
deep "puro" supervisado, injustificable con los datos actuales.**

Razonamiento atado al EDA:
- La escasez (ventana desde 2007, 4 crisis) es **incompatible** con entrenar de
  forma fiable un LSTM/VAE profundo por fold en walk-forward. La tensión de
  cobertura ya señalada en `01_data_and_eda.md` (la GFC pegada al inicio de
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

## Aplicaciones documentadas a regímenes de mercado financiero

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

## Coste de implementación y librería Python recomendada

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

## Referencias

1. Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural
   Computation 9(8): 1735–1780. DOI 10.1162/neco.1997.9.8.1735.
2. Hinton, G. E. & Salakhutdinov, R. R. (2006). *Reducing the Dimensionality of
   Data with Neural Networks*. Science 313(5786): 504–507. DOI 10.1126/science.1127647.
3. Kingma, D. P. & Welling, M. (2014). *Auto-Encoding Variational Bayes*. ICLR
   2014. arXiv:1312.6114.
4. Xie, J., Girshick, R. & Farhadi, A. (2016). *Unsupervised Deep Embedding for
   Clustering Analysis* (DEC). ICML 2016, PMLR 48: 478–487. arXiv:1511.06335.
5. An, J. & Cho, S. (2015). *Variational Autoencoder based Anomaly Detection
   using Reconstruction Probability*. SNU Data Mining Center, Tech. Report.
6. Kohonen, T. (1990). *The Self-Organizing Map*. Proceedings of the IEEE 78(9):
   1464–1480. DOI 10.1109/5.58325.
7. Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*. MIT Press.
8. Bao, W., Yue, J. & Rao, Y. (2017). *A Deep Learning Framework for Financial
   Time Series Using Stacked Autoencoders and Long-Short Term Memory*. PLoS ONE
   12(7): e0180944. DOI 10.1371/journal.pone.0180944.
9. Akioyamen, P., Tang, Y. Z. & Hussien, H. (2021). *A Hybrid Learning Approach
   to Detecting Regime Switches in Financial Markets*. arXiv:2108.05801.
10. Bucci, A. & Ciciretti, V. (2021). *Market Regime Detection via Realized
    Covariances: A Comparison between Unsupervised Learning and Nonlinear Models*.
    arXiv:2104.03667.

(López de Prado 2018, ya en `references.bib`, se invoca en prosa por su crítica
al backtest overfitting; no se re-declara para no duplicar la clave.)

## Candidatas adicionales (para el sintetizador)

- **Híbrido deep+HMM como puente entre familias.** El patrón "AE/VAE reduce →
  HMM sobre el latente" conecta esta familia con la del HMM (fichero de HMM /
  Nystrup et al. 2018, *Dynamic portfolio optimization across hidden market
  regimes*, Quantitative Finance 18(1)). Recomendación al sintetizador:
  considerar **PCA/AE + HMM** como una sola línea comparable (lineal vs. no
  lineal en la reducción) en lugar de tratar deep y HMM como mundos aparte.
- **Clustering del espacio latente** se solapa con la familia "Clustering"
  (k-means/GMM): la decisión AE vs. PCA antes del clustering es el verdadero
  punto de comparación; no duplicar el clustering en sí, sino el *reductor*.
- **Wasserstein k-means de regímenes** (Horvath et al., arXiv:2110.11848) y
  **statistical jump models** (Nystrup et al.) son alternativas no-deep que
  imponen persistencia — pertenecen a Clustering/Change-point pero son rivales
  directos del AE+clustering en muestras pequeñas; el sintetizador debería
  contrastarlas frente a cualquier propuesta deep.
- **AE de anomalías como "stress index"** se solapa conceptualmente con
  reglas/umbrales sobre VIX y spreads (familia Reglas): el error de
  reconstrucción es, en el fondo, otro índice de estrés a umbralar; conviene
  benchmarkearlo contra el simple VIX>percentil para ver si el AE aporta algo.
- (criterio del equipo) Recomendar al sintetizador que, si entra cualquier
  detector deep, sea **uno solo, ligero, no supervisado**, etiquetado como
  exploratorio, con resultado-negativo aceptable como contribución.
