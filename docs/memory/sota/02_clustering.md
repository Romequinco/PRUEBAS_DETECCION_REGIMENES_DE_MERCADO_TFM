# 02 — Clustering

> Estado del arte de la familia **CLUSTERING** como detector de regímenes de
> mercado. Frontera: aquí se trata el clustering *estático* (cada observación se
> asigna a un grupo de forma independiente, sin modelo de transición temporal).
> El HMM y el Markov-Switching (dinámica temporal explícita) pertenecen a otra
> familia; el GMM-HMM también. El **GMM como mixtura estática** (sin cadena de
> Markov sobre los estados) sí es nuestro.

## Definición y supuestos

El clustering es aprendizaje **no supervisado**: dado un panel de observaciones
descritas por *features* (en nuestro caso, las 15 features causales diarias del
EDA), se particiona el espacio en *k* grupos ("regímenes") de modo que las
observaciones del mismo grupo sean más parecidas entre sí que con las de otros
grupos, según una métrica de distancia/similitud. La interpretación financiera:
cada cluster ≈ un "estado de mercado" (calma, estrés, risk-off, etc.).

Supuestos clave que conviene tener presentes para criticar la idoneidad:

- **Asignación independiente por observación.** No hay matriz de transición ni
  noción de persistencia: el régimen del día *t* no condiciona el del día *t+1*.
  Esto es la diferencia esencial frente al HMM (criterio del equipo, alineado con
  la literatura: el HMM añade dinámica markoviana sobre los mismos componentes
  gaussianos que un GMM estático).
- **Métrica de distancia.** k-means asume distancia **euclídea** (clusters
  esféricos, isotrópicos, de tamaño similar). El GMM relaja esto admitiendo
  covarianzas plenas por componente (clusters elípticos y de distinta densidad).
- **Estacionariedad implícita.** El clustering clásico forma los grupos con
  **toda la muestra a la vez** → es, por construcción, **no causal** (mira el
  futuro al definir centroides/medias). Causalizarlo exige un protocolo explícito
  (ver §3 y §4).
- **Distribución de los componentes.** El GMM supone que cada componente es
  **gaussiano**; con fat tails (kurtosis exceso S&P 500 ≈ 25.6, HYG ≈ 39.6) este
  supuesto se viola y la mixtura tiende a "gastar" componentes para absorber las
  colas (Münnix et al., 2012, motivan precisamente el clustering de estados como
  alternativa descriptiva).

## Variantes principales

### k-means (y k-medoids / k-means++)
Minimiza la inercia (suma de distancias euclídeas al cuadrado al centroide).
Rápido y escalable, pero sensible a outliers y a la inicialización (mitigada por
k-means++). Referencias canónicas: Lloyd (1982) y MacQueen (1967). k-medoids
(PAM) usa medoides reales y distancias arbitrarias → más robusto a colas gordas,
a costa de cómputo. Aplicación directa a estados de mercado en Borst y en la
práctica del *quant* retail; variantes robustas con **distancia de Wasserstein**
(k-means sobre distribuciones de retornos) en Horvath, Issa & Muguruza (2021) y
su extensión *sliced* multidimensional de Luan & Hamp (2023).

### GMM estático (mixtura gaussiana, sin cadena de Markov)
Modelo generativo probabilístico estimado por EM. Ventajas sobre k-means:
(i) asignación **blanda** (probabilidad de pertenencia a cada régimen, útil para
ponderar exposición), (ii) covarianzas plenas → captura regímenes con estructura
de correlación distinta (p. ej. el cambio de signo de la correlación
S&P500/bonos del EDA). Es la base estadística del whitepaper de Two Sigma (2021),
que ajusta un GMM sobre factores y obtiene ~4 regímenes (incluido un "crisis").
**Frontera**: si a este GMM se le añade una matriz de transición markoviana entre
componentes, deja de ser nuestro y pasa a la familia HMM/GMM-HMM.

### Clustering jerárquico / aglomerativo
Construye un dendrograma fusionando observaciones (linkage: single, complete,
average, Ward). No exige fijar *k* a priori (se corta el árbol). En regímenes de
mercado se aplica sobre **matrices de correlación** rolling: cada ventana es un
punto, se mide distancia entre matrices y se agrupan estados (Münnix et al.,
2012; Bucci & Ciciretti, 2022, que comparan clustering aglomerativo contra un
modelo no lineal VLSTAR). Marti et al. (2021) revisan dos décadas de clustering
jerárquico de correlaciones en finanzas. También sustenta el **HRP** de López de
Prado (2018) para construcción de cartera, aunque ahí el fin es asignación, no
detección de régimen.

### DBSCAN / HDBSCAN (basados en densidad)
DBSCAN (Ester et al., 1996) agrupa por densidad y **etiqueta outliers como ruido**
en lugar de forzarlos a un cluster → atractivo con fat tails. No exige *k*, pero
sí `eps`/`minPts`. HDBSCAN (Campello et al., 2013; McInnes et al., 2017) elimina
`eps`, maneja clusters de **densidad variable** y devuelve probabilidades de
pertenencia y un score de outlier. Inconveniente para regímenes: tiende a marcar
los días de crisis (raros, dispersos) como "ruido", justo los que más interesan
→ uso más natural como **detector de anomalías** que como particionador de
regímenes (criterio del equipo).

### Selección de *k* (número de regímenes)
- **Silhouette** (Rousseeuw, 1987): cohesión vs separación; maximizar la media.
- **Gap statistic** (Tibshirani, Walther & Hastie, 2001): compara la inercia con
  la esperada bajo una referencia sin estructura.
- **BIC / AIC para GMM** (Schwarz, 1978): penaliza verosimilitud por nº de
  parámetros; es el criterio estándar para elegir nº de componentes y tipo de
  covarianza en mixturas.
- **ONC** (Optimal Number of Clusters) de López de Prado (2020): combina
  silhouette con re-clustering para estabilizar *k* en matrices de correlación
  financieras ruidosas.

## Fortalezas y debilidades

**Fortalezas**
- Simple, transparente, barato y rápido → excelente **baseline**.
- No requiere etiquetas (los regímenes no son observables).
- El GMM da pertenencias blandas y captura covarianzas régimen-dependientes
  (clave dado el cambio de signo de correlaciones del EDA).
- DBSCAN/HDBSCAN aíslan outliers explícitamente.

**Debilidades (atadas a nuestro problema)**
- **¿Capta crisis rápidas?** Parcialmente. Como cada día se etiqueta por sus
  features (VIX, drawdown, momentum), un salto brusco de volatilidad puede caer de
  inmediato en el cluster "estrés" sin esperar a una transición → reacción rápida,
  *pero* sin memoria: si un día aislado se calma, vuelve al cluster "calma".
- **Flickering (parpadeo).** Es el talón de Aquiles: sin término de persistencia
  temporal, la secuencia de etiquetas **parpadea** día a día alrededor de las
  fronteras entre clusters, generando regímenes irrealmente cortos y muchas
  transiciones espurias. El HMM amortigua esto con su matriz de transición; el
  clustering necesitaría post-proceso (suavizado, *minimum dwell time*, HMM sobre
  las etiquetas) que ya lo acerca a la familia temporal.
- **Sensibilidad a fat tails / outliers.** k-means con distancia **euclídea** es
  muy sensible a los outliers, abundantes aquí (kurtosis exceso 25.6 / 39.6): unos
  pocos días extremos desplazan centroides y distorsionan fronteras. Mitigaciones:
  estandarización robusta, k-medoids, GMM con covarianza plena, distancia de
  Wasserstein (Horvath et al., 2021) o densidad (HDBSCAN, que los aparta como
  ruido).
- **Causalidad / online.** Es la mayor debilidad metodológica: el clustering
  clásico usa **toda la muestra** para definir centroides/medias → **no causal**
  (look-ahead). Para usarlo en un marco walk-forward hay que **causalizarlo**
  (§4): no es online de forma nativa.

## Idoneidad para este proyecto

El clustering es el **baseline no temporal** idóneo para cuantificar *cuánto
aporta la dinámica del HMM*. La comparación HMM-gaussiano-2-estados vs.
k-means/GMM-2-clusters sobre las **mismas 15 features causales** aísla
limpiamente el valor añadido de la matriz de transición: si el HMM mejora la
persistencia, reduce el flickering y detecta antes/mejor las crisis del EDA
(GFC-2008, Euro-2011, COVID-2020, Inflación-2022) frente a un clustering que usa
exactamente la misma información instantánea, esa diferencia *es* el aporte de la
dinámica temporal (criterio del equipo).

Atado al EDA:
- **Fat tails** → preferir GMM (covarianza plena) o variantes robustas frente a
  k-means euclídeo puro; reportar sensibilidad a outliers como riesgo conocido.
- **Correlación S&P500/bonos que cambia de signo** → el GMM con covarianzas
  plenas puede separar el régimen "diversificación" (corr negativa) del régimen
  "todo cae junto" (corr positiva); k-means con features ya incluye
  `corr_spx_bond` como dimensión, lo que ayuda.
- **Cobertura 2007-04** → la causalización por re-fit en ventana *expanding* choca
  con que la GFC queda muy cerca del inicio (poca historia previa para formar
  clusters estables); declararlo como en §3 del EDA.

Rol final: **baseline interpretable y barato**, no candidato a detector
definitivo, salvo en su versión causalizada y suavizada (que ya hibrida con lo
temporal).

## Aplicaciones documentadas a regímenes de mercado financiero

- **Münnix, Schäfer & Guhr et al. (2012), *Scientific Reports*** — definen
  "estados" de mercado clusterizando matrices de correlación del S&P 500
  (1992–2010) y detectan puntos de cambio drástico de estructura. Referente
  fundacional del clustering de estados.
- **Two Sigma (2021), whitepaper** — GMM estático sobre los factores del *Two
  Sigma Factor Lens*; identifica ~4 regímenes con medias, vols y correlaciones
  distintas, incluido un "crisis". Uso explícito en *asset allocation*.
- **Horvath, Issa & Muguruza (2021; Journal of Computational Finance, 2024)** —
  Wasserstein k-means: clustering robusto de regímenes sobre distribuciones de
  retornos, sin supuestos de modelo.
- **Luan & Hamp (2023)** — *sliced* Wasserstein k-means para series
  multidimensionales; identifica regímenes en datos FX reales.
- **Bucci & Ciciretti (2022), *Economic Modelling*** — clustering aglomerativo
  sobre covarianzas realizadas vs. VLSTAR; útil porque **compara** clustering no
  supervisado con un modelo de transición no lineal (paralelo a nuestro
  baseline-vs-HMM).
- **López de Prado (2018; 2020)** — clustering de correlaciones para taxonomía de
  activos, HRP y el algoritmo ONC (selección de *k*); marco metodológico de
  referencia para clustering financiero robusto.

## Coste de implementación y librería Python recomendada

Coste **bajo**. Implementación inmediata con:

- **scikit-learn** (Pedregosa et al., 2011): `KMeans`/`MiniBatchKMeans`,
  `GaussianMixture` (con `bic()`/`aic()` y `covariance_type`),
  `AgglomerativeClustering`, `DBSCAN`, `silhouette_score`. Cubre el 90% de las
  variantes y la selección de *k*.
- **hdbscan** (`scikit-learn-contrib/hdbscan`, McInnes et al., 2017) para HDBSCAN
  con probabilidades y scores de outlier.
- **scipy** (`scipy.cluster.hierarchy`) para dendrogramas/linkage y cortes
  jerárquicos personalizados.
- (Opcional) `POT` (Python Optimal Transport) si se explora la variante
  Wasserstein de Horvath et al.

Esfuerzo principal: **no** el algoritmo sino el *wrapper causal* walk-forward
(re-fit expanding + asignación de nuevos puntos por distancia/probabilidad sin
reentrenar con el futuro) y el alineado de etiquetas entre re-fits (los clusters
no tienen identidad estable entre reajustes → hay que **emparejar** clusters por
centroide/medias, p. ej. con asignación húngara). Ese alineado es el coste real
de hacer el baseline comparable y sin look-ahead (criterio del equipo).

## Referencias

1. Münnix, M. C.; Shimada, T.; Schäfer, R.; Leyvraz, F.; Seligman, T. H.; Guhr,
   T.; Stanley, H. E. (2012). *Identifying States of a Financial Market*.
   Scientific Reports 2:644. DOI: 10.1038/srep00644.
2. Horvath, B.; Issa, Z.; Muguruza, A. (2021). *Clustering Market Regimes using
   the Wasserstein Distance*. arXiv:2110.11848; publicado en Journal of
   Computational Finance 28(1), 2024.
3. Luan, Q.; Hamp, J. (2023). *Automated regime detection in multidimensional
   time series data using sliced Wasserstein k-means clustering*.
   arXiv:2310.01285.
4. Bucci, A.; Ciciretti, V. (2022). *Market regime detection via realized
   covariances: A comparison between unsupervised learning and nonlinear models*.
   Economic Modelling. DOI: 10.1016/j.econmod.2022.105792 (arXiv:2104.03667).
5. Two Sigma (2021). *A Machine Learning Approach to Regime Modeling*
   (whitepaper). https://www.twosigma.com/articles/a-machine-learning-approach-to-regime-modeling/
6. López de Prado, M. (2020). *Machine Learning for Asset Managers* (Elements in
   Quantitative Finance). Cambridge University Press. DOI: 10.1017/9781108883658.
   (Algoritmo ONC para nº óptimo de clusters.)
7. Marti, G.; Nielsen, F.; Bińkowski, M.; Donnat, P. (2021). *A review of two
   decades of correlations, hierarchies, networks and clustering in financial
   markets*. En Progress in Information Geometry, Springer (arXiv:1703.00485).
8. Tibshirani, R.; Walther, G.; Hastie, T. (2001). *Estimating the number of
   clusters in a data set via the gap statistic*. JRSS-B 63(2):411–423. DOI:
   10.1111/1467-9868.00293.
9. Rousseeuw, P. J. (1987). *Silhouettes: a graphical aid to the interpretation
   and validation of cluster analysis*. J. Comput. Appl. Math. 20:53–65. DOI:
   10.1016/0377-0427(87)90125-7.
10. Schwarz, G. (1978). *Estimating the dimension of a model* (BIC). Annals of
    Statistics 6(2):461–464. DOI: 10.1214/aos/1176344136.
11. Campello, R. J. G. B.; Moulavi, D.; Sander, J. (2013). *Density-based
    clustering based on hierarchical density estimates* (HDBSCAN). PAKDD. DOI:
    10.1007/978-3-642-37456-2_14.
12. McInnes, L.; Healy, J.; Astels, S. (2017). *hdbscan: Hierarchical density
    based clustering*. JOSS 2(11):205. DOI: 10.21105/joss.00205.
13. Ester, M.; Kriegel, H.-P.; Sander, J.; Xu, X. (1996). *A density-based
    algorithm for discovering clusters in large spatial databases with noise*
    (DBSCAN). KDD-96, pp. 226–231.
14. Lloyd, S. P. (1982). *Least squares quantization in PCM*. IEEE Trans. Inf.
    Theory 28(2):129–137. DOI: 10.1109/TIT.1982.1056489.
15. MacQueen, J. (1967). *Some methods for classification and analysis of
    multivariate observations*. Proc. 5th Berkeley Symp. on Math. Stat. and
    Probability, 1:281–297.
16. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*.
    JMLR 12:2825–2830.

## Candidatas adicionales (para el sintetizador)

> Familias/enfoques relevantes que **no** son CLUSTERING estático y que NO
> desarrollo aquí; las anoto para el sintetizador:

- **HMM / GMM-HMM y Markov-Switching (familia temporal).** Aparecen
  constantemente como el "siguiente paso" sobre el clustering (añaden matriz de
  transición que reduce el flickering). Bucci & Ciciretti (2022) y el whitepaper
  de Two Sigma viven en esta frontera. *No es mía.*
- **Jump models / Statistical Jump Models.** Clustering con penalización de
  saltos temporales (impone persistencia tipo *minimum dwell*): puente exacto
  entre clustering estático y dinámica temporal. Visto en MDPI Mathematics
  13/2837 (regime-switching asset allocation con jump model + MPC). Candidata
  fuerte para el sintetizador como "clustering con persistencia".
- **Change-point detection / segmentación.** Detecta puntos de cambio de
  estructura (el propio Münnix et al. roza esto). Familia distinta.
- **Representation learning / autoencoders + clustering** (p. ej. arXiv:2410.22346,
  "Representation Learning for Regime Detection in Block Hierarchical Financial
  Markets") y modelos basados en redes (CNN/Siamese). Híbrido deep + clustering;
  fuera de mi familia básica.
- **Wasserstein / Optimal Transport como métrica** (Horvath et al.; Luan & Hamp):
  lo cito porque es k-means, pero el componente OT podría tratarse como línea
  metodológica propia por el sintetizador.
