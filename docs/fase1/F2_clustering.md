# F2 — Clustering

> **Cómo leer este documento.** La sección 1 es la teoría de la familia (redactada en la Capa 1, original en el tag `capa1-final`); la 2, los detectores implementados con su configuración v2 y sus resultados del benchmark (ADR-003): [D03 `clustering_gmm`](#d03), [D09 `jump_model`](#d09). Notebook de familia: [`notebooks/06_familia_F2_clustering.ipynb`](../../notebooks/06_familia_F2_clustering.ipynb). Bibliografía: claves `clust_*` y compartidas de [`docs/references.bib`](../references.bib).
>
> **Datos v1 en la teoría.** Las cifras del EDA que cita la sección 1 son de la Capa 1 (15 features, ventana 2007-04 → 2026-06, 4 crisis: 2008, 2011, 2020 y 2022), igual que los nombres de feature `credit_spread_z` o `yield_slope_z` y las referencias a la «tarea previa»; donde el dato importa se indica junto a él. El panel v2 tiene dos pistas (A: 1962-01-02 → 2026-05-29; B: 2007-04-11 → 2026-05-29) y el EDA vigente está en [`eda.md`](eda.md) (curtosis de exceso del S&P 500 = 18,7 en 1928-2026).

## 1. Teoría

> Estado del arte de la familia **CLUSTERING** como detector de regímenes de
> mercado. Frontera: aquí se trata el clustering *estático* (cada observación se
> asigna a un grupo de forma independiente, sin modelo de transición temporal).
> El HMM y el Markov-Switching (dinámica temporal explícita) pertenecen a otra
> familia; el GMM-HMM también. El **GMM como mixtura estática** (sin cadena de
> Markov sobre los estados) sí es nuestro.

### Definición y supuestos

El clustering es aprendizaje **no supervisado**: dado un panel de observaciones
descritas por *features* (en nuestro caso, las features causales diarias del
panel: 9 en la pista A y 14 en la B; 15 en la Capa 1), se particiona el espacio en *k* grupos ("regímenes") de modo que las
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
  **gaussiano**; con fat tails (curtosis de exceso S&P 500 ≈ 25.6 y HYG ≈ 39.6 en la ventana v1 2007-2026; el EDA v2 da 18,7 para el S&P 500 en 1928-2026) este
  supuesto se viola y la mixtura tiende a "gastar" componentes para absorber las
  colas (Münnix et al., 2012, motivan precisamente el clustering de estados como
  alternativa descriptiva).

### Variantes principales

#### k-means (y k-medoids / k-means++)
Minimiza la inercia (suma de distancias euclídeas al cuadrado al centroide).
Rápido y escalable, pero sensible a outliers y a la inicialización (mitigada por
k-means++). Referencias canónicas: Lloyd (1982) y MacQueen (1967). k-medoids
(PAM) usa medoides reales y distancias arbitrarias → más robusto a colas gordas,
a costa de cómputo. Aplicación directa a estados de mercado en Borst y en la
práctica del *quant* retail; variantes robustas con **distancia de Wasserstein**
(k-means sobre distribuciones de retornos) en Horvath, Issa & Muguruza (2021) y
su extensión *sliced* multidimensional de Luan & Hamp (2023).

#### GMM estático (mixtura gaussiana, sin cadena de Markov)
Modelo generativo probabilístico estimado por EM. Ventajas sobre k-means:
(i) asignación **blanda** (probabilidad de pertenencia a cada régimen, útil para
ponderar exposición), (ii) covarianzas plenas → captura regímenes con estructura
de correlación distinta (p. ej. el cambio de signo de la correlación
S&P500/bonos del EDA). Es la base estadística del whitepaper de Two Sigma (2021),
que ajusta un GMM sobre factores y obtiene ~4 regímenes (incluido un "crisis").
**Frontera**: si a este GMM se le añade una matriz de transición markoviana entre
componentes, deja de ser nuestro y pasa a la familia HMM/GMM-HMM.

#### Clustering jerárquico / aglomerativo
Construye un dendrograma fusionando observaciones (linkage: single, complete,
average, Ward). No exige fijar *k* a priori (se corta el árbol). En regímenes de
mercado se aplica sobre **matrices de correlación** rolling: cada ventana es un
punto, se mide distancia entre matrices y se agrupan estados (Münnix et al.,
2012; Bucci & Ciciretti, 2022, que comparan clustering aglomerativo contra un
modelo no lineal VLSTAR). Marti et al. (2021) revisan dos décadas de clustering
jerárquico de correlaciones en finanzas. También sustenta el **HRP** de López de
Prado (2018) para construcción de cartera, aunque ahí el fin es asignación, no
detección de régimen.

#### DBSCAN / HDBSCAN (basados en densidad)
DBSCAN (Ester et al., 1996) agrupa por densidad y **etiqueta outliers como ruido**
en lugar de forzarlos a un cluster → atractivo con fat tails. No exige *k*, pero
sí `eps`/`minPts`. HDBSCAN (Campello et al., 2013; McInnes et al., 2017) elimina
`eps`, maneja clusters de **densidad variable** y devuelve probabilidades de
pertenencia y un score de outlier. Inconveniente para regímenes: tiende a marcar
los días de crisis (raros, dispersos) como "ruido", justo los que más interesan
→ uso más natural como **detector de anomalías** que como particionador de
regímenes (criterio del equipo).

#### Selección de *k* (número de regímenes)
- **Silhouette** (Rousseeuw, 1987): cohesión vs separación; maximizar la media.
- **Gap statistic** (Tibshirani, Walther & Hastie, 2001): compara la inercia con
  la esperada bajo una referencia sin estructura.
- **BIC / AIC para GMM** (Schwarz, 1978): penaliza verosimilitud por nº de
  parámetros; es el criterio estándar para elegir nº de componentes y tipo de
  covarianza en mixturas.
- **ONC** (Optimal Number of Clusters) de López de Prado (2020): combina
  silhouette con re-clustering para estabilizar *k* en matrices de correlación
  financieras ruidosas.

### Fortalezas y debilidades

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
  muy sensible a los outliers, abundantes aquí (curtosis de exceso 25.6 / 39.6 en la ventana v1; 18,7 en el S&P 500 1928-2026): unos
  pocos días extremos desplazan centroides y distorsionan fronteras. Mitigaciones:
  estandarización robusta, k-medoids, GMM con covarianza plena, distancia de
  Wasserstein (Horvath et al., 2021) o densidad (HDBSCAN, que los aparta como
  ruido).
- **Causalidad / online.** Es la mayor debilidad metodológica: el clustering
  clásico usa **toda la muestra** para definir centroides/medias → **no causal**
  (look-ahead). Para usarlo en un marco walk-forward hay que **causalizarlo**
  (§4): no es online de forma nativa.

### Idoneidad para este proyecto

El clustering es el **baseline no temporal** idóneo para cuantificar *cuánto
aporta la dinámica del HMM*. La comparación HMM-gaussiano-2-estados vs.
k-means/GMM-2-clusters sobre las **mismas features causales** (15 en la Capa 1; 9/14 por pista en v2) aísla
limpiamente el valor añadido de la matriz de transición: si el HMM mejora la
persistencia, reduce el flickering y detecta antes/mejor las crisis del EDA
(GFC-2008, Euro-2011, COVID-2020, Inflación-2022 en la Capa 1; 17 crisis evaluables OOS en la pista A y 9 en la B en v2) frente a un clustering que usa
exactamente la misma información instantánea, esa diferencia *es* el aporte de la
dinámica temporal (criterio del equipo).

Atado al EDA:
- **Fat tails** → preferir GMM (covarianza plena) o variantes robustas frente a
  k-means euclídeo puro; reportar sensibilidad a outliers como riesgo conocido.
- **Correlación S&P500/bonos que cambia de signo** → el GMM con covarianzas
  plenas puede separar el régimen "diversificación" (corr negativa) del régimen
  "todo cae junto" (corr positiva); k-means con features ya incluye
  `corr_spx_bond` como dimensión, lo que ayuda.
- **Cobertura 2007-04** (v1; en v2 solo afecta a la pista B) → la causalización por re-fit en ventana *expanding* choca
  con que la GFC queda muy cerca del inicio (poca historia previa para formar
  clusters estables); declararlo como en §3 del EDA.

Rol final: **baseline interpretable y barato**, no candidato a detector
definitivo, salvo en su versión causalizada y suavizada (que ya hibrida con lo
temporal).

### Aplicaciones documentadas a regímenes de mercado financiero

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

### Coste de implementación y librería Python recomendada

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

## 2. Detectores implementados

Las tablas de resultados salen de `results/benchmark/ranking_v2.csv` y `results/benchmark/metrics_master_v2.csv` (benchmark completo 24/24, `results/benchmark/manifest.json` del 2026-09-29T12:41 UTC; cifras de [ADR-003](../decisions/ADR-003-causalidad-calendario-estado-ranking.md) §4). Walk-forward causal y evaluación OOS por pista (17 crisis evaluables en la A y 9 en la B, según `configs/benchmark_spec.yaml`). Ranking de **detección**: un evento se detecta con ≥ 3 sesiones OOS consecutivas en crisis dentro de [pico, suelo]; `score_deteccion` = F1 entre la precisión diaria (1 − FAR) y el recall por evento, con niveles previos (elegible › precisión ≤ azar › parpadeo › degenerado). Cada tabla va entre marcadores `resultados_v2`: tras un benchmark nuevo se regenera desde esos dos CSV, no se edita a mano.

<a id="d03"></a>

### D03 — `clustering_gmm` · mixtura gaussiana estática

Mixtura gaussiana **sin cadena de Markov**: cada día se asigna de forma independiente al componente más probable, así que no hay término de persistencia y se espera parpadeo. La covarianza completa permite separar regímenes con distinta estructura de correlación (cambio de signo acción-bono [`gulko2002`]). Es el baseline no temporal contra el que se mide lo que aporta la dinámica del HMM (D04, [F3](F3_hmm.md)).

**Configuración v2**

- Código: `src/regimenes/detectores/f2_clustering/clustering_gmm.py` (`ClusteringGMM`); registro, id `D03`.
- Modelo: `GaussianMixture` de scikit-learn con **K = 3**, `covariance_type='full'`, `n_init=5`, `reg_covar=1e-6`, `max_iter=300`, semilla 42; `predict_proba` devuelve los posteriores reordenados al orden canónico (0 = calma … K−1 = crisis).
- Features `core` de `regimenes.detectores.registry`: **9 en la pista A** (`CORE_A`: `SP500_ret_z`, `SP500_vol_z`, `SP500_momentum`, `SP500_drawdown`, `FF_MKT_z`, `DGS10_change_z`, `credit_BaaAaa_mensual_z`, `term_spread_hist_z`, `INDPRO_yoy_z`) y **14 en la B** (`CORE_B` = las 9 de A + `VIX_level_z`, `MOVE_level_z`, `credit_BAA10Y_z`, `DFII10_change_z`, `corr_spx_bond`).
- Walk-forward: refit cada 21 sesiones, ventana expanding.

<!-- BEGIN resultados_v2 D03 -->
**Resultados v2 (ADR-003)**

| Métrica | Pista A | Pista B |
|---|---:|---:|
| Variante evaluada | `clustering_gmm_k3` | `clustering_gmm_k3` |
| Puesto de detección | **3** / 12 | **10** / 12 |
| Nivel de etiqueta | `elegible` | `elegible` |
| `score_deteccion` | 0.565 | 0.241 |
| Recall por evento | 0.882 (15/17) | 0.222 (2/9) |
| Precisión diaria (lift vs base) | 0.416 (×2.14) | 0.262 (×1.52) |
| Cobertura media de crisis | 0.530 | 0.034 |
| Tasa de falsas alarmas (FAR) | 0.584 | 0.738 |
| Activación media en trampas | 0.005 | 0.000 |
| Switching rate | 0.0962 | 0.0436 |
| Duración media de régimen (sesiones) | 10.4 | 22.8 |
| Estabilidad de etiqueta | 0.983 | 0.937 |
| Puesto legacy (`rank_medio`) | 11 (8.1) | 10 (9.0) |

**Ventana OOS y eventos por pista**

- **Pista A** · OOS 1970-05-12 → 2026-05-29 (n = 14133) · train inicial 2016 sesiones, refit cada 21 (expanding) · 9 feature(s): `SP500_ret_z|SP500_vol_z|SP500_momentum|SP500_drawdown|FF_MKT_z|DGS10_change_z|credit_BaaAaa_mensual_z|term_spread_hist_z|INDPRO_yoy_z`.
  - Detectados (15): `bear_1969_70`, `oil_stagflation_1973`, `volcker_1980_82`, `black_monday_1987`, `gulf_war_sl_1990`, `ltcm_russia_1998`, `dotcom_2000_02`, `gfc_2007_09`, `flash_crash_euro1_2010`, `us_downgrade_euro2_2011`, `china_oil_2015_16`, `fed_tightening_2018q4`, `covid_2020`, `inflation_bear_2022`, `tariff_selloff_2025`.
  - No detectados (2): `volmageddon_2018q1`, `svb_banking_2023`.
- **Pista B** · OOS 2010-04-12 → 2026-05-29 (n = 4059) · train inicial 756 sesiones, refit cada 21 (expanding) · 14 feature(s): `SP500_ret_z|SP500_vol_z|SP500_momentum|SP500_drawdown|FF_MKT_z|DGS10_change_z|credit_BaaAaa_mensual_z|term_spread_hist_z|INDPRO_yoy_z|VIX_level_z|MOVE_level_z|credit_BAA10Y_z|DFII10_change_z|corr_spx_bond`.
  - Detectados (2): `china_oil_2015_16`, `covid_2020`.
  - No detectados (7): `flash_crash_euro1_2010`, `us_downgrade_euro2_2011`, `volmageddon_2018q1`, `fed_tightening_2018q4`, `inflation_bear_2022`, `svb_banking_2023`, `tariff_selloff_2025`.
<!-- END resultados_v2 D03 -->

Narrativa de la Capa 1 (implementación v1, hallazgos e hipótesis del CHECKPOINT 2): [`historia/fichas_v1.md`](../historia/fichas_v1.md#d03).

<a id="d09"></a>

### D09 — `jump_model` · Statistical Jump Model

Clustering de estados con **penalización de salto** λ [`hmm_nystrup2020`]: minimiza la distancia intra-cluster más λ veces el número de cambios de estado, lo que induce persistencia («histéresis aprendida») sin matriz de transición. Es el rival directo de D03 (mismo espacio de features, sin persistencia) y de D12.

**Configuración v2**

- Código: `src/regimenes/detectores/f2_clustering/jump_model.py` (`JumpModel`, sobre la librería `jumpmodels`); registro, id `D09`.
- Parámetros: **K = 2**, `jump_penalty = 50`, semilla 42. `StandardScaler` ajustado solo con el train del fold; la predicción usa `predict_online` (la etiqueta del día `t` solo usa filas anteriores).
- Features `core` de `regimenes.detectores.registry`: **9 en la pista A** (`CORE_A`: `SP500_ret_z`, `SP500_vol_z`, `SP500_momentum`, `SP500_drawdown`, `FF_MKT_z`, `DGS10_change_z`, `credit_BaaAaa_mensual_z`, `term_spread_hist_z`, `INDPRO_yoy_z`) y **14 en la B** (`CORE_B` = las 9 de A + `VIX_level_z`, `MOVE_level_z`, `credit_BAA10Y_z`, `DFII10_change_z`, `corr_spx_bond`).
- Walk-forward: refit cada 21 sesiones, ventana expanding.

<!-- BEGIN resultados_v2 D09 -->
**Resultados v2 (ADR-003)**

| Métrica | Pista A | Pista B |
|---|---:|---:|
| Variante evaluada | `jump_model_k2_lam50` | `jump_model_k2_lam50` |
| Puesto de detección | **12** / 12 | **11** / 12 |
| Nivel de etiqueta | `elegible` | `precision_no_supera_azar` |
| `score_deteccion` | 0.362 | 0.000 |
| Recall por evento | 0.471 (8/17) | 0.000 (0/9) |
| Precisión diaria (lift vs base) | 0.294 (×1.51) | 0.071 (×0.41) |
| Cobertura media de crisis | 0.321 | 0.009 |
| Tasa de falsas alarmas (FAR) | 0.706 | 0.929 |
| Activación media en trampas | 0.000 | 0.000 |
| Switching rate | 0.0021 | 0.0005 |
| Duración media de régimen (sesiones) | 471.1 | 1353.0 |
| Estabilidad de etiqueta | 0.988 | 0.999 |
| Puesto legacy (`rank_medio`) | 7 (7.4) | 7 (6.6) |

**Ventana OOS y eventos por pista**

- **Pista A** · OOS 1970-05-12 → 2026-05-29 (n = 14133) · train inicial 2016 sesiones, refit cada 21 (expanding) · 9 feature(s): `SP500_ret_z|SP500_vol_z|SP500_momentum|SP500_drawdown|FF_MKT_z|DGS10_change_z|credit_BaaAaa_mensual_z|term_spread_hist_z|INDPRO_yoy_z`.
  - Detectados (8): `bear_1969_70`, `oil_stagflation_1973`, `volcker_1980_82`, `black_monday_1987`, `dotcom_2000_02`, `gfc_2007_09`, `covid_2020`, `svb_banking_2023`.
  - No detectados (9): `gulf_war_sl_1990`, `ltcm_russia_1998`, `flash_crash_euro1_2010`, `us_downgrade_euro2_2011`, `china_oil_2015_16`, `volmageddon_2018q1`, `fed_tightening_2018q4`, `inflation_bear_2022`, `tariff_selloff_2025`.
- **Pista B** · OOS 2010-04-12 → 2026-05-29 (n = 4059) · train inicial 756 sesiones, refit cada 21 (expanding) · 14 feature(s): `SP500_ret_z|SP500_vol_z|SP500_momentum|SP500_drawdown|FF_MKT_z|DGS10_change_z|credit_BaaAaa_mensual_z|term_spread_hist_z|INDPRO_yoy_z|VIX_level_z|MOVE_level_z|credit_BAA10Y_z|DFII10_change_z|corr_spx_bond`.
  - Detectados (0): ninguno.
  - No detectados (9): `flash_crash_euro1_2010`, `us_downgrade_euro2_2011`, `china_oil_2015_16`, `volmageddon_2018q1`, `fed_tightening_2018q4`, `covid_2020`, `inflation_bear_2022`, `svb_banking_2023`, `tariff_selloff_2025`.
<!-- END resultados_v2 D09 -->

Narrativa de la Capa 1 (implementación v1, hallazgos e hipótesis del CHECKPOINT 2): [`historia/fichas_v1.md`](../historia/fichas_v1.md#d09).

## 3. Variantes no implementadas

- **Wasserstein / transporte óptimo como métrica** (Horvath et al.; Luan & Hamp): sustituye la distancia euclídea del k-means; no implementado.
- **Representation learning + clustering** (p. ej. arXiv:2410.22346) y redes CNN/Siamesas: híbridos deep cubiertos de forma exploratoria en [F7](F7_redes_neuronales.md#d12) (D12).
- La persistencia temporal se aborda con los *jump models* ([D09](#d09)) y con las familias HMM y change-point ([F3](F3_hmm.md), [F6](F6_change_point.md)).

## 4. Referencias

1. Münnix, M. C.; Shimada, T.; Schäfer, R.; Leyvraz, F.; Seligman, T. H.; Guhr,
   T.; Stanley, H. E. (2012). *Identifying States of a Financial Market*.
   Scientific Reports 2:644. DOI: 10.1038/srep00644. [`clust_munnix2012`]
2. Horvath, B.; Issa, Z.; Muguruza, A. (2021). *Clustering Market Regimes using
   the Wasserstein Distance*. arXiv:2110.11848; publicado en Journal of
   Computational Finance 28(1), 2024. [`clust_horvath2021wasserstein`]
3. Luan, Q.; Hamp, J. (2023). *Automated regime detection in multidimensional
   time series data using sliced Wasserstein k-means clustering*.
   arXiv:2310.01285. [`clust_luan2023swkmeans`]
4. Bucci, A.; Ciciretti, V. (2022). *Market regime detection via realized
   covariances: A comparison between unsupervised learning and nonlinear models*.
   Economic Modelling. DOI: 10.1016/j.econmod.2022.105792 (arXiv:2104.03667). [`clust_bucci2022realized`]
5. Two Sigma (2021). *A Machine Learning Approach to Regime Modeling*
   (whitepaper). https://www.twosigma.com/articles/a-machine-learning-approach-to-regime-modeling/ [`clust_twosigma2021regime`]
6. López de Prado, M. (2020). *Machine Learning for Asset Managers* (Elements in
   Quantitative Finance). Cambridge University Press. DOI: 10.1017/9781108883658.
   (Algoritmo ONC para nº óptimo de clusters.) [`clust_lopezdeprado2020`]
7. Marti, G.; Nielsen, F.; Bińkowski, M.; Donnat, P. (2021). *A review of two
   decades of correlations, hierarchies, networks and clustering in financial
   markets*. En Progress in Information Geometry, Springer (arXiv:1703.00485). [`clust_marti2021review`]
8. Tibshirani, R.; Walther, G.; Hastie, T. (2001). *Estimating the number of
   clusters in a data set via the gap statistic*. JRSS-B 63(2):411–423. DOI:
   10.1111/1467-9868.00293. [`clust_tibshirani2001gap`]
9. Rousseeuw, P. J. (1987). *Silhouettes: a graphical aid to the interpretation
   and validation of cluster analysis*. J. Comput. Appl. Math. 20:53–65. DOI:
   10.1016/0377-0427(87)90125-7. [`clust_rousseeuw1987silhouette`]
10. Schwarz, G. (1978). *Estimating the dimension of a model* (BIC). Annals of
    Statistics 6(2):461–464. DOI: 10.1214/aos/1176344136. [`clust_schwarz1978bic`]
11. Campello, R. J. G. B.; Moulavi, D.; Sander, J. (2013). *Density-based
    clustering based on hierarchical density estimates* (HDBSCAN). PAKDD. DOI:
    10.1007/978-3-642-37456-2_14. [`clust_campello2013hdbscan`]
12. McInnes, L.; Healy, J.; Astels, S. (2017). *hdbscan: Hierarchical density
    based clustering*. JOSS 2(11):205. DOI: 10.21105/joss.00205. [`clust_mcinnes2017hdbscan`]
13. Ester, M.; Kriegel, H.-P.; Sander, J.; Xu, X. (1996). *A density-based
    algorithm for discovering clusters in large spatial databases with noise*
    (DBSCAN). KDD-96, pp. 226–231. [`clust_ester1996dbscan`]
14. Lloyd, S. P. (1982). *Least squares quantization in PCM*. IEEE Trans. Inf.
    Theory 28(2):129–137. DOI: 10.1109/TIT.1982.1056489. [`clust_lloyd1982kmeans`]
15. MacQueen, J. (1967). *Some methods for classification and analysis of
    multivariate observations*. Proc. 5th Berkeley Symp. on Math. Stat. and
    Probability, 1:281–297. [`clust_macqueen1967`]
16. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*.
    JMLR 12:2825–2830. [`clust_pedregosa2011sklearn`]
