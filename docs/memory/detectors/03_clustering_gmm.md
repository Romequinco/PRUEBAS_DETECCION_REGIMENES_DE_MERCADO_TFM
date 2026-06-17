# D3 — `clustering_gmm` (familia CLUSTERING, baseline NO temporal)

> Mixtura gaussiana estática (`GaussianMixture`, `covariance_type='full'`) sobre
> las 15 features causales. Sin cadena de Markov: cada día se asigna de forma
> independiente al componente más probable. Es el **baseline contra el que D4
> (HMM) medirá cuánto aporta la dinámica temporal** (mismas features, misma
> evaluación walk-forward).

## Implementado

- **Clase** `detectors/clustering_gmm.py::ClusteringGMM(RegimeDetector)`.
  - `name` = `clustering_gmm_k{n_states}`.
  - `bibliography` = `["clust_twosigma2021regime", "clust_munnix2012", "gulko2002",
    "lopezdeprado2018"]` (claves verificadas en `docs/references.bib`).
  - `fit`: `GaussianMixture(covariance_type='full', n_init=5, reg_covar=1e-6,
    max_iter=300)` sobre `X_train.values`; marca `_is_fitted` y llama a
    `label_states_economically(X_train)` para fijar `self._canonical_order`.
  - `_predict_states`: etiquetas INTERNAS (`_model.predict`).
  - **`predict_proba` (override)**: `_model.predict_proba(X)[:, self._canonical_order]`
    → posteriores reales reordenados al orden canónico (0=calma .. n-1=crisis), de
    modo que la última columna es P(crisis). `predict` (duro canónico) y
    `crisis_state` se heredan del núcleo.
  - `score` = `_model.score(X) * len(X)` (log-likelihood TOTAL; sklearn da la media).
  - `n_parameters` = `(k-1) + k·d + k·d(d+1)/2` (pesos + medias + covarianzas full),
    con d=15 → 271 (k=2), 407 (k=3). Habilita AIC/BIC del núcleo.
- **`covariance_type='full'`** elegido a propósito: capta el régimen de correlación
  que cambia de signo (Gulko 2002; feature `corr_spx_bond`), que un k-means euclídeo
  o un GMM diagonal no separan.
- **Causalidad**: el detector solo mira `X_train`; el wrapper causal lo da
  `ev.walk_forward` (re-fit expanding, train inicial 8 años, step 21d). El alineado
  de etiquetas entre folds lo resuelve la canonicalización económica en cada `fit`.
- **Notebook** `notebooks/03_clustering_gmm.ipynb` (construido y ejecutado con
  `notebooks/_build_03.py`, 0 errores, 3 figuras inline): selección de k por BIC,
  sanidad del orden canónico, walk-forward k=2 y k=3, tabla de métricas, histograma
  de duraciones (flickering), S&P 500 coloreado por régimen, timeline + P(crisis), y
  bloque de verificación con asserts sobre 2008/2011 (NaN) y COVID/Inflación.
- **Artefactos**: `results/metrics_03_clustering_gmm.csv` (1 fila, detector k=3
  elegido por BIC), `results/d03_gmm_flickering.png`,
  `results/d03_gmm_sp500_regimes.png`, `results/d03_gmm_timeline.png`.

## Descubierto

### k elegido por BIC
BIC in-sample sobre el set completo: **k=2 → 70 975**, **k=3 → 63 016**. Gana
**k=3** (BIC menor). El detector principal volcado al CSV es `clustering_gmm_k3`.
Un tercer estado intermedio (estrés/transición) sí compensa su coste en parámetros.

### Política de ventana y cobertura por crisis
Las 15 features arrancan en **2007-07**; con train inicial expanding de 8 años el
primer bloque OOS empieza en **2015-09-15** y va hasta **2026-06-12** (n=2649). Por
tanto **2008 (GFC) y 2011 (EuroDebt) NO son OOS-evaluables** —quedan dentro del
primer train— y su cobertura sale **`NaN`**, que es el comportamiento CORRECTO (no
se penaliza lo que el detector no pudo ver). Igualmente la trampa **TaperTantrum
2013** cae fuera de OOS (`NaN`).

Cobertura de crisis OOS (k=3):
- **COVID_2020 = 0.96** (sensibilidad muy alta).
- **Inflation_2022 = 0.87**.
- GFC_2008 = NaN, EuroDebt_2011 = NaN (fuera de OOS, esperado).

Falsos positivos en trampas:
- TaperTantrum_2013 = NaN (fuera de OOS).
- **Selloff_Q4_2018 = 0.00** (no marca crisis sostenida en la trampa de 2018; bien).
- `false_alarm_rate` global = **0.49**: la mitad de los días marcados "crisis" caen
  fuera de las ventanas conocidas — coherente con el flickering (marca crisis
  sueltas dispersas), no con falsas alarmas sostenidas.

Lead/lag (días vs suelo de drawdown): COVID **−160 d**, Inflación **−229 d** (la
señal P(crisis)≥0.5 cruza muy por delante del suelo; ambos suelos sí caen en OOS).

### Flickering medido (talón de Aquiles)
- **switching_rate = 0.126** (k=3) / 0.112 (k=2): conmuta de estado en ~1 de cada 8
  días OOS.
- **mean_regime_duration = 7.9 d** (k=3) / 8.9 d (k=2): rachas de régimen muy
  cortas, irreales para "regímenes" de mercado.
- `label_stability = 0.976`: las etiquetas por fecha son estables entre re-fits
  (el flickering es intra-secuencia, no inestabilidad entre folds).

### Comparación con la hipótesis del CHECKPOINT 2
Hipótesis CP2: *"captará regímenes con estructura de correlación distinta; fallará
por flickering severo; no causal nativo"*. **Se cumple en los tres puntos:**
1. **Capta estructura de correlación**: con covarianza full detecta COVID (0.96) e
   Inflación (0.87) y separa el estado de crisis por retorno/vol decrecientes —la
   covarianza plena ve el cambio de signo de la correlación equity/bonos (Gulko).
2. **Flickering severo**: switching_rate 0.126 y rachas medias ~8 días confirman el
   parpadeo esperado de un modelo sin término de persistencia. Este es justamente el
   número de referencia que D4 (HMM, con matriz de transición) debería **reducir**.
3. **No causal nativo**: el GMM estático mira toda la muestra al ajustarse; la
   causalidad se logra SOLO vía el `walk_forward` (re-fit expanding), no por diseño
   del detector.

**Conclusión**: D3 es un baseline interpretable y sensible a crisis, pero su
flickering lo descarta como detector definitivo; su valor es servir de referencia
NO temporal para aislar el aporte de la dinámica markoviana del HMM.

## Fricción con el núcleo
Ninguna. La interfaz de `RegimeDetector` (override de `predict_proba` con
`_canonical_order`, `score`/`n_parameters` para AIC/BIC) y `evaluation.walk_forward`
cubrieron el caso probabilístico sin necesidad de cambios. No se modificó
`src/detector_base.py`, `src/evaluation.py`, `src/features.py` ni `INDEX.md`.
