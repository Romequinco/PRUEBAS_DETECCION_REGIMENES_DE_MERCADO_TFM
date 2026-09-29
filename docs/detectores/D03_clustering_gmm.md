# D3 — `clustering_gmm` (familia CLUSTERING, baseline NO temporal)

<!-- BEGIN ubicacion_v2 -->
> **Ubicación.** Código: `src/regimenes/detectores/f2_clustering/clustering_gmm.py` (registro: `regimenes.detectores.registry`, id `D03`) · Notebook v2 de familia: [`notebooks/06_familia_F2_clustering.ipynb`](../../notebooks/06_familia_F2_clustering.ipynb) · Resultados v2: sección [Resultados v2 (ADR-003)](#resultados-v2-adr-003) al final. Esta ficha nació en la Capa 1 (v1, congelada en el tag `capa1-final`); el texto histórico se conserva. Cuando menciona el núcleo (`src/`, `detector_base.py`, `evaluation.py`, `features.py`, `data_loader.py`, `INDEX.md`) se refiere al de la Capa 1: hoy `detector_base.py` → `src/regimenes/detectores/base.py`, `evaluation.py` → `regimenes.evaluacion`, `INDEX.md` → `docs/historia/capa1/memoria/INDEX.md`; `features.py`/`data_loader.py` de v1 quedan en el tag `capa1-final`.
<!-- END ubicacion_v2 -->

> Mixtura gaussiana estática (`GaussianMixture`, `covariance_type='full'`) sobre
> las 15 features causales. Sin cadena de Markov: cada día se asigna de forma
> independiente al componente más probable. Es el **baseline contra el que D4
> (HMM) medirá cuánto aporta la dinámica temporal** (mismas features, misma
> evaluación walk-forward).

## Implementado

- **Clase** `src/regimenes/detectores/f2_clustering/clustering_gmm.py::ClusteringGMM(RegimeDetector)`.
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
- **Notebook** `capa1_exploracion/notebooks/03_clustering_gmm.ipynb` (v1, tag `capa1-final`) (construido y ejecutado con
  `_build_03.py` (retirado en el commit `6016f42`), 0 errores, 3 figuras inline): selección de k por BIC,
  sanidad del orden canónico, walk-forward k=2 y k=3, tabla de métricas, histograma
  de duraciones (flickering), S&P 500 coloreado por régimen, timeline + P(crisis), y
  bloque de verificación con asserts sobre 2008/2011 (NaN) y COVID/Inflación.
- **Artefactos**: `docs/historia/capa1/resultados/metrics_03_clustering_gmm.csv` (1 fila, detector k=3
  elegido por BIC), `d03_gmm_flickering.png` (figura v1, no versionada),
  `d03_gmm_sp500_regimes.png` (figura v1, no versionada), `d03_gmm_timeline.png` (figura v1, no versionada).

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

Lead/lag (días vs suelo de drawdown): COVID **−20 d**, Inflación **−219 d** (en
COVID la señal P(crisis)≥0.5 sostenida llega poco antes del suelo; en Inflación
cruza muy por delante; ambos suelos sí caen en OOS).

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

<!-- BEGIN resultados_v2 (generado desde git HEAD; no editar a mano) -->
## Resultados v2 (ADR-003)

> Fuente: `results/benchmark/ranking_v2.csv` y `results/benchmark/metrics_master_v2.csv` del benchmark completo (24/24; `results/benchmark/manifest.json`, generado el 2026-09-29T12:41 UTC), con las cifras de ADR-003 §4. Walk-forward causal, evaluación OOS por pista. Ranking de **detección**: evento detectado = ≥ 3 sesiones OOS consecutivas en crisis dentro de [pico, suelo]; `score_deteccion` = F1 entre precisión diaria (1 − FAR) y recall por evento, con niveles previos (elegible › precisión ≤ azar › parpadeo › degenerado). Ver `docs/decisions/ADR-003-causalidad-calendario-estado-ranking.md`.

**Notebook de familia:** [`notebooks/06_familia_F2_clustering.ipynb`](../../notebooks/06_familia_F2_clustering.ipynb) (F2 — Clustering) · **Teoría:** [`docs/teoria/F2_clustering.md`](../teoria/F2_clustering.md).

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

**Configuración v2 y eventos por pista**

- **Pista A** · OOS 1970-05-12 → 2026-05-29 (n = 14133) · train inicial 2016 sesiones, refit cada 21 (expanding) · 9 feature(s): `SP500_ret_z|SP500_vol_z|SP500_momentum|SP500_drawdown|FF_MKT_z|DGS10_change_z|credit_BaaAaa_mensual_z|term_spread_hist_z|INDPRO_yoy_z`.
  - Detectados (15): `bear_1969_70`, `oil_stagflation_1973`, `volcker_1980_82`, `black_monday_1987`, `gulf_war_sl_1990`, `ltcm_russia_1998`, `dotcom_2000_02`, `gfc_2007_09`, `flash_crash_euro1_2010`, `us_downgrade_euro2_2011`, `china_oil_2015_16`, `fed_tightening_2018q4`, `covid_2020`, `inflation_bear_2022`, `tariff_selloff_2025`.
  - No detectados (2): `volmageddon_2018q1`, `svb_banking_2023`.
- **Pista B** · OOS 2010-04-12 → 2026-05-29 (n = 4059) · train inicial 756 sesiones, refit cada 21 (expanding) · 14 feature(s): `SP500_ret_z|SP500_vol_z|SP500_momentum|SP500_drawdown|FF_MKT_z|DGS10_change_z|credit_BaaAaa_mensual_z|term_spread_hist_z|INDPRO_yoy_z|VIX_level_z|MOVE_level_z|credit_BAA10Y_z|DFII10_change_z|corr_spx_bond`.
  - Detectados (2): `china_oil_2015_16`, `covid_2020`.
  - No detectados (7): `flash_crash_euro1_2010`, `us_downgrade_euro2_2011`, `volmageddon_2018q1`, `fed_tightening_2018q4`, `inflation_bear_2022`, `svb_banking_2023`, `tariff_selloff_2025`.

Lectura: las cifras de la Capa 1 de las secciones anteriores (ventana 2007-2026, 4 crisis, métricas de cobertura sin criterio de detección) **no son comparables** con esta tabla (pistas A y B de `configs/benchmark_spec.yaml`, 17 crisis evaluables OOS en A / 9 crisis evaluables OOS en B, ranking ADR-003). El análisis gráfico por crisis y la comparación intra-familia están en el notebook `notebooks/06_familia_F2_clustering.ipynb`.
<!-- END resultados_v2 -->
