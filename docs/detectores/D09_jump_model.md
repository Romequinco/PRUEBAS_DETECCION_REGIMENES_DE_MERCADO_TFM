# D9 — `jump_model` (FASE 3, Tanda 4 · exploratoria) · Familia F2↔F3 (clustering con persistencia)

<!-- BEGIN ubicacion_v2 -->
> **Ubicación tras ADR-004.** Código: `src/regimenes/detectores/f2_clustering/jump_model.py` (registro: `regimenes.detectores.registry`, id `D09`) · Notebook v2 de familia: [`notebooks/06_familia_F2_clustering.ipynb`](../../notebooks/06_familia_F2_clustering.ipynb) · Resultados v2: sección [Resultados v2 (ADR-003)](#resultados-v2-adr-003) al final. Esta ficha nació en la Capa 1 (v1, congelada en el tag `capa1-final`); el texto histórico se conserva. Cuando menciona el núcleo (`src/`, `detector_base.py`, `evaluation.py`, `features.py`, `data_loader.py`, `INDEX.md`) se refiere al de la Capa 1: hoy `detector_base.py` → `src/regimenes/detectores/base.py`, `evaluation.py` → `regimenes.evaluacion`, `INDEX.md` → `docs/historia/capa1/memoria/INDEX.md`; `features.py`/`data_loader.py` de v1 quedan en el tag `capa1-final`.
<!-- END ubicacion_v2 -->

> Statistical Jump Model (Nystrup et al.): clustering de estados con **penalización
> de salto** λ → histéresis "aprendida", persistencia, online. Rival honesto de D3
> (GMM sin persistencia) y de D12 (AE). Código: `src/regimenes/detectores/f2_clustering/jump_model.py` · Notebook:
> `capa1_exploracion/notebooks/09_jump_model.ipynb` (v1, tag `capa1-final`) · Métricas: `docs/historia/capa1/resultados/metrics_09_jump_model.csv`.

## Implementado

**Modelo.** Librería `jumpmodels` v0.1.1 (`JumpModel`, SJM discreto), `n_states=2`,
**jump_penalty λ=50**. El SJM minimiza distancia intra-cluster + λ·(nº de saltos de
estado) → penaliza cambiar de régimen, induciendo persistencia sin la matriz de
transición de un HMM. `StandardScaler` ajustado **solo con el train** dentro de `fit`
(causal, re-fit por fold) porque 3 de las 15 features (corr, drawdown, momentum) tienen
escala distinta a los z. Mismas 15 features causales y ventana 2007+ que **D3** (es su
rival directo). Bibliografía: `hmm_nystrup2020, hmm_nystrup2017, clust_munnix2012`.

**Causalidad.** `jumpmodels` expone `predict_online` causal (la etiqueta de la fila i
usa solo filas < i). Verificado en el notebook: **0/120 etiquetas del bloque cambian
al añadir futuro**. market_returns a walk_forward Y evaluate.

**Orientación (Arreglo 4).** Verificada en walk-forward: crisis (estado 1) = vol 0.0218
vs calma 0.0089, retorno negativo → **no invertido**, sin warning de fallback.

## Descubierto

**ventana_eval: 2015-09 → 2026-06 (n=2649).** 15 features de 2007 → 2008/2011 en el
train (NaN OOS), como D3.

**La hipótesis anti-flickering se cumple ROTUNDAMENTE.** Comparado con su rival D3
(GMM clustering sin persistencia):

| | switching_rate | duración media | cov COVID | cov Inflation |
|---|---:|---:|---:|---:|
| D3 clustering_gmm_k3 | 0.126 | 7.9 d | 0.96 | 0.87 |
| **D9 jump_model (λ=50)** | **0.005** | **176.6 d** | 0.72 | 0.17 |

La penalización de salto reduce el flickering **~24×** (0.126→0.005) y multiplica la
duración de los episodios ×22 — exactamente lo que prometía Nystrup: histéresis
aprendida sin dinámica de Markov explícita.

**Pero hay un coste claro de sensibilidad.** Con λ=50 (alto), D9 se vuelve tan
persistente que **pierde cobertura de las crisis más lentas/menos extremas**: COVID
72% (vs 96% de D3) e **Inflación 2022 solo 17%** (vs 87% de D3). El bear market lento
de 2022 no genera un salto suficientemente nítido como para vencer la penalización, así
que D9 se queda en "calma" gran parte de él. fa_2018 = 0% (no se dispara en la trampa),
false_alarm_rate 0.62, label_stability ≈0.98.

**Lectura honesta del trade-off.** D9 NO domina a D3: cambia muchísimo flickering por
cobertura. Es el extremo "ultra-persistente" del eje clustering. Su λ es un mando
directo sobre ese trade-off (λ bajo → se parece a D3; λ alto → ultra-persistente y
ciego a estrés suave). Para el TFM es valioso como demostración de que la persistencia
se puede imponer explícitamente, y como punto de comparación frente al GMM (sin
persistencia) y al AE (D12).

## Hipótesis del CHECKPOINT 2 para D9 — veredicto

> *"Histéresis aprendida: estados persistentes, online, menos flickering, drawdowns
> suaves; rival honesto del AE+clustering en muestra pequeña."*

**Se cumple en lo esencial, con matiz.** Persistencia/online/menos flickering: SÍ,
rotundo (switching 0.005). "Drawdowns suaves": a costa de **perder cobertura de crisis
lentas** (Inflación 17%) — la persistencia fuerte es un arma de doble filo. Como rival
del AE (D12): D9 es mucho más limpio (switching 0.005 vs 0.287 del AE) — el jump model
gana de calle al deep como forma de imponer persistencia con pocos datos.

## Fricción con el núcleo

Ninguna; no se tocó `src/`. El Arreglo 4 (vol-primario) orientó D9 sin parche
(candidato a inversión por separar en features; no la hubo). Observación menor (ya
conocida): `walk_forward` tiene `min_train=252*5` que domina sobre `train_size`
pequeños — irrelevante aquí (train=252×8).

<!-- BEGIN resultados_v2 (generado desde git HEAD; no editar a mano) -->
## Resultados v2 (ADR-003)

> Fuente: `results/benchmark/ranking_v2.csv` y `results/benchmark/metrics_master_v2.csv` de la re-ejecución completa del benchmark (24/24) posterior a la unificación ADR-004 (`results/benchmark/manifest.json`, generado el 2026-09-29T12:41 UTC), numéricamente idéntica a la re-ejecución de ADR-003 §4 (ver `docs/revisiones/informe_unificacion.md`). Walk-forward causal, evaluación OOS por pista. Ranking de **detección**: evento detectado = ≥ 3 sesiones OOS consecutivas en crisis dentro de [pico, suelo]; `score_deteccion` = F1 entre precisión diaria (1 − FAR) y recall por evento, con niveles previos (elegible › precisión ≤ azar › parpadeo › degenerado). Ver `docs/decisions/ADR-003-causalidad-calendario-estado-ranking.md`.

**Notebook de familia:** [`notebooks/06_familia_F2_clustering.ipynb`](../../notebooks/06_familia_F2_clustering.ipynb) (F2 — Clustering) · **Teoría:** [`docs/teoria/F2_clustering.md`](../teoria/F2_clustering.md).

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

**Configuración v2 y eventos por pista**

- **Pista A** · OOS 1970-05-12 → 2026-05-29 (n = 14133) · train inicial 2016 sesiones, refit cada 21 (expanding) · 9 feature(s): `SP500_ret_z|SP500_vol_z|SP500_momentum|SP500_drawdown|FF_MKT_z|DGS10_change_z|credit_BaaAaa_mensual_z|term_spread_hist_z|INDPRO_yoy_z`.
  - Detectados (8): `bear_1969_70`, `oil_stagflation_1973`, `volcker_1980_82`, `black_monday_1987`, `dotcom_2000_02`, `gfc_2007_09`, `covid_2020`, `svb_banking_2023`.
  - No detectados (9): `gulf_war_sl_1990`, `ltcm_russia_1998`, `flash_crash_euro1_2010`, `us_downgrade_euro2_2011`, `china_oil_2015_16`, `volmageddon_2018q1`, `fed_tightening_2018q4`, `inflation_bear_2022`, `tariff_selloff_2025`.
- **Pista B** · OOS 2010-04-12 → 2026-05-29 (n = 4059) · train inicial 756 sesiones, refit cada 21 (expanding) · 14 feature(s): `SP500_ret_z|SP500_vol_z|SP500_momentum|SP500_drawdown|FF_MKT_z|DGS10_change_z|credit_BaaAaa_mensual_z|term_spread_hist_z|INDPRO_yoy_z|VIX_level_z|MOVE_level_z|credit_BAA10Y_z|DFII10_change_z|corr_spx_bond`.
  - Detectados (0): ninguno.
  - No detectados (9): `flash_crash_euro1_2010`, `us_downgrade_euro2_2011`, `china_oil_2015_16`, `volmageddon_2018q1`, `fed_tightening_2018q4`, `covid_2020`, `inflation_bear_2022`, `svb_banking_2023`, `tariff_selloff_2025`.

Lectura: las cifras de la Capa 1 de las secciones anteriores (ventana 2007-2026, 4 crisis, métricas de cobertura sin criterio de detección) **no son comparables** con esta tabla (pistas A y B de `configs/benchmark_spec.yaml`, 17 crisis evaluables OOS en A / 9 crisis evaluables OOS en B, ranking ADR-003). El análisis gráfico por crisis y la comparación intra-familia están en el notebook `notebooks/06_familia_F2_clustering.ipynb`.
<!-- END resultados_v2 -->
