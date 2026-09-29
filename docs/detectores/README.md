# docs/detectores — Fichas por detector

Una ficha por detector del banco de pruebas. D01–D12 nacieron en la FASE 3 de la Capa 1 (v1)
en `capa1_exploracion/memory/detectors/NN_<nombre>.md` y se movieron aquí con `git mv` en la
unificación (ADR-004) como `DNN_<nombre>.md`; D13 (ablación HSMM) es nueva. Cada ficha
conserva su texto histórico (Implementado · Descubierto · hipótesis del CHECKPOINT 2 ·
fricción con el núcleo), lleva al principio una nota de ubicación v2 y al final la sección
**Resultados v2 (ADR-003)** con las métricas por pista.

Puesto y `score_deteccion` por pista: `results/benchmark/ranking_v2.csv` en el commit `1f95a9b`
(ranking de detección ADR-003; entre paréntesis el nivel si no es `elegible`).

| Ficha | Detector | Familia | Código (`src/regimenes/detectores/`) | Notebook v2 | Pista A | Pista B |
|---|---|---|---|---|---:|---:|
| [D01](D01_rule_vix_threshold.md) | `rule_vix_threshold` | F1 Reglas | `f1_reglas/rule_vix_threshold.py` | [`05_familia_F1_reglas`](../../notebooks/05_familia_F1_reglas.ipynb) | 2/12 · 0.587 | 3/12 · 0.505 |
| [D02](D02_rule_composite_riskoff.md) | `rule_composite_riskoff` | F1 Reglas | `f1_reglas/rule_composite_riskoff.py` | [`05_familia_F1_reglas`](../../notebooks/05_familia_F1_reglas.ipynb) | 4/12 · 0.559 | 4/12 · 0.501 |
| [D03](D03_clustering_gmm.md) | `clustering_gmm` | F2 Clustering | `f2_clustering/clustering_gmm.py` | [`06_familia_F2_clustering`](../../notebooks/06_familia_F2_clustering.ipynb) | 3/12 · 0.565 | 10/12 · 0.241 |
| [D04](D04_hmm_gaussian_2s.md) | `hmm_gaussian_2s` | F3 HMM | `f3_hmm/hmm_gaussian_2s.py` | [`07_familia_F3_hmm`](../../notebooks/07_familia_F3_hmm.ipynb) | 10/12 · 0.455 | 5/12 · 0.497 |
| [D05](D05_markov_switching_var.md) | `markov_switching_var` | F4 Markov-Switching | `f4_switching/markov_switching_var.py` | [`08_familia_F4_markov_switching`](../../notebooks/08_familia_F4_markov_switching.ipynb) | 5/12 · 0.546 | 1/12 · 0.647 |
| [D06](D06_garch_t_vol.md) | `garch_t_vol` | F5 GARCH | `f5_garch/garch_t_vol.py` | [`09_familia_F5_garch`](../../notebooks/09_familia_F5_garch.ipynb) | 7/12 · 0.540 | 2/12 · 0.597 |
| [D07](D07_changepoint_online.md) | `changepoint_online` | F6 Change-point | `f6_changepoint/changepoint_online.py` | [`10_familia_F6_changepoint`](../../notebooks/10_familia_F6_changepoint.ipynb) | 8/12 · 0.469 | 7/12 · 0.426 |
| [D08](D08_hmm_tstudent.md) | `hmm_tstudent` | F3 HMM | `f3_hmm/hmm_tstudent.py` | [`07_familia_F3_hmm`](../../notebooks/07_familia_F3_hmm.ipynb) | 9/12 · 0.467 | 9/12 · 0.274 |
| [D09](D09_jump_model.md) | `jump_model` | F2 Clustering | `f2_clustering/jump_model.py` | [`06_familia_F2_clustering`](../../notebooks/06_familia_F2_clustering.ipynb) | 12/12 · 0.362 | 11/12 · 0.000 (precision_no_supera_azar) |
| [D10](D10_turbulence_mahalanobis.md) | `turbulence_mahalanobis` | F1 Reglas (multivariante) | `f1_reglas/turbulence_mahalanobis.py` | [`05_familia_F1_reglas`](../../notebooks/05_familia_F1_reglas.ipynb) | 1/12 · 0.614 | 6/12 · 0.457 |
| [D11](D11_msgarch_regime.md) | `msgarch_regime` | F5 GARCH | `f5_garch/msgarch_regime.py` | [`09_familia_F5_garch`](../../notebooks/09_familia_F5_garch.ipynb) | 11/12 · 0.443 | 8/12 · 0.394 |
| [D12](D12_deep_ae_regime.md) | `deep_ae_regime` | F7 Deep | `f7_deep/deep_ae_regime.py` | [`11_familia_F7_deep`](../../notebooks/11_familia_F7_deep.ipynb) | 6/12 · 0.545 | 12/12 · 0.274 (parpadeo) |
| [D13](D13_hsmm_tstudent.md) | `hsmm_tstudent` | F3 HMM (ablación) | `f3_hmm/hsmm_tstudent.py` | [`07_familia_F3_hmm`](../../notebooks/07_familia_F3_hmm.ipynb) | — | — |

D13 no forma parte del benchmark v2 (sin `DetectorSpec` en el registro): sus cifras son las
de la ablación A1 de la Capa 1 frente a D08, recogidas en su ficha.

Regenerar la sección "Resultados v2 (ADR-003)" de las fichas tras un benchmark nuevo: las
tablas están entre los marcadores `<!-- BEGIN resultados_v2 -->` / `<!-- END resultados_v2 -->`
y se derivan solo de `ranking_v2.csv` y `metrics_master_v2.csv`; no editarlas a mano.

Teoría por familia: [`docs/teoria/`](../teoria/README.md).
