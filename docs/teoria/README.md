# docs/teoria — Estado del arte por familia

Teoría de los detectores de regímenes del TFM. Las fichas F1–F7 y el documento maestro se
redactaron en la FASE 2 de la Capa 1 (v1) y se movieron aquí con `git mv` en la unificación
(ADR-004) desde `capa1_exploracion/memory/sota/` y `capa1_exploracion/memory/`; su contenido
no cambia salvo rutas y una nota de ubicación al principio. F8 es nueva (esqueleto).

Cada ficha F1–F7 sigue la misma estructura: definición y supuestos · variantes principales ·
fortalezas y debilidades · idoneidad para este proyecto (EDA, causalidad) · aplicaciones
documentadas a regímenes de mercado · coste de implementación y librería · referencias ·
candidatas adicionales. Los notebooks de familia (`notebooks/05`–`11`) toman de aquí su sección 1
("Teoría de la familia").

| Documento | Familia | Bibliografía | Detectores v2 | Notebook v2 |
|---|---|---|---|---|
| [`00_estado_del_arte.md`](00_estado_del_arte.md) | Síntesis transversal, solapes entre familias y lista D1–D12 (CHECKPOINT 2) | `docs/references.bib` | todos | — |
| [`F1_reglas_umbrales.md`](F1_reglas_umbrales.md) | F1 Reglas / umbrales (incluye turbulencia multivariante) | [`F1_reglas_umbrales.bib`](F1_reglas_umbrales.bib) | [D01](../detectores/D01_rule_vix_threshold.md), [D02](../detectores/D02_rule_composite_riskoff.md), [D10](../detectores/D10_turbulence_mahalanobis.md) | [`05_familia_F1_reglas`](../../notebooks/05_familia_F1_reglas.ipynb) |
| [`F2_clustering.md`](F2_clustering.md) | F2 Clustering estático (y *jump models*) | [`F2_clustering.bib`](F2_clustering.bib) | [D03](../detectores/D03_clustering_gmm.md), [D09](../detectores/D09_jump_model.md) | [`06_familia_F2_clustering`](../../notebooks/06_familia_F2_clustering.ipynb) |
| [`F3_hmm.md`](F3_hmm.md) | F3 HMM con emisiones latentes (y HSMM) | [`F3_hmm.bib`](F3_hmm.bib) | [D04](../detectores/D04_hmm_gaussian_2s.md), [D08](../detectores/D08_hmm_tstudent.md), [D13](../detectores/D13_hsmm_tstudent.md) (ablación) | [`07_familia_F3_hmm`](../../notebooks/07_familia_F3_hmm.ipynb) |
| [`F4_markov_switching.md`](F4_markov_switching.md) | F4 Markov-Switching econométrico | [`F4_markov_switching.bib`](F4_markov_switching.bib) | [D05](../detectores/D05_markov_switching_var.md) | [`08_familia_F4_markov_switching`](../../notebooks/08_familia_F4_markov_switching.ipynb) |
| [`F5_volatilidad_garch.md`](F5_volatilidad_garch.md) | F5 Volatilidad / GARCH / RS-GARCH | [`F5_volatilidad_garch.bib`](F5_volatilidad_garch.bib) | [D06](../detectores/D06_garch_t_vol.md), [D11](../detectores/D11_msgarch_regime.md) | [`09_familia_F5_garch`](../../notebooks/09_familia_F5_garch.ipynb) |
| [`F6_change_point.md`](F6_change_point.md) | F6 Change-point detection | [`F6_change_point.bib`](F6_change_point.bib) | [D07](../detectores/D07_changepoint_online.md) | [`10_familia_F6_changepoint`](../../notebooks/10_familia_F6_changepoint.ipynb) |
| [`F7_redes_neuronales.md`](F7_redes_neuronales.md) | F7 Redes neuronales / no supervisado moderno | [`F7_redes_neuronales.bib`](F7_redes_neuronales.bib) | [D12](../detectores/D12_deep_ae_regime.md) | [`11_familia_F7_deep`](../../notebooks/11_familia_F7_deep.ipynb) |
| [`F8_generadores_sinteticos.md`](F8_generadores_sinteticos.md) | F8 Generadores sintéticos (**esqueleto**) | pendiente | — | `15_sinteticos_generadores` … `18_sinteticos_aumento` |

## Bibliografía

- `docs/references.bib` es la bibliografía única del proyecto: contiene ya todas las claves de
  los siete `.bib` de familia (fusionadas en la FASE 2 de la Capa 1; comprobado en la
  unificación que no falta ninguna).
- Los `.bib` de familia se conservan como *sidecars* de trazabilidad (qué claves aportó cada
  ficha). Prefijos: `reglas_`, `clust_`, `hmm_`, `ms_`, `vol_`, `cp_`, `nn_`; las claves
  compartidas sin prefijo (`hamilton1989`, `guidolintimmermann2007`, `kritzman2012`, …)
  viven solo en `docs/references.bib`.

## Relación con otros documentos

- Fichas por detector (implementación, hallazgos v1, resultados v2):
  [`docs/detectores/`](../detectores/README.md).
- Memoria histórica de la Capa 1 (EDA v1, conclusiones, índice):
  `docs/historia/capa1/memoria/`.
- Decisiones de protocolo y ranking: `docs/decisions/ADR-001` … `ADR-003`.
