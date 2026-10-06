# docs/ — Conocimiento del proyecto (índice)

Todo lo **relevante y durable** del TFM vive aquí o se enlaza desde aquí: la visión del **TFM
completo** (sistema multi-agente), las **decisiones** tomadas, los **datos** y lo encontrado en ellos,
la **teoría** de cada familia de detectores, las **fichas** por detector y la **historia** de la Capa 1.
Salvo `tfm/`, el resto de `docs/` documenta la **Fase 1** (el régimen de mercado). Si algo no está en este
índice, es código (`src/regimenes/`), configuración (`configs/`), datos (`data/`) o resultados
(`results/`).

```
docs/
├── README.md               este índice
├── GLOSARIO.md             conceptos canónicos + términos del TFM + rutas y módulos + pipeline 00–20 (Fase 1)
├── references.bib          bibliografía ÚNICA del TFM
├── tfm/                    el TFM completo: arquitectura, cliente y carteras, datos alternativos, líneas abiertas
├── decisions/              ADR-001 … ADR-005
├── datos/                  SOTA_datos.md · EDA_v2.md · figs_eda/
├── teoria/                 README · 00_estado_del_arte.md · F1…F8_*.md (+ .bib por familia)
├── detectores/             README · D01…D13_*.md (fichas por detector)
├── historia/capa1/         la Capa 1: decisiones, hallazgos, memoria, informe, métricas y datos v1
└── context/                propuesta original, presentación al tutor y resumen de la tarea previa
```

## 0. Qué es el TFM (empieza aquí)
- **[`tfm/README.md`](tfm/README.md)** — el TFM completo: *Multi-Agent RAG System for Regime-Aware
  Macro-Equity Intelligence*, un equipo de agentes (Macro, Equity, Risk, Portfolio y CIO) que, sabiendo en
  qué régimen está el mercado, construye carteras y redacta un *investment memo* con citas. Desde ahí:
  [`ARQUITECTURA`](tfm/ARQUITECTURA.md) · [`CLIENTE_Y_CARTERAS`](tfm/CLIENTE_Y_CARTERAS.md) ·
  [`DATOS_ALTERNATIVOS`](tfm/DATOS_ALTERNATIVOS.md) · [`LINEAS_ABIERTAS`](tfm/LINEAS_ABIERTAS.md).
- **[`decisions/ADR-005-reencuadre-tfm-multiagente.md`](decisions/ADR-005-reencuadre-tfm-multiagente.md)** —
  fija el alcance: cinco fases; el régimen (este repositorio, paquete `regimenes`) es la **Fase 1** y su
  salida es la tool `get_regimen(fecha)`.
- **[`context/TFM_Reunion_Tutor.pdf`](context/TFM_Reunion_Tutor.pdf)** — presentación al tutor (octubre de
  2026) · **[`context/TFM_Proposal_v2.pdf`](context/TFM_Proposal_v2.pdf)** — propuesta original,
  **parcialmente superada** por la ADR-005 (varios de sus elementos siguen por analizar).

El resto de este índice (conceptos, decisiones 001–004, datos, teoría, detectores, historia)
documenta la **Fase 1**.

## 0b. Conceptos
- **[`GLOSARIO.md`](GLOSARIO.md)** — definiciones canónicas: las dos **pistas** (A/B/`ambas`/`validacion`),
  los cinco **roles** (`spine/core/enricher/fallback/validation`), la **causalidad** (computacional y de
  calendario), la **regla anti-fuga**, la lectura correcta de las métricas, **dónde está cada cosa**
  (rutas y módulos de `regimenes`) y el **pipeline 00–20**.

## 1. Decisiones (por qué se hizo cada cosa)
- **[`historia/capa1/README.md`](historia/capa1/README.md)** — punto de partida: qué se decidió al
  construir los 12 detectores v1 de la Capa 1, qué se aprendió y qué falló (§2–§6).
- **[`decisions/ADR-001-rebase-datos.md`](decisions/ADR-001-rebase-datos.md)** — la decisión madre:
  por qué se congelaron los 12 detectores y se re-basó la capa de datos (incomparabilidad 1:1, ~4
  crisis, FRED capado). Su punto *"Capa 1 se mantiene intacta"* lo revierte ADR-004.
- **[`decisions/ADR-002-ajuste-ventanas.md`](decisions/ADR-002-ajuste-ventanas.md)** — reajuste del banco
  congelado: fin de ventana gobernado por la serie diaria más fresca, pool de 106 features, Pista A
  desde 1962 (41 features, 18 crisis) y Pista B desde 2007 (106 features, 10 crisis) con el mismo fin.
- **[`decisions/ADR-003-causalidad-calendario-estado-ranking.md`](decisions/ADR-003-causalidad-calendario-estado-ranking.md)**
  — lags de publicación, propagación de estado en el walk-forward y **ranking por detección**
  (F1 = media armónica, no producto, del recall por evento y la precisión diaria) que sustituye al
  `rank_medio`.
- **[`decisions/ADR-004-unificacion.md`](decisions/ADR-004-unificacion.md)** — un solo repositorio y
  un solo paquete (`regimenes`): la Capa 1 integrada en el paquete, notebooks por familia 00–20 y esta
  estructura de `docs/`. Su afirmación «todo el TFM vive en `regimenes`» la corrige ADR-005.
- **[`decisions/ADR-005-reencuadre-tfm-multiagente.md`](decisions/ADR-005-reencuadre-tfm-multiagente.md)** —
  el TFM es un sistema multi-agente con RAG y el régimen es su Fase 1; requisitos de la tutoría y
  renumeración de la decisión final del detector (notebook 19) a la futura ADR-006.

## 2. Datos (qué se recopiló y qué dicen)
- **[`datos/SOTA_datos.md`](datos/SOTA_datos.md)** — estado del arte de datos: qué series existen para
  detección de regímenes, cuáles se eligieron, su historia, su fuente gratis y el reparto por pista.
- **[`datos/EDA_v2.md`](datos/EDA_v2.md)** — informe EDA completo (figuras en
  [`datos/figs_eda/`](datos/figs_eda/)): colas gordas, clustering de vol, correlación acción-bono que
  cambia de signo, complejo de volatilidad, crédito/curva y **el punto ciego de 2013**, profundidad =
  potencia, ranking causal de features.
- **[`../configs/catalog.yaml`](../configs/catalog.yaml)** — universo declarado (174 series +
  `crisis_catalog` con 22 crisis 1929–2025).
- **[`../configs/benchmark_spec.yaml`](../configs/benchmark_spec.yaml)** — **el banco congelado** por
  pista (ventana + features + crisis + trampas + suelos). Variable controlada de la Fase D.
- Notebooks: [`00_descarga`](../notebooks/00_descarga.ipynb) (panorámica de las 166 series),
  [`01_eda`](../notebooks/01_eda.ipynb) (EDA maestro), [`02_diseno_preprocesado`](../notebooks/02_diseno_preprocesado.ipynb)
  (decisiones del preprocesado) y [`03_preprocesado`](../notebooks/03_preprocesado.ipynb) (paneles causales).

## 3. Teoría y detectores
- **[`teoria/README.md`](teoria/README.md)** y **[`detectores/README.md`](detectores/README.md)** — índices
  de familias y de detectores (código, notebook y puesto en el ranking de cada uno).
- **[`teoria/00_estado_del_arte.md`](teoria/00_estado_del_arte.md)** — tabla transversal de las 7
  familias, solapes (HMM↔MS, GMM↔HMM, RS-GARCH↔MS…) y la lista de 12 detectores.
- **Fichas por familia** (con su `.bib`): [F1 reglas](teoria/F1_reglas_umbrales.md) ·
  [F2 clustering](teoria/F2_clustering.md) · [F3 HMM](teoria/F3_hmm.md) ·
  [F4 Markov-Switching](teoria/F4_markov_switching.md) · [F5 GARCH](teoria/F5_volatilidad_garch.md) ·
  [F6 change-point](teoria/F6_change_point.md) · [F7 redes](teoria/F7_redes_neuronales.md) ·
  [F8 generadores sintéticos](teoria/F8_generadores_sinteticos.md) (fase S: los diez generadores de
  `regimenes.sinteticos`, el espacio de generación, los criterios de validación y niveles del veredicto y
  el laboratorio; notebooks [`15_sinteticos_generadores`](../notebooks/15_sinteticos_generadores.ipynb),
  [`16_sinteticos_validacion`](../notebooks/16_sinteticos_validacion.ipynb) y
  [`17_sinteticos_laboratorio`](../notebooks/17_sinteticos_laboratorio.ipynb)).
- **Fichas por detector** (implementado + descubierto): [D01](detectores/D01_rule_vix_threshold.md) ·
  [D02](detectores/D02_rule_composite_riskoff.md) · [D03](detectores/D03_clustering_gmm.md) ·
  [D04](detectores/D04_hmm_gaussian_2s.md) · [D05](detectores/D05_markov_switching_var.md) (Markov-Switching
  de media y varianza; pese al nombre heredado `markov_switching_var` / "MS-VAR", no es un VAR: ver
  [GLOSARIO](GLOSARIO.md#nombres-de-detectores-que-confunden)) ·
  [D06](detectores/D06_garch_t_vol.md) · [D07](detectores/D07_changepoint_online.md) ·
  [D08](detectores/D08_hmm_tstudent.md) · [D09](detectores/D09_jump_model.md) ·
  [D10](detectores/D10_turbulence_mahalanobis.md) · [D11](detectores/D11_msgarch_regime.md) ·
  [D12](detectores/D12_deep_ae_regime.md) · [D13](detectores/D13_hsmm_tstudent.md) (ablación HSMM, fuera del ranking).
- La teoría y los resultados v2 de cada familia se cuentan juntos en los notebooks `05`–`11`
  (`05_familia_F1_reglas` … `11_familia_F7_deep`); la comparación entre familias, en
  [`12_comparativa`](../notebooks/12_comparativa.ipynb).
- **[`references.bib`](references.bib)** — bibliografía única (fusión de la central, las 7 por familia, la de
  F8 y la del informe v1; deduplicada, con alias `ids` para las claves repetidas).

## 4. Historia
- **[`historia/capa1/README.md`](historia/capa1/README.md)** — qué fue la Capa 1: sus decisiones, 12
  detectores y 5 hallazgos, resumen de su EDA y su comparativa, qué falló, correspondencia de rutas y
  cómo recuperar el código y los notebooks v1 (tag `capa1-final`).
  - [`historia/capa1/memoria/`](historia/capa1/memoria/99_conclusions.md) — `01_data_and_eda` y
    `99_conclusions`.
  - [`historia/capa1/informe/informe_capa1.pdf`](historia/capa1/informe/informe_capa1.pdf) — informe
    LaTeX v1 (con su `.tex`, `references.bib` y la revisión del PDF).
  - [`historia/capa1/resultados/`](historia/capa1/resultados/metrics_master.csv) — métricas v1 por
    detector, master, archivo y ablación HSMM.
- **[`context/`](context/)** — tres ficheros: [`TFM_Proposal_v2.pdf`](context/TFM_Proposal_v2.pdf) (propuesta
  original, parcialmente superada por la ADR-005), [`TFM_Reunion_Tutor.pdf`](context/TFM_Reunion_Tutor.pdf)
  (presentación al tutor, octubre de 2026) y [`RESUMEN_DETECCION_REGIMENES.md`](context/RESUMEN_DETECCION_REGIMENES.md)
  (resumen de la tarea previa, HMM gaussiano).

---

## Orden de lectura

Es el único orden de lectura del repo (el `README.md` raíz remite aquí):

`tfm/README.md` (el TFM completo) → ADR-005 (alcance y fases) → GLOSARIO →
`historia/capa1/README.md` (Capa 1, de dónde venimos) → ADR-001 → SOTA_datos → EDA_v2 → ADR-002 →
benchmark_spec → ADR-003 → ADR-004 (estructura actual) → teoría por familia → notebooks 04 → 05–11 →
12 → 13–14 (fusión) → 15–17 (sintéticos y laboratorio) → 18–20 (aumento, decisión final y validación de
`get_regimen`; esqueletos).

*Visita rápida:* `tfm/README.md` → ADR-005 → GLOSARIO → `historia/capa1/README.md` → ADR-004, y vuelve al orden completo cuando necesites el porqué
de los datos (ADR-001/002) o del ranking (ADR-003).

El orden de **lectura** de los notebooks (familias antes que la comparativa) no es el de
**ejecución** tras un benchmark nuevo (04 → 12 → 05–11): ver [GLOSARIO](GLOSARIO.md#pipeline-0020).
