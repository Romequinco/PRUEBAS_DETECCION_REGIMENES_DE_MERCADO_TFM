# docs/ — Conocimiento del proyecto (índice)

Todo lo **relevante y durable** del TFM vive aquí o se enlaza desde aquí: las **decisiones**
tomadas, lo **encontrado** en los datos, y la **teoría**. Si algo no está en este índice, o es
código (`src/`), datos (`data/`), o material congelado de la Capa 1 (`capa1_exploracion/`).

## 1. Decisiones (por qué se hizo cada cosa)
- **[`decisions/ADR-001-rebase-datos.md`](decisions/ADR-001-rebase-datos.md)** — la decisión madre:
  por qué se congelaron los 12 detectores y se re-basó la capa de datos (con las cifras que lo
  motivaron: incomparabilidad 1:1, ~4 crisis, FRED capado).

## 2. Datos (qué se recopiló y por qué)
- **[`SOTA_datos.md`](SOTA_datos.md)** — estado del arte de datos: qué series existen para detección
  de regímenes, cuáles se eligieron, su historia, su fuente gratis, y el reparto por pista.
- **[`../data/catalog.yaml`](../data/catalog.yaml)** — universo declarado (fuente de verdad): 174
  series + `crisis_catalog` (22 crisis 1929–2025). `../data/raw/coverage_report.csv` = qué hay en disco.
- **[`../data/benchmark_spec.yaml`](../data/benchmark_spec.yaml)** — **el banco congelado** por pista
  (ventana + features + crisis_windows + false_positives + troughs). Variable controlada de la Fase D.

## 3. Hallazgos (lo que dicen los datos)
- **[`EDA_v2.md`](EDA_v2.md)** — informe EDA completo (7 temas, figuras en [`figs_eda/`](figs_eda/)):
  fat tails, clustering de vol, correlación acción-bono que cambia de signo, el complejo de
  volatilidad, crédito/curva y **el punto ciego 2013** (trampa de tipos, no crisis), profundidad =
  potencia, ranking causal de features.
- **[`../notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb)** — recomputo en vivo de los
  hallazgos-cabecera (ejecutable, autónomo).

## 4. Teoría y contexto
- **[`context/`](context/)** — propuesta original del TFM + resumen de la tarea previa (HMM gaussiano).
- **[`references.bib`](references.bib)** — bibliografía central del proyecto.
- **Estado del arte de detectores** (teoría de las 7 familias F1–F7) y **conclusiones de la Capa 1**:
  en el archivo congelado — `../capa1_exploracion/memory/00_state_of_the_art.md`,
  `../capa1_exploracion/memory/99_conclusions.md` e `INDEX.md`.

## 5. La Capa 1 congelada
- **[`../capa1_exploracion/README.md`](../capa1_exploracion/README.md)** — qué se hizo con los 12
  detectores y qué se aprendió (los hallazgos metodológicos que sobreviven al re-base).

---

*Orden de lectura sugerido:* ADR-001 → SOTA_datos → EDA_v2 → benchmark_spec → capa1_exploracion.
