# Detección de regímenes de mercado — TFM (MIAX)

Banco de pruebas para **detección de regímenes de mercado**. El objetivo **no es un detector
concreto**, sino el **marco de evaluación causal y comparable** que juzga a muchos detectores, y
la **base de datos sólida** sobre la que hacerlo.

> **La historia de este repo, en una línea:**
> *hicimos 12 detectores → nos dimos cuenta de que no eran comparables (cada uno usaba datos y
> periodos distintos) → re-basamos los datos desde cero para poder comparar de verdad.*

---

## 🧭 Mapa del repo

```
/
├── README.md                     ← este archivo
├── requirements.txt · .env.example · .gitignore
│
├── src/                          ← código v2 (activo)
│   ├── evaluation.py             ← EL JUEZ (walk-forward + métricas) — reutilizado de Capa 1
│   ├── detector_base.py          ← interfaz RegimeDetector — reutilizada
│   ├── features.py · viz.py      ← primitivas causales · estilo de figuras
│   └── ingest/                   ← descarga dirigida por catálogo (fred/yfinance/ofr/github/académico)
│
├── data/
│   ├── catalog.yaml              ← universo de datos declarado (las 2 pistas + crisis_catalog)
│   ├── benchmark_spec.yaml       ← BANCO CONGELADO por pista (variable controlada de la Fase D)
│   ├── raw/                      ← 166 series .parquet (gitignored) + provenance + coverage
│   └── processed/
│
├── notebooks/                    ← planos, ordenados, ejecutados y autónomos
│   ├── 00_descarga.ipynb         ← panorámica de datos: tabla completa + visualizaciones + descarga
│   ├── 01_eda.ipynb              ← EDA maestro (12 secciones, recomputa los hallazgos)
│   └── 02_diseno_preprocesado.ipynb ← decisiones del preprocesado (03_preprocesado pendiente)
│
├── docs/                         ← TODO el conocimiento del proyecto (empieza por docs/README.md)
│   ├── README.md                 ← índice: decisiones, datos, EDA, teoría
│   ├── GLOSARIO.md               ← conceptos canónicos (pistas A/B, roles, causalidad, anti-fuga)
│   ├── decisions/                ← ADRs (por qué se tomó cada decisión)
│   ├── EDA_v2.md                 ← informe EDA completo   · figs_eda/  figuras del informe
│   ├── SOTA_datos.md             ← estado del arte de datos (fuentes, historia, procedencia)
│   ├── context/ · references.bib ← propuesta TFM + tarea previa · bibliografía
│
└── capa1_exploracion/            ← ❄️ Capa 1 CONGELADA: los 12 detectores + su marco (archivo)
```

**Si acabas de llegar, lee:** `docs/README.md` → **`docs/GLOSARIO.md`** (conceptos: pistas A/B, roles,
anti-fuga) → `docs/decisions/ADR-001-rebase-datos.md` → `capa1_exploracion/README.md`.

## Principio rector (no negociable)

- **Features causales**: toda estandarización es z-score *expanding/rolling* (en `t`, solo
  estadísticos de datos `≤ t`). Nunca de muestra completa — ese *look-ahead* fue el error de la
  tarea previa.
- **Evaluación walk-forward / out-of-sample**: nada se juzga in-sample.
- **Misma interfaz, mismas métricas**: cada detector implementa `RegimeDetector` (`evaluation.py`).

## Las dos pistas de datos (decisión ADR-001)

| Pista | Qué | Ventana | Crisis | Ataca |
|---|---|---|:---:|---|
| **A — Espina profunda** | S&P 500 + vol + factores + crédito/macro profundos | 1927 → 2026 | **22** | el n≈4 (potencia) |
| **B — Panel rico** | crédito, curva, vol, breakevens, sectores, FX | 2003 → 2026 | 10 | el punto ciego de 2013 |

Cada pista congela su ventana + features en `data/benchmark_spec.yaml` → *un leaderboard justo
1:1 dentro de cada pista*.

### Configuración local
```bash
cp .env.example .env       # y pega tu FRED_API_KEY (gratis: fred.stlouisfed.org)
pip install -r requirements.txt
python -m src.ingest.download    # descarga el catálogo -> data/raw/  (o abre notebooks/00_descarga.ipynb)
```

## Estado del proyecto

| Fase | Qué | Estado |
|---|---|---|
| 1 | Reorganización + narrativa | ✅ |
| 2 | Datos: estado del arte + descarga (166/174 series) | ✅ |
| 3 | EDA profundo + benchmark congelado | ✅ |
| — | Limpieza + documentación | ✅ |
| D | Re-evaluar los 12 sobre `benchmark_spec.yaml` + diseñar el detector | 🔜 (siguiente) |

*(La Fase 4 "preprocesado" se pospone: se hará junto con la Fase D, ya con el benchmark congelado.)*
