# Detección de regímenes de mercado — TFM (MIAX)

Banco de pruebas para **detección de regímenes de mercado**. El objetivo del proyecto
**no es un detector concreto**, sino el **marco de evaluación causal y comparable** que
juzga a muchos detectores, y la **base de datos sólida** sobre la que hacerlo.

> **La historia de este repo, en una línea:**
> *hicimos 12 detectores → nos dimos cuenta de que no eran comparables (cada uno usaba
> datos y periodos distintos) → ahora re-basamos los datos desde cero para poder comparar
> de verdad.*

---

## 🧭 Cómo está organizado (mapa)

```
/
├── README.md                     ← este archivo (la historia + el mapa)
├── docs/
│   ├── decisions/                ← ADRs: el hilo de decisiones del proyecto
│   │   └── ADR-001-rebase-datos.md   ← POR QUÉ estamos re-basando los datos (léelo primero)
│   ├── context/                  ← propuesta TFM + resumen de la tarea previa
│   └── references.bib            ← bibliografía central (compartida)
│
├── capa1_exploracion/            ← ❄️ CONGELADO: los 12 detectores + su marco
│   └── README.md                 ← qué se hizo aquí y qué se aprendió
│
├── src/                          ← 🔧 v2 (activo): marco reutilizado + capa de datos nueva
│   ├── evaluation.py             ← EL JUEZ (reutilizado idéntico de Capa 1)
│   ├── detector_base.py          ← interfaz RegimeDetector (reutilizada idéntica)
│   ├── features.py               ← primitivas causales (z-score expanding/rolling)
│   ├── viz.py                    ← estilo de casa para figuras
│   └── ingest/                   ← descargadores por fuente (FASE 2)
│
├── data/                         ← v2 (gitignored salvo metadatos)
│   ├── catalog.yaml              ← universo de datos declarado (las 2 pistas)
│   ├── raw/  ·  processed/
│
└── notebooks/                    ← v2, un subdirectorio por fase
    ├── ingest/                   ← descarga (FASE 2)
    ├── eda_v2/                   ← EDA + análisis (FASE 3)
    └── preprocesado/             ← reordenación + features v2 (FASE 4)
```

**Si acabas de llegar, lee en este orden:** `docs/decisions/ADR-001-rebase-datos.md` →
`capa1_exploracion/README.md` → este mapa.

---

## Principio rector (no negociable)

- **Features causales**: toda estandarización es z-score *expanding/rolling* (en `t` solo se
  usan estadísticos de datos `<= t`). Nunca media/desv de toda la muestra — ese *look-ahead*
  fue el error que se detectó en la tarea previa.
- **Evaluación walk-forward / out-of-sample**: nada se juzga in-sample.
- **Misma interfaz, mismas métricas**: cada detector implementa `RegimeDetector` y se puntúa
  con `evaluation.py`.

## Las dos pistas de datos (decisión ADR-001)

No se puede tener a la vez máxima historia y máxima riqueza de features. Se construyen **dos**:

| Pista | Qué | Ventana | Ataca |
|---|---|---|---|
| **A — Espina histórica profunda** | pocas features (S&P 500 + vol) | desde ~1950 | el **n≈4 crisis** (Pista A tiene 10+) |
| **B — Panel rico multi-activo** | crédito, curva, vol, FX, macro | desde ~1990/2007 | el **punto ciego de 2013** (taxonomía) |

**Benchmark fijo por pista**: cada una congela su ventana + pool de features → *un leaderboard
justo 1:1 dentro de cada pista*. Entre pistas, comparación cualitativa declarada.

## Fuentes de datos (todo gratis)

FRED (con API key en `.env`), yfinance, Stooq, datasets de Kaggle, repos de GitHub, históricos
académicos (Shiller, Ken French) e índices de estrés ya hechos (OFR FSI, NFCI, STLFSI) como
validación externa. Ver `data/catalog.yaml` para el universo declarado y su procedencia.

### Configuración local
```bash
cp .env.example .env      # y pega tu FRED_API_KEY (gratis: fred.stlouisfed.org)
pip install -r requirements.txt
```

## Estado del proyecto

| Fase | Qué | Estado |
|---|---|---|
| 1 | Reorganización + narrativa | ✅ |
| 2 | Datos: estado del arte + descarga máxima | ✅ (145/161 series; ver `data/SOTA_datos.md` y `data/catalog.yaml`) |
| 3 | EDA + análisis profundo | ✅ (7 slices, 62 figs, `docs/EDA_v2.md` + benchmark congelado `data/benchmark_spec.yaml`) |
| 4 | Preprocesado listo-para-ejecutar | ⏳ |
| 5 | Documentación final | ⏳ |
| D | Re-evaluar los 12 sobre el dataset congelado + diseñar el detector | 🔜 (siguiente macro-sesión) |

La memoria viva histórica de la Capa 1 está en `capa1_exploracion/memory/INDEX.md`.
El porqué del giro está en `docs/decisions/ADR-001-rebase-datos.md`.
