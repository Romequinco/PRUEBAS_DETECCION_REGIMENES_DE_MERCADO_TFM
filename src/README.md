# src/ — código v2 (activo)

Marco de evaluación **reutilizado idéntico** de la Capa 1 + capa de datos nueva.

| Módulo | Origen | Rol |
|---|---|---|
| `evaluation.py` | copiado de `capa1_exploracion/src/` | **EL JUEZ** — walk-forward + métricas. No se toca. |
| `detector_base.py` | copiado de `capa1_exploracion/src/` | interfaz `RegimeDetector`. No se toca. |
| `features.py` | copiado de `capa1_exploracion/src/` | primitivas causales (z-score expanding/rolling, vol, drawdown…). Base de las features v2. |
| `viz.py` | copiado de `capa1_exploracion/src/` | estilo de casa para figuras. |
| `ingest/` | **nuevo (FASE 2)** | descargadores por fuente (FRED, yfinance, Stooq, Kaggle, GitHub). |

> **Por qué duplicado y no importado**: Capa 1 es una foto congelada; `src/` v2 evoluciona.
> Mantener copias separadas evita que un cambio en v2 rompa la reproducibilidad de Capa 1.
> El *contrato* del juez (firmas de `evaluate`/`walk_forward` y de `RegimeDetector`) se mantiene
> idéntico en ambas copias — ver `../docs/decisions/ADR-001-rebase-datos.md`.
