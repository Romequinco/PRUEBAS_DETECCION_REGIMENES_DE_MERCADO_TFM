# src/ — código v2 (activo)

Marco de evaluación compatible con la Capa 1 + capa de datos y benchmark v2.

| Módulo | Origen | Rol |
|---|---|---|
| `evaluation.py` | heredado de `capa1_exploracion/src/` | **EL JUEZ** — walk-forward + métricas; v2 añade configuración de eventos por pista. |
| `detector_base.py` | copiado de `capa1_exploracion/src/` | interfaz `RegimeDetector`. No se toca. |
| `features.py` | copiado de `capa1_exploracion/src/` | primitivas causales (z-score expanding/rolling, vol, drawdown…). Base de las features v2. |
| `viz.py` | copiado de `capa1_exploracion/src/` | estilo de casa para figuras. |
| `ingest/` | **nuevo (FASE 2)** | descargadores por fuente (FRED, yfinance, Stooq, Kaggle, GitHub). |
| `detectors/` | **nuevo (FASE D)** | registro/adaptador v2 de las 12 implementaciones congeladas. |
| `benchmark.py` | **nuevo (FASE D)** | carga paneles, gate, walk-forward reanudable, caché con huella y ranking descriptivo. |
| `fusion.py` | **nuevo (FASE E)** | máquina causal normal/vigilancia/confirmado; emparejamiento uno-a-uno, atribución a crisis reales, ablación, sensibilidad y prueba de invariancia al truncado. |

> **Por qué duplicado y no importado**: Capa 1 es una foto congelada; `src/` v2 evoluciona.
> Mantener copias separadas evita que un cambio en v2 rompa la reproducibilidad de Capa 1.
> El *contrato* del juez (firmas de `evaluate`/`walk_forward` y de `RegimeDetector`) se mantiene
> compatible; v2 solo añade la selección explícita de labels A/B antes de evaluar. Ver
> `../docs/decisions/ADR-001-rebase-datos.md`.
