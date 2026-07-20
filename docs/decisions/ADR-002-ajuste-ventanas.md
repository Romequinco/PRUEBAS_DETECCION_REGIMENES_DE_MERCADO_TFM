# ADR-002 — Ajuste de ventanas y ampliación del pool de features (post ADR-001)

- **Estado:** Aceptada · 2026-07-20
- **Rama:** trabajo directo sobre `main`.
- **Ámbito:** afecta a `data/benchmark_spec.yaml` (ventanas + `series_features` + `crisis_windows.pista_A`),
  `notebooks/02_diseno_preprocesado.ipynb` (tabla `FEAT`), `notebooks/00_descarga.ipynb` (cierre) y la
  documentación conceptual (`docs/GLOSARIO.md`, `README.md`). **No** toca `ADR-001` (se preserva íntegro;
  esta ADR lo ajusta, no lo sustituye), ni el marco de evaluación (`src/evaluation.py`,
  `src/detector_base.py`), ni la regla de causalidad.

---

## 1. Contexto

Tras `ADR-001` (re-base de datos, dos pistas A/B) y la Fase 3 (EDA + `benchmark_spec.yaml` congelado),
una revisión conjunta con el usuario detectó dos problemas en el banco congelado v1:

1. **El pool de features usadas era mucho menor que el pool de datos sanos disponibles.** De las 166
   series descargadas, solo 35 (21%) alimentaban alguna feature (8 en Pista A, ~24 en Pista B). El
   resto —sanas, sin NaN, con historia real— quedaban sin usar no por estar "enfermas" sino por una
   selección editorial estrecha hecha en la primera vuelta del diseño (`02_diseno_preprocesado.ipynb` v1).
2. **El fin de ventana se calculaba de forma inconsistente con una regla ya vigente.** Una comprobación
   ad hoc dejó que una serie **mensual** (`GW_PREDICTORS_MONTHLY`, dataset académico con lag de
   publicación real) gobernara el fin de la ventana (2025-12-01), cuando el propio notebook 02 ya había
   fijado la regla de que las series mensuales se tratan con `lag de publicación + ffill` y **nunca**
   truncan el fin (igual que se hace con `INDPRO`). El fin correcto lo debe gobernar la serie **diaria**
   más fresca del panel.

## 2. Análisis: la escalera de cobertura

Se recalculó, directamente sobre `data/raw/coverage_report.csv` (166 series, sin imputar), cuántas
series están "vivas" (con dato) para cada fecha de corte candidata, excluyendo:
- `rol=validation` (20 series: nunca pueden ser feature, regla anti-fuga, sin cambios).
- `rol=fallback` (21 series: sustitutos redundantes, dedup, sin cambios).
- 6 series **descontinuadas** que forzarían el fin de ventana hacia atrás si se incluyen: `VXO`
  (muere 2021-09-23), `DTWEXM`/`TWEXMMTH` (mueren 2019), `EURODOLLAR_TBILL_SPREAD`/`EURODOLLAR_3M`
  (mueren 2016-10-07), `NBER_NY_COMMERCIAL_PAPER` (muere 1971). Quedan disponibles como añadido
  opcional documentado, no en el banco por defecto.

Esto deja un **pool unificado de 125 series candidatas** (119 vivas hoy), del que cada punto de corte
(fecha de inicio) extrae un subconjunto **estrictamente anidado**: toda serie viva en una fecha temprana
también está viva en cualquier fecha posterior. Se construyó una escalera de 11 puntos de corte, desde
1927 (11 features, 22/22 crisis) hasta 2007 (106 features, 10/22 crisis), documentada en detalle en la
conversación de diseño (no archivada aparte; los números clave quedan fijados en `benchmark_spec.yaml`
§`meta.ajuste_v2` y en las `justificacion` de cada pista).

## 3. Decisión

### 3.1 Pista A: escalón elegido = **1962-01-02** (antes 1927-12-30)

Se mueve el inicio de A al momento en que arranca la **curva de tipos completa** (`DGS5`, `DGS10`,
`T10YFF`) — un bloque entero nuevo que en v1 no existía en absoluto en Pista A (solo había un proxy
mensual histórico vía `GS10`/`TB3MS`, que se mantiene como features adicionales para robustez pre-1962).

- **Features**: 8 → **41** series.
- **Crisis vistas**: 22/22 → **18/22**.
- **Coste explícito y aceptado**: se pierden las 4 crisis con pico anterior a 1962-01-02:
  `great_crash_depresion_1929` (−86,2%, **la más severa del catálogo**), `recesion_1937_38` (−54,5%),
  `recesion_1957_58` (−21,5%) y `kennedy_slide_1962` (pico 1961-12-12, 3 semanas antes del corte).
  Se decide conscientemente: 18 crisis siguen siendo una base OOS muy superior a las ~4 de la tarea
  previa que motivó `ADR-001`, y el bloque de curva de tipos es más valioso para el objetivo de
  potencia estadística *moderna* que 4 eventos pre-1962 sin apenas features que los caractericen
  (en 1929/1937 solo hay 14-15/76 series vivas en todo el catálogo).

### 3.2 Pista B: escalón elegido = **2007-04-11** (antes 2003-01-02)

Se mueve el inicio de B al momento en que arranca `HYG_CREDIT` (crédito high-yield).

- **Features**: ~24 → **106** series (de las cuales 41 son la espina compartida con Pista A).
- **Crisis vistas**: 10/22 → **10/22** — **sin coste**: los escalones 2002-07-30, 2003-01-02 y
  2007-04-11 ven exactamente las mismas 10 crisis (la GFC, pico 2007-10-09, sigue dentro). Mover el
  inicio hasta 2007-04-11 es una mejora estrictamente gratis: se ganan crédito HY, vol-of-vol (`VVIX`)
  y breakevens ya consolidados sin sacrificar ningún evento.

### 3.3 `A ⊆ B` por construcción

Al usar el mismo pool unificado y cortar por fecha, toda serie viva en el corte de A (1962) está
automáticamente viva en el corte de B (2007) — **no hace falta mantener dos listas curadas a mano que
puedan divergir**. Esto resuelve una petición explícita del usuario: "todas las features que sirvan en
A deben estar en B". Se verifica programáticamente: `set(pista_A.series_features).issubset(set(pista_B.series_features)) == True`.

### 3.4 Fin de ventana: **igual para A y B, a propósito** (2026-05-29)

Corregido el error de §1.2: el fin lo gobierna la serie diaria más fresca, no una mensual. Con eso:
- Si Pista B **excluyera** los 4 `FF_*` diarios (Fama-French, ya cubiertos por A), su fin natural sería
  2026-07-10 (vía `MOVE`) — 6 semanas más tarde que A.
- **Se decide explícitamente incluir los 4 `FF_*` en B también**, aceptando perder esas 6 semanas, para
  que **A y B terminen exactamente el mismo día**. Prioridad: comparabilidad temporal directa entre
  pistas por encima de exprimir 6 semanas adicionales de datos recientes en B.
- Verificado: `pista_A.ventana_fin == pista_B.ventana_fin == '2026-05-29'`, gobernado en ambas por
  `FF_FACTORS_3_DAILY`.

### 3.5 Ampliación del pool de features (35 → 106 series únicas)

La tabla `FEAT` de `02_diseno_preprocesado.ipynb` se rediseñó por bloques temáticos (equity/breadth,
factores FF, crédito, curva histórica y diaria, velocidad de tipos/breakevens, complejo de
volatilidad, sectores, FX/commodities, macro real, valoración) cubriendo las 106 series de B de forma
explícita: cada una alimenta o bien una feature individual, o bien una feature agregada compartida
(p. ej. `equity_breadth_dispersion_z` sobre 11 índices de amplitud, para no generar 11 z-scores
casi-duplicados del mismo riesgo de mercado), o bien queda marcada como "solo-raw" de apoyo explícito
(9 plazos de curva que alimentarán un PCA de curva en `03_preprocesado`, no z-scores individuales).
Ninguna de las 106 queda descartada en silencio.

## 4. Qué NO cambia (deliberadamente)

- **`ADR-001` se preserva íntegro** — esta ADR lo ajusta (ventanas, pool), no lo contradice ni lo borra.
- **`data/catalog.yaml`** — el campo `pista` por serie (clasificación manual original, A≈1927+/B≈2003+)
  **no se reescribe**. Sigue siendo descriptivo del universo declarado; `benchmark_spec.yaml` es la
  única fuente de verdad operativa. Documentado explícitamente en `docs/GLOSARIO.md` para evitar
  confusión entre los dos ejes.
- **La regla anti-fuga** (`validation` nunca feature) y **la causalidad** (z-score expanding/rolling,
  `assert_causal`) — sin cambios.
- **`src/evaluation.py`, `src/detector_base.py`** — sin cambios.

## 5. Consecuencias

- **Ganamos**: comparabilidad temporal directa A↔B (mismo fin), `A ⊆ B` verificable sin mantenimiento
  manual, y un pool de features 3× más rico (35→106) que aprovecha series ya sanas y descargadas.
- **Coste aceptado**: Pista A pierde la Gran Depresión y otras 3 crisis pre-1962 como eventos "vistos"
  (siguen en `crisis_catalog` general, solo no en `crisis_windows.pista_A`); Pista B renuncia a 6
  semanas de ragged-edge reciente para mantener el fin común.
- **Fuera de alcance de esta ADR**: decidir la columna exacta de `GW_PREDICTORS_MONTHLY` a usar
  (b/m, D/P, u otra) — se deja pendiente y marcada explícitamente para `03_preprocesado`. Tampoco se
  actualiza `data/catalog.yaml` (ver §4).

## 6. Verificación

Reproducible desde `data/raw/coverage_report.csv` (sin necesitar los `.parquet` crudos):

```python
import pandas as pd, yaml
cov = pd.read_csv('data/raw/coverage_report.csv', parse_dates=['inicio','fin'])
ok = cov[cov.status.isin(['OK','CACHE'])]
b = yaml.safe_load(open('data/benchmark_spec.yaml', encoding='utf-8').read())
assert len(b['pista_A']['series_features']) == 41
assert len(b['pista_B']['series_features']) == 106
assert set(b['pista_A']['series_features']).issubset(set(b['pista_B']['series_features']))
assert b['pista_A']['ventana_fin'] == b['pista_B']['ventana_fin'] == '2026-05-29'
assert b['pista_A']['n_crisis_en_ventana'] == 18 and len(b['crisis_windows']['pista_A']) == 18
assert b['pista_B']['n_crisis_en_ventana'] == 10 and len(b['crisis_windows']['pista_B']) == 10
```
