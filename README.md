# Regime Detection — Banco de pruebas comparativo (capa TFM)

Primera capa de un TFM mayor (MIAX). El objetivo **NO** es ningún detector
concreto, sino el **MARCO DE EVALUACIÓN** que juzga a muchos detectores de
regímenes de mercado de forma **comparable y CAUSAL** (sin look-ahead).

> Esta capa NO incluye RAG, agentes ni construcción de portfolio. Solo la
> exploración y evaluación honesta de detectores de régimen.

## Principio rector (no negociable)
- **Features causales**: toda estandarización es z-score *expanding/rolling* (en
  `t` solo se usan estadísticos de datos `<= t`). Nunca media/desv de toda la
  muestra — eso fue el look-ahead detectado en la tarea previa.
- **Evaluación walk-forward / out-of-sample**: nada se juzga in-sample.
- **Misma interfaz, mismas métricas**: cada detector implementa
  `RegimeDetector` y se puntúa con `evaluation.py`.

## Contrato común

### 1. Datos (`src/data_loader.py`)
Set ampliado, descargado SIN imputar; cada serie arranca en su fecha real.

| Categoría | Series | Fuente |
|---|---|---|
| Mercado/riesgo | `^GSPC`, `^VIX`, `TLT`, `IEF`, `HYG`, `^MOVE`*, `GLD` | yfinance |
| Macro/tasas | `T10Y3M` (curva 10Y-3M), `DTWEXBGS` (DXY), `BAMLH0A0HYM2` (HY OAS) | FRED (CSV público) |

*`^MOVE`: si yfinance no lo sirve, se documenta alternativa (FASE 1).*
El set es **ampliable**: si el estado del arte (FASE 2) lo pide, se añaden
correlaciones, breadth, liquidez, etc. Cada serie documenta su fecha de inicio
efectiva y se reporta la **ventana común** (`coverage_report`).

### 2. Features (`src/features.py`)
Causales por construcción: `causal_zscore` (expanding/rolling), retornos log,
vol realizada, spread de crédito, correlación rolling RV/bonos, drawdown,
momentum. Salida en `data/processed/` (gitignored).

### 3. Interfaz de detector (`src/detector_base.py`)
Todo detector hereda `RegimeDetector` y expone:
- `fit(X_train) -> self` — ajuste solo con el tramo de entrenamiento.
- `predict(X) -> np.ndarray` — etiquetas DURAS canónicas (`0`=calma … `n-1`=crisis).
- `predict_proba(X) -> np.ndarray` — probabilidades BLANDAS por estado (one-hot
  si el modelo no es probabilístico).
- `predict_online(X)` — predicción causal día a día para walk-forward.
- `n_states`, `name`, `bibliography` (claves BibTeX), `crisis_state`,
  `crisis_probability(X)`.
- `label_states_economically(...)` — fija el orden canónico de estados por
  criterio económico. Severidad **vol-primaria**: los estados se ordenan por su
  volatilidad de retornos (banda de ancho `VOL_CLOSE_FRAC=15%` de la vol media); el
  retorno medio solo desempata entre estados de vol PRÓXIMA. Así un detector que
  separa solo en varianza (σ GARCH, turbulencia) nunca invierte crisis/calma aunque
  las medias por estado sean casi iguales (Arreglo 4).
- `score / aic / bic` — bondad de ajuste donde el modelo lo permita.

**Causalidad**: en walk-forward la etiqueta de `t` solo puede depender de datos
`<= t`. El suavizado anti-causal (Viterbi sobre toda la muestra) está prohibido
en evaluación online.

### 4. Evaluación (`src/evaluation.py`)
`walk_forward(detector_factory, X, *, market_returns=None, train_size, step,
expanding, min_train)` produce etiquetas/probabilidades out-of-sample;
`evaluate(detector, wf_panel, *, market_returns=None, X_full=None)` calcula la
tabla de métricas estandarizada.

**Etiquetado económico robusto (refinado en FASE 3)**: pasa `market_returns` (la
serie de retornos del S&P 500) a `walk_forward`/`evaluate`. En cada fold se
re-fija el orden de estados (0=calma…n-1=crisis) usando esos retornos, con
severidad **vol-primaria** (volatilidad por estado como criterio principal; el
retorno medio solo desempata entre vols próximas, `VOL_CLOSE_FRAC=15%`). Esto es
necesario para detectores que NO operan sobre retornos crudos (varianza, σ GARCH,
change-point, Mahalanobis): aunque sus dos estados tengan media casi igual, la vol
fija el orden y crisis/calma no se invierte. Sin `market_returns` se conserva el
fallback (ordenar por una columna de retorno de `X`) con *warning*.

Métricas (todas causales, sobre las etiquetas walk-forward):

- **Cobertura de crisis**: % días en "crisis" dentro de 2008/2011/2020/2022.
- **Falsos positivos**: comprobación de NO disparo sostenido en 2013 (taper) y
  Q4 2018.
- **Lead/lag** respecto al suelo del drawdown del S&P 500. Exige señal
  **sostenida** (`p_crisis≥0.5` durante `persist=3` días consecutivos) para no
  premiar el flickering de un cruce suelto.
- **Tasa de falsas alarmas** global.
- **Persistencia / frecuencia de conmutación** (penaliza *flickering*).
- **Log-likelihood / AIC / BIC** donde aplique.
- **Estabilidad de etiquetas** bajo walk-forward. Se mide con un
  `stability_panel` AISLADO (en `wf_panel.attrs`, diagnóstico NO causal): ninguna
  otra métrica lo lee; cobertura/falsas alarmas/lead-lag solo usan las etiquetas
  causales OOS (una por fecha, del fold que la predijo por primera vez).

**Causalidad de los HMM (D4, D8)**: `predict_online`/`predict_proba` usan
**filtrado forward** causal (`detectors/_hmm_utils.py`), NO el Viterbi/forward-
backward suavizado de hmmlearn (que mira días futuros del bloque). El Viterbi
global queda solo para la versión in-sample explícitamente marcada NO causal.

Resultados comparables en `results/` (tabla maestra, una fila por detector).

## Estructura

```
regime-detection/
├── README.md               # este contrato
├── requirements.txt
├── data/{raw,processed}/   # gitignored
├── src/
│   ├── data_loader.py      # yfinance + FRED, sin imputar
│   ├── features.py         # features causales
│   ├── detector_base.py    # interfaz RegimeDetector
│   ├── evaluation.py       # walk-forward + métricas comunes
│   └── viz.py              # visualizaciones estándar
├── detectors/              # una implementación por familia (FASE 3)
├── notebooks/              # 00_eda + un notebook por detector
├── docs/
│   ├── context/            # propuesta TFM + resumen tarea previa
│   ├── memory/             # memoria viva del proyecto (INDEX.md, fases)
│   └── references.bib      # bibliografía central
└── results/                # tablas de métricas, csv, figuras
```

## Fases del proyecto
0. **Estructura** (esta entrega) — interfaz + evaluador + contrato.
1. **Datos + EDA** — descarga, features causales, `00_eda.ipynb`.
2. **Estado del arte** — `docs/memory/00_state_of_the_art.md` + lista final de
   detectores (a aprobar).
3. **Detectores** — un notebook ejecutado por familia, contra la misma interfaz.
4. **Síntesis comparativa** — tabla maestra + conclusiones.

La memoria viva está en `docs/memory/INDEX.md`.

## Uso (a partir de FASE 1)
```bash
pip install -r requirements.txt
python -m src.data_loader      # descarga set ampliado -> data/raw/
python -m src.features         # features causales -> data/processed/
# notebooks/00_eda.ipynb para el EDA
```
