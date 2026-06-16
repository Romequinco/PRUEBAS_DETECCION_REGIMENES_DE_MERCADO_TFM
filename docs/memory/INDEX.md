# INDEX — Memoria viva del proyecto

> Índice vivo del banco de pruebas de detección de regímenes. Si se retoma el
> proyecto en otra sesión, este fichero + los `.md` de esta carpeta deben bastar
> para reconstruir el estado. Se actualiza A MEDIDA que se avanza, no al final.

## Estado por fase
| Fase | Descripción | Estado |
|---|---|---|
| 0 | Estructura del repo + interfaz + evaluador | ✅ Aprobada (CHECKPOINT 0) |
| 1 | Datos + EDA (causal, sin imputar) | ✅ Hecha — pendiente CHECKPOINT 1 |
| 2 | Estado del arte + lista de detectores | ✅ Hecha — **pendiente CHECKPOINT 2** |
| 3 | Implementación de detectores | ⬜ Pendiente |
| 4 | Síntesis comparativa | ⬜ Pendiente |

## Resumen FASE 1 (ver 01_data_and_eda.md)
- **9 series** descargadas sin imputar. FRED inaccesible → fallbacks documentados:
  DXY=`DX-Y.NYB`, curva 10Y-3M=proxy `^TNX−^IRX`, HY_OAS omitida (crédito vía HYG).
  `^MOVE` sí está en yfinance.
- **Ventana común 2007-04-11 → 2026-06-12** (gobernada por HYG; consistente con
  tarea previa). Features causales: 2007-07-06 → 2026-06-12, **15 features**.
- **Causalidad verificada**: `assert_causal` da `max_abs_diff=0.0` en las 15.
- **Fat tails** confirmadas (SP500 kurt exceso 25.6, HYG 39.6) → motiva t-Student.
- **DRAWDOWN_TROUGHS** calculados de datos reales y cableados en evaluation.py.
- Tensión pendiente: GFC 2008 cerca del inicio de datos → cuidado en walk-forward.

## Resumen FASE 2 (ver 00_state_of_the_art.md)
- **7 familias** documentadas en `sota/` + tabla comparativa transversal y
  resolución de solapes (HMM↔MS, GMM↔GMM-HMM, RS-GARCH↔MS, clustering↔redes).
- **Bibliografía fusionada**: 77 entradas nuevas de los 7 sidecars `.bib` añadidas
  a `references.bib` (84 totales, sin claves duplicadas; prefijos por familia).
- **Lista propuesta: 12 detectores** (D1–D12), de baseline a avanzado, pendiente
  de aprobación. Imprescindibles: **D1 `rule_vix_threshold`** y **D4
  `hmm_gaussian_2s`** (puente con la tarea previa). Recomendados: D2 regla
  compuesta, D3 GMM, D5 Markov-Switching, D6 GARCH-t, D7 change-point online,
  D8 HMM t-Student, D9 jump model, D10 turbulencia Mahalanobis.
  Opcional-exploratorio: D11 MS-GARCH, D12 deep AE (escasez de datos). Las 7
  familias quedan representadas.

## Documentos de memoria
- `00_state_of_the_art.md` — estado del arte FASE 2: tabla transversal, solapes y
  lista propuesta de 12 detectores (✅ rellenado).
- `sota/01_reglas_umbrales.md` … `sota/07_redes_neuronales.md` — 7 fichas de
  familia (solo lectura) + sus sidecars `.bib` homónimos:
  - `sota/01_reglas_umbrales.md` — Reglas / umbrales (F1).
  - `sota/02_clustering.md` — Clustering estático (F2).
  - `sota/03_hmm.md` — HMM con emisiones latentes (F3).
  - `sota/04_markov_switching.md` — Markov-Switching econométrico (F4).
  - `sota/05_volatilidad_garch.md` — Volatilidad / GARCH / RS-GARCH (F5).
  - `sota/06_change_point.md` — Change-point detection (F6).
  - `sota/07_redes_neuronales.md` — Redes neuronales / no supervisado (F7).
- `01_data_and_eda.md` — datos descargados, decisiones y hallazgos del EDA (FASE 1).
- `<NN>_<detector>.md` — uno por detector (FASE 3): "Implementado" + "Descubierto".
- `99_conclusions.md` — síntesis y recomendación final (FASE 4).

## Contexto de partida (de docs/context/)
- **Tarea previa**: HMM gaussiano 2 estados, in-sample. Acertaba crisis grandes
  (2008: 98.6%, 2020: 92.3%) pero se perdía correcciones rápidas (taper 2013:
  10.9%, Q4 2018: 20.6%). Limitaciones detectadas → motivan este banco de pruebas:
  look-ahead en z-scores de muestra completa, in-sample sin walk-forward, Viterbi
  duro sin probabilidades, supuesto gaussiano (colas), etiquetado por umbral frágil.
- **Propuesta TFM**: define el sistema mayor (HMM t-Student 4 estados, 14 features,
  BIC, train/val/test 2000-2024). Esta capa es solo la exploración de regímenes.

## Decisiones del núcleo (FASE 0)
- Interfaz `RegimeDetector` con etiquetas canónicas (0=calma … n-1=crisis) para
  comparabilidad entre detectores y entre folds.
- Evaluación walk-forward (expanding por defecto) como único protocolo causal.
- Ventanas de crisis/falsos positivos centralizadas en `evaluation.py`
  (CRISIS_WINDOWS, FALSE_POSITIVE_WINDOWS, DRAWDOWN_TROUGHS); se afinan en FASE 1.

## Decisiones pendientes (a confirmar por el usuario)
- Lista final de familias de detectores (tras FASE 2 / CHECKPOINT 2).
- Set definitivo de features (puede ampliarse según estado del arte FASE 2).
- Estrategia de walk-forward ante el inicio tardío de datos (2007) para cubrir
  2008 OOS (decidir en FASE 3).

## Decisiones resueltas
- `^MOVE`: disponible en yfinance (2002-11), sin alternativa necesaria.
- FRED inaccesible en este entorno → fallbacks yfinance documentados.
