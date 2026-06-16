# INDEX — Memoria viva del proyecto

> Índice vivo del banco de pruebas de detección de regímenes. Si se retoma el
> proyecto en otra sesión, este fichero + los `.md` de esta carpeta deben bastar
> para reconstruir el estado. Se actualiza A MEDIDA que se avanza, no al final.

## Estado por fase
| Fase | Descripción | Estado |
|---|---|---|
| 0 | Estructura del repo + interfaz + evaluador | ✅ Hecha — pendiente CHECKPOINT 0 |
| 1 | Datos + EDA (causal, sin imputar) | ⬜ Pendiente |
| 2 | Estado del arte + lista de detectores | ⬜ Pendiente |
| 3 | Implementación de detectores | ⬜ Pendiente |
| 4 | Síntesis comparativa | ⬜ Pendiente |

## Documentos de memoria
- `00_state_of_the_art.md` — estado del arte (se rellena en FASE 2).
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
- Disponibilidad real de `^MOVE` en yfinance y su alternativa.
- Set definitivo de features (puede ampliarse según estado del arte).
