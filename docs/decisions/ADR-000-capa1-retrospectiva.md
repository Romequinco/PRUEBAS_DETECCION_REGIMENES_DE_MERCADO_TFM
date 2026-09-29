# ADR-000 — Retrospectiva de la Capa 1: el primer banco de 12 detectores

- **Estado:** Registro retrospectivo · redactado 2026-09-29 (describe decisiones tomadas entre la
  FASE 0 y la FASE 5 de la Capa 1, hasta la congelación del 2026-07-18).
- **Por qué existe:** las ADR-001 a ADR-003 parten de la Capa 1 como algo dado. Esta ADR fija por
  escrito **qué se decidió entonces y por qué**, para que la historia del proyecto se lea en orden:
  ADR-000 (Capa 1) → ADR-001 (congelación y re-base de datos) → ADR-002 (ventanas) → ADR-003
  (causalidad de calendario y ranking) → ADR-004 (unificación).
- **Material:** [`../historia/capa1/`](../historia/capa1/README.md) (memoria, informe, métricas v1);
  teoría en [`../teoria/`](../teoria/00_estado_del_arte.md); fichas en [`../detectores/`](../detectores/D01_rule_vix_threshold.md);
  notebooks y código v1 en el tag `capa1-final`.

---

## 1. Contexto

El TFM (MIAX) propone un sistema de detección de regímenes basado en un HMM t-Student multi-estado
([`../context/TFM_Proposal_v2.pdf`](../context/TFM_Proposal_v2.pdf)). La **tarea previa** era un HMM
gaussiano de 2 estados in-sample que acertaba las crisis grandes (2008: 98.6 %, 2020: 92.3 %) pero se
perdía las correcciones rápidas (2013: 10.9 %, Q4 2018: 20.6 %)
([`../context/RESUMEN_DETECCION_REGIMENES.md`](../context/RESUMEN_DETECCION_REGIMENES.md)). Sus
problemas —z-scores de muestra completa (look-ahead), evaluación in-sample, Viterbi duro sin
probabilidades, supuesto gaussiano y etiquetado frágil— no permitían saber si el HMM era una buena
elección o solo una elección.

## 2. Decisiones de la Capa 1

1. **Banco de pruebas, no detector.** El objetivo fue comparar familias bajo un marco común y
   responder "¿cuál es el mejor *para qué*?", no fabricar un campeón.
2. **Interfaz común `RegimeDetector`** con etiquetas canónicas (0 = calma … n−1 = crisis) y orden
   económico de estados **vol-primario** (`_economic_state_order`, banda `VOL_CLOSE_FRAC`), para que
   los estados sean comparables entre detectores y entre folds. Sigue viva en
   `regimenes.detectores.base`.
3. **Walk-forward causal como único protocolo** (expanding, re-ajuste cada `step`), con
   `market_returns` para re-fijar el orden de estados en cada fold y un `stability_panel` aislado como
   diagnóstico no causal.
4. **Features causales** (z-score expanding) verificadas con `assert_causal` (truncar el futuro no
   cambia el pasado).
5. **Filtrado forward en los HMM** en lugar de Viterbi por bloque; `lead_lag` exige cruce sostenido
   (3 días) para no premiar el parpadeo.
6. **Datos sin imputar:** 9 series por yfinance (FRED inaccesible, fallbacks documentados), ventana
   común 2007-04-11 → 2026-06 gobernada por HYG, 15 features.
7. **Estado del arte por familias (7) antes de implementar**, con bibliografía por familia, y lista de
   **12 detectores** de baseline a avanzado aprobada en el CHECKPOINT 2; implementados en 4 tandas.
8. **Honestidad comparativa:** cobertura separada por grupo de ventana (vio 2008 OOS o no); métricas
   de ajuste (logL/AIC/BIC) declaradas in-sample; "estrés agregado" para los multi-estado; los
   resultados negativos (D11, D12) se conservan como evidencia.

| ID | Detector | Familia |
|---|---|---|
| D1 · D2 · D10 | `rule_vix_threshold` · `rule_composite_riskoff` · `turbulence_mahalanobis` | F1 reglas / umbrales |
| D3 · D9 | `clustering_gmm` · `jump_model` | F2 clustering (D9 puente F2↔F3) |
| D4 · D8 (· D13) | `hmm_gaussian_2s` · `hmm_tstudent` (· `hsmm_tstudent`, ablación posterior) | F3 HMM |
| D5 | `markov_switching_var` | F4 Markov-Switching |
| D6 · D11 | `garch_t_vol` · `msgarch_regime` | F5 GARCH |
| D7 | `changepoint_online` | F6 change-point |
| D12 | `deep_ae_regime` | F7 redes |

## 3. Resultados

- **No hay detector dominante:** cuatro familias se reparten seis ejes. Cobertura sistémica en ventana
  larga: D5 ≈ D6 ≈ D1; especificidad, persistencia, anticipación y coste: D7; ajuste (BIC): D8.
- **Cinco hallazgos metodológicos** que sobreviven al re-base (detalle en
  [`../historia/capa1/README.md`](../historia/capa1/README.md) §2): el look-ahead compraba suavidad,
  no acierto; la t-Student mejora el ajuste con holgura (ΔBIC ≈ +10963); el filtrado forward es más
  causal y más estable que Viterbi por bloque; 2013 es un punto ciego universal (la taxonomía de
  features importa); la complejidad extra no se paga con ~4 crisis.
- **Recomendación v1:** núcleo HMM t-Student + change-point como alerta temprana + reglas/vol como
  control. Es el origen de las fusiones D7+D8 y D2+D6 de v2.

## 4. Qué falló (y motivó ADR-001)

- **Incomparabilidad 1:1:** cada detector construía su propia matriz de features (1, 4, 7 o 15) y su
  propia ventana OOS; unos se juzgaban con 3,3× más datos y el doble de crisis que otros.
- **Potencia nula:** ~4 crisis en la ventana común; ningún intervalo de confianza separaba detectores.
- **Taxonomía de features pobre:** casi todo vol/equity; sin crédito ni curva reales → 2013 invisible.
- **Sin banco congelado:** los datos no eran una variable controlada, eran parte del detector.

## 5. Consecuencias

- ADR-001 congeló la Capa 1 y re-basó los datos (dos pistas, 166 series, `configs/benchmark_spec.yaml`).
- El **juez** (walk-forward + métricas) y la **interfaz** de la Capa 1 se reutilizaron en v2; los 12
  detectores se re-evaluaron sobre el banco congelado (ADR-002, ADR-003).
- ADR-004 integra definitivamente la Capa 1 en el paquete `regimenes`; su memoria queda en
  [`../historia/capa1/`](../historia/capa1/README.md) y su código y notebooks originales en el tag
  `capa1-final`.
