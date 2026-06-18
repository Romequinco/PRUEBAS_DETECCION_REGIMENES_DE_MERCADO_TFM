# INDEX — Memoria viva del proyecto

> Índice vivo del banco de pruebas de detección de regímenes. Si se retoma el
> proyecto en otra sesión, este fichero + los `.md` de esta carpeta deben bastar
> para reconstruir el estado. Se actualiza A MEDIDA que se avanza, no al final.

## Estado por fase
| Fase | Descripción | Estado |
|---|---|---|
| 0 | Estructura del repo + interfaz + evaluador | ✅ Aprobada (CHECKPOINT 0) |
| 1 | Datos + EDA (causal, sin imputar) | ✅ Hecha — pendiente CHECKPOINT 1 |
| 2 | Estado del arte + lista de detectores | ✅ Aprobada (CHECKPOINT 2) |
| 3 | Implementación de detectores | ✅ Tandas 1-4 hechas (**12 detectores**) — **pendiente CHECKPOINT 3D** |
| 4 | Síntesis comparativa | ✅ Cerrada (CHECKPOINT 4) |
| 5 | Pulido final (docs + estructura) | ✅ Cerrada — hito `capa1-exploracion` |

## FASE 3 — progreso por tandas
- **Núcleo completado por el orquestador** (fricción de FASE 3): `evaluation.py`
  (walk_forward + métricas + evaluate + results_table) y
  `detector_base._economic_state_order` estaban en esqueleto; se implementaron sin
  cambiar firmas ni el contrato de interfaz (validado con detector dummy).
- **Tarea A (CHECKPOINT 3A-bis) — etiquetado económico robusto**: `walk_forward`
  acepta `market_returns` y re-fija el orden de estados (0=calma..n-1=crisis) por
  el retorno medio del S&P 500 en cada fold; imprescindible para detectores que NO
  operan sobre retornos (varianza, σ GARCH, change-point, Mahalanobis de Tandas
  2-4). Fallback con warning si no se pasa. README actualizado. **Re-ejecutada la
  Tanda 1: métricas IDÉNTICAS** (retro-compatible). Cambio de firma: añadido
  kwarg `market_returns` a `walk_forward` (evaluate ya lo tenía).
- **Arreglos 1-3 (CHECKPOINT 3A-ter)**:
  1. `stability_panel` AISLADO en `wf_panel.attrs` (diagnóstico no causal); solo
     `label_stability` lo lee; el resto de métricas usan las etiquetas causales
     OOS (1/fecha). Documentado en docstring y código.
  2. **Filtrado forward causal en HMM** (`detectors/_hmm_utils.py`): D4 (y patrón
     listo para D8) usan filtrado forward en `predict_online`/`predict_proba`, no
     Viterbi intra-bloque. Viterbi solo en la versión in-sample marcada.
  3. `lead_lag` exige cruce **sostenido** (`persist=3` días) — no premia flicker.
  - Re-ejecutada la Tanda 1: **D1 idéntico**; **D3** solo cambia lead_lag (su
    "anticipación" espuria de COVID -160→-20 d, correcto); **D4 causal** cambia por
    el filtrado forward: switching BAJA 0.124→0.100 y duración SUBE 8.1→9.9 (el
    Viterbi por bloque reiniciaba en cada frontera de 21d creando switching
    artificial; el filtro continuo con burn-in es más persistente Y causal),
    false_alarm_rate 0.76→0.73, cov_Infl 0.89→0.86, lead/lag más realistas.
- **Tanda 1 (D1, D3, D4) — hecha, pendiente CHECKPOINT 3A.** Esquema común de 23
  columnas idéntico en los 3; tabla maestra en `results/metrics_master.csv`.
  - **D1 `rule_vix_threshold`** (F1): ventana larga 1998-2026 (evalúa 2008 OOS).
    Cobertura GFC 93.8%, COVID 90%, Inflación 34.9% (VIX flojo en 2022); histéresis
    suprime 2013/2018 (0% / 6.3%). switching 0.013, duración 75d. → fichas
    `detectors/01_rule_vix_threshold.md`, `detectors/rule_vix_threshold.py`.
  - **D3 `clustering_gmm_k3`** (F2): ventana 2015-2026 (2008/2011 NaN, correcto).
    COVID 96%, Inflación 87%; flickering severo (switching 0.126, duración 7.9d) →
    confirma el coste de no tener dinámica temporal. BIC 63016.
  - **D4 `hmm_gaussian_2s`** (F3, puente): in-sample NO causal (2008 100%, marcado)
    vs causal walk-forward (2008 NaN por inicio 2007; falla 2013/2018: 28%/42%). El
    look-ahead compraba SUAVIDAD (switching 0.047 vs 0.124), no acierto en crisis
    grandes. AIC/BIC disponibles.
- **Tanda 2 (D2, D5, D6) — hecha, pendiente CHECKPOINT 3B.** Esquema común 23 cols;
  master `results/metrics_master.csv` (6 detectores).
  - **D2 `rule_composite_riskoff`** (F1): voto VIX+crédito+curva+drawdown con
    histéresis. Ventana 2007 (2008/2011 NaN). COVID 84%, Inflación 53.8% (mejora a
    D1 en 2022: +19pp), pero más ruido (switching 0.039 vs 0.013). false_alarm 0.73.
  - **D5 `markov_switching_var_2s`** (F4): MS de varianza sobre S&P 500, prob.
    FILTRADAS causales (filtrado forward propio, =statsmodels filtered a 1e-13).
    Ventana larga 1993-2026 → 2008 OOS (GFC **99%**, COVID 96%). 2013 no se dispara
    (3.8%, correcto), 2018 sí (81%). switching 0.056, dur 17.9. crisis=alta varianza
    verificado (no invertido). BIC prefiere k=3 sobre k=2 (desplegado k=2). step=63
    por coste del MS (~131 refits). Build tardó ~33 min.
  - **D6 `garch_t_vol`** (F5): GJR-GARCH(1,1)-t, umbral causal sobre σ con histéresis.
    Ventana 1993-2026 → 2008 OOS (GFC **100%**, COVID 94%, Infl 80%). **Capta 2018
    (87%) que D4 no** (reacción same-day), pero NO 2013 (11%, shock de tipos sin vol
    equity). switching 0.014, dur 70d. false_alarm 0.85.
- **Arreglo 4 (CHECKPOINT 3B-bis) — `_economic_state_order` vol-primario.** Resuelto
  en el núcleo el problema que descubrió D6: con K=2 el `z(std)−z(mean)` dejaba que el
  signo ruidoso de una diferencia de medias casi nula invirtiera crisis/calma en
  detectores que separan solo en varianza. Ahora la severidad es **vol-primaria**:
  orden por banda de volatilidad (ancho `VOL_CLOSE_FRAC=15%` de la vol media); el
  retorno medio solo desempata entre vols próximas. Constante `VOL_CLOSE_FRAC` en
  `detector_base`.
  - **Override local de D6 eliminado**: D6 confía ahora en el núcleo.
  - **No-regresión verificada** (re-ejecución de notebooks 01-04,06 + chequeo de orden
    de D5): los 6 detectores dan métricas IDÉNTICAS. D6 idéntico tras quitar el
    override (la inversión solo se daba in-sample, no en walk-forward). D5 con
    crisis=alta varianza en todas las ventanas de fold (no cambia).
  - README y ficha de D6 actualizados.
- **Tanda 3 (D7, D8, D10) — hecha, pendiente CHECKPOINT 3C.** Master de **9
  detectores** en `results/metrics_master.csv` (23 cols). D8/D10 quedaron cortados por
  el límite de sesión de sus subagentes (detector+build listos); los terminó el
  orquestador ejecutando sus builds (como D5).
  - **D7 `changepoint_online`** (F6): CUSUM online robusto sobre vol del S&P 500,
    histórico largo (2008 OOS, GFC 100%). Detección temprana (lead/lag negativo en las
    4 crisis), NO flickea (switching 0.002, dur 436d). Coste robusto (log|r|) vs
    gaussiano: el gaussiano degenera en alarma permanente por la kurtosis; el robusto
    da regímenes limpios (fa_2013/2018 = 0%). Orientación verificada en walk-forward.
  - **D8 `hmm_tstudent_4s`** (F3 avanzado): HMM t-Student propio, **K=4 por BIC**.
    **BIC 24416 vs D4 35379 → la t mejora el ajuste con holgura** (ΔBIC +10963); ν por
    estado decreciente [10.2,7.6,4.2,2.4] (crisis=colas más pesadas). Orden monótono en
    vol verificado en walk-forward. 2008/2011 en train (como D4). Matiz multi-estado:
    su "crisis" es la cola extrema (cov_COVID 0.66 en crisis; corrección+crisis mayor).
    NO desbloquea 2013 (punto ciego de FEATURES, no distribucional).
  - **D10 `turbulence_mahalanobis`** (F1 multivar.): Mahalanobis con cov expanding
    causal sobre [SP500_ret, VIX_chg, DXY_chg, slope_chg] desde 1990 (2013 OOS). Capta
    sistémicas (GFC 82%, COVID 76%) pero **NO 2013 (12%)**: el taper no fue turbulencia
    conjunta. Flickea (switching 0.087). Orientación OK sin parche (valida Arreglo 4).
- **Hallazgo transversal**: **2013 (taper) es el punto ciego universal** — ningún
  detector lo marca (0-12%). Pero el marco lo trata como TRAMPA (false-positive
  window), no como crisis. Tensión a resolver en FASE 4: ¿2013 es crisis rápida a
  captar o trampa a evitar? Los datos dicen que no fue estrés sistémico/turbulento.
- **Tanda 4 (D9, D11, D12 — exploratorios) — hecha, pendiente CHECKPOINT 3D.** Master
  de **12 detectores** (`results/metrics_master.csv`). Patrón anti-cuelgue: subagentes
  dejaron detector+build listos y verificados; el orquestador ejecutó los builds.
  - **D9 `jump_model`** (F2↔F3): Statistical Jump Model (`jumpmodels`, λ=50). Anti-
    flickering ROTUNDO: switching **0.005** vs D3 0.126 (×24 menos), dur 177d. Pero
    pierde cobertura de crisis lentas (Inflación 2022 solo 17%). λ es el mando del
    trade-off persistencia↔sensibilidad. Orientación OK (Arreglo 4).
  - **D11 `msgarch_regime`** (F5): MS-GARCH-t HMP-2004 implementado desde cero (sin R).
    **RESULTADO NEGATIVO documentado**: causal e implementable, pero **degeneración de
    regímenes en walk-forward** (el fold de la GFC colapsa a 1 régimen → cov_GFC **0%**,
    far 0.95). Confirma la fragilidad que avisó el CP2. D6 cubre el hueco. Se mantiene
    en el master como evidencia del fallo.
  - **D12 `deep_ae_regime`** (F7): AE→GMM vs baseline PCA→GMM. **RESULTADO NEGATIVO
    esperado**: el AE empeora al PCA (switching 0.287 vs 0.091, far 0.60 vs 0.14) sin
    ganar cobertura → la no-linealidad no aporta con ~4 crisis. Se corrigió una aserción
    de causalidad demasiado estricta (ruido FP de torch; 0 estados cambian → causal).
- **Aprendizajes transversales para FASE 4**: (1) 2013 punto ciego universal (¿crisis o
  trampa?); (2) detectores multi-estado → `cov_<crisis>` mide solo la cola extrema; (3)
  regímenes degenerados (un estado sin visitar) → cobertura 0 espuria (D11); (4) deep y
  MS-GARCH (exploratorios) no aportan sobre los baselines → valida la parsimonia.
- FASE 3 COMPLETA (12 detectores). Siguiente: FASE 4 (síntesis comparativa).

## FASE 4 — síntesis comparativa (cerrada)
- **Entregables (CHECKPOINT 4)**:
  - `results/metrics_master_final.csv` — tabla maestra FINAL (12 filas, 34 cols):
    crisis estricta + **estrés agregado** (`cov_estres_*`, `fa_estres_*`,
    `false_alarm_rate_estres`) + equidad de ventana (`vio_2008_oos`) + `clase` y
    `coste`. La estrés de los 3 multi-estado (D3, D8, D12) se recomputa con su
    config exacta; los 8 binarios tienen estrés = crisis (copia directa).
  - `notebooks/13_comparison.ipynb` — **ejecutado, 0 errores**. SANITY CHECK pasa:
    la crisis estricta recomputada coincide (±0.01) con el master → el estrés sale
    del mismo panel walk-forward.
  - `results/fase4_*.png` — 6 figuras de cruce de ejes: `sensibilidad_especificidad`,
    `persistencia_sensibilidad`, `bic`, `leadlag`, `rank_heatmap`, `estres_vs_estricta`.
  - `docs/memory/99_conclusions.md` — conclusiones redactadas (qué gana en qué eje,
    6 hallazgos metodológicos, tensión del estrés agregado, recomendación atada a la
    propuesta TFM, bibliografía).
- **Veredictos centrales (honestidad comparativa total)**:
  1. **NO hay mejor detector único; hay "mejor-para-qué"**. Cuatro familias se
     reparten 6 ejes. Cobertura SIEMPRE separada por ventana (los de ventana corta
     2012+/2015+ NO vieron 2008 OOS; nunca se mezclan ni se penalizan por ello).
  2. **D8 hmm_tstudent_4s RESPALDADO** como núcleo (no "confirmado": el respaldo es
     BIC **in-sample** + estrés agregado **favorable-por-construcción** a K≥3, no
     cobertura OOS estricta). Gana BIC (24416 vs D4 35379, ΔBIC +10963 con mismas
     features) y, por **estrés agregado**, su cobertura iguala a los binarios (COVID
     0.66→0.96, Infl 0.33→0.90); su "baja sensibilidad" era reclasificación a
     corrección, no fallo — PERO el estrés sube también fa_2018 0.034→0.81 y far
     0.52→0.79 (trade-off). Cobertura OOS estricta modesta (cov_COVID 0.66) y nunca
     vio 2008 OOS.
  3. **D7 changepoint_online DOMINA 4 ejes**: especificidad 1.00, persistencia 436 d,
     lead/lag **anticipa de forma sostenida** (días topados por el `lookback`, no cifra
     precisa) y coste bajo. Mejor candidato a "segunda velocidad" / alerta temprana.
  4. **Cobertura sistémica (ventana larga): D5 msvar 0.98 ≈ D6 garch 0.97 ≈ D1 vix
     0.92** — la vol manda; la sofisticación apenas bate a la regla VIX (+6 pp).
  5. **Negativos validan parsimonia**: D11 msgarch degenera (cov_GFC 0%, far 0.95);
     D12 deep_ae empeora a su PCA (switching 0.287) sin ganar cobertura. Con ~4
     crisis, la complejidad no se paga.
  6. **2013 (taper) = punto ciego universal** (6+ detectores independientes 0–12%):
     NO es ruido, es que **la taxonomía de régimen importa** — ninguna definición de
     crisis basada en vol/correlación equity captura un shock de tipos. Se mantiene
     como ventana-trampa.
- **Recomendación TFM**: núcleo HMM t-Student multi-estado (respaldo *consistente con*
  la propuesta —BIC in-sample + lógica corrección↔crisis = "dos velocidades"—, no
  superioridad OOS estricta) + change-point rápido tipo D7 como
  alerta temprana + D1/D5/D6 como control. 2013 = límite de taxonomía, no objetivo.
- **CHECKPOINT 4 alcanzado. FASE 4 cerrada. FASE 3 completa, capa de exploración
  terminada.**

## FASE 5 — pulido final (cerrada · hito `capa1-exploracion`)
Pulido de documentación y estructura SIN cambiar resultados ni re-ejecutar nada.
Tres subagentes auditaron en paralelo (informe de MDs, propuesta de limpieza, dictamen
académico); sus informes se conservan como traza en `docs/memory/_pulido_mds.md`,
`_propuesta_limpieza.md`, `_revision_academica.md`.
- **Cifras stale corregidas** (verificadas contra el master): D3 lead/lag COVID −160→−20
  e Inflación −229→−219; D6 `label_stability` 0.982→0.999; D9 "~1.0"→"≈0.98". Erratas:
  `VIX`, `cusum`.
- **99_conclusions afinado** (A1–A5 + M1, sin tocar números): lead/lag separado por
  ventana y **censura del `lookback` (−252) declarada**; salvedad del `rank_heatmap`
  (su col. de cobertura mezcla ventanas); 2013 como **N/A fuera de OOS** en D3 (los 6
  detectores reales se mantienen); "confirma"→"consistente con"; **logL/AIC/BIC marcados
  in-sample**; nueva §4.1 de **limitación de significancia (n≈4 crisis)** con bootstrap
  por bloques (B1) y sensibilidad de hiperparámetros (B2) como trabajo futuro declarado.
- **Estructura**: `scripts/{builders,verify}/` con los `_build_*.py` y `_verify/_rerun`
  (ROOT `parents[1]→[2]`, referencias actualizadas); `notebooks/` queda solo con `.ipynb`.
  `.gitignore` corregido (typo `2#`) y **whitelist de `data/raw/{provenance.json,
  coverage_report.csv}`** (metadatos de procedencia, sin rutas absolutas — verificado).
- **Hito**: tag `capa1-exploracion` marca el estado entregable de la primera capa del TFM.

## Detectores aprobados (CHECKPOINT 2) — 12, por tandas
T1: D1 rule_vix_threshold, D3 clustering_gmm, D4 hmm_gaussian_2s.
T2: D2 rule_composite_riskoff, D5 markov_switching_var, D6 garch_t_vol.
T3: D7 changepoint_online, D8 hmm_tstudent, D10 turbulence_mahalanobis.
T4: D9 jump_model, D11 msgarch_regime, D12 deep_ae_regime (exploratorios).

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
- `99_conclusions.md` — síntesis y recomendación final FASE 4 (✅ redactado): qué
  familia gana en qué eje, 6 hallazgos metodológicos, tensión del estrés agregado,
  recomendación atada a la propuesta TFM y bibliografía con claves reales.

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
