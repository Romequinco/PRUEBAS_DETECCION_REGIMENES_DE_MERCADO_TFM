# Trazabilidad de notebooks (ADR-004)

Qué notebook/sección antigua corresponde a qué notebook/sección nueva. Mover no es
eliminar; lo eliminado va a `registro_eliminaciones.md`.

| notebook / sección antigua | notebook / sección nueva | nota |
|---|---|---|
| `04_benchmark_detectores.ipynb` (archivo) | `04_protocolo_evaluacion.ipynb` | `git mv`; reenfocado como protocolo; todas las celdas conservadas |
| 04 · cabecera (Fase D, objetivo, entradas, salidas) | 04 · cabecera, bajo "Origen" | rutas → `regimenes.*`, `configs/benchmark_spec.yaml`, `results/benchmark/`; se añade índice |
| 04 · celda de imports (búsqueda de `ROOT` + `sys.path`, `from src import benchmark`) | 04 · celda de imports | `from regimenes import benchmark as bm`, `regimenes.rutas.ROOT` |
| — (nuevo) | 04 · §1 Protocolo walk-forward causal (+ tabla de calendario por pista/detector) | contenido nuevo desde `regimenes.evaluacion.walk_forward` y el registro |
| — (nuevo) | 04 · §1.1 Propagación de estado entre refits (ADR-003) (+ demo sintética `context=0` vs `'auto'`) | contenido nuevo; no lee `results/` |
| 04 · §1 Qué se va a probar | 04 · §2 Qué se va a probar | solo renumerado |
| 04 · §2 Cómo se implementa la comparación | 04 · §3 Cómo se implementa la comparación | párrafo "Estado de los autómatas" conservado como histórico + nota vigente (ADR-003); referencias 05 → 05–11 y 12 |
| — (nuevo) | 04 · §4 Caché con huella de contenido | contenido nuevo desde `regimenes.benchmark.cache` |
| 04 · celda de interruptores (sin encabezado) | 04 · §5 Interruptores `EJECUTAR` / `CACHE_ONLY` | se añade `EJECUTAR = False` (`CACHE_ONLY = not EJECUTAR`); `OUTPUT_DIR = RESULTS_BENCHMARK`; `TRAIN_DAYS = DEFAULT_TRAIN_DAYS` (mismos valores) |
| 04 · §3 Gate previo | 04 · §6 Gate previo | `python -m src.ingest.download` → `python -m regimenes.datos` |
| 04 · §4 Ejecución reanudable (+ re-ejecución paralela) | 04 · §7 Ejecución reanudable y CLI paralela | `python -m src.benchmark` → `python -m regimenes.benchmark`; tabla de opciones de la CLI; la celda imprime el comando si falta caché |
| 04 · §5 Inventario de resultados | 04 · §8 Inventario de resultados | solo renumerado / referencia a 12 |
| 04 · §6 Cierre | 04 · §9 Cierre | referencias a 05–11 y 12 |
| 04 · título `# 04 — Benchmark causal de los 12 detectores (Fase D)` | 04 · título `# 04 — Protocolo de evaluación: walk-forward causal, caché y ejecución del benchmark` | renombrado al reenfocarse como protocolo; el nombre anterior se cita en la nota "Origen" de la cabecera (registrado por el comparador final, `informe_unificacion.md`) |
| 04 · §4 título `## 4. Ejecución reanudable` | 04 · §7 título `## 7. Ejecución reanudable y CLI paralela` | renumerado y ampliado con la CLI paralela (fila §4 → §7 de arriba) |
| `05_comparacion_detectores.ipynb` (archivo) | `12_comparativa.ipynb` | `git mv`; todas las celdas conservadas, mismo orden y numeración §1–§12 |
| 05 · título `# 05 — Comparación de detectores y decisión de diseño (Fase D)` | 12 · título `# 12 — Comparativa de detectores y decisión de diseño (Fase D)` | renumerado y "Comparación" → "Comparativa" (nombre del archivo nuevo); resto del título igual (registrado por el comparador final, `informe_unificacion.md`) |
| — (nuevo) | 12 · celda 2 "Origen" + enlaces a 04, 05–11, 13 y 14 | contenido nuevo |
| 05 · celda de imports (`from src import benchmark as bm`, rutas `data/benchmark_spec.yaml`, `results/benchmark_v2`) | 12 · celda de imports | `load_metrics`/`metrics_provenance` de `regimenes.benchmark`; ranking ADR-003 como `rk` (`regimenes.evaluacion.ranking`); rutas vía `regimenes.rutas`; `bm.X` → `rk.X` en todas las celdas |
| `06_fusion_d07_d08.ipynb` (archivo) | `13_fusion_d07_d08.ipynb` | `git mv`; las 21 celdas conservadas en el mismo orden; salidas guardadas sin re-ejecutar |
| 06 · título `# 06 — D7 como alerta temprana + D8 como confirmación` | 13 · título `# 13 — …` | solo renumerado |
| 06 · celda de imports (búsqueda de `ROOT` por `data/benchmark_spec.yaml` + `sys.path.insert`; `from src import evaluation`, `src.benchmark`, `src.fusion`; `results/benchmark_v2`, `results/fusion_d07_d08`) | 13 · celda de imports | `regimenes.evaluacion`, `regimenes.benchmark.configure_evaluation`, `regimenes.fusion` (recarga de `regimenes.fusion.maquina` + re-export); `BENCHMARK = rutas.RESULTS_BENCHMARK`, `OUTPUT = rutas.RESULTS_FUSION_D07_D08`; se quitan `Path`/`sys` (ya no se usan) |
| 06 · markdown §1 y §8 (`src.fusion.operational_utility`, "notebook 07") | 13 · mismas celdas | → `regimenes.fusion.operational_utility`, "notebook 14" |
| 06 · test económico (`ROOT / 'data' / 'raw' / 'yfinance' / 'SP500.parquet'`) | 13 · misma celda | → `rutas.DATA_RAW / 'yfinance' / 'SP500.parquet'` |
| `07_fusion_d02_d06.ipynb` (archivo) | `14_fusion_d02_d06.ipynb` | `git mv`; las 18 celdas conservadas en el mismo orden; salidas guardadas sin re-ejecutar |
| 07 · título `# 07 — Selección operativa: D2 como alerta + D6 como confirmación` | 14 · título `# 14 — …` | solo renumerado; `src/fusion.py` → `regimenes.fusion` (`src/regimenes/fusion/maquina.py`), "notebook 06" → "notebook 13" |
| 07 · markdown §1 (`src.fusion.operational_utility`, "benchmark (05) y el notebook 06") | 14 · misma celda | → `regimenes.fusion.operational_utility`, "benchmark (12) y el notebook 13" |
| 07 · celda de imports (búsqueda de `ROOT` + `sys.path.insert`; `src.*`; `results/benchmark_v2`, `results/fusion_d02_d06`) | 14 · celda de imports | igual que 13; `OUTPUT = rutas.RESULTS_FUSION_D02_D06` |
| 07 · markdown de paneles (`results/benchmark_v2/panels/*.parquet`) y §8 ("notebook 06") | 14 · mismas celdas | → `results/benchmark/panels/*.parquet`, "notebook 13" |
| — (nuevo) | `15_sinteticos_generadores.ipynb` … `20_pseudolive.ipynb` | esqueletos nuevos (flag `IMPLEMENTADO = False`); no reciben contenido de notebooks antiguos |
| v1 `capa1_exploracion/notebooks/04_hmm_gaussian_2s.ipynb` (archivo) | `07_familia_F3_hmm.ipynb` | contenido rescatado sobre datos v2; el notebook v1 no se mueve: se retira con `capa1_exploracion/` y queda íntegro en el tag `capa1-final` |
| v1 04 · §0 presentación (baseline puente, experimento in-sample vs causal, hipótesis CP2) | 07 · §0 resumen, §1.2, §4.3 y §6 | hipótesis CP2 y veredicto en §6 |
| v1 04 · índice + hilo conductor filtrado vs suavizado | 07 · §0 índice y §1.2 (cita del hilo conductor) | — |
| v1 04 · celda de imports (`sys.path`, `src.evaluation`, `detectors.*`, `data/processed/features.parquet`) | 07 · celda de imports y §3 | `regimenes.*`; datos = paneles v2 de pista vía `load_track_panel` |
| v1 04 · §1 features puente (`BRIDGE_FEATURES`, describe) | 07 · §6 tabla de configuración v1 vs v2 | v2 usa `CORE_A`/`CORE_B` del registro |
| v1 04 · §2 versión IN-SAMPLE (Viterbi global, NO causal) | 07 · §4.3 (ajuste de muestra completa ilustrativo) | recomputado sobre cada pista v2 |
| v1 04 · §3 versión CAUSAL walk-forward (`ev.walk_forward`, `ev.evaluate`) | 07 · §3 (caché del benchmark) y §4.1–4.2 | ya no se ejecuta en el notebook: panel OOS de `results/benchmark` |
| v1 04 · §4 comparación in-sample vs causal (tabla limpia + barras) | 07 · §4.3 (tabla `render_table_figure` por pista) | las barras de cobertura causal quedan en §4.1–4.2 |
| v1 04 · §5 S&P 500 coloreado por régimen | 07 · §4.1–4.2 (`ficha`) | solo la versión causal OOS; la in-sample se resume en la tabla §4.3 |
| v1 04 · §6 probabilidad de crisis y lead/lag | 07 · §4.1–4.2 (`ficha`: P(crisis) filtrada + lead/lag por crisis) | — |
| v1 04 · §7 filtrado vs suavizado | 07 · §4.3 | misma figura por pista |
| v1 04 · §8 emisiones por estado y matriz de transición | 07 · §4.3 | violines de `SP500_vol_z`/`SP500_ret_z` (VIX no existe en A) |
| v1 04 · §9 matriz de transición, timeline e histograma de duraciones | 07 · §4.3 (transición + duración esperada + histograma OOS) y §4.1–4.2 (timeline sobre el precio) | — |
| v1 04 · §10 volcado de métricas a `results/` | 07 · §3 | sustituido por las métricas del benchmark (`results/benchmark/metrics`); no se escriben CSV v1 |
| v1 04 · §11 conclusión y veredicto CP2 | 07 · §6 (texto) + celda con cifras v1 desde `docs/historia/capa1/resultados` | cifras ya no escritas a mano |
| v1 `capa1_exploracion/notebooks/08_hmm_tstudent.ipynb` (archivo) | `07_familia_F3_hmm.ipynb` | contenido rescatado sobre datos v2 |
| v1 08 · §0 presentación (t-Student vs GMM-HMM, K por BIC, causalidad, hipótesis CP2) | 07 · §1.3, §1.5 y §6 | — |
| v1 08 · §1 selección de K por BIC sobre {3, 4} | 07 · §4.6 celda `EXPLORAR_K` (opcional) | en v2 K está fijado a priori en el registro |
| v1 08 · §2 parámetros por estado y monotonía in-sample | 07 · §4.6 (tabla por estado, sin `assert`) | se informa el fallo en lugar de detener el notebook |
| v1 08 · §3 PCA 2D de las features | 07 · §4.6 (figura `D08_<pista>_colas_t_y_pca`) | — |
| v1 08 · §4 colas t vs gaussiana y ν por estado | 07 · §4.6 | — |
| v1 08 · §5–§6 walk-forward causal y fila de métricas | 07 · §3 y §4.4–4.5 | caché del benchmark |
| v1 08 · §7–§8 monotonía y severidad en walk-forward | 07 · §4.6 (severidad OOS desde el panel del benchmark) | — |
| v1 08 · §9–§10 S&P 500 por K estados y P(crisis) filtrada | 07 · §4.4–4.5 (`ficha`) | — |
| v1 08 · §11 matriz de transición y persistencia | 07 · §4.6 | — |
| v1 08 · §12 crisis, trampas y corrección+crisis | 07 · §4.4–4.5 (columnas `corr+crisis`) | — |
| v1 08 · §13 BIC vs D4 | 07 · §5 (BIC del benchmark por pista) y §6 (ΔBIC v1) | — |
| v1 08 · §14 conclusión (problema distribucional vs cobertura de features) | 07 · §6 y §7 | — |
| v1 `capa1_exploracion/notebooks/A1_hsmm_ablation.ipynb` (archivo) | `07_familia_F3_hmm.ipynb` §4.7 | D13 sigue fuera de la matriz oficial |
| v1 A1 · §0 presentación, alcance de la estimación en dos etapas, hipótesis | 07 · §1.4 y §4.7 | — |
| v1 A1 · §1 selección de K por BIC aproximado | 07 · §4.6 celda `EXPLORAR_K` (incluye D13) | — |
| v1 A1 · §2 parámetros y monotonía in-sample | 07 · §4.7.3 (tabla de duración) y §4.7.2 (severidad OOS) | — |
| v1 A1 · §3 PCA 2D | 07 · §4.7.3 | etiquetas = argmax del filtro (no Viterbi segmental) |
| v1 A1 · §4 PMF y hazard explícitos vs geométricos | 07 · §4.7.3 | — |
| v1 A1 · §5–§6 walk-forward semi-Markov y fila de métricas | 07 · §4.7.2 (`EJECUTAR_ABLACION`, `run_one` con el protocolo del benchmark) | salida en `results/detectores/f3_hmm/ablacion_d13/` |
| v1 A1 · §7–§12 monotonía, severidad, S&P 500, P(crisis), cadena embebida, fragmentación | 07 · §4.7.2 (`ficha` + severidad + cambios de estado) y §4.7.3 (cadena embebida) | — |
| v1 A1 · §13 comparación controlada D13 vs D8 | 07 · §4.7.1 (tabla v1 desde `docs/historia/capa1/resultados/ablation_hsmm`) y §4.7.2 (v2) | — |
| v1 A1 · §14 criterios previos + conclusión final | 07 · §4.7.1 (criterios recalculados en celda; conclusión reformulada sin cifras a mano) y §7 | — |
| v1 `capa1_exploracion/notebooks/05_markov_switching_var.ipynb` · cabecera (salvo la hipótesis CP2, ver dos filas más abajo), §2 parámetros/μ-σ, §8 filtrada, §9 filtrada vs smoothed, §10 transición | `08_familia_F4_markov_switching.ipynb` · §1 teoría (1.1–1.4), §4.1, §4.2 a–c | reescrito sobre datos v2 y paneles OOS del benchmark; ajustes de muestra completa marcados como ilustrativos no causales |
| v1 05 · §3 selección de k (AIC/BIC), §7 retorno por régimen, §11 crisis/trampas, §12 conclusión CP2 | 08 · §4.2 e (opcional `EXPLORAR_K`, antes `COMPARAR_K`), §4.2 d, §4.3, §6 hallazgos v1 | cifras v1 leídas de `docs/historia/capa1/resultados/metrics_05_*.csv`, nunca escritas a mano |
| v1 05 · cabecera «Hipótesis CHECKPOINT 2 (D5)» y §12 contraste con la hipótesis (+ veredicto de la ficha `D05_markov_switching_var.md`) | 08 · §6.1 Hipótesis del CHECKPOINT 2 y veredicto (v1 → v2) | hipótesis citada literalmente; veredicto v1 punto por punto sin cifras; cifras v1 (`metrics_05_*.csv`) y lectura v2 (caché + ranking ADR-003 + curtosis por régimen del panel OOS) generadas en celda |
| v1 05 · §1 endog: retorno log del S&P 500 (histórico largo desde 1985) | 08 · §2 configuración (features y ventana por pista desde el registro) | en v2 la serie y su inicio salen del registro/pista, no de una carga propia del notebook |
| v1 05 · §4 walk-forward causal | 08 · §3 ejecución y carga de la caché del benchmark (+ §1.4 causalidad en el walk-forward) | ya no se ejecuta en el notebook |
| v1 05 · §5 evaluación estandarizada y fila de métricas | 08 · §4.4 métricas y puesto en el ranking ADR-003 | sustituido por el benchmark; CSV v1 leído en §6 |
| v1 05 · §6 S&P 500 coloreado por régimen (OOS) | 08 · §4.1 figura `D05_<pista>_estados_oos` | — |
| v1 `06_garch_t_vol.ipynb` · cabecera (salvo la hipótesis CP2, ver fila siguiente), §1 parámetros, §1.1 news impact, §1.2 QQ-t, §2 causalidad, §4 σ+umbral, §4.1 σ por régimen, §4.2 σ vs realizada, §5–§7, §9 conclusión | `09_familia_F5_garch.ipynb` · §1.1–1.3 teoría, §4.1.1–4.1.4, §6 hallazgos v1 | la verificación de causalidad de v1 se cita en §1.3 (no se re-ejecuta); cifras v1 desde `metrics_06_*.csv` |
| v1 06 · cabecera «Hipótesis CP2» y §9 «¿se cumple la hipótesis CP2?» (+ veredicto de la ficha `D06_garch_t_vol.md`) | 09 · §6.1 Hipótesis del CHECKPOINT 2 y veredicto (v1 → v2) | hipótesis citada literalmente (ficha); veredicto v1 «se cumple PARCIALMENTE» (2018 sí, 2013 no); lectura v2 generada en celda (Q4-2018 ahora crisis, *taper* 2013 trampa) |
| v1 06 · §2 verificación de causalidad de la σ | 09 · §1.3 (se cita) | no se re-ejecuta (fila en `registro_eliminaciones.md`) |
| v1 06 · §3 walk-forward causal | 09 · §3 ejecución y carga de la caché del benchmark | ya no se ejecuta en el notebook |
| v1 06 · §8 volcado de métricas | 09 · §4.1.4 métricas y ranking ADR-003 | sustituido por el benchmark; CSV v1 leído en §6 |
| v1 `11_msgarch_regime.ipynb` · cabecera (ver filas siguientes), §4 P(crisis), §6 cobertura, §7 degeneración fold a fold, §8 causa raíz, §11 conclusión (salvo el veredicto CP2, ver fila siguiente) | `09_familia_F5_garch.ipynb` · §1.4–1.5 teoría, §4.2.1–4.2.6, §5 comparación D06 vs D11, §6 | diagnóstico fold a fold recalculado desde el panel OOS v2; ajuste in-sample opcional (`AJUSTE_D11_COMPLETO`); cifras v1 desde `metrics_11_*.csv` |
| v1 11 · §11 «Hipótesis CP2 → veredicto» (RECHAZADA out-of-sample) + veredicto de la ficha `D11_msgarch_regime.md` (se cumple la parte de la fragilidad) | 09 · §6.1 Hipótesis del CHECKPOINT 2 y veredicto (v1 → v2) | las dos formulaciones de la hipótesis citadas literalmente; lectura v2 generada en celda (score D11 frente a D05/D06 en el ranking ADR-003, folds en esquina de §4.2.2) |
| v1 11 · cabecera "La decisión técnica (honesta)" (sin librería MS-GARCH madura en Python; implementación propia sin R) | 09 · §1.4 (párrafo "D11 implementa HMP desde cero (numpy/scipy, sin R)") | el detalle de entorno (`rpy2` no instalado, `arch` 8.0 sin MS-GARCH) no se rescata (fila en `registro_eliminaciones.md`) |
| v1 11 · cabecera "Por qué HMP-2004 sí es tratable (y la naive no)" | 09 · §1.4 (dependencia de la trayectoria, ecuación HMP, filtro de Hamilton O(TK²), μ única y ν compartido) | — |
| v1 11 · cabecera "Posición frente a D5 y D6 (no es redundante)" | 09 · §0 (D11 = síntesis de F4 y F5) y §1.6 relación con otras familias | — |
| v1 11 · cabecera "Honestidad del alcance" | 09 · §1.4 párrafo "Alcance (honesto)" | añadido en la corrección de la revisión |
| v1 11 · §1 ajuste in-sample: parámetros por régimen | 09 · §4.2.4 ajuste de muestra completa opcional (`AJUSTE_D11_COMPLETO`, ILUSTRATIVO) | desactivado por defecto por coste |
| v1 11 · §2 verificación de causalidad del posterior filtrado | 09 · §1.4 (el posterior de Hamilton es causal por construcción) y §4.2.1 | no se re-ejecuta (fila en `registro_eliminaciones.md`) |
| v1 11 · §3 walk-forward causal | 09 · §3 ejecución y carga de la caché del benchmark | ya no se ejecuta en el notebook |
| v1 11 · §5 S&P 500 coloreado por régimen (OOS) | 09 · §4.2.1 figura `D11_<pista>_estados_oos` | — |
| v1 11 · §9 tabla fold a fold sobre las ventanas conocidas | 09 · §4.2.2–4.2.3 diagnóstico por fold desde el panel OOS | — |
| v1 11 · §10 volcado de métricas | 09 · §4.2.6 métricas y ranking ADR-003 | sustituido por el benchmark; CSV v1 leído en §6 |
| v1 11 · §9 tabla fold a fold (figura `render_table_figure`) | 09 · §4.2.2 (tabla `display` de los folds que solapan crisis) | misma información en tabla interactiva en lugar de figura |
| `capa1_exploracion/notebooks/07_changepoint_online.ipynb` · cabecera, online vs offline, hipótesis CP2 | `10_familia_F6_changepoint.ipynb` · §0 resumen y §1.2/§1.7 teoría | reescrito en v2 (no `git mv`: el notebook v1 se retira con `capa1_exploracion/` y queda íntegro en el tag `capa1-final`) |
| 07 v1 · §1 ajuste in-sample y verificación de etiquetado | 10 · §1.7 (polaridad vía núcleo) y §4.2 (ajuste ilustrativo sobre la muestra completa, marcado NO causal) | sobre los paneles v2 de las pistas A y B |
| 07 v1 · §2 verificación de causalidad del CUSUM | 10 · §1.7 (patrón *burn-in*) y §6 (hallazgo v1) | el test ya no se re-ejecuta en el notebook; causalidad garantizada por el protocolo del benchmark |
| 07 v1 · §3/§7 oráculo PELT y retardo causal | 10 · §4.2 "Oráculo PELT" | opcional: solo si `ruptures` está instalado (extra `[changepoint]`) |
| 07 v1 · §4 walk-forward causal | 10 · §3 (carga de la caché del benchmark) | ya no se ejecuta en el notebook: `python -m regimenes.benchmark --detector D07` |
| 07 v1 · §5/§6 estadístico, change-points y mecanismo de acumuladores C± | 10 · §4.2 figura `D07_<pista>_cusum_acumuladores` | |
| 07 v1 · §8 S&P 500 por régimen; §14 timeline y P(crisis) | 10 · §4.1 figura `D07_<pista>_estados_oos` | |
| 07 v1 · §9/§10 lead/lag y retardos de confirmación; §11 cobertura crisis/trampas | 10 · §4.3 tabla por crisis + figura `D07_<pista>_cobertura_crisis` | ventanas `[pico, suelo]` v2 de `configs/benchmark_spec.yaml` |
| 07 v1 · §12/§13 coste gaussiano vs robusto | 10 · §5 ablación con el protocolo de `run_one` + puesto hipotético en el ranking ADR-003 | |
| 07 v1 · §15 volcado de métricas | — | sustituido por `results/benchmark/metrics/*.csv`; el CSV v1 se lee en §6 desde `docs/historia/capa1/resultados/` |
| 07 v1 · §16 conclusión CP2 | 10 · §6 (hallazgos v1) y §7 (conclusión con cifras generadas) | |
| `capa1_exploracion/notebooks/12_deep_ae_regime.ipynb` · cabecera y encuadre CP2 | `11_familia_F7_deep.ipynb` · §0 resumen y §1 teoría | reescrito en v2 (el notebook v1 queda en el tag `capa1-final`) |
| 12 v1 · §1/§1b ajuste in-sample, latente y error de reconstrucción | 11 · §4.2 figuras `D12_<pista>_latente_ae_vs_pca` y `D12_<pista>_error_reconstruccion` | ajuste ilustrativo NO causal |
| 12 v1 · §1c curva de pérdida train/val | 11 · §4.2 figura `D12_curva_perdida` | |
| 12 v1 · §1d error de reconstrucción por régimen | 11 · §4.2 (dentro/fuera de ventanas de crisis del ground-truth) | se compara contra el ground-truth, no contra los estados del propio modelo |
| 12 v1 · §2 verificación de causalidad | 11 · §1.7 y §6 | no se re-ejecuta |
| 12 v1 · §3/§4 walk-forward AE y baseline PCA | 11 · §3 (caché del benchmark para D12) y §5 (PCA→GMM con el protocolo de `run_one`) | |
| 12 v1 · §4b/§5 latente vs PCA y contraste ablativo | 11 · §4.2 y §5 (+ D03 como referencia sin reductor) | |
| 12 v1 · §6/§7 S&P 500 por régimen, timelines y duraciones | 11 · §4.1, §4.2 "Parpadeo OOS" y §5 | |
| 12 v1 · §8 volcado de métricas | — | sustituido por el benchmark; CSV v1 leído en §6 |
| 12 v1 · §9 conclusión | 11 · §6 y §7 | |
| v1 `capa1_exploracion/notebooks/01_rule_vix_threshold.ipynb` (archivo) | `05_familia_F1_reglas.ipynb` | contenido rescatado sobre datos v2; el notebook v1 no se mueve: se retira con `capa1_exploracion/` y queda íntegro en el tag `capa1-final` |
| v1 01 · cabecera (baseline, histéresis, dwell, política de ventana, hipótesis CP2) + índice | 05 · §0 resumen, §1.2 autómata, §1.3 D01, §6 y §6.1 (hipótesis CP2 literal) | la variante de la pista A (vol realizada) se declara en §1.3 y §2 |
| v1 01 · §1 feature causal VIX + verificación de truncado | 05 · §2 (features del registro) | la causalidad de las features la garantiza `03_preprocesado`; no se repite el test (fila en `registro_eliminaciones.md`) |
| v1 01 · §2 umbrales τ_in/τ_out (ajuste de inspección) | 05 · §4.1 diagnóstico (umbrales de ajuste único, ILUSTRATIVO) | — |
| v1 01 · §3–§4 walk-forward y fila de métricas | 05 · §3 (caché del benchmark) y §4.1 (cobertura + ranking ADR-003) | ya no se ejecuta en el notebook |
| v1 01 · §5 S&P 500 por régimen, §6 serie + timeline, §8 panel de histéresis | 05 · §4.1 figura `D01_<pista>_estados_oos` | el zoom GFC se sustituye por la serie OOS completa con la banda muerta |
| v1 01 · §7 distribución del VIX por régimen | 05 · §4.4 separación de la señal por régimen (3 reglas) | — |
| v1 01 · §9 duración de rachas | 05 · §5 tabla de persistencia (switching, duración, episodios de crisis) | sin histograma propio (fila en `registro_eliminaciones.md`) |
| v1 01 · §10–§11 verificación y tabla de cobertura por ventana | 05 · §4.1 figura/tabla de cobertura por crisis (IC bootstrap, detección por evento, lead/lag) | ventanas v2 = catálogo de la pista |
| v1 01 · §12 conclusión CP2 | 05 · §6 (hallazgos: texto + celda con cifras v1 desde `docs/historia/capa1/resultados/metrics_master.csv`) y §6.1 (hipótesis, veredicto v1 «cumplida en lo esencial» y lectura v2 en celda) | cifras ya no escritas a mano |
| v1 `capa1_exploracion/notebooks/02_rule_composite_riskoff.ipynb` (archivo) | `05_familia_F1_reglas.ipynb` | contenido rescatado sobre datos v2 |
| v1 02 · cabecera (por qué un voto, señales y orientación, ventana, hipótesis CP2) + índice | 05 · §0, §1.4, §6 y §6.1 (hipótesis CP2 literal, partes (a) y (b)) | señales v2 impresas desde el registro (§2) |
| v1 02 · §1 señales del voto + retorno del S&P 500 (`market_returns`) | 05 · §2 configuración (señales de D02 desde el registro) y §4.2 figura de estados sobre el S&P 500 | el S&P 500 lo aporta `inf.cargar_contexto_pistas` |
| v1 02 · §2 dirección de estrés empírica por ventana (grieta de la curva) y §9 descomposición por señal | 05 · §4.2 heatmap `D02_contribucion_por_senal` (z orientada media por crisis) + señales que restan | ILUSTRATIVO (μ/σ de ajuste único) |
| v1 02 · §3 score compuesto y problema de calibración | 05 · §1.4 (teoría) y §4.2 diagnóstico (score + umbrales) | — |
| v1 02 · §4–§6 walk-forward, métricas, S&P 500 por régimen, §7 score + timeline | 05 · §3 y §4.2 figura `D02_<pista>_estados_oos` | — |
| v1 02 · §8 distribución del score por régimen | 05 · §4.4 | — |
| v1 02 · §10 espacio PCA de las 4 señales | — | no se rescata (redundante con §4.4; tag `capa1-final`) (fila en `registro_eliminaciones.md`) |
| v1 02 · §11 verificación, §12 comparación D1 vs D2 y conclusión | 05 · §4.2 cobertura, §5 comparación intra-familia, §6 contraste v1→v2 generado en celda y §6.1 (veredicto CP2 v1 y lectura v2, incluidas las señales que restan al voto) | — |
| v1 `capa1_exploracion/notebooks/10_turbulence_mahalanobis.ipynb` (archivo) | `05_familia_F1_reglas.ipynb` | contenido rescatado sobre datos v2 |
| v1 10 · cabecera (ecuación, hipótesis CP2 sobre 2013, ventana) + índice | 05 · §0, §1.5, §6 y §6.1 (hipótesis CP2 literal) | — |
| v1 10 · §1 ajuste in-sample, §4 serie de turbulencia con umbral | 05 · §4.3 diagnóstico (d_t expanding causal, umbrales ILUSTRATIVOS) | — |
| v1 10 · §2 verificación de causalidad de d_t | 05 · §1.5 (la construcción expanding se explica; no se re-ejecuta) (fila en `registro_eliminaciones.md`) | — |
| v1 10 · §3 walk-forward, §5 S&P 500 por régimen, §6 eventos, §10–§11 timeline y métricas | 05 · §3, §4.3 figuras de estados y cobertura, §5 persistencia | — |
| v1 10 · §7–§7b estructura de covarianza por régimen y por qué 2013 se escapa | 05 · §6 (hallazgo v1 en texto + activación en `taper_2013` v1 vs v2 en celda) | los heatmaps Σ/Σ⁻¹ no se rescatan (tag `capa1-final`) (fila en `registro_eliminaciones.md`) |
| v1 10 · §8 turbulencia vs volatilidad realizada | 05 · §4.3 figura `D10_turbulencia_vs_volatilidad` (Spearman por pista) | — |
| v1 10 · §9 distribución de d_t por régimen | 05 · §4.4 | — |
| v1 10 · §12 conclusión CP2 | 05 · §6, §6.1 (veredicto v1 «solo en parte; 2013 no se sostiene» y lectura v2 en celda) y §7 | — |
| v1 `capa1_exploracion/notebooks/03_clustering_gmm.ipynb` (archivo) | `06_familia_F2_clustering.ipynb` | contenido rescatado sobre datos v2 |
| v1 03 · cabecera (GMM full, baseline no temporal, hipótesis CP2, ventana) + índice | 06 · §0, §1.1–1.2, §6 y §6.1 (hipótesis CP2 literal) | — |
| v1 03 · §1 selección de k por BIC y §7 paisaje BIC | 06 · §1.2 (fórmula de BIC y nº de parámetros) + celda de §2 con el nº de parámetros por pista | K fijado a priori en el registro; no se repite el barrido de BIC (fila en `registro_eliminaciones.md`) |
| v1 03 · §2 sanidad económica (retorno por estado) | 06 · §4.3 perfil de estados (retorno y vol anualizados por estado, ILUSTRATIVO) | — |
| v1 03 · §3–§4 walk-forward y tabla de métricas | 06 · §3 y §4.1 cobertura + ranking | — |
| v1 03 · §5 espacio PCA por componente y §6 correlación por componente | 06 · §4.3 heatmap de medias estandarizadas por estado | ni la PCA 2D ni los heatmaps de correlación por componente se rescatan (tag `capa1-final`) (fila en `registro_eliminaciones.md`) |
| v1 03 · §7 (derecha) confianza de las asignaciones | 06 · §4.1 días-frontera (0.2 < P(crisis) < 0.8) desde el panel OOS | — |
| v1 03 · §8 flickering, §9 S&P 500 por régimen, §10 timeline + P(crisis) | 06 · §4.1 figura `D03_<pista>_estados_oos` y §5 histograma de duraciones | — |
| v1 03 · §11 verificación y §12 conclusión | 06 · §4.1 cobertura, §6, §6.1 (veredicto CP2 v1 y lectura v2 en celda) y §7 | — |
| v1 `capa1_exploracion/notebooks/09_jump_model.ipynb` (archivo) | `06_familia_F2_clustering.ipynb` | contenido rescatado sobre datos v2 |
| v1 09 · cabecera (objetivo SJM, rival de D3, hipótesis D9, vía `jumpmodels`, escalado) + índice | 06 · §0, §1.3, §6 y §6.1 (hipótesis D9 literal; las cifras de D3 que citaba se generan en celda) | la justificación del `StandardScaler` pasa a 06 §1.3 en forma cualitativa; la cifra v1 (3 de 15 features con escala ~0.1–0.3) queda en el tag (fila en `registro_eliminaciones.md`) |
| v1 09 · §1 ajuste in-sample y orientación, §2 espacio de features por estado | 06 · §4.3 perfil de estados (ILUSTRATIVO) | — |
| v1 09 · §3 barrido de λ in-sample | 06 · §4.2 celda `BARRIDO_LAMBDA` (opcional, ILUSTRATIVO) | — |
| v1 09 · §4 verificación de causalidad de `predict_online` | 06 · §1.3 (explicación) | no se re-ejecuta (fila en `registro_eliminaciones.md`) |
| v1 09 · §5 walk-forward, §9 S&P 500, §10 cobertura, §11 timeline, §12 métricas | 06 · §3, §4.2 figuras de estados (diagnóstico = edad del régimen) y cobertura | — |
| v1 09 · §6–§8 persistencia y trade-off D9 vs D3, duración de episodios | 06 · §5 (histogramas de duración, tabla `PERSIST`, lectura generada) | — |
| v1 09 · §13 conclusión mejor-para-qué | 06 · §6, §6.1 (veredicto CP2 v1, reparto de usos D9/D3 y lectura v2 en celda) y §7 | — |
| v1 `capa1_exploracion/notebooks/00_eda.ipynb` (archivo) | `docs/historia/capa1/README.md` §3 (resumen) + `docs/historia/capa1/memoria/01_data_and_eda.md` (memoria completa) | no se rescata como notebook: el EDA vigente es `01_eda.ipynb` sobre los datos v2 (`docs/datos/EDA_v2.md`); el `.ipynb` v1 ejecutado se retira con `capa1_exploracion/` y queda íntegro en el tag `capa1-final` |
| v1 00 · §1 cobertura y fechas de inicio, §2 timeline de disponibilidad, §3 procedencia (yfinance + fallbacks), §4 periodos faltantes | historia/capa1 README §3 (bullets «Datos» y «Ventana común») + `docs/historia/capa1/datos_v1/{provenance.json,coverage_report.csv}` | las figuras de §2 y §4 no se rehacen (fila en `registro_eliminaciones.md`) |
| v1 00 · §5 estadísticos de retornos, §6 colas gordas, §7 S&P 500 crisis vs calma, §8 correlaciones | historia/capa1 README §3 (bullet «Colas gordas») + memoria `01_data_and_eda.md` e informe v1 (figuras `eda_fat_tails`, `eda_corr`) | cifras v1 históricas, citadas de la memoria v1 (no recalculadas) |
| v1 00 · §9 S&P 500, drawdown y ventanas, §10 suelos de drawdown (`DRAWDOWN_TROUGHS`) | historia/capa1 README §3 (bullet «Patrón oro») + memoria `01_data_and_eda.md` | en v2 las ventanas `[pico, suelo]` salen de `configs/benchmark_spec.yaml` |
| v1 00 · §11 correlación rolling S&P 500 / Treasuries (Gulko 2002) | historia/capa1 README §3 (bullet «Correlación rolling») + memoria `01_data_and_eda.md` (figura `eda_rolling_corr`) | en v2 es la feature `corr_spx_bond` de la pista B |
| v1 00 · §12 features causales + verificación de no look-ahead, §13 panel-resumen (15 small multiples), §14 tabla por feature | historia/capa1 README §3 (bullet «15 features causales») + memoria `01_data_and_eda.md` | la causalidad de las features v2 la verifica `03_preprocesado`; el panel y la tabla por feature v1 no se rehacen (fila en `registro_eliminaciones.md`) |
| v1 `capa1_exploracion/notebooks/13_comparison.ipynb` (archivo) | `docs/historia/capa1/README.md` §4 (resumen) + `docs/historia/capa1/memoria/99_conclusions.md` y `memoria/pdf_src/` (hallazgos, tablas y figuras del informe v1) | no se rescata como notebook: la comparativa vigente es `12_comparativa.ipynb` (juez ADR-003); el `.ipynb` v1 ejecutado se retira con `capa1_exploracion/` y queda íntegro en el tag `capa1-final` |
| v1 13 · cabecera (tesis de la síntesis, qué hace, principios no negociables) | historia/capa1 README §4 (bullets «Tesis» y «Equidad de ventana») | — |
| v1 13 · §1 métrica de estrés agregado, §2–§2b recompute de los multi-estado (D3, D8, D12) y *sanity check* ±0.01 | historia/capa1 README §4 (bullet «Estrés agregado») | el recompute no se rehace; en v2 el juez es ADR-003 (fila en `registro_eliminaciones.md`) |
| v1 13 · §3 tabla maestra, §4 ranking por eje, §5 scorecard, §6 radar, §7 podio por eje, §8 planos de cruce | historia/capa1 README §4 (bullet «Veredictos por eje») + `docs/historia/capa1/resultados/metrics_master.csv` + informe v1 (`synth_scorecard`, `synth_coverage_ci`) | radar, podio y planos de cruce no se rehacen (fila en `registro_eliminaciones.md`); el ranking vigente es el de 12 (ADR-003) |
| v1 13 · §9 resumen estructurado (recomendación v1) | historia/capa1 README §4 (bullet «Recomendación v1») + memoria `99_conclusions.md` | — |
| 05–11 · celdas de utilidades repetidas (carga de caché con `run_benchmark(cache_only=True)`, lectura de paneles OOS con `_panel_path`, lectura y verificación de `ranking*.csv`, precio del S&P 500 con `_sp500_path`, `franjas`/`franjas_eventos`/`sombrear_crisis`, `panel_estados`/`pintar_estados`/`figura_estados`/`figura_detector`, `tabla_cobertura`/`tabla_crisis`/`tabla_trampas`, `figura_cobertura`, `retardo_confirmacion`, `coincidencia` (Jaccard), tabla de configuración, plano recall × precisión, heatmaps anotados) | `src/regimenes/informes.py` (`cargar_resultados_familia`, `cargar_contexto_pistas`, `tabla_configuracion`, `tabla_cobertura`, `tabla_trampas`, `matriz_jaccard`, `retardo_confirmacion`, `figura_estados_oos`, `figura_cobertura`, `dibujar_plano_ranking`, `dibujar_heatmap`, `guardar_figura`); tests en `tests/informes/test_informes.py` | homogeneización de plantilla (optimizador): los notebooks conservan envoltorios finos con los nombres locales usados aguas abajo |
| 05–11 · §2 celda de imports y rutas (cada notebook con sus variantes: `IDS`/`DETECTORES`, `FAMILIA` como título o carpeta, `BENCH_DIR`/`PANEL_DIR`) | 05–11 · §2 primera celda de código, idéntica salvo las constantes de familia | variables comunes: `FAMILIA` (carpeta), `NOMBRE_FAMILIA`, `DETECTORES`, `PISTAS`, `OUTPUT_DIR`, `FIG_DIR`, `SPECS`, `MIN_RUN`, `BETA`, `rel`, `guardar`; en 07–09 la celda pasa de final de §1 al inicio de §2 |
| 05–11 · §3 celdas de ejecución, ranking y datos de apoyo | 05–11 · §3 celda `EJECUTAR`/`FORZAR`/`COMANDO` + `RES = inf.cargar_resultados_familia(...)` y celda `CTX = inf.cargar_contexto_pistas(...)` | nombres comunes `METRICAS` (índice pista,id), `PANELES`, `RANKING`, `RANK`, `N_PISTA`, `DATOS`, `INDICE`, `SP500`, `CRISIS`, `TRAMPAS`; 05/06: `PANELES_OOS`→`PANELES`, `SPX`→`SP500`; 08–11: `PANEL_PISTA`→`DATOS`; 10/11: `BENCH_DIR`→`OUTPUT_DIR`, ranking siempre desde `ranking*.csv` verificado (antes se recalculaba si la matriz 2×12 estaba vigente) |
| 08/09 · §3.1 "Datos de apoyo, ranking ADR-003 y utilidades de figura" | 08/09 · §3.1 "Datos de apoyo y utilidades de figura" | la verificación del ranking pasa a `cargar_resultados_familia` (falla si no es coherente; antes solo imprimía `RANKING_COHERENTE`) |
| 09 · `## 4.1 D06` / `## 4.2 D11` (sin `## 4.`) | 09 · `## 4. Detector a detector` + `### 4.1 D06` / `### 4.2 D11` + `#### 4.x.y` | solo niveles de encabezado |
| 05–11 · nombres de figura | `<Dxx>_<pista>_estados_oos.png`, `<Dxx>_<pista>_cobertura_crisis.png`, `<Fk>_comparativa_<tema>.png` | antes: `Dxx_<P>_estados_diagnostico`, `Dxx_cobertura_por_crisis` (dos pistas en una figura, 05/06), `Dxx_<P>_cobertura` (08/09), `comparativa_*`/`perfil_de_estados`/`separacion_senal_por_regimen` sin prefijo (05/06), `F3_comparacion_intra_familia`, `F3_<P>_p_crisis_D04_vs_D08`, `D05_vecindad_ranking`, `F5_cobertura_D06_vs_D11`, `D07_robusto_vs_gaussiano`, `D07_contraste_coste`, `D12_ae_vs_pca_estados`, `D12_contraste_ablativo` |
| 05–11 · interruptores opcionales | `EJECUTAR`, `FORZAR`, `DIAGNOSTICO_ILUSTRATIVO`, `EXPLORAR_K`, `EJECUTAR_ABLACION`, `FORZAR_ABLACION` | 08/09 `AJUSTE_ILUSTRATIVO`→`DIAGNOSTICO_ILUSTRATIVO`; 08 `COMPARAR_K`→`EXPLORAR_K`; 10/11 `ABLACION`→`EJECUTAR_ABLACION`, `RECALCULAR`→`FORZAR_ABLACION` (mismos valores por defecto) |
| 07/08/09 · conclusiones con `print` | 07–11 · §7 conclusión con `display(Markdown(...))` + celda final común con la lista de figuras | 08/09: la lista de figuras y `RANKING_COHERENTE` salen de la celda de conclusión a la celda final común |
| 04 · §3 viñeta «Crisis realmente evaluadas» (cifras escritas a mano) | 04 · §3 celda nueva `ventanas` (crisis fuera / parciales y días OOS por pista, mismo cálculo que 12 §2.1) | corrección de texto (fila en `registro_eliminaciones.md`) |
| 07 · veredictos v1 del CHECKPOINT 2 (§6, sin veredicto v2) | 07 · §6.1 nueva (markdown + celda de veredicto v2 por hipótesis de D04 y D08) | usa `PUENTE_D04` (guardado en §4.3) y `SEVERIDAD` (§4.6) |
| 10 · §6 «Detección temprana» v1 y §7 lectura del lead/lag | 10 · §4.3 (lectura del lead/lag con `inf.clasificar_lead_lag`), §6.1 nueva (veredicto v2 de la hipótesis CP2) y §7 | plantilla homogénea con 05, 06, 08 y 09 |
| 11 · §6 hallazgos v1 (sin veredicto v2) y §7 lectura | 11 · §6.1 nueva (veredicto v2 por afirmación v1) y §7 | usa `PERDIDA_AE` (guardado en §4.2), `AE_IS` y `CONTRASTE` |
| 05–11 · lectura del lead/lag como anticipación | `regimenes.informes.clasificar_lead_lag` / `lookback_lead_lag` (tests en `tests/informes/test_informes.py`) | usado en 06 §6.1 y 10 §4.3/§6.1 |
