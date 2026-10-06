# Capa 1 (v1) — historia del primer banco de detectores

> **Reencuadre (ADR-005).** La Capa 1 es la **primera vuelta de la Fase 1** (régimen de mercado) del TFM,
> que es un sistema multi-agente RAG ([ADR-005](../../decisions/ADR-005-reencuadre-tfm-multiagente.md)).
> «Capa 1» es el nombre de esta vuelta exploratoria y **no** guarda relación con la «Capa 1» (42 ETFs) ni
> la «Capa 2» (acciones) de la propuesta.
>
> Archivo histórico de la **primera vuelta** de la Fase 1 del TFM: 12 detectores de régimen (7 familias) bajo un
> marco causal común, evaluados sobre un set de datos pequeño (9 series, 15 features). Se **congeló**
> el 2026-07-18 ([ADR-001](../../decisions/ADR-001-rebase-datos.md)); sus detectores e interfaz forman
> hoy parte del paquete `regimenes` ([ADR-004](../../decisions/ADR-004-unificacion.md)). Este README es
> también el registro de **qué se decidió en la Capa 1 y por qué** (§2), para que ADR-001 … ADR-004
> se lean en orden.
>
> Esta carpeta conserva **los artefactos que no se re-ejecutan** (memoria, informe, métricas v1,
> procedencia de datos v1). El código y la teoría que siguen vivos están en el paquete `regimenes`,
> `docs/teoria/` y `docs/detectores/` (tabla de §7); el original completo, en el tag `capa1-final` (§8).

---

## 1. Qué fue la Capa 1

Un **banco de pruebas comparativo** cuyo objetivo no era "el mejor detector" sino **"el mejor para
qué"**: misma interfaz `RegimeDetector`, mismo protocolo walk-forward causal y mismas métricas
(cobertura por crisis, falsas alarmas en las trampas 2013/2018, lead/lag al suelo del drawdown,
switching, duración, estabilidad, BIC) para los 12 detectores.

La propuesta del TFM (MIAX) es un sistema multi-agente RAG con conciencia de régimen
([`../../context/TFM_Proposal_v2.pdf`](../../context/TFM_Proposal_v2.pdf)), cuya pieza de régimen
planteaba un HMM t-Student multi-estado ([ADR-005](../../decisions/ADR-005-reencuadre-tfm-multiagente.md)).
La Capa 1 partía de la
**tarea previa**: un HMM gaussiano de 2 estados in-sample, con z-scores de muestra completa, que
acertaba las crisis grandes (2008: 98.6 %, 2020: 92.3 %) pero se perdía las correcciones rápidas
(2013: 10.9 %, Q4 2018: 20.6 %)
([`../../context/RESUMEN_DETECCION_REGIMENES.md`](../../context/RESUMEN_DETECCION_REGIMENES.md)).
Sus problemas —look-ahead de los z-scores, evaluación in-sample, Viterbi duro sin probabilidades,
supuesto gaussiano y etiquetado frágil— no permitían saber si el HMM era una buena elección o solo
una elección: de ahí el banco.

Fases internas de la Capa 1 (numeración propia, **independiente** de la hoja de ruta v2): 0 estructura
+ interfaz + evaluador · 1 datos + EDA · 2 estado del arte · 3 implementación en 4 tandas · 4 síntesis
comparativa · 5 pulido (cerradas todas; el detalle de cada tanda está en el tag `capa1-final`, §8).

### Los 12 detectores (+ ablación D13)

| ID | Detector | Familia | Clase v1 | Ventana OOS v1 | ¿Vio 2008 OOS? |
|---|---|---|---|---|:---:|
| D1 | `rule_vix_threshold` | F1 reglas/umbrales | baseline | 1998–2026 | sí |
| D2 | `rule_composite_riskoff` | F1 reglas/umbrales | baseline | 2015–2026 | no |
| D3 | `clustering_gmm` (K=3) | F2 clustering | baseline | 2015–2026 | no |
| D4 | `hmm_gaussian_2s` | F3 HMM (puente con la tarea previa) | baseline | 2012–2026 | no |
| D5 | `markov_switching_var` (2 estados) | F4 Markov-Switching | avanzado | 1993–2026 | sí |
| D6 | `garch_t_vol` (GJR-GARCH-t) | F5 GARCH | avanzado | 1993–2026 | sí |
| D7 | `changepoint_online` (CUSUM robusto) | F6 change-point | avanzado | 1993–2026 | sí |
| D8 | `hmm_tstudent` (K=4) | F3 HMM avanzado | avanzado | 2012–2026 | no |
| D9 | `jump_model` (λ=50) | F2↔F3 jump model | avanzado | 2015–2026 | no |
| D10 | `turbulence_mahalanobis` | F1 multivariante (Kritzman) | avanzado | 1998–2026 | sí |
| D11 | `msgarch_regime` | F5 MS-GARCH | exploratorio-negativo | 1991–2026 | sí |
| D12 | `deep_ae_regime` (AE→GMM) | F7 redes | exploratorio-negativo | 2015–2026 | no |
| D13 | `hsmm_tstudent` | F3 (ablación de D8) | fuera del ranking | = D8 | no |

Fuente: [`memoria/99_conclusions.md`](memoria/99_conclusions.md) y
[`resultados/metrics_master.csv`](resultados/metrics_master.csv). La ablación D8↔D13 está en
[`resultados/ablation_hsmm/`](resultados/ablation_hsmm/README.md): la duración explícita **no** reduce
el switching ni alarga los regímenes → se mantuvo D8 por parsimonia.

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

## 3. Los 5 hallazgos que sobreviven

1. **El look-ahead de los z-scores in-sample compraba suavidad, no acierto.** D4 causal frente a su
   versión in-sample: el switching sube de 0.047 a 0.100 y sigue fallando 2013/2018; el acierto en
   crisis grandes nunca dependió del look-ahead.
2. **La t-Student mejora el ajuste con holgura:** ΔBIC ≈ +10963 de D8 frente a D4 con las mismas 7
   features; ν por estado decreciente [10.2, 7.6, 4.2, 2.4] (crisis = colas más pesadas).
3. **Viterbi-por-bloque era menos causal y menos estable que el filtrado forward:** en D4 el switching
   bajó 0.124 → 0.100 y la duración subió 8.1 → 9.9 días al pasar a filtrado forward.
4. **2013 (taper tantrum) es el punto ciego universal:** 6 detectores independientes que lo tenían OOS
   lo marcan entre 0 % y ~12 % → *la taxonomía de features importa* (todas eran de vol/equity).
5. **La complejidad extra no se paga con ~4 crisis:** D11 MS-GARCH degenera (cobertura GFC 0 %) y D12
   AE empeora a su PCA (switching 0.287) → parsimonia validada.

(Las cifras son las de la memoria v1; detalle y evidencia en
[`memoria/99_conclusions.md`](memoria/99_conclusions.md) §2 y en el
[informe](informe/informe_capa1.pdf).)

## 4. Resumen del EDA v1 (tag `capa1-final`: `capa1_exploracion/notebooks/00_eda.ipynb`)

- **Datos:** 9 series descargadas **sin imputar** (S&P 500, VIX, MOVE, TLT, IEF, HYG, GLD, DXY y la
  pendiente 10Y−3M). FRED era inaccesible en aquel entorno → fallbacks documentados: DXY por
  `DX-Y.NYB`, curva por el proxy `^TNX − ^IRX` y HY OAS omitido (crédito vía HYG y spread HYG−IEF).
  Procedencia en [`datos_v1/provenance.json`](datos_v1/provenance.json) y
  [`datos_v1/coverage_report.csv`](datos_v1/coverage_report.csv).
- **Ventana común 2007-04-11 → 2026-06**, gobernada por HYG (la serie que arranca más tarde): se
  prefirió perder la DotCom antes que imputar hacia atrás.
- **Colas gordas:** kurtosis de exceso del S&P 500 25.6 y de HYG 39.6 → motiva la t-Student.
  Crisis frente a calma: la distribución se ensancha y se sesga a pérdidas (partición de referencia,
  no un detector).
- **Patrón oro:** 4 crisis sistémicas (2008, 2011, 2020, 2022) y 2 trampas (2013, 2018); suelos de
  drawdown calculados sobre la serie real (`DRAWDOWN_TROUGHS`). Con **n≈4** crisis se puede describir
  comportamiento, no hacer tests de significancia.
- **Correlación rolling S&P 500/Treasuries** cambia de signo entre regímenes (Gulko 2002) → feature
  `corr_spx_bond`.
- **15 features causales** (z-scores expanding) verificadas con `assert_causal` (`max_abs_diff = 0`).

Memoria completa: [`memoria/01_data_and_eda.md`](memoria/01_data_and_eda.md). El EDA v2 que lo
sustituye (166 series, 22 crisis) está en [`../../datos/EDA_v2.md`](../../datos/EDA_v2.md).

## 5. Resumen de la comparativa v1 (tag `capa1-final`: `capa1_exploracion/notebooks/13_comparison.ipynb`)

- **Tesis:** no hay detector dominante; 4 familias se reparten 6 ejes (cobertura sistémica,
  especificidad, persistencia, lead/lag, BIC, coste). Es un resultado, no un fracaso.
- **Equidad de ventana:** la cobertura se compara **por grupo de ventana** (vio 2008 OOS o no); nunca se
  penaliza a un detector por lo que no pudo ver.
- **Estrés agregado:** para los multi-estado (D3, D8, D12) se define estrés = unión de los dos estados
  más severos, para compararlos con los binarios; el notebook recomputa esos tres y comprueba (±0.01)
  que la crisis estricta coincide con el master.
- **Veredictos por eje:** cobertura sistémica en ventana larga D5 0.98 ≈ D6 0.97 ≈ D1 0.92 (la vol
  manda; la sofisticación apenas bate a la regla VIX); especificidad, persistencia, lead/lag y coste:
  D7 (CUSUM); BIC: D8.
- **Recomendación v1** (superada por el benchmark v2, donde D8 queda 9.º en ambas pistas, ADR-003, y por
  ADR-005): núcleo HMM t-Student multi-estado (respaldo *consistente con* la propuesta, no
  superioridad OOS estricta) + change-point tipo D7 como alerta temprana + D1/D5/D6 como control.
  Es el origen de las fusiones D7+D8 y D2+D6 de v2 (notebooks `13`–`14`).

Figuras y tablas del informe: [`informe/informe_capa1.pdf`](informe/informe_capa1.pdf). Sus cifras se
verificaron contra la tabla maestra (20 cifras muestreadas, 0 discrepancias; las 4 citas centrales,
correctas) y las cautelas exigidas (consistente con, no «confirma»; n≈4 crisis sin tests; BIC in-sample;
lead/lag censurado a 252 días; cobertura por ventana; D11/D12 exploratorio-negativos) están presentes.

**Tabla maestra.** `resultados/metrics_master.csv` (43 columnas) es la única canónica: el superset del
antiguo `metrics_master_final.csv` (que aportaba `clase`, `coste`, `vio_2008_oos` y las columnas de
estrés) y de la primera tabla (que aportaba `silhouette` y los IC de cobertura). Los dos originales,
las figuras `fase4_*` y la revisión del PDF siguen en el tag `capa1-final`.

## 6. Qué falló y qué vino después

- **Incomparabilidad 1:1:** cada detector construía su propia matriz de features (1, 4, 7 o 15) y su
  propia ventana OOS; unos se juzgaban con 3,3× más datos y el doble de crisis que otros (4 crisis y
  ~8000 días frente a 2 y ~2600).
- **Potencia nula:** ~4 crisis en la ventana común; ningún intervalo de confianza separaba detectores.
- **Taxonomía de features pobre:** casi todo vol/equity; sin crédito ni curva reales → 2013 invisible.
- **Sin banco congelado:** los datos no eran una variable controlada, eran parte del detector.

Consecuencias:

- **[ADR-001](../../decisions/ADR-001-rebase-datos.md) (2026-07-18):** se congeló la Capa 1 y se
  re-basó la capa de datos (dos pistas, 166 series, `configs/benchmark_spec.yaml`). El **juez**
  (walk-forward + métricas) y la **interfaz** de la Capa 1 se reutilizaron en v2, y los 12 detectores
  se re-evaluaron sobre el banco congelado ([ADR-002](../../decisions/ADR-002-ajuste-ventanas.md),
  [ADR-003](../../decisions/ADR-003-causalidad-calendario-estado-ranking.md)).
- **[ADR-004](../../decisions/ADR-004-unificacion.md) (2026-09-29):** los detectores y la interfaz de
  la Capa 1 forman parte del paquete `regimenes`; cada familia se cuenta en su notebook (05–11) y esta
  carpeta queda como archivo.

## 7. Correspondencia de rutas (Capa 1 → hoy)

| Ruta en la Capa 1 | Hoy |
|---|---|
| `capa1_exploracion/detectors/*.py` (12 + `hsmm_tstudent` + utilidades HMM) | `src/regimenes/detectores/f1_reglas/` … `f7_deep/` |
| `capa1_exploracion/src/detector_base.py` | `src/regimenes/detectores/base.py` (era idéntica a la de v2) |
| `capa1_exploracion/src/{data_loader,evaluation,features,viz}.py` (marco v1) | no forman parte del paquete; en el tag `capa1-final` (§8). El juez vigente es `regimenes.evaluacion` |
| `capa1_exploracion/memory/00_state_of_the_art.md` | [`docs/teoria/00_estado_del_arte.md`](../../teoria/00_estado_del_arte.md) |
| `capa1_exploracion/memory/sota/0k_*.md` y `.bib` | `docs/teoria/Fk_*.md` y `.bib` (p. ej. [`F3_hmm.md`](../../teoria/F3_hmm.md)) |
| `capa1_exploracion/memory/detectors/NN_*.md` | `docs/detectores/DNN_*.md` (p. ej. [`D08_hmm_tstudent.md`](../../detectores/D08_hmm_tstudent.md)) |
| `capa1_exploracion/memory/{01_data_and_eda,99_conclusions}.md` | [`memoria/`](memoria/99_conclusions.md) (aquí) |
| `capa1_exploracion/memory/{INDEX.md,pdf_src/}`, `report/_revision_pdf.md`, `results/_archive/` | documentos de proceso: solo en el tag `capa1-final` |
| `capa1_exploracion/report/` (tex, pdf, bib) | [`informe/`](informe/informe_capa1.pdf) (aquí) |
| `capa1_exploracion/results/*.csv`, `ablation_hsmm/` | [`resultados/`](resultados/metrics_master.csv) (aquí) |
| `capa1_exploracion/data/raw/{provenance.json,coverage_report.csv}` | [`datos_v1/`](datos_v1/provenance.json) (aquí) |
| `capa1_exploracion/notebooks/01..12_<detector>.ipynb` | su teoría y hallazgos se rescatan en los notebooks de familia `notebooks/05_familia_F1_reglas` … `11_familia_F7_deep`; los `.ipynb` v1 ejecutados, en el tag `capa1-final` |
| `capa1_exploracion/notebooks/A1_hsmm_ablation.ipynb` | D13 como ablación en `notebooks/07_familia_F3_hmm`; original en el tag |
| `capa1_exploracion/notebooks/00_eda.ipynb`, `13_comparison.ipynb` | resumidos en §4–§5 de este README; originales en el tag |
| bibliografía `report/references.bib` | fusionada en la bibliografía única [`docs/references.bib`](../../references.bib) (se conserva aquí para que el `.tex` compile) |

**Rutas dentro de estos documentos.** La memoria y el informe son históricos y **no se reescriben**:
sus rutas (`docs/memory/…`, `results/…`, `notebooks/…`, `report/…`, `detectors/…`) son relativas a la
antigua raíz de la Capa 1 (`capa1_exploracion/`). Tradúcelas con la tabla anterior.

## 8. Recuperar la Capa 1 original (tag `capa1-final`)

El tag `capa1-final` apunta al último commit con la Capa 1 completa
(notebooks v1 ejecutados con sus salidas, marco v1 y detectores en su sitio original):

```bash
git show capa1-final:capa1_exploracion/src/evaluation.py        # un archivo concreto
git show capa1-final:capa1_exploracion/notebooks/13_comparison.ipynb > /tmp/13_comparison.ipynb
git worktree add ../capa1-v1 capa1-final                         # árbol completo, sin tocar tu rama
```

Código v1 que solo vive en el tag: `data_loader.py`, `evaluation.py`, `features.py`, `viz.py`
(marco v1) y los 15 notebooks v1. El código v2 con la Capa 1 como árbol aparte está en el tag
`v2-pre-unificacion`; la historia anterior al re-base, en la rama remota `backup-main-pre-datos-v2`.

> Si re-ejecutas un notebook v1 desde un worktree del tag, hazlo **desde dentro de
> `capa1_exploracion/notebooks/`**: descubren su raíz subiendo hasta encontrar `src/`.
