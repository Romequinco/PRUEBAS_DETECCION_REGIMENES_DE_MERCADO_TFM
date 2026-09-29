# D5 — `markov_switching_var` (FASE 3, Tanda 2) · Familia F4 (Markov-Switching)

<!-- BEGIN ubicacion_v2 -->
> **Ubicación tras ADR-004.** Código: `src/regimenes/detectores/f4_switching/markov_switching_var.py` (registro: `regimenes.detectores.registry`, id `D05`) · Notebook v2 de familia: [`notebooks/08_familia_F4_markov_switching.ipynb`](../../notebooks/08_familia_F4_markov_switching.ipynb) · Resultados v2: sección [Resultados v2 (ADR-003)](#resultados-v2-adr-003) al final. Esta ficha nació en la Capa 1 (v1, congelada en el tag `capa1-final`); el texto histórico se conserva. Cuando menciona el núcleo (`src/`, `detector_base.py`, `evaluation.py`, `features.py`, `data_loader.py`, `INDEX.md`) se refiere al de la Capa 1: hoy `detector_base.py` → `src/regimenes/detectores/base.py`, `evaluation.py` → `regimenes.evaluacion`, `INDEX.md` → `docs/historia/capa1/memoria/INDEX.md`; `features.py`/`data_loader.py` de v1 quedan en el tag `capa1-final`.
<!-- END ubicacion_v2 -->

> Baseline econométrico **interpretable**. Markov-Switching de varianza sobre el
> retorno del S&P 500 (Hamilton 1989), con probabilidades **filtradas causales**.
> Código: `src/regimenes/detectores/f4_switching/markov_switching_var.py` · Notebook:
> `capa1_exploracion/notebooks/05_markov_switching_var.ipynb` (v1, tag `capa1-final`) (constructor `_build_05.py` (retirado en el commit `6016f42`))
> · Métricas: `docs/historia/capa1/resultados/metrics_05_markov_switching_var.csv`.

## Implementado

**Modelo.** `statsmodels.tsa.regime_switching.MarkovRegression` sobre el retorno
log del S&P 500 (endog), `trend='c'`, `switching_variance=True`, `k_regimes=2`
(baseline interpretable calma/crisis). Los regímenes difieren en media y varianza.
Bibliografía: `hamilton1989, ms_kim1994, ms_guidolin2011, ms_kimnelson1999`
(verificadas en `docs/references.bib`).

**Causalidad — probabilidades FILTRADAS, no smoothed.** `MarkovRegression` da
`filtered_marginal_probabilities` P(S_t|y≤t) (causal) y
`smoothed_marginal_probabilities` P(S_t|y₁..T) (look-ahead). En walk-forward NO se
puede reestimar por bloque, así que `predict_online`/`predict_proba` usan un
**filtrado forward propio** (univariante gaussiano con media y varianza por
régimen + matriz de transición) sobre `train_burn-in + bloque`, devolviendo solo el
bloque — el mismo patrón que D4 (`_hmm_utils`) pero 1-D. Verificado contra
statsmodels: `max|forward-filter propio − filtered statsmodels| = 1.2e-13` (≈0).
Las smoothed solo se usan IN-SAMPLE, marcadas NO causales (comparación en el
notebook).

**Etiquetado económico robusto.** Pasa `market_returns` (retorno log S&P 500) a
`walk_forward` y `evaluate`. **Sin warning de fallback.** Orientación verificada:
crisis (canónico 1) = media **−0.115**, varianza **3.88**; calma (0) = media
**+0.084**, varianza **0.50** → **crisis = ALTA varianza, confirmado y NO
invertido** (a diferencia de D6: aquí los regímenes MS separan también en media, así
que `z(std)−z(mean)` del núcleo orienta sin ambigüedad).

**Coste / ventana.** Modela solo el retorno del S&P 500, disponible desde 1985 →
**ventana LARGA**. `walk_forward(train_size=252*8, step=63, expanding=True)`:
**step trimestral** (no 21) porque con ventana expanding los folds tardíos
reajustan el MS sobre 10 000+ obs (~10 s cada uno) y el EM tarda; con step=63 son
~131 reestimaciones (no ~394). Los regímenes de varianza son persistentes, así que
el refit trimestral es adecuado. El filtrado forward dentro de cada bloque sigue
siendo diario y causal.

## Descubierto

**`ventana_eval` causal: 1993-03-23 → 2026-06-12 (n=8278).** Como D1 y D6 (y a
diferencia del HMM puente D4), el histórico largo permite **evaluar 2008 y 2011
OOS**.

**Cobertura por crisis (OOS, causal):**

| Ventana | Cobertura | Lectura |
|---|---:|---|
| GFC_2008 | **99.3 %** | excelente (2008 sí evaluable OOS) |
| EuroDebt_2011 | 74.1 % | buena |
| COVID_2020 | 96.0 % | excelente |
| Inflation_2022 | 73.7 % | buena |
| TaperTantrum_2013 (trampa) | **3.8 %** | no se dispara (correcto) |
| Selloff_Q4_2018 (trampa) | 81.0 % | se dispara fuerte (la vol equity de Q4-2018 fue real) |

**¿Capta 2013/2018?** 2013 NO (3.8 %, correcto: el taper fue shock de tipos sin
vol equity) y 2018 SÍ (81 %). Mismo perfil que el HMM gaussiano (D4) pero con mejor
cobertura de las crisis grandes evaluables OOS gracias a la ventana larga.

**Persistencia / flickering.** `switching_rate=0.056`, duración media **17.9 días**,
persistencia esperada de la matriz de transición ≈ 84 d (calma) / 27 d (crisis),
`label_stability=0.998`. Mucho más estable que el clustering GMM (D3, switching
0.126) — la dinámica de Markov aporta persistencia, como predecía el estado del
arte. `false_alarm_rate=0.774` (alto, como D6: marca crisis en episodios de vol
reales fuera de las 4 ventanas: 1998 LTCM, 2002, 2010, 2011 US downgrade, 2015-16,
2023 SVB).

**Efecto del look-ahead (filtered vs smoothed, in-sample).** Correlación
filtered/smoothed = 0.896; media |smoothed − filtered| = 0.087. Las smoothed son
más nítidas y "anticipadas" porque miran el futuro — ilustra por qué la evaluación
online DEBE usar filtradas (lo que hace `predict_online`).

**Selección de nº de estados.** Por AIC y BIC el óptimo es **k=3** (BIC 27 247 vs
28 024 de k=2): un tercer régimen de varianza intermedia (varianzas k=3 ≈
[0.32, 1.32, 9.34]) mejora el ajuste. El detector desplegado es **k=2** (baseline
interpretable calma/crisis, comparable con D4); k=3 queda como mejora documentada
(separa "corrección" de "crisis sistémica", en línea con D8). logL/AIC/BIC se
exponen para la comparativa.

## Hipótesis del CHECKPOINT 2 para D5 — veredicto

> *"Baseline econométrico interpretable; capta calma/estrés; punto ciego en crisis
> rápidas; univariante → no ve correlación cross-asset; gaussiano insuficiente para
> colas."*

**Se cumple.** (1) Interpretable: medias/varianzas/persistencia por régimen
legibles, crisis = alta varianza. (2) Capta calma/estrés y las 4 crisis evaluables
OOS (incl. 2008 al 99 %). (3) Punto ciego en crisis rápidas confirmado: 2013 no se
dispara (3.8 %). (4) Univariante (solo S&P 500) → no usa crédito/curva ni la
correlación cross-asset; es su límite estructural frente a un detector multivariante.
(5) Gaussiano por régimen: la mezcla de 2-3 gaussianas mitiga pero no elimina las
colas (kurt 25-40) — coherente con que BIC pida un tercer régimen de varianza muy
alta para capturar los extremos.

## Fricción con el núcleo

Ninguna que requiera cambiar firmas. Dos observaciones:
- **Orientación robusta**: pasar `market_returns` al núcleo bastó; D5 NO sufre la
  inversión de `_economic_state_order` que sí afectó a D6, porque los regímenes MS
  separan en media además de en varianza. (Aun así, ver la nota de D6 sobre la
  fragilidad de `z(std)−z(mean)` cuando las medias por estado son casi iguales:
  afecta a detectores de pura varianza/umbral, no a este.)
- **Coste**: el walk-forward del MS es caro (reestimación por fold); se usó
  `step=63`. Si en el futuro se quiere `step=21` para comparabilidad estricta,
  habría que cachear/acelerar la reestimación o usar ventana rolling de tamaño
  fijo (no se hizo aquí).

<!-- BEGIN resultados_v2 (generado desde git HEAD; no editar a mano) -->
## Resultados v2 (ADR-003)

> Fuente: `results/benchmark/ranking_v2.csv` y `results/benchmark/metrics_master_v2.csv` de la re-ejecución completa del benchmark (24/24) posterior a la unificación ADR-004 (`results/benchmark/manifest.json`, generado el 2026-09-29T12:41 UTC), numéricamente idéntica a la re-ejecución de ADR-003 §4 (ver `docs/revisiones/informe_unificacion.md`). Walk-forward causal, evaluación OOS por pista. Ranking de **detección**: evento detectado = ≥ 3 sesiones OOS consecutivas en crisis dentro de [pico, suelo]; `score_deteccion` = F1 entre precisión diaria (1 − FAR) y recall por evento, con niveles previos (elegible › precisión ≤ azar › parpadeo › degenerado). Ver `docs/decisions/ADR-003-causalidad-calendario-estado-ranking.md`.

**Notebook de familia:** [`notebooks/08_familia_F4_markov_switching.ipynb`](../../notebooks/08_familia_F4_markov_switching.ipynb) (F4 — Markov-Switching) · **Teoría:** [`docs/teoria/F4_markov_switching.md`](../teoria/F4_markov_switching.md).

| Métrica | Pista A | Pista B |
|---|---:|---:|
| Variante evaluada | `markov_switching_var_2s` | `markov_switching_var_2s` |
| Puesto de detección | **5** / 12 | **1** / 12 |
| Nivel de etiqueta | `elegible` | `elegible` |
| `score_deteccion` | 0.546 | 0.647 |
| Recall por evento | 1.000 (17/17) | 1.000 (9/9) |
| Precisión diaria (lift vs base) | 0.375 (×1.94) | 0.478 (×2.77) |
| Cobertura media de crisis | 0.731 | 0.499 |
| Tasa de falsas alarmas (FAR) | 0.625 | 0.522 |
| Activación media en trampas | 0.032 | 0.000 |
| Switching rate | 0.0683 | 0.0434 |
| Duración media de régimen (sesiones) | 14.6 | 22.9 |
| Estabilidad de etiqueta | 0.995 | 0.997 |
| Puesto legacy (`rank_medio`) | 6 (7.0) | 4 (4.6) |

**Configuración v2 y eventos por pista**

- **Pista A** · OOS 1970-05-12 → 2026-05-29 (n = 14133) · train inicial 2016 sesiones, refit cada 63 (expanding) · 1 feature(s): `SP500_ret`.
  - Detectados (17): `bear_1969_70`, `oil_stagflation_1973`, `volcker_1980_82`, `black_monday_1987`, `gulf_war_sl_1990`, `ltcm_russia_1998`, `dotcom_2000_02`, `gfc_2007_09`, `flash_crash_euro1_2010`, `us_downgrade_euro2_2011`, `china_oil_2015_16`, `volmageddon_2018q1`, `fed_tightening_2018q4`, `covid_2020`, `inflation_bear_2022`, `svb_banking_2023`, `tariff_selloff_2025`.
  - No detectados (0): ninguno.
- **Pista B** · OOS 2010-04-12 → 2026-05-29 (n = 4059) · train inicial 756 sesiones, refit cada 63 (expanding) · 1 feature(s): `SP500_ret`.
  - Detectados (9): `flash_crash_euro1_2010`, `us_downgrade_euro2_2011`, `china_oil_2015_16`, `volmageddon_2018q1`, `fed_tightening_2018q4`, `covid_2020`, `inflation_bear_2022`, `svb_banking_2023`, `tariff_selloff_2025`.
  - No detectados (0): ninguno.

Lectura: las cifras de la Capa 1 de las secciones anteriores (ventana 2007-2026, 4 crisis, métricas de cobertura sin criterio de detección) **no son comparables** con esta tabla (pistas A y B de `configs/benchmark_spec.yaml`, 17 crisis evaluables OOS en A / 9 crisis evaluables OOS en B, ranking ADR-003). El análisis gráfico por crisis y la comparación intra-familia están en el notebook `notebooks/08_familia_F4_markov_switching.ipynb`.
<!-- END resultados_v2 -->
