# D5 — `markov_switching_var` (FASE 3, Tanda 2) · Familia F4 (Markov-Switching)

> Baseline econométrico **interpretable**. Markov-Switching de varianza sobre el
> retorno del S&P 500 (Hamilton 1989), con probabilidades **filtradas causales**.
> Código: `detectors/markov_switching_var.py` · Notebook:
> `notebooks/05_markov_switching_var.ipynb` (constructor `scripts/builders/_build_05.py`)
> · Métricas: `results/metrics_05_markov_switching_var.csv`.

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
