# D6 — `garch_t_vol` (FASE 3, Tanda 2) · Familia F5 (Volatilidad)

> Detector univariante sobre el **retorno log del S&P 500** (desde 1985). Ajusta un
> **GJR-GARCH(1,1)-t**, obtiene la **sigma condicional** y define el régimen
> **umbralizando** esa sigma (percentil del train + histéresis + dwell). 2 estados.
>
> Código: `detectors/garch_t_vol.py` · Notebook: `notebooks/06_garch_t_vol.ipynb`
> (constructor `scripts/builders/_build_06.py`) · Métricas:
> `results/metrics_06_garch_t_vol.csv` · Figuras: `results/d06_*.png`.

## Implementado

**Modelo.** `arch_model(ret×100, mean='Constant', vol='GARCH', p=1, o=1, q=1,
dist='t')` = **GJR-GARCH(1,1)-t** (librería `arch` de Sheppard, instalada con
`pip install arch`). El término `o=1` (Glosten-Jagannathan-Runkle) capta la
**asimetría/leverage**; `dist='t'` capta las **colas gordas** (kurtosis 25–40 del
EDA). Parámetros in-sample (full): `gamma[1]=+0.162` (>0 → leverage real),
`beta[1]=0.89`, `nu≈6.1` (colas claramente no gaussianas). Escala ×100 recomendada
por `arch`. `n_parameters=6` (mu, omega, alpha, gamma, beta, nu); `score`=logL del
ajuste → AIC/BIC comparables.

**De sigma a estado (hipótesis CP2 declarada).** El GARCH **no da estados tipados**,
solo sigma condicional. El régimen se define **umbralizando** sigma: crisis si
`sigma_t > τ_in` (percentil **p80** de la sigma in-sample del train), con
**histéresis** (`τ_out` = p60) y **dwell-time mínimo** (5 días) para no parpadear.
Autómata de 2 estados idéntico en espíritu al de D1 (rule_vix), pero sobre la sigma
GARCH en vez del nivel de VIX. `predict_proba` = **sigmoide monótona** de
`(sigma − τ_in)` (≈0.5 en el umbral), reordenada al orden canónico.

**Sigma CAUSAL en walk-forward (lo importante).** GARCH es causal nativo (sigma_t
depende solo del pasado), pero hay que evitar reestimar con el test. Vía elegida:

1. `fit(train)`: estima por ML y **congela** `res.params`; el umbral τ se fija con
   `res.conditional_volatility` del train (percentil). Guarda el train para burn-in.
2. Para cada bloque de test, **NO reestima**: reconstruye un `arch_model` sobre
   `[burn-in de train anterior al bloque] + bloque`, fija los parámetros con
   `am.fix(self._params)` y lee `conditional_volatility`, quedándose con la parte del
   bloque. La recursión de varianza se propaga hacia delante desde el burn-in, así
   que sigma_t usa solo retornos ≤ t. El único uso de estadística de muestra (el
   *backcast* inicial que arranca la recursión) queda diluido por el burn-in.

**Verificación de causalidad (notebook §2).** Ocultar el futuro NO cambia la sigma
del bloque: `max|sigma_bloque(ver futuro) − sigma_bloque(ocultar futuro)| = 0.0`
sobre 2008. Estrictamente causal.

**Etiquetado económico ROBUSTO (fricción con el núcleo → Arreglo 4, ya en el núcleo).**
D6 fue **el caso que destapó** el problema: la señal es VOLATILIDAD, no retorno. El
criterio ANTIGUO del núcleo (`z(std_ret) − z(mean_ret)`) **invertía** crisis/calma
sobre el histórico largo aunque se pasara `market_returns`: en 1985–2026 el estado de
alta sigma tiene un **retorno medio casi idéntico** al de calma (0.00037 vs 0.00036),
y con K=2 el z-score de la media amplifica esa diferencia nula y cancela la señal
correcta (la std, 0.0172 vs 0.0076). El efecto se veía in-sample (el walk-forward, con
ventanas cortas y la relación vol↔retorno más nítida, no invertía). **Se arregló en el
NÚCLEO (Arreglo 4):** `_economic_state_order` ahora es **vol-primario** (ordena por
banda de volatilidad; el retorno solo desempata entre vols próximas), así que crisis =
ALTA σ de forma determinista. D6 **ya NO sobrescribe** `label_states_economically`:
confía en el núcleo como el resto. Verificado: crisis canónica = σ alta (1.59 vs 0.74),
**sin warning de fallback**, y las métricas de D6 **no cambian** al quitar el override.

**Ventana.** `returns = log(SP500/SP500.shift(1)).dropna()`, índice 1985-01 →
2026-06. Walk-forward expanding, `train_size = 252×8` (~8 años), `step = 21`.
**ventana_eval = 1993-03-23 → 2026-06-12 (n=8278)** → **2008 y 2011 son OOS**, a
diferencia de D4 (atado a HYG desde 2007).

## Descubierto

**Cobertura por crisis (CAUSAL OOS), incluida 2008:**

| Ventana | Cobertura OOS | Lectura |
|---|---:|---|
| GFC_2008 | **100.0 %** | OOS gracias al histórico largo; sigma se dispara y se mantiene |
| EuroDebt_2011 | **74.1 %** | OOS, bien captada |
| COVID_2020 | **94.0 %** | reacción same-day al crash |
| Inflation_2022 | **80.4 %** | bear market de tipos, vol sostenida |
| TaperTantrum_2013 (trampa) | **11.3 %** | apenas — ver abajo |
| Selloff_Q4_2018 (trampa) | **87.3 %** | **captado** — ver abajo |

**¿Captó 2013/2018? (clave de la hipótesis CP2).** Resultado **mixto**, muy
informativo:
- **2018 (Q4): SÍ (87 %).** La hipótesis acierta: como la sigma reacciona el **mismo
  día**, el sell-off de octubre–diciembre 2018 (caída real de la vol del equity)
  dispara el régimen. D4 (HMM gaussiano) solo marcaba ~46 % ahí → D6 lo supera
  claramente.
- **2013 (taper tantrum): NO (11 %).** Matiz a la hipótesis: el taper fue un shock de
  **tipos/bonos** con **poca volatilidad realizada en el equity**; un GARCH
  *univariante sobre el S&P 500* simplemente no ve ese estrés. Reacciona el mismo
  día… pero solo a la vol del **activo que modela**. La coletilla "univariante sobre
  equity" de la propia hipótesis es la que explica el fallo.

**false_alarm_rate = 0.845 (alto).** Precio de la ventana larga + señal de vol pura:
marca crisis muchos picos de vol **fuera** de las 4 ventanas etiquetadas — LTCM 1998,
dotcom 2000–02, flash-crash 2010, 2015–16 (China/Brexit), SVB 2023… Son episodios de
**alta vol reales** pero no las 4 crisis canónicas del set, así que cuentan como
"falsa alarma" contra ese ground-truth laxo. No es ruido día-a-día (ver flickering).

**Flickering / persistencia (muy buena).** `switching_rate = 0.0141`,
**duración media ≈ 70 días**, `label_stability = 0.999`. La histéresis+dwell sobre la
sigma, más la propia **persistencia de la varianza GARCH** (`beta≈0.89`), dan
episodios largos y estables — mucho menos flicker que el HMM gaussiano causal de D4
(switching 0.100, dur 9.9 d) y que el GMM estático de D3.

**Comparación con D1 / D4.**
- **vs D1 (rule_vix, OOS 1998+):** ambos son detectores de vol con ventana larga e
  histéresis. D1 usa el **nivel de VIX** (vol *implícita*, exógena); D6 usa la sigma
  **realizada/condicional** del propio S&P 500 (endógena, sin depender del VIX). D6
  evalúa incluso desde 1993 (antes del VIX usable). Esperable que D1 capte mejor 2013
  (el VIX sí reflejó algo de estrés) y D6 sea más limpio en persistencia.
- **vs D4 (hmm_gaussian, OOS 2012+):** D6 evalúa 2008/2011 OOS (D4 no podía: en train)
  y **supera a D4 en 2018** (87 % vs 46 %) — confirma que la reacción same-day de la
  sigma capta correcciones rápidas que las emisiones gaussianas latentes subestiman.
  D4 tiene `predict_proba` posterior real (HMM); D6 lo aproxima con una sigmoide.

**Bondad de ajuste.** logL = −13285.6, AIC = 26583.1, BIC = 26626.6 (k=6). Referencia
para comparar contra otros modelos de vol/colas de la Tanda 2.

## Hipótesis del CHECKPOINT 2 para D6 — veredicto

> *"El GARCH no da estados tipados (se umbraliza sigma); causal por construcción,
> reacciona el MISMO día → debería captar 2013/2018; univariante sobre equity."*

**Se cumple PARCIALMENTE.** ✓ No da estados tipados → se umbraliza sigma (con
histéresis+dwell), crisis = alta sigma (verificado, no invertido). ✓ Causal por
construcción (sigma del bloque idéntica al ocultar el futuro). ✓ Reacciona same-day y
capta **2018 (87 %)** donde D4 fallaba, además de 2008/2011/2020/2022 OOS. ✗ **2013
se le escapa (11 %)**: la propia hipótesis lo predice implícitamente con "univariante
sobre equity" — el taper fue un shock de **tipos** con poca vol *equity*, invisible
para un GARCH del solo S&P 500. Conclusión: la promesa "capta 2013 **y** 2018" solo se
verifica en 2018; el límite es la **naturaleza univariante-equity** de la señal, no la
causalidad ni la reactividad.

## Fricción con el núcleo

- **`_economic_state_order` frágil para señales de volatilidad → Arreglo 4 (resuelto
  en el núcleo).** El criterio antiguo `z(std_ret) − z(mean_ret)` podía **invertir**
  crisis/calma cuando los estados tienen retorno medio casi igual (detector de vol
  sobre histórico largo). Lo destapó D6. **El núcleo ahora ordena por volatilidad
  primaria** (banda de vol; retorno solo desempata vols próximas), así que ya no
  invierte. D6 **dejó de sobrescribir** `label_states_economically` y confía en el
  núcleo; sus métricas no cambiaron al quitar el override (la inversión solo se
  manifestaba in-sample, no en el walk-forward).
- **`false_alarm_rate` con ground-truth laxo.** Penaliza como falsa alarma toda alta
  vol fuera de las 4 ventanas canónicas; para un detector de vol con histórico largo
  (que ve LTCM, dotcom, 2010, 2023…) infla la métrica. No es defecto del núcleo, pero
  conviene leerla junto a la cobertura, no aislada.
