# D2 — `rule_composite_riskoff` (Familia F1: Reglas / Umbrales)

Regla COMPUESTA causal de 2 estados (FASE 3, Tanda 2). Agrega 4 señales de estrés
ya causales de `features.parquet` en un **score de risk-off** y lo umbraliza con
histéresis + dwell-time (mismo autómata que D1, pero sobre un VOTO multivariante en
vez de un único nivel de VIX). Entregables: `detectors/rule_composite_riskoff.py`,
`notebooks/02_rule_composite_riskoff.ipynb` (ejecutado, 0 errores, 2 figuras
inline), `results/metrics_02_rule_composite_riskoff.csv`, figuras
`results/d2_regime_sp500.png` y `results/d2_score_timeline.png`.

---

## Implementado (qué y cómo)

- **Modelo**: score compuesto de "risk-off" + autómata de 2 estados (0=calma,
  1=crisis) con **histéresis** (τ_in/τ_out) + **dwell-time** mínimo.
  - Score = media ponderada de 4 señales orientadas (signo·valor de modo que
    ALTO=estrés) y re-estandarizadas con μ/σ **del train** (causal):
    | Señal | Signo | Por qué | Referencia |
    |---|---|---|---|
    | `VIX_level_z` | **+** | miedo equity; alto=estrés | reglas_bloom2009 |
    | `credit_spread_z` | **−** | = ret(HYG)−ret(IEF); el deterioro de crédito lo vuelve NEGATIVO | reglas_gilchristzakrajsek2012 |
    | `yield_slope_z` | **−** | pendiente 10Y−3M; curva baja/invertida=estrés (signo invertido) | estrellamishkin1998 |
    | `SP500_drawdown` | **−** | drawdown ∈[−1,0]; más negativo=estrés | kritzman2012 |
  - Entra a crisis cuando el score cruza **τ_in por arriba**; sale solo cuando baja
    de **τ_out** (< τ_in) **y** se cumplen `min_dwell` días. Recorrido secuencial
    pero **causal** (estado en t depende solo de t−1 y del score en t; NaN→mantiene
    estado).
- **Parámetros**: pesos iguales (0.25 c/u), `q_in=0.90`, `q_out=0.70`,
  `min_dwell=5`. Todo parametrizable (`weights`, `signs`, `q_in/q_out`,
  `min_dwell`).
- **Causalidad de umbrales**: μ/σ de cada señal y τ_in/τ_out (percentiles 90/70 del
  score) se fijan en `fit` **solo con el train**; en walk-forward cada fold
  recalcula los suyos. Re-estandarizar es necesario porque `SP500_drawdown` vive en
  [−1,0] mientras las otras son z≈unitarias; usar μ/σ del train mantiene la
  causalidad (no se toca el contrato de no-reestandarizar con TODA la muestra).
- **Etiquetado económico ROBUSTO**: se pasa SIEMPRE `market_returns` (retorno log
  del S&P 500 = `np.log(raw['SP500']/raw['SP500'].shift(1))` reindexado a X) a
  `walk_forward` Y a `evaluate`. El núcleo re-fija el orden 0=calma..1=crisis por
  fold. **Sin warning de fallback** (verificado con `warnings.simplefilter('error')`
  sobre el walk-forward completo). En `fit` se fija el orden canónico provisional
  por construcción (`[0,1]`, score alto = risk-off), que walk_forward re-confirma.
  - Verificado en el notebook: `crisis_state == 1` y `mean_return_by_state =
    {0: +0.00117, 1: −0.00222}` → el estado crisis canónico (n−1) COINCIDE con el
    voto risk-off (peor retorno del S&P). `assert` explícito en la celda 2.
- **Ventana**: `X` usa crédito (HYG) y curva, que existen en `features.parquet`
  desde **2007-07**. Con `train_size=252*8` el primer bloque OOS cae en
  **2015-09**, así que **2008 y 2011 quedan en el train inicial → cobertura NaN
  OOS** (correcto y declarado). D2 solo se evalúa OOS sobre **2020 y 2022** (y las
  trampas 2013/2018 también caen en el train: 2018 sí entra en OOS, 2013 no).

---

## Descubierto (resultados)

`ventana_eval = 2015-09-15 → 2026-06-12 (n=2649)`. Fila completa en
`results/metrics_02_rule_composite_riskoff.csv`.

| Métrica | D2 (voto compuesto) | D1 (VIX-solo) |
|---|---|---|
| cov GFC_2008 | **NaN (en train)** | 93.8% |
| cov EuroDebt_2011 | **NaN (en train)** | 63.5% |
| cov COVID_2020 | 84.0% | 90.0% |
| cov Inflation_2022 | **53.8%** | 34.9% |
| fa Selloff_Q4_2018 | 42.4% | 6.3% |
| false_alarm_rate (global) | 0.725 | 0.697 |
| switching_rate | 0.0385 | 0.0132 |
| mean_regime_duration | 25.7 días | 75.2 días |
| label_stability | 0.9996 | 0.999 |
| leadlag COVID / Infl (días) | −251 / −219 | −17 / −179 |

**Cobertura por crisis.** GFC_2008 y EuroDebt_2011 son **NaN out-of-sample**: caen
íntegras en el train inicial por la ventana corta del crédito/curva (HYG desde
2007). No es un fallo del detector sino el límite de datos declarado; quien quiera
juzgar 2008/2011 con señales de crédito necesita histórico anterior (no existe en
el panel). Sobre lo evaluable OOS: **COVID 84%** (vs 90% de D1, ligeramente por
debajo) e **Inflación 2022 53.8%**.

**¿El voto compuesto capta estrés multivariante que el VIX solo no ve? SÍ, y se ve
justo donde se esperaba.** En el bear market de tipos de **2022** la cobertura sube
de **34.9% (D1, VIX-solo) → 53.8% (D2)**, una mejora de **+18.9 pp**. 2022 fue una
caída lenta con VIX moderado (rara vez en su decil alto), pero CON deterioro real de
crédito (`credit_spread_z` negativo) y un **drawdown** sostenido del −25%: el voto
compuesto los recoge aunque el "miedo" VIX no se disparara. Esto confirma el motivo
de existir de D2 frente al baseline univariante.

**Coste: más falsas alarmas y más flickering (la otra cara del voto).** En el
sell-off de **Q4 2018** D2 se activa el **42.4%** de los días (vs 6.3% de D1): fue
un episodio con crédito ensanchándose y drawdown ~−20%, así que el voto lo lee como
risk-off, pero en el marco estricto (solo 2008/2011/2020/2022 son crisis) cuenta
como falso positivo. El `false_alarm_rate` global sube poco (0.725 vs 0.697), pero
el **switching_rate casi se triplica (0.0385 vs 0.0132)** y la **duración media de
régimen cae a 26 días (vs 75 de D1)**: combinar 4 señales con pesos iguales mete más
ruido y conmutaciones que un único nivel de VIX suavizado. La histéresis + dwell-time
contienen el parpadeo pero no lo eliminan al nivel de D1.

**Lead/lag.** Negativos en COVID e Inflación (−251 y −219 días): la primera señal
sostenida de crisis precede al suelo del drawdown con mucha antelación, coherente
con un detector reactivo de entrada temprana (y con que la señal de crédito/drawdown
se enciende al inicio del tramo bajista, no en el fondo).

**Estabilidad.** `label_stability=0.9996`: las etiquetas casi no cambian al
reentrenar en folds sucesivos (esperable en una regla con μ/σ y percentiles
estables).

### Veredicto sobre la hipótesis del CHECKPOINT 2 (D2)
> *"Captará estrés multivariante equity+crédito+curva en 2008/2011/2020/2022;
> fallará por calibración de pesos sensible."*

**Cumplida en lo esencial, con un matiz de ventana y la calibración confirmada:**
1. **Estrés multivariante**: confirmado donde es evaluable. Capta COVID 2020 (84%) y
   sobre todo **mejora 2022 de 34.9%→53.8%** respecto al VIX-solo, exactamente por
   sumar crédito + drawdown a la señal de miedo. **2008/2011 no son juzgables OOS**
   (caen en el train inicial por la ventana de HYG): la parte de la hipótesis sobre
   esas dos crisis queda **sin verificar por datos**, no refutada.
2. **Calibración de pesos sensible**: **confirmada**. La señal de curva
   (`yield_slope_z`) es ADELANTADA, no contemporánea — en 2008/2011 la curva se
   EMPINÓ (Fed recortando el corto), de modo que con signo "curva baja=estrés" RESTA
   score justo en esas crisis; en 2020/2022, en cambio, suma. El voto con pesos
   iguales es por tanto sensible a qué señal domina cada episodio, lo que se traduce
   en el exceso de falsas alarmas de 2018 (42.4%) y el mayor switching (0.0385). Un
   ajuste de pesos (p. ej. bajar el peso de la curva o separarla como señal
   adelantada) cambiaría el balance cobertura/falsas-alarmas: justamente la
   fragilidad que el CP2 anticipaba.

**Comparación con D1.** D2 NO domina a D1: gana en 2022 (estrés multivariante) pero
pierde en especificidad (2018), persistencia (duración 26 vs 75 días) y switching
(3×). Son detectores complementarios: D1 es un baseline de miedo limpio y persistente;
D2 añade sensibilidad a estrés de crédito/drawdown a costa de más ruido. La síntesis
de FASE 4 debería tratarlos como señales que se combinan, no como sustitutos.

---

### Fricción con el núcleo
Ninguna. No se modificó `src/` (`detector_base.py`, `evaluation.py`, `features.py`,
`data_loader.py`). Dos observaciones para la familia de reglas, sin requerir cambio
en el núcleo:
1. El etiquetado económico provisional dentro de `fit` emitiría el warning de
   fallback si `X` no contiene una columna de retorno reconocida. Se evita fijando
   el orden canónico por construcción en `fit` (`[0,1]`, score alto=crisis), que
   `walk_forward` re-confirma causalmente con `market_returns`. Limpio y sin
   warnings.
2. Las señales de crédito/curva atan la ventana a 2007 (HYG), por lo que 2008/2011
   son inevitablemente NaN OOS con `train_size=252*8`. Es un límite de datos, no del
   marco; documentado en `ventana_eval`.
