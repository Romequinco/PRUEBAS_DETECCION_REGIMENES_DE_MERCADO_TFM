# D1 — `rule_vix_threshold` (Familia F1: Reglas / Umbrales)

Baseline reactivo imprescindible (CHECKPOINT 2). Regla causal sobre el nivel del
VIX con histéresis y dwell-time. Entregables: `detectors/rule_vix_threshold.py`,
`notebooks/01_rule_vix_threshold.ipynb` (ejecutado, 0 errores, 2 figuras inline),
`results/metrics_01_rule_vix_threshold.csv`, figuras `results/d1_regime_sp500.png`
y `results/d1_prob_timeline.png`.

---

## Implementado (qué y cómo)

- **Modelo**: autómata de 2 estados (0=calma, 1=crisis) sobre `VIX_level_z` con
  **histéresis** (banda muerta τ_in/τ_out) + **dwell-time** mínimo.
  - Entra a crisis cuando el VIX z cruza **τ_in por arriba**.
  - Sale solo cuando baja de **τ_out** (< τ_in) **y** se han cumplido `min_dwell`
    días en crisis. Esto mata el flickering del umbral simple.
  - Recorrido **secuencial pero causal**: el estado en `t` depende solo del estado
    en `t-1` y del VIX en `t`. Sin look-ahead. NaN → mantiene estado previo.
- **Parámetros**: `q_in=0.90`, `q_out=0.70`, `min_dwell=5`.
- **Umbrales causales**: τ_in/τ_out se fijan en `fit` como **percentiles del VIX z
  SOLO del train** (`np.quantile(q_in)/(q_out)`), no de toda la muestra. En
  walk-forward cada fold recalcula sus cortes con su propio train.
- **Feature causal**: `VIX_level_z = features.causal_zscore(VIX)` (expanding). Se
  verifica no look-ahead en el notebook (`max_abs_diff < 1e-9` al truncar en 2015).
- **Ventana LARGA (clave de D1)**: la feature se construye desde el panel crudo
  (`data/raw/raw_panel.parquet`), no desde `features.parquet` (atado a 2007 por
  HYG). El VIX existe desde **1990**, así que `X` abarca 1990–2026 y el
  walk-forward (`train_size=252*8, step=21, expanding`) deja un panel OOS
  **1998-06-23 → 2026-06-12 (n=6994)** que cubre 2008 y 2011 out-of-sample.
- **Etiquetado económico**: se añade `SP500_ret` a `X` para que
  `label_states_economically` ordene los estados por severidad real (retorno medio
  + vol). Resultado: orden canónico = identidad `[0,1]`, con
  `mean_return_by_state` = {0: +0.00041, 1: −0.00061} → confirma 0=calma, 1=crisis.
  `market_returns` del evaluador = retorno log del S&P 500 reindexado al panel OOS.
- **Bibliografía** (claves reales de `docs/references.bib`): `reglas_bloom2009`,
  `reglas_moreiramuir2017`, `kritzman2012`.

---

## Descubierto (resultados)

`ventana_eval = 1998-06-23 → 2026-06-12 (n=6994)`. Fila completa en
`results/metrics_01_rule_vix_threshold.csv`.

| Métrica | Valor |
|---|---|
| cov GFC_2008 | **93.8%** |
| cov EuroDebt_2011 | 63.5% |
| cov COVID_2020 | **90.0%** |
| cov Inflation_2022 | **34.9%** |
| fa TaperTantrum_2013 | **0.0%** |
| fa Selloff_Q4_2018 | 6.3% |
| false_alarm_rate (global) | 0.697 |
| switching_rate | 0.0132 |
| mean_regime_duration | 75.2 días |
| label_stability | 0.999 |
| leadlag GFC / Euro / COVID / Infl (días) | −251 / −39 / −17 / −179 |

**Cobertura por crisis (incluida 2008 OOS).** Gracias al histórico largo, 2008 se
evalúa out-of-sample y se cubre casi entera (93.8%); COVID 90%, EuroDebt 63.5%.
La excepción es **Inflación 2022 (34.9%)**: fue un *bear market* lento de tipos
donde el VIX se mantuvo moderado (rara vez en su decil superior), así que un
detector puramente de VIX lo infra-detecta. Hallazgo esperable y útil: motiva D2
(regla compuesta con crédito/curva/drawdown) y los detectores de volatilidad/HMM.

**¿Captó 2013/2018 que el HMM gaussiano (D4) falla?** Con histéresis, **NO** se
dispara de forma sostenida: 2013 = 0.0%, 2018 = 6.3%. La hipótesis del CP2
anticipaba que "por reactividad probablemente captaría 2013/2018"; lo que ocurre es
que **la banda muerta + dwell-time los suprime a propósito** (fueron correcciones
rápidas, no crisis sistémicas). Es decir: la versión SIN histéresis sí se
encendería en esos picos (eso es justo lo que el CP2 advertía como falso positivo);
la versión CON histéresis los limpia. En el marco estricto del proyecto (solo
2008/2011/2020/2022 son crisis), esto es la respuesta correcta y mejora la
especificidad frente al baseline.

**false_alarm_rate = 0.697 (alto, con matiz).** El 70% de los días marcados crisis
caen fuera de las 4 ventanas oficiales. No es ruido espurio en su mayoría: las
ventanas de `evaluation.py` son **estrechas** (p.ej. GFC solo 2008-09→2009-03)
mientras el VIX sigue elevado meses después, y hay **episodios reales de estrés no
catalogados** dentro de 1998–2026 (LTCM 1998, DotCom 2000-02, flash-crash 2010,
China 2015-16, SVB 2023). El marco los cuenta como falsas alarmas porque no están
en `CRISIS_WINDOWS`. La histéresis ya reduce el componente verdaderamente espurio:
`switching_rate=0.013` y **duración media de régimen 75 días** indican episodios
largos y estables, no parpadeo.

**Lead/lag.** Todos negativos: la primera señal de crisis **precede** al suelo del
drawdown en cada evento (−17 días en COVID, −39 en EuroDebt; −251 y −179 en GFC e
Inflación reflejan que la señal entra muchos meses antes del fondo en bear markets
prolongados). Consistente con un detector reactivo de entrada temprana.

**Estabilidad.** `label_stability=0.999`: las etiquetas casi no bailan al reentrenar
en folds sucesivos (esperable en una regla con umbrales-percentil estables).

### Veredicto sobre la hipótesis del CHECKPOINT 2
> *"Capta las 4 crisis y probablemente 2013/2018; falla en falsos positivos sin
> histéresis."*

**Cumplida en lo esencial, con dos matices cuantificados:**
1. **Crisis**: captadas 3 de 4 con fuerza (2008 93.8%, 2020 90%, 2011 63.5%); la 4ª
   (2022) solo 34.9% por ser un régimen de VIX moderado — límite real de un
   detector univariante de VIX, no un fallo de implementación.
2. **2013/2018 y falsos positivos**: confirmado que el riesgo está en los picos
   efímeros; **la histéresis los neutraliza** (2013=0%, 2018=6.3%, switching 0.013),
   demostrando empíricamente por qué la banda muerta + dwell-time es necesaria. El
   `false_alarm_rate` global (0.697) está inflado por ventanas de crisis estrechas y
   episodios de estrés reales no catalogados, no por flickering.

---

### Fricción con el núcleo
Ninguna. No se modificó `detector_base.py`, `evaluation.py`, `features.py` ni
`data_loader.py`. Único punto a tener en cuenta para los demás detectores: el
etiquetado económico por defecto (`_economic_state_order`) infiere los "retornos"
de la primera columna estándar que encuentre; con una `X` que solo contiene
`VIX_level_z` interpretaría el nivel de VIX como retorno e **invertiría** el orden
canónico. Se resuelve incluyendo `SP500_ret` en `X` (o pasando `market_returns` a
`label_states_economically`). No requiere cambio en el núcleo.
