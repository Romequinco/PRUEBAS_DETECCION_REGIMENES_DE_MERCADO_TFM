# Fichas v1 de los detectores (Capa 1)

Narrativa original de las fichas D01–D12, redactadas en la FASE 3 de la Capa 1 (originales en el tag `capa1-final`). Se conservan íntegras salvo rutas: qué se implementó, qué se descubrió, el veredicto sobre la hipótesis del CHECKPOINT 2 y la fricción con el núcleo de entonces. La configuración y los resultados vigentes (v2, ADR-003) de cada detector están en su documento de familia de [`docs/fase1/`](../fase1/); la ablación D13 (HSMM), en [F3](../fase1/F3_hmm.md#d13).

> **Cómo leerlas.** Son datos v1: ventana 2007-2026 (desde 1985-1990 en los detectores de histórico largo), 4 crisis oficiales (GFC 2008, deuda europea 2011, COVID 2020 e inflación 2022) más las trampas 2013 y Q4-2018, y métricas de cobertura sin criterio de detección. **No son comparables** con los resultados v2. Cuando el texto cita el núcleo de la Capa 1: `detector_base.py` → `src/regimenes/detectores/base.py`; `evaluation.py` → `regimenes.evaluacion`; `features.py`, `data_loader.py`, `INDEX.md`, la memoria, los constructores `_build_*.py` y los notebooks `capa1_exploracion/` quedan en el tag `capa1-final`. Las métricas v1 están en [`capa1/resultados/`](capa1/resultados/); las figuras v1 no se versionan.

| Ficha v1 | Detector | Configuración y resultados v2 |
|---|---|---|
| [D01](#d01) | `rule_vix_threshold` | [F1 — Reglas y umbrales](../fase1/F1_reglas.md#d01) |
| [D02](#d02) | `rule_composite_riskoff` | [F1 — Reglas y umbrales](../fase1/F1_reglas.md#d02) |
| [D03](#d03) | `clustering_gmm` | [F2 — Clustering](../fase1/F2_clustering.md#d03) |
| [D04](#d04) | `hmm_gaussian_2s` | [F3 — Modelos ocultos de Markov (HMM)](../fase1/F3_hmm.md#d04) |
| [D05](#d05) | `markov_switching_var` | [F4 — Markov-Switching econométrico](../fase1/F4_markov_switching.md#d05) |
| [D06](#d06) | `garch_t_vol` | [F5 — Volatilidad y GARCH](../fase1/F5_volatilidad_garch.md#d06) |
| [D07](#d07) | `changepoint_online` | [F6 — Change-point](../fase1/F6_change_point.md#d07) |
| [D08](#d08) | `hmm_tstudent` | [F3 — Modelos ocultos de Markov (HMM)](../fase1/F3_hmm.md#d08) |
| [D09](#d09) | `jump_model` | [F2 — Clustering](../fase1/F2_clustering.md#d09) |
| [D10](#d10) | `turbulence_mahalanobis` | [F1 — Reglas y umbrales](../fase1/F1_reglas.md#d10) |
| [D11](#d11) | `msgarch_regime` | [F5 — Volatilidad y GARCH](../fase1/F5_volatilidad_garch.md#d11) |
| [D12](#d12) | `deep_ae_regime` | [F7 — Redes neuronales](../fase1/F7_redes_neuronales.md#d12) |

<a id="d01"></a>

## D1 — `rule_vix_threshold` (Familia F1: Reglas / Umbrales)

> Configuración y resultados v2: [F1 — Reglas y umbrales](../fase1/F1_reglas.md#d01).

Baseline reactivo imprescindible (CHECKPOINT 2). Regla causal sobre el nivel del
VIX con histéresis y dwell-time. Entregables: `src/regimenes/detectores/f1_reglas/rule_vix_threshold.py`,
`capa1_exploracion/notebooks/01_rule_vix_threshold.ipynb` (v1, tag `capa1-final`) (ejecutado, 0 errores, 2 figuras inline),
`docs/historia/capa1/resultados/metrics_01_rule_vix_threshold.csv`, figuras `d1_regime_sp500.png` (figura v1, no versionada)
y `d1_prob_timeline.png` (figura v1, no versionada).

---

### Implementado (qué y cómo)

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

### Descubierto (resultados)

`ventana_eval = 1998-06-23 → 2026-06-12 (n=6994)`. Fila completa en
`docs/historia/capa1/resultados/metrics_01_rule_vix_threshold.csv`.

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

#### Veredicto sobre la hipótesis del CHECKPOINT 2
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

#### Fricción con el núcleo
Ninguna. No se modificó `detector_base.py`, `evaluation.py`, `features.py` ni
`data_loader.py`. Único punto a tener en cuenta para los demás detectores: el
etiquetado económico por defecto (`_economic_state_order`) infiere los "retornos"
de la primera columna estándar que encuentre; con una `X` que solo contiene
`VIX_level_z` interpretaría el nivel de VIX como retorno e **invertiría** el orden
canónico. Se resuelve incluyendo `SP500_ret` en `X` (o pasando `market_returns` a
`label_states_economically`). No requiere cambio en el núcleo.

<a id="d02"></a>

## D2 — `rule_composite_riskoff` (Familia F1: Reglas / Umbrales)

> Configuración y resultados v2: [F1 — Reglas y umbrales](../fase1/F1_reglas.md#d02).

Regla COMPUESTA causal de 2 estados (FASE 3, Tanda 2). Agrega 4 señales de estrés
ya causales de `features.parquet` en un **score de risk-off** y lo umbraliza con
histéresis + dwell-time (mismo autómata que D1, pero sobre un VOTO multivariante en
vez de un único nivel de VIX). Entregables: `src/regimenes/detectores/f1_reglas/rule_composite_riskoff.py`,
`capa1_exploracion/notebooks/02_rule_composite_riskoff.ipynb` (v1, tag `capa1-final`) (ejecutado, 0 errores, 2 figuras
inline), `docs/historia/capa1/resultados/metrics_02_rule_composite_riskoff.csv`, figuras
`d2_regime_sp500.png` (figura v1, no versionada) y `d2_score_timeline.png` (figura v1, no versionada).

---

### Implementado (qué y cómo)

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

### Descubierto (resultados)

`ventana_eval = 2015-09-15 → 2026-06-12 (n=2649)`. Fila completa en
`docs/historia/capa1/resultados/metrics_02_rule_composite_riskoff.csv`.

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

#### Veredicto sobre la hipótesis del CHECKPOINT 2 (D2)
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

#### Fricción con el núcleo
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

<a id="d03"></a>

## D3 — `clustering_gmm` (familia CLUSTERING, baseline NO temporal)

> Configuración y resultados v2: [F2 — Clustering](../fase1/F2_clustering.md#d03).

> Mixtura gaussiana estática (`GaussianMixture`, `covariance_type='full'`) sobre
> las 15 features causales. Sin cadena de Markov: cada día se asigna de forma
> independiente al componente más probable. Es el **baseline contra el que D4
> (HMM) medirá cuánto aporta la dinámica temporal** (mismas features, misma
> evaluación walk-forward).

### Implementado

- **Clase** `src/regimenes/detectores/f2_clustering/clustering_gmm.py::ClusteringGMM(RegimeDetector)`.
  - `name` = `clustering_gmm_k{n_states}`.
  - `bibliography` = `["clust_twosigma2021regime", "clust_munnix2012", "gulko2002",
    "lopezdeprado2018"]` (claves verificadas en `docs/references.bib`).
  - `fit`: `GaussianMixture(covariance_type='full', n_init=5, reg_covar=1e-6,
    max_iter=300)` sobre `X_train.values`; marca `_is_fitted` y llama a
    `label_states_economically(X_train)` para fijar `self._canonical_order`.
  - `_predict_states`: etiquetas INTERNAS (`_model.predict`).
  - **`predict_proba` (override)**: `_model.predict_proba(X)[:, self._canonical_order]`
    → posteriores reales reordenados al orden canónico (0=calma .. n-1=crisis), de
    modo que la última columna es P(crisis). `predict` (duro canónico) y
    `crisis_state` se heredan del núcleo.
  - `score` = `_model.score(X) * len(X)` (log-likelihood TOTAL; sklearn da la media).
  - `n_parameters` = `(k-1) + k·d + k·d(d+1)/2` (pesos + medias + covarianzas full),
    con d=15 → 271 (k=2), 407 (k=3). Habilita AIC/BIC del núcleo.
- **`covariance_type='full'`** elegido a propósito: capta el régimen de correlación
  que cambia de signo (Gulko 2002; feature `corr_spx_bond`), que un k-means euclídeo
  o un GMM diagonal no separan.
- **Causalidad**: el detector solo mira `X_train`; el wrapper causal lo da
  `ev.walk_forward` (re-fit expanding, train inicial 8 años, step 21d). El alineado
  de etiquetas entre folds lo resuelve la canonicalización económica en cada `fit`.
- **Notebook** `capa1_exploracion/notebooks/03_clustering_gmm.ipynb` (v1, tag `capa1-final`) (construido y ejecutado con
  `_build_03.py` (retirado en el commit `6016f42`), 0 errores, 3 figuras inline): selección de k por BIC,
  sanidad del orden canónico, walk-forward k=2 y k=3, tabla de métricas, histograma
  de duraciones (flickering), S&P 500 coloreado por régimen, timeline + P(crisis), y
  bloque de verificación con asserts sobre 2008/2011 (NaN) y COVID/Inflación.
- **Artefactos**: `docs/historia/capa1/resultados/metrics_03_clustering_gmm.csv` (1 fila, detector k=3
  elegido por BIC), `d03_gmm_flickering.png` (figura v1, no versionada),
  `d03_gmm_sp500_regimes.png` (figura v1, no versionada), `d03_gmm_timeline.png` (figura v1, no versionada).

### Descubierto

#### k elegido por BIC
BIC in-sample sobre el set completo: **k=2 → 70 975**, **k=3 → 63 016**. Gana
**k=3** (BIC menor). El detector principal volcado al CSV es `clustering_gmm_k3`.
Un tercer estado intermedio (estrés/transición) sí compensa su coste en parámetros.

#### Política de ventana y cobertura por crisis
Las 15 features arrancan en **2007-07**; con train inicial expanding de 8 años el
primer bloque OOS empieza en **2015-09-15** y va hasta **2026-06-12** (n=2649). Por
tanto **2008 (GFC) y 2011 (EuroDebt) NO son OOS-evaluables** —quedan dentro del
primer train— y su cobertura sale **`NaN`**, que es el comportamiento CORRECTO (no
se penaliza lo que el detector no pudo ver). Igualmente la trampa **TaperTantrum
2013** cae fuera de OOS (`NaN`).

Cobertura de crisis OOS (k=3):
- **COVID_2020 = 0.96** (sensibilidad muy alta).
- **Inflation_2022 = 0.87**.
- GFC_2008 = NaN, EuroDebt_2011 = NaN (fuera de OOS, esperado).

Falsos positivos en trampas:
- TaperTantrum_2013 = NaN (fuera de OOS).
- **Selloff_Q4_2018 = 0.00** (no marca crisis sostenida en la trampa de 2018; bien).
- `false_alarm_rate` global = **0.49**: la mitad de los días marcados "crisis" caen
  fuera de las ventanas conocidas — coherente con el flickering (marca crisis
  sueltas dispersas), no con falsas alarmas sostenidas.

Lead/lag (días vs suelo de drawdown): COVID **−20 d**, Inflación **−219 d** (en
COVID la señal P(crisis)≥0.5 sostenida llega poco antes del suelo; en Inflación
cruza muy por delante; ambos suelos sí caen en OOS).

#### Flickering medido (talón de Aquiles)
- **switching_rate = 0.126** (k=3) / 0.112 (k=2): conmuta de estado en ~1 de cada 8
  días OOS.
- **mean_regime_duration = 7.9 d** (k=3) / 8.9 d (k=2): rachas de régimen muy
  cortas, irreales para "regímenes" de mercado.
- `label_stability = 0.976`: las etiquetas por fecha son estables entre re-fits
  (el flickering es intra-secuencia, no inestabilidad entre folds).

#### Comparación con la hipótesis del CHECKPOINT 2
Hipótesis CP2: *"captará regímenes con estructura de correlación distinta; fallará
por flickering severo; no causal nativo"*. **Se cumple en los tres puntos:**
1. **Capta estructura de correlación**: con covarianza full detecta COVID (0.96) e
   Inflación (0.87) y separa el estado de crisis por retorno/vol decrecientes —la
   covarianza plena ve el cambio de signo de la correlación equity/bonos (Gulko).
2. **Flickering severo**: switching_rate 0.126 y rachas medias ~8 días confirman el
   parpadeo esperado de un modelo sin término de persistencia. Este es justamente el
   número de referencia que D4 (HMM, con matriz de transición) debería **reducir**.
3. **No causal nativo**: el GMM estático mira toda la muestra al ajustarse; la
   causalidad se logra SOLO vía el `walk_forward` (re-fit expanding), no por diseño
   del detector.

**Conclusión**: D3 es un baseline interpretable y sensible a crisis, pero su
flickering lo descarta como detector definitivo; su valor es servir de referencia
NO temporal para aislar el aporte de la dinámica markoviana del HMM.

### Fricción con el núcleo
Ninguna. La interfaz de `RegimeDetector` (override de `predict_proba` con
`_canonical_order`, `score`/`n_parameters` para AIC/BIC) y `evaluation.walk_forward`
cubrieron el caso probabilístico sin necesidad de cambios. No se modificó
`src/detector_base.py`, `src/evaluation.py`, `src/features.py` ni `INDEX.md`.

<a id="d04"></a>

## D4 — `hmm_gaussian_2s` (FASE 3, Tanda 1) · Familia F3 (HMM)

> Configuración y resultados v2: [F3 — Modelos ocultos de Markov (HMM)](../fase1/F3_hmm.md#d04).

> **Baseline puente** con la tarea previa (`Tarea_riesgos.ipynb`). Reproduce
> honestamente el GaussianHMM 2 estados y MIDE el efecto del look-ahead comparando
> una versión in-sample (no causal) con una causal walk-forward.
>
> Código: `src/regimenes/detectores/f3_hmm/hmm_gaussian_2s.py` · Notebook: `capa1_exploracion/notebooks/04_hmm_gaussian_2s.ipynb` (v1, tag `capa1-final`)
> (constructor `_build_04.py` (retirado en el commit `6016f42`)) · Métricas: `docs/historia/capa1/resultados/metrics_04_hmm_gaussian_2s.csv`
> (causal) y `docs/historia/capa1/resultados/metrics_04_hmm_gaussian_2s_insample.csv` (in-sample, no causal).

### Implementado

**Modelo.** `GaussianHMM(n_components=2, covariance_type='full', n_iter=200)` de
`hmmlearn`. En `fit` se prueban **5 inicializaciones** (semillas 42–46) y se elige la
de mayor log-verosimilitud (mitiga óptimos locales del EM, como la tarea previa, que
usaba 10 seeds 42–51). El orden económico de estados (0=calma, 1=crisis) lo fija
`label_states_economically` del núcleo, usando el retorno log del S&P 500.

**Features puente (7).** Subconjunto de las 15 de `features.parquet`, idéntico a la
tarea previa: `SP500_ret_z, TLT_ret_z, IEF_ret_z, HYG_ret_z, SP500_vol_z,
credit_spread_z, VIX_level_z`. Constante `BRIDGE_FEATURES` en el módulo. Ventana
2007-07-06 → 2026-06-12 (4 665 obs).

**Override.** `predict` (duro, vía `_predict_states`) usa **Viterbi** → modo
in-sample NO causal. `predict_online` y `predict_proba` usan **FILTRADO FORWARD
causal** (`src/regimenes/detectores/f3_hmm/_hmm_utils.py::filtered_posterior`) con contexto de burn-in: la
etiqueta/probabilidad de t solo usa observaciones <= t, reordenadas al orden
canónico. `score` = logL; `n_parameters` = `k²−1 + k·d + k·d(d+1)/2` (transición +
medias + covarianzas full) = 73 con k=2, d=7. Bibliografía: `hamilton1989,
hmm_rabiner1989, hmm_bulla2011, guidolintimmermann2007` (verificadas en
`docs/references.bib`).

**Dos versiones (núcleo de D4).**
1. **IN-SAMPLE (NO causal, marcada).** Ajuste + Viterbi sobre TODA la muestra.
   Reproduce la tarea previa; tiene look-ahead (el EM ve el futuro y el Viterbi
   suaviza sobre todo el histórico). Se reporta aparte, NO comparable con causales.
2. **CAUSAL walk-forward.** `ev.walk_forward(lambda: HMMGaussian2S(n_states=2),
   X, train_size=252*5, step=21, expanding=True)`. Cada fold reentrena solo con
   pasado y decodifica el bloque de test de 21 días.

**Nota de causalidad (CHECKPOINT 3A-ter — Arreglo 2).** La versión causal usa
**filtrado forward estricto**, NO Viterbi intra-bloque. El Viterbi y el
`predict_proba` de `hmmlearn` suavizan con todo el bloque (miran días futuros del
mes), look-ahead de hasta `step`=21 días. `predict_online`/`predict_proba`
sobrescritos calculan el forward filter (`_hmm_utils`) sobre `train_burn-in + bloque`
y devuelven solo el bloque: cada t usa solo obs <= t. El Viterbi sobre toda la
muestra queda EXCLUSIVAMENTE en la versión in-sample marcada NO causal. Patrón
reutilizable por D8 (hmm_tstudent).

### Descubierto

**`ventana_eval` causal:** OOS = 2012-07 → 2026-06 aprox. (train inicial de ~5 años).
Con esa ventana, **2008 (GFC) y 2011 (deuda europea) caen DENTRO del train inicial**
→ no son out-of-sample → su cobertura sale **NaN** en la versión causal. Es el punto
metodológico central: con datos que empiezan en 2007-04 (por HYG) no se puede evaluar
la GFC 2008 de forma causal sin sacrificar casi todo el train. La versión in-sample sí
las cubre (in-sample). Las crisis evaluables OOS son **2020 y 2022**, y las trampas
**2013 y 2018**.

**Cobertura in-sample vs causal por crisis** (números finales del notebook
ejecutado; `ventana_eval` causal = 2012-07-20 → 2026-06-12, n=3405):

| Ventana | In-sample | Causal OOS | Lectura |
|---|---:|---:|---|
| GFC_2008 | 100.0 % | **NaN** | en train, no OOS |
| EuroDebt_2011 | 81.0 % | **NaN** | en train, no OOS |
| COVID_2020 | 96.0 % | 96.0 % | OOS, idéntica (no era look-ahead) |
| Inflation_2022 | 82.7 % | 86.1 % | OOS, sube ligeramente |
| TaperTantrum_2013 (trampa) | 12.0 % | 25.0 % | no sostenida, algo peor causal |
| Selloff_Q4_2018 (trampa) | 30.5 % | 45.8 % | parcial, peor causal |

*(cifras causales tras el Arreglo 2 = filtrado forward; ver "Re-evaluación" abajo.)*

**Efecto del look-ahead.** Para las crisis que SÍ son OOS (2020, 2022) la cobertura
**no cae** al pasar de in-sample a causal — incluso sube ligeramente en 2022. Es decir,
el acierto del modelo en crisis **grandes y persistentes** NO era un artefacto del
look-ahead: un HMM gaussiano reentrenado solo con pasado las sigue capturando. El
look-ahead de la tarea previa afectaba sobre todo a la *estabilidad de las etiquetas
históricas* (re-decodificación de 2008/2011 al reentrenar), no a la detección OOS de
las crisis recientes. La pérdida real al "ir causal" es que **2008/2011 dejan de ser
evaluables** por la ventana, no que el modelo empeore.

**¿2008 fuera de OOS?** Sí: con train inicial de varios años 2008 (y 2011) quedan en el
train → NaN en la versión causal. Documentado en `01_data_and_eda.md` §3 (tensión de
cobertura). Para evaluar 2008 causalmente haría falta un detector con histórico más
largo (p. ej. solo S&P 500+VIX desde 1990, como D1), no el HMM puente atado a las 7
features con HYG desde 2007.

**Fallo en 2013/2018.** Confirmado: el HMM gaussiano apenas marca crisis sostenida en
el taper tantrum 2013 y solo parcialmente en el sell-off Q4 2018. Coherente con las
fat tails del EDA (kurtosis 25–40): las emisiones gaussianas subestiman las colas y el
modelo prioriza episodios largos de alta varianza, perdiéndose las correcciones
rápidas. Es justo la motivación de los detectores posteriores (HMM t-Student / GMM-HMM,
D8; K=3–4 para separar corrección de crisis sistémica).

**Flickering / persistencia.** In-sample (Viterbi global): `switching_rate=0.047`,
duración media **21.2 días**. Causal con filtrado forward: `switching_rate=0.100`,
duración media **9.9 días**. La causal flickea ~2.1× más que la in-sample suavizada,
y por eso sus falsas alarmas en 2013/2018 son mayores (25 % y 46 %). El beneficio del
look-ahead in-sample era la suavidad del recorrido, no la detección OOS de crisis
grandes. `label_stability=0.994` (muy alto): reentrenar con más datos casi no cambia
la etiqueta OOS ya emitida.

**Bondad de ajuste.** logL, AIC y BIC se exponen vía `score`/`aic`/`bic` (k=73 params).
Sirven de referencia para comparar contra K=3 y contra emisiones t-Student en detectores
posteriores (se espera que BIC mejore con t pese a más parámetros si las colas se
ajustan mejor).

### Re-evaluación tras Arreglos 1-3 (CHECKPOINT 3A-ter) — hallazgo

El Arreglo 2 sustituyó el Viterbi/forward-backward intra-bloque por **filtrado
forward causal**. La hipótesis de partida era que quitar el suavizado intra-bloque
SUBIRÍA el switching. Ocurrió lo **contrario**: switching **bajó 0.124 → 0.100** y la
duración media **subió 8.1 → 9.9 días**.

**Por qué (explicación).** El esquema anterior decodificaba con Viterbi cada bloque de
21 días **por separado**, rearrancando desde `startprob_` en cada frontera de mes →
inestabilidad en bloques cortos y saltos artificiales en las costuras. El filtrado
forward, en cambio, propaga la creencia de estado **de forma continua** a través de
todo el histórico (con burn-in del train), gobernada por la diagonal alta de la matriz
de transición → más persistencia. Conclusión: el filtrado forward es a la vez **más
causal** (cero look-ahead intra-bloque) **y más estable** que el Viterbi-por-bloque.
El resto de cambios menores: `false_alarm_rate` 0.76 → 0.73, `cov_Inflation_2022`
0.894 → 0.861, `fa_2013` 0.28 → 0.25, `fa_2018` 0.42 → 0.46, y lead/lag más realistas
(COVID causal −240 → −151 d; la in-sample, con `p_crisis` ahora filtrada en vez de
suavizada, pasa de −158 a −20 d). 2008/2011 siguen NaN (en train).

### Hipótesis del CHECKPOINT 2 para D4 — veredicto

> *"Captará 2008 y 2020 (crisis grandes y persistentes); fallará 2013 y 2018
> (correcciones rápidas; el gaussiano subestima las colas); la versión in-sample es
> NO causal."*

**Se cumple.** (1) In-sample capta 2008 (~100 %) y 2020 (~96 %); causalmente 2020/2022
se mantienen altas OOS (2008 no evaluable OOS por la ventana, ver arriba). (2) Falla
2013/2018 en ambas versiones (baja cobertura sostenida) → el supuesto gaussiano pierde
las correcciones rápidas. (3) La versión in-sample está claramente etiquetada como NO
causal y se separa de la causal comparable. El único matiz frente al enunciado de la
hipótesis: "capta 2008" es verificable solo in-sample; OOS 2008 queda fuera de la
ventana de evaluación causal (no es que falle, es que no se puede ver sin look-ahead).

### Fricción con el núcleo

Ninguna que requiera cambiar firmas del núcleo. La causalidad estricta intra-bloque
(Arreglo 2) se resolvió **en el detector**, sobrescribiendo `predict_online`/
`predict_proba` con filtrado forward (`src/regimenes/detectores/f3_hmm/_hmm_utils.py`), exactamente como el
docstring de `walk_forward` contempla. Observación:
- La cobertura NaN de 2008/2011 en la versión causal es comportamiento correcto de
  `crisis_coverage` (ventana fuera del rango OOS → NaN), no un defecto.

<a id="d05"></a>

## D5 — `markov_switching_var` (FASE 3, Tanda 2) · Familia F4 (Markov-Switching)

> Configuración y resultados v2: [F4 — Markov-Switching econométrico](../fase1/F4_markov_switching.md#d05).

> Baseline econométrico **interpretable**. Markov-Switching de varianza sobre el
> retorno del S&P 500 (Hamilton 1989), con probabilidades **filtradas causales**.
> Código: `src/regimenes/detectores/f4_switching/markov_switching_var.py` · Notebook:
> `capa1_exploracion/notebooks/05_markov_switching_var.ipynb` (v1, tag `capa1-final`) (constructor `_build_05.py` (retirado en el commit `6016f42`))
> · Métricas: `docs/historia/capa1/resultados/metrics_05_markov_switching_var.csv`.

### Implementado

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

### Descubierto

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

### Hipótesis del CHECKPOINT 2 para D5 — veredicto

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

### Fricción con el núcleo

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

<a id="d06"></a>

## D6 — `garch_t_vol` (FASE 3, Tanda 2) · Familia F5 (Volatilidad)

> Configuración y resultados v2: [F5 — Volatilidad y GARCH](../fase1/F5_volatilidad_garch.md#d06).

> Detector univariante sobre el **retorno log del S&P 500** (desde 1985). Ajusta un
> **GJR-GARCH(1,1)-t**, obtiene la **sigma condicional** y define el régimen
> **umbralizando** esa sigma (percentil del train + histéresis + dwell). 2 estados.
>
> Código: `src/regimenes/detectores/f5_garch/garch_t_vol.py` · Notebook: `capa1_exploracion/notebooks/06_garch_t_vol.ipynb` (v1, tag `capa1-final`)
> (constructor `_build_06.py` (retirado en el commit `6016f42`)) · Métricas:
> `docs/historia/capa1/resultados/metrics_06_garch_t_vol.csv` · Figuras: `d06_*.png` (figura v1, no versionada).

### Implementado

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

### Descubierto

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

### Hipótesis del CHECKPOINT 2 para D6 — veredicto

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

### Fricción con el núcleo

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

<a id="d07"></a>

## D7 — `changepoint_online` (FASE 3, Tanda 3) · Familia F6 (Change-point)

> Configuración y resultados v2: [F6 — Change-point](../fase1/F6_change_point.md#d07).

> Detector de **cambio estructural ONLINE y CAUSAL** sobre el **retorno log del S&P 500**
> (desde 1985). Núcleo: un **CUSUM de Page (1954)** secuencial sobre un estadístico de
> volatilidad; un autómata de 2 CUSUM de una cara lo mapea a 2 estados (0=calma,
> 1=crisis). El umbral `h` da persistencia. 2 estados.
>
> Código: `src/regimenes/detectores/f6_changepoint/changepoint_online.py` · Notebook: `capa1_exploracion/notebooks/07_changepoint_online.ipynb` (v1, tag `capa1-final`)
> (constructor `_build_07.py` (retirado en el commit `6016f42`)) · Métricas:
> `docs/historia/capa1/resultados/metrics_07_changepoint_online.csv` · Figuras: `d07_*.png` (figura v1, no versionada).

### Implementado

**Método ONLINE causal (CUSUM de Page 1954).** La distinción clave de la familia F6:
la variante potente de `ruptures` (**PELT** — cp_killick2012; BinSeg — cp_truong2020) es
**OFFLINE y ANTI-causal** (segmenta mirando toda la serie → vería el futuro del bloque en
walk-forward). D7 usa el **CUSUM secuencial**, la variante online: acumula evidencia de un
cambio de nivel y dispara una alarma cuando el cumulativo supera `h`, usando solo datos
≤ t (detecta el cambio **con cierto retardo**, como toca a un método causal). PELT se usa
SOLO en el notebook (§3) como **oráculo in-sample**, explícitamente marcado **NO causal**.

**De change-point a 2 estados recurrentes (reto de integración).** Un change-point
*segmenta*, no etiqueta estados. Mapeo elegido: **autómata de 2 estados con dos CUSUM de
una cara** (Page):
- En CALMA (0) se vigila un cambio **al alza** de la vol con `C⁺ = max(0, C⁺ + z − k)`; si
  `C⁺ > h` → se entra en CRISIS (1) y se resetean los acumuladores.
- En CRISIS (1) se vigila un cambio **a la baja** con `C⁻ = max(0, C⁻ − z − k)`; si
  `C⁻ > h` → se vuelve a CALMA (0).
El umbral `h` da **persistencia por construcción** (hace falta evidencia acumulada para
conmutar) → poco flickering, sin dwell explícito. **QUÉ** tramo es "crisis" lo decide el
**NÚCLEO** vía `label_states_economically` (Arreglo 4, vol-primario): D7 solo separa por
nivel de vol; el núcleo pone la polaridad → crisis = el tramo de **mayor σ** de los
retornos. "Etiqueta de régimen por día = en tramo de alta vol tras un cambio al alza" = 1.

**Estadístico monitorizado y coste (gaussiano vs robusto — eje de CP2).**
- `cost='robust'` (DEFECTO): estadístico = **log|retorno|**, estandarizado con
  **MEDIANA / MAD** del train (escala robusta de Gauss, `1.4826·MAD`) y **winsorizado** a
  ±`clip`. log|r| es ~**simétrico** (log-normalidad de la vol), lo que permite que el
  CUSUM acumule en ambos sentidos y el autómata **conmute** correctamente; mediana/MAD +
  winsor **acotan la influencia de cada día de cola**.
- `cost='gaussian'`: estadístico = **retorno²** (coste cuadrático L2), estandarizado con
  **media / desv** del train. Fuertemente asimétrico y dominado por las colas (kurt 25–40).

  Defaults: `k=0.5` (holgura/slack), `h=5.0`, `clip=3.0`. `predict_proba` = **sigmoide
  monótona** del EWMA (halflife 10) del estadístico de vol estandarizado, centrada con su
  media/desv in-sample → ~0.5 en el nivel medio, ↑ con la vol reciente; reordenada al
  orden canónico (como D6/GMM). El detector NO es generativo → `score`/AIC/BIC = NaN.

**Causalidad en walk-forward (patrón burn-in, como D6).** El CUSUM es causal nativo (C±ₜ
depende solo de z con s ≤ t), pero el walk-forward usa bloques de `step=21` días con
detector NUEVO por fold; arrancar en frío en CALMA cada bloque perdería tramos de crisis
en curso. Se evita re-ejecutando el CUSUM sobre `[retornos de train anteriores al bloque]
+ bloque` con la base (centro/escala) **CONGELADA** del train, devolviendo solo la parte
del bloque. Como el CUSUM solo mira atrás, el estado al inicio del bloque refleja la
historia real → causal y continuo entre folds. **Verificado (notebook §2):** ocultar el
futuro (2008 dentro de 2008–2010) NO cambia ni un día del bloque (`ndiff=0`).

**Ventana (LARGA).** `returns = log(SP500/SP500.shift(1)).dropna()`, índice 1985-01 →
2026-06. Walk-forward expanding, `train_size = 252×8` (~8 años), `step = 21`.
**ventana_eval = 1993-03 → 2026-06 (n≈8278)** → **2008 y 2011 son OOS**, a diferencia de
D4 (atado a HYG desde 2007). step=21 va sobrado en coste (~45 s por walk-forward).

### Descubierto

**Orientación VERIFICADA en walk-forward (no solo in-sample).** Pasando `market_returns`
a `walk_forward` Y a `evaluate`, el estado canónico crisis = `crisis_state = 1` tiene
**mayor σ de retornos OOS** que calma (≈0.0158 vs ≈0.0082) → **NO invertido**. Con el
núcleo vol-primario (Arreglo 4) el riesgo de inversión de un detector que separa por vol
queda neutralizado. **Sin warning de fallback peligroso**: X contiene `SP500_ret`
(columna reconocida como retorno real), así que el etiquetado provisional dentro de `fit`
usa el retorno correcto; dentro de `walk_forward` el aviso se silencia y se re-fija el
orden con `market_returns`. (El único warning posible es el advisory benigno de
"recognized-column proxy", idéntico al de D6, no la variante peligrosa que invierte.)

**Cobertura por crisis (CAUSAL OOS, coste robusto):**

| Ventana | Cobertura OOS | Lectura |
|---|---:|---|
| GFC_2008 | **100 %** | OOS gracias al histórico largo; CUSUM entra y se mantiene |
| EuroDebt_2011 | **~67 %** | OOS, captada (con algún tramo de salida/reentrada) |
| COVID_2020 | **~84 %** | el crash dispara C⁺ rápido |
| Inflation_2022 | **~77 %** | bear market de tipos, vol sostenida |
| TaperTantrum_2013 (trampa) | **0 %** | el robusto NO se dispara (poca vol equity) |
| Selloff_Q4_2018 (trampa) | **0 %** | el robusto NO se dispara |

**Lead/lag SOSTENIDO (persist=3) vs los troughs — DETECCIÓN TEMPRANA confirmada.** El
`lead_lag` del núcleo (cruce de p_crisis≥0.5 mantenido ≥3 días) es **negativo en las 4
crisis** (la señal de crisis está activa **bien antes** del suelo de drawdown; en varias
ventanas alcanza el tope del `lookback`=252 → ≥1 año de antelación). Es exactamente la
"detección temprana" que predice CP2. Matiz honesto: parte de esa antelación es porque el
detector también marca como crisis la fase de subida de vol previa (alto recall, a costa
de precisión global — ver false_alarm_rate).

**Switching / flickering (bueno).** `switching_rate ≈ 0.002`, **duración media ≈ 430 días
global** (episodios de crisis de decenas–cientos de días), `label_stability = 1.0`. El
umbral `h` del CUSUM actúa como histéresis natural → **NO flickea** por outliers en el
coste robusto. Comparable en estabilidad a D6 (GARCH).

**Falsas alarmas por outliers Y si el coste robusto ayuda (núcleo de CP2).** Resultado
**nítido**:
- **Coste GAUSSIANO (retorno², L2): degenera.** La desv de `r²` está inflada por la
  kurtosis (25–40), de modo que la base no distingue calma de crisis: el autómata entra en
  crisis con el primer shock (1987) y **no consigue salir** → **alarma casi permanente**
  (frac_crisis≈0.93, switching≈0, **trampas 2013 y 2018 al 100 %**, false_alarm_rate≈0.94).
  Es la materialización del "riesgo de falsas alarmas con outliers" de CP2.
- **Coste ROBUSTO (log|r| + mediana/MAD + winsor): lo arregla.** Recupera regímenes
  limpios — **trampas 2013/2018 ≈ 0 %**, switching bajo, episodios largos — manteniendo
  cobertura plena de 2008/2020 y buena de 2011/2022. → **el coste robusto SÍ reduce las
  falsas alarmas**, como pide CP2 ("preferir kernel/robusto").

**false_alarm_rate ≈ 0.87 (alto, mismo matiz que D6).** Precio de la ventana larga + señal
de vol: marca crisis muchos picos de alta vol reales **fuera** de las 4 ventanas canónicas
(1987, LTCM 1998, dotcom 2000–02, flash-crash 2010, 2015–16, SVB 2023…). No es flickering
día-a-día (switching bajísimo) sino episodios de alta vol legítimos que el ground-truth
laxo de 4 ventanas cuenta como falsa alarma. Léase junto a la cobertura.

**Oráculo PELT (offline, NO causal).** `ruptures.Pelt(model='l2')` sobre la log-vol
localiza los change-points "ideales" con información perfecta; sirve para ver que las
conmutaciones ONLINE del CUSUM caen cerca pero **con retardo** (precio de la causalidad).
Solo comparación visual; NUNCA entra en la evaluación.

### Hipótesis del CHECKPOINT 2 para D7 — veredicto

> *"Detección **temprana** (lead/lag) pero **riesgo de falsas alarmas con outliers**;
> **preferir kernel/robusto** frente al CUSUM **gaussiano**."*

**Se CUMPLE.** ✓ **Online causal** (CUSUM de Page; causalidad verificada, PELT solo como
oráculo). ✓ **Detección temprana**: lead/lag sostenido **negativo en las 4 crisis**. ✓
**Falsas alarmas con outliers**: el coste **gaussiano** (r², L2) se deja arrastrar por las
colas y degenera en alarma permanente (trampas al 100 %, far≈0.94). ✓ **El coste robusto
las reduce**: log|r| + mediana/MAD + winsor → trampas ≈0 %, regímenes limpios con cobertura
intacta. Conclusión: para datos financieros de cola gorda, **el CUSUM gaussiano sobre la
varianza es inutilizable y el coste robusto es claramente preferible** — exactamente la
recomendación de CP2.

### Fricción con el núcleo

- **Ninguna que obligara a tocar `src/`.** El núcleo cubrió bien a D7: (1) el etiquetado
  **vol-primario (Arreglo 4)** ordena correctamente un detector que separa por vol, sin
  inversión y sin necesidad de override (igual que D6); (2) la métrica `lead_lag` con
  `persist=3` del núcleo es justo la que mide la "detección temprana SOSTENIDA" de CP2.
- **(Observación, no bloqueante)** `false_alarm_rate` con ground-truth laxo (4 ventanas)
  penaliza como falsa alarma toda alta vol fuera de ellas; para un detector de change-point
  de vol con histórico largo infla la métrica. Mismo comentario que D6: no es defecto del
  núcleo, pero conviene leerla junto a la cobertura, no aislada.
- **(Patrón reutilizado)** El burn-in causal por bloque (concatenar train previo + bloque
  con la base congelada) es el mismo patrón que D6 (GARCH). Útil que el contrato del núcleo
  (`predict_online(test)` con detector nuevo por fold) admita este patrón sin cambios.

<a id="d08"></a>

## D8 — `hmm_tstudent` (FASE 3, Tanda 3) · Familia F3 (HMM avanzado)

> Configuración y resultados v2: [F3 — Modelos ocultos de Markov (HMM)](../fase1/F3_hmm.md#d08).

> HMM con **emisiones t-Student multivariantes** (colas pesadas) y **K estados por
> BIC**. La mejora directa sobre el baseline gaussiano D4: ataca de frente las fat
> tails del EDA (kurtosis 25-40). Código: `src/regimenes/detectores/f3_hmm/hmm_tstudent.py` (+
> `src/regimenes/detectores/f3_hmm/_hmm_t_utils.py`, filtrado forward t) · Notebook:
> `capa1_exploracion/notebooks/08_hmm_tstudent.ipynb` (v1, tag `capa1-final`) · Métricas: `docs/historia/capa1/resultados/metrics_08_hmm_tstudent.csv`.

### Implementado

**Modelo.** HMM con emisiones **t-Student multivariantes** (location mᵢ, matriz de
escala Sᵢ y grados de libertad **νᵢ** por estado), estimado por EM con variable de
escala latente (cada t-Student = mezcla gaussiana de escala continua). Se eligió la
t propia frente a GMM-HMM porque: (1) **BIC justo** — la t añade solo `k` parámetros
(un ν por estado) sobre el gaussiano, mientras un GMM multiplicaría medias/covarianzas
por nº de mezclas y dispararía el BIC; (2) robustez sin sobre-parametrizar.
Bibliografía: `hmm_bulla2011, hmm_rabiner1989, guidolintimmermann2007, hamilton1989`.

**Causalidad — filtrado forward t.** `predict_online`/`predict_proba` usan filtrado
forward causal con la **emisión t** (en `src/regimenes/detectores/f3_hmm/_hmm_t_utils.py`, adaptando el
patrón de D4 `_hmm_utils` cuya emisión era gaussiana), con contexto de burn-in del
train. Viterbi solo en la versión in-sample marcada NO causal. **No se modificó
`_hmm_utils.py` ni `src/`.**

**Features/ventana = D4 (BIC comparable).** Las mismas 7 features puente
(`BRIDGE_FEATURES`), ventana 2007+, `train_size=252*5` → como D4, 2008/2011 caen en el
train (NaN OOS). Así el BIC de D8 (t) es directamente comparable con el de D4
(gaussiano) sobre los mismos datos. market_returns = retorno log S&P 500.

**Coste.** El EM-t es caro (Baum-Welch + actualización de ν por fold). `step=126`
(refit ~semestral), declarado; el walk-forward tarda ~5 min. K seleccionado fuera del
walk-forward (fits in-sample para BIC).

### Descubierto

**K=4 por BIC (y AIC).** Grid {3,4}: BIC k=3 = 28103, **k=4 = 24416** → se despliega
**K=4** (calma · corrección leve · corrección · crisis). Los ν por estado canónico son
**decrecientes [10.2, 7.6, 4.2, 2.4]**: el estado de crisis tiene ν≈2.4 (colas muy
pesadas), el de calma ν≈10 (casi gaussiano). Económicamente impecable: la crisis es
donde viven los outliers.

**BIC vs D4 — la t-Student MEJORA el ajuste con holgura:**

| | n_states | logL | n_params | BIC |
|---|---:|---:|---:|---:|
| D4 gaussiano | 2 | −17381 | ~73 | **35379** |
| D8 t-Student | 4 | −11536 | 159 | **24416** |

**ΔBIC ≈ +10 963 a favor de D8** pese a tener más del doble de parámetros: el supuesto
gaussiano de D4 pagaba un coste enorme por no modelar las colas. Es la confirmación
cuantitativa de la crítica del EDA (kurtosis 25-40) y del estado del arte
(`hmm_bulla2011`).

**Orden de estados MONÓTONO en severidad** (no solo "no invertido"). Verificado
**en walk-forward** (con market_returns): vol anualizada por estado canónico OOS =
[10.1%, 13.8%, 21.0%, 37.1%] → estrictamente creciente 0→3; crisis (3) = mayor vol.
`monotonía vol walk-forward = True`, sin warning de fallback. Con K=4 el binning por
bandas de `VOL_CLOSE_FRAC` del Arreglo 4 ordena bien los 4 estados.

**Cobertura (OOS causal).** ventana 2012-2026 → 2008/2011 NaN (en train, como D4).
COVID_2020 = 66% en el estado CRISIS, Inflation_2022 = 33% en crisis. **Matiz
importante de multi-estado**: con K=4 el estado "crisis" es el más EXTREMO y estrecho;
el estrés más amplio cae en "corrección" (estado K−2). Sumando corrección+crisis,
Q4-2018 = 81%. Por eso `cov_COVID`/`cov_Inflation` de D8 parecen MÁS BAJAS que las de
D4 (2 estados, crisis ancha): **no es que D8 detecte peor, es que su "crisis" es la
cola extrema**. Hay que leer su fila del master con esta lente (la comparación justa de
"estrés" sería corrección+crisis).

*(Lectura v1. En v2 este matiz depende de la pista: en la pista A el estado de crisis OOS de D8 es el más poblado y no una cola estrecha; en la B sí es estrecho. Ver `notebooks/08_familia_F3_hmm.ipynb` §1.3 y §4.6. El taper de 2013 es en v2 una trampa: no activarse en él es correcto, no un punto ciego.)*

**¿Captó 2013/2018?** Activación del estado crisis: 2013 = **0%**, 2018 = 3.4%
(corrección+crisis: 2013 = 0%, 2018 = 81%). Es decir, capta 2018 como corrección pero
**NO 2013**. La t-Student y los 4 estados mejoran el ajuste y separan corrección de
crisis, pero **2013 sigue siendo invisible**: el taper fue un shock de tipos sin
volatilidad de equity, y D8 usa solo features equity/crédito (las mismas que D4). Es
un punto ciego de las FEATURES, no del supuesto distribucional → lo debe tapar D10
(multivariante con tipos/dólar).

**Flickering.** switching 0.052, duración media 19.1 d, persistencia esperada por
estado [calma 31d, leve 18d, corrección 38d, crisis 83d]. Más persistente que el GMM
(D3, 0.126) y similar a D4 causal (0.100).

### Hipótesis del CHECKPOINT 2 para D8 — veredicto

> *"Emisiones t + más estados atacan fat tails y POTENCIALMENTE captan 2013/2018
> donde el gaussiano falla; riesgo de sobreajuste con pocas obs por estado."*

**Se cumple a medias, con un matiz nítido.** (1) Fat tails: **SÍ, rotundo** — ΔBIC
+10963 vs D4, ν decreciente hasta 2.4 en crisis. (2) Más estados: SÍ, separa
calma/corrección/crisis de forma monótona y económicamente coherente. (3)
**¿2013/2018? NO desbloquea 2013** (sigue invisible: es shock de tipos, no de vol
equity; límite de las features, no de la t); 2018 se capta como corrección. Conclusión:
la t-Student arregla el problema DISTRIBUCIONAL (colas) que tenía D4, pero el agujero
de 2013 es de COBERTURA DE FEATURES, no distribucional → no lo cierra D8. (4)
Sobreajuste: con 159 params y ~212 días en el estado crisis OOS hay menos soporte por
estado, pero el BIC (que penaliza params) sigue prefiriéndolo con holgura, así que el
ajuste extra está justificado.

### Fricción con el núcleo

Ninguna que obligue a tocar `src/`. El filtrado forward t se resolvió en el detector
(`_hmm_t_utils.py`) reusando el patrón de D4. El Arreglo 4 (vol-primario) ordenó
correctamente los 4 estados (monotonía verificada). Observación de lectura, no de
núcleo: para detectores multi-estado, `cov_<crisis>` mide solo el estado más extremo;
al comparar contra detectores de 2 estados conviene mirar también "corrección+crisis"
(lo hace el notebook), no solo la columna del master.

<a id="d09"></a>

## D9 — `jump_model` (FASE 3, Tanda 4 · exploratoria) · Familia F2↔F3 (clustering con persistencia)

> Configuración y resultados v2: [F2 — Clustering](../fase1/F2_clustering.md#d09).

> Statistical Jump Model (Nystrup et al.): clustering de estados con **penalización
> de salto** λ → histéresis "aprendida", persistencia, online. Rival honesto de D3
> (GMM sin persistencia) y de D12 (AE). Código: `src/regimenes/detectores/f2_clustering/jump_model.py` · Notebook:
> `capa1_exploracion/notebooks/09_jump_model.ipynb` (v1, tag `capa1-final`) · Métricas: `docs/historia/capa1/resultados/metrics_09_jump_model.csv`.

### Implementado

**Modelo.** Librería `jumpmodels` v0.1.1 (`JumpModel`, SJM discreto), `n_states=2`,
**jump_penalty λ=50**. El SJM minimiza distancia intra-cluster + λ·(nº de saltos de
estado) → penaliza cambiar de régimen, induciendo persistencia sin la matriz de
transición de un HMM. `StandardScaler` ajustado **solo con el train** dentro de `fit`
(causal, re-fit por fold) porque 3 de las 15 features (corr, drawdown, momentum) tienen
escala distinta a los z. Mismas 15 features causales y ventana 2007+ que **D3** (es su
rival directo). Bibliografía: `hmm_nystrup2020, hmm_nystrup2017, clust_munnix2012`.

**Causalidad.** `jumpmodels` expone `predict_online` causal (la etiqueta de la fila i
usa solo filas < i). Verificado en el notebook: **0/120 etiquetas del bloque cambian
al añadir futuro**. market_returns a walk_forward Y evaluate.

**Orientación (Arreglo 4).** Verificada en walk-forward: crisis (estado 1) = vol 0.0218
vs calma 0.0089, retorno negativo → **no invertido**, sin warning de fallback.

### Descubierto

**ventana_eval: 2015-09 → 2026-06 (n=2649).** 15 features de 2007 → 2008/2011 en el
train (NaN OOS), como D3.

**La hipótesis anti-flickering se cumple ROTUNDAMENTE.** Comparado con su rival D3
(GMM clustering sin persistencia):

| | switching_rate | duración media | cov COVID | cov Inflation |
|---|---:|---:|---:|---:|
| D3 clustering_gmm_k3 | 0.126 | 7.9 d | 0.96 | 0.87 |
| **D9 jump_model (λ=50)** | **0.005** | **176.6 d** | 0.72 | 0.17 |

La penalización de salto reduce el flickering **~24×** (0.126→0.005) y multiplica la
duración de los episodios ×22 — exactamente lo que prometía Nystrup: histéresis
aprendida sin dinámica de Markov explícita.

**Pero hay un coste claro de sensibilidad.** Con λ=50 (alto), D9 se vuelve tan
persistente que **pierde cobertura de las crisis más lentas/menos extremas**: COVID
72% (vs 96% de D3) e **Inflación 2022 solo 17%** (vs 87% de D3). El bear market lento
de 2022 no genera un salto suficientemente nítido como para vencer la penalización, así
que D9 se queda en "calma" gran parte de él. fa_2018 = 0% (no se dispara en la trampa),
false_alarm_rate 0.62, label_stability ≈0.98.

**Lectura honesta del trade-off.** D9 NO domina a D3: cambia muchísimo flickering por
cobertura. Es el extremo "ultra-persistente" del eje clustering. Su λ es un mando
directo sobre ese trade-off (λ bajo → se parece a D3; λ alto → ultra-persistente y
ciego a estrés suave). Para el TFM es valioso como demostración de que la persistencia
se puede imponer explícitamente, y como punto de comparación frente al GMM (sin
persistencia) y al AE (D12).

### Hipótesis del CHECKPOINT 2 para D9 — veredicto

> *"Histéresis aprendida: estados persistentes, online, menos flickering, drawdowns
> suaves; rival honesto del AE+clustering en muestra pequeña."*

**Se cumple en lo esencial, con matiz.** Persistencia/online/menos flickering: SÍ,
rotundo (switching 0.005). "Drawdowns suaves": a costa de **perder cobertura de crisis
lentas** (Inflación 17%) — la persistencia fuerte es un arma de doble filo. Como rival
del AE (D12): D9 es mucho más limpio (switching 0.005 vs 0.287 del AE) — el jump model
gana de calle al deep como forma de imponer persistencia con pocos datos.

### Fricción con el núcleo

Ninguna; no se tocó `src/`. El Arreglo 4 (vol-primario) orientó D9 sin parche
(candidato a inversión por separar en features; no la hubo). Observación menor (ya
conocida): `walk_forward` tiene `min_train=252*5` que domina sobre `train_size`
pequeños — irrelevante aquí (train=252×8).

<a id="d10"></a>

## D10 — `turbulence_mahalanobis` (FASE 3, Tanda 3) · Familia F1 (multivariante)

> Configuración y resultados v2: [F1 — Reglas y umbrales](../fase1/F1_reglas.md#d10).

> Índice de **turbulencia financiera** (Kritzman, Page & Turkington 2012): distancia
> de Mahalanobis multivariante con covarianza **expanding causal**. Mide el colapso de
> correlaciones / "rareza" del vector de mercado respecto a su historia.
> Código: `src/regimenes/detectores/f1_reglas/turbulence_mahalanobis.py` · Notebook:
> `capa1_exploracion/notebooks/10_turbulence_mahalanobis.ipynb` (v1, tag `capa1-final`) · Métricas:
> `docs/historia/capa1/resultados/metrics_10_turbulence_mahalanobis.csv`.

### Implementado

**Modelo.** Turbulencia `d_t = (x_t − μ)ᵀ Σ⁻¹ (x_t − μ)`, con μ y Σ estimados de forma
**CAUSAL expanding** (datos < t). Estado por umbral causal sobre `d_t`: percentiles del
train τ_in=p90 / τ_out=p70 con histéresis + dwell. 2 estados (0=calma, 1=crisis).
Bibliografía: `kritzman2012`, `gulko2002`.

**Features / ventana — 2013 OOS (el contraste clave).** Vector multivariante de
4 cambios causales desde **1990**: `[SP500_ret, VIX_change, DXY_change,
yield_slope_chg]` (equity, miedo, dólar, tipos). Se eligió este set de histórico LARGO
(no las 15 features de 2007, que incluyen HYG) precisamente para que **2013, 2008, 2011
sean OOS**. Ventana efectiva OOS = **1998-06-02 → 2026-06-12 (n=6987)**; confirmado que
2013 < OOS-start es falso, i.e. 2013 cae dentro del OOS. market_returns = retorno log
S&P 500.

### Descubierto

**Orientación verificada (Arreglo 4 funciona).** Turbulencia media por estado:
**crisis = 11.30 vs calma = 2.13** → crisis = ALTA turbulencia = alta vol de retornos,
**NO invertido**, sin warning de fallback. El núcleo vol-primario orientó bien un
detector que separa por varianza/turbulencia (era candidato a inversión; no la hubo).

**Cobertura (CAUSAL OOS):**

| Ventana | Cobertura | |
|---|---:|---|
| GFC_2008 | 82.2 % | sólida (2008 OOS) |
| EuroDebt_2011 | 48.2 % | parcial |
| COVID_2020 | 76.0 % | buena |
| Inflation_2022 | 43.1 % | floja |
| TaperTantrum_2013 (trampa) | 12.3 % | apenas se enciende |
| Selloff_Q4_2018 (trampa) | 30.2 % | parcial |

**El test estrella (2013) NO se cumple como esperaba la hipótesis.** La premisa del
CP2 era que la turbulencia multivariante captaría el colapso de correlaciones de 2013
que los univariantes no ven. Resultado: **2013 marca solo 12.3 %** — esencialmente NO
se enciende, igual que D6 (GARCH equity, ~11 %) y D4 (HMM gaussiano, que tampoco lo
veía). Conclusión honesta: **el taper de 2013 NO fue un evento de turbulencia
multivariante** en este espacio de 4 features; fue una repreciación ordenada de tipos
sin "rareza" conjunta de equity/vol/dólar/curva. Añadir la curva al Mahalanobis no lo
ilumina. **2013 sigue siendo el punto ciego universal del banco** (D1 0%, D5 3.8%,
D6 11%, D7 0%, D8 0%, D10 12%).

**Tensión conceptual a resolver (para FASE 4 / decisión del usuario).** El marco actual
clasifica 2013 como **ventana-TRAMPA** (`FALSE_POSITIVE_WINDOWS`): firmar crisis ahí
cuenta como FALSO POSITIVO, así que `fa_2013 = 12.3 %` es en realidad BUENO
(especificidad). Pero el prompt de D10 lo planteaba como un evento DESEABLE de captar
("el agujero que D10 debería tapar"). Las dos lecturas chocan: ¿2013 es una crisis
rápida que un buen detector debe ver, o una trampa que no debe disparar? Los datos
zanjan la cuestión empírica (2013 no es turbulencia conjunta), pero la **etiqueta
del marco** (crisis vs trampa) es una decisión que conviene revisar en la síntesis.

**Flickering.** switching 0.087, duración media 11.4 d (305 episodios de crisis,
~5 d cada uno): **flickea más** que D6/D7 (la turbulencia es ruidosa día a día pese a
la histéresis). `false_alarm_rate` = 0.815 (alto, como todos los de histórico largo:
ve LTCM 1998, dotcom, 2010, 2015-16, 2023 fuera de las 4 ventanas canónicas).
`label_stability` = 1.000.

### Hipótesis del CHECKPOINT 2 para D10 — veredicto

> *"Capta el colapso de correlaciones multivariante que las reglas univariantes no
> ven."*

**Se cumple solo en parte.** SÍ capta los eventos sistémicos donde las correlaciones
sí colapsan (GFC 82 %, COVID 76 %) usando un único índice multivariante barato y
causal — eso valida el mecanismo de Kritzman. Pero **NO** captura 2013, que era el
contraste que lo justificaba frente a D4/D6: porque 2013 simplemente **no fue** un
episodio de turbulencia conjunta. La contribución real de D10 no es "tapar 2013", sino
ofrecer una señal de estrés sistémico multivariante de muy bajo coste; su debilidad es
el flickering y que no añade nada sobre el agujero de 2013.

### Fricción con el núcleo

Ninguna. El Arreglo 4 (vol-primario) orientó D10 correctamente sin necesidad de parche
local (a diferencia de lo que D6 tuvo que hacer antes del Arreglo 4): verificado
crisis = alta turbulencia en walk-forward, sin warning de fallback. Confirma que el
Arreglo 4 cubre el caso de los detectores que separan en varianza.

<a id="d11"></a>

## D11 — `msgarch_regime` (FASE 3, Tanda 4 · exploratoria) · Familia F5 (RS-GARCH)

> Configuración y resultados v2: [F5 — Volatilidad y GARCH](../fase1/F5_volatilidad_garch.md#d11).

> MS-GARCH(1,1)-t de Haas-Mittnik-Paolella (2004), **implementado desde cero en
> numpy/scipy** (filtro de Hamilton propio), SIN R ni rpy2. **RESULTADO NEGATIVO
> documentado**: implementable y causal, pero su asignación de regímenes degenera en
> walk-forward y NO sirve como detector. Código: `src/regimenes/detectores/f5_garch/msgarch_regime.py` ·
> Notebook: `capa1_exploracion/notebooks/11_msgarch_regime.ipynb` (v1, tag `capa1-final`) · Métricas:
> `docs/historia/capa1/resultados/metrics_11_msgarch_regime.csv`.

### Decisión de implementar (no declararlo fuera)

El CP2 avisó: "no hay librería madura de MS-GARCH en Python". Eso aplica al MS-GARCH
*naive* path-dependiente (verosimilitud sobre 2^t trayectorias, intratable). Pero la
variante **HMP-2004 SÍ es tratable en Python puro**: K recursiones GARCH en PARALELO
(cada una alimentada solo por su propia varianza pasada → sin path dependence),
verosimilitud por filtro de Hamilton en O(T·K²). El subagente la implementó como modelo
genuino (no un apaño), lo cual era lo honesto. Bibliografía:
`vol_haasmittnikpaolella2004, vol_gray1996, vol_marcucci2005`.

### Implementado

**Modelo.** 2 regímenes, cada uno GARCH(1,1)-t (retornos×100); μ única, ν compartido por
parsimonia. Estimación ML (scipy) con filtro de Hamilton. **Walk-forward**: rolling 6
años, `step=126` (semestral), `n_init=1, maxiter=100` para acotar el coste del ML no
convexo (~13-36 min según contención). Dentro de cada bloque el posterior de crisis
filtrado es **diario y causal** (verificado: `max|P(crisis) ver vs ocultar futuro| =
0.0`). market_returns a walk_forward Y evaluate.

**In-sample (n_init=3, maxiter=300) el modelo SÍ es sensato**: régimen calma α+β=0.997
(ω=0.002), régimen crisis α=0.21/ω=0.042 (más reactivo), ν=5.29 (colas gordas),
transiciones persistentes (p00=0.996, p11=0.995). Orientación in-sample correcta
(crisis = alta vol).

### Descubierto — PATOLOGÍA en walk-forward (resultado negativo)

**Las métricas OOS son aberrantes para un modelo de volatilidad:**

| | valor | esperado en un vol-model |
|---|---:|---|
| cov_GFC_2008 | **0.0 %** | debería ser ~100% (mayor vol de la muestra) |
| cov_COVID_2020 | 20 % | alto |
| cov_Inflation_2022 | 36 % | medio-alto |
| fa_Selloff_Q4_2018 (trampa) | **93.7 %** | bajo |
| false_alarm_rate | **0.949** | bajo-medio |

**Diagnóstico (causa raíz, no es inversión del núcleo).** En el fold rolling de 6 años
que predice la GFC (sept-2008), el modelo decodifica **100 % régimen 0** en todo el
train: **nunca visita el régimen de crisis**. El GARCH casi-integrado del régimen
dominante (α+β=0.997) absorbe toda la volatilidad, dejando el 2º régimen **muerto**.
Como el régimen crisis no se observa, el núcleo (vol-primario, Arreglo 4) lo deja como
estado canónico 1 sin soporte → durante la GFC la "crisis" se dispara 0 %. En otros
folds el régimen 1 SÍ se activa, pero en días no-crisis (de ahí far 0.95 y fa_2018
0.94). Es decir: **asignación de regímenes degenerada e inestable entre folds**, típica
del ML no convexo del MS-GARCH con poca multistart, agravada por regímenes casi-IGARCH.

**No es un fallo del núcleo ni una inversión simple.** El Arreglo 4 orientó bien lo que
había; el problema es que el MODELO no entrega dos regímenes estables en walk-forward
con `n_init=1`. Subir a `n_init=3, maxiter=300` por fold (como el in-sample) podría
estabilizarlo, pero multiplicaría un coste ya alto (~36 min) por ~3 → inviable y, sobre
todo, **innecesario**: el subagente ya anticipó que si D11 rendía mal, **D6
`garch_t_vol` cubre limpiamente el hueco** de la señal de vol con colas (GFC 100 %,
COVID 94 %) sin la fragilidad del cambio de régimen.

### Hipótesis del CHECKPOINT 2 para D11 — veredicto

> *"Extensión natural (heteroscedasticidad GARCH dentro de cada régimen + prob.
> filtrada); coste alto y fragilidad (path dependence, óptimos locales,
> label-switching); sin librería madura en Python."*

**Se cumple la parte de la FRAGILIDAD, que era el riesgo declarado.** Es implementable
y causal (mérito), pero los óptimos locales y la degeneración de regímenes hacen que su
crisis OOS NO rastree las crisis reales (GFC 0 %). Como detector **no es usable**; como
ejercicio, confirma empíricamente la advertencia del estado del arte. **Se mantiene en
el master como resultado negativo documentado** (no se excluye): su fila es la evidencia
de la patología.

### Fricción con el núcleo

Ninguna en `src/`. El núcleo orientó correctamente los estados observados; la
degeneración es del modelo MS-GARCH, no del marco. Observación transversal útil para la
síntesis: detectores con **estados que pueden quedar sin visitar** en un fold producen
un `crisis_state` canónico "vacío" → cobertura 0 espuria; conviene vigilar en FASE 4
los casos de regímenes degenerados (frac de un estado ≈ 0).

<a id="d12"></a>

## D12 — `deep_ae_regime` (FASE 3, Tanda 4 · exploratoria) · Familia F7 (redes neuronales)

> Configuración y resultados v2: [F7 — Redes neuronales](../fase1/F7_redes_neuronales.md#d12).

> Autoencoder ligero → GMM sobre el latente, como **contraste ablativo honesto** frente
> al baseline lineal PCA→GMM. Objetivo: ¿aporta la no linealidad del AE algo con ~4
> crisis? Código: `src/regimenes/detectores/f7_deep/deep_ae_regime.py` (`DeepAERegime` + `PCAGMMBaseline`) ·
> Notebook: `capa1_exploracion/notebooks/12_deep_ae_regime.ipynb` (v1, tag `capa1-final`) · Métricas:
> `docs/historia/capa1/resultados/metrics_12_deep_ae_regime.csv` (2 filas: AE y baseline PCA).

### Implementado

**Modelo.** Autoencoder denso PyTorch (15→8→2→8→15, ReLU + dropout 0.10 + weight-decay
1e-3, 40 épocas, Adam, semillas fijas) que comprime las 15 features causales a un
**latente 2D**; encima, `GaussianMixture` full con **K=3** (mismo esquema que D3 pero no
lineal). Estandarización **causal por fold** (μ/σ del train). Se expone también
`reconstruction_error()` como score de anomalía (complemento, no principal).
Bibliografía: `nn_kingma2014, nn_akioyamen2021, nn_bucci2021, lopezdeprado2018`.

**Baseline ablativo.** Clase hermana `PCAGMMBaseline` (PCA 2D → GMM K=3) en el mismo
fichero, pasa por el MISMO `walk_forward` → comparación limpia AE-no-lineal vs
PCA-lineal con idéntica dim latente, K y ventana.

**Causalidad (con nota — se corrigió una aserción).** El encoder es puntual (eval() +
no_grad + scaler congelado del train). La verificación inicial del notebook usaba
`assert maxdiff_latente < 1e-9`, que **fallaba por ruido de coma flotante de torch**
(BLAS sobre tensores de distinto tamaño en float32 da ~2.4e-7), NO por look-ahead. Se
corrigió la aserción al criterio REAL de causalidad: **0/247 estados del bloque cambian
al ocultar el futuro** (y maxdiff latente < 1e-4, tolerancia FP). Confirmado causal.
market_returns a walk_forward Y evaluate.

**Orientación (Arreglo 4).** Verificada en walk-forward: crisis (estado 2) vol 0.0275
vs calma 0.0070; PCA igual (crisis vol 0.0524 vs 0.0066). No invertido, sin fallback.

### Descubierto — RESULTADO NEGATIVO (esperado y válido)

**ventana_eval: 2015-09 → 2026-06 (n=2649)** (15 features de 2007 → 2008/2011 NaN OOS).

**Contraste ablativo AE→GMM vs PCA→GMM (mismo K, dim, ventana):**

| | cov COVID | cov Inflation | fa 2018 | false_alarm_rate | switching | dur |
|---|---:|---:|---:|---:|---:|---:|
| **D12 AE→GMM** | 0.54 | 0.10 | 0.15 | 0.60 | **0.287** | 3.5 d |
| PCA→GMM (lineal) | 0.62 | 0.005 | 0.00 | 0.14 | 0.091 | 10.9 d |

**El AE NO mejora al baseline lineal — al contrario, lo empeora.** La no linealidad
**añade flickering** (switching 0.287 vs 0.091, ~3×) **y falsas alarmas** (far 0.60 vs
0.14) **sin ganar cobertura** (COVID 0.54 vs 0.62). Es el resultado que anticipaba el
CHECKPOINT 2: con ~4 crisis reales no hay señal suficiente para que la capacidad extra
del AE generalice; solo ajusta ruido y produce un latente más inestable que la PCA.

**Valor del hallazgo (negativo pero útil).** La conclusión metodológica es limpia: en
este problema y con estos datos, **un reductor lineal (PCA) es preferible a un AE** como
front-end de clustering de regímenes. El deep learning solo se justificaría con muchos
más datos o features de alta frecuencia (intradía), como ya señaló el estado del arte.
No se sobre-optimizó para "ganar": el resultado es la comparación honesta.

### Hipótesis del CHECKPOINT 2 para D12 — veredicto

> *"Contraste ablativo honesto; con ~4 crisis el resultado negativo es aceptable; el
> deep solo se justificaría con más datos."*

**Se cumple exactamente.** El AE no aporta sobre PCA (resultado negativo), lo cual es
una contribución válida: documenta que la complejidad deep no está justificada aquí. La
comparación se hizo limpia (mismo pipeline) y sin sobreajustar para inflar el AE.

### Fricción con el núcleo

Ninguna en `src/`. Única incidencia: la aserción de causalidad del notebook era
demasiado estricta (`<1e-9`) para float32 de torch; se relajó a tolerancia FP + chequeo
de igualdad de estados (el criterio correcto). El detector es causal; no se tocó el
núcleo. Aprendizaje transversal: **las verificaciones de causalidad de detectores deep
deben comprobar invariancia de ESTADOS/probabilidades, no igualdad exacta del latente
en float** (el FP de torch da ~1e-7 inocuo).
