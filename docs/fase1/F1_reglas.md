# F1 — Reglas y umbrales

> **Cómo leer este documento.** La sección 1 es la teoría de la familia (redactada en la Capa 1, original en el tag `capa1-final`); la 2, los detectores implementados con su configuración v2 y sus resultados del benchmark (ADR-003): [D01 `rule_vix_threshold`](#d01), [D02 `rule_composite_riskoff`](#d02), [D10 `turbulence_mahalanobis`](#d10). Notebook de familia: [`notebooks/05_familia_F1_reglas.ipynb`](../../notebooks/05_familia_F1_reglas.ipynb). Bibliografía: claves `reglas_*` y compartidas de [`docs/references.bib`](../references.bib).
>
> **Datos v1 en la teoría.** Las cifras del EDA que cita la sección 1 son de la Capa 1 (15 features, ventana 2007-04 → 2026-06, 4 crisis: 2008, 2011, 2020 y 2022), igual que los nombres de feature `credit_spread_z` o `yield_slope_z` y las referencias a la «tarea previa»; donde el dato importa se indica junto a él. El panel v2 tiene dos pistas (A: 1962-01-02 → 2026-05-29; B: 2007-04-11 → 2026-05-29) y el EDA vigente está en [`eda.md`](eda.md) (curtosis de exceso del S&P 500 = 18,7 en 1928-2026).

## 1. Teoría

> Familia de detectores NO estadísticos / heurísticos: el régimen se decide
> comparando una o varias variables observables (VIX, drawdown, spread de
> crédito, pendiente de curva, momentum) contra umbrales fijos o adaptativos.
> Son los candidatos naturales a **baseline** del banco de pruebas.

### Definición y supuestos

Un detector de reglas/umbrales asigna el régimen en el instante `t` mediante una
función explícita y determinista de variables observables hasta `t`:

```
estado_t = crisis   si  g(x_t) > τ
estado_t = calma    en otro caso
```

donde `g(·)` es típicamente la identidad sobre un nivel (VIX), un z-score, un
drawdown o una combinación lógica (`VIX>τ1 AND drawdown<τ2`). Formalmente es el
caso degenerado/observable de un modelo de cambio de régimen con variable de
transición conocida: el **Self-Exciting Threshold AutoRegressive (SETAR / TAR)**
de Tong (1978, 1990), donde el estado no es latente sino que se dispara cuando
una variable cruza un umbral. Esto lo diferencia del HMM/Markov-switching de
Hamilton (1989) [`hamilton1989`], donde el estado es **no observable** y se
infiere por verosimilitud.

Supuestos clave (y por qué importan aquí):
- **No asume una distribución generadora** de los retornos: no hay
  verosimilitud gaussiana ni t-Student que estimar. El umbral encapsula todo el
  "modelo". Esto es una ventaja directa frente al supuesto de normalidad del HMM
  gaussiano, criticado por las fat tails del EDA.
- **Causalidad/online por construcción**: `g(x_t)` sólo usa información hasta
  `t`. No requiere ajustar nada con la muestra completa (a diferencia del HMM,
  que con `fit/predict` sobre todo el panel introduce look-ahead si no se hace
  walk-forward). Un umbral fijo (VIX>30) es trivialmente causal; un umbral
  adaptativo (z-score expanding, percentil rolling) lo es si la normalización es
  expanding/rolling — exactamente el diseño de las features causales del proyecto
  (`src/regimenes/features/`).
- **El umbral es el (hiper)parámetro**: o se fija por convención de mercado
  (VIX 20/30) o se calibra (percentil histórico, grid out-of-sample). La
  calibración es el punto donde puede colarse look-ahead si se elige `τ` mirando
  toda la muestra; debe fijarse con datos pasados (expanding) o por convención.

### Variantes principales

1. **Umbral de nivel sobre volatilidad implícita (VIX/MOVE).** La regla más
   extendida en la práctica: bandas VIX <15 (calma), 15–20 (normal), 20–30
   (elevado), >30 (crisis). Respaldo académico en que los picos de VIX marcan
   shocks de incertidumbre con efecto real (Bloom 2009 [`reglas_bloom2009`]).
   Variante multi-banda → régimen ordinal (3–4 estados) en vez de binario.

2. **Volatility/Vol-target timing.** Reducir exposición cuando la volatilidad
   (realizada o implícita) supera su media; Moreira & Muir (2017)
   [`reglas_moreiramuir2017`] muestran que escalar inversamente a la varianza
   reciente mejora ratios de Sharpe. Caso continuo del umbral: en vez de
   crisis/calma, peso ∝ 1/σ²; se discretiza fácilmente en régimen con un corte.

3. **Reglas de drawdown / trend (momentum de precio).** Salir del activo de
   riesgo cuando el drawdown desde máximos supera un umbral, o cuando el precio
   cae por debajo de una media móvil. El modelo canónico es Faber (2007)
   [`reglas_faber2007`]: largo si precio > media móvil de 10 meses, si no a
   liquidez; "equity-like returns with bond-like drawdowns". El drawdown como
   disparador conecta directamente con la feature `SP500_drawdown` y los
   `DRAWDOWN_TROUGHS` ya cableados.

4. **Umbral de spread de crédito.** Régimen de estrés cuando el spread HY se
   amplía por encima de un nivel/percentil. Gilchrist & Zakrajšek (2012)
   [`reglas_gilchristzakrajsek2012`] muestran que el componente "excess bond
   premium" del spread predice contracciones; conecta con la feature
   `credit_spread_z` (HYG−IEF) y con el alto co-movimiento HYG/equity (+0.67) del
   EDA.

5. **Umbral de pendiente de curva (term spread).** Señal de recesión cuando la
   pendiente 10Y−3M se invierte (cruza 0) o cae bajo un umbral. Estrella &
   Mishkin (1998) [`estrellamishkin1998`] y Estrella & Hardouvelis (1991)
   [`reglas_estrellahardouvelis1991`] documentan el poder predictivo; conecta con
   `yield_slope_z`. Es señal **lead** (anticipa con meses), complementaria a las
   coincidentes (VIX, drawdown).

6. **Reglas de fecha/algoritmos de dating.** Pagan & Sossounov (2003)
   [`reglas_pagansossounov2003`] adaptan el algoritmo Bry–Boschan para fechar
   bull/bear por reglas (mínima duración de fase, magnitud), sin suavizar ni
   eliminar outliers. Es rule-based puro y sirve como **etiquetado de referencia**
   ex-post (cuidado: la versión clásica usa ventana centrada → no causal; existe
   variante online).

7. **Reglas con persistencia / contador (estilo Sahm).** La Sahm rule (2019)
   [`reglas_sahm2019`] dispara recesión cuando la media 3m del paro sube ≥0.5pp
   sobre su mínimo de 12m: combina umbral + suavizado + memoria, y es **real-time
   por diseño**. Plantilla de cómo construir una regla causal y robusta a ruido.

8. **Reglas compuestas / scores de "risk-on/risk-off".** Combinación lógica o
   por voto de varios umbrales (VIX z-score + spread z-score + slope + breadth),
   tal como hacen los "regime indicators" prácticos de gestoras. Kritzman, Page &
   Turkington (2012) [`kritzman2012`] formalizan un índice de turbulencia
   (Mahalanobis) que, umbralizado, define régimen de estrés — puente entre regla
   simple y método estadístico multivariante.

### Fortalezas y debilidades

**¿Capta crisis rápidas?** Sí — es precisamente su ventaja sobre el HMM de la
tarea previa. Un umbral sobre VIX o drawdown reacciona **el mismo día** que la
variable cruza el corte, sin esperar a que la verosimilitud acumule evidencia
para reasignar el estado. Esto ataca directamente el fallo documentado del HMM
gaussiano (se perdía el taper-tantrum 2013 y el Q4-2018): una regla
`VIX_level_z>τ OR drawdown<−τ2` habría marcado ambos. El coste es el espejo:
mayor tasa de **falsos positivos** ante picos efímeros.

**Flickering/parpadeo y su mitigación.** El talón de Aquiles. Cuando la variable
oscila cerca del umbral, el estado parpadea (calma↔crisis) generando
transiciones espurias. Mitigaciones estándar (todas causales):
- **Histéresis / doble umbral (banda muerta):** entrar en crisis exige cruzar
  `τ_alto` pero sólo se sale al bajar de `τ_bajo < τ_alto` (analogía del
  termostato). Reduce drásticamente el whipsaw sin retrasar la entrada.
- **Persistencia / dwell-time mínimo:** exigir que la condición se cumpla `k`
  días seguidos, o imponer duración mínima de fase (Pagan–Sossounov).
- **Suavizado previo:** umbralizar una media móvil/EWMA de la señal (Sahm
  suaviza a 3m; Faber usa media de 10m) en vez del valor instantáneo.
- **Contador con memoria** (estilo Sahm: nivel vs mínimo móvil).
Hay un trade-off explícito: más histéresis/persistencia ⇒ menos parpadeo pero
**más retraso** en señalar el giro. El banco de pruebas debe medir ambos
(latencia de detección vs nº de transiciones / falsos positivos en 2013 y 2018).

**¿Asume normalidad?** No. No hay supuesto distribucional; el umbral funciona
igual de bien (o mal) con colas gordas. Frente al HMM gaussiano —cuestionado por
la curtosis de exceso de 25.6 (S&P) y 39.6 (HYG) de la ventana v1 2007-2026, 18,7
para el S&P 500 en 1928-2026 según el EDA v2— las reglas son **robustas a fat
tails**: de hecho los outliers son justo la señal que quieren capturar, no ruido
a modelar. Limitación: el umbral "óptimo" no es estacionario (el "nivel normal"
de VIX cambia entre décadas), de ahí la preferencia por umbrales
adaptativos/z-score sobre niveles fijos.

**¿Causal/online o necesita toda la muestra?** Causal y online si la
normalización es expanding/rolling (caso de todas las features del proyecto). No necesita la
muestra completa para inferir — a diferencia del HMM/Viterbi smoothed, que usa el
futuro. Único riesgo de look-ahead: **elegir el umbral** mirando todo el
histórico; se neutraliza fijándolo por convención (VIX>30) o calibrándolo
expanding/walk-forward.

**Otras debilidades:** univariante por defecto (ignora interacciones — el cambio
de signo de la correlación S&P/bonos no lo capta un umbral simple sobre VIX);
sensible a la elección de τ; no da probabilidad ni cuantifica incertidumbre (es
duro 0/1, salvo que se construya un score); puede sobre-ajustarse si se tunea el
umbral contra eventos conocidos.

### Idoneidad para este proyecto

**Es un baseline imprescindible**, por tres razones atadas al EDA y a la tarea
previa:

1. **Robustez a fat tails.** Con curtosis de exceso de 25.6 (S&P) y 39.6 (HYG) en la
   ventana v1 2007-2026 (18,7 en el S&P 500 1928-2026, EDA v2),
   el supuesto gaussiano del HMM es frágil; un umbral no asume distribución y
   trata los outliers como señal. Una regla bien diseñada es el contraste justo
   para cuantificar cuánto pierde realmente el HMM por la mala especificación.

2. **Captura de crisis rápidas.** El HMM previo acertó 2008 y 2020 pero falló
   2013 y 2018. Reglas sobre `VIX_level_z`, `SP500_drawdown` y `credit_spread_z`
   reaccionan en el día y son la hipótesis directa para no perder esas
   correcciones rápidas. El banco de pruebas puede así separar "qué eventos sólo
   se capturan con reactividad" de "cuáles necesitan modelo estadístico".

3. **Features ya disponibles y causales.** Las 15 features de la Capa 1 encajaban 1:1 con las
   variantes: `VIX_level_z`/`MOVE_level_z` (variante 1), `SP500_vol_z` (2),
   `SP500_drawdown`/`SP500_momentum` (3), `credit_spread_z` (4), `yield_slope_z`
   (5). Implementar las reglas es casi gratis: comparaciones sobre columnas que ya
   pasaron el test de no look-ahead (`max_abs_diff=0.0`). En v2 los equivalentes por
   pista son los de D01 y D02 (sección 2): `SP500_vol_z`, `credit_BaaAaa_mensual_z` y
   `term_spread_hist_z` en A; `VIX_level_z`, `credit_BAA10Y_z` y `slope_10y2y_z` en B.

Matiz importante (cambio de signo de la correlación): una regla univariante NO
capta que la correlación S&P/bonos cambia de signo entre regímenes (Gulko 2002
[`gulko2002`]); para eso conviene una **regla compuesta** (voto de varios
umbrales) o pasar a la familia estadística multivariante. Es decir, reglas como
baseline sólido y techo de referencia, no como detector final.

Sobre la ventana (v1: 2007-04 → 2026-06; en v2 solo la pista B arranca en 2007-04): las reglas no sufren el problema de
"entrenamiento" que tensiona el walk-forward (la GFC 2008 pegada al inicio de
datos), porque un umbral por convención no necesita train. Esto las hace además
útiles para **evaluar 2008 out-of-sample** cuando los detectores que sí entrenan
no pueden. Un umbral sobre VIX/SP500 (disponibles desde 1990) podría incluso
extender la evaluación a DotCom 2002, fuera de la ventana común.

### Aplicaciones documentadas a regímenes de mercado financiero

- **VIX como termómetro de régimen / incertidumbre:** Bloom (2009)
  [`reglas_bloom2009`] usa picos de VIX para fechar shocks de incertidumbre con
  caída-rebote de actividad real; base teórica del umbral VIX > nivel.
- **Timing por volatilidad:** Moreira & Muir (2017) [`reglas_moreiramuir2017`]
  reducen exposición cuando sube la varianza realizada en múltiples factores
  (market, value, momentum, carry) con ganancias de Sharpe — evidencia de que
  umbralizar la volatilidad mejora resultados ajustados a riesgo.
- **Drawdown/trend defensivo:** Faber (2007) [`reglas_faber2007`] (media móvil
  10m sobre múltiples clases de activo) como regla de salida que recorta
  drawdowns manteniendo retorno; aplicación directa de regla de tendencia/umbral.
- **Spread de crédito como señal de ciclo:** Gilchrist & Zakrajšek (2012)
  [`reglas_gilchristzakrajsek2012`]; el excess bond premium umbralizado se usa en
  notas de la Fed como indicador de riesgo de recesión.
- **Pendiente de curva:** Estrella & Mishkin (1998) [`estrellamishkin1998`] y
  Estrella & Hardouvelis (1991) [`reglas_estrellahardouvelis1991`]: inversión de
  la curva como regla líder de recesión.
- **Dating rule-based de bull/bear:** Pagan & Sossounov (2003)
  [`reglas_pagansossounov2003`] aplican reglas (Bry–Boschan) al S&P para fechar
  mercados alcistas/bajistas.
- **Regla real-time de ciclo:** Sahm (2019) [`reglas_sahm2019`]; umbral+suavizado
  +memoria sobre el paro, adoptado por la Fed de St. Louis (serie FRED).
- **Índice de turbulencia umbralizado:** Kritzman, Page & Turkington (2012)
  [`kritzman2012`] definen régimen de estrés cortando la distancia de
  Mahalanobis multivariante.

### Coste de implementación y librería Python recomendada

**Coste: muy bajo.** Las reglas son comparaciones vectorizadas sobre el
`features.parquet` ya existente. No hay ajuste iterativo (segundos, no minutos).

- **NumPy / pandas:** núcleo de todo. Umbral fijo (`df['VIX_level_z'] > tau`),
  z-scores/percentiles rolling (`.rolling()/.expanding().quantile()`), drawdown
  (`cummax`), media móvil (Faber). Histéresis y persistencia se implementan con un
  pequeño bucle de estado o `np.where` + `ffill` sobre las dos condiciones de
  entrada/salida (banda muerta).
- **statsmodels:** si se quiere el primo estadístico de la regla, los modelos
  SETAR/TAR de umbral (transición conocida) y Markov-switching están en
  `statsmodels.tsa` (`MarkovRegression`) para comparar regla dura vs umbral
  estimado.
- **scipy.signal:** útil para suavizados (filtfilt no causal — evitar; usar EWMA
  causal de pandas) y para detección de histéresis tipo Schmitt.
- **(Opcional) ruptures:** detección de puntos de cambio; relevante sólo si se
  quiere comparar la regla con change-point — anótese para la familia
  correspondiente, no es necesaria aquí.

Recomendación: implementar la familia entera con **pandas + NumPy** y encapsular
la histéresis en una función reutilizable `apply_hysteresis(signal, tau_in,
tau_out, min_dwell)`. Cero dependencias nuevas respecto al stack actual.

## 2. Detectores implementados

Las tablas de resultados salen de `results/benchmark/ranking_v2.csv` y `results/benchmark/metrics_master_v2.csv` (benchmark completo 24/24, `results/benchmark/manifest.json` del 2026-09-29T12:41 UTC; cifras de [ADR-003](../decisions/ADR-003-causalidad-calendario-estado-ranking.md) §4). Walk-forward causal y evaluación OOS por pista (17 crisis evaluables en la A y 9 en la B, según `configs/benchmark_spec.yaml`). Ranking de **detección**: un evento se detecta con ≥ 3 sesiones OOS consecutivas en crisis dentro de [pico, suelo]; `score_deteccion` = F1 entre la precisión diaria (1 − FAR) y el recall por evento, con niveles previos (elegible › precisión ≤ azar › parpadeo › degenerado). Cada tabla va entre marcadores `resultados_v2`: tras un benchmark nuevo se regenera desde esos dos CSV, no se edita a mano.

<a id="d01"></a>

### D01 — `rule_vix_threshold` · regla de volatilidad / VIX

Autómata causal de dos estados (calma/crisis) sobre una sola señal de estrés, con **histéresis** (entra al cruzar `τ_in` por arriba y solo sale al bajar de `τ_out < τ_in`) y **permanencia mínima** (`min_dwell`). Los umbrales son percentiles de la señal calculados **solo con el train** de cada fold; el estado en `t` depende del estado en `t−1` y de la señal en `t`. Es el baseline reactivo de la familia.

**Configuración v2**

- Código: `src/regimenes/detectores/f1_reglas/rule_vix_threshold.py` (`RuleVixThreshold`); registro `regimenes.detectores.registry`, id `D01`.
- Señal por pista: **A** `SP500_vol_z` (volatilidad realizada; variante declarada, porque el VIX no existe en la pista histórica); **B** `VIX_level_z` (regla VIX original).
- Parámetros: `q_in = 0.90`, `q_out = 0.70`, `min_dwell = 5` (defectos del constructor).
- Walk-forward: refit cada 21 sesiones, ventana expanding.

<!-- BEGIN resultados_v2 D01 -->
**Resultados v2 (ADR-003)**

| Métrica | Pista A | Pista B |
|---|---:|---:|
| Variante evaluada | `rule_vix_threshold` | `rule_vix_threshold` |
| Puesto de detección | **2** / 12 | **3** / 12 |
| Nivel de etiqueta | `elegible` | `elegible` |
| `score_deteccion` | 0.587 | 0.505 |
| Recall por evento | 0.941 (16/17) | 0.667 (6/9) |
| Precisión diaria (lift vs base) | 0.427 (×2.20) | 0.406 (×2.35) |
| Cobertura media de crisis | 0.603 | 0.254 |
| Tasa de falsas alarmas (FAR) | 0.573 | 0.594 |
| Activación media en trampas | 0.000 | 0.000 |
| Switching rate | 0.0074 | 0.0069 |
| Duración media de régimen (sesiones) | 133.3 | 140.0 |
| Estabilidad de etiqueta | 1.000 | 1.000 |
| Puesto legacy (`rank_medio`) | 1 (3.6) | 2 (4.2) |

**Ventana OOS y eventos por pista**

- **Pista A** · OOS 1970-05-12 → 2026-05-29 (n = 14133) · train inicial 2016 sesiones, refit cada 21 (expanding) · 1 feature(s): `SP500_vol_z` · variante: A: umbral de volatilidad realizada; VIX no existe en la pista histórica.
  - Detectados (16): `bear_1969_70`, `oil_stagflation_1973`, `volcker_1980_82`, `black_monday_1987`, `gulf_war_sl_1990`, `ltcm_russia_1998`, `dotcom_2000_02`, `gfc_2007_09`, `flash_crash_euro1_2010`, `us_downgrade_euro2_2011`, `china_oil_2015_16`, `fed_tightening_2018q4`, `covid_2020`, `inflation_bear_2022`, `svb_banking_2023`, `tariff_selloff_2025`.
  - No detectados (1): `volmageddon_2018q1`.
- **Pista B** · OOS 2010-04-12 → 2026-05-29 (n = 4059) · train inicial 756 sesiones, refit cada 21 (expanding) · 1 feature(s): `VIX_level_z` · variante: B: regla VIX original.
  - Detectados (6): `us_downgrade_euro2_2011`, `china_oil_2015_16`, `volmageddon_2018q1`, `covid_2020`, `inflation_bear_2022`, `tariff_selloff_2025`.
  - No detectados (3): `flash_crash_euro1_2010`, `fed_tightening_2018q4`, `svb_banking_2023`.
<!-- END resultados_v2 D01 -->

Narrativa de la Capa 1 (implementación v1, hallazgos e hipótesis del CHECKPOINT 2): [`historia/fichas_v1.md`](../historia/fichas_v1.md#d01).

<a id="d02"></a>

### D02 — `rule_composite_riskoff` · regla compuesta risk-off

**Score de risk-off** = media ponderada de cuatro señales orientadas (signo · valor, de modo que alto = estrés) y re-estandarizadas con μ/σ **del train**; el score se umbraliza con el mismo autómata de histéresis y permanencia mínima que D01. Es la versión multivariante (voto) de la regla simple.

**Configuración v2**

- Código: `src/regimenes/detectores/f1_reglas/rule_composite_riskoff.py` (`RuleCompositeRiskoff`); registro, id `D02`.
- Señales y signos por pista: **A** `SP500_vol_z` (+), `credit_BaaAaa_mensual_z` (+), `term_spread_hist_z` (−), `SP500_drawdown` (−); **B** `VIX_level_z` (+), `credit_BAA10Y_z` (+), `slope_10y2y_z` (−), `SP500_drawdown` (−). En la Capa 1 eran `VIX_level_z`, `credit_spread_z` (HYG−IEF), `yield_slope_z` y `SP500_drawdown`.
- Parámetros: pesos iguales, `q_in = 0.90`, `q_out = 0.70`, `min_dwell = 5`.
- Walk-forward: refit cada 21 sesiones, ventana expanding.

<!-- BEGIN resultados_v2 D02 -->
**Resultados v2 (ADR-003)**

| Métrica | Pista A | Pista B |
|---|---:|---:|
| Variante evaluada | `rule_composite_riskoff` | `rule_composite_riskoff` |
| Puesto de detección | **4** / 12 | **4** / 12 |
| Nivel de etiqueta | `elegible` | `elegible` |
| `score_deteccion` | 0.559 | 0.501 |
| Recall por evento | 0.706 (12/17) | 0.667 (6/9) |
| Precisión diaria (lift vs base) | 0.462 (×2.38) | 0.402 (×2.33) |
| Cobertura media de crisis | 0.446 | 0.294 |
| Tasa de falsas alarmas (FAR) | 0.538 | 0.598 |
| Activación media en trampas | 0.000 | 0.000 |
| Switching rate | 0.0026 | 0.0039 |
| Duración media de régimen (sesiones) | 371.9 | 238.8 |
| Estabilidad de etiqueta | 0.998 | 0.998 |
| Puesto legacy (`rank_medio`) | 3 (4.4) | 3 (4.4) |

**Ventana OOS y eventos por pista**

- **Pista A** · OOS 1970-05-12 → 2026-05-29 (n = 14133) · train inicial 2016 sesiones, refit cada 21 (expanding) · 4 feature(s): `SP500_vol_z|credit_BaaAaa_mensual_z|term_spread_hist_z|SP500_drawdown`.
  - Detectados (12): `bear_1969_70`, `oil_stagflation_1973`, `volcker_1980_82`, `black_monday_1987`, `dotcom_2000_02`, `gfc_2007_09`, `flash_crash_euro1_2010`, `us_downgrade_euro2_2011`, `covid_2020`, `inflation_bear_2022`, `svb_banking_2023`, `tariff_selloff_2025`.
  - No detectados (5): `gulf_war_sl_1990`, `ltcm_russia_1998`, `china_oil_2015_16`, `volmageddon_2018q1`, `fed_tightening_2018q4`.
- **Pista B** · OOS 2010-04-12 → 2026-05-29 (n = 4059) · train inicial 756 sesiones, refit cada 21 (expanding) · 4 feature(s): `VIX_level_z|credit_BAA10Y_z|slope_10y2y_z|SP500_drawdown`.
  - Detectados (6): `china_oil_2015_16`, `fed_tightening_2018q4`, `covid_2020`, `inflation_bear_2022`, `svb_banking_2023`, `tariff_selloff_2025`.
  - No detectados (3): `flash_crash_euro1_2010`, `us_downgrade_euro2_2011`, `volmageddon_2018q1`.
<!-- END resultados_v2 D02 -->

Narrativa de la Capa 1 (implementación v1, hallazgos e hipótesis del CHECKPOINT 2): [`historia/fichas_v1.md`](../historia/fichas_v1.md#d02).

<a id="d10"></a>

### D10 — `turbulence_mahalanobis` · turbulencia de Mahalanobis (regla multivariante)

Índice de **turbulencia financiera** de Kritzman, Page y Turkington [`kritzman2012`]: distancia de Mahalanobis `d_t = (x_t − μ)ᵀ Σ⁻¹ (x_t − μ)` del vector de mercado del día, con μ y Σ **expanding causales** (solo observaciones anteriores a `t`; el train se antepone como contexto al predecir cada bloque). Mide magnitudes extremas y co-movimientos atípicos (colapso de correlaciones) que una regla univariante no ve. El estado sale de umbralizar `d_t` con percentiles del train, histéresis y permanencia mínima.

**Configuración v2**

- Código: `src/regimenes/detectores/f1_reglas/turbulence_mahalanobis.py` (`TurbulenceMahalanobis`); registro, id `D10`.
- Features por pista: **A** `SP500_ret`, `SP500_vol_z`, `DGS10_change_z`, `FF_MKT_z`; **B** `SP500_ret`, `VIX_change_z`, `DXY_change_z`, `slope_10y2y_z`. En la Capa 1 eran cuatro cambios crudos desde 1990 (`SP500_ret`, `VIX_change`, `DXY_change`, `yield_slope_chg`).
- Parámetros: `q_in = 0.90`, `q_out = 0.70`, `min_dwell = 5`, `cov_min_periods = 252`, `ridge = 1e-6`.
- Walk-forward: refit cada 21 sesiones, ventana expanding.

<!-- BEGIN resultados_v2 D10 -->
**Resultados v2 (ADR-003)**

| Métrica | Pista A | Pista B |
|---|---:|---:|
| Variante evaluada | `turbulence_mahalanobis` | `turbulence_mahalanobis` |
| Puesto de detección | **1** / 12 | **6** / 12 |
| Nivel de etiqueta | `elegible` | `elegible` |
| `score_deteccion` | 0.614 | 0.457 |
| Recall por evento | 0.941 (16/17) | 1.000 (9/9) |
| Precisión diaria (lift vs base) | 0.455 (×2.35) | 0.296 (×1.71) |
| Cobertura media de crisis | 0.481 | 0.600 |
| Tasa de falsas alarmas (FAR) | 0.545 | 0.704 |
| Activación media en trampas | 0.000 | 0.047 |
| Switching rate | 0.0417 | 0.0567 |
| Duración media de régimen (sesiones) | 24.0 | 17.6 |
| Estabilidad de etiqueta | 0.999 | 0.996 |
| Puesto legacy (`rank_medio`) | 4 (5.4) | 8 (7.6) |

**Ventana OOS y eventos por pista**

- **Pista A** · OOS 1970-05-12 → 2026-05-29 (n = 14133) · train inicial 2016 sesiones, refit cada 21 (expanding) · 4 feature(s): `SP500_ret|SP500_vol_z|DGS10_change_z|FF_MKT_z`.
  - Detectados (16): `bear_1969_70`, `oil_stagflation_1973`, `volcker_1980_82`, `black_monday_1987`, `gulf_war_sl_1990`, `ltcm_russia_1998`, `dotcom_2000_02`, `gfc_2007_09`, `flash_crash_euro1_2010`, `us_downgrade_euro2_2011`, `china_oil_2015_16`, `volmageddon_2018q1`, `fed_tightening_2018q4`, `covid_2020`, `inflation_bear_2022`, `tariff_selloff_2025`.
  - No detectados (1): `svb_banking_2023`.
- **Pista B** · OOS 2010-04-12 → 2026-05-29 (n = 4059) · train inicial 756 sesiones, refit cada 21 (expanding) · 4 feature(s): `SP500_ret|VIX_change_z|DXY_change_z|slope_10y2y_z`.
  - Detectados (9): `flash_crash_euro1_2010`, `us_downgrade_euro2_2011`, `china_oil_2015_16`, `volmageddon_2018q1`, `fed_tightening_2018q4`, `covid_2020`, `inflation_bear_2022`, `svb_banking_2023`, `tariff_selloff_2025`.
  - No detectados (0): ninguno.
<!-- END resultados_v2 D10 -->

Narrativa de la Capa 1 (implementación v1, hallazgos e hipótesis del CHECKPOINT 2): [`historia/fichas_v1.md`](../historia/fichas_v1.md#d10).

## 3. Variantes no implementadas

- **Jump models estadísticos** (arXiv:2402.05272): clustering temporal que penaliza las transiciones; es la versión «aprendida» de la histéresis (D09, [F2](F2_clustering.md#d09)).
- **Clasificadores supervisados** (logística, árboles) sobre features de régimen: fuera de alcance, porque exigen etiquetas de régimen que el banco reserva para evaluar.

## 4. Referencias

- Bloom, N. (2009). *The Impact of Uncertainty Shocks*. Econometrica, 77(3),
  623–685. DOI: 10.3982/ECTA6248. [`reglas_bloom2009`]
- Estrella, A. & Hardouvelis, G. A. (1991). *The Term Structure as a Predictor of
  Real Economic Activity*. The Journal of Finance, 46(2), 555–576. DOI:
  10.1111/j.1540-6261.1991.tb02674.x. [`reglas_estrellahardouvelis1991`]
- Estrella, A. & Mishkin, F. S. (1998). *Predicting U.S. Recessions: Financial
  Variables as Leading Indicators*. Review of Economics and Statistics, 80(1),
  45–61. DOI: 10.1162/003465398557320. [`estrellamishkin1998`]
- Faber, M. T. (2007). *A Quantitative Approach to Tactical Asset Allocation*. The
  Journal of Wealth Management, 9(4), 69–79. DOI: 10.3905/jwm.2007.674809. [`reglas_faber2007`]
- Gilchrist, S. & Zakrajšek, E. (2012). *Credit Spreads and Business Cycle
  Fluctuations*. American Economic Review, 102(4), 1692–1720. DOI:
  10.1257/aer.102.4.1692. [`reglas_gilchristzakrajsek2012`]
- Gulko, L. (2002). *Decoupling*. Journal of Portfolio Management, 28(3), 59–66.
  DOI: 10.3905/jpm.2002.319843. [`gulko2002`]
- Hamilton, J. D. (1989). *A New Approach to the Economic Analysis of
  Nonstationary Time Series and the Business Cycle*. Econometrica, 57(2),
  357–384. DOI: 10.2307/1912559. [`hamilton1989`]
- Kritzman, M., Page, S. & Turkington, D. (2012). *Regime Shifts: Implications for
  Dynamic Strategies*. Financial Analysts Journal, 68(3), 22–39. DOI:
  10.2469/faj.v68.n3.3. [`kritzman2012`]
- Moreira, A. & Muir, T. (2017). *Volatility-Managed Portfolios*. The Journal of
  Finance, 72(4), 1611–1644. DOI: 10.1111/jofi.12513. [`reglas_moreiramuir2017`]
- Pagan, A. R. & Sossounov, K. A. (2003). *A Simple Framework for Analysing Bull
  and Bear Markets*. Journal of Applied Econometrics, 18(1), 23–46. DOI:
  10.1002/jae.664. [`reglas_pagansossounov2003`]
- Sahm, C. (2019). *Direct Stimulus Payments to Individuals*. En Boushey, Nunn &
  Shambaugh (eds.), *Recession Ready: Fiscal Policies to Stabilize the American
  Economy*. The Hamilton Project / Brookings Institution, pp. 67–92. URL:
  https://www.brookings.edu/articles/direct-stimulus-payments-to-individuals/
  (serie FRED: SAHMREALTIME). [`reglas_sahm2019`]
- Tong, H. (1990). *Non-Linear Time Series: A Dynamical System Approach*. Oxford
  University Press. ISBN: 9780198522249. (modelos de umbral TAR/SETAR). [`reglas_tong1990`]
