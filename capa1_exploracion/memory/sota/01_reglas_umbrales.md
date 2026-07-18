# 01 — Reglas / Umbrales

> Familia de detectores NO estadísticos / heurísticos: el régimen se decide
> comparando una o varias variables observables (VIX, drawdown, spread de
> crédito, pendiente de curva, momentum) contra umbrales fijos o adaptativos.
> Son los candidatos naturales a **baseline** del banco de pruebas.

## Definición y supuestos

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
  expanding/rolling — exactamente el diseño de las 15 features del proyecto.
- **El umbral es el (hiper)parámetro**: o se fija por convención de mercado
  (VIX 20/30) o se calibra (percentil histórico, grid out-of-sample). La
  calibración es el punto donde puede colarse look-ahead si se elige `τ` mirando
  toda la muestra; debe fijarse con datos pasados (expanding) o por convención.

## Variantes principales

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

## Fortalezas y debilidades

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
la kurtosis de exceso de 25.6 (S&P) y 39.6 (HYG)— las reglas son **robustas a fat
tails**: de hecho los outliers son justo la señal que quieren capturar, no ruido
a modelar. Limitación: el umbral "óptimo" no es estacionario (el "nivel normal"
de VIX cambia entre décadas), de ahí la preferencia por umbrales
adaptativos/z-score sobre niveles fijos.

**¿Causal/online o necesita toda la muestra?** Causal y online si la
normalización es expanding/rolling (caso de las 15 features). No necesita la
muestra completa para inferir — a diferencia del HMM/Viterbi smoothed, que usa el
futuro. Único riesgo de look-ahead: **elegir el umbral** mirando todo el
histórico; se neutraliza fijándolo por convención (VIX>30) o calibrándolo
expanding/walk-forward.

**Otras debilidades:** univariante por defecto (ignora interacciones — el cambio
de signo de la correlación S&P/bonos no lo capta un umbral simple sobre VIX);
sensible a la elección de τ; no da probabilidad ni cuantifica incertidumbre (es
duro 0/1, salvo que se construya un score); puede sobre-ajustarse si se tunea el
umbral contra eventos conocidos.

## Idoneidad para este proyecto

**Es un baseline imprescindible**, por tres razones atadas al EDA y a la tarea
previa:

1. **Robustez a fat tails.** Con kurtosis de exceso de 25.6 (S&P) y 39.6 (HYG),
   el supuesto gaussiano del HMM es frágil; un umbral no asume distribución y
   trata los outliers como señal. Una regla bien diseñada es el contraste justo
   para cuantificar cuánto pierde realmente el HMM por la mala especificación.

2. **Captura de crisis rápidas.** El HMM previo acertó 2008 y 2020 pero falló
   2013 y 2018. Reglas sobre `VIX_level_z`, `SP500_drawdown` y `credit_spread_z`
   reaccionan en el día y son la hipótesis directa para no perder esas
   correcciones rápidas. El banco de pruebas puede así separar "qué eventos sólo
   se capturan con reactividad" de "cuáles necesitan modelo estadístico".

3. **Features ya disponibles y causales.** Las 15 features encajan 1:1 con las
   variantes: `VIX_level_z`/`MOVE_level_z` (variante 1), `SP500_vol_z` (2),
   `SP500_drawdown`/`SP500_momentum` (3), `credit_spread_z` (4), `yield_slope_z`
   (5). Implementar las reglas es casi gratis: comparaciones sobre columnas que ya
   pasaron el test de no look-ahead (`max_abs_diff=0.0`).

Matiz importante (cambio de signo de la correlación): una regla univariante NO
capta que la correlación S&P/bonos cambia de signo entre regímenes (Gulko 2002
[`gulko2002`]); para eso conviene una **regla compuesta** (voto de varios
umbrales) o pasar a la familia estadística multivariante. Es decir, reglas como
baseline sólido y techo de referencia, no como detector final.

Sobre la ventana (2007-04 → 2026-06): las reglas no sufren el problema de
"entrenamiento" que tensiona el walk-forward (la GFC 2008 pegada al inicio de
datos), porque un umbral por convención no necesita train. Esto las hace además
útiles para **evaluar 2008 out-of-sample** cuando los detectores que sí entrenan
no pueden. Un umbral sobre VIX/SP500 (disponibles desde 1990) podría incluso
extender la evaluación a DotCom 2002, fuera de la ventana común.

## Aplicaciones documentadas a regímenes de mercado financiero

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

## Coste de implementación y librería Python recomendada

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

## Referencias

- Bloom, N. (2009). *The Impact of Uncertainty Shocks*. Econometrica, 77(3),
  623–685. DOI: 10.3982/ECTA6248.
- Estrella, A. & Hardouvelis, G. A. (1991). *The Term Structure as a Predictor of
  Real Economic Activity*. The Journal of Finance, 46(2), 555–576. DOI:
  10.1111/j.1540-6261.1991.tb02674.x.
- Estrella, A. & Mishkin, F. S. (1998). *Predicting U.S. Recessions: Financial
  Variables as Leading Indicators*. Review of Economics and Statistics, 80(1),
  45–61. DOI: 10.1162/003465398557320. [clave existente: `estrellamishkin1998`]
- Faber, M. T. (2007). *A Quantitative Approach to Tactical Asset Allocation*. The
  Journal of Wealth Management, 9(4), 69–79. DOI: 10.3905/jwm.2007.674809.
- Gilchrist, S. & Zakrajšek, E. (2012). *Credit Spreads and Business Cycle
  Fluctuations*. American Economic Review, 102(4), 1692–1720. DOI:
  10.1257/aer.102.4.1692.
- Gulko, L. (2002). *Decoupling*. Journal of Portfolio Management, 28(3), 59–66.
  DOI: 10.3905/jpm.2002.319843. [clave existente: `gulko2002`]
- Hamilton, J. D. (1989). *A New Approach to the Economic Analysis of
  Nonstationary Time Series and the Business Cycle*. Econometrica, 57(2),
  357–384. DOI: 10.2307/1912559. [clave existente: `hamilton1989`]
- Kritzman, M., Page, S. & Turkington, D. (2012). *Regime Shifts: Implications for
  Dynamic Strategies*. Financial Analysts Journal, 68(3), 22–39. DOI:
  10.2469/faj.v68.n3.3. [clave existente: `kritzman2012`]
- Moreira, A. & Muir, T. (2017). *Volatility-Managed Portfolios*. The Journal of
  Finance, 72(4), 1611–1644. DOI: 10.1111/jofi.12513.
- Pagan, A. R. & Sossounov, K. A. (2003). *A Simple Framework for Analysing Bull
  and Bear Markets*. Journal of Applied Econometrics, 18(1), 23–46. DOI:
  10.1002/jae.664.
- Sahm, C. (2019). *Direct Stimulus Payments to Individuals*. En Boushey, Nunn &
  Shambaugh (eds.), *Recession Ready: Fiscal Policies to Stabilize the American
  Economy*. The Hamilton Project / Brookings Institution, pp. 67–92. URL:
  https://www.brookings.edu/articles/direct-stimulus-payments-to-individuals/
  (serie FRED: SAHMREALTIME).
- Tong, H. (1990). *Non-Linear Time Series: A Dynamical System Approach*. Oxford
  University Press. ISBN: 9780198522249. (modelos de umbral TAR/SETAR).

## Candidatas adicionales (para el sintetizador)

Familias detectadas durante la búsqueda que NO son reglas/umbrales y que deben
desarrollarse en su memoria correspondiente (sólo se anotan aquí):

- **Markov-switching / HMM (estado latente).** Núcleo de la tarea previa
  (`hamilton1989`); emisiones t-Student/mixturas para fat tails. Es la familia de
  contraste directa al baseline de reglas.
- **Change-point detection / segmentación retrospectiva y online.** `ruptures`,
  CUSUM, Bayesian online change-point (Adams & MacKay). Prima de la regla con
  histéresis pero con base estadística de cambio de media/varianza.
- **GARCH y regímenes de volatilidad (clustering).** Umbralizar la vol
  condicional GARCH/EWMA; puente entre regla sobre vol y modelo estadístico.
- **Statistical Jump Models (clustering temporal con penalización de saltos).**
  Aparecieron como alternativa reciente al HMM con drawdowns más suaves
  (arXiv:2402.05272); penalizan transiciones, lo que es la versión "aprendida" de
  la histéresis.
- **Índice de turbulencia / Mahalanobis multivariante (Kritzman et al. 2012).**
  Aunque se umbraliza, su construcción es multivariante-estadística; encaja mejor
  como método de distancia/anomalía que como regla simple.
- **Clasificadores ML supervisados (logística/árboles/RL) sobre features de
  régimen.** Varias referencias (preprints VIX-spike, RegimeFolio) entrenan
  clasificadores; familia ML, fuera de alcance aquí.
