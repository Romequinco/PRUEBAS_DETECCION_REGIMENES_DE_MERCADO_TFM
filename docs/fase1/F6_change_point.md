# F6 — Change-point

> **Cómo leer este documento.** La sección 1 es la teoría de la familia (redactada en la Capa 1, original en el tag `capa1-final`); la 2, los detectores implementados con su configuración v2 y sus resultados del benchmark (ADR-003): [D07 `changepoint_online`](#d07). Notebook de familia: [`notebooks/11_familia_F6_changepoint.ipynb`](../../notebooks/11_familia_F6_changepoint.ipynb). Bibliografía: claves `cp_*` y compartidas de [`docs/references.bib`](../references.bib).
>
> **Datos v1 en la teoría.** Las cifras del EDA que cita la sección 1 son de la Capa 1 (15 features, ventana 2007-04 → 2026-06, 4 crisis: 2008, 2011, 2020 y 2022), igual que los nombres de feature `credit_spread_z` o `yield_slope_z` y las referencias a la «tarea previa»; donde el dato importa se indica junto a él. El panel v2 tiene dos pistas (A: 1962-01-02 → 2026-05-29; B: 2007-04-11 → 2026-05-29) y el EDA vigente está en [`eda.md`](eda.md) (curtosis de exceso del S&P 500 = 18,7 en 1928-2026).

## 1. Teoría

> Familia de la FASE 2 (estado del arte). Enfoque: detectar **instantes de cambio
> estructural** que separan tramos homogéneos, en vez de modelar estados latentes
> recurrentes. Referencias: claves `cp_*` de [`docs/references.bib`](../references.bib).

### Definición y supuestos

El **change-point detection** (CPD) busca los instantes \(t_1 < t_2 < \dots < t_k\)
en que las propiedades estadísticas de una serie temporal cambian de forma abrupta,
particionándola en \(k+1\) **segmentos** internamente homogéneos. Formalmente se
asume que dentro de cada segmento las observaciones siguen una distribución (o un
conjunto de parámetros) constante —media, varianza, distribución completa,
estructura de covarianza— y que esos parámetros saltan en los breakpoints. El
problema combina dos decisiones: **dónde** están los cambios y **cuántos** hay
(Truong, Oudre & Vayatis, 2020).

Supuestos habituales:

- **Homogeneidad por tramos**: cada segmento es estacionario; el cambio es un salto,
  no una transición suave.
- **Independencia / pre-blanqueo**: muchas formulaciones clásicas (CUSUM gaussiano,
  ICSS) suponen observaciones i.i.d. dentro del segmento. Con retornos financieros
  esto obliga a trabajar sobre residuos o features ya desestacionalizados.
- **Forma del cambio conocida**: se especifica *qué* cambia (media, varianza,
  distribución) eligiendo la función de coste; un coste mal elegido no ve el cambio
  relevante o dispara falsos.
- **Número de cambios**: o se fija \(k\), o se controla con una **penalización** que
  crece con el número de breakpoints (BIC, AIC, penalización lineal \(\beta\), o
  el coste por cambio de PELT).

Diferencia conceptual clave con HMM / Markov-switching: el CPD **segmenta**, no
etiqueta estados recurrentes. No dice "esto es el régimen crisis que ya vimos en
2008", solo dice "aquí hay una frontera entre dos tramos distintos". Para encajar
en la interfaz de detectores de este proyecto (estados `0..n-1`, crisis = el peor)
hay que **post-etiquetar** cada segmento por criterio económico (ver §Idoneidad).

### Variantes principales

#### CUSUM (Page, 1954) — online, secuencial
La suma acumulada de desviaciones respecto a un valor objetivo. Se mantiene un
estadístico \(S_t = \max(0,\, S_{t-1} + (x_t - \mu_0 - k))\) y se declara cambio
cuando \(S_t\) supera un umbral \(h\). Es el detector secuencial original de control
de calidad, **causal por construcción**: solo usa el pasado y dispara con cierto
**retardo de detección** tras el cambio real. Variante de varianza:
**ICSS** (Inclán & Tiao, 1994), suma acumulada de *cuadrados* para detectar cambios
en la varianza incondicional —muy usado en finanzas para quiebres de volatilidad.

#### BOCPD — Bayesian Online Change Point Detection (Adams & MacKay, 2007) — online
Estima en cada instante la distribución posterior del **run length** \(r_t\)
(tiempo transcurrido desde el último cambio) mediante un *message-passing*
recursivo: \(P(r_t \mid x_{1:t})\). Cada paso solo necesita el dato nuevo y el
posterior anterior, por lo que es **exactamente causal y online**. Usa una
distribución predictiva por segmento (típicamente de la familia exponencial con
prior conjugado) y un *hazard* que modela la probabilidad a priori de cambio. Su
fortaleza es que entrega **incertidumbre** (probabilidad de cambio) en tiempo real,
no solo un punto. Fearnhead & Liu (2007) proponen, en paralelo, filtrado online
exacto para múltiples change-points con coste reducible a lineal vía partículas.

#### `ruptures` — métodos principalmente OFFLINE (Truong et al., 2020)
La librería de referencia en Python organiza los métodos como combinación de
*(coste, método de búsqueda, restricción sobre nº de cambios)*:

- **PELT** (Pruned Exact Linear Time; Killick, Fearnhead & Eckley, 2012):
  minimización **exacta** del coste penalizado con una poda que da coste esperado
  **lineal** \(O(n)\) (peor caso \(O(n^2)\)). Requiere fijar la penalización
  \(\beta\); es el caballo de batalla para nº de cambios desconocido.
- **Binary Segmentation (BinSeg)**: greedy, detecta el cambio más fuerte, parte la
  serie y recurre. Rápido (\(O(n\log n)\)) y aproximado; puede colocar mal cambios
  cuando hay varios cercanos.
- **Window**: desliza dos ventanas adyacentes y mide la discrepancia de coste entre
  ellas; aproximado y barato, intuitivamente "online-like" pero en `ruptures` se
  aplica offline sobre la serie completa.
- **Dynamic Programming (Dynp)**: óptimo para un número de cambios \(k\) **fijo y
  conocido**, coste \(O(k n^2)\).
- **Kernel change-point** (Arlot, Celisse & Harchaoui, 2019; Harchaoui & Cappé,
  2007): mapea los datos a un RKHS y detecta cambios en la **distribución completa**
  (no solo media/varianza). Con un kernel RBF es **no paramétrico** y robusto, capaz
  de ver cambios donde media y varianza son constantes pero cambia la forma.

#### Coste / penalización del número de cambios
El número de breakpoints es un problema de **selección de modelo**: sin penalizar,
el óptimo coloca un cambio en cada punto (sobreajuste). Se controla con
penalización lineal \(\beta\) (PELT), BIC/AIC, o el coste por cambio en kernel CPD
(Arlot et al., 2019). **La calibración de \(\beta\)/umbral es la decisión crítica**:
gobierna directamente el *trade-off* entre detección temprana y falsas alarmas.

#### Online vs offline — distinción central para este proyecto
- **Online / secuencial (CAUSAL)**: CUSUM, ICSS secuencial, BOCPD, Fearnhead–Liu.
  Detectan el cambio **en cuanto va ocurriendo**, usando solo \(x_{1:t}\), con un
  **retardo de detección**. Son los apropiados para un marco walk-forward sin
  look-ahead.
- **Offline / retrospectivo (ANTI-CAUSAL)**: PELT, BinSeg, Dynp y, en la práctica,
  Window de `ruptures`. Colocan los breakpoints minimizando un coste sobre **toda la
  serie**, es decir **usan el futuro** para decidir dónde estuvo cada frontera. Tal
  cual, **violan la causalidad** del proyecto.
- **Rescate del offline en walk-forward**: un método offline puede usarse de forma
  causal si se **re-aplica en ventana expanding/rolling** y solo se confía en el
  *último* breakpoint detectado con datos hasta \(t\). Esto reintroduce el retardo y
  el coste (re-segmentar en cada paso), y puede provocar que un breakpoint "salte"
  de posición al añadir datos (inestabilidad retrospectiva). Es el camino para
  integrar PELT, pero hay que declarar el coste y el flickering que induce.

### Fortalezas y debilidades

**Fortalezas**
- **Detección temprana de crisis rápidas**: su punto fuerte. Un crash o un salto de
  volatilidad es exactamente un change-point en media/varianza; CUSUM y BOCPD lo
  señalan con poco retardo, lo que encaja con la métrica **lead/lag** del proyecto
  (anticipar el suelo del drawdown). En liquidez/volatilidad se han documentado
  *change-points que preceden* a los giros de precio.
- **Sin estados predefinidos**: no exige fijar el número de regímenes de antemano
  (a diferencia de HMM/k-means); el nº de segmentos emerge de la penalización.
- **Variantes no paramétricas robustas**: kernel CPD detecta cambios en la
  distribución completa sin asumir gaussianidad.
- **BOCPD entrega probabilidad de cambio** en tiempo real (incertidumbre explícita).

**Debilidades**
- **Supuesto gaussiano y fat tails**: CUSUM en media e ICSS en varianza asumen
  normalidad/i.i.d. Con la **curtosis de exceso ≈ 25.6** del S&P 500 (y ≈ 39.6 en
  HYG) del EDA v1 (2007-2026; 18,7 en 1928-2026 según el EDA v2), un único outlier leptocúrtico puede superar el umbral y disparar un
  **falso change-point**. Los métodos basados en momentos gaussianos son
  especialmente vulnerables; los **no paramétricos / kernel** (y costes robustos
  tipo Huber o sobre rangos) mitigan esto.
- **Flickering y falsas alarmas**: un umbral / \(\beta\) demasiado sensible genera
  **muchos breakpoints** y segmentos espurios — justo la patología que mide la
  métrica de **tasa de falsas alarmas y flickering** del proyecto. Se penaliza
  subiendo \(\beta\)/el umbral, imponiendo un tamaño mínimo de segmento
  (`min_size`), o añadiendo histéresis/persistencia. Hay tensión directa con la
  detección temprana: menos falsas alarmas ⇒ más retardo.
- **Retardo de detección (online)**: el lado causal cuesta latencia; el cambio se
  confirma algunos pasos después de ocurrir.
- **Causalidad del offline**: PELT/BinSeg/Dynp son anti-causales salvo
  reaplicación en ventana (con su coste e inestabilidad retrospectiva).
- **No reconoce regímenes recurrentes**: segmenta pero no etiqueta; requiere
  post-procesado económico (ver §Idoneidad).

### Idoneidad para este proyecto

1. **Variante relevante = online/secuencial.** El proyecto es causal y walk-forward,
   así que las piezas de primera clase son **CUSUM/ICSS** y **BOCPD**. PELT y el
   resto de `ruptures` solo entran como baseline offline *re-aplicado en ventana
   expanding*, declarando explícitamente que la versión "sobre toda la serie" es
   anti-causal y sirve únicamente de referencia/oráculo.

2. **EDA → elegir métodos robustos.** Dado el perfil de **fat tails** (curtosis
   ≈ 25.6 en SP500 en v1; 18,7 en 1928-2026), priorizar variantes **no paramétricas / kernel** o costes
   robustos frente al CUSUM gaussiano puro, para no confundir outliers con cambios
   de régimen. Conviene además operar sobre **features ya causales y
   z-scoreadas** (`VIX_level_z`, `SP500_vol_z`, `credit_spread_z`,
   `SP500_drawdown`), que están más cerca de la i.i.d. por tramos que los precios
   crudos. D07 sigue esta recomendación: su coste por defecto es robusto (`log|r|`,
   mediana/MAD y winsorización).

3. **Métrica lead/lag — donde puede lucir.** La detección temprana es la ventaja
   natural del CPD: medir el **lead/lag respecto al suelo del drawdown**
   (`DRAWDOWN_TROUGHS` de la Capa 1: 2009-03-09, 2011-10-03, 2020-03-23, 2022-10-12) es la prueba
   reina para esta familia. Un CUSUM/BOCPD bien calibrado debería **anticipar** o
   acompañar de cerca la entrada en crisis.

4. **Reto de integración — post-etiquetado económico.** El CPD produce segmentos sin
   nombre. Para encajar en la interfaz de detectores (estados `0..n-1`, **crisis =
   peor estado**) hay que **etiquetar cada segmento a posteriori** por un criterio
   económico **causal** (p. ej. signo del retorno medio del tramo, nivel de
   volatilidad/VIX, profundidad de drawdown), asignando el índice "crisis" al
   segmento de peor riesgo/retorno. Cuidado: ese etiquetado debe usar solo
   información disponible hasta \(t\) para no reintroducir look-ahead. Es el
   principal coste conceptual de adoptar esta familia.

5. **Control de flickering.** Calibrar \(\beta\)/umbral y `min_size` con la propia
   métrica de falsas alarmas/flickering del proyecto, evitando segmentaciones
   excesivas (especialmente alrededor de 2018/2013, las "trampas" / falsos positivos
   del `evaluation.py` de la Capa 1; hoy `regimenes.evaluacion`).

**Veredicto**: familia **complementaria y valiosa como detector temprano** (sobre
todo CUSUM/BOCPD online), no como clasificador de regímenes recurrentes. Encaja
bien como *baseline causal* y como *fuente de la métrica lead/lag*, con la cautela
de fat tails (preferir kernel/robusto) y el coste de post-etiquetado.

### Aplicaciones documentadas a regímenes de mercado financiero

- **Quiebres de varianza/volatilidad**: ICSS (Inclán & Tiao, 1994) es el estándar
  para detectar cambios en la varianza incondicional de retornos; ampliamente
  aplicado a series bursátiles para datar fases de alta/baja volatilidad.
- **Volatilidad de precios de activos**: Lavielle & Teyssière (2007) detectan
  múltiples change-points en la volatilidad de precios de activos (incl. procesos
  de memoria larga y multivariantes), datando episodios de estrés.
- **Segmentación de series financieras con PELT**: el método de Killick et al.
  (2012) se motiva explícitamente con datos donde el nº de cambios crece con \(n\)
  (genética, **finanzas**); estudios recientes lo aplican a series financieras para
  identificar regímenes y quiebres estructurales.
- **Filtrado online en finanzas**: tanto Adams & MacKay (2007) como Fearnhead & Liu
  (2007) citan finanzas entre las áreas de aplicación del CPD online; BOCPD se ha
  usado para señalización de cambios de régimen en tiempo real.
- **Topological / non-parametric CPD**: líneas recientes combinan TDA y kernels para
  detectar transiciones de régimen sin supuesto gaussiano (criterio del equipo:
  línea emergente, no madura como baseline).

### Coste de implementación y librería Python recomendada

- **`ruptures`** (Truong et al., 2020) — librería de referencia, madura y bien
  documentada. Cubre PELT, BinSeg, Window, Dynp y kernel CPD con API uniforme
  (`fit`/`predict`), múltiples costes (`l1`, `l2`, `rbf`, `normal`, `linear`) y
  parámetros `pen`/`n_bkps`/`min_size`. **Limitación**: es esencialmente
  **offline** — no expone un modo streaming nativo; el uso causal exige envolverlo
  en un bucle expanding/rolling (coste de recomputar por paso, O(n) por ventana).
- **`bayesian_changepoint_detection`** (J. Kulick et al., implementación de Adams &
  MacKay 2007 y Fearnhead 2006) — provee BOCPD online y variantes offline en Python.
  Más artesanal; hay que definir la distribución predictiva y el *hazard*, y vigilar
  el coste del run-length (truncar el historial de runs para mantenerlo acotado).
- **CUSUM/ICSS**: triviales de implementar a mano (pocas líneas) o vía `ruptures`
  con coste adecuado; ventaja de control total sobre la causalidad y el umbral.
- **Coste global**: **bajo-medio**. Lo barato es ejecutar; lo costoso es (a) calibrar
  \(\beta\)/umbral/`min_size` contra la métrica de falsas alarmas, (b) montar el
  envoltorio expanding causal para los métodos offline, y (c) diseñar el
  **post-etiquetado económico causal** de segmentos. Recomendación: **CUSUM + BOCPD
  online** como detectores causales propios y **PELT (kernel `rbf`) de `ruptures`**
  como baseline offline/oráculo re-aplicado en ventana.

## 2. Detectores implementados

Las tablas de resultados salen de `results/benchmark/ranking_v2.csv` y `results/benchmark/metrics_master_v2.csv` (benchmark completo 24/24, `results/benchmark/manifest.json` del 2026-09-29T12:41 UTC; cifras de [ADR-003](../decisions/ADR-003-causalidad-calendario-estado-ranking.md) §4). Walk-forward causal y evaluación OOS por pista (17 crisis evaluables en la A y 9 en la B, según `configs/benchmark_spec.yaml`). Ranking de **detección**: un evento se detecta con ≥ 3 sesiones OOS consecutivas en crisis dentro de [pico, suelo]; `score_deteccion` = F1 entre la precisión diaria (1 − FAR) y el recall por evento, con niveles previos (elegible › precisión ≤ azar › parpadeo › degenerado). Cada tabla va entre marcadores `resultados_v2`: tras un benchmark nuevo se regenera desde esos dos CSV, no se edita a mano.

<a id="d07"></a>

### D07 — `changepoint_online` · CUSUM online robusto

Detector de cambio estructural **online y causal** sobre la volatilidad del retorno del S&P 500: dos CUSUM de una cara de Page [`cp_page1954`] forman un autómata de dos estados (en calma se vigila una subida de la volatilidad, `C⁺ > h`; en crisis, una bajada, `C⁻ > h`). El umbral `h` da persistencia por construcción. PELT [`cp_killick2012`] solo se usa en el notebook como oráculo in-sample, marcado no causal.

**Configuración v2**

- Código: `src/regimenes/detectores/f6_changepoint/changepoint_online.py` (`ChangepointOnline`); registro, id `D07`.
- **Coste robusto** (`cost='robust'`, el que usa el registro): estadístico `log|r|`, estandarizado con la **mediana y 1.4826·MAD del train** y **winsorizado** a ±`clip`. La alternativa `cost='gaussian'` (`r²` con media/desviación) queda como contraste en el notebook.
- Parámetros: `k = 0.5`, `h = 5.0`, `clip = 3.0`; `predict_proba` es una logística del z EWMA (vida media 10 sesiones). Burn-in del train en cada bloque.
- Feature: `SP500_ret` en ambas pistas. Walk-forward: refit cada 21 sesiones, ventana expanding.

<!-- BEGIN resultados_v2 D07 -->
**Resultados v2 (ADR-003)**

| Métrica | Pista A | Pista B |
|---|---:|---:|
| Variante evaluada | `changepoint_online` | `changepoint_online` |
| Puesto de detección | **8** / 12 | **7** / 12 |
| Nivel de etiqueta | `elegible` | `elegible` |
| `score_deteccion` | 0.469 | 0.426 |
| Recall por evento | 0.824 (14/17) | 0.556 (5/9) |
| Precisión diaria (lift vs base) | 0.328 (×1.69) | 0.345 (×2.00) |
| Cobertura media de crisis | 0.646 | 0.285 |
| Tasa de falsas alarmas (FAR) | 0.672 | 0.655 |
| Activación media en trampas | 0.000 | 0.000 |
| Switching rate | 0.0022 | 0.0030 |
| Duración media de régimen (sesiones) | 441.7 | 312.2 |
| Estabilidad de etiqueta | 1.000 | 0.995 |
| Puesto legacy (`rank_medio`) | 2 (3.8) | 5 (5.8) |

**Ventana OOS y eventos por pista**

- **Pista A** · OOS 1970-05-12 → 2026-05-29 (n = 14133) · train inicial 2016 sesiones, refit cada 21 (expanding) · 1 feature(s): `SP500_ret`.
  - Detectados (14): `bear_1969_70`, `oil_stagflation_1973`, `volcker_1980_82`, `black_monday_1987`, `gulf_war_sl_1990`, `ltcm_russia_1998`, `dotcom_2000_02`, `gfc_2007_09`, `flash_crash_euro1_2010`, `us_downgrade_euro2_2011`, `china_oil_2015_16`, `covid_2020`, `inflation_bear_2022`, `svb_banking_2023`.
  - No detectados (3): `volmageddon_2018q1`, `fed_tightening_2018q4`, `tariff_selloff_2025`.
- **Pista B** · OOS 2010-04-12 → 2026-05-29 (n = 4059) · train inicial 756 sesiones, refit cada 21 (expanding) · 1 feature(s): `SP500_ret`.
  - Detectados (5): `us_downgrade_euro2_2011`, `china_oil_2015_16`, `covid_2020`, `inflation_bear_2022`, `svb_banking_2023`.
  - No detectados (4): `flash_crash_euro1_2010`, `volmageddon_2018q1`, `fed_tightening_2018q4`, `tariff_selloff_2025`.
<!-- END resultados_v2 D07 -->

Narrativa de la Capa 1 (implementación v1, hallazgos e hipótesis del CHECKPOINT 2): [`historia/fichas_v1.md`](../historia/fichas_v1.md#d07).

## 3. Variantes no implementadas

- **TDA / spectral-residual CPD:** transiciones por homología persistente, línea aún no madura como baseline.
- **Costes no paramétricos** (CUSUM sobre rangos, coste Huber, RuLSIF / density-ratio), relevantes por las colas gordas del EDA; no implementados. El coste robusto `log|r|` + mediana/MAD + winsorización **sí está implementado** en D07 (es su coste por defecto).
- **BOCPD** [`cp_adamsmackay2007`] (p. ej. con `bayesian_changepoint_detection`) como pieza online causal alternativa al CUSUM; no implementado.

## 4. Referencias

- **Page (1954)** — *Continuous Inspection Schemes*. Biometrika 41(1/2):100-115.
  DOI: 10.1093/biomet/41.1-2.100. (CUSUM seminal.) [`cp_page1954`]
- **Inclán & Tiao (1994)** — *Use of Cumulative Sums of Squares for Retrospective
  Detection of Changes of Variance*. JASA 89(427):913-923.
  DOI: 10.1080/01621459.1994.10476824. (ICSS, cambios de varianza.) [`cp_inclantiao1994`]
- **Adams & MacKay (2007)** — *Bayesian Online Changepoint Detection*. arXiv:0710.3742.
  https://arxiv.org/abs/0710.3742. (BOCPD, online causal.) [`cp_adamsmackay2007`]
- **Fearnhead & Liu (2007)** — *On-line Inference for Multiple Changepoint Problems*.
  JRSS-B 69(4):589-605. DOI: 10.1111/j.1467-9868.2007.00601.x. (Filtrado online.) [`cp_fearnheadliu2007`]
- **Killick, Fearnhead & Eckley (2012)** — *Optimal Detection of Changepoints with a
  Linear Computational Cost*. JASA 107(500):1590-1598.
  DOI: 10.1080/01621459.2012.737745. (PELT.) [`cp_killick2012`]
- **Lavielle & Teyssière (2007)** — *Adaptive Detection of Multiple Change-Points in
  Asset Price Volatility*. En *Long Memory in Economics* (Springer), pp. 129-156.
  DOI: 10.1007/978-3-540-34625-8_5. (Aplicación financiera.) [`cp_lavielleteyssiere2007`]
- **Arlot, Celisse & Harchaoui (2019)** — *A Kernel Multiple Change-point Algorithm
  via Model Selection*. JMLR 20(162):1-56.
  https://jmlr.org/papers/v20/16-155.html. (Kernel CPD no paramétrico.) [`cp_arlot2019`]
- **Truong, Oudre & Vayatis (2020)** — *Selective Review of Offline Change Point
  Detection Methods*. Signal Processing 167:107299.
  DOI: 10.1016/j.sigpro.2019.107299. (Survey + librería `ruptures`.) [`cp_truong2020`]
