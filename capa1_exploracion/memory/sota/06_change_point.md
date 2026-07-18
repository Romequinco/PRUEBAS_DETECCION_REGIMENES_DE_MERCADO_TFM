# 06 — Change-point detection

> Familia de la FASE 2 (estado del arte). Enfoque: detectar **instantes de cambio
> estructural** que separan tramos homogéneos, en vez de modelar estados latentes
> recurrentes. Referencias en `06_change_point.bib`.

## Definición y supuestos

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

## Variantes principales

### CUSUM (Page, 1954) — online, secuencial
La suma acumulada de desviaciones respecto a un valor objetivo. Se mantiene un
estadístico \(S_t = \max(0,\, S_{t-1} + (x_t - \mu_0 - k))\) y se declara cambio
cuando \(S_t\) supera un umbral \(h\). Es el detector secuencial original de control
de calidad, **causal por construcción**: solo usa el pasado y dispara con cierto
**retardo de detección** tras el cambio real. Variante de varianza:
**ICSS** (Inclán & Tiao, 1994), suma acumulada de *cuadrados* para detectar cambios
en la varianza incondicional —muy usado en finanzas para quiebres de volatilidad.

### BOCPD — Bayesian Online Change Point Detection (Adams & MacKay, 2007) — online
Estima en cada instante la distribución posterior del **run length** \(r_t\)
(tiempo transcurrido desde el último cambio) mediante un *message-passing*
recursivo: \(P(r_t \mid x_{1:t})\). Cada paso solo necesita el dato nuevo y el
posterior anterior, por lo que es **exactamente causal y online**. Usa una
distribución predictiva por segmento (típicamente de la familia exponencial con
prior conjugado) y un *hazard* que modela la probabilidad a priori de cambio. Su
fortaleza es que entrega **incertidumbre** (probabilidad de cambio) en tiempo real,
no solo un punto. Fearnhead & Liu (2007) proponen, en paralelo, filtrado online
exacto para múltiples change-points con coste reducible a lineal vía partículas.

### `ruptures` — métodos principalmente OFFLINE (Truong et al., 2020)
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

### Coste / penalización del número de cambios
El número de breakpoints es un problema de **selección de modelo**: sin penalizar,
el óptimo coloca un cambio en cada punto (sobreajuste). Se controla con
penalización lineal \(\beta\) (PELT), BIC/AIC, o el coste por cambio en kernel CPD
(Arlot et al., 2019). **La calibración de \(\beta\)/umbral es la decisión crítica**:
gobierna directamente el *trade-off* entre detección temprana y falsas alarmas.

### Online vs offline — distinción central para este proyecto
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

## Fortalezas y debilidades

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
  normalidad/i.i.d. Con la **kurtosis de exceso ≈ 25.6** del S&P 500 (y ≈ 39.6 en
  HYG) del EDA, un único outlier leptocúrtico puede superar el umbral y disparar un
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

## Idoneidad para este proyecto

1. **Variante relevante = online/secuencial.** El proyecto es causal y walk-forward,
   así que las piezas de primera clase son **CUSUM/ICSS** y **BOCPD**. PELT y el
   resto de `ruptures` solo entran como baseline offline *re-aplicado en ventana
   expanding*, declarando explícitamente que la versión "sobre toda la serie" es
   anti-causal y sirve únicamente de referencia/oráculo.

2. **EDA → elegir métodos robustos.** Dado el perfil de **fat tails** (kurtosis
   ≈ 25.6 en SP500), priorizar variantes **no paramétricas / kernel** o costes
   robustos frente al CUSUM gaussiano puro, para no confundir outliers con cambios
   de régimen. Conviene además operar sobre **features ya causales y
   z-scoreadas** (`VIX_level_z`, `SP500_vol_z`, `credit_spread_z`,
   `SP500_drawdown`), que están más cerca de la i.i.d. por tramos que los precios
   crudos.

3. **Métrica lead/lag — donde puede lucir.** La detección temprana es la ventaja
   natural del CPD: medir el **lead/lag respecto al suelo del drawdown**
   (`DRAWDOWN_TROUGHS`: 2009-03-09, 2011-10-03, 2020-03-23, 2022-10-12) es la prueba
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
   del `evaluation.py`).

**Veredicto**: familia **complementaria y valiosa como detector temprano** (sobre
todo CUSUM/BOCPD online), no como clasificador de regímenes recurrentes. Encaja
bien como *baseline causal* y como *fuente de la métrica lead/lag*, con la cautela
de fat tails (preferir kernel/robusto) y el coste de post-etiquetado.

## Aplicaciones documentadas a regímenes de mercado financiero

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

## Coste de implementación y librería Python recomendada

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

## Referencias

- **Page (1954)** — *Continuous Inspection Schemes*. Biometrika 41(1/2):100-115.
  DOI: 10.1093/biomet/41.1-2.100. (CUSUM seminal.)
- **Inclán & Tiao (1994)** — *Use of Cumulative Sums of Squares for Retrospective
  Detection of Changes of Variance*. JASA 89(427):913-923.
  DOI: 10.1080/01621459.1994.10476824. (ICSS, cambios de varianza.)
- **Adams & MacKay (2007)** — *Bayesian Online Changepoint Detection*. arXiv:0710.3742.
  https://arxiv.org/abs/0710.3742. (BOCPD, online causal.)
- **Fearnhead & Liu (2007)** — *On-line Inference for Multiple Changepoint Problems*.
  JRSS-B 69(4):589-605. DOI: 10.1111/j.1467-9868.2007.00601.x. (Filtrado online.)
- **Killick, Fearnhead & Eckley (2012)** — *Optimal Detection of Changepoints with a
  Linear Computational Cost*. JASA 107(500):1590-1598.
  DOI: 10.1080/01621459.2012.737745. (PELT.)
- **Lavielle & Teyssière (2007)** — *Adaptive Detection of Multiple Change-Points in
  Asset Price Volatility*. En *Long Memory in Economics* (Springer), pp. 129-156.
  DOI: 10.1007/978-3-540-34625-8_5. (Aplicación financiera.)
- **Arlot, Celisse & Harchaoui (2019)** — *A Kernel Multiple Change-point Algorithm
  via Model Selection*. JMLR 20(162):1-56.
  https://jmlr.org/papers/v20/16-155.html. (Kernel CPD no paramétrico.)
- **Truong, Oudre & Vayatis (2020)** — *Selective Review of Offline Change Point
  Detection Methods*. Signal Processing 167:107299.
  DOI: 10.1016/j.sigpro.2019.107299. (Survey + librería `ruptures`.)

## Candidatas adicionales (para el sintetizador)

- **HMM / Markov-switching (familia ajena)**: alternativa que sí modela **regímenes
  recurrentes** y los etiqueta nativamente (crisis vs calma). Complementaria al CPD:
  el CPD detecta *cuándo* cambia, el HMM dice *a qué estado conocido* va. Sugerencia:
  comparar lead/lag de CUSUM/BOCPD frente al HMM gaussiano/t-Student.
- **Spectral residual / TDA-CPD (línea emergente)**: detección de transiciones por
  homología persistente; reportada como pre-señal de giros de volatilidad (criterio
  del equipo: aún no madura como baseline, vigilar).
- **Costes robustos / no paramétricos adicionales**: variantes de CUSUM sobre rangos
  o con coste Huber, y RuLSIF / density-ratio change-point — relevantes dado el
  perfil fat-tails del EDA (criterio del equipo).
- **`bayesian_changepoint_detection`** como implementación directa de BOCPD para la
  pieza online causal (software, no paper).
