# F5 — Volatilidad y GARCH

> **Cómo leer este documento.** La sección 1 es la teoría de la familia (redactada en la Capa 1, original en el tag `capa1-final`); la 2, los detectores implementados con su configuración v2 y sus resultados del benchmark (ADR-003): [D06 `garch_t_vol`](#d06), [D11 `msgarch_regime`](#d11). Notebook de familia: [`notebooks/10_familia_F5_garch.ipynb`](../../notebooks/10_familia_F5_garch.ipynb). Bibliografía: claves `vol_*` y compartidas de [`docs/references.bib`](../references.bib).
>
> **Datos v1 en la teoría.** Las cifras del EDA que cita la sección 1 son de la Capa 1 (15 features, ventana 2007-04 → 2026-06, 4 crisis: 2008, 2011, 2020 y 2022), igual que los nombres de feature `credit_spread_z` o `yield_slope_z` y las referencias a la «tarea previa»; donde el dato importa se indica junto a él. El panel v2 tiene dos pistas (A: 1962-01-02 → 2026-05-29; B: 2007-04-11 → 2026-05-29) y el EDA vigente está en [`eda.md`](eda.md) (curtosis de exceso del S&P 500 = 18,7 en 1928-2026).

## 1. Teoría

> Familia de la FASE 2 centrada en la **modelización explícita de la varianza
> condicional** (heteroscedasticidad) y su acoplamiento con regímenes de mercado.
> Idea rectora: el régimen se detecta a través de la **volatilidad condicional**
> y sus cambios de estado. Estos modelos son **naturalmente causales** (la
> varianza en `t` depende solo del pasado), lo que los hace idóneos para un marco
> walk-forward / online sin look-ahead.
>
> **Nota de frontera (solape declarado):** el Markov-Switching "puro" sobre la
> *media/varianza* (Hamilton 1989, sin dinámica GARCH dentro del régimen) lo
> trata la ficha [F4](F4_markov_switching.md). Aquí el foco es la **varianza condicional
> heteroscedástica** y su combinación con regímenes (RS-GARCH / MS-GARCH). El
> solape concreto está en los modelos de cambio de régimen: la diferencia es que
> el MS "puro" asume varianza constante *dentro* de cada estado, mientras que
> RS-GARCH deja que la varianza evolucione tipo GARCH *dentro* de cada estado.

### Definición y supuestos

Los modelos de la familia ARCH descomponen el retorno en una **media
condicional** y una **innovación con varianza condicional variable en el
tiempo**:

```
r_t = μ_t + ε_t ,   ε_t = σ_t · z_t ,   z_t ~ iid D(0,1)
```

- **ARIMA(p,d,q)** modela la *media condicional* `μ_t` (componente
  autorregresivo–media móvil sobre la serie, posiblemente diferenciada). En
  retornos financieros la media suele ser casi ruido blanco, así que su papel es
  modesto; importa sobre todo como *paso previo* para limpiar autocorrelación
  antes de modelar la varianza de los residuos.
- **ARCH(q)** (Engle, 1982): `σ_t²` es función lineal de los `q` cuadrados de las
  innovaciones pasadas. Captura el **clustering de volatilidad** (rachas
  tranquilas y turbulentas alternadas).
- **GARCH(p,q)** (Bollerslev, 1986): añade términos autorregresivos de la propia
  varianza, dando una representación parsimoniosa (típicamente GARCH(1,1)) de
  memoria larga en la volatilidad.

**Supuestos clave:**
1. La media condicional es (casi) impredecible; la estructura explotable está en
   la **varianza**.
2. La varianza condicional es una función determinista del pasado observable →
   **causalidad estricta** (sin uso del futuro).
3. La innovación estandarizada `z_t` sigue una distribución fija `D`
   (gaussiana por defecto; t-Student o GED para colas).
4. Estacionariedad en covarianza: en GARCH(1,1), `α + β < 1`.

### Variantes principales

| Modelo | Aporta | Ecuación de la varianza (esquema) | Referencia |
|---|---|---|---|
| **ARIMA** | media condicional | `φ(L)(1−L)^d r_t = θ(L) ε_t` | Box–Jenkins (clásico) |
| **ARCH(q)** | clustering de vol | `σ_t² = ω + Σ α_i ε_{t−i}²` | Engle (1982) |
| **GARCH(1,1)** | memoria de vol parsimoniosa | `σ_t² = ω + α ε_{t−1}² + β σ_{t−1}²` | Bollerslev (1986) |
| **EGARCH** | **asimetría** (log-var, sin restricción de signo) | `ln σ_t² = ω + β ln σ_{t−1}² + α(\|z_{t−1}\|−E\|z\|) + γ z_{t−1}` | Nelson (1991) |
| **GJR-GARCH** | asimetría vía indicador de signo | `σ_t² = ω + α ε_{t−1}² + γ ε_{t−1}² 1_{ε_{t−1}<0} + β σ_{t−1}²` | Glosten–Jagannathan–Runkle (1993) |
| **TGARCH** | asimetría sobre la desviación típica | `σ_t = ω + α\|ε_{t−1}\| + γ ε_{t−1}^- + β σ_{t−1}` | Zakoian (1994) |
| **GARCH-t / GARCH-GED** | **colas gruesas** en `z_t` | misma recursión, `z_t ~ t_ν` (o GED) | Bollerslev (1987) |
| **SWARCH** | ARCH con **cambio de régimen** markoviano en el nivel de varianza | ARCH escalado por un factor de estado `g_{s_t}` | Hamilton–Susmel (1994) |
| **RS-GARCH (Gray)** | GARCH dentro de cada régimen; rompe la *path dependence* integrando la varianza sobre los estados | varianza recursiva con `σ_{t−1}²` colapsada vía probabilidades de régimen | Gray (1996) |
| **RS-GARCH (Klaassen)** | mejora la recursión usando la **esperanza condicional** de la varianza dado el régimen actual; previsión multi-periodo recursiva | usa `E[σ_{t−1}² \| s_t]` en lugar del colapso de Gray | Klaassen (2002) |
| **MS-GARCH (Haas–Mittnik–Paolella)** | **K procesos GARCH paralelos**, uno por régimen, sin path dependence → tratable analíticamente | `σ_{k,t}² = ω_k + α_k ε_{t−1}² + β_k σ_{k,t−1}²` por estado `k` | Haas–Mittnik–Paolella (2004) |
| **MRS-GARCH (aplicación)** | todos los parámetros conmutan; `z_t` t-Student con `ν` por régimen | combinación de los anteriores, evaluado para predicción de vol bursátil | Marcucci (2005) |

**El "asunto" de la path dependence en RS-GARCH:** un GARCH dentro de un régimen
necesita `σ_{t−1}²`, que a su vez dependería de *toda* la trayectoria de estados
pasada (`2^t` caminos) → estimación inviable. Las soluciones difieren en cómo
colapsan esa dependencia:
- **Gray (1996):** sustituye `σ_{t−1}²` por su valor *integrado* sobre las
  probabilidades de régimen (esperanza ponderada).
- **Klaassen (2002):** integra usando la probabilidad *condicionada al régimen
  actual* `s_t` (más información), lo que da previsiones multi-periodo más
  limpias y mejor desempeño out-of-sample.
- **Haas–Mittnik–Paolella (2004):** evitan el problema de raíz: cada régimen
  lleva su **propia** recursión de varianza independiente, de modo que el modelo
  es markoviano y analíticamente tratable (momentos, estacionariedad cerrados).

### Fortalezas y debilidades

**¿Detecta régimen o solo modela varianza?**
- ARIMA/ARCH/GARCH y sus variantes asimétricas **no detectan régimen**: producen
  una **serie continua de volatilidad condicional**. Para obtener una "señal de
  régimen" hay que añadir una capa (umbral sobre `σ_t`, o clasificación de
  niveles de vol). Es modelado de varianza, no segmentación de estados.
- **SWARCH / RS-GARCH / MS-GARCH sí detectan régimen**: el filtro de Hamilton
  entrega `P(s_t = crisis | información hasta t)`, una **probabilidad suave de
  régimen** directamente comparable con el `predict_proba` del HMM previo. Esta
  es la rama relevante para detectar régimen.

**¿Capta crisis rápidas?** Sí, y es su ventaja diferencial frente al HMM
gaussiano (que en la tarea previa **se perdió** Taper Tantrum 2013 y el sell-off
Q4 2018). La varianza GARCH **reacciona en el mismo día** a una innovación grande
(`ε_{t−1}²` salta), de modo que sube de nivel mucho antes de que el HMM acumule
suficiente evidencia para conmutar de estado. La asimetría (EGARCH/GJR/TGARCH)
refuerza esto: un shock **negativo** amplifica la volatilidad más que uno
positivo (efecto apalancamiento), lo que es exactamente el patrón de los
drawdowns de equity.

**¿Flickering?** El GARCH simple no "parpadea" porque no emite etiquetas
discretas; entrega un nivel continuo. En RS-GARCH/MS-GARCH el flickering depende
de la **persistencia de la matriz de transición**: con regímenes muy persistentes
(diagonal alta, como las `P≈0.94–0.98` halladas en la tarea previa) la
conmutación es estable. Las probabilidades suaves permiten además histéresis /
umbral con banda muerta para evitar oscilación.

**¿Asume normalidad y cómo se arregla?** El GARCH gaussiano captura el clustering
pero **deja colas residuales**: tras estandarizar por `σ_t`, los `z_t` de equity
y crédito siguen siendo leptocúrticos. La solución estándar es **GARCH-t**
(Bollerslev, 1987) o GED: una `z_t ~ t_ν` con `ν` bajo (≈4–8) absorbe la curtosis
sobrante. En RS-GARCH es habitual dejar `ν` **distinto por régimen** (Marcucci,
2005), capturando curtosis dependiente del estado.

**¿Causal / online?** **Sí, por construcción** — es la mayor fortaleza para este
proyecto. `σ_t²` se calcula con datos hasta `t−1`; no hay estadístico de muestra
completa. La estimación de parámetros puede hacerse por ventana expanding/rolling
(walk-forward) y la varianza se actualiza recursivamente un paso a la vez, ideal
para detección online sin re-mirar el futuro.

**Coste de estimación de RS-GARCH:** ahí está la debilidad. El path dependence
hace que Gray/Klaassen requieran aproximaciones, y la verosimilitud combina el
filtro de Hamilton con la recursión GARCH → **optimización no convexa, sensible a
valores iniciales, con riesgo de óptimos locales y de problemas de
identificación/label-switching**. Es bastante más caro y frágil de estimar que un
GARCH(1,1) o que un HMM gaussiano. La formulación de Haas–Mittnik–Paolella alivia
esto al ser markoviana, pero sigue siendo no trivial.

### Idoneidad para este proyecto

Conexión directa con los hallazgos del EDA de la Capa 1 (memoria v1, tag `capa1-final`; EDA vigente
en [`eda.md`](eda.md)):

1. **Clustering de volatilidad evidente** → es precisamente el fenómeno que
   ARCH/GARCH fueron diseñados para modelar. La feature `SP500_vol_z` (vol
   realizada 21d estandarizada) ya usada en el HMM es una versión *no
   paramétrica* de lo que GARCH estima de forma paramétrica y un paso por delante.
2. **Fat tails muy fuertes** (curtosis de exceso S&P 500 ≈ 25.6 y HYG ≈ 39.6 en la ventana v1 2007-2026; el EDA v2 da 18,7 para el S&P 500 en 1928-2026) → el
   GARCH gaussiano dejaría colas residuales; **motiva GARCH-t / GED** como
   distribución de error. Esto ataca de frente la limitación nº5 del detector
   anterior ("supuesto gaussiano subestima colas").
3. **Reacción rápida de la varianza** → candidata natural para capturar las
   crisis medianas/rápidas (2013, 2018) que el HMM gaussiano falló (limitación
   nº2).
4. **Marco causal walk-forward** → GARCH encaja sin fricción; no introduce el
   look-ahead sutil de los z-scores de muestra completa (limitación nº6).

**Cómo derivar una señal de régimen desde esta familia (dos vías):**

- **Vía A — umbral sobre la volatilidad condicional:** ajustar GARCH-t(1,1) (o
  GJR-GARCH-t por la asimetría del equity) sobre el retorno del S&P 500; tomar
  `σ_t` anualizada y clasificar régimen por nivel (p.ej. percentil expanding) o
  por cruce de umbral con histéresis. Simple, robusto, plenamente causal. No da
  estados "tipados", solo calma/estrés por intensidad de vol.
- **Vía B — estados del RS-GARCH/MS-GARCH:** ajustar un MS-GARCH de 2–3 estados y
  usar la **probabilidad filtrada de régimen** `P(s_t | F_t)` como salida. Esto es
  lo más parecido conceptualmente al HMM previo, pero con **heteroscedasticidad
  explícita dentro de cada régimen** (en vez de varianza constante por estado).
  Es la extensión natural del GARCH-t hacia un detector de régimen.

**Recomendación de encaje:** usar GARCH-t (Vía A) como **baseline econométrico
fuerte y barato** que probablemente mejore la detección de crisis rápidas, y
MS-GARCH (Vía B) como **detector avanzado** que aporta heteroscedasticidad
intra-régimen y probabilidades suaves, asumiendo su mayor coste/fragilidad de
estimación. Ambos son univariantes sobre el S&P 500 (o crédito HYG); para el
panel multi-activo del proyecto sirven como sensor focalizado de la vol de
equity, complementando al HMM multivariante.

### Aplicaciones documentadas a regímenes de mercado financiero

- **Hamilton–Susmel (1994):** SWARCH aplicado a retornos semanales de la bolsa
  de EE.UU.; identifican regímenes de alta y baja volatilidad y muestran que
  ignorar el cambio de régimen sobreestima la persistencia de la volatilidad.
- **Gray (1996):** regímenes en el tipo de interés a corto plazo de EE.UU.;
  modelo generalizado que anida GARCH y raíz cuadrada con probabilidades de
  transición dependientes del estado.
- **Klaassen (2002):** divisas; el RS-GARCH mejora significativamente las
  previsiones de volatilidad out-of-sample frente al GARCH simple, corrigiendo el
  sesgo al alza de éste en periodos volátiles.
- **Haas–Mittnik–Paolella (2004):** series de tipos de cambio; formulación MS-GARCH
  tratable con mejores propiedades dinámicas.
- **Marcucci (2005):** **predicción de la volatilidad del S&P 500** comparando
  GARCH estándar contra MRS-GARCH (con `z_t` t-Student y `ν` por régimen); el
  modelo de cambio de régimen domina en horizontes cortos. Es la referencia más
  alineada con este proyecto (mismo activo, foco en regímenes de volatilidad).
- **Ardia et al. (2019):** documentan el paquete MSGARCH con aplicaciones de
  Value-at-Risk y previsión de volatilidad bajo cambio de régimen.

### Coste de implementación y librería Python recomendada

- **GARCH y variantes (ARCH, GARCH, EGARCH, GJR/TARCH, GARCH-t/GED): `arch` de
  Kevin Sheppard** (`pip install arch`, repo `bashtage/arch`). Madura, bien
  documentada, soporta media (Constant/AR/HAR), volatilidad (ARCH/GARCH/EGARCH/
  TARCH) y distribuciones (Normal/StudentsT/GED/SkewStudent). Estimación por MLE,
  previsión recursiva y simulación incluidas. **Coste bajo**: un GARCH(1,1) ajusta
  en milisegundos; encaja directamente en un bucle walk-forward.
- **RS-GARCH / MS-GARCH:** **no hay equivalente maduro y de referencia en Python
  puro.** El estándar de facto es el paquete **MSGARCH en R** (Ardia et al.,
  2019), muy completo (varios tipos de GARCH × distribuciones × nº de estados,
  MLE y MCMC bayesiano, VaR). En Python las opciones son: (a) puentear a R vía
  `rpy2`; (b) implementación propia del filtro de Hamilton sobre la recursión
  MS-GARCH (formulación Haas–Mittnik–Paolella, la más tratable); (c) paquetes
  parciales/menos mantenidos. **Coste alto** por la optimización no convexa,
  label-switching e inicialización sensible.
- **ARIMA (media):** `statsmodels` (`ARIMA`/`SARIMAX`) si se necesita filtrar la
  media antes del GARCH; en retornos diarios suele bastar con media constante o
  cero, evitando complejidad.

**Limitaciones de implementación a declarar:** (1) RS-GARCH en Python implica
trabajo a medida o dependencia de R, lo que choca con el stack Python del
proyecto; conviene presupuestarlo o limitarse a GARCH-t univariante + capa de
régimen. (2) GARCH es **univariante** por activo: no sustituye al HMM
multivariante sobre el panel de features, lo complementa como sensor de vol de equity/
crédito. (3) En walk-forward, reestimar parámetros en cada ventana es caro; una
alternativa causal barata es estimar en una ventana de calibración y solo
**actualizar recursivamente** `σ_t` (los parámetros cambian despacio).

## 2. Detectores implementados

Las tablas de resultados salen de `results/benchmark/ranking_v2.csv` y `results/benchmark/metrics_master_v2.csv` (benchmark completo 24/24, `results/benchmark/manifest.json` del 2026-09-29T12:41 UTC; cifras de [ADR-003](../decisions/ADR-003-causalidad-calendario-estado-ranking.md) §4). Walk-forward causal y evaluación OOS por pista (17 crisis evaluables en la A y 9 en la B, según `configs/benchmark_spec.yaml`). Ranking de **detección**: un evento se detecta con ≥ 3 sesiones OOS consecutivas en crisis dentro de [pico, suelo]; `score_deteccion` = F1 entre la precisión diaria (1 − FAR) y el recall por evento, con niveles previos (elegible › precisión ≤ azar › parpadeo › degenerado). Cada tabla va entre marcadores `resultados_v2`: tras un benchmark nuevo se regenera desde esos dos CSV, no se edita a mano.

<a id="d06"></a>

### D06 — `garch_t_vol` · GJR-GARCH(1,1)-t con umbral sobre la sigma condicional

El GARCH no da estados tipados: da una **sigma condicional** causal. D06 ajusta un GJR-GARCH(1,1) con innovaciones t [`vol_glosten1993`; `vol_bollerslev1987`] sobre el retorno del S&P 500 y define el régimen **umbralizando la sigma** con percentiles del train, histéresis y permanencia mínima (la «vía A» de la teoría).

**Configuración v2**

- Código: `src/regimenes/detectores/f5_garch/garch_t_vol.py` (`GarchTVol`); registro, id `D06`.
- Modelo: `arch_model(ret×100, mean='Constant', vol='GARCH', p=1, o=1, q=1, dist='t')`; parámetros estimados en el train y **congelados** (`.fix`); la sigma de cada bloque se reconstruye con burn-in del train.
- Umbral: `q_in = 0.80`, `q_out = 0.60` (percentiles de la sigma in-sample del train), `min_dwell = 5`; `predict_proba` es una sigmoide monótona de `σ − τ_in`.
- Feature: `SP500_ret` en ambas pistas. Walk-forward: refit cada 21 sesiones, ventana expanding.

<!-- BEGIN resultados_v2 D06 -->
**Resultados v2 (ADR-003)**

| Métrica | Pista A | Pista B |
|---|---:|---:|
| Variante evaluada | `garch_t_vol` | `garch_t_vol` |
| Puesto de detección | **7** / 12 | **2** / 12 |
| Nivel de etiqueta | `elegible` | `elegible` |
| `score_deteccion` | 0.540 | 0.597 |
| Recall por evento | 1.000 (17/17) | 0.889 (8/9) |
| Precisión diaria (lift vs base) | 0.370 (×1.91) | 0.449 (×2.60) |
| Cobertura media de crisis | 0.786 | 0.512 |
| Tasa de falsas alarmas (FAR) | 0.630 | 0.551 |
| Activación media en trampas | 0.079 | 0.000 |
| Switching rate | 0.0137 | 0.0133 |
| Duración media de régimen (sesiones) | 72.9 | 73.8 |
| Estabilidad de etiqueta | 0.998 | 0.999 |
| Puesto legacy (`rank_medio`) | 5 (6.2) | 1 (3.4) |

**Ventana OOS y eventos por pista**

- **Pista A** · OOS 1970-05-12 → 2026-05-29 (n = 14133) · train inicial 2016 sesiones, refit cada 21 (expanding) · 1 feature(s): `SP500_ret`.
  - Detectados (17): `bear_1969_70`, `oil_stagflation_1973`, `volcker_1980_82`, `black_monday_1987`, `gulf_war_sl_1990`, `ltcm_russia_1998`, `dotcom_2000_02`, `gfc_2007_09`, `flash_crash_euro1_2010`, `us_downgrade_euro2_2011`, `china_oil_2015_16`, `volmageddon_2018q1`, `fed_tightening_2018q4`, `covid_2020`, `inflation_bear_2022`, `svb_banking_2023`, `tariff_selloff_2025`.
  - No detectados (0): ninguno.
- **Pista B** · OOS 2010-04-12 → 2026-05-29 (n = 4059) · train inicial 756 sesiones, refit cada 21 (expanding) · 1 feature(s): `SP500_ret`.
  - Detectados (8): `flash_crash_euro1_2010`, `us_downgrade_euro2_2011`, `china_oil_2015_16`, `volmageddon_2018q1`, `fed_tightening_2018q4`, `covid_2020`, `inflation_bear_2022`, `tariff_selloff_2025`.
  - No detectados (1): `svb_banking_2023`.
<!-- END resultados_v2 D06 -->

Narrativa de la Capa 1 (implementación v1, hallazgos e hipótesis del CHECKPOINT 2): [`historia/fichas_v1.md`](../historia/fichas_v1.md#d06).

<a id="d11"></a>

### D11 — `msgarch_regime` · MS-GARCH de Haas, Mittnik y Paolella

Markov-Switching GARCH en la especificación **paralela** de Haas, Mittnik y Paolella [`vol_haasmittnikpaolella2004`]: dos recursiones GARCH(1,1)-t, una por régimen, cada una alimentada solo por su propia varianza pasada, lo que evita la dependencia de la trayectoria y permite la verosimilitud exacta con el filtro de Hamilton (la «vía B» de la teoría). Implementado desde cero en numpy/scipy, sin R.

**Configuración v2**

- Código: `src/regimenes/detectores/f5_garch/msgarch_regime.py` (`MSGarchRegime`); registro, id `D11`.
- Modelo: 2 regímenes GARCH(1,1)-t, media μ única y ν compartido; máxima verosimilitud con `n_init = 1`, `maxiter = 100`; posterior de crisis **filtrado** (Hamilton), diario y causal dentro de cada bloque.
- Feature: `SP500_ret` en ambas pistas.
- Walk-forward: refit cada **126** sesiones con ventana **rolling** (`expanding=False`, longitud = train inicial de la pista); detector lento.

<!-- BEGIN resultados_v2 D11 -->
**Resultados v2 (ADR-003)**

| Métrica | Pista A | Pista B |
|---|---:|---:|
| Variante evaluada | `msgarch_regime` | `msgarch_regime` |
| Puesto de detección | **11** / 12 | **8** / 12 |
| Nivel de etiqueta | `elegible` | `elegible` |
| `score_deteccion` | 0.443 | 0.394 |
| Recall por evento | 0.765 (13/17) | 0.778 (7/9) |
| Precisión diaria (lift vs base) | 0.312 (×1.61) | 0.264 (×1.53) |
| Cobertura media de crisis | 0.437 | 0.392 |
| Tasa de falsas alarmas (FAR) | 0.688 | 0.736 |
| Activación media en trampas | 0.000 | 0.167 |
| Switching rate | 0.0173 | 0.0719 |
| Duración media de régimen (sesiones) | 57.7 | 13.9 |
| Estabilidad de etiqueta | 0.897 | 0.872 |
| Puesto legacy (`rank_medio`) | 12 (8.8) | 12 (9.6) |

**Ventana OOS y eventos por pista**

- **Pista A** · OOS 1970-05-12 → 2026-05-29 (n = 14133) · train inicial 2016 sesiones, refit cada 126 (rolling) · 1 feature(s): `SP500_ret`.
  - Detectados (13): `oil_stagflation_1973`, `volcker_1980_82`, `black_monday_1987`, `gulf_war_sl_1990`, `ltcm_russia_1998`, `dotcom_2000_02`, `gfc_2007_09`, `us_downgrade_euro2_2011`, `china_oil_2015_16`, `volmageddon_2018q1`, `fed_tightening_2018q4`, `covid_2020`, `tariff_selloff_2025`.
  - No detectados (4): `bear_1969_70`, `flash_crash_euro1_2010`, `inflation_bear_2022`, `svb_banking_2023`.
- **Pista B** · OOS 2010-04-12 → 2026-05-29 (n = 4059) · train inicial 756 sesiones, refit cada 126 (rolling) · 1 feature(s): `SP500_ret`.
  - Detectados (7): `flash_crash_euro1_2010`, `us_downgrade_euro2_2011`, `china_oil_2015_16`, `volmageddon_2018q1`, `covid_2020`, `inflation_bear_2022`, `tariff_selloff_2025`.
  - No detectados (2): `fed_tightening_2018q4`, `svb_banking_2023`.
<!-- END resultados_v2 D11 -->

Narrativa de la Capa 1 (implementación v1, hallazgos e hipótesis del CHECKPOINT 2): [`historia/fichas_v1.md`](../historia/fichas_v1.md#d11).

## 3. Variantes no implementadas

- **Volatilidad realizada / HAR (Corsi, 2009):** alternativa no paramétrica a GARCH, ligada a la feature `vol 21d`; pertenece a las features de volatilidad.
- **DCC-GARCH (Engle):** extensión multivariante para la correlación condicional dinámica (feature `corr_spx_bond`); fuera de la familia univariante de esta ficha y no implementada.

## 4. Referencias

- Engle, R. F. (1982). *Autoregressive Conditional Heteroscedasticity with
  Estimates of the Variance of United Kingdom Inflation.* Econometrica 50(4),
  987–1008. DOI: 10.2307/1912773 [`vol_engle1982`]
- Bollerslev, T. (1986). *Generalized Autoregressive Conditional
  Heteroskedasticity.* Journal of Econometrics 31(3), 307–327. DOI:
  10.1016/0304-4076(86)90063-1 [`vol_bollerslev1986`]
- Bollerslev, T. (1987). *A Conditionally Heteroskedastic Time Series Model for
  Speculative Prices and Rates of Return.* The Review of Economics and Statistics
  69(3), 542–547. DOI: 10.2307/1925546 [`vol_bollerslev1987`]
- Nelson, D. B. (1991). *Conditional Heteroskedasticity in Asset Returns: A New
  Approach.* Econometrica 59(2), 347–370. DOI: 10.2307/2938260 [`vol_nelson1991`]
- Glosten, L. R., Jagannathan, R. & Runkle, D. E. (1993). *On the Relation
  between the Expected Value and the Volatility of the Nominal Excess Return on
  Stocks.* The Journal of Finance 48(5), 1779–1801. DOI:
  10.1111/j.1540-6261.1993.tb05128.x [`vol_glosten1993`]
- Zakoian, J.-M. (1994). *Threshold Heteroskedastic Models.* Journal of Economic
  Dynamics and Control 18(5), 931–955. DOI: 10.1016/0165-1889(94)90039-6 [`vol_zakoian1994`]
- Hamilton, J. D. & Susmel, R. (1994). *Autoregressive Conditional
  Heteroskedasticity and Changes in Regime.* Journal of Econometrics 64(1–2),
  307–333. DOI: 10.1016/0304-4076(94)90067-1 [`vol_hamiltonsusmel1994`]
- Gray, S. F. (1996). *Modeling the Conditional Distribution of Interest Rates as
  a Regime-Switching Process.* Journal of Financial Economics 42(1), 27–62. DOI:
  10.1016/0304-405X(96)00875-6 [`vol_gray1996`]
- Klaassen, F. (2002). *Improving GARCH Volatility Forecasts with Regime-Switching
  GARCH.* Empirical Economics 27(2), 363–394. DOI: 10.1007/s001810100100 [`vol_klaassen2002`]
- Haas, M., Mittnik, S. & Paolella, M. S. (2004). *A New Approach to
  Markov-Switching GARCH Models.* Journal of Financial Econometrics 2(4),
  493–530. DOI: 10.1093/jjfinec/nbh020 [`vol_haasmittnikpaolella2004`]
- Marcucci, J. (2005). *Forecasting Stock Market Volatility with Regime-Switching
  GARCH Models.* Studies in Nonlinear Dynamics & Econometrics 9(4), art. 6. DOI:
  10.2202/1558-3708.1145 [`vol_marcucci2005`]
- Ardia, D., Bluteau, K., Boudt, K., Catania, L. & Trottier, D.-A. (2019).
  *Markov-Switching GARCH Models in R: The MSGARCH Package.* Journal of
  Statistical Software 91(4), 1–38. DOI: 10.18637/jss.v091.i04 [`vol_ardia2019`]
- Sheppard, K. (s.f.). *arch: ARCH models in Python.* Software.
  https://github.com/bashtage/arch [`vol_sheppard_arch`]
