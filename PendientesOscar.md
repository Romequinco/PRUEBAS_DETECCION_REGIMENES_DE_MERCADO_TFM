# Pendientes Oscar · Detección de regímenes

Notas personales (17-10-2026). Recogen lo hablado sobre el estado del arte y los huecos de la parte de
**detectores** (Capa 1).

Las referencias entre corchetes usan la numeración de `docs/estado_del_arte/estado_del_arte.md`.

---

## 1. Conceptos aclarados

### Look-ahead en agentes LLM (FINSABER [12])

Hay dos tipos de fuga del futuro:

1. **En los datos de entrada.** Se le pasa al agente una noticia, un dato revisado o un precio que aún
   no existía en la fecha simulada. Lo tapan el TimeGate, los vintages de ALFRED y los lags (ADR-003).
2. **En la memoria del modelo.** El LLM se entrenó con internet hasta su fecha de corte. En un
   backtest de 2020 a 2022 ya «sabe» lo que pasó, aunque solo le pases la información de ese día. El
   TimeGate no lo tapa; hay que medirlo con las pruebas L1–L5 del plan de datos, todavía sin implementar.

FINSABER reevalúa las estrategias LLM con más de 20 años y más de 100 símbolos, y controla tres
sesgos: look-ahead, supervivencia y data snooping. Con eso la ventaja desaparece. Además, los LLM son
demasiado conservadores en mercados alcistas y demasiado agresivos en bajistas, así que los autores
piden **controles de riesgo según régimen**, que es el hueco que cubre el TFM.

### OOS y OOD

- **OOS (out-of-sample, fuera de muestra):** el modelo se evalúa con datos posteriores, que no vio al
  ajustarse. Es la única evaluación honesta.
  - Dentro de muestra el modelo ya ha visto la crisis. Ejemplo propio: el HMM gaussiano con
    look-ahead daba 98,6 % en 2008 y 10,9 % en 2013.
- **OOD (out-of-distribution, fuera de distribución):** los datos de evaluación vienen de una
  distribución distinta a la de entrenamiento. Por ejemplo, una crisis de un tipo nuevo, como el COVID.
  Hammond lo aproxima separando crisis «convencionales» y «novedosas».

### Benchmark de Hammond (2026) [4] · arXiv 2605.17117

- **Qué hace:** propone observables geométricos sobre un embedding QCML y los compara con métodos
  clásicos (Berry Phase Rate, Spectral Entropy, Reduced Purity, QFI, Multi-Lag Fidelity…).
- **Datos:** SPY y DIA de 1998 a 2024, sin macro con vintages.
- **Crisis:** 17 (9 convencionales y 8 novedosas).
- **Métodos:** los «46» son en realidad 22 canales geométricos propios más 24 baselines.
  - Clásicos: Absorption Ratio, Hamilton Markov-switching, GARCH(1,1), CUSUM, z-score de volatilidad
    móvil, BOCPD, turbulencia, y VIX como oráculo.
  - Machine learning: Random Forest *leave-one-crisis-out* y un Random Forest móvil con etiqueta
    VIX > 25.
- **Protocolo:** walk-forward anidado desde 2005; cada año reajusta los hiperparámetros (Optuna, 100
  pruebas) solo con las crisis ya terminadas, lo que da 9 pares crisis–ventana. Métricas: d de Cohen
  con bootstrap, más tasa de detección, retraso y falsas alarmas al año.
- **Resultados dentro de muestra:**
  - Reduced Purity 0,83 · Absorption Ratio 0,80 · Hamilton Markov-switching 0,71.
  - Sesgo de optimismo de unos +0,4 en d; Reduced Purity cae a unos 0,26 fuera de muestra.
- **Resultados fuera de muestra:**
  - Berry Phase Rate tiene d mediano 0,72 y es el mejor en 5 de 9 crisis. **El mejor detector cambia
    según la crisis.**
  - Random Forest: 7/7 crisis detectadas, 5 días de retraso, 3,6 falsas alarmas al año.
  - Berry Phase Rate: 5/7 crisis, 24 días de retraso, 1,2 falsas alarmas al año.
- **Qué usamos de él:** ningún método. Lo usamos como comparación externa, porque confirma que lo
  complejo no se sostiene fuera de muestra, y como ideas de protocolo pendientes (sección 10, P1).
- ⚠️ **Corregir** la ficha `hammond2026` de `fichas.json` y la sección 1.2 del estado del arte: no
  son «46 métodos clásicos y de ML». Verificar las cifras en el PDF antes de citarlas.

### Variables de los detectores (`src/regimenes/detectores/registry.py`)

- **CORE_A (9):** `SP500_ret_z`, `SP500_vol_z`, `SP500_momentum`, `SP500_drawdown`, `FF_MKT_z`,
  `DGS10_change_z`, `credit_BaaAaa_mensual_z`, `term_spread_hist_z`, `INDPRO_yoy_z`.
- **CORE_B (14):** CORE_A más `VIX_level_z`, `MOVE_level_z`, `credit_BAA10Y_z`, `DFII10_change_z` y
  `corr_spx_bond`.
- **Qué usa cada detector:**
  - Núcleo: D03, D04, D08, D09 y D12.
  - Solo `SP500_ret`: D05, D06, D07 y D11.
  - D01: volatilidad realizada en A y VIX en B.
  - D02 y D10: 4 features, distintas en cada pista.
- **Pool por pista:** la pista A tiene 41 series, que dan 35 columnas; la pista B, 106 series, que
  dan 77 columnas.
  - El pool es el **dataset congelado del benchmark** (ADR-001 y ADR-002), no la entrada de cada
    detector.
  - El núcleo se fijó a priori para evitar el data snooping. Además, con d² parámetros y 17 crisis,
    usar más variables sobreajusta.
  - **Debilidad:** el pool está infrautilizado. Se amplió para atacar 2013, pero ningún detector lo
    explota y el núcleo no salió de ningún proceso de selección.
  - Frase para la memoria: «el pool es el dataset del benchmark; los detectores son hipótesis con
    subconjuntos fijados a priori; explotar el pool completo queda como trabajo futuro». Mejor aún,
    tapar el hueco con los puntos B5 o B9.

---

## 2. Qué tenemos ya al nivel de la literatura

- Walk-forward causal: z-scores con ventana expansiva, lags de publicación y `assert_causal`. Es más
  estricto que Hammond.
- 17 crisis en la pista A, igual que Hammond.
- Score por evento y lift sobre el azar, más `lead_lag`, `false_alarm_rate` y el bootstrap por bloques
  de la cobertura (`evaluacion/metricas.py`).
- Parsimonia validada: los ganadores son D10 (4 series) y D05 (1 serie). Coincide con Hansen-Lunde
  [22], TCPDBench [24] y Hammond [4].

---

## 3. Huecos y cómo taparlos (solo detectores)

### A. Evaluación (prioridad alta, coste bajo)

- [ ] **A1. Significatividad del ranking con el Model Confidence Set** [46].
  - Bootstrap por crisis del score de cada detector y MCS sobre esas réplicas.
  - Objetivo: poder escribir «los mejores, estadísticamente indistinguibles, son {…}». ¿D10 (0,614)
    y D01 (0,587) se distinguen de verdad?
- [ ] **A2. Baselines triviales en el ranking** (TCPDBench [24]).
  - Tres detectores tontos: siempre calma, aleatorio con la tasa base y volatilidad por encima de un
    percentil 90 fijo.
  - Si alguno gana a nuestros detectores, hay que saberlo.
  - **Estado (19-10-2026): casi hecho.** `trivial_baselines` en `evaluacion/ranking.py` ya genera
    `SIEMPRE_CRISIS`, `SIEMPRE_CALMA`, `AZAR_IID` y `AZAR_PERSISTENTE`, y el notebook de comparativa
    los usa. Falta `VOL_P90`, con un percentil expansivo causal y no fijo con toda la muestra.
- [ ] **A3. Curva de detección frente a falsas alarmas y retraso** (Hammond [4], KLR [5],
  Sarlin [6]).
  - Calibrar el umbral de cada detector a la misma tasa de falsas alarmas, por ejemplo 1 al año.
  - Reportar tasa de detección, retraso mediano y ratio ruido-señal.
  - Reutiliza `lead_lag`. Sirve para cuantificar el resultado negativo de que la fusión «no anticipa
    el inicio».
- [ ] **A4. Ranking por defecto frente a ranking ajustado, con walk-forward anidado** (Hammond [4],
  TCPDBench [24]).
  - Mantener el ranking actual como «por defecto, hipótesis a priori».
  - Añadir un ranking «ajustado», donde cada año se tunea solo con las crisis ya terminadas.
  - La diferencia entre ambos es nuestro sesgo de optimismo, comparable al de Hammond.
  - Es el punto más caro.

### B. Detectores que faltan

- [ ] **B5. Absorption ratio** (Kritzman et al. [30]; 2º de 46 en Hammond). **Prioridad 1 entre los
  detectores.**
  - PCA causal sobre las **rentabilidades crudas** de unos 20 activos del pool (sectores, índices,
    bonos, crédito, oro, materias primas, divisas), no sobre los z-scores. Así lo define Kritzman.
  - **Solo en la pista B** (decisión del 19-10-2026): en la A solo hay unos 9 activos diarios.
  - Responde a Hammond y aprovecha el pool sin disparar el número de parámetros.
- [ ] **B6. Jump model bien especificado («D09-Shu»)** (Shu-Yu-Mulvey [3], Nystrup [35, 36]).
  - Problema actual: D09 queda último con penalización fija de 50 y las features del núcleo.
  - Shu usa features derivadas de rentabilidades (medias exponenciales, desviación a la baja) y elige
    la penalización fuera de muestra. Hacer una variante declarada con esa especificación.
  - Si sigue perdiendo, el resultado matiza a Mulvey con más fuerza; si gana, era la configuración.
  - Revisar las features exactas en el paper antes de implementar.
- [ ] **B7. Baseline supervisado** (Random Forest *leave-one-crisis-out* de Hammond [4], logit de
  Beutel [44], Bluwstein [45]).
  - Logit y Random Forest entrenados solo con crisis pasadas, como techo de referencia.
  - Usa las etiquetas, así que va en una categoría aparte del ranking no supervisado.
- [ ] **B8. BOCPD** (Adams-MacKay [32]; baseline en Hammond).
  - Solo tenemos CUSUM en la familia change-point. Prioridad baja.
- [ ] **B9. Detector sobre el panel rico con selección de variables** (Sparse JM de Nystrup [35]).
  - Alternativa a B5: el modelo elige variables del pool dentro del walk-forward.
  - Prioridad media, más caro.

### C. Fusión

- [ ] **C1. Agregación online de los 12 detectores** [61].
  - Pesos actualizados cada mes según el acierto pasado, en lugar de la regla fija (D2 alerta + D6
    confirma).
  - Comparar con la regla fija y con el mejor detector individual.

### No hacer

- Más familias deep ni MS-GARCH. La literatura y nuestros datos coinciden en que la complejidad no se
  paga con 17 crisis.

---

## 4. Orden recomendado (sustituido)

El orden original (A1+A2 → B5 → A3 → B6+B7 → resto) queda sustituido por el plan de la sección 6, que
reorganiza el trabajo en torno a usar el pool completo.

## 5. Otras tareas de documentación

- [ ] Corregir la descripción de Hammond en `fichas.json` y en la sección 1.2 del estado del arte (ver
  arriba). `docs/estado_del_arte/estado_del_arte.md` (línea 36) sigue diciendo «46 métodos».
- [ ] Comprobar en los PDF las cifras de Hammond y de FINSABER antes de citarlas en la memoria. Hay que
  localizar los PDF; no están en el repo.
- [ ] Comprobar las citas de la sección 6.2 (Boivin-Ng, NFCI, Stock-Watson, TCPDBench) antes de
  llevarlas a la memoria.

---

## 6. Plan acordado (19-10-2026): usar todo el pool

### 6.1 Diagnóstico

- **Usamos pocos datos.** Solo llegan a algún detector 8 de las 41 series de la pista A (20 %) y 15
  de las 106 de la B (14 %). Hay bloques enteros sin usar: inflación esperada 0/3, macro 1/18 y renta
  variable 2/31 en la B.
- **El ranking mezcla método y datos.** Cada detector recibe un subconjunto distinto. D10 gana con 4
  columnas y D04 queda décimo con 14: no sabemos si es mejor el método o los datos.
- **Corrección sobre 2013.** El *taper* de 2013 es una **trampa** (`false_positive_windows`): el S&P
  cae un 5,8 % y no es una crisis. Un buen detector no debe marcarlo. El bloque de velocidad de tipos
  sirve para **distinguirlo** como estrés de tipos, no para detectarlo como crisis.

### 6.2 Qué dice la literatura sobre dar los mismos datos a todos

- **Mismo conjunto de datos para todos los métodos.** Así lo hacen M4, Hansen-Lunde (330 ARCH sobre
  las mismas rentabilidades), Hammond (SPY y DIA para todos) y la alerta temprana (Beutel y Bluwstein
  comparan logit y ML con los mismos indicadores).
- **Cada método lo usa según su naturaleza.** Un GARCH modela una sola serie de rentabilidades.
  TCPDBench separa los métodos univariantes de los multivariantes.
- **Con muchas variables se comprime o se regulariza.** El NFCI es un factor de 105 indicadores y los
  índices de difusión de Stock-Watson son PCA sobre cientos de series. Boivin-Ng (2006), «Are more data
  always better for factor analysis?», responden que no siempre.
- **Por qué no las 77 columnas a todos.** Parámetros por estado de un HMM o GMM con covarianza
  completa: 8 entradas → 44, 14 → 119, 77 → 3.080. Con 9 crisis en la pista B, el modelo memoriza o no
  converge. Solo los métodos regularizados (sparse JM, random forest, logit L1) admiten el pool en bruto.

### 6.3 Termómetros por bloque: una PCA por bloque, no una PCA global

- 77 columnas (35 en la A) → 8 bloques (5 en la A) → una PCA pequeña por bloque → **8 termómetros**.
- Una PCA global quedaría dominada por los bloques más grandes (macro, tipos) y no se podría
  interpretar.
- Cada termómetro es una **columna causal nueva del panel**. La PCA se reajusta cada mes solo con el
  pasado y su primera componente da el valor de cada día. El signo se fija con una columna ancla
  declarada a priori, para que el termómetro siempre suba con el estrés. Los detectores no cambian por
  dentro, solo su lista de features.
- Si un bloque tiene una sola columna, el termómetro es esa columna. Si no supera el umbral de
  varianza explicada, se usa la media de sus columnas.

| Termómetro | Pista B (77) | Pista A (35) |
|---|---|---|
| T1 · Renta variable | SP500_ret_z, SP500_vol_z, SP500_momentum, SP500_drawdown, SP500_megacap_gap_z, equity_breadth_dispersion_z, smallcap_largecap_spread_z, style_spread_z, FF_MKT_z, FF_MOM_z, FF_RMW_z, FF_industry_dispersion_z, sector_dispersion, cyc_def_ratio_z (14) | Las 4 de SP500, FF_MKT_z, FF_MOM_z, FF_industry_dispersion_z (7) |
| T2 · Tipos: nivel y curva | slope_10y2y_z, slope_10y3m_z, curve_curvature_z, mid_curve_z, curve_vs_policy_10y_z, curve_vs_policy_5y_z, short_rate_z, EFFR_z, TB3MS_z, GS1_z, GS5_z, GS10_z, term_spread_hist_z, fed_stance_z, corr_spx_bond (15) | mid_curve_z, curve_vs_policy_10y_z, curve_vs_policy_5y_z, short_rate_z, TB3MS_z, GS1_z, GS5_z, GS10_z, term_spread_hist_z, fed_stance_z (10) |
| T3 · Velocidad de tipos | DGS10_change_z, DFII10_change_z, MOVE_change_z (3) | DGS10_change_z (1) |
| T4 · Inflación y tipo real | DFII5_z, breakeven_10y_z, breakeven_5y_z, breakeven_5y_fwd_z (4) | — |
| T5 · Volatilidad implícita | VIX_level_z, VIX_change_z, MOVE_level_z, VVIX_z, vix_term_z, vol_complex_dispersion_z, SKEW_z (7) | — |
| T6 · Crédito | credit_BAA10Y_z, credit_BaaAaa_diaria_z, credit_BaaAaa_mensual_z, credit_BaaFF_z, credit_HYG_IEF_z, credit_HYG_LQD_z (6) | credit_BaaAaa_mensual_z, credit_BaaFF_z (2) |
| T7 · Divisas y materias primas | DXY_change_z, JPY_ret_z, CHF_ret_z, AUD_ret_z, GOLD_ret_z, GSCI_ret_z, BCOM_ret_z, WTI_yoy_z, broad_dollar_real_z, broad_dollar_nominal_z (10) | — |
| T8 · Macro y valoración | INDPRO, PPI (×3), CPI (×2), UNRATE, PAYEMS, CFNAI (×2), PMI_proxy, ICSA, UMCSENT, HOUST, MANEMP, GDP_growth, CAPE_z, GW_valuation_z (18) | Las mismas salvo CFNAI, PMI e ICSA (15) |

> **Reparto final (30-10-2026):** 10 bloques en B y 7 en A, con T1 separado en estado y rentabilidad diaria y
> la macro separada en actividad e inflación + valoración. Lo fija el ADR-008 §2.1 y `configs/bloques.yaml`; esta
> tabla queda como propuesta inicial.

La lista de la pista A sale de `construir_diarias`/`construir_mensuales` sobre los datos crudos, más
FF_industry_dispersion_z y CAPE_z. El reparto en bloques es la propuesta inicial y se congela en el
ADR-008.

### 6.4 Menús de datos comunes (pendiente de confirmar)

En lugar de una lista de entradas propia para cada detector, hay **5 menús fijos**. Cada detector se
ejecuta con todos los menús que admite por naturaleza.

| Menú | Contenido | A / B |
|---|---|---|
| U · Univariante | SP500_ret (lo mismo que usa Hammond) | 1 / 1 |
| N · Núcleo | CORE_A / CORE_B (el actual) | 9 / 14 |
| T · Termómetros | Pool completo comprimido | 5 / 8 |
| P · Pool completo | Todas las columnas | 35 / 77 |
| R · Rentabilidades multiactivo | Unas 20 series crudas de activos | — / 20 |

| Detector | U | N | T | P | R |
|---|---|---|---|---|---|
| D01 Regla de volatilidad | ✓ | | | | |
| D02 Regla compuesta | | ✓ | ✓ (media de termómetros) | | |
| D03 GMM, D04 HMM, D08 HMM-t | ✓ | ✓ | ✓ | | |
| D05 MS varianza, D06 GARCH, D07 CUSUM, D11 MS-GARCH | ✓ | | | | |
| D09 Jump model (D15 Shu = menú U con las features de Shu) | ✓ | ✓ | ✓ | | |
| D10 Turbulencia | | ✓ | ✓ | ✓ (con shrinkage) | ✓ (definición original de Kritzman-Li) |
| D12 Autoencoder | | ✓ | ✓ | ✓ | |
| D14 Absorption ratio | | | | | ✓ |
| D16 Logit L1, D17 Random forest (categoría supervisada aparte) | | | ✓ | ✓ | |
| D18 Sparse jump model | | | | ✓ | |

- **Por columnas:** mismos datos, distintos métodos. Es la comparación justa. En el menú T, nueve
  detectores ven exactamente lo mismo y todo el pool. En el menú U compiten los univariantes, como en
  Hammond.
- **Por filas:** mismo método, distintos datos. Mide si el pool aporta.
- El ranking v2 se conserva como «configuración a priori de cada detector», el equivalente a la
  configuración por defecto de TCPDBench.
- Los 12 detectores originales no se tocan. Las variantes nuevas van al lado y `registry.py` pasa a
  listar pares detector × menú.

### 6.5 Cómo validar los termómetros antes de dárselos a ningún detector

Ninguna de estas tres pruebas usa las etiquetas de crisis.

1. **Bien calculados (tests).** Recuperar un factor plantado en datos de juguete. Prueba de prefijo:
   calcular hasta t debe dar lo mismo que calcular con toda la muestra y recortar. El signo no salta
   entre reajustes. Los NaN de las columnas que empiezan tarde no se propagan.
2. **Con sentido estadístico y económico.** Varianza explicada por la primera componente (umbral a fijar
   probando). Cargas estables entre reajustes. Ninguna columna pesa más del 50 %. Los signos tienen
   lógica (los spreads de crédito cargan en positivo en T6).
3. **Validez externa.** Correlación con los índices de estrés `rol=validation` que ya están en
   `data/raw` y nunca entran como features: NFCI, NFCI_CREDIT, NFCI_RISK, NFCI_LEVERAGE, ANFCI, OFR_FSI
   y sus subíndices de crédito, volatilidad y financiación, STLFSI4 y KCFSI. Por ejemplo, T6 frente a
   NFCI_CREDIT y OFR_FSI_CREDIT, y T5 frente a OFR_FSI_VOLATILITY.

Después vienen las pruebas con crisis:

4. **Relevancia.** Cada termómetro como regla de umbral, comparado con `persistent_random_null`. Mapa
   de bloques × crisis y comportamiento en las trampas.
5. **Valor añadido.** Comparar por pares cada detector con N y con T. Quitar un bloque cada vez. Dos
   controles: la media simple en vez de la PCA, y 100 reparticiones aleatorias de columnas a bloques.
   Si los bloques aleatorios rinden igual, la agrupación económica no aporta.
6. **Significación.** MCS dentro de cada menú, con los baselines incluidos. Solo hay «mejora» si se
   cumple en las dos pistas.

Todo esto se **pre-registra en el ADR-008 antes de ejecutar**: bloques, anclas, menús, parámetros y
criterios de éxito y de fracaso. Si nada mejora, también es un resultado válido: «con el pool completo,
lo simple sigue ganando».

### 6.6 Pasada de referencia antes de tocar nada

- **No volver a descargar.** Los 166 parquet crudos están en disco y una descarga nueva traería
  revisiones de FRED y Yahoo, que romperían la comparabilidad con v2.
- **Volver a ejecutar `03_preprocesado`.** `data/processed` no está en disco. Hay que comprobar que los
  sha256 coinciden con los que guarda la última ejecución del notebook (celda de escritura).
- **Regenerar los paneles walk-forward** de los 12 detectores (`results/benchmark/panels/`, que no
  está en disco; unas 8 h). Comprobar que las métricas coinciden con `metrics_master_v2.csv` y poner
  el tag `v2-reproducido`.
- Los notebooks posteriores no hace falta volver a ejecutarlos todavía: darían lo mismo.

### 6.7 Nuevo orden de los notebooks

Lo nuevo va antes de los sintéticos, que se rehacen al final con todos los detectores. La
renumeración va en un commit aparte con `git mv`; el test de enlaces de la documentación detecta los
enlaces rotos.

```
00 descarga · 01 eda · 02 diseño preprocesado · 03 preprocesado
04 termómetros (NUEVO: construcción y validación 1–3)
05 protocolo de evaluación (+ VOL_P90, MCS, curva a igual tasa de falsas alarmas)
06–12 familias F1–F7 (+ variantes por menú; D14 en F1; D15 y D18 en F2)
13 familia F8 supervisado (NUEVO: D16 y D17)
14 comparativa (matriz detector × menú, ranking v3, ablaciones, «¿sirve el pool?»)
15–16 fusiones actuales · 17 fusión online C1 (NUEVO)
18–21 sintéticos (generadores ampliados al pool y re-ejecutados)
22 decisión final · 23 pseudolive
```

Los generadores sintéticos (`sinteticos/datos.py`) solo fabrican CORE + `SP500_ret`. Para probar ahí
los menús T y P hay que ampliarlos al pool.

### 6.8 Fases

**Fase 1 · Cimientos: todos los datos y la vara de medir**

- [ ] 1.0 Rama de trabajo y **ADR-008 de pre-registro**: bloques, anclas, menús, parámetros y criterios.
- [ ] 1.1 Pasada de referencia (6.6).
- [ ] 1.2 `features/bloques.py` con sus tests y notebook 04 (validación 1–3).
- [ ] 1.3 Vara de medir sobre los 12: VOL_P90 (A2), MCS (A1) y curva (A3). Van en `evaluacion/mcs.py`
  y `evaluacion/curvas.py`, y en el notebook 05.
- [ ] 1.4 Correcciones de documentación (sección 5).
- [ ] **CP1.** Qué termómetros sobreviven y tabla MCS de los 12.

**Fase 2 · Detectores sobre el pool**

- [ ] 2.1 Matriz detector × menú (variantes en `registry.py`).
- [ ] 2.2 D14 absorption ratio (B5, solo pista B).
- [ ] 2.3 D15 jump Shu (B6) y D18 sparse JM (B9, con `jumpmodels.SparseJumpModel`).
- [ ] 2.4 D16 logit L1 y D17 random forest (B7), entrenados solo con crisis terminadas.
- [ ] 2.5 Ablaciones y controles (6.5, puntos 4–6).
- [ ] 2.6 C1 fusión online.
- [ ] 2.7 Ranking v3, sintéticos rehechos, decisión final, memoria y ADR-009 de resultados.
- [ ] **CP2.**

Opcionales: A4 (ranking ajustado, el más caro) y B8 (BOCPD).

### 6.9 Decisiones

**Tomadas (19-10-2026)**

- Tipos dividido en «nivel y curva» (T2) y «velocidad de tipos» (T3).
- Umbral de varianza explicada de los termómetros: se fija probando, con criterios que no usan crisis.
- Notebooks nuevos antes de los sintéticos y sintéticos rehechos al final (6.7).
- Absorption ratio solo en la pista B.

**Pendientes (con la recomendación entre paréntesis)**

- [ ] Ranking principal en el menú T (sí; MCS dentro de cada menú).
- [ ] D11 y D12 solo en sus menús actuales, U y N (sí, por coste y porque ya son negativos controlados).
- [ ] Menú T para los univariantes, por ejemplo un GARCH sobre la media de termómetros (no; deja de
  ser el método de la literatura).
- [ ] Pérdida del MCS (acierto por crisis + días de falsa alarma) y remuestreo (por bloques de días,
  α = 0,10).
- [ ] Tasa objetivo de falsas alarmas para A3 (1 al año).
- [ ] Parámetros del absorption ratio (los de Kritzman: ventana de 500 días, 1/5 de autovectores,
  cambio a 15 días frente a 1 año por encima de 1σ).
- [ ] Penalización del jump Shu (utilidad de estrategia, sin etiquetas).
- [ ] Rama de trabajo para la implementación (`feat/detectores-pool` desde `main`).
