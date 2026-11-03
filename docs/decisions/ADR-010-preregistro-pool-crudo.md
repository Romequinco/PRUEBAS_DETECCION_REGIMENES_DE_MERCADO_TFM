# ADR-010 — Pre-registro de la Fase 2: ¿aporta el pool sin comprimir frente al núcleo N?

- **Estado:** Aceptado y **congelado** · 2026-11-03, antes de escribir el código de los detectores nuevos y antes
  de ejecutar ninguno. Cualquier cambio posterior se declara aquí como errata fechada.
- **Origen:** [`PendientesOscar.md`](../../PendientesOscar.md) §6.8, Fase 2 replanteada el 02-11-2026; decisiones
  del usuario del 02-11-2026.
- **Parte de:** [ADR-008](ADR-008-preregistro-pool-termometros.md), con sus menús, su vara de medir y su MCS, y de
  [ADR-009](ADR-009-resultados-pool.md), que descarta el menú T: el núcleo es N, con U para los univariantes.
- **No cambia:** los 12 detectores de v2, sus 24 resultados y su caché, el notebook 21, las ventanas, las crisis ni
  las trampas.
- **Numeración:** ADR-006 sigue reservada a la decisión final del detector de régimen. Los resultados de esta fase
  se escriben en el §9 de este ADR.

---

## 1. Pregunta

El ADR-009 mostró que **comprimir** el pool en termómetros por bloque no mejora a los 6 multivariantes frente a N.
Queda la otra mitad de la pregunta «¿sirve el pool?»: **¿aporta el pool sin comprimir?** Se prueba de dos formas:

- **Con P**, el pool entero: 35 columnas en A y 77 en B, las de `configs/bloques.yaml`. Para que no se sobreajuste
  con tantas columnas, se usan detectores que las regularizan o que eligen ellos mismos con cuáles quedarse.
- **Con R**, 22 rentabilidades crudas de activos, solo en la pista B. Es el dato que usan Kritzman y Li (2010)
  para la turbulencia y Kritzman et al. (2011) para el absorption ratio.

## 2. Menús nuevos

### 2.1 P · Pool completo

- **Columnas:** las de `configs/bloques.yaml` (35 en A y 77 en B), con la transformación del preprocesado (z-scores
  causales).
- **Regla de disponibilidad** (causal y sin etiquetas). En A hay columnas que empiezan tarde:
  - `UMCSENT_z` empieza en 1969-12;
  - `HOUST_yoy_z` en 1965-02;
  - `CPI_core_yoy_z` y `PPI_metals_yoy_z` en 1963-02;
  - `GDP_growth_z` en 1962-05.

  En B, las dos de HYG empiezan en 2007-07. Ninguna tiene huecos en el OOS (comprobado el 03-11-2026). La regla es:
  - en cada reajuste, una columna entra solo si tiene **al menos 756 observaciones (3 años)** en la ventana de
    entrenamiento;
  - el ajuste usa las filas de la ventana en las que están todas las columnas que entran;
  - hasta el siguiente reajuste se predice con esas mismas columnas.

  Se implementa como un envoltorio del detector, no dentro de cada detector. En la práctica solo afecta a
  `UMCSENT_z`, que entra en el primer reajuste posterior a 1972-12, y a nada en B.

### 2.2 R · Rentabilidades multiactivo (solo pista B)

- **Series** (22; todas empiezan no después del 2007-04-11, inicio de la pista B; `HYG_CREDIT` justo ese día):

| Grupo | Series (`data/raw/`) |
|---|---|
| Renta variable | `SPDR_XLB`, `SPDR_XLE`, `SPDR_XLF`, `SPDR_XLI`, `SPDR_XLK`, `SPDR_XLP`, `SPDR_XLU`, `SPDR_XLV`, `SPDR_XLY`, `SP500`, `NASDAQ_COMP`, `RUSSELL2000` |
| Renta fija | `TLT_TREASURY`, `IEF_TREASURY` |
| Crédito | `LQD_IGCREDIT`, `HYG_CREDIT` |
| Materias primas | `GOLD_GLD`, `SPGSCI` |
| Divisas | `DXY` (yfinance), `DEXJPUS`, `DEXSZUS`, `DEXUSAL` (FRED) |

- **Transformación:** rentabilidad logarítmica diaria del cierre, sin z-score, porque así lo definen Kritzman y Li.
  Las divisas de FRED se orientan como rentabilidad del dólar (`DEXUSAL` se invierte) y, como el resto de series
  diarias de mercado, **no llevan retraso** (ADR-003 §2.1; `features.lags` no las incluye, igual que en el
  preprocesado de la pista B). Los días sin cotización se dejan en NaN, sin rellenar.
- **Índice y hueco inicial:** el de `pistaB_diaria.parquet`. `HYG_CREDIT` empieza el 2007-04-11, así que la primera
  rentabilidad es del 2007-04-12. Antes del OOS (2010-04-12) hay 750 filas con las 22 series (las FRED pierden 6
  días festivos). Se aplica la misma regla de disponibilidad de P.
- **Panel:** `data/processed/pistaB_rentabilidades.parquet`. Se reconstruye desde `data/raw` sin descargar nada y
  se guarda su sha256.

## 3. Detectores nuevos y variantes

Todas las variantes usan el mismo walk-forward (expansivo, salvo D15, que usa la ventana móvil de 3.000 días del
paper, §3.1), el mismo índice OOS que v2 (se comprueba día a día), la misma vara
de medir (`score_deteccion` y MCS diario del ADR-008 §4) y semilla 42. Los hiperparámetros quedan fijados aquí y no
se afinan con crisis.

| Id | Detector | Menús | Especificación pre-registrada |
|---|---|---|---|
| D10-P | Turbulencia de Mahalanobis | P (A y B) | Los mismos parámetros que D10 (`q_in = 0,90`, `q_out = 0,70`, `min_dwell = 5`, `cov_min_periods = 252`); solo cambian las columnas |
| D10-R | Turbulencia de Mahalanobis | R (B) | Ídem; es la especificación original de Kritzman y Li (2010) |
| D14 | Absorption ratio (Kritzman et al., 2011) | R (B) | Matriz de covarianza de las 22 rentabilidades en una ventana de 500 filas completas (las 22 con dato), con ponderación exponencial (semivida de 250, la mitad de la ventana). AR es la fracción de varianza que explican los n = 4 primeros autovectores (≈ 1/5 de 22, redondeado). Desplazamiento estandarizado: ΔAR = (media del AR en 15 días − media del AR en 252) / desviación típica del AR en 252. Crisis = ΔAR ≥ 1. Sin ajuste a crisis: todo se calcula con datos ≤ t. Si en un día del OOS aún no hay 252 AR (calentamiento, a lo sumo los primeros días), se predice calma y se reporta cuántos días |
| D15 | Jump model de Shu, Yu y Mulvey (2024) | U (A y B) | Features del paper (§3.4.1, tabla 2), sobre la rentabilidad del S&P 500 en exceso de `DTB3`: desviación a la baja EWM (semivida 10) y ratio de Sortino EWM (semividas 20 y 60), estandarizadas con media y desviación de la ventana de ajuste. `jumpmodels.JumpModel`, 2 estados, 10 arranques. **Penalización elegida en cada reajuste** por utilidad de estrategia, sin etiquetas (§3.1) |
| D18 | Sparse jump model (Nystrup et al., 2021) | P (A y B) | `jumpmodels.SparseJumpModel` (v0.1.1), `n_components = 2`, `jump_penalty = 50` (la de D09; la librería la divide internamente por √p) y `max_feats = 9` en A y 14 en B, el tamaño de N. `max_feats` es κ², el número **efectivo** de variables (restricción L1 sobre los pesos), no un número exacto: se reportan las columnas con peso > 0 |
| D16 | Logit L1 (supervisado) | P (principal) y N (comparación) | `C = 1`, `class_weight = "balanced"`, `solver = "liblinear"`. Crisis si la probabilidad es ≥ 0,5. Etiquetas en §3.2. La comparación barata pasa de T (ADR-009 §7) a N, porque T está descartado |
| D17 | Random forest (supervisado) | P (principal) y N (comparación) | 500 árboles, `max_depth = 5`, `min_samples_leaf = 50`, `class_weight = "balanced_subsample"`, semilla 42. Crisis si la probabilidad es ≥ 0,5. Etiquetas en §3.2 |

**Estado de crisis** (el `crisis_state = n_states − 1` que leen el walk-forward y el MCS):

- **D15 y D18:** la canonicalización económica de `base.py`, igual que D09, con `market_returns = SP500_ret` del
  panel de la pista. El paper de Shu asigna el estado bajista por rentabilidad acumulada en exceso; se usa la
  canonicalización común para que D15 sea comparable con D09, y se reporta si ambas reglas discrepan en algún reajuste.
- **D10-P, D10-R y D14:** estado interno 1 = la regla dispara (turbulencia o ΔAR ≥ 1), con la misma
  canonicalización que D10. Se reporta si invierte el orden en algún reajuste.
- **D16 y D17:** clase 1 = crisis, sin canonicalización económica.

### 3.1 Penalización de D15 (sin etiquetas)

Lo que hace el paper (§3.4.1–3.4.3, v3 de arXiv:2402.05272, = versión de la revista): ajusta Θ cada 6 meses en una
ventana móvil de 3.000 días; cada mes elige λ por el Sharpe de la estrategia 0/1 sobre los estados **online** de
una ventana de validación de 8 años; señal con un día de retraso (la de t se opera desde t + 2), 10 pb por
operación en cada sentido; en bajista, al activo libre de riesgo. No publica la rejilla (su tabla 3 ilustra
λ ∈ {0, 5, 15, 35, 70, 150}; la v1 usaba {10, 22, 50, 100, 220, 500, 1000} y eligió 100 para el S&P 500).

- **Rejilla pre-registrada:** λ ∈ {5, 10, 22, 50, 100, 220}, espaciada en log; cubre el valor típico del paper (50)
  y el elegido en la v1 (100).
- **Elección en cada reajuste t** (paso 21, como D09; el paper reajusta Θ cada 6 meses, desviación declarada):
  - Con n días de entrenamiento, la validación son los últimos V = min(2.016, ⌊n/2⌋) días (8 años si caben) y el
    ajuste, los hasta 3.000 días anteriores a ella. Con cada λ se ajusta ahí y se infiere **online** en la
    validación con Θ fijo.
  - Se elige la λ con mayor **Sharpe de la estrategia 0/1 del paper** en la validación: S&P 500 en calma y `DTB3`
    (con su rentabilidad diaria) en crisis, señal de t operada desde t + 2, 10 pb por operación. Empates: la λ mayor.
  - Con la λ elegida se reajusta en los últimos min(3.000, n) días.
- **No usa etiquetas de crisis.** La validación es pasado del reajuste.
- **Historia previa** (decidido el 03-11-2026). Con solo la ventana de la pista, D15 no llega a los 3.000 + 2.016
  días del paper: en B nunca, y en A tarda años. Se usa la historia anterior a la pista, como hizo el menú T (ADR-008
  §2.1, `historia_previa: toda`). Las fuentes son `data/raw/yfinance/SP500.parquet` (desde 1927-12) y
  `data/raw/fred/DTB3.parquet` (desde 1954-01), y las features se cortan a ese inicio. Es causal: en cada reajuste
  solo se usan días ≤ t. El OOS y su índice son los de v2.

### 3.2 Etiquetas de los supervisados (sin fuga)

- **En cada reajuste t,** un día del entrenamiento es «crisis» si cae en la ventana [inicio, suelo] de una crisis
  de `data/processed/labels.parquet` (columnas `inicio` y `trough_date`; en las 28 filas `fin = trough_date`) de esa
  pista cuyo **suelo es anterior a t**. Las crisis que aún no han terminado en t no
  existen como etiqueta: **sus días cuentan como calma**, igual que los vería alguien en tiempo real. Es la variante
  estrictamente causal, y es la principal (decidido el 03-11-2026). Como sensibilidad, sin peso en el veredicto, se
  reporta la variante que saca del ajuste los días de crisis aún sin terminar. Evita enseñar que el principio de una
  crisis es calma, pero elige las filas con una etiqueta futura.
- **Trampas:** las de `regimenes.evaluacion.ranking.track_false_positive_windows` (`taper_2013` y
  `debtceiling_2013`; `selloff_2018`, aunque está bajo `false_positive_windows`, es crisis del catálogo y no trampa).
  Cuentan como calma.
- **Sin crisis terminadas:** si en un reajuste aún no hay ninguna, el detector predice calma en todo ese tramo.
- **Categoría aparte.** D16 y D17 usan etiquetas, así que no entran en el ranking principal ni en el criterio del §5,
  que es solo de no supervisados. Tienen su propio veredicto (§6).

## 4. Implementación (restricciones)

- **Código nuevo en `src/regimenes/detectores/pool/`**, una carpeta sin la «f» inicial. Las especificaciones van en
  `regimenes.detectores.menus` (o en un módulo nuevo junto a él), **nunca en `registry.py`, `base.py`,
  `detectores/__init__.py`, `evaluacion/walk_forward.py`, `evaluacion/metricas.py` ni `detectores/f*/`**, y sin
  tocar `configs/benchmark_spec.yaml` ni los parquet de `data/processed`. Esos ficheros entran en la huella de la
  caché v2 (`benchmark.cache._cache_code_paths` y `_cache_fingerprint`) y,
  a través de ella, en la del notebook 21. Tocarlos invalidaría los 24 resultados oficiales y el cálculo del
  aumento.
- **Comprobación al cerrar:** la caché v2 sigue vigente (`comprobar_benchmark`) y el `estado` del notebook 21 no
  recupera trabajos ya hechos.
- **Tests:**
  - prefijo: calcular hasta t da lo mismo que calcular con todo y recortar en t;
  - disponibilidad (§2.1);
  - fuga de etiquetas (§3.2): una crisis sin terminar no aparece;
  - D14 con un factor plantado: el AR sube cuando se inyecta un factor común;
  - semilla fija.
- **Resultados:** `results/detectores/pool/`, con los mismos esquemas que `results/detectores/menus/`.

## 5. Criterio de éxito del pool sin comprimir (no supervisados)

**El pool sin comprimir aporta** si, en **cada una de las dos pistas**, algún detector no supervisado de P (o de R,
en B) cumple las dos condiciones. Se pide «alguno» y no «la mayoría», como en el ADR-008 §6(a): aquí hay 2 o 3
detectores por pista y cada uno usa el pool de una forma distinta. La exigencia está en las dos pistas y en (a) y
(b) a la vez.

- **(a) Mejora a su par en N** en `score_deteccion` (estrictamente mayor, sin margen):
  - D10-P y D10-R frente a D10 de v2 (sus 4 columnas de N, no las 9/14 del núcleo completo);
  - D18 frente a D09, la misma familia con las columnas elegidas a priori;
  - D14 frente a D10, que es la turbulencia sobre N y el detector de N más cercano en espíritu.
- **(b) Entra en el MCS al 10 %** del conjunto de todos los no supervisados de N, U, P y R de esa pista, con los
  baselines incluidos (`SIEMPRE_*`, `AZAR_*`, `VOL_P90`). Es la misma vara de medir que el ADR-008 §4: pérdida
  diaria 1 − J de Youden con crisis equiponderadas, bloque de 66 días en A y 50 en B, 5.000 réplicas.

Notas:

- **R solo existe en B,** así que en A el criterio solo puede cumplirse con P, y así se declara.
- **Si se cumple solo en una pista,** se reporta como **indicio**, no como aporte (mismo criterio que el ADR-008).
- **Si no se cumple,** se reporta como resultado: «ni comprimido ni en bruto: con datos limpios y sin sesgo, lo
  simple sigue ganando».
- **Se reporta además, sin peso en el veredicto:**
  - si el MCS deja fuera a `VOL_P90`;
  - qué columnas elige D18 en cada reajuste: frecuencia por columna y por bloque;
  - la comparación de D15 con D09.

## 6. Veredicto aparte de los supervisados («techo»)

D16 y D17 sobre P y sobre N se comparan con el mejor no supervisado de cada pista, con `score_deteccion` y con un
MCS de supervisados y no supervisados juntos, solo como lectura.

- **Si ninguno entra en el MCS o mejora al mejor no supervisado:** ni con etiquetas se gana a lo simple. Refuerza
  la conclusión.
- **Si alguno lo mejora:** hay información en los datos que los no supervisados no extraen. Se reporta, pero no
  cambia el ranking principal, porque usa etiquetas.

## 7. Coste y ejecución

- **Piloto de coste** antes del cálculo completo: un detector y una pista de cada tipo.
  - Si D15 o D18 superan, en conjunto, unas 60 h de CPU, la elección de λ de D15 se hace **una vez al año** (cada 12
    reajustes) en vez de en cada reajuste. Si hace falta, se declara aquí como errata **antes** de calcular.
  - El paso de reajuste de cada detector es el de su familia en v2: 21 días, como D09 y D10 (`DetectorSpec.step`
    por defecto en `registry.py`). D14 no se ajusta.
- **Ejecución:** con `Start-Process`, `--horas` y orden LPT, reanudable, en paralelo al notebook 21.

## 8. Verificaciones antes de congelar

- [x] Features, semividas, ventana de validación y coste de transacción de D15, contrastados con Shu, Yu y Mulvey
  (2024), arXiv:2402.05272v3 (17-09-2024) §3.4.1–3.4.3 y tabla 2 (03-11-2026). El borrador citaba features que no
  son las del paper (media de rentabilidad, desviación a la baja en log): corregido en §3 y §3.1.
- [x] Parámetros de D14 (03-11-2026), contrastados con la descripción que hacen Kinlaw, Kritzman y Turkington (2011,
  MIT Sloan WP 4940-11, ec. 2), «following Kritzman et al. (2011)»: 500 días, ≈ 1/5 de autovectores, semivida de 250
  días, ΔAR = (AR 15 días − AR 1 año) / σ del AR de 1 año, riesgo alto si ΔAR ≥ 1. El artículo de JPM no se ha
  podido leer (de pago).
- [x] Interfaz de `jumpmodels.SparseJumpModel` 0.1.1 instalada: `n_components`, `max_feats` (κ², número efectivo),
  `jump_penalty` (escalada por 1/√p), `random_state`, `n_init_jm` (03-11-2026).
- [x] Coherencia con el ADR-008 (vara de medir y MCS), el ADR-009 (núcleo N) y PendientesOscar §6.8 (03-11-2026):
  22 series en R (no 21), trampas por `track_false_positive_windows`, FX sin retraso (ADR-003), D10 de v2 con 4
  columnas, comparación de D16/D17 en N en vez de T.

## 8.1 Erratas de implementación (03-11-2026, antes de calcular)

Las ha encontrado la implementación. No cambian ningún parámetro ni el criterio: precisan casos que el texto no
cubría o corrigen un dato.

- **La regla de disponibilidad sí afecta a B** (corrige el §2.1, que decía que no).
  - En P, las dos columnas de HYG tienen 696 observaciones en el primer reajuste y quedan fuera de los 3 primeros.
  - En R, `HYG_CREDIT` (755) y las 3 divisas de FRED (750) quedan fuera del primero.
  - Se aplica la regla tal cual.
- **NaN de R dentro del OOS.**
  - Hay 42 festivos de FRED en las divisas y 2 días sin `DXY`. Se dejan en NaN, sin rellenar, como dice el §2.2.
  - Solo en el menú R se admite NaN en el OOS; en P, U y N un NaN en el OOS sigue siendo un error.
  - D10-R mantiene en esos días el trato de D10, que conserva el estado anterior.
  - D14 da a un día incompleto el ΔAR de la última fila completa, porque no se puede calcular el AR sin las 22
    series.
- **D14 no lleva el envoltorio de disponibilidad.** Su propia regla ya fija las filas: las 500 últimas con las 22
  series.
- **D14 sin canonicalización económica** (corrige el §3, que la pedía «como D10»). El revisor encontró que con la
  canonicalización por volatilidad el orden de estados salía invertido en los reajustes de 2011-02 a 2019-06. La
  causa son los ~750 días de calentamiento sin ΔAR, que abarcan toda la crisis de 2008 y se predicen como estado
  0, así que inflan la volatilidad de ese estado. Con el orden invertido, la «crisis» habría sido ΔAR < 1, en torno
  al 85 % de los días. ΔAR ≥ 1 es por definición el riesgo alto de Kritzman, así que **estado interno 1 = crisis,
  sin canonicalizar**, como D16 y D17. Se decidió antes de calcular ningún resultado de D14. D10-P y D10-R
  conservan la canonicalización de D10, para seguir siendo comparables con él, y se reporta si invierten el orden.
- **D18 estandariza** con `StandardScaler` ajustado en el entrenamiento, igual que D09.
- **«N ∪ U» en el MCS del §5(b)** son los 12 detectores de v2, que ya incluyen los univariantes sobre `SP500_ret`,
  más D15-U. Las variantes U de los multivariantes del ADR-008 §3 (D03-U, D04-U, D08-U y D09-U) no se han calculado
  y no entran.
- **D15** (detalles que el paper deja abiertos):
  - el orden de estados se fija con su ventana de ajuste de 3.000 días;
  - el Sharpe de la validación es el de la rentabilidad en exceso (pie de la tabla 4 del paper);
  - la inferencia online arranca al inicio de la ventana de ajuste;
  - el ajuste es el mismo en A y en B para una misma fecha, porque la historia cruda es común.
- **D16 y D17:** una crisis existe como etiqueta si su suelo es ≤ el último día del entrenamiento. Con
  scikit-learn 1.8, la penalización L1 del logit se pide con `l1_ratio = 1`.

## 9. Resultados

Calculado el 03-11-2026 entre las 10:05 y las 15:21: 20 variantes (9 en A y 11 en B), 4 procesos y 0 errores. Las
tablas están en `results/detectores/pool/`: `comparacion_pares.csv`, `mcs_NUPR_{A,B}.csv`, `criterio_adr010.csv`,
`techo_supervisados.csv`, `mcs_supervisados_{A,B}.csv` y `d18_{columnas,bloques}_{A,B}.csv`. Lo único que se salió
de lo previsto fue el coste de A_D18-P: el piloto estimaba 3,2 h y tardó unas 5,3 h, porque cada reajuste se
encarece a medida que crece la ventana.

### 9.1 Criterio (a): mejora a su par en N (`score_deteccion`)

| Pista | Variante | Par | Score | Score del par | Δ | (a) |
|---|---|---|---|---|---|---|
| A | D10-P | D10 | 0,527 | 0,614 | −0,087 | no |
| A | D18-P | D09 | 0,395 | 0,362 | +0,033 | sí |
| B | D10-P | D10 | 0,370 | 0,457 | −0,087 | no |
| B | D10-R | D10 | 0,475 | 0,457 | +0,018 | sí |
| B | D14-R | D10 | 0,350 | 0,457 | −0,107 | no |
| B | D18-P | D09 | 0,140 | 0,000 | +0,140 | sí |

### 9.2 Criterio (b): MCS al 10 % de N∪U∪P∪R con baselines

- **A:** quedan dentro 14 de 20. La pérdida más baja es la de D06 (0,550), seguido de D05, D01, D10, D12, VOL_P90,
  D15-U, D03, D04 y D02. **D18-P** queda dentro en el puesto 11 (0,761), y **D10-P** queda fuera (p = 0,038).
- **B:** quedan dentro 13 de 22. La pérdida más baja es la de D05 (0,610), seguido de D06 y D10. **D10-R** queda
  dentro (0,798), y también **D14-R**, **D10-P** y **D18-P**, todos con p = 0,113. VOL_P90 queda **fuera**.

### 9.3 Veredicto (§5)

**El pool sin comprimir aporta, según el criterio pre-registrado.** Se cumple en las dos pistas: en A con D18-P, y
en B con D10-R y D18-P.

**Lectura, sin peso en el veredicto.** El aporte es formal y estrecho:

- **Ningún detector de P o R supera al mejor de N.** En A, D06 tiene la pérdida más baja y D10 el mejor score
  (0,614). En B, D05 es el mejor en las dos cosas. D18-P y D10-R entran en el MCS porque el MCS al 10 % es amplio,
  no porque estén cerca de la cabeza.
- **El par de D18 es D09,** que es débil: queda fuera del MCS en las dos pistas y tiene score 0 en B. La condición
  (a) de D18-P se cumple, por tanto, con poco.
- **La única comparación de par fuerte es D10-R frente a D10 en B,** y la mejora es de +0,018.
- **D10 y D14 no ganan con el pool.** D10 sobre P empeora en las dos pistas (−0,087), y D14 sobre R también
  (−0,107).
- **Qué columnas elige D18:**
  - en A, sobre todo tipos (T2, en el 100 % de los reajustes) e inflación y valoración (T9, en el 98 %);
  - en B, crédito (BaaFF, en el 100 %), CAPE (82 %), drawdown del SP500 (68 %) y peticiones de subsidio (62 %).
- **D15-U frente a D09:** +0,164 en A y +0,523 en B. Queda dentro del MCS en las dos pistas, en el puesto 7 en A y
  el 4 en B.

### 9.4 Supervisados («techo», §6)

**Ningún supervisado mejora al mejor no supervisado:**

- **Pista A:** el mejor no supervisado es D10, con 0,614. El mejor supervisado es D16-N, con 0,498.
- **Pista B:** el mejor no supervisado es D05, con 0,647. El mejor supervisado es D16-N, con 0,441.

**Con el pool rinden peor que con N:**

| Pista | Modelo | Score con P | Score con N |
|---|---|---|---|
| A | D16 | 0,19 | 0,50 |
| A | D17 | 0,06 | 0,46 |
| B | D16 | 0,14 | 0,44 |
| B | D17 | 0,06 | 0,41 |

Las variantes `-excl` mejoran a la P base, pero siguen por debajo de N. En el MCS conjunto entran D16-N y D17-N en
las dos pistas, y además las dos `-excl` en B.

**Conclusión del §6:** ni con etiquetas se gana a lo simple.

### 9.5 Consecuencias

- **Ranking principal v3:** entran P y R, porque el §5 se cumple. En la práctica la cabeza no cambia: D06 en A y
  D05 en B.
- **`get_regimen`:** sigue con D06 en A y D05 en B hasta la decisión de CP2.
- **Para el TFM:** el pool sin comprimir aporta formalmente con D18-P y D10-R, pero no desplaza a los detectores
  simples sobre N.
