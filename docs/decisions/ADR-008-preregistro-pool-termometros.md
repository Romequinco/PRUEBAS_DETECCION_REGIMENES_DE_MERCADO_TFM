# ADR-008 — Pre-registro: termómetros por bloque, menús de datos y vara de medir

- **Estado:** Aceptada y **congelada** · 2026-10-30, antes de ejecutar ningún detector con los menús nuevos y antes
  de cruzar los termómetros con las crisis. Bloques, anclas, umbral, calentamiento, menús, vara de medir y criterios
  de éxito (§6) quedan fijados; cualquier cambio posterior se declara y se fecha aquí.
- **Origen:** plan de [`PendientesOscar.md`](../../PendientesOscar.md) §6 (19-10-2026) y decisiones del 30-10-2026.
- **No cambia:** ventanas, series, crisis ni trampas de [`configs/benchmark_spec.yaml`](../../configs/benchmark_spec.yaml)
  (ADR-001/002/003); la interfaz `RegimeDetector`; los 12 detectores originales y su ranking v2, que se conserva
  como «configuración a priori de cada detector».

---

## 1. Contexto

El ranking v2 mezcla método y datos: cada detector recibe un subconjunto distinto del pool (D10 gana con 4
columnas, D04 queda décimo con 14). Solo 8 de las 41 series de la pista A y 15 de las 106 de la B llegan a algún
detector. Dar las 77 columnas de B a un HMM con covarianza completa no es viable (3.080 parámetros por estado con 9
crisis), así que el pool se **comprime por bloques económicos** y se ofrece a todos los detectores en **menús
comunes**.

## 2. Termómetros por bloque

- **Una PCA por bloque**, no una global (la global la dominan los bloques grandes y no se interpreta).
- **Causal:** reajuste mensual con datos ≤ fecha de reajuste (ventana expansiva); las cargas se aplican a los días
  estrictamente posteriores hasta el siguiente reajuste. Re-estandarización dentro de la ventana.
- **Signo** fijado por una **columna ancla** con signo declarado: el termómetro sube con el estrés tal como se
  declara.
- **Código:** `regimenes.features.bloques`; construcción, pruebas de diseño y validación en el notebook
  [`04_termometros`](../../notebooks/04_termometros.ipynb).

### 2.1 Decisiones de diseño (30-10-2026, sin etiquetas de crisis)

Todas se tomaron con criterios que no miran las crisis: varianza de la PC1, cambios de método, saltos de signo
entre reajustes y correlación con índices `rol=validation`. Son estadísticos de **toda la muestra** (OOS incluido);
no usan etiquetas, pero no son una elección walk-forward, y así se declara. Detalle y tablas en `04_termometros`,
secciones 4–6.

| Decisión | Elegido | Evidencia (prueba de humo) |
|---|---|---|
| Reparto en bloques | **10 en B, 7 en A** ([`configs/bloques.yaml`](../../configs/bloques.yaml)): T1 renta variable (estado) · T1b rentabilidad diaria · T2 tipos nivel y curva · T3 velocidad de tipos · T4 inflación esperada y tipo real (B) · T5 vol implícita (B) · T6 crédito · T7 divisas y materias primas (B) · T8 actividad · T9 inflación realizada y valoración | La propuesta de §6.3 mezclaba en T1 columnas de estado (autocorrelación ≈ 1) con rentabilidades diarias (≈ 0, correlación ≈ 0 con la vol): PC1 ≈ 0,29 en B, 69 % de tramos en media y signo al azar. Separadas: PC1 ≈ 0,62, 0 saltos, ρ con STLFSI4/VIX de 0,58 → 0,78 (B) y de 0,41 → 0,81 (A). En A la PC1 de la macro era inflación (4 saltos de signo); separada, T8 tiene 0 saltos y ρ con la regla de Sahm de 0,51 → 0,75. Se separan en vez de descartar para no perder pool |
| Anclas | T1 `SP500_vol_z` +1 · T1b `SP500_ret_z` −1 · T2 `short_rate_z` +1 · T3 `DGS10_change_z` +1 · T4 `breakeven_10y_z` +1 · T5 `VIX_level_z` +1 · T6 `credit_BAA10Y_z` +1 (A: `credit_BaaAaa_mensual_z`) · T7 `DXY_change_z` +1 · T8 `UNRATE_chg_z` +1 · T9 `PPI_yoy_z` +1 | T2: la pendiente cargaba poco y daba 2 saltos de signo en A; el tipo corto, 0. T9: PPI 0 saltos frente a 2 con CPI. La inflación realizada no va a T4 en B (PC1 0,68 → 0,43 y 3 saltos) |
| Umbral de la PC1 | **30 %** | Con el reparto final la PC1 nunca baja de 0,30: el umbral no se activa (= siempre PCA). 40 y 50 % meten 4–8 cambios PCA ↔ media sin ganar validez externa. Se mantiene como red de seguridad; vigilar T7 (B), con mínimo 0,30 |
| Calentamiento | **Toda la historia anterior a la ventana** (`historia_previa: toda`), recortada a la ventana | Solo dentro de la ventana, B no tiene termómetros hasta 2008-05 y el OOS del menú T no casaría con el del N. Con 5 años o más previos coincide con toda la historia (ρ ≈ 0,99); con 2–3, los primeros reajustes difieren. Causal: solo usa pasado, como los z-scores |

## 3. Menús de datos

| Menú | Contenido | A / B |
|---|---|---|
| U · Univariante | `SP500_ret` | 1 / 1 |
| N · Núcleo | CORE_A / CORE_B (actual) | 9 / 14 |
| T · Termómetros | pool comprimido (§2.1) | 7 / 10 |
| P · Pool completo | todas las columnas | 35 / 77 |
| R · Rentabilidades multiactivo | ~20 series crudas de activos | — / ~20 |

Cada detector se ejecuta **solo en los menús que admite por naturaleza**:

| Detector | U | N | T | P | R |
|---|:-:|:-:|:-:|:-:|:-:|
| D01 regla de volatilidad | ✓ | | | | |
| D02 regla compuesta | | ✓ | ✓ | | |
| D03 GMM · D04 HMM · D08 HMM-t · D09 jump | ✓ | ✓ | ✓ | | |
| D05 MS · D06 GARCH · D07 CUSUM | ✓ | | | | |
| D10 turbulencia | | ✓ | ✓ | ✓ | ✓ |
| D11 MS-GARCH | ✓ | | | | |
| D12 autoencoder | | ✓ | | | |
| D14 absorption ratio (solo B) | | | | | ✓ |
| D15 jump Shu | ✓ (features de Shu) | | | | |
| D16 logit L1 · D17 random forest (supervisados, categoría aparte) | | | ✓ | ✓ | |
| D18 sparse jump model | | | | ✓ | |

Decisiones del 30-10-2026: D11 y D12 solo en sus menús actuales (coste, y ya son negativos controlados); los
univariantes **no** van al menú T (dejarían de ser el método de la literatura). Salen 31 combinaciones
detector × menú en B y 29 en A, frente a las 90 de «todos en todo» (18 detectores × 5 menús) en B.

**Numeración:** D13 sigue siendo `hsmm_tstudent` (ablación de D08). Nuevos: D14 absorption ratio, D15 jump Shu,
D16 logit L1, D17 random forest, D18 sparse jump model.

## 4. Vara de medir

- **Ranking principal v3:** el del **menú T** (los seis multivariantes no supervisados, D02, D03, D04, D08, D09 y
  D10, ven exactamente los mismos datos; D16 y D17 van en su categoría supervisada). Además, un MCS dentro de cada
  menú.
- **MCS** (Hansen, Lunde y Nason, 2011; `regimenes.evaluacion.mcs`): pérdida = fallo por crisis + días de falsa
  alarma; bootstrap por bloques de días; α = 0,10. Baselines triviales incluidos (`SIEMPRE_*`, `AZAR_*`, `VOL_P90`).
  *Versión preliminar* (sin paneles diarios): remuestreo por crisis sobre `results/benchmark/metrics/*.csv`.
  - *Pérdida y remuestreo definitivos (02-11-2026, cierran el punto del §7).* Pérdida diaria sobre los paneles
    walk-forward OOS, con media (1/K)·Σ_c (1 − cobertura_c) + λ·FPR y λ = 1 (1 − J de Youden con las crisis
    equiponderadas; los baselines sin información pierden 1). «Fallo por crisis» es la fracción de días de la
    ventana [pico, suelo] no marcados; «días de falsa alarma», la fracción de días de calma marcados (las trampas
    cuentan como calma). Estado de crisis = `n_states − 1`, leído del CSV de métricas. Réplicas «de razón»:
    cobertura y FPR se recalculan sobre los días de cada réplica (con pesos fijos, `SIEMPRE_CALMA` tendría
    desviación bootstrap 0,17–0,23 y en B entrarían en el MCS los 17 modelos, `SIEMPRE_CRISIS` incluido); una
    crisis ausente de la réplica sale del promedio. Bootstrap estacionario con bloque medio = mediana de sesiones
    OOS de las ventanas de crisis (66 en A, 50 en B; solo usa las ventanas del juez). `T_max`, 5.000 réplicas,
    semilla 0. Sensibilidades: bloque ×½, ×2, Politis-White e iid; `T_R`; pesos fijos (el estándar de `arch`);
    versión por crisis. `VOL_P90` usa `SP500_vol_z` (z causal de la volatilidad a 21 días) con percentil 90
    expansivo, `shift(1)` y 252 observaciones mínimas; los `AZAR_*` del MCS son una realización con la tasa y la
    racha medianas de los detectores de la pista.
- **Curva a igual tasa de falsas alarmas** (`regimenes.evaluacion.curvas`): 0,5, 1 y 2 falsas alarmas al año; tasa
  de detección, retraso mediano y ratio ruido-señal.
- **VOL_P90:** baseline con percentil 90 expansivo y causal de la volatilidad.

## 5. Validación de los termómetros (antes de dárselos a ningún detector)

Sin etiquetas de crisis:

1. **Bien calculados (tests):** factor plantado, prefijo, signo estable, NaN que no se propagan.
2. **Sentido estadístico y económico**: la PC1 supera el umbral elegido en ≥ 80 % de los
   reajustes (si no, el bloque queda como media simple, que sigue siendo válida); ninguna columna pesa > 50 % en
   ≥ 80 % de los reajustes (no aplica con 1 o 2 columnas); coseno mediano de las cargas entre reajustes consecutivos
   ≥ 0,9 y sin saltos de signo; signos económicamente lógicos.
3. **Validez externa:** correlación con las series `rol=validation` (NFCI y subíndices, ANFCI, OFR_FSI y subíndices,
   STLFSI4, KCFSI, regla de Sahm, probabilidad de recesión), que nunca entran como features. Criterio: |ρ| ≥ 0,3 con
   alguna de sus referencias (T1 con STLFSI4 y el VIX; T5 con OFR_FSI_VOLATILITY y el VIX; T6 con NFCI_CREDIT y
   OFR_FSI_CREDIT; T8 con la regla de Sahm y la probabilidad de recesión).

Con crisis:

4. **Relevancia:** cada termómetro como regla de umbral frente a `persistent_random_null`; mapa bloques × crisis y
   comportamiento en las trampas (2013 se espera en T3, no como crisis).
5. **Valor añadido:** cada detector con N frente a T; quitar un bloque cada vez; controles de media simple en vez de
   PCA y de 100 reparticiones aleatorias de columnas a bloques.
6. **Significación:** MCS dentro de cada menú con los baselines.

## 6. Criterios de éxito y de fracaso (aprobados el 30-10-2026)

- **El pool aporta** si, en **las dos pistas**: (a) la mayoría de los multivariantes del menú T (D02, D03, D04, D08,
  D09, D10) mejora su `score_deteccion` frente a su versión N; (b) la mediana de esos scores con bloques económicos
  supera al percentil 90 de las 100 reparticiones aleatorias; y (c) algún detector del menú T está en el MCS del
  conjunto «T ∪ N».
- **Si no se cumple**, se reporta como resultado: «con el pool completo, lo simple sigue ganando».

## 7. Abierto (se decide al llegar a la Fase 2 del plan)

- Parámetros del absorption ratio (D14): candidatos de Kritzman (ventana de 500 días, 1/5 de autovectores, cambio a
  15 días frente a 1 año por encima de 1σ).
- Penalización del jump de Shu (D15): selección por utilidad sin etiquetas frente al valor publicado.

Resueltos el 30-10-2026 (§2.1): reparto en bloques, anclas, umbral y calentamiento. Resueltos el 02-11-2026: la
pérdida definitiva del MCS con días de falsa alarma (§4) y el cálculo con historia previa en el pipeline del menú T
(`regimenes.benchmark.menus_oficial`): su OOS coincide día a día con el del menú N en las dos pistas
(`results/detectores/menus/indice_{A,B}.json`), así que N se toma de v2 sin re-ejecutar (re-ejecutar B/D02-N
reproduce v2 celda a celda y con el panel idéntico).

## 8. Estado de la ejecución (02-11-2026)

**Pasada de referencia (plan 1.1).** En el portátil bueno, el notebook 03 reconstruye los 5 parquet de
`data/processed` con el mismo sha256 que v2. El benchmark completo (`--force`, 24 trabajos) reproduce v2 bit a bit:
24 de 24 paneles y estados idénticos, diferencia máxima en métricas de 0,0. Etiqueta local `v2-reproducido`.

**Termómetros oficiales** (`data/processed/pista{A,B}_termometros.parquet`, sha256 `2fb84668…` en A y `41d3fdd6…`
en B; notebook 04). Validación 2: cumplen 10 de 10 en B y 5 de 6 aplicables en A; el que falla es **T9 de A**, que
pierde la estabilidad por un salto de signo cuando entran `CPI_core_yoy_z`, `PPI_metals_yoy_z` y `CAPE_z`. El salto
cae antes del OOS de A y se reporta tal cual: T9 sigue en el menú T. Validación 3: cumplen todos los que tienen
referencia (3 de 3 en A y 4 de 4 en B). Validación 4: superan al nulo persistente T1, T1b y T3 en A, y T1b, T3, T5 y
T7 en B. Con Holm al 5 % por pista (exploratorio) solo cae T1b de B. T3 es el que más marca el *taper* de 2013.

**Vara de medir definitiva** (§4; notebook 05 §9–12, `results/benchmark/vara/`). Al 10 %, el MCS de los 12 con
baselines retiene 11 en A (fuera D09) y 8 en B (fuera D12, D08, D03 y D09). D06 y D05 encabezan las dos pistas.
`VOL_P90` entra en el MCS de A (p 0,39) y queda fuera en B (p 0,004). En `score_deteccion` (0,600 en A y 0,525 en B) solo lo superan D10 en A, y D05 y D06
en B. Los baselines triviales quedan fuera en las dos pistas. Frente a
`arch.bootstrap.MCS` con 1000 a 2000 réplicas, la diferencia máxima de p está entre 0,02 y 0,07. El conjunto
coincide en 3 de 4 casos y difiere con T_max en B, dentro del error Monte Carlo que documenta el notebook 05
(§10.5). La revisión crítica corrigió `senales_crisis`:
un detector que no visita su estado de crisis se leía como «siempre crisis». No cambió ninguna cifra.

**Menú N frente a menú T, oficial** (`results/detectores/menus/`, `python -m regimenes.benchmark.menus_oficial`).
Mismo OOS que v2 en todos los casos (`indice_{A,B}.json`), y el D02 de N en B reproduce v2. Mejoran con T **1 de 6 en
A** (solo D02, +0,011) y **5 de 6 en B** (todos salvo D04, −0,079). En A, D03 cae −0,220 y D04 −0,106. En el MCS de
«T ∪ N» (18 modelos, sin baselines), A no retiene ninguna variante T (11 de 18, todas en N) y B retiene cuatro (D02_T
con p 0,36, D10_T, D04_T y D08_T; 12 de 18). Datos en `comparacion_N_T.csv`, `mcs_T_union_N_{A,B}.csv` y
`criterios_adr008.csv`.

**Veredicto del §6.** (a) se cumple en B y no en A. (c) se cumple en B y no en A. (b) queda para la Fase 2 (2.5),
pero ya no puede cambiar el veredicto. Como el §6 exige las dos pistas, **el pool no aporta tal como se
pre-registró**: «con el pool completo, lo simple sigue ganando». En B los termómetros sí ayudan a los multivariantes,
lo que justifica seguir con la Fase 2 como análisis, sin cambiar el veredicto pre-registrado.

### 8.1 Prueba de humo (30-10-2026, superada por lo anterior)

Portátil secundario sin `data/processed` ni paneles walk-forward y sin tiempo para el benchmark. Hecho como **prueba
de humo**: paneles construidos en memoria (≈ 4 s por pista, sin escribir a disco ni verificar sha256), pruebas de
diseño, termómetros y validación 1–3, y MCS preliminar. Los crudos de esta máquina son una descarga anterior a
ADR-003: faltan `CAPE_z` (no está `SHILLER_SP500_CAPE`), `FF_industry_dispersion_z` y `FF_RMW_z` (`FF_5_INDUSTRY` y
`FF_FACTORS_5_DAILY` en su versión de una columna). Con el reparto final, los 17 termómetros cumplen las validaciones
2 y 3 (T3 de A es una sola columna y T6 de A tiene dos, donde algún criterio no aplica). La versión oficial se hará
sobre los datos verificados.

Verificación adversarial (ola 3): sin fugas de causalidad (prefijo exacto con paneles reales, cargas aplicadas
solo a días posteriores al reajuste); MCS idéntico a `arch.bootstrap.MCS` dentro del error Monte Carlo. Corregidos
en `curvas.py`: con una puntuación binaria se elegía «marcar todos los días», fallaba sin días de calma y el detalle
en `attrs` rompía `pd.concat`.

Primer cruce con las crisis (prueba de humo, `04_termometros` §10–11; datos en `results/detectores/menus_humo/`): el
menú T empeora en 5 de 6 casos frente al N (solo D10 en B sube, +0,020, con 2 crisis menos). Las 6 cifras de N
coinciden con `ranking_v2.csv` hasta el tercer decimal. En B ya empeoran 3 de 6 multivariantes, así que el criterio
(a) del §6 no se cumpliría si se confirma. No es definitivo: el menú T va sin 3 columnas, faltan D04 y D09 en las dos
pistas y D08 y D10 en la A, y no hay MCS. Bloques y anclas no se han retocado tras ver estos resultados.
