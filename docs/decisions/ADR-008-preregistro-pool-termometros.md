# ADR-008 — Pre-registro: termómetros por bloque, menús de datos y vara de medir

- **Estado:** Borrador · 2026-10-30. Se congela (pasa a *Aceptada*) **antes** de ejecutar ningún detector con los
  menús nuevos. Lo marcado como *propuesta* espera el visto bueno del equipo.
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
- **Umbral de varianza de la PC1:** se prueban 30, 40 y 50 % y se elige el que dé **cargas y signos más estables
  entre reajustes, sin usar etiquetas de crisis**. Si en un reajuste la PC1 no llega, ese tramo usa la media simple
  de las columnas orientadas.
- **Bloques y anclas:** [`configs/bloques.yaml`](../../configs/bloques.yaml) (8 bloques en B, 5 en A; reparto de
  PendientesOscar.md §6.3). Las anclas son *propuesta*.
- **Código:** `regimenes.features.bloques`; construcción y validación en el notebook
  [`04_termometros`](../../notebooks/04_termometros.ipynb).

## 3. Menús de datos

| Menú | Contenido | A / B |
|---|---|---|
| U · Univariante | `SP500_ret` | 1 / 1 |
| N · Núcleo | CORE_A / CORE_B (actual) | 9 / 14 |
| T · Termómetros | pool comprimido | 5 / 8 |
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

- **Ranking principal v3:** el del **menú T** (nueve detectores ven exactamente los mismos datos). Además, un MCS
  dentro de cada menú.
- **MCS** (Hansen, Lunde y Nason, 2011; `regimenes.evaluacion.mcs`): pérdida = fallo por crisis + días de falsa
  alarma; bootstrap por bloques de días; α = 0,10. Baselines triviales incluidos (`SIEMPRE_*`, `AZAR_*`, `VOL_P90`).
  *Versión preliminar* (sin paneles diarios): remuestreo por crisis sobre `results/benchmark/metrics/*.csv`.
- **Curva a igual tasa de falsas alarmas** (`regimenes.evaluacion.curvas`): 0,5, 1 y 2 falsas alarmas al año; tasa
  de detección, retraso mediano y ratio ruido-señal.
- **VOL_P90:** baseline con percentil 90 expansivo y causal de la volatilidad.

## 5. Validación de los termómetros (antes de dárselos a ningún detector)

Sin etiquetas de crisis:

1. **Bien calculados (tests):** factor plantado, prefijo, signo estable, NaN que no se propagan.
2. **Sentido estadístico y económico** (*propuesta de umbrales*): la PC1 supera el umbral elegido en ≥ 80 % de los
   reajustes (si no, el bloque queda como media simple, que sigue siendo válida); ninguna columna pesa > 50 % en
   ≥ 80 % de los reajustes; correlación media de cargas entre reajustes consecutivos ≥ 0,9; signos económicamente
   lógicos.
3. **Validez externa:** correlación con las series `rol=validation` (NFCI y subíndices, ANFCI, OFR_FSI y subíndices,
   STLFSI4, KCFSI), que nunca entran como features. *Propuesta:* |ρ| ≥ 0,3 con su índice de referencia (T6 con
   NFCI_CREDIT / OFR_FSI_CREDIT, T5 con OFR_FSI_VOLATILITY…).

Con crisis:

4. **Relevancia:** cada termómetro como regla de umbral frente a `persistent_random_null`; mapa bloques × crisis y
   comportamiento en las trampas (2013 se espera en T3, no como crisis).
5. **Valor añadido:** cada detector con N frente a T; quitar un bloque cada vez; controles de media simple en vez de
   PCA y de 100 reparticiones aleatorias de columnas a bloques.
6. **Significación:** MCS dentro de cada menú con los baselines.

## 6. Criterios de éxito y de fracaso (*propuesta*)

- **El pool aporta** si, en **las dos pistas**: (a) la mayoría de los multivariantes del menú T (D02, D03, D04, D08,
  D09, D10) mejora su `score_deteccion` frente a su versión N; (b) la mediana de esos scores con bloques económicos
  supera al percentil 90 de las 100 reparticiones aleatorias; y (c) algún detector del menú T está en el MCS del
  conjunto «T ∪ N».
- **Si no se cumple**, se reporta como resultado: «con el pool completo, lo simple sigue ganando».

## 7. Abierto (se decide al llegar a la Fase 2 del plan)

- Parámetros del absorption ratio (D14): candidatos de Kritzman (ventana de 500 días, 1/5 de autovectores, cambio a
  15 días frente a 1 año por encima de 1σ).
- Penalización del jump de Shu (D15): selección por utilidad sin etiquetas frente al valor publicado.
- Pérdida definitiva del MCS con días de falsa alarma, que requiere los paneles walk-forward.
- **Calentamiento de los termómetros** (`min_obs = 252`): en B no hay valor hasta ~abril de 2008. Si los detectores
  del menú T arrancan su entrenamiento ahí, su OOS empieza más tarde que el del menú N y la comparación deja de ser
  1:1. Opciones: (a) calcular los termómetros con la historia anterior a `ventana_inicio` de las series que la
  tienen (causal, no usa futuro); (b) bajar `min_obs`; (c) recortar el menú N al mismo OOS. Propuesta: (a), y
  verificar que el OOS de T coincide con el de N.
- Tabla de §6.3 de PendientesOscar.md: T8 de la pista A son 15 columnas porque incluye `WTI_yoy_z` (A no tiene T7).

## 8. Estado de la ejecución (30-10-2026)

Portátil secundario sin `data/processed` ni paneles walk-forward y sin tiempo para el benchmark. Hecho como **prueba
de humo**: paneles construidos en memoria (≈ 4 s por pista, sin escribir a disco ni verificar sha256), termómetros y
validación 1–3, y MCS preliminar. Falta en este portátil `data/raw/academico/SHILLER_SP500_CAPE.parquet`, así que
`CAPE_z` queda fuera de T8 en la prueba de humo. La versión oficial se hará sobre `data/processed` verificado.
