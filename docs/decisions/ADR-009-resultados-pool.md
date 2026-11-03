# ADR-009 — Resultados del pool de termómetros: ablaciones, controles y decisión sobre el menú T

- **Estado:** Aceptado · 2026-11-02. Resultados de 2.5 completos: 1.338 trabajos, 0 errores, ejecutados el
  02-11-2026 de 14:48 a 21:56. Decisión del §6: opción 1, descartar el menú T.
- **Origen:** plan de [`PendientesOscar.md`](../../PendientesOscar.md) §6.8, punto 2.5 (ablaciones y controles de
  §6.5, puntos 4–6; el 4, relevancia, ya está en la sección 10 del notebook 04).
- **Completa:** [ADR-008](ADR-008-preregistro-pool-termometros.md), el pre-registro. No cambia sus bloques, anclas,
  menús, vara de medir ni criterios, ni su veredicto (§8).
- **Numeración:** ADR-006 sigue reservada a la decisión final del detector de régimen.

---

## 1. Contexto

El ADR-008 pre-registró los termómetros por bloque (una PCA causal por bloque económico), el menú T y tres criterios
de éxito (§6). Exigen las dos pistas: (a) la mayoría de los 6 multivariantes mejora con T frente a N; (b) la mediana
de sus scores con bloques económicos supera al percentil 90 de 100 reparticiones aleatorias; (c) algún detector T
entra en el MCS de «T ∪ N».

Veredicto oficial (ADR-008 §8, 02-11-2026): **el pool no aporta tal como se pre-registró**. En A fallan (a) y (c):
mejora 1 de 6 y ningún detector T entra en el MCS. En B se cumplen las dos, pero es un indicio y no una mejora
general. Dos de las mejoras son mínimas y, con la pérdida del MCS, solo mejoran 3 de 6. (b) no puede cambiar el
veredicto.

## 2. Qué pregunta responde 2.5

Con el veredicto cerrado, 2.5 no busca rescatar el pool. Busca **entender** el resultado para decidir qué hacer con
el menú T en el ranking v3 (plan 2.7):

1. **¿Por qué B sí y A no?** ¿El indicio de B viene de los bloques que A no tiene (T4 inflación esperada, T5 vol
   implícita y T7 divisas y materias primas)? ¿O de que B tiene menos crisis y más pool?
2. **¿Aporta la agrupación económica frente al azar?** Es el criterio (b): bloques económicos frente a 100
   reparticiones aleatorias de las mismas columnas. Si el azar rinde igual, lo que aporta es comprimir, no la
   economía de los bloques.
3. **¿Qué bloque ayuda y cuál estorba?** Se quita un bloque cada vez. Es descriptivo. T9 de A merece atención
   especial: es el único termómetro que no pasó la validación 2, por un salto de signo.
4. **¿Aporta la PCA sobre la media simple** de cada bloque?
5. **Significación (§5, punto 6):** MCS dentro de cada menú, con los baselines.

## 3. Diseño

Variantes, semillas, reajuste anual (`step = 252`) y criterio de validez del atajo: errata del ADR-008, §5.1. Código:
`regimenes.benchmark.ablaciones` (ejecución) y `regimenes.benchmark.ablaciones_lectura` (lectura y tablas).
Resultados: `results/detectores/ablaciones/`. Lectura: notebook [`04_termometros`](../../notebooks/04_termometros.ipynb),
sección 13.

- **Validez del atajo.** Con reajuste anual, el signo de `REF-T − REF-N` debe ser el oficial (mensual) en cada
  detector y pista. Un piloto lo reprodujo en 12 de 12. Todas las ablaciones se comparan con `REF-T`, nunca con la
  T oficial.
- **Criterio (b)** (reglas de la errata §5.1). Mediana de los 6 `score_deteccion` con `REF-T` frente al percentil
  90 (interpolación lineal) de las medianas `AZAR`. Superarlo es estricto: un empate no cumple.
  - Una semilla con algún detector sin resultado sale entera del cálculo y no se repone.
  - (b) exige al menos 95 semillas completas y `REF-T` con los 6 detectores. Si no, no se puede evaluar y cuenta
    como no cumplido.
  - Como sensibilidad, sin peso en el veredicto, se rellenan los scores que faltan con el mínimo y con el máximo de
    cada detector y se recalcula el p90 (`p90_cota_baja`, `p90_cota_alta`).
- **Reparto al azar.** Cada semilla permuta las columnas del pool y las corta con los tamaños de los bloques
  económicos. La ancla de cada bloque aleatorio se elige en el primer reajuste r0 y su signo se re-estima en cada
  reajuste, con datos hasta ese reajuste. La regla exacta está en la errata §5.1.
  - **Reparto factible.** Si un termómetro queda sin datos en el OOS, el reparto se rechaza y se toma la siguiente
    permutación del mismo generador. En A se rechaza el 29 % de los intentos (41 de 141), y lo necesita el 30 % de
    las semillas, con un máximo de 4 intentos. La causa son bloques de 1 o 2 columnas que reciben series que
    empiezan tarde. En B no se rechaza ninguno. Equivale a muestrear de manera uniforme entre los repartos
    factibles y solo depende de la disponibilidad de datos, no de las crisis.
  - **Desventaja estructural de `AZAR`.** Las anclas y el reparto económico se eligieron con estadísticos de toda la
    muestra para evitar saltos de signo (ADR-008 §2.1); los bloques aleatorios no tienen esa ventaja. Si (b) se
    cumpliera por poco, podría deberse en parte a eso.
- **MEDIA.** En A coincide con `REF-T` en T3 (una columna) y en T6 (dos columnas, donde la PC1 es la media con
  signos), así que solo cambia 5 de los 7 termómetros. En B todos los bloques tienen más de dos columnas.
- **Quitar un bloque.** `SIN-T<k>` − `REF-T` por detector; el bloque «estorba» si la mediana de los 6 cambios es
  positiva y «ayuda» si es negativa.
- **MCS por menú.** Sobre los paneles **oficiales** (reajuste mensual) y con la vara de medir definitiva (ADR-008
  §4). Los baselines (`SIEMPRE_*`, `AZAR_*`, `VOL_P90`) son los mismos en todos los menús: los de
  `results/benchmark/vara/mcs_diario.csv`. Menús: N (los 12 de v2, es `mcs_diario.csv`), N6 (los 6 multivariantes
  con N) y T (los 6 con T).

## 4. Resultados

### 4.1 Significación: MCS dentro de cada menú (calculado el 02-11-2026)

`results/detectores/ablaciones/mcs_menu_{T,N6}_{A,B}.csv`; N es `results/benchmark/vara/mcs_diario.csv`. α = 0,10;
bloque 66 en A y 50 en B; 5.000 réplicas de razón.

| Pista · menú | Primero (pérdida) | En el MCS (p) | Baselines dentro |
|---|---|---|---|
| A · N (12) | D06 (0,550) | 11 detectores, todos salvo D09 | `VOL_P90` (0,39) |
| A · N6 | D10 (0,652) | D10, D03, D04, D02, D08 | `VOL_P90` (0,82) |
| A · T | **`VOL_P90`** (0,662) | D10_T, D09_T, D02_T, D08_T (0,45) | `VOL_P90`, **`AZAR_PERSISTENTE` (0,11)** |
| B · N (12) | D05 (0,610) | 8 detectores | ninguno (`VOL_P90` 0,004) |
| B · N6 | D10 (0,653) | D10, D04 (0,15) | ninguno (`VOL_P90` 0,012) |
| B · T | D02_T (0,707) | D02_T, D10_T (0,38), D04_T, D08_T (0,27) | **`VOL_P90` (0,27)** |

Lectura. El MCS es relativo al mejor de cada conjunto, así que un p-valor no se compara entre menús.

- **A.** Ningún detector T pierde menos que `VOL_P90`, la regla de un solo umbral sobre la volatilidad. Además, el
  MCS del menú T no descarta `AZAR_PERSISTENTE`, un baseline sin información. Con N y con N6 lo descarta. Refuerza
  el fallo de A del ADR-008 §8.
- **B.** El menú T deja fuera a los baselines sin información y lo encabeza D02_T. Pero no descarta `VOL_P90`, que
  sí queda fuera con N y con N6. El indicio de B no basta para separar a los detectores con T de la regla trivial
  de volatilidad.
- **Veredicto del ADR-008 §8:** no cambia. La lectura de A se refuerza y la de B se matiza: es un indicio frente a
  N, no frente a `VOL_P90`.

### 4.2 Validez del atajo anual

`results/detectores/ablaciones/validez_atajo.csv`. **12 de 12 pares** reproducen el signo de T − N oficial. En A,
T mejora solo en D02; en B, en 5 de 6, todos salvo D04. Ningún par se lee con cautela. La magnitud sí cambia con el
reajuste anual: B·D03 pasa de +0,04 a +0,21 y A·D08 de −0,008 a −0,038. Por eso las ablaciones se leen por signo
y orden, no por el tamaño exacto.

### 4.3 Criterio (b)

`results/detectores/ablaciones/criterio_b.csv` y `medianas_azar.csv`. Sin semillas incompletas, así que las cotas
de sensibilidad coinciden con el p90.

| Pista | Mediana `REF-T` | p90 `AZAR` | Percentil de `REF-T` | `AZAR` completas | (b) |
|---|---|---|---|---|---|
| A | 0,402 | 0,424 | 83 | 100 de 100 | **no cumple** |
| B | 0,354 | 0,341 | 93 | 100 de 100 | **cumple, por 0,012** |

- **Referencias:** sin comprimir (`REF-N`), la mediana es 0,504 en A, por encima de las 100 reparticiones, y 0,340
  en B, en su p90.
- **Mediana de las 100 `AZAR`:** 0,360 en A, con rango de 0,27 a 0,48, y 0,265 en B, con rango de 0,10 a 0,45.

En B el margen es pequeño y cabe dentro de la desventaja estructural de `AZAR` (§3). El agente de ejecución detectó
además una fuente concreta: en B, 214 de 1.000 bloques aleatorios cuya ancla viene de T9 (`GW_valuation_z`,
`CAPE_z`) quedan dominados por columnas de inflación y correlan en negativo con el termómetro económico del que
sale su ancla. No es un error de signo: el signo se re-estima en cada reajuste. Pero esos bloques miden otra cosa
que su ancla y rebajan la distribución `AZAR`.

### 4.4 Quitar un bloque

`results/detectores/ablaciones/sin_bloque.csv`. Las cifras son la mediana de los 6 cambios y cuántos detectores
mejoran al quitar el bloque.

- **En A estorban T6, T9, T2 y T8.**
  - Quitar T6 mejora los 6 detectores (+0,060).
  - T9, el del salto de signo de la validación 2, es el segundo que más estorba (+0,044, 4 de 6).
- **En A ayudan T1b, T1 y T3**, todos con menos de 0,04.
- **En B estorban T9, T8, T2 y T3**, todos con menos de 0,016.
- **En B ayudan T5, T6, T4, T1, T7 y T1b.**
  - El que más ayuda es T5, vol implícita: −0,075, 0 de 6 mejoran sin él.
  - Le siguen T6 (−0,038) y T4, inflación esperada (−0,034, 0 de 6).
  - T7, divisas y materias primas, ayuda poco (−0,017).
- **Dos de los tres bloques exclusivos de B** (T5 y T4) están entre los tres que más ayudan.
- **T6 cambia de papel según la pista:** estorba en A y ayuda en B.

### 4.5 Media simple frente a PCA

`results/detectores/ablaciones/media.csv`.

- **En A, la media mejora a la PCA en 4 de 6 detectores.** La mediana pasa de 0,402 a 0,468 (+0,066), sin llegar a
  la de `REF-N` (0,504). La comparación solo afecta a 5 de los 7 termómetros: en T3 y T6 la media y la PC1
  coinciden (§3).
- **En B empatan:** la media mejora en 3 de 6 y la mediana es la misma (0,354). Hay cambios grandes por detector,
  por ejemplo D08 −0,15.

## 5. Respuestas

1. **¿Por qué B sí y A no?** En parte es estructural. Los bloques que más ayudan en B son la vol implícita (T5) y la
   inflación esperada (T4), y ninguno de los dos existe en A. A carga además con dos bloques que estorban de forma
   clara, T6 y T9. El T9 es el del salto de signo. El tamaño de la muestra no se puede separar con este diseño: B
   tiene menos crisis.
2. **¿Aporta la agrupación económica frente al azar?** No en A: queda en el percentil 83 y cualquier partición
   comprimida pierde frente a no comprimir. En B aporta, por poco: percentil 93, margen de 0,012, y parte de ese
   margen puede venir de la desventaja de `AZAR` (§4.3). Es un indicio débil, coherente con el de ADR-008 §8.
3. **¿Qué bloque ayuda y cuál estorba?** Lo resume el §4.4. Es descriptivo y no hay contraste formal.
4. **¿Aporta la PCA sobre la media?** No. En A la media simple es mejor en 4 de 6 detectores y en B empata. Lo que
   comprime cada bloque no gana nada con la PC1.

Las cuatro respuestas apuntan en el mismo sentido que el MCS (§4.1). En A, el menú T pierde frente a N, frente a
`VOL_P90` y frente a repartir las columnas al azar. En B hay una señal pequeña, ligada a bloques que A no tiene, que
no se separa de `VOL_P90`.

## 6. Decisión sobre el menú T

Opciones definidas antes de ver los resultados:

1. **Descartar el menú T.** El ranking v3 se queda con el menú N (y U, P y R donde apliquen). El menú T se reporta
   como resultado negativo pre-registrado: «con el pool completo, lo simple sigue ganando». Encaja si (b) no se
   cumple en ninguna pista o si el indicio de B no resiste frente a `VOL_P90` (§4.1).
2. **Menú T solo en B.** Se mantiene en B como menú del ranking v3 y se retira de A. Exige que (b) se cumpla en B y
   que la explicación de §5 sea estructural (bloques que A no tiene), no un ajuste a la muestra. Se declara que no
   cumple el §6 del ADR-008, que pedía las dos pistas.
3. **Menú T reducido** (sin los bloques que estorban). **Solo exploratorio:** los bloques se eligen a la vista de los
   resultados de §4.4, así que no puede entrar en el ranking principal ni usarse para revisar el veredicto del
   ADR-008. Si se ejecuta, se reporta aparte con esa etiqueta.

Cómo encajan los resultados con cada opción, según las reglas escritas arriba antes de verlos:

- **Opción 1.** Su condición se cumple, porque el indicio de B no resiste frente a `VOL_P90` (§4.1).
- **Opción 2.** Sus dos condiciones se cumplen a medias:
  - (b) se cumple en B, pero por 0,012;
  - la explicación es en parte estructural (T5 y T4).
- **Opción 3.** Solo puede ser un anexo exploratorio.

**Decisión (02-11-2026, usuario): opción 1, descartar el menú T.**

- **El núcleo es N, y U para los univariantes.** El ranking principal v3 pasa del menú T al N.
- **Es una desviación declarada del ADR-008 §4,** justificada por el veredicto del §8 y por los §4–5 de este ADR.
- **T se publica como resultado negativo pre-registrado,** con su propio MCS, como ranking secundario.
- **No se ejecuta la opción 3.**
- **La pregunta «¿sirve el pool?» sigue abierta solo para el pool sin comprimir,** con los menús P y R (§7).

## 7. Consecuencias

- **Ranking v3 (2.7).** El principal es el del menú N, con los univariantes en U, y se declara como desviación del
  ADR-008 §4. El MCS dentro de cada menú se mantiene: N, U, P, R y, como secundario, T.
- **Lo que queda de la Fase 2 responde una sola pregunta: ¿aporta algo el pool sin comprimir frente a N?**
  - **D18, sparse jump model sobre P:** es la prueba principal, porque elige él mismo las columnas del pool.
  - **D10 sobre P y R.**
  - **D14, absorption ratio sobre R** (solo en B): es una idea distinta, no depende de T, y sigue igual.
  - **D15, jump de Shu sobre U:** no depende del pool y sigue igual.
- **D16 y D17.** Los supervisados se evalúan sobre todo en P y, como comparación barata, en N (no en T, que está
  descartado; precisado en el ADR-010 §3 el 03-11-2026). No tienen peso en el ranking principal. Responden si un
  modelo con etiquetas saca partido del pool.
- **`get_regimen` provisional, sintéticos del notebook 21 y XAI.** Se apoyan en N y U, como hasta ahora. Ya no hace
  falta ampliar los generadores sintéticos al menú T (PendientesOscar §6.7): solo a P, y solo si P aporta.
- **Memoria del TFM.** El resultado negativo es pre-registrado y se explica bien: comprimir el pool en
  termómetros por bloque no mejora a los mismos detectores con los datos de siempre. Tampoco supera repartir las
  columnas al azar en A, y la PCA no aporta sobre la media simple.
