# ADR-009 — Resultados del pool de termómetros: ablaciones, controles y decisión sobre el menú T

- **Estado:** Borrador (pendiente de resultados de 2.5) · 2026-11-02. Las secciones marcadas **PENDIENTE DE
  RESULTADOS** se rellenan cuando termine la ejecución. La decisión (§6) no se toma antes.
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

**PENDIENTE DE RESULTADOS.** Pares con el signo de T − N reproducido: [k de 12]. Pares que se leen con cautela:
[ninguno / …].

### 4.3 Criterio (b)

**PENDIENTE DE RESULTADOS.**

| Pista | Mediana `REF-T` | p90 `AZAR` | Percentil de `REF-T` | `AZAR` completas | (b) |
|---|---|---|---|---|---|
| A | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE |
| B | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE |

### 4.4 Quitar un bloque

**PENDIENTE DE RESULTADOS.** Estorban en A: […]; ayudan en A: […]. Estorban en B: […]; ayudan en B: […]. Lugar de
T9 de A: […]. ¿T4, T5 y T7 de B ayudan?: […].

### 4.5 Media simple frente a PCA

**PENDIENTE DE RESULTADOS.** La media mejora a la PCA en [k de 6] detectores en A y en [k de 6] en B. En A, la
comparación solo afecta a 5 de los 7 termómetros: en T3 y T6 la media y la PC1 coinciden (§3).

## 5. Respuestas

**PENDIENTE DE RESULTADOS.** Una por pregunta del §2: por qué B sí y A no; si la agrupación económica aporta frente
al azar; qué bloque ayuda o estorba; si la PCA aporta sobre la media.

## 6. Decisión sobre el menú T

**PENDIENTE DE RESULTADOS.** Se elige una opción cuando estén los §4.2–4.5:

1. **Descartar el menú T.** El ranking v3 se queda con el menú N (y U, P y R donde apliquen). El menú T se reporta
   como resultado negativo pre-registrado: «con el pool completo, lo simple sigue ganando». Encaja si (b) no se
   cumple en ninguna pista o si el indicio de B no resiste frente a `VOL_P90` (§4.1).
2. **Menú T solo en B.** Se mantiene en B como menú del ranking v3 y se retira de A. Exige que (b) se cumpla en B y
   que la explicación de §5 sea estructural (bloques que A no tiene), no un ajuste a la muestra. Se declara que no
   cumple el §6 del ADR-008, que pedía las dos pistas.
3. **Menú T reducido** (sin los bloques que estorban). **Solo exploratorio:** los bloques se eligen a la vista de los
   resultados de §4.4, así que no puede entrar en el ranking principal ni usarse para revisar el veredicto del
   ADR-008. Si se ejecuta, se reporta aparte con esa etiqueta.

Decisión: **PENDIENTE DE RESULTADOS.**

## 7. Consecuencias

**PENDIENTE DE RESULTADOS.** Qué cambia en el ranking v3 (2.7), en los detectores supervisados D16 y D17 (que usan
el menú T, ADR-008 §3) y en la memoria del TFM.
