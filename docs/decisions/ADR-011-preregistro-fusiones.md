# ADR-011 — Pre-registro de las fusiones: ¿mejora una combinación de detectores al mejor detector solo?

- **Estado:** Aceptado y **congelado** · 03-11-2026, antes de escribir el código de las fusiones y antes de
  calcular ninguna. Desde ese momento, cualquier cambio se declara aquí como errata fechada.
- **Origen:** [`PendientesOscar.md`](../tfm/PendientesOscar.md) §C1 y §6.8, punto 2.6; decisiones del usuario del
  03-11-2026 (numeración, candidatos, familias, vara, control de multiplicidad y fusión online) y, tras la revisión
  del borrador, sobre la binarización de (a), el MCS del criterio (ii), η, el arranque de los pesos y las mitades
  (§8).
- **Parte de:** [ADR-003](ADR-003-causalidad-calendario-estado-ranking.md) (`score_deteccion`),
  [ADR-008](ADR-008-preregistro-pool-termometros.md) §4 (MCS) y [ADR-010](ADR-010-preregistro-pool-crudo.md), cuyos
  paneles OOS y etiquetas causales se reutilizan.
- **No cambia:** los 12 detectores de v2, las variantes de la Fase 2 (ADR-010), sus paneles y métricas, el ranking
  v3, el notebook 21, las ventanas, las crisis ni las trampas. **No se reentrena nada:** las fusiones combinan los
  paneles OOS que ya existen.
- **Numeración:** ADR-006 sigue reservada a la decisión final del detector de régimen. Los resultados de esta fase
  se escriben en el §9 de este ADR.

---

## 1. Pregunta

Los ADR-008 a ADR-010 comparan detectores **sueltos**. Las fusiones que hay ([notebook 15](../../notebooks/fase1/15_fusiones_fijas.ipynb),
D7 alerta + D8 confirma, y [notebook 16](../../notebooks/fase1/15_fusiones_fijas.ipynb), D6 confirmador fijo con cribado de
alertas) se eligieron después de ver el benchmark y se midieron con una utilidad operativa propia, no con la vara del
ranking. Queda la pregunta: **¿hay alguna combinación de detectores no supervisados que, medida con la misma vara que
los individuales y con la selección controlada, mejore al mejor detector solo?** Se prueba de dos formas:

- **Fusiones fijas** (notebook 15): reglas sin parámetros aprendidos (alerta→confirmación, votación, OR y AND),
  barridas sobre todos los candidatos.
- **Fusión online** (notebook 17, C1, inspirada en la agregación de modelos de crisis de Fouliard et al., 2020): media
  ponderada de probabilidades con pesos que se actualizan cada mes según el acierto pasado, medido solo con crisis ya
  terminadas.

## 2. Candidatos

Todos los detectores **no supervisados** con panel OOS. D16 y D17 (supervisados) quedan fuera: ya usan etiquetas, y
mezclarlos haría que la fusión heredara esa ventaja.

| Pista | Candidatos | n |
|---|---|---|
| A | D01–D12 (v2) + D15-U + D10-P + D18-P | 15 |
| B | D01–D12 (v2) + D15-U + D10-P + D18-P + D10-R + D14-R | 17 |

- **Paneles:** `results/benchmark/panels/{A,B}_Dxx.parquet` (v2) y `results/detectores/pool/{A,B}_{id}.parquet`
  (Fase 2). Columnas `state`, `p_crisis` y `fold`.
- **Índice:** todos los paneles de una pista tienen **el mismo índice OOS**, sin NaN (comprobado, §8). No hace falta
  alinear ni intersecar: no se pierde ningún día. A: 14.133 sesiones, de 1970-05-12 a 2026-05-29. B: 4.059 sesiones,
  de 2010-04-12 a 2026-05-29.
- **NaN:** si un panel trae NaN en `state` o `p_crisis`, es un error (como en `mcs.cargar_paneles` y
  `fusion.load_panel`); no se rellena.

### 2.1 Señal binaria y probabilidad de cada candidato

- **Señal binaria de crisis** (la que puntúan `score_deteccion` y el MCS): `state == n_states − 1`, con `n_states`
  del CSV de métricas de cada detector (`mcs.n_states_metricas`; `results/benchmark/metrics/` para v2 y
  `results/detectores/pool/{P}_{id}.csv` para la Fase 2). Es lo mismo que hacen `mcs.senales_crisis` y
  `pool_lectura.senales`. D14-R no se canonicaliza (ADR-010 §8.1), pero su estado 1 = ΔAR ≥ 1 = `n_states − 1`, así
  que la regla común vale. D03 tiene 3 estados y D08 4: su crisis es el estado 2 y el 3.
- **Probabilidad de crisis:** la columna `p_crisis` (= `proba[:, crisis_state]`). Existe para todos. En los
  deterministas (D01, D02, D09, D15-U, D18-P y, en B, D14-R) vale exactamente 0 o 1 y coincide con la señal binaria:
  la conversión estado→probabilidad ya está hecha en el panel.
- **Ojo:** en cinco detectores `p_crisis ≥ 0,5` **no** es la señal binaria, porque su estado lleva histéresis o
  suavizado (§8): D06, D07, D10, D10-P y D10-R (este solo en B) discrepan entre el 7,7 % y el 26,1 % de los días. En
  A, D03 y D08 discrepan además en 14 y 2 días. Por eso las fusiones fijas usan siempre la señal binaria, y la
  online usa `p_crisis` con la binaria como sensibilidad (§4).

## 3. Fusiones fijas (notebook 15)

Todas se construyen con la **señal binaria** de cada candidato (§2.1), sobre todo el OOS de una vez (son causales,
§7.1), y se recortan después para puntuar cada tramo, como `fusion.period_scorecard`. No tienen parámetros que se
ajusten a crisis.

### 3.1 Familia (a) · Parejas alerta→confirmación

- **Pares:** todos los **ordenados** de candidatos distintos (X alerta, Y confirma): 15 × 14 = 210 en A y 17 × 16 =
  272 en B.
- **Máquina:** `fusion.fuse_early_warning` sin cambios, con la configuración de los notebooks 15 y 16:
  `FusionConfig(alert_source="state", alert_crisis_state=n_states_X − 1, alert_persistence=1,
  confirmation_threshold=0,50, confirmation_persistence=3, warning_horizon=63)`.
- **Confirmador binario:** la máquina confirma con `p_crisis ≥ umbral`, no con el estado. Para que el confirmador sea
  la misma señal que se puntúa, se le pasa un panel con `p_crisis` sustituida por su señal binaria (0,0/1,0); con el
  umbral 0,50 la confirmación es «señal binaria de Y en 3 sesiones seguidas». No se toca `maquina.py`.
- **Binarización principal: accionable = vigilancia ∪ confirmado** (`decision_code ≥ 1`, columna `actionable`).
  Decidido por el usuario el 03-11-2026, tras la revisión.
  - **Por qué no «confirmado».** En la máquina, `decision_code = 2` es exactamente `confirmation_active`, que solo
    mira a Y: la alerta no interviene. Las 210 (o 272) parejas se reducirían a 15 (o 17) señales distintas, una por
    confirmador, y cada una sería el propio confirmador con persistencia 3. Eso no es una fusión: sería medir los
    individuales con un filtro de persistencia y llamarlo «combinación».
  - «Accionable» es la decisión que defendían los notebooks 15 y 16, y la única de las tres en la que X y Y
    intervienen los dos. Es crisis:
    - el día de un flanco de subida de X con Y aún inactivo y las 63 sesiones siguientes (`warning_window`, edad 0 a
      63). La ventana no se corta cuando Y confirma ni cuando X se apaga, y un flanco nuevo la reinicia;
    - y los días en que Y confirma.
  - Es una propiedad de la máquina, no un resultado: cuanto más marca Y, menos flancos de X caen con Y inactivo, y
    más se parece la pareja a Y con persistencia 3. Las familias (b) y (c) combinan día a día y no tienen ese
    sesgo.
- **Sensibilidad, sin peso:** «confirmación tras aviso» (`maquina._gated_confirmation`). Solo cuenta como crisis una
  confirmación de Y precedida por un aviso de X emparejado dentro del horizonte. Es la lectura secuencial estricta, y
  el notebook 16 vio que casi nunca ocurre (en A, el 1,3 % de las confirmaciones de D6 tienen aviso de D2).
- **Referencia, sin peso:** «confirmado» puro, es decir, cada candidato con persistencia 3.

### 3.2 Familia (b) · Votación k-de-n

Crisis el día t si al menos k de los n detectores marcan crisis ese día. No se añade persistencia.

| Regla | n detectores | k |
|---|---|---|
| 2 de 3 | los 3 primeros del ranking de la **primera mitad** (§5.2) | 2 |
| 3 de 5 | los 5 primeros del ranking de la primera mitad | 3 |
| Mayoría | **todos** los candidatos (sin selección) | 8 de 15 en A, 9 de 17 en B |

**Por qué el ranking de la primera mitad.** El top-n del `pool_lectura.ranking_v3(menus=("N", "U", "P", "R"))` de todo
el OOS es una selección hecha con las crisis que luego se usan para puntuar. Se elige con la primera mitad (§5.2) y
se evalúa en la segunda. El ranking v3 de todo el OOS se reporta solo como lectura.

- **Top-n:** los n primeros individuales en el orden de H1 del §5.2, sea cual sea su nivel.
- **Mayoría y media:** con n impar, «al menos ⌈n/2⌉ de n» es lo mismo que «media de las señales binarias ≥ 0,50».
  Por eso `VOTO_mayoria` coincide día a día con la sensibilidad de la online con η = 0 y entrada binaria (§6.3).

### 3.3 Familia (c) · OR y AND de parejas

Para cada pareja **no ordenada** {X, Y} de candidatos distintos: OR (crisis si marca alguno) y AND (crisis si marcan
los dos), día a día. Son C(15, 2) × 2 = 210 señales en A y C(17, 2) × 2 = 272 en B.

### 3.4 Recuento

| Pista | (a) parejas | (b) votación | (c) OR y AND | Total principal | Sensibilidad (a) tras aviso |
|---|---|---|---|---|---|
| A | 210 | 3 | 210 | **423** | 210 |
| B | 272 | 3 | 272 | **547** | 272 |

**Identificadores:** `PAR_<X>><Y>`, `OR_<X>+<Y>`, `AND_<X>+<Y>` (X < Y en orden lexicográfico), `VOTO_2de3`,
`VOTO_3de5` y `VOTO_mayoria`. Si dos fusiones dan señales idénticas, se mantienen las dos; el desempate del §5.2 elige
una.

## 4. Fusión online (notebook 17)

### 4.1 Combinación

- **Probabilidad fusionada:** `p̄_t = Σ_i w_i(m) · p_i,t`, donde la suma recorre todos los candidatos de la pista
  (§2; sin selección), `p_i,t` es la `p_crisis` del panel y `w_i(m)` son los pesos vigentes en el mes m de t.
- **Señal:** crisis si `p̄_t ≥ 0,50`, sin persistencia añadida. El umbral es el de los supervisados y el de la máquina.
  Se fija a priori y no se barre.
- **NaN:** no hay ninguno (§8). Si apareciera un `p_i,t` en NaN, ese día el detector sale de la suma y los pesos se
  renormalizan entre los demás.

### 4.2 Etiquetas causales y calendario

- **Frecuencia:** los pesos se recalculan al cierre de la **última sesión OOS de cada mes natural** (τ) y se aplican
  desde la **primera sesión OOS del mes siguiente**. Nunca se usa un dato del mismo día: hay al menos una sesión de
  retraso entre la información y su uso. En el primer mes del OOS los pesos son iguales. Hay 673 meses OOS en A y
  194 en B.
- **Etiquetas vistas en τ:** la regla causal de D16 y D17 (ADR-010 §3.2,
  `detectores.pool.supervisados.etiquetas_causales` con `excluir_sin_terminar=False`).
  - Una crisis existe si su suelo es ≤ τ.
  - Los días de una crisis aún sin terminar cuentan como calma.
  - Las trampas cuentan como calma.

  Las ventanas son las del juez (`ranking.track_crisis_windows`), que coinciden día a día con
  `data/processed/labels.parquet` (`inicio`, `trough_date`; §8). No se añade retraso de publicación: es el mismo
  criterio que D16 y D17. Hay una sensibilidad con 63 sesiones de retraso de reconocimiento (§6.3).
- **Implementación:** `etiquetas_causales(idx_OOS[idx_OOS ≤ τ], crisis, trampas)`. La función toma el último día del
  índice como fecha de corte, así que basta con pasarle el OOS recortado en τ. Las trampas de 2013 no se solapan con
  ninguna ventana de crisis, así que esta regla y la de `mcs.perdidas_diarias` (todo lo que no es ventana es calma)
  dan las mismas etiquetas.
- **Pesos iguales hasta la primera crisis terminada que tenga días OOS.** En A es bear_1969_70, con suelo el
  1970-05-26 y solo 11 días OOS. Los primeros pesos aprendidos se calculan el 1970-05-29 y rigen desde junio de 1970.
  En B es flash_crash_euro1_2010, con suelo el 2010-07-02: los pesos se calculan el 2010-07-30 y rigen desde agosto
  de 2010. Las crisis sin días OOS (credit_crunch_1966 en A y gfc_2007_09 en B) nunca cuentan.
  - En A, los primeros pesos se aprenden con solo 14 días (11 de crisis y 3 de calma). Se mantiene la regla tal cual
    (decisión del usuario del 03-11-2026).
  - **Sensibilidad, sin peso:** pesos iguales mientras las crisis terminadas no sumen al menos **21 días OOS**. En
    A, los primeros pesos aprendidos pasan a calcularse el 1974-10-31 (suelo de oil_stagflation_1973) y rigen desde
    noviembre de 1974. En B no cambia nada, porque flash_crash_euro1_2010 ya tiene 50 días OOS.

### 4.3 Pérdida y pesos

- **Pérdida de cada detector en τ:** la misma 1 − J de Youden con crisis equiponderadas de la vara (ADR-008 §4,
  `mcs.perdidas_diarias`), calculada sobre su **señal binaria** en los días OOS ≤ τ y con las etiquetas vistas en τ.
  Es decir, `L_i(τ) = (1/K_τ) · Σ_c (1 − cobertura_i,c) + FPR_i`, donde:
  - K_τ son las crisis terminadas con algún día OOS ≤ τ;
  - FPR_i es la fracción de días de calma (según τ) que marca el detector.

  Se recalcula entera en cada τ, no se acumula: cuando una crisis termina, sus días pasan de calma a crisis y el
  detector que la marcaba deja de estar penalizado. Una crisis que termina después de τ no entra en K_τ y sus días
  ≤ τ cuentan como calma, así que penaliza a quien la marca mientras dura: es lo que vería alguien en tiempo real.
  Todos los τ con pesos aprendidos tienen días de calma (el primero de A tiene 3), así que FPR_i siempre está
  definida.
- **Pesos:** `w_i(τ) = exp(−η · L_i(τ)) / Σ_j exp(−η · L_j(τ))`. Son los pesos exponenciales de *Hedge*, con la
  pérdida media en vez de la acumulada: η es una temperatura que no crece con el tiempo.
- **Por qué 1 − J y no log-loss.** Es la pérdida con la que se juzga al resultado (el MCS), así que los pesos premian
  exactamente lo que se mide. Trata igual a los deterministas, que con log-loss tendrían pérdida infinita en cada
  fallo (p ∈ {0, 1}) y obligarían a recortar las probabilidades a mano. Y con crisis equiponderadas no deja que las
  crisis largas (la de 1973, la punto com) dominen.

### 4.4 Tasa de aprendizaje

- **Valor principal: η = 10.** Lo justifica la escala de la pérdida, no el resultado de ninguna fusión. Las
  pérdidas de los individuales en todo el OOS van de 0,55 a 0,98 en A y de 0,61 a 1,00 en B (ADR-010 §9.2,
  `mcs_NUPR_{A,B}.csv`). Con η = 10:
  - una diferencia de 0,1 en la pérdida multiplica el peso por e ≈ 2,7;
  - el mejor pesa unas 70 veces más que el peor en A y unas 50 en B.

  Concentra los pesos sin llegar a quedarse solo con el líder.
- **Uso de información de toda la muestra (se declara).** η = 10 se fijó mirando el rango de pérdidas de los
  individuales en **todo** el OOS, que incluye las crisis con las que luego se puntúa. Es un uso suave: fija solo
  la escala de una temperatura, no mira ninguna fusión y no elige entre valores por su resultado. Como control, se
  reporta sin peso qué η de la rejilla gana en H1 (orden del §5.2; si empatan, el η menor) y cómo rinde ese η en H2
  frente a η = 10 (§6.3).
- **Sensibilidad, sin peso:** η ∈ {0, 1, 2, 5, 10, 20, 50, ∞}.
  - η = 0 es la media simple (comparador M, §4.5).
  - η = ∞ es *follow the leader*: todo el peso al detector con la pérdida más baja en τ, y si empatan, pesos iguales
    entre los empatados.

### 4.5 Comparadores (misma vara, mismos días)

- **O:** la fusión online con η = 10 (principal).
- **M:** la media simple (η = 0, pesos iguales siempre).
- **V:** la votación por mayoría (`VOTO_mayoria`, §3.2), que tampoco elige detectores.
- **F\*:** la mejor fusión fija, elegida en la primera mitad (§5.2). Si no hay F\* (§5.2), los comparadores y el
  MCS-O se quedan sin ella.
- **I\*:** el mejor individual. Para el criterio (i), el puesto 1 del ranking v3 ampliado de todo el OOS (D10 en A y
  D05 en B). Para el criterio (iii), el mejor de la primera mitad (§5.2).

## 5. Vara de medir y control de la multiplicidad

### 5.1 Vara (la de los individuales)

- **`score_deteccion`** (ADR-003): F1 entre la precisión diaria y el recall por evento, con `min_run = 3`
  (`ranking.DETECTION_DEFAULTS`). Se calcula con `metricas.detection_summary(senal, ventanas)` y
  `ranking.detection_score(det_precision, det_event_recall)` sobre la señal binaria. Para los 32 individuales
  reproduce al decimal el `score_deteccion` oficial (§8).
- **Nivel** del ranking (`ranking.rank_detection`): degenerado, parpadeo y precisión que no supera al azar, con los
  umbrales de `DETECTION_DEFAULTS`. Se calcula sobre la **señal binaria**, tanto para las fusiones como para los
  individuales (`metricas.switching_rate` y `metricas.mean_regime_duration` sobre la serie 0/1). Para los
  individuales da el mismo nivel que el ranking oficial: B_D09 «precisión que no supera al azar» y B_D12 «parpadeo»
  (§8). `label_stability` no se puede calcular para una fusión y se quita del desempate.
  - **D03 y D08** tienen 3 y 4 estados. Su `switching_rate` oficial cuenta los cambios entre todos los estados; aquí
    se cuenta sobre la binaria y sale menor (en A, D03 0,080 frente a 0,096 y D08 0,009 frente a 0,016). Se usa la
    binaria para todos, individuales y fusiones, para que el desempate mida lo mismo. Con todo el OOS, el orden de
    los 32 no cambia (§8).
- **MCS al 10 %:** `mcs.mcs_senales(senales, ventanas, block_len=…, **ablaciones_lectura.MCS_KW)`, con λ = 1,
  α = 0,10, `T_max`, bootstrap estacionario, 5.000 réplicas de razón (`replicas_perdida`) y semilla 0. Los baselines
  son los 5 de `ablaciones_lectura.senales_baselines_vara(pista)`, los mismos que en `mcs_NUPR`. Es la construcción de
  `benchmark.pool.mcs_conjunto` con otras columnas.
  - **Bloque fijo:** 66 en A y 50 en B, el de todo el OOS (`mcs.longitud_bloque_crisis`). En las mitades no se
    recalcula, para no cambiar la vara con el corte (recalculado sería 72/58 en A y 79/35 en B).
  - **MCS-F** (criterio de las fijas): los 15 o 17 individuales, los 5 baselines y F\*.
  - **MCS-O** (criterio de la online): los mismos más O, M, V y F\*.
  - **Señales idénticas.** Si dos señales de un MCS son idénticas día a día, entra una sola, para no dejar en el MCS
    una diferencia de pérdidas con varianza 0. Se queda la primera en este orden: individuales, baselines, F\*, V, M
    y O. La que sale hereda el resultado (dentro o fuera) de la que queda.
  - **Lectura del MCS (criterio ii).** Se lee el de **todo** el OOS, como con los individuales en ADR-008 a
    ADR-010 (decisión del usuario del 03-11-2026). Así (ii) se mide con la misma vara que el MCS publicado, y el
    control de la selección en H1 lo hace (iii). El MCS en H2 se reporta sin peso (§6.3): tiene 12 crisis en A y solo
    5 en B.
- **Utilidad operativa** de los notebooks 15 y 16 (`fusion.signal_scores` y `operational_utility`): se mantiene como
  lectura secundaria, sin peso.

### 5.2 Selección en la primera mitad, evaluación en la segunda

- **Mitades:** se corta en ⌊n/2⌋ sesiones OOS, igual que el notebook 16. Ninguna ventana de crisis cruza el corte
  (§8).

| Pista | H1 (selección) | Crisis en H1 | H2 (evaluación) | Crisis en H2 |
|---|---|---|---|---|
| A | 1970-05-12 a 1998-04-24 (7.066) | 5 | 1998-04-27 a 2026-05-29 (7.067) | 12 |
| B | 2010-04-12 a 2018-05-01 (2.029) | 4 | 2018-05-02 a 2026-05-29 (2.030) | 5 |

  Las trampas de 2013 caen en H2 en A y en H1 en B. Solo cuentan como calma.
- **H1 de B tiene solo 4 crisis** (flash_crash_euro1_2010, us_downgrade_euro2_2011, china_oil_2015_16 y
  volmageddon_2018q1, esta de 10 días). El recall por evento de H1 se mueve en saltos de 0,25, así que la selección
  en B y el control por mitades en B son ruidosos. Se declara y no se cambia el corte. En A, H1 tiene 5 crisis, una
  de ellas bear_1969_70 con 11 días.
- **Puntuación en un tramo:** la señal se construye sobre todo el OOS y se recorta. `detection_summary` usa las
  ventanas con días en el tramo y las demás salen como NaN, igual que credit_crunch_1966 (A) y gfc_2007_09 (B) en el
  OOS completo. El nivel, la tasa base y el `switching_rate` también se calculan sobre el tramo.
- **Orden de un tramo** (para el top-n de la votación, F\* e I\*):
  1. nivel en el tramo;
  2. `score_deteccion` en el tramo, redondeado a 3 decimales y de mayor a menor;
  3. `switching_rate` en el tramo, de menor a mayor;
  4. id.

  Es el orden de `rank_detection` sin `label_stability`. Con todo el OOS reproduce el `puesto_deteccion` oficial de
  los 32 individuales (§8). Los ids de las fusiones se comparan como cadenas.
- **F\*:** la primera de las 423 (A) o 547 (B) fusiones principales en el orden de H1, entre las de nivel 0 en H1.
  Si ninguna tiene nivel 0, no hay F\* y las fijas no aportan. Se elige entre las tres familias juntas, sin reservar
  un puesto por familia.
- **I\*_H1:** el primero de los individuales en el orden de H1.

## 6. Criterio de éxito

### 6.1 Fusiones fijas

**La fusión fija aporta** si, en **cada una de las dos pistas**, F\* cumple a la vez:

- **(i)** tiene nivel 0 en todo el OOS y un `score_deteccion` en todo el OOS **estrictamente mayor** que el del mejor
  individual (el puesto 1 del ranking v3 ampliado: 0,614 de D10 en A y 0,647 de D05 en B);
- **(ii)** entra en el MCS-F al 10 % de todo el OOS;
- **(iii)** aguanta el corte: en H2 tiene nivel 0 y un `score_deteccion` estrictamente mayor que el de I\*_H1 en
  H2.

**Comparaciones de score:** sin redondear y sin margen, como el criterio (a) del ADR-010 §5. Un empate exacto no
cumple. El redondeo a 3 decimales solo se usa para ordenar (§5.2).

**Si se cumple en una sola pista,** es un **indicio**, no un aporte. **Si no se cumple,** se reporta como resultado:
«combinar no mejora al mejor detector solo».

### 6.2 Fusión online

**La fusión online aporta** si, en **cada una de las dos pistas**, O cumple (i), (ii) en el MCS-O y (iii) frente a
I\*_H1. η está fijada a priori, así que O no tiene selección en H1; (iii) es solo su comparación en H2. Una sola pista
es un indicio.

**Veredicto secundario, «los pesos aprenden»:** si en las dos pistas el score de O es estrictamente mayor que el de M,
en todo el OOS y en H2. Se reporta aparte y no cambia el veredicto principal. Las comparaciones son como en el
§6.1.

### 6.3 Se reporta además, sin peso en el veredicto

- **Selección:**
  - el ganador de cada familia en H1 y su puesto en H2;
  - el optimismo de seleccionar en todo el OOS: el score de la mejor fusión de todo el OOS menos el de F\*;
  - el top-n del ranking v3 de todo el OOS y en qué difiere del de H1.
- **Variantes de las fijas:**
  - (a) con «confirmación tras aviso»;
  - «confirmado» puro;
  - la anticipación de los avisos de F\*, si F\* es de la familia (a) (`warning_ground_truth_table` frente a
    `warning_phase_base_rates`).
- **MCS-F y MCS-O** en H2.
- **Utilidad operativa** de F\*, O, M, V e I\*.
- **Online:**
  - los pesos a lo largo del tiempo, con su entropía efectiva;
  - las sensibilidades:
    - la rejilla de η, y qué η gana en H1 y cómo rinde en H2 frente a η = 10 (§4.4);
    - la entrada binaria en vez de `p_crisis` (con η = 0 es `VOTO_mayoria`, §3.2);
    - 63 sesiones de retraso de reconocimiento: en τ, una crisis cuenta si su suelo es ≤ la sesión OOS que está 63
      posiciones antes de τ. La pérdida sigue usando todos los días ≤ τ, y los de crisis aún no reconocidas cuentan
      como calma. En A los primeros pesos aprendidos se calculan el 1970-08-31 y en B el 2010-10-29;
    - pesos iguales hasta 21 días OOS de crisis terminadas (§4.2);
    - la variante `excluir_sin_terminar=True`, que saca de la pérdida los días de crisis empezadas y sin suelo.
      Usa una fecha futura (la de inicio), igual que en el ADR-010.

## 7. Implementación, coste y notebooks

### 7.1 Restricciones

- **Código nuevo** en `src/regimenes/fusion/`, en módulos aparte (p. ej. `fijas.py` y `online.py`). **Sin tocar:**
  - `maquina.py`, salvo que haga falta reexportar algo;
  - `registry.py`, `base.py`, `detectores/__init__.py`, `detectores/f*/`, `detectores/pool/*.py`, `walk_forward.py`
    y `metricas.py`;
  - `benchmark/pool.py`, `configs/benchmark_spec.yaml`, `configs/bloques.yaml` y `data/processed`;
  - los módulos de los sintéticos (`anclas.py`, `muestras.py` y `envoltorio.py`) y `features/rentabilidades.py`.

  `fusion/` no entra en la huella de la caché v2 (`benchmark.cache._cache_code_paths`) ni en la del notebook 21
  (`sinteticos.aumento.config.huella_generador`).
- **Resultados:** `results/fusion/fijas/` y `results/fusion/online/`, con rutas nuevas en `rutas.py`. Se conservan
  `results/fusion/d07_d08/` y `results/fusion/d02_d06/`.
- **Tests:**
  - prefijo: para cada familia y para la online, calcular hasta t da lo mismo que calcular con todo y recortar en t
    (`fusion.assert_prefix_causal`, ampliado a la columna de «confirmación tras aviso»);
  - fuga de etiquetas: una crisis sin terminar en τ no cambia los pesos de τ;
  - pesos iguales hasta la primera crisis terminada;
  - η = 0 da M;
  - el confirmador binario reproduce la señal binaria con persistencia 3;
  - la votación por mayoría con k = n es un AND, y con k = 1 un OR;
  - la online con η = 0 y entrada binaria coincide con `VOTO_mayoria`;
  - el orden del §5.2 con todo el OOS reproduce el `puesto_deteccion` oficial de los individuales.

### 7.2 Coste

Es barato y no hace falta piloto: no se ajusta ningún modelo.

- Unas 1.450 señales fijas entre las dos pistas, contando la sensibilidad (a).
- `fuse_early_warning` recorre el OOS una vez por pareja: unos 13 ms en A (medido con D7 → D6).
- La online son 8 valores de η más 4 sensibilidades por pista, sobre 673 y 194 actualizaciones vectorizables.
- Lo más caro son los MCS: 2 por pista en todo el OOS y 2 en H2, con 21 a 26 modelos y 5.000 réplicas. Son minutos
  cada uno.

Se ejecuta con **un solo proceso**, en paralelo al notebook 21 sin competir con él.

### 7.3 Numeración de notebooks

- **El 15 y el 16 se unifican** en `notebooks/fase1/15_fusiones_fijas.ipynb` y el 16 se borra con `git mv` / `git rm`.
  - El 15 nuevo empieza por los antecedentes (D7 + D8 y D2 + D6, leídos de sus `results/fusion/` actuales, sin peso).
  - Después vienen las tres familias de este ADR.
- **El 17 sigue siendo la fusión online** (`17_fusion_online.ipynb`, hoy un esqueleto). Su objetivo cambia: compara
  con F\*, M, V e I\*, en vez de con D2 + D6 y el mejor del 14.
- **No se renumera 18–23** hasta que termine el notebook 21. Mientras tanto queda el hueco del 16. La renumeración irá
  en un commit aparte con `git mv` y el test de enlaces de la documentación.

## 8. Verificaciones antes de congelar

- [x] **Paneles** (03-11-2026). Los 15 de A y los 17 de B tienen columnas `state`, `p_crisis` y `fold`, el mismo
  índice OOS dentro de cada pista (A 14.133 y B 4.059 sesiones) y 0 NaN. No hay pérdida por alineación.
- [x] **Estado de crisis = `n_states − 1`** según los CSV de métricas: D03 tiene 3 estados, D08 4 y el resto 2.
  D14-R, sin canonicalizar, marca el 18,1 % del OOS de B como crisis (estado 1).
- [x] **`p_crisis` frente a la señal binaria.**
  - En los deterministas, `p_crisis ∈ {0, 1}` y coincide día a día con la binaria: D01, D02, D09, D15-U y D18-P en
    las dos pistas, y D14-R en B.
  - Discrepan, con `p ≥ 0,5` frente a la binaria:
    - en A: D06 (1.763 días, 12,5 %), D07 (3.688 días, 26,1 %), D10 (1.453 días, 10,3 %), D10-P (1.082 días,
      7,7 %), D03 (14 días) y D08 (2 días);
    - en B: D06 (332 días, 8,2 %), D07 (883 días, 21,8 %), D10 (783 días, 19,3 %), D10-P (373 días, 9,2 %) y D10-R
      (501 días, 12,3 %).
- [x] **Máquina** (`maquina.fuse_early_warning`). `decision_code = 2` es `confirmation_active`, independiente de la
  alerta. La confirmación se lee de `p_crisis`, no de `state`. Por eso la binarización principal es «accionable» y el
  confirmador se binariza (§3.1).
  - Con el confirmador binarizado (D7 → D6 en A), `confirmation_active` es idéntica a «binaria de D6 en 3 sesiones
    seguidas» y `assert_prefix_causal` pasa.
  - `actionable` es causal: `warning_window` solo mira flancos pasados de X y el estado actual de Y, y
    `_gated_confirmation` empareja cada confirmación con avisos anteriores a ella.
- [x] **`score_deteccion` reproducido.** `detection_summary` + `detection_score` sobre la señal binaria reproducen el
  score del ranking v3 ampliado de los 32 individuales (diferencia máxima 1 × 10⁻¹⁶). El nivel calculado sobre la
  binaria coincide con el oficial.
- [x] **Orden.** Nivel → score a 3 decimales → `switching_rate` de la binaria → id reproduce el `puesto_deteccion`
  oficial de los 15 de A y los 17 de B. El puesto 1 es D10 (0,614) en A y D05 (0,647) en B.
- [x] **Mitades.** Ninguna ventana de crisis cruza el corte de ⌊n/2⌋. ltcm_russia_1998 empieza el 1998-07-17, después
  del corte de A (1998-04-24/27), y fed_tightening_2018q4 empieza después del de B (2018-05-01/02). Crisis con días
  OOS: 5 en H1 y 12 en H2 en A; 4 y 5 en B. El bloque recalculado en cada mitad sería 72 y 58 en A, y 79 y 35 en B.
- [x] **Calendario de la online.** 673 meses OOS en A y 194 en B. Primeros pesos aprendidos: 1970-05-29 en A y
  2010-07-30 en B (con 21 días: 1974-10-31 en A; con 63 sesiones de retraso: 1970-08-31 y 2010-10-29).
- [x] **Etiquetas.** Las ventanas del juez y `labels.parquet` (`inicio`, `trough_date`) coinciden en las 28 crisis.
  Las primeras crisis terminadas con días OOS son bear_1969_70 en A y flash_crash_euro1_2010 en B.
- [x] **Huellas.** `fusion/` no aparece en `benchmark.cache._cache_code_paths` ni en
  `sinteticos.aumento.config.huella_generador`.
- [x] **Decisiones del usuario** (03-11-2026), sobre las dudas del borrador:
  - la binarización principal de (a) es «accionable», con «confirmación tras aviso» como sensibilidad y
    «confirmado» puro como referencia (§3.1);
  - (ii) se lee en el MCS de todo el OOS, como en los individuales, y el de H2 queda sin peso (§5.1 y §6.3);
  - η = 10 se mantiene, declarando que se fijó con el rango de pérdidas de todo el OOS, con el control del η de H1
    (§4.4);
  - los pesos iguales se mantienen hasta la primera crisis terminada, con la sensibilidad de 21 días (§4.2);
  - H1 de B tiene solo 4 crisis y su control por mitades es ruidoso (§5.2).
- [x] **Revisión** (03-11-2026). Se precisaron reglas que el borrador dejaba abiertas: comparaciones de score sin
  redondeo, nivel 0 en H2 para (iii), señales idénticas en el MCS, F\* inexistente, top-n sin filtro de nivel y el
  `switching_rate` binario de D03 y D08.

### 8.1 Erratas

- **03-11-2026 · enlaces del §1.** Los enlaces a los notebooks 15 y 16 antiguos (`15_fusion_d07_d08` y
  `16_fusion_d02_d06`) apuntan ahora a [`15_fusiones_fijas`](../../notebooks/fase1/15_fusiones_fijas.ipynb), por la
  unificación prevista en el §7.3. El texto no cambia.

## 9. Resultados

Calculado el 03-11-2026 después de congelar. Hubo una sola ejecución limpia: primero las fijas, después la online,
en un proceso, con 5.000 réplicas en todos los MCS. Las tablas están en `results/fusion/fijas/` y
`results/fusion/online/`. La lectura se hace en los notebooks [15](../../notebooks/fase1/15_fusiones_fijas.ipynb) y
[17](../../notebooks/fase1/16_fusion_online.ipynb).

### 9.1 Fusiones fijas (§6.1)

| Pista | F\* (elegida en H1) | Score de F\* en todo el OOS | I\* | (i) | (ii) MCS-F | (iii) H2: F\* frente a I\*_H1 | Cumple |
|---|---|---|---|---|---|---|---|
| A | AND_D01+D10-P (c) | 0,597 | D10 0,614 | no | no (p = 0,063) | sí: 0,541 frente a 0,471 (D10-P) | no |
| B | VOTO_3de5 (b) | 0,614 | D05 0,647 | no | sí (p = 0,412) | sí: 0,543 frente a 0,428 (D01) | no |

**Veredicto: no aporta.** Combinar detectores no mejora al mejor detector solo.

Lectura sin peso (§6.3):

- **Hay fusiones mejores que I\*, pero no se pueden elegir de antemano.** Con todo el OOS superan a I\* 21 de las 423
  fusiones de A (casi todas AND con D10, D01 o D05) y 2 de las 547 de B. Elegirlas mirando todo el OOS infla el
  score en +0,045 en A y +0,043 en B.
- **El orden de H1 predice mal el de H2:** la correlación de Spearman entre las fusiones es 0,36 en A y 0,45 en B.
- **(iii) se cumple porque I\*_H1 se hunde en H2, no porque F\* sea buena.** En H2, D10 da 0,706 en A y D05 da 0,619
  en B, los dos por encima de F\*.
- **Las fusiones de antes, medidas con esta vara, quedan por debajo del mejor individual:**
  - D2 + D6 queda en el puesto 136 de 423 en A y en el 41 de 547 en B;
  - D7 + D8 queda en el 206 de 423 en A y en el 362 de 547 en B.

### 9.2 Fusión online (§6.2)

| Pista | Score de O (η = 10) en todo el OOS | I\* | (i) | (ii) MCS-O | O en H2 frente a I\*_H1 | (iii) | Cumple |
|---|---|---|---|---|---|---|---|
| A | 0,655 | D10 0,614 | sí | sí (p = 0,91) | 0,683 frente a 0,471 (D10-P) | sí | **sí** |
| B | 0,581 | D05 0,647 | no | sí (p = 0,35) | 0,568 frente a 0,428 (D01) | sí | no |

**Veredicto: indicio.** Se cumple solo en la pista A.

**Veredicto secundario: los pesos no aprenden.** En A, O supera a M (la media simple) en todo el OOS, 0,655 frente
a 0,632, pero no en H2, 0,683 frente a 0,696. En B, O sí supera a M en los dos tramos, pero en B no cumple el
criterio principal.

Lectura sin peso (§6.3):

- **En el MCS-O de A, O entra pero no encabeza.** La pérdida más baja sigue siendo la de D06 (0,550), seguida de V,
  D05 y M. O tiene 0,577, mejor que D10 (0,652), y F\* queda fuera.
- **En B, F\* sí supera a O** en score: 0,614 frente a 0,581.
- **η:** en H1 gana η = 5 en A, que en H2 da 0,688 frente a 0,683 con η = 10. En B gana η = 10.
- **Sensibilidades con η = 10** (score en todo el OOS):

  | Sensibilidad | A | B |
  |---|---|---|
  | Entrada binaria | 0,650 | 0,560 |
  | Retraso de 63 sesiones | 0,663 | 0,609 |
  | Pesos iguales hasta 21 días | 0,619 | 0,581 |
  | `excluir_sin_terminar` | 0,660 | 0,602 |

  Ninguna llega a D05 en B.
- **Los pesos no se concentran:**
  - A: número efectivo medio de 8,9 detectores sobre 15; al final pesan más D06 (22 %), D05 (19 %) y D01 (13 %).
  - B: 7,4 sobre 17; al final pesan más D05 (25 %), D06 (23 %) y D10 (16 %).

### 9.3 Consecuencias

- **Ninguna fusión desplaza a los mejores detectores solos en las dos pistas.** La fusión online es la única que
  gana en una, la A, y esa ventaja no viene de que los pesos aprendan: la media simple rinde igual o más en H2.
- **`get_regimen`** sigue con un detector individual, D06 en A y D05 en B, hasta la decisión final (ADR-006,
  notebook 22). La fusión online de A queda como candidata que hay que contrastar allí.
- **Para el TFM:** se suma al resultado del ADR-009 y del ADR-010: con 17 crisis, la complejidad no se paga.
