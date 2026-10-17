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
  - PCA causal sobre el pool completo, 35 o 77 columnas.
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

## 4. Orden recomendado

| Paso | Puntos | Por qué |
|---|---|---|
| 1 | A1 + A2 | 1-2 días, sin detectores nuevos; dan la conclusión estadística |
| 2 | B5 | Un detector sencillo que responde a Hammond y al pool infrautilizado |
| 3 | A3 | Reutiliza `lead_lag` |
| 4 | B6 + B7 | Cierran las comparaciones directas con los papers |
| 5 | A4, B8, B9, C1 | Si queda tiempo; A4 es el más caro |

Trabajar por checkpoints: parar al cerrar cada paso y revisar antes de seguir.

## 5. Otras tareas de documentación

- [ ] Corregir la descripción de Hammond en `fichas.json` y en la sección 1.2 del estado del arte (ver
  arriba).
- [ ] Comprobar en los PDF las cifras de Hammond y de FINSABER antes de citarlas en la memoria.
