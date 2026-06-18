# Revisión académica (dictamen de tribunal) — capa "banco de pruebas" de detección de regímenes

> Documento de SOLO LECTURA emitido por un revisor externo en la fase de pulido.
> No modifica código, resultados, notebooks ni el resto de la memoria. Coteja las
> conclusiones contra `results/metrics_master_final.csv`, el código de
> `src/evaluation.py` / `src/detector_base.py` / `detectors/_hmm_utils.py`, las
> fichas `docs/memory/detectors/*.md`, las figuras `results/fase4_*.png` y
> `docs/references.bib`. Fecha: 2026-06-18.

---

## 0. Resumen ejecutivo

El trabajo está **a nivel de entrega de TFM** en su componente metodológico: el
marco de evaluación es causal de verdad (no solo en prosa), la honestidad
comparativa es notable (resultados negativos documentados, separación por
ventana, no se proclama un "mejor detector único") y la bibliografía central es
**real y bien atribuida**. La nota de solidez metodológica que le pondría es
**8/10 (notable alto)**. Lo que separa ese 8 de un sobresaliente no es el código
—que es sólido— sino tres cosas: (i) un par de **inconsistencias entre el texto
cuidadoso y las figuras/claims auxiliares**, (ii) **ausencia total de
significancia estadística** en las comparaciones, y (iii) algunas
**sobre-afirmaciones** que un tribunal atacaría aunque el fondo sea correcto.

---

## 1. Fortalezas (lo que se defiende bien ante un tribunal)

1. **La causalidad es real, no retórica.** Verificado en código:
   - `walk_forward` (`evaluation.py:212-246`) entrena solo con `X.iloc[:t]` y
     predice el bloque `[t:t+step]`; el re-etiquetado económico por fold usa
     `market_returns.reindex(train.index)` (líneas 228-230) — **solo train**. El
     orden de estados se congela con datos de entrenamiento y se aplica al bloque
     futuro: **no hay fuga** en el etiquetado.
   - `_hmm_utils.filtered_posterior` (`detectors/_hmm_utils.py:38-62`) es un
     forward filtering escalado en log donde `log_alpha[t]` depende únicamente de
     `log_alpha[t-1]` y `logB[t]`: **estrictamente causal**. La sustitución de
     Viterbi-por-bloque por filtrado forward está bien justificada y, además,
     mejora la estabilidad (hallazgo (b)), que es un punto genuinamente
     interesante.
   - El `stability_panel` está **bien aislado**: se guarda en `panel.attrs`
     (`evaluation.py:253`) y `evaluate` solo lo lee para `label_stability`
     (`evaluation.py:463`); todas las demás métricas tocan `wf_panel["state"]` /
     `["p_crisis"]`. El diseño hace imposible que el diagnóstico no causal
     contamine cobertura/falsas alarmas/lead-lag. Defendible al 100%.

2. **Honestidad comparativa.** Los dos negativos (D11 MS-GARCH degenerado,
   cov_GFC 0%; D12 deep AE peor que su PCA) se conservan en el master como
   evidencia, no se ocultan. El hallazgo "no hay mejor detector único, hay
   mejor-para-qué" está bien sostenido por el reparto del podio entre 4 familias.

3. **El hallazgo cuantitativo estrella es verificable.** D8 vs D4 sobre idénticas
   features: BIC 24415.9 vs 35379.4 (CSV, filas `hmm_tstudent_4s` y
   `hmm_gaussian_2s`) → ΔBIC ≈ 10963. La cifra del texto es exacta. La estructura
   de ν decreciente por estado [10.2, 7.6, 4.2, 2.4] es económicamente coherente
   con los hechos estilizados. Es el argumento más fuerte del trabajo.

4. **Arreglo 4 (severidad vol-primaria) bien motivado.** El problema que resuelve
   (con K=2 el signo ruidoso de una diferencia de medias casi nula invierte
   crisis/calma en detectores de varianza) es real y la solución por bandas de
   vol (`detector_base.py:281-289`) es razonable y está aislada en el núcleo, no
   en parches por detector.

5. **Separación por ventana en el texto.** La §1.1 de `99_conclusions.md`
   presenta dos tablas separadas (vio/no vio 2008 OOS) y nunca penaliza a un
   detector por lo que no pudo ver. Es exactamente la disciplina que se espera.

---

## 2. Debilidades y sobre-afirmaciones (con ubicación)

### D1 — Lead/lag: ni separado por ventana ni protegido del censurado *(la más seria)*
- **Ubicación:** `99_conclusions.md` §1.4 ("D7 cusum −240 d > D2 comp −235 d >
  D4 hmm −185 d") y figura `results/fase4_leadlag.png`.
- **Problema A (mezcla de ventanas):** la disciplina de "nunca mezclar ventana
  larga y corta" que se aplica con rigor a la cobertura **NO se aplica al
  lead/lag**. El −240 de D7 es media sobre **4 troughs** (incl. GFC 2008 −252 y
  Euro 2011 −252; CSV); el −235 de D2 y el −185 de D4 son media sobre **solo 2
  troughs** (COVID + Inflación, porque sus ventanas arrancan en 2015/2012). Se
  rankea sobre conjuntos de eventos distintos: no es apples-to-apples.
- **Problema B (censurado en el tope del lookback):** `lead_lag` usa
  `lookback=252` (`evaluation.py:329`). D7 da exactamente −252 en GFC, Euro e
  Inflación: significa que la señal ya estaba en crisis sostenida el **primer
  día** de la ventana de 252 → el verdadero lead está **censurado** en −252, no
  es "240 días de anticipación". Para un detector con `false_alarm_rate ≈ 0.87`,
  ese −252 refleja más baja precisión (crisis casi permanente alrededor del
  trough) que capacidad de alerta genuina. **La propia ficha D7 lo admite**
  ("alcanza el tope del lookback=252"; "parte de esa antelación es porque el
  detector también marca la fase de subida de vol previa"), pero la conclusión lo
  presenta como dominancia limpia. Como el lead/lag sostiene la recomendación
  "D7 = segunda velocidad / alerta temprana", esta debilidad es estructural.

### D2 — La figura `fase4_rank_heatmap.png` contradice el principio de "no mezclar ventanas"
- **Ubicación:** columna "Cob.sistemica" del heatmap (citado como evidencia en
  §1) vs. el texto §1.1.
- **Problema:** el heatmap rankea **los 12 detectores juntos** en cobertura
  sistémica, dando a D3 gmm y D4 hmm rank 3 — junto a D5/D6 (rank 1-2). Pero D3 y
  D4 **no vieron 2008 OOS**; su "cobertura sistémica" es COVID-only y compite en
  la misma columna contra la media GFC+COVID de los de ventana larga. Es
  precisamente la comparación incomparable que el texto prohíbe. El texto es la
  versión cuidadosa; **la figura la deshace**. Un tribunal que mire la figura verá
  la inconsistencia. (Es figura, no número del master; arreglable documentando que
  esa columna NO respeta la separación, o marcando en blanco a los de ventana
  corta.)

### D3 — Sobre-afirmación en el hallazgo (d): D3 listado como evidencia de 2013 sin haber visto 2013
- **Ubicación:** `99_conclusions.md` §2(d): "...D3 GMM (0.00 en su ventana)" y
  "Seis o más detectores independientes coinciden en NO ver 2013".
- **Problema:** D3 (clustering_gmm) tiene `ventana_eval 2015-09→2026` y en el CSV
  `fa_TaperTantrum_2013` está **vacío (NaN)**: 2013 cae **fuera** de su OOS, no es
  "0.00 en su ventana". Atribuirle un 0.00 en 2013 es incorrecto: no lo evaluó.
  El núcleo del hallazgo (6 detectores reales: D1, D5, D6, D7, D8, D10, todos con
  valor numérico de fa_2013 en el CSV) **sí se sostiene**, pero padear la lista
  con D3 debilita un argumento que no lo necesitaba.

### D4 — El BIC (pilar de la recomendación) es in-sample, pero el banner dice "todo sale del walk-forward"
- **Ubicación:** `99_conclusions.md` cabecera ("todo sale del walk-forward... sin
  look-ahead", "Los números de este documento son los reales de esa tabla") vs.
  `evaluate` (`evaluation.py:468-474`), que calcula `score/aic/bic` sobre
  `X_full`.
- **Problema:** logL/AIC/BIC son **in-sample** por construcción (es correcto: el
  BIC es un criterio de selección in-sample). Pero la afirmación global de que
  "todo" sale del walk-forward es imprecisa: la columna que más peso da a la
  recomendación de D8 **no** es out-of-sample. No es un error metodológico (el BIC
  debe ser in-sample), es una **imprecisión de redacción** que conviene matizar
  para no exponerse a la pregunta "¿su mejor evidencia es in-sample?".

### D5 — La recomendación de D8 se apoya en una métrica favorable-por-construcción
- **Ubicación:** §3 y §4 de `99_conclusions.md`.
- **Problema:** la cobertura OOS *estricta* de D8 es floja (cov_COVID 0.66,
  cov_Inflación 0.33; CSV) y **nunca evaluó 2008 OOS**. D8 "iguala a los binarios"
  solo bajo **estrés agregado** (estado ≥ n−2), métrica introducida en FASE 4 que
  por diseño favorece a los multi-estado, y que el propio texto reconoce que sube
  fa_2018 de 0.034 a 0.81. Es decir: el caso empírico OOS de D8 sobre un D1/D6
  simple es **delgado**, y descansa en (a) un BIC in-sample y (b) una métrica
  generosa con K≥3. El framing "la exploración CONFIRMA el núcleo de la propuesta"
  roza la **confirmación de la hipótesis previa**: la propuesta ya pedía un HMM
  t-Student de 4 estados, y la capa lo "confirma". El texto es honesto en las
  limitaciones (§4 las lista), pero el verbo "confirma" es más fuerte de lo que el
  OOS estricto sostiene. Mejor: "es consistente con / no contradice".

### D6 — `false_alarm_rate` con ground-truth laxo se reporta sin intervalo interpretativo
- Las fichas D6/D7 ya advierten que el far ≈0.85-0.87 infla por marcar como falsa
  alarma toda alta vol fuera de las 4 ventanas (1987, LTCM, dotcom, SVB...). Bien
  visto en las fichas, pero en el master `false_alarm_rate` aparece como número
  pelado; un lector que solo mire la tabla penalizará a D6/D7 injustamente. Es un
  problema de **presentación**, no de cálculo.

---

## 3. Verificación de citas centrales (a mano sobre `docs/references.bib`)

| Clave | ¿Existe? | ¿Bien atribuida? | Notas |
|---|---|---|---|
| `hamilton1989` | Sí (líneas 8-17) | **Correcta** | Hamilton, *Econometrica* 57(2):357-384, 1989. Es el paper seminal de Markov-Switching. Autor/año/título/páginas reales. |
| `kritzman2012` | Sí (41-50) | **Correcta (con matiz menor)** | Kritzman, Page, Turkington, "Regime Shifts: Implications for Dynamic Strategies", *Financial Analysts Journal* 68(3):22-39, 2012. Real. Matiz: el índice de turbulencia de Mahalanobis se atribuye canónicamente a **Kritzman & Li (2010)**, "Skulls, Financial Turbulence, and Risk Management"; el paper de 2012 usa turbulencia pero como aplicación. La atribución para D10 es **defendible**; añadir Kritzman-Li 2010 como fuente primaria reforzaría. |
| `hmm_nystrup2020` | Sí (480-488) | **Correcta** | Nystrup, Lindström, Madsen, "Learning Hidden Markov Models with Persistent States by Penalizing Jumps", *Expert Systems with Applications* 150:113307, 2020. Real y es la base legítima del statistical jump model (penalización de saltos). Bien usada para D9. |
| `vol_haasmittnikpaolella2004` | Sí (702-711) | **Correcta** | Haas, Mittnik, Paolella, "A New Approach to Markov-Switching GARCH Models", *Journal of Financial Econometrics* 2(4):493-530, 2004. Real; es exactamente la formulación HMP que D11 implementa. |

**Otras citas de soporte verificadas como reales y bien atribuidas:**
`hmm_bulla2011` (Quantitative Finance 11(3), HMM con componentes t), `hmm_rabiner1989`,
`vol_bollerslev1987` (la GARCH-t, correcta para D6), `reglas_bloom2009`,
`cp_page1954` (CUSUM seminal), `clust_schwarz1978bic` (BIC).

**Conclusión de citas:** **no se detectan citas inventadas ni mal atribuidas**
entre las centrales ni en la muestra de soporte. La gestión de la fusión
bibliográfica (claves duplicadas `hmm_*`/`ms_*`/`nn_*` para el mismo paper) está
**documentada explícitamente** en el encabezado del .bib (líneas 94-104) — es algo
inusual (BibTeX las trata como entradas independientes) pero honesto y con
instrucción de unificación para la redacción final. Esto es un punto a favor, no
en contra. Único riesgo cosmético: si en el PDF final se citan ambas claves del
mismo trabajo, aparecería duplicado en la lista de referencias; conviene unificar
antes de compilar.

---

## 4. Riesgos de defensa: preguntas de tribunal que el repo NO responde

1. **Significancia estadística (la más peligrosa).** No hay ningún test de si las
   diferencias entre detectores son significativas. ¿D5 0.98 vs D1 0.92 en
   cobertura es señal o ruido con ~4 crisis? ¿El ΔBIC sobrevive a bootstrap por
   bloques? Sin bandas de confianza ni tests (Diebold-Mariano, bootstrap de
   bloques, etc.), todo el ranking es **puntual**. Un tribunal lo preguntará seguro.
2. **n efectivo = ~4 crisis.** Toda la sección de cobertura descansa en 4 eventos
   (2 de ellos OOS solo para ventana larga). La pregunta "¿cuánta de su conclusión
   es generalizable y cuánta es overfitting a 4 puntos?" no tiene respuesta
   cuantitativa.
3. **Sensibilidad a hiperparámetros del walk-forward.** `train_size=252*8`,
   `step=21` (y `step=63/126` en los caros) están fijados sin análisis de
   sensibilidad. ¿Cambia el podio si `train_size` o `step` cambian? No se muestra.
   Para D5/D8 el `step` se subió por coste, lo que mezcla "elección metodológica"
   con "presupuesto de cómputo".
4. **Elección de ventanas de crisis/trampa.** `CRISIS_WINDOWS` y
   `FALSE_POSITIVE_WINDOWS` (`evaluation.py:32-44`) son fechas fijadas a mano
   ("Lehman→suelo", etc.). ¿Por qué esos cortes y no otros? La cobertura es
   sensible a los bordes de ventana y no hay análisis de robustez a desplazarlos.
5. **Inicio en 2007 y la GFC.** La decisión de que HYG gobierne la ventana común
   (2007-04) deja la GFC pegada al inicio; los detectores de ventana corta
   (2012+/2015+) **nunca** la evalúan OOS. La validación sistémica de D8 (el
   recomendado) se apoya solo en COVID. ¿Es suficiente un único evento sistémico
   OOS para recomendar el núcleo?
6. **Validación del etiquetado económico.** El orden de estados se fija por
   vol-primaria con `VOL_CLOSE_FRAC=15%` (constante elegida, `detector_base.py:57`).
   ¿Es robusto el podio a ese 15%? No hay sensibilidad. Para un detector con
   estados de vol próxima el resultado podría cambiar.
7. **Coherencia predict_online / predict_proba con burn-in.** `walk_forward`
   llama por separado a `predict_online(test)` y `predict_proba(test)`
   (`evaluation.py:231-232`). La causalidad y la continuidad entre bloques dependen
   de que el detector preprenda el burn-in de train en AMBOS; está descrito en las
   fichas pero no hay un test unificado que lo verifique para todos los HMM. Un
   tribunal técnico podría pedirlo.

---

## 5. Lista priorizada de mejoras

### ALTA prioridad (mueve la nota; arreglable en pulido, sin re-cómputo pesado)
- **A1.** Corregir/anotar el lead/lag (Debilidad D1): separar por grupo de
  ventana igual que la cobertura, y declarar explícitamente que los valores en
  −252 están **censurados al lookback** (no son "240 d de anticipación"). Si no se
  reseparan los números, al menos un párrafo de caveat en §1.4.
- **A2.** Resolver la contradicción figura↔texto del `rank_heatmap` (Debilidad
  D2): documentar en el pie de figura que la columna "Cob.sistemica" no respeta la
  separación por ventana, o regenerar la figura marcando en blanco a los de
  ventana corta. (Regenerar la figura es barato si se quiere.)
- **A3.** Corregir el hallazgo (d) (Debilidad D3): quitar "D3 GMM (0.00 en su
  ventana)" de la lista de 2013 o sustituirlo por "N/A (2013 fuera de su OOS)". El
  argumento de 6 detectores reales se mantiene intacto.
- **A4.** Matizar el verbo "confirma" de la recomendación (Debilidad D5):
  reescribir como "consistente con / no contradice la propuesta", y dejar
  explícito que el caso de D8 descansa en BIC in-sample + estrés agregado, no en
  cobertura OOS estricta.
- **A5.** Matizar la cabecera de `99_conclusions.md`: "todo sale del walk-forward"
  → aclarar que cobertura/falsas alarmas/lead-lag/persistencia son OOS, pero
  logL/AIC/BIC son in-sample por definición.

### MEDIA prioridad (documentar; refuerza la defensa)
- **M1.** Añadir un párrafo de "limitación: sin significancia estadística (n≈4
  crisis)" reconociéndolo explícitamente — convierte una pregunta peligrosa de
  tribunal en una limitación asumida (mucho mejor recibido).
- **M2.** Documentar la elección de `train_size`, `step` y `VOL_CLOSE_FRAC`, y
  reconocer que la sensibilidad no se exploró (declararlo como trabajo futuro, no
  fingir que no existe).
- **M3.** En el master, acompañar `false_alarm_rate` de una nota de lectura (ya
  está en fichas; subirla a la tabla/conclusiones para que no se lea aislada).
- **M4.** Unificar las claves bib duplicadas antes de compilar el PDF para evitar
  entradas repetidas en la lista de referencias.

### BAJA prioridad / fuera de alcance de esta fase (requiere cómputo; solo señalar)
- **B1.** Bootstrap por bloques / tests de significancia sobre las diferencias de
  cobertura y sobre el ΔBIC. (Cómputo: medio; daría el salto a sobresaliente.)
- **B2.** Análisis de sensibilidad del walk-forward (barrer `train_size`, `step`)
  para 2-3 detectores representativos. (Cómputo: alto por los caros D5/D8/D11.)
- **B3.** Re-estabilizar D11 con más multistart por fold — pero el propio repo
  argumenta bien que es innecesario (D6 cubre el hueco). Mantener como negativo.

---

## 6. Veredicto global

**¿Está a nivel de entrega de TFM?** Sí, en su núcleo metodológico. El marco
causal es correcto en código (no solo en prosa), la honestidad comparativa es
ejemplar para una capa exploratoria, y la bibliografía central es real y bien
atribuida. Como **primera capa de evaluación** de un TFM mayor, cumple su función:
deja un banco de pruebas reutilizable, una recomendación trazable y unos negativos
honestos.

**Nota de solidez metodológica: 8/10 (notable alto).** Lo defendería con
confianza salvo en tres frentes que un tribunal exigente atacaría: (1) el lead/lag
mezcla ventanas y sobre-vende un valor censurado; (2) la figura rank_heatmap
contradice el principio de separación que el texto defiende; (3) no hay ninguna
medida de significancia con n≈4 crisis, lo que deja todo el ranking en terreno
puntual. Ninguno de los tres invalida el trabajo, y los dos primeros son
**arreglables en pulido sin re-cómputo** (puntos A1-A5). El tercero es la
limitación de fondo que conviene **declarar abiertamente** en lugar de esperar la
pregunta. Con A1-A5 aplicados, la nota subiría a 8.5-9; con B1 (significancia), a
sobresaliente.

**Las 3 debilidades más serias, en una línea cada una:**
1. Lead/lag: ranking sobre conjuntos de troughs distintos + valores censurados en
   −252 presentados como "anticipación real" (§1.4 y ficha D7).
2. `fase4_rank_heatmap.png` rankea cobertura sistémica mezclando ventana larga y
   corta, justo lo que el texto §1.1 prohíbe.
3. Sin significancia estadística ni sensibilidad de hiperparámetros: todo el
   podio es puntual sobre ~4 crisis.
