# 99 — Conclusiones de la FASE 4: síntesis comparativa de los 12 detectores

> Síntesis honesta del banco de pruebas. La tesis central de esta capa NO es
> "cuál es el mejor detector", sino **"cuál es el mejor para qué"**: cada familia
> domina un eje distinto, y la comparación solo es legítima si se hace **causal** y
> **con equidad de ventana** (un detector de ventana corta no vio la GFC 2008 OOS y
> nunca se penaliza por lo que no pudo ver).
>
> **Qué es OOS y qué no.** Las métricas de comportamiento —cobertura, falsas
> alarmas, lead/lag, persistencia/switching, estabilidad— se calculan sobre las
> etiquetas **out-of-sample** del walk-forward de la FASE 3, **sin look-ahead**. En
> cambio, las de **bondad de ajuste (logL / AIC / BIC) son in-sample por
> definición** (se evalúan sobre la verosimilitud del modelo en sus datos), así que
> el eje de BIC (§1.5, §2c) NO es una métrica causal OOS: mide especificación, no
> capacidad predictiva fuera de muestra. Se interpreta como tal.
>
> Fuentes: `results/metrics_master_final.csv` (tabla maestra final), notebook
> `notebooks/13_comparison.ipynb` (ejecutado, 0 errores) y figuras
> `results/fase4_*.png`. Los números de este documento son los reales de esa
> tabla y ese notebook.

Mapa de detectores y familias (F1–F7):

| ID | Detector (clave master) | Familia | Clase | Ventana OOS | Vio 2008 OOS |
|---|---|---|---|---|---|
| D1 | `rule_vix_threshold` | F1 reglas | baseline | 1998–2026 | sí |
| D2 | `rule_composite_riskoff` | F1 reglas | baseline | 2015–2026 | no |
| D3 | `clustering_gmm_k3` | F2 clustering | baseline | 2015–2026 | no |
| D4 | `hmm_gaussian_2s` | F3 HMM | baseline (puente) | 2012–2026 | no |
| D5 | `markov_switching_var_2s` | F4 Markov-Switching | avanzado | 1993–2026 | sí |
| D6 | `garch_t_vol` | F5 GARCH | avanzado | 1993–2026 | sí |
| D7 | `changepoint_online` | F6 change-point | avanzado | 1993–2026 | sí |
| D8 | `hmm_tstudent_4s` | F3 HMM (avanzado) | avanzado | 2012–2026 | no |
| D9 | `jump_model` | F2↔F3 jump model | avanzado | 2015–2026 | no |
| D10 | `turbulence_mahalanobis` | F1 multivar. | avanzado | 1998–2026 | sí |
| D11 | `msgarch_regime` | F5 MS-GARCH | **exploratorio-negativo** | 1991–2026 | sí |
| D12 | `deep_ae_regime` | F7 redes | **exploratorio-negativo** | 2015–2026 | no |

---

## 1. Qué familia gana en qué eje (no hay mejor único)

El resultado más importante de la comparación es que **ningún detector domina
todos los ejes**. El mapa de ganadores por eje (notebook celda "GANADOR POR EJE",
y figura `results/fase4_rank_heatmap.png`) reparte el podio entre cuatro familias
distintas. La separación por ventana es **obligatoria**: la cobertura de
sistémicas grandes nunca mezcla a los de ventana larga (vieron 2008 OOS) con los
de ventana corta (2012+/2015+, que NO la vieron).

> **Salvedad sobre `fase4_rank_heatmap.png`.** En el heatmap, la columna
> *"Cob. sistémica"* rankea a los 12 detectores **juntos**, sin separar por
> ventana, por lo que asigna un rango alto a D3/D4 (ventana corta, solo COVID) al
> lado de D5/D6 (ventana larga, GFC+COVID) — precisamente la mezcla que el resto
> del documento prohíbe. Esa columna del heatmap **debe leerse solo como vista
> panorámica**; la comparación de cobertura legítima es la de §1.1, separada por
> grupo de ventana. Las otras columnas del heatmap (especificidad, persistencia,
> coste, BIC) sí son comparables entre todos.

### 1.1. Cobertura de crisis sistémicas grandes (GFC 2008 + COVID 2020)

Separada por grupo de ventana para no comparar lo incomparable
(`results/fase4_sensibilidad_especificidad.png`):

| Grupo | Ranking (cob. = media GFC+COVID, alto = mejor) |
|---|---|
| **Vio 2008 OOS (ventana larga)** | D5 msvar **0.98** > D6 garch **0.97** > D7 cusum **0.92** ≈ D1 vix **0.92** > D10 turb 0.79 > D11 msg 0.10 (degenerado) |
| **No vio 2008 (ventana corta)** | D3 gmm **0.96** = D4 hmm **0.96** > D2 comp 0.84 > D9 jump 0.72 > D8 hmm-t 0.66 > D12 ae 0.54 |

**Ganador en ventana larga: D5 markov_switching_var (F4)**, con D6 GARCH-t (F5) y
D1 reglas VIX (F1) muy cerca. La conclusión cualitativa es que para cubrir
*crisis grandes y lentas* basta con cualquier detector que reaccione a la
**volatilidad/varianza** del S&P 500 con suficiente histórico: MS de varianza,
GARCH y una regla de umbral sobre VIX dan todos cobertura ≥0.92 de GFC+COVID. La
sofisticación econométrica de D5/D6 no compra una cobertura sustancialmente mayor
que la regla D1 en estas dos crisis (apenas +6 pp), un primer aviso de
parsimonia.

### 1.2. Especificidad (no disparar en las trampas 2013/2018)

Especificidad = `1 − media(fa_2013, fa_2018)`. Se reporta en versión **crisis
estricta** y **estrés agregado**:

| Versión | Top-3 (alto = mejor) |
|---|---|
| Estricta | D7 cusum **1.00**, D3 gmm **1.00**, D9 jump **1.00** |
| Estrés | D7 cusum **1.00**, D9 jump **1.00**, D1 vix **0.97** |

**Ganador: D7 changepoint_online (F6)**, único que mantiene especificidad 1.00 en
ambas versiones (no se contamina al ampliar a "corrección"). D9 jump model
también es perfecto en estricta y estrés. Nótese que la especificidad estricta de
D3 (1.00) es engañosa: al medirla por estrés agregado se desploma (ver §3), de ahí
que no figure en el top-3 de estrés.

### 1.3. Flickering / persistencia

Persistencia = duración media de régimen (alto = mejor); anti-flicker =
`switching_rate` (bajo = mejor). Figura `results/fase4_persistencia_sensibilidad.png`:

| Eje | Top-3 |
|---|---|
| Persistencia (días) | D7 cusum **436** > D9 jump **177** > D1 vix **75** |
| Anti-flicker (switching) | D7 cusum **0.002** < D9 jump **0.005** < D1 vix **0.013** |

**Ganador: D7 (F6)** por amplísimo margen. En el extremo opuesto, D12 deep_ae
(switching 0.287, duración 3.5 d) y D3 gmm (0.126, 7.9 d) flickean: confirman el
coste de no incorporar dinámica temporal en el clasificador.

### 1.4. Lead/lag (anticipación al suelo del drawdown)

`leadlag_*` (negativo = anticipa; señal sostenida 3 días para no premiar el cruce
suelto). Figura `results/fase4_leadlag.png`. **Dos cautelas obligatorias antes de
rankear**, ambas señaladas en la revisión académica:

1. **Comparar los MISMOS eventos** (igual que en la cobertura). La media de los 4
   eventos NO es comparable entre detectores: D7 promedia GFC+EuroDebt+COVID+Inflación
   (vio 2008 OOS), mientras los de ventana corta (D2, D4, D8, D9…) solo tienen
   COVID+Inflación. Por eso el "−240 d" de D7 frente al "−235 d" de D2 mezcla
   conjuntos de eventos distintos. Abajo se separa por grupo de evento.
2. **Censura en el tope del `lookback` (= 252 días).** El lead/lag se busca en los
   252 días previos al suelo; un valor de **−252 NO significa "252 días exactos de
   anticipación"**, sino que la señal de crisis sostenida **ya estaba activa al
   inicio de esa ventana** → el lead real es **≥252 y queda sin medir** (censurado
   por la derecha). D7 toca ese tope en **3 de sus 4 eventos** (GFC, EuroDebt,
   Inflación = −252), así que su media −240 es un **límite inferior**, no una cifra
   precisa.

| Eventos comunes a todos (COVID 2020 + Inflación 2022) | Top-3 |
|---|---|
| anticipación media | D2 comp **−235 d** > D7 cusum **−228 d** (Infl. censurada) ≈ D5 msvar **−190 d** / D4 hmm **−185 d** |

| Sistémicas tempranas (GFC + EuroDebt, solo ventana larga) | |
|---|---|
| D7 anticipa de forma sostenida ambas (GFC −252c, EuroDebt −252c, **censuradas**); D5/D6/D1/D10 anticipan la GFC (−250…−252) pero entran tarde en EuroDebt (−39…−43). |

**Lectura: D7 (F6) anticipa de forma sostenida y temprana en todos los eventos y
sin flickering**, lo que sí lo distingue como señal de **alerta temprana**; pero el
**número de días no debe citarse como preciso** (está topado por el `lookback` y,
para un detector con `false_alarm_rate ≈ 0.87`, "entrar pronto" convive con baja
precisión). La virtud defendible de D7 es *cualitativa* —cruza pronto y se mantiene—
no "240 días" literales.

### 1.5. Ajuste estadístico (BIC, solo modelos generativos)

`results/fase4_bic.png`. Menor = mejor; comparable estrictamente solo entre
modelos con **las mismas features/ventana**:

| Top-3 BIC | Aviso de comparabilidad |
|---|---|
| D8 hmm-t **24416** < D6 garch **26627** < D11 msg 26823 | comparación legítima D4↔D8 (mismas 7 features puente): D4 35379 vs D8 **24416**, **ΔBIC ≈ +10963** a favor de la t-Student |

**Ganador: D8 hmm_tstudent_4s (F3)**. El salto de BIC frente a su gemelo gaussiano
D4 sobre idénticas features es el hallazgo cuantitativo más nítido del banco (ver
§2c).

### 1.6. Coste computacional

Escala cualitativa {bajo, medio, alto} documentada desde las notas de build (no
inventada con falsa precisión):

| Coste | Detectores |
|---|---|
| **bajo** | D1 vix, D2 comp, D7 cusum, D10 turb |
| medio | D3 gmm, D4 hmm, D6 garch, D9 jump, D12 ae |
| **alto** | D5 msvar (~33 min), D8 hmm-t (~5 min), D11 msg |

**Ganador (más barato y útil): D7 (F6)**, cálculo cerrado por día. D7 barre cuatro
ejes simultáneos —especificidad, persistencia, lead/lag y coste— siendo a la vez
el más barato.

### 1.7. Resumen del reparto del podio

| Eje | Familia ganadora | Detector |
|---|---|---|
| Cobertura sistémica (ventana larga) | F4 Markov-Switching | D5 msvar (D6, D1 cerca) |
| Especificidad / trampas | F6 change-point | D7 cusum |
| Persistencia / anti-flicker | F6 change-point | D7 cusum |
| Lead/lag | F6 change-point | D7 cusum |
| Ajuste BIC | F3 HMM t-Student | D8 hmm-t |
| Coste | F6 change-point / F1 reglas | D7 cusum / D1 vix |

Cuatro familias se reparten seis ejes. **No existe un detector dominante**: la
elección depende del eje que el sistema aguas abajo priorice.

---

## 2. Hallazgos metodológicos transversales (el corazón del TFM)

Más allá del ranking, lo que esta capa aporta al TFM son seis lecciones
metodológicas, cada una con su evidencia causal.

### (a) El look-ahead de los z-scores in-sample compraba SUAVIDAD, no acierto

La tarea previa (HMM gaussiano in-sample, `docs/context/RESUMEN_DETECCION_REGIMENES.md`)
acertaba 2008 (98.6%) y 2020 (92.3%) con un perfil de régimen muy suave. Al
reimplementar ese mismo modelo de forma **causal walk-forward** (D4), la suavidad
se evapora: el switching sube de **0.047** (in-sample) a **0.100** (causal) y el
modelo **falla 2013 (25%) y 2018 (46%)** igual que antes, pero ahora sin la
ilusión de continuidad. La conclusión es incómoda y honesta: **el suavizado
anti-causal de los z-scores de muestra completa regalaba persistencia falsa** —
estabilidad prestada del futuro—, no capacidad real de clasificación. El acierto
en crisis grandes nunca dependió del look-ahead; lo que el look-ahead aportaba era
cosmética temporal.

### (b) Viterbi-por-bloque era MENOS causal Y menos estable que el filtrado forward

Al exigir causalidad estricta hubo que sustituir el Viterbi (que mira días futuros
dentro de su bloque) por **filtrado forward** en D4 y D8 (`detectors/_hmm_utils.py`).
Contra la intuición, el cambio **mejoró** la estabilidad: en D4 el switching
**bajó 0.124 → 0.100** y la duración media **subió 8.1 → 9.9 días**. La razón es
que el Viterbi-por-bloque **reiniciaba la decodificación en cada frontera de
bloque** (cada 21 días), inyectando switching artificial en los empalmes. El
filtro forward con burn-in es continuo: más causal *y* más persistente. Lección:
en evaluación online, la elección del algoritmo de decodificación no es neutral —
el "óptimo" offline (Viterbi) introduce un artefacto de frontera que el filtrado
causal evita.

### (c) La t-Student mejora el ajuste con holgura: el gaussiano estaba mal especificado

El EDA de la FASE 1 ya avisaba: curtosis en exceso de 25–40 (SP500 25.6, HYG 39.6)
— colas mucho más pesadas que la normal. La comparación D4↔D8 sobre **idénticas 7
features puente** cuantifica el coste de ignorarlo: **ΔBIC ≈ +10963** a favor de
la t (D8 24416 vs D4 35379). Además, D8 estima ν (grados de libertad) **por estado
de forma decreciente: [10.2, 7.6, 4.2, 2.4]** — es decir, el estado de crisis tiene
las colas más pesadas (ν más bajo), exactamente la estructura que predice la teoría
de los hechos estilizados financieros (`hmm_bulla2011`, `hmm_nystrup2015`). La
conclusión: **el supuesto gaussiano estaba mal especificado**, y la t-Student no es
un adorno sino la corrección distribucional que los datos exigen.

### (d) 2013 (taper tantrum) es el PUNTO CIEGO UNIVERSAL — la taxonomía importa

Seis detectores **independientes y de familias distintas** que SÍ tuvieron 2013 en
su tramo out-of-sample coinciden en NO verlo: D1 reglas (fa 0.00), D5 MS-VAR
(0.038), D6 GARCH (0.113), D7 change-point (0.00), D8 HMM-t (0.00), D10 turbulencia
(0.123). La activación en 2013 oscila entre **0% y ~12%** en todos ellos. (Los de
ventana corta —D3, D2, D9, D12— tienen 2013 **fuera** de su OOS: N/A, no cuentan ni
a favor ni en contra.)

El matiz honesto y exigido: el hallazgo **NO** es "2013 es ruido reclasificable".
Es que **ninguna definición de crisis basada en volatilidad/correlación equity
captura 2013**, porque el taper fue un shock de *tipos* sin estrés sistémico de
renta variable. Que seis detectores construidos sobre features de vol/correlación
converjan en lo mismo es **evidencia robusta**, no un fallo común: **la taxonomía
de régimen importa**. Por eso 2013 se mantiene deliberadamente como
**ventana-trampa** (false-positive window) para medir especificidad, y NO se
reclasifica como crisis. Capturar 2013 exigiría ampliar la *taxonomía de features*
(curva de tipos, MOVE) y/o el *espacio de regímenes*, no afinar un umbral.

### (e) MS-GARCH (D11) y deep (D12) no superan a los baselines: parsimonia validada

Los dos detectores más complejos del banco son **resultados negativos explícitos**:

- **D11 msgarch_regime (MS-GARCH-t, F5)**: en walk-forward el modelo **degenera** —
  el fold de la GFC colapsa a un único régimen → **cobertura GFC 2008 = 0%**, false
  alarm rate **0.95**. Implementable y causal, pero frágil: la riqueza paramétrica
  (transición × dinámica GARCH × colas t) no se identifica con tan pocas crisis.
- **D12 deep_ae_regime (AE→GMM, F7)**: el autoencoder **empeora** a su baseline
  PCA→GMM (switching 0.287 vs ~0.091, false alarm 0.60 vs ~0.14) **sin ganar
  cobertura**. La no-linealidad no aporta.

Con ~4 crisis en la muestra efectiva, la complejidad extra no se paga: a veces es
neutra (D12) y a veces destructiva (D11, cov_GFC 0%). **La parsimonia queda
validada empíricamente**: los baselines (D1 reglas, D6 GARCH) cubren los huecos que
los complejos dejan.

### (f) Lecciones de diseño del MARCO de evaluación

Dos arreglos del núcleo resultaron imprescindibles para comparar **con justicia**
detectores que NO operan sobre retornos:

- **Severidad vol-primaria (Arreglo 4)**: con K=2, el criterio `z(std) − z(mean)`
  dejaba que el **signo ruidoso de una diferencia de medias casi nula** invirtiera
  crisis↔calma en detectores que separan solo en varianza (σ GARCH, turbulencia
  Mahalanobis). La solución: ordenar los estados por **banda de volatilidad**
  (ancho `VOL_CLOSE_FRAC=15%` de la vol media); el retorno medio solo desempata
  entre vols próximas. Así la vol manda y la etiqueta nunca se invierte por ruido
  de medias. (Lo descubrió D6; se resolvió en el núcleo, no con parches locales.)
- **Etiquetado económico robusto por fold**: en cada ventana walk-forward se re-fija
  el orden 0=calma…n−1=crisis usando los retornos del S&P 500 de *ese* fold. Sin
  ello, un detector que clasifica sobre σ o sobre Mahalanobis no tiene forma de
  saber cuál de sus estados es "crisis". Es la pieza que hace **comparables** entre
  sí familias que ni siquiera ven los retornos.

Sin estos dos arreglos, la mitad de los detectores (varianza, GARCH, change-point,
Mahalanobis) habrían sido incomparables o se habrían invertido aleatoriamente.

---

## 3. La tensión de la métrica de estrés agregado

Los detectores **multi-estado** (D8 k=4, D3/D12 k=3) quedan en desventaja si solo
se mide su **crisis estricta** (`state == n−1`, la cola extrema), porque los
**binarios** colapsan todo el estrés en su único estado de riesgo. Para comparar
**con justicia** se introduce la máscara de **estrés agregado**:

| K del detector | máscara de estrés | significado |
|---|---|---|
| binarios (K=2) | `state == 1` | idéntico a crisis estricta |
| multi-estado (K≥3) | `state ≥ n−2` | unión de los dos estados más severos (corrección + crisis) |

Esta métrica es **necesaria** para que un D8 de 4 estados, cuya "crisis" es la cola
extrema, se compare en igualdad con un binario cuyo único estado de riesgo ya
engloba corrección + crisis.

El doble filo, con números reales (`results/fase4_estres_vs_estricta.png` y celda
"ESTRÉS AGREGADO vs CRISIS ESTRICTA"):

| Detector | cov_COVID | cov_Inflación | fa_2018 (trampa) | far global |
|---|---|---|---|---|
| **D8 hmm-t** estricta → estrés | 0.66 → **0.96** | 0.33 → **0.90** | 0.034 → **0.81** | 0.52 → 0.79 |
| **D3 gmm** estricta → estrés | 0.96 → 0.96 | 0.87 → 0.87 | 0.00 → **0.73** | 0.49 → 0.83 |
| **D12 ae** estricta → estrés | 0.54 → **0.96** | 0.10 → **0.79** | 0.15 → 0.19 | 0.60 → 0.72 |

Lectura honesta de los **dos lados**:

- **El lado bueno**: lo que parecía baja sensibilidad de D8 NO era un fallo, sino
  **reclasificación a "corrección"**. Medido con justicia (estrés agregado), D8
  **iguala a los binarios** en COVID (0.96) e Inflación (0.90). El HMM t-Student no
  se perdía 2022; lo etiquetaba como corrección, no como cola extrema — que es
  precisamente la distinción de régimen que un modelo de 4 estados debe hacer.
- **El lado malo**: ampliar a "corrección" **también** sube la activación en las
  trampas. fa_2018 de D8 salta 0.034 → **0.81** y su far global 0.52 → 0.79; D3
  sube fa_2018 0.00 → **0.73**. Es decir, la mayor cobertura de crisis lentas
  **no es gratis**: viene con más falsos positivos en las correcciones-trampa.

Conclusión: el estrés agregado es **el corte justo para comparar multi-estado con
binarios**, pero deja explícito un **trade-off**, no una mejora gratuita. D12,
además, sigue siendo negativo incluso medido por estrés (switching 0.287, far 0.72):
su problema no era la métrica, era el modelo.

---

## 4. Recomendación para la siguiente capa del TFM

La propuesta original del TFM mayor (`docs/context/`) define un **HMM t-Student de
4 estados** con lógica de **"dos velocidades"**. La pregunta que esta exploración
debía responder con evidencia es: **¿confirma los datos esa elección, o sugiere
ajustarla?**

**Veredicto: la evidencia es CONSISTENTE CON el núcleo de la propuesta y NO lo
contradice; sugiere complementarlo.** (Se evita el verbo "confirma": el respaldo a
D8 es fuerte pero descansa en un eje **in-sample** —BIC— y en una métrica
**favorable-por-construcción** a K≥3 —el estrés agregado—, no en cobertura OOS
estricta, donde D8 es modesto; ver la cautela al final de esta sección.)

1. **El núcleo HMM t-Student multi-estado sale reforzado**, por dos vías —ambas con
   su asterisco de honestidad:
   - **Por BIC (in-sample)**: D8 gana el eje de ajuste con holgura (ΔBIC ≈ +10963
     sobre el gaussiano D4 con las mismas features). La t-Student no es opcional; es
     la corrección que la curtosis 25–40 de los datos exige (§2c). *Pero el BIC es
     bondad de ajuste in-sample, no capacidad predictiva OOS* (§A5 de la cabecera).
   - **Por la lógica corrección↔crisis**: medido con **estrés agregado**, D8 iguala
     a los binarios en cobertura (COVID 0.96, Inflación 0.90), lo que da sentido al
     cuarto estado y a la **"segunda velocidad"** (corrección rápida vs crisis
     sistémica) que la propuesta persigue (§3). *Pero el estrés agregado es una
     métrica que, por construcción, favorece a los multi-estado (suma dos estados),
     y su cobertura OOS estricta es modesta (cov_COVID 0.66) — D8 nunca vio 2008
     OOS.* El argumento es de **plausibilidad estructural**, no de superioridad OOS
     demostrada.

2. **Complementar con un detector de change-point rápido y barato (D7) como
   segunda velocidad / alerta temprana.** D7 domina **cuatro ejes a la vez**
   (especificidad, persistencia, lead/lag, coste) y **anticipa de forma sostenida**
   los suelos en todos los eventos (los días concretos están topados por el
   `lookback`, no son una cifra precisa; §1.4). Donde el HMM-t da la *taxonomía* (qué tipo de régimen), el change-point da la
   *reactividad* (cuándo cambia el nivel), barato y sin flickering. Conecta
   directamente con "dos velocidades": D7 = alerta temprana / velocidad rápida;
   HMM-t = clasificación de régimen / velocidad lenta.

3. **Mantener baselines como control**: D1 (regla VIX) y D5/D6 (MS-VAR / GARCH-t),
   que dan la **mejor cobertura sistémica en ventana larga** y sirven de referencia
   honesta. Si un modelo complejo no bate a D1 en cobertura de GFC+COVID, no se
   justifica su coste.

Honestidad sobre las **limitaciones** que la siguiente capa hereda:

- **D8 tiene crisis estricta estrecha y coste alto**: su cola extrema sola es poco
  sensible (cov_COVID estricta 0.66); depende del estrés agregado para lucir, y su
  refit es caro (~5 min/build). No es un detector "para producción ligera".
- **2013 seguirá siendo punto ciego** salvo que se **amplíe la taxonomía** de
  features (curva de tipos, MOVE) o de regímenes. Ningún ajuste de hiperparámetros
  lo resuelve (§2d).
- **La GFC cae cerca del inicio de datos** para las ventanas cortas (2012+/2015+):
  D8 nunca la evaluó OOS. Su validación sistémica se apoya en COVID 2020, no en
  2008.

**Recomendación final, matizada**: llevar a la siguiente capa un **núcleo HMM
t-Student multi-estado** (respaldado por BIC y por la lógica corrección↔crisis, con
las cautelas de arriba), **emparejado con un change-point rápido tipo D7 como
segunda velocidad / sistema de alerta temprana**, y con **D1/D5/D6 como controles
baseline**. Descartar D11 y D12 como detectores operativos (validan la parsimonia,
no la aportan). Tratar 2013 como límite conocido de la taxonomía, no como objetivo
a forzar.

### 4.1. Limitación estructural: sin significancia estadística (n ≈ 4 crisis)

Toda la comparación de este banco descansa sobre **una muestra de ~4 crisis
sistémicas** (GFC 2008, EuroDebt 2011, COVID 2020, Inflación 2022) y dos
ventanas-trampa (2013, 2018). Con tan pocos eventos efectivos, **los rankings por
eje son puntuales: no hay tests de significancia ni intervalos de confianza**, y
diferencias de pocos puntos de cobertura o de lead/lag entre detectores **no son
estadísticamente distinguibles**. El podio (§1.7) debe leerse como *evidencia
cualitativa direccional* —qué familia tiende a dominar qué eje— y NO como una
ordenación con respaldo inferencial. Esta es una **limitación asumida del diseño**,
inherente a evaluar regímenes de crisis (que por definición son raros), no un
descuido.

Se declara como **trabajo futuro** de la siguiente capa, para convertir el podio
cualitativo en uno con incertidumbre cuantificada:
- **(B1) Bootstrap por bloques** (stationary/circular block bootstrap) sobre las
  series OOS para obtener bandas de confianza de cobertura, falsas alarmas y
  switching, y contrastar pares de detectores respetando la autocorrelación.
- **(B2) Análisis de sensibilidad de hiperparámetros** del protocolo walk-forward
  (`train_size`, `step`) y del etiquetado económico (`VOL_CLOSE_FRAC`), para
  comprobar que el reparto del podio es robusto y no un artefacto de la
  configuración elegida.

---

## 5. Bibliografía

Técnicas en juego, con las claves reales de `docs/references.bib`:

- **HMM (emisiones latentes)**: tutorial y algoritmos de inferencia/aprendizaje
  `hmm_rabiner1989`, `hmm_baumwelch1970`, decodificación `hmm_viterbi1967`; manual
  de referencia `hmm_zucchini2016`.
- **HMM con colas pesadas (t-Student)**: `hmm_bulla2011` (componentes t, mayor
  persistencia), hechos estilizados que lo motivan `hmm_nystrup2015`,
  `hmm_bullabulla2006`.
- **Markov-Switching econométrico**: `hamilton1989` (modelo seminal), filtro de Kim
  `ms_kim1994`, manual `ms_kimnelson1999`, revisión `ms_angtimmermann2012`.
- **GARCH / asimetría (GJR)**: `vol_bollerslev1986` (GARCH), `vol_glosten1993`
  (GJR-GARCH), `vol_engle1982` (ARCH).
- **MS-GARCH (el negativo D11)**: `vol_haasmittnikpaolella2004` (formulación HMP),
  `vol_ardia2019` (paquete MSGARCH), `vol_marcucci2005`.
- **Change-point / CUSUM (D7)**: `cp_page1954` (CUSUM seminal), survey
  `cp_truong2020`, variantes online `cp_adamsmackay2007`, cambios de varianza
  financiera `cp_inclantiao1994`, `cp_lavielleteyssiere2007`.
- **Jump model / penalización de saltos (D9)**: `hmm_nystrup2020` (HMM con estados
  persistentes penalizando saltos, base del statistical jump model).
- **Turbulencia financiera (D10)**: `kritzman2012` (distancia de Mahalanobis como
  índice de turbulencia y régimen).
- **Reglas / umbrales sobre VIX (D1, D2)**: `reglas_bloom2009` (shocks de
  incertidumbre), `reglas_moreiramuir2017` (gestión por volatilidad),
  `reglas_gilchristzakrajsek2012` (spreads de crédito).
- **Clustering (D3) y selección de K (BIC)**: `clust_pedregosa2011sklearn`
  (scikit-learn/GMM), `clust_lopezdeprado2020`, criterio `clust_schwarz1978bic`
  (BIC).
- **Redes / autoencoder (el negativo D12)**: `nn_hinton2006` (reducción de
  dimensionalidad con autoencoders), `nn_goodfellow2016`, clustering profundo
  `nn_xie2016`.
- **Metodología ML financiero (causalidad, walk-forward)**: `lopezdeprado2018`.

> Nota de citación (de la fusión de la FASE 2): preferir `clust_bucci2022realized`
> sobre `nn_bucci2021` (mismo trabajo, versión de revista) y unificar
> Ang & Timmermann / Hardy por su clave `hmm_*`.
