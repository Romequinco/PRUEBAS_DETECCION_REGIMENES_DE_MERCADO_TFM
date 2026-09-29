# 01 — Hallazgos (materia prima para el PDF ejecutivo científico)

> **Rol de este documento.** Recopilación trazada de hallazgos para el redactor LaTeX.
> NO es el PDF. Cada afirmación lleva su DATO y su FUENTE (fichero + sección).
> Nada inventado: todo está copiado/condensado de los `.md` ya pulidos (FASE 5).
> Fuentes citadas con claves cortas:
> - `[99]` = `docs/memory/99_conclusions.md`
> - `[IDX]` = `docs/memory/INDEX.md`
> - `[SOTA]` = `docs/memory/00_state_of_the_art.md`
> - `[EDA]` = `docs/memory/01_data_and_eda.md`
> - `[Dn]` = `docs/memory/detectors/<NN>_<nombre>.md`
> - `[PREV]` = `docs/context/RESUMEN_DETECCION_REGIMENES.md`

---

## 1. Puente narrativo (de dónde viene esta capa, qué es y qué no es)

### 1.1. La tarea previa y sus limitaciones
**Origen.** Esta capa nace del detector de regímenes de la práctica previa
`Tarea_riesgos.ipynb` (sistema de stress testing multi-activo): un **HMM gaussiano
de 2 estados, in-sample y con look-ahead** [PREV §3.2]. Datos: 5 indicadores
(`^GSPC`, `^VIX`, `TLT`, `IEF`, `HYG`), ventana **2007-04-11 → 2026-02-10**
(gobernada por HYG), 7 features estandarizadas [PREV §2, §3.1].

**Lo que acertaba y lo que fallaba** [PREV §4.3]:
| Evento | % días en Crisis | Veredicto |
|---|---|---|
| Lehman 2008 | 98.6 % | ✓ alta coherencia |
| Deuda Europea 2011 | 67.2 % | ✓ |
| COVID-19 2020 | 92.3 % | ✓ |
| Inflación 2022 | 78.9 % | ✓ |
| Taper Tantrum 2013 | 10.9 % | ✗ no detectado |
| Sell-off Q4 2018 | 20.6 % | ✗ no detectado |

→ Acertaba **crisis sistémicas grandes**, se perdía **correcciones rápidas** (2013/2018).

**Las 8 limitaciones documentadas del detector previo** [PREV §7] — son el motivo
de existir de este banco de pruebas:
1. **Solo 2 estados**: colapsa corrección / crisis sistémica / estanflación en uno.
2. **Ciego a crisis rápidas/medianas** (2013, Q4 2018).
3. **Etiquetado por umbral post-hoc** (vol + VIX), frágil ante régimen intermedio o
   divergencia VIX/vol.
4. **Look-ahead / sin walk-forward**: entrenado in-sample sobre toda la muestra; sin
   detección online causal.
5. **Supuesto gaussiano** de las emisiones (subestima colas; retornos leptocúrticos).
6. **Z-scores con media/desv de TODA la muestra** → fuga de futuro (otro look-ahead sutil).
7. **`P(estado 0 inicial)=1`**: inicialización degenerada.
8. **Sin medida de incertidumbre** (Viterbi "duro", no `predict_proba` suave).

> Nota de trazabilidad: `[SOTA §1]` y `[99 §2a]` resumen estas mismas debilidades en
> cuatro grandes ejes (look-ahead en z-scores, ausencia de walk-forward, Viterbi duro
> sin probabilidades, supuesto gaussiano). Las 8 finas están en `[PREV §7]`.

### 1.2. Qué propone el TFM mayor
La propuesta del TFM mayor (`docs/context/`) define el **sistema mayor**: un
**HMM t-Student de 4 estados**, ~14 features, selección de K por BIC, split
train/val/test 2000-2024, con lógica de **"dos velocidades"** (corrección rápida vs
crisis sistémica lenta) [99 §4; IDX "Contexto de partida"; PREV §8].

### 1.3. Qué ES esta capa (y qué NO es)
Esta capa **NO es un detector concreto**: es un **MARCO DE EVALUACIÓN causal,
comparativo y honesto** que enfrenta **12 detectores de 7 familias** bajo una sola
interfaz (`RegimeDetector`, etiquetas canónicas `0=calma…n-1=crisis`) y un protocolo
walk-forward / out-of-sample común sin look-ahead [SOTA §1; 99 cabecera].
La tesis central **no** es "cuál es el mejor detector" sino **"cuál es el mejor para
qué"**: cada familia domina un eje distinto, y la comparación solo es legítima si se
hace **causal** y **con equidad de ventana** [99 cabecera; IDX veredicto 1].

---

## 2. Los 6 hallazgos metodológicos transversales (el corazón del TFM)
> Fuente principal: `[99 §2]` (a–f). Cada uno con su evidencia numérica y su matiz.

### (a) El look-ahead de los z-scores in-sample compraba SUAVIDAD, no acierto
- **Dato.** Al reimplementar el HMM gaussiano de la tarea previa de forma causal
  walk-forward (D4), el switching sube de **0.047 (in-sample) → 0.100 (causal)** y el
  modelo sigue fallando **2013 (25%) y 2018 (46%)** [99 §2a; D4 "Re-evaluación"].
- **Matiz.** El acierto en crisis grandes NUNCA dependió del look-ahead; lo que el
  look-ahead aportaba era **cosmética temporal** (persistencia falsa, "estabilidad
  prestada del futuro"), no capacidad de clasificación [99 §2a].

### (b) Viterbi-por-bloque era MENOS causal Y menos estable que el filtrado forward
- **Dato.** Al sustituir Viterbi intra-bloque por **filtrado forward** en D4/D8
  (`detectors/_hmm_utils.py`), el switching **bajó 0.124 → 0.100** y la duración media
  **subió 8.1 → 9.9 días** [99 §2b; D4 "Re-evaluación"; IDX FASE 3 Arreglo 2].
- **Matiz / causa.** Contra la intuición, el cambio MEJORÓ la estabilidad: el
  Viterbi-por-bloque **reiniciaba la decodificación en cada frontera de 21 días**,
  inyectando switching artificial en los empalmes; el filtro forward con burn-in es
  continuo → más causal *y* más persistente. "El óptimo offline (Viterbi) introduce un
  artefacto de frontera que el filtrado causal evita" [99 §2b].

### (c) La t-Student está bien especificada; el gaussiano estaba mal especificado
- **Dato.** El EDA avisa de curtosis en exceso de **25–40** (SP500 25.6, HYG 39.6)
  [EDA §4; 99 §2c]. La comparación D4↔D8 sobre las **mismas 7 features puente**:
  **ΔBIC ≈ +10963** a favor de la t (**D8 24416 vs D4 35379**) [99 §1.5, §2c; D8].
  D8 estima ν (grados de libertad) **decreciente por estado [10.2, 7.6, 4.2, 2.4]** →
  la crisis tiene las colas más pesadas (ν≈2.4), la calma casi gaussiana (ν≈10.2) [99 §2c; D8].
- **Matiz.** La t-Student "no es un adorno sino la corrección distribucional que los
  datos exigen"; la estructura ν decreciente coincide con la teoría de hechos
  estilizados (`hmm_bulla2011`, `hmm_nystrup2015`) [99 §2c].

### (d) 2013 (taper tantrum) es el PUNTO CIEGO UNIVERSAL — la taxonomía importa
- **Dato.** Seis detectores **independientes de familias distintas** que SÍ tuvieron
  2013 en su tramo OOS coinciden en NO verlo (activación 0–~12%): D1 reglas (0.00),
  D5 MS-VAR (3.8%), D6 GARCH (11.3%), D7 change-point (0.00), D8 HMM-t (0.00),
  D10 turbulencia (12.3%) [99 §2d; D10]. (Los de ventana corta —D2, D3, D9, D12— tienen
  2013 **fuera** de su OOS: N/A.)
- **Matiz (exigido).** El hallazgo **NO** es "2013 es ruido reclasificable". Es que
  **ninguna definición de crisis basada en volatilidad/correlación equity captura 2013**,
  porque el taper fue un shock de *tipos* sin estrés sistémico de renta variable. Que
  seis detectores converjan es **evidencia robusta**, no fallo común: **la taxonomía de
  régimen importa**. 2013 se mantiene como **ventana-trampa** (false-positive window);
  capturarlo exigiría **ampliar la taxonomía de features** (curva de tipos, MOVE) y/o de
  regímenes, no afinar un umbral [99 §2d].

### (e) MS-GARCH (D11) y deep (D12) no superan a los baselines: parsimonia validada
- **Dato.** D11 msgarch-t **degenera** en walk-forward: el fold de la GFC colapsa a un
  único régimen → **cobertura GFC 2008 = 0%**, false alarm rate **0.95** [99 §2e; D11].
  D12 deep_ae **empeora** a su baseline PCA→GMM (switching **0.287 vs ~0.091**, false
  alarm **0.60 vs 0.14**) **sin ganar cobertura** [99 §2e; D12].
- **Matiz.** Con ~4 crisis, la complejidad extra no se paga: a veces neutra (D12), a
  veces destructiva (D11, cov_GFC 0%). **La parsimonia queda validada empíricamente**;
  los baselines (D1 reglas, D6 GARCH) cubren los huecos que los complejos dejan [99 §2e].

### (f) Lecciones de diseño del MARCO de evaluación (dos arreglos imprescindibles)
- **Severidad vol-primaria (Arreglo 4)** [99 §2f; D6; IDX Arreglo 4]. Con K=2, el
  criterio `z(std) − z(mean)` dejaba que el **signo ruidoso de una diferencia de medias
  casi nula** invirtiera crisis↔calma en detectores que separan solo en varianza (σ
  GARCH, turbulencia Mahalanobis). Solución: ordenar los estados por **banda de
  volatilidad** (ancho `VOL_CLOSE_FRAC = 15%` de la vol media); el retorno medio solo
  desempata entre vols próximas. Lo destapó D6; se resolvió en el núcleo, no con parches.
- **Etiquetado económico robusto por fold** [99 §2f; IDX Tarea A]. En cada ventana
  walk-forward se re-fija el orden `0=calma…n-1=crisis` usando los retornos del S&P 500
  de *ese* fold. Es la pieza que hace **comparables** entre sí familias que ni siquiera
  ven los retornos (varianza, σ, change-point, Mahalanobis).
- **Matiz.** Sin estos dos arreglos, la mitad de los detectores habrían sido
  incomparables o se habrían invertido aleatoriamente [99 §2f].

---

## 3. Veredicto por detector (D1–D12)
> Familia + qué es + veredicto, de cada ficha `[Dn]` y de `[99]`/`[IDX]`. Ventana OOS y
> "vio 2008 OOS" según la tabla de `[99]` (mapa de detectores).

- **D1 `rule_vix_threshold`** · F1 reglas · baseline · OOS 1998–2026 · vio 2008 OOS.
  Autómata de 2 estados sobre `VIX_level_z` con histéresis + dwell. Cobertura GFC 93.8%,
  COVID 90%, Inflación 34.9% (VIX flojo en 2022); 2013 = 0%, 2018 = 6.3% (la histéresis
  los suprime a propósito). switching 0.013, duración 75 d. **Baseline de "miedo" limpio
  y persistente; mejor cobertura sistémica de ventana larga junto a D5/D6** [D1; 99 §1.1].

- **D2 `rule_composite_riskoff`** · F1 reglas · baseline · OOS 2015–2026 · NO vio 2008.
  Voto compuesto VIX+crédito+curva+drawdown con histéresis. Mejora a D1 en 2022
  (53.8% vs 34.9%, +18.9 pp) pero más ruido (switching 0.039 vs 0.013, dur 26 d). 2008/2011
  NaN OOS (HYG desde 2007). **Complementa a D1, no lo sustituye; calibración de pesos
  sensible confirmada** [D2; IDX].

- **D3 `clustering_gmm_k3`** · F2 clustering · baseline NO temporal · OOS 2015–2026 · NO
  vio 2008. GMM full K=3 (por BIC) sobre 15 features, sin cadena de Markov. COVID 0.96,
  Inflación 0.87, **flickering severo (switching 0.126, dur 7.9 d)**. **Baseline que aísla
  el aporte de la dinámica temporal; flickering lo descarta como detector definitivo**
  [D3; 99 §1.3].

- **D4 `hmm_gaussian_2s`** · F3 HMM · **baseline puente** con la tarea previa · OOS
  2012–2026 · NO vio 2008 OOS. GaussianHMM 2 estados, 7 features puente. In-sample capta
  2008 (~100%)/2011 (81%); causal: 2008/2011 NaN (en train), COVID 96%, Inflación 86%,
  falla 2013 (25%)/2018 (46%). **Mide el efecto del look-ahead; el gaussiano subestima las
  colas → motiva D8** [D4; 99 §2a].

- **D5 `markov_switching_var_2s`** · F4 Markov-Switching · avanzado · OOS 1993–2026 · vio
  2008 OOS. MS de varianza sobre S&P 500, probabilidades **filtradas** causales. GFC 99.3%,
  COVID 96%, Inflación 73.7%; 2013 no se dispara (3.8%, correcto), 2018 sí (81%).
  switching 0.056, dur 17.9 d. Coste alto (~33 min). **Ganador de cobertura sistémica en
  ventana larga (mean GFC+COVID 0.98); control baseline interpretable** [D5; 99 §1.1].

- **D6 `garch_t_vol`** · F5 GARCH · avanzado · OOS 1993–2026 · vio 2008 OOS. GJR-GARCH(1,1)-t,
  umbral causal sobre σ con histéresis. GFC 100%, COVID 94%, Inflación 80%; **capta 2018
  (87%)** que D4 no, pero NO 2013 (11%, shock de tipos sin vol equity). switching 0.014,
  dur 70 d. BIC 26627. **Sensor de volatilidad fuerte y barato; control baseline** [D6; 99 §1.1].

- **D7 `changepoint_online`** · F6 change-point · avanzado · OOS 1993–2026 · vio 2008 OOS.
  CUSUM de Page online robusto (log|r| + mediana/MAD + winsor) sobre vol del S&P 500.
  GFC 100%, EuroDebt ~67%, COVID ~84%, Inflación ~77%; **especificidad 1.00** (trampas
  2013/2018 = 0%). **DOMINA 4 ejes** (especificidad, persistencia, lead/lag, coste);
  switching ~0.002, duración media ~436 d. El coste gaussiano (r²) degenera en alarma
  permanente; el robusto lo arregla. **Mejor candidato a "segunda velocidad" / alerta
  temprana** [D7; 99 §1.2–1.4, §1.6].

- **D8 `hmm_tstudent_4s`** · F3 HMM avanzado · avanzado · OOS 2012–2026 · NO vio 2008 OOS.
  HMM t-Student multivariante, **K=4 por BIC** (calma·corrección leve·corrección·crisis),
  ν por estado [10.2, 7.6, 4.2, 2.4]. **Gana el eje BIC (24416 vs D4 35379, ΔBIC +10963)**.
  Su "crisis" es la cola extrema (cov_COVID estricta 0.66; corrección+crisis 0.96). NO
  desbloquea 2013 (punto ciego de FEATURES, no distribucional). Coste ~5 min. **Núcleo
  recomendado, con asteriscos (ver §5)** [D8; 99 §1.5, §3, §4].

- **D9 `jump_model`** · F2↔F3 jump model · avanzado · OOS 2015–2026 · NO vio 2008.
  Statistical Jump Model (λ=50): histéresis "aprendida". **Anti-flickering rotundo**
  (switching 0.005 vs D3 0.126, ×24 menos; dur 177 d) pero **pierde cobertura de crisis
  lentas** (Inflación 2022 solo 17%). Especificidad perfecta en estricta y estrés. λ es el
  mando del trade-off persistencia↔sensibilidad [D9; 99 §1.2–1.3].

- **D10 `turbulence_mahalanobis`** · F1 multivariante · avanzado · OOS 1998–2026 · vio 2008
  OOS. Mahalanobis con covarianza expanding causal sobre [SP500_ret, VIX_chg, DXY_chg,
  slope_chg]. Capta sistémicas (GFC 82%, COVID 76%) pero **NO 2013 (12%)**: el taper no fue
  turbulencia conjunta. Flickea (switching 0.087, dur 11.4 d). **Señal de estrés sistémico
  multivariante barata; no tapa 2013** [D10].

- **D11 `msgarch_regime`** · F5 MS-GARCH · **EXPLORATORIO-NEGATIVO** · OOS 1991–2026 · vio
  2008 OOS. MS-GARCH(1,1)-t de Haas-Mittnik-Paolella, implementado desde cero (sin R).
  **RESULTADO NEGATIVO EXPLÍCITO**: causal e implementable, pero **degenera en walk-forward**
  — el fold de la GFC colapsa a 1 régimen → **cov_GFC 0%**, far **0.95**, fa_2018 93.7%. Se
  mantiene en el master como evidencia de la patología. **D6 cubre el hueco** [D11; 99 §2e].

- **D12 `deep_ae_regime`** · F7 redes · **EXPLORATORIO-NEGATIVO** · OOS 2015–2026 · NO vio
  2008. Autoencoder denso → GMM K=3 sobre el latente, vs baseline PCA→GMM. **RESULTADO
  NEGATIVO EXPLÍCITO**: el AE **empeora** al PCA (switching 0.287 vs 0.091, far 0.60 vs
  0.14) **sin ganar cobertura** (COVID 0.54 vs 0.62). La no-linealidad no aporta con ~4
  crisis; un reductor lineal (PCA) es preferible [D12; 99 §2e].

---

## 4. Reparto del podio por eje
> Fuente: `[99 §1]` (qué familia gana en qué eje). Cuatro familias se reparten seis ejes;
> **no existe detector dominante**.

| Eje | Familia ganadora | Detector(es) | Dato clave |
|---|---|---|---|
| **Cobertura sistémica — ventana larga** (vio 2008 OOS) | F4 Markov-Switching | D5 msvar (D6, D1 cerca) | D5 **0.98** > D6 **0.97** > D7 0.92 ≈ D1 0.92 > D10 0.79 > D11 0.10 (degenerado). Media GFC+COVID [99 §1.1] |
| **Cobertura sistémica — ventana corta** (NO vio 2008) | F2 / F3 baseline | D3 gmm = D4 hmm | D3 **0.96** = D4 **0.96** > D2 0.84 > D9 0.72 > D8 0.66 > D12 0.54 [99 §1.1] |
| **Especificidad** (no disparar en trampas 2013/2018) | F6 change-point | D7 cusum | Estricta: D7 1.00, D3 1.00, D9 1.00. Estrés: D7 1.00, D9 1.00, D1 0.97. D7 único 1.00 en ambas [99 §1.2] |
| **Persistencia / anti-flicker** | F6 change-point | D7 cusum | Persistencia: D7 **436 d** > D9 177 > D1 75. Anti-flicker (switching): D7 **0.002** < D9 0.005 < D1 0.013 [99 §1.3] |
| **Lead/lag** (anticipación al suelo) | F6 change-point | D7 cusum | Eventos comunes (COVID+Infl): D2 −235 d > D7 −228 (Infl. censurada) ≈ D5 −190 / D4 −185. D7 anticipa sostenido y temprano en TODOS [99 §1.4] |
| **Ajuste BIC** (in-sample, solo generativos) | F3 HMM t-Student | D8 hmm-t | D8 **24416** < D6 26627 < D11 26823; comparación legítima D4↔D8 (mismas features): ΔBIC **+10963** [99 §1.5] |
| **Coste computacional** | F6 change-point / F1 reglas | D7 cusum / D1 vix | bajo: D1, D2, D7, D10; medio: D3, D4, D6, D9, D12; alto: D5 (~33 min), D8 (~5 min), D11 [99 §1.6] |

**Lectura sistémica (parsimonia, §1.1).** Para cubrir crisis grandes y lentas basta con
cualquier detector que reaccione a la **volatilidad/varianza** del S&P 500 con suficiente
histórico: MS de varianza, GARCH y una regla de umbral sobre VIX dan todos cobertura
≥0.92 de GFC+COVID. La sofisticación de D5/D6 apenas bate a la regla D1 (+6 pp) [99 §1.1].

---

## 5. Recomendación para la siguiente capa
> Fuente: `[99 §4]`. **Verbo obligatorio: "consistente con / no contradice", NUNCA "confirma".**

**Veredicto.** La evidencia es **CONSISTENTE CON** el núcleo de la propuesta del TFM mayor
y **NO lo contradice**; sugiere complementarlo [99 §4]. (Se evita "confirma": el respaldo a
D8 descansa en un eje **in-sample** —BIC— y en una métrica **favorable-por-construcción** a
K≥3 —el estrés agregado—, NO en cobertura OOS estricta, donde D8 es modesto.)

1. **Núcleo HMM t-Student multi-estado (D8) sale reforzado**, por dos vías, ambas con su
   asterisco de honestidad [99 §4.1, pto 1]:
   - **Por BIC (in-sample)**: gana el eje de ajuste con holgura (ΔBIC +10963 sobre D4 con
     mismas features). *Pero el BIC es bondad de ajuste in-sample, no capacidad predictiva OOS.*
   - **Por la lógica corrección↔crisis**: con **estrés agregado** D8 iguala a los binarios
     (COVID 0.66→0.96, Inflación 0.33→0.90), lo que da sentido al cuarto estado y a la
     "segunda velocidad". *Pero el estrés agregado favorece por construcción a los
     multi-estado, y la cobertura OOS estricta es modesta (cov_COVID 0.66); D8 nunca vio
     2008 OOS.* Argumento de **plausibilidad estructural**, no de superioridad OOS demostrada.

2. **Complementar con un change-point rápido y barato (D7)** como **segunda velocidad /
   alerta temprana**: domina 4 ejes y anticipa de forma sostenida (días topados por el
   `lookback`, no cifra precisa). D7 = reactividad (cuándo cambia el nivel); HMM-t =
   taxonomía (qué tipo de régimen) [99 §4 pto 2].

3. **Mantener baselines como control**: D1 (regla VIX) y D5/D6 (MS-VAR / GARCH-t), mejor
   cobertura sistémica en ventana larga; referencia honesta [99 §4 pto 3].

4. **Descartar D11 y D12** como detectores operativos: validan la parsimonia, no la aportan
   [99 §4; IDX veredicto 5].

**Nota crítica que el redactor DEBE preservar (§4 cabecera).** El caso de D8 descansa en
**BIC in-sample + estrés agregado** (favorable-por-construcción a K≥3), **no** en cobertura
OOS estricta. Donde se mide OOS estricta, D8 es modesto (cov_COVID 0.66) y nunca evaluó 2008
OOS [99 §4; IDX veredicto 2]. Tratar 2013 como límite conocido de la taxonomía, no como
objetivo a forzar [99 §4].

---

## 6. Cautelas obligatorias (el redactor las DEBE preservar)
> Fuente: `[99]` cabecera, §1.4, §3, §4.1 y la salvedad del rank_heatmap (§1).

(i) **Lead/lag censurado al lookback.** El lead/lag se busca en los **252 días previos** al
suelo; un valor de **−252 NO significa "252 días exactos de anticipación"**, sino que la
señal de crisis sostenida ya estaba activa al inicio de la ventana → el lead real es **≥252 y
queda sin medir** (censura por la derecha). D7 toca ese tope en **3 de sus 4 eventos** (GFC,
EuroDebt, Inflación = −252), así que su media −240 d es un **límite inferior**, no una cifra
precisa [99 §1.4]. La virtud de D7 es **cualitativa** (cruza pronto y se mantiene), no "240
días" literales.

(ii) **logL / AIC / BIC son in-sample por definición.** Las métricas de comportamiento
(cobertura, falsas alarmas, lead/lag, persistencia, estabilidad) son OOS sin look-ahead;
pero las de **bondad de ajuste (logL/AIC/BIC) son in-sample** (verosimilitud del modelo en
sus datos). El eje de BIC (§1.5, §2c, §4) **NO es una métrica causal OOS**: mide
especificación, no capacidad predictiva fuera de muestra [99 cabecera; IDX veredicto 2].

(iii) **Limitación de significancia (n ≈ 4 crisis).** Toda la comparación descansa en ~4
crisis sistémicas (GFC 2008, EuroDebt 2011, COVID 2020, Inflación 2022) + 2 ventanas-trampa
(2013, 2018). **No hay tests de significancia ni intervalos de confianza**; diferencias de
pocos puntos NO son estadísticamente distinguibles. El podio es **evidencia cualitativa
direccional**, no una ordenación con respaldo inferencial. Es una **limitación asumida del
diseño** (las crisis son raras por definición), no un descuido. Trabajo futuro declarado:
**(B1) bootstrap por bloques** (stationary/circular block bootstrap) para bandas de confianza
y contraste de pares respetando la autocorrelación; **(B2) análisis de sensibilidad de
hiperparámetros** del walk-forward (`train_size`, `step`) y del etiquetado económico
(`VOL_CLOSE_FRAC`) [99 §4.1].

(iv) **Salvedad del `rank_heatmap`.** En `results/fase4_rank_heatmap.png`, la columna
*"Cob. sistémica"* rankea a los 12 detectores **juntos sin separar por ventana**, asignando
rango alto a D3/D4 (ventana corta, solo COVID) junto a D5/D6 (ventana larga, GFC+COVID) —
justo la mezcla que el resto del documento prohíbe. Esa columna **debe leerse solo como
vista panorámica**; la cobertura legítima es la de §1.1, separada por grupo de ventana. Las
otras columnas del heatmap (especificidad, persistencia, coste, BIC) sí son comparables [99 §1].

(v) **2013 está N/A (fuera de OOS) para los detectores de ventana corta.** Los de ventana
corta (2012+/2015+: D2, D3, D9, D12) tienen 2013 **fuera** de su OOS → N/A, no cuentan ni a
favor ni en contra. El "punto ciego universal" de 2013 se sostiene sobre los **6 detectores
reales** que SÍ tuvieron 2013 OOS (D1, D5, D6, D7, D8, D10) [99 §2d]. Análogamente, la GFC
2008 cae cerca del inicio de datos para las ventanas cortas → **D8 nunca evaluó 2008 OOS**;
su validación sistémica se apoya en COVID 2020 [99 §4 limitaciones].

---

## 7. Apéndice de trazabilidad — las 7 familias y los 12 detectores
> Fuente: `[SOTA §2, §5]`. Útil para la sección de método del PDF.

**Las 7 familias (F1–F7)** [SOTA §2–3]:
- **F1 Reglas / Umbrales** — régimen por umbral sobre observables (VIX, drawdown, spread,
  curva); reactivo, robusto a colas; talón = flickering (mitigable con histéresis/dwell).
- **F2 Clustering estático** — particiona el espacio de features sin matriz de transición
  (GMM); baseline NO temporal; flickering severo; no causal nativo.
- **F3 HMM (emisiones latentes)** — cadena de Markov latente + emisiones continuas;
  persistencia explícita; gaussiano subestima colas → t-Student/GMM-HMM; Viterbi/suavizado
  anti-causales → filtrado forward.
- **F4 Markov-Switching econométrico** — parámetros que conmutan con estado markoviano
  (Hamilton); univariante interpretable; probabilidades filtradas causales nativas.
- **F5 Volatilidad / GARCH / RS-GARCH** — varianza condicional; causal por construcción,
  reacción same-day; GARCH-t para colas; RS-GARCH caro y frágil.
- **F6 Change-point** — detecta instantes de cambio estructural (segmenta, no etiqueta
  estados recurrentes); CUSUM/BOCPD online causal; familia natural de la métrica lead/lag.
- **F7 Redes neuronales / no supervisado** — autoencoder + clustering latente; no asume
  normalidad pero exige datos abundantes que aquí no hay → overfitting; solo representante
  exploratorio con resultado negativo aceptable.

**Mapa detector → familia** [99 tabla; SOTA §5]:
| ID | Detector | Familia | Clase |
|---|---|---|---|
| D1 | rule_vix_threshold | F1 reglas | baseline |
| D2 | rule_composite_riskoff | F1 reglas | baseline |
| D3 | clustering_gmm_k3 | F2 clustering | baseline |
| D4 | hmm_gaussian_2s | F3 HMM | baseline (puente) |
| D5 | markov_switching_var_2s | F4 Markov-Switching | avanzado |
| D6 | garch_t_vol | F5 GARCH | avanzado |
| D7 | changepoint_online | F6 change-point | avanzado |
| D8 | hmm_tstudent_4s | F3 HMM (avanzado) | avanzado |
| D9 | jump_model | F2↔F3 jump model | avanzado |
| D10 | turbulence_mahalanobis | F1 multivariante | avanzado |
| D11 | msgarch_regime | F5 MS-GARCH | exploratorio-negativo |
| D12 | deep_ae_regime | F7 redes | exploratorio-negativo |

**Datos y EDA (contexto de método)** [EDA]:
- Ventana común **2007-04-11 → 2026-06-12** (4 737 obs sin NaN), gobernada por HYG;
  features causales **2007-07-06 → 2026-06-12** (4 665 obs, **15 features**) [EDA §3, §5].
- **Política SIN imputar**; causalidad verificada (`max_abs_diff = 0.0` en las 15 features) [EDA §2, §5].
- **Fat tails**: curtosis exceso SP500 25.6, HYG 39.6 → motiva t-Student [EDA §4].
- **DRAWDOWN_TROUGHS** reales: GFC 2009-03-09 (−56.8%), EuroDebt 2011-10-03 (−29.8%),
  COVID 2020-03-23 (−33.9%), Inflación 2022-10-12 (−25.4%) [EDA §6].
- Tensión declarada: la GFC 2008 cae cerca del inicio de datos → tensiona el walk-forward
  (2008 difícil de evaluar OOS sin histórico más largo) [EDA §3].

**Bibliografía** (claves reales en `docs/references.bib`): bloque completo en `[99 §5]`
(HMM `hmm_rabiner1989`/`hmm_bulla2011`/`hmm_nystrup2015`; Markov-Switching `hamilton1989`/
`ms_kim1994`; GARCH `vol_bollerslev1986`/`vol_glosten1993`; MS-GARCH
`vol_haasmittnikpaolella2004`/`vol_ardia2019`; change-point `cp_page1954`/`cp_truong2020`;
jump model `hmm_nystrup2020`; turbulencia `kritzman2012`; reglas `reglas_bloom2009`;
clustering/BIC `clust_schwarz1978bic`; redes `nn_hinton2006`; metodología `lopezdeprado2018`).
