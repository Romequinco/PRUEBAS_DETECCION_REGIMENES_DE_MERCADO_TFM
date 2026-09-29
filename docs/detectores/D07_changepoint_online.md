# D7 — `changepoint_online` (FASE 3, Tanda 3) · Familia F6 (Change-point)

<!-- BEGIN ubicacion_v2 -->
> **Ubicación tras ADR-004.** Código: `src/regimenes/detectores/f6_changepoint/changepoint_online.py` (registro: `regimenes.detectores.registry`, id `D07`) · Notebook v2 de familia: [`notebooks/10_familia_F6_changepoint.ipynb`](../../notebooks/10_familia_F6_changepoint.ipynb) · Resultados v2: sección [Resultados v2 (ADR-003)](#resultados-v2-adr-003) al final. Esta ficha nació en la Capa 1 (v1, congelada en el tag `capa1-final`); el texto histórico se conserva. Cuando menciona el núcleo (`src/`, `detector_base.py`, `evaluation.py`, `features.py`, `data_loader.py`, `INDEX.md`) se refiere al de la Capa 1: hoy `detector_base.py` → `src/regimenes/detectores/base.py`, `evaluation.py` → `regimenes.evaluacion`, `INDEX.md` → `docs/historia/capa1/memoria/INDEX.md`; `features.py`/`data_loader.py` de v1 quedan en el tag `capa1-final`.
<!-- END ubicacion_v2 -->

> Detector de **cambio estructural ONLINE y CAUSAL** sobre el **retorno log del S&P 500**
> (desde 1985). Núcleo: un **CUSUM de Page (1954)** secuencial sobre un estadístico de
> volatilidad; un autómata de 2 CUSUM de una cara lo mapea a 2 estados (0=calma,
> 1=crisis). El umbral `h` da persistencia. 2 estados.
>
> Código: `src/regimenes/detectores/f6_changepoint/changepoint_online.py` · Notebook: `capa1_exploracion/notebooks/07_changepoint_online.ipynb` (v1)
> (constructor `_build_07.py` (retirado en el commit `6016f42`)) · Métricas:
> `docs/historia/capa1/resultados/metrics_07_changepoint_online.csv` · Figuras: `d07_*.png` (figura v1, no versionada).

## Implementado

**Método ONLINE causal (CUSUM de Page 1954).** La distinción clave de la familia F6:
la variante potente de `ruptures` (**PELT** — cp_killick2012; BinSeg — cp_truong2020) es
**OFFLINE y ANTI-causal** (segmenta mirando toda la serie → vería el futuro del bloque en
walk-forward). D7 usa el **CUSUM secuencial**, la variante online: acumula evidencia de un
cambio de nivel y dispara una alarma cuando el cumulativo supera `h`, usando solo datos
≤ t (detecta el cambio **con cierto retardo**, como toca a un método causal). PELT se usa
SOLO en el notebook (§3) como **oráculo in-sample**, explícitamente marcado **NO causal**.

**De change-point a 2 estados recurrentes (reto de integración).** Un change-point
*segmenta*, no etiqueta estados. Mapeo elegido: **autómata de 2 estados con dos CUSUM de
una cara** (Page):
- En CALMA (0) se vigila un cambio **al alza** de la vol con `C⁺ = max(0, C⁺ + z − k)`; si
  `C⁺ > h` → se entra en CRISIS (1) y se resetean los acumuladores.
- En CRISIS (1) se vigila un cambio **a la baja** con `C⁻ = max(0, C⁻ − z − k)`; si
  `C⁻ > h` → se vuelve a CALMA (0).
El umbral `h` da **persistencia por construcción** (hace falta evidencia acumulada para
conmutar) → poco flickering, sin dwell explícito. **QUÉ** tramo es "crisis" lo decide el
**NÚCLEO** vía `label_states_economically` (Arreglo 4, vol-primario): D7 solo separa por
nivel de vol; el núcleo pone la polaridad → crisis = el tramo de **mayor σ** de los
retornos. "Etiqueta de régimen por día = en tramo de alta vol tras un cambio al alza" = 1.

**Estadístico monitorizado y coste (gaussiano vs robusto — eje de CP2).**
- `cost='robust'` (DEFECTO): estadístico = **log|retorno|**, estandarizado con
  **MEDIANA / MAD** del train (escala robusta de Gauss, `1.4826·MAD`) y **winsorizado** a
  ±`clip`. log|r| es ~**simétrico** (log-normalidad de la vol), lo que permite que el
  CUSUM acumule en ambos sentidos y el autómata **conmute** correctamente; mediana/MAD +
  winsor **acotan la influencia de cada día de cola**.
- `cost='gaussian'`: estadístico = **retorno²** (coste cuadrático L2), estandarizado con
  **media / desv** del train. Fuertemente asimétrico y dominado por las colas (kurt 25–40).

  Defaults: `k=0.5` (holgura/slack), `h=5.0`, `clip=3.0`. `predict_proba` = **sigmoide
  monótona** del EWMA (halflife 10) del estadístico de vol estandarizado, centrada con su
  media/desv in-sample → ~0.5 en el nivel medio, ↑ con la vol reciente; reordenada al
  orden canónico (como D6/GMM). El detector NO es generativo → `score`/AIC/BIC = NaN.

**Causalidad en walk-forward (patrón burn-in, como D6).** El CUSUM es causal nativo (C±ₜ
depende solo de z con s ≤ t), pero el walk-forward usa bloques de `step=21` días con
detector NUEVO por fold; arrancar en frío en CALMA cada bloque perdería tramos de crisis
en curso. Se evita re-ejecutando el CUSUM sobre `[retornos de train anteriores al bloque]
+ bloque` con la base (centro/escala) **CONGELADA** del train, devolviendo solo la parte
del bloque. Como el CUSUM solo mira atrás, el estado al inicio del bloque refleja la
historia real → causal y continuo entre folds. **Verificado (notebook §2):** ocultar el
futuro (2008 dentro de 2008–2010) NO cambia ni un día del bloque (`ndiff=0`).

**Ventana (LARGA).** `returns = log(SP500/SP500.shift(1)).dropna()`, índice 1985-01 →
2026-06. Walk-forward expanding, `train_size = 252×8` (~8 años), `step = 21`.
**ventana_eval = 1993-03 → 2026-06 (n≈8278)** → **2008 y 2011 son OOS**, a diferencia de
D4 (atado a HYG desde 2007). step=21 va sobrado en coste (~45 s por walk-forward).

## Descubierto

**Orientación VERIFICADA en walk-forward (no solo in-sample).** Pasando `market_returns`
a `walk_forward` Y a `evaluate`, el estado canónico crisis = `crisis_state = 1` tiene
**mayor σ de retornos OOS** que calma (≈0.0158 vs ≈0.0082) → **NO invertido**. Con el
núcleo vol-primario (Arreglo 4) el riesgo de inversión de un detector que separa por vol
queda neutralizado. **Sin warning de fallback peligroso**: X contiene `SP500_ret`
(columna reconocida como retorno real), así que el etiquetado provisional dentro de `fit`
usa el retorno correcto; dentro de `walk_forward` el aviso se silencia y se re-fija el
orden con `market_returns`. (El único warning posible es el advisory benigno de
"recognized-column proxy", idéntico al de D6, no la variante peligrosa que invierte.)

**Cobertura por crisis (CAUSAL OOS, coste robusto):**

| Ventana | Cobertura OOS | Lectura |
|---|---:|---|
| GFC_2008 | **100 %** | OOS gracias al histórico largo; CUSUM entra y se mantiene |
| EuroDebt_2011 | **~67 %** | OOS, captada (con algún tramo de salida/reentrada) |
| COVID_2020 | **~84 %** | el crash dispara C⁺ rápido |
| Inflation_2022 | **~77 %** | bear market de tipos, vol sostenida |
| TaperTantrum_2013 (trampa) | **0 %** | el robusto NO se dispara (poca vol equity) |
| Selloff_Q4_2018 (trampa) | **0 %** | el robusto NO se dispara |

**Lead/lag SOSTENIDO (persist=3) vs los troughs — DETECCIÓN TEMPRANA confirmada.** El
`lead_lag` del núcleo (cruce de p_crisis≥0.5 mantenido ≥3 días) es **negativo en las 4
crisis** (la señal de crisis está activa **bien antes** del suelo de drawdown; en varias
ventanas alcanza el tope del `lookback`=252 → ≥1 año de antelación). Es exactamente la
"detección temprana" que predice CP2. Matiz honesto: parte de esa antelación es porque el
detector también marca como crisis la fase de subida de vol previa (alto recall, a costa
de precisión global — ver false_alarm_rate).

**Switching / flickering (bueno).** `switching_rate ≈ 0.002`, **duración media ≈ 430 días
global** (episodios de crisis de decenas–cientos de días), `label_stability = 1.0`. El
umbral `h` del CUSUM actúa como histéresis natural → **NO flickea** por outliers en el
coste robusto. Comparable en estabilidad a D6 (GARCH).

**Falsas alarmas por outliers Y si el coste robusto ayuda (núcleo de CP2).** Resultado
**nítido**:
- **Coste GAUSSIANO (retorno², L2): degenera.** La desv de `r²` está inflada por la
  kurtosis (25–40), de modo que la base no distingue calma de crisis: el autómata entra en
  crisis con el primer shock (1987) y **no consigue salir** → **alarma casi permanente**
  (frac_crisis≈0.93, switching≈0, **trampas 2013 y 2018 al 100 %**, false_alarm_rate≈0.94).
  Es la materialización del "riesgo de falsas alarmas con outliers" de CP2.
- **Coste ROBUSTO (log|r| + mediana/MAD + winsor): lo arregla.** Recupera regímenes
  limpios — **trampas 2013/2018 ≈ 0 %**, switching bajo, episodios largos — manteniendo
  cobertura plena de 2008/2020 y buena de 2011/2022. → **el coste robusto SÍ reduce las
  falsas alarmas**, como pide CP2 ("preferir kernel/robusto").

**false_alarm_rate ≈ 0.87 (alto, mismo matiz que D6).** Precio de la ventana larga + señal
de vol: marca crisis muchos picos de alta vol reales **fuera** de las 4 ventanas canónicas
(1987, LTCM 1998, dotcom 2000–02, flash-crash 2010, 2015–16, SVB 2023…). No es flickering
día-a-día (switching bajísimo) sino episodios de alta vol legítimos que el ground-truth
laxo de 4 ventanas cuenta como falsa alarma. Léase junto a la cobertura.

**Oráculo PELT (offline, NO causal).** `ruptures.Pelt(model='l2')` sobre la log-vol
localiza los change-points "ideales" con información perfecta; sirve para ver que las
conmutaciones ONLINE del CUSUM caen cerca pero **con retardo** (precio de la causalidad).
Solo comparación visual; NUNCA entra en la evaluación.

## Hipótesis del CHECKPOINT 2 para D7 — veredicto

> *"Detección **temprana** (lead/lag) pero **riesgo de falsas alarmas con outliers**;
> **preferir kernel/robusto** frente al CUSUM **gaussiano**."*

**Se CUMPLE.** ✓ **Online causal** (CUSUM de Page; causalidad verificada, PELT solo como
oráculo). ✓ **Detección temprana**: lead/lag sostenido **negativo en las 4 crisis**. ✓
**Falsas alarmas con outliers**: el coste **gaussiano** (r², L2) se deja arrastrar por las
colas y degenera en alarma permanente (trampas al 100 %, far≈0.94). ✓ **El coste robusto
las reduce**: log|r| + mediana/MAD + winsor → trampas ≈0 %, regímenes limpios con cobertura
intacta. Conclusión: para datos financieros de cola gorda, **el CUSUM gaussiano sobre la
varianza es inutilizable y el coste robusto es claramente preferible** — exactamente la
recomendación de CP2.

## Fricción con el núcleo

- **Ninguna que obligara a tocar `src/`.** El núcleo cubrió bien a D7: (1) el etiquetado
  **vol-primario (Arreglo 4)** ordena correctamente un detector que separa por vol, sin
  inversión y sin necesidad de override (igual que D6); (2) la métrica `lead_lag` con
  `persist=3` del núcleo es justo la que mide la "detección temprana SOSTENIDA" de CP2.
- **(Observación, no bloqueante)** `false_alarm_rate` con ground-truth laxo (4 ventanas)
  penaliza como falsa alarma toda alta vol fuera de ellas; para un detector de change-point
  de vol con histórico largo infla la métrica. Mismo comentario que D6: no es defecto del
  núcleo, pero conviene leerla junto a la cobertura, no aislada.
- **(Patrón reutilizado)** El burn-in causal por bloque (concatenar train previo + bloque
  con la base congelada) es el mismo patrón que D6 (GARCH). Útil que el contrato del núcleo
  (`predict_online(test)` con detector nuevo por fold) admita este patrón sin cambios.

<!-- BEGIN resultados_v2 (generado desde git HEAD; no editar a mano) -->
## Resultados v2 (ADR-003)

> Fuente: `results/benchmark/ranking_v2.csv` y `results/benchmark/metrics_master_v2.csv` en el commit `1f95a9b` (benchmark completo 24/24, ADR-003). Walk-forward causal, evaluación OOS por pista. Ranking de **detección**: evento detectado = ≥ 3 sesiones OOS consecutivas en crisis dentro de [pico, suelo]; `score_deteccion` = F1 entre precisión diaria (1 − FAR) y recall por evento, con niveles previos (elegible › precisión ≤ azar › parpadeo › degenerado). Ver `docs/decisions/ADR-003-causalidad-calendario-estado-ranking.md`.

**Notebook de familia:** [`notebooks/10_familia_F6_changepoint.ipynb`](../../notebooks/10_familia_F6_changepoint.ipynb) (F6 — Change-point) · **Teoría:** [`docs/teoria/F6_change_point.md`](../teoria/F6_change_point.md).

| Métrica | Pista A | Pista B |
|---|---:|---:|
| Variante evaluada | `changepoint_online` | `changepoint_online` |
| Puesto de detección | **8** / 12 | **7** / 12 |
| Nivel de etiqueta | `elegible` | `elegible` |
| `score_deteccion` | 0.469 | 0.426 |
| Recall por evento | 0.824 (14/17) | 0.556 (5/9) |
| Precisión diaria (lift vs base) | 0.328 (×1.69) | 0.345 (×2.00) |
| Cobertura media de crisis | 0.646 | 0.285 |
| Tasa de falsas alarmas (FAR) | 0.672 | 0.655 |
| Activación media en trampas | 0.000 | 0.000 |
| Switching rate | 0.0022 | 0.0030 |
| Duración media de régimen (sesiones) | 441.7 | 312.2 |
| Estabilidad de etiqueta | 1.000 | 0.995 |
| Puesto legacy (`rank_medio`) | 2 (3.8) | 5 (5.8) |

**Configuración v2 y eventos por pista**

- **Pista A** · OOS 1970-05-12 → 2026-05-29 (n = 14133) · train inicial 2016 sesiones, refit cada 21 (expanding) · 1 feature(s): `SP500_ret`.
  - Detectados (14): `bear_1969_70`, `oil_stagflation_1973`, `volcker_1980_82`, `black_monday_1987`, `gulf_war_sl_1990`, `ltcm_russia_1998`, `dotcom_2000_02`, `gfc_2007_09`, `flash_crash_euro1_2010`, `us_downgrade_euro2_2011`, `china_oil_2015_16`, `covid_2020`, `inflation_bear_2022`, `svb_banking_2023`.
  - No detectados (3): `volmageddon_2018q1`, `fed_tightening_2018q4`, `tariff_selloff_2025`.
- **Pista B** · OOS 2010-04-12 → 2026-05-29 (n = 4059) · train inicial 756 sesiones, refit cada 21 (expanding) · 1 feature(s): `SP500_ret`.
  - Detectados (5): `us_downgrade_euro2_2011`, `china_oil_2015_16`, `covid_2020`, `inflation_bear_2022`, `svb_banking_2023`.
  - No detectados (4): `flash_crash_euro1_2010`, `volmageddon_2018q1`, `fed_tightening_2018q4`, `tariff_selloff_2025`.

Lectura: las cifras de la Capa 1 de las secciones anteriores (ventana 2007-2026, 4 crisis, métricas de cobertura sin criterio de detección) **no son comparables** con esta tabla (pistas A y B de `configs/benchmark_spec.yaml`, 17 crisis evaluables OOS en A / 9 crisis evaluables OOS en B, ranking ADR-003). El análisis gráfico por crisis y la comparación intra-familia están en el notebook `notebooks/10_familia_F6_changepoint.ipynb`.
<!-- END resultados_v2 -->
