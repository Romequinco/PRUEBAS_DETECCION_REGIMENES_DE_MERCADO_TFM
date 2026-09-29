# D11 — `msgarch_regime` (FASE 3, Tanda 4 · exploratoria) · Familia F5 (RS-GARCH)

<!-- BEGIN ubicacion_v2 -->
> **Ubicación.** Código: `src/regimenes/detectores/f5_garch/msgarch_regime.py` (registro: `regimenes.detectores.registry`, id `D11`) · Notebook v2 de familia: [`notebooks/09_familia_F5_garch.ipynb`](../../notebooks/09_familia_F5_garch.ipynb) · Resultados v2: sección [Resultados v2 (ADR-003)](#resultados-v2-adr-003) al final. Esta ficha nació en la Capa 1 (v1, congelada en el tag `capa1-final`); el texto histórico se conserva. Cuando menciona el núcleo (`src/`, `detector_base.py`, `evaluation.py`, `features.py`, `data_loader.py`, `INDEX.md`) se refiere al de la Capa 1: hoy `detector_base.py` → `src/regimenes/detectores/base.py`, `evaluation.py` → `regimenes.evaluacion`, `INDEX.md` → `docs/historia/capa1/memoria/INDEX.md`; `features.py`/`data_loader.py` de v1 quedan en el tag `capa1-final`.
<!-- END ubicacion_v2 -->

> MS-GARCH(1,1)-t de Haas-Mittnik-Paolella (2004), **implementado desde cero en
> numpy/scipy** (filtro de Hamilton propio), SIN R ni rpy2. **RESULTADO NEGATIVO
> documentado**: implementable y causal, pero su asignación de regímenes degenera en
> walk-forward y NO sirve como detector. Código: `src/regimenes/detectores/f5_garch/msgarch_regime.py` ·
> Notebook: `capa1_exploracion/notebooks/11_msgarch_regime.ipynb` (v1, tag `capa1-final`) · Métricas:
> `docs/historia/capa1/resultados/metrics_11_msgarch_regime.csv`.

## Decisión de implementar (no declararlo fuera)

El CP2 avisó: "no hay librería madura de MS-GARCH en Python". Eso aplica al MS-GARCH
*naive* path-dependiente (verosimilitud sobre 2^t trayectorias, intratable). Pero la
variante **HMP-2004 SÍ es tratable en Python puro**: K recursiones GARCH en PARALELO
(cada una alimentada solo por su propia varianza pasada → sin path dependence),
verosimilitud por filtro de Hamilton en O(T·K²). El subagente la implementó como modelo
genuino (no un apaño), lo cual era lo honesto. Bibliografía:
`vol_haasmittnikpaolella2004, vol_gray1996, vol_marcucci2005`.

## Implementado

**Modelo.** 2 regímenes, cada uno GARCH(1,1)-t (retornos×100); μ única, ν compartido por
parsimonia. Estimación ML (scipy) con filtro de Hamilton. **Walk-forward**: rolling 6
años, `step=126` (semestral), `n_init=1, maxiter=100` para acotar el coste del ML no
convexo (~13-36 min según contención). Dentro de cada bloque el posterior de crisis
filtrado es **diario y causal** (verificado: `max|P(crisis) ver vs ocultar futuro| =
0.0`). market_returns a walk_forward Y evaluate.

**In-sample (n_init=3, maxiter=300) el modelo SÍ es sensato**: régimen calma α+β=0.997
(ω=0.002), régimen crisis α=0.21/ω=0.042 (más reactivo), ν=5.29 (colas gordas),
transiciones persistentes (p00=0.996, p11=0.995). Orientación in-sample correcta
(crisis = alta vol).

## Descubierto — PATOLOGÍA en walk-forward (resultado negativo)

**Las métricas OOS son aberrantes para un modelo de volatilidad:**

| | valor | esperado en un vol-model |
|---|---:|---|
| cov_GFC_2008 | **0.0 %** | debería ser ~100% (mayor vol de la muestra) |
| cov_COVID_2020 | 20 % | alto |
| cov_Inflation_2022 | 36 % | medio-alto |
| fa_Selloff_Q4_2018 (trampa) | **93.7 %** | bajo |
| false_alarm_rate | **0.949** | bajo-medio |

**Diagnóstico (causa raíz, no es inversión del núcleo).** En el fold rolling de 6 años
que predice la GFC (sept-2008), el modelo decodifica **100 % régimen 0** en todo el
train: **nunca visita el régimen de crisis**. El GARCH casi-integrado del régimen
dominante (α+β=0.997) absorbe toda la volatilidad, dejando el 2º régimen **muerto**.
Como el régimen crisis no se observa, el núcleo (vol-primario, Arreglo 4) lo deja como
estado canónico 1 sin soporte → durante la GFC la "crisis" se dispara 0 %. En otros
folds el régimen 1 SÍ se activa, pero en días no-crisis (de ahí far 0.95 y fa_2018
0.94). Es decir: **asignación de regímenes degenerada e inestable entre folds**, típica
del ML no convexo del MS-GARCH con poca multistart, agravada por regímenes casi-IGARCH.

**No es un fallo del núcleo ni una inversión simple.** El Arreglo 4 orientó bien lo que
había; el problema es que el MODELO no entrega dos regímenes estables en walk-forward
con `n_init=1`. Subir a `n_init=3, maxiter=300` por fold (como el in-sample) podría
estabilizarlo, pero multiplicaría un coste ya alto (~36 min) por ~3 → inviable y, sobre
todo, **innecesario**: el subagente ya anticipó que si D11 rendía mal, **D6
`garch_t_vol` cubre limpiamente el hueco** de la señal de vol con colas (GFC 100 %,
COVID 94 %) sin la fragilidad del cambio de régimen.

## Hipótesis del CHECKPOINT 2 para D11 — veredicto

> *"Extensión natural (heteroscedasticidad GARCH dentro de cada régimen + prob.
> filtrada); coste alto y fragilidad (path dependence, óptimos locales,
> label-switching); sin librería madura en Python."*

**Se cumple la parte de la FRAGILIDAD, que era el riesgo declarado.** Es implementable
y causal (mérito), pero los óptimos locales y la degeneración de regímenes hacen que su
crisis OOS NO rastree las crisis reales (GFC 0 %). Como detector **no es usable**; como
ejercicio, confirma empíricamente la advertencia del estado del arte. **Se mantiene en
el master como resultado negativo documentado** (no se excluye): su fila es la evidencia
de la patología.

## Fricción con el núcleo

Ninguna en `src/`. El núcleo orientó correctamente los estados observados; la
degeneración es del modelo MS-GARCH, no del marco. Observación transversal útil para la
síntesis: detectores con **estados que pueden quedar sin visitar** en un fold producen
un `crisis_state` canónico "vacío" → cobertura 0 espuria; conviene vigilar en FASE 4
los casos de regímenes degenerados (frac de un estado ≈ 0).

<!-- BEGIN resultados_v2 (generado desde git HEAD; no editar a mano) -->
## Resultados v2 (ADR-003)

> Fuente: `results/benchmark/ranking_v2.csv` y `results/benchmark/metrics_master_v2.csv` del benchmark completo (24/24; `results/benchmark/manifest.json`, generado el 2026-09-29T12:41 UTC), con las cifras de ADR-003 §4. Walk-forward causal, evaluación OOS por pista. Ranking de **detección**: evento detectado = ≥ 3 sesiones OOS consecutivas en crisis dentro de [pico, suelo]; `score_deteccion` = F1 entre precisión diaria (1 − FAR) y recall por evento, con niveles previos (elegible › precisión ≤ azar › parpadeo › degenerado). Ver `docs/decisions/ADR-003-causalidad-calendario-estado-ranking.md`.

**Notebook de familia:** [`notebooks/09_familia_F5_garch.ipynb`](../../notebooks/09_familia_F5_garch.ipynb) (F5 — GARCH) · **Teoría:** [`docs/teoria/F5_volatilidad_garch.md`](../teoria/F5_volatilidad_garch.md).

| Métrica | Pista A | Pista B |
|---|---:|---:|
| Variante evaluada | `msgarch_regime` | `msgarch_regime` |
| Puesto de detección | **11** / 12 | **8** / 12 |
| Nivel de etiqueta | `elegible` | `elegible` |
| `score_deteccion` | 0.443 | 0.394 |
| Recall por evento | 0.765 (13/17) | 0.778 (7/9) |
| Precisión diaria (lift vs base) | 0.312 (×1.61) | 0.264 (×1.53) |
| Cobertura media de crisis | 0.437 | 0.392 |
| Tasa de falsas alarmas (FAR) | 0.688 | 0.736 |
| Activación media en trampas | 0.000 | 0.167 |
| Switching rate | 0.0173 | 0.0719 |
| Duración media de régimen (sesiones) | 57.7 | 13.9 |
| Estabilidad de etiqueta | 0.897 | 0.872 |
| Puesto legacy (`rank_medio`) | 12 (8.8) | 12 (9.6) |

**Configuración v2 y eventos por pista**

- **Pista A** · OOS 1970-05-12 → 2026-05-29 (n = 14133) · train inicial 2016 sesiones, refit cada 126 (rolling) · 1 feature(s): `SP500_ret`.
  - Detectados (13): `oil_stagflation_1973`, `volcker_1980_82`, `black_monday_1987`, `gulf_war_sl_1990`, `ltcm_russia_1998`, `dotcom_2000_02`, `gfc_2007_09`, `us_downgrade_euro2_2011`, `china_oil_2015_16`, `volmageddon_2018q1`, `fed_tightening_2018q4`, `covid_2020`, `tariff_selloff_2025`.
  - No detectados (4): `bear_1969_70`, `flash_crash_euro1_2010`, `inflation_bear_2022`, `svb_banking_2023`.
- **Pista B** · OOS 2010-04-12 → 2026-05-29 (n = 4059) · train inicial 756 sesiones, refit cada 126 (rolling) · 1 feature(s): `SP500_ret`.
  - Detectados (7): `flash_crash_euro1_2010`, `us_downgrade_euro2_2011`, `china_oil_2015_16`, `volmageddon_2018q1`, `covid_2020`, `inflation_bear_2022`, `tariff_selloff_2025`.
  - No detectados (2): `fed_tightening_2018q4`, `svb_banking_2023`.

Lectura: las cifras de la Capa 1 de las secciones anteriores (ventana 2007-2026, 4 crisis, métricas de cobertura sin criterio de detección) **no son comparables** con esta tabla (pistas A y B de `configs/benchmark_spec.yaml`, 17 crisis evaluables OOS en A / 9 crisis evaluables OOS en B, ranking ADR-003). El análisis gráfico por crisis y la comparación intra-familia están en el notebook `notebooks/09_familia_F5_garch.ipynb`.
<!-- END resultados_v2 -->
