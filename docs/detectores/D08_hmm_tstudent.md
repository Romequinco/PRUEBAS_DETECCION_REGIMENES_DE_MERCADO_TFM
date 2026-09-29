# D8 — `hmm_tstudent` (FASE 3, Tanda 3) · Familia F3 (HMM avanzado)

<!-- BEGIN ubicacion_v2 -->
> **Ubicación tras ADR-004.** Código: `src/regimenes/detectores/f3_hmm/hmm_tstudent.py` (registro: `regimenes.detectores.registry`, id `D08`) · Notebook v2 de familia: [`notebooks/07_familia_F3_hmm.ipynb`](../../notebooks/07_familia_F3_hmm.ipynb) · Resultados v2: sección [Resultados v2 (ADR-003)](#resultados-v2-adr-003) al final. Esta ficha nació en la Capa 1 (v1, congelada en el tag `capa1-final`); el texto histórico se conserva. Cuando menciona el núcleo (`src/`, `detector_base.py`, `evaluation.py`, `features.py`, `data_loader.py`, `INDEX.md`) se refiere al de la Capa 1: hoy `detector_base.py` → `src/regimenes/detectores/base.py`, `evaluation.py` → `regimenes.evaluacion`, `INDEX.md` → `docs/historia/capa1/memoria/INDEX.md`; `features.py`/`data_loader.py` de v1 quedan en el tag `capa1-final`.
<!-- END ubicacion_v2 -->

> HMM con **emisiones t-Student multivariantes** (colas pesadas) y **K estados por
> BIC**. La mejora directa sobre el baseline gaussiano D4: ataca de frente las fat
> tails del EDA (kurtosis 25-40). Código: `src/regimenes/detectores/f3_hmm/hmm_tstudent.py` (+
> `src/regimenes/detectores/f3_hmm/_hmm_t_utils.py`, filtrado forward t) · Notebook:
> `capa1_exploracion/notebooks/08_hmm_tstudent.ipynb` (v1) · Métricas: `docs/historia/capa1/resultados/metrics_08_hmm_tstudent.csv`.

## Implementado

**Modelo.** HMM con emisiones **t-Student multivariantes** (location mᵢ, matriz de
escala Sᵢ y grados de libertad **νᵢ** por estado), estimado por EM con variable de
escala latente (cada t-Student = mezcla gaussiana de escala continua). Se eligió la
t propia frente a GMM-HMM porque: (1) **BIC justo** — la t añade solo `k` parámetros
(un ν por estado) sobre el gaussiano, mientras un GMM multiplicaría medias/covarianzas
por nº de mezclas y dispararía el BIC; (2) robustez sin sobre-parametrizar.
Bibliografía: `hmm_bulla2011, hmm_rabiner1989, guidolintimmermann2007, hamilton1989`.

**Causalidad — filtrado forward t.** `predict_online`/`predict_proba` usan filtrado
forward causal con la **emisión t** (en `src/regimenes/detectores/f3_hmm/_hmm_t_utils.py`, adaptando el
patrón de D4 `_hmm_utils` cuya emisión era gaussiana), con contexto de burn-in del
train. Viterbi solo en la versión in-sample marcada NO causal. **No se modificó
`_hmm_utils.py` ni `src/`.**

**Features/ventana = D4 (BIC comparable).** Las mismas 7 features puente
(`BRIDGE_FEATURES`), ventana 2007+, `train_size=252*5` → como D4, 2008/2011 caen en el
train (NaN OOS). Así el BIC de D8 (t) es directamente comparable con el de D4
(gaussiano) sobre los mismos datos. market_returns = retorno log S&P 500.

**Coste.** El EM-t es caro (Baum-Welch + actualización de ν por fold). `step=126`
(refit ~semestral), declarado; el walk-forward tarda ~5 min. K seleccionado fuera del
walk-forward (fits in-sample para BIC).

## Descubierto

**K=4 por BIC (y AIC).** Grid {3,4}: BIC k=3 = 28103, **k=4 = 24416** → se despliega
**K=4** (calma · corrección leve · corrección · crisis). Los ν por estado canónico son
**decrecientes [10.2, 7.6, 4.2, 2.4]**: el estado de crisis tiene ν≈2.4 (colas muy
pesadas), el de calma ν≈10 (casi gaussiano). Económicamente impecable: la crisis es
donde viven los outliers.

**BIC vs D4 — la t-Student MEJORA el ajuste con holgura:**

| | n_states | logL | n_params | BIC |
|---|---:|---:|---:|---:|
| D4 gaussiano | 2 | −17381 | ~73 | **35379** |
| D8 t-Student | 4 | −11536 | 159 | **24416** |

**ΔBIC ≈ +10 963 a favor de D8** pese a tener más del doble de parámetros: el supuesto
gaussiano de D4 pagaba un coste enorme por no modelar las colas. Es la confirmación
cuantitativa de la crítica del EDA (kurtosis 25-40) y del estado del arte
(`hmm_bulla2011`).

**Orden de estados MONÓTONO en severidad** (no solo "no invertido"). Verificado
**en walk-forward** (con market_returns): vol anualizada por estado canónico OOS =
[10.1%, 13.8%, 21.0%, 37.1%] → estrictamente creciente 0→3; crisis (3) = mayor vol.
`monotonía vol walk-forward = True`, sin warning de fallback. Con K=4 el binning por
bandas de `VOL_CLOSE_FRAC` del Arreglo 4 ordena bien los 4 estados.

**Cobertura (OOS causal).** ventana 2012-2026 → 2008/2011 NaN (en train, como D4).
COVID_2020 = 66% en el estado CRISIS, Inflation_2022 = 33% en crisis. **Matiz
importante de multi-estado**: con K=4 el estado "crisis" es el más EXTREMO y estrecho;
el estrés más amplio cae en "corrección" (estado K−2). Sumando corrección+crisis,
Q4-2018 = 81%. Por eso `cov_COVID`/`cov_Inflation` de D8 parecen MÁS BAJAS que las de
D4 (2 estados, crisis ancha): **no es que D8 detecte peor, es que su "crisis" es la
cola extrema**. Hay que leer su fila del master con esta lente (la comparación justa de
"estrés" sería corrección+crisis).

**¿Captó 2013/2018?** Activación del estado crisis: 2013 = **0%**, 2018 = 3.4%
(corrección+crisis: 2013 = 0%, 2018 = 81%). Es decir, capta 2018 como corrección pero
**NO 2013**. La t-Student y los 4 estados mejoran el ajuste y separan corrección de
crisis, pero **2013 sigue siendo invisible**: el taper fue un shock de tipos sin
volatilidad de equity, y D8 usa solo features equity/crédito (las mismas que D4). Es
un punto ciego de las FEATURES, no del supuesto distribucional → lo debe tapar D10
(multivariante con tipos/dólar).

**Flickering.** switching 0.052, duración media 19.1 d, persistencia esperada por
estado [calma 31d, leve 18d, corrección 38d, crisis 83d]. Más persistente que el GMM
(D3, 0.126) y similar a D4 causal (0.100).

## Hipótesis del CHECKPOINT 2 para D8 — veredicto

> *"Emisiones t + más estados atacan fat tails y POTENCIALMENTE captan 2013/2018
> donde el gaussiano falla; riesgo de sobreajuste con pocas obs por estado."*

**Se cumple a medias, con un matiz nítido.** (1) Fat tails: **SÍ, rotundo** — ΔBIC
+10963 vs D4, ν decreciente hasta 2.4 en crisis. (2) Más estados: SÍ, separa
calma/corrección/crisis de forma monótona y económicamente coherente. (3)
**¿2013/2018? NO desbloquea 2013** (sigue invisible: es shock de tipos, no de vol
equity; límite de las features, no de la t); 2018 se capta como corrección. Conclusión:
la t-Student arregla el problema DISTRIBUCIONAL (colas) que tenía D4, pero el agujero
de 2013 es de COBERTURA DE FEATURES, no distribucional → no lo cierra D8. (4)
Sobreajuste: con 159 params y ~212 días en el estado crisis OOS hay menos soporte por
estado, pero el BIC (que penaliza params) sigue prefiriéndolo con holgura, así que el
ajuste extra está justificado.

## Fricción con el núcleo

Ninguna que obligue a tocar `src/`. El filtrado forward t se resolvió en el detector
(`_hmm_t_utils.py`) reusando el patrón de D4. El Arreglo 4 (vol-primario) ordenó
correctamente los 4 estados (monotonía verificada). Observación de lectura, no de
núcleo: para detectores multi-estado, `cov_<crisis>` mide solo el estado más extremo;
al comparar contra detectores de 2 estados conviene mirar también "corrección+crisis"
(lo hace el notebook), no solo la columna del master.

<!-- BEGIN resultados_v2 (generado desde git HEAD; no editar a mano) -->
## Resultados v2 (ADR-003)

> Fuente: `results/benchmark/ranking_v2.csv` y `results/benchmark/metrics_master_v2.csv` en el commit `1f95a9b` (benchmark completo 24/24, ADR-003). Walk-forward causal, evaluación OOS por pista. Ranking de **detección**: evento detectado = ≥ 3 sesiones OOS consecutivas en crisis dentro de [pico, suelo]; `score_deteccion` = F1 entre precisión diaria (1 − FAR) y recall por evento, con niveles previos (elegible › precisión ≤ azar › parpadeo › degenerado). Ver `docs/decisions/ADR-003-causalidad-calendario-estado-ranking.md`.

**Notebook de familia:** [`notebooks/07_familia_F3_hmm.ipynb`](../../notebooks/07_familia_F3_hmm.ipynb) (F3 — HMM) · **Teoría:** [`docs/teoria/F3_hmm.md`](../teoria/F3_hmm.md).

| Métrica | Pista A | Pista B |
|---|---:|---:|
| Variante evaluada | `hmm_tstudent_4s` | `hmm_tstudent_4s` |
| Puesto de detección | **9** / 12 | **9** / 12 |
| Nivel de etiqueta | `elegible` | `elegible` |
| `score_deteccion` | 0.467 | 0.274 |
| Recall por evento | 0.824 (14/17) | 0.333 (3/9) |
| Precisión diaria (lift vs base) | 0.326 (×1.68) | 0.233 (×1.35) |
| Cobertura media de crisis | 0.554 | 0.081 |
| Tasa de falsas alarmas (FAR) | 0.674 | 0.767 |
| Activación media en trampas | 0.000 | 0.000 |
| Switching rate | 0.0157 | 0.0177 |
| Duración media de régimen (sesiones) | 63.4 | 55.6 |
| Estabilidad de etiqueta | 0.773 | 0.664 |
| Puesto legacy (`rank_medio`) | 8 (7.6) | 10 (9.0) |

**Configuración v2 y eventos por pista**

- **Pista A** · OOS 1970-05-12 → 2026-05-29 (n = 14133) · train inicial 2016 sesiones, refit cada 126 (expanding) · 9 feature(s): `SP500_ret_z|SP500_vol_z|SP500_momentum|SP500_drawdown|FF_MKT_z|DGS10_change_z|credit_BaaAaa_mensual_z|term_spread_hist_z|INDPRO_yoy_z`.
  - Detectados (14): `oil_stagflation_1973`, `volcker_1980_82`, `black_monday_1987`, `gulf_war_sl_1990`, `ltcm_russia_1998`, `dotcom_2000_02`, `gfc_2007_09`, `flash_crash_euro1_2010`, `us_downgrade_euro2_2011`, `china_oil_2015_16`, `covid_2020`, `inflation_bear_2022`, `svb_banking_2023`, `tariff_selloff_2025`.
  - No detectados (3): `bear_1969_70`, `volmageddon_2018q1`, `fed_tightening_2018q4`.
- **Pista B** · OOS 2010-04-12 → 2026-05-29 (n = 4059) · train inicial 756 sesiones, refit cada 126 (expanding) · 14 feature(s): `SP500_ret_z|SP500_vol_z|SP500_momentum|SP500_drawdown|FF_MKT_z|DGS10_change_z|credit_BaaAaa_mensual_z|term_spread_hist_z|INDPRO_yoy_z|VIX_level_z|MOVE_level_z|credit_BAA10Y_z|DFII10_change_z|corr_spx_bond`.
  - Detectados (3): `china_oil_2015_16`, `covid_2020`, `inflation_bear_2022`.
  - No detectados (6): `flash_crash_euro1_2010`, `us_downgrade_euro2_2011`, `volmageddon_2018q1`, `fed_tightening_2018q4`, `svb_banking_2023`, `tariff_selloff_2025`.

Lectura: las cifras de la Capa 1 de las secciones anteriores (ventana 2007-2026, 4 crisis, métricas de cobertura sin criterio de detección) **no son comparables** con esta tabla (pistas A y B de `configs/benchmark_spec.yaml`, 17 crisis evaluables OOS en A / 9 crisis evaluables OOS en B, ranking ADR-003). El análisis gráfico por crisis y la comparación intra-familia están en el notebook `notebooks/07_familia_F3_hmm.ipynb`.
<!-- END resultados_v2 -->
