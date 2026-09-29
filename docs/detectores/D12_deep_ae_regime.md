# D12 — `deep_ae_regime` (FASE 3, Tanda 4 · exploratoria) · Familia F7 (redes neuronales)

<!-- BEGIN ubicacion_v2 -->
> **Ubicación tras ADR-004.** Código: `src/regimenes/detectores/f7_deep/deep_ae_regime.py` (registro: `regimenes.detectores.registry`, id `D12`) · Notebook v2 de familia: [`notebooks/11_familia_F7_deep.ipynb`](../../notebooks/11_familia_F7_deep.ipynb) · Resultados v2: sección [Resultados v2 (ADR-003)](#resultados-v2-adr-003) al final. Esta ficha nació en la Capa 1 (v1, congelada en el tag `capa1-final`); el texto histórico se conserva. Cuando menciona el núcleo (`src/`, `detector_base.py`, `evaluation.py`, `features.py`, `data_loader.py`, `INDEX.md`) se refiere al de la Capa 1: hoy `detector_base.py` → `src/regimenes/detectores/base.py`, `evaluation.py` → `regimenes.evaluacion`, `INDEX.md` → `docs/historia/capa1/memoria/INDEX.md`; `features.py`/`data_loader.py` de v1 quedan en el tag `capa1-final`.
<!-- END ubicacion_v2 -->

> Autoencoder ligero → GMM sobre el latente, como **contraste ablativo honesto** frente
> al baseline lineal PCA→GMM. Objetivo: ¿aporta la no linealidad del AE algo con ~4
> crisis? Código: `src/regimenes/detectores/f7_deep/deep_ae_regime.py` (`DeepAERegime` + `PCAGMMBaseline`) ·
> Notebook: `capa1_exploracion/notebooks/12_deep_ae_regime.ipynb` (v1, tag `capa1-final`) · Métricas:
> `docs/historia/capa1/resultados/metrics_12_deep_ae_regime.csv` (2 filas: AE y baseline PCA).

## Implementado

**Modelo.** Autoencoder denso PyTorch (15→8→2→8→15, ReLU + dropout 0.10 + weight-decay
1e-3, 40 épocas, Adam, semillas fijas) que comprime las 15 features causales a un
**latente 2D**; encima, `GaussianMixture` full con **K=3** (mismo esquema que D3 pero no
lineal). Estandarización **causal por fold** (μ/σ del train). Se expone también
`reconstruction_error()` como score de anomalía (complemento, no principal).
Bibliografía: `nn_kingma2014, nn_akioyamen2021, nn_bucci2021, lopezdeprado2018`.

**Baseline ablativo.** Clase hermana `PCAGMMBaseline` (PCA 2D → GMM K=3) en el mismo
fichero, pasa por el MISMO `walk_forward` → comparación limpia AE-no-lineal vs
PCA-lineal con idéntica dim latente, K y ventana.

**Causalidad (con nota — se corrigió una aserción).** El encoder es puntual (eval() +
no_grad + scaler congelado del train). La verificación inicial del notebook usaba
`assert maxdiff_latente < 1e-9`, que **fallaba por ruido de coma flotante de torch**
(BLAS sobre tensores de distinto tamaño en float32 da ~2.4e-7), NO por look-ahead. Se
corrigió la aserción al criterio REAL de causalidad: **0/247 estados del bloque cambian
al ocultar el futuro** (y maxdiff latente < 1e-4, tolerancia FP). Confirmado causal.
market_returns a walk_forward Y evaluate.

**Orientación (Arreglo 4).** Verificada en walk-forward: crisis (estado 2) vol 0.0275
vs calma 0.0070; PCA igual (crisis vol 0.0524 vs 0.0066). No invertido, sin fallback.

## Descubierto — RESULTADO NEGATIVO (esperado y válido)

**ventana_eval: 2015-09 → 2026-06 (n=2649)** (15 features de 2007 → 2008/2011 NaN OOS).

**Contraste ablativo AE→GMM vs PCA→GMM (mismo K, dim, ventana):**

| | cov COVID | cov Inflation | fa 2018 | false_alarm_rate | switching | dur |
|---|---:|---:|---:|---:|---:|---:|
| **D12 AE→GMM** | 0.54 | 0.10 | 0.15 | 0.60 | **0.287** | 3.5 d |
| PCA→GMM (lineal) | 0.62 | 0.005 | 0.00 | 0.14 | 0.091 | 10.9 d |

**El AE NO mejora al baseline lineal — al contrario, lo empeora.** La no linealidad
**añade flickering** (switching 0.287 vs 0.091, ~3×) **y falsas alarmas** (far 0.60 vs
0.14) **sin ganar cobertura** (COVID 0.54 vs 0.62). Es el resultado que anticipaba el
CHECKPOINT 2: con ~4 crisis reales no hay señal suficiente para que la capacidad extra
del AE generalice; solo ajusta ruido y produce un latente más inestable que la PCA.

**Valor del hallazgo (negativo pero útil).** La conclusión metodológica es limpia: en
este problema y con estos datos, **un reductor lineal (PCA) es preferible a un AE** como
front-end de clustering de regímenes. El deep learning solo se justificaría con muchos
más datos o features de alta frecuencia (intradía), como ya señaló el estado del arte.
No se sobre-optimizó para "ganar": el resultado es la comparación honesta.

## Hipótesis del CHECKPOINT 2 para D12 — veredicto

> *"Contraste ablativo honesto; con ~4 crisis el resultado negativo es aceptable; el
> deep solo se justificaría con más datos."*

**Se cumple exactamente.** El AE no aporta sobre PCA (resultado negativo), lo cual es
una contribución válida: documenta que la complejidad deep no está justificada aquí. La
comparación se hizo limpia (mismo pipeline) y sin sobreajustar para inflar el AE.

## Fricción con el núcleo

Ninguna en `src/`. Única incidencia: la aserción de causalidad del notebook era
demasiado estricta (`<1e-9`) para float32 de torch; se relajó a tolerancia FP + chequeo
de igualdad de estados (el criterio correcto). El detector es causal; no se tocó el
núcleo. Aprendizaje transversal: **las verificaciones de causalidad de detectores deep
deben comprobar invariancia de ESTADOS/probabilidades, no igualdad exacta del latente
en float** (el FP de torch da ~1e-7 inocuo).

<!-- BEGIN resultados_v2 (generado desde git HEAD; no editar a mano) -->
## Resultados v2 (ADR-003)

> Fuente: `results/benchmark/ranking_v2.csv` y `results/benchmark/metrics_master_v2.csv` de la re-ejecución completa del benchmark (24/24) posterior a la unificación ADR-004 (`results/benchmark/manifest.json`, generado el 2026-09-29T12:41 UTC), numéricamente idéntica a la re-ejecución de ADR-003 §4 (ver `docs/revisiones/informe_unificacion.md`). Walk-forward causal, evaluación OOS por pista. Ranking de **detección**: evento detectado = ≥ 3 sesiones OOS consecutivas en crisis dentro de [pico, suelo]; `score_deteccion` = F1 entre precisión diaria (1 − FAR) y recall por evento, con niveles previos (elegible › precisión ≤ azar › parpadeo › degenerado). Ver `docs/decisions/ADR-003-causalidad-calendario-estado-ranking.md`.

**Notebook de familia:** [`notebooks/11_familia_F7_deep.ipynb`](../../notebooks/11_familia_F7_deep.ipynb) (F7 — Deep learning) · **Teoría:** [`docs/teoria/F7_redes_neuronales.md`](../teoria/F7_redes_neuronales.md).

| Métrica | Pista A | Pista B |
|---|---:|---:|
| Variante evaluada | `deep_ae_regime_k2` | `deep_ae_regime_k2` |
| Puesto de detección | **6** / 12 | **12** / 12 |
| Nivel de etiqueta | `elegible` | `parpadeo` |
| `score_deteccion` | 0.545 | 0.274 |
| Recall por evento | 0.882 (15/17) | 0.222 (2/9) |
| Precisión diaria (lift vs base) | 0.395 (×2.03) | 0.358 (×2.07) |
| Cobertura media de crisis | 0.568 | 0.159 |
| Tasa de falsas alarmas (FAR) | 0.605 | 0.642 |
| Activación media en trampas | 0.005 | 0.021 |
| Switching rate | 0.1911 | 0.0552 |
| Duración media de régimen (sesiones) | 5.2 | 18.0 |
| Estabilidad de etiqueta | 0.995 | 0.998 |
| Puesto legacy (`rank_medio`) | 9 (7.7) | 9 (8.0) |

**Configuración v2 y eventos por pista**

- **Pista A** · OOS 1970-05-12 → 2026-05-29 (n = 14133) · train inicial 2016 sesiones, refit cada 21 (expanding) · 9 feature(s): `SP500_ret_z|SP500_vol_z|SP500_momentum|SP500_drawdown|FF_MKT_z|DGS10_change_z|credit_BaaAaa_mensual_z|term_spread_hist_z|INDPRO_yoy_z`.
  - Detectados (15): `bear_1969_70`, `oil_stagflation_1973`, `volcker_1980_82`, `black_monday_1987`, `gulf_war_sl_1990`, `ltcm_russia_1998`, `dotcom_2000_02`, `gfc_2007_09`, `flash_crash_euro1_2010`, `us_downgrade_euro2_2011`, `china_oil_2015_16`, `fed_tightening_2018q4`, `covid_2020`, `inflation_bear_2022`, `svb_banking_2023`.
  - No detectados (2): `volmageddon_2018q1`, `tariff_selloff_2025`.
- **Pista B** · OOS 2010-04-12 → 2026-05-29 (n = 4059) · train inicial 756 sesiones, refit cada 21 (expanding) · 14 feature(s): `SP500_ret_z|SP500_vol_z|SP500_momentum|SP500_drawdown|FF_MKT_z|DGS10_change_z|credit_BaaAaa_mensual_z|term_spread_hist_z|INDPRO_yoy_z|VIX_level_z|MOVE_level_z|credit_BAA10Y_z|DFII10_change_z|corr_spx_bond`.
  - Detectados (2): `covid_2020`, `inflation_bear_2022`.
  - No detectados (7): `flash_crash_euro1_2010`, `us_downgrade_euro2_2011`, `china_oil_2015_16`, `volmageddon_2018q1`, `fed_tightening_2018q4`, `svb_banking_2023`, `tariff_selloff_2025`.

Lectura: las cifras de la Capa 1 de las secciones anteriores (ventana 2007-2026, 4 crisis, métricas de cobertura sin criterio de detección) **no son comparables** con esta tabla (pistas A y B de `configs/benchmark_spec.yaml`, 17 crisis evaluables OOS en A / 9 crisis evaluables OOS en B, ranking ADR-003). El análisis gráfico por crisis y la comparación intra-familia están en el notebook `notebooks/11_familia_F7_deep.ipynb`.
<!-- END resultados_v2 -->
