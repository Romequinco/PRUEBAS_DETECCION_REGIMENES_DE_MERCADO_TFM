# Capa 1 — Exploración de detectores (❄️ CONGELADO)

> Esta carpeta es una **foto congelada** de la primera vuelta del TFM: 12 detectores de
> régimen (7 familias) implementados y evaluados bajo un marco causal común. **No se re-ejecuta
> ni se modifica.** Se conserva como *baseline* y como registro de los hallazgos que motivaron
> el re-base de datos (ver `../docs/decisions/ADR-001-rebase-datos.md`).

## Qué se hizo aquí

Un **banco de pruebas comparativo**: misma interfaz `RegimeDetector`, mismo protocolo
walk-forward causal, mismas métricas para los 12 detectores.

| ID | Detector | Familia |
|---|---|---|
| D1 | `rule_vix_threshold` | F1 reglas/umbrales |
| D2 | `rule_composite_riskoff` | F1 reglas/umbrales |
| D3 | `clustering_gmm` | F2 clustering |
| D4 | `hmm_gaussian_2s` | F3 HMM (puente con tarea previa) |
| D5 | `markov_switching_var` | F4 Markov-Switching |
| D6 | `garch_t_vol` | F5 GARCH |
| D7 | `changepoint_online` | F6 change-point |
| D8 | `hmm_tstudent` | F3 HMM avanzado |
| D9 | `jump_model` | F2↔F3 jump model |
| D10 | `turbulence_mahalanobis` | F1 multivariante (Kritzman) |
| D11 | `msgarch_regime` | F5 MS-GARCH (exploratorio-negativo) |
| D12 | `deep_ae_regime` | F7 redes (exploratorio-negativo) |

## Qué se aprendió (hallazgos que sobreviven al re-base)

1. El *look-ahead* de los z-scores in-sample compraba **suavidad, no acierto**.
2. La t-Student mejora el ajuste con holgura: **ΔBIC ≈ +10963** vs. gaussiano (mismas features).
3. El Viterbi-por-bloque metía *switching* artificial → el **filtrado forward** es más causal *y* más estable.
4. **2013 (taper) = punto ciego universal**: 6 detectores independientes lo fallan → *la taxonomía de features importa* (todas eran de vol/equity).
5. La complejidad extra (MS-GARCH, deep AE) **no se paga** con ~4 crisis → parsimonia validada.

## Por qué se congeló (el problema)

Los 12 detectores **no son comparables 1:1**: cada uno usa su propio subconjunto de features
(1, 4, 7 o 15) y su propia ventana OOS (unos ven 4 crisis y ~8000 días, otros 2 crisis y ~2600).
El detalle y la decisión de re-basar los datos están en
**`../docs/decisions/ADR-001-rebase-datos.md`**.

## Estructura interna (auto-contenida)

```
capa1_exploracion/
├── src/          marco que usó (data_loader, features, detector_base, evaluation, viz)
├── detectors/    los 12 detectores
├── notebooks/    00_eda + 01..12 (un detector c/u) + 13_comparison
├── scripts/      builders (regeneran+ejecutan cada notebook), figs, verify
├── results/      métricas por detector + tabla maestra + figuras
├── report/       informe_capa1 (LaTeX + PDF)
├── data/         metadatos de procedencia de los datos v1 (raw gitignored)
└── memory/       memoria viva: INDEX.md (estado), sota/, detectors/, 99_conclusions.md
```

**Reproducibilidad**: las rutas son relativas a esta carpeta (los builders usan
`parents[2]` → `capa1_exploracion`), así que sigue siendo auto-contenida tras el re-base.
Para reconstruir un notebook: `python capa1_exploracion/scripts/builders/_build_NN.py`
(lento: la tanda completa son 2-3 h; **no es necesario** para el trabajo v2).

> ⚠️ **Re-ejecutar SOLO vía los builders**, nunca lanzando la notebook a mano desde la raíz
> del repo. Las notebooks descubren su ROOT subiendo hasta encontrar `src/`; como ahora existe
> un `src/` v2 en la raíz, ejecutarlas ad-hoc con `cwd=raíz` resolvería al `src/` equivocado.
> Los builders fijan el `cwd` a `capa1_exploracion/notebooks/`, por eso resuelven bien.
