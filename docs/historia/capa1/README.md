# Capa 1 (v1) — historia del primer banco de detectores

> Archivo histórico de la **primera vuelta** del TFM: 12 detectores de régimen (7 familias) bajo un
> marco causal común, evaluados sobre un set de datos pequeño (9 series, 15 features). Se **congeló**
> en 2026-07-18 ([ADR-001](../../decisions/ADR-001-rebase-datos.md)) y se **unificó** con v2 en
> 2026-09-29 ([ADR-004](../../decisions/ADR-004-unificacion.md)). Retrospectiva completa en
> [ADR-000](../../decisions/ADR-000-capa1-retrospectiva.md).
>
> Esta carpeta conserva **los artefactos que no se re-ejecutan** (memoria, informe, métricas v1,
> procedencia de datos v1). El código y la teoría que siguen vivos ya no están aquí: se movieron al
> paquete `regimenes` y a `docs/teoria/` y `docs/detectores/` (tabla de §5).

---

## 1. Qué fue la Capa 1

Un **banco de pruebas comparativo** cuyo objetivo no era "el mejor detector" sino **"el mejor para
qué"**: misma interfaz `RegimeDetector`, mismo protocolo walk-forward causal y mismas métricas
(cobertura por crisis, falsas alarmas en las trampas 2013/2018, lead/lag al suelo del drawdown,
switching, duración, estabilidad, BIC) para los 12 detectores.

Partía de la **tarea previa** (HMM gaussiano de 2 estados, in-sample, con z-scores de muestra
completa; ver [`../../context/RESUMEN_DETECCION_REGIMENES.md`](../../context/RESUMEN_DETECCION_REGIMENES.md)),
cuyas limitaciones (look-ahead, sin walk-forward, Viterbi duro, supuesto gaussiano) motivaron el banco.

Fases internas de la Capa 1 (numeración propia, **independiente** de la hoja de ruta v2): 0 estructura
+ interfaz + evaluador · 1 datos + EDA · 2 estado del arte · 3 implementación en 4 tandas · 4 síntesis
comparativa · 5 pulido. Estado detallado por fase en [`memoria/INDEX.md`](memoria/INDEX.md).

### Los 12 detectores (+ ablación D13)

| ID | Detector | Familia | Clase v1 | Ventana OOS v1 | ¿Vio 2008 OOS? |
|---|---|---|---|---|:---:|
| D1 | `rule_vix_threshold` | F1 reglas/umbrales | baseline | 1998–2026 | sí |
| D2 | `rule_composite_riskoff` | F1 reglas/umbrales | baseline | 2015–2026 | no |
| D3 | `clustering_gmm` (K=3) | F2 clustering | baseline | 2015–2026 | no |
| D4 | `hmm_gaussian_2s` | F3 HMM (puente con la tarea previa) | baseline | 2012–2026 | no |
| D5 | `markov_switching_var` (2 estados) | F4 Markov-Switching | avanzado | 1993–2026 | sí |
| D6 | `garch_t_vol` (GJR-GARCH-t) | F5 GARCH | avanzado | 1993–2026 | sí |
| D7 | `changepoint_online` (CUSUM robusto) | F6 change-point | avanzado | 1993–2026 | sí |
| D8 | `hmm_tstudent` (K=4) | F3 HMM avanzado | avanzado | 2012–2026 | no |
| D9 | `jump_model` (λ=50) | F2↔F3 jump model | avanzado | 2015–2026 | no |
| D10 | `turbulence_mahalanobis` | F1 multivariante (Kritzman) | avanzado | 1998–2026 | sí |
| D11 | `msgarch_regime` | F5 MS-GARCH | exploratorio-negativo | 1991–2026 | sí |
| D12 | `deep_ae_regime` (AE→GMM) | F7 redes | exploratorio-negativo | 2015–2026 | no |
| D13 | `hsmm_tstudent` | F3 (ablación de D8) | fuera del ranking | = D8 | no |

Fuente: [`memoria/99_conclusions.md`](memoria/99_conclusions.md) y
[`resultados/metrics_master.csv`](resultados/metrics_master.csv). La ablación D8↔D13 está en
[`resultados/ablation_hsmm/`](resultados/ablation_hsmm/README.md): la duración explícita **no** reduce
el switching ni alarga los regímenes → se mantuvo D8 por parsimonia.

## 2. Los 5 hallazgos que sobreviven

1. **El look-ahead de los z-scores in-sample compraba suavidad, no acierto.** D4 causal frente a su
   versión in-sample: el switching sube de 0.047 a 0.100 y sigue fallando 2013/2018; el acierto en
   crisis grandes nunca dependió del look-ahead.
2. **La t-Student mejora el ajuste con holgura:** ΔBIC ≈ +10963 de D8 frente a D4 con las mismas 7
   features; ν por estado decreciente [10.2, 7.6, 4.2, 2.4] (crisis = colas más pesadas).
3. **Viterbi-por-bloque era menos causal y menos estable que el filtrado forward:** en D4 el switching
   bajó 0.124 → 0.100 y la duración subió 8.1 → 9.9 días al pasar a filtrado forward.
4. **2013 (taper tantrum) es el punto ciego universal:** 6 detectores independientes que lo tenían OOS
   lo marcan entre 0 % y ~12 % → *la taxonomía de features importa* (todas eran de vol/equity).
5. **La complejidad extra no se paga con ~4 crisis:** D11 MS-GARCH degenera (cobertura GFC 0 %) y D12
   AE empeora a su PCA (switching 0.287) → parsimonia validada.

(Las cifras son las de la memoria v1; detalle y evidencia en
[`memoria/99_conclusions.md`](memoria/99_conclusions.md) §2 y en el
[informe](informe/informe_capa1.pdf).)

## 3. Resumen del EDA v1 (tag `capa1-final`: `capa1_exploracion/notebooks/00_eda.ipynb`)

- **Datos:** 9 series descargadas **sin imputar** (S&P 500, VIX, MOVE, TLT, IEF, HYG, GLD, DXY y la
  pendiente 10Y−3M). FRED era inaccesible en aquel entorno → fallbacks documentados: DXY por
  `DX-Y.NYB`, curva por el proxy `^TNX − ^IRX` y HY OAS omitido (crédito vía HYG y spread HYG−IEF).
  Procedencia en [`datos_v1/provenance.json`](datos_v1/provenance.json) y
  [`datos_v1/coverage_report.csv`](datos_v1/coverage_report.csv).
- **Ventana común 2007-04-11 → 2026-06**, gobernada por HYG (la serie que arranca más tarde): se
  prefirió perder la DotCom antes que imputar hacia atrás.
- **Colas gordas:** kurtosis de exceso del S&P 500 25.6 y de HYG 39.6 → motiva la t-Student.
  Crisis frente a calma: la distribución se ensancha y se sesga a pérdidas (partición de referencia,
  no un detector).
- **Patrón oro:** 4 crisis sistémicas (2008, 2011, 2020, 2022) y 2 trampas (2013, 2018); suelos de
  drawdown calculados sobre la serie real (`DRAWDOWN_TROUGHS`). Con **n≈4** crisis se puede describir
  comportamiento, no hacer tests de significancia.
- **Correlación rolling S&P 500/Treasuries** cambia de signo entre regímenes (Gulko 2002) → feature
  `corr_spx_bond`.
- **15 features causales** (z-scores expanding) verificadas con `assert_causal` (`max_abs_diff = 0`).

Memoria completa: [`memoria/01_data_and_eda.md`](memoria/01_data_and_eda.md). El EDA v2 que lo
sustituye (166 series, 22 crisis) está en [`../../datos/EDA_v2.md`](../../datos/EDA_v2.md).

## 4. Resumen de la comparativa v1 (tag `capa1-final`: `capa1_exploracion/notebooks/13_comparison.ipynb`)

- **Tesis:** no hay detector dominante; 4 familias se reparten 6 ejes (cobertura sistémica,
  especificidad, persistencia, lead/lag, BIC, coste). Es un resultado, no un fracaso.
- **Equidad de ventana:** la cobertura se compara **por grupo de ventana** (vio 2008 OOS o no); nunca se
  penaliza a un detector por lo que no pudo ver.
- **Estrés agregado:** para los multi-estado (D3, D8, D12) se define estrés = unión de los dos estados
  más severos, para compararlos con los binarios; el notebook recomputa esos tres y comprueba (±0.01)
  que la crisis estricta coincide con el master.
- **Veredictos por eje:** cobertura sistémica en ventana larga D5 0.98 ≈ D6 0.97 ≈ D1 0.92 (la vol
  manda; la sofisticación apenas bate a la regla VIX); especificidad, persistencia, lead/lag y coste:
  D7 (CUSUM); BIC: D8.
- **Recomendación v1:** núcleo HMM t-Student multi-estado (respaldo *consistente con* la propuesta, no
  superioridad OOS estricta) + change-point tipo D7 como alerta temprana + D1/D5/D6 como control.

Figuras y tablas del informe: [`memoria/pdf_src/`](memoria/pdf_src/01_hallazgos.md).

## 5. Por qué se congeló (ADR-001) y por qué se unifica (ADR-004)

- **Congelación (2026-07-18):** los 12 detectores **no eran comparables 1:1**: cada uno usaba su
  subconjunto de features (1, 4, 7 o 15) y su ventana OOS (unos veían 4 crisis y ~8000 días; otros 2 y
  ~2600). Los datos no eran una variable controlada. Se congeló la Capa 1 como foto y se re-basó la
  capa de datos (dos pistas, `configs/benchmark_spec.yaml`). Ver
  [ADR-001](../../decisions/ADR-001-rebase-datos.md).
- **Unificación (2026-09-29):** v2 acabó importando los detectores de la Capa 1 (vía `sys.path`) y
  duplicando su interfaz. Mantener dos árboles costaba más que lo que protegía. Por decisión del
  usuario se revierte el punto *"Capa 1 se mantiene intacta"* de ADR-001: un solo paquete
  (`regimenes`), un notebook por familia y este archivo histórico. Ver
  [ADR-004](../../decisions/ADR-004-unificacion.md).

## 6. Dónde está hoy cada cosa

| Antes (Capa 1) | Hoy |
|---|---|
| `capa1_exploracion/detectors/*.py` (12 + `hsmm_tstudent` + utilidades HMM) | `src/regimenes/detectores/f1_reglas/` … `f7_deep/` |
| `capa1_exploracion/src/detector_base.py` | `src/regimenes/detectores/base.py` (era idéntica a la de v2) |
| `capa1_exploracion/src/{data_loader,evaluation,features,viz}.py` (marco v1) | retirados al cerrar la unificación (`git rm -r capa1_exploracion`); recuperables con el tag `capa1-final` (§7). El juez vigente es `regimenes.evaluacion` |
| `capa1_exploracion/memory/00_state_of_the_art.md` | [`docs/teoria/00_estado_del_arte.md`](../../teoria/00_estado_del_arte.md) |
| `capa1_exploracion/memory/sota/0k_*.md` y `.bib` | `docs/teoria/Fk_*.md` y `.bib` (p. ej. [`F3_hmm.md`](../../teoria/F3_hmm.md)) |
| `capa1_exploracion/memory/detectors/NN_*.md` | `docs/detectores/DNN_*.md` (p. ej. [`D08_hmm_tstudent.md`](../../detectores/D08_hmm_tstudent.md)) |
| `capa1_exploracion/memory/{INDEX,01_data_and_eda,99_conclusions}.md`, `pdf_src/` | [`memoria/`](memoria/INDEX.md) (aquí) |
| `capa1_exploracion/report/` (tex, pdf, bib, revisión) | [`informe/`](informe/informe_capa1.pdf) (aquí) |
| `capa1_exploracion/results/*.csv`, `_archive/`, `ablation_hsmm/` | [`resultados/`](resultados/metrics_master.csv) (aquí) |
| `capa1_exploracion/data/raw/{provenance.json,coverage_report.csv}` | [`datos_v1/`](datos_v1/provenance.json) (aquí) |
| `capa1_exploracion/notebooks/01..12_<detector>.ipynb` | su teoría y hallazgos se rescatan en los notebooks de familia `notebooks/05_familia_F1_reglas` … `11_familia_F7_deep`; los `.ipynb` v1 ejecutados, en el tag `capa1-final` |
| `capa1_exploracion/notebooks/A1_hsmm_ablation.ipynb` | D13 como ablación en `notebooks/07_familia_F3_hmm`; original en el tag |
| `capa1_exploracion/notebooks/00_eda.ipynb`, `13_comparison.ipynb` | resumidos en §3–§4 de este README; originales en el tag |
| bibliografía `report/references.bib` | fusionada en la bibliografía única [`docs/references.bib`](../../references.bib) (se conserva aquí para que el `.tex` compile) |

**Rutas dentro de estos documentos.** La memoria y el informe son históricos y **no se reescriben**:
sus rutas (`docs/memory/…`, `results/…`, `notebooks/…`, `report/…`, `detectors/…`) son relativas a la
antigua raíz de la Capa 1 (`capa1_exploracion/`). Tradúcelas con la tabla anterior.

## 7. Recuperar la Capa 1 original (tag `capa1-final`)

El tag `capa1-final` apunta al último commit anterior a la unificación, con la Capa 1 completa
(notebooks v1 ejecutados con sus salidas, marco v1 y detectores en su sitio original):

```bash
git show capa1-final:capa1_exploracion/src/evaluation.py        # un archivo concreto
git show capa1-final:capa1_exploracion/notebooks/13_comparison.ipynb > /tmp/13_comparison.ipynb
git worktree add ../capa1-v1 capa1-final                         # árbol completo, sin tocar tu rama
```

Código v1 que solo vive en el tag desde que se retiró `capa1_exploracion/`: `data_loader.py`,
`evaluation.py`, `features.py`, `viz.py` (marco v1) y los 15 notebooks v1. El código v2 previo a la
unificación está en el tag `v2-pre-unificacion`; la historia anterior al re-base, en la rama remota
`backup-main-pre-datos-v2`.

> Si re-ejecutas un notebook v1 desde un worktree del tag, hazlo **desde dentro de
> `capa1_exploracion/notebooks/`**: descubren su raíz subiendo hasta encontrar `src/`.
