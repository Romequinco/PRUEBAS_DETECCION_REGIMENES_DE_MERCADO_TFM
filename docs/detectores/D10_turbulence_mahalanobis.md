# D10 — `turbulence_mahalanobis` (FASE 3, Tanda 3) · Familia F1 (multivariante)

<!-- BEGIN ubicacion_v2 -->
> **Ubicación.** Código: `src/regimenes/detectores/f1_reglas/turbulence_mahalanobis.py` (registro: `regimenes.detectores.registry`, id `D10`) · Notebook v2 de familia: [`notebooks/05_familia_F1_reglas.ipynb`](../../notebooks/05_familia_F1_reglas.ipynb) · Resultados v2: sección [Resultados v2 (ADR-003)](#resultados-v2-adr-003) al final. Esta ficha nació en la Capa 1 (v1, congelada en el tag `capa1-final`); el texto histórico se conserva. Cuando menciona el núcleo (`src/`, `detector_base.py`, `evaluation.py`, `features.py`, `data_loader.py`, `INDEX.md`) se refiere al de la Capa 1: hoy `detector_base.py` → `src/regimenes/detectores/base.py`, `evaluation.py` → `regimenes.evaluacion`, `INDEX.md` → `docs/historia/capa1/memoria/INDEX.md`; `features.py`/`data_loader.py` de v1 quedan en el tag `capa1-final`.
<!-- END ubicacion_v2 -->

> Índice de **turbulencia financiera** (Kritzman, Page & Turkington 2012): distancia
> de Mahalanobis multivariante con covarianza **expanding causal**. Mide el colapso de
> correlaciones / "rareza" del vector de mercado respecto a su historia.
> Código: `src/regimenes/detectores/f1_reglas/turbulence_mahalanobis.py` · Notebook:
> `capa1_exploracion/notebooks/10_turbulence_mahalanobis.ipynb` (v1, tag `capa1-final`) · Métricas:
> `docs/historia/capa1/resultados/metrics_10_turbulence_mahalanobis.csv`.

## Implementado

**Modelo.** Turbulencia `d_t = (x_t − μ)ᵀ Σ⁻¹ (x_t − μ)`, con μ y Σ estimados de forma
**CAUSAL expanding** (datos < t). Estado por umbral causal sobre `d_t`: percentiles del
train τ_in=p90 / τ_out=p70 con histéresis + dwell. 2 estados (0=calma, 1=crisis).
Bibliografía: `kritzman2012`, `gulko2002`.

**Features / ventana — 2013 OOS (el contraste clave).** Vector multivariante de
4 cambios causales desde **1990**: `[SP500_ret, VIX_change, DXY_change,
yield_slope_chg]` (equity, miedo, dólar, tipos). Se eligió este set de histórico LARGO
(no las 15 features de 2007, que incluyen HYG) precisamente para que **2013, 2008, 2011
sean OOS**. Ventana efectiva OOS = **1998-06-02 → 2026-06-12 (n=6987)**; confirmado que
2013 < OOS-start es falso, i.e. 2013 cae dentro del OOS. market_returns = retorno log
S&P 500.

## Descubierto

**Orientación verificada (Arreglo 4 funciona).** Turbulencia media por estado:
**crisis = 11.30 vs calma = 2.13** → crisis = ALTA turbulencia = alta vol de retornos,
**NO invertido**, sin warning de fallback. El núcleo vol-primario orientó bien un
detector que separa por varianza/turbulencia (era candidato a inversión; no la hubo).

**Cobertura (CAUSAL OOS):**

| Ventana | Cobertura | |
|---|---:|---|
| GFC_2008 | 82.2 % | sólida (2008 OOS) |
| EuroDebt_2011 | 48.2 % | parcial |
| COVID_2020 | 76.0 % | buena |
| Inflation_2022 | 43.1 % | floja |
| TaperTantrum_2013 (trampa) | 12.3 % | apenas se enciende |
| Selloff_Q4_2018 (trampa) | 30.2 % | parcial |

**El test estrella (2013) NO se cumple como esperaba la hipótesis.** La premisa del
CP2 era que la turbulencia multivariante captaría el colapso de correlaciones de 2013
que los univariantes no ven. Resultado: **2013 marca solo 12.3 %** — esencialmente NO
se enciende, igual que D6 (GARCH equity, ~11 %) y D4 (HMM gaussiano, que tampoco lo
veía). Conclusión honesta: **el taper de 2013 NO fue un evento de turbulencia
multivariante** en este espacio de 4 features; fue una repreciación ordenada de tipos
sin "rareza" conjunta de equity/vol/dólar/curva. Añadir la curva al Mahalanobis no lo
ilumina. **2013 sigue siendo el punto ciego universal del banco** (D1 0%, D5 3.8%,
D6 11%, D7 0%, D8 0%, D10 12%).

**Tensión conceptual a resolver (para FASE 4 / decisión del usuario).** El marco actual
clasifica 2013 como **ventana-TRAMPA** (`FALSE_POSITIVE_WINDOWS`): firmar crisis ahí
cuenta como FALSO POSITIVO, así que `fa_2013 = 12.3 %` es en realidad BUENO
(especificidad). Pero el prompt de D10 lo planteaba como un evento DESEABLE de captar
("el agujero que D10 debería tapar"). Las dos lecturas chocan: ¿2013 es una crisis
rápida que un buen detector debe ver, o una trampa que no debe disparar? Los datos
zanjan la cuestión empírica (2013 no es turbulencia conjunta), pero la **etiqueta
del marco** (crisis vs trampa) es una decisión que conviene revisar en la síntesis.

**Flickering.** switching 0.087, duración media 11.4 d (305 episodios de crisis,
~5 d cada uno): **flickea más** que D6/D7 (la turbulencia es ruidosa día a día pese a
la histéresis). `false_alarm_rate` = 0.815 (alto, como todos los de histórico largo:
ve LTCM 1998, dotcom, 2010, 2015-16, 2023 fuera de las 4 ventanas canónicas).
`label_stability` = 1.000.

## Hipótesis del CHECKPOINT 2 para D10 — veredicto

> *"Capta el colapso de correlaciones multivariante que las reglas univariantes no
> ven."*

**Se cumple solo en parte.** SÍ capta los eventos sistémicos donde las correlaciones
sí colapsan (GFC 82 %, COVID 76 %) usando un único índice multivariante barato y
causal — eso valida el mecanismo de Kritzman. Pero **NO** captura 2013, que era el
contraste que lo justificaba frente a D4/D6: porque 2013 simplemente **no fue** un
episodio de turbulencia conjunta. La contribución real de D10 no es "tapar 2013", sino
ofrecer una señal de estrés sistémico multivariante de muy bajo coste; su debilidad es
el flickering y que no añade nada sobre el agujero de 2013.

## Fricción con el núcleo

Ninguna. El Arreglo 4 (vol-primario) orientó D10 correctamente sin necesidad de parche
local (a diferencia de lo que D6 tuvo que hacer antes del Arreglo 4): verificado
crisis = alta turbulencia en walk-forward, sin warning de fallback. Confirma que el
Arreglo 4 cubre el caso de los detectores que separan en varianza.

<!-- BEGIN resultados_v2 (generado desde git HEAD; no editar a mano) -->
## Resultados v2 (ADR-003)

> Fuente: `results/benchmark/ranking_v2.csv` y `results/benchmark/metrics_master_v2.csv` del benchmark completo (24/24; `results/benchmark/manifest.json`, generado el 2026-09-29T12:41 UTC), con las cifras de ADR-003 §4. Walk-forward causal, evaluación OOS por pista. Ranking de **detección**: evento detectado = ≥ 3 sesiones OOS consecutivas en crisis dentro de [pico, suelo]; `score_deteccion` = F1 entre precisión diaria (1 − FAR) y recall por evento, con niveles previos (elegible › precisión ≤ azar › parpadeo › degenerado). Ver `docs/decisions/ADR-003-causalidad-calendario-estado-ranking.md`.

**Notebook de familia:** [`notebooks/05_familia_F1_reglas.ipynb`](../../notebooks/05_familia_F1_reglas.ipynb) (F1 — Reglas / umbrales) · **Teoría:** [`docs/teoria/F1_reglas_umbrales.md`](../teoria/F1_reglas_umbrales.md).

| Métrica | Pista A | Pista B |
|---|---:|---:|
| Variante evaluada | `turbulence_mahalanobis` | `turbulence_mahalanobis` |
| Puesto de detección | **1** / 12 | **6** / 12 |
| Nivel de etiqueta | `elegible` | `elegible` |
| `score_deteccion` | 0.614 | 0.457 |
| Recall por evento | 0.941 (16/17) | 1.000 (9/9) |
| Precisión diaria (lift vs base) | 0.455 (×2.35) | 0.296 (×1.71) |
| Cobertura media de crisis | 0.481 | 0.600 |
| Tasa de falsas alarmas (FAR) | 0.545 | 0.704 |
| Activación media en trampas | 0.000 | 0.047 |
| Switching rate | 0.0417 | 0.0567 |
| Duración media de régimen (sesiones) | 24.0 | 17.6 |
| Estabilidad de etiqueta | 0.999 | 0.996 |
| Puesto legacy (`rank_medio`) | 4 (5.4) | 8 (7.6) |

**Configuración v2 y eventos por pista**

- **Pista A** · OOS 1970-05-12 → 2026-05-29 (n = 14133) · train inicial 2016 sesiones, refit cada 21 (expanding) · 4 feature(s): `SP500_ret|SP500_vol_z|DGS10_change_z|FF_MKT_z`.
  - Detectados (16): `bear_1969_70`, `oil_stagflation_1973`, `volcker_1980_82`, `black_monday_1987`, `gulf_war_sl_1990`, `ltcm_russia_1998`, `dotcom_2000_02`, `gfc_2007_09`, `flash_crash_euro1_2010`, `us_downgrade_euro2_2011`, `china_oil_2015_16`, `volmageddon_2018q1`, `fed_tightening_2018q4`, `covid_2020`, `inflation_bear_2022`, `tariff_selloff_2025`.
  - No detectados (1): `svb_banking_2023`.
- **Pista B** · OOS 2010-04-12 → 2026-05-29 (n = 4059) · train inicial 756 sesiones, refit cada 21 (expanding) · 4 feature(s): `SP500_ret|VIX_change_z|DXY_change_z|slope_10y2y_z`.
  - Detectados (9): `flash_crash_euro1_2010`, `us_downgrade_euro2_2011`, `china_oil_2015_16`, `volmageddon_2018q1`, `fed_tightening_2018q4`, `covid_2020`, `inflation_bear_2022`, `svb_banking_2023`, `tariff_selloff_2025`.
  - No detectados (0): ninguno.

Lectura: las cifras de la Capa 1 de las secciones anteriores (ventana 2007-2026, 4 crisis, métricas de cobertura sin criterio de detección) **no son comparables** con esta tabla (pistas A y B de `configs/benchmark_spec.yaml`, 17 crisis evaluables OOS en A / 9 crisis evaluables OOS en B, ranking ADR-003). El análisis gráfico por crisis y la comparación intra-familia están en el notebook `notebooks/05_familia_F1_reglas.ipynb`.
<!-- END resultados_v2 -->
