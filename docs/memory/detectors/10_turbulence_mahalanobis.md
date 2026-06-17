# D10 — `turbulence_mahalanobis` (FASE 3, Tanda 3) · Familia F1 (multivariante)

> Índice de **turbulencia financiera** (Kritzman, Page & Turkington 2012): distancia
> de Mahalanobis multivariante con covarianza **expanding causal**. Mide el colapso de
> correlaciones / "rareza" del vector de mercado respecto a su historia.
> Código: `detectors/turbulence_mahalanobis.py` · Notebook:
> `notebooks/10_turbulence_mahalanobis.ipynb` · Métricas:
> `results/metrics_10_turbulence_mahalanobis.csv`.

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
