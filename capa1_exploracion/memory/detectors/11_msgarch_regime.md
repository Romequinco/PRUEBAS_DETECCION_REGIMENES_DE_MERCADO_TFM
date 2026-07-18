# D11 — `msgarch_regime` (FASE 3, Tanda 4 · exploratoria) · Familia F5 (RS-GARCH)

> MS-GARCH(1,1)-t de Haas-Mittnik-Paolella (2004), **implementado desde cero en
> numpy/scipy** (filtro de Hamilton propio), SIN R ni rpy2. **RESULTADO NEGATIVO
> documentado**: implementable y causal, pero su asignación de regímenes degenera en
> walk-forward y NO sirve como detector. Código: `detectors/msgarch_regime.py` ·
> Notebook: `notebooks/11_msgarch_regime.ipynb` · Métricas:
> `results/metrics_11_msgarch_regime.csv`.

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
