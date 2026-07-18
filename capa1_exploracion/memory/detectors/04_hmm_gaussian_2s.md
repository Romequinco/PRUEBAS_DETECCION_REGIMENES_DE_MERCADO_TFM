# D4 — `hmm_gaussian_2s` (FASE 3, Tanda 1) · Familia F3 (HMM)

> **Baseline puente** con la tarea previa (`Tarea_riesgos.ipynb`). Reproduce
> honestamente el GaussianHMM 2 estados y MIDE el efecto del look-ahead comparando
> una versión in-sample (no causal) con una causal walk-forward.
>
> Código: `detectors/hmm_gaussian_2s.py` · Notebook: `notebooks/04_hmm_gaussian_2s.ipynb`
> (constructor `notebooks/_build_04.py`) · Métricas: `results/metrics_04_hmm_gaussian_2s.csv`
> (causal) y `results/metrics_04_hmm_gaussian_2s_insample.csv` (in-sample, no causal).

## Implementado

**Modelo.** `GaussianHMM(n_components=2, covariance_type='full', n_iter=200)` de
`hmmlearn`. En `fit` se prueban **5 inicializaciones** (semillas 42–46) y se elige la
de mayor log-verosimilitud (mitiga óptimos locales del EM, como la tarea previa, que
usaba 10 seeds 42–51). El orden económico de estados (0=calma, 1=crisis) lo fija
`label_states_economically` del núcleo, usando el retorno log del S&P 500.

**Features puente (7).** Subconjunto de las 15 de `features.parquet`, idéntico a la
tarea previa: `SP500_ret_z, TLT_ret_z, IEF_ret_z, HYG_ret_z, SP500_vol_z,
credit_spread_z, VIX_level_z`. Constante `BRIDGE_FEATURES` en el módulo. Ventana
2007-07-06 → 2026-06-12 (4 665 obs).

**Override.** `predict` (duro, vía `_predict_states`) usa **Viterbi** → modo
in-sample NO causal. `predict_online` y `predict_proba` usan **FILTRADO FORWARD
causal** (`detectors/_hmm_utils.filtered_posterior`) con contexto de burn-in: la
etiqueta/probabilidad de t solo usa observaciones <= t, reordenadas al orden
canónico. `score` = logL; `n_parameters` = `k²−1 + k·d + k·d(d+1)/2` (transición +
medias + covarianzas full) = 73 con k=2, d=7. Bibliografía: `hamilton1989,
hmm_rabiner1989, hmm_bulla2011, guidolintimmermann2007` (verificadas en
`docs/references.bib`).

**Dos versiones (núcleo de D4).**
1. **IN-SAMPLE (NO causal, marcada).** Ajuste + Viterbi sobre TODA la muestra.
   Reproduce la tarea previa; tiene look-ahead (el EM ve el futuro y el Viterbi
   suaviza sobre todo el histórico). Se reporta aparte, NO comparable con causales.
2. **CAUSAL walk-forward.** `ev.walk_forward(lambda: HMMGaussian2S(n_states=2),
   X, train_size=252*5, step=21, expanding=True)`. Cada fold reentrena solo con
   pasado y decodifica el bloque de test de 21 días.

**Nota de causalidad (CHECKPOINT 3A-ter — Arreglo 2).** La versión causal usa
**filtrado forward estricto**, NO Viterbi intra-bloque. El Viterbi y el
`predict_proba` de `hmmlearn` suavizan con todo el bloque (miran días futuros del
mes), look-ahead de hasta `step`=21 días. `predict_online`/`predict_proba`
sobrescritos calculan el forward filter (`_hmm_utils`) sobre `train_burn-in + bloque`
y devuelven solo el bloque: cada t usa solo obs <= t. El Viterbi sobre toda la
muestra queda EXCLUSIVAMENTE en la versión in-sample marcada NO causal. Patrón
reutilizable por D8 (hmm_tstudent).

## Descubierto

**`ventana_eval` causal:** OOS = 2012-07 → 2026-06 aprox. (train inicial de ~5 años).
Con esa ventana, **2008 (GFC) y 2011 (deuda europea) caen DENTRO del train inicial**
→ no son out-of-sample → su cobertura sale **NaN** en la versión causal. Es el punto
metodológico central: con datos que empiezan en 2007-04 (por HYG) no se puede evaluar
la GFC 2008 de forma causal sin sacrificar casi todo el train. La versión in-sample sí
las cubre (in-sample). Las crisis evaluables OOS son **2020 y 2022**, y las trampas
**2013 y 2018**.

**Cobertura in-sample vs causal por crisis** (números finales del notebook
ejecutado; `ventana_eval` causal = 2012-07-20 → 2026-06-12, n=3405):

| Ventana | In-sample | Causal OOS | Lectura |
|---|---:|---:|---|
| GFC_2008 | 100.0 % | **NaN** | en train, no OOS |
| EuroDebt_2011 | 81.0 % | **NaN** | en train, no OOS |
| COVID_2020 | 96.0 % | 96.0 % | OOS, idéntica (no era look-ahead) |
| Inflation_2022 | 82.7 % | 86.1 % | OOS, sube ligeramente |
| TaperTantrum_2013 (trampa) | 12.0 % | 25.0 % | no sostenida, algo peor causal |
| Selloff_Q4_2018 (trampa) | 30.5 % | 45.8 % | parcial, peor causal |

*(cifras causales tras el Arreglo 2 = filtrado forward; ver "Re-evaluación" abajo.)*

**Efecto del look-ahead.** Para las crisis que SÍ son OOS (2020, 2022) la cobertura
**no cae** al pasar de in-sample a causal — incluso sube ligeramente en 2022. Es decir,
el acierto del modelo en crisis **grandes y persistentes** NO era un artefacto del
look-ahead: un HMM gaussiano reentrenado solo con pasado las sigue capturando. El
look-ahead de la tarea previa afectaba sobre todo a la *estabilidad de las etiquetas
históricas* (re-decodificación de 2008/2011 al reentrenar), no a la detección OOS de
las crisis recientes. La pérdida real al "ir causal" es que **2008/2011 dejan de ser
evaluables** por la ventana, no que el modelo empeore.

**¿2008 fuera de OOS?** Sí: con train inicial de varios años 2008 (y 2011) quedan en el
train → NaN en la versión causal. Documentado en `01_data_and_eda.md` §3 (tensión de
cobertura). Para evaluar 2008 causalmente haría falta un detector con histórico más
largo (p. ej. solo S&P 500+VIX desde 1990, como D1), no el HMM puente atado a las 7
features con HYG desde 2007.

**Fallo en 2013/2018.** Confirmado: el HMM gaussiano apenas marca crisis sostenida en
el taper tantrum 2013 y solo parcialmente en el sell-off Q4 2018. Coherente con las
fat tails del EDA (kurtosis 25–40): las emisiones gaussianas subestiman las colas y el
modelo prioriza episodios largos de alta varianza, perdiéndose las correcciones
rápidas. Es justo la motivación de los detectores posteriores (HMM t-Student / GMM-HMM,
D8; K=3–4 para separar corrección de crisis sistémica).

**Flickering / persistencia.** In-sample (Viterbi global): `switching_rate=0.047`,
duración media **21.2 días**. Causal con filtrado forward: `switching_rate=0.100`,
duración media **9.9 días**. La causal flickea ~2.1× más que la in-sample suavizada,
y por eso sus falsas alarmas en 2013/2018 son mayores (25 % y 46 %). El beneficio del
look-ahead in-sample era la suavidad del recorrido, no la detección OOS de crisis
grandes. `label_stability=0.994` (muy alto): reentrenar con más datos casi no cambia
la etiqueta OOS ya emitida.

**Bondad de ajuste.** logL, AIC y BIC se exponen vía `score`/`aic`/`bic` (k=73 params).
Sirven de referencia para comparar contra K=3 y contra emisiones t-Student en detectores
posteriores (se espera que BIC mejore con t pese a más parámetros si las colas se
ajustan mejor).

## Re-evaluación tras Arreglos 1-3 (CHECKPOINT 3A-ter) — hallazgo

El Arreglo 2 sustituyó el Viterbi/forward-backward intra-bloque por **filtrado
forward causal**. La hipótesis de partida era que quitar el suavizado intra-bloque
SUBIRÍA el switching. Ocurrió lo **contrario**: switching **bajó 0.124 → 0.100** y la
duración media **subió 8.1 → 9.9 días**.

**Por qué (explicación).** El esquema anterior decodificaba con Viterbi cada bloque de
21 días **por separado**, rearrancando desde `startprob_` en cada frontera de mes →
inestabilidad en bloques cortos y saltos artificiales en las costuras. El filtrado
forward, en cambio, propaga la creencia de estado **de forma continua** a través de
todo el histórico (con burn-in del train), gobernada por la diagonal alta de la matriz
de transición → más persistencia. Conclusión: el filtrado forward es a la vez **más
causal** (cero look-ahead intra-bloque) **y más estable** que el Viterbi-por-bloque.
El resto de cambios menores: `false_alarm_rate` 0.76 → 0.73, `cov_Inflation_2022`
0.894 → 0.861, `fa_2013` 0.28 → 0.25, `fa_2018` 0.42 → 0.46, y lead/lag más realistas
(COVID causal −240 → −151 d; la in-sample, con `p_crisis` ahora filtrada en vez de
suavizada, pasa de −158 a −20 d). 2008/2011 siguen NaN (en train).

## Hipótesis del CHECKPOINT 2 para D4 — veredicto

> *"Captará 2008 y 2020 (crisis grandes y persistentes); fallará 2013 y 2018
> (correcciones rápidas; el gaussiano subestima las colas); la versión in-sample es
> NO causal."*

**Se cumple.** (1) In-sample capta 2008 (~100 %) y 2020 (~96 %); causalmente 2020/2022
se mantienen altas OOS (2008 no evaluable OOS por la ventana, ver arriba). (2) Falla
2013/2018 en ambas versiones (baja cobertura sostenida) → el supuesto gaussiano pierde
las correcciones rápidas. (3) La versión in-sample está claramente etiquetada como NO
causal y se separa de la causal comparable. El único matiz frente al enunciado de la
hipótesis: "capta 2008" es verificable solo in-sample; OOS 2008 queda fuera de la
ventana de evaluación causal (no es que falle, es que no se puede ver sin look-ahead).

## Fricción con el núcleo

Ninguna que requiera cambiar firmas del núcleo. La causalidad estricta intra-bloque
(Arreglo 2) se resolvió **en el detector**, sobrescribiendo `predict_online`/
`predict_proba` con filtrado forward (`detectors/_hmm_utils.py`), exactamente como el
docstring de `walk_forward` contempla. Observación:
- La cobertura NaN de 2008/2011 en la versión causal es comportamiento correcto de
  `crisis_coverage` (ventana fuera del rango OOS → NaN), no un defecto.
