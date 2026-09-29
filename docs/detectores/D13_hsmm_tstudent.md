# D13 — `hsmm_tstudent` (ablación A1) · Familia F3 (HMM semi-Markov)

<!-- BEGIN ubicacion_v2 -->
> **Ubicación tras ADR-004.** Código: `src/regimenes/detectores/f3_hmm/hsmm_tstudent.py`
> (clase `HSMMTStudent`, hereda de `HMMTStudent` de D08) · Notebook v1:
> `capa1_exploracion/notebooks/A1_hsmm_ablation.ipynb` (tag `capa1-final`) · Métricas v1:
> `docs/historia/capa1/resultados/ablation_hsmm/metrics_d13_hsmm.csv` y
> `metrics_d8_reference.csv` (README de la ablación en la misma carpeta) · Notebook v2 de familia:
> [`notebooks/07_familia_F3_hmm.ipynb`](../../notebooks/07_familia_F3_hmm.ipynb) (sección de
> ablación) · Teoría: [`docs/teoria/F3_hmm.md`](../teoria/F3_hmm.md).
> Ficha nueva (no existía en `capa1_exploracion/memory/detectors/`): se redacta a partir del
> README de la ablación y del markdown/salidas del notebook A1.
<!-- END ubicacion_v2 -->

> Un *Hidden Semi-Markov Model* (HSMM) relaja el supuesto temporal más restrictivo del HMM
> de D8: en un HMM la duración de cada régimen es **geométrica y sin memoria** (hazard de
> salida constante `h(d) = 1 − A_ss`). D13 modela explícitamente cuánto dura cada estado, de
> modo que la probabilidad de salir depende de la **edad del episodio**
> (`h_s(d) = P(D = d | D ≥ d)`). Bibliografía: `hmm_bullabulla2006`, `hmm_bulla2011`,
> `hmm_rabiner1989`, `guidolintimmermann2007`.

## Implementado

**Experimento controlado frente a D8.** Mismas 7 features puente de la Capa 1, misma
ventana (2007-07-06 → 2026-08-26), mismo rango de K (`{3, 4}`), mismas emisiones t-Student
multivariantes, mismo `train_size = 252·5` y misma frecuencia de refit (`step = 126`). Solo
cambia la dinámica temporal:

- **D8 — t-HMM:** permanencia geométrica inducida por la diagonal de la matriz de transición.
- **D13 — t-HSMM:** duración **Gamma desplazada** por estado, truncada en `D_MAX = 126`
  sesiones (la última celda acumula la cola, sin perder masa) y **regularizada** hacia la
  geometría del HMM de arranque (`duration_prior_strength = 12` episodios equivalentes;
  `transition_prior_strength = 2` sobre la cadena de saltos).

**Estimación en dos etapas (no EM conjunto).** (1) Se ajusta el HMM-t de D8 sobre el train;
(2) se decodifica el train y se estiman las duraciones Gamma y la cadena **embebida** (diagonal
cero: la permanencia la da la PMF de duración, no las auto-transiciones). Consecuencia: el
BIC de D13 es **aproximado** y sirve para elegir K dentro de D13, no como test frente a D8.

**Causalidad.** En walk-forward se filtra el estado ampliado
`P(S_t, edad_t | observaciones ≤ t)` con parámetros ajustados solo en el pasado; `p_crisis`
marginaliza la edad. La ruta segmental completa (Viterbi HSMM) solo se usa como diagnóstico
**in-sample, NO causal**.

**Coste.** Mayor que D8 (filtro sobre K·D_MAX estados ampliados). Por eso, y por el
resultado, D13 **no entra en el benchmark v2**: no tiene `DetectorSpec` en
`regimenes.detectores.registry` (solo aparece en `_FAMILIA_POR_MODULO` para poder importarse) y
`hsmm_tstudent.py` está excluido de la huella de caché del benchmark
(`regimenes.benchmark.cache`).

## Descubierto (notebook A1, Capa 1)

Cifras copiadas de las salidas del notebook A1 y de los dos CSV de
`docs/historia/capa1/resultados/ablation_hsmm/`.

**Selección de K (BIC aproximado).** K = 3: BIC ≈ 28 205; **K = 4: BIC ≈ 24 296** → se
despliega K = 4 (calma · leve · corrección · crisis). ν por estado canónico decreciente
[10.55, 8.06, 3.94, 2.42] y duración esperada explícita E[D] = [33.0, 21.1, 43.7, 71.5]
sesiones (la geometría del HMM de arranque daba [31.9, 19.6, 48.1, 123.4]). Monotonía de
volatilidad verificada in-sample y en walk-forward.

**Walk-forward causal** (OOS 2012-07-31 → 2026-08-26, n = 3432, 28 folds; 2008 y 2011 caen en
el train, como en D4/D8).

| Métrica (OOS, misma ventana) | D8 t-HMM | D13 t-HSMM | Lectura |
|---|---:|---:|---|
| *Switching rate* | 0.0475 | 0.0495 | empeora |
| Duración media de régimen | 20.93 | 20.07 | empeora |
| Tasa de falsas alarmas | 0.411 | 0.414 | sin cambio material |
| Estabilidad de etiqueta | 0.896 | 0.898 | mejora insignificante |
| Cobertura COVID-2020 | 0.660 | 0.680 | +2 pp |
| Cobertura Inflación-2022 | 0.643 | 0.623 | −1.9 pp |
| Activación en trampa 2013 / Q4-2018 | 0.000 / 0.017 | 0.000 / 0.017 | idéntico |
| BIC (D13 aproximado) | 24 129 | 24 296 | D13 peor (con cautela) |
| *Lead/lag* COVID / Inflación | −149 / −171 | −149 / −171 | idéntico |

**Criterios declarados antes de mirar los resultados:** (1) menos *switching* → **falla**;
(2) mayor duración media → **falla**; (3) la cobertura media de crisis OOS no cae más de 10 pp →
OK (65.2 % vs 65.1 %); (4) la tasa de falsas alarmas no sube más de 10 pp → OK.
**Veredicto: hipótesis no apoyada.** La duración explícita produce una segmentación
prácticamente idéntica a D8 y no mejora las dos variables que motivaban el modelo.

## Decisión y papel en v2

- **D13 no se selecciona**: resultado negativo informativo. Por parsimonia, rendimiento
  equivalente y menor complejidad, **D8 `hmm_tstudent_4s` sigue siendo el representante de
  F3** (ver [`D08_hmm_tstudent.md`](D08_hmm_tstudent.md)).
- En v2 se conserva como **ablación** dentro del notebook de familia F3; no tiene fila en
  `results/benchmark/ranking_v2.csv` ni en `metrics_master_v2.csv`, por lo que esta ficha no
  lleva la tabla "Resultados v2 (ADR-003)" del resto.
- Trabajo futuro solo si la duración de régimen pasa a ser pregunta central: sensibilidad a
  `D_MAX` y a la fuerza del prior, y un EM HSMM conjunto.

## Fricción con el núcleo

Ninguna. `HSMMTStudent` reutiliza `t_log_emission` de `_hmm_t_utils.py` y la API de
`HMMTStudent`; el orden económico de estados lo fija el núcleo común (vol-primario).
