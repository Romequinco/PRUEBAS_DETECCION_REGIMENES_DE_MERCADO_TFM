# ADR-003 — Causalidad de calendario, propagación de estado y ranking por detección

- **Estado:** Aceptada · 2026-09-29
- **Rama:** trabajo directo sobre `main`.
- **Origen:** revisión completa de los notebooks 00–07 ([`../revisiones/REVISION_2026-09-29.md`](../revisiones/REVISION_2026-09-29.md)).
- **Ámbito:** `src/features.py` (lags de publicación, alineación), `src/ingest/` (CAPE, Ken French,
  vol realizada), `src/evaluation.py` (walk-forward), `src/detectors/registry.py` (semillas),
  `src/benchmark.py` (ranking, huella de caché, ejecución paralela), `data/benchmark_spec.yaml`
  (solo comentarios), notebooks 02–07. **No** cambia ventanas, crisis, series del banco ni los
  detectores congelados de `capa1_exploracion/`.
- **Consecuencia:** invalida los resultados de `results/benchmark_v2/` y de la Fase E; el benchmark
  se re-ejecuta completo una vez con los tres cambios juntos.

<!-- BEGIN ubicacion_v2 -->
> **Ubicación tras ADR-004.** Esta ADR se redactó antes de la unificación; el texto conserva los
> nombres de entonces. Traducción: `src/features.py::LAG_PUBLICACION` → `regimenes.features.lags.LAG_PUBLICACION`;
> `src/evaluation.py` → `regimenes.evaluacion` (`walk_forward`, `metricas`); `src/detectors/registry.py` →
> `regimenes.detectores.registry`; `src/benchmark.py` → `regimenes.benchmark` (`ejecucion`, `cache`, `cli`)
> y `regimenes.evaluacion.ranking`; `bm.DETECTION_DEFAULTS` → `regimenes.evaluacion.ranking.DETECTION_DEFAULTS`;
> `python -m src.benchmark` → `python -m regimenes.benchmark`; `data/benchmark_spec.yaml` →
> `configs/benchmark_spec.yaml`; `results/benchmark_v2/` → `results/benchmark/` (p. ej.
> `results/benchmark/ranking_v2.csv`); `results/fusion_*` → `results/fusion/{d07_d08,d02_d06}/`;
> notebooks `04`/`05`/`06`/`07` → `04_protocolo_evaluacion`/`12_comparativa`/`13_fusion_d07_d08`/`14_fusion_d02_d06`
> (así, "05 §8" es hoy `12_comparativa` §8); detectores de `capa1_exploracion/` → `regimenes.detectores.f1_reglas` … `f7_deep`.
> "MS-VAR" (D05, `markov_switching_var`) no es un VAR sino un Markov-Switching univariante de media y
> varianza, y "D01 VIX" en la pista A es un umbral de volatilidad realizada (no hay VIX antes de 1990):
> ver [GLOSARIO](../GLOSARIO.md#nombres-de-detectores-que-confunden).
<!-- END ubicacion_v2 -->

---

## 1. Contexto

La revisión encontró tres problemas que **cambian métricas** y que, por tanto, no se corrigieron
durante la revisión, sino que se elevaron a decisión:

1. **Fuga de calendario.** El test de truncado prueba causalidad *computacional* (ningún cálculo en `t`
   usa filas `> t`), pero no causalidad de *calendario* (el dato fechado en `t` ya estaba publicado en
   `t`). Varias series mensuales de FRED son la media del mes M fechada el día 1 de M, y el lag macro
   de 1 mes era corto. Afectaba a 3 features del núcleo.
2. **Reinicio de estado en cada refit.** Los detectores con autómata de histéresis (D01, D02, D06,
   D10) arrancaban en calma al inicio de cada bloque OOS, recortando cobertura de forma artificial;
   D07 sí propagaba estado, por lo que la comparación no era justa.
3. **Ranking sesgado a la inactividad.** `rank_medio` promediaba 5 ejes de los que 3 (trampas,
   switching, estabilidad) los gana un detector que no hace nada: la línea base "siempre crisis" quedaba 4ª en ambas pistas
   ([revisión 2026-09-29](../revisiones/REVISION_2026-09-29.md), banco previo a esta ADR; con el banco
   re-ejecutado queda 5ª por `rank_medio` en ambas, `12_comparativa` §8).

## 2. Decisión

### 2.1 Datos: lags de publicación
Toda serie no diaria entra a las features con su lag de publicación, definido en una única tabla
(`src/features.py::LAG_PUBLICACION`, con la fuente de cada lag) y aplicado a la serie cruda **antes**
de cualquier transformación.

| Series | Lag | Motivo |
|---|---|---|
| GS10, GS5, GS1, TB3MS, FEDFUNDS, TBILL3M_MINUS_FEDFUNDS, MOODYS_BAA/AAA, BAAFFM | +1 mes | media del mes M fechada el día 1 |
| SHILLER CAPE, GW b/m | +1 mes | precio medio / fin de mes fechado el día 1 |
| WTI_SPOT_MONTHLY | +1 mes +10 días | media mensual, EIA con ~1 semana de retraso |
| INDPRO, PPI, CPI, HOUST, CFNAI, BIS, UNRATE, PAYEMS, MANEMP | +2 meses | publicación a mitad de M+1 (empleo: conservador) |
| UMCSENT, Philly Fed | +1 mes | se publican dentro del propio mes |
| GDP_GROWTH_QOQ, GDPC1 | +4 meses | avance del BEA a fin del mes siguiente al trimestre |
| ICSA | +5 días | sábado de referencia → jueves de publicación |

La alineación diaria pasa a ser *as-of* por columna (recupera `ICSA_z`, antes vacía en la pista B).
Se corrige la ingesta de CAPE (PE10 real), vol realizada y Ken French (todas las columnas): en
consecuencia `FF_industry_dispersion_z` es una dispersión real y `FF_RMW_z` sustituye al duplicado
`FF_MKT5_z`. Solo se re-descargaron esas 5 series; las otras 161 son idénticas byte a byte.

**Impacto en el núcleo:** solo cambian `credit_BaaAaa_mensual_z`, `term_spread_hist_z` e
`INDPRO_yoy_z` (corr 0,92–0,98 con la versión anterior), con el error concentrado en estrés
(2008-10-01: diferencia de 1,31 en z). El resto del núcleo diario es idéntico.

Un test de truncado **por fecha de publicación** verifica el pipeline de extremo a extremo.

### 2.2 Evaluación: propagación de estado
`walk_forward(..., context="auto")` predice cada bloque OOS sobre **[cola del train] + bloque** y
conserva solo las predicciones del bloque. El contexto por defecto es un año en la frecuencia del panel
(252 sesiones); `None` = train completo, `0` = protocolo anterior. Es causal: las filas de contexto son
anteriores al bloque y ya formaban parte del train del fold (tests de truncado en
`tests/test_walkforward_context.py`).

- Corrige el reinicio en calma de los autómatas de histéresis (D01, D02, D06, D10): en A, 54 de 75
  salidas de crisis de D01 caían en un borde de bloque; con contexto, 1 de 53.
- Un año basta: los paneles de D01/D02/D06 son idénticos a los obtenidos con el train completo.
- Los detectores sin estado (D03, D12) y los que ya anteponían el train como *burn-in* (D07, HMM,
  MS, D11) no cambian; D07 da el mismo panel byte a byte. D09 sí cambia (arranca con historia).
- Efecto medido aislado (datos anteriores a 2.1), cobertura media: D01 A 0,52→0,60, B 0,20→0,25;
  D02 A 0,41→0,46, B 0,18→0,29; D06 A 0,73→0,79, B 0,49→0,51; D07 sin cambios. A cambio, la tasa
  de falsas alarmas sube ligeramente.
- Semillas explícitas (`SEED = 42`) en D03, D04, D08, D09, D12. El ajuste final es determinista en
  una misma máquina; entre máquinas AIC/BIC pueden diferir por BLAS/hilos (no por la semilla).

### 2.3 Ranking: criterio de detección
El ranking principal de cada pista pasa a ser un criterio de **detección** explícito:

- **Evento detectado:** ≥ 3 sesiones OOS **consecutivas** marcadas como crisis dentro de su ventana
  [pico, suelo] (evita el *point-adjust* criticado por Kim et al. 2022 y Tatbul et al. 2018).
- **Score:** F1 entre la **precisión diaria** (1 − `false_alarm_rate`) y el **recall por evento**
  (*composite F-score*, Garg et al. 2021).
- **Niveles previos al score** (un detector de nivel peor nunca supera a uno de nivel mejor):
  0 elegible · 1 precisión que no supera la tasa base (lift ≤ 1; criterio de azar de Kaminsky,
  Lizondo y Reinhart 1998 y Alessi y Detken 2011) · 2 parpadeo (duración media de régimen < 5 o
  episodios de crisis < 3 sesiones) · 3 salida degenerada.
- **Desempate:** menor `switching_rate` y después mayor `label_stability`. La persistencia ya no se
  promedia con la detección.
- Umbrales explícitos en `bm.DETECTION_DEFAULTS` (hoy `regimenes.evaluacion.ranking.DETECTION_DEFAULTS`);
  sensibilidad documentada en 05 §8 (hoy [`12_comparativa`](../../notebooks/12_comparativa.ipynb) §8).
- Líneas base triviales (siempre crisis, siempre calma, azar diario, azar persistente) quedan al
  fondo por construcción, verificado por test.
- La frontera de Pareto recall/precisión se publica como vista complementaria. `rank_medio` se
  conserva como columna legacy para trazar el cambio.

### 2.4 Reproducibilidad
- Huella de caché por **contenido** (no bytes) de parquet y YAML, `.py` normalizado a LF, y hash de
  la copia de `detector_base` que realmente importan los detectores.
- Semillas fijadas en `registry.py` para los detectores estocásticos (AIC/BIC deterministas).
- Ejecución paralela por detector: `python -m src.benchmark --track A B --jobs N` (hoy
  `python -m regimenes.benchmark --track A B --jobs N`).
- Metadatos `data/raw/provenance.json` y `coverage_report.csv` regenerados desde los parquet locales
  (`--offline`).

## 3. Alternativas descartadas
- **Documentar las tres cosas como limitaciones sin corregir:** coste nulo, pero contradice el
  principio rector del proyecto (causal, comparable) y deja un ranking que premia la inactividad.
- **Solo lag para las mensuales de mercado:** dejaba una fuga conocida en `INDPRO_yoy_z` (núcleo).
- **Normalizar los 5 ejes contra líneas base / solo Pareto:** más difícil de explicar o sin ganador
  único para alimentar la fusión.
- **Vintages en tiempo real (ALFRED):** más exacto, pero coste alto; queda como limitación declarada.

## 4. Resultado de la re-ejecución (2026-09-29)

Benchmark completo re-ejecutado (24/24 combinaciones `ok`, ~1 h 55 min con 9 procesos).

| Puesto | Pista A | Pista B |
|---|---|---|
| 1 | D10 turbulencia (0,614) | D05 MS-VAR (0,647) |
| 2 | D01 VIX (0,587) | D06 GARCH-t (0,597) |
| 3 | D03 GMM (0,565) | D01 VIX (0,505) |
| 4 | D02 riesgo compuesto (0,559) | D02 riesgo compuesto (0,501) |
| 5 | D05 MS-VAR (0,546) | D04 HMM gaussiano (0,497) |

(score F1 entre paréntesis; tabla completa en `results/benchmark_v2/ranking_v2.csv`, hoy
`results/benchmark/ranking_v2.csv`). D07, 1º en A (5º en B) con el `rank_medio` del banco previo a
esta ADR, cae al 8º (A) y 7º (B) del ranking de detección: era persistente, no preciso. (Con el
banco re-ejecutado su `rank_medio` le daría el 2º en A y el 5º en B, columna `puesto_legacy`.) En B, D09 no supera la precisión del
azar y D12 parpadea.

Fase E: D2+D6 sigue mejorando a sus dos sensores, pero D2 y D7 quedan empatados como alerta, en A
D7+D6 y D7 solo la superan, y ningún aviso anticipa el inicio de una crisis. La regla a congelar para
el pseudolive queda como decisión abierta.

## 5. Consecuencias
- `results/benchmark_v2/` y `results/fusion_*` se regeneran; el veredicto D2+D6 de la Fase E se
  **re-evalúa** con el banco nuevo.
- Las métricas pueden bajar respecto a la v2 anterior: la anterior tenía ventaja indebida (fuga) y
  desventaja indebida (reinicio). Ninguna comparación con los números previos a esta ADR es válida.
