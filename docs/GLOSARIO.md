# Glosario — conceptos del proyecto (fuente canónica)

> Definiciones **echadas a tierra** de los términos que gobiernan todo el repo. Es la **fuente
> única**: los notebooks (`00`/`01`/`02`) y el resto de docs enlazan aquí en vez de redefinir.

---

## Las dos pistas (y `ambas` / `validacion`)

Cada serie/feature se asigna a una **pista** = *para qué banco de pruebas sirve*. Nacen de la
decisión [`ADR-001`](decisions/ADR-001-rebase-datos.md): **no se puede tener a la vez máxima historia
y máxima riqueza de features**, así que se construyen dos bancos en paralelo. Las ventanas y recuentos
de esta tabla fueron **reajustados por [`ADR-002`](decisions/ADR-002-ajuste-ventanas.md)** (2026-07-20):
el fin de ventana pasó a estar gobernado por la serie diaria más fresca (nunca una mensual) y el pool
de features candidatas creció de 35 a 106 series únicas (64% de las 166 descargadas).

| Pista | Qué es | Ventana del banco | Nº features | Nº crisis | Objetivo |
|---|---|---|:---:|:---:|---|
| **A — espina histórica + curva de tipos** | equity + vol realizada + factores + crédito/macro profundos + curva de tipos completa (DGS5/10, T10YFF) | **1962-01-02 → 2026-05-29** (gobiernan DGS10/DGS5/T10YFF al inicio; FF_FACTORS_3_DAILY al fin) | **41** | **18** | **potencia estadística** — se sacrifican conscientemente las 4 crisis con pico < 1962 (incluida la Gran Depresión) a cambio del bloque de curva completo |
| **B — panel rico multi-activo** | espina de A **+** crédito HY/IG, curva/breakevens/velocidad, complejo de vol, 9 sectores, 11 índices de amplitud (breadth), FX/commodities | **2007-04-11 → 2026-05-29** (gobierna HYG_CREDIT al inicio; FF_FACTORS_3_DAILY al fin — **el mismo fin que A, a propósito**) | **106** | 10 | **riqueza / discriminación** — separar mejor y **atacar el punto ciego de 2013**; mover el inicio de 2003 a 2007 no cuesta ninguna crisis |
| **ambas** | serie que existe en las dos ventanas (p. ej. VIX, pendientes de curva); por construcción **A ⊆ B** (toda serie viva en 1962 lo está en 2007) | — | — | — | se **cuenta una sola vez** (dedup) pero alimenta los dos bancos |
| **validacion** | índices de estrés ya hechos y labels (OFR FSI, NFCI, NBER) | — | — | — | **ground truth laxo** para *juzgar* regímenes — **nunca** entra como feature |

> **`pista` y `rol` son ejes independientes.** `pista` = a qué banco sirve; `rol` = qué papel juega la
> serie (abajo). Por eso el recuento de "validacion" difiere según el eje (25 series con
> `pista=validacion` vs 21 con `rol=validation`).
>
> ⚠️ **El campo `pista` de `data/catalog.yaml` (por serie) NO se reescribió con ADR-002** — sigue
> reflejando la clasificación manual original (A≈1927+, B≈2003+) y es solo descriptivo del universo
> declarado. La fuente de verdad **operativa** (qué serie entra en qué ventana congelada) es
> exclusivamente [`data/benchmark_spec.yaml`](../data/benchmark_spec.yaml).

---

## Los cinco roles

El campo `rol` (en `data/catalog.yaml`) dice **qué papel** juega cada serie. No es un ranking de
calidad: es una función.

| Rol | Qué es (echado a tierra) | Ejemplo |
|---|---|---|
| **`spine`** | **columna vertebral** de la pista: la serie imprescindible que la define y le da su historia | `SP500` (Pista A), `DGS10` (curva) |
| **`core`** | **feature principal** del pool: entra al detector con peso propio | `VIX`, `MOVE`, `MOODYS_BAA_AAA_SPREAD` |
| **`enricher`** | **enriquecedor opcional**: añade matiz, no es imprescindible | sectores SPDR, breakevens, VVIX |
| **`fallback`** | **sustituto redundante** de una serie ya presente: solo se usa si falla la fuente primaria (regla de dedup del catálogo) | `GOLD_FUT` (fallback de `GOLD_GLD`) |
| **`validation`** | **ground truth** para evaluar/etiquetar — **jamás feature** | `OFR_FSI`, `NFCI`, `NBER_RECESSION_DAILY` |

Recuento actual: spine 23 · core 44 · enricher 62 · fallback 24 · validation 21 (sobre 166 series
descargadas; 174 declaradas, 8 no descargables y declaradas).

---

## Regla de oro anti-fuga (la más importante)

Una serie de **`rol=validation`** (índices de estrés, recesión NBER) es *ground truth*: sabe "esto
**fue** una crisis". **Nunca** puede entrar a la vez como **feature** y como **etiqueta** — sería
**fuga de información** (el detector "adivinaría" mirando la respuesta). Por eso `validation` solo se
usa para **juzgar** los detectores en la Fase D, jamás en la matriz de features.

## Causalidad (no look-ahead)

Toda feature es **causal**: en el instante `t` solo usa estadísticos de datos `≤ t` (z-score
*expanding*/*rolling*, nunca de muestra completa — ese fue el error de la tarea previa). Se verifica
con `assert_causal` (truncar el futuro y recomputar debe dar `max|Δ| = 0`). Se demuestra en
[`02_diseno_preprocesado.ipynb`](../notebooks/02_diseno_preprocesado.ipynb) §4-5.

## El banco congelado (benchmark)

[`data/benchmark_spec.yaml`](../data/benchmark_spec.yaml) **congela**, por pista, la ventana + las
features + las etiquetas (crisis, falsos positivos, troughs). Es la **variable controlada** de la
Fase D: un detector puede cambiar su algoritmo, pero **no** estas ventanas/etiquetas.

---

*El pipeline del repo:* `00_descarga` (datos) → `01_eda` (análisis) → `02_diseno_preprocesado`
(decisiones) → `03_preprocesado` (features, pendiente) → **Fase D** (detectores).
