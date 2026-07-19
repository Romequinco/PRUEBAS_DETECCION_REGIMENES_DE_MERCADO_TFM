# Glosario — conceptos del proyecto (fuente canónica)

> Definiciones **echadas a tierra** de los términos que gobiernan todo el repo. Es la **fuente
> única**: los notebooks (`00`/`01`/`02`) y el resto de docs enlazan aquí en vez de redefinir.

---

## Las dos pistas (y `ambas` / `validacion`)

Cada serie/feature se asigna a una **pista** = *para qué banco de pruebas sirve*. Nacen de la
decisión [`ADR-001`](decisions/ADR-001-rebase-datos.md): **no se puede tener a la vez máxima historia
y máxima riqueza de features**, así que se construyen dos bancos en paralelo.

| Pista | Qué es | Ventana del banco | Nº crisis | Objetivo |
|---|---|---|:---:|---|
| **A — espina histórica profunda** | pocas features robustas (S&P 500 + vol realizada + factores + crédito/macro profundos) | **1927+** (la gobierna el S&P 500 diario `^GSPC`; incorpora crédito/macro mensual desde 1913-1919) | **22** | **potencia estadística** — muchas crisis atacan el "n≈4" de la Capa 1 |
| **B — panel rico multi-activo** | muchas features (crédito, curva, breakevens, vol, sectores, FX) | **2003+** (la gobiernan los breakevens TIPS; el sub-banco vol-of-vol arranca en 2007) | 10 | **riqueza / discriminación** — separar mejor y **atacar el punto ciego de 2013** |
| **ambas** | serie que existe en las dos ventanas (p. ej. VIX, pendientes de curva) | — | — | se **cuenta una sola vez** (dedup) pero alimenta los dos bancos |
| **validacion** | índices de estrés ya hechos y labels (OFR FSI, NFCI, NBER) | — | — | **ground truth laxo** para *juzgar* regímenes — **nunca** entra como feature |

> **`pista` y `rol` son ejes independientes.** `pista` = a qué banco sirve; `rol` = qué papel juega la
> serie (abajo). Por eso el recuento de "validacion" difiere según el eje (25 series con
> `pista=validacion` vs 21 con `rol=validation`).

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
