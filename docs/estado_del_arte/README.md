# Estado del arte

Tres documentos, de lo general a lo particular, y un catálogo navegable de las obras.

| Documento | Qué cubre | Corte |
|---|---|---|
| [`estado_del_arte.md`](estado_del_arte.md) | Estado del arte general del TFM: régimen y alerta temprana, agentes LLM en inversión, RAG financiero, look-ahead y datos point-in-time, evaluación, datos alternativos y sintéticos, industria y regulación; posicionamiento frente a los sistemas más cercanos y qué se adopta en cada fase (P1–P5). 273 obras. | 2026-10-13 |
| [`detectores.md`](detectores.md) | Detectores de régimen de la Fase 1 por familia (F1–F7): tabla comparativa, síntesis, solapes, candidatas y qué se implementó (D01–D13). | Capa 1 · etapa 2, actualizado |
| [`fuentes_datos.md`](fuentes_datos.md) | Fuentes de datos gratuitas para detectar regímenes: qué existe, qué se eligió, desde cuándo y para qué pista; corregido con el [EDA](../fase1/eda.md). | Descargas 2026-07-18, corregido 2026-10-14 |

**Numeración.** P1–P5 son las fases del TFM (P1 régimen, P2 ingesta/RAG/tools, P3 agentes,
P4 cartera y memo, P5 evaluación y backtest); F1–F7 son las familias de detectores y F8 los
generadores sintéticos ([`../fase1/sinteticos.md`](../fase1/sinteticos.md)).

## Datos del estado del arte general

| Fichero | Contenido |
|---|---|
| [`obras.csv`](obras.csv) | Las 273 obras de `estado_del_arte.md`, en el orden de sus citas: la fila *n* es la cita [*n*] del dossier. Columnas: bloque, clave canónica en [`references.bib`](../references.bib) (para citar en la memoria), título, autores, año, venue, tipo, seminal, URL, fase P1–P5 donde se usan y rasgos (régimen, multi-agente, RAG con citas, anti-look-ahead, cartera, datos alternativos). |
| [`competidores_2026.json`](competidores_2026.json) | Pasada extra de competidores de junio–octubre de 2026 (§9.bis), con rasgos y nivel de amenaza. |
| [`fichas.json`](fichas.json) | Ficha breve de cada obra (clave de `references.bib`, qué hace, diferencia con el TFM, fase, rasgos), redactada por los agentes buscadores a partir del resumen y la página de la obra. Es la entrada del catálogo. |
| [`catalogo.py`](catalogo.py) | Genera `catalogo.html`. |

Las referencias BibTeX de las 273 obras están en la bibliografía única del repo,
[`../references.bib`](../references.bib), con clave autor-año (bloque «Estado del arte del
TFM»). Las 19 obras que ya estaban con otra clave conservan su entrada antigua y la clave nueva
es un alias; el mapa está en la cabecera del `.bib`.

## Catálogo navegable

`catalogo.html` es una página con una tarjeta por obra (298: las 273 del documento y 25 de la
pasada extra), con buscador y filtro por frente: qué hace, en qué se diferencia del TFM, en
qué secciones se cita, qué se adopta de ella y si es un hito de la línea temporal o una fila
de la matriz de posicionamiento (§9). Se genera y no se versiona (`.gitignore`):

```bash
python docs/estado_del_arte/catalogo.py
```

Lee `estado_del_arte.md`, `fichas.json` y `competidores_2026.json` de esta carpeta y escribe
`catalogo.html` junto a ellos. Solo usa la biblioteca estándar de Python.

## Historia

La lista de detectores que se propuso en la Capa 1 · etapa 2, antes de implementarlos, está
en [`../historia/estado_del_arte_v1.md`](../historia/estado_del_arte_v1.md).
