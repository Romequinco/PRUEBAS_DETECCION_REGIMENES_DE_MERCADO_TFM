# docs/ — índice de la documentación

Único índice de la documentación del TFM. Aquí está el mapa de carpetas y el orden de lectura según quién lee; el
contenido vive en cada documento. Qué es el TFM y en qué estado está cada fase: [`tfm/README.md`](tfm/README.md).
Lo que no está en `docs/` es código (`src/regimenes/` para la Fase 1, `src/ingesta/` para la Fase 2), configuración
(`configs/`), datos (`data/`), resultados (`results/`) o notebooks (`notebooks/`).

Las cifras (puestos del ranking, recuentos de series, métricas) no se copian en este índice: viven en los
notebooks, en `results/` y en los documentos que las comentan.

## Mapa de carpetas

```
docs/
├── README.md            este índice
├── GLOSARIO.md          definiciones de los términos del proyecto
├── references.bib       bibliografía única del TFM
├── decisions/           ADR-001 … ADR-005 y ADR-007 (ADR-006 reservada a la decisión final de régimen)
├── tfm/                 visión del TFM: arquitectura, cliente y carteras, líneas abiertas, documentos de origen
├── estado_del_arte/     estado del arte: dossier general, detectores y fuentes de datos, con su catálogo
├── fase1/               Fase 1 (régimen): pipeline, mapa del código, EDA, teoría y detectores por familia, sintéticos
├── fase2/               Fase 2 (datos de los agentes): reparto en cuatro temas, plan de datos, datos alternativos
└── historia/            Capa 1, narrativa v1 de los detectores, estado del arte v1 y tarea previa
```

### `tfm/` — el sistema completo
- [`tfm/README.md`](tfm/README.md) — visión del TFM, cinco fases y su estado real, dónde vive cada fase.
- [`tfm/arquitectura.md`](tfm/arquitectura.md) — fuentes, índices, tools, TimeGate, anatomía de un agente, catálogo de
  tools y contrato provisional de `get_regimen(fecha)`.
- [`tfm/cliente_y_carteras.md`](tfm/cliente_y_carteras.md) — cliente minorista, perfil, dos carteras, memoria de
  cartera y botón de cataclismo.
- [`tfm/lineas_abiertas.md`](tfm/lineas_abiertas.md) — System One, Briefly, elementos de la propuesta por analizar y
  ubicación del código de las fases 3–5.
- [`tfm/fuentes/`](tfm/fuentes/) — [`propuesta_v2.pdf`](tfm/fuentes/propuesta_v2.pdf) (propuesta original, parcialmente
  superada por ADR-005) y [`reunion_tutor.pdf`](tfm/fuentes/reunion_tutor.pdf) (presentación al tutor, octubre de 2026).

### `decisions/` — por qué se hizo cada cosa
- [ADR-001](decisions/ADR-001-rebase-datos.md) — se congela la Capa 1 y se re-basa la capa de datos en dos pistas.
- [ADR-002](decisions/ADR-002-ajuste-ventanas.md) — ventanas del banco congelado y pool de features por pista.
- [ADR-003](decisions/ADR-003-causalidad-calendario-estado-ranking.md) — lags de publicación, propagación de estado
  y ranking por detección.
- [ADR-004](decisions/ADR-004-unificacion.md) — un solo repositorio y un solo paquete (`regimenes`); notebooks por familia.
- [ADR-005](decisions/ADR-005-reencuadre-tfm-multiagente.md) — el TFM es un sistema multi-agente RAG y el régimen es
  su Fase 1.
- [ADR-007](decisions/ADR-007-fase2-ingesta-y-docs.md) — la Fase 2 vive en `src/ingesta` (cuatro temas, una rama
  cada uno) y esta estructura de `docs/`.

### `estado_del_arte/` — qué existe fuera del proyecto
- [`estado_del_arte/README.md`](estado_del_arte/README.md) — índice de los tres documentos y de su catálogo.
- [`estado_del_arte/estado_del_arte.md`](estado_del_arte/estado_del_arte.md) — dossier general del TFM (sistemas
  multi-agente, RAG financiero, detección de regímenes, carteras).
- [`estado_del_arte/detectores.md`](estado_del_arte/detectores.md) — familias de detectores de régimen y sus solapes.
- [`estado_del_arte/fuentes_datos.md`](estado_del_arte/fuentes_datos.md) — series para detectar regímenes, cuáles se
  eligieron y su reparto por pista.
- `obras.csv`, `competidores_2026.json`, `fichas.json` — catálogo de obras y fichas; `catalogo.py` genera
  `catalogo.html` (no versionado).

### `fase1/` — el régimen de mercado (`src/regimenes`, notebooks `00`–`23`)
- [`fase1/README.md`](fase1/README.md) — pipeline `00`–`23`, orden de ejecución, mapa del código y de las rutas, e
  índice de detectores D01–D13 por familia.
- [`fase1/eda.md`](fase1/eda.md) — informe EDA (figuras en [`fase1/figs/`](fase1/figs/)).
- Teoría de cada familia con sus detectores: [F1 reglas](fase1/F1_reglas.md) · [F2 clustering](fase1/F2_clustering.md) ·
  [F3 HMM](fase1/F3_hmm.md) · [F4 Markov-Switching](fase1/F4_markov_switching.md) ·
  [F5 GARCH](fase1/F5_volatilidad_garch.md) · [F6 change-point](fase1/F6_change_point.md) ·
  [F7 redes](fase1/F7_redes_neuronales.md).
- [`fase1/sinteticos.md`](fase1/sinteticos.md) — generadores sintéticos, validación y laboratorio (notebooks `18`–`21`).
- Configuración congelada de la Fase 1: [`../configs/catalog.yaml`](../configs/catalog.yaml) (universo de series y
  catálogo de crisis) y [`../configs/benchmark_spec.yaml`](../configs/benchmark_spec.yaml) (el banco congelado).

### `fase2/` — los datos de los agentes (`src/ingesta`)
- [`fase2/README.md`](fase2/README.md) — los cuatro temas, ramas, dónde escribe cada tema y contrato común.
- [`fase2/plan_datos.md`](fase2/plan_datos.md) — plan de datos vigente: arquitectura, TimeGate, qué entra por tema,
  tratamiento RAG, look-ahead del LLM, evaluación y preguntas abiertas.
- [`fase2/datos_alternativos.md`](fase2/datos_alternativos.md) — candidatas: Pelosi tracker, Polymarket, EWS y otras.
- README de cada tema: [mercados y series](../src/ingesta/mercados_series/README.md) ·
  [bancos centrales y riesgo](../src/ingesta/bancos_riesgo/README.md) · [empresas](../src/ingesta/empresas/README.md) ·
  [noticias y alternativos](../src/ingesta/noticias_alt/README.md).

### `historia/` — de dónde venimos
- [`historia/capa1/README.md`](historia/capa1/README.md) — la Capa 1 (12 detectores v1): decisiones, hallazgos, qué
  falló, correspondencia de rutas y cómo recuperarla del tag `capa1-final`. Sus métricas v1, en
  [`historia/capa1/resultados/`](historia/capa1/resultados/).
- [`historia/fichas_v1.md`](historia/fichas_v1.md) — narrativa v1 de los detectores D01–D12.
- [`historia/estado_del_arte_v1.md`](historia/estado_del_arte_v1.md) — estado del arte de la Capa 1.
- [`historia/tarea_previa.md`](historia/tarea_previa.md) — la práctica previa (HMM de dos estados) que originó la Fase 1.

### Términos y bibliografía
- [`GLOSARIO.md`](GLOSARIO.md) — pistas, roles, causalidad, regla anti-fuga, métricas del ranking, sintéticos y
  términos del TFM.
- [`references.bib`](references.bib) — bibliografía única, deduplicada. Cuando una obra ya estaba con otra clave, la
  entrada antigua es la canónica y la clave nueva del estado del arte es un alias (campo `ids`): 16 alias, más 3
  obras que ya tenían la misma clave. Con BibTeX clásico, que no entiende `ids`, se cita la clave canónica.

---

## Orden de lectura

Este es el único orden de lectura del repositorio (el `README.md` raíz remite aquí). Elige el recorrido según para qué
lees.

**Equipo (empezar a trabajar en la Fase 2).**
[`tfm/README.md`](tfm/README.md) → [ADR-005](decisions/ADR-005-reencuadre-tfm-multiagente.md) →
[ADR-007](decisions/ADR-007-fase2-ingesta-y-docs.md) → [`fase2/README.md`](fase2/README.md) →
[`fase2/plan_datos.md`](fase2/plan_datos.md) → README de tu tema en `src/ingesta/` →
[`tfm/arquitectura.md`](tfm/arquitectura.md) (catálogo de tools). El [GLOSARIO](GLOSARIO.md) se consulta cuando
aparece un término.

**Tutor (estado del trabajo).**
[`tfm/README.md`](tfm/README.md) → [`tfm/arquitectura.md`](tfm/arquitectura.md) →
[`tfm/cliente_y_carteras.md`](tfm/cliente_y_carteras.md) → [`fase1/README.md`](fase1/README.md) (qué está hecho en la
Fase 1) → [`fase2/plan_datos.md`](fase2/plan_datos.md) (preguntas abiertas) → [`tfm/lineas_abiertas.md`](tfm/lineas_abiertas.md).

**Memoria del TFM (redacción, de principio a fin).**
[`estado_del_arte/README.md`](estado_del_arte/README.md) → [`historia/capa1/README.md`](historia/capa1/README.md) →
[ADR-001](decisions/ADR-001-rebase-datos.md) → [`estado_del_arte/fuentes_datos.md`](estado_del_arte/fuentes_datos.md) →
[`fase1/eda.md`](fase1/eda.md) → [ADR-002](decisions/ADR-002-ajuste-ventanas.md) → `benchmark_spec.yaml` →
[ADR-003](decisions/ADR-003-causalidad-calendario-estado-ranking.md) → [ADR-004](decisions/ADR-004-unificacion.md) →
teoría por familia (`fase1/F1`…`F7`) → notebooks `04` (termómetros) → `05` → `06`–`13` → `14` → `15`–`17` (fusión) →
[`fase1/sinteticos.md`](fase1/sinteticos.md) y `18`–`20` → `21`–`23` (aumento, decisión final y pseudolive) →
[`fase2/plan_datos.md`](fase2/plan_datos.md).

*Visita rápida:* `tfm/README.md` → ADR-005 → `fase1/README.md` → `fase2/README.md`.

El orden de **lectura** de los notebooks (familias antes que la comparativa) no es el de **ejecución** tras un
benchmark nuevo (`05` → `14` → `06`–`12`): ver [`fase1/README.md`](fase1/README.md#orden-de-ejecución).

---

## Solo local

Hay material de trabajo que **no se versiona** (la carpeta está en el `.gitignore` de la raíz) y vive en `docs/tfm/internos/` en la
copia de quien lo genera:

| Carpeta | Qué contiene | Por qué no se versiona |
|---|---|---|
| `tfm/internos/presentacion/` | Presentaciones (HTML y PDF) y su script de construcción | Material de exposición, cambia con cada reunión |
| `tfm/internos/plan_datos/` | Plan de datos completo en LaTeX con figuras y resumen de fuentes | Documento de trabajo; su contenido vigente está destilado en [`fase2/plan_datos.md`](fase2/plan_datos.md) |
| `tfm/internos/estado_del_arte/` | Material de trabajo del estado del arte: el diario de su construcción (`_journal_estado_del_arte.jsonl`), la bibliografía de trabajo previa a la fusión (`referencias.bib`, 273 entradas), el catálogo navegable generado (`catalogo.html`) y el guion de la diapositiva del estado del arte (`slide1_estado_del_arte.md`) | Registro de proceso y salidas generadas; el resultado está en [`estado_del_arte/`](estado_del_arte/README.md) y la bibliografía fusionada, en [`references.bib`](references.bib) |

Si un documento versionado necesita algo de `internos/`, se destila a Markdown en su carpeta y se enlaza ese destilado,
nunca la ruta local.
