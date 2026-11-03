# ADR-007 — La Fase 2 vive en `src/ingesta`; estructura de `docs/` por fases

- **Estado:** Aceptada · 2026-10-14 (decisión del equipo).
- **Completa:** [ADR-005](ADR-005-reencuadre-tfm-multiagente.md) §3, que dejaba abierto dónde vivirían las fases 2–5:
  aquí se decide para la **Fase 2**; las fases 3–5 siguen abiertas ([`tfm/lineas_abiertas.md`](../tfm/lineas_abiertas.md) §4).
- **Sustituye:** el árbol de `docs/` de [ADR-004](ADR-004-unificacion.md) §2 (punto 4) y §3 (el resto de ADR-004 sigue vigente).
- **Numeración:** ADR-006 sigue reservada a la decisión final del detector de régimen (notebook `22_decision_final`).

---

## 1. Contexto

1. **La Fase 2 empezó antes de cerrar la Fase 1.** Con el plan de datos ya medido, el equipo se repartió la
   ingesta de los agentes en cuatro temas, con un responsable por tema. Hacía falta un sitio para su código que no
   tocara el de la Fase 1.
2. **La Fase 1 tiene una huella de caché frágil.** El benchmark de los detectores guarda sus resultados por huella de
   contenido: el código de `regimenes.detectores` y `regimenes.evaluacion`, `configs/benchmark_spec.yaml`, los paneles
   procesados y las versiones de numpy, pandas, torch y otras librerías numéricas. Cambiar cualquiera de ellos obliga
   a re-ejecutar el benchmark completo (horas de cómputo). Las dependencias de la Fase 2 (embeddings, bases
   vectoriales, torch con CUDA) chocan con eso.
3. **`docs/` había crecido por capas.** La estructura de ADR-004 (`teoria/`, `detectores/`, `datos/`, `context/`) era
   solo de la Fase 1; la visión del TFM se había copiado en cuatro sitios (índice, GLOSARIO, ADR-005 y la tarea
   previa); el índice y `tfm/README.md` se presentaban a la vez como «fuente de verdad»; y la teoría de cada familia y
   la ficha de cada detector contaban lo mismo en dos ficheros. Con la Fase 2 en marcha, el árbol no tenía sitio para
   ella ni para el estado del arte general del TFM.

## 2. Decisión

### 2.1 Dónde vive la Fase 2

1. **Código:** paquete `src/ingesta/` en este repositorio, junto a `regimenes` y sin importarlo ni modificarlo. Un
   subpaquete por tema: `mercados_series`, `bancos_riesgo`, `empresas` y `noticias_alt`; lo compartido, en
   `src/ingesta/comun/` (esquema `Documento`, fichas de fuente, cliente HTTP educado, rutas).
2. **Ramas:** una rama por tema (`fase2/t1-mercados-series`, `fase2/t2-bancos-riesgo`, `fase2/t3-empresas`,
   `fase2/t4-noticias-alt`), con un PR a `main` por pieza cerrada. Lo común solo cambia con un PR pequeño a `main`.
   **Excepción, por decisión expresa del responsable del Tema 1:** la reorganización de `docs/` de esta ADR (§2.2), los ajustes de lo común que
   la acompañan (`docs/fase2/README.md`, enlaces de los README de los otros temas) y esta misma ADR se han hecho en la
   rama del Tema 1 (`fase2/t1-mercados-series`) a propósito, no en un PR aparte. Todo llegará a `main` cuando se
   unifiquen las ramas de la Fase 2; hasta entonces, quien trabaje en otra rama verá en `main` el árbol anterior.
3. **Salidas:** tests en `tests/ingesta/<tema>/` (sin red por defecto), salidas pequeñas versionadas en
   `results/ingesta/<tema>/`, notebooks en `notebooks/fase2/<tema>/`, dependencias en
   `requirements/ingesta-<tema>.txt`. El corpus pesado y el texto con derechos, **fuera del repositorio**
   (`INGESTA_CORPUS_DIR`).
4. **Entornos:** lo pesado (torch con CUDA, embeddings, bases vectoriales) en un entorno aparte, para no alterar las
   versiones que entran en la huella de la Fase 1.
5. **Prohibido para la Fase 2:** `src/regimenes/`, `configs/` y `data/`.
6. **Documentación:** `docs/fase2/` (reparto y contrato común, plan de datos, datos alternativos) y un README por tema
   en su subpaquete. El reparto y las reglas: [`docs/fase2/README.md`](../fase2/README.md).

### 2.2 Estructura de `docs/`

`docs/` se ordena por **fases del TFM** y por **público**, con un único índice y una única copia de la visión:

```
docs/
├── README.md            índice único: mapa de carpetas, orden de lectura por público, qué es solo local
├── GLOSARIO.md          solo términos
├── references.bib       bibliografía única
├── decisions/           ADR
├── tfm/                 visión del TFM (README), arquitectura, cliente y carteras, líneas abiertas,
│   ├── fuentes/         documentos de origen (propuesta, presentación al tutor)
│   └── internos/        SOLO LOCAL: presentaciones, plan de datos en LaTeX, diario del estado del arte
├── estado_del_arte/     estado del arte general, de detectores y de fuentes de datos, con su catálogo
├── fase1/               pipeline 00–23 y mapa del código (README), EDA, F1–F7 (teoría + detectores), sintéticos
├── fase2/               reparto en temas (README), plan de datos, datos alternativos
└── historia/            Capa 1, narrativa v1 de los detectores, estado del arte v1, tarea previa
```

Reglas:

1. **Una sola copia de cada cosa.** La visión del TFM y el estado de sus fases viven en `tfm/README.md`; el mapa de la
   documentación, en `docs/README.md`; las definiciones, en `GLOSARIO.md`; lo operativo de la Fase 1 (pipeline,
   orden de ejecución, rutas y módulos, índice de detectores), en `fase1/README.md`. Los demás documentos enlazan.
2. **Teoría y detector juntos.** Cada familia `fase1/Fk_*.md` contiene su teoría y sus detectores v2; la narrativa
   original v1 de cada detector pasa a `historia/fichas_v1.md`.
3. **Lo histórico, en `historia/`.** Lo que solo cuenta de dónde venimos (Capa 1, estado del arte v1, tarea previa)
   no se mezcla con lo vigente. La memoria, el informe y la procedencia de datos de la Capa 1 se retiran del árbol:
   eran copias idénticas de lo que guarda el tag `capa1-final`, salvo `memoria/99_conclusions.md`, que llevaba una
   nota propia de 2026-10 (la recomendación v1 queda superada); esa nota se recoge en
   [`historia/capa1/README.md`](../historia/capa1/README.md) §5.
4. **Lo interno, solo local.** Presentaciones, documentos LaTeX de trabajo y diarios de proceso viven en
   `docs/tfm/internos/`, excluido en el `.gitignore` de la raíz (vale para todos los clones). Si un documento versionado necesita algo de ahí, se destila a Markdown
   (como `fase2/plan_datos.md`) y se enlaza el destilado.
5. **Nombres:** en minúsculas con guion bajo, salvo `README.md`, `GLOSARIO.md` y las fichas `F1_…`–`F7_…`.

**Rutas fijas** (no se mueven, aunque el árbol cambie):

| Ruta | Por qué |
|---|---|
| `docs/references.bib` | La citan los documentos y la bibliografía de la memoria; un solo fichero deduplicado: si una obra ya estaba con otra clave, la entrada antigua es la canónica y la clave nueva del estado del arte es un alias (campo `ids`; 16 alias más 3 obras con la misma clave) |
| `docs/decisions/` | Las ADR se enlazan desde el código, los notebooks y el README raíz |
| `docs/fase2/README.md` | La enlazan los README de los cuatro temas, que viven en ramas distintas: moverla rompería enlaces en ramas que no controla quien reorganiza |
| `docs/GLOSARIO.md` | La enlazan notebooks, ADR y el README raíz, también por ancla; sus anclas de términos se conservan |
| `docs/historia/capa1/resultados/` | La leen los notebooks de familia y el `22` como datos (métricas v1); moverla obligaría a tocar notebooks ya ejecutados |

## 3. Alternativas consideradas

| Alternativa | Por qué se descartó |
|---|---|
| **Repositorio aparte para la Fase 2** | Duplica utilidades, separa `get_regimen` de sus consumidores y obliga a versionar dos repositorios para un mismo backtest |
| **Código de la Fase 2 dentro de `regimenes`** | Mezcla dependencias y cambia la huella de caché del benchmark de la Fase 1 |
| **Una rama por persona en lugar de por tema** | El tema es la unidad de entrega (fuentes, tools, ficha); si cambia el responsable, la rama sigue teniendo sentido |
| **Mantener el árbol de ADR-004 y añadir `fase2/` al lado** | Deja la teoría y la ficha de cada detector duplicadas, el estado del arte general sin sitio y cuatro copias de la visión que ya divergían |
| **Versionar las presentaciones y el plan de datos en LaTeX** | Son documentos de trabajo que cambian con cada reunión y pesan; su contenido vigente cabe en un destilado en Markdown |

## 4. Consecuencias

- **Ganamos:** la Fase 2 avanza en paralelo sin tocar la Fase 1; cada tema se revisa y se fusiona por piezas; quien
  llega a `docs/` encuentra un índice, un recorrido según su papel y una sola versión de cada contenido.
- **Coste:** la reorganización cambia muchas rutas de `docs/`; los enlaces del README raíz, de los notebooks y de las
  ramas abiertas se actualizan en la misma tanda (las rutas fijas existen para acotar ese coste). De `src/regimenes/`
  solo cambian su `README.md` y el docstring de `__init__.py` (un enlace), que no entran en la huella de caché; ningún
  fichero de `regimenes.detectores`, `regimenes.evaluacion`, `regimenes.benchmark`, `configs/` ni `data/` cambia, así
  que la caché del benchmark sigue vigente. Dos comentarios de `configs/` (`benchmark_spec.yaml:8` y `catalog.yaml:3`)
  conservan rutas antiguas de `docs/datos/` (hoy `fase1/eda.md` y `estado_del_arte/fuentes_datos.md`): se dejan como
  están porque `configs/` no se toca en la Fase 2, y el test de enlaces (`tests/test_docs_enlaces.py`) los documenta en
  su lista de excepciones.
- **Reversibilidad:** total con la historia de git; la Capa 1 completa sigue en el tag `capa1-final`.
- **Pendiente:** dónde viven las fases 3–5 se decidirá en otra ADR.
