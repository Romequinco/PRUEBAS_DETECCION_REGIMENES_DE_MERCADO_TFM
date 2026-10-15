# Fase 2 — Datos de los agentes: reparto y forma de trabajar

Contexto: el TFM es un sistema multi-agente con RAG ([`docs/tfm/`](../tfm/README.md), [ADR-005](../decisions/ADR-005-reencuadre-tfm-multiagente.md)).
La Fase 1 (régimen, `src/regimenes`) está en marcha. La Fase 2 reúne **todos los datos que consultan los agentes**, y se
reparte en **cuatro temas**: cada uno tiene responsable, rama y carpetas propias.

## 1. Los cuatro temas

| Tema | Carpeta | Rama | Agentes | Tools que alimenta | Responsable |
|---|---|---|---|---|---|
| 1 · Mercados y series numéricas | [`src/ingesta/mercados_series`](../../src/ingesta/mercados_series/README.md) | `fase2/t1-mercados-series` | Portfolio, Equity, Macro, Risk | `get_precios`, `get_serie` | Romen (@Romequinco) |
| 2 · Bancos centrales, riesgo sistémico y literatura | [`src/ingesta/bancos_riesgo`](../../src/ingesta/bancos_riesgo/README.md) | `fase2/t2-bancos-riesgo` | Macro, Risk (Portfolio y CIO: conocimiento) | `buscar_macro`, `buscar_riesgo` | _por asignar_ |
| 3 · Empresas | [`src/ingesta/empresas`](../../src/ingesta/empresas/README.md) | `fase2/t3-empresas` | Equity, Risk | `buscar_filings`, `get_fundamentales` | Daniel García |
| 4 · Noticias y datos alternativos | [`src/ingesta/noticias_alt`](../../src/ingesta/noticias_alt/README.md) | `fase2/t4-noticias-alt` | Macro, Equity, Risk (CIO: cataclismo) | `buscar_noticias`, `get_polymarket` | Romen (@Romequinco) |

**Estado** (2026-10-15): los temas 1 (mercados y series), 3 (sonda EDGAR) y 4 (sonda de noticias y alternativos con
validez) están integrados en `main`; el Tema 2 aún no tiene rama ni responsable. Cada responsable rellena su celda
de la tabla.

**Notebooks por tema:**
- Tema 1: [`notebooks/fase2/mercados_series/`](../../notebooks/fase2/mercados_series/) (`01`–`09`).
- Tema 3: [`notebooks/fase2/t3_01_sonda_empresas.ipynb`](../../notebooks/fase2/t3_01_sonda_empresas.ipynb).
- Tema 4: [`notebooks/fase2/t4_01_sonda_noticias_alt.ipynb`](../../notebooks/fase2/t4_01_sonda_noticias_alt.ipynb) y
  [`t4_02_validez_alternativos.ipynb`](../../notebooks/fase2/t4_02_validez_alternativos.ipynb).

Cada README de tema dice qué entra, qué entrega y qué preguntas quedan abiertas. El plan de datos común (arquitectura,
TimeGate, tratamiento RAG, look-ahead del LLM, evaluación y preguntas abiertas) está en [`plan_datos.md`](plan_datos.md);
las candidatas a datos alternativos, en [`datos_alternativos.md`](datos_alternativos.md).

## 2. Dónde escribe cada tema

```
src/ingesta/<tema>/                código del tema (un módulo por fuente)
tests/ingesta/<tema>/              sus tests (sin red por defecto; los de red con @pytest.mark.live)
results/ingesta/<tema>/            salidas pequeñas versionadas: fichas/, catálogos, tablas
notebooks/fase2/<tema>/            notebooks del tema (p. ej. notebooks/fase2/mercados_series/; los temas 3 y 4
                                   los tienen sueltos como notebooks/fase2/tN_*.ipynb)
requirements/ingesta-<tema>.txt    dependencias propias del tema
<INGESTA_CORPUS_DIR>/<tema>/       corpus pesado, FUERA del repo (por defecto ./corpus, ignorado)
```

**Compartido** (solo cambia con un PR pequeño a `main`): `src/ingesta/comun/`, `tests/ingesta/conftest.py`,
`pyproject.toml`, `docs/fase2/README.md`, `.env.example`. **Nadie toca** `src/regimenes`, `configs/` ni `data/` (Fase 1).

## 3. Flujo de ramas

```bash
git switch main && git pull
git switch -c fase2/t1-mercados-series        # tu tema
# ... trabajo, commits pequeños ...
git fetch origin && git merge origin/main     # traer cambios comunes cuando los haya
git push -u origin fase2/t1-mercados-series   # y PR a main cuando una pieza esté cerrada
```

- Un PR por pieza cerrada (p. ej. «extractor de X con tests»), no uno gigante al final.
- La CI (`.github/workflows/tests.yml`) debe pasar: sin red, los tests `live` se saltan.
- Si necesitas algo en `comun/`, abre un PR aparte y avisa al resto.

## 4. Contrato común (para poder juntarlo todo después)

1. **TimeGate.** Todo dato lleva `fecha_publicacion` (día en que se hizo público, no el del hecho). Una tool con
   `fecha` nunca devuelve nada publicado después. **Hora de decisión común (propuesta):** las 16:30
   America/New_York del día `fecha`, la que ya usa el Tema 1; es visible lo que se hizo público antes de esa hora.
   Una fuente puede ser más conservadora, nunca menos: los filings de EDGAR aceptados a partir de las 16:00 ET pasan
   al día hábil siguiente porque la hora de aceptación del JSON trae desfases de horas ([plan de
   datos](plan_datos.md) §3.1).
2. **Textos** → objetos `Documento` (`ingesta.comun.esquema`) guardados en jsonl: un documento por línea con todos sus
   metadatos. El troceado y los vectores se deciden después, en común.
3. **Series** → parquet en tabla larga: `serie, fecha_referencia, fecha_publicacion, valor, unidad, fuente` (+ `vintage`
   si existe).
4. **Cada fuente tiene una ficha** (`FichaFuente`, `ingesta.comun.fichas.guardar_ficha`): qué es, cómo se consigue,
   cobertura, retraso, volumen, licencia y estado.
5. **Acceso legal y educado.** Usar `ingesta.comun.http.get`, respetar robots.txt (`robots_permite`) y los límites de
   cada API, y no esquivar CAPTCHAs, logins ni muros de pago. Las claves van en `.env`, nunca en el código. Las
   fuentes que ya se descartaron por acceso o términos de uso (Google News, SEDAR+, PBoC…) están en el
   [plan de datos](plan_datos.md#34-acceso-legal-fuentes-excluidas).
6. **Copyright.** El texto con derechos solo se guarda en local (corpus). Al repo van código, fichas y metadatos.
7. **Dependencias pesadas** (torch, transformers, bases vectoriales) en un entorno aparte. Cambiar torch, numpy o
   pandas en el entorno principal invalida la caché del benchmark de la Fase 1.

## 5. Cuándo está «hecha» una fuente

- [ ] Extractor en `src/ingesta/<tema>/<fuente>.py` con funciones reutilizables (listar, descargar, parsear).
- [ ] Ficha guardada con cobertura y retraso **medidos** sobre una muestra real.
- [ ] Test sin red sobre una muestra pequeña en `tests/ingesta/<tema>/fixtures/` y test `live`.
- [ ] `fecha_publicacion` correcta y justificada (cómo se obtiene y qué lag tiene).
- [ ] Decidido si va a la base de series, al RAG base o al cajón del agente.

## 6. Configuración inicial

```bash
cp .env.example .env                    # FRED_API_KEY, SEC_EDGAR_USER_AGENT, INGESTA_CORPUS_DIR
python -m pip install -e ".[dev]"
python -m pip install -r requirements/ingesta-<tema>.txt
python -m pytest tests/ingesta -q       # INGESTA_LIVE=1 para los tests con red
```
