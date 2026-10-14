# Tema 3 · Empresas: filings, fundamentales y sentimiento

**Rama:** `fase2/t3-empresas` · **Agentes:** Equity y Risk · **Tools:** `buscar_filings(tk, q, fecha)`,
`get_fundamentales(tk, fecha)`, `composite_score(bucket, fecha)` · **Destino:** RAG base + cajón (texto), base de series
(fundamentales). Reglas comunes: [`docs/fase2/README.md`](../../../docs/fase2/README.md).

## Alcance

Información de empresas:
- informes oficiales (10-K, 10-Q, 8-K y equivalentes fuera de EE. UU.) troceados por sección;
- fundamentales contables **tal como se conocían en cada fecha**;
- sentimiento del lenguaje de riesgo;
- resultados trimestrales.

Cobertura mundial cuando haya fuentes abiertas.

## Fuentes a investigar

- [ ] SEC EDGAR: submissions API, índices completos, descargas masivas, búsqueda de texto completo
- [ ] Extracción de secciones de 10-K / 10-Q (Item 1A Risk Factors, Item 7 MD&A, Item 8…) en todos los formatos históricos
- [ ] Fundamentales XBRL (companyfacts, frames, Financial Statement Data Sets) y qué hacer antes de que existan
- [ ] Resultados trimestrales (8-K) y transcripciones de *earnings calls* (legalidad de cada fuente)
- [ ] Datasets abiertos ya construidos (p. ej. EDGAR-CORPUS)
- [ ] Sentimiento: diccionario Loughran-McDonald, FinBERT
- [ ] Fuera de EE. UU.: ESEF (UE), CNMV, Companies House, EDINET, DART, SEDAR+, HKEX, cninfo, CVM…
- [ ] Empresas deslistadas (supervivencia)

> **Acceso legal.** Las sondas del plan ya descartaron **SEDAR+** (redirección anti-bot) y **BSE/NSE y HKEX**
> (anti-bot o términos que lo prohíben): no se descargan de forma automática. Tampoco se usan **Motley Fool** ni
> **Seeking Alpha** (sus términos lo prohíben); para los resultados trimestrales, la fuente legal es el 8-K Item 2.02
> con su anexo 99.1. Lista completa y motivos:
> [plan de datos §3.4](../../../docs/fase2/plan_datos.md#34-acceso-legal-fuentes-excluidas); puntos abiertos del
> tema: [anexo B](../../../docs/fase2/plan_datos.md#anexo-b-puntos-abiertos-por-tema).

## Entregables

1. Un extractor por fuente con su ficha y sus tests; secciones como `Documento` (con `tickers`, `seccion` y
   `fecha_publicacion` = fecha de presentación).
2. Tabla de fundamentales *point-in-time* (por fecha de presentación, con reexpresiones).
3. Propuesta de qué va a la base (p. ej. 10-K del universo) y qué al cajón.

## Preguntas abiertas

- ¿Universo de empresas (S&P 500, STOXX 600, emergentes)?
- ¿Fuente de fundamentales anteriores a XBRL (acceso universitario a WRDS/Compustat)?
- ¿Registrar claves gratuitas de registros extranjeros?
