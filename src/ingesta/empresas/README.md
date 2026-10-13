# Tema 3 · Empresas: filings, fundamentales y sentimiento

**Rama:** `fase2/t3-empresas` · **Responsable:** Daniel · **Agentes:** Equity y Risk · **Tools:**
`buscar_filings(tk, q, fecha)`, `get_fundamentales(tk, fecha)`, `composite_score(bucket, fecha)` · **Destino:** RAG
base + cajón (texto), base de series (fundamentales). Reglas comunes: [`docs/fase2/README.md`](../../../docs/fase2/README.md).

## Estado: sonda pequeña (10 empresas)

Acordado en el grupo (12-oct-2026): antes de la descarga grande, cada tema prueba sus fuentes con poco volumen y
mide qué hay, desde cuándo, con qué retraso y cuánto ocupa. Universo, ventana y almacenamiento se deciden después.

Muestra: AAPL, MSFT, JPM, XOM, JNJ, WMT, KO, GE (S&P 500 actual, sectores distintos, historia larga) + **Lehman** y
**Enron** (desaparecidas: comprueban que el corpus no tiene sesgo de supervivencia). Ver `universo.MUESTRA_SONDA`.

Se ejecuta con el notebook [`notebooks/fase2/t3_01_sonda_empresas.ipynb`](../../../notebooks/fase2/t3_01_sonda_empresas.ipynb).

## Módulos

| Módulo | Qué hace |
|---|---|
| `universo.py` | ticker ↔ CIK (`company_tickers.json` de la SEC) y la muestra de la sonda. El identificador interno es el **CIK**: los tickers cambian y se reutilizan |
| `sec_edgar.py` | todos los filings de una empresa (incluidas las páginas antiguas), **TimeGate**, descarga comprimida del documento principal y del anexo 99.1 de los 8-K de resultados |
| `secciones.py` | HTML/SGML → texto; secciones de 10-K (Items) y 10-Q (Parte-Item); filas jsonl con los metadatos de `Documento` + `texto` |
| `xbrl.py` | fundamentales `companyfacts` en tabla larga **point-in-time**; `vista_a_fecha` y `reexpresiones` |
| `sentimiento.py` | Loughran-McDonald (proporciones por categoría y tono) y `risk_delta` del Item 1A entre años |
| `sonda.py` | la prueba de punta a punta: mediciones, tablas y fichas |

Sin dependencias nuevas: solo librería estándar + `pandas`, `pyarrow` y `requests` del paquete.

## Decisiones tomadas en el código

- **TimeGate.** `fecha_publicacion` = `filingDate`, salvo aceptación ≥ 16:00 (Nueva York) ese mismo día → siguiente
  día hábil de la NYSE. La hora se lee de la cabecera oficial de cada filing (`.hdr.sgml`, `verificar_aceptacion`,
  cacheada): la de `acceptanceDateTime` de la API **no es fiable** (prueba real, 10-K de Apple del 31-oct-2025:
  cabecera 06:01:26, API 14:01:26Z). Mientras un filing no se verifica, lleva la fecha prudente (día hábil siguiente).
  `contrastar_timegate` mide el desfase de la API.
- **Secciones.** Para cada Item se toma el tramo más largo entre su encabezado y el siguiente (el índice da tramos
  cortos). Las secciones que solo dicen *incorporated by reference* (MD&A antiguo remitido al anexo 13) se cuentan pero
  no generan documento.
- **Base y cajón.** Van a la base los 10-K (Items 1, 1A, 1C, 7, 7A, 8) y los 10-Q (Parte I Items 2 y 3, Parte II
  Item 1A), sin enmiendas. El resto, al cajón. Los 8-K de resultados, al cajón.
- **Fundamentales.** Cada cifra se guarda una vez por filing en que aparece, con su `fecha_publicacion`: las
  correcciones posteriores no se filtran al pasado.
- **Texto en el jsonl.** `Documento` no tiene campo de texto; de momento va en la clave `texto` de la misma línea.
  Pendiente de acordar en `comun`.

## Dónde queda cada cosa

```
<INGESTA_CORPUS_DIR>/empresas/            (fuera del repo)
├── sec/submissions/      listas de filings (json.gz)
├── sec/crudo/<cik>/<año>/ documentos originales (gzip)
├── sec/companyfacts/     XBRL (json.gz)
├── documentos/<cik>.jsonl secciones y notas de resultados (una por línea)
├── fundamentales/<cik>.parquet
└── lm/                   diccionario Loughran-McDonald (descarga manual)
results/ingesta/empresas/                 (versionado: solo mediciones)
├── fichas/*.json
└── cobertura.csv · registro_descargas.csv · extraccion_secciones.csv · timegate.csv · volumen.csv · ...
```

## Fuentes

- [x] SEC EDGAR submissions API (lista completa de filings, también los anteriores a los ~1.000 recientes)
- [x] Extracción de secciones de 10-K / 10-Q (HTML moderno y texto de los 90)
- [x] Fundamentales XBRL (`companyfacts`)
- [x] 8-K de resultados (Item 2.02) con su nota de prensa (anexo 99.1)
- [x] Sentimiento Loughran-McDonald + `risk_delta`
- [x] Empresas deslistadas (Lehman, Enron)
- [ ] EDGAR-CORPUS (Hugging Face), FSDS / frames, búsqueda de texto completo
- [ ] Transcripciones de calls (Alpha Vantage con clave), FinBERT
- [ ] Fuera de EE. UU.: ESEF (UE), CNMV, Companies House, EDINET, DART…

## Preguntas abiertas

- ¿Universo de empresas (S&P 500 histórico con constituyentes por fecha, STOXX 600, emergentes)?
- ¿Ventana del backtest? Determina cuántos años de filings se bajan.
- ¿Fuente de fundamentales anteriores a XBRL (2009): acceso universitario a WRDS/Compustat?
- ¿Campo de texto en `Documento` (`comun`)?
