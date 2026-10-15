# Tema 4 · Noticias y datos alternativos

**Rama:** `fase2/t4-noticias-alt` · **Responsable:** Romen (@Romequinco) · **Agentes:** Macro, Equity, Risk (CIO: botón
de cataclismo) · **Tools:** `buscar_noticias(q, fecha)`, `get_polymarket(evento, fecha)` y las de alternativos ·
**Destino:** noticias al cajón, hechos del día al RAG base, alternativos a la base de series. Reglas comunes:
[`docs/fase2/README.md`](../../../docs/fase2/README.md). Candidatas de la tutoría:
[`docs/fase2/datos_alternativos.md`](../../../docs/fase2/datos_alternativos.md).

## Estado: sonda pequeña sobre días testigo + validez de los alternativos

Como el Tema 3 (10 empresas), no se descarga todo: se mide **qué hay, desde cuándo, con qué retraso de publicación,
cuánto ocupa y qué falla**. La muestra son **días testigo** comunes a todas las fuentes de texto
([`muestra.py`](muestra.py)): Lehman (2008-09-15), elecciones de EE. UU. (2016-11-09), COVID (2020-03-16), SVB
(2023-03-10), aranceles (2025-04-07) y el día de la sonda. Las series de alternativos (GPR, EPU, CBOE, FINRA, tono,
visitas, mercados de predicción) se bajan **enteras**, porque la validez necesita toda su historia.

- Sonda: 39 fuentes en 12 módulos. Por estado: 22 ok, 6 parcial, 7 no probadas y 4 fallo. Hay 332 ejemplos de
  TimeGate y ninguno tiene un lag negativo.
- Notebooks:
  - [`t4_01_sonda_noticias_alt.ipynb`](../../../notebooks/fase2/t4_01_sonda_noticias_alt.ipynb): cobertura por
    épocas, cada fuente, limpieza, TimeGate y volumen.
  - [`t4_02_validez_alternativos.ipynb`](../../../notebooks/fase2/t4_02_validez_alternativos.ipynb): calidad,
    look-ahead, señal frente a las crisis de la Fase 1, calibración, rentabilidad anormal y semáforo.

  Los dos solo leen `results/` y el corpus.
- Ejecutar de nuevo: `python -m ingesta.noticias_alt.sonda` (unos 5 min con la caché caliente; `--rapido` para
  probar sin pisar la pasada completa) y después `python -c "from ingesta.noticias_alt import validez; validez.ejecutar()"`.

## Módulos

| Módulo | Qué hace |
|---|---|
| `muestra.py` | Días testigo, consultas fijas, artículos de Wikipedia, tickers y crisis de la Fase 1 (`configs/benchmark_spec.yaml`, solo lectura) |
| `base.py` | Contrato interno: `ConfigSonda`, `Resultado`, registro, cobertura, TimeGate, `Documento` en jsonl y series en tabla larga |
| `limpieza.py` | Fechas heterogéneas a UTC, RSS/Atom sin dependencias, URL canónica, idioma, duplicados (URL → Jaccard ≥ 0,75 → SimHash ≤ 3) quedándose con el más antiguo |
| `ccnews.py` | CC-NEWS de Common Crawl: lectura parcial (Range) de un WARC por día testigo, extracción y regla de fecha |
| `hf_datasets.py` | Metadatos de datasets de Hugging Face (FNSPID, stanford-oval/ccnews, Bloomberg 2006-13, Reuters/CNBC 2018, multisource) sin descargarlos |
| `prensa_api.py` | NYT Archive y Guardian Open Platform: clientes listos, sin clave (no probados) |
| `gdelt.py` | GKG 2.1 de 15 min (titulares y tono), Events 1.0/2.0 (`DATEADDED`) y DOC API (titulares y serie de tono) |
| `wikipedia.py` | Portal:Current events con la revisión de las 23:59 UTC y visitas por artículo |
| `rss.py` | 36 RSS de 15 países con robots.txt y Bing News (solo en vivo); Google News queda bloqueado por robots.txt |
| `social.py` | Archivo de Truth Social, short volume de FINRA, y Reddit, X, Trends, AAII y CNN (fichas) |
| `congreso.py` | PTR de la Cámara (índices 2014→ y PDF electrónicos con pypdf), Senado, Stock Watcher, agregadores y ETF NANC/KRUZ |
| `insiders.py` | Form 4 de la SEC (índice diario y XML) y datasets trimestrales Form 3/4/5 |
| `prediccion.py` | Polymarket (Gamma + CLOB), Kalshi y PredictIt: liquidez por año, mercados resueltos y precio a fecha |
| `indices.py` | GPR (diario y mensual), EPU (EE. UU. diario y global), SKEW, VIX y put/call de CBOE |
| `ews.py` | EWS de jets privados: archivo, z publicado frente a z causal y niveles |
| `sonda.py` | Orquestador: tablas comunes, fichas, volumen y embudo de duplicados por día testigo |
| `validez.py` | Calidad, look-ahead, estudio de eventos, AUC frente al VIX, lead-lag, estabilidad, Brier, CAR y semáforo |
| `graficos.py` | Figuras de los dos notebooks (estilo de la Fase 1, crisis sombreadas) |

Dependencias nuevas: `xlrd` y `pypdf` ([`requirements/ingesta-noticias_alt.txt`](../../../requirements/ingesta-noticias_alt.txt)).
No tocan numpy, pandas ni torch, así que la huella de la Fase 1 no cambia.

## Decisiones tomadas en el código

- **TimeGate por fuente.** `fecha_publicacion` es el día en que el dato se hizo público:

  | Fuente | Regla |
  |---|---|
  | CC-NEWS | Fecha del metadato si es ≤ `WARC-Date`; si no, la de `WARC-Date` |
  | GDELT | `DATEADDED`, nunca `SQLDATE` |
  | Wikipedia Current Events | D+1: la revisión de las 23:59 UTC es posterior a las 16:30 de Nueva York |
  | Visitas de Wikipedia | D+1 |
  | PTR del Congreso | `FilingDate` (+1 día hábil), **nunca la fecha de la operación** |
  | Form 4 | Fecha de presentación |
  | Polymarket | Último precio ≤ 16:30 de Nueva York |
  | CBOE y FINRA | D+1 |
  | GPR y EPU | Lag de calendario y descarga guardada como vintage |
- **Copyright.** El texto de los medios solo va al corpus (`<INGESTA_CORPUS_DIR>/noticias_alt`). En `results/` y en
  los notebooks solo hay metadatos, cifras y titulares. Los fixtures de los tests son inventados.
- **Acceso educado.** Todo pasa por `ingesta.comun.http.get` y se respeta robots.txt. Las fuentes que bloquean
  (Google News, AAII, CNN, Stock Watcher, Senado) quedan en `fallo` o `no_probada` con su motivo; no se esquivan.
- **Duplicados.** Se conserva la versión más antigua y el resto de medios se anota en `extra['sindicada_en']`. En
  los días testigo queda entre el 90 % y el 96 %; la sindicación de agencias es lo que más se elimina.
- **Validez causal.** Los z-scores son expanding con un mínimo de 252 observaciones. Las series se alinean por fecha
  de publicación. El AUC se compara con el VIX en los mismos días. El CAR se mide desde la publicación, y desde la
  operación solo para enseñar la diferencia.

## Cifras principales (de `results/ingesta/noticias_alt`)

- **Texto por épocas:**
  - 2016-08 en adelante: CC-NEWS, con retraso de rastreo de 1,6 h de mediana y hasta unos 800 GB/mes.
  - 2006-12 en adelante: hechos diarios de Wikipedia.
  - 2015 en adelante: titulares de GDELT GKG.
  - Antes de 2016, el texto completo depende de FNSPID, Bloomberg 2006-13 y NYT/Guardian, ninguno medido. **Lehman
    (2008) no tiene texto de noticias medido.**
- **Retrasos de publicación (mediana):**

  | Fuente | Retraso |
  |---|---|
  | GDELT | 2,3 h |
  | FINRA | 1,3 h |
  | CBOE | 4,9 h |
  | GPR | 15,8 h |
  | EPU | 23,9 h |
  | Form 4 | 96 h |
  | PTR del Congreso | 41 días desde la operación (el 38,6 % fuera del plazo de 45 días) |
- **Mercados de predicción:**
  - Liquidez de Polymarket despreciable antes de 2024: 76,5 M USD en 2020 (73 eventos), frente a 18.450 M en 2024 y
    26.050 M en 2025 (1.000 eventos con más volumen de cada año).
  - Brier a 7 días en mercados indecisos: Polymarket 0,200 frente a 0,249 climatológico (n = 57); Kalshi 0,150 frente
    a 0,230 (n = 28).
- **Señal frente a crisis:**
  - VIX: AUC 0,80 (12 crisis); ninguna serie alternativa lo supera en los mismos días.
  - Mejor alternativo: visitas a *Recession* en Wikipedia, AUC 0,815 frente a 0,818 del VIX (6 crisis).
- **Rentabilidad anormal:**
  - Compras de congresistas: CAR a 20 días desde la publicación de −1,2 %, con un IC que incluye el 0 (n = 141).
  - Insiders: no evaluable, porque el 45 % de los eventos no tiene precio (deslistados).
- **Semáforo** (`validez_semaforo.csv`): 11 aptas para backtest, 20 solo en vivo, 39 de contexto y 11 descartadas.
  - Aptas: CC-NEWS, ccnews de HF, GDELT (GKG, DOC y tono), Wikipedia Current Events, visitas a *Recession*,
    put/call total (hasta 2019) y VIX.
  - Solo en vivo: EWS, Polymarket, Kalshi, RSS, Bing, PredictIt, Truth Social, FINRA y PTR del Congreso.

  El resumen de t4_01 cuenta las fuentes «aptas por cobertura», un criterio distinto; manda el semáforo de t4_02.

## Dónde queda cada cosa

```
<INGESTA_CORPUS_DIR>/noticias_alt/      (fuera del repo; por defecto ./corpus)
  crudo/<modulo>/                       respuestas crudas cacheadas
  documentos/*.jsonl                    Documento + texto (ccnews, gdelt, rss, bing, wiki, truth)
  series/*.parquet                      tabla larga: serie, fecha_referencia, fecha_publicacion, valor, ...
results/ingesta/noticias_alt/
  fichas/*.json                         una ficha por fuente (39)
  cobertura.csv, timegate.csv, registro_descargas.csv, volumen.csv, fuentes_resumen.csv
  <fuente>_*.csv                        tablas de cada módulo (sin texto)
  limpieza_*.csv                        embudo, idiomas, sindicación
  validez_*.csv                         calidad, look-ahead, eventos, AUC, lead-lag, estabilidad, calibración, CAR, semáforo
```

## Fuentes

- [x] Noticias con texto: CC-NEWS (WARC) y metadatos de Hugging Face (FNSPID, ccnews, Bloomberg, Reuters/CNBC, multisource)
- [ ] NYT y Guardian: clientes listos, **sin clave** (decisión del 2026-10-14)
- [x] GDELT: GKG con tono, eventos y DOC API
- [x] Wikipedia: Current Events y visitas
- [x] En vivo: RSS con robots.txt y Bing News; Google News prohibido por robots.txt
- [x] Congreso: PTR de la Cámara; Senado no probado, Stock Watcher caído, agregadores de pago
- [x] Insiders: Form 4 y datasets Form 3/4/5
- [x] Mercados de predicción: Polymarket, Kalshi y PredictIt
- [x] Índices: GPR, EPU, SKEW, VIX, put/call (congelado en 2019-10-04) y short volume de FINRA
- [x] EWS de jets privados (archivo de un año móvil; solo en vivo)
- [x] Redes: Truth Social (archivo hasta 2026-05); Reddit, X y Trends no probados (OAuth, de pago o sin API)
- [ ] Briefly: su lógica de RSS, Bing y relevancia está repensada en `rss.py` y `limpieza.py`; no se integra nada más

## Preguntas abiertas

- ¿Qué cobertura de texto basta para el backtest? Sin texto medido antes de 2016, la ventana sería 2016-2026, o
  escalonada por fuente. Se decide con la ventana de la Fase 1 y el tutor.
- ¿Se piden las claves de NYT y Guardian, y se aceptan las condiciones de multisource (HF) y del Senado eFD?
- Licencias de los datasets de terceros: la apache-2.0 declarada por Bloomberg no es fiable.
- Umbrales del semáforo y tabla `SIGNO_ESPERADO`: validarlos en grupo antes de usarlos.
- Transformación log/nivel fijada a priori: hoy se elige con la serie entera, con riesgo en 6 series.
- Vintages revisados de GPR y EPU: solo hay el vintage de hoy.
- Archivo diario propio de las fuentes solo en vivo (EWS, RSS, Bing, PredictIt, Polymarket, Kalshi) para ganar historia.
- Put/call de CBOE después de 2019-10: buscar una continuación o cortar ahí.
- OCR de los PTR escaneados (el 29 % del total, sobre todo antes de 2018).
- Umbral del EWS para el cataclismo. El z publicado mira el futuro: usar el nivel publicado o el z causal en vivo.
