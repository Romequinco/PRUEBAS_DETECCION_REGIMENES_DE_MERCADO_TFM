# Plan de datos de la Fase 2

Plan vigente de los datos que consultan los cinco agentes: de dónde sale cada dato, cómo se guarda, cómo se trocea e
indexa, cómo se respeta el TimeGate y qué queda por decidir. Es el destilado del plan de datos completo (documento
LaTeX local, `docs/tfm/internos/plan_datos/`, octubre de 2026), actualizado con lo que ya ha resuelto el Tema 1.

- Reparto en cuatro temas, ramas y contrato común: [`README.md`](README.md).
- Visión del sistema y catálogo de tools: [`tfm/README.md`](../tfm/README.md) · [`tfm/arquitectura.md`](../tfm/arquitectura.md).
- Candidatas a datos alternativos: [`datos_alternativos.md`](datos_alternativos.md).

**Cómo se midió.** Antes de escribir el plan, cada fuente se probó con un extractor-sonda: una descarga pequeña y real
que midió cobertura, retraso de publicación, volumen, límites y licencia. Esas sondas vivieron en una rama local de
exploración que no se publica; el código definitivo se escribe en cada tema (`src/ingesta/<tema>/`) con su ficha y sus
tests. Las cifras de este documento son las de las sondas (octubre de 2026) y valen como orden de magnitud.

Marcas de estado: **[Tema 1 ✔]** = resuelto e implementado en
[`src/ingesta/mercados_series`](../../src/ingesta/mercados_series/README.md) (rama `fase2/t1-mercados-series`);
**[propuesta]** = decisión del plan pendiente de implementar; **[abierta]** = sin decidir.

---

## 1. Qué consulta cada agente

Cada agente tiene dos RAG (base y cajón, §2.2) y acceso a la base de series.

| Agente | RAG base (fijo) | Cajón (lo traen sus tools) | Series y cálculo |
|---|---|---|---|
| **Macro** | Comunicados, actas y ruedas de prensa de FOMC, BCE, BoE y BoJ; Beige Book; informe anual del BIS | Discursos de bancos centrales de todo el mundo (vía BIS), bancos menores, transcripciones, FMI, OCDE, noticias | FRED/ALFRED, OCDE, BCE, Eurostat, FMI, con vintages |
| **Equity** | 10-K del universo por sección (1, 1A, 7, 7A, notas) | 10-Q, 8-K, resultados (8-K 2.02), filings antiguos, ESEF, CNMV, CVM, Form 4 | Precios, fundamentales XBRL a fecha, diccionario Loughran-McDonald |
| **Risk** | Informes de estabilidad financiera (Fed, BCE, BoE), BIS Quarterly, papers de crisis abiertos, huellas de crisis | BIS WP, FSB, ESRB, OFR, noticias, Polymarket, operaciones del Congreso | VIX, spreads, curva, GPR, EPU, SKEW |
| **Portfolio** | Literatura abierta de construcción de carteras | — | Precios, libre de riesgo, baselines, costes |
| **CIO** | Idoneidad MiFID II (ESMA, CNMV), plantilla del memo | Memos y decisiones propios (memoria de cartera) | Perfil del cliente, límites |

Hallazgos de las sondas que condicionan el plan:

1. **Hay muchos datos gratuitos y legales.** Algo más de la mitad de las fuentes probadas funciona sin coste. Los
   huecos importantes son pocos: texto de noticias anterior a 2016, fundamentales anteriores a 2009 y empresas
   deslistadas en Yahoo.
2. **Los vintages de ALFRED son baratos y cambian el pasado.** En 2008 el empleo de EE. UU. se publicaba mucho menos
   negativo de lo que dice el dato revisado; los lags fijos de la Fase 1 dejan ver el PIB antes de tiempo en los
   cierres del gobierno federal.
3. **Algunos metadatos de fecha mienten.** El BIS fecha todo el día 1; la hora de aceptación del JSON de EDGAR tiene
   desfases de horas; Yahoo reescribe hacia atrás el precio ajustado. Cada caso tiene su corrección.
4. **La literatura llega tarde.** La mayoría de los papers sobre una crisis aparecen más de un año después de su
   inicio (solo 18 de 77 papers sobre la crisis financiera global eran visibles el 15-sep-2008; mediana, 1,6 años):
   el TimeGate del conocimiento importa tanto como el de los datos.
5. **Polymarket y EWS no sirven para el backtest largo** (volumen despreciable antes de 2024; histórico de EWS desde
   octubre de 2025 y con un z-score que mira el futuro).
6. **El TimeGate cabe dentro de la base vectorial** como pre-filtro, sin fugas medidas; filtrar después de buscar
   hunde la recuperación (§5.4).

Las cifras medidas que sostienen estos hallazgos (retrasos de publicación, recall, coste) están en el
[anexo A](#anexo-a-cifras-de-las-sondas); los puntos abiertos de cada tema, en el
[anexo B](#anexo-b-puntos-abiertos-por-tema).

## 2. Arquitectura de datos

### 2.1 Del dato a la tool: cinco capas

```
1 Fuentes ──> 2 Crudo ──> 3 Normalizado ──┬──> 4a Trozos + vectores (RAG base / cajón) ──┐
  (APIs,       (tal cual,    (Documento +   │                                              ├──> 5 Tools
   descargas,   hash y        texto limpio, └──> 4b Base de series (niveles + vintages) ───┘    buscar_*(q, fecha)
   librerías)   procedencia)  o serie larga)                                                    get_*(id, fecha)
```

Cada capa se regenera desde la anterior. El crudo se guarda tal cual, identificado por el hash de su contenido y con
su procedencia (URL, fecha de descarga, versión). Las tools son la única puerta de los agentes a los datos y todas
reciben la fecha de decisión.

### 2.2 Dos RAG por agente: base y cajón

| | RAG base | Cajón |
|---|---|---|
| Qué es | Conocimiento preparado de antemano: libros, informes, papers y los documentos más relevantes de las tools del agente | Lo que traen las tools del agente al llamarlas: documentos menos relevantes, más antiguos o de entidades fuera del universo |
| Cuándo se llena | Ingesta planificada, antes del backtest | Bajo demanda, en la primera llamada que lo necesita |
| Persistencia | Fija y versionada | Crece con el uso, se comparte entre ejecuciones y deduplica por hash |
| En el backtest | Pasa el TimeGate | Solo fuentes con fecha fiable; la búsqueda web abierta queda para el modo en vivo |

**[propuesta]** El cajón es una *vista*, no un segundo almacén: un único almacén por capas con un catálogo (SQLite); la
base de un agente son los documentos marcados `rag=base` para él y su cajón, los que han visto sus tools.

**Las skills no van en el RAG.** Son procedimientos («leer un comunicado FOMC», «comparar Risk Factors entre dos 10-K»)
que el agente carga enteros; se guardan como ficheros versionados aparte. Así conocimiento, tools y skills se evalúan
por separado.

### 2.3 Un almacén por tipo de dato

| Dato | Almacén | Estado |
|---|---|---|
| Documentos crudos y normalizados | Carpetas por fuente y año, direccionadas por hash, en el corpus local (`INGESTA_CORPUS_DIR`, fuera del repo) | [propuesta] |
| Series | Parquet en tabla larga con `fecha_publicacion` y `vintage` | [Tema 1 ✔] |
| Precios | Parquet as-traded por ticker; el ajuste por dividendos y splits se reconstruye a la fecha de decisión | [Tema 1 ✔] |
| Fundamentales | Tabla larga *point-in-time* por fecha de filing | [propuesta] Tema 3 |
| Estado del cliente y de la cartera (perfil, memoria, registros) | SQLite de solo inserción, un fichero por ejecución | [propuesta] |
| Vectores | Índice embebido local (§5.4) | [propuesta] |

### 2.4 Corpus fuera del repositorio

El texto con derechos (noticias, libros, muchos papers) solo se guarda en el corpus local. Al repositorio, que es
público, van el código, las fichas, los catálogos de metadatos y, en su momento, los vectores con sus metadatos y los
scripts para regenerar el texto (§5.7).

### 2.5 Código y entornos

El código de la Fase 2 vive en `src/ingesta/`, junto a `regimenes` y sin tocarlo, con un subpaquete por tema
([ADR-007](../decisions/ADR-007-fase2-ingesta-y-docs.md)). Lo pesado (torch con CUDA, embeddings, bases vectoriales)
se instala en un entorno aparte: la huella de caché del benchmark de la Fase 1 incluye las versiones de torch y numpy, y
cambiarlas invalidaría las ejecuciones ya verificadas.

## 3. Contrato común

Las reglas mínimas (TimeGate, `Documento`, tabla larga de series, fichas, acceso legal, copyright) están en
[`README.md`](README.md) §4. El plan añade:

### 3.1 TimeGate

1. Todo documento y todo valor lleva `fecha_publicacion` (día en que se hizo público, no el del hecho): las actas del
   FOMC, unas tres semanas después de la reunión; un 10-K, la fecha de presentación ante la SEC; un informe de
   operaciones del Congreso, la del informe y no la de la operación; un dato macro, la de su vintage.
2. Una tool llamada con `fecha` nunca devuelve nada con `fecha_publicacion > fecha`. En los índices vectoriales el
   filtro va **dentro** de la consulta (pre-filtro por metadato), no después.
3. Lo que vale para los datos vale para el conocimiento: un libro de 2015 sobre la crisis de 2008 no es visible en una
   decisión simulada de 2008. Solo se marcan `atemporal` las piezas sin dependencia temporal (manuales, escalas de
   perfil, plantillas), cada una justificada.
4. **Jerarquía de la fecha de publicación:** (1) sello exacto de la fuente (vintage de ALFRED, BCE u OCDE; aceptación
   del filing en EDGAR; comunicado de prensa); (2) calendario oficial de publicaciones; (3) lag fijo de la Fase 1
   (`LAG_PUBLICACION`). Si solo se conoce el mes o el año, se toma el último día del periodo. **[Tema 1 ✔]** para las
   series (`metodo_publicacion`: `vintage_exacto`, `calendario`, `lag_fijo`…).
5. **Hora de decisión: después del cierre de Nueva York.** **[Tema 1 ✔]** fijada en las 16:30 America/New_York de
   cada día natural; un cierre de mercado es visible a su hora local de cierre + 30 min. **[propuesta]** Esa misma
   hora, las 16:30 America/New_York, es la hora de decisión del contrato común para todos los temas
   ([`README.md`](README.md) §4). Una fuente puede aplicar una regla más conservadora, nunca menos: para los filings
   de EDGAR, uno aceptado a partir de las 16:00 ET cuenta el día hábil siguiente, porque la hora de aceptación del JSON
   trae desfases de 4–10 h y el corte de las 16:00 cubre ese margen. No es una segunda hora de decisión, sino una
   regla de fecha de publicación que solo retrasa documentos.
6. Hay un TimeGate que no se puede filtrar: el que llevan los pesos del LLM (§7).

### 3.2 Documentos

Los textos se guardan como `Documento` (`ingesta.comun.esquema`) en jsonl. Además de los campos obligatorios
(`doc_id` = hash del contenido, `fuente`, `tipo_documento`, `titulo`, `fecha_publicacion`, `url`, `agente`, `rag`),
conviene rellenar todos los opcionales, porque cada uno es un filtro: `fecha_referencia`, `institucion`, `idioma`,
`tickers`, `entidades`, `paises`, `seccion` (cita exacta en el memo), `licencia`, `atemporal`, `version` y `extra`
(enriquecimiento al ingerir: tono, sentimiento, `risk_delta`…).

### 3.3 Series y precios

**[Tema 1 ✔]** Contrato completo de `get_serie(id, fecha)` y `get_precios(tk, fecha)` (columnas, consulta *as-of*,
vintages, precios as-traded con ajuste reconstruido a fecha, conversión de divisa con el fixing del BCE visible) en el
[README del Tema 1](../../src/ingesta/mercados_series/README.md#contrato-bloque-1). Las series revisables se cargan
siempre con sus vintages, nunca con el histórico de hoy.

### 3.4 Acceso legal: fuentes excluidas

Decisión del plan, vigente para los cuatro temas: se respetan robots.txt y los límites de cada API, y no se esquiva
ningún CAPTCHA, muro de pago, login, puerta de términos ni anti-bot. Las sondas encontraron estas fuentes cerradas
para un robot o vetadas por sus términos; **no se usan** (ni se intenta rodearlas) aunque parezcan útiles:

| Fuente | Motivo | Alternativa legal |
|---|---|---|
| Google News | Su robots.txt no permite `/rss/search`; el mecanismo de decodificación no está documentado y las fechas vienen truncadas al día | RSS directos permitidos, Bing News y GDELT DOC en vivo (Tema 4) |
| GFSR del FMI | El cortafuegos de aplicaciones web (WAF) del FMI bloquea a los clientes que no son navegador | Descarga manual desde el navegador, si se aprueba (pregunta Q5) |
| SEDAR+ (Canadá) | Redirección anti-bot | Ninguna automática; las empresas canadienses que presentan en EE. UU., por EDGAR (Tema 3) |
| BSE / NSE (India) y HKEX | Anti-bot (403 de Akamai) o términos que lo prohíben | Ninguna automática |
| PBoC (Banco Popular de China) | robots.txt con `Disallow: /` para todos salvo el buscador Baidu | Discursos vía BIS y datasets abiertos con fecha estimada (Tema 2) |
| Senado de EE. UU. (eFD) | Puerta de aceptación de términos más anti-bot: no se automatiza (aceptarla a mano es parte de Q8) | Informes PTR de la Cámara de Representantes, publicados por el House Clerk (Tema 4) |
| CoinGecko | Exige clave de API y sus condiciones no encajan con el uso del plan | Coinbase y Binance (Tema 1) |
| Motley Fool, Seeking Alpha | Sus términos de uso prohíben la descarga | Para resultados trimestrales, el 8-K Item 2.02 con su anexo 99.1 en EDGAR, que es la fuente legal (Tema 3) |

Dos avisos más:

- **Yahoo** (`query1/query2.finance.yahoo.com`, el endpoint de yfinance) tiene `Disallow: /` en su robots.txt y sus
  términos limitan el uso a fines personales. Es la única fuente central en esa situación y sigue como pregunta
  abierta (Q1).
- **El `Last-Modified` de los PDF antiguos no vale como fecha de publicación:** cambia cada vez que el PDF se vuelve
  a subir. Un PDF sin fecha propia se fecha con la página que lo publica o con una regla conservadora (que nunca
  adelante la fecha real) y se marca `fecha_estimada`.

## 4. Qué entra por tema

### Tema 1 · Mercados y series numéricas — **completado**

Tools `get_precios`, `get_serie`. Destino: base de series (no vectorial). Detalle, estado de cada fuente y deudas
pendientes: [README del Tema 1](../../src/ingesta/mercados_series/README.md). Los puntos abiertos de los cuatro temas
están en el [anexo B](#anexo-b-puntos-abiertos-por-tema).

| Bloque | Fuentes | Estado |
|---|---|---|
| Precios y universo | Yahoo as-traded (núcleo y nivel ampliado), constituyentes point-in-time del S&P 500, universo por capas con equivalentes UCITS, cripto (Coinbase, Binance de control) | [Tema 1 ✔] |
| Series macro y de mercado | Las series de la Fase 1 como niveles, FRED vivo, vintages exactos de ALFRED, académicas (French, Shiller, Goyal-Welch, OFR FSI, JST) | [Tema 1 ✔] |
| Macro internacional y curvas | OCDE, BCE, Eurostat, FMI (parcial), Banco Mundial, BIS, Philadelphia Fed; curvas soberanas oficiales | [Tema 1 ✔] (con huecos declarados) |
| Cartera | Libre de riesgo, baselines (buy & hold, 60/40, 1/N, momentum sectorial…) y modelo de costes | [Tema 1 ✔] |
| Sesgo de supervivencia | Se mantiene, se mide y se declara; la corrección completa exige CRSP (WRDS) | [Tema 1 ✔] medido; corrección [abierta] |

### Tema 2 · Bancos centrales, riesgo sistémico y literatura

Tools `buscar_macro`, `buscar_riesgo`. Destino: RAG base (documentos clave) + cajón. Detalle:
[README del Tema 2](../../src/ingesta/bancos_riesgo/README.md).

- **Política monetaria.** Fed (comunicados, actas, ruedas de prensa, Beige Book, transcripciones), BCE, BoE y BoJ en la
  base; discursos de todo el mundo vía BIS y bancos menores en el cajón. Retrasos medidos: actas FOMC +21 días,
  *accounts* del BCE +28 días, transcripciones FOMC ≈ +6 años. Rastreo propio como fuente primaria (es el único
  con fecha de publicación fiable); los datasets abiertos sirven de semilla, verificación y etiquetas, no de fuente.
- **Estabilidad financiera.** BIS Quarterly (desde 1996), FSR del BCE (2004), del BoE (1996) y de la Fed (2018), OFR
  (2012) y ESRB (2012) en la base; BIS WP, FSB y ESRB en el cajón. El hueco de EE. UU. de 2000 a 2017 lo cubren el
  BIS, el FSOC y la literatura; el GFSR del FMI está bloqueado (§3.4). TimeGate conservador: con fecha dudosa se
  retrasa el documento, nunca se adelanta.
- **Papers de crisis y literatura de cartera.** Catálogo de metadatos versionado (sin texto); `fecha_publicacion` =
  primera versión pública fechable (working paper, no el número de la revista).
- **Huellas de crisis** (indicadores antes y durante cada crisis), deterministas a partir de los datos de la Fase 1.
- **Enriquecimiento:** diferencias entre comunicados consecutivos y tono *hawkish/dovish*. Modelo elegido:
  `gtfintechlab/FOMC-RoBERTa` (entrenado sobre FOMC; licencia CC BY-NC 4.0, compatible con el uso académico y
  pendiente de aceptar en Q10), en un entorno aparte. Es ayuda de lectura, no señal de decisión: se entrenó con frases
  de 1996–2022 etiquetadas con conocimiento posterior.
- **Multi-idioma:** se indexa la versión inglesa oficial siempre que el banco la publique (BCE, BoJ, Riksbank,
  Norges, SNB, BCB, BoC), marcada `traduccion=True`; los documentos solo en español (Banxico, BdE) o en portugués se
  indexan en su idioma con un modelo de embeddings multilingüe (`multilingual-e5-large`). No se traduce con un LLM al
  ingerir: sería caro y metería errores sin cita verificable.

### Tema 3 · Empresas

Tools `buscar_filings`, `get_fundamentales`. Detalle: [README del Tema 3](../../src/ingesta/empresas/README.md).

- **SEC EDGAR** como fuente principal: `fecha_publicacion` = fecha de presentación, corregida al día hábil siguiente
  si la aceptación es a partir de las 16:00 ET (§3.1); nunca un lag fijo desde el cierre del ejercicio. Retrasos
  medianos medidos desde el cierre del periodo: 10-K +58 días, 10-Q +34.
- **Base:** 10-K del universo en cada año (incluidas las empresas que salieron del índice), secciones 1, 1A, 1C, 7 y 7A
  más las notas del Item 8, y el 20-F de las extranjeras. **Cajón:** 10-Q, 8-K, filings antiguos, empresas fuera del
  universo y documentos internacionales (ESEF con la fecha del registro nacional, CNMV, CVM).
- **Fundamentales XBRL** point-in-time por hecho en tabla larga: cada valor, el del filing más reciente presentado
  hasta la fecha. Antes de 2009 no hay fuente abierta.
- **Resultados trimestrales:** 8-K Item 2.02 y su anexo 99.1 como fuente legal; las transcripciones, solo como piloto
  con clave gratuita.
- **Enriquecimiento:** sentimiento Loughran-McDonald y `risk_delta` entre 10-K consecutivos.

### Tema 4 · Noticias y datos alternativos

Tools `buscar_noticias`, `get_polymarket` y las de alternativos. Detalle:
[README del Tema 4](../../src/ingesta/noticias_alt/README.md) y candidatas en [`datos_alternativos.md`](datos_alternativos.md).

- **Texto histórico por épocas** (el punto más débil del plan): CC-NEWS da texto abierto desde 2016; antes solo hay
  FNSPID, Wikipedia (*Current events*) y, con clave gratuita, Guardian y NYT. GDELT da titulares, hora y tono, no texto.
  En vivo: RSS permitidos, Bing News y GDELT DOC.
- **Destino:** noticias al cajón; un resumen diario (Wikipedia *Current events*, titulares GDELT por tema) a la base;
  los alternativos numéricos (tono GDELT, GPR, EPU, SKEW, put/call, Polymarket, EWS) a la base de series.
- **Limpieza:** duplicados por URL, titular y SimHash, conservando la versión más antigua.
- **Historia corta:** Polymarket (útil desde 2024), EWS (desde octubre de 2025) y los ETF del Congreso (desde 2023)
  solo en vivo o en la ventana reciente del backtest.
- **Operaciones del Congreso:** el informe (PTR) se publica una mediana de +41 días después de la operación (sonda del
  Tema 4, 531 operaciones; el 38,6 % fuera del plazo legal de 45 días), y su `fecha_publicacion` es la del informe.

## 5. Tratamiento RAG

### 5.1 Troceado por tipo de documento

Medido sobre cinco tipos de documento con consultas sintéticas: el **padre-hijo** (hijos de unos 128 tokens para
buscar, padre = sección o documento para el LLM) gana en los cuatro tipos largos; trozos pequeños buscan mejor; el
troceado semántico no compensa su coste; la cabecera de contexto embebida no ayuda (se guarda como metadato).

| Tipo | Estrategia propuesta |
|---|---|
| Comunicado de banco central | Padre-hijo: hijos de 128 tokens, padre = el comunicado entero. Las frases casi idénticas entre reuniones solo se distinguen por la fecha: pre-filtro temporal y recencia imprescindibles |
| Actas FOMC | Padre-hijo dentro de cada sección (la sección dice quién habla y va como metadato) |
| Ruedas de prensa y Q&A | Un trozo por par pregunta-respuesta |
| Beige Book | Por distrito |
| 10-K / 10-Q | Padre-hijo: padre = Item o factor de riesgo del Item 1A; empresa e Item como metadatos |
| Informe largo en PDF (FSR, BIS, FMI, papers) | Padre-hijo por sección (padres de 512–800 tokens); resumen del paper y *overview* del informe indexados aparte |
| Noticia | Corta: documento entero. Larga: trozos de 100–200 tokens por frases con el titular delante |

### 5.2 Embeddings

**[propuesta]** Embeber en local (GPU), con búsqueda **híbrida**: vector + BM25 fusionados por RRF (BM25 solo ya
superó a todos los densos en las sondas y protege nombres propios, tickers y cifras). Las APIs comerciales son baratas,
pero su modelo puede cambiar o retirarse (rompe la reproducibilidad) y sus vectores no se pueden publicar con garantías.

Medido en las sondas (recall@5 medio sobre cinco tipos de documento, 1.114 consultas): e5-large-v2 es el mejor denso
(0,76); BM25 solo llega a 0,83; el híbrido e5-large-v2 + BM25, a 0,84, y bge-small + BM25, a 0,82 con unas diez veces
menos GPU.

**[abierta]** Modelo: provisionalmente e5-large-v2 + BM25 para las bases (bancos centrales, 10-K y PDF) y bge-small +
BM25 para el cajón de noticias (calidad parecida con una fracción de la GPU). Para los textos solo en español o en
portugués, `multilingual-e5-large` (§4, Tema 2). Se fijará con las preguntas RAGAS
revisadas por el equipo, tras medir también bge-m3 y FinModernBERT. Criterio: recall@5 en la base de cada agente,
coste de embebido acotado y licencia MIT o Apache.

### 5.3 Almacén por capas y partición por año

Un único almacén con catálogo; el escalado lo marcan los documentos distintos (del orden de millones de trozos con
todas las fuentes). La partición por año resuelve a la vez la exactitud y la velocidad del filtro temporal: una decisión
de 2000 solo abre las particiones ≤ 2000.

### 5.4 Base vectorial con TimeGate como pre-filtro

Medido: el **post-filtro no vale** (con el corte al inicio del backtest devuelve casi nada); todas las bases con
pre-filtro devolvieron cero documentos posteriores a la fecha. Un índice HNSW con filtro muy selectivo pierde vecinos,
así que en el backtest se usa búsqueda plana sobre la partición o un índice por partición.

**[propuesta]** **LanceDB embebida**, con una tabla por (modelo, troceador, fuente, año) y pre-filtro
`fecha_publicacion <= :fecha AND fuente IN (...)`: ficheros en disco sin servidor, versiones de tabla y el mismo formato
sirve para publicar. OpenSearch solo si el modo en vivo se despliega en AWS, a través de un adaptador común.
Descartados: Chroma (latencia con filtros, tamaño) y Qdrant local (escala).

### 5.5 Framework

**[propuesta]** LangGraph para orquestar los agentes y recuperación propia: el TimeGate es la propiedad crítica del
sistema y debe ser auditable en nuestro código, con un test, no depender del conector de un framework. LlamaIndex,
como mucho, de lector puntual de formatos complejos.

### 5.6 Infraestructura y coste

**[propuesta]** Corpus, índices y embeddings en local (coste cero); el LLM por API con *prompt caching*, Batch API para
el backtest y un modelo por agente (medio para Macro, Equity y Risk; el mejor para el CIO). AWS solo si se despliega el
modo en vivo. **[abierta]** Presupuesto de LLM: depende de la frecuencia del backtest, el número de ablaciones y el
modelo por agente. La estimación de las sondas (precios públicos de octubre de 2026; unos 270k tokens de entrada y
16k de salida por comité) es de unos **315 $** para un backtest semanal 2000–2024 (unos 1.300 comités) con el CIO en
el mejor modelo, el resto en un modelo medio, caché y lotes; unos 73 $ si es mensual y unos 1.260 $ con cuatro
variantes. Antes de comprometerlo se mide el consumo real del Macro Agent en unas diez fechas.

### 5.7 Publicación de vectores y metadatos

**[propuesta]** Un dataset de Hugging Face (o un *release*) con vectores en `float16` y metadatos, **sin texto**: por
cada trozo, su documento, fuente, fecha, URL, posiciones de inicio y fin y el hash del trozo, más un manifiesto y un
script que vuelve a bajar cada URL, normaliza con la misma versión del código y comprueba el hash. La regla es la misma
para todas las fuentes. La inversión de embeddings existe pero es aproximada; se declara como riesgo residual.

## 6. Cartera y cliente

Requisitos de la tutoría en [`tfm/cliente_y_carteras.md`](../tfm/cliente_y_carteras.md); lo que propone el plan desde
los datos:

| Aspecto | Propuesta | Estado |
|---|---|---|
| RAG base de Portfolio y CIO | Literatura abierta y legal de construcción de carteras e idoneidad; los papers **no** son atemporales (un resultado empírico sobre 2008 no se ve en 2007) | [propuesta] Tema 2 |
| Perfil del cliente | Cuatro bloques tipo MiFID II; perfil 1–7 = mínimo (capacidad, tolerancia) traducido a límites numéricos; versionado con `vigente_desde`, fuera del TimeGate de mercado; en el backtest, clientes sintéticos por perfil | [propuesta] |
| Divisa de los datos | Los datos se guardan en su divisa y se convierten al leer con el fixing del BCE visible en la fecha. Las tools del Tema 1 convierten a USD (`divisa='original'` o `'USD'`), la divisa base decidida en el Tema 1 | [Tema 1 ✔] |
| Divisa del cliente | USD por defecto, como la base del Tema 1. EUR queda como opción del cliente (minorista español): el plan LaTeX lo proponía, con la cobertura FX simulada por paridad cubierta (Euribor 3M/€STR frente a DTB3); exigiría añadir la salida en EUR a las tools | [abierta] (Q2) |
| Libre de riesgo y baselines | DTB3 para los agentes, French RF ex post; €STR (EONIA antes) en EUR; baselines con proxies para cubrir los primeros años | [Tema 1 ✔] |
| Costes | Comisión pública de un bróker minorista + spread base por clase escalado en crisis por un estimador de spread. Para un minorista con órdenes de 500–1.000 €, la comisión fija pesa 30–60 pb por orden, más que el spread | [Tema 1 ✔] (tarifas sin verificar) |
| Universo invertible del minorista europeo | ETF de EE. UU. como proxy de sus equivalentes UCITS en el backtest, declarando la diferencia; UCITS en la demo | [Tema 1 ✔] mapeo UCITS y empalme |
| Rebalanceo | Base mensual + rebalanceo obligatorio cuando el detector entra en crisis | [propuesta] |
| Memo | La plantilla es una skill; memos de terceros al RAG base con su fecha; memos propios al cajón del CIO, visibles solo si son anteriores y de la misma ejecución | [propuesta] |
| Estado | SQLite de solo inserción por ejecución (perfil, memoria de cartera, registros) | [propuesta] |

## 7. Look-ahead del LLM y evaluación

### 7.1 El TimeGate que no se puede filtrar

El TimeGate garantiza que las tools no devuelven nada posterior a la fecha, pero el LLM se entrenó con texto hasta su
fecha de corte, posterior a todo el backtest: un memo «de agosto de 2008» puede acertar porque el modelo recuerda lo
que pasó. La literatura (Glasserman y Lin, 2023; Sarkar y Vafa, 2024; He et al., 2025, ChronoBERT/ChronoGPT) coincide
en que una instrucción en el *prompt* no basta y en que la solución limpia son modelos entrenados solo con texto
anterior a cada fecha.

Protocolo de verificación (medir y acotar, no eliminar):

| Prueba | Qué se hace | Indica fuga si… | Estado |
|---|---|---|---|
| L1 Sondas | Preguntas cuya respuesta solo se conoce después de la fecha simulada, con y sin instrucción TimeGate | acierta con la instrucción puesta | Probada con un LLM local pequeño |
| L2 Datación | Pedir el año de un comunicado FOMC, original y anonimizado; control fuera del corte | data los textos de su ventana de entrenamiento y no los posteriores | Probada con un LLM local pequeño |
| L3 Ablación | Mismas decisiones con entradas originales y anonimizadas | el rendimiento cae al anonimizar dentro de la ventana de entrenamiento y no fuera | Diseñada |
| L4 Fuera del corte | Repetir el backtest después del corte del modelo | rendimiento dentro ≫ fuera | Diseñada |
| L5 Modelo con corte | Nodo de control con un modelo cronológicamente consistente en tareas de clasificación | divergencia sistemática en fechas antiguas | Opcional |

La mini-demostración con un modelo diminuto ya mostró fuga con la instrucción puesta y la huella de datación; la
anonimización la reduce sin eliminarla. L1–L2 deben repetirse con el LLM elegido. Las pruebas se numeran L1–L5 para
no confundirlas con las fases P1–P5 ni con las preguntas Q1–Q11 ([GLOSARIO](../GLOSARIO.md)).

**Mitigaciones [propuesta]:** anonimización opcional en las tools (fechas relativas, episodios enmascarados; empresas
solo en tareas de sentimiento); instrucción TimeGate en cada agente (necesaria, no suficiente); el LLM no decide pesos
(motores deterministas y el CIO solo ajusta ±5 pp); y **reporte doble** de todas las métricas: dentro de la ventana de
entrenamiento (potencialmente contaminadas) y fuera del corte (resultado limpio). **[abierta]** La ventana limpia es
corta y la fija el modelo elegido: compromiso entre calidad y limpieza a decidir con el tutor.

### 7.2 Evaluación RAGAS por RAG

RAGAS (Es et al., 2024) separa recuperación (*context recall*, *context precision*) y generación (*faithfulness*, base
de `verificar_citas`; *answer relevancy*). Con dos RAG por agente, hacer a mano 50–100 pares por RAG no es viable:
**[propuesta]** generar con LLM y revisar por humanos.

1. **Muestreo estratificado** de trozos por fuente, año y régimen (calma / alerta / crisis de `get_regimen`).
2. **Generación** con un LLM distinto del evaluado. Tipos: factual, comparativa entre fechas, razonamiento y «sin
   respuesta a esa fecha» (mide abstención y TimeGate). Cada pregunta lleva su `fecha_decision` ≥ `fecha_publicacion`.
3. **Revisión humana** con rúbrica (respondible, no ambigua, fechada, útil); una parte con doble revisión para medir el
   acuerdo.
4. **Congelación y versión** del conjunto, con una parte reservada como test; nunca se ajusta y evalúa con lo mismo.
5. **Ejecución** por lotes en cada cambio de troceado, modelo o índice.

Tamaño objetivo: unas 100 preguntas aceptadas por RAG base y 60 por cajón; basta para detectar diferencias de 10–15
puntos entre configuraciones con tests pareados, no menores. El coste dominante no es el dinero sino las **horas de
revisión** del equipo; se priorizan las bases de Macro y Risk.

## 8. Preguntas abiertas

Las once preguntas del plan, con su estado a octubre de 2026. Se numeran Q1–Q11 (en el LaTeX local eran P1–P11).

| # | Pregunta | Estado |
|:---:|---|---|
| Q1 | **Yahoo y robots.txt.** El endpoint del que descarga yfinance lo prohíbe en `robots.txt`. ¿Se asume para el corpus de precios o se usa una fuente de pago (EODHD, unos 20 €/mes, y un mes basta para la descarga inicial; Tiingo) o CRSP si llega la licencia? | **Abierta.** El Tema 1 usa Yahoo y deja la fuente de la versión final como pregunta abierta |
| Q2 | **Hora de decisión y divisa base.** ¿Después del cierre de EE. UU. o antes de la apertura? ¿Cliente en EUR, con cobertura FX simulada, o en USD? | **Resuelta en parte (Tema 1):** decisión a las 16:30 de Nueva York; datos en su divisa con conversión al leer, hoy a USD (divisa base del Tema 1). Queda la divisa de referencia del cliente: EUR es una opción (el plan LaTeX la proponía) que exigiría salida en EUR en las tools (§6) |
| Q3 | **Universo del minorista europeo.** Por PRIIPs no puede comprar ETF domiciliados en EE. UU. ¿UCITS equivalentes en la cartera del cliente y los de EE. UU. solo para el análisis? | **Resuelta en datos (Tema 1):** universo con equivalentes UCITS mapeados y empalme con el proxy de EE. UU. Falta confirmarlo con el tutor |
| Q4 | **Claves gratuitas** (NYT, Guardian, EDINET, OpenDART, Alpha Vantage, BEA, BLS v2): ¿se registran? Abrirían noticias con texto anteriores a 2016, Japón, Corea y transcripciones | Abierta (Temas 3 y 4) |
| Q5 | **Descargas manuales** de unos 45 PDF abiertos bloqueados para robots (SSRN, GFSR del FMI, EUR-Lex): ¿se bajan a mano desde el navegador? Son legales y gratuitos; en el Tema 2, con los de la AEA, son unos 94 documentos | Abierta (Tema 2) |
| Q6 | **Acceso universitario a WRDS** (CRSP, Compustat, TAQ): resolvería supervivencia, fundamentales anteriores a 2009 y spreads reales (TAQ) | Abierta: pendiente de la licencia de MIAX (el Tema 1 mide el sesgo sin ella) |
| Q7 | **OCR** de papers antiguos escaneados (15 papers NBER anteriores a 1995) y de parte de los PTR del Congreso (del 11 al 61 % según el año): ¿se instala Tesseract? | Abierta (Temas 2 y 4) |
| Q8 | **Licencias de datasets** de Hugging Face dudosas (corpus Bloomberg/Reuters 2006–2013) o con aceptación manual de términos (*Financial News Multisource*, 57 M filas desde 1990; la misma pregunta vale para la puerta del Senado eFD, §3.4): ¿se usan? | Abierta (Tema 4) |
| Q9 | **Fase 1 con vintages.** Con ALFRED disponible, ¿se reevalúa el detector con datos en tiempo real? Exigiría una ADR nueva sobre la limitación de ADR-003 | **Abierta, con los datos listos:** el Tema 1 ya sirve los vintages exactos de las series revisables |
| Q10 | **Clasificador hawkish/dovish:** el mejor disponible es CC BY-NC. ¿Se acepta por ser uso académico? | Abierta (Tema 2) |
| Q11 | **Cataclismo:** el z-score publicado por EWS mira el futuro. ¿Qué nivel publicado dispara la desinversión en vivo? | Abierta (Tema 4; propuesta: nivel máximo publicado en dos instantáneas seguidas) |

Decisiones transversales que siguen pendientes con el tutor:

- **Ventana del backtest de los agentes.** El texto la acota: noticias con texto completo desde 2016, fundamentales
  XBRL desde 2009, Polymarket útil desde 2024, FSR de la Fed desde 2018, Yahoo fuera de EE. UU. desde 2000, bancos
  centrales y EDGAR desde 1993–1994. Opciones: (a) 2007–2026,
  alineada con la pista B de la Fase 1, aceptando noticias pobres antes de 2016; (b) 2016–2026, fuentes completas pero
  solo dos crisis grandes; (c) ventanas escalonadas por fuente, declarando qué ve el agente en cada época.
- **Frecuencia de rebalanceo** (propuesta: mensual + crisis); su coste está en las llamadas al LLM (§5.6) y, en un
  minorista, en las comisiones fijas: 30–60 pb por orden, más que el spread.

## 9. Siguientes pasos

| Paso | Contenido | Estado |
|---|---|---|
| 1 · Cerrar la Fase 1 | Notebooks 18–20 y `get_regimen(fecha)` | Pendiente |
| 2 · Base de series | `get_serie` y `get_precios` en producción con sus tests de TimeGate | **Hecho (Tema 1)** |
| 3 · Primer RAG completo | Macro: comunicados, actas y ruedas de prensa de FOMC y BCE en la base, discursos en el cajón, con troceado, embeddings y pre-filtro elegidos | Siguiente (Tema 2) |
| 4 · Primer agente | Macro Agent sobre episodios concretos, con evaluación de la recuperación y prueba de look-ahead | Pendiente (Fase 3) |
| 5 · Escalar | Equity (10-K por sección), Risk (BIS, FSR, papers) y News (CC-NEWS, GDELT); después Portfolio y CIO | Pendiente |

---

## Anexo A. Cifras de las sondas

Medidas por los extractores-sonda en octubre de 2026 (plan LaTeX local, resumen ejecutivo y secciones por dominio).
Valen como orden de magnitud: el código definitivo de cada tema vuelve a medirlas en su ficha.

**Alcance.** 188 fuentes sondadas: 117 funcionan, 22 parciales, 12 bloqueadas y 37 con clave o de pago (el 62 % de
las probadas funciona sin coste). 440 papers y libros catalogados. 166 series de la Fase 1 más 16 series revisables
con vintages de ALFRED. Bancos centrales y EDGAR, desde 1993–1994. Gasto en datos: 0 €.

**Retrasos de publicación medidos** (mediana):

| Documento | Retraso | Desde |
|---|---|---|
| Actas del FOMC | +21 días | la reunión |
| *Accounts* del BCE | +28 días | la reunión |
| PDF de las ruedas de prensa de la Fed | +7 a +18 días (el vídeo, el mismo día) | la rueda |
| 10-K | +58 días | el cierre del ejercicio |
| 10-Q | +34 días | el cierre del trimestre |
| PTR del Congreso | +41 días (mediana medida; plazo legal 45) | la operación |
| Transcripciones del FOMC | ≈ +6 años | la reunión |

**Vintages y lags.**

- La historia completa de vintages de las 16 series revisables ocupa 1,3 MB y se descarga en 7 minutos.
- En 2008 el empleo de EE. UU. se publicaba en torno a −100k/mes; el dato revisado es −450k. La producción industrial
  cambia de signo en el 22 % de las revisiones.
- Los lags fijos de la Fase 1 dejan ver el PIB antes de publicarse en el 4,3 % de los trimestres (cierres del
  gobierno federal); en el conjunto de series, las fugas medidas afectan al 0,2–4,5 % de los meses (Q9).

**Metadatos de fecha que mienten.** El BIS fecha todas las Quarterly Review el día 1; la hora de aceptación del JSON
de EDGAR tiene desfases de 4–10 h; Yahoo reescribe hacia atrás el `Adj Close`; el `Last-Modified` de los PDF antiguos
cambia al volver a subirlos (§3.4).

**Supervivencia y literatura.**

- EDGAR conserva a las empresas deslistadas (Lehman, 6.341 filings; Enron). Yahoo solo cubre el 15 % de la etapa en
  el índice de 40 valores que salieron del S&P 500 (la medición completa del Tema 1 está en su README).
- Solo 18 de 77 papers sobre la crisis financiera global eran visibles el 15-sep-2008; la mediana de publicación
  llega 1,6 años después del inicio de la crisis.

**Noticias y alternativos.** CC-NEWS da texto completo y abierto desde 2016 (del orden de 900 M de artículos). El
volumen de Polymarket anterior a 2024 es menos del 1 % del de 2025. El z-score publicado de EWS mira el futuro: una
reconstrucción causal con un modelo simple dio 411 falsas alarmas.

**Recuperación** (§5.1–5.4).

| Medida | Resultado |
|---|---|
| Troceado padre-hijo (128/768), recall@5 por tipo | comunicados 0,70 · actas 0,76 · 10-K 0,86 · FSR 0,80, el mejor en los cuatro tipos largos (en noticias, todas las estrategias dan 0,95–0,96) |
| Embeddings, recall@5 medio | e5-large-v2 0,76 (mejor denso) · BM25 0,83 · e5-large-v2 + BM25 0,84 · bge-small + BM25 0,82 |
| Post-filtro temporal | el recall cae a 0,02 con el corte al inicio del backtest |
| Pre-filtro temporal | 0 documentos posteriores a la fecha en las cinco bases probadas |
| Embeber el corpus completo (unos 2.400 M de tokens) | unas 226 h de GPU local con e5-large frente a unas 23 h con bge-small; por API, 50–310 $ por pasada |
| Publicar vectores en `float16` | el recall@5 cambia como máximo 0,004 |

**Coste de LLM** (backtest 2000–2024, precios públicos de octubre de 2026, unos 270k tokens de entrada y 16k de salida
por comité, 60 % de la entrada cacheable):

| Escenario | Mensual (300 comités) | Semanal (1.300) | Semanal × 4 variantes |
|---|---:|---:|---:|
| Mejor modelo en todo, sin optimizar | 420 $ | 1.820 $ | 7.280 $ |
| Mixto (CIO en el mejor modelo, resto en uno medio) + caché + lotes | 73 $ | **315 $** | 1.261 $ |
| Modelo pequeño + caché + lotes | 31 $ | 133 $ | 530 $ |

Un LLM abierto en local no es alternativa para el backtest: en la GPU disponible, un modelo de 7B en 4 bits genera
unos 10 tokens/s (unas 650 h para 1.300 comités). Sí sirve para tareas mecánicas, como L1–L2 (§7.1).

**Cartera y costes.** Spread estimado (Abdi y Ranaldo, 2017), calma 2017 frente a COVID 2020: SPY de 18 a 170 pb,
AGG de 7 a 28, XLK de 27 a 201, EEM de 25 a 112, IEF de 11 a 31. Los estimadores de rango sobrestiman el nivel en un
orden de magnitud (el spread cotizado de SPY ronda 1 pb), pero capturan el multiplicador de crisis (×3 a ×10). Para un
minorista con 10.000 € y órdenes de 500–1.000 €, la comisión fija pesa 30–60 pb por orden, más que el spread.

## Anexo B. Puntos abiertos por tema

Lo que cada sección del plan dejaba por decidir o completar, con las preguntas Q1–Q11 (§8) a las que se liga. Lo ya
resuelto por un tema no se repite aquí; las deudas propias del Tema 1 están en su
[README](../../src/ingesta/mercados_series/README.md#deudas-pendientes).

**Tema 1 · Mercados y series**

- Fuente de precios de la versión final: Yahoo, EODHD o Tiingo, o WRDS/CRSP si llega la licencia (Q1, Q6).
- Universo de emergentes: el top 200 por capitalización de la propuesta o índices nacionales.
- Fechas de vintage aproximadas (OCDE: fin de mes de la edición; RTDSM: día 15): ¿bastan para el backtest o se
  cruzan con calendarios oficiales?
- Previsiones (WEO, Economic Outlook de la OCDE, SEP): el TimeGate filtra por edición, no por año de referencia; falta
  decidir si van a una tabla de proyecciones aparte.
- Cobertura de China y emergentes: la NBS no tiene API automatizable; FMI, OCDE y BIS (abiertos) o CEIC (de pago).
- Licencias de redistribución: las series de Moody's, ICE y CBOE en FRED permiten el uso, no la redistribución; la
  base de series vive solo en local.
- Fase 1 con vintages (Q9).

**Tema 2 · Bancos centrales, riesgo y literatura**

- FMI y OCDE bloquean a los clientes que no son navegador: ¿descarga manual única de unos 40 WEO y unos 40 Economic
  Outlook (2007–2026), solo los PDF y con su fecha de publicación, o se prescinde de ellos? Junto con el GFSR y los PDF
  bloqueados de SSRN y la AEA (unos 94 documentos), es Q5.
- Bancos inaccesibles (RBA, RBNZ, PBoC, RBI): propuesta, BIS más el dataset `aufklarer` con fechas estimadas, sin
  pedir los datos a los bancos (son cola larga para un universo centrado en EE. UU. y Europa).
- CSV de discursos del BCE y ZIP del BIS: descarga reanudable y lenta (12–70 KB/s); una sola pasada nocturna cacheada.
- Fecha de los PDF de las ruedas de prensa de la Fed: el `Last-Modified` no sirve para los antiguos; ¿regla fija de
  +7 días o la fecha de la página de la rueda?
- Licencia del clasificador hawkish/dovish (CC BY-NC): compatible con el TFM académico, no con un uso comercial (Q10).
- ¿Los unos 1.380 BIS Working Papers (unos 30 M de palabras) van a la base o al cajón?
- Unas 42 entradas del catálogo de papers solo tienen fecha con precisión de año, lo que las retrasa hasta 12 meses:
  ¿se completan a mano?
- Huellas de crisis fuera de EE. UU. (Asia 1997, México 1994, SME 1992, deuda europea): ¿con series de qué países?
- `robots_permite` del módulo común da falsos negativos con la Fed, el BoE y el FSB: lo decide quien mantiene
  `src/ingesta/comun/`.
- Por sondear con más detalle: Greenbook anterior a 1994 (Philadelphia Fed), Banco de Corea, SARB, Banco de Turquía y
  testimonios ante el Congreso.

**Tema 3 · Empresas**

- Constituyentes históricos del STOXX 600 (los del S&P 500 ya los da el Tema 1).
- Fundamentales anteriores a 2009: acceso a WRDS/Compustat (Q6).
- Claves gratuitas de EDINET, OpenDART y Alpha Vantage, a nombre de un miembro del equipo (Q4).
- STOXX 600 y MSCI EM: ESEF solo cubre desde el ejercicio 2020; ¿se reduce la Capa 2 a EE. UU. para el backtest largo?
- Librerías candidatas en el entorno aparte: `edgartools` (MIT) y `sec-edgar-downloader`.
- Validar el extractor a escala: unos 200 10-K del S&P 500 (2001–2025) contrastados con EDGAR-CORPUS hasta 2020, y
  mejorar la detección del MD&A en el anexo EX-13 con títulos no estándar.

**Tema 4 · Noticias y alternativos**

- Claves gratuitas de Guardian y NYT, las únicas fuentes con fecha fiable para 2000–2016 con texto o resumen (Q4).
  Sin ellas, 2008 se apoya en Wikipedia, GDELT y FNSPID.
- Riesgo de licencia del corpus Bloomberg/Reuters 2006–2013 y aceptación manual de *Financial News Multisource* y del
  Senado eFD (Q8).
- OCR de los PTR escaneados, del 11 al 61 % según el año (Q7).
- Lista definitiva de medios y países del filtro de CC-NEWS (propuesta: 300 dominios de 30 países).
- Umbral del botón de cataclismo (propuesta: `emergencyLevel` publicado = 5 en dos instantáneas seguidas) y si anula
  al CIO (Q11).
- Fuente del ratio put/call después de octubre de 2019 (los CSV de CBOE se cortan ahí).

**Cartera y cliente** (preguntas de [`tfm/cliente_y_carteras.md`](../tfm/cliente_y_carteras.md) §6 que dependen de
datos)

- Escalas del formulario y tabla perfil → límites: ¿Grable-Lytton traducida o preguntas propias validadas con ESMA?
- Memoria de cartera: ¿se añade el memo previo (más tokens, riesgo de anclaje)? Evaluarlo exige dos ejecuciones
  aisladas y el Sharpe deflactado.
- Cambio de perfil en el tiempo: el esquema lo soporta; falta decidir si el backtest lo simula.
- Rol de gestor: cambiaría el almacén de estado (Postgres) solo si entra en el alcance.
- Comisiones históricas: solo hay tarifas vigentes y DEGIRO prohíbe el rastreo en robots.txt; ¿tarifa fija para todo
  el backtest o reconstrucción con Wayback Machine?
- Divisa del cliente (Q2, §6).

**Transversales (RAG y LLM)**

- Troceado semántico: no compensa en las sondas; se deja fuera salvo que las preguntas RAGAS lo justifiquen.
- Modelo de embeddings (§5.2), presupuesto de LLM (§5.6) y ventana limpia del LLM (§7.1).
- Ventana del backtest y frecuencia de rebalanceo (§8).
