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
   inicio: el TimeGate del conocimiento importa tanto como el de los datos.
5. **Polymarket y EWS no sirven para el backtest largo** (volumen despreciable antes de 2024; histórico de EWS desde
   octubre de 2025 y con un z-score que mira el futuro).
6. **El TimeGate cabe dentro de la base vectorial** como pre-filtro, sin fugas medidas; filtrar después de buscar
   hunde la recuperación (§5.4).

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
   cada día natural; un cierre de mercado es visible a su hora local de cierre + 30 min. Para los filings, uno aceptado
   a partir de las 16:00 ET cuenta el día hábil siguiente.
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

## 4. Qué entra por tema

### Tema 1 · Mercados y series numéricas — **completado**

Tools `get_precios`, `get_serie`. Destino: base de series (no vectorial). Detalle, estado de cada fuente y deudas
pendientes: [README del Tema 1](../../src/ingesta/mercados_series/README.md).

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
  base; discursos de todo el mundo vía BIS y bancos menores en el cajón. Retrasos medidos: actas FOMC ≈ +3 semanas,
  *accounts* del BCE ≈ +4 semanas, transcripciones FOMC ≈ +6 años. Rastreo propio como fuente primaria (es el único
  con fecha de publicación fiable); los datasets abiertos sirven de semilla, verificación y etiquetas, no de fuente.
- **Estabilidad financiera.** BIS Quarterly, FSR del BCE, BoE y Fed, OFR y ESRB en la base; BIS WP, FSB y ESRB en el
  cajón. TimeGate conservador: con fecha dudosa se retrasa el documento, nunca se adelanta.
- **Papers de crisis y literatura de cartera.** Catálogo de metadatos versionado (sin texto); `fecha_publicacion` =
  primera versión pública fechable (working paper, no el número de la revista).
- **Huellas de crisis** (indicadores antes y durante cada crisis), deterministas a partir de los datos de la Fase 1.
- **Enriquecimiento:** diferencias entre comunicados consecutivos y tono *hawkish/dovish* (ayuda de lectura, no señal
  de decisión: los clasificadores disponibles se entrenaron con etiquetas posteriores).

### Tema 3 · Empresas

Tools `buscar_filings`, `get_fundamentales`. Detalle: [README del Tema 3](../../src/ingesta/empresas/README.md).

- **SEC EDGAR** como fuente principal: `fecha_publicacion` = fecha de presentación, corregida al día hábil siguiente
  si la aceptación es a partir de las 16:00 ET; nunca un lag fijo desde el cierre del ejercicio.
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

**[abierta]** Modelo: provisionalmente e5-large-v2 + BM25 para las bases (bancos centrales, 10-K y PDF) y bge-small +
BM25 para el cajón de noticias (calidad parecida con una fracción de la GPU). Se fijará con las preguntas RAGAS
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
modelo por agente; se mide primero el consumo real del Macro Agent en unas pocas fechas.

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
| Divisa | Los datos se guardan en su divisa y se convierten al leer con el fixing del BCE visible en la fecha | [Tema 1 ✔] |
| Libre de riesgo y baselines | DTB3 para los agentes, French RF ex post; €STR (EONIA antes) en EUR; baselines con proxies para cubrir los primeros años | [Tema 1 ✔] |
| Costes | Comisión pública de un bróker minorista + spread base por clase escalado en crisis por un estimador de spread | [Tema 1 ✔] (tarifas sin verificar) |
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
| P1 Sondas | Preguntas cuya respuesta solo se conoce después de la fecha simulada, con y sin instrucción TimeGate | acierta con la instrucción puesta | Probada con un LLM local pequeño |
| P2 Datación | Pedir el año de un comunicado FOMC, original y anonimizado; control fuera del corte | data los textos de su ventana de entrenamiento y no los posteriores | Probada con un LLM local pequeño |
| P3 Ablación | Mismas decisiones con entradas originales y anonimizadas | el rendimiento cae al anonimizar dentro de la ventana de entrenamiento y no fuera | Diseñada |
| P4 Fuera del corte | Repetir el backtest después del corte del modelo | rendimiento dentro ≫ fuera | Diseñada |
| P5 Modelo con corte | Nodo de control con un modelo cronológicamente consistente en tareas de clasificación | divergencia sistemática en fechas antiguas | Opcional |

La mini-demostración con un modelo diminuto ya mostró fuga con la instrucción puesta y la huella de datación; la
anonimización la reduce sin eliminarla. P1–P2 deben repetirse con el LLM elegido.

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

Las once preguntas del plan, con su estado a octubre de 2026.

| # | Pregunta | Estado |
|:---:|---|---|
| P1 | **Yahoo y robots.txt.** El endpoint del que descarga yfinance lo prohíbe en `robots.txt`. ¿Se asume para el corpus de precios o se usa una fuente de pago (EODHD, Tiingo) o CRSP si llega la licencia? | **Abierta.** El Tema 1 usa Yahoo y deja la fuente de la versión final como pregunta abierta |
| P2 | **Hora de decisión y divisa base.** ¿Después del cierre de EE. UU. o antes de la apertura? ¿Cliente en EUR o en USD? | **Resuelta en parte (Tema 1):** decisión a las 16:30 de Nueva York; datos en su divisa con conversión al leer. Queda la divisa de referencia del cliente (el plan propone EUR) |
| P3 | **Universo del minorista europeo.** Por PRIIPs no puede comprar ETF domiciliados en EE. UU. ¿UCITS equivalentes en la cartera del cliente y los de EE. UU. solo para el análisis? | **Resuelta en datos (Tema 1):** universo con equivalentes UCITS mapeados y empalme con el proxy de EE. UU. Falta confirmarlo con el tutor |
| P4 | **Claves gratuitas** (NYT, Guardian, EDINET, OpenDART, Alpha Vantage, BEA, BLS v2): ¿se registran? Abrirían noticias con texto anteriores a 2016, Japón, Corea y transcripciones | Abierta (Temas 3 y 4) |
| P5 | **Descargas manuales** de PDF abiertos bloqueados para robots (SSRN, GFSR del FMI, EUR-Lex): ¿se bajan a mano desde el navegador? | Abierta (Tema 2) |
| P6 | **Acceso universitario a WRDS** (CRSP, Compustat): resolvería supervivencia, fundamentales anteriores a 2009 y spreads reales | Abierta: pendiente de la licencia de MIAX (el Tema 1 mide el sesgo sin ella) |
| P7 | **OCR** de papers antiguos escaneados y de parte de los informes del Congreso: ¿se instala Tesseract? | Abierta (Temas 2 y 4) |
| P8 | **Licencias de datasets** de Hugging Face dudosas o con aceptación manual de términos: ¿se usan? | Abierta (Tema 4) |
| P9 | **Fase 1 con vintages.** Con ALFRED disponible, ¿se reevalúa el detector con datos en tiempo real? Exigiría una ADR nueva sobre la limitación de ADR-003 | **Abierta, con los datos listos:** el Tema 1 ya sirve los vintages exactos de las series revisables |
| P10 | **Clasificador hawkish/dovish:** el mejor disponible es CC BY-NC. ¿Se acepta por ser uso académico? | Abierta (Tema 2) |
| P11 | **Cataclismo:** el z-score publicado por EWS mira el futuro. ¿Qué nivel publicado dispara la desinversión en vivo? | Abierta (Tema 4; propuesta: nivel máximo publicado en dos instantáneas seguidas) |

Decisiones transversales que siguen pendientes con el tutor:

- **Ventana del backtest de los agentes.** El texto la acota: noticias con texto completo desde 2016, fundamentales
  XBRL desde 2009, Polymarket útil desde 2024, bancos centrales y EDGAR desde 1993–1994. Opciones: (a) 2007–2026,
  alineada con la pista B de la Fase 1, aceptando noticias pobres antes de 2016; (b) 2016–2026, fuentes completas pero
  solo dos crisis grandes; (c) ventanas escalonadas por fuente, declarando qué ve el agente en cada época.
- **Frecuencia de rebalanceo** (propuesta: mensual + crisis); su coste está en las llamadas al LLM y, en un
  minorista, en las comisiones fijas.

## 9. Siguientes pasos

| Paso | Contenido | Estado |
|---|---|---|
| 1 · Cerrar la Fase 1 | Notebooks 18–20 y `get_regimen(fecha)` | Pendiente |
| 2 · Base de series | `get_serie` y `get_precios` en producción con sus tests de TimeGate | **Hecho (Tema 1)** |
| 3 · Primer RAG completo | Macro: comunicados, actas y ruedas de prensa de FOMC y BCE en la base, discursos en el cajón, con troceado, embeddings y pre-filtro elegidos | Siguiente (Tema 2) |
| 4 · Primer agente | Macro Agent sobre episodios concretos, con evaluación de la recuperación y prueba de look-ahead | Pendiente (Fase 3) |
| 5 · Escalar | Equity (10-K por sección), Risk (BIS, FSR, papers) y News (CC-NEWS, GDELT); después Portfolio y CIO | Pendiente |
