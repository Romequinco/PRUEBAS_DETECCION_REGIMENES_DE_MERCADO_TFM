# Bancos centrales, riesgo sistémico y literatura

**Notebook de selección de fuentes · revisión: 12 de octubre de 2026**

Objetivo: decidir de dónde obtener cada tipo de información de la captura, priorizando fuentes gratuitas, originales y verificables. Los **14 apartados** reproducen sus tarjetas; las tablas separan los subtipos de documentos.

**Orientación actual del proyecto:** construir una biblioteca amplia de literatura, informes y comunicaciones para que los agentes busquen evidencias y fundamenten decisiones. Se recopilará por lotes; la prioridad es cobertura, calidad y consulta útil, sin exigir ingesta inmediata. Las fechas y versiones aportan contexto. Las indicaciones de TimeGate de este cuaderno se aplican cuando se pide una evaluación histórica, y no bloquean por defecto la incorporación al corpus general.

Una publicación sin fecha pública exacta puede ser útil si conocemos su procedencia, términos y versión y el texto es fiable; la incertidumbre se muestra al agente. Mantener papers e informes antiguos como contexto, identificar ediciones y no confundir una guía sustituida con una vigente. El detalle del diseño está en [Corpus de conocimiento para agentes](../docs/CORPUS_PARA_AGENTES.md).

Este cuaderno es documental: se puede leer sin ejecutar celdas. Los enlaces son las referencias que respaldan las recomendaciones. Se han consultado portales, páginas de publicaciones y fichas de datasets; todavía no se ha descargado ni validado un corpus completo ni probado la automatización de todas las fuentes.

**Cómo interpretar “calidad”.** Para comunicaciones institucionales, elegir el emisor original: texto íntegro, fecha, versión y procedencia. Para papers, evaluar también metodología, revisión académica y versión. Un repositorio cómodo de descargar no convierte sus textos o etiquetas en información de mayor calidad.

**Cómo interpretar “gratis”.** Se refiere al acceso digital indicado. Descargar sin pagar no implica permiso ilimitado para redistribuir, comercializar o realizar descargas masivas. Las restricciones expresas más relevantes aparecen en sus apartados. El futuro almacenamiento y procesamiento tienen sus propios costes.

Las cantidades, tamaños y estados OK/MIXTO/BLOQUEADA de la captura son referencias del proyecto, **no inventarios verificados**. Un bloqueo técnico de una web no demuestra que el contenido sea de pago.

## Vista rápida e índice

| Apartado | Primera opción recomendada | ¿Gratis? | Prioridad de incorporación |
|---|---|---|---|
| [1. Fed](#fed) | Federal Reserve | Sí, documentos públicos | Alta |
| [2. BCE](#bce) | Banco Central Europeo | Sí | Alta |
| [3. BoE y BoJ](#boe-boj) | Cada banco emisor | Sí | Alta |
| [4. Discursos BIS](#discursos-bis) | Descarga oficial BIS | Sí; uso no comercial en el paquete | Alta: facilita un corpus amplio |
| [5. Otros bancos centrales](#otros-bancos) | Portales de cada banco | Publicaciones públicas gratuitas; acceso variable | Media |
| [6. Organismos](#organismos) | Banco Mundial, FMI y OCDE | Sí para las fuentes propuestas | Media |
| [7. Datasets abiertos](#datasets) | Fichas y archivos originales del dataset | Sí o cuenta gratuita; licencias distintas | Complementaria |
| [8. Estabilidad financiera](#estabilidad) | Fed, BCE, BoE, ESRB, OFR y FSB | Sí | Alta |
| [9. BIS QR y Working Papers](#bis-papers) | Publicaciones oficiales BIS | Sí | Alta |
| [10. FMI GFSR](#gfsr) | FMI: PDF y datos por edición | Sí | Alta |
| [11. Papers de crisis](#papers-crisis) | Emisor, autor o repositorio institucional | Muchos sí; NBER/editoriales, parcialmente | Alta: desde el piloto |
| [12. Huellas de crisis](#huellas) | ESRB + FMI + Macrohistory; elaboración propia | Bases gratuitas con condiciones | Después de validar las bases |
| [13. Literatura de cartera y CIO](#cartera) | Autores/instituciones y EUR-Lex, ESMA, CNMV | Mixto en literatura; normativa propuesta gratuita | Alta: selección abierta desde el piloto |
| [14. La literatura llega tarde](#literatura-tarde) | Historial de versiones y evidencia de publicación | Metadatos gratuitos; análisis propio | Opcional: evaluación histórica posterior |

## 1. Fed

<a id="fed"></a>

**Elección:** Federal Reserve como fuente principal. Conserva el documento institucional y permite relacionarlo con la reunión correspondiente. Los datasets externos sirven de complemento.

| Información solicitada | Fuente de mayor calidad recomendada | ¿Gratis? | Acceso previsto |
|---|---|---|---|
| Comunicados y decisiones FOMC | [Calendarios y documentos FOMC](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm) | Sí, sin cuenta | HTML y PDF enlazados por reunión |
| Actas | [Mismo calendario FOMC](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm) y [archivo histórico](https://www.federalreserve.gov/monetarypolicy/fomc_historical.htm) | Sí | HTML/PDF; publicación separada del comunicado |
| Ruedas de prensa | [Calendario FOMC](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm) | Sí | Transcripción cuando esté publicada; vídeo como complemento |
| Beige Book | [Archivo oficial Beige Book](https://www.federalreserve.gov/monetarypolicy/beigebook/default.htm) | Sí | HTML/PDF por edición y distrito |
| Transcripciones y material histórico | [FOMC Historical Materials](https://www.federalreserve.gov/monetarypolicy/fomc_historical.htm) | Sí | PDF y documentos por año |
| Discursos | [Speeches de la Fed](https://www.federalreserve.gov/newsevents/speeches.htm) | Sí | HTML/PDF con autor y fecha |

**Calidad y tiempo:** preferir HTML para extraer párrafos y conservar el PDF original cuando exista. Las actas actuales suelen publicarse unas tres semanas después; las transcripciones completas se publican con aproximadamente cinco años de retraso. Usar la fecha efectiva de cada publicación: los “+21 días” y “+6 años” de la captura no son reglas universales para todo el histórico. El archivo enlazado documenta el régimen de publicación.

## 2. BCE

<a id="bce"></a>

**Elección:** portal del BCE. Mantiene decisiones, comparecencias y deliberaciones como series distintas, evitando mezclar documentos de la misma reunión.

| Información solicitada | Fuente recomendada | ¿Gratis? | Acceso previsto |
|---|---|---|---|
| Decisiones de política monetaria | [Monetary policy decisions](https://www.ecb.europa.eu/press/govcdec/mopo/html/index.en.html) | Sí | Listados y texto HTML |
| Declaración y preguntas/respuestas | [Conferencias de prensa](https://www.ecb.europa.eu/press/press_conference/html/index.en.html) | Sí | HTML; separar declaración y cada Q&A |
| Accounts, equivalentes a actas resumidas | [Monetary policy accounts](https://www.ecb.europa.eu/press/accounts/html/index.en.html) | Sí | HTML con fecha de reunión y de publicación |
| Discursos | [Speeches](https://www.ecb.europa.eu/press/key/html/index.en.html) | Sí | HTML/PDF según documento |

**Calidad y tiempo:** guardar el idioma original y vincular traducciones como versiones relacionadas. Las accounts están disponibles después de la reunión; usar la fecha real de publicación, sin imponer “+28 días” a todas las ediciones. La fecha de reunión identifica el evento; la fecha de publicación decide cuándo podía conocerlo el pipeline.

## 3. BoE y BoJ

<a id="boe-boj"></a>

**Elección:** Bank of England y Bank of Japan, respectivamente. Para análisis textual, sus informes y actas completos tienen más valor que un resumen de prensa externo.

| Banco e información | Fuente recomendada | ¿Gratis? | Acceso previsto |
|---|---|---|---|
| BoE: decisión y actas MPC | [Última decisión y enlaces a documentos](https://www.bankofengland.co.uk/monetary-policy/the-interest-rate-bank-rate); [ejemplo publicado: julio de 2026](https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/2026/july-2026) | Sí | HTML/PDF; descubrir ediciones desde enlaces oficiales |
| BoE: Monetary Policy Report | [Archivo MPR](https://www.bankofengland.co.uk/sitemap/monetary-policy-report) | Sí | Informes HTML/PDF por edición |
| BoE: discursos | [Speeches](https://www.bankofengland.co.uk/news/speeches) | Sí | HTML y documentos enlazados |
| BoJ: comunicados, opiniones y actas | [Calendario y documentos de reuniones](https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm) | Sí | Enlaces a declaraciones, Summary of Opinions y Minutes |
| BoJ: informes de perspectivas | [Outlook Report](https://www.boj.or.jp/en/mopo/outlook/index.htm) | Sí | PDF de informes y perspectivas |
| BoJ: discursos incluidos en BIS | [Discursos de bancos centrales BIS](https://www.bis.org/speeches/central-bank) | Sí | Complemento; contrastar con el original del BoJ |

**Calidad y tiempo:** en BoJ, las opiniones y las actas son documentos diferentes. Su [explicación oficial del procedimiento](https://www.boj.or.jp/en/about/education/oshiete/seisaku/b20.htm) vincula la publicación de actas a su aprobación en una reunión posterior. El “+44 días” de la captura no debe convertirse en retraso fijo. Las traducciones al inglés pueden aparecer después del original japonés.

## 4. Discursos BIS y Annual Report

<a id="discursos-bis"></a>

**Elección para un corpus internacional:** descarga oficial del BIS. Para el texto definitivo de un discurso concreto, contrastar con el banco emisor si hay discrepancias.

| Información solicitada | Fuente recomendada | ¿Gratis? | Acceso previsto |
|---|---|---|---|
| Discursos de bancos centrales | [Descarga oficial de discursos BIS](https://www.bis.org/speeches/central-bank/download) | Sí, sin cuenta; paquete para uso no comercial según sus condiciones | Archivos CSV UTF-8 comprimidos, por año o conjunto, desde 1996 |
| Documento individual y metadatos | [Catálogo BIS](https://www.bis.org/speeches/central-bank) | Sí | Página y documento enlazado |
| Informe económico anual | [Annual Economic Report](https://www.bis.org/publications/aer) | Sí | Capítulos e informe completo |
| Informe institucional anual | [Annual Report](https://www.bis.org/about/areport/index.htm) | Sí | Informe de actividad institucional |

**Calidad:** el paquete es especialmente útil para empezar sin reconstruir miles de páginas. BIS advierte que la cobertura no es exhaustiva y que la conversión automática puede introducir errores. Validar muestras contra los originales y guardar la fecha/versión de la descarga. Annual Economic Report y Annual Report tienen objetivos diferentes: catalogarlos por separado. No asumir que todas las publicaciones BIS comparten la licencia del paquete de discursos.

## 5. Otros bancos centrales

<a id="otros-bancos"></a>

**Elección:** emisor original para decisiones, actas, informes y discursos. Las publicaciones digitales públicas indicadas son la primera ruta gratuita; no se ha probado todavía una descarga masiva de cada banco.

| Banco | Punto de entrada oficial | Información que buscar | Gratuidad y acceso |
|---|---|---|---|
| BCB, Brasil | [Comunicados Copom](https://www.bcb.gov.br/en/monetarypolicy/copomstatements/cronologicos) | Decisiones; localizar también actas e informes | Comunicados gratuitos; conservar original portugués cuando corresponda |
| BoC, Canadá | [Monetary policy](https://www.bankofcanada.ca/core-functions/monetary-policy/) | Decisiones, MPR, deliberaciones y discursos | Publicaciones gratuitas; HTML/PDF |
| SNB, Suiza | [Monetary policy decisions](https://www.snb.ch/en/the-snb/mandates-goals/monetary-policy/decisions) | Evaluaciones y decisiones monetarias | Gratuitas; documentos por evaluación |
| Riksbank, Suecia | [Monetary policy](https://www.riksbank.se/en-gb/monetary-policy/) | Decisiones, actas e informes | Gratuitas; HTML/PDF |
| Norges Bank, Noruega | [Monetary policy](https://www.norges-bank.no/en/topics/Monetary-policy/) | Decisiones, informes y discursos | Gratuitas; HTML/PDF |
| RBNZ, Nueva Zelanda | [Monetary policy](https://www.rbnz.govt.nz/monetary-policy) | Statements y decisiones del comité | Publicaciones gratuitas; archivos por edición |
| RBI, India | [Portal oficial](https://www.rbi.org.in/) | Press Releases, MPC, Speeches y Reports | Seleccionar documentos públicos gratuitos; no confundir con publicaciones ofrecidas por suscripción |
| Banxico, México | [Anuncios de decisiones](https://www.banxico.org.mx/publicaciones-y-prensa/anuncios-de-las-decisiones-de-politica-monetaria/anuncios-politica-monetaria-t.html) | Anuncios; localizar minutas e informes relacionados | Anuncios gratuitos; PDF |
| Banco de España | [Publicaciones](https://www.bde.es/wbe/es/publicaciones/) | Proyecciones, análisis y estabilidad financiera | Gratuitas; HTML/PDF |
| RBA, Australia | [Monetary policy](https://www.rba.gov.au/monetary-policy/) | Decisiones, actas e informes monetarios | Gratuitas; HTML/PDF |
| PBoC, China | [Portal inglés](https://www.pbc.gov.cn/en/); [ejemplo de informe publicado](https://www.pbc.gov.cn/en/3688229/3688353/3688362/5385746/index.html) | Monetary Policy Reports y comunicaciones | Ejemplo de acceso público; cobertura y automatización por comprobar |

**Calidad:** registrar país, institución, idioma y tipo de documento. Banco de España pertenece al Eurosistema: las decisiones monetarias del euro se obtienen del BCE; una traducción o republicación no es una decisión adicional. Para discursos internacionales, BIS es una alternativa gratuita útil, con cobertura parcial. Preferir descargas o feeds oficialmente ofrecidos antes de desarrollar extracción de páginas; no presuponer que todos ofrecen JSON/RSS/API.

## 6. Organismos: Banco Mundial, FMI WEO y OCDE

<a id="organismos"></a>

**Elección:** informe original más los datos de su edición. El texto explica supuestos y riesgos; las tablas permiten conservar valores y proyecciones sin errores de extracción de PDF.

| Información solicitada | Fuente recomendada | ¿Gratis? | Acceso previsto |
|---|---|---|---|
| Banco Mundial: Global Economic Prospects | [GEP oficial](https://www.worldbank.org/en/publication/global-economic-prospects) | Sí | Informe completo y capítulos PDF |
| Datos GEP | [Catálogo de datos GEP](https://datacatalog.worldbank.org/search/dataset/0037888/global-economic-prospects) | Sí; ficha declara CC BY 4.0 | Descargas de tablas; formatos según recurso |
| FMI: World Economic Outlook | [WEO oficial](https://www.imf.org/en/Publications/WEO) | Sí para los informes digitales propuestos | PDF por edición y materiales asociados |
| Series y proyecciones WEO | [Dataset WEO del FMI](https://data.imf.org/Datasets/WEO); [ejemplo de edición histórica](https://www.imf.org/en/publications/weo/weo-database/2024/october) | Sí | Portal de datos y archivos de edición |
| OCDE: Economic Outlook | [Serie oficial](https://www.oecd.org/en/publications/oecd-economic-outlook_16097408.html) | Sí, publicación digital | Informes y enlaces a datos |

**Acceso:** la OCDE [anunció acceso gratuito a sus datos, publicaciones y análisis desde julio de 2024](https://www.oecd.org/en/about/news/press-releases/2024/07/oecd-data-publications-and-analysis-become-freely-accessible.html). El estado “bloqueada” de la captura no justifica comprar una suscripción. Un problema técnico se registra y se resuelve mediante rutas oficiales alternativas o descarga manual.

**Tiempo:** preservar cada edición o vintage. Una serie revisada hoy no reproduce lo conocido en 2008; una proyección WEO tampoco es un dato observado. Conservar unidad, país, periodo, condición observado/proyectado y edición.

## 7. Datasets abiertos: Hugging Face, FOMC y Beige Book

<a id="datasets"></a>

**Elección:** útiles para prototipos y anotaciones; las fuentes oficiales de los apartados anteriores siguen siendo la referencia para texto y fechas.

| Dataset o tipo | Fuente recomendada | ¿Gratis? | Calidad, licencia y limitación |
|---|---|---|---|
| Comunicaciones de múltiples bancos | [aufklarer/central-bank-communications](https://huggingface.co/datasets/aufklarer/central-bank-communications) | Descarga pública gratuita; ficha declara CC BY 4.0 | Candidato compatible con la descripción de la captura; no se ha demostrado que sea exactamente su dataset. Incluye corpus/anotaciones; contrastar procedencia y fechas |
| Frases FOMC con tono anotado | [gtfintechlab/fomc_communication](https://huggingface.co/datasets/gtfintechlab/fomc_communication) | Sí; ficha declara CC BY-NC 4.0 | Adecuado para etiquetas hawkish/dovish/neutral; restringe uso comercial. No sustituye un archivo completo de documentos |
| Discursos de bancos centrales | [istat-ai/cb-speeches](https://huggingface.co/datasets/istat-ai/cb-speeches) | Acceso condicionado a cuenta y aceptación/solicitud según ficha | Revisar permisos del contenido original además de la licencia declarada. Para empezar, preferir descarga directa BIS |
| Beige Book | [Archivo oficial Fed](https://www.federalreserve.gov/monetarypolicy/beigebook/default.htm) | Sí, sin cuenta | Mejor procedencia para el texto completo; la captura no identifica un dataset concreto de Beige Book |

**Calidad y tiempo:** una etiqueta de sentimiento es una interpretación, no una afirmación oficial del banco. Registrar quién anotó, método y versión. Si un dataset contiene `meeting_date` pero no evidencia de publicación, esa fecha no basta para TimeGate. Usarlo como semilla y reconstruir la disponibilidad desde el original. No atribuir CC BY a todos los datasets de una plataforma.

Para la biblioteca general, aceptar textos cuya procedencia, licencia y extracción se hayan validado aunque su fecha exacta sea desconocida. Reconstruir la disponibilidad precisa solo cuando haga falta para el modo histórico; no dejar estos corpus fuera únicamente por ese dato ausente.

## 8. Estabilidad financiera

<a id="estabilidad"></a>

**Elección:** informe completo de la institución que lo produce, junto con tablas y anexos publicados. Son fuentes primarias para saber qué riesgos comunicaba cada organismo.

| Información solicitada | Fuente recomendada | ¿Gratis? | Acceso previsto |
|---|---|---|---|
| Fed Financial Stability Report | [FSR Fed](https://www.federalreserve.gov/publications/financial-stability-report.htm) | Sí | PDF/HTML y ediciones anteriores |
| BCE Financial Stability Review | [FSR BCE](https://www.ecb.europa.eu/pub/financial-stability/fsr/html/index.en.html) | Sí | Informe, capítulos y artículos |
| BoE Financial Stability Report | [Archivo FSR BoE](https://www.bankofengland.co.uk/sitemap/financial-stability-report) | Sí | HTML/PDF por edición |
| ESRB: informes de riesgo sistémico | [Reports ESRB](https://www.esrb.europa.eu/pub/reports/html/index.en.html) | Sí | Informes temáticos, anuales y anexos |
| OFR: evaluación y análisis | [Annual Reports OFR](https://www.financialresearch.gov/annual-reports/) | Sí | Informes anuales y material asociado |
| FSB: supervisión y vulnerabilidades | [Publications FSB](https://www.fsb.org/publications/) | Sí | Informes por tema y fecha |

**Calidad:** conservar gráficos y referencias a tablas junto al texto. “Estabilidad financiera” agrupa series distintas: no tratar un informe anual OFR y un FSR semestral como observaciones idénticas. Deduplicar informe completo y capítulos, sin perder su relación. Los recuentos de la captura no representan necesariamente todo el archivo actual.

## 9. BIS Quarterly Review y Working Papers

<a id="bis-papers"></a>

**Elección:** catálogo oficial BIS, tanto para el documento como para número de serie, autores y fecha.

| Información solicitada | Fuente recomendada | ¿Gratis? | Acceso previsto |
|---|---|---|---|
| Quarterly Review completo y artículos | [BIS Quarterly Review](https://www.bis.org/publications/qr) | Sí | Ediciones y artículos HTML/PDF |
| Working Papers | [BIS Working Papers](https://www.bis.org/publications/working-paper) | Sí | Catálogo y PDFs individuales |

**Calidad y tiempo:** separar el informe trimestral de sus artículos y vincularlos con un documento padre. Un Working Paper no equivale automáticamente a un artículo revisado por una revista. Guardar la versión y el número de la serie. Para TimeGate, contrastar la fecha de lanzamiento con la portada y la ficha: la fecha de modificación de una página migrada puede ser posterior a la publicación original.

## 10. FMI Global Financial Stability Report — GFSR

<a id="gfsr"></a>

**Elección:** FMI como editor original. Es una fuente central para riesgos financieros globales y debe formar parte del corpus aunque alguna ruta de acceso automático falle.

| Información solicitada | Fuente recomendada | ¿Gratis? | Acceso previsto |
|---|---|---|---|
| GFSR completo | [Serie oficial GFSR](https://www.imf.org/en/Publications/GFSR) | Sí, PDF digital | Ediciones, capítulos y anexos |
| Ejemplo de PDF completo | [GFSR abril de 2026, texto inglés](https://www.imf.org/-/media/files/publications/gfsr/2026/april/english/text.pdf) | Sí, enlace público | PDF original descargable |
| Ruta editorial alternativa | [IMF eLibrary: estabilidad financiera](https://www.elibrary.imf.org/subject/082) | Acceso gratuito a materiales abiertos; comprobar el documento concreto | Ficha editorial y versiones disponibles |

**Calidad y acceso:** la captura marca “BLOQUEADA” por WAF; eso describe un obstáculo de adquisición, no un precio. Si falla la descarga automática, usar el enlace oficial al PDF o descarga manual con registro de procedencia. No es necesario empezar comprando el informe impreso. Una traducción de un resumen ejecutivo no sustituye el informe completo en inglés.

**Tiempo:** ciertos registros de ediciones antiguas pueden reflejar la incorporación al portal. Comprobar fecha original de edición/lanzamiento y conservar evidencia; no usar sin más la fecha reciente de la ficha migrada.

## 11. Papers de crisis

<a id="papers-crisis"></a>

**Elección:** buscar primero el documento publicado por su institución o una versión autorizada del autor. La calidad científica depende de cada trabajo; el proveedor de descarga, por sí solo, no demuestra revisión o robustez.

| Tipo de fuente | Recomendación y enlace | ¿Gratis? | Uso y límites |
|---|---|---|---|
| Working papers institucionales | [BIS](https://www.bis.org/publications/working-paper) y [BCE](https://www.ecb.europa.eu/press/research-publications/working-papers/html/index.en.html) | Sí, publicaciones propuestas | Original, autores y serie; buenas primeras fuentes sobre crisis financieras |
| Preprints | [arXiv](https://arxiv.org/) y [documentación de su API](https://info.arxiv.org/help/api/index.html) | Lectura pública gratuita | API de metadatos; PDF y versiones individuales. Cobertura económica selectiva; no confundir preprint y versión revisada |
| Working papers y manuscritos | [SSRN](https://www.ssrn.com/); [explicación del acceso del proveedor](https://blog.ssrn.com/2020/08/03/what-is-the-ssrn-login-worth-core-values-and-commitments-to-researchers/) | Muchos gratuitos; puede requerir cuenta gratuita | Comprobar archivo concreto y condiciones; no asumir API masiva pública |
| NBER Working Papers | [Condiciones de acceso NBER](https://www.nber.org/subscribe) | Parcial: cupos, acceso elegible o suscripción | Priorizar copia autorizada del autor/institución si el PDF exige pago; no es una fuente totalmente gratuita |
| Descubrimiento por DOI | [Crossref REST API](https://www.crossref.org/documentation/retrieve-metadata/rest-api/) | Metadatos públicos gratuitos | Título, autores, DOI y fechas; no entrega necesariamente texto completo |
| Localizar copias de acceso abierto | [Unpaywall API v2](https://unpaywall.org/api/v2) | API gratuita con identificación por email y límites | Consultar por DOI; enlaza ubicaciones abiertas, sin garantizar una copia para todo paper |

**Orden recomendado:** original institucional → versión editorial abierta → manuscrito autorizado del autor → preprint documentado. Si solo queda una versión de pago, guardar referencia y estado pendiente; decidir más adelante si merece comprarla.

**Alcance:** la imagen no aporta la bibliografía de sus 386 papers/37 crisis. No se puede afirmar que se haya localizado ese conjunto exacto. Antes de recolectarlo, definir crisis, criterios de inclusión y lista de referencias. Deduplicar por DOI y conservar separadas las versiones que cambian resultados.

## 12. Huellas de crisis

<a id="huellas"></a>

**Elección:** bases documentadas de crisis como referencia, y construcción propia de las “huellas”. Las 18 crisis y 7 familias de la captura son una selección/taxonomía del proyecto; no identifican por sí mismas una descarga existente.

| Información necesaria | Fuente recomendada | ¿Gratis? | Calidad y limitación |
|---|---|---|---|
| Episodios de crisis europeas | [Financial crises database ESRB](https://www.esrb.europa.eu/pub/financial-crises/html/index.en.html) | Sí, descarga pública | Definiciones, episodios y documentación institucional |
| Crisis bancarias internacionales | [FMI: Systemic Banking Crises Database 1970–2025](https://www.elibrary.imf.org/abstract/journals/001/2026/094/article-A001-en.xml) | Trabajo digital abierto | Actualización documentada; localizar anexos y distinguir episodio/país |
| Referencia histórica del FMI | [Laeven y Valencia, 2018](https://www.elibrary.imf.org/abstract/journals/001/2018/206/article-A001-en.xml) | Texto digital abierto | Versión anterior útil para reproducibilidad; no mezclar vintages |
| Series macrofinancieras de largo plazo | [Jordà–Schularick–Taylor Macrohistory Database](https://www.macrohistory.net/database/) | Sí; CC BY-NC-SA 4.0 según sus términos | Datos, cronología y documentación; uso no comercial y condiciones de atribución/compartición |
| Familias, señales y huellas | Elaboración propia a partir de las bases y documentos anteriores | Sin compra de un dataset propio; procesamiento aparte | Definir variables, reglas, evidencias y versión de la clasificación |

**Tiempo:** una crisis identificada retrospectivamente en una base publicada en 2026 no era una etiqueta disponible para operar en 2008. Separar el catálogo ex post para evaluación de los indicadores y documentos disponibles en cada fecha. Reconciliar diferencias de definición entre fuentes; no fusionar episodios solo por compartir año.

En consulta general, el análisis retrospectivo es conocimiento útil para los agentes y puede incorporarse con su metodología y procedencia. La restricción de disponibilidad anterior al evento se reserva a backtests o simulaciones históricas.

## 13. Literatura de cartera y CIO

<a id="cartera"></a>

**Elección:** textos académicos originales o versiones autorizadas para construcción de carteras; organismos oficiales para normativa y guías de idoneidad. Son dos grupos relacionados, con criterios de calidad distintos.

| Información solicitada | Fuente recomendada | ¿Gratis? | Criterio de calidad |
|---|---|---|---|
| Teoría y evidencia sobre carteras | Versiones abiertas del editor, repositorios universitarios y working papers [BIS](https://www.bis.org/publications/working-paper) / [BCE](https://www.ecb.europa.eu/press/research-publications/working-papers/html/index.en.html) | Muchos trabajos sí; libros y artículos comerciales pueden costar | Autoría, método, revisión y versión; usar DOI y rutas abiertas del apartado 11 |
| Datos de factores y carteras para contrastar papers | [Kenneth French Data Library, Dartmouth](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) | Descargas públicas gratuitas | Fuente académica de datos; complemento, no sustituto de la literatura |
| MiFID II | [Directiva 2014/65/UE en EUR-Lex](https://eur-lex.europa.eu/legal-content/ES/TXT/?uri=CELEX:32014L0065) | Sí | Texto oficial; seleccionar versión consolidada y fecha aplicable |
| Idoneidad MiFID II | [ESMA: directrices actualizadas](https://www.esma.europa.eu/document/guidelines-certain-aspects-mifid-ii-suitability-requirements-1) | Sí | Directrices oficiales; conservar edición, traducción y fechas |
| Guías nacionales | [Guías técnicas CNMV](https://www.cnmv.es/portal/legislacion/guias-tecnicas?lang=es) | Sí | Publicaciones del supervisor español; identificar documento y vigencia |

**Calidad y alcance:** la captura no identifica las 34 referencias de cartera ni los 20 documentos de idoneidad: hace falta su bibliografía para reproducir la colección exacta. Idoneidad y conveniencia son evaluaciones distintas; etiquetar correctamente las guías. Los memos privados de un CIO no son necesariamente públicos ni sustituibles por un paper: incorporar únicamente material publicado o autorizado.

**Tiempo:** conservar versiones históricas de normativa, guías y literatura. Un texto actualizado no representa automáticamente las reglas o conocimientos disponibles años antes; los datasets empíricos también pueden sufrir revisiones.

## 14. La literatura llega tarde

<a id="literatura-tarde"></a>

**Elección:** construir una auditoría de disponibilidad por versión. Esta tarjeta describe un análisis derivado, no un proveedor adicional del que descargar un corpus ya terminado.

**Prioridad actual:** conservar fechas y versiones durante la ingesta, pero posponer esta auditoría detallada hasta que se necesite un estudio histórico. Un paper posterior a una crisis puede aportar mucho a la biblioteca actual de los agentes.

| Evidencia necesaria | Fuente de mayor calidad | ¿Gratis? | Qué registrar |
|---|---|---|---|
| Primera versión pública de un preprint | Historial de versiones en [arXiv](https://arxiv.org/) o ficha original [SSRN](https://www.ssrn.com/) | Metadatos públicos; acceso concreto según portal | Identificador, versión y fecha pública respaldada |
| Publicación anticipada/editorial | Ficha del editor y [Crossref](https://www.crossref.org/documentation/retrieve-metadata/rest-api/) | Metadatos normalmente gratuitos; texto puede ser de pago | Online first, publicación formal y fuente de cada fecha |
| Copia previa autorizada | Repositorio institucional del autor; [Unpaywall](https://unpaywall.org/api/v2) ayuda a localizarla | Copias abiertas gratuitas cuando existen | Fecha de depósito y de apertura pública, que pueden ser distintas |
| Recuento y retraso frente a una crisis | Cálculo propio con corpus, versiones y fecha de corte explícitos | Fuentes anteriores; coste de procesamiento aparte | Incluido/excluido/incierto y razón verificable |

**Regla:** incluir solo la versión cuya disponibilidad pública anterior o igual al corte esté respaldada por evidencia. Una copia encontrada hoy no prueba que pudiera descargarse entonces. Si falta la fecha, marcar “incierta” y excluir de la evaluación histórica estricta hasta resolverla; no inventar un retraso medio.

Para comprobar “18 de 77 disponibles el 15-09-2008”, primero necesitamos los 77 identificadores y su historial. Tampoco se ha verificado el “+1,6 años” de la imagen. Informar denominador, tratamiento de versiones y casos inciertos antes de presentar esos valores como resultado.

## Reglas comunes para el futuro pipeline

La columna “Cómo se trata” de la captura se traduce en estas decisiones de diseño; se implementarán cuando empiece la fase de código.

| Necesidad | Qué conservar o hacer |
|---|---|
| Original verificable | HTML/PDF original, URL, institución, tipo, idioma, fecha de descarga y hash del archivo |
| Catálogo y versiones | Título, autores, identificador/DOI, versión, tema, país, idioma, estado editorial conocido, términos y relación con traducciones y documento padre |
| Fechas como contexto | Fecha del evento/publicación cuando se conozcan y fecha de descarga; mostrar precisión e incertidumbre sin bloquear el corpus general |
| TimeGate opcional | Para backtests: primera disponibilidad pública de la versión y evidencia; no confundir con descarga ni inventar horas |
| Troceado de bancos centrales | Párrafos y una unidad por pregunta/respuesta, conservando contexto y orden; el rango de 60–350 palabras de la captura es una pauta inicial |
| Troceado de informes | Sección, aproximadamente 800 tokens como pauta, con vínculo al documento padre y referencias a tablas |
| Búsqueda y RAG | Catálogo y búsqueda textual del texto validado; embeddings por lotes prioritarios, ampliables según utilidad y coste; filtros temporales solo en modo histórico |
| Consulta de agentes | Buscar por pregunta/tema/fuente/tipo, recuperar pasajes, ampliar contexto y abrir el documento completo con fuente, versión, página/sección y URL |
| Cobertura útil | Medir documentos únicos utilizables por familia y tema, errores y referencias sin texto; evaluar relevancia y respaldo de citas con preguntas de referencia |
| Tono y diferencias | Resultados derivados con modelo/método/versión y fecha de cálculo; vinculados al texto fuente |
| Destino Cloudflare | R2 para originales, texto y JSONL; D1 para metadatos consultables. Las fuentes no se conectan directamente a R2: el recolector escribe los objetos y el catálogo |

No es necesario contratar una fuente de datos para comenzar la colección propuesta. Primero validar un pequeño lote gratuito y diverso; ampliar luego la carga retrospectiva. Como punto de partida, refrescos semanales para comunicaciones y mensuales para literatura, ajustables al piloto. Mantener los corpus fuera de Git y conservar en el repositorio documentación, configuración y, más adelante, el código de adquisición.

## Orden recomendado para empezar sin pagar por fuentes

1. **Piloto diverso:** paquete de discursos BIS, muestra de BIS QR/Working Papers y papers de crisis, informes de estabilidad/GFSR, comunicaciones Fed/BCE y referencias de cartera abiertas. Validar formatos y utilidad de varias familias.
2. **Consulta inicial:** buscar pasajes en texto validado, abrir contexto y comprobar citas; medir cobertura y calidad. Subir PDFs a R2 requiere además esa capa de recuperación para que los agentes los aprovechen.
3. **Cargas amplias:** ampliar series y años del piloto, BoE/BoJ, GEP/WEO/OCDE, otros bancos y literatura de cartera/regulación. Priorizar descargas masivas oficiales y versiones abiertas autorizadas.
4. **RAG y agentes:** combinar búsqueda textual y semántica, ampliar embeddings por lotes y conectar una interfaz de consulta con citas. Mantener accesibles los documentos sin embeddings.
5. **Derivados opcionales:** tono, diferencias y huellas si aportan utilidad; reconstrucción temporal detallada y estudio de literatura tardía cuando se soliciten evaluaciones históricas.

**Criterio de aceptación para la biblioteca general:** original accesible y conservado, procedencia y términos registrados, texto suficientemente completo, versión/hash e identificador reproducible. Las fechas conocidas se conservan y las ausentes se marcan; no se exige precisión extrema. Referencias sin texto y fallos graves de extracción permanecen como pendientes. Para análisis histórico estricto se añade evidencia de disponibilidad antes del corte. El volumen se evalúa junto con relevancia, diversidad de fuentes y calidad de las citas.
