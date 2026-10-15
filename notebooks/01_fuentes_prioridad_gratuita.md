# Bancos centrales, riesgo sistémico y literatura — sonda de fuentes y disponibilidad temporal

**Fase 2 del TFM · Responsable: Fernando Dapena Tauste · Revisión: 15 de octubre de 2026.**

Antes de ampliar el corpus, probamos cada fuente con poco volumen y medimos **qué entrega, qué fechas acredita, con qué retraso publica, cuánto texto se puede extraer, cuánto ocupa y qué falla**. La estructura sigue la [sonda de empresas de Daniel](https://github.com/Romequinco/PRUEBAS_DETECCION_REGIMENES_DE_MERCADO_TFM/blob/fase2/t3-empresas/notebooks/fase2/t3_01_sonda_empresas.ipynb): configuración común, adquisición, cobertura, auditoría temporal, texto, volumen y resumen calculado.

Se mantienen los **14 apartados independientes**. Fed, BCE, cada banco y cada familia aportan información distinta; sus filas y gráficos conservan esa separación. El código de adquisición y medición vive en `src/macro_riesgo/sonda/`; este notebook lo ejecuta y presenta resultados. Las URLs y las 94 familias de muestras están en `config/sources/sonda_fuentes.json`.

Una sonda pequeña no demuestra el inicio del histórico ni representa todo el archivo. Los recuentos de la captura del proyecto tampoco son resultados de esta ejecución. La biblioteca general admite texto validado con incertidumbre temporal explícita; el ejemplo de **TimeGate histórico es estricto por versión**.

### Antes de ejecutar

1. Usa el kernel `Python (.venv - Bancos centrales TFM)` e instala `requirements.txt` si recreas el entorno.
2. Ejecuta la configuración y después el apartado que quieras. `MODO_RAPIDO=True` limita a 3 documentos por familia descubierta, 3 filas y 40 páginas por PDF. Las URLs de edición fija entregan una edición. El modo ampliado sigue siendo una sonda, no una descarga de todo el histórico.
3. `HF_TOKEN` y `UNPAYWALL_EMAIL` se leen de variables de entorno si se necesitan. Las demás muestras no requieren API key. Los errores de acceso y los permisos pendientes quedan en sus tablas.
4. Se conservan originales por SHA-256 en `data/cache/sonda/`, texto y fragmentos en `data/local/sonda/` y tablas en `data/exports/sonda/`, fuera de Git. La caché se reutiliza; `usar_cache=False` vuelve a consultar y conserva los objetos antiguos.

La lectura completa de la referencia y las decisiones de adaptación están en [Análisis del notebook de Daniel](../docs/ANALISIS_NOTEBOOK_REFERENCIA.md).

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

## Configuración común

Los apartados comparten únicamente la configuración y las funciones de medición. Cada apartado se puede ejecutar por separado después de esta celda. `CORTE_UTC` incluye zona horaria; el corte es una elección del análisis, no la hora de descarga.

```python
from pathlib import Path
import sys

import pandas as pd
from IPython.display import display

RAIZ = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / "config/sources/sonda_fuentes.json").exists()), None)
if RAIZ is None:
    raise RuntimeError("Abre el notebook desde este repositorio.")
if str(RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(RAIZ / "src"))

from macro_riesgo.sonda import ConfigSonda, Sonda
from macro_riesgo.sonda.analysis import mostrar_apartado

pd.set_option("display.max_columns", 24)
pd.set_option("display.max_colwidth", 100)

MODO_RAPIDO = True
CORTE_UTC = "2026-10-14T00:00:00+00:00"

cfg = ConfigSonda(
    raiz=RAIZ,
    max_por_familia=3 if MODO_RAPIDO else 10,
    max_filas=3 if MODO_RAPIDO else 10,
    max_paginas=40 if MODO_RAPIDO else 200,
    usar_cache=True,
    corte_utc=CORTE_UTC,
)
sonda = Sonda(cfg)
print("Responsable: Fernando Dapena Tauste")
print("Raíz:", RAIZ)
print("Modo:", "rápido" if MODO_RAPIDO else "ampliado (no es el histórico completo)")
print("Corte histórico:", CORTE_UTC)
```

## Cómo leer las mediciones y evitar look-ahead

| Campo | Qué significa y qué permite |
|---|---|
| `fecha_evento` | Reunión, fin de periodo o referencia explícita. No fecha de publicación. En el apartado 14 es el corte ilustrativo de la GFC. |
| `fecha_publicacion` | Fecha anunciada para el documento o metadato consultado. La evidencia y la precisión acompañan cada fila. No demuestra por sí sola que el texto actual sea la versión original. |
| `fecha_aplicacion` | Entrada en vigor/effectividad cuando el texto la declara. Si no existe o no se extrae, queda vacía; un paper no tiene por defecto una fecha de aplicación. |
| `retraso_evento_dias` | Publicación menos evento/referencia en días de calendario. Solo se calcula con un par de fechas identificadas. |
| `retraso_aplicacion_dias` | Publicación menos aplicación. Negativo significa anuncio anterior a la entrada en vigor. No se confunde con retraso respecto a la reunión. |
| `retrieved_at` / `consulta_utc` | Primera descarga de esos bytes / ejecución de la medición. Nunca sustituyen publicación. |
| `version_id` / `sha256` | Identifican los bytes del original; el hash evita mezclar revisiones, pero no prueba su historia pública. |
| `disponible_desde_utc` | Límite conservador para el texto completo de una versión acreditada; no se rellena con estimaciones. |

**Política `sonda-timegate-v1`:** la vista histórica admite únicamente texto completo, versión acreditada y disponibilidad anterior o igual al corte. Una fecha con hora y zona se normaliza a UTC. Si solo hay día y conocemos la zona, una versión acreditada se admite desde el **inicio del día local siguiente**; esa es una política conservadora, no una hora de publicación observada. Mes/año, hora sin zona y fecha desconocida no bastan para el filtro. No se modelan calendarios bursátiles ni aperturas/cierres de mercado.

Los HTML/PDF actuales de los portales conservan su publicación conocida, pero permanecen **pendientes de acreditar la versión histórica**: descargar hoy no demuestra qué texto había antes. arXiv consulta y descarga **v1 explícitamente** para unir el texto con su fecha. Un PDF truncado por el límite de páginas, una ficha bibliográfica, un abstract, un dataset revisado o una etiqueta ex post no se presentan como texto completo disponible en el pasado. Last-Modified, CreationDate, fecha de reunión y fecha de depósito no se convierten automáticamente en publicación.

### Salidas de cada apartado

1. Inventario y descarga limitada por familia.
2. Cobertura de la muestra: primera/última publicación, pares válidos, mediana y P10–P90 de retrasos, extracción y volumen.
3. Se conservan los gráficos de **publicaciones y volumen textual observado**. La cobertura y los retrasos respecto a aplicación quedan en las tablas.
4. Un único gráfico de **retraso entre evento y publicación** muestra una fila por documento: fuente y familia, fechas «evento → publicación» y días transcurridos. **0 días** significa publicación el mismo día; un valor negativo indica que se publicó antes del evento. Solo se dibuja cuando hay pares de fechas precisos y verificables; una fecha ausente no se transforma en cero. En familias sin evento asociado, sin ambas fechas o con un corte meramente ilustrativo se omite el gráfico y se explica el motivo.
5. Documento/fila de ejemplo con URL y hash, errores y TimeGate al corte. Cuando hay dos textos completos de la misma familia se compara su vocabulario; es una medida descriptiva, no sentimiento ni señal predictiva.

Los PDFs se leen hasta el límite configurado; las tablas conservan filas y unidades cuando las entrega la fuente. El número de palabras mide extracción, no calidad científica. Las primeras/últimas publicaciones de una muestra pequeña no permiten extrapolar cobertura ni GB del histórico.

**Ejemplo de lectura:** si una reunión se celebra el 28/01 y sus actas se publican el 18/02, la barra indica **21 días**. La fecha de aplicación de una decisión, el fin de un periodo estadístico y un corte elegido para analizar una crisis son referencias diferentes; no sustituyen a la fecha del evento. Las fechas del gráfico tienen precisión diaria y no permiten inferir un retraso en horas.

**Figuras de esta revisión (15/10/2026):** regeneradas a partir de los CSV y JSONL locales de la muestra guardada, sin repetir las descargas. Esta revisión actualiza la presentación; las fechas de consulta de cada registro se conservan.

## 1. Fed

<a id="fed"></a>

**Qué es y qué decide.** La [Reserva Federal o Fed](https://www.federalreserve.gov/aboutthefed.htm) es el banco central de Estados Unidos. Busca mantener los precios estables y favorecer el máximo empleo. Dentro de la Fed, el **FOMC** (Comité Federal de Mercado Abierto) es el comité que toma las principales decisiones de política monetaria, como el nivel objetivo del tipo de interés a corto plazo y las medidas sobre su cartera de activos.

**Cómo influye.** Sus decisiones afectan al coste de los préstamos en dólares, la financiación de empresas y las condiciones de ahorro. Una subida de tipos tiende a frenar el gasto y la inversión; una bajada tiende a facilitarlos. También pueden cambiar las valoraciones de bonos y acciones, el dólar y la financiación internacional. Las actas y los discursos ayudan a entender qué preocupa a la Fed y qué podría decidir después; el mercado reacciona especialmente a lo que cambia respecto a lo esperado.

**Elección:** Federal Reserve como fuente principal. Conserva el documento institucional y permite relacionarlo con la reunión correspondiente. Los datasets externos sirven de complemento.

| Información solicitada | Fuente de mayor calidad recomendada | ¿Gratis? | Acceso previsto |
|---|---|---|---|
| Comunicados y decisiones FOMC | [Calendarios y documentos FOMC](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm) | Sí, sin cuenta | HTML y PDF enlazados por reunión |
| Actas | [Mismo calendario FOMC](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm) y [archivo histórico](https://www.federalreserve.gov/monetarypolicy/fomc_historical.htm) | Sí | HTML/PDF; publicación separada del comunicado |
| Ruedas de prensa | [Calendario FOMC](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm) | Sí | Transcripción cuando esté publicada; vídeo como complemento |
| Beige Book | [Archivo oficial Beige Book](https://www.federalreserve.gov/monetarypolicy/beige-book-default.htm) | Sí | HTML/PDF por edición y distrito |
| Transcripciones y material histórico | [FOMC Historical Materials](https://www.federalreserve.gov/monetarypolicy/fomc_historical.htm) | Sí | PDF y documentos por año |
| Discursos | [Speeches de la Fed](https://www.federalreserve.gov/newsevents/speeches.htm) | Sí | HTML/PDF con autor y fecha |

**Calidad y tiempo:** preferir HTML para extraer párrafos y conservar el PDF original cuando exista. Las actas actuales suelen publicarse unas tres semanas después; las transcripciones completas se publican con aproximadamente cinco años de retraso. Usar la fecha efectiva de cada publicación: los “+21 días” y “+6 años” de la captura no son reglas universales para todo el histórico. El archivo enlazado documenta el régimen de publicación.

### Sonda: adquisición y mediciones

**Medición específica.** Comunicado, actas, rueda de prensa, Beige Book, transcripción histórica y discurso son series distintas. En las actas se cruza el identificador oficial de la reunión con la etiqueta **Released** del calendario; no se suma una constante de 21 días. Un aviso `For release at` con EST/EDT permite conservar la hora real del comunicado. La publicación de una transcripción histórica no se obtiene sumando cinco años ni usando el año de su carpeta.

```python
inventario_01 = sonda.inventario(1)
display(inventario_01[["fuente", "tipo", "adaptador"]])
muestra_01 = sonda.ejecutar(1)
```

```python
resumen_01 = mostrar_apartado(sonda, 1, '1. Fed')
```

## 2. BCE

<a id="bce"></a>

**Qué es y qué decide.** El [Banco Central Europeo (BCE)](https://www.ecb.europa.eu/ecb/orga/tasks/html/index.en.html) dirige la política monetaria de los países que utilizan el euro. Junto con sus bancos centrales nacionales forma el **Eurosistema**. Su objetivo principal es mantener la estabilidad de precios. El Consejo de Gobierno decide los tipos oficiales y otras medidas que influyen en las condiciones de financiación.

**Cómo influye.** Sus decisiones se transmiten al coste del crédito para familias, empresas y Estados, y pueden afectar al euríbor, las hipotecas variables, los bonos y el euro. Por ejemplo, unos tipos más altos tienden a encarecer los nuevos préstamos y a moderar la demanda. Las comparecencias y las actas resumidas permiten interpretar cómo valora la inflación y el crecimiento. Esta [transmisión a la economía](https://www.ecb.europa.eu/mopo/intro/transmission/html/index.en.html) lleva tiempo y depende también de la situación de los bancos y de los prestatarios.

**Elección:** portal del BCE. Mantiene decisiones, comparecencias y deliberaciones como series distintas, evitando mezclar documentos de la misma reunión.

| Información solicitada | Fuente recomendada | ¿Gratis? | Acceso previsto |
|---|---|---|---|
| Decisiones de política monetaria | [Monetary policy decisions](https://www.ecb.europa.eu/press/govcdec/mopo/html/index.en.html) | Sí | Listados y texto HTML |
| Declaración y preguntas/respuestas | [Conferencias de prensa](https://www.ecb.europa.eu/press/press_conference/html/index.en.html) | Sí | HTML; separar declaración y cada Q&A |
| Accounts, equivalentes a actas resumidas | [Monetary policy accounts](https://www.ecb.europa.eu/press/accounts/html/index.en.html) | Sí | HTML con fecha de reunión y de publicación |
| Discursos | [Speeches](https://www.ecb.europa.eu/press/key/html/index.en.html) | Sí | HTML/PDF según documento |

**Calidad y tiempo:** guardar el idioma original y vincular traducciones como versiones relacionadas. Las accounts están disponibles después de la reunión; usar la fecha real de publicación, sin imponer “+28 días” a todas las ediciones. La fecha de reunión identifica el evento; la fecha de publicación decide cuándo podía conocerlo el pipeline.

### Sonda: adquisición y mediciones

**Medición específica.** Decisión, declaración, Q&A, accounts y discursos se mantienen separados. Las accounts identifican la reunión en el encabezado y la publicación en `ecb-publicationDate`; el gráfico usa ese par real. Las decisiones pueden indicar una aplicación futura. La sección Q&A hereda la publicación del padre, pero no acredita la fecha de subida de la transcripción ni sustituye el documento completo.

```python
inventario_02 = sonda.inventario(2)
display(inventario_02[["fuente", "tipo", "adaptador"]])
muestra_02 = sonda.ejecutar(2)
```

```python
resumen_02 = mostrar_apartado(sonda, 2, '2. BCE')
```

## 3. BoE y BoJ

<a id="boe-boj"></a>

**Qué entidades son.** El [Bank of England (BoE)](https://www.bankofengland.co.uk/about) es el banco central del Reino Unido; su **MPC**, o Comité de Política Monetaria, decide el tipo oficial. El [Bank of Japan (BoJ)](https://www.boj.or.jp/en/about/outline/index.htm) es el banco central de Japón y decide su política monetaria mediante su Consejo de Política. Ambos trabajan por la estabilidad de precios y del sistema financiero, con instrumentos y contextos económicos propios.

**Cómo influyen.** El BoE afecta al crédito y las hipotecas británicas, la libra y los bonos del Reino Unido. El BoJ influye en los tipos japoneses, el yen y el coste de financiarse en esa moneda. Los cambios en la financiación en yenes también pueden afectar a inversores que piden prestado en Japón para invertir en otros mercados. Sus informes y actas ayudan a distinguir una decisión puntual de un cambio más duradero en la orientación de la política monetaria.

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

### Sonda: adquisición y mediciones

**Medición específica.** BoE y BoJ conservan filas distintas. En BoJ se leen por separado las columnas **Date of MPM** y **Release Schedule** del calendario: Summary of Opinions y Minutes pueden salir en días distintos. No se interpreta la fecha de reunión codificada en el nombre del PDF como publicación. La fecha del texto inglés no se presume idéntica a la del original japonés.

```python
inventario_03 = sonda.inventario(3)
display(inventario_03[["fuente", "tipo", "adaptador"]])
muestra_03 = sonda.ejecutar(3)
```

```python
resumen_03 = mostrar_apartado(sonda, 3, '3. BoE y BoJ')
```

## 4. Discursos BIS y Annual Report

<a id="discursos-bis"></a>

**Qué es.** El **BIS**, llamado en español [Banco de Pagos Internacionales](https://www.bis.org/about), facilita la cooperación entre bancos centrales y les presta servicios financieros. También produce investigación y alberga comités que trabajan sobre estabilidad y regulación financiera. Su archivo de discursos reúne intervenciones de responsables de distintos bancos: cada intervención debe atribuirse a su autor y a su institución.

**Cómo influye.** Su influencia procede de la cooperación, el análisis y el trabajo sobre estándares que después pueden incorporar las autoridades competentes. Los discursos permiten conocer cómo interpretan los bancos la inflación, el crédito o los riesgos. El **Annual Economic Report** analiza la economía y el sistema financiero; el **Annual Report** explica la actividad de la propia institución. Para el TFM, esta diferencia ayuda a separar información sobre la economía de información sobre el funcionamiento del BIS.

**Elección para un corpus internacional:** descarga oficial del BIS. Para el texto definitivo de un discurso concreto, contrastar con el banco emisor si hay discrepancias.

| Información solicitada | Fuente recomendada | ¿Gratis? | Acceso previsto |
|---|---|---|---|
| Discursos de bancos centrales | [Descarga oficial de discursos BIS](https://www.bis.org/speeches/central-bank/download) | Sí, sin cuenta; paquete para uso no comercial según sus condiciones | Archivos CSV UTF-8 comprimidos, por año o conjunto, desde 1996 |
| Documento individual y metadatos | [Catálogo BIS](https://www.bis.org/speeches/central-bank) | Sí | Página y documento enlazado |
| Informe económico anual | [Annual Economic Report](https://www.bis.org/publications/aer) | Sí | Capítulos e informe completo |
| Informe institucional anual | [Annual Report](https://www.bis.org/about/areport/index.htm) | Sí | Informe de actividad institucional |

**Calidad:** el paquete es especialmente útil para empezar sin reconstruir miles de páginas. BIS advierte que la cobertura no es exhaustiva y que la conversión automática puede introducir errores. Validar muestras contra los originales y guardar la fecha/versión de la descarga. Annual Economic Report y Annual Report tienen objetivos diferentes: catalogarlos por separado. No asumir que todas las publicaciones BIS comparten la licencia del paquete de discursos.

### Sonda: adquisición y mediciones

**Medición específica.** El discurso individual, el paquete CSV de 1996, el Annual Economic Report y el Annual Report institucional tienen registros propios. La fecha de un discurso es un evento; la fecha de subida al BIS o de creación del paquete es otra. El CSV conserva sus nombres de campos y no entra en TimeGate hasta contrastar originales y versiones.

```python
inventario_04 = sonda.inventario(4)
display(inventario_04[["fuente", "tipo", "adaptador"]])
muestra_04 = sonda.ejecutar(4)
```

```python
resumen_04 = mostrar_apartado(sonda, 4, '4. Discursos BIS y Annual Report')
```

## 5. Otros bancos centrales

<a id="otros-bancos"></a>

**Qué entidades son.** Este grupo amplía la muestra a bancos centrales de distintas economías. En general, sus decisiones sobre tipos y condiciones de financiación influyen en el crédito, el gasto, la inflación y su moneda. La tabla resume qué representa cada uno y un canal por el que sus decisiones pueden llegar a los mercados.

| Entidad | Explicación sencilla | Dónde pueden sentirse sus decisiones |
|---|---|---|
| **BCB — Banco Central do Brasil** | Banco central de Brasil; su comité Copom decide el tipo Selic. | Crédito en reales, deuda brasileña y valor del real. |
| **BoC — Bank of Canada** | Banco central de Canadá. | Hipotecas y financiación canadienses, bonos y dólar canadiense. |
| **SNB — Swiss National Bank** | Banco central de Suiza. | Tipos en francos suizos y condiciones monetarias que influyen en el franco. |
| **Riksbank** | Banco central de Suecia. | Coste del crédito, mercado de vivienda y corona sueca. |
| **Norges Bank** | Banco central de Noruega. | Financiación en coronas noruegas, demanda interna y tipo de cambio. |
| **RBNZ — Reserve Bank of New Zealand** | Banco central de Nueva Zelanda. | Hipotecas, actividad económica y dólar neozelandés. |
| **RBI — Reserve Bank of India** | Banco central de India. | Condiciones de crédito y liquidez en rupias y financiación de su economía. |
| **Banxico — Banco de México** | Banco central de México. | Tipos en pesos, inflación y valor del peso mexicano. |
| **Banco de España** | Banco central español, integrante del Eurosistema; participa en sus decisiones y tiene funciones de supervisión y análisis. | Aplicación de la política del euro en España, vigilancia bancaria y análisis del crédito y los riesgos. |
| **RBA — Reserve Bank of Australia** | Banco central de Australia. | Préstamos e hipotecas, demanda interna y dólar australiano. |
| **PBoC — People's Bank of China** | Banco central de China; utiliza diversos instrumentos para influir en las condiciones monetarias y el crédito. | Financiación de empresas y hogares chinos, liquidez y condiciones del renminbi o yuan. |

**Cómo interpretar el conjunto.** Una decisión local también puede afectar a otros países a través del comercio, los flujos de inversión y las divisas. Estos son canales de influencia, no predicciones sobre la dirección de cada mercado. El Banco de España participa en la política común del euro; los tipos oficiales de esa moneda se deciden en el BCE.

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

### Sonda: adquisición y mediciones

**Medición específica.** Cada banco y cada tipo conserva su propia fila en tablas y gráficos. Se consulta la edición o el listado indicado; el modo ampliado no reconstruye todo el archivo de esos bancos. Un PDF sin fecha explícita, un portal JavaScript o un bloqueo HTTP queda pendiente, sin inventar retrasos comunes. Banco de España se mantiene como fuente de análisis del Eurosistema; sus documentos no se cuentan como otra decisión monetaria del BCE.

```python
inventario_05 = sonda.inventario(5)
display(inventario_05[["fuente", "tipo", "adaptador"]])
muestra_05 = sonda.ejecutar(5)
```

```python
resumen_05 = mostrar_apartado(sonda, 5, '5. Otros bancos centrales')
```

## 6. Organismos: Banco Mundial, FMI WEO y OCDE

<a id="organismos"></a>

**Qué entidades son.** El [Banco Mundial](https://www.worldbank.org/ext/en/who-we-are) financia proyectos de desarrollo y publica análisis de la economía mundial. El [Fondo Monetario Internacional (FMI)](https://www.imf.org/en/About) vigila la evolución económica, asesora a los países y puede prestar apoyo financiero ante dificultades externas. La [OCDE](https://www.oecd.org/en/about.html), Organización para la Cooperación y el Desarrollo Económicos, compara economías y elabora datos y recomendaciones de políticas públicas.

**Cómo influyen.** Los préstamos y proyectos del Banco Mundial pueden facilitar inversión y desarrollo. Los programas del FMI pueden afectar a la financiación de un país y a las políticas acordadas dentro del programa. La OCDE influye principalmente mediante análisis y recomendaciones. Sus informes **GEP**, **WEO** y **Economic Outlook** presentan perspectivas de crecimiento y riesgos: pueden ayudar a gobiernos e inversores a revisar sus escenarios, aunque una previsión publicada es una estimación, no una decisión de tipos ni un resultado asegurado.

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

### Sonda: adquisición y mediciones

**Medición específica.** Texto y datos de GEP, WEO y Economic Outlook tienen filas distintas. El periodo de una cifra y su condición observado/proyectado no son publicación. DataMapper sirve datos actuales revisables: sin vintage acreditado no hay point-in-time. Los anexos GEP consultados no heredan la fecha de una edición distinta.

```python
inventario_06 = sonda.inventario(6)
display(inventario_06[["fuente", "tipo", "adaptador"]])
muestra_06 = sonda.ejecutar(6)
```

```python
resumen_06 = mostrar_apartado(sonda, 6, '6. Organismos: Banco Mundial, FMI WEO y OCDE')
```

## 7. Datasets abiertos: Hugging Face, FOMC y Beige Book

<a id="datasets"></a>

**Qué son estas fuentes.** **Hugging Face** es una plataforma donde distintos autores comparten modelos y conjuntos de datos. Los datasets de este apartado recopilan comunicaciones de bancos centrales y, en algunos casos, etiquetas sobre su tono. El **FOMC** es el comité de política monetaria de la Fed. El **Beige Book** es un informe de la Fed que describe la actividad económica de sus regiones a partir de contactos con empresas y otras organizaciones.

**Para qué sirven.** Estos materiales permiten estudiar si una comunicación muestra más preocupación por la inflación o por la debilidad económica. Por ejemplo, **hawkish** suele señalar preferencia por una política más restrictiva y **dovish**, por una más expansiva. Las etiquetas las asignan los autores del dataset; la plataforma y el dataset no deciden los tipos. En el TFM ayudan a analizar comunicaciones que pueden cambiar expectativas sobre futuras decisiones, mientras que el Beige Book aporta contexto sobre empleo, precios y actividad.

**Elección:** útiles para prototipos y anotaciones; las fuentes oficiales de los apartados anteriores siguen siendo la referencia para texto y fechas.

| Dataset o tipo | Fuente recomendada | ¿Gratis? | Calidad, licencia y limitación |
|---|---|---|---|
| Comunicaciones de múltiples bancos | [aufklarer/central-bank-communications](https://huggingface.co/datasets/aufklarer/central-bank-communications) | Descarga pública gratuita; ficha declara CC BY 4.0 | Candidato compatible con la descripción de la captura; no se ha demostrado que sea exactamente su dataset. Incluye corpus/anotaciones; contrastar procedencia y fechas |
| Frases FOMC con tono anotado | [gtfintechlab/fomc_communication](https://huggingface.co/datasets/gtfintechlab/fomc_communication) | Sí; ficha declara CC BY-NC 4.0 | Adecuado para etiquetas hawkish/dovish/neutral; restringe uso comercial. No sustituye un archivo completo de documentos |
| Discursos de bancos centrales | [istat-ai/cb-speeches](https://huggingface.co/datasets/istat-ai/cb-speeches) | Acceso condicionado a cuenta y aceptación/solicitud según ficha | Revisar permisos del contenido original además de la licencia declarada. Para empezar, preferir descarga directa BIS |
| Beige Book | [Archivo oficial Fed](https://www.federalreserve.gov/monetarypolicy/beige-book-default.htm) | Sí, sin cuenta | Mejor procedencia para el texto completo; la captura no identifica un dataset concreto de Beige Book |

**Calidad y tiempo:** una etiqueta de sentimiento es una interpretación, no una afirmación oficial del banco. Registrar quién anotó, método y versión. Si un dataset contiene `meeting_date` pero no evidencia de publicación, esa fecha no basta para TimeGate. Usarlo como semilla y reconstruir la disponibilidad desde el original. No atribuir CC BY a todos los datasets de una plataforma.

Para la biblioteca general, aceptar textos cuya procedencia, licencia y extracción se hayan validado aunque su fecha exacta sea desconocida. Reconstruir la disponibilidad precisa solo cuando haga falta para el modo histórico; no dejar estos corpus fuera únicamente por ese dato ausente.

### Sonda: adquisición y mediciones

**Medición específica.** Cada dataset HF mantiene identificador, campos y etiquetas originales. Un `publication_date` del dataset es evidencia secundaria; no prueba la disponibilidad de cada frase ni la fecha de sus anotaciones. La falta de permisos se registra sin detener el Beige Book original. Para un backtest habría que reconstruir documentos padres, fechas, versiones y fecha de disponibilidad de las etiquetas.

```python
inventario_07 = sonda.inventario(7)
display(inventario_07[["fuente", "tipo", "adaptador"]])
muestra_07 = sonda.ejecutar(7)
```

```python
resumen_07 = mostrar_apartado(sonda, 7, '7. Datasets abiertos: Hugging Face, FOMC y Beige Book')
```

## 8. Estabilidad financiera

<a id="estabilidad"></a>

**Qué vigilan estas entidades.** La **estabilidad financiera** consiste en que bancos, mercados y sistemas de pago puedan seguir funcionando incluso ante dificultades. Fed, BCE y BoE estudian estas amenazas en sus respectivas economías. La [Junta Europea de Riesgo Sistémico (ESRB)](https://www.esrb.europa.eu/about/html/index.en.html) vigila riesgos del conjunto del sistema financiero de la UE. La [Office of Financial Research (OFR)](https://www.financialresearch.gov/about/) aporta datos e investigación a las autoridades estadounidenses. El [Financial Stability Board (FSB), o Consejo de Estabilidad Financiera](https://www.fsb.org/about/), coordina a autoridades de distintos países para abordar vulnerabilidades financieras.

**Cómo influyen.** Sus análisis pueden orientar medidas de las autoridades, como exigir más recursos propios a los bancos para absorber pérdidas o limitar determinadas exposiciones. El ESRB emite alertas y recomendaciones; la OFR aporta evidencia y el FSB promueve coordinación y recomendaciones internacionales. Los informes ayudan a interpretar problemas como endeudamiento excesivo, dificultades para obtener efectivo o contagio entre entidades. Publicar una advertencia y adoptar una medida son hechos distintos, con fechas y efectos diferentes.

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

### Sonda: adquisición y mediciones

**Medición específica.** Fed, BCE, BoE, ESRB, OFR y FSB conservan series separadas. Fecha de edición, fin del periodo analizado, corte de datos y publicación no se intercambian. Se miden páginas, palabras y tamaño; un informe que exceda el límite de páginas se señala como extracción parcial. Los gráficos no atribuyen igual frecuencia a todas las instituciones.

```python
inventario_08 = sonda.inventario(8)
display(inventario_08[["fuente", "tipo", "adaptador"]])
muestra_08 = sonda.ejecutar(8)
```

```python
resumen_08 = mostrar_apartado(sonda, 8, '8. Estabilidad financiera')
```

## 9. BIS Quarterly Review y Working Papers

<a id="bis-papers"></a>

**Qué publica el BIS.** El Banco de Pagos Internacionales, presentado en el apartado 4, también difunde investigación sobre bancos y mercados. Su **Quarterly Review** es una revisión trimestral de los acontecimientos financieros y de temas relevantes. Los **Working Papers** son trabajos de investigación que estudian preguntas concretas mediante datos, modelos o análisis económico.

**Cómo influyen.** Estas publicaciones aportan explicaciones sobre crédito, deuda, movimientos de capital y riesgos que pueden utilizar bancos centrales, supervisores e inversores. Por ejemplo, un estudio puede ayudar a entender por qué el endeudamiento en dólares vuelve vulnerable a una economía. Su utilidad para el TFM es ofrecer mecanismos y evidencia con los que interpretar los cambios de mercado. Las conclusiones de un trabajo deben atribuirse a sus autores; no constituyen por sí mismas una decisión de política del BIS ni de los bancos centrales.

**Elección:** catálogo oficial BIS, tanto para el documento como para número de serie, autores y fecha.

| Información solicitada | Fuente recomendada | ¿Gratis? | Acceso previsto |
|---|---|---|---|
| Quarterly Review completo y artículos | [BIS Quarterly Review](https://www.bis.org/publications/qr) | Sí | Ediciones y artículos HTML/PDF |
| Working Papers | [BIS Working Papers](https://www.bis.org/publications/working-paper) | Sí | Catálogo y PDFs individuales |

**Calidad y tiempo:** separar el informe trimestral de sus artículos y vincularlos con un documento padre. Un Working Paper no equivale automáticamente a un artículo revisado por una revista. Guardar la versión y el número de la serie. Para TimeGate, contrastar la fecha de lanzamiento con la portada y la ficha: la fecha de modificación de una página migrada puede ser posterior a la publicación original.

### Sonda: adquisición y mediciones

**Medición específica.** Quarterly Review, artículo y Working Paper son familias distintas. La ficha consultada puede contener solo resumen/metadatos; no se presume el informe completo. Se registran publicación y número/URL cuando existen. La portada de un WP con mes y año no se transforma en el primer día del mes.

```python
inventario_09 = sonda.inventario(9)
display(inventario_09[["fuente", "tipo", "adaptador"]])
muestra_09 = sonda.ejecutar(9)
```

```python
resumen_09 = mostrar_apartado(sonda, 9, '9. BIS Quarterly Review y Working Papers')
```

## 10. FMI Global Financial Stability Report — GFSR

<a id="gfsr"></a>

**Qué entidad y qué informe son.** El **Fondo Monetario Internacional (FMI)** es el organismo de cooperación económica y apoyo financiero descrito en el apartado 6. Su **Global Financial Stability Report (GFSR)** examina la salud del sistema financiero mundial: bancos, deuda, mercados y conexiones por las que un problema puede propagarse entre sectores o países.

**Cómo influye.** El informe puede ayudar a autoridades e inversores a identificar vulnerabilidades y revisar sus medidas o su exposición al riesgo. Por ejemplo, puede analizar cómo unos costes de financiación elevados dificultan el pago de deudas. El **WEO** se centra en las perspectivas de la economía; el **GFSR**, en los riesgos financieros que pueden afectarla. Las recomendaciones del GFSR orientan el debate, pero no cambian automáticamente las normas o los tipos oficiales de los países.

**Elección:** FMI como editor original. Es una fuente central para riesgos financieros globales y debe formar parte del corpus aunque alguna ruta de acceso automático falle.

| Información solicitada | Fuente recomendada | ¿Gratis? | Acceso previsto |
|---|---|---|---|
| GFSR completo | [Serie oficial GFSR](https://www.imf.org/en/Publications/GFSR) | Sí, PDF digital | Ediciones, capítulos y anexos |
| Ejemplo de PDF completo | [GFSR abril de 2026, texto inglés](https://www.imf.org/-/media/files/publications/gfsr/2026/april/english/text.pdf) | Sí, enlace público | PDF original descargable |
| Ruta editorial alternativa | [IMF eLibrary: estabilidad financiera](https://www.elibrary.imf.org/subject/082) | Acceso gratuito a materiales abiertos; comprobar el documento concreto | Ficha editorial y versiones disponibles |

**Calidad y acceso:** la captura marca “BLOQUEADA” por WAF; eso describe un obstáculo de adquisición, no un precio. Si falla la descarga automática, usar el enlace oficial al PDF o descarga manual con registro de procedencia. No es necesario empezar comprando el informe impreso. Una traducción de un resumen ejecutivo no sustituye el informe completo en inglés.

**Tiempo:** ciertos registros de ediciones antiguas pueden reflejar la incorporación al portal. Comprobar fecha original de edición/lanzamiento y conservar evidencia; no usar sin más la fecha reciente de la ficha migrada.

### Sonda: adquisición y mediciones

**Medición específica.** Se conserva por separado la ficha descubierta, el PDF fijo de abril de 2026 y la ruta editorial alternativa. **El PDF fijo no hereda la fecha de la última ficha descubierta:** pueden ser ediciones distintas. La falta de fecha verificada queda visible; un bloqueo de acceso es un resultado de la sonda, no prueba de que el informe sea de pago.

```python
inventario_10 = sonda.inventario(10)
display(inventario_10[["fuente", "tipo", "adaptador"]])
muestra_10 = sonda.ejecutar(10)
```

```python
resumen_10 = mostrar_apartado(sonda, 10, '10. FMI Global Financial Stability Report — GFSR')
```

## 11. Papers de crisis

<a id="papers-crisis"></a>

**Quién produce y quién distribuye estos trabajos.** Los **papers de crisis** son investigaciones sobre las causas, el desarrollo y las consecuencias de episodios financieros. BIS, BCE y universidades publican estudios propios. El **NBER** es una organización estadounidense de investigación económica. **arXiv** y **SSRN** difunden manuscritos; **Crossref** mantiene identificadores DOI y metadatos bibliográficos, y **Unpaywall** ayuda a localizar copias de acceso abierto. Estas últimas herramientas facilitan encontrar un trabajo, pero no son sus autores ni avalan por sí solas sus conclusiones.

**Cómo pueden influir.** La investigación aporta evidencia para diseñar regulación, evaluar respuestas a crisis y construir modelos de riesgo. Para el TFM permite relacionar señales observadas con mecanismos como quiebras bancarias, exceso de deuda o contagio. Su influencia suele ser indirecta y gradual: un artículo puede cambiar cómo se entiende un problema, pero no equivale a un anuncio de política económica ni demuestra que un patrón vaya a repetirse.

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

### Sonda: adquisición y mediciones

**Medición específica.** Texto institucional, preprint, ficha SSRN/NBER, metadatos Crossref y copia localizada por Unpaywall tienen registros independientes. Crossref conserva online/print por separado; no acredita acceso al PDF. arXiv fija el documento a v1. Unpaywall localiza acceso abierto actual, pero no demuestra cuándo se abrió esa copia. La bibliografía completa de la captura sigue pendiente.

```python
inventario_11 = sonda.inventario(11)
display(inventario_11[["fuente", "tipo", "adaptador"]])
muestra_11 = sonda.ejecutar(11)
```

```python
resumen_11 = mostrar_apartado(sonda, 11, '11. Papers de crisis')
```

## 12. Huellas de crisis

<a id="huellas"></a>

**Qué significa «huella de crisis».** Es un conjunto de rasgos con los que describimos un episodio: por ejemplo, expansión previa del crédito, caída de activos, problemas bancarios y contracción económica. El **ESRB** aporta una base de episodios europeos; los investigadores del **FMI** documentan crisis bancarias internacionales. La **Jordà–Schularick–Taylor Macrohistory Database** es una base académica de series económicas y financieras de largo plazo que permite estudiar esos procesos.

**Para qué sirve.** Comparar estas huellas ayuda a investigar qué señales aparecen antes, durante o después de distintos tipos de crisis y a evaluar modelos de detección de regímenes. Son herramientas de análisis: la clasificación de un episodio puede conocerse años después. En el TFM hay que distinguir entre utilizar esa clasificación para evaluar un modelo y tratarla como información que un inversor ya conocía al comienzo de la crisis.

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

### Sonda: adquisición y mediciones

**Medición específica.** Las filas ESRB y Macrohistory son ejemplos de bases revisables; las metodologías FMI pertenecen a ediciones identificadas. La fecha de inicio de una crisis es una etiqueta retrospectiva, no una señal disponible ese día. Las huellas reproducibles exigirían el vintage, variables y reglas; esta sonda muestra los datos de ejemplo y no fabrica las 18 crisis ni las siete familias de la captura.

```python
inventario_12 = sonda.inventario(12)
display(inventario_12[["fuente", "tipo", "adaptador"]])
muestra_12 = sonda.ejecutar(12)
```

```python
resumen_12 = mostrar_apartado(sonda, 12, '12. Huellas de crisis')
```

## 13. Literatura de cartera y CIO

<a id="cartera"></a>

**Qué entidades y conceptos aparecen aquí.** **CIO** significa *Chief Investment Officer*: la persona responsable de orientar las inversiones de una organización. La literatura de cartera estudia cómo combinar activos y gestionar riesgos. La **Kenneth French Data Library** ofrece series académicas para contrastar modelos de rentabilidad. **EUR-Lex** es el portal de legislación de la UE; **MiFID II** es un marco normativo sobre servicios de inversión. La [ESMA](https://www.esma.europa.eu/about-esma) es la autoridad europea de mercados de valores y la [Comisión Nacional del Mercado de Valores (CNMV)](https://www.cnmv.es/portal/quees/funciones-cnmv?lang=es), el supervisor de los mercados de valores en España.

**Cómo influyen.** La investigación puede orientar la diversificación, la distribución entre activos y la medición del riesgo. La normativa y las directrices condicionan cómo se prestan y comercializan los servicios de inversión, qué información se ofrece y cómo se evalúa si una recomendación resulta adecuada para un cliente. Para el TFM, la literatura ayuda a fundamentar decisiones de cartera; los textos normativos aportan el contexto de las obligaciones y la protección del inversor.

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

### Sonda: adquisición y mediciones

**Medición específica.** Literatura, factores empíricos y normativa tienen filas distintas. MiFID II requiere distinguir publicación, entrada en vigor, aplicación y versión consolidada; el extractor solo registra la aplicación cuando encuentra una frase explícita. La traducción ESMA conserva su propia versión. Los factores de French actuales no son un vintage histórico; su mes de observación no se usa como publicación.

```python
inventario_13 = sonda.inventario(13)
display(inventario_13[["fuente", "tipo", "adaptador"]])
muestra_13 = sonda.ejecutar(13)
```

```python
resumen_13 = mostrar_apartado(sonda, 13, '13. Literatura de cartera y CIO')
```

## 14. La literatura llega tarde

<a id="literatura-tarde"></a>

**Qué se analiza aquí.** Este apartado estudia cuándo pudo conocerse una explicación sobre una crisis. Intervienen los autores, las revistas y los repositorios que publican distintas versiones de un trabajo. Un primer manuscrito puede aparecer antes que el artículo definitivo, y cada versión puede contener información diferente. Por eso importa la fecha pública de la versión concreta utilizada.

**Por qué afecta al análisis de decisiones.** Un estudio publicado después de una crisis puede explicar muy bien lo ocurrido y resultar útil para decisiones futuras. Sin embargo, incorporarlo a una simulación como si ya existiera antes daría al modelo conocimiento del futuro y podría exagerar sus resultados. En el TFM, esta auditoría permite separar lo que hoy sabemos de lo que podía saberse al decidir en cada fecha.

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

### Sonda: adquisición y mediciones

**Medición específica.** Se usa el corte ilustrativo **15 de septiembre de 2008** como referencia de comparación con la publicación. Los valores negativos indican publicaciones anteriores al corte; los positivos, posteriores. La tabla incluye evidencia y estado de cada versión; un paper anterior puede quedar excluido si no se acredita su texto. No se reproducen «18 de 77» ni «+1,6 años» sin los identificadores y el historial del corpus completo.

No se muestra aquí el gráfico de retraso evento → publicación: el corte de 2008 es una referencia ilustrativa y no acredita el evento que origina cada paper. La distancia al corte se conserva en la tabla; las figuras muestran publicaciones y volumen textual observado.

```python
inventario_14 = sonda.inventario(14)
display(inventario_14[["fuente", "tipo", "adaptador"]])
muestra_14 = sonda.ejecutar(14)
```

```python
resumen_14 = mostrar_apartado(sonda, 14, '14. La literatura llega tarde')
```

## Volumen, exportación y resumen para el equipo

Todo se calcula desde los apartados ejecutados. Se conservan los fallos y las fechas ausentes. El tamaño cuenta originales únicos por hash **dentro de cada apartado**; una republicación puede aparecer en varios apartados y sus MB no deben sumarse como corpus deduplicado global. No se extrapola al universo sin inventario de documentos y una muestra representativa.

Los CSV de auditoría incluyen fuente, familia, fechas, precisión, evidencias, URL, hash, bytes, errores y política temporal. Los JSONL locales conservan además texto, fragmentos y filas de ejemplo. No hay subida a Cloudflare ni despliegue en esta sonda.

```python
resumen_equipo = sonda.resumen()
display(resumen_equipo.round(2))
directorio_tablas = sonda.guardar_tablas()
print("Tablas de medición:", directorio_tablas)
print("Textos y fragmentos:", RAIZ / "data/local/sonda")
print("Originales:", RAIZ / "data/cache/sonda/objects")
```

## Limitaciones que quedan medidas

La sonda registra problemas de acceso, cambios de HTML, fechas incompletas y extracción parcial. El texto HTML puede incluir material accesorio y no se ha aplicado OCR a PDFs escaneados. La descarga del PDF completo puede ocupar más que sus páginas extraídas; las métricas separan bytes, páginas y texto completo.

La biblioteca de consulta general conserva estos textos y su contexto temporal. Para un backtest, deben acreditarse las versiones históricas de los portales, los vintages de tablas y la fecha de disponibilidad de traducciones y anotaciones. También hay que revisar el conocimiento futuro incorporado en cualquier modelo entrenado que produzca tono o resúmenes. Los gráficos muestran lo realmente medido y hacen visibles esas lagunas.
