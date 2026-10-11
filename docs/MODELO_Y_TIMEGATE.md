# Modelo de datos y disponibilidad histórica

Diseño conceptual; no se han creado tablas SQL ni contratos ejecutables.

## Registros del catálogo D1

| Entidad | Información principal |
|---|---|
| sources | Institución, familias, URLs, acceso, licencia y política temporal |
| documents | ID lógico, fuente, título, autores, idioma, tipo y tema |
| document_versions | Versión, URL, hash, edición, fechas y claves R2 |
| source_observations | Cuándo y dónde se observó cada versión, estado HTTP y evidencia |
| ingestion_runs | Fuente, lote, etapas, errores, conteos, bytes y checkpoint |
| derived_artifacts | Método/modelo, versión del pipeline, entradas y referencias R2 |

Los textos, archivos completos y listas extensas de fragmentos se guardan en R2.
El catálogo mantiene IDs y claves de los artefactos. Se evaluará el tamaño de
índices y metadatos antes de duplicar cada fragmento en D1.

## Documento normalizado en R2

Campos previstos: document_id, version_id, source_id, institución, familia,
título, autores, idioma, URL canónica, URL de descarga, formato, licencia,
SHA-256, tamaño, texto/segmentos, referencia al original, versión del extractor,
calidad de extracción y política RAG (key o archive).

Los fragmentos tendrán chunk_id, version_id, parent_id, sección, página,
posición en el texto, texto, tokens y versión de segmentación. Su disponibilidad
histórica se obtiene de la versión padre, no de la fecha de generación del chunk.

## Fechas distintas que debemos conservar

| Fecha | Significado |
|---|---|
| event_at | Fecha de reunión, discurso o hecho que se describe |
| publication_at | Publicación anunciada por la fuente para esa versión |
| public_available_at | Momento más temprano respaldado por evidencia de acceso al contenido de esa versión |
| retrieved_at | Momento de descarga por nuestro pipeline |
| processed_at | Momento de extracción/normalización |
| inferred_available_at | Estimación por una regla histórica, con evidencia y confianza |

Todas las fechas operativas se normalizan a UTC, conservando zona original y
precisión (hora exacta, solo día o fecha desconocida). Si solo se conoce el día,
las consultas intradía aplicarán una regla conservadora explícita; no se inventará
la hora de publicación. El campo policy_version permite reproducir esa decisión.

## Regla TimeGate

Una consulta as_of solo puede usar versiones cuyo contenido estuviera disponible
antes de la fecha/hora de corte, con evidencia suficiente y derechos de acceso
compatibles. Para el modo histórico estricto, una fecha desconocida o meramente
estimada excluye el documento. Un modo exploratorio podría admitir estimaciones,
pero debe etiquetarlas y no mezclarlas con resultados históricos estrictos.

Descargar hoy un paper de 2008 no demuestra que su texto actual fuera accesible
en 2008. Hay que conservar la primera versión pública y sus revisiones. Fecha
de publicación en revista, aparición del abstract y acceso al texto completo
pueden diferir. Las versiones nuevas no reemplazarán a las antiguas.

## Cómo interpretar los retrasos de la imagen

- Actas FOMC +21 días y accounts BCE +28 días son referencias de planificación:
  se verificará cada publicación y la política vigente en cada período histórico.
- BoJ +44 días no se codificará como una constante universal; se utilizará el
  calendario y la fecha efectiva de cada acta.
- La imagen indica +6 años para transcripciones FOMC. La Fed describe un retraso
  aproximado de cinco años; una entrega anual puede cruzar años naturales.
  Debe conservarse la fecha efectiva de publicación, no sumar automáticamente
  cinco o seis años a todas las reuniones.
- Un dataset sin fecha de publicación sirve como material auxiliar, pero requiere
  recuperar evidencia temporal antes de entrar en consultas históricas.
- El estudio de literatura tardía se reproducirá a partir de versiones y fechas
  verificadas, sin tratar los resultados de la imagen como ya validados.

Referencias: [material histórico Fed](https://www.federalreserve.gov/monetarypolicy/fomc_historical.htm),
[accounts BCE](https://www.ecb.europa.eu/press/accounts/html/index.en.html)
y [actas BoJ](https://www.boj.or.jp/en/mopo/mpmsche_minu/minu_all/index.htm).

## Repetibilidad y ausencia de duplicados

Un ID lógico identifica el documento; el hash identifica los bytes de una
versión. El mismo archivo visto en otra URL no obliga a duplicar bytes, pero
sí puede aportar otra observación de procedencia. La reejecución de un lote
debe reutilizar esos IDs y completar el checkpoint.

El pipeline no debe asignar disponibilidad histórica a partir de la fecha
del archivo local, del header HTTP Last-Modified sin validar, ni del orden
actual de un listado web.

## RAG y derivados

Aplicar filtros temporales al recuperar candidatos y comprobar otra vez la
versión padre antes de entregarlos. Registrar la fecha de corte y el conjunto
de versiones usado en cada análisis. Cualquier extracción bajo demanda del
archivo debe pasar por la misma comprobación.

Un score de tono o un resumen conserva entradas, fecha de corte, modelo, prompt,
versión y fecha de cálculo. Para evaluación histórica debe revisarse también
el conocimiento incorporado en el modelo: filtrar documentos no elimina por
sí solo el conocimiento de hechos futuros aprendido durante su entrenamiento.
