# Modelo de datos, consulta general y TimeGate opcional

Diseño conceptual; no se han creado tablas SQL ni contratos ejecutables.

La sonda del notebook implementa una primera política local
`sonda-timegate-v1` en `src/macro_riesgo/sonda/temporal.py`: texto completo,
versión acreditada y disponibilidad con zona antes del corte. Una fecha con
solo día y zona conocida se admite desde el inicio del día local siguiente;
es un límite conservador, no una hora observada. La biblioteca general conserva
también muestras con fecha incierta. Este código no implementa todavía el
catálogo D1 ni el filtrado de un servicio RAG.

La consulta general del corpus es el modo predeterminado: admite literatura
relevante aunque no se conozca la fecha exacta de su primera disponibilidad.
Se conservan fechas, versiones y dudas para aportar contexto. El modo histórico
estricto se solicita explícitamente y tiene requisitos adicionales de evidencia.

## Registros del catálogo D1

| Entidad | Información principal |
|---|---|
| sources | Institución, familias, URLs, acceso, términos y frecuencia de refresco |
| documents | ID lógico, fuente, título, autores, idioma, tipo, tema, país y estado editorial cuando se conozca |
| document_versions | Versión, URL, hash, edición, fechas, calidad de extracción, estado histórico y claves R2 |
| indexing_runs | Motor/versión, lotes, estado de indexación textual o semántica, errores y cobertura |
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
calidad de extracción, temas, país, estado editorial, relación con traducciones,
prioridad de indexación y estado de búsqueda textual/semántica. La prioridad
determina el orden de indexación, no la exclusión permanente del corpus.

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

## Requisitos de consulta general

Una versión ready necesita procedencia, original recuperable, términos de acceso
registrados y extracción suficientemente completa. La ausencia de una fecha
pública exacta se comunica como incertidumbre; no bloquea esta modalidad. Una
referencia sin texto permanece en el catálogo como pendiente y no se devuelve
como pasaje de evidencia. Los fallos de OCR y las tablas omitidas se registran.

Por defecto se consulta la versión identificada como vigente o más reciente de
cada documento; las relaciones de sustitución se registran cuando la fuente las
confirma. Las versiones anteriores siguen disponibles para contexto o comparación.
Si el estado es desconocido, se indica; no se presenta automáticamente una guía
antigua como vigente ni se eliminan papers por edad.

El resultado incluye texto, document_id/version_id, título, institución/autores,
tipo, fecha conocida y su precisión, página/sección, URL y referencia al original.
El agente puede ampliar contexto o abrir el documento completo. Aplicar filtros
temáticos y de acceso en búsqueda y recuperación; el corte temporal es opcional.

## Regla TimeGate, solo para consultas históricas

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

En consultas históricas, aplicar filtros temporales al recuperar candidatos y
comprobar otra vez la versión padre antes de entregarlos, también al abrir el
archivo completo. En consultas generales, conservar contexto temporal sin
imponer un corte. Registrar modo de consulta, filtros y versiones usados; el
corte se registra cuando exista.

Un score de tono o un resumen conserva entradas, modo de consulta y fecha de
corte si corresponde, modelo, prompt,
versión y fecha de cálculo. Para evaluación histórica debe revisarse también
el conocimiento incorporado en el modelo: filtrar documentos no elimina por
sí solo el conocimiento de hechos futuros aprendido durante su entrenamiento.
