# Orden de implementación

## Fase 0: diseño y cuenta

Completado localmente: estructura de carpetas, documentación, inventario de la
imagen y propuesta de almacenamiento. Pendientes: cuenta Cloudflare, activación
R2 y decisión de presupuesto. No se ha conectado Cloudflare ni creado un recurso
remoto. El repositorio GitHub existente es
`feraltauste1/Bancos-centrales-riesgo-sistem-tico-y-literatura`.

Objetivo actualizado: biblioteca amplia para agentes, con cargas por lotes y
actualizaciones sin urgencia. Priorizar cobertura, texto fiable y recuperación
con citas. Propuesta de piloto: una muestra de discursos BIS, papers BIS/BCE,
informes de estabilidad/GFSR, comunicaciones Fed/BCE y literatura de cartera
abierta. Incorporar documentos antiguos y recientes; no exigir una reconstrucción
histórica completa para validar su utilidad general.

## Fase 1: almacenamiento y catálogo del piloto

Crear entorno dev, credenciales acotadas, contrato de documento y catálogo D1.
Implementar una prueba de subir/descargar/verificar hash y registrar metadatos.
Comprobar repetición segura, error parcial entre R2 y D1 y recuperación.

## Fase 2: recolección amplia y verificable

Implementar primero importación de paquetes oficiales y PDFs/HTML de varias
familias del piloto. Conservar originales, extraer texto, deduplicar y emitir
manifiestos. Registrar fechas conocidas y marcar las desconocidas sin bloquear
la consulta general. Revisar muestras de extracción, páginas, tablas y citas.
Implementar carga manual e inventario de referencias pendientes de texto.

## Fase 3: consulta útil para agentes

Catálogo por tema/fuente/tipo/idioma/país y búsqueda textual del corpus validado.
Decidir el motor de búsqueda en el piloto. Permitir recuperar pasajes, ampliar
su contexto y abrir el texto completo, con fuente, versión y página/sección.
Evaluar preguntas repartidas entre riesgos, comunicaciones y carteras; comprobar
relevancia, citas y respuestas cuando no hay suficiente evidencia.

## Fase 4: automatización y cobertura

Utilizar el remoto GitHub existente y revisar su visibilidad; habilitar tests y jobs
manuales/programados. Ampliar todas las familias de la captura, incluyendo
BoE/BoJ, otros bancos, organismos, estabilidad y literatura. Hacer la carga
retrospectiva por fuente/año,
con checkpoints y seguimiento de volumen, coste y calidad. El proceso se
ejecutará en la nube; GitHub y R2 cumplirán funciones distintas.

Propuesta inicial: refrescos semanales para comunicaciones y mensuales para
literatura y catálogos; ajustar por fuente según el piloto. Los cambios del
pipeline pasan los checks de CI cuando se configure. El refresco de información
se ejecuta por calendario o acción manual; no depende de hacer commits con datos.
Los workflows no se han creado ni activado todavía.

## Fase 5: RAG progresivo y conexión de agentes

Fragmentar texto con páginas y documento padre; incorporar embeddings de las
familias prioritarias y ampliar al medir utilidad y coste. Combinar búsqueda
textual y semántica; los textos sin embeddings siguen localizables. Añadir una
API de consulta autenticada cuando se conecten agentes, con límites de contexto,
contrato de citas y acceso al original. Comparar resultados contra las preguntas
de referencia antes de ampliar la indexación.

## Fase 6: derivados y evaluación histórica opcional

Recibir inventarios concretos de papers/datasets/CIO para contrastar cobertura
con la referencia. Agregar huellas de crisis, tono, diferencias y otros derivados
si aportan utilidad a los agentes. Activar TimeGate estricto para backtests y
estudio de literatura tardía; comprobar evidencia temporal de las versiones
incluidas. Esta fase no condiciona la puesta en marcha de la biblioteca general.

## Criterios para pasar a producción

Originales recuperables y verificados por hash; catálogo coherente; procedencia
y condiciones de acceso registradas; calidad de extracción revisada; cobertura
medida por familia; reejecuciones sin duplicados; recuperación de fallos; búsqueda
relevante con citas y originales accesibles; costes medidos y restauración probada.
La aptitud histórica se certifica por separado cuando se habilite ese modo.
El alcance y los umbrales se fijarán durante el piloto.
