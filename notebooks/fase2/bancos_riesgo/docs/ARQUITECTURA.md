# Arquitectura del corpus y del pipeline

## Objetivo

Recopilar las familias de información de la imagen: comunicaciones de bancos
centrales, informes de estabilidad, organismos multilaterales, papers de crisis,
literatura de cartera y documentos regulatorios. Conservar los originales y
construir una biblioteca amplia para agentes: búsqueda de evidencias, lectura
de documentos completos y RAG progresivo. Cobertura y utilidad de consulta son
prioritarias; no se exige ingesta inmediata. Las consultas históricas son un
modo opcional que respeta la información disponible en la fecha solicitada.

El objetivo y los criterios de consulta están en
[Corpus de conocimiento para agentes](CORPUS_PARA_AGENTES.md).

## Flujo propuesto

```mermaid
flowchart TD
  S[Fuentes oficiales, repositorios abiertos y carga manual] --> C[Recolectores Python]
  C --> F[Descarga, hash y registro de procedencia]
  F --> R[R2: HTML, PDF, ZIP y CSV originales]
  F --> D[D1: catálogo, versiones y estado de jobs]
  R --> P[Extracción y normalización]
  P --> T[Calidad, acceso, deduplicación y metadatos]
  T --> J[R2: texto y documentos JSONL]
  T --> D
  J --> A[Archivo completo bajo demanda]
  J --> CH[Fragmentos con referencia al documento padre]
  CH --> L[Búsqueda textual del corpus validado]
  CH --> V[Índice semántico por lotes prioritarios]
  D --> G[Consulta general o histórica opcional]
  L --> G
  V --> G
  A --> G
  G --> Q[Resultados con fuente, versión y citas]
```

## Qué hace cada componente

| Componente | Responsabilidad |
|---|---|
| Python | Recolección, PDF/OCR, extracción de tablas, normalización y procesamiento por lotes |
| R2 | Almacenar bytes originales y artefactos grandes con rutas reproducibles |
| D1 | Consultar metadatos por fuente, familia, tema, fecha, versión y estado |
| GitHub Actions | Ejecutar inicialmente lotes acotados y refrescos programados |
| Worker opcional | API privada de consulta o coordinación ligera |
| Búsqueda textual | Recuperar términos, autores e identificadores en textos validados; motor por decidir en el piloto |
| Índice vectorial opcional | Recuperar pasajes por significado; ampliar cobertura según calidad y coste |

Los originales y los textos no van en Git. R2 conserva el corpus; D1 mantiene
las referencias para encontrarlo. Los embeddings no se almacenarán en una tabla
D1 como sustituto de un motor de búsqueda vectorial. Antes de adoptar Vectorize
u otro índice, se medirá volumen, coste, relevancia de resultados y soporte de
filtros. La elección del motor de búsqueda queda pendiente del piloto; R2
conserva archivos y D1 sus metadatos, pero eso no resuelve por sí solo la búsqueda.

## Etapas y recuperación

1. **Descubrir:** guardar URL, fuente y señal de publicación; checkpoint por fuente.
2. **Descargar:** respetar límites y condiciones de acceso, registrar resultado
   HTTP, tipo y tamaño. Validar que un supuesto PDF no sea una página de bloqueo.
3. **Conservar:** calcular SHA-256, guardar el original y registrar la versión.
4. **Extraer:** conservar párrafos, páginas, títulos, autores y estructura Q&A;
   OCR solo cuando sea necesario y con calidad registrada.
5. **Normalizar:** producir documentos JSONL con un contrato versionado.
6. **Validar:** idioma, acceso, términos, completitud, duplicados y trazabilidad;
   registrar fechas conocidas e incertidumbres, sin exigir precisión histórica
   extrema para la consulta general.
7. **Catalogar:** guardar metadatos y referencias de objetos; marcar la versión ready.
8. **Fragmentar e indexar:** texto validado localizable; embeddings por lotes
   prioritarios y reglas de segmentación según familia.
9. **Consultar o analizar:** devolver pasajes y contexto con citas al original
   y versión. Activar TimeGate solo para solicitudes históricas explícitas.

Estados propuestos: discovered, fetched, extracted, validated, ready, failed,
manual_required y quarantined. Cada etapa debe poder repetirse sin duplicar datos.
R2 y D1 no comparten una transacción global: un lote debe tolerar objetos escritos
sin catálogo confirmado y reconciliarlos al reanudarse. Los resultados solo serán
visibles para consulta cuando su versión esté validada y marcada ready. El
estado de indexación textual/semántica y la aptitud para evaluación histórica
se registran por separado: no tener embeddings o una fecha pública exacta no
impide que un texto validado sea útil en consulta general.

## Organización de objetos en cada bucket

```text
raw/<fuente>/<año>/<sha256>.<extensión>
normalized/<fuente>/<año>/<lote>.jsonl
chunks/<fuente>/<año>/<lote>.jsonl
manifests/<fecha>/<run-id>.jsonl
derived/<tipo>/<versión-pipeline>/<run-id>.jsonl
```

Esta es una convención de claves, no objetos ya subidos. Los textos derivados
deben enlazar al hash y versión del original. Un manifiesto recoge objetos,
hashes, conteos y estados, sin secretos. Un mismo contenido puede reutilizarse
entre documentos, conservando por separado cada procedencia y fecha.

## Ejecución sin depender del ordenador

Al implementar, GitHub Actions ofrecerá ejecución manual del piloto y una
programación incremental sin requisitos de ejecución inmediata. Propuesta:
refrescos semanales para comunicaciones y mensuales para literatura, ajustables
tras medir el piloto. La carga retrospectiva se dividirá por fuente y año,
con checkpoints persistidos en D1/R2; no dependerá del disco temporal del runner.
Las tareas costosas de OCR se separarán de los refrescos ligeros.

Los cron de GitHub pueden retrasarse y no garantizan ejecutar al minuto exacto.
Para cargas grandes se evaluará un ejecutor dedicado o una cola según volumen
y coste; el objetivo actual no requiere ejecución al minuto.
La elección final debe revisar minutos disponibles y coste de la cuenta GitHub.

Referencia: [ejecuciones programadas de GitHub Actions](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule).

## RAG y análisis de la imagen

- Bancos centrales: empezar por párrafos de 60–350 palabras y Q&A por pregunta,
  ajustando la longitud con muestras reales.
- Informes: fragmentos por sección, hasta aproximadamente 800 tokens, más vínculo
  al padre. Son parámetros iniciales, no límites universales.
- Registrar tokenizer, versión de reglas, página y posición de cada fragmento.
- Hacer localizable el texto validado mediante catálogo y búsqueda textual.
  Incorporar embeddings por lotes prioritarios y ampliar el RAG según utilidad
  y coste; conservar acceso al documento completo y al contexto de cada pasaje.
- Entregar título, fuente, versión, fecha conocida, página/sección y URL en cada
  resultado. Evaluar relevancia y respaldo de las citas con preguntas de referencia.
- Tono hawkish/dovish y diferencias entre comunicados son resultados derivados:
  conservar método, modelo, prompt, versión y documentos de entrada.
- Cada resultado debe distinguir texto fuente de interpretación generada.

## Límites de acceso

Las etiquetas de bloqueo de la imagen sirven para reservar carga manual y
revisar el acceso actual. Ante autenticación, bloqueo o paywall, se registrará
la causa y se buscará una API, feed o descarga autorizada. No se diseñan saltos
de controles de acceso. La licencia y las condiciones de redistribución se
registrarán por fuente/documento; una copia pública no implica permiso universal
para redistribuirla.
