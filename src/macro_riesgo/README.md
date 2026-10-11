# Módulos reservados

Este directorio será el paquete Python del pipeline. Solo contiene estructura;
los `.gitkeep` permiten conservar las carpetas en Git.

| Carpeta | Trabajo que contendrá |
|---|---|
| collectors | Adaptadores por institución: listar e identificar documentos |
| fetching | Cliente HTTP, descargas, hashes, reintentos y límites de peticiones |
| parsing | Extracción HTML/PDF/tablas y OCR selectivo |
| normalization | Contrato común, idioma, fechas y calidad |
| storage | Repositorios R2 y D1; claves de objetos y manifiestos |
| temporal | Reglas TimeGate, precisión de fechas y evidencia |
| chunking | Fragmentos de párrafo, Q&A y sección, con vínculo al padre |
| retrieval | Consultas del archivo y del índice RAG |
| analysis | Tono, diferencias entre comunicados y derivados de crisis |
| orchestration | Jobs, checkpoints, estados e idempotencia |

Los adaptadores de fuentes no accederán directamente a credenciales ni a SQL:
utilizarán los contratos y la capa de almacenamiento. Los módulos de consulta
aplicarán las mismas reglas temporales a documentos y fragmentos.
