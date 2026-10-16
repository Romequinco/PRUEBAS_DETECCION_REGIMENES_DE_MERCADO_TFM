# Módulos del corpus y sonda

`sonda/` implementa las consultas pequeñas del notebook: HTTP con caché de
originales, adaptadores de documentos/tablas/APIs, extracción, política temporal,
mediciones y gráficos. `ConfigSonda` limita volumen y páginas; `Sonda.ejecutar`
mantiene independiente cada apartado. El inventario declarativo vive en
`config/sources/sonda_fuentes.json`.

La adquisición no convierte fechas de evento o periodos en publicación.
Las versiones de portales actuales conservan sus fechas conocidas, pero no
entran en el TimeGate estricto hasta acreditar el texto histórico. arXiv fija
la descarga a v1. Las pruebas de `tests/unit/` cubren los límites temporales.

Las demás carpetas conservan la estructura prevista para el pipeline completo;
no representan integraciones R2/D1 ni un recolector masivo implementado.

| Carpeta | Trabajo que contendrá |
|---|---|
| collectors | Adaptadores por institución: listar e identificar documentos |
| fetching | Cliente HTTP, descargas, hashes, reintentos y límites de peticiones |
| parsing | Extracción HTML/PDF/tablas y OCR selectivo |
| normalization | Contrato común, idioma, fechas y calidad |
| storage | Repositorios R2 y D1; claves de objetos y manifiestos |
| temporal | Fechas y versiones; TimeGate opcional para evaluación histórica |
| chunking | Fragmentos de párrafo, Q&A y sección, con vínculo al padre |
| retrieval | Búsqueda textual/semántica, filtros temáticos, contexto, citas y documento completo |
| analysis | Tono, diferencias entre comunicados y derivados de crisis |
| orchestration | Jobs, checkpoints, estados e idempotencia |

Los adaptadores de fuentes no accederán directamente a credenciales ni a SQL:
utilizarán los contratos y la capa de almacenamiento. Los módulos de consulta
aplicarán filtros de acceso y calidad a documentos y fragmentos. La consulta
general será el modo predeterminado; las reglas temporales estrictas se aplicarán
cuando se solicite el modo histórico.
