# Jobs futuros

Se distinguirán descubrimiento, refresco incremental, backfill histórico,
importación manual, extracción, fragmentación y reconciliación de catálogo.
Cada job tendrá fuente, rango, límites, checkpoint y versión de configuración.

La configuración del job vive aquí; la lógica se implementará en
`src/macro_riesgo/orchestration/`. Los calendarios se activarán en el ejecutor
solo después de verificar acceso y consumo. No hay jobs ejecutables todavía.
