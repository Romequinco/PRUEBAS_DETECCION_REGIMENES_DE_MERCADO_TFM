# Ejecución de la sonda y jobs futuros

`ejecutar_notebook_sonda.py` ejecuta el cuaderno completo con el kernel local y
guarda sus salidas. `exportar_notebook.py` sincroniza Markdown y HTML desde ese
notebook, sin repetir consultas de red. Usar el Python del entorno `.venv`.

La recolección masiva y los calendarios de refresco descritos abajo siguen
pendientes; estos dos scripts solo ejecutan y exportan la sonda.

Se distinguirán descubrimiento, refresco incremental, backfill histórico,
importación manual, extracción, fragmentación y reconciliación de catálogo.
Cada job tendrá fuente, rango, límites, checkpoint y versión de configuración.

Priorizar cargas amplias reanudables y descargas oficiales masivas. Incorporar
indexación textual, embeddings por lotes y medición de cobertura por familia.
Frecuencia propuesta: semanal para comunicaciones y mensual para literatura;
ajustar según consumo y utilidad. No hay requisito de ejecución inmediata.

La configuración del job vive aquí; la lógica se implementará en
`src/macro_riesgo/orchestration/`. Los calendarios se activarán en el ejecutor
solo después de verificar acceso y consumo. No hay jobs masivos ni programados.
