# Configuración futura

- `sources/`: un registro por fuente con listado oficial, método permitido,
  tipos de documentos, formatos, temas, prioridad y frecuencia de lotes.
- `policies/`: calidad, deduplicación, acceso, indexación progresiva, retención
  y modo histórico opcional. Propuesta de refresco semanal/mensual, ajustable.
- `environments/`: nombres de buckets, IDs de catálogo y parámetros no secretos
  de desarrollo y producción.

Las reglas y configuraciones reales se escribirán al implementar. El inventario
inicial está en `docs/FUENTES.md`. Las credenciales pertenecerán a `.env` local
ignorado o al gestor de secretos del ejecutor, no a estas carpetas.
