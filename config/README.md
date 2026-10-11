# Configuración futura

- `sources/`: un registro por fuente con listado oficial, método permitido,
  tipos de documentos, formatos y frecuencia.
- `policies/`: selección RAG, reglas históricas, límites de descarga y retención.
- `environments/`: nombres de buckets, IDs de catálogo y parámetros no secretos
  de desarrollo y producción.

Las reglas y configuraciones reales se escribirán al implementar. El inventario
inicial está en `docs/FUENTES.md`. Las credenciales pertenecerán a `.env` local
ignorado o al gestor de secretos del ejecutor, no a estas carpetas.
