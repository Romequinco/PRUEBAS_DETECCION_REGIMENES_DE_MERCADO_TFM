# Verificaciones futuras

- `unit/`: extracción, normalización, hashing, claves, fechas, TimeGate y reglas
  de fragmentación con muestras pequeñas.
- `integration/`: subida/descarga R2, catálogo D1, reejecución y recuperación de
  escrituras parciales, solo contra desarrollo.
- `fixtures/`: muestras pequeñas sintéticas o con licencia que permita guardarlas
  en Git; sin credenciales ni corpus de producción.

La prueba histórica clave usará dos versiones del mismo documento y una fecha
de corte intermedia: la versión publicada después debe quedar excluida tanto
de búsqueda como del análisis derivado. No hay tests implementados todavía.
