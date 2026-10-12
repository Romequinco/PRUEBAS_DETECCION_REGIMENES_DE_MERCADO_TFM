# Verificaciones futuras

- `unit/`: extracción, normalización, hashing, claves, fechas, TimeGate y reglas
  de fragmentación con muestras pequeñas.
- `integration/`: subida/descarga R2, catálogo D1, reejecución y recuperación de
  escrituras parciales, solo contra desarrollo.
- `fixtures/`: muestras pequeñas sintéticas o con licencia que permita guardarlas
  en Git; sin credenciales ni corpus de producción.

La verificación principal del corpus usará preguntas de referencia de varias
familias: recuperar pasajes pertinentes, abrir su contexto y comprobar que la
cita respalda la afirmación. Incluir fuentes discrepantes y preguntas sin evidencia.
Comprobar que un documento validado sin fecha pública exacta puede consultarse
en modo general, mientras un fallo grave de extracción no se presenta como texto listo.

La prueba del modo histórico opcional usará dos versiones del mismo documento y una fecha
de corte intermedia: la versión publicada después debe quedar excluida tanto
de búsqueda como del análisis derivado. No hay tests implementados todavía.
