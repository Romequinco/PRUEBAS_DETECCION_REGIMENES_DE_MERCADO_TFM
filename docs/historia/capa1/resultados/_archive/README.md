# results/_archive — artefactos archivados (saneamiento Ola 0, 2026-06-21)

Carpeta de archivo reversible. Nada aquí se borra: se mueve para retirarlo del
flujo vigente conservando el historial.

## `metrics_master_final.csv`
**Obsoleto.** Era el segundo master, con esquema complementario al antiguo
`metrics_master.csv` (aportaba `clase`, `coste`, `vio_2008_oos`, `cov_estres_*`,
`fa_estres_*`, `false_alarm_rate_estres`, `nota`; le faltaban `silhouette` y los
IC bootstrap de cobertura `cov_*_lo/_hi`).

A partir de Ola 0 existe **UN ÚNICO master canónico**:
`results/metrics_master.csv`, que es el **SUPERSET** de ambos (43 columnas) y lo
reconstruye `scripts/verify/_rebuild_master.py`. Todos los scripts de figuras
(`build_pdf_figs.py`, `build_synthesis_figs.py`) leen ese canónico.

Este `_final.csv` se conserva solo como referencia/origen de los valores de
ESTRÉS de los 3 multi-estado (incrustados en `_rebuild_master.py` como
`MULTI_ESTRES`). No debe volver a usarse como entrada de ningún script.

## `fase4_*.png`
Generación intermedia de figuras de síntesis (FASE 4), producidas por
`scripts/builders/_build_13.py`. Quedaron **superseded** por las figuras
curadas `results/pdf_*.png` (las que incluye `report/informe_capa1.tex`).
No las referencia ni el informe ni los scripts de figuras vigentes. Si se
re-ejecuta `_build_13.py` se regeneran en `results/` (están gitignored).
