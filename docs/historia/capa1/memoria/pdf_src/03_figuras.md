# 03 — Curaduría de figuras para el PDF ejecutivo

**Agente 3 (Curador de figuras).** Decide qué figuras entran al PDF científico, cuáles se
reúsan tal cual y cuáles se regeneran corregidas. Todas las figuras nuevas se generan
**verbatim** desde `results/metrics_master_final.csv` con un único script reproducible
(`scripts/figs_pdf/build_pdf_figs.py`). No se re-ejecutan detectores ni walk-forward; no se
modifica ningún CSV ni ninguna figura `fase4_*`/`d0X_*` existente.

## Principios de equidad aplicados (revisión académica)
- **La cobertura SIEMPRE se separa por grupo de ventana** (`vio_2008_oos`). Seis detectores de
  ventana larga vieron la GFC-2008 fuera de muestra; seis de ventana corta NO. Compararlos en
  una única "cobertura sistémica" agregada es injusto.
- **Eje común de sensibilidad = cobertura de COVID-2020**, la única ventana de crisis OOS
  compartida por los 12 detectores. La cobertura de GFC-2008 se ranquea **solo** dentro del
  bloque de ventana larga.
- **Lead/lag censurado**: `src/evaluation.lead_lag` usa `lookback=252` días. Un valor en el
  borde (|d| ≥ 249) significa que la señal ya estaba activa al inicio de la ventana → **no es
  anticipación genuina** (refleja sesgo *always-on* / alta tasa de falsa alarma). Se marca con
  `*` y borde discontinuo.
- **Paleta sobria y consistente** en todas las figuras nuevas: azul `#2b5d8a` (ventana larga),
  ámbar `#c98a2b` (ventana corta), gris `#7a7a7a` (exploratorio-negativo). Se elimina la escala
  rojo-verde (`RdYlGn`) de las figuras `fase4_*` (problema de daltonismo). DPI = 200.

---

## Tabla de decisiones

| # | Figura (ruta final) | Decisión | Sección del PDF | Pie (\caption) propuesto |
|---|---------------------|----------|-----------------|--------------------------|
| 1 | `results/eda_sp500_drawdown.png` | **REUSAR** | 2. Datos y eventos de evaluación | S&P 500 (escala log) y su drawdown desde 1985, con las ventanas de **crisis** (rojo) y de **trampa / falso positivo** (naranja) que definen el protocolo de evaluación. Estas ventanas son la verdad-terreno frente a la que se mide cobertura, especificidad y lead/lag. |
| 2 | `results/eda_fat_tails.png` | **REUSAR** | 2. Datos y hechos estilizados | Distribución de retornos diarios frente a la Normal (eje log) para los seis activos. El exceso de curtosis (25.6 en el S&P 500, 39.6 en HYG) motiva el uso de colas extremas y de modelos t-Student / GARCH en la fase de detección. |
| 3 | `results/eda_corr.png` | **REUSAR** | 2. Datos | Matriz de correlación de retornos diarios (muestra completa). La estructura renta-fija/renta-variable (TLT–IEF 0.91; S&P–HYG 0.67) justifica el conjunto multiactivo usado por los detectores de turbulencia y clustering. |
| 4 | `results/pdf_rank_heatmap.png` | **REGENERAR** (corrige el defecto) | 5. Síntesis comparativa | Ranking por eje (1 = mejor; más oscuro = mejor rango). **Corrige el defecto de la versión `fase4`**: la cobertura se separa por ventana — la GFC-2008 solo ranquea entre los seis detectores de ventana larga (`n/a` para los cortos) y COVID-2020 (ventana OOS común) ranquea a los doce. Filas agrupadas en bloque de ventana larga y bloque de ventana corta. † El lead/lag medio está censurado en ±252 d (ver Fig. 8). |
| 5 | `results/pdf_sensibilidad_especificidad.png` | **REGENERAR** | 4. Resultados: trade-off sensibilidad–especificidad | Plano sensibilidad ↔ especificidad. La sensibilidad se mide en la **ventana OOS común (COVID-2020)** para una comparación justa entre las dos longitudes de ventana. Eje X: especificidad = 1 − media de falsa alarma en las trampas de 2013 y 2018. Color y forma codifican el grupo de ventana; D11/D12 son exploratorio-negativos. |
| 6 | `results/pdf_persistencia_sensibilidad.png` | **REGENERAR** | 4. Resultados: trade-off persistencia–sensibilidad | Plano persistencia (duración media de régimen, eje log) ↔ sensibilidad (cobertura COVID-2020 común). D3/D12 flickean (duración baja); D7 (CUSUM) y las reglas son muy persistentes. Mismo código de color/forma por grupo de ventana. |
| 7 | `results/pdf_bic.png` | **REGENERAR** | 4. Resultados: ajuste estadístico | BIC de los modelos generativos (menor = mejor ajuste). **Aviso:** el BIC solo es estrictamente comparable sobre las MISMAS features/ventana (p. ej. D4 vs D8); las barras de distinto conjunto de features no son directamente comparables. Color por grupo de ventana; gris = negativo. |
| 8 | `results/pdf_leadlag.png` | **REGENERAR** (censura marcada) | 4. Resultados: anticipación (lead/lag) | Lead/lag por evento (días de trading; negativo = la señal sostenida anticipa el suelo del drawdown). `*` y borde discontinuo = valor **censurado** en ±252 d: la señal ya estaba activa al inicio de la ventana de búsqueda, por lo que no constituye anticipación genuina sino sesgo *always-on*. Filas separadas por grupo de ventana. |
| 9 | `results/pdf_estres_vs_estricta.png` | **REGENERAR** | 4. Resultados: sensibilidad al criterio de crisis | Detectores multi-estado: cobertura con definición de crisis **estricta** (cola extrema) vs **estrés agregado** (corrección + crisis), en las ventanas OOS comunes COVID-2020 e Inflación-2022. Muestra cuánto sube la cobertura al relajar el criterio (lectura honesta de los dos lados). |
| — | `results/fase4_rank_heatmap.png` | **DESCARTAR** | — (sustituida por #4) | Defecto: la columna "Cob. sistémica" ranquea los 12 detectores juntos sin separar por ventana, mezclando ventana larga (vio 2008 OOS) con ventana corta (no la vio). Además usa paleta rojo-verde. Se sustituye por `pdf_rank_heatmap.png`. |
| — | `results/fase4_*` (5 restantes) | **DESCARTAR** | — (sustituidas por #5–#9) | Versiones DPI 120 con paleta rojo-verde y ejes de cobertura no separados por ventana. Sustituidas por sus equivalentes `pdf_*`. |
| — | `results/d0X_*`, `d1_*`, `d2_*`, `d5_*`, `d8_*`, `metrics_04_*` (≈45 figuras) | **DESCARTAR del PDF** (→ apéndice) | Apéndice / notebooks | Diagnósticos por detector (timelines, transición, coverage individual). Nivel de detalle no ejecutivo: pertenecen al apéndice o a los notebooks `0X_*.ipynb`, no al cuerpo del PDF. |

### Cómo se resolvió el `rank_heatmap`
Se **REGENERA corregido** (no se descarta a favor de una tabla). La columna ambigua
"Cob. sistémica" se parte en dos columnas honestas:
1. **Cob. GFC 08** — ranqueada **solo** entre los seis detectores de ventana larga; los seis de
   ventana corta aparecen como `n/a` (gris), no se penalizan ni se comparan en una ventana que
   no vieron.
2. **Cob. COVID 20** — ventana OOS común a los doce, ranqueada entre todos.

Además, las filas se agrupan en dos bloques con separador (ventana larga arriba, corta abajo),
las etiquetas de detector se colorean por grupo de ventana, se sustituye la paleta rojo-verde
por una secuencial azul sobria (más oscuro = mejor rango) y se añade una nota † avisando de que
el lead/lag medio está censurado.

---

## Ficheros `results/pdf_*.png` generados (6)
```
results/pdf_rank_heatmap.png
results/pdf_sensibilidad_especificidad.png
results/pdf_persistencia_sensibilidad.png
results/pdf_bic.png
results/pdf_leadlag.png
results/pdf_estres_vs_estricta.png
```

## Reproducir
```bash
python scripts/figs_pdf/build_pdf_figs.py
```
Regenera las seis figuras `results/pdf_*.png` desde `results/metrics_master_final.csv`.
Determinista, sin dependencias de red ni de re-cómputo de detectores.

## Coherencia con el texto del PDF (checklist)
- [x] Cobertura presentada **separada por grupo de ventana** en todas las figuras
      (heatmap con bloques + columnas GFC/COVID; scatters con sensibilidad = COVID común).
- [x] Lead/lag **etiquetado como censurado** donde |d| ≥ 249 (Fig. 8).
- [x] BIC con aviso de comparabilidad solo sobre las mismas features.
- [x] D11/D12 marcados como exploratorio-negativos en todas las figuras donde aparecen.
- [x] Ninguna figura nueva pisa un fichero `fase4_*` o `d0X_*` existente (prefijo `pdf_`).
```
