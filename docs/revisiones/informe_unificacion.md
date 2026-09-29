# Informe de unificación (ADR-004): comparación final con la foto de referencia

Fecha: 2026-09-29 · rama `unificacion` · comparador final.

**Foto previa:** `docs/revisiones/baseline/`, generada en el commit `0bf0251` antes de la
unificación. **Foto nueva:** la genera el mismo script,
`python docs/revisiones/baseline/make_baseline.py --out <scratchpad>/baseline_final --results results --processed data/processed --notebooks notebooks --src src/regimenes`.
Los contenidos previos de `results/` se leen con `git show 0bf0251:results/...`, sin
checkout. No se relanzó el benchmark ni se modificó nada en `results/benchmark`.

Traducción de rutas aplicada: `benchmark_v2/` → `benchmark/`,
`fusion_d07_d08/` → `fusion/d07_d08/` y `fusion_d02_d06/` → `fusion/d02_d06/`.

## Resumen por comprobación

| # | comprobación | resultado | detalle |
|---|---|---|---|
| a | `results_hashes`: 79 CSV/JSON | **OK** | Los 79 archivos existen en ambos lados tras traducir las rutas. **28 tienen hash idéntico**: los 27 de `fusion/` y `benchmark/metrics_master_v2.csv`. Los otros **51** difieren, pero solo en columnas o claves de tiempo, huella o ruta (ver «Diferencias aceptadas»). Toda columna numérica de resultados coincide exactamente (`Series.equals`), con la misma forma y las mismas columnas en el mismo orden. |
| b | `panels_hashes` | **OK** | Los 29 paneles previos (24 `results/benchmark/panels/*.parquet` y 5 de `data/processed`) tienen un hash de contenido **idéntico**. Hay 4 paneles nuevos sin equivalente previo, que son salidas nuevas de las ablaciones de los notebooks de familia: `results/detectores/f6_changepoint/variante_D07_gauss_{A,B}.parquet` y `results/detectores/f7_deep/variante_D12_pca_{A,B}.parquet`. |
| c | notebooks: títulos | **OK con diferencias explicadas** | El inventario previo contiene 97 títulos (8 notebooks v2). **94** aparecen tal cual en algún notebook nuevo una vez normalizada la numeración (p. ej. «## 1. Qué se va a probar» → «## 2. Qué se va a probar»). Los otros **3** son renombrados, y cada uno tiene ahora su fila en `trazabilidad_notebooks.md`: el título de 04 pasa a «Protocolo de evaluación…»; «## 4. Ejecución reanudable» pasa a «## 7. Ejecución reanudable y CLI paralela»; el título de 05 pasa a «12 — Comparativa de detectores…». Los notebooks v1 de `capa1_exploracion/` (238 títulos) no estaban en el inventario previo. Se reescribieron dentro de los notebooks de familia 05–11, y su destino, sección a sección, está en `trazabilidad_notebooks.md`. Lo que no se rescató tiene su fila en `registro_eliminaciones.md`, y los 15 notebooks v1 aparecen en ambos documentos. |
| c | notebooks: errores | **OK (0 errores, 0 celdas sin ejecutar)** | Los 21 notebooks nuevos (00–20) tienen **0 salidas de error** y contadores de ejecución monótonos. El comparador dejó 3 celdas «Veredicto v2» de §6.1 sin ejecutar (`07_familia_F3_hmm.ipynb` celda 48, `10_familia_F6_changepoint.ipynb` celda 31, `11_familia_F7_deep.ipynb` celda 31). El corrector final re-ejecutó los 3 notebooks con los flags por defecto (ver «Correcciones del corrector final»): ahora todas sus celdas de código tienen `execution_count` y salida, sin errores. |
| d | API pública previa | **OK** | El inventario previo tiene 149 defs y 51 constantes, 200 nombres en total, y cada uno tiene su fila en el contrato §6. **198 son accesibles** en su módulo destino. Los otros 2 son `LEGACY_ROOT` (`src/benchmark.py` y `src/detectors/registry.py`), eliminados con fila en el registro. También se comprobó lo siguiente: están los re-exports exigidos en los `__init__` (`evaluacion`, `features`, `detectores`, `datos`, `viz`, `fusion`, que incluye sus 19 defs y 3 constantes); `features.LAG_PUBLICACION` es el mismo objeto que `features.lags.LAG_PUBLICACION`; `CRISIS_WINDOWS` es el mismo objeto en `evaluacion`, `evaluacion.metricas` y `viz`; y las 24 `factory()` (pistas A y B) se instancian sin error. |
| e | tests | **OK: 199 passed** | `pytest --collect-only` recoge **199 tests** (≥ 158). De los 158 previos se conservan 157 por nombre (ahora en subcarpetas de `tests/`). El que falta, `test_benchmark_parallel.py::test_clean_sys_path_removes_legacy_root_temporarily`, está eliminado con fila en el registro. El corrector final ejecutó `python -m pytest`: **199 passed**. `pytest -m "not datos and not resultados"` (`make test-rapido`) da 163 passed y 36 deselected. |
| f | enlaces `.md` | **OK** | 33 `.md` con enlaces relativos y 306 enlaces en total, con **0 rotos** (antes: 43 enlaces, 0 rotos). |

**Veredicto:** los números de la unificación son idénticos a la foto previa (a, b), la API
(d) y la documentación (f) cuadran. Los tests pasan (e: 199 passed) y los notebooks 07, 10 y
11 están ejecutados por completo (c). No queda nada pendiente para cerrar.

## Diferencias aceptadas (solo tiempo, huella y ruta)

| archivo(s) | columnas o claves que difieren | por qué es aceptable |
|---|---|---|
| `benchmark/metrics/{A,B}_D01..D12.csv` (24) | `cache_fingerprint`, `elapsed_seconds` | La huella hashea el código del paquete nuevo (rutas `src/regimenes/...`) y el tiempo depende de la ejecución. Las otras 155 columnas coinciden exactamente. |
| `benchmark/ranking_v2.csv` | `elapsed_seconds` | Solo lo arrastra de las métricas. Puestos, scores y el resto de métricas coinciden exactamente. |
| `benchmark/run_status.csv` | `segundos`, `inicio_utc`, `fin_utc`, `pid`, `detalle` | Son tiempos y el proceso. `detalle` coincide al traducir `benchmark_v2` → `benchmark`, y `estado` es `ok` 24/24 en ambos. |
| `benchmark/status/{A,B}_D*.json` (24) | `segundos`, `inicio_utc`, `fin_utc`, `pid`, `detalle` | Igual que `run_status.csv`: `detalle` coincide al traducir la ruta y `estado = ok` en ambos. |
| `benchmark/manifest.json` | `generated_utc`, `cache_fingerprints/{A,B}/D01..D12`, `benchmark_spec_sha256` | Son huellas y fecha. `benchmark_spec_sha256` cambia porque `configs/benchmark_spec.yaml` (antes `data/benchmark_spec.yaml`) solo cambió en **5 líneas de comentario** (referencias `src/features.py` → `regimenes.features...`, `data/catalog.yaml` → `configs/catalog.yaml`). El hash del manifest coincide con el archivo vigente (`c7cdfade…`). Las demás claves (`runtime_versions`, `train_days`, `tracks`, …) son idénticas. |

No hay ninguna otra diferencia: los 27 CSV de `fusion/` y `metrics_master_v2.csv` son
idénticos byte a byte en su contenido.

## Resumen del registro de eliminaciones

`docs/revisiones/registro_eliminaciones.md` tiene 83 filas en tres tablas. La tabla
general (47 filas) se desglosa aquí en los puntos 1 y 2; la de retirada de
`capa1_exploracion/` tiene 14 filas y la de correcciones de texto, 22.

1. **Código (filas 11–34, 56–57).** Recoge lo siguiente:
   - copias duplicadas (`capa1_exploracion/src/detector_base.py`);
   - `__init__` antiguos fusionados (`src/detectors`, `src/ingest`);
   - `LEGACY_ROOT` ×2, el guard de import y `_clean_sys_path_for_workers`, junto con su test (el único test eliminado);
   - el bloque `__main__` de `src/benchmark.py` y el fallback de import de `viz`;
   - 2 imports sin uso;
   - los archivos partidos (`benchmark.py`, `features.py`, `detector_base.py`, `src/README.md`), que git ve como borrados;
   - la deduplicación de `references.bib`;
   - el arranque con `sys.path` de los notebooks 04 y 05;
   - la reversión del cálculo de huecos en `resolve_context`, que mantiene vigente la caché;
   - el comentario falso «el codigo ya es robusto a ambos» de `pyproject.toml` (fila 57).

   Todo queda en `src/regimenes/…` o en los tags `v2-pre-unificacion` y `capa1-final`.
2. **Homogeneización 05–11 y rescate de notebooks v1 (filas 35–55).** Son las secciones v1 que no se rehicieron: celdas de test de causalidad, barridos BIC, PCA y heatmaps redundantes, y el recompute del marco v1 de `13_comparison`. Cada una cita su recuperación con `git show capa1-final:…`.
3. **Retirada de `capa1_exploracion/` (filas 68–81).** Cubre los 15 notebooks v1, que tienen como destino un notebook de familia o `docs/historia/capa1/`. También cubre el README, el marco `src/` v1, los `.gitkeep`, el bytecode no versionado y los patrones obsoletos de `.gitignore` y `pyproject.toml`.
4. **Correcciones científicas de texto en 04–13 (filas 91–112).** Son 22 frases sustituidas porque contradecían sus propias salidas o ADR-003. El texto original se conserva en la tabla.

## Cambios hechos por el comparador

- `docs/revisiones/trazabilidad_notebooks.md`: 3 filas nuevas para los títulos renombrados
  (04, 04 §4 → §7 y 05 → 12), que antes no figuraban de forma literal.
- Este informe.

## Correcciones del corrector final

Hallazgos del code review y del comparador. Ninguno cambia resultados numéricos: los 80
CSV/JSON/parquet de `results/benchmark` y `results/detectores` tienen el mismo md5 antes y
después de todo lo siguiente, y no se relanzó el benchmark.

1. **`tests/benchmark/test_cache_versionada_vigente.py`: marcadores y guarda de skip.** El test
   lee `data/raw`, `data/processed` y `results/benchmark`, pero no llevaba marcador, así que
   `make test-rapido` lo seleccionaba y podía fallar de forma transitoria. Ahora lleva
   `@pytest.mark.datos` y `@pytest.mark.resultados`. Además se salta si
   `data/raw/<fuente>/SP500.parquet` no es unívoco. Antes, `_sp500_path()` lanzaba
   `FileNotFoundError`, la caché marcaba las 24 combinaciones como `sin_cache` y el test fallaba
   con «Caché del benchmark NO vigente» en lugar de saltarse. La lógica de la comprobación no
   cambia.
2. **`pyproject.toml`: comentario de la cota `pandas<3`.** Decía que el código «ya es robusto» a la
   resolución `us` de pandas 3, y es falso desde que se revirtió `resolve_context`.
   Reproducido: con un índice mensual `as_unit('us')` o `as_unit('s')`, `resolve_context('auto', …)`
   devuelve 252 en lugar de 12. El comentario documenta ahora esa limitación y por qué no se
   corrige en `walk_forward.py` (el archivo entra en la huella de la caché). Hoy no tiene efecto
   numérico: el pin es `pandas<3`, los índices son `datetime64[ns]` y un índice diario da 252 en
   cualquier unidad. La frase retirada tiene su fila en `registro_eliminaciones.md` (fila 57).
3. **Notebooks 07, 10 y 11 re-ejecutados** con `nbconvert --execute --inplace` y los flags por
   defecto: 07 con `EJECUTAR = False` y `EJECUTAR_ABLACION = False`; 10 y 11 con
   `EJECUTAR_ABLACION = True` y `FORZAR_ABLACION = False`. Tiempos: 82 s, 22 s y 35 s, todo desde
   caché, sin recalcular ningún detector ni variante. Las celdas «Veredicto v2» de §6.1 tienen
   ahora salida, y sus cifras coinciden con las tablas del mismo notebook (p. ej. F1 de D07
   0.469/0.426 en A/B, igual que el `RESUMEN` de §4).
4. **Tests:** `python -m pytest` da **199 passed**.
5. **`trazabilidad_notebooks.md`:** comprobado que están las 3 filas de títulos renombrados que
   añadió el comparador (sin cambios adicionales).

## Cierre del coordinador

- **Benchmark completo re-ejecutado con el paquete nuevo:** 24/24 combinaciones `ok`; los 24 paneles OOS
  y todas las columnas numéricas de métricas son **idénticos** a la foto previa (comprobación
  independiente con `assert_frame_equal(check_exact=True)`).
- **Notebooks 00–20:** ejecutados en orden con kernel limpio sobre la caché vigente (24/24 «caché
  verificada»): 0 errores, 0 celdas sin ejecutar.
- **Revisión científica final:** texto contrastado con salidas en 04–14; corregidas afirmaciones que
  contradecían los datos v2 (p. ej. 07: el estado de crisis del HMM-t con K=4 no es la «cola estrecha» en
  la pista A; 05: D02 es la alerta, no el confirmador, en 14; `taper_2013` es ventana trampa en v2).
  Detalle en `registro_eliminaciones.md` §«Revisión científica final».
- **Pendiente conocido (no bloqueante):** `evaluacion.walk_forward.resolve_context` asume índices en
  nanosegundos (`index.asi8`). Con `pandas<3` fijado en `pyproject.toml` es correcto; el arreglo para
  pandas 3 cambia el hash de `walk_forward.py`, que forma parte de la huella de caché, así que se
  aplicará la próxima vez que se re-ejecute el benchmark completo.
- Las figuras de los notebooks de familia (`results/detectores/**/*.png`, 26 MB) no se versionan: van
  embebidas en los notebooks y se regeneran al ejecutarlos.
