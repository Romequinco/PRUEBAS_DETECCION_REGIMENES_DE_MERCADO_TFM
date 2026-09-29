# Contrato de unificación (ADR-004): `src/` + `capa1_exploracion/` → paquete `regimenes`

Rama `unificacion`. Foto previa: `docs/revisiones/baseline/` (149 defs públicas en
`api_inventory.json`, 158 tests en `tests_inventory.txt`, hashes de paneles y resultados).
Tags de respaldo: `capa1-final` (Capa 1 congelada) y `v2-pre-unificacion` (código v2 previo).

Este documento es **vinculante** para los agentes de zona P1, P2 y P3. Si algo no encaja,
se reporta; no se improvisa otra ubicación.

## 0. Estado que deja el arquitecto (ya hecho)

- `pyproject.toml` (setuptools, `package-dir = src`, paquete `regimenes`, `python>=3.11`,
  extras `[deep]` torch, `[jump]` jumpmodels, `[changepoint]` ruptures, `[dev]` pytest,
  nbconvert, nbformat, jupyter, ipykernel; `[tool.pytest.ini_options] testpaths = ["tests"]`,
  **sin** `pythonpath`: el paquete se instala).
- `requirements.txt` conserva la lista completa (entorno de notebooks incluido) y termina en
  `-e .`. Decisión: se mantiene porque fija además Jupyter y los extras; las dependencias de
  ejecución de la librería viven en `pyproject.toml`. `ruptures` pasa a extra opcional
  (el código no lo importa; solo se cita como oráculo en D07).
- Instalado: `python -m pip install --no-build-isolation -e .` (import verificado desde `/`).
- Esqueleto completo de `src/regimenes/` con todos los subpaquetes, `rutas.py` implementado y
  `sinteticos/` completo (interfaz `Generador`, registro vacío, `validacion.py` con firmas y
  `NotImplementedError`).
- Movidos con `git mv`: `data/catalog.yaml` → `configs/catalog.yaml`;
  `data/benchmark_spec.yaml` → `configs/benchmark_spec.yaml`; `results/benchmark_v2` →
  `results/benchmark` (los 24 `panels/*.parquet`, gitignored, se movieron físicamente con la
  carpeta); `results/fusion_d07_d08` → `results/fusion/d07_d08`; `results/fusion_d02_d06` →
  `results/fusion/d02_d06`.
- Nuevos: `configs/sinteticos.yaml` (esqueleto), `.gitkeep` en `data/sinteticos/`,
  `results/detectores/`, `results/sinteticos/`, `results/pseudolive/`. `.gitignore`
  actualizado (`results/benchmark/panels/*.parquet`, `data/sinteticos/*` salvo `.gitkeep`,
  `*.parquet` bajo las nuevas carpetas de resultados).
- **No editados** (se regeneran con el benchmark completo tras la fase):
  `results/benchmark/manifest.json`, `metrics/*.csv`, `status/*.json` siguen conteniendo
  rutas `results/benchmark_v2/...` y huellas antiguas.
- Estado intermedio esperado: hasta que P1–P3 terminen, `src/*.py` antiguo sigue apuntando a
  `data/benchmark_spec.yaml` y `tests/test_preprocesado_causalidad.py` falla en la colección.

## 1. Reglas comunes (recordatorio operativo)

1. La lógica **se mueve, no se reescribe**. Solo cambian: líneas de import, definición de
   rutas (→ `regimenes.rutas`) y lo que este contrato indique explícitamente.
2. `git mv` del archivo antiguo al módulo **principal** indicado; los demás módulos del
   reparto se crean nuevos cortando/pegando bloques íntegros (funciones completas, con sus
   comentarios de sección).
3. Imports internos **absolutos**: `from regimenes.x import y`. Prohibido `from src...`,
   `from detectors...`, imports relativos, `sys.path.insert/append` y cualquier referencia a
   `capa1_exploracion` en el código del paquete.
4. **Rutas**: siempre vía `regimenes.rutas` (`ROOT`, `CONFIGS`, `CATALOG`, `BENCHMARK_SPEC`,
   `SINTETICOS_CONFIG`, `DATA`, `DATA_RAW`, `DATA_PROCESSED`, `DATA_SINTETICOS`, `RESULTS`,
   `RESULTS_BENCHMARK`, `RESULTS_FUSION`, `RESULTS_FUSION_D07_D08`, `RESULTS_FUSION_D02_D06`,
   `RESULTS_DETECTORES`, `RESULTS_SINTETICOS`, `RESULTS_PSEUDOLIVE`, `DOCS`, `ENV_FILE`).
   Ningún módulo calcula `Path(__file__).parents[...]`. Las constantes públicas antiguas de
   ruta (`ROOT`, `RAW`, `CATALOG`, `PROCESSED`, `DEFAULT_OUTPUT`) se **conservan como alias**
   en su módulo destino: `from regimenes.rutas import ROOT` / `DEFAULT_OUTPUT = RESULTS_BENCHMARK`.
   Los tests hacen lo mismo (nada de `ROOT / "results" / "benchmark_v2"`).
5. Firmas públicas intactas: mismo nombre, mismos parámetros y defaults; solo cambia el módulo.
6. Toda eliminación de archivo o bloque grande → fila en `docs/revisiones/registro_eliminaciones.md`
   (`elemento | motivo | dónde queda`). Mover no es eliminar.
7. Nada de commit/push/stash/reset/checkout; nada de benchmark completo. Sí `pytest` sobre tu zona.
8. No tocar `notebooks/`, `docs/` (salvo el registro), `capa1_exploracion/` fuera de
   `detectors/` y `src/`.
9. Borrar `__pycache__/` de carpetas que queden vacías (no están versionados).

## 2. Tabla de módulos: antiguo → nuevo

| Antiguo | Nuevo | Operación | Zona |
|---|---|---|---|
| `src/ingest/__init__.py` | `src/regimenes/datos/__init__.py` | contenido fusionado en el `__init__` ya creado; `git rm` del antiguo | P1 |
| `src/ingest/download.py` | `src/regimenes/datos/descarga.py` | `git mv` (principal) | P1 |
| (bloque `load_catalog` de download.py) | `src/regimenes/datos/catalogo.py` | nuevo | P1 |
| (bloque `if __name__ == "__main__"` de download.py) | `src/regimenes/datos/__main__.py` | nuevo (`python -m regimenes.datos`) | P1 |
| `src/ingest/sources.py` | `src/regimenes/datos/fuentes.py` | `git mv` | P1 |
| `src/features.py` §1 y §2 | `src/regimenes/features/transformaciones.py` | `git mv` (principal) | P1 |
| `src/features.py` §3 | `src/regimenes/features/lags.py` | nuevo | P1 |
| `src/features.py` §4 | `src/regimenes/features/paneles.py` | nuevo | P1 |
| `src/features.py` §5 | `src/regimenes/features/causalidad.py` | nuevo | P1 |
| `src/detector_base.py` | `src/regimenes/detectores/base.py` | `git mv` | P2 |
| `capa1_exploracion/src/detector_base.py` | (idéntica byte a byte a la anterior) | `git rm` + fila en registro | P2 |
| `src/detectors/registry.py` | `src/regimenes/detectores/registry.py` | `git mv` | P2 |
| `src/detectors/__init__.py` | `src/regimenes/detectores/__init__.py` | re-exports fusionados; `git rm` + fila | P2 |
| `capa1_exploracion/detectors/rule_vix_threshold.py` | `detectores/f1_reglas/rule_vix_threshold.py` | `git mv` | P2 |
| `capa1_exploracion/detectors/rule_composite_riskoff.py` | `detectores/f1_reglas/rule_composite_riskoff.py` | `git mv` | P2 |
| `capa1_exploracion/detectors/turbulence_mahalanobis.py` | `detectores/f1_reglas/turbulence_mahalanobis.py` | `git mv` | P2 |
| `capa1_exploracion/detectors/clustering_gmm.py` | `detectores/f2_clustering/clustering_gmm.py` | `git mv` | P2 |
| `capa1_exploracion/detectors/jump_model.py` | `detectores/f2_clustering/jump_model.py` | `git mv` | P2 |
| `capa1_exploracion/detectors/hmm_gaussian_2s.py` | `detectores/f3_hmm/hmm_gaussian_2s.py` | `git mv` | P2 |
| `capa1_exploracion/detectors/hmm_tstudent.py` | `detectores/f3_hmm/hmm_tstudent.py` | `git mv` | P2 |
| `capa1_exploracion/detectors/hsmm_tstudent.py` | `detectores/f3_hmm/hsmm_tstudent.py` | `git mv` | P2 |
| `capa1_exploracion/detectors/_hmm_utils.py` | `detectores/f3_hmm/_hmm_utils.py` | `git mv` | P2 |
| `capa1_exploracion/detectors/_hmm_t_utils.py` | `detectores/f3_hmm/_hmm_t_utils.py` | `git mv` | P2 |
| `capa1_exploracion/detectors/markov_switching_var.py` | `detectores/f4_switching/markov_switching_var.py` | `git mv` | P2 |
| `capa1_exploracion/detectors/garch_t_vol.py` | `detectores/f5_garch/garch_t_vol.py` | `git mv` | P2 |
| `capa1_exploracion/detectors/msgarch_regime.py` | `detectores/f5_garch/msgarch_regime.py` | `git mv` | P2 |
| `capa1_exploracion/detectors/changepoint_online.py` | `detectores/f6_changepoint/changepoint_online.py` | `git mv` | P2 |
| `capa1_exploracion/detectors/deep_ae_regime.py` | `detectores/f7_deep/deep_ae_regime.py` | `git mv` | P2 |
| `src/evaluation.py` (resultado, protocolo, orquestador) | `src/regimenes/evaluacion/walk_forward.py` | `git mv` (principal) | P3 |
| `src/evaluation.py` (ventanas + métricas individuales) | `src/regimenes/evaluacion/metricas.py` | nuevo | P3 |
| `src/benchmark.py` (ejecución) | `src/regimenes/benchmark/ejecucion.py` | `git mv` (principal) | P3 |
| `src/benchmark.py` (rutas, huella, verificación de caché) | `src/regimenes/benchmark/cache.py` | nuevo | P3 |
| `src/benchmark.py` (CLI) | `src/regimenes/benchmark/cli.py` + `__main__.py` | nuevos | P3 |
| `src/benchmark.py` (ranking ADR-003 + líneas base) | `src/regimenes/evaluacion/ranking.py` | nuevo | P3 |
| `src/benchmark.py` (detección por evento) | `src/regimenes/evaluacion/metricas.py` | bloque añadido | P3 |
| `src/fusion.py` | `src/regimenes/fusion/maquina.py` | `git mv`; `fusion/__init__.py` re-exporta la API pública | P3 |
| `src/viz.py` | `src/regimenes/viz/figuras.py` | `git mv`; `viz/__init__.py` re-exporta la API pública | P3 |
| `src/README.md` | `src/regimenes/README.md` | `git mv` + tabla de módulos actualizada | P3 |

`capa1_exploracion/src/{data_loader,evaluation,features,viz}.py` **no se tocan** en esta fase
(versión congelada de Capa 1; su destino lo decide la fase de notebooks/docs). El paquete no
los importa.

## 3. P1 — `datos/` + `features/`

### datos/
- `catalogo.py`: `CATALOG` (alias de `rutas.CATALOG`) y `load_catalog()` **leyendo el global
  `CATALOG` del módulo en tiempo de llamada** (igual que hoy).
- `descarga.py`: resto de `download.py` (`ROOT`, `RAW` alias de `rutas`, `MIN_FRAC_VS_CACHE`,
  `MIN_OBS`, `COLUMNA_POR_SERIE`, `DERIVADAS`, `download_all` y privadas). Importa
  `from regimenes.datos import fuentes as sources` (el nombre local `sources` se mantiene
  para no tocar el cuerpo) y `from regimenes.datos.catalogo import load_catalog`.
- `fuentes.py`: `sources.py` íntegro; `ROOT` alias y `.env` vía `rutas.ENV_FILE`.
- `__main__.py`: el bloque CLI de `download.py` literal, llamando a `download_all`.
  Docstring de `descarga.py`: CLI pasa a `python -m regimenes.datos [--offline|--force --only A,B]`.
- `__init__.py`: re-exporta `catalogo`, `descarga`, `fuentes`, `download_all`, `load_catalog`
  (y alias `sources = fuentes`, `download = descarga` para compatibilidad).
- Tests: `tests/test_ingest_download.py` pasa a `from regimenes.datos import descarga as download, fuentes as sources`;
  el fixture que parchea `download.CATALOG` debe parchear `regimenes.datos.catalogo.CATALOG`
  (es donde lo lee `load_catalog`); `RAW` sigue en `descarga`.

### features/
Corte por las secciones numeradas que ya tiene `src/features.py`:
- `transformaciones.py` (principal, `git mv`): §1 primitivas causales + §2 recetario v2
  (`TRADING_DAYS`, `RawDict`, `BREADTH_11`, `SPDR_9`, `_get`, … `lag_publicacion`, `yoy`,
  `macro_z`, …, `fed_stance`).
- `lags.py`: §3 (`LAG_PUBLICACION`, `lag_de`, `aplicar_lag_publicacion`,
  `tabla_lags_publicacion`); importa `RawDict`, `lag_publicacion` de `transformaciones`.
- `paneles.py`: §4 (`N_FEAT_*`, `MENSUAL_*`, `PANELES_MULTICOLUMNA`, `GW_COLUMN`,
  `seleccionar_raw`, `_col`, `_zscore_col`, `ff_industry_dispersion`, `_ensamblar`,
  `construir_diarias`, `construir_mensuales`, `_asof`, `alinear_*`, `rejilla_nyse`,
  `construir_paneles`); importa de `transformaciones` y `lags`.
- `causalidad.py`: §5 (`truncar`, `truncation_max_abs_diff`, `assert_causal`).
- Dependencias en una sola dirección: transformaciones ← lags ← paneles; causalidad solo usa
  `RawDict`.
- `__init__.py`: re-exporta **toda** la API pública anterior (defs y constantes, incluidas
  `LAG_PUBLICACION` y `RawDict`, que el inventario no lista pero los tests usan) para que
  `from regimenes import features as ft; ft.X` funcione. `LAG_PUBLICACION` debe ser el
  **mismo objeto** dict (import, no copia).
- Tests: `test_features_causalidad.py`, `test_preprocesado_causalidad.py` →
  `from regimenes import features as ft`; `ft._col` → `from regimenes.features.paneles import _col`;
  rutas `data/benchmark_spec.yaml`/`data/catalog.yaml` → `rutas.BENCHMARK_SPEC`/`rutas.CATALOG`,
  `data/raw`/`data/processed` → `rutas.DATA_RAW`/`rutas.DATA_PROCESSED`.

## 4. P2 — `detectores/`

- `base.py` ← `git mv src/detector_base.py`. `git rm capa1_exploracion/src/detector_base.py`
  (idéntica; fila en registro, dónde queda: `regimenes/detectores/base.py` y tag `capa1-final`).
- 15 archivos de `capa1_exploracion/detectors` → `detectores/fN_*/` según la tabla §2. Únicos
  cambios permitidos en ellos: las líneas de import
  - `from src.detector_base import RegimeDetector` → `from regimenes.detectores.base import RegimeDetector`
  - `from detectors._hmm_utils import …` → `from regimenes.detectores.f3_hmm._hmm_utils import …`
  - `from detectors._hmm_t_utils import …` → `from regimenes.detectores.f3_hmm._hmm_t_utils import …`
  - `from detectors.hmm_gaussian_2s import …` / `from detectors.hmm_tstudent import …` → `regimenes.detectores.f3_hmm.…`
  - y, si algún docstring menciona `capa1_exploracion/detectors/X.py` como ruta de uso, puede
    actualizarse (no es obligatorio).
- `registry.py` ← `git mv src/detectors/registry.py`:
  - `ROOT` alias de `rutas.ROOT`; **`LEGACY_ROOT` se elimina** (fila en registro; motivo:
    regla ADR-004 de cero referencias a `capa1_exploracion`).
  - `DetectorSpec.module` **conserva sus valores actuales** (`"rule_vix_threshold"`, …) para
    no alterar la configuración declarada. `factory()` deja de tocar `sys.path` y resuelve
    `import_module(f"regimenes.detectores.{_FAMILIA_POR_MODULO[self.module]}.{self.module}")`
    con un dict privado módulo→subpaquete (los 12 módulos oficiales + `hsmm_tstudent`).
  - `SEED`, `CORE_A`, `CORE_B`, `detector_specs`, `specs_table`: sin cambios.
- `detectores/__init__.py`: re-exporta `DetectorSpec`, `detector_specs`, `specs_table` (como
  `src/detectors/__init__.py`) y `RegimeDetector`. `git rm src/detectors/__init__.py` (+ fila).
- `f*/__init__.py`: ya existen con docstring; no importan nada (import perezoso vía registry).
- Borrar `src/detectors/` y `capa1_exploracion/detectors/` cuando queden vacíos (solo `__pycache__`).
- Tests: `tests/test_registry_specs.py`. `test_hashed_detector_base_is_identical_to_the_one_imported`
  pierde su motivo (ya hay una sola copia): reescribirlo para comprobar que
  `regimenes.benchmark.cache._resolved_detector_base()` == archivo de `regimenes.detectores.base`
  y que está en `_cache_code_paths()`. Rutas de YAML vía `rutas.BENCHMARK_SPEC`.
  `bm` → `from regimenes.benchmark import cache as bm_cache` (nombres de P3 fijados en §5).
- Comprobar al final: `python -c "from regimenes.detectores import detector_specs; [s.factory() for t in 'AB' for s in detector_specs(t)]"`.

## 5. P3 — `evaluacion/`, `benchmark/`, `fusion/`, `viz/`

### evaluacion/
- `walk_forward.py` (principal, `git mv src/evaluation.py`): `EvaluationResult`,
  `CONTEXT_YEARS`, `_OBS_PER_YEAR`, `resolve_context`, `_ctx_start`, `_predict_block`,
  `walk_forward`, `evaluate`, `results_table`. Importa de `metricas` las métricas y las
  ventanas.
- `metricas.py`: `CRISIS_WINDOWS`, `FALSE_POSITIVE_WINDOWS`, `DRAWDOWN_TROUGHS`,
  `configure_event_windows`, `_in_any_window`, `crisis_coverage`, `false_alarm_in_windows`,
  `false_alarm_rate`, `_first_sustained`, `lead_lag`, `switching_rate`,
  `mean_regime_duration`, `label_stability`, `silhouette_states`,
  `block_bootstrap_coverage_ci` **+ desde benchmark.py**: `_longest_true_run`,
  `event_detection_table`, `detection_summary` (métricas de detección por evento que
  `run_one` escribe como `det_*`). Los tres dicts de ventanas son **objetos únicos**:
  `configure_event_windows` ya los muta in situ; todos los demás módulos (walk_forward,
  fusion, viz, ranking) los importan de `metricas` o acceden como `ev.CRISIS_WINDOWS`.
- `ranking.py` (criterio ADR-003; sale de benchmark.py): `add_comparison_metrics`,
  `rank_within_track`, `DETECTION_DEFAULTS`, `DETECTION_LEVELS`, `track_crisis_windows`,
  `track_false_positive_windows`, `track_oos_index`, `_detection_from_coverage`,
  `_approx_crisis_run`, `add_detection_metrics`, `detection_score`, `rank_detection`,
  `RANKING_COLUMNS`, `pareto_mask`, `_markov_flags`, `_baseline_row`, `trivial_baselines`,
  `persistent_random_null`. Importa `load_benchmark_spec`, `_panel_path`, `DEFAULT_OUTPUT`
  de `regimenes.benchmark.cache` y las métricas de `regimenes.evaluacion.metricas`.
- `evaluacion/__init__.py`: re-exporta la API pública de `walk_forward` y `metricas`
  (incluidos los tres dicts), de modo que `from regimenes import evaluacion as ev; ev.walk_forward`,
  `ev.CRISIS_WINDOWS` sigan funcionando. **No importa `ranking`** (evita ciclo
  ranking → benchmark → evaluacion).

### benchmark/
Grafo de imports obligatorio (sin ciclos):
`evaluacion.metricas` ← `evaluacion.walk_forward` ← `benchmark.cache` ← `benchmark.ejecucion`
← `benchmark.cli`; `evaluacion.ranking` → (`benchmark.cache`, `evaluacion.metricas`).
**Nada en `benchmark/` importa `evaluacion.ranking`.**

- `cache.py`: `ROOT`, `PROCESSED` (= `rutas.DATA_PROCESSED`), `DEFAULT_OUTPUT`
  (= `rutas.RESULTS_BENCHMARK`), `CACHE_SCHEMA_VERSION`, `load_benchmark_spec` (lee
  `rutas.BENCHMARK_SPEC`), `processed_paths`, `processed_available`, `_sp500_path`
  (`rutas.DATA_RAW`), `_metric_path`, `_panel_path`, `_status_path`, `_display_path`,
  `_write_atomic`, `_normalized_text_bytes`, `_parquet_content_digest`,
  `_yaml_content_digest`, `_sha256_file_version`, `_sha256_file`, `_relative_hashes`,
  `_resolved_detector_base`, `_runtime_versions`, `_benchmark_model_sha256`,
  `_cache_code_paths`, `_cache_fingerprint`, `_cache_matches`, `_write_manifest`,
  `_read_verified_cache`, `metrics_provenance`, `load_metrics`.
- `ejecucion.py` (principal, `git mv src/benchmark.py`): `DEFAULT_TRAIN_DAYS`,
  `configure_evaluation`, `load_track_panel`, `common_track_index`, `_dependency_for`,
  `preflight`, `run_one`, `_spec_for`, `run_job`, `_thread_limits`, `_write_run_status`,
  `_collect_metrics`, `run_benchmark`, `consolidate_run`. `run_one` importa
  `detection_summary` de `regimenes.evaluacion.metricas` y `_cache_fingerprint`,
  `_cache_matches`, … de `cache`.
- `cli.py`: `_parse_args` (`prog="python -m regimenes.benchmark"`), `main`.
  `__main__.py`: `from regimenes.benchmark.cli import main; raise SystemExit(main())`
  (el truco de reimportar como `src.benchmark` ya no hace falta: las funciones de los hijos
  viven en `regimenes.benchmark.ejecucion`, serializables).
- `benchmark/__init__.py`: puede re-exportar API pública de `cache`, `ejecucion`, `cli`;
  **nunca** de `evaluacion.ranking`.
- Eliminaciones (con fila en el registro): `LEGACY_ROOT`; `_clean_sys_path_for_workers`
  (existía solo porque `factory()` anteponía `capa1_exploracion` a `sys.path`; con el paquete
  instalado no hay manipulación de `sys.path`) y su uso en `run_benchmark` (el `with` se
  quita, el cuerpo queda igual); test `test_clean_sys_path_removes_legacy_root_temporarily`.
  La doble copia de `detector_base` en la huella desaparece (una sola).

#### Huella de caché (`_cache_code_paths` y compañía) — debe apuntar a los nuevos módulos
```
_cache_code_paths() = [
    <src/regimenes/evaluacion/walk_forward.py>,
    <src/regimenes/evaluacion/metricas.py>,
    <src/regimenes/detectores/__init__.py>,
    <src/regimenes/detectores/registry.py>,
    <src/regimenes/detectores/base.py>,          # = _resolved_detector_base()
    *sorted(<src/regimenes/detectores/f*/*.py>) excepto hsmm_tstudent.py,
]
```
- Rutas del paquete a partir de `rutas.ROOT / "src" / "regimenes"` o de
  `importlib.util.find_spec(...).origin`; no `Path(__file__)`.
- `_resolved_detector_base()` devuelve el archivo de `regimenes.detectores.base`;
  la clave `detector_base_resolved` de la huella queda `"src/regimenes/detectores/base.py"`.
- `_benchmark_model_sha256()` conserva **el mismo conjunto de 10 funciones**
  (`configure_evaluation`, `processed_paths`, `_sp500_path`, `load_track_panel`,
  `common_track_index`, `_dependency_for`, `run_one`, `_longest_true_run`,
  `event_detection_table`, `detection_summary`) y el mismo orden en el `ast.Module`, pero
  ahora las lee de tres módulos (`benchmark.ejecucion`, `benchmark.cache`,
  `evaluacion.metricas`) por `importlib.util.find_spec(mod).origin` + `ast.parse`, sin importar
  módulos que creen ciclos. Si falta alguna → `RuntimeError` como hoy.
- `input_sha256` usa `rutas.BENCHMARK_SPEC` (antes `data/benchmark_spec.yaml`).
- Consecuencia asumida: todas las huellas cambian → `load_metrics`/`metrics_provenance` sobre
  `results/benchmark` marcarán caché obsoleta hasta el benchmark completo posterior a esta
  fase. **No** se editan `manifest.json` ni los CSV para "arreglarlo".

### fusion/ y viz/
- `fusion/maquina.py` ← `git mv src/fusion.py`; `from src import evaluation as ev` →
  `from regimenes import evaluacion as ev`. `fusion/__init__.py` re-exporta los 19 defs + 3
  constantes públicas del inventario.
- `viz/figuras.py` ← `git mv src/viz.py`; el `try: from src.evaluation import … except: from evaluation import …`
  pasa a `from regimenes.evaluacion.metricas import CRISIS_WINDOWS, FALSE_POSITIVE_WINDOWS, DRAWDOWN_TROUGHS`
  (fila en registro por el fallback eliminado). `viz/__init__.py` re-exporta la API pública
  del inventario **y** los tres dicts (los tests usan `viz.CRISIS_WINDOWS`).

### Tests de P3
`test_benchmark_parallel.py`, `test_benchmark_ranking.py`, `test_benchmark_safety.py`,
`test_evaluation_metrics.py`, `test_fusion.py`, `test_fusion_extra.py`,
`test_ranking_detection.py`, `test_walkforward_context.py`:
- `from src import evaluation as ev` → `from regimenes import evaluacion as ev`;
  `from src.detector_base import RegimeDetector` → `from regimenes.detectores.base import RegimeDetector`;
  `from src.fusion import …` → `from regimenes.fusion import …`; `from src import viz` → `from regimenes import viz`.
- `bm.X` → importar del módulo destino de X (tabla §6). Los `mock.patch.object(bm, "run_one")`
  y `mock.patch.object(bm, "_cache_fingerprint")` deben parchear **el módulo donde se busca el
  nombre en el camino probado** (`regimenes.benchmark.ejecucion` para `run_one` dentro de
  `run_job`/`run_benchmark`; `_cache_fingerprint` se usa en `ejecucion.run_one` y en
  `cache._read_verified_cache/_write_manifest/load_metrics`: parchear ambos si el test
  atraviesa los dos).
- `RESULTS = ROOT / "results" / "benchmark_v2"` → `rutas.RESULTS_BENCHMARK`.
- Número de tests: se conserva salvo los eliminados con fila en el registro.

## 6. Destino de cada definición pública del inventario (149 defs)

Las privadas (`_x`) siguen a su bloque según §3–§5.

#### `src/benchmark.py`

| nombre | tipo | modulo destino |
|---|---|---|
| `load_benchmark_spec` | def/clase | `regimenes.benchmark.cache` |
| `configure_evaluation` | def/clase | `regimenes.benchmark.ejecucion` |
| `processed_paths` | def/clase | `regimenes.benchmark.cache` |
| `processed_available` | def/clase | `regimenes.benchmark.cache` |
| `load_track_panel` | def/clase | `regimenes.benchmark.ejecucion` |
| `common_track_index` | def/clase | `regimenes.benchmark.ejecucion` |
| `preflight` | def/clase | `regimenes.benchmark.ejecucion` |
| `run_one` | def/clase | `regimenes.benchmark.ejecucion` |
| `run_job` | def/clase | `regimenes.benchmark.ejecucion` |
| `run_benchmark` | def/clase | `regimenes.benchmark.ejecucion` |
| `consolidate_run` | def/clase | `regimenes.benchmark.ejecucion` |
| `metrics_provenance` | def/clase | `regimenes.benchmark.cache` |
| `load_metrics` | def/clase | `regimenes.benchmark.cache` |
| `add_comparison_metrics` | def/clase | `regimenes.evaluacion.ranking` |
| `rank_within_track` | def/clase | `regimenes.evaluacion.ranking` |
| `event_detection_table` | def/clase | `regimenes.evaluacion.metricas` |
| `detection_summary` | def/clase | `regimenes.evaluacion.metricas` |
| `track_crisis_windows` | def/clase | `regimenes.evaluacion.ranking` |
| `track_false_positive_windows` | def/clase | `regimenes.evaluacion.ranking` |
| `track_oos_index` | def/clase | `regimenes.evaluacion.ranking` |
| `add_detection_metrics` | def/clase | `regimenes.evaluacion.ranking` |
| `detection_score` | def/clase | `regimenes.evaluacion.ranking` |
| `rank_detection` | def/clase | `regimenes.evaluacion.ranking` |
| `pareto_mask` | def/clase | `regimenes.evaluacion.ranking` |
| `trivial_baselines` | def/clase | `regimenes.evaluacion.ranking` |
| `persistent_random_null` | def/clase | `regimenes.evaluacion.ranking` |
| `main` | def/clase | `regimenes.benchmark.cli` |
| `ROOT` | constante | `regimenes.benchmark.cache (alias de regimenes.rutas.ROOT)` |
| `PROCESSED` | constante | `regimenes.benchmark.cache (alias de regimenes.rutas.DATA_PROCESSED)` |
| `DEFAULT_OUTPUT` | constante | `regimenes.benchmark.cache (alias de regimenes.rutas.RESULTS_BENCHMARK)` |
| `DEFAULT_TRAIN_DAYS` | constante | `regimenes.benchmark.ejecucion` |
| `CACHE_SCHEMA_VERSION` | constante | `regimenes.benchmark.cache` |
| `LEGACY_ROOT` | constante | `ELIMINADO (ver registro_eliminaciones.md)` |
| `RANKING_COLUMNS` | constante | `regimenes.evaluacion.ranking` |

#### `src/detector_base.py`

| nombre | tipo | modulo destino |
|---|---|---|
| `RegimeDetector` | def/clase | `regimenes.detectores.base` |

#### `src/detectors/registry.py`

| nombre | tipo | modulo destino |
|---|---|---|
| `DetectorSpec` | def/clase | `regimenes.detectores.registry` |
| `detector_specs` | def/clase | `regimenes.detectores.registry` |
| `specs_table` | def/clase | `regimenes.detectores.registry` |
| `ROOT` | constante | `regimenes.detectores.registry (alias de regimenes.rutas.ROOT)` |
| `LEGACY_ROOT` | constante | `ELIMINADO (ver registro_eliminaciones.md)` |
| `SEED` | constante | `regimenes.detectores.registry` |
| `CORE_A` | constante | `regimenes.detectores.registry` |
| `CORE_B` | constante | `regimenes.detectores.registry` |

#### `src/evaluation.py`

| nombre | tipo | modulo destino |
|---|---|---|
| `configure_event_windows` | def/clase | `regimenes.evaluacion.metricas` |
| `EvaluationResult` | def/clase | `regimenes.evaluacion.walk_forward` |
| `resolve_context` | def/clase | `regimenes.evaluacion.walk_forward` |
| `walk_forward` | def/clase | `regimenes.evaluacion.walk_forward` |
| `crisis_coverage` | def/clase | `regimenes.evaluacion.metricas` |
| `false_alarm_in_windows` | def/clase | `regimenes.evaluacion.metricas` |
| `false_alarm_rate` | def/clase | `regimenes.evaluacion.metricas` |
| `lead_lag` | def/clase | `regimenes.evaluacion.metricas` |
| `switching_rate` | def/clase | `regimenes.evaluacion.metricas` |
| `mean_regime_duration` | def/clase | `regimenes.evaluacion.metricas` |
| `label_stability` | def/clase | `regimenes.evaluacion.metricas` |
| `silhouette_states` | def/clase | `regimenes.evaluacion.metricas` |
| `block_bootstrap_coverage_ci` | def/clase | `regimenes.evaluacion.metricas` |
| `evaluate` | def/clase | `regimenes.evaluacion.walk_forward` |
| `results_table` | def/clase | `regimenes.evaluacion.walk_forward` |

#### `src/features.py`

| nombre | tipo | modulo destino |
|---|---|---|
| `causal_zscore` | def/clase | `regimenes.features.transformaciones` |
| `log_returns` | def/clase | `regimenes.features.transformaciones` |
| `realized_vol` | def/clase | `regimenes.features.transformaciones` |
| `rolling_correlation` | def/clase | `regimenes.features.transformaciones` |
| `drawdown` | def/clase | `regimenes.features.transformaciones` |
| `momentum` | def/clase | `regimenes.features.transformaciones` |
| `logret` | def/clase | `regimenes.features.transformaciones` |
| `ret_z` | def/clase | `regimenes.features.transformaciones` |
| `zscore_raw` | def/clase | `regimenes.features.transformaciones` |
| `change_z` | def/clase | `regimenes.features.transformaciones` |
| `spread_ret_z` | def/clase | `regimenes.features.transformaciones` |
| `cross_sectional_std_ret` | def/clase | `regimenes.features.transformaciones` |
| `cross_sectional_std_level` | def/clase | `regimenes.features.transformaciones` |
| `zscore_or_none` | def/clase | `regimenes.features.transformaciones` |
| `lag_publicacion` | def/clase | `regimenes.features.transformaciones` |
| `yoy` | def/clase | `regimenes.features.transformaciones` |
| `macro_z` | def/clase | `regimenes.features.transformaciones` |
| `mercado_z` | def/clase | `regimenes.features.transformaciones` |
| `mercado_spread_z` | def/clase | `regimenes.features.transformaciones` |
| `fed_stance` | def/clase | `regimenes.features.transformaciones` |
| `lag_de` | def/clase | `regimenes.features.lags` |
| `aplicar_lag_publicacion` | def/clase | `regimenes.features.lags` |
| `tabla_lags_publicacion` | def/clase | `regimenes.features.lags` |
| `seleccionar_raw` | def/clase | `regimenes.features.paneles` |
| `ff_industry_dispersion` | def/clase | `regimenes.features.paneles` |
| `construir_diarias` | def/clase | `regimenes.features.paneles` |
| `construir_mensuales` | def/clase | `regimenes.features.paneles` |
| `alinear_diaria` | def/clase | `regimenes.features.paneles` |
| `alinear_mensual_con_edad` | def/clase | `regimenes.features.paneles` |
| `rejilla_nyse` | def/clase | `regimenes.features.paneles` |
| `construir_paneles` | def/clase | `regimenes.features.paneles` |
| `truncar` | def/clase | `regimenes.features.causalidad` |
| `truncation_max_abs_diff` | def/clase | `regimenes.features.causalidad` |
| `assert_causal` | def/clase | `regimenes.features.causalidad` |
| `TRADING_DAYS` | constante | `regimenes.features.transformaciones` |
| `BREADTH_11` | constante | `regimenes.features.transformaciones` |
| `SPDR_9` | constante | `regimenes.features.transformaciones` |
| `N_FEAT_DIARIAS` | constante | `regimenes.features.paneles` |
| `N_FEAT_MENSUALES` | constante | `regimenes.features.paneles` |
| `MENSUAL_MERCADO` | constante | `regimenes.features.paneles` |
| `MENSUAL_MACRO` | constante | `regimenes.features.paneles` |
| `PANELES_MULTICOLUMNA` | constante | `regimenes.features.paneles` |
| `GW_COLUMN` | constante | `regimenes.features.paneles` |

#### `src/fusion.py`

| nombre | tipo | modulo destino |
|---|---|---|
| `FusionConfig` | def/clase | `regimenes.fusion.maquina` |
| `load_panel` | def/clase | `regimenes.fusion.maquina` |
| `crisis_state_from_metrics` | def/clase | `regimenes.fusion.maquina` |
| `operational_utility` | def/clase | `regimenes.fusion.maquina` |
| `fuse_early_warning` | def/clase | `regimenes.fusion.maquina` |
| `warning_episode_table` | def/clase | `regimenes.fusion.maquina` |
| `confirmation_episode_table` | def/clase | `regimenes.fusion.maquina` |
| `warning_phase_base_rates` | def/clase | `regimenes.fusion.maquina` |
| `warning_ground_truth_table` | def/clase | `regimenes.fusion.maquina` |
| `per_crisis_scorecard` | def/clase | `regimenes.fusion.maquina` |
| `signal_scores` | def/clase | `regimenes.fusion.maquina` |
| `ablation_scorecard` | def/clase | `regimenes.fusion.maquina` |
| `fusion_verdict` | def/clase | `regimenes.fusion.maquina` |
| `warning_summary` | def/clase | `regimenes.fusion.maquina` |
| `event_scorecard` | def/clase | `regimenes.fusion.maquina` |
| `period_scorecard` | def/clase | `regimenes.fusion.maquina` |
| `select_then_evaluate` | def/clase | `regimenes.fusion.maquina` |
| `sensitivity_table` | def/clase | `regimenes.fusion.maquina` |
| `assert_prefix_causal` | def/clase | `regimenes.fusion.maquina` |
| `DECISION_LABELS` | constante | `regimenes.fusion.maquina` |
| `PHASES` | constante | `regimenes.fusion.maquina` |
| `SWITCHING_PENALTY` | constante | `regimenes.fusion.maquina` |

#### `src/ingest/download.py`

| nombre | tipo | modulo destino |
|---|---|---|
| `load_catalog` | def/clase | `regimenes.datos.catalogo` |
| `download_all` | def/clase | `regimenes.datos.descarga` |
| `ROOT` | constante | `regimenes.datos.descarga (alias de regimenes.rutas.ROOT)` |
| `RAW` | constante | `regimenes.datos.descarga (alias de regimenes.rutas.DATA_RAW)` |
| `CATALOG` | constante | `regimenes.datos.catalogo (alias de regimenes.rutas.CATALOG)` |
| `MIN_FRAC_VS_CACHE` | constante | `regimenes.datos.descarga` |
| `MIN_OBS` | constante | `regimenes.datos.descarga` |
| `DERIVADAS` | constante | `regimenes.datos.descarga` |

#### `src/ingest/sources.py`

| nombre | tipo | modulo destino |
|---|---|---|
| `fred_key` | def/clase | `regimenes.datos.fuentes` |
| `fetch_fred` | def/clase | `regimenes.datos.fuentes` |
| `fetch_yahoo` | def/clase | `regimenes.datos.fuentes` |
| `fetch_ofr` | def/clase | `regimenes.datos.fuentes` |
| `fetch_github_csv` | def/clase | `regimenes.datos.fuentes` |
| `fetch_fred_spread` | def/clase | `regimenes.datos.fuentes` |
| `fetch_academico` | def/clase | `regimenes.datos.fuentes` |
| `fetch_stooq` | def/clase | `regimenes.datos.fuentes` |
| `es_ken_french` | def/clase | `regimenes.datos.fuentes` |
| `fetch` | def/clase | `regimenes.datos.fuentes` |
| `ROOT` | constante | `regimenes.datos.fuentes (alias de regimenes.rutas.ROOT)` |
| `_UA` | constante | `regimenes.datos.fuentes` |
| `SHILLER_XLS_URLS` | constante | `regimenes.datos.fuentes` |
| `MISSING_FRENCH` | constante | `regimenes.datos.fuentes` |

#### `src/viz.py`

| nombre | tipo | modulo destino |
|---|---|---|
| `canonical_detector` | def/clase | `regimenes.viz.figuras` |
| `detector_short` | def/clase | `regimenes.viz.figuras` |
| `use_house_style` | def/clase | `regimenes.viz.figuras` |
| `regime_color` | def/clase | `regimenes.viz.figuras` |
| `shade_regime` | def/clase | `regimenes.viz.figuras` |
| `episode_durations` | def/clase | `regimenes.viz.figuras` |
| `plot_price_by_regime` | def/clase | `regimenes.viz.figuras` |
| `plot_regime_timeline` | def/clase | `regimenes.viz.figuras` |
| `plot_crisis_probability` | def/clase | `regimenes.viz.figuras` |
| `plot_duration_histogram` | def/clase | `regimenes.viz.figuras` |
| `plot_transition_matrix` | def/clase | `regimenes.viz.figuras` |
| `plot_metric_comparison` | def/clase | `regimenes.viz.figuras` |
| `plot_distribution_by_regime` | def/clase | `regimenes.viz.figuras` |
| `plot_feature_space_scatter` | def/clase | `regimenes.viz.figuras` |
| `plot_regime_correlation_heatmaps` | def/clase | `regimenes.viz.figuras` |
| `plot_fold_panel` | def/clase | `regimenes.viz.figuras` |
| `render_table_figure` | def/clase | `regimenes.viz.figuras` |
| `plot_grouped_bars` | def/clase | `regimenes.viz.figuras` |
| `SHORT` | constante | `regimenes.viz.figuras` |
| `_K_SUFFIX` | constante | `regimenes.viz.figuras` |
| `C_LONG` | constante | `regimenes.viz.figuras` |
| `C_SHORT` | constante | `regimenes.viz.figuras` |
| `C_NEG` | constante | `regimenes.viz.figuras` |
| `C_NA` | constante | `regimenes.viz.figuras` |
| `C_CRISIS` | constante | `regimenes.viz.figuras` |
| `C_FP` | constante | `regimenes.viz.figuras` |
| `REGIME_COLORS` | constante | `regimenes.viz.figuras` |
| `HOUSE_RC` | constante | `regimenes.viz.figuras` |

#### `capa1_exploracion/detectors/_hmm_t_utils.py`

| nombre | tipo | modulo destino |
|---|---|---|
| `t_log_emission` | def/clase | `regimenes.detectores.f3_hmm._hmm_t_utils` |
| `StudentTHMM` | def/clase | `regimenes.detectores.f3_hmm._hmm_t_utils` |
| `filtered_posterior_t` | def/clase | `regimenes.detectores.f3_hmm._hmm_t_utils` |
| `fit_student_t_hmm` | def/clase | `regimenes.detectores.f3_hmm._hmm_t_utils` |
| `_TINY` | constante | `regimenes.detectores.f3_hmm._hmm_t_utils` |

#### `capa1_exploracion/detectors/_hmm_utils.py`

| nombre | tipo | modulo destino |
|---|---|---|
| `filtered_posterior` | def/clase | `regimenes.detectores.f3_hmm._hmm_utils` |
| `_TINY` | constante | `regimenes.detectores.f3_hmm._hmm_utils` |

#### `capa1_exploracion/detectors/changepoint_online.py`

| nombre | tipo | modulo destino |
|---|---|---|
| `ChangepointOnline` | def/clase | `regimenes.detectores.f6_changepoint.changepoint_online` |

#### `capa1_exploracion/detectors/clustering_gmm.py`

| nombre | tipo | modulo destino |
|---|---|---|
| `ClusteringGMM` | def/clase | `regimenes.detectores.f2_clustering.clustering_gmm` |

#### `capa1_exploracion/detectors/deep_ae_regime.py`

| nombre | tipo | modulo destino |
|---|---|---|
| `DeepAERegime` | def/clase | `regimenes.detectores.f7_deep.deep_ae_regime` |
| `PCAGMMBaseline` | def/clase | `regimenes.detectores.f7_deep.deep_ae_regime` |

#### `capa1_exploracion/detectors/garch_t_vol.py`

| nombre | tipo | modulo destino |
|---|---|---|
| `GarchTVol` | def/clase | `regimenes.detectores.f5_garch.garch_t_vol` |

#### `capa1_exploracion/detectors/hmm_gaussian_2s.py`

| nombre | tipo | modulo destino |
|---|---|---|
| `HMMGaussian2S` | def/clase | `regimenes.detectores.f3_hmm.hmm_gaussian_2s` |
| `BRIDGE_FEATURES` | constante | `regimenes.detectores.f3_hmm.hmm_gaussian_2s` |

#### `capa1_exploracion/detectors/hmm_tstudent.py`

| nombre | tipo | modulo destino |
|---|---|---|
| `HMMTStudent` | def/clase | `regimenes.detectores.f3_hmm.hmm_tstudent` |

#### `capa1_exploracion/detectors/hsmm_tstudent.py`

| nombre | tipo | modulo destino |
|---|---|---|
| `HSMMTStudent` | def/clase | `regimenes.detectores.f3_hmm.hsmm_tstudent` |
| `_TINY` | constante | `regimenes.detectores.f3_hmm.hsmm_tstudent` |

#### `capa1_exploracion/detectors/jump_model.py`

| nombre | tipo | modulo destino |
|---|---|---|
| `JumpModel` | def/clase | `regimenes.detectores.f2_clustering.jump_model` |

#### `capa1_exploracion/detectors/markov_switching_var.py`

| nombre | tipo | modulo destino |
|---|---|---|
| `MarkovSwitchingVar` | def/clase | `regimenes.detectores.f4_switching.markov_switching_var` |
| `_TINY` | constante | `regimenes.detectores.f4_switching.markov_switching_var` |

#### `capa1_exploracion/detectors/msgarch_regime.py`

| nombre | tipo | modulo destino |
|---|---|---|
| `msgarch_filter` | def/clase | `regimenes.detectores.f5_garch.msgarch_regime` |
| `MSGarchRegime` | def/clase | `regimenes.detectores.f5_garch.msgarch_regime` |
| `_TINY` | constante | `regimenes.detectores.f5_garch.msgarch_regime` |
| `_PERSIST_CAP` | constante | `regimenes.detectores.f5_garch.msgarch_regime` |

#### `capa1_exploracion/detectors/rule_composite_riskoff.py`

| nombre | tipo | modulo destino |
|---|---|---|
| `RuleCompositeRiskoff` | def/clase | `regimenes.detectores.f1_reglas.rule_composite_riskoff` |

#### `capa1_exploracion/detectors/rule_vix_threshold.py`

| nombre | tipo | modulo destino |
|---|---|---|
| `RuleVixThreshold` | def/clase | `regimenes.detectores.f1_reglas.rule_vix_threshold` |

#### `capa1_exploracion/detectors/turbulence_mahalanobis.py`

| nombre | tipo | modulo destino |
|---|---|---|
| `TurbulenceMahalanobis` | def/clase | `regimenes.detectores.f1_reglas.turbulence_mahalanobis` |

Total defs publicas mapeadas: 149.

## 7. Verificación de cierre de fase (integrador)

1. `grep -rnE "from src|import src|sys\.path|capa1_exploracion|from detectors" src/regimenes tests` → vacío
   (salvo menciones en docstrings/registro justificadas).
2. `src/` solo contiene `regimenes/` (y `regimenes.egg-info/`, ignorado).
3. `python -m pytest` verde; recuento = 158 − tests eliminados con fila en el registro.
4. Cada una de las 149 defs de §6 importable desde su módulo destino
   (script contra `docs/revisiones/baseline/api_inventory.json`).
5. `python -m regimenes.benchmark --help` y `python -m regimenes.datos --offline` (sin red).
6. Benchmark completo (fase posterior, no en esta): regenerar `results/benchmark/` y comparar
   contra `docs/revisiones/baseline/results_hashes.json` / `panels_hashes.json` — los
   números deben coincidir; solo pueden cambiar huellas, rutas y marcas de tiempo.
