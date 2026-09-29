# ADR-004 — Unificación de la Capa 1 y v2 en un solo repositorio y un solo paquete

- **Estado:** Aceptada · 2026-09-29 (aprobada por el usuario)
- **Rama:** `unificacion` (fase 0: foto de referencia; fase 1: paquete `src/regimenes`; fase 2:
  notebooks por familia + documentación).
- **Revierte:** [ADR-001](ADR-001-rebase-datos.md) §7, último punto (*"Capa 1 se mantiene intacta"*),
  y el coste asumido en ADR-001 §5 (*"duplicación intencionada del framework"*). **Decisión explícita
  del usuario, 2026-09-29.** El resto de ADR-001 (re-base de datos, dos pistas), ADR-002 y ADR-003
  siguen vigentes sin cambios.
- **Respaldo:** tags `capa1-final` (Capa 1 congelada completa) y `v2-pre-unificacion` (código v2
  previo); foto de referencia en [`../revisiones/baseline/`](../revisiones/baseline/) (inventario de
  149 definiciones públicas, 158 tests, huellas de paneles y resultados).
- **Contrato de ejecución:** [`../revisiones/unificacion_contrato.md`](../revisiones/unificacion_contrato.md)
  (mapa módulo antiguo → nuevo). Eliminaciones: [`../revisiones/registro_eliminaciones.md`](../revisiones/registro_eliminaciones.md).

---

## 1. Contexto

ADR-001 congeló la Capa 1 (`capa1_exploracion/`) como una foto y construyó v2 en la raíz, aceptando
una duplicación del marco. Tres meses después esa separación ya no era real:

1. **v2 dependía de la Capa 1 en tiempo de ejecución.** `src/detectors/registry.py` anteponía
   `capa1_exploracion` a `sys.path` (`LEGACY_ROOT`) para instanciar los 12 detectores congelados;
   `src/benchmark.py` necesitaba `_clean_sys_path_for_workers()` y un guard de importación para que
   los procesos hijos no resolvieran el `src/` equivocado; la huella de caché hasheaba **dos** copias
   idénticas de `detector_base.py`.
2. **Dos árboles, un solo significado.** La teoría de las familias, las fichas por detector y las
   conclusiones vivían en `capa1_exploracion/memory/`, mientras los resultados vigentes (ADR-003)
   vivían en `results/benchmark_v2/` y en los notebooks 04–07. Para explicar un detector había que
   leer tres sitios.
3. **Notebooks por fase, no por familia.** `04_benchmark_detectores` ejecutaba los 12 detectores a la
   vez y `05_comparacion_detectores` los comparaba, pero ningún notebook contaba *una familia*: qué
   detecta, qué supone, cómo se comporta en cada crisis y qué cambió respecto a v1.
4. **Lo que viene necesita un paquete.** La fase de datos sintéticos (generadores, validación,
   laboratorio, aumento) y el pseudolive van a importar detectores, evaluación y features desde otros
   notebooks y scripts; con `sys.path` y dos `src/` eso no escala.

## 2. Decisión

Se unifica todo en **un repositorio plano y un paquete instalable**, `regimenes`
(`pip install -e .`), y la Capa 1 pasa a ser **historia documentada**, no un segundo árbol de código.

1. **Código:** `src/` + `capa1_exploracion/{src,detectors}` → `src/regimenes/` (mover, no
   reescribir: solo cambian imports y rutas). Una sola interfaz `RegimeDetector`, cero `sys.path`,
   cero referencias a `capa1_exploracion` en el paquete, rutas centralizadas en `regimenes.rutas`.
2. **Configuración y resultados:** `data/{catalog,benchmark_spec}.yaml` → `configs/`;
   `results/benchmark_v2` → `results/benchmark`; `results/fusion_*` → `results/fusion/*`.
3. **Notebooks:** carpeta plana `notebooks/00–20`, **un notebook por familia** (05–11) con plantilla
   común, más protocolo (04), comparativa (12), fusión (13–14) y esqueletos de sintéticos,
   decisión final y pseudolive (15–20).
4. **Documentación:** la teoría pasa a `docs/teoria/`, las fichas por detector a `docs/detectores/`,
   los datos a `docs/datos/`, las revisiones a `docs/revisiones/` y el archivo de la Capa 1 a
   `docs/historia/capa1/`. Bibliografía **única** en `docs/references.bib`.
5. **Regla de no pérdida:** todo se mueve con `git mv`; lo que se elimina tiene fila en
   `registro_eliminaciones.md`; lo que cambia de notebook tiene fila en
   `trazabilidad_notebooks.md`; los notebooks y el código v1 originales quedan en el tag
   `capa1-final`.
6. **Resultados:** la unificación **no cambia números**. Cambian las huellas de caché (rutas de
   módulos), así que el benchmark completo se re-ejecuta una vez tras la fase 1 y se compara contra
   la foto de referencia (`results_hashes.json`, `panels_hashes.json`).

## 3. Estructura resultante

```
/
├── README.md · pyproject.toml · requirements.txt · Makefile · .env.example
├── configs/
│   ├── catalog.yaml              universo de datos (174 series declaradas + crisis_catalog)
│   ├── benchmark_spec.yaml       banco congelado por pista (ventanas, features, crisis, trampas)
│   └── sinteticos.yaml           configuración de generadores (esqueleto)
├── src/regimenes/                paquete único (pip install -e .)
│   ├── rutas.py                  rutas centralizadas (única fuente)
│   ├── datos/                    catalogo · descarga · fuentes · __main__ (python -m regimenes.datos)
│   ├── features/                 transformaciones · lags · paneles · causalidad
│   ├── detectores/               base (RegimeDetector) · registry
│   │   ├── f1_reglas/            D01 rule_vix_threshold · D02 rule_composite_riskoff · D10 turbulence_mahalanobis
│   │   ├── f2_clustering/        D03 clustering_gmm · D09 jump_model
│   │   ├── f3_hmm/               D04 hmm_gaussian_2s · D08 hmm_tstudent · D13 hsmm_tstudent (ablación) · utilidades
│   │   ├── f4_switching/         D05 markov_switching_var
│   │   ├── f5_garch/             D06 garch_t_vol · D11 msgarch_regime
│   │   ├── f6_changepoint/       D07 changepoint_online
│   │   └── f7_deep/              D12 deep_ae_regime
│   ├── evaluacion/               walk_forward · metricas · ranking (ADR-003)
│   ├── benchmark/                ejecucion · cache · cli · __main__ (python -m regimenes.benchmark)
│   ├── fusion/                   maquina (normal / vigilancia / confirmado)
│   ├── viz/                      figuras (estilo de casa)
│   ├── informes.py               utilidades comunes de los notebooks de familia 05–11 (carga verificada, tablas, figuras)
│   └── sinteticos/               base (Generador) · registry · validacion · parametricos/ · neuronales/
├── tests/                        por subpaquete: datos, features, detectores, evaluacion, benchmark, fusion, informes, sinteticos
├── notebooks/                    00–20, carpeta plana (tabla abajo)
├── data/                         raw/ · processed/ · sinteticos/ (gitignored salvo procedencia)
├── results/
│   ├── benchmark/                metrics/ · status/ · panels/ (gitignored) · manifest.json · ranking_v2.csv · metrics_master_v2.csv
│   ├── fusion/                   d07_d08/ · d02_d06/
│   ├── detectores/<fk_nombre>/   figuras de los notebooks de familia
│   ├── sinteticos/ · pseudolive/
└── docs/
    ├── README.md · GLOSARIO.md · references.bib (bibliografía única)
    ├── decisions/                ADR-000 … ADR-004
    ├── datos/                    SOTA_datos.md · EDA_v2.md · figs_eda/
    ├── teoria/                   README.md · 00_estado_del_arte.md · F1…F7_*.md + .bib por familia · F8_generadores_sinteticos.md (esqueleto, fase S)
    ├── detectores/               README.md · D01…D12_*.md (fichas por detector) · D13_hsmm_tstudent.md (ablación, fuera del ranking)
    ├── historia/capa1/           memoria, informe, resultados y datos v1 (+ README)
    ├── revisiones/               contrato, registro de eliminaciones, trazabilidad, baseline, REVISION_2026-09-29
    └── context/                  propuesta TFM + resumen de la tarea previa
```

### Notebooks

| Nº | Notebook | Contenido | Origen |
|---|---|---|---|
| 00 | `00_descarga` | panorámica de datos y descarga | se conserva |
| 01 | `01_eda` | EDA maestro | se conserva |
| 02 | `02_diseno_preprocesado` | decisiones del preprocesado | se conserva |
| 03 | `03_preprocesado` | paneles causales por pista + labels | se conserva |
| 04 | `04_protocolo_evaluacion` | walk-forward, gate/preflight, caché, CLI paralela | `git mv` de `04_benchmark_detectores` |
| 05 | `05_familia_F1_reglas` | D01, D02, D10 | nuevo (teoría v1 + resultados v2) |
| 06 | `06_familia_F2_clustering` | D03, D09 | nuevo |
| 07 | `07_familia_F3_hmm` | D04, D08, D13 (ablación) | nuevo |
| 08 | `08_familia_F4_markov_switching` | D05 | nuevo |
| 09 | `09_familia_F5_garch` | D06, D11 | nuevo |
| 10 | `10_familia_F6_changepoint` | D07 | nuevo |
| 11 | `11_familia_F7_deep` | D12 | nuevo |
| 12 | `12_comparativa` | scorecards y ranking por pista | `git mv` de `05_comparacion_detectores` |
| 13 | `13_fusion_d07_d08` | D7 alerta + D8 confirma | `git mv` de `06_fusion_d07_d08` |
| 14 | `14_fusion_d02_d06` | D2 alerta + D6 confirma | `git mv` de `07_fusion_d02_d06` |
| 15–18 | `15_sinteticos_generadores` · `16_sinteticos_validacion` · `17_sinteticos_laboratorio` · `18_sinteticos_aumento` | fase de datos sintéticos | esqueletos |
| 19 | `19_decision_final` | elección del sistema a congelar | esqueleto |
| 20 | `20_pseudolive` | prueba pseudolive independiente | esqueleto |

**Plantilla de notebook de familia (05–11):** título y resumen · teoría de la familia (de
`docs/teoria/`, fichas de detector y markdown de los notebooks v1) · configuración generada desde
`regimenes.detectores.registry` · ejecución con `EJECUTAR = False` por defecto (carga la caché de
`results/benchmark` y falla con el comando exacto si no está vigente) · por detector y pista: estados
OOS con franjas de crisis, diagnóstico propio (marcado ilustrativo no-causal si se ajusta sobre la
muestra completa), cobertura por crisis, métricas y puesto en el ranking ADR-003 · comparación
intra-familia · hallazgos v1 y qué cambia en v2 · fortalezas, limitaciones y conclusión. Toda cifra
del texto sale de una celda; figuras en `results/detectores/<fk_nombre>/` vía `regimenes.viz`.

## 4. Alternativas consideradas

| Alternativa | Por qué se descartó |
|---|---|
| **Mantener la congelación (ADR-001)** y seguir importando la Capa 1 por `sys.path` | La dependencia ya existía de hecho; mantenerla exige guardas de `sys.path`, doble hash y dos `src/` con el mismo nombre de paquete. Cada fase nueva (sintéticos, pseudolive) multiplicaría el problema. |
| **Copiar** los detectores a `src/` y dejar la Capa 1 intacta como archivo ejecutable | Dos copias del mismo código que divergen en silencio; la huella de caché no sabría cuál es la buena. |
| **Git submodule / repo aparte** para la Capa 1 | Añade fricción (dos repos, versiones cruzadas) para un código que ya es parte del sistema v2. |
| **Paquete sin notebooks por familia** (solo 04 benchmark + 05 comparativa) | Deja sin contar la teoría y el comportamiento de cada familia, que es el contenido académico del TFM; la teoría seguiría en un archivo congelado. |
| **Borrar la Capa 1** sin archivo | Pierde la memoria de decisiones, el informe v1 y los 5 hallazgos que motivaron v2. |

## 5. Consecuencias

- **Ganamos:** un solo `import regimenes` para notebooks, tests y scripts; rutas en un único módulo;
  una sola interfaz `RegimeDetector` y una sola huella de caché; cada familia contada en su notebook
  con la teoría v1 y los resultados v2 juntos; base limpia para sintéticos y pseudolive.
- **Coste:** todas las huellas de caché cambian (rutas de módulos) → `results/benchmark` queda
  marcado como caché obsoleta hasta una re-ejecución completa del benchmark (≈7,6 h de cómputo en
  serie según los `elapsed_seconds` de `ranking_v2.csv`, 6,9 h de ellas en la pista A; ≈2 h con 9
  procesos, acotado por D04-A ≈1,9 h); los números deben
  coincidir con la foto de referencia. Los enlaces y rutas de los documentos históricos
  (`docs/historia/capa1/`, `docs/revisiones/REVISION_2026-09-29.md`) no se reescriben: se traducen
  con [`../historia/capa1/README.md`](../historia/capa1/README.md) §6 y el glosario.
- **Reversibilidad:** total vía tags (`capa1-final`, `v2-pre-unificacion`) y la foto de referencia.
- **Fuera de alcance:** cambiar detectores, ventanas, crisis o el criterio de ranking (ADR-003); la
  elección del sistema final y el pseudolive (notebooks 19–20).
