# ADR-004 — Un solo repositorio y un solo paquete (`regimenes`)

- **Estado:** Aceptada · 2026-09-29 (decisión del usuario).
- **Revisión posterior (2026-10-06):** [ADR-005](ADR-005-reencuadre-tfm-multiagente.md) corrige §2:
  «todo el TFM vive en `regimenes`» pasa a ser «`regimenes` es el código de la **Fase 1** (régimen) del TFM».
  Estado actual de lo que §3 llamaba esqueletos: los notebooks 15–17 están **hechos** (`configs/sinteticos.yaml`
  y la validación F8 completos; subpaquetes de `regimenes.sinteticos`: `parametricos`, `neuronales`,
  `validacion` y `laboratorio`); el 18 se mantiene y los 19 (`decision_final`) y 20 (`pseudolive`) son el cierre
  de la Fase 1: la señal de régimen que consumirán los agentes, no «el sistema final del TFM».
- **Revierte:** [ADR-001](ADR-001-rebase-datos.md) §7, último punto (*"Capa 1 se mantiene intacta"*),
  y el coste asumido en ADR-001 §5 (*"duplicación intencionada del framework"*). El resto de ADR-001
  (re-base de datos, dos pistas), ADR-002 y ADR-003 siguen vigentes sin cambios.
- **Estado anterior recuperable:** tag `capa1-final` (Capa 1 completa: código, notebooks v1
  ejecutados, memoria) y tag `v2-pre-unificacion` (código v2 con la Capa 1 como árbol aparte).

---

## 1. Contexto

ADR-001 congeló la Capa 1 (los 12 detectores v1, en una carpeta propia) y construyó v2 en la raíz,
aceptando duplicar el marco de evaluación. Con el benchmark v2 terminado esa separación ya no era real:

1. **v2 dependía de la Capa 1 en tiempo de ejecución.** El registro de detectores anteponía la
   carpeta de la Capa 1 a `sys.path` para instanciar los 12 detectores congelados; el benchmark
   necesitaba guardas para que los procesos hijos no resolvieran el `src/` equivocado, y la huella de
   caché hasheaba dos copias idénticas de la interfaz `RegimeDetector`.
2. **Dos árboles, un solo significado.** La teoría de las familias, las fichas por detector y las
   conclusiones vivían en la memoria de la Capa 1, mientras los resultados vigentes (ADR-003) vivían en
   v2. Para explicar un detector había que leer tres sitios.
3. **Notebooks por fase, no por familia.** Un notebook ejecutaba los 12 detectores y otro los
   comparaba, pero ninguno contaba *una familia*: qué detecta, qué supone, cómo se comporta en cada
   crisis y qué cambió respecto a v1.
4. **Lo que viene necesita un paquete.** La fase de datos sintéticos (generadores, validación,
   laboratorio, aumento) y el pseudolive importan detectores, evaluación y features desde notebooks y
   scripts; con `sys.path` y dos `src/` eso no escala.

## 2. Decisión

Todo el TFM vive en **un repositorio plano y un paquete instalable**, `regimenes`
(`pip install -e .`). La Capa 1 pasa a ser **historia documentada** en
[`docs/historia/capa1/`](../historia/capa1/README.md), no un segundo árbol de código.

1. **Código:** un único paquete `src/regimenes/`. Los detectores de la Capa 1 se integran **sin
   reescribir su lógica** (solo imports y rutas), agrupados por familia. Una sola interfaz
   `RegimeDetector`, ningún acceso a `sys.path`, rutas centralizadas en `regimenes.rutas`.
2. **Configuración y resultados:** `configs/` (catálogo, banco congelado, sintéticos);
   `results/benchmark/`, `results/fusion/{d07_d08,d02_d06}/`, `results/detectores/<fk_nombre>/`.
3. **Notebooks:** carpeta plana `notebooks/00–20`, **un notebook por familia** (05–11) con plantilla
   común, más protocolo (04), comparativa (12), fusiones (13–14) y la fase de sintéticos, decisión
   final y pseudolive (15–20).
4. **Documentación:** teoría en `docs/teoria/`, fichas por detector en `docs/detectores/`, datos en
   `docs/datos/`, archivo de la Capa 1 en `docs/historia/capa1/`. Bibliografía **única** en
   `docs/references.bib`.
5. **Sin pérdida:** los ficheros se trasladan con su historia git; el código y los notebooks v1 que no
   pasan al paquete quedan íntegros en el tag `capa1-final`.
6. **Resultados:** la reorganización **no cambia números**. Cambian las huellas de caché (rutas de
   módulos), así que el benchmark completo se re-ejecuta una vez y debe reproducir las cifras de
   ADR-003 §4.

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
│   ├── informes.py               utilidades comunes de los notebooks de familia 05–11
│   └── sinteticos/               base (Generador) · registry · validacion · parametricos/ · neuronales/
├── tests/                        por subpaquete: datos, features, detectores, evaluacion, benchmark, fusion, informes, sinteticos
├── notebooks/                    00–20, carpeta plana (tabla abajo)
├── data/                         raw/ · processed/ · sinteticos/ (gitignored salvo procedencia)
├── results/
│   ├── benchmark/                metrics/ · status/ · panels/ (gitignored) · manifest.json · ranking_v2.csv · metrics_master_v2.csv
│   ├── fusion/                   d07_d08/ · d02_d06/
│   ├── detectores/<fk_nombre>/   figuras de los notebooks de familia
│   └── sinteticos/ · pseudolive/
└── docs/
    ├── README.md · GLOSARIO.md · references.bib (bibliografía única)
    ├── decisions/                ADR-001 … ADR-004
    ├── datos/                    SOTA_datos.md · EDA_v2.md · figs_eda/
    ├── teoria/                   README.md · 00_estado_del_arte.md · F1…F7_*.md + .bib por familia · F8_generadores_sinteticos.md (esqueleto)
    ├── detectores/               README.md · D01…D12_*.md · D13_hsmm_tstudent.md (ablación, fuera del ranking)
    ├── historia/capa1/           la Capa 1: decisiones, hallazgos, memoria, informe, métricas y datos v1
    └── context/                  propuesta TFM + resumen de la tarea previa
```

### Notebooks

| Nº | Notebook | Contenido |
|---|---|---|
| 00–03 | `00_descarga` · `01_eda` · `02_diseno_preprocesado` · `03_preprocesado` | datos, EDA y paneles causales por pista |
| 04 | `04_protocolo_evaluacion` | walk-forward, gate/preflight, caché, CLI paralela |
| 05–11 | `05_familia_F1_reglas` … `11_familia_F7_deep` | una familia cada uno (D01–D13) |
| 12 | `12_comparativa` | scorecards y ranking por pista |
| 13–14 | `13_fusion_d07_d08` · `14_fusion_d02_d06` | alerta + confirmación |
| 15–18 | `15_sinteticos_generadores` · `16_sinteticos_validacion` · `17_sinteticos_laboratorio` · `18_sinteticos_aumento` | fase de datos sintéticos (esqueletos) |
| 19–20 | `19_decision_final` · `20_pseudolive` | sistema a congelar y prueba pseudolive (esqueletos) |

**Plantilla de notebook de familia (05–11):** título y resumen · teoría de la familia (de
`docs/teoria/` y las fichas de detector) · configuración generada desde
`regimenes.detectores.registry` · ejecución con `EJECUTAR = False` por defecto (carga la caché de
`results/benchmark` y falla con el comando exacto si no está vigente) · por detector y pista: estados
OOS con franjas de crisis, diagnóstico propio (marcado ilustrativo no-causal si se ajusta sobre la
muestra completa), cobertura por crisis, métricas y puesto en el ranking ADR-003 · comparación
intra-familia · hallazgos de la Capa 1 y qué cambia en v2 · fortalezas, limitaciones y conclusión.
Toda cifra del texto sale de una celda; figuras en `results/detectores/<fk_nombre>/` vía
`regimenes.viz`.

## 4. Alternativas consideradas

| Alternativa | Por qué se descartó |
|---|---|
| **Mantener la congelación (ADR-001)** y seguir importando la Capa 1 por `sys.path` | La dependencia ya existía de hecho; mantenerla exige guardas de `sys.path`, doble hash y dos `src/` con el mismo nombre de paquete. Cada fase nueva (sintéticos, pseudolive) multiplicaría el problema. |
| **Copiar** los detectores a `src/` y dejar la Capa 1 intacta como archivo ejecutable | Dos copias del mismo código que divergen en silencio; la huella de caché no sabría cuál es la buena. |
| **Git submodule / repo aparte** para la Capa 1 | Añade fricción (dos repos, versiones cruzadas) para un código que ya es parte del sistema v2. |
| **Paquete sin notebooks por familia** (solo benchmark + comparativa) | Deja sin contar la teoría y el comportamiento de cada familia, que es el contenido académico del TFM. |
| **Borrar la Capa 1** sin archivo | Pierde la memoria de decisiones, el informe v1 y los 5 hallazgos que motivaron v2. |

## 5. Consecuencias

- **Ganamos:** un solo `import regimenes` para notebooks, tests y scripts; rutas en un único módulo;
  una sola interfaz `RegimeDetector` y una sola huella de caché; cada familia contada en su notebook
  con la teoría v1 y los resultados v2 juntos; base limpia para sintéticos y pseudolive.
- **Coste:** todas las huellas de caché cambian (rutas de módulos) y el benchmark completo se
  re-ejecuta una vez (≈7,4 h de cómputo en serie según los `elapsed_seconds` de `ranking_v2.csv`,
  ≈6,6 h de ellas en la pista A; ≈2 h con 9 procesos, acotado por D04-A ≈1,8 h). La re-ejecución
  (24/24, `results/benchmark/manifest.json` del 2026-09-29) reproduce exactamente las cifras de
  ADR-003 §4. Los documentos históricos de [`docs/historia/capa1/`](../historia/capa1/README.md)
  conservan sus rutas originales; su README §7 las traduce.
- **Reversibilidad:** total vía los tags `capa1-final` y `v2-pre-unificacion`.
- **Fuera de alcance:** cambiar detectores, ventanas, crisis o el criterio de ranking (ADR-003); la
  elección del sistema final y el pseudolive (notebooks 19–20).
