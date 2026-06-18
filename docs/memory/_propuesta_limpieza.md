# Propuesta de limpieza del repo (Subagente 2 — Auditor de estructura)

> **NATURALEZA DE ESTE DOCUMENTO:** es una PROPUESTA. No se ha borrado, movido ni
> renombrado nada. Todo lo de abajo lo decide y ejecuta el orquestador (humano).
> Ante la duda → "revisar", nunca "borrar". Es un TFM: borrar es irreversible.

Fecha auditoría: 2026-06-18 · Rama: `main` · Working tree con 1 modificado + 4 sin trackear.

---

## 0. Inventario rápido (qué hay y para qué sirve)

| Carpeta / fichero | Contenido | Función |
|---|---|---|
| `README.md` | Descripción del banco de pruebas comparativo | Doc principal. **Esencial.** |
| `requirements.txt` | Dependencias | **Esencial.** |
| `.gitignore` | Reglas de ignore (plantilla Python + bloque del proyecto) | **Esencial** (ver §4). |
| `src/` | `data_loader.py`, `features.py`, `detector_base.py`, `evaluation.py`, `viz.py` | Librería núcleo (datos causales + marco de evaluación walk-forward). **Esencial.** |
| `detectors/` | 12 detectores `DNN_*.py` + `_hmm_utils.py`, `_hmm_t_utils.py` + `.gitkeep` | Implementaciones `RegimeDetector`. Los dos `_hmm*` con guion bajo **NO son basura**: son módulos compartidos importados por D4/D8. **Esencial.** |
| `notebooks/` | `00..13 *.ipynb` (14 ejecutados) + 10 `_build_*.py` + 3 `_rerun/_verify_*.py` | Notebooks de cada detector + scripts auxiliares (ver §1, §2). |
| `results/` | 13 `metrics_*.csv` + `metrics_master*.csv` + ~40 `*.png` | Salidas. CSV versionados; PNG ignorados (§4). **Esencial los CSV.** |
| `data/raw/` | `raw_panel.parquet`, `yfinance_raw.parquet`, `coverage_report.csv`, `provenance.json` + `.gitkeep` | Datos crudos regenerables. Solo `.gitkeep` trackeado (ignorados por `data/raw/*`). |
| `data/processed/` | `features.parquet` + `.gitkeep` | Features regenerables. Solo `.gitkeep` trackeado. |
| `docs/context/` | `RESUMEN_DETECCION_REGIMENES.md`, `TFM_Proposal_v2.pdf` | Contexto del TFM. **Esencial.** |
| `docs/memory/` | `00_state_of_the_art.md`, `01_data_and_eda.md`, `99_conclusions.md`, `INDEX.md` | Memoria del proyecto. **Esencial.** |
| `docs/memory/detectors/` | `01..12 *.md` | Ficha de cada detector. **Esencial.** |
| `docs/memory/sota/` | 7 pares `*.md` + `*.bib` | Estado del arte + bibliografía. **Esencial.** |
| `docs/references.bib` | Bibliografía global | **Esencial.** |

---

## 1. CANDIDATOS A BORRAR

Solo regenerables o cache. **Nada de contenido.** Recordatorio: las 3 carpetas
`__pycache__/` ya están cubiertas por `.gitignore` y **no están trackeadas**
(`git ls-files` no las lista), así que ni siquiera ensucian el repo en git; borrarlas
es puramente cosmético del disco.

| Ruta | Tipo | Motivo | Riesgo |
|---|---|---|---|
| `src/__pycache__/` | cache Python | Regenerable, no trackeado, ya en `.gitignore` | **seguro** |
| `detectors/__pycache__/` | cache Python | Regenerable, no trackeado, ya en `.gitignore` | **seguro** |
| `notebooks/__pycache__/` | cache Python | Regenerable, no trackeado, ya en `.gitignore` | **seguro** |

No hay `.ipynb_checkpoints/`, `.pytest_cache/`, `*.bak`, `*~`, ni carpetas
`_t1_backup`/`_t12_backup`/`/tmp/t1_backup` en disco (esas rutas solo se mencionan
dentro de scripts históricos; los directorios ya no existen). **Nada más se propone borrar.**

Comando (opcional, cosmético) que el orquestador podría ejecutar:
```bash
# (NO ejecutado por el auditor) limpieza de bytecode regenerable
find . -type d -name '__pycache__' -not -path './.git/*' -prune -exec rm -rf {} +
```

---

## 2. CANDIDATOS A MOVER

Los scripts auxiliares sueltos en `notebooks/` mezclan "generadores" con los
notebooks finales. Propuesta: crear `scripts/` con dos subcarpetas y dejar
`notebooks/` **solo con los 14 `.ipynb` ejecutados**.

> **CAVEAT TÉCNICO IMPORTANTE (leer antes de mover):** todos estos scripts calculan
> la raíz con `ROOT = Path(__file__).resolve().parents[1]`, asumiendo que están en
> `notebooks/` (un nivel bajo la raíz). Si se mueven a `scripts/builders/` o
> `scripts/verify/` quedan **dos** niveles bajo la raíz → habría que cambiar
> `parents[1]` por `parents[2]` en cada uno para que sigan funcionando. Como los
> notebooks ya están ejecutados y versionados, estos generadores son **artefactos
> históricos / reproducibilidad** y NO necesitan volver a correr para el TFM, por lo
> que el path roto tras el move es tolerable; aun así, lo honesto es ajustar el índice.

### 2a. Generadores de notebooks (`_build_*.py`) → `scripts/builders/`

Valor: **histórico / reproducible**. Cada uno construyó (con `nbformat`) y ejecutó su
notebook homónimo, que ya está versionado. (Nota: no existen builders trackeados para
`00`, `01`, `03`, `04` — esos notebooks se generaron por otra vía o su builder se descartó.)

```bash
mkdir -p scripts/builders
git mv notebooks/_build_02.py scripts/builders/_build_02.py
git mv notebooks/_build_05.py scripts/builders/_build_05.py
git mv notebooks/_build_06.py scripts/builders/_build_06.py
git mv notebooks/_build_07.py scripts/builders/_build_07.py
git mv notebooks/_build_08.py scripts/builders/_build_08.py
git mv notebooks/_build_09.py scripts/builders/_build_09.py
git mv notebooks/_build_10.py scripts/builders/_build_10.py
git mv notebooks/_build_11.py scripts/builders/_build_11.py
git mv notebooks/_build_12.py scripts/builders/_build_12.py
git mv notebooks/_build_13.py scripts/builders/_build_13.py   # (aún sin trackear: usar 'mv', no 'git mv')
```

### 2b. Verificación / re-ejecución puntual (`_rerun_*`, `_verify_*`) → `scripts/verify/`

One-offs de no-regresión usados durante el desarrollo. `_rerun_tanda1.py` apunta a
`/tmp/t1_backup` y `_verify_arreglo4.py` a `_t12_backup/` (dirs ya inexistentes), así que
**ya no son re-ejecutables** tal cual: su valor es puramente histórico/documental.
Alternativa razonable: muchos los pondrían en "revisar para BORRAR" — aquí, por prudencia
de TFM, se proponen **mover**, no borrar.

```bash
mkdir -p scripts/verify
git mv notebooks/_rerun_tanda1.py    scripts/verify/_rerun_tanda1.py
git mv notebooks/_verify_arreglo4.py scripts/verify/_verify_arreglo4.py
git mv notebooks/_verify_tarea_a.py  scripts/verify/_verify_tarea_a.py
```

| Origen | Destino propuesto | Motivo |
|---|---|---|
| `notebooks/_build_02..13.py` (10) | `scripts/builders/` | Separar generadores de los `.ipynb` finales |
| `notebooks/_rerun_tanda1.py` | `scripts/verify/` | One-off de re-ejecución histórico |
| `notebooks/_verify_arreglo4.py` | `scripts/verify/` | One-off de no-regresión histórico |
| `notebooks/_verify_tarea_a.py` | `scripts/verify/` | One-off de no-regresión histórico |

---

## 3. DEJAR INTACTO (esencial)

- **Los 14 notebooks ejecutados** `notebooks/00_eda.ipynb … 13_comparison.ipynb`. Son el
  entregable visible del TFM.
- **`src/`** completo (núcleo: datos causales + evaluación walk-forward).
- **`detectors/`** completo, **incluidos `_hmm_utils.py` y `_hmm_t_utils.py`** (módulos
  compartidos activos, importados por D4/D8 — el guion bajo NO indica que sobren) y `.gitkeep`.
- **`results/`**: todos los `metrics_*.csv` y `metrics_master*.csv` (resultados versionables);
  los `.png` se quedan en disco pero git los ignora (ok).
- **`docs/`** entero: context, memory, memory/detectors, memory/sota, `references.bib`, `*.bib`.
- **`data/`**: `.gitkeep` de `raw/` y `processed/`. Los `.parquet`/`.csv`/`.json` crudos se
  quedan en disco como cache local; git los ignora (regenerables con `src/data_loader.py`).
- **`README.md`, `requirements.txt`, `.gitignore`**.

---

## 4. GITIGNORE / RUTAS ABSOLUTAS

### .gitignore — cobertura
- **Cubre bien lo regenerable:** `__pycache__/`, `.ipynb_checkpoints`, `.pytest_cache/`,
  `data/raw/*`, `data/processed/*` (con excepción `!*.gitkeep`), `results/*.png`,
  `results/*.parquet`, `.env`, `venv/`. Coherente con que `git ls-files` solo trackee
  `.gitkeep` en `data/` y ningún `.png/.parquet` en `results/`.
- **BUG cosmético:** la **primera línea** del `.gitignore` es `2# Byte-compiled / optimized / DLL files`
  — hay un `2` colado al principio del comentario (probable typo de pegado). No rompe nada
  (sigue siendo comentario), pero conviene corregir a `# Byte-compiled / optimized / DLL files`.
- **Para revisar (whitelisting opcional):** `data/raw/provenance.json` y
  `data/raw/coverage_report.csv` son **metadatos de procedencia/cobertura de datos**, no datos
  pesados. Hoy quedan ignorados por `data/raw/*`. Para un TFM puede interesar versionar la
  procedencia (reproducibilidad documental). Sugerencia (decide el orquestador):
  ```gitignore
  !data/raw/provenance.json
  !data/raw/coverage_report.csv
  ```

### Sin trackear que SÍ deberían versionarse (commit pendiente, no borrar)
`git status` muestra trabajo nuevo de la Fase 4 todavía no añadido:
- `M  docs/memory/INDEX.md`
- `?? docs/memory/99_conclusions.md`
- `?? notebooks/13_comparison.ipynb`
- `?? notebooks/_build_13.py`  (si se adopta §2a, va a `scripts/builders/`)
- `?? results/metrics_master_final.csv`

Son entregables/resultados → **commitear**, no son candidatos a borrar.

### Rutas absolutas hardcodeadas — CÓDIGO LIMPIO
- **`src/` y `detectors/`:** 0 rutas absolutas. Todo relativo.
- **Celdas de CÓDIGO de los `.ipynb`:** 0 rutas absolutas. Usan el patrón correcto
  `ROOT = Path.cwd(); if ... ROOT = ROOT.parent`. **Verificado: ningún incumplimiento.**
- **Único hallazgo real en código:** `notebooks/_rerun_tanda1.py` línea 13 →
  `BACKUP = Path("/tmp/t1_backup")` (ruta absoluta POSIX). Es un script histórico no
  re-ejecutable (el dir no existe); se va a `scripts/verify/` en §2b. Riesgo: nulo.
- **Cosmético (no es violación de código):** las salidas (output cells) de
  `01_`, `03_`, `05_`, `13_*.ipynb` contienen `C:\Users\oscar\Downloads\PRUEBAS_...\src\detector_base.py`
  dentro de mensajes `UserWarning` impresos. **No están en celdas de código**, solo en
  outputs guardados; filtran el nombre de usuario. Se regeneran limpios al re-ejecutar, pero
  como NO se debe re-ejecutar en esta fase, se dejan tal cual (anotado para conocimiento).

---

## 5. ESTRUCTURA FINAL PROPUESTA (árbol)

```
PRUEBAS_DETECCION_REGIMENES_DE_MERCADO_TFM/
├── README.md
├── requirements.txt
├── .gitignore                      (corregir typo línea 1; valorar whitelist provenance)
├── data/
│   ├── raw/.gitkeep                (parquet/json crudos en disco, ignorados)
│   └── processed/.gitkeep          (features.parquet en disco, ignorado)
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── detector_base.py
│   ├── evaluation.py
│   └── viz.py
├── detectors/
│   ├── .gitkeep
│   ├── _hmm_utils.py               (módulo compartido — NO mover/borrar)
│   ├── _hmm_t_utils.py             (módulo compartido — NO mover/borrar)
│   ├── rule_vix_threshold.py … msgarch_regime.py   (12 detectores)
├── notebooks/                      (SOLO .ipynb ejecutados)
│   ├── 00_eda.ipynb
│   ├── 01_… … 12_…
│   └── 13_comparison.ipynb
├── scripts/                        (NUEVO — antes sueltos en notebooks/)
│   ├── builders/                   (_build_02..13.py — ajustar parents[1]→parents[2])
│   └── verify/                     (_rerun_tanda1.py, _verify_arreglo4.py, _verify_tarea_a.py)
├── results/                        (metrics_*.csv versionados; *.png en disco, ignorados)
│   ├── metrics_01..12_*.csv
│   ├── metrics_master.csv
│   └── metrics_master_final.csv    (commitear)
└── docs/
    ├── context/  (RESUMEN_*.md, TFM_Proposal_v2.pdf)
    ├── references.bib
    └── memory/
        ├── INDEX.md, 00_state_of_the_art.md, 01_data_and_eda.md, 99_conclusions.md
        ├── detectors/ (01..12 *.md)
        └── sota/ (01..07 *.md + *.bib)
```

(`__pycache__/` desaparece del árbol: regenerable, no versionado.)
