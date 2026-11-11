# ADR-012 — Estructura del repositorio para todas las fases del TFM

- **Estado:** Aceptada · 2026-11-11 (decisión del usuario).
- **Completa:** [ADR-005](ADR-005-reencuadre-tfm-multiagente.md) §3 y [ADR-007](ADR-007-fase2-ingesta-y-docs.md):
  fija dónde viven las fases 3–5, que seguían abiertas.
- **Sustituye en parte:** la instalación con `requirements.txt` de [ADR-004](ADR-004-unificacion.md) y los
  `requirements/ingesta-<tema>.txt` de [ADR-007](ADR-007-fase2-ingesta-y-docs.md) §2.1 (§2 de este ADR).
- **Numeración:** ADR-006 sigue reservada a la decisión final del detector de régimen (notebook `21_decision_final`).

---

## 1. Contexto

1. **El repositorio había crecido por capas.** Tenía 2.458 ficheros versionados: 1.338 eran CSV de un solo trabajo
   de las ablaciones. Además había seis ficheros de dependencias, dos documentos sueltos en la raíz, notebooks de
   Fase 2 con dos convenciones de nombre y una línea de sintéticos cerrada («el aumento no se adopta») que seguía
   en `main`.
2. **La huella de caché del benchmark depende de las rutas.** `regimenes.benchmark.cache` hashea el código de
   `regimenes/detectores` y `regimenes/evaluacion`, `configs/benchmark_spec.yaml` y los paneles de `data/`, cada uno
   con su **ruta relativa** como clave. Mover o renombrar cualquiera de ellos obliga a re-ejecutar el benchmark
   (horas). Las huellas del pool (`detectores/pool`, `benchmark/pool.py`, `features/rentabilidades.py`,
   `configs/bloques.yaml`) y de las ablaciones (`benchmark/ablaciones.py`, `features/bloques.py`, `bloques.yaml`)
   funcionan igual.
3. **`get_regimen` no funcionaba en un clon limpio.** Lee `results/benchmark/panels/{A_D06,B_D05}.parquet`, que
   estaban ignorados.
4. **Faltaba un contrato estable** entre la Fase 1 y los agentes de las fases 3–5.

## 2. Decisión

### 2.1 Árbol

```
regime-aware-macro-equity/
├── README.md · pyproject.toml · constraints.txt · requirements.txt (alias) · Makefile · LICENSE
├── configs/        Fase 1 intacta (benchmark_spec, bloques, catalog, regimen); lo nuevo en configs/<paquete>/
├── data/           fuera de git (raw, processed)
├── src/
│   ├── regimenes/      Fase 1 · régimen → get_regimen (CONGELADO por la huella)
│   ├── ingesta/        Fase 2 · un subpaquete por tema
│   ├── herramientas/   contrato de tools tipado para los agentes (Fase 2)
│   ├── memoria/        Fase 2 · RAG con TimeGate (se crea al empezar)
│   ├── agentes/        Fases 3–4 · grafo LangGraph (se crea al empezar)
│   └── cartera/        Fases 4–5 · motores, límites, backtest (se crea al empezar)
├── notebooks/      fase1/NN_*.ipynb · fase2/<tema>/NN_*.ipynb · faseN/ al llegar
├── results/        salidas pequeñas versionadas; binarios regenerables ignorados
├── tests/          espejo de src/
└── docs/           índice, ADR, estado del arte, una carpeta por fase
```

**Reglas:**
- Un solo `pyproject.toml` y varios paquetes de primer nivel. No se crea un paquete paraguas, porque obligaría a
  mover `regimenes`.
- Cada carpeta nueva se crea **en el primer PR que la necesite**, nunca vacía.
- Las dependencias entre capas van en un solo sentido: `regimenes`, `ingesta` → `memoria` → `herramientas` →
  `agentes`. `cartera` es numérico puro.
- **Las fases 3–5 importan solo de `herramientas`**, nunca de `regimenes.*` ni de `ingesta.*`.

### 2.2 Dependencias

- Todas en `pyproject.toml`, con un extra por uso:
  - `deep`, `jump`, `changepoint` y `dev`.
  - Uno por tema de la Fase 2: `ingesta-mercados`, `ingesta-bancos` e `ingesta-noticias`. El Tema 3 no necesita
    dependencias extra.
  - `ingesta`, que reúne los tres anteriores.
  - `todo`, que lo reúne todo.
- El candado de versiones pasa a ser `constraints.txt` en la raíz. Antes era
  `requirements/ingesta-mercados_series.constraints.txt`.
- Instalación: `pip install -e ".[todo]" -c constraints.txt`. El candado fija las 8 librerías de la huella; sin él,
  una actualización invalida la caché.
- `requirements.txt` se queda como alias de una línea, porque un mensaje de error de `benchmark/ejecucion.py`
  (función en la huella AST) lo cita por su nombre.
- Fuera: `requests-cache` y `sdmx1` (no los importaba nadie); `xlrd` y `openpyxl` repetidos (ya están en el núcleo).

### 2.3 Compactación de resultados

- **Ablaciones.** Los 1.338 CSV de trabajo se consolidan en `results/detectores/ablaciones/trabajos.csv`, que
  escribe `ablaciones_lectura.escribir_tablas()`.
  - `ablaciones_lectura.cargar()` lee los CSV sueltos si existen (ejecución local) y, si no, el consolidado.
  - Los CSV sueltos quedan en local, ignorados: son la marca de «trabajo hecho» de `ablaciones.py`, que no se toca.
  - Verificado: desde el consolidado, las tablas derivadas (`criterio_b`, `sin_bloque`, `media`, `validez_atajo`,
    `medianas_azar`) salen idénticas byte a byte.
- **Paneles de `get_regimen`.** `A_D06.parquet` y `B_D05.parquet` se versionan como excepción. El resto de paneles
  sigue ignorado y se regenera con `python -m regimenes.benchmark`.
- **Fichas de ingesta.** Se quedan como un JSON por fuente. Es la API común de los cuatro temas
  (`ingesta.comun.fichas`) y cambiarla obligaría a tocar los tests y módulos de cada tema por poca ganancia.

### 2.4 Archivo de la línea de sintéticos

- Va a la etiqueta **`sinteticos-final`** (commit `10b65d2`, último `main` con la línea entera):
  - `regimenes.sinteticos`, `tests/sinteticos`, `configs/sinteticos.yaml`;
  - los notebooks `17`–`20`;
  - `results/sinteticos`.
- Se archiva porque el veredicto es «el aumento no se adopta» y `get_regimen` no la usa.
- Para consultarla o re-ejecutarla: `git switch --detach sinteticos-final`, e instalar con
  `pip install -r requirements.txt`, como entonces.
- Solo dejan de valer sus propias cachés (aumento y laboratorio), que no se van a reanudar. Las del benchmark, el
  pool y las ablaciones salen idénticas.
- Los notebooks `21` y `22` (esqueletos) ya no importan `regimenes.sinteticos`.
- **No se renumeran:** queda el hueco 17–20 para no desfasar los comentarios de los ficheros con huella.

### 2.5 Otros movimientos

| Antes | Ahora |
|---|---|
| `notebooks/fase2/t2_01_sonda_bancos_riesgo.ipynb` | `notebooks/fase2/bancos_riesgo/01_sonda.ipynb` |
| `notebooks/fase2/t3_01_sonda_empresas.ipynb` | `notebooks/fase2/empresas/01_sonda.ipynb` |
| `notebooks/fase2/t4_01_sonda_noticias_alt.ipynb` | `notebooks/fase2/noticias_alt/01_sonda.ipynb` |
| `notebooks/fase2/t4_02_validez_alternativos.ipynb` | `notebooks/fase2/noticias_alt/02_validez_alternativos.ipynb` |
| `notebooks/NN_*.ipynb` (Fase 1, carpeta plana) | `notebooks/fase1/NN_*.ipynb` |
| `PendientesOscar.md` (raíz) | `docs/tfm/PendientesOscar.md` |
| `PlanPendientesOscar.pdf` (raíz) | `docs/tfm/internos/` (solo local) |
| `requirements/ingesta-*.txt` | extras de `pyproject.toml` (§2.2) |
| distribución `regimenes` 0.4.0 | distribución `regime-aware-macro-equity` 0.5.0 (los paquetes no cambian de nombre) |
| repositorio `PRUEBAS_DETECCION_REGIMENES_DE_MERCADO_TFM` | `regime-aware-macro-equity` (renombrado en GitHub) |

### 2.6 Contrato de tools (`src/herramientas`)

- **`contrato.py`:** `Cita`, `EntradaTool` y `SalidaTool` (pydantic, sin campos extra).
- **`registro.py`:** `registrar`, `catalogo`, `json_schema` e `invocar`. La **fecha de la decisión la pone el
  sistema**: `registrar` rechaza un modelo de entrada con campo `fecha`, y el esquema que ve el LLM no la incluye.
- **`regimen.py`:** `get_regimen` tipado sobre `regimenes.regimen.get_regimen`, con sus citas (panel y fusiones,
  con sha256).
- **Pendientes:** `get_serie` y `get_precios` (Tema 1), y las tools de búsqueda cuando exista `memoria`.

## 3. Excepciones conocidas (no se corrigen a propósito)

Estos textos citan nombres antiguos, pero están en ficheros que entran en una huella o son pre-registros congelados.
Corregirlos invalidaría una caché o reescribiría un pre-registro.

- **`src/regimenes/benchmark/ejecucion.py:208`:** «pip install -r requirements.txt». Por eso se conserva el alias.
- **Comentarios con la numeración de ADR-004 en ficheros con huella** («notebook 23», «22_decision_final»…):
  `evaluacion/metricas.py`, `detectores/__init__.py` y `detectores/pool/__init__.py`.
- **`configs/bloques.yaml` y `features/bloques.py`:** citan `PendientesOscar.md` solo por su nombre, sin ruta.
- **`configs/benchmark_spec.yaml` y `configs/bloques.yaml`:** sus comentarios citan `notebooks/NN_*.ipynb` sin la
  carpeta `fase1/` (la Fase 1 se movió a `notebooks/fase1/` después; los notebooks no están en ninguna huella).
- **ADR-004, ADR-005 y ADR-007 a ADR-011:** usan la numeración de notebooks de su fecha. Equivalencias: notebook
  «21» de ADR-009–011 = aumento (hoy archivado, antes 20); «17» = fusión online (hoy 16); «22»/«23» de ADR-004–005
  = decisión final y pseudolive (hoy 21 y 22).

## 4. Consecuencias

- **El repo baja de 2.458 a 932 ficheros versionados**, sin re-ejecutar nada: las cachés vigentes del
  benchmark, el pool y las ablaciones siguen igual.
- **`get_regimen` y su test real** corren en un clon limpio y en la CI.
- **Cada persona del equipo tiene que reinstalar una vez**, porque la distribución cambia de nombre. Los pasos están
  en el README raíz, «Instalación».
- **Comprobaciones que tiene que pasar cada PR de reorganización:**
  - `pytest` en verde, incluidos `tests/test_docs_enlaces.py` y `tests/benchmark/test_cache_versionada_vigente.py`;
  - `git diff main -- src/regimenes/detectores src/regimenes/evaluacion configs/benchmark_spec.yaml` vacío.
