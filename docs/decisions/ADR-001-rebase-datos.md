# ADR-001 — Re-base de la capa de datos antes de comparar detectores

- **Estado:** Aceptada · 2026-07-18
- **Rama:** trabajo directo sobre `main`; el estado previo queda en `backup-main-pre-datos-v2` (y tag `capa1-exploracion`).
- **Ámbito:** afecta a la estructura del repo, la capa de datos y el protocolo de comparación. **No** toca el marco de evaluación (`src/evaluation.py`) ni la interfaz `RegimeDetector`.

---

## 1. Contexto — la historia hasta aquí

Este repo es la **Capa 1** de un TFM mayor (MIAX): un banco de pruebas cuyo objetivo
**no es un detector**, sino el **marco de evaluación causal y comparable** que juzga a
muchos detectores de régimen de mercado.

En la primera vuelta se implementaron **12 detectores** (7 familias, D1–D12) bajo una
interfaz común y un protocolo walk-forward. El trabajo fue riguroso y dejó hallazgos
metodológicos sólidos que **siguen siendo válidos** (ver `capa1_exploracion/`):

- El *look-ahead* de los z-scores in-sample compraba **suavidad, no acierto**.
- La t-Student mejora el ajuste con holgura (**ΔBIC ≈ +10963** vs. gaussiano, mismas features).
- El Viterbi-por-bloque introducía *switching* artificial en las fronteras de bloque.
- **2013 (taper tantrum) es punto ciego universal**: 6 detectores independientes lo fallan
  → *la taxonomía de features importa* (todas son de vol/equity; falta tipos/crédito).
- La complejidad extra (MS-GARCH, deep AE) **no se paga** con ~4 crisis → parsimonia validada.

## 2. El problema que nos hizo parar

Al intentar poner los 12 detectores en **un único ranking**, se ve que **no son comparables
1:1**. Cada detector se construye *su propia* matriz de features y *su propia* ventana:

| Grupo | Detectores | OOS empieza | n_oos | Crisis vistas OOS |
|---|---|---|---:|:---:|
| Ventana larga | D11, D5, D6, D7, D1, D10 | 1991–1998 | 6987–8782 | **4** |
| Ventana media | D4, D8 | 2012 | 3405 | 2 |
| Ventana corta | D2, D3, D9, D12 | 2015 | 2649 | 2 |

Unos se juzgan con **3,3× más datos y el doble de crisis** que otros, y con **subconjuntos
de features distintos** (1, 4, 7 o 15). El repo lo maneja honestamente ("comparar por grupo
de ventana, nunca mezclar"), pero eso es una **tirita**: *los datos no son una variable
controlada, son parte del detector.*

**Diagnóstico raíz:** la FASE 1 (datos) se cerró deprisa y con carencias estructurales:

1. **Datos comprometidos**: FRED fue inaccesible en el entorno original → crédito (HY OAS)
   omitido, curva por proxy, DXY por *fallback*. La ventana común la **gobernaba HYG (2007)**,
   no una decisión de diseño. Solo 9 series → 15 features, **casi todas de vol/equity**.
2. **~4 crisis** → cero potencia estadística (sin intervalos de confianza entre eventos).
3. **Sin dataset-benchmark fijo**: no existe *un* conjunto de datos congelado sobre el que competir.

## 3. Decisión

**No se tira nada.** Los 12 detectores **se congelan** como Capa 1 de *exploración* (cumplieron
su papel: mapear familias + dejar las lecciones metodológicas). Y **se re-basa la capa de datos**
antes de volver a comparar. En concreto:

1. **Reorganizar el repo** para que cuente la historia: `capa1_exploracion/` (congelado) +
   estructura v2 en la raíz. → *esta entrega (FASE 1)*.
2. **Adquisición de datos máxima y reproducible**, todo gratis: FRED (con API key), yfinance,
   Stooq, datasets de Kaggle y repos de GitHub, históricos académicos (Shiller, Ken French) e
   **índices de estrés ya hechos** (OFR FSI, NFCI, STLFSI) como validación externa.
3. **Dos pistas de datos en paralelo** (no se puede todo a la vez; definiciones canónicas en
   [`docs/GLOSARIO.md`](../GLOSARIO.md), cifras finales congeladas en `data/benchmark_spec.yaml`):
   - **Pista A — Espina histórica profunda**: pocas features (S&P 500 + vol + factores + crédito/macro
     profundos); banco **desde 1927** (lo gobierna el S&P 500 diario; crédito/macro mensual desde
     1913-1919) → **22 crisis**, ataca el n≈4.
   - **Pista B — Panel rico multi-activo**: crédito, curva, breakevens, vol, sectores, FX; banco
     **desde 2003** (breakevens TIPS; sub-banco vol-of-vol 2007) → 10 crisis, ataca el punto ciego de 2013.
4. **Benchmark fijo por pista**: cada pista tiene su ventana + pool de features **congelado** →
   *un leaderboard justo 1:1 dentro de cada pista*; entre pistas, comparación cualitativa declarada.
   Así desaparece el "3,3× datos".

## 4. Qué NO cambia (deliberadamente)

- **El juez**: `src/evaluation.py` (walk-forward + métricas) y `src/detector_base.py` (interfaz
  `RegimeDetector`) se **reutilizan idénticos**. Todo el cambio es *aguas arriba* (datos), no en
  la evaluación.
- **La causalidad**: toda feature sigue siendo z-score *expanding/rolling* (nunca de muestra
  completa) y se verifica con `assert_causal`.

## 5. Consecuencias

- **Ganamos**: comparabilidad real, potencia estadística (más crisis en Pista A), y un ataque
  directo al punto ciego de 2013 (features de tipos/crédito en Pista B).
- **Coste**: duplicación intencionada del framework (una copia congelada en `capa1_exploracion/src/`,
  otra viva en `src/`). Es honesto: Capa 1 es una foto; v2 evoluciona.
- **Fuera de alcance de esta vuelta**: re-ejecutar los 12 sobre el dataset congelado (FASE D) y
  diseñar "el" detector. Eso es la siguiente macro-sesión, ya con datos sólidos debajo.

## 6. Hoja de ruta (fases con revisión agéntica)

| Fase | Qué | Estado |
|---|---|---|
| 1 | Reorganización + narrativa | ✅ |
| 2 | Datos: estado del arte + descarga máxima (166/174 series) | ✅ |
| 3 | EDA + análisis profundo (con ola de re-verificación) + benchmark congelado | ✅ |
| — | Limpieza intensa + documentación | ✅ |
| D | Re-evaluar los 12 sobre `benchmark_spec.yaml` + diseñar el detector | 🔜 |

*(La Fase 4 "preprocesado" se pospone y se hará junto con la Fase D, ya con el benchmark congelado.)*

## 7. Nota de limpieza (2026-07-18)

Tras la Fase 3 se hizo una **limpieza intensa** del repo para dejar solo lo relevante:
- **Notebooks planos y ordenados**: `notebooks/00_descarga.ipynb`, `notebooks/01_eda.ipynb`
  (sin subcarpetas, sin scripts `_build_*.py` — el notebook ejecutado y autónomo es el entregable;
  la *lógica* vive en `src/`).
- **Retirada la maquinaria de builders**: de v2 y de `capa1_exploracion/scripts/` (Capa 1 está
  congelada; sus notebooks ya están ejecutados). El histórico íntegro sigue en la rama
  `backup-main-pre-datos-v2` y el tag `capa1-exploracion`.
- **Traza temporal fuera**: `data/_catalog_research/` (consolidado en `docs/SOTA_datos.md`).
- **`docs/` = hogar único del conocimiento** (ver `docs/README.md`): decisiones, datos, EDA, teoría.
  Ningún hallazgo, decisión ni teoría se perdió: se preservó en `docs/` antes de borrar.
- **Capa 1 se mantiene intacta** (decisión del usuario) salvo la retirada de `scripts/`.
