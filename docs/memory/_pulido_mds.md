# Informe de pulido de documentación (.md) — Subagente 1 (Revisor de documentación)

> Fase de pulido final del TFM de detección de regímenes. Este informe NO toca
> resultados ni números: documenta el pulido de REDACCIÓN/FORMATO y SEÑALA las
> discrepancias de cifras para decisión del orquestador. Fuente de verdad de
> cifras: `results/metrics_master_final.csv` (final) y `results/metrics_master.csv`
> (previa); además se cotejó cada ficha contra su CSV individual
> `results/metrics_NN_*.csv`.

Fecha: 2026-06-18.

---

## 1. Alcance revisado

Se leyeron y cotejaron TODOS los `.md` del proyecto:

- `README.md` (raíz)
- `docs/memory/INDEX.md`
- `docs/memory/00_state_of_the_art.md`
- `docs/memory/01_data_and_eda.md`
- `docs/memory/99_conclusions.md`
- `docs/memory/detectors/01..12_*.md` (12 fichas de detector)
- `docs/memory/sota/01..07_*.md` (7 fichas de familia)
- `docs/context/RESUMEN_DETECCION_REGIMENES.md`

No se encontraron otros `.md` fuera de `node_modules` (no hay `node_modules`).
Este informe (`_pulido_mds.md`) y los demás informes de la fase de pulido NO se
tocan entre sí.

Verificaciones globales realizadas:
- **Cifras del master**: todas las cifras principales de `99_conclusions.md` e
  `INDEX.md` (medias de cobertura GFC+COVID, BIC, ΔBIC, switching, duraciones,
  lead/lag, far, estrés agregado) se recomputaron desde
  `metrics_master_final.csv` y **cuadran exactamente**. La documentación de
  síntesis es fiable.
- **Enlaces a figuras** `results/*.png`: todas las figuras referenciadas en las
  fichas y conclusiones EXISTEN (verificado con glob). Sin enlaces rotos.
- **Notebooks** `notebooks/00..13`: los 14 existen. Sin enlaces rotos.
- **CSVs** `results/metrics_*`: todos existen. Sin enlaces rotos.
- **Codificación**: sin artefactos de mojibake. Las tildes y símbolos (canónico,
  ν, σ, ≈, →, ×, −) se renderizan bien en UTF-8 en todos los documentos.
- **Terminología**: nombres de detector D1..D12 y familias F1..F7 consistentes
  entre todos los documentos. "msvar" se usa de forma uniforme (no hay variantes
  "msVAR"/"MsVar"). No aparece la errata "cusup" en ningún `.md` (sí en un
  notebook; ver §5).

---

## 2. Ficheros tocados (correcciones seguras aplicadas)

### `docs/memory/sota/01_reglas_umbrales.md`
- **Errata corregida** (línea ~183, sección "Aplicaciones documentadas"):
  `umbral VI> nivel` → `umbral VIX > nivel`. Errata de transcripción inequívoca
  ("VI>" por "VIX >"); no afecta a ningún número ni conclusión.

No se aplicó ninguna otra edición: el resto del pulido se limita a SEÑALAR
discrepancias (ver §3), por las reglas de la fase (no alterar cifras salvo errata
de transcripción 100% verificable; las de §3 son valores OBSOLETOS, no erratas de
transcripción, y su prosa de interpretación depende del valor → decisión del
orquestador).

---

## 3. DISCREPANCIAS DE CIFRAS (NO corregidas — para decisión del orquestador)

Las tres discrepancias siguientes son valores **obsoletos** en una ficha `.md`
(quedaron de una versión anterior del pipeline). En los tres casos, **el CSV
individual del detector Y ambos masters coinciden** en el valor correcto, así que
la ficha es la única desalineada. NO se han cambiado porque (a) no son erratas de
transcripción sino valores stale, y (b) en el caso del lead/lag de D3 la prosa
contigua ("cruza muy por delante del suelo") interpreta el valor y reescribirla
sería alterar contenido.

| # | Fichero:línea aprox. | Afirmación | Valor en el doc | Valor correcto (master + CSV individual) | Recomendación |
|---|---|---|---|---|---|
| 1 | `docs/memory/detectors/03_clustering_gmm.md` : 68 | Lead/lag D3 COVID vs suelo del drawdown | **−160 d** | **−20 d** (`metrics_03_clustering_gmm.csv` y `metrics_master_final.csv`: `leadlag_COVID_2020 = -20.0`) | Actualizar a −20 d. El propio `INDEX.md` (FASE 3, Arreglo 3) documenta la corrección "−160→−20 d, correcto": la ficha quedó pre-Arreglo-3. Al hacerlo, suavizar la frase "cruza muy por delante del suelo" (a −20 d ya no es "muy por delante"). |
| 2 | `docs/memory/detectors/03_clustering_gmm.md` : 68 | Lead/lag D3 Inflación vs suelo del drawdown | **−229 d** | **−219 d** (`leadlag_Inflation_2022 = -219.0` en CSV individual y master) | Actualizar a −219 d. Mismo origen stale que #1. |
| 3 | `docs/memory/detectors/06_garch_t_vol.md` : 97 | `label_stability` de D6 | **0.982** | **0.999** (`metrics_06_garch_t_vol.csv` y master: `label_stability = 0.99939…`) | Actualizar a 0.999 (≈1.0). No altera ninguna interpretación; la frase es puramente factual. |

Notas de cotejo (todo lo demás cuadra):
- D1, D2, D4, D5, D7, D8, D9, D10, D11, D12: todas las cifras de sus fichas
  coinciden con el master y su CSV individual (cobertura por crisis, fa_2013/2018,
  switching, duración, far, BIC/AIC/logL, lead/lag donde se citan).
- D6: salvo la #3, el resto cuadra (cov_GFC 100%, cov_COVID 94%, fa_2013 11.3%,
  fa_2018 87.3%, switching 0.0141, dur ≈70 d, logL/AIC/BIC = −13285.6/26583.1/26626.6).
- D3: salvo #1 y #2, el resto cuadra (cov_COVID 0.96, cov_Infl 0.87, switching
  0.126, dur 7.9 d, BIC 63016, far 0.49, fa_2018 0.00).
- Observación menor (NO discrepancia dura): `09_jump_model.md` (línea ~47)
  describe `label_stability ~1.0`; el master da 0.983 para `jump_model`. Es una
  aproximación cualitativa ("~1.0") algo generosa, no un número afirmado; opcional
  precisar a "≈0.98".

---

## 4. ENLACES ROTOS

Ninguno. Todas las rutas relativas a figuras (`results/*.png`), notebooks
(`notebooks/*.ipynb`), CSVs (`results/metrics_*.csv`) y `.md` internos (INDEX →
sota/detectores/fases) apuntan a ficheros que existen.

---

## 5. PROBLEMAS DE CODIFICACIÓN / FORMATO

- **Codificación**: sin incidencias. No hay mojibake ni comillas tipográficas
  rotas; UTF-8 consistente.
- **Formato**: tablas markdown bien alineadas; encabezados coherentes; bloques de
  código con o sin lenguaje pero correctos. No se detectaron viñetas
  inconsistentes ni dobles espacios significativos en prosa.
- **Fuera de alcance (.ipynb), se SEÑALA**: la errata conocida "cusup" → "cusum"
  está en un **notebook**, no en un `.md`:
  `notebooks/13_comparison.ipynb` (celda ~línea 1760, cadena de `print`:
  `'...D7 cusup y reglas son persistentes'`). No se ha tocado por estar fuera del
  alcance `.md` y por la regla de no editar notebooks; se recomienda corregirla a
  "cusum" en una pasada separada (es solo una cadena de impresión, no afecta a
  resultados). En todos los `.md` el término correcto "cusum" ya se usa de forma
  consistente.

---

## 6. Veredicto final

**La documentación queda esencialmente lista para entrega.** No hay bloqueantes:
los enlaces, la codificación, el formato y la terminología están correctos, y la
síntesis (`99_conclusions.md`, `INDEX.md`) es numéricamente fiel al master.

Quedan **3 cifras obsoletas en 2 fichas de detector** (§3, D3 lead/lag ×2 y D6
label_stability), todas menores y todas con el valor correcto verificado en el
master + CSV individual. NO se corrigieron por las reglas de esta fase (son
valores stale, no erratas de transcripción, y uno arrastra prosa interpretativa).
Se recomienda que el orquestador aplique las tres actualizaciones de §3 (y,
opcionalmente, la nota de D9 y el "cusup" del notebook de §5) antes de la entrega
final. Ninguna de ellas cambia conclusiones ni veredictos del TFM.
