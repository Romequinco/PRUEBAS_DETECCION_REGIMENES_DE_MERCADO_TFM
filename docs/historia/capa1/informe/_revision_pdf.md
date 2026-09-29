# Revisión del PDF `report/informe_capa1.pdf` — Dictamen del Agente 5 (Revisor / Tribunal)

Documento revisado: `report/informe_capa1.pdf` (12 páginas).
Fuente de verdad numérica: `results/metrics_master_final.csv`.
Fuente del texto/figuras: `report/informe_capa1.tex`, `report/references.bib`, `results/pdf_*.png`, `results/eda_*.png`.
Fecha de revisión: 2026-06-18.

---

## (a) Verificación de números (PDF vs CSV, verbatim)

Truth = `results/metrics_master_final.csv`. Todas las cifras del PDF están redondeadas de forma consistente con el CSV.

| # | Cifra (ubicación PDF) | Valor PDF | Valor CSV | ¿Coincide? |
|---|------------------------|-----------|-----------|-----------|
| 1 | ΔBIC D4−D8 (Tabla 7, Hallazgo c) | 35379.4 − 24415.9 = **10963.5** | 35379.41 − 24415.89 = 10963.52 | ✅ |
| 2 | D8 BIC (Tabla 2 / 7) | 24 416 / 24 415.9 | 24415.886 | ✅ |
| 3 | D4 BIC (Tabla 2 / 7) | 35 379 / 35 379.4 | 35379.408 | ✅ |
| 4 | D8 COVID estricta (Tabla 3/4, Sec 7) | 0.660 → 0.66 | 0.66 | ✅ |
| 5 | D8 COVID estrés (Tabla 4, Sec 7) | 0.960 → 0.96 | cov_estres_COVID 0.96 | ✅ |
| 6 | D8 Selloff-2018 estricta (Tabla 5, Sec 5.2) | 0.034 | fa_Selloff 0.033898 | ✅ |
| 7 | D8 Selloff-2018 estrés (Tabla 5, Sec 5.2) | 0.814 (texto "0.81") | fa_estres_Selloff 0.81356 | ✅ |
| 8 | D8 Inflación estricta→estrés (Sec 7) | 0.33 → 0.90 | 0.33173 → 0.90385 | ✅ |
| 9 | D8 FA global estricta→estrés (Sec 5.2) | 0.52 → 0.79 | 0.51887 → 0.79407 | ✅ |
| 10 | D7 switching (Tabla 2, podio) | 0.0022 / 0.002 | 0.0021744 | ✅ |
| 11 | D7 duración (Tabla 2) | 435.68 | 435.684 | ✅ |
| 12 | D12 switching (Tabla 2, Hallazgo e) | 0.2873 / 0.287 | 0.2872782 | ✅ |
| 13 | D6 cov GFC 2008 (Tabla 3) | 1.000 | 1.0 | ✅ |
| 14 | D11 cov GFC 2008 (Tabla 3, Hallazgo e) | 0.000 / 0 % | 0.0 | ✅ |
| 15 | D6 FA global / BIC (Tabla 2) | 0.845 / 26 627 | 0.84514 / 26626.56 | ✅ |
| 16 | D5 BIC / switching (Tabla 2) | 28 024 / 0.0557 | 28023.77 / 0.055690 | ✅ |
| 17 | D8 ν por estado (Hallazgo c) | [10.2, 7.6, 4.2, 2.4] | no está en el CSV (parámetro interno) | — (no verificable; plausible) |
| 18 | D4 2013 / 2018 (Hallazgo a) | 25 % / 46 % | fa_Taper 0.25 / fa_Selloff 0.45763 | ✅ |
| 19 | D3 trampa 2018 estricta→estrés (Tabla 5) | 0.000 → 0.729 | 0.0 → 0.72881 | ✅ |
| 20 | Cobertura GFC+COVID podio (Tabla 6) | D5 0.98 > D6 0.97 ≈ D1/D7 0.92 | medias (0.993+0.96)/2=0.98; (1.0+0.94)/2=0.97; (0.938+0.90)/2=0.92; (1.0+0.84)/2=0.92 | ✅ |

**Muestreadas además y correctas**: D6/D5/D7/D1/D10/D11 fila completa de Tabla 3 (cov estricta); Tabla 7 completa (logL/AIC/BIC de D8/D6/D11/D5/D4/D3); lead/lag de Fig. leadlag (D7 −252/−252/−204/−252; D5 −252/−43/−159/−220; D6 −252/−42/−158/−217) todos verbatim del CSV.

**Discrepancias numéricas detectadas: 0.** El único valor no verificable contra el CSV (ν por estado de D8, ítem 17) es un parámetro interno del modelo, no una métrica del master; es coherente con la narrativa (colas decrecientes calma→crisis) y no contradice ninguna cifra.

---

## (b) Verificación de las 4 citas centrales (contra `references.bib` y lista [1]–[18] del PDF)

| Cita central | Clave bib | Atribución en texto | Datos bibliográficos | Veredicto |
|---|---|---|---|---|
| Hamilton 1989 | `hamilton1989` | D5 Markov-Switching ([6] en PDF) | Econometrica **57(2):357–384** | ✅ real y bien atribuida |
| Kritzman–Page–Turkington 2012 | `kritzman2012` | D10 turbulencia Mahalanobis ([9] en PDF) | FAJ **68(3):22–39** | ✅ real; atribución **defendible** (ver nota) |
| Nystrup et al. 2020 (jump model) | `hmm_nystrup2020` | D9 jump model ([11] en PDF) | ESWA **150:113307** | ✅ real y bien atribuida |
| Haas–Mittnik–Paolella 2004 (MS-GARCH) | `vol_haasmittnikpaolella2004` | D11 MS-GARCH ([5] en PDF) | JFE **2(4):493–530** | ✅ real y bien atribuida |

**Nota sobre Kritzman (turbulencia):** el índice de turbulencia de Mahalanobis se asocia canónicamente a **Kritzman & Li (2010)**, "Skulls, Financial Turbulence, and Risk Management" (FAJ). El PDF cita Kritzman–Page–Turkington (2012) "Regime Shifts", que también desarrolla y aplica la medida de turbulencia/Mahalanobis para regímenes. La atribución es por tanto **defendible** (no es errónea), aunque la fuente primaria más precisa de la *medida* sería la de 2010. No bloqueante; opcional añadir `kritzman2010` como cita secundaria si se quiere blindar ante un tribunal estricto.

**Citas/refs rotas:** NINGUNA. El texto extraído no contiene `[?]` ni `??`. Las 18 entradas citadas resuelven a la bibliografía y todas las `\cite`/`\ref` del `.tex` están definidas.

---

## (c) Figura ↔ texto (una línea por figura clave embebida)

- **`pdf_rank_heatmap.png` (Fig. 8):** ✅ separa cobertura por ventana — "GFC 08 (solo v. larga)" con `n/a` para los de ventana corta; "COVID 20 (común a 12)"; filas agrupadas por bloque; footnote marca lead/lag censurado ±252. NO mezcla larga y corta. Coherente con el pie.
- **`pdf_sensibilidad_especificidad.png` (Fig. 4):** ✅ eje Y = "cobertura COVID-2020 (OOS común a los 12)"; eje X = especificidad 1−media(FA 2013,2018); color/forma por grupo de ventana; D11/D12 marcados negativos. La sensibilidad común es COVID-2020, como exige el texto.
- **`pdf_persistencia_sensibilidad.png` (Fig. 5):** ✅ sensibilidad = COVID-2020 común; D7 cusum a la derecha (≈436 d, persistente), D3/D12 a la izquierda (flicker); escala log. Coherente con el pie.
- **`pdf_estres_vs_estricta.png` (Fig. 7):** ✅ D8 0.66→0.96 (COVID) y 0.33→0.90 (Inflación); D3 0.96/0.96 y 0.87/0.87; D12 0.54→0.96 y 0.10→0.79. Todas las barras coinciden con CSV/Tablas 4–5.
- **`pdf_bic.png` (Fig. 6):** ✅ D8 24416 (menor) → D3 63016 (mayor); D11 en gris (negativo); aviso "solo comparable sobre las MISMAS features (D4 vs D8)". Coherente.
- **`pdf_leadlag.png`:** la figura existe en `results/` y es correcta (asteriscos `*` marcan censura en ±252; D7 toca −252 en 3 de 4 eventos, coincide con Sec. 8), **pero NO está embebida en el PDF**. El lead/lag se trata solo en texto (Sec. 8) y en el footnote de la Fig. 8. No hay contradicción figura↔texto; es una decisión de maquetación, no un fallo.

---

## (d) Checklist de matizaciones del pulido

| Matización exigida | Estado | Evidencia |
|---|---|---|
| "consistente con / no contradice", NUNCA "confirma" sobre D8 | ✅ PRESENTE | Resumen: "de forma consistente con (no 'confirmando')"; Sec. 7: "es consistente… y no lo contradice… Se evita deliberadamente el verbo 'confirma'" |
| Limitación n≈4 crisis, sin tests de significancia | ✅ PRESENTE | Sec. 8 "Significancia con n ≈ 4 crisis… No hay tests de significancia ni intervalos de confianza" |
| logL/AIC/BIC marcados in-sample | ✅ PRESENTE | Sec. 8 "logL/AIC/BIC son in-sample por definición"; Tabla 6 "Ajuste BIC (in-sample)"; Sec. 7 "eje in-sample (BIC)" |
| Lead/lag censurado al lookback (252) | ✅ PRESENTE | Sec. 8 "se busca en los 252 días previos… censura por la derecha… límite inferior"; footnote Fig. 8 |
| Cobertura SIEMPRE separada por ventana | ✅ PRESENTE | Tablas 2/3 con bloques larga/corta; Fig. 8 separa GFC vs COVID; Sec. 5.1 "La separación por ventana es obligatoria" |
| D11/D12 marcados exploratorio-negativos | ✅ PRESENTE | Tabla 1, Hallazgo (e), Sec. 7 pto 4, todas las figuras los marcan "(neg.)/(-)" |

Las 6 matizaciones están presentes.

---

## (e) Veredicto de autocontención

**APROBADO.** Un tribunal que no ha visto el repo puede seguir el documento de principio a fin:
- **Qué se hizo**: marco de evaluación causal y comparativo, no un detector (Resumen + Sec. 1–2).
- **Datos**: universo multiactivo sin imputar, ventana común, 15 features causales, hechos estilizados (Sec. 3.1).
- **Método causal**: z-score expandido, walk-forward OOS sin look-ahead, canonicalización vol-primaria por fold, métricas definidas (Sec. 3.2).
- **Los 12 detectores**: Tabla 1 + descripción de cada uno con familia y rol (Sec. 4).
- **Resultados**: tablas maestras, cobertura estricta/estrés, especificidad, planos de trade-off, podio (Sec. 5).
- **Hallazgos** (el producto central), **recomendación**, **limitaciones** y **nota de reproducibilidad** (Sec. 6–8).

Sin saltos lógicos graves ni términos sin definir relevantes ("trampa", "estrés agregado", "censura", "vol-primaria" se definen al introducirse).

---

## (f) Lista priorizada de fallos a corregir

**BLOQUEANTES: NINGUNO.**

**COSMÉTICOS / opcionales (no impiden la entrega):**

1. **[Cosmético] Frase deslavazada en Sec. 3.2** (`informe_capa1.tex` línea 177): "Tres principios hacen que la comparación sea legítima: La metodología causal sigue las recomendaciones de \cite{lopezdeprado2018}." — la oposición "principios :" + oración suelta + subtítulo "Features causales." queda algo abrupta. Reubicar la cita de López de Prado como pie introductorio o integrarla en el primer principio mejoraría la lectura. No afecta a la corrección.
2. **[Cosmético] Atribución de turbulencia** (Sec. 4, D10; `.tex` línea 259): considerar añadir `kritzman2010` (Kritzman & Li, "Skulls…") como cita primaria de la medida de Mahalanobis, junto a `kritzman2012`. La actual es defendible; esto solo la blinda.
3. **[Cosmético/opcional] `pdf_leadlag.png` no embebida**: existe y es correcta. Si se desea reforzar visualmente la censura ±252 (mencionada solo en texto), podría incluirse como figura; alternativamente, dejarlo como está es legítimo. Decisión editorial, no fallo.

---

## (g) Veredicto global

**EL PDF ESTÁ A NIVEL DE ENTREGA DE TFM.** ✅

- **Discrepancias numéricas: 0** (todas las cifras muestreadas coinciden verbatim con `metrics_master_final.csv`, incluidas las 7 obligatorias del encargo).
- **Citas: 0 rotas, 0 mal atribuidas.** Las 4 centrales son reales y correctamente referenciadas; la de turbulencia (Kritzman 2012) es defendible.
- **Figuras: 5/5 embebidas coherentes con su pie**; la cobertura se separa por ventana en heatmap y tablas; sensibilidad común = COVID-2020.
- **Matizaciones del pulido: 6/6 presentes.**
- **Autocontenido: sí.**
- **Fallos: ningún bloqueante; 3 cosméticos opcionales.**

El documento puede defenderse ante tribunal sin correcciones obligatorias.
