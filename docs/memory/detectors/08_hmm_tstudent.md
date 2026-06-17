# D8 — `hmm_tstudent` (FASE 3, Tanda 3) · Familia F3 (HMM avanzado)

> HMM con **emisiones t-Student multivariantes** (colas pesadas) y **K estados por
> BIC**. La mejora directa sobre el baseline gaussiano D4: ataca de frente las fat
> tails del EDA (kurtosis 25-40). Código: `detectors/hmm_tstudent.py` (+
> `detectors/_hmm_t_utils.py`, filtrado forward t) · Notebook:
> `notebooks/08_hmm_tstudent.ipynb` · Métricas: `results/metrics_08_hmm_tstudent.csv`.

## Implementado

**Modelo.** HMM con emisiones **t-Student multivariantes** (location mᵢ, matriz de
escala Sᵢ y grados de libertad **νᵢ** por estado), estimado por EM con variable de
escala latente (cada t-Student = mezcla gaussiana de escala continua). Se eligió la
t propia frente a GMM-HMM porque: (1) **BIC justo** — la t añade solo `k` parámetros
(un ν por estado) sobre el gaussiano, mientras un GMM multiplicaría medias/covarianzas
por nº de mezclas y dispararía el BIC; (2) robustez sin sobre-parametrizar.
Bibliografía: `hmm_bulla2011, hmm_rabiner1989, guidolintimmermann2007, hamilton1989`.

**Causalidad — filtrado forward t.** `predict_online`/`predict_proba` usan filtrado
forward causal con la **emisión t** (en `detectors/_hmm_t_utils.py`, adaptando el
patrón de D4 `_hmm_utils` cuya emisión era gaussiana), con contexto de burn-in del
train. Viterbi solo en la versión in-sample marcada NO causal. **No se modificó
`_hmm_utils.py` ni `src/`.**

**Features/ventana = D4 (BIC comparable).** Las mismas 7 features puente
(`BRIDGE_FEATURES`), ventana 2007+, `train_size=252*5` → como D4, 2008/2011 caen en el
train (NaN OOS). Así el BIC de D8 (t) es directamente comparable con el de D4
(gaussiano) sobre los mismos datos. market_returns = retorno log S&P 500.

**Coste.** El EM-t es caro (Baum-Welch + actualización de ν por fold). `step=126`
(refit ~semestral), declarado; el walk-forward tarda ~5 min. K seleccionado fuera del
walk-forward (fits in-sample para BIC).

## Descubierto

**K=4 por BIC (y AIC).** Grid {3,4}: BIC k=3 = 28103, **k=4 = 24416** → se despliega
**K=4** (calma · corrección leve · corrección · crisis). Los ν por estado canónico son
**decrecientes [10.2, 7.6, 4.2, 2.4]**: el estado de crisis tiene ν≈2.4 (colas muy
pesadas), el de calma ν≈10 (casi gaussiano). Económicamente impecable: la crisis es
donde viven los outliers.

**BIC vs D4 — la t-Student MEJORA el ajuste con holgura:**

| | n_states | logL | n_params | BIC |
|---|---:|---:|---:|---:|
| D4 gaussiano | 2 | −17381 | ~73 | **35379** |
| D8 t-Student | 4 | −11536 | 159 | **24416** |

**ΔBIC ≈ +10 963 a favor de D8** pese a tener más del doble de parámetros: el supuesto
gaussiano de D4 pagaba un coste enorme por no modelar las colas. Es la confirmación
cuantitativa de la crítica del EDA (kurtosis 25-40) y del estado del arte
(`hmm_bulla2011`).

**Orden de estados MONÓTONO en severidad** (no solo "no invertido"). Verificado
**en walk-forward** (con market_returns): vol anualizada por estado canónico OOS =
[10.1%, 13.8%, 21.0%, 37.1%] → estrictamente creciente 0→3; crisis (3) = mayor vol.
`monotonía vol walk-forward = True`, sin warning de fallback. Con K=4 el binning por
bandas de `VOL_CLOSE_FRAC` del Arreglo 4 ordena bien los 4 estados.

**Cobertura (OOS causal).** ventana 2012-2026 → 2008/2011 NaN (en train, como D4).
COVID_2020 = 66% en el estado CRISIS, Inflation_2022 = 33% en crisis. **Matiz
importante de multi-estado**: con K=4 el estado "crisis" es el más EXTREMO y estrecho;
el estrés más amplio cae en "corrección" (estado K−2). Sumando corrección+crisis,
Q4-2018 = 81%. Por eso `cov_COVID`/`cov_Inflation` de D8 parecen MÁS BAJAS que las de
D4 (2 estados, crisis ancha): **no es que D8 detecte peor, es que su "crisis" es la
cola extrema**. Hay que leer su fila del master con esta lente (la comparación justa de
"estrés" sería corrección+crisis).

**¿Captó 2013/2018?** Activación del estado crisis: 2013 = **0%**, 2018 = 3.4%
(corrección+crisis: 2013 = 0%, 2018 = 81%). Es decir, capta 2018 como corrección pero
**NO 2013**. La t-Student y los 4 estados mejoran el ajuste y separan corrección de
crisis, pero **2013 sigue siendo invisible**: el taper fue un shock de tipos sin
volatilidad de equity, y D8 usa solo features equity/crédito (las mismas que D4). Es
un punto ciego de las FEATURES, no del supuesto distribucional → lo debe tapar D10
(multivariante con tipos/dólar).

**Flickering.** switching 0.052, duración media 19.1 d, persistencia esperada por
estado [calma 31d, leve 18d, corrección 38d, crisis 83d]. Más persistente que el GMM
(D3, 0.126) y similar a D4 causal (0.100).

## Hipótesis del CHECKPOINT 2 para D8 — veredicto

> *"Emisiones t + más estados atacan fat tails y POTENCIALMENTE captan 2013/2018
> donde el gaussiano falla; riesgo de sobreajuste con pocas obs por estado."*

**Se cumple a medias, con un matiz nítido.** (1) Fat tails: **SÍ, rotundo** — ΔBIC
+10963 vs D4, ν decreciente hasta 2.4 en crisis. (2) Más estados: SÍ, separa
calma/corrección/crisis de forma monótona y económicamente coherente. (3)
**¿2013/2018? NO desbloquea 2013** (sigue invisible: es shock de tipos, no de vol
equity; límite de las features, no de la t); 2018 se capta como corrección. Conclusión:
la t-Student arregla el problema DISTRIBUCIONAL (colas) que tenía D4, pero el agujero
de 2013 es de COBERTURA DE FEATURES, no distribucional → no lo cierra D8. (4)
Sobreajuste: con 159 params y ~212 días en el estado crisis OOS hay menos soporte por
estado, pero el BIC (que penaliza params) sigue prefiriéndolo con holgura, así que el
ajuste extra está justificado.

## Fricción con el núcleo

Ninguna que obligue a tocar `src/`. El filtrado forward t se resolvió en el detector
(`_hmm_t_utils.py`) reusando el patrón de D4. El Arreglo 4 (vol-primario) ordenó
correctamente los 4 estados (monotonía verificada). Observación de lectura, no de
núcleo: para detectores multi-estado, `cov_<crisis>` mide solo el estado más extremo;
al comparar contra detectores de 2 estados conviene mirar también "corrección+crisis"
(lo hace el notebook), no solo la columna del master.
