# F8 — Generadores de datos sintéticos

<!-- BEGIN nota_v2 -->
> **Origen y ubicación.** Ficha nueva de la fase S (sin equivalente en la Capa 1). Bibliografía: [`F8_generadores_sinteticos.bib`](F8_generadores_sinteticos.bib) (todas sus claves están ya fusionadas en `docs/references.bib`). Código: `src/regimenes/sinteticos/` (un fichero por generador en `parametricos/` y `neuronales/`). Configuración: [`configs/sinteticos.yaml`](../../configs/sinteticos.yaml). Notebook: [`notebooks/15_sinteticos_generadores.ipynb`](../../notebooks/15_sinteticos_generadores.ipynb). Esta ficha describe **qué hace cada generador y qué no puede hacer por construcción**; no contiene resultados: las cifras de ajuste y de muestreo salen del notebook 15 y la validación, del 16.
<!-- END nota_v2 -->

> **Estado (2026-10-06).** Notebook 15 (generadores) y 16 (validación) hechos: **ningún generador resulta `apto_laboratorio`**; `apto_aumento` son, en la pista A, todos salvo `jitter` y `bootstrap_regimen`, y en la B `gaussiano`, `var`, `garch_regimen`, `rbig` y `flow_matching`. El notebook 17 (laboratorio) está hecho como **simulación controlada** con `gaussiano`, `var` y `garch_regimen` (18 celdas por pista, 4356 trabajos) y cuatro hipótesis: H1 (los detectores recuperan la señal sintética) se cumple (92 % en A, 100 % en B); H2, el score baja al acortar y atenuar los episodios (en parte por la prevalencia; el recall por evento pasa de 0,86 a 0,54 en A); H3, el ranking sintético se correlaciona con el real (ρ 0,87, p 0,001 en A; 0,48, p 0,16 en B); H4, hay indicio de circularidad para D03 (+0,23 relativa en B) y ninguna para GARCH. El 18 (aumento) sigue pendiente.

> Familia: generadores de trayectorias multivariantes de features **con la
> secuencia de régimen conocida**. Núcleo: seis generadores paramétricos (jitter,
> bootstrap por régimen, gaussiano por régimen, VAR por régimen, GJR-GARCH por
> régimen, RBIG) y cuatro neuronales (flow matching, difusión, CVAE, CGAN), todos
> detrás de la misma interfaz `Generador` (`fit` / `sample` / `name`).
> Aviso de honestidad: un generador **no crea información** sobre crisis que no
> esté en su tramo de entrenamiento; con 8 episodios de crisis en el ajuste de la
> pista A, lo que aquí se fabrica son variaciones sobre esos episodios, no crisis
> nuevas. Distinguimos abajo lo que cada modelo reproduce por construcción de lo
> que solo puede decidir la validación (notebook 16).

## Para qué sirven en este TFM

Con 17 crisis evaluables en la pista A y 9 en la pista B, el banco de pruebas tiene
poca potencia estadística para separar detectores (ver la limitación de
significancia de la Capa 1 en `docs/historia/capa1/memoria/99_conclusions.md`).
Los generadores sintéticos se usan para tres cosas, por este orden de fiabilidad:

1. **Laboratorio con verdad conocida** (notebook `17`, hecho como simulación controlada): trayectorias cuya cadena de
   regímenes se conoce exactamente, para medir recall y precisión de los detectores
   sin la ambigüedad del etiquetado histórico.
2. **Aumento de datos** (notebook `18`): entrenar detectores con más episodios de
   crisis y evaluarlos en real (TSTR, *train on synthetic, test on real*
   [synth_esteban2017]) sin tocar el test real.
3. **Pruebas de estrés** de la máquina de fusión y de la señal de régimen que consumirán los agentes (tool `get_regimen`).

Ninguno de los tres usos sustituye al benchmark real: el laboratorio responde a
«¿detecta este detector un cambio de ley cuando lo hay?», no a «¿detecta las crisis
del catálogo?».

## Definición y supuestos

Un generador aprende, solo con el tramo de entrenamiento, la ley de un panel de
features condicionada a un régimen `s_t ∈ {0, 1}` y devuelve trayectorias nuevas con
su columna `regime`:

```
fit(train, regimes)                 ajusta p(x_t | pasado, s_t) con datos <= fin_train
sample(n_paths, length, regimes)    n_paths trayectorias de `length` sesiones + regime
```

Tres decisiones de diseño son comunes a los diez generadores y viven fuera de ellos
(`comun.py`, `datos.py`, `espacio.py`), para que las diferencias que midan los
notebooks 16–18 sean del modelo y no de la fontanería.

**1. Régimen de referencia: ventanas de crisis, nunca un detector.** El régimen es
binario: 1 = el día cae dentro de alguna ventana `[pico, suelo]` (extremos incluidos)
de `crisis_windows` de la pista en `configs/benchmark_spec.yaml`; 0 = calma
(`datos.regimen_referencia`). No sale de ningún detector del benchmark: si el
generador se condicionara a las etiquetas de un HMM, ese HMM jugaría en casa al
evaluarse sobre lo generado (circularidad). El corte de entrenamiento no puede partir
un episodio (sesgaría duraciones y transiciones; `cargar_entrenamiento` lo
comprueba): **pista A hasta 2006-12-31, pista B hasta 2017-12-31**.

**2. El régimen es exógeno.** Ningún generador modela la transición: rellenan la
secuencia que reciben. La secuencia es (a) **impuesta** por quien llama, o (b)
**simulada** por la base con la cadena estimada en train, en dos variantes
(`datos.simular_regimenes`):

```
Markov:       P[i,j] = P(s_{t+1}=j | s_t=i) por conteo; duración geométrica de media 1/(1 − P[k,k])
semi-Markov:  cada racha dura una duración REAL de train de su régimen, remuestreada con reemplazo
arranque:     "ultimo" (continúa el último estado de train) | "estacionaria" (estado inicial ~ π, π P = π)
```

**3. Semántica del régimen sintético.** En el histórico, `regime = 1` significa
«tramo pico→suelo de una crisis del catálogo». En una trayectoria sintética significa
otra cosa: **«día extraído de la ley condicional de crisis»**. Un tramo sintético con
`regime = 1` no tiene por qué empezar en un máximo ni acabar en un mínimo del precio
generado, ni su drawdown tiene que parecerse al de un episodio real. El laboratorio
mide si un detector reconoce ese cambio de ley, no si encuentra picos y suelos.

**Supuesto transversal y crítico:** condicionado al régimen, el proceso es
estacionario; un día de la crisis de 1974 es intercambiable con uno de la de 2001.
Es falso en el detalle (los episodios no se parecen entre sí) y es el precio de tener
solo 8 episodios de crisis en el ajuste de la pista A.

## Espacio de generación y re-derivación

Cuatro columnas del núcleo (`SP500_ret_z`, `SP500_vol_z`, `SP500_momentum`,
`SP500_drawdown`) son funciones **deterministas** de la senda del S&P 500. Un
generador que las modelara como variables libres produciría paneles imposibles (un
drawdown que no corresponde a los retornos generados, una volatilidad realizada
incoherente con ellos) y los detectores verían una inconsistencia que no existe en
datos reales. Por eso hay dos espacios (`espacio.EspacioGeneracion`):

- **Espacio de generación (de trabajo):** el log-retorno crudo `SP500_ret` más las
  columnas no derivables del panel, estandarizadas con media y desviación **solo de
  train**. Es lo único que ven los generadores (6 columnas en la pista A, 11 en la B).
- **Vuelta al espacio público (re-derivación):** se deshace la estandarización y las
  cuatro columnas se recalculan con las mismas primitivas causales de
  `regimenes.features` sobre [historia real del S&P 500 hasta `fin_train`] +
  [retornos sintéticos], quedándose con el tramo sintético:

```
P_t = P_fin_train · exp(Σ r_sint)          senda de precios que continúa la historia real
SP500_ret_z, SP500_vol_z   = z expanding (sobre historia real + tramo sintético)
SP500_momentum             = 12M−1M sobre P_t
SP500_drawdown             = P_t / max(P_{≤t}) − 1     (el máximo incluye el histórico real)
```

Cada trayectoria es así una **continuación hipotética del mercado tras `fin_train`**.
La prueba de que la receta es la del preprocesado es `error_rederivacion_`: re-derivar
las cuatro columnas sobre la historia real reproduce el panel de entrenamiento con
error 0. No se retiene futuro: el espacio y el pickle del generador solo conservan
precios `<= fin_train`.

Limitación declarada: las trayectorias se fechan con días hábiles de lunes a viernes,
que no es el calendario de la NYSE. Las fechas sintéticas son una etiqueta ordenada,
no deben cruzarse por fecha con datos reales.

## Variantes principales

| Generador | Familia | Idea | Dinámica dentro de régimen | ¿Crea días nuevos? | Código |
|---|---|---|---|---|---|
| `jitter` | remuestreo | tramos reales + ruido gaussiano | la del tramo copiado | no (bola alrededor de un real) | `parametricos/jitter.py` |
| `bootstrap_regimen` | remuestreo | bootstrap estacionario dentro de cada régimen | la del bloque | no | `parametricos/bootstrap_regimen.py` |
| `gaussiano_regimen` | paramétrico | normal multivariante i.i.d. por régimen | ninguna | sí | `parametricos/gaussiano_regimen.py` |
| `var_regimen` | paramétrico | VAR(p) con coeficientes por régimen observado | lineal en media | sí | `parametricos/var_regimen.py` |
| `garch_regimen` | paramétrico | GJR-GARCH-t por régimen (mercado) + VAR(1) (resto) | media y varianza | sí | `parametricos/garch_regimen.py` |
| `rbig` | flujo sin gradiente | gaussianización iterativa de bloques | dentro del bloque | sí, con soporte acotado | `parametricos/rbig.py` |
| `flow_matching` | neuronal (flujo continuo) | campo de velocidades ruido → bloque | dentro del bloque + contexto | sí | `neuronales/flow_matching.py` |
| `difusion` | neuronal (difusión) | denoiser, muestreo DDIM | ídem | sí | `neuronales/difusion.py` |
| `cvae` | neuronal (variable latente) | ELBO con decoder heterocedástico | ídem | sí | `neuronales/cvae.py` |
| `cgan` | neuronal (adversarial) | WGAN-GP condicional | ídem | sí | `neuronales/cgan.py` |

### Remuestreo: `bootstrap_regimen` y `jitter`

- **Bootstrap estacionario condicionado a régimen.** No ajusta nada: rellena la
  secuencia de régimen pedida con bloques contiguos de días reales de ese mismo
  régimen. La longitud de bloque es geométrica (bootstrap estacionario de Politis y
  Romano [synth_politisromano1994]), lo que evita el artefacto de periodicidad de los
  bloques de longitud fija de Künsch [synth_kunsch1989]:

  ```
  L ~ Geom(p),  p = 1 / longitud_media        inicio uniforme entre los días del régimen
  ```

  A diferencia del original no hay *wrap-around*: un bloque nunca cruza el final de
  una racha real, así que la longitud efectiva es menor que `longitud_media`.
  **Reproduce** exactamente las marginales y la dependencia transversal de cada
  régimen (cada fila sintética es una fila real) y la dependencia temporal hasta el
  orden del bloque (agrupamiento de volatilidad, escalones mensuales). **No
  reproduce** nada fuera del soporte de train (nunca sale un día peor que el peor
  observado), salta en las costuras las columnas persistentes y no modela la dinámica
  propia del cambio de régimen (un bloque de crisis puede empezar en mitad de un
  episodio real). En memorización es copia literal por construcción. Coste: nulo.
- **Jitter.** La misma mecánica con bloques largos y ruido añadido:

  ```
  x_sint = x_real + σ · sd_train(columna) · ε ,   ε ~ N(0, I)
  ```

  `σ` es relativa a la escala de cada columna. **Memoriza por diseño**: cada día
  sintético está a distancia ~`σ·√d` de un día real concreto. Es el **control
  positivo** de las métricas de memorización del notebook 16 (si una métrica no marca
  al jitter como copia, la métrica no sirve) y el suelo de utilidad: un generador que
  no lo mejore no aporta más que regularización por ruido. El ruido blanco rompe los
  escalones de las columnas mensuales y se acumula en el precio re-derivado.

### Paramétricos por régimen observado

En los tres el régimen **no es latente**: es el de referencia. Eso elimina el filtro
de Hamilton y el EM [hamilton1989], separa la verosimilitud por régimen y deja
estimadores cerrados o de máxima verosimilitud exacta.

- **`gaussiano_regimen` — la línea base «solo segundo orden».**

  ```
  x_t | s_t = k  ~  N(μ_k, Σ_k)      Σ_k por contracción de Ledoit–Wolf, muestreo por Cholesky
  ```

  La covarianza es el estimador de contracción de Ledoit y Wolf [synth_ledoitwolf2004]
  (con `d` pequeña la contracción es casi nula; se mantiene por los pocos días
  efectivamente independientes de la crisis). **Reproduce** media, desviación y
  correlaciones contemporáneas por régimen; la mezcla de regímenes da colas
  incondicionales más gruesas que una normal, pero solo por esa vía. **No reproduce,
  por diseño,** colas ni asimetría dentro de régimen (curtosis 3), ni ninguna
  dependencia temporal: las columnas de nivel y los escalones mensuales salen como
  ruido blanco alrededor de la media del régimen. Esa pobreza es su propósito: todo
  lo que otro generador gane sobre este es atribuible a dinámica o a colas.
- **`var_regimen` — VAR por régimen observado.**

  ```
  x_t = c_k + A_k1 x_{t−1} + … + A_kp x_{t−p} + e_t ,     s_t = k
  ```

  Es el MS-VAR de Hamilton [hamilton1989] y Krolzig [synth_krolzig1997] con el
  régimen conocido: MCO por régimen (con una penalización *ridge* numérica), usando
  solo pares que no cruzan una frontera de régimen. Dos salvaguardas declaradas: el
  radio espectral de la matriz compañera se contrae por debajo de `radio_max`, y la
  constante se ancla para que el punto fijo de cada régimen sea su media muestral
  (`anclar_media`). Las innovaciones se remuestrean por filas enteras de residuos
  reales del régimen. **Reproduce** la persistencia lineal (autocorrelaciones y
  correlaciones cruzadas retardadas), la dependencia contemporánea y las colas de las
  innovaciones, y transiciones suaves entre regímenes. **No reproduce** agrupamiento
  de volatilidad dentro de régimen (innovaciones i.i.d.: toda la heterocedasticidad
  viene del cambio de régimen) ni no linealidades; la media por régimen solo se
  alcanza como punto fijo, y con raíces casi unitarias una crisis corta no llega a
  él. *Aviso de nombre:* no confundir con el detector D05 `markov_switching_var`, que
  pese a su nombre no es un VAR (ver [GLOSARIO](../GLOSARIO.md#nombres-de-detectores-que-confunden)).
- **`garch_regimen` — GJR-GARCH-t por régimen + VAR(1) condicionado.** El único
  paramétrico que reproduce, dentro de cada régimen, los tres hechos estilizados del
  retorno diario. Factor de mercado (`SP500_ret`):

  ```
  r_t = μ[s_t] + e_t ,   e_t = √h_t · z_t ,   z_t ~ t_ν[s_t] estandarizada
  h_t = ω[s_t] + (α[s_t] + γ[s_t] · 1{e_{t−1} < 0}) · e_{t−1}² + β[s_t] · h_{t−1}
  ```

  GJR-GARCH [vol_glosten1993] con innovación t de Student [vol_bollerslev1987] y todos
  los parámetros dependientes del régimen. Como `s_t` es observado, no hay la
  dependencia de la senda que obliga a Gray [vol_gray1996] a colapsar la varianza o a
  Haas, Mittnik y Paolella [vol_haasmittnikpaolella2004] a llevar K recursiones en
  paralelo: la recursión es única, atraviesa los cambios de régimen y su verosimilitud
  es exacta (máxima verosimilitud conjunta, partiendo de un ajuste por régimen con
  `arch`). El resto de columnas sigue un VAR(1) por régimen con el mercado
  contemporáneo como regresor:

  ```
  y_t = c[s_t] + A[s_t] x_{t−1} + b[s_t] m_t + u_t
  ```

  con dependencia entre columnas lineal y constante por régimen, en el espíritu de la
  correlación condicional constante de Bollerslev [synth_bollerslev1990].
  **Reproduce** nivel de volatilidad por régimen, agrupamiento de volatilidad,
  apalancamiento y colas pesadas. **No reproduce** asimetría de la innovación (t
  simétrica), memoria larga de la volatilidad (un GARCH(1,1) decae geométricamente),
  dependencia de cola entre columnas ni heterocedasticidad propia de las columnas que
  no venga del mercado. Lleva dos topes de simulación declarados y contados (techo de
  varianza por régimen y techo de nivel): si la fracción de celdas recortadas crece,
  el techo está haciendo de modelo.

### RBIG: gaussianización iterativa por bloques

RBIG (*Rotation-Based Iterative Gaussianization*, Laparra, Camps-Valls y Malo
[synth_laparra2011], sobre la gaussianización de Chen y Gopinath
[synth_chengopinath2000]) es un flujo normalizante que **no se entrena por
gradiente**. Cada capa encadena dos operaciones cerradas e invertibles:

```
1. gaussianización marginal:  z_d = Φ⁻¹( F̂_d(x_d) )      (CDF empírica + probit, por dimensión)
2. rotación PCA:              z ← R z
```

Iterando, la multi-información se destruye capa a capa hasta un gaussiano
factorizado; generar es muestrear `z ~ N(0, I)` y desandar las capas. RBIG modela un
vector, no una serie, así que aquí se ajusta **un modelo por régimen** sobre bloques
de `largo_bloque` sesiones aplanados, y el contexto entra por la condicional
gaussiana en el espacio gaussianizado (complemento de Schur):

```
E[g_blk | g_ctx] = m_b + A (g_ctx − m_c) ,   A = S_bc S_cc⁻¹ ;    RBIG modela r = g_blk − E[g_blk | g_ctx]
```

**Reproduce** la marginal de cada columna por régimen (colas y asimetría incluidas),
la dependencia cruzada y la dinámica dentro del bloque, y la continuidad de las
columnas persistentes en las costuras. **No reproduce:** días fuera del soporte (la
inversa satura en el mínimo y el máximo de train: no inventa un día peor que el peor
observado), herencia de volatilidad entre bloques (el contexto solo desplaza la
media; sin efecto GARCH entre bloques) ni los escalones mensuales (salen como
derivas suaves). Coste: segundos a minutos en CPU, `numpy`/`scipy` puros.

### Neuronales de «siguiente bloque»

Los cuatro necesitan el extra `[deep]` (torch) y resuelven el mismo problema
(sección siguiente); solo cambian la red y la pérdida. Flow matching y difusión
comparten literalmente red, transformación de datos, equilibrio de lotes y media
móvil de pesos (`_redes_flujo.py`); CVAE y CGAN comparten la representación
(`_redes_latentes.py`).

- **`flow_matching` — flujo continuo con caminos rectos.** Aprende el campo de
  velocidades que transporta ruido a datos [synth_lipman2023; synth_liu2023]:

  ```
  x_τ = (1 − τ) x_0 + τ x_1 ,  x_0 ~ N(0, I) ;    pérdida = E ‖ v_θ(x_τ, τ, c) − (x_1 − x_0) ‖²
  muestreo: integrar dx/dτ = v_θ(x, τ, c) de τ = 0 a 1 (Heun o Euler)
  ```

  No hay que integrar nada para entrenar. La pérdida no tiende a cero (su mínimo es
  la varianza condicional de `x_1 − x_0` dado `x_τ`) y una pérdida baja no garantiza
  buenas muestras: el error se integra a lo largo de la trayectoria y de los bloques
  encadenados.
- **`difusion` — predicción de ruido, muestreo DDIM.** Proceso directo en forma
  cerrada [synth_ho2020] con planificador coseno [synth_nicholdhariwal2021]:

  ```
  x_τ = √ᾱ_τ · x_0 + √(1 − ᾱ_τ) · ε ;    pérdida = E ‖ ε_θ(x_τ, τ, c) − ε ‖²
  ```

  y muestreo determinista DDIM sobre una subsecuencia de niveles [synth_song2021].
  El régimen se anula en una fracción de los ejemplos para poder usar *classifier-free
  guidance* [synth_hosalimans2022] (por defecto, condicional puro). Es el mismo
  transporte que flow matching con otra parametrización; sus pérdidas no son
  comparables entre sí. La calidad la deciden los parámetros de **muestreo** (dónde
  arranca la cadena inversa), no la curva de entrenamiento.
- **`cvae` — autoencoder variacional condicional.** Modelo de variable latente
  [nn_kingma2014] condicionado en encoder y decoder [synth_sohn2015]:

  ```
  q(z | x, c) = N(μ(x,c), diag σ²(x,c)) ,  p(z | c) = N(0, I) ,  p(x | z, c) = N(m(z,c), diag s²(z,c))
  ELBO = E_q[ log p(x | z, c) ] − β · KL( q(z | x, c) ‖ p(z | c) )
  ```

  con el peso `β` de β-VAE [synth_higgins2017] en rampa. La varianza del decoder es
  **aprendida** y se muestrea `x = m + s·ε`: muestrear solo la media produce
  infradispersión, porque la media condicional promedia todo lo que `z` no explica.
  Supone gaussiana diagonal condicionada a `z`: si el latente colapsa, degenera en
  ruido independiente por día.
- **`cgan` — GAN condicional entrenada como WGAN-GP.** Un generador `G(z, c)` y un
  crítico `D(x, c)` que comparten condición [synth_goodfellow2014;
  synth_mirzaosindero2014], con pérdida de Wasserstein y penalización de gradiente
  [synth_arjovsky2017; synth_gulrajani2017]:

  ```
  min_G max_D  E[D(x, c)] − E[D(G(z, c), c)] − λ · E[(‖∇_x̂ D(x̂, c)‖ − 1)²]
  ```

  Sin verosimilitud ni selección de época por validación. Riesgo específico con una
  condición tan informativa: **colapso condicional** (G ignora `z` y emite una única
  continuación por contexto; las marginales pueden parecer correctas).

**Qué reproducen y qué no, como clase.** Pueden reproducir escala por régimen,
persistencia de las columnas de nivel y colas más pesadas que una gaussiana (por la
transformación `arcsinh` de las colas, no por el ruido latente). **No garantizan**
los escalones exactos de las columnas mensuales, el agrupamiento de volatilidad más
allá de lo que quepa en un bloque más su contexto, ni eventos del tamaño de octubre
de 1987 (una sola observación no se aprende). Con latente gaussiano y una red de
momentos finitos, la cola **condicional** tiende a quedar corta; es la predicción de
Wiese et al. [synth_wiese2020] y lo que midió el taller previo (más abajo).

## Formulación de «siguiente bloque» y encadenado

Una red no puede generar de una vez las 2 520 sesiones de una trayectoria con un
tramo de entrenamiento de ~11 000. RBIG y los cuatro neuronales modelan

```
p( bloque_{t+1 … t+L_blk}  |  contexto_{t−L_ctx+1 … t} ,  régimen de CADA día del bloque )
```

con `L_blk = 21` sesiones (contexto de 21 en los neuronales y de 5 en RBIG). Que el
régimen sea un valor por día hace que una transición calma→crisis a mitad de bloque
sea una condición más, no un caso especial. `bloques.py` contiene las dos mitades,
idénticas para todos:

- **Trocear** (`construir_bloques`): ventanas deslizantes (contexto, bloque, régimen
  del bloque) solo dentro de tramos contiguos de train.
- **Encadenar** (`encadenar`): el primer contexto son las últimas filas reales de
  train; cada bloque generado pasa a ser contexto del siguiente, `⌈length / L_blk⌉`
  veces.

Supone dependencia de Markov de orden `L_ctx` entre bloques. El riesgo propio del
encadenado es la **realimentación**: un bloque extremo saca el contexto de lo que la
red ha visto y el siguiente extrapola. Cada familia lleva una salvaguarda declarada
(saturación del contexto de entrada en CVAE/CGAN, tope sobre el extremo histórico en
flow matching y difusión, saturación de la inversa en RBIG), y la fracción de
coordenadas afectadas queda en `diagnostico_muestreo()`.

Los paramétricos recursivos (`var_regimen`, `garch_regimen`) no necesitan bloques:
simulan día a día desde el último estado real. `gaussiano_regimen`, `bootstrap_regimen`
y `jitter` ignoran el contexto: su primera fila no continúa el último día real.

## Fortalezas y debilidades

**Fortalezas:**
- **Verdad de régimen exacta.** La etiqueta de cada día sintético se conoce por
  construcción; el histórico solo ofrece ventanas de catálogo discutibles.
- **Escalera de complejidad.** De «solo medias y covarianzas» (gaussiano) a «dinámica
  no lineal» (neuronales): cada peldaño aísla qué aporta la dinámica, las colas o la
  no linealidad.
- **Paneles coherentes.** La re-derivación garantiza que drawdown, momentum y
  volatilidad realizada corresponden a los retornos generados.
- **Controles incorporados.** Jitter (control positivo de memorización) y gaussiano
  (suelo de fidelidad) dan escala a cualquier métrica.

**Debilidades (las decisivas):**
- **Pocas crisis.** El régimen de crisis se aprende de un puñado de episodios; las
  ventanas solapadas de un mismo episodio no son observaciones independientes.
- **Régimen exógeno.** La cadena no depende de los retornos generados: no hay
  deterioro previo al pico ni retroalimentación mercado→régimen.
- **Estacionariedad condicional.** Las crisis se tratan como intercambiables.
- **Topes y salvaguardas.** Varios generadores acotan varianza, nivel o contexto para
  no divergir al encadenar; son parte del modelo y hay que contarlos.

## Estado del arte no implementado

El esqueleto de esta ficha anunciaba TimeGAN, QuantGAN, un «MS-VAR» y un generador
HMM-t. Lo que cambia y por qué:

- **TimeGAN** [synth_yoon2019] (cuatro redes; pérdida adversarial + supervisada paso
  a paso en un latente aprendido) y **Quant GANs** [synth_wiese2020] (generador y
  discriminador con convoluciones temporales dilatadas; reportan agrupamiento de
  volatilidad y apalancamiento) son la referencia en series financieras. **No se
  implementan**: su entrenamiento adversarial recurrente/convolucional es caro e
  inestable en CPU y no son condicionales a un régimen por día sin rediseñarlos. Se
  sustituyen por los cuatro neuronales portados del taller previo del equipo,
  reescritos en torch puro como modelos de siguiente bloque.
- **«MS-VAR»** pasa a ser **VAR por régimen observado** (`var_regimen`): con el
  régimen de referencia conocido no hay nada latente que filtrar.
- **No hay generador HMM-t separado**: simular de un HMM ajustado daría ventaja a los
  detectores D04/D08 (circularidad) y su contenido (mezcla por régimen + cadena de
  Markov) ya lo cubren `gaussiano_regimen` y la cadena simulada.
- **TimeVAE** [synth_desai2021]: VAE convolucional con bloques interpretables de
  tendencia y estacionalidad; referencia natural del CVAE, cuya estacionalidad no
  aporta nada a retornos diarios.
- **Generadores con firmas**: SigCWGAN [synth_liao2024] sustituye el discriminador por
  una métrica analítica sobre la firma del camino; el *market simulator* de Buehler
  et al. [synth_buehler2020] usa un VAE sobre firmas y está pensado para pocos datos.
  La dimensión de la firma crece geométricamente con el número de canales.
- **Difusión para series**: Diffusion-TS [synth_yuanqiao2024] (transformer con
  descomposición tendencia/estacionalidad) y CSDI [synth_tashiro2021] (difusión
  condicionada a observaciones, orientada a imputación). Arquitecturas con atención,
  fuera del presupuesto de CPU.
- **Tail-GAN** [synth_conttailgan2026]: pérdida basada en la elicitabilidad conjunta
  de VaR y ES para preservar el riesgo de cola de estrategias. Apunta justo al hueco
  que dejan los generadores de aquí (la cola), pero su objetivo es una cartera, no un
  panel de features con régimen.
- **Procesos con saltos / volatilidad estocástica** (Heston, Hawkes): calibración
  multivariante frágil; fuera de alcance.

## Antecedente: qué enseñó el taller previo

Los cuatro neuronales, RBIG, jitter y el gaussiano vienen de un taller previo del
mismo equipo (generación de ventanas de 60 sesiones × 20 canales, 2003–2026, régimen
etiquetado con un HMM de tres estados; sus decisiones D9, D16, D25 y D26). Su montaje
**no es el de este TFM** y sus cifras no se trasladan, pero sus lecciones fijaron cómo
se lee el notebook 16:

- **Solo RBIG y flow matching pasaron a la vez memorización y discriminador.** El
  CVAE quedó rechazado por infradispersión (cociente de distancias al vecino más
  cercano 0,843 en crisis, bajo el suelo de 0,9) y la CGAN fue trivialmente separable
  (AUC discriminativo 0,991 global y 0,999 en crisis). De ahí las correcciones de
  aquí: decoder heterocedástico en el CVAE y WGAN-GP en la CGAN.
- **Fidelidad y utilidad no van juntas.** RBIG, el mejor en fidelidad, quedó último
  en utilidad (balanced accuracy 0,574 frente a 0,575 sin sintéticos); el jitter, que
  suspende memorización, fue el primero (0,612), por delante de flow matching
  (0,605). El sintético actuó como regularizador, no como fuente de información:
  ninguna tabla puede titularse «el mejor generador» sin decir en qué eje.
- **Ningún generador reprodujo la cola condicional.** La curtosis de los residuos
  estandarizados por la volatilidad previa quedó entre 3,40 y 4,22 en los seis
  generativos, fuera de la banda real [5,57 – 10,30]; la cola incondicional se puede
  acertar por mezcla de escalas sin tener cola condicional.
- **La memorización se mide con el índice de vecindad completo.** Buscando el vecino
  más cercano en una submuestra del entrenamiento, el jitter aprobaba (cociente
  0,912); con el índice íntegro suspende (0,046). Solo se puede submuestrear el
  conjunto que estima la distancia real–real. Además, el cociente no distingue copiar
  de encoger: se separan por magnitud.
- **Las métricas de crisis se leen por episodio.** Las 587 ventanas de crisis de su
  entrenamiento eran 8 rachas contiguas: el n efectivo es el número de episodios.
  Toda métrica restringida a crisis lleva banda por *jackknife* de episodios, y del
  AUC discriminativo en crisis se publica el orden, no el valor.

## Criterios de validación (notebook 16)

La validación vive en el subpaquete `regimenes.sinteticos.validacion`, un módulo por
dimensión, y la ejecuta `16_sinteticos_validacion`. Cada generador se compara **solo con
el tramo real de entrenamiento** con el que se ajustó (lo posterior a `fin_train` queda
reservado a 17–20) y solo con las trayectorias de **cadena simulada**; las de secuencia
impuesta son para el laboratorio. Real y sintético nunca se cruzan por fecha: se
comparan por distribución o por posición dentro de cada trayectoria.

| Módulo | Funciones | Papel |
|---|---|---|
| `_comun` | `separar_real`, `separar_sintetico`, `tramos`, `id_episodio`, `estandarizador` | convenciones compartidas: rachas, episodios, estandarización con media y desviación de train |
| `fidelidad` | `fidelidad_marginal`, `fidelidad_dependencia`, `distancia_correlaciones`, `fidelidad_regimenes`, `condicionamiento` | marginales, hechos estilizados, correlaciones y señal de régimen; define el **contrato de salida** común |
| `discriminador` | `discriminador` | test de dos muestras con clasificador |
| `memorizacion` | `memorizacion` | distancia al vecino real más cercano |
| `utilidad` | `referencia_trtr`, `utilidad_tstr`, `unir_utilidad` | TSTR frente a TRTR con las métricas por evento del benchmark |
| `veredicto` | `veredicto` | aplica umbrales, reglas y niveles del yaml |

**Contrato de salida.** Todas las funciones devuelven una tabla larga con una fila por
(régimen, columna, métrica) y las columnas `real`, `sintetico`, `banda_inf`/`banda_sup`
(banda de referencia del real), `cociente` (sintético/real, solo para magnitudes de
escala positivas) y `en_banda`. Una fila **pasa** si el valor sintético cae en la banda
del 95 % del real **o** su cociente cae en `banda_cociente` (0,80–1,25). Un criterio
exige que pasen **todas sus filas clave** (`validacion.reglas`); el resto de filas se
publica como información, de modo que ninguna métrica compensa a otra. Los umbrales se
fijaron en `validacion.umbrales` de `configs/sinteticos.yaml` antes de ver ningún
resultado de validación.

**Bandas del real.** Salen de `n_boot` remuestreos del tramo de entrenamiento: en
**calma**, bootstrap por bloques contiguos de 63 sesiones tomados dentro de las rachas
de calma (Künsch [synth_kunsch1989]; Politis y Romano [synth_politisromano1994]); en
**crisis**, remuestreo por **episodio** completo, porque el n efectivo es el número de
episodios y no el de días. La banda es el intervalo 2,5–97,5 % de los remuestreos. Para
las distancias (Wasserstein, KS, Frobenius) la referencia es la distancia entre el real
y un remuestreo del propio real, que es estricta (comparten días). Las bandas miden la
variabilidad del real; la del sintético, con cien trayectorias apiladas, es casi nula y
no se añade. Excepción: la **referencia de igual a igual** de la dependencia (abajo), cuya
banda es la de ventanas reales del largo de las trayectorias.

| Criterio | Función | Qué mide (filas clave) | Papel en el veredicto |
|---|---|---|---|
| Fidelidad marginal | `fidelidad_marginal` | por régimen: desviación de **cada** columna modelada y amplitud de cola relativa de `q01` y `q99` de `SP500_ret`, `(q_s − mediana_s) / (q_r − mediana_r)`. Momentos, Wasserstein, KS y re-derivadas, informativos | laboratorio |
| Fidelidad de dependencia | `fidelidad_dependencia` | por régimen y **dentro de racha**: `acf_abs_1`, `curtosis_condicional_tray` y `vida_media_persistentes`; referencia real de igual a igual (ventanas del largo de las trayectorias) donde caben al menos `min_ventanas_no_solapadas` (3) ventanas no solapadas | laboratorio |
| Condicionamiento | `condicionamiento` | cociente de la separación de volatilidad `sd(r ∣ crisis) / sd(r ∣ calma)`. Deriva en crisis y AUC de régimen, informativos | laboratorio y aumento |
| Discriminador | `discriminador` | AUC global real frente a sintético con todas las columnas `≤ discriminador_auc_max` (0,75) [synth_lopezpaz2017] | laboratorio |
| Memorización | `memorizacion` | `cociente_nn` en el espacio de las columnas **modeladas**, global y en crisis, `≥ memorizacion_cociente_min` (0,90) | aumento |
| Correlaciones | `distancia_correlaciones` | Frobenius por régimen y del salto calma→crisis | informativo |
| Utilidad (TSTR) | `utilidad_tstr` | detector ajustado en sintético y evaluado en real frente a TRTR [synth_esteban2017] | informativo |
| Regímenes | `fidelidad_regimenes` | fracción de días, duraciones, `P_kk`, KS de duraciones | control (la cadena es común) |

**Cómo se lee cada dimensión.**

- **Hechos estilizados dentro de racha.** Las autocorrelaciones solo usan pares
  `(t, t+k)` de la misma racha, para no mezclar el salto de nivel entre regímenes con la
  dinámica propia (el notebook 15 las medía sobre la trayectoria entera; esas filas se
  conservan como `regimen='todos'`). En el sintético se toma la mediana entre
  trayectorias. La **curtosis condicional** es la de Pearson de `z_t = r_t / σ_t`, con
  `σ_t` EWMA de RiskMetrics (λ = 0,94) sobre retornos **pasados**, sin reiniciar en los
  cambios de racha. Como un estadístico de cola por trayectoria tiene la mediana sesgada
  a la baja y la banda real de crisis descansa en muy pocos episodios, la fila clave
  invierte la pregunta (`curtosis_condicional_tray`): ¿es el valor real típico del
  generador, es decir, cae en la banda 2,5–97,5 % de los valores por trayectoria? La
  persistencia de las features lentas se mide como **vida media** `h = ln 0,5 / ln ρ` de
  un AR(1) con la acf1 de cada columna persistente: el cociente de dos acf1 cercanas a 1
  apenas discrimina.
- **Referencia de igual a igual** (`reglas.dependencia.referencia_ventanas` y
  `min_ventanas_no_solapadas`). La autocorrelación estimada en una muestra corta está
  sesgada a la baja, y más cuanto más cerca de 1: comparar trayectorias de
  `muestreo.length` (2.520) sesiones con el real entero castiga incluso a un generador
  perfecto, sobre todo en la vida media. Por eso el valor real y su banda de las ACF
  (`acf_r_k`, `acf_abs_k`, `acf_sq_k`), del apalancamiento y de `vida_media_persistentes`
  (filas de calma y crisis) se calculan sobre **ventanas reales del mismo largo** que las
  trayectorias, deslizantes con paso de 63 sesiones: mediana entre ventanas y banda
  2,5–97,5 % entre ventanas. Solo se aplica si el real admite al menos 3 ventanas **no
  solapadas** de ese largo: en la pista A sí; en la B no (su tramo de entrenamiento mide
  casi lo mismo que una trayectoria, y unas pocas ventanas casi idénticas dejarían la banda
  en un punto que suspendería hasta al propio real troceado), y allí se mantiene el real
  entero con banda bootstrap. El control es el real troceado en ventanas del largo de las
  trayectorias, pasado por `fidelidad_dependencia` como si fuera un generador.
- **Regímenes y condicionamiento.** Duraciones y transiciones las simula la base común
  con la cadena de train, la misma ley para todos: `fidelidad_regimenes` es un control,
  no un criterio. Lo que depende del generador es si un día con `regime = 1` se parece a
  un día de crisis: la separación de volatilidad es el criterio; el AUC de una logística
  que predice el régimen del día es informativo (el agrupado del real es bajo por la
  heterogeneidad de los episodios, así que un generador fiel da uno mayor; el de ajuste
  es el comparable).
- **Discriminador** (*classifier two-sample test* [synth_lopezpaz2017]). Ventanas no
  solapadas de 21 sesiones, resumidas por columna (media, desviación, mínimo, máximo,
  último − primero, acf1; curtosis y media de `|r|` para el retorno) y clasificadas con un
  `HistGradientBoostingClassifier` modesto de parámetros fijos, con validación cruzada
  **agrupada** (bloques cronológicos del real, trayectorias enteras del sintético) y
  clases equilibradas. Sustituye a la red recurrente de la métrica discriminativa de
  TimeGAN [synth_yoon2019]: si un clasificador sencillo ya separa, el generador no pasa.
  En crisis solo se publica el **orden** entre generadores. **Artefacto con copias:** si
  un generador copia días reales, las copias de las ventanas reales de prueba están en
  los pliegues de entrenamiento etiquetadas como sintéticas y el AUC cae **por debajo de
  0,5**; el discriminador no detecta la copia (la premia), y el veredicto anota el
  indicio para que lo resuelva la memorización.
- **Memorización.** Cociente de medianas
  `d_NN(sintético → real) / d_NN(real → real)`, con la distancia real–real excluyendo
  vecinos a menos de 21 sesiones, en el espacio estandarizado con train. El índice de
  vecindad es el real **íntegro** (solo se submuestrean consultas sintéticas; lección del
  taller previo) y la banda es *jackknife* por racha. `frac_copias` y `dispersion`
  separan copiar (cociente ≈ 0, dispersión normal, copias literales) de encoger
  (cociente < 1, dispersión baja, sin copias). El veredicto lee el espacio de las
  columnas **modeladas**: en el de todas, las re-derivadas continúan la senda sintética
  y no la del día copiado, esconden la copia y los controles positivos (`jitter`,
  `bootstrap_regimen`) aprobarían, lo que invalida la métrica por la regla a priori de
  esta ficha.
- **Utilidad: TSTR frente a TRTR** [synth_esteban2017]. Detectores baratos del banco
  (`validacion.tstr.detectores`) se ajustan una vez por trayectoria sintética (orden
  económico de estados con el retorno sintético) y se evalúan, con parámetros
  congelados, en los mismos días fuera de muestra del **TRTR walk-forward** del tramo
  real de entrenamiento, con el `score_deteccion` de ADR-003. Se publica también el
  **TRTR fijo** (un solo ajuste con todo el real, evaluado igual que el TSTR) para separar
  el efecto de los datos (TSTR frente a TRTR fijo) del del protocolo (fijo frente a
  walk-forward), y un nulo aleatorio persistente con la agresividad de la propia señal
  TSTR. **Es informativo** por dos razones: es dentro de muestra respecto al generador
  (vio los días evaluados) y, sobre todo, un **control negativo** de filas reales i.i.d.,
  sin dinámica y con régimen no informativo, aprueba la regla en la pista A; en la B la
  suspende, pero con 3 crisis evaluables el TSTR de B tampoco tiene potencia. Los
  detectores son no supervisados: ajustados en sintético aprenden estados por nivel de
  volatilidad, así que el TSTR mide si la ley marginal permite calibrarlos, no si el
  sintético transmite la señal de crisis.

**Niveles del veredicto** (`validacion.niveles`). Cada nivel es la conjunción de sus
criterios; todos los criterios se calculan y se publican, y solo deciden los que figuran
en algún nivel.

- **`apto_laboratorio`** = marginal + dependencia + condicionamiento + discriminador: lo
  generado se parece al real lo bastante para estresar detectores con verdad de régimen
  exacta (notebook 17) y un clasificador no lo distingue del real.
- **`laboratorio_condicionado`** = marginal + dependencia + condicionamiento: el mismo
  nivel sin el discriminador. No sustituye a `apto_laboratorio`: el 17 puede usar estos
  generadores **advirtiendo** que son distinguibles del real.
- **`apto_aumento`** = memorización + condicionamiento, igual en las dos pistas: lo
  generado no es copia del entrenamiento y lleva la señal de régimen (notebook 18).

**Decisiones declaradas.** Todas del 2026-10-01; el motivo y las cifras que justifican
cada una están en los comentarios de la sección `validacion` de
`configs/sinteticos.yaml` y en el notebook 16.

- *Antes de ver ningún resultado de validación:* umbrales de `validacion.umbrales`
  (banda de cocientes, suelo de memorización, techo del AUC del discriminador, regla de
  utilidad). No se retocan a la vista de las cifras.
- *Tras las pruebas de humo de las funciones y antes de calcular ningún veredicto:*
  (1) variante del discriminador solo con `SP500_ret` (`columnas_informativas`), fuera del
  veredicto, porque con todas las columnas los escalones y la persistencia de las series
  mensuales bastan para separar; el criterio sigue usando todas; (2) espacio de
  memorización `modeladas` (`espacio_veredicto`), por la regla de los controles
  positivos; (3) regla «todas las filas clave» por criterio.
- *Tras la revisión metodológica del veredicto:* (4) `curtosis_condicional_tray` sustituye
  a la curtosis condicional mediana frente a la banda real; (5) `vida_media_persistentes`
  sustituye a `acf1_persistentes`; (6) las correlaciones salen del veredicto
  (informativas), porque con el factor sobre el p95 no discriminaban; (7) utilidad
  informativa en las dos pistas (antes decidía en la pista A) y `apto_aumento` =
  memorización + condicionamiento (antes, utilidad + memorización), porque el control
  negativo i.i.d. aprobaba el TSTR en la pista A; (8) referencia de igual a igual en la
  dependencia (`referencia_ventanas`, `min_ventanas_no_solapadas` = 3), tras ver que el
  propio real troceado en ventanas del largo de las trayectorias casi suspendía la vida
  media; se aplica en la pista A y no en la B (antes, real entero en las dos).
- *Después de ver el veredicto:* (9) nivel `laboratorio_condicionado`, porque el
  discriminador con todas las columnas separaba a todos los generadores y ninguno quedaba
  `apto_laboratorio`.

**Checklist de hechos estilizados** (Cont [synth_cont2001]; se contrastan los
aplicables a datos diarios sin volumen, con métricas de `fidelidad` por régimen y
dentro de racha):

| Hecho | Contraste implementado | Papel | Quién puede reproducirlo por construcción |
|---|---|---|---|
| Ausencia de autocorrelación lineal en retornos | `acf_r_k` | informativo | todos |
| Colas pesadas incondicionales | amplitud de cola `q01`, `q99` (la curtosis depende de un solo dato) | criterio marginal | remuestreo, RBIG, GARCH-t; el gaussiano solo por mezcla |
| Asimetría ganancia/pérdida | `asimetria` | informativo | remuestreo, RBIG; no el GARCH-t simétrico |
| Agrupamiento de volatilidad | `acf_abs_1` (fila clave); `acf_abs_k`, `acf_sq_k` informativas | criterio dependencia | GARCH; remuestreo y bloques solo hasta el largo del bloque; no gaussiano ni VAR dentro de régimen |
| Colas pesadas condicionales | `curtosis_condicional_tray` (EWMA causal, banda por trayectoria) | criterio dependencia | GARCH-t; dudoso en neuronales (antecedente del taller) |
| Decaimiento lento de la ACF de `\|r_t\|` | `pendiente_loglog_abs` (retardos 1–21) | informativo | ninguno por construcción (GARCH(1,1) decae geométricamente) |
| Efecto apalancamiento | `apalancamiento_k` = `corr(r_t, r²_{t+k})` | informativo | GARCH (término GJR); remuestreo dentro del bloque |
| Gaussianidad por agregación | no se contrasta | — | poco discriminante por sí sola |
| Correlaciones que cambian en crisis | `distancia_correlaciones` (Frobenius por régimen y `salto_correlacion`) | informativo | todos los condicionados a régimen |
| Persistencia de las features lentas (propio del panel, no de Cont) | `vida_media_persistentes`; `escalones` informativo | criterio dependencia | remuestreo (escalones incluidos); VAR, GARCH y neuronales con contexto |

**Salidas.** En `results/sinteticos/validacion/` (versionable salvo las figuras PNG): una
tabla larga por dimensión y pista (`<dimension>_pista<X>.csv`), la referencia TRTR, el
TSTR por trayectoria, los tiempos, la ficha de la configuración usada
(`ficha_pista<X>.json`) y el veredicto generador × criterio (`veredicto_pista<X>.csv`).

## Riesgos metodológicos

- **Fuga de información.** El generador solo puede ver datos `<= fin_train`:
  estandarización, matriz de transición, contexto de arranque e historia de precios
  se calculan con train. Un detector evaluado con sintéticos de un generador que vio
  el tramo de test está contaminado aunque el detector no lo haya visto.
- **Circularidad.** Un generador de la familia de un detector lo favorece
  (`garch_regimen` frente a D06/D11; `var_regimen` y `gaussiano_regimen` frente a
  los modelos de cambio de régimen gaussianos). El régimen de referencia sale de
  ventanas de crisis, no de un detector, y el laboratorio debe usar generadores de
  varias familias y declararlo.
- **Semántica del régimen.** `regime = 1` sintético es «ley condicional de crisis»,
  no «pico→suelo». Un recall medido en el laboratorio no es el recall por evento del
  benchmark y no sustituye a la taxonomía histórica (p. ej. el punto ciego de 2013).
- **Memorización.** Bootstrap y jitter copian por diseño; los neuronales pueden
  hacerlo con ventanas muy solapadas de pocos episodios. Se mide con índice íntegro y
  se distingue de la infradispersión.
- **Sobreconfianza por volumen.** Cien trayectorias de diez años no añaden
  información sobre crisis que el generador no sabe producir; el error estándar que
  encogen es el del muestreo, no el de la estimación.
- **Pocas crisis.** El n efectivo del régimen de crisis es el número de episodios de
  train; cualquier diferencia entre generadores dentro de crisis debe leerse con
  banda por episodio.
- **Salvaguardas que hacen de modelo.** Techos de varianza, topes y saturaciones son
  necesarios para encadenar, pero si se activan a menudo el resultado es el tope y no
  el generador. Por eso cada `sample` deja su diagnóstico.

## Coste de implementación y librería Python recomendada

- **Remuestreo y paramétricos:** `numpy`/`scipy`; `scikit-learn` (`LedoitWolf`);
  `arch` para el punto de partida del GJR-GARCH-t. Ajuste de segundos a pocos minutos.
- **RBIG:** `numpy`/`scipy` puros, determinista.
- **Neuronales:** `PyTorch` (extra `[deep]`, el mismo de D12). MLP pequeños (menos de
  un millón de parámetros) entrenados en CPU con un solo hilo para que una semilla dé
  el mismo resultado bit a bit en cualquier máquina; minutos por generador y pista.
- **Registro perezoso:** `regimenes.sinteticos.registry.crear(nombre, **params)`
  importa cada generador bajo demanda; el paquete funciona sin torch instalado.
- **Persistencia:** en `data/sinteticos/<nombre>/pista<X>/` (no se versiona) el
  modelo (`generador.pkl`) y las trayectorias de los dos modos de régimen del
  notebook 15: `trayectorias.parquet` (cadena simulada) y
  `trayectorias_impuesto.parquet` (secuencia impuesta común). En
  `results/sinteticos/generadores/` (versionable) las fichas de ajuste y muestreo
  (`*_resumen.json`), el historial de convergencia (`*_historial.csv`) y las tablas
  de sanidad (`sanidad_*.csv`); las figuras PNG de esa carpeta no se versionan.

## Referencias

- Arjovsky, M., Chintala, S. & Bottou, L. (2017). *Wasserstein Generative Adversarial
  Networks.* ICML 2017, PMLR 70: 214–223. [synth_arjovsky2017]
- Bollerslev, T. (1990). *Modelling the Coherence in Short-Run Nominal Exchange
  Rates: A Multivariate Generalized ARCH Model.* The Review of Economics and
  Statistics 72(3), 498–505. DOI: 10.2307/2109358 [synth_bollerslev1990]
- Buehler, H., Horvath, B., Lyons, T., Perez Arribas, I. & Wood, B. (2020). *A
  Data-driven Market Simulator for Small Data Environments.* arXiv:2006.14498.
  [synth_buehler2020]
- Chen, S. S. & Gopinath, R. A. (2000). *Gaussianization.* Advances in Neural
  Information Processing Systems 13. [synth_chengopinath2000]
- Cont, R. (2001). *Empirical Properties of Asset Returns: Stylized Facts and
  Statistical Issues.* Quantitative Finance 1(2), 223–236. DOI: 10.1080/713665670
  [synth_cont2001]
- Cont, R., Cucuringu, M., Xu, R. & Zhang, C. (2026). *Tail-GAN: Learning to Simulate
  Tail Risk Scenarios.* Management Science 72(4), 2917–2936. DOI:
  10.1287/mnsc.2023.00936 [synth_conttailgan2026]
- Desai, A., Freeman, C., Wang, Z. & Beaver, I. (2021). *TimeVAE: A Variational
  Auto-Encoder for Multivariate Time Series Generation.* arXiv:2111.08095.
  [synth_desai2021]
- Esteban, C., Hyland, S. L. & Rätsch, G. (2017). *Real-valued (Medical) Time Series
  Generation with Recurrent Conditional GANs.* arXiv:1706.02633. [synth_esteban2017]
- Goodfellow, I. et al. (2014). *Generative Adversarial Nets.* Advances in Neural
  Information Processing Systems 27. [synth_goodfellow2014]
- Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V. & Courville, A. (2017).
  *Improved Training of Wasserstein GANs.* Advances in Neural Information Processing
  Systems 30. arXiv:1704.00028. [synth_gulrajani2017]
- Higgins, I. et al. (2017). *beta-VAE: Learning Basic Visual Concepts with a
  Constrained Variational Framework.* ICLR 2017. [synth_higgins2017]
- Ho, J., Jain, A. & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models.*
  Advances in Neural Information Processing Systems 33. arXiv:2006.11239.
  [synth_ho2020]
- Ho, J. & Salimans, T. (2022). *Classifier-Free Diffusion Guidance.*
  arXiv:2207.12598. [synth_hosalimans2022]
- Krolzig, H.-M. (1997). *Markov-Switching Vector Autoregressions.* Lecture Notes in
  Economics and Mathematical Systems 454, Springer. DOI: 10.1007/978-3-642-51684-9
  [synth_krolzig1997]
- Künsch, H. R. (1989). *The Jackknife and the Bootstrap for General Stationary
  Observations.* The Annals of Statistics 17(3), 1217–1241. DOI:
  10.1214/aos/1176347265 [synth_kunsch1989]
- Laparra, V., Camps-Valls, G. & Malo, J. (2011). *Iterative Gaussianization: From
  ICA to Random Rotations.* IEEE Transactions on Neural Networks 22(4), 537–549.
  DOI: 10.1109/TNN.2011.2106511 [synth_laparra2011]
- Ledoit, O. & Wolf, M. (2004). *A Well-Conditioned Estimator for Large-Dimensional
  Covariance Matrices.* Journal of Multivariate Analysis 88(2), 365–411. DOI:
  10.1016/S0047-259X(03)00096-4 [synth_ledoitwolf2004]
- Liao, S., Ni, H., Sabate-Vidales, M., Szpruch, L., Wiese, M. & Xiao, B. (2024).
  *Sig-Wasserstein GANs for Conditional Time Series Generation.* Mathematical Finance
  34(2), 622–670. DOI: 10.1111/mafi.12423 [synth_liao2024]
- Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M. & Le, M. (2023). *Flow
  Matching for Generative Modeling.* ICLR 2023. arXiv:2210.02747. [synth_lipman2023]
- Liu, X., Gong, C. & Liu, Q. (2023). *Flow Straight and Fast: Learning to Generate
  and Transfer Data with Rectified Flow.* ICLR 2023. arXiv:2209.03003.
  [synth_liu2023]
- Lopez-Paz, D. & Oquab, M. (2017). *Revisiting Classifier Two-Sample Tests.* ICLR
  2017. arXiv:1610.06545. [synth_lopezpaz2017]
- Mirza, M. & Osindero, S. (2014). *Conditional Generative Adversarial Nets.*
  arXiv:1411.1784. [synth_mirzaosindero2014]
- Nichol, A. & Dhariwal, P. (2021). *Improved Denoising Diffusion Probabilistic
  Models.* ICML 2021, PMLR 139: 8162–8171. [synth_nicholdhariwal2021]
- Politis, D. N. & Romano, J. P. (1994). *The Stationary Bootstrap.* Journal of the
  American Statistical Association 89(428), 1303–1313. DOI:
  10.1080/01621459.1994.10476870 [synth_politisromano1994]
- Sohn, K., Lee, H. & Yan, X. (2015). *Learning Structured Output Representation
  using Deep Conditional Generative Models.* Advances in Neural Information
  Processing Systems 28. [synth_sohn2015]
- Song, J., Meng, C. & Ermon, S. (2021). *Denoising Diffusion Implicit Models.* ICLR
  2021. arXiv:2010.02502. [synth_song2021]
- Tashiro, Y., Song, J., Song, Y. & Ermon, S. (2021). *CSDI: Conditional Score-based
  Diffusion Models for Probabilistic Time Series Imputation.* Advances in Neural
  Information Processing Systems 34. arXiv:2107.03502. [synth_tashiro2021]
- Wiese, M., Knobloch, R., Korn, R. & Kretschmer, P. (2020). *Quant GANs: Deep
  Generation of Financial Time Series.* Quantitative Finance 20(9), 1419–1440. DOI:
  10.1080/14697688.2020.1730426 [synth_wiese2020]
- Yoon, J., Jarrett, D. & van der Schaar, M. (2019). *Time-series Generative
  Adversarial Networks.* Advances in Neural Information Processing Systems 32.
  [synth_yoon2019]
- Yuan, X. & Qiao, Y. (2024). *Diffusion-TS: Interpretable Diffusion for General Time
  Series Generation.* ICLR 2024. arXiv:2403.01742. [synth_yuanqiao2024]

(Hamilton 1989 [hamilton1989], Kingma & Welling 2014 [nn_kingma2014], Glosten,
Jagannathan & Runkle 1993 [vol_glosten1993], Bollerslev 1987 [vol_bollerslev1987],
Gray 1996 [vol_gray1996] y Haas, Mittnik & Paolella 2004
[vol_haasmittnikpaolella2004] están ya en `references.bib` con esas claves y no se
re-declaran.)

## Variantes no implementadas

- **Régimen endógeno:** un generador cuya transición dependa del estado del mercado (probabilidades de transición variables en el tiempo) cerraría la debilidad del régimen exógeno; no implementado.
- Los solapes con otras familias (`garch_regimen` ~ D06/D11 de F5; la cadena de Markov simulada ~ F3/F4; el CVAE ~ el autoencoder D12 de F7) deben declararse al usar el laboratorio. Ningún resultado sintético se cita sin su generador y su lectura de memorización.
