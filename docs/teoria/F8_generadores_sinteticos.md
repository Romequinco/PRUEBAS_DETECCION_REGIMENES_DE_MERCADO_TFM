# F8 — Generadores de datos sintéticos

<!-- BEGIN nota_v2 -->
> **Origen y ubicación.** Ficha nueva de la fase S (sin equivalente en la Capa 1). Bibliografía: [`F8_generadores_sinteticos.bib`](F8_generadores_sinteticos.bib) (todas sus claves están ya fusionadas en `docs/references.bib`). Código: `src/regimenes/sinteticos/` (un fichero por generador en `parametricos/` y `neuronales/`). Configuración: [`configs/sinteticos.yaml`](../../configs/sinteticos.yaml). Notebook: [`notebooks/15_sinteticos_generadores.ipynb`](../../notebooks/15_sinteticos_generadores.ipynb). Esta ficha describe **qué hace cada generador y qué no puede hacer por construcción**; no contiene resultados: las cifras de ajuste y de muestreo salen del notebook 15 y la validación, del 16 (pendiente).
<!-- END nota_v2 -->

> Familia: generadores de trayectorias multivariantes de features **con la
> secuencia de régimen conocida**. Núcleo: seis generadores paramétricos (jitter,
> bootstrap por régimen, gaussiano por régimen, VAR por régimen, GJR-GARCH por
> régimen, RBIG) y cuatro neuronales (flow matching, difusión, CVAE, CGAN), todos
> detrás de la misma interfaz `Generador` (`fit` / `sample` / `name`).
> Aviso de honestidad: un generador **no crea información** sobre crisis que no
> esté en su tramo de entrenamiento; con 8 episodios de crisis en el ajuste de la
> pista A, lo que aquí se fabrica son variaciones sobre esos episodios, no crisis
> nuevas. Distinguimos abajo lo que cada modelo reproduce por construcción de lo
> que solo puede decidir la validación (notebook 16, pendiente).

## Para qué sirven en este TFM

Con 17 crisis evaluables en la pista A y 9 en la pista B, el banco de pruebas tiene
poca potencia estadística para separar detectores (ver la limitación de
significancia de la Capa 1 en `docs/historia/capa1/memoria/99_conclusions.md`).
Los generadores sintéticos se usan para tres cosas, por este orden de fiabilidad:

1. **Laboratorio con verdad conocida** (notebook `17`): trayectorias cuya cadena de
   regímenes se conoce exactamente, para medir recall y precisión de los detectores
   sin la ambigüedad del etiquetado histórico.
2. **Aumento de datos** (notebook `18`): entrenar detectores con más episodios de
   crisis y evaluarlos en real (TSTR, *train on synthetic, test on real*
   [synth_esteban2017]) sin tocar el test real.
3. **Pruebas de estrés** de la máquina de fusión y del *pseudo-live* (notebook `20`).

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
**no es el de este TFM** y sus cifras no se trasladan, pero sus lecciones fijan cómo
se leerá el notebook 16:

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

## Criterios de validación (pendientes: notebook 16)

`regimenes.sinteticos.validacion` contiene hoy **solo las firmas** de cinco funciones
(levantan `NotImplementedError`); su lógica se implementará en
`16_sinteticos_validacion`. Son las de `validacion.metricas` de
`configs/sinteticos.yaml`:

| Criterio | Función | Qué comparará |
|---|---|---|
| Fidelidad marginal | `fidelidad_marginal` | distribución por feature y por régimen: momentos, cuantiles de cola, KS |
| Fidelidad de dependencia | `fidelidad_dependencia` | ACF de retornos y de `\|r\|`, correlaciones cruzadas por régimen |
| Fidelidad de regímenes | `fidelidad_regimenes` | duraciones y matriz de transición real frente a sintética |
| Utilidad (TSTR) | `utilidad_tstr` | detector ajustado en sintético y evaluado en real con el walk-forward y el ranking ADR-003 [synth_esteban2017] |
| Memorización | `memorizacion` | distancia al vecino real más cercano, con índice de vecindad íntegro |

Complemento previsto: un discriminador real/sintético (*classifier two-sample test*
[synth_lopezpaz2017]). Las comparaciones contra el tramo real posterior al corte se
hacen por distribución o por posición, nunca cruzando fechas.

**Checklist de hechos estilizados** (Cont [synth_cont2001]; se contrastan los
aplicables a datos diarios sin volumen):

| Hecho | Contraste | Quién puede reproducirlo por construcción |
|---|---|---|
| Ausencia de autocorrelación lineal en retornos | ACF de `r_t` | todos |
| Colas pesadas incondicionales | cuantiles de cola (la curtosis depende de un solo dato) | remuestreo, RBIG, GARCH-t; el gaussiano solo por mezcla |
| Asimetría ganancia/pérdida | asimetría muestral | remuestreo, RBIG; no el GARCH-t simétrico |
| Agrupamiento de volatilidad | ACF de `\|r_t\|` y de `r_t²` | GARCH; remuestreo y bloques solo hasta el largo del bloque; no gaussiano ni VAR dentro de régimen |
| Colas pesadas condicionales | curtosis de los residuos estandarizados | GARCH-t; dudoso en neuronales (antecedente del taller) |
| Decaimiento lento de la ACF de `\|r_t\|` | pendiente log-log | ninguno por construcción (GARCH(1,1) decae geométricamente) |
| Efecto apalancamiento | `corr(r_t, r²_{t+τ})`, τ > 0 | GARCH (término GJR); remuestreo dentro del bloque |
| Gaussianidad por agregación | curtosis frente al horizonte | poco discriminante por sí sola |
| Correlaciones que cambian en crisis | correlación por régimen | todos los condicionados a régimen |

Los contrastes se leen contra bandas de remuestreo de la serie real, no contra un
valor puntual, y en crisis por episodio.

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

## Candidatas adicionales (para el sintetizador)

- **Solape con F5 (GARCH):** `garch_regimen` es el generador gemelo de D06/D11; al
  usarlo en el laboratorio hay que declarar la ventaja de esos detectores.
- **Solape con F3/F4 (HMM, Markov-Switching):** la cadena de regímenes simulada es
  una cadena de Markov de primer orden (o semi-Markov con duraciones empíricas, el
  análogo del HSMM de D13); los detectores de esas familias asumen justo esa
  estructura.
- **Solape con F7 (redes):** el CVAE comparte herramienta con el autoencoder de D12;
  la advertencia de F7 sobre muestra pequeña aplica aquí con más fuerza.
- **Régimen endógeno:** un generador en el que la transición dependa del estado del
  mercado (probabilidades de transición variables en el tiempo) cerraría la
  debilidad de régimen exógeno; queda como extensión.
- (criterio del equipo) Ningún resultado del laboratorio sintético se cita sin el
  generador que lo produjo y sin su lectura de memorización al lado.
