"""GAN condicional (CGAN) de "siguiente bloque", entrenada como WGAN-GP.

Que hace
--------
Aprende a muestrear ``p(bloque | contexto, regimen)`` sin densidad explicita
(Goodfellow et al., 2014): un generador ``G(z, c)`` transforma ruido gaussiano
y la condicion ``c`` en el bloque de las ``largo_bloque`` sesiones siguientes,
y un critico ``D(x, c)`` compara bloques reales y generados CON la misma
condicion (Mirza y Osindero, 2014: la condicion entra en las dos redes; si el
critico no la viera no podria penalizar un bloque incoherente con su contexto o
con su regimen). ``c`` = contexto aplanado + regimen de cada dia del bloque; la
representacion (arcsinh de las colas, ancla de las columnas persistentes) es la
de ``_redes_latentes.PreparadorBloques``, la misma del CVAE. Las trayectorias
largas salen de ``encadenar``.

Que cambia respecto a la CGAN del taller previo y por que
---------------------------------------------------------
Aquella (Keras, entropia cruzada no saturante, BatchNorm en G, suavizado de
etiquetas y banda muerta sobre la precision de D, ventana de 60x20) fue
RECHAZADA: un discriminador externo la separaba con AUC 0,99, D clasificaba
casi todo como falso, las perdidas nunca se acercaron a log 2 y, sobre todo,
la perdida no distinguia exito de fracaso. Cambios, todos estandar y baratos:

- **Perdida de Wasserstein con penalizacion de gradiente** (Arjovsky et al.,
  2017; Gulrajani et al., 2017). El critico no clasifica: estima la distancia
  de Wasserstein-1 entre real y generado, que da gradiente util a G aunque las
  dos distribuciones no se solapen (la entropia cruzada satura justo ahi, que
  es lo que ocurrio en el taller). La restriccion 1-Lipschitz se impone
  penalizando ``(||grad_x D|| - 1)^2`` en interpolaciones real-generado.
- **Sin BatchNorm** en ninguna red: en G acopla las muestras del lote (y aqui
  cada muestra lleva una condicion distinta) y es incompatible con la
  penalizacion de gradiente por muestra en D.
- **n_critico pasos de D por cada paso de G** y **tasas distintas**
  (``lr_d > lr_g``): la estimacion de la distancia solo vale con el critico
  cerca de su optimo. Adam con ``beta1 = 0.5``.
- **Media movil exponencial de los pesos de G** (``ema``): se muestrea de la
  copia promediada, que amortigua la oscilacion del juego.
- **Abandono del contexto** (``abandono_contexto``): en una fraccion de las
  filas de cada lote el contexto se pone a cero, con la MISMA mascara para el
  bloque real y el generado. Sin el, G aprende la volatilidad del contexto y
  no de la etiqueta de regimen (en train van juntas), y al encadenar, donde
  el contexto es el suyo propio, genera la misma volatilidad en calma que en
  crisis (medido en la pista A: 1,11 % diario en ambos regimenes frente a
  0,83 % / 1,28 % reales). Obligarle a generar tambien sin contexto hace que
  la etiqueta importe.
- Muestreo de lotes equilibrado por clase de bloque (calma / crisis / mixto);
  real y generado comparten condicion, asi que el critico no puede usar la
  frecuencia de la condicion como atajo.

Que supone, que reproduce y que no
----------------------------------
G es determinista dado ``(z, c)``: puede emitir escalones exactos y colas sin
ruido de observacion, pero nada le obliga a usar ``z``. Con una condicion tan
informativa (21 dias de contexto) el riesgo especifico de una GAN condicional
es el colapso CONDICIONAL: G aprende una continuacion plausible por contexto e
ignora ``z``; las marginales pueden parecer correctas mientras dos
trayectorias con el mismo contexto salen casi iguales. ``diversidad_z`` lo
mide. Tampoco hay verosimilitud ni seleccion de epoca por validacion: se usa
la ultima epoca (pesos EMA).

Como leer la curva de convergencia (``historial``)
--------------------------------------------------
En una WGAN la perdida SI es interpretable, pero no como en un modelo de
verosimilitud. Una fila por epoca (una pasada del critico por los bloques):

- ``wasserstein`` = media D(real) - media D(generado) sobre los lotes del
  critico: estimacion de W1 (hasta la constante de Lipschitz). Empieza alta y
  debe DECRECER hacia una meseta baja y positiva; 0 exacto desde el principio
  es un critico que no aprende; creciente es G perdiendo.
- ``perdida_critico`` = -wasserstein + lambda * penalizacion;
  ``penalizacion_gradiente`` debe quedar pequena (<~ 0,1): si no, la
  restriccion de Lipschitz no se cumple y ``wasserstein`` no es una distancia.
- ``perdida_generador`` = -media D(G(z, c)). Su nivel absoluto no significa
  nada (el critico esta definido salvo constante); no tiene ancla en log 2.
- ``acierto_critico``: fraccion de muestras que un umbral en el punto medio
  de las dos medias separa bien; 0,5 = indistinguibles para ESTE critico, 1 =
  separacion perfecta. Sustituye a las precisiones real/falso del taller.
- ``cociente_dispersion``: desviacion entre muestras generadas / reales
  (media sobre dimensiones, G promediado, ruido fijo). << 1 = colapso de modos.
- ``diversidad_z``: desviacion entre dos muestras con la MISMA condicion y
  distinto ``z``, relativa a la desviacion real. ~0 = colapso condicional.
- ``wasserstein_val`` (solo si ``fraccion_validacion > 0``): la misma distancia
  sobre una cola cronologica que el critico no ve. Muy por debajo de
  ``wasserstein`` indica que el critico memoriza train.

Convergencia aqui significa: ``wasserstein`` en meseta, penalizacion pequena,
``cociente_dispersion`` cerca de 1 y ``diversidad_z`` claramente positiva. Aun
asi, el veredicto lo dan las metricas externas de los notebooks 16-18, no esta
curva.

Referencias
-----------
Goodfellow et al. (2014), Generative Adversarial Nets. Mirza y Osindero (2014),
Conditional Generative Adversarial Nets. Arjovsky, Chintala y Bottou (2017),
Wasserstein GAN. Gulrajani et al. (2017), Improved Training of Wasserstein
GANs. Heusel et al. (2017), GANs Trained by a Two Time-Scale Update Rule.
"""

from __future__ import annotations

import copy

import numpy as np

from regimenes.sinteticos.bloques import encadenar
from regimenes.sinteticos.neuronales._redes_latentes import (
    PreparadorBloques,
    cociente_dispersion,
    mlp,
    partir_validacion,
    probabilidades_lote,
    submuestra_diagnostico,
    torch,
)
from regimenes.sinteticos.neuronales._torch import (
    GeneradorNeuronal,
    a_numpy,
    a_tensor,
    generador_torch,
    sembrar,
)
from regimenes.sinteticos.registry import registrar

N_DIAGNOSTICO = 2048


@registrar
class CGAN(GeneradorNeuronal):
    """WGAN-GP condicional de siguiente bloque (ver docstring del modulo).

    Hiperparametros (``PARAMS``)
    ----------------------------
    largo_bloque, largo_contexto : ``L_blk`` y ``L_ctx`` en sesiones.
    dim_latente : dimension del ruido ``z``.
    ocultas_g, ocultas_d : anchuras de las capas ocultas de G y del critico.
    epocas : pasadas del critico por los bloques (``ceil(m / tam_lote)`` pasos cada una).
    tam_lote, n_critico : lote y pasos del critico por cada paso de G.
    lr_g, lr_d, beta1, beta2 : Adam de cada red.
    lambda_gp : peso de la penalizacion de gradiente.
    ema : decaimiento de la media movil de los pesos de G (0 = sin promediar).
    abandono_contexto : fraccion de filas de cada lote entrenadas sin contexto.
    fraccion_validacion : cola cronologica excluida del ajuste y usada SOLO para
        ``wasserstein_val`` (0 = sin validacion, todo train entra en el ajuste).
    equilibrio, escala_colas, curtosis_colas, umbral_ancla, blanquear, saturar_contexto :
        como en el CVAE (ver ``PreparadorBloques``).

    Atributos tras ``fit``: ``red_`` (G promediado, el que muestrea),
    ``critico_`` (para auditar el juego) y ``prep_``.
    """

    nombre = "cgan"
    familia = "neuronales"
    PARAMS = {
        "largo_bloque": 21,
        "largo_contexto": 21,
        "dim_latente": 32,
        "ocultas_g": (256, 256),
        "ocultas_d": (256, 256),
        "epocas": 150,
        "tam_lote": 256,
        "n_critico": 3,
        "lr_g": 3e-4,
        "lr_d": 6e-4,
        "beta1": 0.5,
        "beta2": 0.9,
        "lambda_gp": 10.0,
        "ema": 0.99,
        "abandono_contexto": 0.3,
        "fraccion_validacion": 0.0,
        "equilibrio": 0.5,
        "escala_colas": 2.0,
        "curtosis_colas": 3.0,
        "umbral_ancla": 0.9,
        "blanquear": True,
        "saturar_contexto": True,
    }
    PARAMS_RAPIDOS = {"epocas": 60, "tam_lote": 64, "ocultas_g": (64, 64), "ocultas_d": (64, 64), "ema": 0.98}

    # -------------------------------------------------------------------- fit
    def _fit(self, X, reg, fechas):
        g = sembrar(self.random_state)
        rng = self.rng_ajuste()
        prep = self.prep_ = PreparadorBloques(
            self.largo_bloque, self.largo_contexto, self.n_regimenes_, self.escala_colas, self.umbral_ancla,
            self.curtosis_colas, self.blanquear, self.saturar_contexto,
        ).ajustar(X)
        total = prep.lb + prep.lc
        partes = partir_validacion(len(X), self.fraccion_validacion, total, total)
        validacion = None
        if partes is not None:
            cabeza, cola = partes
            validacion = prep.pares(X[cola], reg[cola], fechas[cola])
            X, reg, fechas = X[cabeza], reg[cabeza], fechas[cabeza]
        cond, obj, reg_blk = prep.pares(X, reg, fechas)
        C, Xo = a_tensor(cond), a_tensor(obj)
        m, J = len(obj), int(self.dim_latente)

        gen = mlp(J + cond.shape[1], self.ocultas_g, obj.shape[1], activacion="leaky")
        critico = mlp(obj.shape[1] + cond.shape[1], self.ocultas_d, 1, activacion="leaky")
        gen_ema = copy.deepcopy(gen).requires_grad_(False)
        betas = (float(self.beta1), float(self.beta2))
        opt_g = torch.optim.Adam(gen.parameters(), lr=float(self.lr_g), betas=betas)
        opt_d = torch.optim.Adam(critico.parameters(), lr=float(self.lr_d), betas=betas)

        p = probabilidades_lote(reg_blk, self.equilibrio)
        tam = min(int(self.tam_lote), m)
        pasos = max(1, -(-m // tam))
        n_critico = max(int(self.n_critico), 1)
        lam, decaimiento = float(self.lambda_gp), float(self.ema)

        # diagnostico: subconjunto y ruido fijos para que las curvas sean comparables entre epocas
        diag = torch.as_tensor(submuestra_diagnostico(m, N_DIAGNOSTICO, self.random_state))
        gd = torch.Generator().manual_seed(0)
        z_a, z_b = torch.randn(len(diag), J, generator=gd), torch.randn(len(diag), J, generator=gd)
        desv_real = Xo[diag].std(dim=0)
        validas = desv_real > 1e-6
        if validacion is not None:
            Cv, Xv = a_tensor(validacion[0]), a_tensor(validacion[1])
            z_v = torch.randn(len(Xv), J, generator=gd)

        n_ctx = prep.lc * prep.d
        abandono = float(self.abandono_contexto)

        def condicion_lote(idx):
            """Condicion del lote; con ``abandono_contexto`` se anula el contexto de parte de las filas."""
            c = C[idx]
            if abandono > 0 and n_ctx:
                c = c.clone()
                c[torch.rand(len(c), generator=g) < abandono, :n_ctx] = 0.0
            return c

        contador, pasos_g = 0, 0
        ultima_g = float("nan")
        for epoca in range(int(self.epocas)):
            suma = np.zeros(4)
            sumas_g: list[float] = []
            for _ in range(pasos):
                idx = torch.as_tensor(rng.choice(m, size=tam, p=p))
                real, c = Xo[idx], condicion_lote(idx)
                with torch.no_grad():
                    falso = gen(torch.cat([torch.randn(tam, J, generator=g), c], dim=1))
                d_real = critico(torch.cat([real, c], dim=1))
                d_falso = critico(torch.cat([falso, c], dim=1))
                eps = torch.rand(tam, 1, generator=g)
                mezcla = (eps * real + (1.0 - eps) * falso).requires_grad_(True)
                d_mezcla = critico(torch.cat([mezcla, c], dim=1))
                (grad,) = torch.autograd.grad(d_mezcla.sum(), mezcla, create_graph=True)
                penal = ((grad.norm(2, dim=1) - 1.0) ** 2).mean()
                distancia = d_real.mean() - d_falso.mean()
                perdida_d = -distancia + lam * penal
                opt_d.zero_grad()
                perdida_d.backward()
                opt_d.step()
                with torch.no_grad():
                    umbral = 0.5 * (d_real.mean() + d_falso.mean())
                    acierto = 0.5 * ((d_real > umbral).float().mean() + (d_falso < umbral).float().mean())
                suma += (float(perdida_d.detach()), float(distancia.detach()), float(penal.detach()), float(acierto))

                contador += 1
                if contador % n_critico == 0:
                    idx = torch.as_tensor(rng.choice(m, size=tam, p=p))
                    c = condicion_lote(idx)
                    generado = gen(torch.cat([torch.randn(tam, J, generator=g), c], dim=1))
                    perdida_g = -critico(torch.cat([generado, c], dim=1)).mean()
                    opt_g.zero_grad()
                    perdida_g.backward()
                    opt_g.step()
                    pasos_g += 1
                    sumas_g.append(float(perdida_g.detach()))
                    # arranque rapido de la media movil: al principio copia a G
                    d_t = min(decaimiento, (1.0 + pasos_g) / (10.0 + pasos_g))
                    with torch.no_grad():
                        for p_ema, p_g in zip(gen_ema.parameters(), gen.parameters()):
                            p_ema.mul_(d_t).add_(p_g, alpha=1.0 - d_t)

            media = suma / pasos
            ultima_g = float(np.mean(sumas_g)) if sumas_g else ultima_g
            if not np.isfinite(media).all():
                raise FloatingPointError(f"cgan: la perdida del critico diverge en la epoca {epoca}.")
            fila = {"epoca": epoca, "perdida_critico": media[0], "perdida_generador": ultima_g,
                    "wasserstein": media[1], "penalizacion_gradiente": media[2], "acierto_critico": media[3]}
            with torch.no_grad():
                f_a = gen_ema(torch.cat([z_a, C[diag]], dim=1))
                f_b = gen_ema(torch.cat([z_b, C[diag]], dim=1))
                fila["cociente_dispersion"] = cociente_dispersion(f_a, Xo[diag])
                entre_z = (f_a - f_b).std(dim=0) / np.sqrt(2.0)
                fila["diversidad_z"] = float((entre_z[validas] / desv_real[validas]).mean())
                fila["wasserstein_val"] = np.nan
                if validacion is not None:
                    f_v = gen_ema(torch.cat([z_v, Cv], dim=1))
                    fila["wasserstein_val"] = float(
                        critico(torch.cat([Xv, Cv], dim=1)).mean() - critico(torch.cat([f_v, Cv], dim=1)).mean()
                    )
            self.registrar(**fila)

        self.red_ = gen_ema.eval()
        self.critico_ = critico.eval().requires_grad_(False)

    # ----------------------------------------------------------------- sample
    def _sample(self, reg, rng, contexto):
        g = generador_torch(rng)
        red, prep = self.red_, self.prep_
        J = int(self.dim_latente)

        def generar_bloque(ctx, reg_bloque, _rng):
            c = a_tensor(prep.condicion(ctx, reg_bloque))
            with torch.no_grad():
                x = red(torch.cat([torch.randn(len(c), J, generator=g), c], dim=1))
            return prep.a_bloque(a_numpy(x), ctx)

        return encadenar(generar_bloque, reg, contexto, prep.lb, prep.lc, rng)
