"""Autoencoder variacional condicional (CVAE) de "siguiente bloque".

Que hace
--------
Aprende ``p(bloque | contexto, regimen)``: el bloque son las ``largo_bloque``
sesiones siguientes (aplanadas, ``L_blk * d`` dimensiones), el contexto las
``largo_contexto`` anteriores y el regimen el de CADA dia del bloque. Modelo de
variable latente (Kingma y Welling, 2014) condicionado en encoder y decoder
(Sohn, Lee y Yan, 2015):

    encoder  q(z | x, c) = N(mu(x, c), diag sigma^2(x, c))
    prior    p(z | c)    = N(0, I)
    decoder  p(x | z, c) = N(m(z, c), diag s^2(z, c))      <- varianza APRENDIDA

Las trayectorias largas salen de ``encadenar``: se muestrea un bloque, se anade
al contexto y se repite, en lote sobre todas las trayectorias. La
representacion (compresion arcsinh de las colas, ancla de las columnas
persistentes, condicion) es la de ``_redes_latentes.PreparadorBloques`` y es la
misma que usa la CGAN.

Que corrige respecto al CVAE del taller previo (Keras, ventana de 60x20)
------------------------------------------------------------------------
- **Infradispersion**. Aquel decoder tenia varianza fija y se muestreaba solo
  su media: la desviacion sintetica era 0,65 de la real (0,45 en crisis),
  porque la media condicional de un decoder gaussiano promedia todo lo que
  ``z`` no explica. Aqui el decoder emite media y log-varianza por dimension
  (heterocedastico, log-varianza acotada con una sigmoide a
  ``[log_var_min, log_var_max]`` para que la verosimilitud no diverja) y
  ``_sample`` muestrea ``x = m + s * eps``: el ruido de observacion forma parte
  del modelo. ``ruido_observacion=False`` reproduce el defecto del taller. El
  historial lleva ``cociente_dispersion`` (con ruido) y
  ``cociente_dispersion_media`` (solo la media) para que la diferencia se vea.
- **Sobreajuste** (alli: reconstruccion 278 en train frente a 1068 en
  validacion). La cola cronologica de train (``fraccion_validacion``, separada
  por un hueco de ``L_ctx + L_blk`` filas) no entra en el ajuste de la fase 1 y
  fija la mejor epoca por ELBO de validacion, con parada temprana
  (``paciencia``). Ademas se regulariza con ``dropout`` (0,1) y un
  ``weight_decay`` pequeno: en la pista A el dropout baja el ELBO de validacion
  de ~63 a ~33 nats por bloque y retrasa la mejor epoca de la ~35 a la ~110. Con ``reajuste_completo=True`` (defecto) la fase 2 reentrena
  desde cero con TODO train durante ese numero de epocas: la cola contiene lo
  mas reciente (y el ultimo episodio de crisis) y no debe perderse. Con
  ``False`` se conservan los pesos de la mejor epoca de la fase 1.
- Se mantienen la rampa lineal de beta (Bowman et al., 2016; el peso es el de
  beta-VAE, Higgins et al., 2017), los free bits (Kingma et al., 2016) y el
  recorte de gradiente. Ambos terminos del ELBO se SUMAN sobre dimensiones y
  se promedian solo sobre el lote; con varianza aprendida ``beta = 1`` es el
  ELBO exacto y ``beta`` deja de ser "la varianza del decoder".
- Desaparecen la PCA y la etiqueta unica por ventana: la condicion lleva el
  regimen de cada dia, asi que un bloque puede contener una transicion.

Que supone
----------
- Gaussiana diagonal condicionada a ``z``: la dependencia entre columnas y
  entre dias dentro del bloque solo puede pasar por ``z`` y por la media. Si
  el latente colapsa (``unidades_activas`` -> 0) el modelo degenera en ruido
  independiente por dia alrededor de una media que solo depende del contexto.
- Markov de orden ``L_ctx`` entre bloques; regimen exogeno.
- La estandarizacion (la de ``GeneradorBase`` y la re-estandarizacion tras el
  arcsinh) usa todo train, tambien la cola de validacion: fuga despreciable
  (dos momentos por columna) que solo afecta a la eleccion de la epoca.

Que reproduce y que no
----------------------
Reproduce la escala por regimen, la persistencia de las columnas con ancla y,
gracias al arcsinh, colas mas pesadas que una gaussiana (la inversa ``sinh``
estira el ruido). No garantiza los escalones exactos de las columnas mensuales
(el ruido de observacion anade una vibracion diaria del tamano de ``s``, que
en esas columnas cae hacia ``exp(log_var_min / 2)``), ni el agrupamiento de
volatilidad dentro del bloque mas alla de lo que capture ``z``.

Como leer la curva de convergencia (``historial``)
--------------------------------------------------
Una fila por epoca: ``fase`` (1 = ajuste con validacion, 2 = reajuste con todo
train), ``beta``, ``perdida`` (la optimizada), ``reconstruccion`` (log-
verosimilitud negativa en nats por bloque; con varianza aprendida puede ser
NEGATIVA, no es un error cuadratico), ``kl`` (nats, sin el suelo de free
bits), ``val_reconstruccion``, ``val_kl``, ``val_elbo`` (= suma, sin beta; NaN
en la fase 2), ``unidades_activas`` (dimensiones latentes con KL > 0,01 nats),
``sigma_decoder`` (desviacion media del ruido de observacion) y los dos
cocientes de dispersion. Un ajuste sano: ``reconstruccion`` baja y se aplana;
``kl`` SUBE desde ~0 y se estabiliza en una meseta positiva; ``val_elbo`` baja
y despues sube (ahi esta ``mejor_epoca_``); ``cociente_dispersion`` cerca de 1.
``kl`` ~ 0 con ``unidades_activas`` = 0 es colapso del posterior.

Referencias
-----------
Kingma y Welling (2014), Auto-Encoding Variational Bayes. Sohn, Lee y Yan
(2015), Learning Structured Output Representation using Deep Conditional
Generative Models. Higgins et al. (2017), beta-VAE. Bowman et al. (2016),
Generating Sentences from a Continuous Space. Kingma et al. (2016), Improved
Variational Inference with Inverse Autoregressive Flow (free bits).
"""

from __future__ import annotations

import copy
import math
import warnings

import numpy as np

from regimenes.sinteticos.bloques import encadenar
from regimenes.sinteticos.neuronales._redes_latentes import (
    PreparadorBloques,
    cociente_dispersion,
    mlp,
    nn,
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

UMBRAL_UNIDAD_ACTIVA = 0.01
N_DIAGNOSTICO = 2048
_LOG_2PI = math.log(2.0 * math.pi)


class RedCVAE(nn.Module):
    """Encoder ``[x, c] -> (mu, log sigma^2)`` y decoder ``[z, c] -> (m, log s^2)``."""

    def __init__(self, dim_x, dim_c, dim_latente, ocultas, dropout, log_var_min, log_var_max):
        super().__init__()
        self.dim_x, self.dim_latente = int(dim_x), int(dim_latente)
        self.lv_min, self.lv_max = float(log_var_min), float(log_var_max)
        self.encoder = mlp(dim_x + dim_c, ocultas, 2 * dim_latente, dropout=dropout)
        self.decoder = mlp(dim_latente + dim_c, tuple(ocultas)[::-1], 2 * dim_x, dropout=dropout)

    def codificar(self, x, c):
        mu, log_var = self.encoder(torch.cat([x, c], dim=1)).chunk(2, dim=1)
        return mu, log_var.clamp(-12.0, 6.0)

    def decodificar(self, z, c):
        media, cruda = self.decoder(torch.cat([z, c], dim=1)).chunk(2, dim=1)
        return media, self.lv_min + (self.lv_max - self.lv_min) * torch.sigmoid(cruda)

    def elbo(self, x, c, ruido):
        """``(nll por muestra (n,), kl por muestra y dimension (n, J), log_var del decoder)``."""
        mu, log_var = self.codificar(x, c)
        z = mu + torch.exp(0.5 * log_var) * ruido
        media, lv_x = self.decodificar(z, c)
        nll = 0.5 * (lv_x + (x - media) ** 2 * torch.exp(-lv_x) + _LOG_2PI).sum(dim=1)
        kl = -0.5 * (1.0 + log_var - mu**2 - torch.exp(log_var))
        return nll, kl, lv_x


@registrar
class CVAE(GeneradorNeuronal):
    """CVAE de siguiente bloque con decoder heterocedastico (ver docstring del modulo).

    Hiperparametros (``PARAMS``)
    ----------------------------
    largo_bloque, largo_contexto : ``L_blk`` y ``L_ctx`` en sesiones.
    dim_latente : dimension ``J`` de ``z``.
    ocultas : anchuras del encoder (el decoder es su espejo).
    beta, epocas_rampa_beta : peso final del KL y epocas de su rampa lineal desde 0.
    free_bits : suelo en nats por dimension latente (0 lo desactiva).
    epocas, paciencia : maximo de epocas y epocas sin mejora de ``val_elbo`` antes de parar.
    tam_lote, lr, clip, weight_decay, dropout : optimizacion (Adam) y regularizacion.
    fraccion_validacion : cola cronologica de train reservada (0 = sin validacion).
    reajuste_completo : reentrenar con todo train las epocas elegidas por validacion.
    equilibrio : 0 = lotes con la frecuencia empirica; 1 = calma / crisis / mixto a partes iguales.
    escala_colas, curtosis_colas : ``c`` de ``c * arcsinh(x / c)`` (``None`` = sin transformar)
        y exceso de curtosis a partir del cual se aplica a una columna sin ancla.
    umbral_ancla : autocorrelacion a partir de la cual una columna se modela en diferencias.
    blanquear, saturar_contexto : ver ``PreparadorBloques``.
    log_var_min, log_var_max : cotas de la log-varianza del decoder.
    ruido_observacion : muestrear ``m + s * eps`` (True) o solo ``m`` (False, el defecto del taller).

    Atributos tras ``fit``: ``red_`` (RedCVAE), ``prep_`` (PreparadorBloques),
    ``mejor_epoca_`` (indice de la mejor epoca de la fase 1, o ``None`` sin
    validacion) y ``val_elbo_invalido_`` (True si ``val_elbo`` no fue finito en
    ninguna epoca: se toma entonces la ultima epoca de la fase 1 y queda anotado
    en ``resumen()``).
    """

    nombre = "cvae"
    familia = "neuronales"
    PARAMS = {
        "largo_bloque": 21,
        "largo_contexto": 21,
        "dim_latente": 16,
        "ocultas": (256, 256),
        "beta": 1.0,
        "epocas_rampa_beta": 20,
        "free_bits": 0.05,
        "epocas": 120,
        "paciencia": 25,
        "tam_lote": 256,
        "lr": 1e-3,
        "clip": 5.0,
        "weight_decay": 1e-5,
        "dropout": 0.1,
        "fraccion_validacion": 0.15,
        "reajuste_completo": True,
        "equilibrio": 0.5,
        "escala_colas": 2.0,
        "curtosis_colas": 3.0,
        "umbral_ancla": 0.9,
        "blanquear": True,
        "saturar_contexto": True,
        "log_var_min": -7.0,
        "log_var_max": 2.0,
        "ruido_observacion": True,
    }
    PARAMS_RAPIDOS = {"epocas": 30, "paciencia": 10, "epocas_rampa_beta": 5, "ocultas": (64, 64), "dim_latente": 8}

    # -------------------------------------------------------------------- fit
    def _fit(self, X, reg, fechas):
        self.prep_ = PreparadorBloques(
            self.largo_bloque, self.largo_contexto, self.n_regimenes_, self.escala_colas, self.umbral_ancla,
            self.curtosis_colas, self.blanquear, self.saturar_contexto,
        ).ajustar(X)
        total = self.prep_.lb + self.prep_.lc
        partes = partir_validacion(len(X), self.fraccion_validacion, total, total)
        self.mejor_epoca_ = None
        self.val_elbo_invalido_ = False
        if partes is None:
            self.red_ = self._entrenar(self.prep_.pares(X, reg, fechas), None, int(self.epocas), fase=1)
            return
        cabeza, cola = partes
        entreno = self.prep_.pares(X[cabeza], reg[cabeza], fechas[cabeza])
        validacion = self.prep_.pares(X[cola], reg[cola], fechas[cola])
        red = self._entrenar(entreno, validacion, int(self.epocas), fase=1)
        if self.reajuste_completo:
            red = self._entrenar(self.prep_.pares(X, reg, fechas), None, self.mejor_epoca_ + 1, fase=2)
        self.red_ = red

    def _beta_en(self, epoca: int) -> float:
        if int(self.epocas_rampa_beta) <= 0:
            return float(self.beta)
        return float(self.beta) * min(1.0, (epoca + 1) / int(self.epocas_rampa_beta))

    def _entrenar(self, datos, validacion, epocas: int, fase: int):
        """Entrena una red nueva; con ``validacion`` devuelve los pesos de la mejor epoca."""
        g = sembrar(self.random_state)
        rng = self.rng_ajuste()
        cond, obj, reg_blk = datos
        C, Xo = a_tensor(cond), a_tensor(obj)
        m, J = len(obj), int(self.dim_latente)
        red = RedCVAE(
            obj.shape[1], cond.shape[1], J, tuple(self.ocultas), self.dropout, self.log_var_min, self.log_var_max
        )
        parametros = list(red.parameters())
        optim = torch.optim.Adam(parametros, lr=float(self.lr), weight_decay=float(self.weight_decay))
        p = probabilidades_lote(reg_blk, self.equilibrio)
        tam = min(int(self.tam_lote), m)
        pasos = max(1, -(-m // tam))
        diag = torch.as_tensor(submuestra_diagnostico(m, N_DIAGNOSTICO, self.random_state))
        if validacion is not None:
            Cv, Xv = a_tensor(validacion[0]), a_tensor(validacion[1])
        suelo = float(self.free_bits)
        mejor_valor, mejor_estado, sin_mejora = np.inf, None, 0

        for epoca in range(int(epocas)):
            beta = self._beta_en(epoca)
            red.train()
            suma = np.zeros(3)
            for _ in range(pasos):
                idx = torch.as_tensor(rng.choice(m, size=tam, p=p))
                optim.zero_grad()
                nll, kl, _ = red.elbo(Xo[idx], C[idx], torch.randn(tam, J, generator=g))
                kl_dim = kl.mean(dim=0)
                recon = nll.mean()
                perdida = recon + beta * kl_dim.clamp(min=suelo).sum()
                perdida.backward()
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(parametros, float(self.clip))
                optim.step()
                suma += (float(perdida.detach()), float(recon.detach()), float(kl_dim.sum().detach()))
            media = suma / pasos
            if not np.isfinite(media).all():
                raise FloatingPointError(f"cvae: la perdida diverge en la epoca {epoca} (fase {fase}).")

            fila = {"epoca": epoca, "fase": fase, "beta": beta, "perdida": media[0],
                    "reconstruccion": media[1], "kl": media[2],
                    "val_reconstruccion": np.nan, "val_kl": np.nan, "val_elbo": np.nan}
            red.eval()
            with torch.no_grad():
                # ruido comun entre epocas: las curvas de validacion y de diagnostico
                # cambian por el modelo, no por el sorteo
                gd = torch.Generator().manual_seed(0)
                if validacion is not None:
                    nll_v, kl_v, _ = red.elbo(Xv, Cv, torch.randn(len(Xv), J, generator=gd))
                    fila["val_reconstruccion"] = float(nll_v.mean())
                    fila["val_kl"] = float(kl_v.sum(dim=1).mean())
                    fila["val_elbo"] = fila["val_reconstruccion"] + fila["val_kl"]
                mu, log_var = red.codificar(Xo[diag], C[diag])
                kl_d = (-0.5 * (1.0 + log_var - mu**2 - torch.exp(log_var))).mean(dim=0)
                m_x, lv_x = red.decodificar(torch.randn(len(diag), J, generator=gd), C[diag])
                s_x = torch.exp(0.5 * lv_x)
                muestra = m_x + s_x * torch.randn(m_x.shape, generator=gd)
                fila["unidades_activas"] = int((kl_d > UMBRAL_UNIDAD_ACTIVA).sum())
                fila["sigma_decoder"] = float(s_x.mean())
                fila["cociente_dispersion"] = cociente_dispersion(muestra, Xo[diag])
                fila["cociente_dispersion_media"] = cociente_dispersion(m_x, Xo[diag])
            self.registrar(**fila)

            if validacion is not None:
                if fila["val_elbo"] < mejor_valor:
                    mejor_valor, sin_mejora = fila["val_elbo"], 0
                    self.mejor_epoca_ = epoca
                    mejor_estado = copy.deepcopy(red.state_dict())
                else:
                    sin_mejora += 1
                    if self.paciencia and sin_mejora >= int(self.paciencia) and epoca >= int(self.epocas_rampa_beta):
                        break
        if mejor_estado is not None:
            red.load_state_dict(mejor_estado)
        elif validacion is not None:
            # ``val_elbo`` no fue finito en ninguna epoca (NaN no es menor que nada): sin criterio
            # de seleccion se toma la ultima epoca ejecutada y se deja constancia.
            self.mejor_epoca_ = int(epoca)
            self.val_elbo_invalido_ = True
            warnings.warn(
                f"cvae: val_elbo no es finito en ninguna epoca; se usa la ultima ({epoca}).",
                RuntimeWarning, stacklevel=2,
            )
        return red.eval()

    def resumen(self):
        """Ficha de ``GeneradorBase`` mas ``mejor_epoca`` y ``val_elbo_invalido``."""
        ficha = super().resumen()
        ficha["mejor_epoca"] = self.mejor_epoca_
        ficha["val_elbo_invalido"] = bool(getattr(self, "val_elbo_invalido_", False))
        return ficha

    # ----------------------------------------------------------------- sample
    def _sample(self, reg, rng, contexto):
        g = generador_torch(rng)
        red, prep = self.red_.eval(), self.prep_

        def generar_bloque(ctx, reg_bloque, _rng):
            c = a_tensor(prep.condicion(ctx, reg_bloque))
            with torch.no_grad():
                z = torch.randn(len(c), red.dim_latente, generator=g)
                x, log_var = red.decodificar(z, c)
                if self.ruido_observacion:
                    x = x + torch.exp(0.5 * log_var) * torch.randn(x.shape, generator=g)
            return prep.a_bloque(a_numpy(x), ctx)

        return encadenar(generar_bloque, reg, contexto, prep.lb, prep.lc, rng)
