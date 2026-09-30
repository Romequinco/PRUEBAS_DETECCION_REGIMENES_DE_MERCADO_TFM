"""GJR-GARCH(1,1)-t con regimen observado para el mercado + VAR(1) condicionado para el resto.

Que hace
--------
Es el unico generador parametrico del catalogo que reproduce, DENTRO de cada
regimen, los tres hechos estilizados del retorno diario de renta variable:
agrupamiento de volatilidad, apalancamiento (las caidas suben la volatilidad mas
que las subidas) y colas pesadas. El panel se parte en dos bloques:

1. **Factor de mercado** (una columna: ``SP500_ret`` si existe; si no, la primera
   columna o la que diga ``columna_mercado``). Con ``s_t`` el regimen del dia t::

       r_t = mu[s_t] + e_t,        e_t = sqrt(h_t) * z_t,   z_t ~ t_nu[s_t] estandarizada
       h_t = omega[s_t] + (alfa[s_t] + gamma[s_t] * 1{e_{t-1} < 0}) * e_{t-1}^2 + beta[s_t] * h_{t-1}

   Es el GJR-GARCH de Glosten-Jagannathan-Runkle (1993) con innovacion t de
   Student (Bollerslev 1987) y TODOS los parametros dependientes del regimen. La
   varianza condicional es una unica recursion que atraviesa los cambios de
   regimen: al entrar en crisis ``h_t`` hereda el valor de calma y converge al
   nivel de crisis al ritmo ``1 - persistencia``.

2. **Resto de columnas**: VAR(1) por regimen con el mercado contemporaneo como
   regresor exogeno::

       y_t = c[s_t] + A[s_t] x_{t-1} + b[s_t] m_t + u_t

   ``x_{t-1}`` es el panel completo retardado (incluido el mercado), ``m_t`` el
   retorno de mercado del dia (o su shock estandarizado ``z_t``, parametro
   ``regresor_mercado``) y ``u_t`` un residuo remuestreado por filas de los
   residuos del regimen (o gaussiano). Las columnas heredan asi del mercado su
   heterocedasticidad y conservan su propia memoria y su correlacion mutua.

Como se ajusta (y por que asi)
------------------------------
El regimen es OBSERVADO (no latente), asi que no hay la dependencia de la senda
que obliga a Gray (1996) a colapsar la varianza o a Haas-Mittnik-Paolella (2004)
a llevar K recursiones en paralelo: con ``s_t`` conocido la recursion de arriba
es unica y la verosimilitud es exacta. El ajuste tiene dos etapas:

a. ``arch`` ajusta un GJR-GARCH-t por regimen sobre sus rachas concatenadas. Es
   una aproximacion (en cada empalme la varianza arrastra el final de la racha
   anterior del mismo regimen, separada quiza por anos) y solo se usa como punto
   de partida y como respaldo.
b. ``ajuste="conjunto"`` (defecto): maxima verosimilitud exacta de los
   parametros de todos los regimenes a la vez sobre la serie contigua, con la
   MISMA recursion que luego se simula (SLSQP, persistencia
   ``alfa + gamma/2 + beta <= persistencia_max`` por regimen). Se acepta solo si
   mejora la log-verosimilitud conjunta del punto de partida. Con
   ``ajuste="por_regimen"`` se queda en la etapa (a).

Un regimen con menos de ``min_obs_regimen`` observaciones no se estima: toma la
dinamica del modelo agrupado (todas las observaciones) con su media y su nivel de
varianza reescalados (o, si no tiene ninguna, copia el agrupado tal cual).

El VAR se estima por MCO con ridge pequeno sobre los pares ``(t-1, t)`` con
``s_t = k`` (la transicion se atribuye al regimen de destino; el retardo es el
valor real anterior aunque sea de otro regimen, asi que no hay empalmes). Si el
radio espectral del bloque propio ``A_yy`` supera ``radio_max`` se contrae ese
bloque y se recalcula el intercepto.

Que supone
----------
- Regimen exogeno al mercado: la cadena no depende de los retornos generados.
- El mercado no depende del resto de columnas (orden recursivo tipo CCC de
  Bollerslev 1990 en el sentido de que la dependencia entre columnas es lineal y
  constante por regimen; aqui pasa por ``b`` y por los residuos remuestreados).
- ``alfa + gamma/2 + beta`` es la persistencia solo si la innovacion es simetrica
  (lo es: t de Student).

Que reproduce y que no
----------------------
Reproduce: nivel de volatilidad por regimen, agrupamiento de volatilidad y
apalancamiento dentro de regimen, colas pesadas, persistencia de las columnas
lentas y su relacion lineal con el mercado.
No reproduce: asimetria de la innovacion (t simetrica); colas del tamano de
octubre de 1987 (un dia a -24 desviaciones cae fuera de cualquier t razonable);
memoria larga de la volatilidad (un GARCH(1,1) decae geometricamente); la forma
de escalon mensual exacta de las columnas mensuales (los saltos remuestreados
llegan en dias aleatorios y el nivel decae entre saltos); dependencia no lineal o
de cola entre columnas; heterocedasticidad propia de las columnas que no venga
del mercado.

Salvaguardas de simulacion (declaradas y contadas)
--------------------------------------------------
- **Techo de varianza**: ``h_t <= techo_varianza * cuantil(cuantil_techo)`` de la
  varianza condicional FILTRADA en train en los dias del regimen vigente
  (``h_max_``, un valor por regimen). Por que un cuantil y por regimen: con
  persistencia ~0,994 e innovacion t la recursion simulada tiene rafagas que el
  tramo real no tiene, y el techo anterior (2 x el maximo global) no frenaba
  nada: ese maximo (103 en unidades estandarizadas, ~9,5 % de volatilidad
  diaria) son los cinco dias posteriores al 19 de octubre de 1987, y el doble
  (206) dejaba pasar trayectorias con 51 dias de ``|ret| > 10 %``. Medido en la
  pista A hasta 2006 (500 x 2520 sesiones, semilla 7; real: maximo ``|ret|``
  22,9 %, maxima volatilidad anualizada de una ventana de 2520 sesiones 18,3 %;
  bootstrap por regimen con la misma cadena: 0,4 % de trayectorias con drawdown
  < -0,9, que vienen de crisis largas de la cadena, no del modelo de retornos)::

      techo de h_t                      dd<-0,9  max|ret|  vol anual max  dias>10 % (max)  sd calma / crisis
      2 x maximo global (anterior)       1,4 %    34,4 %      49,5 %           51            1,02 / 1,03
      1 x maximo del regimen             1,2 %    33,0 %      30,0 %           16            0,97 / 1,01
      1 x p99,5 del regimen              0,8 %    22,4 %      27,6 %            8            0,94 / 0,99
      1,5 x p99 del regimen              0,8 %    17,5 %      25,9 %            5            0,95 / 0,98
      1 x p99 del regimen (defecto)      0,4 %    14,3 %      23,7 %            3            0,92 / 0,96
      persistencia_max = 0,98            0,4 %    27,6 %      23,0 %            7            0,82 / 0,86

  El maximo del regimen (``cuantil_techo = 1``) actua pero no elimina las
  trayectorias patologicas; bajar ``persistencia_max`` las elimina hundiendo la
  desviacion por regimen (la MV restringida no conserva la varianza
  incondicional). El p99 del regimen (5,0 en calma y 19,0 en crisis: 2,1 % y
  4,1 % de volatilidad diaria) deja el drawdown extremo en el nivel del
  bootstrap y cuesta un 4-8 % de desviacion: el techo toca el ~0,1 % de las
  celdas. Consecuencia declarada: el generador no produce volatilidad
  condicional por encima del percentil 99 de su regimen en train (la semana
  posterior al crash de 1987 queda fuera, como ya quedaba el propio crash).
  En la pista B (hasta 2017, 2008 en train) el mismo techo deja 2,0 % de
  trayectorias con drawdown < -0,9 y 9 de 500 con >= 5 dias de ``|ret| > 10 %``
  (bootstrap por regimen: 3,0 % y 49), con cociente de desviaciones 0,88 / 0,86
  (0,99 / 0,91 con 2 x maximo del regimen) y el 0,4 % de las celdas recortadas.
  Son cifras del barrido de desarrollo (500 trayectorias, semilla 7, umbral de
  drawdown -0,9): NO son las del muestreo del proyecto. Con la cadena compartida
  de ``configs/sinteticos.yaml`` los cocientes, los recortes y el porcentaje de
  trayectorias que superan el peor drawdown REAL (otra vara de medir, mucho mas
  exigente que -0,9) salen de ``notebooks/15_sinteticos_generadores`` §4.5 y §6.
- **Techo de nivel**: cada columna se acota por ``techo_nivel`` veces su maximo
  absoluto de train (el mercado tambien).

``diagnostico_muestreo()`` devuelve tras cada ``sample`` la fraccion de celdas
que toco cada techo: ``fraccion_recorte_varianza`` (sobre trayectoria x dia),
``fraccion_recorte_mercado`` (idem) y ``fraccion_recorte_nivel`` (sobre
trayectoria x dia x columna del resto), mas ``fraccion_recorte_varianza_regimen``.
Si crecen, el techo esta haciendo de modelo.

Arranque: la simulacion continua el tramo de train. La varianza y el residuo
iniciales son los filtrados en la ultima fecha de train (``h_final_``,
``e_final_``) y el retardo del VAR es la ultima fila del ``contexto``.

Referencias
-----------
Glosten, Jagannathan y Runkle (1993); Bollerslev (1987, 1990); Gray (1996);
Haas, Mittnik y Paolella (2004).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import gammaln

from regimenes.sinteticos.comun import GeneradorBase
from regimenes.sinteticos.datos import COL_RET
from regimenes.sinteticos.registry import registrar

# Orden de las columnas de la matriz de parametros (una fila por regimen).
MU, OMEGA, ALFA, GAMMA, BETA, NU = range(6)
NU_MIN, NU_MAX = 2.05, 200.0


def _persistencia(par: np.ndarray) -> np.ndarray:
    return par[..., ALFA] + 0.5 * par[..., GAMMA] + par[..., BETA]


def _filtrar(r: np.ndarray, reg: np.ndarray, par: np.ndarray, h0: float) -> tuple[np.ndarray, np.ndarray]:
    """Varianza condicional ``h`` y residuo ``e`` de la recursion con regimen observado.

    Dentro de una racha ``beta`` es constante y la recursion es un filtro lineal
    de primer orden sobre ``omega + (alfa + gamma 1{e<0}) e^2``: se resuelve por
    racha con ``lfilter`` encadenando la condicion inicial.
    """
    e = r - par[reg, MU]
    n = e.size
    h = np.empty(n)
    h[0] = h0
    if n == 1:
        return h, e
    k = reg[1:]
    u = np.empty(n)
    u[1:] = par[k, OMEGA] + (par[k, ALFA] + par[k, GAMMA] * (e[:-1] < 0)) * e[:-1] ** 2
    cortes = np.unique(np.r_[1, np.flatnonzero(np.diff(reg) != 0) + 1, n])
    for a, b in zip(cortes[:-1], cortes[1:]):
        bk = par[reg[a], BETA]
        h[a:b] = lfilter([1.0], [1.0, -bk], u[a:b], zi=[bk * h[a - 1]])[0]
    return h, e


def _loglik_t(e: np.ndarray, h: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """Log-densidad por observacion de ``e ~ sqrt(h) * t_nu`` estandarizada."""
    return (
        gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu) - 0.5 * np.log(np.pi * (nu - 2.0))
        - 0.5 * np.log(h) - 0.5 * (nu + 1.0) * np.log1p(e * e / (h * (nu - 2.0)))
    )


@registrar
class GarchRegimen(GeneradorBase):
    """GJR-GARCH-t con regimen para el mercado y VAR(1) por regimen para el resto.

    Hiperparametros (``PARAMS``)
    ----------------------------
    columna_mercado : nombre de la columna de mercado; ``None`` = ``SP500_ret`` si
        existe y si no la primera columna.
    ajuste : ``"conjunto"`` (MV exacta con regimen observado, partiendo de
        ``arch``) o ``"por_regimen"`` (solo ``arch`` sobre rachas concatenadas).
    persistencia_max : cota de ``alfa + gamma/2 + beta`` por regimen (< 1).
    min_obs_regimen : observaciones minimas para estimar un regimen por separado.
    techo_varianza, cuantil_techo : cota de ``h_t`` = ``techo_varianza`` veces el cuantil
        ``cuantil_techo`` (en (0, 1]; 1 = maximo) de la varianza filtrada en train en los dias
        del regimen vigente.
    techo_nivel : cota de cada columna en multiplos de su maximo absoluto en train.
    regresor_mercado : ``"retorno"`` (``m_t`` = retorno de mercado) o ``"shock"`` (``z_t``).
    residuos : ``"bootstrap"`` (filas de residuos del regimen) o ``"gaussiano"``.
    ridge : penalizacion relativa del MCO del VAR (no afecta al intercepto).
    radio_max : radio espectral maximo del bloque propio del VAR.
    largo_contexto : filas de contexto (solo se usa la ultima, como retardo del VAR).

    Atributos tras ``fit``
    ----------------------
    i_mercado_, i_resto_ : posiciones de las columnas
    garch_ : (K, 6) ``[mu, omega, alfa, gamma, beta, nu]`` por regimen (espacio estandarizado)
    h_final_, e_final_ : estado del filtro al final de train
    h_max_ : (K,) techo de varianza por regimen
    var_c_ (K, dr), var_A_ (K, dr, d), var_b_ (K, dr) : coeficientes del VAR
    var_res_ : lista de K matrices de residuos; var_chol_ : (K, dr, dr)
    limite_ : (d,) cota absoluta por columna
    """

    nombre = "garch_regimen"
    familia = "parametricos"
    PARAMS = {
        "columna_mercado": None,
        "ajuste": "conjunto",
        "persistencia_max": 0.995,
        "min_obs_regimen": 100,
        "techo_varianza": 1.0,
        "cuantil_techo": 0.99,
        "techo_nivel": 1.5,
        "regresor_mercado": "retorno",
        "residuos": "bootstrap",
        "ridge": 1e-4,
        "radio_max": 0.999,
        "largo_contexto": 1,
    }
    PARAMS_RAPIDOS: dict = {}

    # ------------------------------------------------------------------ fit
    def _fit(self, X: np.ndarray, reg: np.ndarray, fechas: pd.DatetimeIndex) -> None:
        if self.ajuste not in {"conjunto", "por_regimen"}:
            raise ValueError("ajuste debe ser 'conjunto' o 'por_regimen'.")
        if self.regresor_mercado not in {"retorno", "shock"}:
            raise ValueError("regresor_mercado debe ser 'retorno' o 'shock'.")
        if self.residuos not in {"bootstrap", "gaussiano"}:
            raise ValueError("residuos debe ser 'bootstrap' o 'gaussiano'.")
        if not 0.0 < self.persistencia_max < 1.0:
            raise ValueError("persistencia_max debe estar en (0, 1).")
        if not 0.0 < float(self.cuantil_techo) <= 1.0 or not float(self.techo_varianza) > 0.0:
            raise ValueError("cuantil_techo debe estar en (0, 1] y techo_varianza ser > 0.")
        columnas = list(self.columnas_trabajo_)
        if self.columna_mercado is not None:
            if self.columna_mercado not in columnas:
                raise KeyError(f"columna_mercado={self.columna_mercado!r} no esta en {columnas}.")
            self.i_mercado_ = columnas.index(self.columna_mercado)
        else:
            self.i_mercado_ = columnas.index(COL_RET) if COL_RET in columnas else 0
        self.i_resto_ = np.array([j for j in range(X.shape[1]) if j != self.i_mercado_], dtype=int)

        r = X[:, self.i_mercado_]
        diag = self._ajustar_garch(r, reg)
        h, e = _filtrar(r, reg, self.garch_, self._h0(r, reg))
        self.h_final_, self.e_final_ = float(h[-1]), float(e[-1])
        # techo por regimen: cuantil de la varianza FILTRADA en los dias de ese regimen
        # (un regimen sin dias en train usa el cuantil de toda la muestra)
        self.h_max_ = np.array([
            float(self.techo_varianza) * float(np.quantile(h[reg == k] if (reg == k).any() else h, self.cuantil_techo))
            for k in range(self.n_regimenes_)
        ])
        self.limite_ = self.techo_nivel * np.abs(X).max(axis=0)
        diag_var = self._ajustar_var(X, reg, e / np.sqrt(h))

        ll = _loglik_t(e, h, self.garch_[reg, NU])
        escala = float(self.espacio_.escala_[self.i_mercado_])
        n_rachas = self.rachas_.groupby("regimen").size()
        for k in range(self.n_regimenes_):
            p = self.garch_[k]
            pers = float(_persistencia(p))
            vol = float(np.sqrt(p[OMEGA] / (1.0 - pers)))
            self.registrar(
                regimen=k,
                n_obs=int((reg == k).sum()),
                n_rachas=int(n_rachas.get(k, 0)),
                origen=diag[k]["origen"],
                mu=float(p[MU]),
                omega=float(p[OMEGA]),
                alfa=float(p[ALFA]),
                gamma=float(p[GAMMA]),
                beta=float(p[BETA]),
                persistencia=pers,
                nu=float(p[NU]),
                vol_incondicional=vol,
                vol_incondicional_original=vol * escala,
                techo_h=float(self.h_max_[k]),
                h_filtrada_max=float(h[reg == k].max()) if (reg == k).any() else float("nan"),
                frac_train_sobre_techo=float((h[reg == k] > self.h_max_[k]).mean()) if (reg == k).any() else 0.0,
                loglik=float(ll[reg == k].sum()),
                loglik_arch=diag[k]["loglik_arch"],
                convergencia_arch=diag[k]["convergencia_arch"],
                convergencia=diag[k]["convergencia"],
                loglik_conjunta=float(ll.sum()),
                loglik_conjunta_inicial=self.loglik_inicial_,
                **diag_var[k],
            )

    @staticmethod
    def _h0(r: np.ndarray, reg: np.ndarray) -> float:
        """Varianza inicial del filtro: la muestral del regimen de la primera observacion."""
        propio = r[reg == reg[0]]
        v = float(np.var(propio)) if propio.size > 1 else float(np.var(r))
        return v if v > 0 else 1.0

    def _factible(self, p: np.ndarray, var: float) -> np.ndarray:
        """Proyecta ``[mu, omega, alfa, gamma, beta, nu]`` a la region admisible."""
        p = np.array(p, dtype=float)
        p[ALFA], p[GAMMA], p[BETA] = np.clip(p[[ALFA, GAMMA, BETA]], 0.0, 1.0)
        pers = _persistencia(p)
        if pers > self.persistencia_max:
            p[[ALFA, GAMMA, BETA]] *= 0.999 * self.persistencia_max / pers
        p[OMEGA] = float(np.clip(p[OMEGA], 1e-8 * var, 10.0 * var))
        p[NU] = float(np.clip(p[NU], NU_MIN, NU_MAX))
        return p

    def _arch(self, y: np.ndarray) -> tuple[np.ndarray, float, bool]:
        """GJR-GARCH(1,1)-t de ``arch`` sobre ``y``: (parametros, loglik, convergio)."""
        var = float(np.var(y))
        try:
            from arch import arch_model

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = arch_model(
                    y, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t", rescale=False
                ).fit(disp="off", show_warning=False)
            p = np.asarray(res.params, dtype=float)
            if p.shape != (6,) or not np.isfinite(p).all():
                raise ValueError("parametros no finitos")
            return self._factible(p, var), float(res.loglikelihood), bool(res.convergence_flag == 0)
        except Exception:  # respaldo: varianza casi constante con dinamica suave
            p = np.array([float(np.mean(y)), 0.05 * var, 0.03, 0.06, 0.88, 8.0])
            return self._factible(p, var), float("nan"), False

    def _ajustar_garch(self, r: np.ndarray, reg: np.ndarray) -> list[dict]:
        K = self.n_regimenes_
        var_total = float(np.var(r)) or 1.0
        par = np.zeros((K, 6))
        diag: list[dict] = []
        libres: list[int] = []
        agrupado = None
        for k in range(K):
            y = r[reg == k]
            if y.size >= max(int(self.min_obs_regimen), 10) and np.var(y) > 0:
                par[k], ll, ok = self._arch(y)
                libres.append(k)
                diag.append({"origen": "arch", "loglik_arch": ll, "convergencia_arch": ok, "convergencia": ok})
                continue
            if agrupado is None:
                agrupado = self._arch(r)
            p = agrupado[0].copy()
            if y.size >= 2 and np.var(y) > 0:
                p[MU] = float(y.mean())
                p[OMEGA] *= float(np.var(y)) / var_total
            par[k] = self._factible(p, var_total)
            diag.append({
                "origen": "respaldo", "loglik_arch": float("nan"),
                "convergencia_arch": agrupado[2], "convergencia": agrupado[2],
            })
        self.garch_ = par
        h0 = self._h0(r, reg)

        def loglik(p: np.ndarray) -> float:
            h, e = _filtrar(r, reg, p, h0)
            if not np.isfinite(h).all() or h.min() <= 0:
                return -np.inf
            return float(_loglik_t(e, h, p[reg, NU]).sum())

        self.loglik_inicial_ = loglik(par)
        if self.ajuste != "conjunto" or not libres:
            return diag

        n = r.size
        idx = np.array(libres)

        def montar(theta: np.ndarray) -> np.ndarray:
            p = par.copy()
            p[idx] = theta.reshape(len(idx), 6)
            return p

        def objetivo(theta: np.ndarray) -> float:
            valor = loglik(montar(theta))
            return 1e6 if not np.isfinite(valor) else -valor / n

        sd = float(np.sqrt(var_total))
        cotas = [
            (-10.0 * sd, 10.0 * sd), (1e-8 * var_total, 10.0 * var_total),
            (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (NU_MIN, NU_MAX),
        ] * len(idx)
        restricciones = [{
            "type": "ineq",
            "fun": lambda th: self.persistencia_max - _persistencia(th.reshape(len(idx), 6)),
        }]
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            sol = minimize(
                objetivo, par[idx].ravel(), method="SLSQP", bounds=cotas, constraints=restricciones,
                options={"maxiter": 500, "ftol": 1e-10},
            )
        candidato = montar(np.asarray(sol.x, dtype=float))
        for k in idx:
            candidato[k] = self._factible(candidato[k], var_total)
        ll_nueva = loglik(candidato)
        if np.isfinite(ll_nueva) and ll_nueva >= self.loglik_inicial_:
            self.garch_ = candidato
            for k in idx:
                diag[k]["origen"] = "conjunto"
                diag[k]["convergencia"] = bool(sol.success)
        return diag

    def _ajustar_var(self, X: np.ndarray, reg: np.ndarray, shock: np.ndarray) -> list[dict]:
        K, d = self.n_regimenes_, X.shape[1]
        resto, dr = self.i_resto_, len(self.i_resto_)
        self.var_c_ = np.zeros((K, dr))
        self.var_A_ = np.zeros((K, dr, d))
        self.var_b_ = np.zeros((K, dr))
        self.var_chol_ = np.zeros((K, dr, dr))
        self.var_res_ = [np.zeros((1, dr)) for _ in range(K)]
        vacio = {"n_var": 0, "radio_espectral_var": 0.0, "radio_recortado": False, "r2_var_medio": float("nan")}
        if dr == 0 or len(X) < 3:
            return [dict(vacio) for _ in range(K)]
        lag, Y, destino = X[:-1], X[1:, resto], reg[1:]
        m = X[1:, self.i_mercado_] if self.regresor_mercado == "retorno" else shock[1:]
        q = d + 2
        minimo = max(int(self.min_obs_regimen), 3 * q)

        def estimar(filas: np.ndarray) -> dict:
            Lg, Yk, mk = lag[filas], Y[filas], m[filas]
            Z = np.column_stack([np.ones(len(Lg)), Lg, mk])
            pen = self.ridge * len(Z) * np.eye(q)
            pen[0, 0] = 0.0
            coef = np.linalg.solve(Z.T @ Z + pen, Z.T @ Yk)
            c, A, b = coef[0].copy(), coef[1:1 + d].T.copy(), coef[1 + d].copy()
            radio = float(np.abs(np.linalg.eigvals(A[:, resto])).max())
            recortado = radio > self.radio_max
            if recortado:
                A[:, resto] *= self.radio_max / radio
                c = (Yk - Lg @ A.T - np.outer(mk, b)).mean(axis=0)
            res = Yk - c - Lg @ A.T - np.outer(mk, b)
            sst = ((Yk - Yk.mean(axis=0)) ** 2).sum(axis=0)
            r2 = 1.0 - (res ** 2).sum(axis=0) / np.where(sst > 0, sst, np.nan)
            cov = np.atleast_2d(np.cov(res, rowvar=False)) + 1e-10 * np.eye(dr)
            return {
                "c": c, "A": A, "b": b, "res": res, "chol": np.linalg.cholesky(cov), "n": len(Z),
                "radio": min(radio, float(self.radio_max)), "recortado": bool(recortado),
                "r2": float(np.nanmean(r2)) if np.isfinite(r2).any() else float("nan"),
            }

        agrupado = None
        diag = []
        for k in range(K):
            filas = np.flatnonzero(destino == k)
            if filas.size >= minimo:
                est = estimar(filas)
            else:
                if agrupado is None:
                    agrupado = estimar(np.arange(len(lag)))
                est = agrupado
            self.var_c_[k], self.var_A_[k], self.var_b_[k] = est["c"], est["A"], est["b"]
            self.var_chol_[k], self.var_res_[k] = est["chol"], est["res"]
            diag.append({
                "n_var": int(est["n"]), "radio_espectral_var": est["radio"],
                "radio_recortado": est["recortado"], "r2_var_medio": est["r2"],
            })
        return diag

    # --------------------------------------------------------------- sample
    def _sample(self, reg: np.ndarray, rng: np.random.Generator, contexto: np.ndarray) -> np.ndarray:
        P, T = reg.shape
        d, im, resto = self.d_, self.i_mercado_, self.i_resto_
        dr = len(resto)
        par = self.garch_
        nu = par[reg, NU]
        z = rng.standard_t(nu) * np.sqrt((nu - 2.0) / nu)
        if dr and self.residuos == "bootstrap":
            tam = np.array([len(u) for u in self.var_res_])
            fila = np.minimum((rng.random((P, T)) * tam[reg]).astype(int), tam[reg] - 1)
        elif dr:
            eps = rng.standard_normal((P, T, dr))

        salida = np.empty((P, T, d))
        x = np.tile(contexto[-1], (P, 1)) if len(contexto) else np.zeros((P, d))
        h = np.full(P, self.h_final_)
        e = np.full(P, self.e_final_)
        lim_m = self.limite_[im]
        h_max = np.broadcast_to(np.asarray(self.h_max_, dtype=float), (self.n_regimenes_,))
        n_h = np.zeros(self.n_regimenes_, dtype=np.int64)
        n_mercado = n_nivel = 0
        for t in range(T):
            k = reg[:, t]
            pk = par[k]
            h = pk[:, OMEGA] + (pk[:, ALFA] + pk[:, GAMMA] * (e < 0)) * e * e + pk[:, BETA] * h
            techo = h_max[k]
            n_h += np.bincount(k[h > techo], minlength=self.n_regimenes_)
            h = np.clip(h, 1e-12, techo)
            raiz = np.sqrt(h)
            mercado = pk[:, MU] + raiz * z[:, t]
            n_mercado += int((np.abs(mercado) > lim_m).sum())
            mercado = np.clip(mercado, -lim_m, lim_m)
            e = mercado - pk[:, MU]
            nuevo = np.empty((P, d))
            nuevo[:, im] = mercado
            if dr:
                if self.residuos == "bootstrap":
                    u = np.empty((P, dr))
                    for j in np.unique(k):
                        sel = k == j
                        u[sel] = self.var_res_[j][fila[sel, t]]
                else:
                    u = np.einsum("pij,pj->pi", self.var_chol_[k], eps[:, t])
                m = mercado if self.regresor_mercado == "retorno" else e / raiz
                y = self.var_c_[k] + np.einsum("pij,pj->pi", self.var_A_[k], x) + self.var_b_[k] * m[:, None] + u
                n_nivel += int((np.abs(y) > self.limite_[resto]).sum())
                nuevo[:, resto] = np.clip(y, -self.limite_[resto], self.limite_[resto])
            salida[:, t] = nuevo
            x = nuevo
        dias = np.bincount(reg.ravel(), minlength=self.n_regimenes_)
        self.diagnostico_muestreo_.update(
            fraccion_recorte_varianza=float(n_h.sum() / (P * T)),
            fraccion_recorte_varianza_regimen=[float(a / b) if b else 0.0 for a, b in zip(n_h, dias)],
            fraccion_recorte_mercado=float(n_mercado / (P * T)),
            fraccion_recorte_nivel=float(n_nivel / (P * T * dr)) if dr else 0.0,
            techo_h=[float(v) for v in h_max],
        )
        return salida
