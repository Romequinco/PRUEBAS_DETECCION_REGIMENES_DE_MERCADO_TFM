"""
detector_base.py — Interfaz común para TODOS los detectores de régimen.

Principio rector del proyecto: el objetivo no es ningún detector concreto, sino el
MARCO DE EVALUACIÓN que los juzga a todos de forma comparable y CAUSAL (sin
look-ahead). Para que la comparación sea honesta, cada familia de detector
(reglas, clustering, HMM, Markov-Switching, change-point, redes, GARCH-RS...)
debe implementar EXACTAMENTE esta misma interfaz y producir el mismo tipo de
salida, de modo que `evaluation.py` pueda puntuarlos con las mismas métricas.

Contrato de causalidad (NO negociable)
--------------------------------------
- `fit(X_train)` solo puede ver el tramo de entrenamiento.
- En walk-forward, la etiqueta del día t debe depender únicamente de datos <= t.
  Es responsabilidad del detector NO usar información futura (p. ej. filtrado
  forward/Viterbi sobre toda la muestra es ANTI-causal y está prohibido en
  evaluación online; ver `predict_online`).
- Las features de entrada ya vienen estandarizadas de forma causal desde
  `features.py` (z-scores expanding/rolling). El detector NO debe reestandarizar
  con estadísticos de toda la muestra.

Salida de estado
----------------
- `predict(X)` -> etiquetas de estado DURAS (enteros 0..n_states-1), ya
  *canonicalizadas* por criterio económico (ver `_economic_state_order`): el
  estado 0 es el más "calmado/risk-on" y el último el de "crisis". Esto hace que
  las etiquetas sean comparables entre detectores y entre ventanas walk-forward.
- `predict_proba(X)` -> matriz (n_obs, n_states) de probabilidades blandas por
  estado, en el mismo orden canónico. Detectores que no son probabilísticos
  (p. ej. reglas duras, k-means) devuelven one-hot.
- `crisis_state` -> índice del estado etiquetado como "crisis" (el de peor
  retorno / mayor volatilidad). Permite construir una probabilidad de crisis
  continua: `predict_proba(X)[:, crisis_state]`.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import pandas as pd

# Columnas de X que se aceptan como proxy de "retorno de mercado" cuando NO se
# pasa market_returns explícito. El orden refleja preferencia. Si ninguna está,
# el etiquetado económico cae a la primera columna y AVISA (puede invertir
# crisis/calma para detectores que no operan sobre retornos: varianza, sigma,
# distancia de Mahalanobis, etc.).
_RETURN_COLS: tuple[str, ...] = ("SP500_ret_z", "SP500_ret", "ret", "SP500")

# Dos estados se consideran de "volatilidad próxima" (y entonces el retorno medio
# desempata) si sus desviaciones difieren menos de esta fracción de la vol media.
# Si están claramente separados en vol, la VOLATILIDAD fija el orden y el retorno
# NO puede invertirlo (Arreglo 4: robustez ante estados que solo separan en
# varianza, p. ej. detectores de sigma/turbulencia con K=2 y medias ~iguales).
VOL_CLOSE_FRAC: float = 0.15


class RegimeDetector(ABC):
    """Clase base abstracta que todo detector de régimen debe heredar.

    Subclases mínimas a implementar: `fit`, `_predict_states`, `name`,
    `bibliography`. El resto (canonicalización económica, predict_proba por
    defecto, walk-forward online) se hereda.

    Parameters
    ----------
    n_states : int
        Número de estados/regímenes del modelo. Para detectores con K fijo
        (reglas binarias) suele ser 2; para HMM/GMM se selecciona por BIC/AIC.
    random_state : int | None
        Semilla para reproducibilidad de detectores estocásticos.
    """

    def __init__(self, n_states: int = 2, random_state: int | None = 42) -> None:
        self._n_states = int(n_states)
        self.random_state = random_state
        # Mapa que reordena las etiquetas internas del modelo (arbitrarias) al
        # orden económico canónico 0=calma ... n-1=crisis. Lo fija
        # `label_states_economically()` tras el fit.
        self._canonical_order: np.ndarray | None = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------ #
    # Identidad del detector
    # ------------------------------------------------------------------ #
    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre legible y único del detector (p. ej. 'GaussianHMM-k3')."""

    @property
    @abstractmethod
    def bibliography(self) -> list[str]:
        """Lista de claves BibTeX (en docs/references.bib) de las técnicas usadas.

        Requisito transversal del TFM: cada detector declara de qué literatura
        proviene para construir la bibliografía final automáticamente.
        """

    @property
    def n_states(self) -> int:
        """Número de estados del modelo."""
        return self._n_states

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    # ------------------------------------------------------------------ #
    # API de ajuste y predicción (la que ve el evaluador)
    # ------------------------------------------------------------------ #
    @abstractmethod
    def fit(self, X_train: pd.DataFrame) -> "RegimeDetector":
        """Ajusta el detector usando SOLO el tramo de entrenamiento.

        Debe terminar marcando `self._is_fitted = True` y, normalmente, llamar a
        `self.label_states_economically(X_train)` para fijar el orden canónico.

        Parameters
        ----------
        X_train : pd.DataFrame
            Features causales de entrenamiento, indexadas por fecha.

        Returns
        -------
        self
        """

    @abstractmethod
    def _predict_states(self, X: pd.DataFrame) -> np.ndarray:
        """Predice etiquetas de estado INTERNAS del modelo (sin canonicalizar).

        Implementación específica de cada familia. El método público `predict`
        aplica encima la reordenación económica. Devuelve un array de enteros de
        longitud len(X).
        """

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Etiquetas de estado DURAS, ya canonicalizadas (0=calma..n-1=crisis).

        Parameters
        ----------
        X : pd.DataFrame
            Features causales sobre las que predecir.

        Returns
        -------
        np.ndarray de int, shape (len(X),).
        """
        self._check_fitted()
        raw = np.asarray(self._predict_states(X))
        return self._apply_canonical(raw)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilidades blandas por estado, en orden canónico.

        Por defecto devuelve one-hot a partir de `predict` (válido para
        detectores no probabilísticos). Los detectores con posterior real
        (HMM, GMM, Markov-Switching) sobrescriben este método con el
        filtrado/forward-backward correspondiente.

        Returns
        -------
        np.ndarray, shape (len(X), n_states), filas que suman 1.
        """
        self._check_fitted()
        hard = self.predict(X)
        proba = np.zeros((len(hard), self.n_states), dtype=float)
        proba[np.arange(len(hard)), hard] = 1.0
        return proba

    def predict_online(self, X: pd.DataFrame, refit: bool = False) -> np.ndarray:
        """Predicción CAUSAL día a día para evaluación walk-forward.

        Para cada t devuelve la etiqueta usando solo información <= t. La
        implementación por defecto asume que `_predict_states` ya es causal
        (filtrado, no suavizado) y delega en `predict`. Detectores que por
        defecto suavizan sobre toda la muestra (p. ej. Viterbi) DEBEN
        sobrescribir esto para garantizar causalidad.

        Parameters
        ----------
        X : pd.DataFrame
            Features causales del tramo de test.
        refit : bool
            Si True, el orquestador de walk-forward reajustará el modelo en cada
            ventana (lo gestiona `evaluation.walk_forward`, no este método).
        """
        return self.predict(X)

    # ------------------------------------------------------------------ #
    # Etiquetado económico de estados
    # ------------------------------------------------------------------ #
    def label_states_economically(
        self, X: pd.DataFrame, market_returns: pd.Series | None = None
    ) -> "RegimeDetector":
        """Fija el orden canónico de estados por criterio económico.

        Los modelos no supervisados asignan etiquetas arbitrarias (0/1/2...).
        Aquí las reordenamos a un orden estable y comparable entre detectores:
        **0 = régimen más calmado (mejor retorno / menor vol)** ... **n-1 =
        crisis (peor retorno / mayor vol)**. El criterio usa el retorno medio y
        la volatilidad del mercado dentro de cada estado.

        Parameters
        ----------
        X : pd.DataFrame
            Features (o datos) sobre los que se han estimado los estados.
        market_returns : pd.Series | None
            Retornos del activo de referencia (S&P 500) alineados con X. Si es
            None, se intenta inferir de una columna estándar de X
            (ver `_economic_state_order`).

        Returns
        -------
        self
        """
        raw = np.asarray(self._predict_states(X))
        self._canonical_order = self._economic_state_order(raw, X, market_returns)
        return self

    def _economic_state_order(
        self,
        raw_labels: np.ndarray,
        X: pd.DataFrame,
        market_returns: pd.Series | None,
    ) -> np.ndarray:
        """Devuelve el array de permutación: orden[i] = etiqueta interna que
        pasa a ser el estado canónico i.

        Ordena los estados por "severidad" creciente. Severidad ≈ -retorno_medio
        ponderado con +volatilidad del mercado condicionada al estado. El estado
        más severo queda como crisis (índice n-1).
        """
        raw = np.asarray(raw_labels)
        # 1. retornos de mercado: PRIORIDAD al parámetro explícito (lo pasa
        #    walk_forward/evaluate). Si no, fallback a una columna de X reconocida
        #    como retorno; y si tampoco, a la primera columna AVISANDO del riesgo.
        if market_returns is not None:
            r = pd.Series(np.asarray(market_returns), index=X.index)
        else:
            recognized = next((c for c in _RETURN_COLS if c in X.columns), None)
            if recognized is not None:
                r = X[recognized]
                warnings.warn(
                    f"{self.name}: label_states_economically sin market_returns "
                    f"explícito; uso la columna '{recognized}' de X como proxy de "
                    f"retorno. Pásalo vía walk_forward/evaluate(market_returns=) "
                    f"para mayor robustez.",
                    stacklevel=2,
                )
            else:
                col = X.columns[0]
                warnings.warn(
                    f"{self.name}: SIN market_returns y SIN columna de retorno "
                    f"reconocida en X; ordeno los estados por '{col}', lo que PUEDE "
                    f"INVERTIR crisis/calma. Pasa market_returns a "
                    f"walk_forward/evaluate.",
                    stacklevel=2,
                )
                r = X[col]
        rv = r.values

        # 2. media y desviación (nan-safe) de los retornos por etiqueta observada.
        observed = np.unique(raw)
        means = np.array([np.nanmean(rv[raw == lab]) for lab in observed])
        stds = np.array([np.nanstd(rv[raw == lab]) for lab in observed])

        # 3. severidad: VOLATILIDAD primaria, retorno medio solo como DESEMPATE
        #    cuando las vols están próximas. Con K=2 el z-score de 2 elementos es
        #    siempre {-1,+1}, así que el viejo `z(std)-z(media)` dejaba que el signo
        #    (ruidoso) de una diferencia de medias casi nula INVIRTIERA crisis/calma
        #    en detectores que separan solo en varianza (D6). Aquí la vol manda:
        #    los estados se agrupan en bandas de volatilidad (ancho VOL_CLOSE_FRAC ×
        #    vol media); el orden lo fija la banda y, SOLO dentro de una misma banda
        #    (vols próximas), desempata el retorno medio (menor => más severo).
        #    Coherente con la tarea previa: "crisis = mayor vol Y menor retorno, con
        #    fallback a solo vol".
        vol_scale = float(np.nanmean(stds))
        tol = VOL_CLOSE_FRAC * vol_scale
        if tol > 0 and np.isfinite(tol):
            band = np.round(stds / tol)
        else:  # vols degeneradas (todas ~0): solo desempata el retorno
            band = np.zeros(len(stds))
        # Orden ascendente calma->crisis: banda de vol asc (primario); dentro de la
        # misma banda, retorno medio desc (menor retorno = más severo => más tarde).
        sev_order = sorted(range(len(observed)), key=lambda j: (band[j], -means[j]))
        ordered = list(observed[sev_order])

        # 4. completar con estados nunca observados (al final) para cubrir n_states.
        ordered += [s for s in range(self.n_states) if s not in ordered]
        return np.asarray(ordered, dtype=int)

    @property
    def crisis_state(self) -> int:
        """Índice canónico del estado 'crisis' (el más severo): n_states - 1."""
        return self.n_states - 1

    def crisis_probability(self, X: pd.DataFrame) -> np.ndarray:
        """Serie continua P(crisis) = predict_proba(X)[:, crisis_state]."""
        return self.predict_proba(X)[:, self.crisis_state]

    # ------------------------------------------------------------------ #
    # Bondad de ajuste (donde el modelo lo permita)
    # ------------------------------------------------------------------ #
    def score(self, X: pd.DataFrame) -> float:
        """Log-likelihood total del modelo sobre X (si aplica).

        Los detectores generativos (HMM, GMM, Markov-Switching, GARCH) lo
        sobrescriben. Los no generativos (reglas, k-means duro) devuelven NaN y
        `evaluation.py` simplemente omite AIC/BIC para ellos.
        """
        return float("nan")

    def n_parameters(self) -> int:
        """Nº de parámetros libres del modelo, para AIC/BIC. NaN-equivalente=-1."""
        return -1

    def aic(self, X: pd.DataFrame) -> float:
        """AIC = 2k - 2logL (NaN si el modelo no expone logL)."""
        ll, k = self.score(X), self.n_parameters()
        if np.isnan(ll) or k < 0:
            return float("nan")
        return 2 * k - 2 * ll

    def bic(self, X: pd.DataFrame) -> float:
        """BIC = k·ln(n) - 2logL (NaN si el modelo no expone logL)."""
        ll, k = self.score(X), self.n_parameters()
        if np.isnan(ll) or k < 0:
            return float("nan")
        return k * np.log(len(X)) - 2 * ll

    # ------------------------------------------------------------------ #
    # Utilidades internas
    # ------------------------------------------------------------------ #
    def _apply_canonical(self, raw_labels: np.ndarray) -> np.ndarray:
        """Mapea etiquetas internas -> canónicas usando `self._canonical_order`."""
        if self._canonical_order is None:
            return raw_labels
        inverse = np.empty(self.n_states, dtype=int)
        for canonical_idx, internal_label in enumerate(self._canonical_order):
            inverse[internal_label] = canonical_idx
        return inverse[raw_labels]

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(f"{self.name}: llama a fit() antes de predecir.")

    def __repr__(self) -> str:  # pragma: no cover
        state = "fitted" if self._is_fitted else "unfitted"
        return f"<{self.name} n_states={self.n_states} [{state}]>"
