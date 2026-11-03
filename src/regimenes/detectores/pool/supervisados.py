"""supervisados.py — D16 (logit L1) y D17 (random forest) con etiquetas causales (ADR-010 §3, §3.2).

Etiquetas en cada reajuste
--------------------------
El reajuste t es el primer día del bloque que se predice; el train son los días < t, así que su
última fecha ``fin`` es el último día visto. Un día del train es **crisis** si cae en
``[inicio, suelo]`` de una crisis de ``data/processed/labels.parquet`` de la pista cuyo suelo es
anterior a t (``trough_date <= fin``). Las crisis cuyo suelo aún no ha llegado:

- variante principal (``excluir_sin_terminar=False``): sus días cuentan como **calma**, igual que
  los vería alguien en tiempo real;
- sensibilidad (``excluir_sin_terminar=True``): sus días del train se sacan del ajuste.

Las trampas de ``regimenes.evaluacion.ranking.track_false_positive_windows`` cuentan como calma.
Si en el reajuste no hay ninguna crisis terminada (o el train no tiene las dos clases), el detector
predice calma en todo el tramo.

Estado de crisis: clase 1 = crisis = ``crisis_state`` (``n_states = 2``), sin canonicalización
económica: ``label_states_economically`` deja el orden identidad. Crisis si P(crisis) ≥ 0,5.

Estos detectores no transforman las columnas: reciben las de P o N ya en z-scores causales. Las filas
con algún NaN se sacan del ajuste; la regla de disponibilidad de P (§2.1) la aplica el envoltorio
``ConDisponibilidad``.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from regimenes import rutas
from regimenes.detectores.base import RegimeDetector

UMBRAL: float = 0.5
RUTA_LABELS = rutas.DATA_PROCESSED / "labels.parquet"


# --------------------------------------------------------------------------- #
# Catálogo de crisis y trampas
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=4)
def _crisis_catalogo(pista: str) -> tuple[tuple[pd.Timestamp, pd.Timestamp], ...]:
    labels = pd.read_parquet(RUTA_LABELS)
    parte = labels[labels["pista"].astype(str).str.upper() == pista.upper()]
    return tuple((pd.Timestamp(i), pd.Timestamp(s))
                 for i, s in zip(parte["inicio"], parte["trough_date"]))


@lru_cache(maxsize=1)
def _trampas_catalogo() -> tuple[tuple[pd.Timestamp, pd.Timestamp], ...]:
    from regimenes.evaluacion.ranking import track_false_positive_windows

    return tuple((pd.Timestamp(a), pd.Timestamp(b))
                 for a, b in track_false_positive_windows().values())


def etiquetas_causales(
    index: pd.DatetimeIndex,
    crisis: tuple[tuple[pd.Timestamp, pd.Timestamp], ...] | list,
    trampas: tuple[tuple[pd.Timestamp, pd.Timestamp], ...] | list = (),
    *,
    excluir_sin_terminar: bool = False,
) -> pd.Series:
    """Etiqueta 0/1 de cada día de ``index`` vista al final de ``index`` (ADR-010 §3.2).

    Devuelve una serie float con 1 = crisis terminada, 0 = calma y NaN = fila excluida (solo con
    ``excluir_sin_terminar``: días de una crisis empezada y aún sin suelo).
    """
    index = pd.DatetimeIndex(index)
    fin = index[-1]
    y = pd.Series(0.0, index=index)
    for inicio, suelo in crisis:
        inicio, suelo = pd.Timestamp(inicio), pd.Timestamp(suelo)
        dentro = (index >= inicio) & (index <= suelo)
        if suelo <= fin:
            y[dentro] = 1.0
        elif excluir_sin_terminar and inicio <= fin:
            y[dentro] = np.nan
    for a, b in trampas:
        y[(index >= pd.Timestamp(a)) & (index <= pd.Timestamp(b))] = 0.0
    return y


# --------------------------------------------------------------------------- #
# Base común
# --------------------------------------------------------------------------- #
class _Supervisado(RegimeDetector):
    """Clasificador binario 0 = calma / 1 = crisis con las etiquetas causales del §3.2.

    Parameters
    ----------
    pista : ``'A'`` o ``'B'``; elige las crisis de ``labels.parquet``.
    excluir_sin_terminar : variante de sensibilidad (saca del ajuste las crisis sin suelo).
    crisis, trampas : ventanas ``[(inicio, suelo), ...]`` explícitas (para tests); por defecto, las
        del catálogo de la pista y las de ``track_false_positive_windows``.
    random_state : semilla.
    """

    def __init__(
        self,
        pista: str = "B",
        excluir_sin_terminar: bool = False,
        crisis: list | tuple | None = None,
        trampas: list | tuple | None = None,
        random_state: int | None = 42,
    ) -> None:
        super().__init__(n_states=2, random_state=random_state)
        self.pista = str(pista).upper()
        self.excluir_sin_terminar = bool(excluir_sin_terminar)
        self._crisis = crisis
        self._trampas = trampas
        self._clf = None
        self._columnas: list[str] = []
        # Para el informe.
        self.n_crisis_terminadas: int = 0
        self.n_dias_crisis: int = 0
        self.n_filas_ajuste: int = 0
        self.siempre_calma: bool = False

    # Subclases ---------------------------------------------------------- #
    def _nuevo_clasificador(self):  # pragma: no cover - abstracto
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    def _ventanas(self):
        crisis = _crisis_catalogo(self.pista) if self._crisis is None else self._crisis
        trampas = _trampas_catalogo() if self._trampas is None else self._trampas
        return crisis, trampas

    def fit(self, X_train: pd.DataFrame) -> "_Supervisado":
        crisis, trampas = self._ventanas()
        fin = pd.Timestamp(X_train.index[-1])
        y = etiquetas_causales(X_train.index, crisis, trampas,
                               excluir_sin_terminar=self.excluir_sin_terminar)
        self.n_crisis_terminadas = sum(pd.Timestamp(s) <= fin for _, s in crisis)
        validas = y.notna() & X_train.notna().all(axis=1)
        Xf, yf = X_train.loc[validas], y.loc[validas].astype(int)
        self._columnas = list(X_train.columns)
        self.n_filas_ajuste = int(len(yf))
        self.n_dias_crisis = int(yf.sum())
        self.siempre_calma = self.n_crisis_terminadas == 0 or yf.nunique() < 2
        if self.siempre_calma:
            self._clf = None
        else:
            self._clf = self._nuevo_clasificador().fit(Xf.to_numpy(dtype=float), yf.to_numpy())
        self._canonical_order = np.arange(self.n_states)
        self._is_fitted = True
        return self

    def diagnostico(self) -> dict:
        """Etiquetas del último ``fit`` (lo recoge ``ConDisponibilidad`` en el menú P)."""
        return {"sup_n_crisis_terminadas": int(self.n_crisis_terminadas),
                "sup_n_dias_crisis": int(self.n_dias_crisis),
                "sup_n_filas_ajuste": int(self.n_filas_ajuste),
                "sup_siempre_calma": bool(self.siempre_calma)}

    def resumen_reajuste(self) -> dict:
        """Ficha del último ``fit`` para el ejecutor (menú N, sin envoltorio)."""
        return self.diagnostico() if self._is_fitted else {}

    def label_states_economically(self, X: pd.DataFrame, market_returns: pd.Series | None = None
                                  ) -> "_Supervisado":
        """Sin canonicalización económica (ADR-010 §3): clase 1 = crisis."""
        self._canonical_order = np.arange(self.n_states)
        return self

    def crisis_probability(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        if self._clf is None:
            return np.zeros(len(X))
        A = X[self._columnas].to_numpy(dtype=float)
        p = np.zeros(len(X))
        ok = np.isfinite(A).all(axis=1)
        if ok.any():
            clases = list(self._clf.classes_)
            p[ok] = self._clf.predict_proba(A[ok])[:, clases.index(1)]
        return p

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.crisis_probability(X)
        return np.column_stack([1.0 - p, p])

    def _predict_states(self, X: pd.DataFrame) -> np.ndarray:
        return (self.crisis_probability(X) >= UMBRAL).astype(int)


class LogitL1(_Supervisado):
    """D16: regresión logística con penalización L1 (``C = 1``, ``balanced``, ``liblinear``)."""

    def __init__(self, C: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.C = float(C)

    @property
    def name(self) -> str:
        return "logit_l1" + ("_excl" if self.excluir_sin_terminar else "")

    @property
    def bibliography(self) -> list[str]:
        return []

    def _nuevo_clasificador(self):
        import sklearn
        from sklearn.linear_model import LogisticRegression

        # Desde scikit-learn 1.8, ``penalty`` está obsoleto y L1 se pide con ``l1_ratio = 1``; antes,
        # ``l1_ratio`` sin ``penalty="elasticnet"`` se ignoraba. Mismo modelo en ambos casos.
        version = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
        l1 = {"l1_ratio": 1.0} if version >= (1, 8) else {"penalty": "l1"}
        return LogisticRegression(C=self.C, class_weight="balanced", solver="liblinear",
                                  random_state=self.random_state, **l1)

    @property
    def coeficientes(self) -> pd.Series:
        """Coeficientes del último ajuste (0 si predice siempre calma)."""
        if self._clf is None:
            return pd.Series(0.0, index=self._columnas)
        return pd.Series(self._clf.coef_[0], index=self._columnas)


class BosqueAleatorio(_Supervisado):
    """D17: random forest (500 árboles, ``max_depth = 5``, ``min_samples_leaf = 50``,
    ``balanced_subsample``, semilla 42)."""

    def __init__(self, n_estimators: int = 500, max_depth: int = 5, min_samples_leaf: int = 50,
                 n_jobs: int | None = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.n_jobs = n_jobs

    @property
    def name(self) -> str:
        return "random_forest" + ("_excl" if self.excluir_sin_terminar else "")

    @property
    def bibliography(self) -> list[str]:
        return []

    def _nuevo_clasificador(self):
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                      min_samples_leaf=self.min_samples_leaf,
                                      class_weight="balanced_subsample",
                                      random_state=self.random_state, n_jobs=self.n_jobs)
