"""shu.py — D15 · Jump model de Shu, Yu y Mulvey (2024) con λ elegida sin etiquetas (ADR-010 §3, §3.1).

Qué hace
--------
- **Features del paper** (arXiv:2402.05272v3, §3.4.1 y tabla 2), sobre la rentabilidad del S&P 500
  en exceso del T-bill a 3 meses (``DTB3``):

  - desviación a la baja EWM, semivida 10: ``sqrt(EWM(min(r, 0)²))``;
  - ratio de Sortino EWM, semividas 20 y 60: ``EWM(r) / DD_EWM`` con la misma semivida.

  Se estandarizan con la media y la desviación de la ventana de ajuste (el escalado del paper por
  √2 y 1/√2 desaparece al estandarizar).
- **Historia previa** (ADR-010 §3.1): las features se calculan desde
  ``data/raw/yfinance/SP500.parquet`` y ``data/raw/fred/DTB3.parquet`` y empiezan con el primer
  ``DTB3`` (1954-01). En cada llamada los crudos se recortan a la última fecha de la ``X`` recibida:
  nunca se usa un día posterior. La ``X`` de la pista (``SP500_ret``) solo aporta las fechas.
- **Elección de λ en cada reajuste** sobre la rejilla {5, 10, 22, 50, 100, 220}: con n días de
  historia ≤ fin del train, la validación son los últimos V = min(2.016, ⌊n/2⌋) días y el ajuste, los
  hasta 3.000 anteriores. Con cada λ se ajusta, se infiere online en la validación con Θ fijo y se
  calcula el Sharpe de la estrategia 0/1 (S&P 500 en calma, ``DTB3`` en crisis, señal de t operada
  desde t + 2, 10 pb por operación en cada sentido). Gana el mayor Sharpe; empates, la λ mayor.
  Con la λ elegida se reajusta en los últimos min(3.000, n) días.
- **Estado de crisis:** la canonicalización económica de ``base.py`` (como D09) con el retorno
  logarítmico del S&P 500 de la ventana de ajuste, que es la misma definición que ``SP500_ret`` de
  la pista (``benchmark.ejecucion.load_track_panel``). Se guarda si la regla del paper (bajista =
  menor rentabilidad acumulada en exceso) discrepa (``discrepa_regla_paper``).

Sharpe de la validación: media / desviación de la rentabilidad **en exceso** de la estrategia
(«average excess return over volatility», pie de la tabla 4 del paper), anualizado con √252. Si la
desviación es 0 (siempre en el activo libre), el Sharpe es 0.

Causalidad: ``fit`` usa días ≤ fin del train; ``predict_online`` y ``predict_proba`` usan el
filtrado online de ``jumpmodels`` sobre los días desde el inicio de la ventana de ajuste hasta la
última fecha de ``X``; la etiqueta de un día depende solo de días ≤ él.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from regimenes import rutas
from regimenes.detectores.base import RegimeDetector

LAMBDAS: tuple[float, ...] = (5.0, 10.0, 22.0, 50.0, 100.0, 220.0)
VENTANA_AJUSTE: int = 3000
VALIDACION_MAX: int = 2016
COSTE: float = 0.0010           # 10 pb por operación en cada sentido
RETRASO: int = 2                # la señal de t se opera desde t + 2
DIAS_ANIO: int = 252
SEMIVIDA_DD: int = 10
SEMIVIDAS_SORTINO: tuple[int, ...] = (20, 60)
FEATURES: tuple[str, ...] = ("DD_10", "sortino_20", "sortino_60")

RUTA_SP500: Path = rutas.DATA_RAW / "yfinance" / "SP500.parquet"
RUTA_DTB3: Path = rutas.DATA_RAW / "fred" / "DTB3.parquet"


# --------------------------------------------------------------------------- #
# Datos y features
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=4)
def _crudos(ruta_sp500: str, ruta_dtb3: str) -> tuple[pd.Series, pd.Series]:
    precios = pd.read_parquet(ruta_sp500)["SP500"].sort_index().astype(float)
    tipos = pd.read_parquet(ruta_dtb3)["DTB3"].sort_index().astype(float)
    precios.index = pd.to_datetime(precios.index)
    tipos.index = pd.to_datetime(tipos.index)
    return precios, tipos


def base_exceso(precios: pd.Series, tipos: pd.Series) -> pd.DataFrame:
    """Rentabilidades diarias del S&P 500 y del T-bill, en el calendario de ``precios``.

    - ``ret``: rentabilidad simple del cierre; ``logret``: logarítmica (= ``SP500_ret`` de la pista).
    - ``rf``: rentabilidad diaria del T-bill, ``DTB3`` (en %, anual) / 100 / 252, con el tipo vigente
      al cierre del día anterior (as-of ≤ t − 1).
    - ``exceso`` = ``ret`` − ``rf``.

    Empieza en el primer día con ``rf`` (las features se cortan al inicio de ``DTB3``).
    """
    precios = precios.dropna().sort_index()
    tipo_dia = tipos.dropna().sort_index().reindex(precios.index, method="ffill")
    rf = (tipo_dia / 100.0 / DIAS_ANIO).shift(1)
    out = pd.DataFrame({
        "ret": precios.pct_change(),
        "logret": np.log(precios / precios.shift(1)),
        "rf": rf,
    })
    out["exceso"] = out["ret"] - out["rf"]
    return out.dropna()


def features_shu(exceso: pd.Series) -> pd.DataFrame:
    """Features de la tabla 2 del paper sobre la rentabilidad en exceso (EWM, causales)."""
    neg2 = np.minimum(exceso, 0.0) ** 2
    feats = {}
    dd10 = np.sqrt(neg2.ewm(halflife=SEMIVIDA_DD).mean())
    feats["DD_10"] = dd10
    for h in SEMIVIDAS_SORTINO:
        dd = np.sqrt(neg2.ewm(halflife=h).mean())
        feats[f"sortino_{h}"] = exceso.ewm(halflife=h).mean() / dd
    out = pd.DataFrame(feats, index=exceso.index).replace([np.inf, -np.inf], np.nan)
    return out.dropna()


def sharpe_estrategia(senal_crisis: pd.Series, exceso: pd.Series, *, desde: int,
                      coste: float = COSTE, retraso: int = RETRASO) -> float:
    """Sharpe anualizado de la estrategia 0/1 del paper en las filas ``[desde:]``.

    ``senal_crisis`` (0/1, 1 = crisis) y ``exceso`` comparten índice. La señal de la fila t fija la
    posición desde t + ``retraso``; en crisis se está en el T-bill (rentabilidad en exceso 0). Cada
    cambio de posición cuesta ``coste``.
    """
    pos = (1.0 - senal_crisis.astype(float)).shift(retraso).fillna(1.0)
    cambios = pos.diff().abs().fillna(0.0)
    r = (pos * exceso - coste * cambios).iloc[desde:]
    sd = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    if not np.isfinite(sd) or sd <= 0:
        return 0.0
    return float(r.mean() / sd * np.sqrt(DIAS_ANIO))


def elegir_lambda(sharpes: dict[float, float], tol: float = 1e-12) -> float:
    """λ con mayor Sharpe; en empate (|ΔSharpe| ≤ ``tol``), la mayor."""
    mejor = max(sharpes.values())
    return max(lam for lam, s in sharpes.items() if s >= mejor - tol)


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
class ShuJumpModel(RegimeDetector):
    """D15: jump model de 2 estados de Shu, Yu y Mulvey (2024) con λ elegida por Sharpe online.

    Parameters
    ----------
    lambdas : rejilla de penalizaciones (ADR-010 §3.1).
    ventana_ajuste : máximo de días de ajuste (3.000, ventana móvil del paper).
    validacion_max : máximo de días de validación (2.016 = 8 años).
    coste : coste por operación en cada sentido (10 pb).
    n_init : arranques del descenso por coordenadas de ``jumpmodels`` (10).
    random_state : semilla de ``jumpmodels``.
    precios, tipos : series crudas del S&P 500 y de ``DTB3`` (solo para tests); por defecto se leen
        de ``data/raw``.
    """

    def __init__(
        self,
        lambdas: tuple[float, ...] | list[float] = LAMBDAS,
        ventana_ajuste: int = VENTANA_AJUSTE,
        validacion_max: int = VALIDACION_MAX,
        coste: float = COSTE,
        n_init: int = 10,
        random_state: int | None = 42,
        precios: pd.Series | None = None,
        tipos: pd.Series | None = None,
    ) -> None:
        super().__init__(n_states=2, random_state=random_state)
        self.lambdas = tuple(float(x) for x in lambdas)
        self.ventana_ajuste = int(ventana_ajuste)
        self.validacion_max = int(validacion_max)
        self.coste = float(coste)
        self.n_init = int(n_init)
        self._precios = precios
        self._tipos = tipos
        self._model = None
        self._scaler: StandardScaler | None = None
        self._inicio_ajuste: pd.Timestamp | None = None
        # Para el informe (ADR-010 §3.1 y §5).
        self.lambda_elegida: float | None = None
        self.sharpe_validacion: dict[float, float] = {}
        self.n_historia: int = 0
        self.n_validacion: int = 0
        self.fin_ajuste: pd.Timestamp | None = None
        self.discrepa_regla_paper: bool | None = None

    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        return "shu_jump_model_k2"

    @property
    def bibliography(self) -> list[str]:
        return ["shu2024downside", "hmm_nystrup2020"]

    @property
    def inicio_ajuste(self) -> pd.Timestamp | None:
        return self._inicio_ajuste

    # ------------------------------------------------------------------ #
    def _base(self, hasta: pd.Timestamp) -> pd.DataFrame:
        """Rentabilidades y features con datos ≤ ``hasta`` (nunca días posteriores)."""
        if self._precios is not None and self._tipos is not None:
            precios, tipos = self._precios, self._tipos
        else:
            precios, tipos = _crudos(str(RUTA_SP500), str(RUTA_DTB3))
        base = base_exceso(precios.loc[:hasta], tipos.loc[:hasta])
        feats = features_shu(base["exceso"])
        return base.loc[feats.index].join(feats)

    def _ajustar(self, F: pd.DataFrame, lam: float):
        from jumpmodels.jump import JumpModel as _JM

        scaler = StandardScaler().fit(F.values)
        Xs = pd.DataFrame(scaler.transform(F.values), index=F.index, columns=F.columns)
        model = _JM(n_components=2, jump_penalty=float(lam), cont=False,
                    random_state=self.random_state, n_init=self.n_init, max_iter=1000)
        model.fit(Xs, sort_by=None)
        return model, scaler

    @staticmethod
    def _escalar(scaler: StandardScaler, F: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(scaler.transform(F.values), index=F.index, columns=F.columns)

    def fit(self, X_train: pd.DataFrame) -> "ShuJumpModel":
        fin = pd.Timestamp(X_train.index[-1])
        datos = self._base(fin)
        F = datos[list(FEATURES)]
        n = len(F)
        if n < 4:
            raise ValueError(f"{self.name}: solo {n} días de historia hasta {fin.date()}")
        V = min(self.validacion_max, n // 2)
        ini_fit = max(0, n - V - self.ventana_ajuste)
        F_fit = F.iloc[ini_fit:n - V]
        tramo = F.iloc[ini_fit:n]                   # ajuste + validación (contexto online)
        self.sharpe_validacion = {}
        for lam in self.lambdas:
            model, scaler = self._ajustar(F_fit, lam)
            raw_fit = np.asarray(model.predict(self._escalar(scaler, F_fit)))
            orden = self._economic_state_order(raw_fit, F_fit, datos["logret"].iloc[ini_fit:n - V])
            crisis_interna = int(orden[-1])
            online = np.asarray(model.predict_online(self._escalar(scaler, tramo)))
            senal = pd.Series((online == crisis_interna).astype(int), index=tramo.index)
            self.sharpe_validacion[lam] = sharpe_estrategia(
                senal, datos["exceso"].iloc[ini_fit:n], desde=len(F_fit), coste=self.coste)
        self.lambda_elegida = elegir_lambda(self.sharpe_validacion)

        ini = max(0, n - self.ventana_ajuste)
        F_final = F.iloc[ini:n]
        self._model, self._scaler = self._ajustar(F_final, self.lambda_elegida)
        self._inicio_ajuste = pd.Timestamp(F_final.index[0])
        self.fin_ajuste = fin
        self.n_historia, self.n_validacion = n, V
        self._datos_ajuste = datos.iloc[ini:n]
        self._is_fitted = True
        self.label_states_economically(X_train)
        return self

    # ------------------------------------------------------------------ #
    def label_states_economically(self, X: pd.DataFrame, market_returns: pd.Series | None = None
                                  ) -> "ShuJumpModel":
        """Canonicalización de ``base.py`` sobre la ventana de ajuste (no sobre todo ``X``).

        El modelo se ajusta en los últimos ≤ 3.000 días de la historia, no en todo el train
        expansivo que recibe ``fit``; por eso el orden se fija con esos días y con su retorno
        logarítmico del S&P 500, que es la definición de ``SP500_ret``. ``X`` y ``market_returns``
        se aceptan por compatibilidad con ``walk_forward`` y no se usan.
        """
        self._check_fitted()
        d = self._datos_ajuste
        F = d[list(FEATURES)]
        raw = np.asarray(self._model.predict(self._escalar(self._scaler, F)))
        self._canonical_order = self._economic_state_order(raw, F, d["logret"])
        # Regla del paper: bajista = menor rentabilidad acumulada en exceso.
        cum = {s: float(d["exceso"].values[raw == s].sum()) for s in np.unique(raw)}
        bajista_paper = min(cum, key=cum.get) if len(cum) == 2 else None
        self.discrepa_regla_paper = (bajista_paper is not None
                                     and int(bajista_paper) != int(self._canonical_order[-1]))
        return self

    def resumen_reajuste(self) -> dict:
        """Ficha del último ``fit`` para ``reajustes/<P>_D15-U.csv`` (ADR-010 §3 y §3.1): λ elegida,
        Sharpe de validación por λ, tamaños y si la regla del paper discrepa del orden canónico."""
        if not self._is_fitted:
            return {}
        out = {"fin_train": self.fin_ajuste, "inicio_ajuste": self._inicio_ajuste,
               "n_historia": int(self.n_historia), "n_validacion": int(self.n_validacion),
               "lambda_elegida": self.lambda_elegida,
               "discrepa_regla_paper": self.discrepa_regla_paper,
               "sharpe_validacion": "|".join(f"{lam:g}={s:.6g}" for lam, s in self.sharpe_validacion.items())}
        if self._canonical_order is not None:
            out["orden_canonico"] = "|".join(str(int(o)) for o in self._canonical_order)
        return out

    # ------------------------------------------------------------------ #
    def _tramo_prediccion(self, X: pd.DataFrame) -> pd.DataFrame:
        hasta = pd.Timestamp(X.index[-1])
        F = self._base(hasta)[list(FEATURES)]
        desde = min(self._inicio_ajuste, pd.Timestamp(X.index[0]))
        return F.loc[desde:]

    def _a_fechas(self, estados: np.ndarray, F: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
        # Una fecha de X anterior al primer día con features (no ocurre en las pistas) recibe el
        # estado interno de calma.
        calma = 0 if self._canonical_order is None else int(self._canonical_order[0])
        s = pd.Series(estados, index=F.index).reindex(X.index, method="ffill")
        return s.fillna(calma).astype(int).to_numpy()

    def _predict_states(self, X: pd.DataFrame) -> np.ndarray:
        """Etiquetas internas con la DP sobre toda la ventana (solo uso in-sample)."""
        F = self._tramo_prediccion(X)
        raw = np.asarray(self._model.predict(self._escalar(self._scaler, F)))
        return self._a_fechas(raw, F, X)

    def predict_online(self, X: pd.DataFrame, refit: bool = False) -> np.ndarray:
        """Estados canónicos causales: filtrado online desde el inicio de la ventana de ajuste."""
        self._check_fitted()
        F = self._tramo_prediccion(X)
        raw = np.asarray(self._model.predict_online(self._escalar(self._scaler, F)))
        return self._apply_canonical(self._a_fechas(raw, F, X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """One-hot de ``predict_online`` (el jump model discreto da estados duros)."""
        estados = self.predict_online(X)
        proba = np.zeros((len(estados), self.n_states), dtype=float)
        proba[np.arange(len(estados)), estados] = 1.0
        return proba
