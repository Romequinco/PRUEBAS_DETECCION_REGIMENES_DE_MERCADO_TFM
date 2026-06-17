"""
deep_ae_regime.py — D12 `deep_ae_regime` (FASE 3, Tanda 4 — EXPLORATORIA).

Contraste ABLATIVO honesto, no una apuesta por "ganar". Pregunta única:
¿un reductor de dimensión NO LINEAL (autoencoder) sobre las 15 features causales
aporta algo, para detectar regímenes, frente al reductor LINEAL (PCA) del baseline
D3 (clustering GMM)? Con ~4 crisis reales en la muestra, el aprendizaje profundo
está muy limitado y un resultado NEGATIVO (AE≈PCA o peor) es ACEPTABLE y esperado
(ver CHECKPOINT 2). El valor está en la comparación limpia, gane o pierda.

Variante PRINCIPAL: **AE → GMM sobre el latente**.
- Un autoencoder LIGERO y regularizado (1 capa oculta, dropout + weight-decay,
  pocas épocas) comprime las 15 features a un espacio latente de baja dimensión
  (`latent_dim`=2 por defecto). Sobre ese latente se ajusta una mixtura gaussiana
  (`GaussianMixture`, full) con K=`n_states` estados, igual que D3 pero sobre el
  código no lineal en vez de sobre las features crudas.
- Complemento expuesto (no es la variante principal): `reconstruction_error(X)`
  da el MSE de reconstrucción por día, un score de anomalía/"rareza" que puede
  inspeccionarse como detector de crisis alternativo.

El BASELINE ablativo lineal **PCA → GMM** (misma dimensión latente, mismo K) está
implementado en `PCAGMMBaseline` (mismo archivo) para que el notebook lo pase por
el MISMO `walk_forward` y aísle exclusivamente el efecto de la NO LINEALIDAD.

Causalidad
----------
- El AE (y la estandarización por fold) se ajustan SOLO con `X_train`. Las features
  ya son z causales; aquí se re-centra/escala con estadísticos del TRAIN del fold
  (causal, nunca con toda la muestra) porque tres columnas — `corr_spx_bond`,
  `SP500_drawdown`, `SP500_momentum` — NO son z-scores y conviene homogeneizar la
  escala antes de la red. Los estadísticos se CONGELAN tras el fit.
- `predict`/`predict_proba` codifican el bloque de test con esos parámetros
  congelados (model.eval(), sin gradiente) → sin look-ahead. El wrapper causal
  walk-forward (re-fit expanding por fold) lo da `evaluation.walk_forward`.

Reproducibilidad: semillas fijas (torch + numpy) en cada fit.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture

from src.detector_base import RegimeDetector


def _seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)


class _Autoencoder(nn.Module):
    """AE denso minúsculo: 15 -> hidden -> latent -> hidden -> 15.

    Una sola capa oculta por rama, ReLU y dropout en el encoder. Deliberadamente
    pequeño para no sobreajustar con tan pocos datos/crisis.
    """

    def __init__(self, n_features: int, hidden: int, latent: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class DeepAERegime(RegimeDetector):
    """D12: autoencoder ligero (PyTorch) + GMM sobre el latente.

    Detector PROBABILÍSTICO: sobreescribe `predict_proba` con los posteriores del
    GMM ajustado en el espacio latente, reordenados al orden canónico económico
    (0=calma .. n-1=crisis). No expone logL del modelo de datos (el AE no es
    generativo); `score` devuelve NaN → AIC/BIC quedan NaN, como en otros
    detectores no generativos.
    """

    def __init__(
        self,
        n_states: int = 2,
        *,
        latent_dim: int = 2,
        hidden: int = 8,
        epochs: int = 40,
        lr: float = 1e-2,
        weight_decay: float = 1e-3,
        dropout: float = 0.10,
        gmm_n_init: int = 3,
        random_state: int | None = 42,
    ) -> None:
        super().__init__(n_states=n_states, random_state=random_state)
        self.latent_dim = int(latent_dim)
        self.hidden = int(hidden)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.dropout = float(dropout)
        self.gmm_n_init = int(gmm_n_init)
        # Estado ajustado (congelado tras fit).
        self._ae: _Autoencoder | None = None
        self._gmm: GaussianMixture | None = None
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None
        self._n_features: int | None = None

    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        return f"deep_ae_regime_k{self.n_states}"

    @property
    def bibliography(self) -> list[str]:
        # Claves verificadas en docs/references.bib.
        return [
            "nn_kingma2014",      # VAE / autoencoders variacionales (reductor no lineal)
            "nn_akioyamen2021",   # deep learning para detección de regímenes de mercado
            "nn_bucci2021",       # redes neuronales para regímenes/volatilidad
            "lopezdeprado2018",   # ML financiero / pipeline clustering + validación
        ]

    # ------------------------------------------------------------------ #
    def _standardize_fit(self, X) -> np.ndarray:
        """Calcula mu/sd con el TRAIN (causal) y devuelve la matriz estandarizada."""
        M = np.asarray(X.values, dtype=np.float64)
        self._mu = M.mean(axis=0)
        self._sd = M.std(axis=0)
        self._sd[self._sd < 1e-8] = 1.0
        return (M - self._mu) / self._sd

    def _standardize_apply(self, X) -> np.ndarray:
        M = np.asarray(X.values, dtype=np.float64)
        return (M - self._mu) / self._sd

    def fit(self, X_train) -> "DeepAERegime":
        _seed_everything(self.random_state)
        Z = self._standardize_fit(X_train)
        self._n_features = Z.shape[1]
        Xt = torch.tensor(Z, dtype=torch.float32)

        self._ae = _Autoencoder(self._n_features, self.hidden, self.latent_dim, self.dropout)
        opt = torch.optim.Adam(
            self._ae.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.MSELoss()
        self._ae.train()
        for _ in range(self.epochs):  # full-batch: datos pequeños, barato y estable
            opt.zero_grad()
            out = self._ae(Xt)
            loss = loss_fn(out, Xt)
            loss.backward()
            opt.step()

        # Latente del train -> GMM (mismo esquema que D3 pero sobre el código AE).
        self._ae.eval()
        with torch.no_grad():
            lat = self._ae.encode(Xt).numpy()
        self._gmm = GaussianMixture(
            n_components=self.n_states,
            covariance_type="full",
            random_state=self.random_state,
            n_init=self.gmm_n_init,
            reg_covar=1e-6,
            max_iter=300,
        ).fit(lat)

        self._is_fitted = True
        self.label_states_economically(X_train)
        return self

    # ------------------------------------------------------------------ #
    def _latent(self, X) -> np.ndarray:
        Z = self._standardize_apply(X)
        self._ae.eval()
        with torch.no_grad():
            return self._ae.encode(torch.tensor(Z, dtype=torch.float32)).numpy()

    def _predict_states(self, X) -> np.ndarray:
        """Etiquetas INTERNAS del GMM sobre el latente (sin canonicalizar)."""
        return self._gmm.predict(self._latent(X))

    def predict_proba(self, X) -> np.ndarray:
        """Posteriores del GMM en el latente, reordenados al orden CANÓNICO."""
        self._check_fitted()
        raw = self._gmm.predict_proba(self._latent(X))
        if self._canonical_order is None:
            return raw
        return raw[:, self._canonical_order]

    def reconstruction_error(self, X) -> np.ndarray:
        """MSE de reconstrucción por fila (score de anomalía/"rareza"; complemento).

        No es la variante principal: se expone para inspección (días donde el AE
        no sabe reconstruir el vector de mercado = configuraciones atípicas).
        """
        self._check_fitted()
        Z = self._standardize_apply(X)
        self._ae.eval()
        with torch.no_grad():
            t = torch.tensor(Z, dtype=torch.float32)
            rec = self._ae(t).numpy()
        return ((Z - rec) ** 2).mean(axis=1)

    # AE no es generativo del dato -> sin logL comparable. NaN (como D1/D7/D10).
    def score(self, X) -> float:
        return float("nan")

    def n_parameters(self) -> int:
        return -1


class PCAGMMBaseline(RegimeDetector):
    """Baseline ablativo LINEAL: PCA (mismas componentes) -> GMM (mismo K).

    Idéntico a `DeepAERegime` salvo que el reductor es PCA lineal en vez del AE.
    Comparten estandarización causal por fold, K y dimensión latente, de modo que
    cualquier diferencia de métricas en el MISMO walk-forward se atribuye SOLO a la
    NO LINEALIDAD del autoencoder. Es el contraste limpio de la ficha D12.
    """

    def __init__(
        self,
        n_states: int = 2,
        *,
        latent_dim: int = 2,
        gmm_n_init: int = 3,
        random_state: int | None = 42,
    ) -> None:
        super().__init__(n_states=n_states, random_state=random_state)
        self.latent_dim = int(latent_dim)
        self.gmm_n_init = int(gmm_n_init)
        self._pca = None
        self._gmm: GaussianMixture | None = None
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None

    @property
    def name(self) -> str:
        return f"pca_gmm_baseline_k{self.n_states}"

    @property
    def bibliography(self) -> list[str]:
        return ["lopezdeprado2018", "nn_akioyamen2021"]

    def _standardize_fit(self, X) -> np.ndarray:
        M = np.asarray(X.values, dtype=np.float64)
        self._mu = M.mean(axis=0)
        self._sd = M.std(axis=0)
        self._sd[self._sd < 1e-8] = 1.0
        return (M - self._mu) / self._sd

    def _standardize_apply(self, X) -> np.ndarray:
        M = np.asarray(X.values, dtype=np.float64)
        return (M - self._mu) / self._sd

    def fit(self, X_train) -> "PCAGMMBaseline":
        from sklearn.decomposition import PCA

        _seed_everything(self.random_state)
        Z = self._standardize_fit(X_train)
        self._pca = PCA(n_components=self.latent_dim, random_state=self.random_state)
        lat = self._pca.fit_transform(Z)
        self._gmm = GaussianMixture(
            n_components=self.n_states,
            covariance_type="full",
            random_state=self.random_state,
            n_init=self.gmm_n_init,
            reg_covar=1e-6,
            max_iter=300,
        ).fit(lat)
        self._is_fitted = True
        self.label_states_economically(X_train)
        return self

    def _latent(self, X) -> np.ndarray:
        return self._pca.transform(self._standardize_apply(X))

    def _predict_states(self, X) -> np.ndarray:
        return self._gmm.predict(self._latent(X))

    def predict_proba(self, X) -> np.ndarray:
        self._check_fitted()
        raw = self._gmm.predict_proba(self._latent(X))
        if self._canonical_order is None:
            return raw
        return raw[:, self._canonical_order]

    def score(self, X) -> float:
        return float("nan")

    def n_parameters(self) -> int:
        return -1
