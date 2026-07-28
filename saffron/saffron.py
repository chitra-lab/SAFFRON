"""
saffron.saffron
================
The `SAFFRON` class ties together the two steps of the workflow: fit an SAE on
embeddings `Z`, then evaluate the resulting sparse features against a known
spatial-variation measure — first by per-feature correlation, then by finding
a minimal reconstructing subset with OMP.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .correlate import CorrelationResult, spearman_scan
from .omp import OMPCurveResult, OMPSelection, omp_r2_curve, omp_select
from .sae import SAEResult, SparseAutoencoder, train_sae


def _is_adata(obj) -> bool:
    return hasattr(obj, "obsm") and hasattr(obj, "obs")


@dataclass
class SAFFRONResult:
    """Output of `SAFFRON.evaluate`: per-feature correlation, plus the OMP curve
    and selection if OMP was run (both `None` if the correlation was too weak)."""
    correlation: CorrelationResult
    omp_curve: OMPCurveResult | None
    omp_selection: OMPSelection | None


class SAFFRON:
    """Fit an SAE once on a set of embeddings, then evaluate it against any number
    of known spatial-variation measures without retraining.

    Example
    -------
        model = SAFFRON()
        model.fit(Z)                       # Z: (N, D) embeddings or gene expression
        result = model.evaluate(tau)       # tau: (N,) known spatial-variation measure
        print(result.correlation.best_rho, result.omp_selection.selected_features)
    """

    def __init__(
        self,
        latent_dim: int | None = None,
        k: int | None = None,
        sparsity_mode: str = "batchtopk",
        matryoshka_dims: list[int] | None = None,
        rho_threshold: float = 0.3,
        omp_max_features: int = 20,
        **train_kwargs,
    ):
        self.latent_dim = latent_dim
        self.k = k
        self.sparsity_mode = sparsity_mode
        self.matryoshka_dims = matryoshka_dims
        self.rho_threshold = rho_threshold
        self.omp_max_features = omp_max_features
        self.train_kwargs = train_kwargs

        self.sae_result_: SAEResult | None = None

    @property
    def model(self) -> SparseAutoencoder:
        if self.sae_result_ is None:
            raise RuntimeError("Call .fit(Z) before accessing the trained model.")
        return self.sae_result_.model

    def fit(self, data, embedding_key: str = "X_embedding", features_key: str | None = None,
            **override_kwargs) -> "SAFFRON":
        """Train the SAE on embeddings/expression matrix `data` (N, D), or an AnnData with
        `data.obsm[embedding_key]`. If `features_key` is given and already in `data.obsm`, load
        it instead of training; otherwise train and store the result there."""
        if _is_adata(data):
            if features_key and features_key in data.obsm:
                self.sae_result_ = SAEResult(Z=np.asarray(data.obsm[features_key]))
                return self
            if embedding_key not in data.obsm:
                raise KeyError(f"{embedding_key!r} not in adata.obsm; available keys: {list(data.obsm.keys())}")
            self.fit(np.asarray(data.obsm[embedding_key]), **override_kwargs)
            if features_key:
                data.obsm[features_key] = self.sae_result_.Z
            return self

        kwargs = {**self.train_kwargs, **override_kwargs}
        self.sae_result_ = train_sae(
            data,
            latent_dim=self.latent_dim,
            k=self.k,
            sparsity_mode=self.sparsity_mode,
            matryoshka_dims=self.matryoshka_dims,
            **kwargs,
        )
        return self

    def transform(self, Z: np.ndarray) -> np.ndarray:
        """Encode Z into sparse SAE features using the fitted model."""
        if self.model is None:
            raise RuntimeError("No trained encoder available (features were loaded from "
                                "adata.obsm, not trained) — call .fit() without features_key to train one.")
        return self.model.encode(Z)

    def evaluate(
        self,
        tau,
        H: np.ndarray | None = None,
        tau_key: str | None = None,
        features_key: str | None = None,
        restrict_support: bool = True,
        min_support: int = 10,
        run_omp: bool | None = None,
        omp_max_features: int | None = None,
    ) -> SAFFRONResult:
        """Correlate feature matrix `H` against `tau`, then run OMP if warranted. `tau` may be
        an (N,) array, or an AnnData with `tau.obs[tau_key]` (and `H` read from
        `tau.obsm[features_key or "sae_features"]`, unless `H` is given explicitly)."""
        if _is_adata(tau):
            adata = tau
            if H is None:
                key = features_key or "sae_features"
                if key not in adata.obsm:
                    raise KeyError(f"{key!r} not in adata.obsm; available keys: {list(adata.obsm.keys())}. "
                                    "Pass H explicitly, or features_key matching a key in adata.obsm.")
                H = np.asarray(adata.obsm[key])
            if tau_key is None:
                raise ValueError("tau_key is required when tau is an AnnData (which column of adata.obs to use).")
            if tau_key not in adata.obs:
                raise KeyError(f"{tau_key!r} not in adata.obs; available keys: {list(adata.obs.columns)}")
            tau = np.asarray(adata.obs[tau_key].values)
        elif H is None:
            if self.sae_result_ is None:
                raise RuntimeError("Call .fit(Z) first, or pass H explicitly.")
            H = self.sae_result_.Z

        if H.shape[0] != len(tau):
            raise ValueError(f"H has {H.shape[0]} rows but tau has {len(tau)} — they must cover "
                              "the same cells (e.g. subset H/adata to tau's cells first).")

        corr = spearman_scan(H, tau, restrict_support=restrict_support, min_support=min_support)

        do_omp = run_omp if run_omp is not None else (abs(corr.best_rho) >= self.rho_threshold)
        omp_curve = omp_selection = None
        if do_omp:
            max_feat = omp_max_features or self.omp_max_features
            omp_curve = omp_r2_curve(H, tau, max_features=max_feat)
            omp_selection = omp_select(H, tau, max_features=max_feat)

        return SAFFRONResult(correlation=corr, omp_curve=omp_curve, omp_selection=omp_selection)
