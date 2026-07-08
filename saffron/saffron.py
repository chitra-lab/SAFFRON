"""
saffron.saffron
================
The `SAFFRON` class ties together the two steps of the workflow: fit an SAE on
embeddings `Z`, then evaluate the resulting sparse features against a known
spatial-variation measure  — first by per-feature correlation, then by finding a minimal reconstructing subset with OMP.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .correlate import CorrelationResult, spearman_scan
from .omp import OMPCurveResult, OMPSelection, omp_r2_curve, omp_select
from .sae import SAEResult, SparseAutoencoder, train_sae


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

    def fit(self, Z: np.ndarray, **override_kwargs) -> "SAFFRON":
        """Train the SAE on embeddings/expression matrix Z (N, D)."""
        kwargs = {**self.train_kwargs, **override_kwargs}
        self.sae_result_ = train_sae(
            Z,
            latent_dim=self.latent_dim,
            k=self.k,
            sparsity_mode=self.sparsity_mode,
            matryoshka_dims=self.matryoshka_dims,
            **kwargs,
        )
        return self

    def transform(self, Z: np.ndarray) -> np.ndarray:
        """Encode Z into sparse SAE features using the fitted model."""
        return self.model.encode(Z)

    def evaluate(
        self,
        tau: np.ndarray,
        H: np.ndarray | None = None,
        restrict_support: bool = True,
        min_support: int = 10,
        run_omp: bool | None = None,
        omp_max_features: int | None = None,
    ) -> SAFFRONResult:
        """Correlate SAE features against `tau`, then run OMP if warranted."""
        if H is None:
            if self.sae_result_ is None:
                raise RuntimeError("Call .fit(Z) first, or pass H explicitly.")
            H = self.sae_result_.Z

        corr = spearman_scan(H, tau, restrict_support=restrict_support, min_support=min_support)

        do_omp = run_omp if run_omp is not None else (abs(corr.best_rho) >= self.rho_threshold)
        omp_curve = omp_selection = None
        if do_omp:
            max_feat = omp_max_features or self.omp_max_features
            omp_curve = omp_r2_curve(H, tau, max_features=max_feat)
            omp_selection = omp_select(H, tau, max_features=max_feat)

        return SAFFRONResult(correlation=corr, omp_curve=omp_curve, omp_selection=omp_selection)
