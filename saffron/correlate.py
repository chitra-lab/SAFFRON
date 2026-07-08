"""
saffron.correlate
==================
Correlate SAE-learned features against a known measure of spatial variation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata


@dataclass
class CorrelationResult:
    rho: np.ndarray          # (M,) signed Spearman correlation per feature
    best_feature: int        # argmax_m |rho[m]|
    best_rho: float          # rho[best_feature]
    support_size: np.ndarray  # (M,) number of cells used per feature (== N if restrict_support=False)


def _spearman_cols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized Spearman rho between every column of X and y."""
    n = X.shape[0]
    if n < 2:
        return np.zeros(X.shape[1])
    ry = rankdata(y).astype(np.float64) - (n + 1) / 2
    denom = np.sqrt((ry ** 2).sum())
    if denom == 0:
        return np.zeros(X.shape[1])
    ry = ry / denom
    Xr = np.apply_along_axis(rankdata, 0, X).astype(np.float64) - (n + 1) / 2
    norms = np.sqrt((Xr ** 2).sum(axis=0))
    norms[norms == 0] = 1.0
    return (Xr * ry[:, None]).sum(axis=0) / norms


def spearman_scan(
    H: np.ndarray,
    tau: np.ndarray,
    restrict_support: bool = True,
    min_support: int = 10,
) -> CorrelationResult:
    """Compute the Spearman correlation between every SAE feature (column of H) and τ.

    Parameters
    ----------
    H : (N, M) sparse SAE feature matrix (e.g. `SAEResult.Z`).
    tau : (N,) known measure of spatial variation (e.g. an isodepth axis, a disease
        signature score).
    restrict_support : if True (default), each feature's correlation is computed
        only over cells where that feature is nonzero. Features with fewer than
        `min_support` active cells get rho=0.
    min_support : minimum active-cell count required to compute a correlation
        when `restrict_support=True`.
    """
    H = np.asarray(H, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    N, M = H.shape

    if not restrict_support:
        rho = _spearman_cols(H, tau)
        support_size = np.full(M, N)
    else:
        rho = np.zeros(M)
        support_size = np.zeros(M, dtype=int)
        for m in range(M):
            mask = H[:, m] > 0
            support_size[m] = int(mask.sum())
            if support_size[m] < min_support:
                continue
            n = support_size[m]
            ry = rankdata(tau[mask]) - (n + 1) / 2
            rx = rankdata(H[mask, m]) - (n + 1) / 2
            denom = np.sqrt((ry ** 2).sum()) * np.sqrt((rx ** 2).sum())
            rho[m] = float((rx * ry).sum() / denom) if denom > 0 else 0.0

    best_feature = int(np.argmax(np.abs(rho)))
    return CorrelationResult(
        rho=rho, best_feature=best_feature, best_rho=float(rho[best_feature]),
        support_size=support_size,
    )
