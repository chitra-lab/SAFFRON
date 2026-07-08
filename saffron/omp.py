"""
saffron.omp
===========
Identify a minimal set of SAE features that linearly reconstruct a known measure
of spatial variation, via orthogonal matching pursuit (OMP).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler


@dataclass
class OMPCurveResult:
    k_values: np.ndarray          # (max_features,) number of features, 1..max_features
    r2: np.ndarray                # (max_features,) cross-validated R2 at each k
    ridge_r2: float                # cross-validated R2 using all M features (RidgeCV reference)


@dataclass
class OMPSelection:
    selected_features: list[int]   # feature indices in the order OMP added them
    coefficients: np.ndarray       # (len(selected_features),) coefficient for each, same order
    r2: float                      # in-sample R2 of the final fit at max_features


def _standardize(H: np.ndarray, tau: np.ndarray):
    Hs = StandardScaler().fit_transform(np.asarray(H, dtype=np.float64))
    ts = np.asarray(tau, dtype=np.float64)
    ts = (ts - ts.mean()) / (ts.std() + 1e-12)
    return Hs, ts


def omp_r2_curve(
    H: np.ndarray,
    tau: np.ndarray,
    max_features: int = 20,
    cv_folds: int = 5,
    seed: int = 0,
) -> OMPCurveResult:
    """Cross-validated R^2 of OMP reconstruction of τ from H, for k=1..max_features.

    Also fits a RidgeCV model using all M features as a dense reference (how much
    reconstruction quality is achievable in principle vs with a sparse subset).
    """
    Hs, ts = _standardize(H, tau)
    max_features = min(max_features, Hs.shape[1])
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    def cv_r2(model):
        return r2_score(ts, cross_val_predict(model, Hs, ts, cv=kf))

    r2_values = np.array([
        cv_r2(OrthogonalMatchingPursuit(n_nonzero_coefs=k))
        for k in range(1, max_features + 1)
    ])
    ridge_r2 = cv_r2(RidgeCV(alphas=np.logspace(-2, 6, 20)))

    return OMPCurveResult(k_values=np.arange(1, max_features + 1), r2=r2_values, ridge_r2=ridge_r2)


def omp_select(H: np.ndarray, tau: np.ndarray, max_features: int = 20) -> OMPSelection:
    """Fit OMP once at `max_features` and recover the greedy feature-selection order.

    Refits OMP at every k=1..max_features (in-sample) and records, at each step, the
    single new feature index that entered the solution — this is the standard way to
    recover OMP's greedy order since `n_nonzero_coefs=k` is not guaranteed to return a
    strict superset of the `k-1` solution, but in practice does so almost always for
    well-separated sparse signals.
    """
    Hs, ts = _standardize(H, tau)
    max_features = min(max_features, Hs.shape[1])

    selected_at: dict[int, list[int]] = {}
    for k in range(1, max_features + 1):
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=k, fit_intercept=True)
        omp.fit(Hs, ts)
        selected_at[k] = list(np.where(omp.coef_ != 0)[0])

    ordered: list[int] = []
    prev: list[int] = []
    for k in range(1, max_features + 1):
        new = [f for f in selected_at[k] if f not in prev]
        if new:
            ordered.append(new[0])
        prev = selected_at[k]

    final_omp = OrthogonalMatchingPursuit(n_nonzero_coefs=len(ordered) or 1, fit_intercept=True)
    final_omp.fit(Hs, ts)
    coef = final_omp.coef_[ordered]
    r2 = float(r2_score(ts, final_omp.predict(Hs)))

    return OMPSelection(selected_features=ordered, coefficients=coef, r2=r2)
