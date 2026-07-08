"""
saffron.utils
=============
AnnData loaders and small plotting utilities used by demo.ipynb.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import griddata
from sklearn.neighbors import NearestNeighbors


def load_embedding(h5ad_path: str, embed_key: str = "X_embedding") -> np.ndarray:
    """Load an (N, D) embedding matrix from `adata.obsm[embed_key]`."""
    import anndata

    adata = anndata.read_h5ad(h5ad_path)
    if embed_key not in adata.obsm:
        raise KeyError(f"{embed_key!r} not found in adata.obsm; available keys: {list(adata.obsm.keys())}")
    X = np.asarray(adata.obsm[embed_key])
    return X.astype(np.float32)


def load_spatial_coords(h5ad_path: str, spatial_key: str = "spatial") -> np.ndarray | None:
    """Load (N, 2) spatial coordinates from `adata.obsm[spatial_key]`, if present."""
    import anndata

    adata = anndata.read_h5ad(h5ad_path)
    if spatial_key not in adata.obsm:
        return None
    return np.asarray(adata.obsm[spatial_key]).astype(np.float32)


def rotate_by_theta(coords: np.ndarray, theta: float) -> np.ndarray:
    """Rotate an (N, 2) coordinate array by angle theta (radians), about the origin."""
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array(((c, -s), (s, c)))
    return (rotation_matrix @ coords.T).T


def norm01(vals: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1] so different quantities share one color scale."""
    lo, hi = vals.min(), vals.max()
    return (vals - lo) / (hi - lo) if hi > lo else vals


def topo_map(ax, x, y, vals, fig=None, cmap="coolwarm", n_fill=20, n_lines=5,
             lw=0.8, label_fmt="%.2f", colorbar_label="", add_colorbar=True,
             grid_res=100, mask_factor=1.8, normalize=True):
    """Filled contour map with labeled contour lines over scattered (x, y, vals)."""
    if normalize:
        vals = norm01(np.asarray(vals, dtype=np.float64))

    xi = np.linspace(x.min(), x.max(), grid_res)
    yi = np.linspace(y.min(), y.max(), grid_res)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), vals, (Xi, Yi), method="linear")

    pts = np.c_[x, y]
    med_spacing = np.median(NearestNeighbors(n_neighbors=2).fit(pts).kneighbors(pts)[0][:, 1])
    grid_pts = np.c_[Xi.ravel(), Yi.ravel()]
    dists = NearestNeighbors(n_neighbors=1).fit(pts).kneighbors(grid_pts)[0].reshape(Xi.shape)
    Zi[dists > mask_factor * med_spacing] = np.nan
    Zi = np.ma.masked_invalid(Zi)

    cf = ax.contourf(Xi, Yi, Zi, levels=n_fill, cmap=cmap)
    if n_lines > 0:
        CS = ax.contour(Xi, Yi, Zi, levels=n_lines, colors="k", linewidths=lw, linestyles="solid")
        ax.clabel(CS, CS.levels, inline=True, fontsize=6, fmt=label_fmt)
    ax.set_aspect("equal")
    ax.axis("off")
    if add_colorbar and fig is not None:
        fig.colorbar(cf, ax=ax, shrink=0.75, label=colorbar_label)
    return cf
