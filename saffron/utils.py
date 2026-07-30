"""
saffron.utils
=============
AnnData loaders and small data/plotting utilities for spatial features and gene expression.
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


def select_domains(adata, domains: list, domain_key: str = "domain"):
    """Subset `adata` to rows whose `domain_key` is in `domains`."""
    return adata[adata.obs[domain_key].isin(domains)].copy()


def to_dense(X) -> np.ndarray:
    """Convert `X` to a dense `np.ndarray`, densifying first if it's a scipy sparse matrix."""
    return np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)


def gene_row(adata, gene: str, tau_key: str = "isodepth", spatial_key: str = "spatial",
             reconstructed: np.ndarray | None = None, reconstructed_mask: np.ndarray | None = None,
             color: str | None = None, group: np.ndarray | None = None,
             group_colors: dict | None = None) -> dict:
    """Build one `plot_gene_tracks` row for `gene`'s expression in `adata`, against
    `adata.obs[tau_key]` and an optional `reconstructed` axis. `reconstructed_mask` excludes
    points from that axis's trend line only; `group`/`group_colors` color the axis scatters
    by category instead of by `color`."""
    idx = list(adata.var_names).index(gene)
    expr = to_dense(adata.X[:, idx]).ravel()
    axes = {tau_key: adata.obs[tau_key].values}
    trend_mask = {}
    if reconstructed is not None:
        axes["reconstructed axis"] = np.asarray(reconstructed, dtype=np.float64)
        if reconstructed_mask is not None:
            trend_mask["reconstructed axis"] = np.asarray(reconstructed_mask, dtype=bool)
    row = {"label": gene, "spatial": adata.obsm[spatial_key], "expr": expr, "axes": axes}
    if trend_mask:
        row["trend_mask"] = trend_mask
    if color is not None:
        row["color"] = color
    if group is not None and group_colors is not None:
        row["group"] = np.asarray(group)
        row["group_colors"] = dict(group_colors)
    return row


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

    cf = ax.contourf(Xi, Yi, Zi, levels=n_fill, cmap=cmap, extend="both")
    if n_lines > 0:
        CS = ax.contour(Xi, Yi, Zi, levels=n_lines, colors="k", linewidths=lw, linestyles="solid")
        ax.clabel(CS, CS.levels, inline=True, fontsize=6, fmt=label_fmt)
    ax.set_aspect("equal")
    ax.axis("off")
    if add_colorbar and fig is not None:
        fig.colorbar(cf, ax=ax, shrink=0.75, label=colorbar_label)
    return cf


def plot_spatial_grid(adata, panels: dict, spatial_key: str = "spatial", cmap: str = "coolwarm",
                       figsize: tuple | None = None, suptitle: str | None = None):
    """Plot one `topo_map` per entry of `panels` (title -> per-cell values), side by side."""
    import matplotlib.pyplot as plt

    x, y = adata.obsm[spatial_key][:, 0], adata.obsm[spatial_key][:, 1]
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=figsize or (4 * n, 4))
    axes = [axes] if n == 1 else axes
    for ax, (title, vals) in zip(axes, panels.items()):
        topo_map(ax, x, y, np.asarray(vals), fig=fig, cmap=cmap, colorbar_label=title)
        ax.set_title(title, fontsize=10)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, y=1.02)
    plt.tight_layout()
    return fig, axes


def plot_spatial_scatter(adata, panels: dict, spatial_key: str = "spatial", cmap: str = "viridis",
                          markers: np.ndarray | None = None, dot_size: float = 18,
                          figsize: tuple | None = None, suptitle: str | None = None):
    """Per-cell scatter of `panels` (title -> per-cell values), side by side.

    `markers`, if given, is an (M, 2) array of extra point coordinates (e.g. plaque centroids)
    overlaid as red crosses on every panel.
    """
    import matplotlib.pyplot as plt

    x, y = adata.obsm[spatial_key][:, 0], adata.obsm[spatial_key][:, 1]
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=figsize or (4 * n, 4))
    axes = [axes] if n == 1 else axes
    for ax, (title, vals) in zip(axes, panels.items()):
        vals = norm01(np.asarray(vals, dtype=np.float64))
        sc = ax.scatter(x, y, c=vals, cmap=cmap, vmin=0, vmax=1, s=dot_size, linewidths=0, zorder=2)
        if markers is not None and len(markers):
            ax.scatter(markers[:, 0], markers[:, 1], marker="x", c="white", s=250, linewidths=5, zorder=4)
            ax.scatter(markers[:, 0], markers[:, 1], marker="x", c="red", s=140, linewidths=3, zorder=5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=10)
        fig.colorbar(sc, ax=ax, shrink=0.75, label=title)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, y=1.02)
    plt.tight_layout()
    return fig, axes


def _trend_line(x: np.ndarray, y: np.ndarray, frac: float = 0.35, mask: np.ndarray | None = None):
    """LOESS-smoothed trend of `y` vs `x`. If `mask` is given and leaves at least 20 points,
    the fit uses only those points; otherwise it uses all of them. Returns (sorted_x, smoothed_y)."""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    if mask is not None and np.sum(mask) >= 20:
        x, y = x[mask], y[mask]
    fit = lowess(y, x, frac=frac, return_sorted=True)
    return fit[:, 0], fit[:, 1]


def _auto_dot_size(ax, spatial: np.ndarray, fill_frac: float = 0.85) -> float:
    """Marker size (points^2) so neighboring points in `spatial` fill `fill_frac` of the gap
    between them, based on the axes' current data-to-points scale and the median
    nearest-neighbor spacing."""
    spacing = np.median(NearestNeighbors(n_neighbors=2).fit(spatial).kneighbors(spatial)[0][:, 1])
    p0 = ax.transData.transform((0, 0))
    p1 = ax.transData.transform((spacing, 0))
    px_per_unit = np.hypot(*(p1 - p0))
    pts_per_unit = px_per_unit * 72.0 / ax.figure.dpi
    return (fill_frac * pts_per_unit) ** 2


def plot_gene_tracks(rows: list[dict], cmap: str = "viridis", dot_size: float | None = None,
                      trend_frac: float = 0.35, figsize: tuple | None = None,
                      suptitle: str | None = None):
    """One row per entry of `rows`: a spatial scatter of expression, plus a scatter and
    LOESS trend line against every array in that entry's `axes` (values normalized to [0, 1]).
    Each entry needs `label`, `spatial`, `expr`, `axes`, and may set `trend_mask`, `color`,
    and `group`/`group_colors`; see `gene_row` for how these are built."""
    import matplotlib.pyplot as plt

    n_rows = len(rows)
    n_axes = len(rows[0]["axes"])
    n_cols = 1 + n_axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize or (3.6 * n_cols, 3.0 * n_rows))
    axes = np.atleast_2d(axes)
    colors = plt.get_cmap("tab10").colors

    for r, row in enumerate(rows):
        color = row.get("color") or colors[r % len(colors)]
        raw_expr = np.asarray(row["expr"], dtype=np.float64)
        vals = norm01(raw_expr)
        dropout_mask = raw_expr > 0

        ax = axes[r, 0]
        spatial = row["spatial"]
        ax.set_aspect("equal")
        pad = 0.02 * max(np.ptp(spatial[:, 0]), np.ptp(spatial[:, 1]))
        ax.set_xlim(spatial[:, 0].min() - pad, spatial[:, 0].max() + pad)
        ax.set_ylim(spatial[:, 1].min() - pad, spatial[:, 1].max() + pad)
        fig.canvas.draw()
        size = dot_size if dot_size is not None else _auto_dot_size(ax, spatial)
        sc = ax.scatter(spatial[:, 0], spatial[:, 1], c=vals, cmap=cmap, vmin=0, vmax=1,
                         s=size, linewidths=0)
        ax.axis("off")
        ax.set_title(row["label"], fontsize=11, fontweight="bold", color=color, loc="left")
        fig.colorbar(sc, ax=ax, shrink=0.75)

        group_colors = row.get("group_colors")
        point_colors = ([group_colors.get(g, color) for g in row["group"]]
                         if group_colors is not None else color)

        for c, (axis_label, xvals) in enumerate(row["axes"].items(), start=1):
            axc = axes[r, c]
            xvals = norm01(np.asarray(xvals, dtype=np.float64))
            axc.scatter(xvals, vals, s=8, alpha=0.35, color=point_colors, linewidths=0)
            extra = row.get("trend_mask", {}).get(axis_label)
            mask = (dropout_mask & extra) if extra is not None else dropout_mask
            xt, yt = _trend_line(xvals, vals, frac=trend_frac, mask=mask)
            axc.plot(xt, yt, color=color, lw=2)
            axc.set_xlabel(f"{axis_label} (normalized)")
            axc.set_xlim(-0.05, 1.05)
            axc.set_ylim(-0.05, 1.05)
            if c == 1:
                axc.set_ylabel("expression (normalized)")
            axc.spines[["top", "right"]].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=1.01)
    plt.tight_layout()
    return fig, axes


def plot_omp_curve(curve, tau_label: str = "τ", figsize: tuple = (5.5, 4.5)):
    """Plot cross-validated R2 vs. number of OMP features, against a dense RidgeCV reference."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(curve.k_values, curve.r2, "o-", color="#2166AC", label="OMP (sparse)")
    ax.axhline(curve.ridge_r2, color="#888888", ls="--", label="RidgeCV (all features)")
    ax.set_xlabel("Number of SAE features (k)")
    ax.set_ylabel(f"Cross-validated R² with {tau_label}")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax
