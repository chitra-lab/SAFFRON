"""
SAFFRON — a Sparse Autoencoder Framework For Representing Omics Natural variation.
"""

from .correlate import CorrelationResult, spearman_scan, top_features
from .omp import OMPCurveResult, OMPReconstruction, OMPSelection, omp_r2_curve, omp_reconstruct, omp_select
from .saffron import SAFFRON, SAFFRONResult
from .sae import SAEResult, SparseAutoencoder, load_checkpoint, save_checkpoint, train_sae
from .utils import (
    gene_row, plot_gene_tracks, plot_omp_curve,
    plot_spatial_grid, plot_spatial_scatter, select_domains, to_dense,
)

__all__ = [
    "SAFFRON",
    "SAFFRONResult",
    "SparseAutoencoder",
    "SAEResult",
    "train_sae",
    "save_checkpoint",
    "load_checkpoint",
    "CorrelationResult",
    "spearman_scan",
    "top_features",
    "OMPCurveResult",
    "OMPSelection",
    "OMPReconstruction",
    "omp_r2_curve",
    "omp_select",
    "omp_reconstruct",
    "select_domains",
    "to_dense",
    "gene_row",
    "plot_spatial_grid",
    "plot_spatial_scatter",
    "plot_gene_tracks",
    "plot_omp_curve",
]

__version__ = "0.1.0"
