"""
SAFFRON — a Sparse Autoencoder Framework For Representing Omics Natural variation.
"""

from .correlate import CorrelationResult, spearman_scan
from .omp import OMPCurveResult, OMPSelection, omp_r2_curve, omp_select
from .saffron import SAFFRON, SAFFRONResult
from .sae import SAEResult, SparseAutoencoder, load_checkpoint, save_checkpoint, train_sae

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
    "OMPCurveResult",
    "OMPSelection",
    "omp_r2_curve",
    "omp_select",
]

__version__ = "0.1.0"
