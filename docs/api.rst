API
===

.. currentmodule:: saffron

Pipeline
--------

The two-step SAFFRON workflow: fit an SAE, then evaluate it against a known
spatial-variation measure.

.. autosummary::
    :toctree: generated
    :nosignatures:

    SAFFRON
    SAFFRONResult

Sparse autoencoder
-------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    SparseAutoencoder
    SAEResult
    train_sae
    save_checkpoint
    load_checkpoint

Correlation
-----------

.. autosummary::
    :toctree: generated
    :nosignatures:

    CorrelationResult
    spearman_scan
    top_features

Orthogonal matching pursuit
-----------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    OMPCurveResult
    OMPSelection
    OMPReconstruction
    omp_r2_curve
    omp_select
    omp_reconstruct

Utilities
---------

.. autosummary::
    :toctree: generated
    :nosignatures:

    select_domains
    to_dense
    gene_row
    plot_spatial_grid
    plot_spatial_scatter
    plot_gene_tracks
    plot_omp_curve
