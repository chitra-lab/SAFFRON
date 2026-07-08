# SAFFRON

SAFFRON is a mechanistic interpretability framework for evaluating whether spatial
transcriptomics foundation model (SFM) embeddings capture known sources of spatial
variation in gene expression, using a Matryoshka sparse autoencoder (SAE) to
decompose embeddings into sparse, human-interpretable features.

Given (1) a set of embeddings `Z` (from an SFM, a single-cell FM, or even raw gene
expression) and (2) a known measure of spatial variation `τ` for the same cells
(e.g. a 1-D isodepth axis, a disease-signature score, distance to a pathological
feature), SAFFRON trains a sparse autoencoder on `Z` and evaluates which of its
learned features track `τ`.

## 🚀 Quick Start

```bash
git clone <this-repo> && cd SAFFRON
pip install -e ".[demo]"   # core library + everything notebooks/demo.ipynb needs
```

We provide a complete interactive walkthrough in the notebook using the Novae
embeddings on the colorectal tumor dataset:

See `notebooks/demo.ipynb` to learn
(i) how to train a Matryoshka SAE on SFM embeddings to learn sparse, interpretable
features with SAFFRON, and
(ii) how to evaluate those features against a known spatial-variation measure (e.g.
a 1-D tumor-to-stroma axis) — correlating each feature with the measure and using
orthogonal matching pursuit (OMP) to find a minimal reconstructing subset.

## 📦 Dependencies

`pip install -e .` installs the core library (numpy, scipy, scikit-learn, torch).
Optional extras:

- `pip install -e ".[io]"` — adds `anndata`, for loading embeddings from `.h5ad`.
- `pip install -e ".[demo]"` — adds everything `notebooks/demo.ipynb` needs (`anndata`,
  `matplotlib`, `statsmodels`, `jupyter`, `ipykernel`).
