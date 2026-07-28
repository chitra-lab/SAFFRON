# SAFFRON

SAFFRON is a mechanistic interpretability framework for evaluating whether spatial
transcriptomics foundation model (SFM) embeddings capture known sources of spatial
variation in gene expression, using a Matryoshka sparse autoencoder (SAE) to
decompose embeddings into sparse, human-interpretable features.

Given (1) a set of embeddings `Z` (from an SFM, a single-cell FM, or even raw gene
expression) and (2) a known measure of spatial variation `τ` for the same cells
(e.g. a 1-D spatial axis, a disease-signature score, distance to a pathological
feature), SAFFRON trains a sparse autoencoder on `Z` and evaluates which of its
learned features track `τ`.

## 🚀 Quick Start

```bash
git clone https://github.com/chitra-lab/SAFFRON && cd SAFFRON
pip install -e .
```

Two interactive walkthroughs:

- `notebooks/global_spatial_variation_tutorial.ipynb` — Novae embeddings on the colorectal tumor
  dataset. Learn (i) how to train a Matryoshka SAE on SFM embeddings to learn sparse,
  interpretable features with SAFFRON, and (ii) how to evaluate those features
  against a **global** spatial-variation measure (a 1-D tumor-to-stroma axis) —
  correlating each feature with the measure and using orthogonal matching pursuit
  (OMP) to find a minimal reconstructing subset. Also checks the SAE features
  against raw gene expression by running the same correlation directly on genes.
- `notebooks/local_spatial_variation_tutorial.ipynb` — MERFISH Alzheimer's brain dataset. The
  same evaluation applied to **local** microenvironment measures around amyloid-β
  plaques instead of a global gradient, comparing a gene-expression SAE against an
  SFM's SAE.

## 📦 Dependencies

`pip install -e .` installs everything needed for the core SAFFRON pipeline and
both notebooks (numpy, scipy, scikit-learn, torch, anndata, matplotlib, jupyter,
ipykernel).
