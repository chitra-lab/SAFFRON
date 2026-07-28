# SAFFRON

SAFFRON is a mechanistic interpretability framework for evaluating whether spatial
transcriptomics foundation model (SFM) embeddings capture known sources of spatial
variation in gene expression, using a Matryoshka sparse autoencoder (SAE) to
decompose embeddings into sparse, human-interpretable features.

Given (1) a set of embeddings `Z` (from an SFM, a single-cell FM, or even gene
expression) and (2) a known measure of spatial variation `τ` for the same cells
(e.g. a 1-D spatial axis, a disease-signature score, distance to a pathological
feature), SAFFRON trains a sparse autoencoder on `Z` and evaluates which of its
learned features track `τ`.

## 🚀 Quick Start

```bash
git clone https://github.com/chitra-lab/SAFFRON && cd SAFFRON
pip install -e .
```

## Getting started

Check out our [readthedocs](https://saffron-sae.readthedocs.io/en/latest/index.html), which
includes tutorials for two analyses:
- `notebooks/global_spatial_variation_tutorial.ipynb` — **Spatial gradients in the colorectal
  tumor-stroma boundary.** Novae embeddings on 10x Visium data, evaluated against a **global**
  1-D tumor-to-stroma gradient, including a check of the SAE features against raw gene expression.
- `notebooks/local_spatial_variation_tutorial.ipynb` — **Local microenvironment variation around
  Aβ plaques (MERFISH).** Compares a gene-expression SAE against a SFM's SAE on **local**
  microenvironment measures in an Alzheimer's disease brain dataset.

## 📦 Dependencies

`pip install -e .` installs everything needed for the core SAFFRON pipeline and
both notebooks (numpy, scipy, scikit-learn, torch, anndata, matplotlib, jupyter,
ipykernel).
