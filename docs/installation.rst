Installation
============

Clone the repository and install with pip::

    git clone https://github.com/chitra-lab/SAFFRON
    cd SAFFRON
    pip install -e .

This installs everything needed for both the core SAFFRON pipeline and the
tutorial notebooks (numpy, scipy, scikit-learn, torch, anndata, matplotlib,
jupyter, ipykernel).

To build these docs locally, also install the ``docs`` extra::

    pip install -e ".[docs]"
    cd docs
    make html
