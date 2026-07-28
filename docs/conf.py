# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from datetime import datetime

project = "SAFFRON"
author = "Chitra Lab"
copyright = f"{datetime.now():%Y}, {author}"

extensions = [
    "sphinx_copybutton",
    "nbsphinx",
]

source_suffix = {".rst": "restructuredtext"}
master_doc = "index"
exclude_patterns = ["**.ipynb_checkpoints", "build"]

# notebooks already have their outputs baked in (executed once, checked in) — don't re-run at build time
nbsphinx_execute = "never"

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {"navigation_depth": 4}
html_show_sphinx = False
