project   = "CBOED"
author    = "Mohamed Doumbouya"
release   = "0.1.0"

extensions = [
    "myst_parser",
    "autoapi.extension",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

autoapi_dirs   = ["../src/cboed"]
html_theme     = "furo"
source_suffix  = {".rst": "restructuredtext", ".md": "markdown"}
