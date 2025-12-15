# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
project = "Ordinis"
copyright = "2025, Ordinis Team"
author = "Ordinis Team"

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinxcontrib.mermaid",
    "sphinx_copybutton",
    "sphinx_inline_tabs",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for MyST Parser -------------------------------------------------
myst_enable_extensions = ["colon_fence", "deflist", "html_admonition", "html_image"]

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
