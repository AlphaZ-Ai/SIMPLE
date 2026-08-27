"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add src/ to Python path for autodoc
sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SIMPLE'
copyright = '2025, Songlin Wei'
author = 'Songlin Wei'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Google/Numpy docstrings
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinxemoji.sphinxemoji",
]

# Theme
html_theme = "furo"

# Optional: control sidebar behavior
html_theme_options = {
    "sidebar_hide_name": False,  # show project name at top
    # Furo auto-expands to current section by default
}

# MyST markdown config
# Generate anchors for h1-h4 so in-page links like [x](#some-heading) resolve
myst_heading_anchors = 4

myst_enable_extensions = [
    "dollarmath",  # For inline LaTeX
    "colon_fence"  # ::: fenced blocks
]

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"

templates_path = ['_templates']
# Superseded pages kept in the tree but not built:
#   nix-runtime.md / robo-nix.md -> replaced by nix-setup/runtime.md and
#   nix-setup/installation.md; *.old.md -> previous revisions of a tutorial;
#   tutorials/replay.md and tutorials/eval.md -> superseded by the per-pipeline
#   Replay & Render / Evaluation pages under decoupled-wbc/ and sonic-wbc/.
exclude_patterns = [
    "nix-runtime.md",
    "robo-nix.md",
    "**/*.old.md",
    "tutorials/replay.md",
    "tutorials/eval.md",
]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_static_path = ['_static']


autodoc_mock_imports = ["omni", "carb", "isaacsim"]
