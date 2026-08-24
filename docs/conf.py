"""Sphinx configuration for Aether documentation."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "Aether Runtime"
copyright = "2026, Muhammad Kaleem Sajjad"
author = "Muhammad Kaleem Sajjad"
version = "1.2.4.charlie"
release = "1.2.4.charlie"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_click",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "sphinx_rtd_theme"
html_static_path = []
html_title = "Aether Runtime"

autodoc_typehints = "description"
autodoc_member_order = "bysource"

intersphinx_mapping = {}
suppress_warnings = ["myst.header", "intersphinx.external"]

