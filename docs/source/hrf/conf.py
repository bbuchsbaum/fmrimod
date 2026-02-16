# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import fmrimod

# -- Project information -----------------------------------------------------

project = "fmrimod"
copyright = "2024, Bradley Buchsbaum"
author = "Bradley Buchsbaum"
release = fmrimod.__version__
version = ".".join(release.split(".")[:2])
language = "en"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Autodoc / autosummary --------------------------------------------------

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autosummary_generate = True

# Suppress duplicate object warnings from dataclass fields
suppress_warnings = ["autodoc.duplicate_object"]

# -- Napoleon settings -------------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# -- Intersphinx mapping ----------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# -- Plot directive ----------------------------------------------------------

plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = [("png", 150)]

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    # Sidebar
    "show_nav_level": 2,
    "navigation_depth": 3,
    "collapse_navigation": False,
    # Header
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "use_edit_page_button": True,
    "search_bar_text": "Search the docs...",
    # Icon links
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/bbuchsbaum/fmrimod",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/fmrimod",
            "icon": "fa-brands fa-python",
        },
    ],
}

html_context = {
    "default_mode": "auto",
    "github_user": "bbuchsbaum",
    "github_repo": "fmrimod",
    "github_version": "main",
    "doc_path": "docs/source",
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]

# -- Copybutton settings ----------------------------------------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
