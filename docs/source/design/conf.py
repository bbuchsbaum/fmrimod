from importlib.util import find_spec
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))
else:
    sys.path.insert(0, str(ROOT))

import fmrimod

project = "fmrimod"
copyright = "2024, fmrimod developers"
author = "fmrimod developers"
release = fmrimod.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
]

if find_spec("sphinx_autodoc_typehints") is not None:
    extensions.append("sphinx_autodoc_typehints")

HAS_NBSPHINX = find_spec("nbsphinx") is not None
if HAS_NBSPHINX:
    extensions.append("nbsphinx")
    tags.add("has_nbsphinx")

templates_path = ["_templates"]
exclude_patterns = ["api/generated/pyfmridesign*.rst"]
if not HAS_NBSPHINX:
    exclude_patterns.append("notebooks/*.ipynb")

if find_spec("sphinx_rtd_theme") is not None:
    html_theme = "sphinx_rtd_theme"
else:
    html_theme = "alabaster"

STATIC_DIR = Path(__file__).parent / "_static"
html_static_path = ["_static"] if STATIC_DIR.exists() else []

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autodoc_typehints = "description"
autosummary_generate = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
]

nbsphinx_execute = "never"
nbsphinx_allow_errors = True
nbsphinx_kernel_name = "python3"
