# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import datetime

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sys
from pathlib import Path

source_path = Path("../../../src").resolve()
sys.path.insert(0, str(source_path))

# -- Project information -----------------------------------------------------

project = "tbp.monty"
year = str(datetime.datetime.now().year)
copyright = f"{year}, Thousand Brains Project"  # noqa: A001
author = "Thousand Brains Project"

# The full version, including alpha/beta/rc tags

# Get version from package
import tbp.monty  # noqa: E402 isort:skip

version = tbp.monty.__version__
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"


html_theme_options = {
    "fixed_sidebar": True,
    "description": f"version {release}",
    "github_banner": True,
    "github_button": True,
    "github_user": "thousandbrainsproject",
    "github_repo": "tbp.monty",
    "navigation_depth": 5,
    "show_nav_level": 3,
    "show_toc_level": 3,
    "collapse_navigation": False,
    "logo": {
        "image_light": "_static/logo_light.png",
        "image_dark": "_static/logo_dark.png",
    },
    "icon_links": [
        {
            "name": "Twitter",
            "url": "https://x.com/1000brainsproj",
            "icon": "fa-brands fa-x-twitter",
        },
        {
            "name": "Bluesky",
            "url": "https://bsky.app/profile/thousandbrains.org",
            "icon": "fa-brands fa-bluesky",
        },
        {
            "name": "GitHub",
            "url": "https://github.com/thousandbrainsproject/tbp.monty",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "Discourse",
            "url": "https://thousandbrains.discourse.group/",
            "icon": "fa-brands fa-discourse",
        },
        {
            "name": "Docs",
            "url": "https://thousandbrainsproject.readme.io/",
            "icon": "fa-brands fa-readme",
        },
    ],
}

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_logo = "_static/logo.png"
html_favicon = "_static/favicon.png"

# Set the maximum depth for the table of contents
toc_maxdepth = 5


# -- Extension configuration -------------------------------------------------
autodoc_member_order = "groupwise"
autodoc_inherit_docstrings = True
autodoc_mock_imports = [
    "PIL",
    "gym",
    "habitat_sim",
    "magnum",
    "matplotlib",
    "numpy",
    "pandas",
    "quaternion",
    "scipy",
    "skimage",
    "sklearn",
    "torch",
    "torch_geometric",
]
autodoc_default_options = {
    # Make sphinx generate docs for specific dunder methods
    "special-members": "__init__",
}
# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False

# The autosummary directive can also optionally serve as a toctree entry for
# the included items. Optionally, stub .rst files for these items can also be
# automatically generated when autosummary_generate is True.
# autosummary_generate = True

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "habitat_sim": ("https://aihabitat.org/docs/habitat-sim/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "pillow": ("https://pillow.readthedocs.io/en/stable/", None),
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "torch_geometric": ("https://pytorch-geometric.readthedocs.io/en/latest/", None),
}

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True
todo_link_only = True

# remove the Built with Sphinx message
html_show_sphinx = False

# -- Options for sphinx_autodoc_typehints extension ----------------------------

always_use_bars_union = True
typehints_use_signature = True
typehints_use_signature_return = True
