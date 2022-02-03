# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
os.environ['pyna_language'] = 'python'
import sys
sys.path.insert(0, os.path.abspath('../../pyNA/'))
sys.path.insert(0, os.path.abspath('../../pyNA/run/'))
sys.path.insert(0, os.path.abspath('../../pyNA/src/'))
sys.path.insert(0, os.path.abspath('../../pyNA/src/noise_src_py/'))
sys.path.insert(0, os.path.abspath('../../pyNA/src/trajectory_src/'))

# Command to build documentation in the docs folder
# sphinx-build ./_rst .   

# -- Project information -----------------------------------------------------

project = 'pyNA'
copyright = '2021, Laurens Voet'
author = 'Laurens Voet'

# The full version, including alpha/beta/rc tags
release = '1.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
			  'sphinx.ext.todo',
			  'sphinx.ext.coverage',
			  'sphinx.ext.mathjax',
			  'sphinx.ext.viewcode',
			  'sphinx.ext.githubpages',
			  'sphinx.ext.napoleon',]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Make sure not to skip __init__ functions --------------------------------
def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip

def setup(app):
    app.connect("autodoc-skip-member", skip)
