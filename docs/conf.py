# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ovrolwasolar'
copyright = '2023, OVROLWA SolarTeam'
author = 'OVROLWA SolarTeam'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
              'myst_parser',
              'sphinx.ext.napoleon',
              'sphinx.ext.autodoc', 
              'sphinx.ext.coverage', 
              'sphinxcontrib.programoutput',
              'sphinx.ext.autosummary']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

myst_enable_extensions = [
    "colon_fence",  # allows ::: fenced blocks
    "deflist",      # definition lists
    "linkify",      # auto-detect URLs
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_theme_options = {
    'navigation_depth': 2,
    'collapse_navigation': False,
    'titles_only': False
}

extensions.append('autoapi.extension')
autoapi_type = 'python'
autoapi_dirs = ['../ovrolwasolar']
