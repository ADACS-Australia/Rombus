# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = docs
BUILDDIR      = docs/_build

# Exclude the following (spec-separated) paths from the docs
PATH_EXCLUDE_LIST = docs/content python/${PKG_PROJECT}/tests python/${PKG_PROJECT}/models python/${PKG_PROJECT}/cli.py

# Set some variables needed by the documentation
PKG_PROJECT := $(shell poetry run python3 -c 'from tomllib import load;print(load(open("pyproject.toml","rb"))["project"]["name"])')
PKG_AUTHOR  := $(shell poetry run python3 -c 'from importlib.metadata import metadata; print(metadata("${PKG_PROJECT}")["author"])')
PKG_VERSION := $(word 2,$(shell poetry version))
PKG_RELEASE := ${PKG_VERSION}

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

clean-docs:
	@rm -rf build docs/_build
	@rm -f docs/conf.py docs/make.bat docs/Makefile
	@rm -f docs/*.rst docs/*.md

clean: clean-docs
	@rm -rf dist
	@rm -rf .tests
	@rm -rf .pytest_cache
	@rm -rf .cache
	@rm -rf .ropeproject
	@find . -type d -name "__pycache__" -exec rm -rf {} \;

apidoc: clean-docs
	@echo "Building documentation for:"
	@echo "   project: "${PKG_PROJECT}
	@echo "   author:  "${PKG_AUTHOR}
	@echo "   version: "${PKG_VERSION}
	@sphinx-apidoc -o docs --doc-project ${PKG_PROJECT} --doc-author "${PKG_AUTHOR}" --doc-version ${PKG_VERSION} --doc-release ${PKG_RELEASE} -t docs/_templates -T --extensions sphinx_click,sphinx.ext.doctest,sphinx.ext.mathjax,sphinx.ext.autosectionlabel,myst_parser,sphinx.ext.todo,sphinx_copybutton,sphinx.ext.napoleon -d 3 -E -f -F python/${PKG_PROJECT} ${PATH_EXCLUDE_LIST}

content: apidoc
	@cp docs/content/* docs/

docs: content
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
