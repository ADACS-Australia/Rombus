# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = docs
BUILDDIR      = build

# Set some variables needed by the documentation
# TODO: Ideally, PKG_PROJECT would be extracted from the TOML file correctly.  This will be easier with Python 3.11, which will include tomllib automatically
# TODO: Ideally, PKG_VERSION would be extracted from the TOML file in the same way, and version would be set there from the git tag.
PKG_PROJECT := rombus
PKG_AUTHOR  := `poetry run python3 -c 'from importlib.metadata import metadata; print(metadata("${PKG_PROJECT}")["author"])'`
PKG_VERSION := `git describe --tags --abbrev=0`
PKG_RELEASE := ${PKG_VERSION}

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

clean:
	@rm -rf build docs/_build
	@rm docs/conf.py docs/make.bat docs/Makefile
	@find docs/ -type f -name '*.rst' -not -name 'overview.rst' -delete
apidoc:
	sphinx-apidoc -o docs --doc-project ${PKG_PROJECT} --doc-author "${PKG_AUTHOR}" --doc-version ${PKG_VERSION} --doc-release ${PKG_RELEASE} -t docs/_templates -T --extensions sphinx.ext.doctest,sphinx.ext.mathjax,sphinx.ext.autosectionlabel,myst_parser,sphinx.ext.todo -d 3 -E -f -F python/${PKG_PROJECT}

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile apidoc
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
