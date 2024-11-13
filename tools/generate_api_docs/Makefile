# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXAPIDOC  ?= sphinx-apidoc
SOURCEDIR     = source
BUILDDIR      = build
TEMPLATEDIR   = source/_templates
PACKAGEDIR    = ../../src/tbp

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo " "
	@echo "\033[1mtbp.monty\033[0m specific targets:"
	@echo " "
	@echo "  \033[0;34mapidoc\033[0m      to create api doc files in the source directory"
	@echo "  \033[0;34mall\033[0m         alias to 'clean apidoc html monty_plan'"

.PHONY: help Makefile

all: clean apidoc html monty_plan

apidoc:
	rm -f ${SOURCEDIR}/modules.rst ${SOURCEDIR}/tbp.* 
	@${SPHINXAPIDOC} -o "${SOURCEDIR}" -t "${TEMPLATEDIR}" -f -M --implicit-namespaces "${PACKAGEDIR}"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
.DEFAULT:
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
