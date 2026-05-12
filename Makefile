PYTHON ?= python
QUARTO_PYTHON ?= $(shell $(PYTHON) -c "import sys; print(sys.executable)")

.PHONY: docs docs-preview

docs:
	cd docs && quartodoc build && QUARTO_PYTHON=$(QUARTO_PYTHON) quarto render

docs-preview:
	cd docs && quartodoc build && QUARTO_PYTHON=$(QUARTO_PYTHON) quarto preview
