# ============================================================================
# Makefile for the kerr-dirac-scattering companion repository
# Authors: Dr. Davide Batic and Dr. Denys Dutykh
#          (Mathematics Department, Khalifa University of Science and
#           Technology, Abu Dhabi, UAE)
#
# TARGETS
#   make            benchmarks + figures (everything)
#   make benchmarks regenerate results/ (tables of Section XI of the paper)
#   make figures    compile TikZ diagram sources and regenerate data figures
#                   into ../latex/figures/
#   make test       fast smoke test (coarse tolerances, asserts diagnostics)
#   make clean      remove build litter (never touches results/)
# ============================================================================

PY     := python3
LATEX  := pdflatex -interaction=nonstopmode -halt-on-error
FIGDIR := ../latex/figures

TIKZ_SRC := $(wildcard figsrc/fig0*.tex)
TIKZ_PDF := $(patsubst figsrc/%.tex,$(FIGDIR)/%.pdf,$(TIKZ_SRC))

.PHONY: all benchmarks figures test clean help

all: benchmarks figures

benchmarks:
	$(PY) schwarzschild_dirac_matching.py
	$(PY) schwarzschild_dirac_matching.py --refined
	$(PY) kerr_dirac_matching.py
	$(PY) kerr_dirac_matching.py --refined
	$(PY) convergence_study.py
	$(PY) massgap_reflection.py
	$(PY) flat_limit_check.py
	$(PY) quasibound_search.py

figures: $(TIKZ_PDF)
	$(PY) make_fig06.py
	$(PY) make_fig07.py

$(FIGDIR)/%.pdf: figsrc/%.tex figsrc/figstyle.tex
	@mkdir -p $(FIGDIR)
	cd figsrc && $(LATEX) -output-directory=../$(FIGDIR) $(notdir $<) >/dev/null
	@rm -f $(FIGDIR)/*.aux $(FIGDIR)/*.log

test:
	$(PY) flat_limit_check.py
	$(PY) massgap_reflection.py

clean:
	rm -rf __pycache__ figsrc/*.aux figsrc/*.log $(FIGDIR)/*.aux $(FIGDIR)/*.log

help:
	@echo "Targets: all (default), benchmarks, figures, test, clean, help"
