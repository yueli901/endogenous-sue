# Three commands do the work: install, reproduce, rebuild.
PYTHON ?= python

.PHONY: help install data test reproduce corpus tables figures validate validate-strict provenance lint clean

help:
	@echo "install     install the package and its dependencies"
	@echo "data        clone the benchmark networks at the pinned commit"
	@echo "test        run the test suite"
	@echo "reproduce   run a few cells of each exhibit into results/smoke/, to show the"
	@echo "            pipeline runs (the canonical sweeps are launched individually)"
	@echo "tables      rebuild the paper's tables from stored results"
	@echo "figures     rebuild the paper's figures from stored results"
	@echo "corpus      run the exclusion criteria over every network in the collection"
	@echo "validate    check the stored results before anything is read off them"
	@echo "validate-strict  the manuscript gate: missing sinks and unconverged runs fail"
	@echo "provenance  record the environment this machine would reproduce in"

install:
	$(PYTHON) -m pip install -e ".[fast,figures,dev]"

data:
	$(PYTHON) reproduction/tools/fetch_networks.py

test:
	$(PYTHON) tests/run_tests.py

# The small reproduction: a few cells of each sweep, to show the pipeline runs. It writes to
# results/smoke/ and never to results/reproduce/, because its rows are produced under the same settings as
# the reported ones and would therefore mix with them undetected -- the settings check compares rows, and
# these would agree. The corpus and certificate sweeps proper are hours to days and are launched
# individually into results/reproduce/.
reproduce:
	$(PYTHON) reproduction/exhibits/witness.py --out results/smoke/witness.jsonl --no-subsets
	$(PYTHON) reproduction/exhibits/support_comparison.py --networks SiouxFalls,Eastern-Massachusetts \
		--dispersions 1 --out results/smoke/support_comparison.jsonl
	$(PYTHON) reproduction/exhibits/corpus_sweep.py --networks Braess-Example,SiouxFalls --dispersions 1 \
		--out results/smoke/corpus_sweep.jsonl
	$(PYTHON) reproduction/exhibits/certificate_sweep.py --networks SiouxFalls --dispersions 10 \
		--out results/smoke/certificate_sweep.jsonl --no-keep-flows
	@echo
	@echo "Wrote results/smoke/. These are not results: validate and build tables from"
	@echo "results/reproduce/, which the canonical sweeps write."

tables:
	$(PYTHON) reproduction/tools/build_tables.py

figures:
	$(PYTHON) reproduction/tools/build_figures.py

corpus:
	$(PYTHON) reproduction/tools/corpus_verdicts.py --check

validate:
	$(PYTHON) reproduction/tools/validate_results.py

# The manuscript gate. An absent sink and an unconverged record are failures here, not warnings, because
# a table cannot be built from a result set that does not exist or from a run that did not converge.
validate-strict:
	$(PYTHON) reproduction/tools/validate_results.py --strict
	$(PYTHON) reproduction/tools/splice_tables.py --check \
		--tex manuscript/submission/main.tex --tex manuscript/submission/supplement.tex

provenance:
	$(PYTHON) reproduction/tools/record_provenance.py

lint:
	ruff check src reproduction tests

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf build dist src/*.egg-info
