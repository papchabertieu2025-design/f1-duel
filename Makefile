PY ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

all: data evaluate reset tracks notebooks tables forecast

data:
	cd data/raw && unzip -o -q f1db-csv.zip -d f1db && sha256sum -c f1db-csv.zip.sha256
	$(PY) -m f1duel.data

evaluate: data
	$(PY) -m f1duel.evaluate

# the proposal's question: the 2014 and 2026 resets, history vs weekend pace, power-unit variables
reset: data
	$(PY) -m f1duel.reset

# one model or two? normal tracks vs grid-dependent (Monaco-like) tracks
tracks: data
	$(PY) -m f1duel.tracks

notebooks: evaluate
	$(PY) notebooks/build_notebooks.py

PR = reports/1_progress_report_25oct
# LaTeX tables for the reports (copied into the progress report folder for Overleaf), from results/summary.json
tables:
	$(PY) -m f1duel.report_tables
	mkdir -p $(PR)/tables
	cp reports/tables/*.tex $(PR)/tables/

# next race: make forecast YEAR=2026 ROUND=18
YEAR ?= 2026
ROUND ?= 18
forecast:
	$(PY) -m f1duel.forecast $(YEAR) $(ROUND)

site:
	cd app && $(PY) app.py

.PHONY: all data evaluate reset tracks notebooks tables forecast site
