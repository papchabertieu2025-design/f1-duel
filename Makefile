PY ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

all: data evaluate notebooks tables forecast

data:
	cd data/raw && unzip -o -q f1db-csv.zip -d f1db && sha256sum -c f1db-csv.zip.sha256
	$(PY) -m f1duel.data

evaluate: data
	$(PY) -m f1duel.evaluate

notebooks: evaluate
	$(PY) notebooks/build_notebooks.py

PR = reports/1_progress_report_25oct
# LaTeX tables for the reports (copied into the progress report folder for Overleaf), from results/summary.json
tables:
	$(PY) -m f1duel.report_tables
	mkdir -p $(PR)/tables $(PR)/figures
	cp reports/tables/*.tex $(PR)/tables/
	cp figures/02_win_by_grid.png figures/05_duel_weights.png $(PR)/figures/

# next race: make forecast YEAR=2026 ROUND=18
YEAR ?= 2026
ROUND ?= 18
forecast:
	$(PY) -m f1duel.forecast $(YEAR) $(ROUND)

site:
	cd app && $(PY) app.py

.PHONY: all data evaluate notebooks tables forecast site
