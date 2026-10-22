# Reset: Can Machine Learning Forecast Formula 1 Through Its Biggest Rule Change in a Decade?

**Question (proposal, 11 Oct 2026).** Can a pre-race model predict which drivers score points (top ten)
through a regulation reset, and should it rely on past results or on the current weekend's pace?
H1 weekend pace beats past results after a reset · H2 past results regain value during the season ·
H3 live forecasts match the simulation. Progress report: `reports/Progress_Report_25oct.pdf`;
code for the reset analysis: `f1duel/reset.py`, notebook `07_reset`.

| Brier score (lower = better) | 2014 reset, rounds 1–5 / 6–19 | 2026 reset, rounds 1–5 / 6–17 |
|---|---|---|
| Past results only | 0.166 / 0.145 | 0.186 / 0.156 |
| Weekend pace only | 0.155 / 0.155 | 0.176 / 0.153 |
| Full model | 0.146 / 0.140 | 0.173 / 0.148 |

H1 and H2 supported in both resets; the new power-unit variables do not improve the full model.
Against the professor's grid baseline (2020–2025): 7.92 vs 7.71 of 10 points scorers (+0.21, 95% CI +0.06 to +0.35).

## F1 Duel (the website built on the same models): can a model beat the starting grid?

**Question.** Before a Grand Prix starts, can we predict who will finish ahead of whom better than the
starting grid does, and will the model be right on the races still to come?

Paul, Luis, Eugenio — Machine Learning (BAM, IE University), Fall 2026.
Live website: **https://paulchabert.pythonanywhere.com** · Code: https://github.com/papchabertieu2025-design/f1-duel

## The answer so far

Tested honestly: each season 2020–2025 is predicted by models trained only on earlier seasons, and
each 2026 race by models re-trained on everything before it.

| Question | Simple rule to beat | Our model | Verdict |
|---|---|---|---|
| Who finishes ahead in a duel? (24,134 pairs, 2020–2025) | "whoever starts ahead": 75.5% | **77.4%** (+2.0 pts, 95% CI +1.1 to +2.8) | beats the grid |
| Same, 2026 new rules (3,685 pairs) | 78.9% | **81.2%** (+2.3 pts, 95% CI +0.5 to +4.2) | beats the grid |
| Who scores points? (131 races) | grid top 10: 7.71 of 10 | **7.87 of 10** (+0.16, 95% CI +0.02 to +0.30) | beats the grid |
| Who wins? | pole sitter wins 55% | favourite wins 55% (it is the pole sitter in 86% of races) | no better pick, but calibrated chances: log-loss 1.37 vs 1.59 for the win rate of each grid slot |
| At Monaco (professor's question) | grid: 9.2 of 10 scorers | 8.0 of 10 | the grid wins at Monaco; on the other grid-dependent tracks the model still wins (duels 77.0% vs 74.8%) |
| Before qualifying (Monday) | grid, once known: 75.5% | duels 73.1% | below the grid: qualifying matters most |

## How it is built: one course method per step

| Part | Method (course session) | Its job | Notebook |
|---|---|---|---|
| I | Data (pandas) | official results 2010–2026, 7,231 starts, pre-race information only | `01_data` |
| II | Linear regression, Ridge, Lasso, CV (S2–5) | one grid place ≈ 0.64 finishing places; the front rows count more → log(grid) | `02_linear_regression` |
| III | PCA (S9) | 7 team statistics → 2 axes: car performance (76%) and reliability (13%) | `03_pca` |
| IV | k-means (S10–11) | 3 car families: top, midfield, back (elbow + silhouette) | `04_kmeans` |
| V | Logistic regression + softmax (S6–8) | P(points), P(A ahead of B), P(win): the main model | `05_logistic_regression` |
| VI | Verdict | model vs reality, 2026 race by race, then live races | `06_verdict` |
| VII | The reset (proposal question) | past results vs weekend pace in 2014 and 2026; power-unit variables | `07_reset` |

Final model (after qualifying): 8 features known on Saturday night: grid, log(grid), qualifying gap,
practice gap, car performance (PC1), reliability (PC2), top-car and back-of-grid flags (k-means).
Before qualifying the site uses the 6 features known on Monday: the 4 car features and the driver's recent form.

## Data and sources

One source: official Formula 1 results via [F1DB](https://github.com/f1db/f1db) (CC BY 4.0),
snapshot in `data/raw/` with its SHA-256 checksum. Checked line by line against formula1.com:
see [VERIFICATION.md](VERIFICATION.md).

## Run it

```bash
pip install -r requirements.txt
make all            # data -> evaluation -> notebooks (with outputs) -> forecast -> app/model.json
cd app && python app.py   # website on http://127.0.0.1:5050
```

## Live forecasts (the real test)

| When | Command |
|---|---|
| Monday before a race | `python -m f1duel.forecast 2026 18` → forecast from history only |
| Saturday after qualifying | copy `data/upcoming/TEMPLATE_2026_r18.csv` to `data/upcoming/2026_r18.csv`, fill grid and best lap times from formula1.com, run the same command |
| Before lights out | `git add forecasts app/model.json && git commit -m "Forecast US GP" && git push` (the timestamp proves it) |
| After the race | download the new F1DB release into `data/raw/`, then `make all` (scoreboard updates) |

Upcoming: United States (25 Oct), Mexico City (1 Nov), São Paulo (8 Nov), Las Vegas (22 Nov), Qatar (29 Nov), Abu Dhabi (6 Dec).

## Deploy on PythonAnywhere

See [app/DEPLOY.md](app/DEPLOY.md). The site needs only Flask (+ `openai` for the AI box), because
the model is exported as plain numbers in `app/model.json`.

## Repository

```
f1duel/      data.py (Part I) · model.py (Parts II-VI) · evaluate.py (VII) · forecast.py (live) · plots.py
notebooks/   01 ... 07, one per part, executed with outputs; build_notebooks.py rebuilds them
app/         Flask website: app.py, templates/index.html, model.json
results/     summary.json, predictions.csv, duels.csv      reports/   LaTeX tables + deliverable plans      figures/   every figure of the notebooks
forecasts/   one file per race, committed before the start
```
