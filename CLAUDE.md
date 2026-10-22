# CLAUDE.md — context for Claude Code

Read this first. It summarises everything decided in the earlier Claude conversation that built this project.

**Then read the documents in `docs/`** (index: `docs/README.md`):
- `docs/course/project_brief_ML_CommSkills_2026.txt`: the official project rules (deliverables, page limits, report structure, grading, AI policy).
- `docs/course/syllabus_Machine_Learning.txt` and `docs/course/syllabus_Technology_with_Impact.txt`: what Paul studied.
- `docs/course/GenAI_OpenAI_tutorial_Gadi.pdf`: the OpenAI API tutorial used for the website's AI box.
- `docs/teacher_feedback.md`: the professor's feedback, word for word, and where each point is answered.
- `docs/student_preferences_and_history.md`: what Paul expects and how the project evolved.
- `docs/our_submissions/`: the proposal submitted on 11 Oct and the progress report draft.

## Who and what
- Student: Paul (team: Paul, Luis, Eugenio), Bachelor in Applied Mathematics, IE University Madrid. Speaks French, but wants everything (chat, docs, deliverables) in English.
- Courses: **Machine Learning** (Prof. Idris Kouadri; covered up to session 15: linear regression, Ridge/Lasso/CV, logistic regression + softmax + ROC/AUC, PCA, k-means; KNN deliberately NOT used, at Paul's request) and **Technology with Impact** (Prof. Manoel Gadi; covered up to session 8: Flask model deployment as API, HTML/CSS/JS, v0.dev, OpenAI API, PythonAnywhere).
- Goal: the best possible grade, a clear project, "wow effect", every method from the course used with one clear role.

## The project: F1 Duel
**Question:** before a Grand Prix starts, can we predict who finishes ahead of whom better than the starting grid, and will the model be right on the real races still to come?

| Part | Method | Role | Notebook |
|---|---|---|---|
| I | data (pandas) | official results 2014–2026, 5,435 starts, 269 races, pre-race info only | 01_data |
| II | linear regression, Ridge, Lasso, CV | 1 grid place ≈ 0.64 finishing places; front rows count more → log(grid) | 02 |
| III | PCA | 7 rolling team stats → PC1 car performance (76%), PC2 reliability (13%) | 03 |
| IV | k-means (k=3, elbow + silhouette) | car families: top / midfield / back (top-10 rate 84/55/20%) | 04 |
| V | logistic regression + softmax | P(top 10), P(A ahead of B) (logistic on differences, no intercept), P(win) (softmax over drivers, hand-written gradient) | 05 |
| VI | verdict | 2026 race by race, then live races | 06 |
| — | two models (tracks.py, forecast.py) | Paul's decision (Option A, 22 Oct): forecast uses the model of the next race's track type (hard-to-overtake list: Monaco, Zandvoort, Hungaroring, Marina Bay, Madring, vs normal). Honest numbers: 7.86 vs 7.87 overall, 7.85 vs 7.95 on hard tracks; single model kept alongside, both compared live (website table) | 06 |
| VII | reset (proposal question) | past results vs weekend pace, 2014 and 2026 resets, power-unit variables | 07 |

Final model after qualifying: 8 features (grid, log grid, quali_gap, practice_gap, car_perf, car_rel, tier_top, tier_back). Before qualifying: 6 features (4 car features + driver form).

## Results (results/summary.json — regenerate with `make all`)
- Duels: 77.4% vs grid rule 75.5% (24,134 pairs, 2020–2025, +2.0 pts, 95% CI +1.1 to +2.8); 2026: 81.2% vs 78.9% (+2.3, CI +0.5 to +4.2).
- Points scorers: 7.87 vs 7.71 of 10 per race (+0.16, CI +0.02 to +0.30); 2026 gain not significant (17 races).
- Winner: favourite = pole sitter in 86% of races, wins 55% like pole; win probabilities calibrated (log-loss 1.37 vs 1.59 for grid-slot win rate).
- Before qualifying the model is below the grid (duels 73.1%).
- Monaco (teacher's point): grid wins there (9.2 vs 8.0 of 10 scorers, 5 races); on the most grid-dependent third of tracks (by 2014–2019) the model still wins (duels 77.0% vs 74.8%). summary.json → "tracks".
- Limitation: top-10 probabilities slightly high for outsiders (19% predicted vs 12% observed).
- Evaluation: season-ahead 2020–2025 (train on earlier seasons only); 2026 re-trained before every race. An independent review found no leakage and verified all numbers.

## Sources
Single source: official F1 results via F1DB (github.com/f1db/f1db), snapshot data/raw/f1db-csv.zip (2026-10-21, after Singapore R17), SHA-256 checked. Cross-checked against formula1.com: see VERIFICATION.md. The user insists every statistic is sourced and verified.

## Professor's feedback on the earlier proposal (grade 7, "unclear paper") — must stay answered
"Specify which features are used as predictors… Are you using grid position as a predictor? If so, show a baseline: top 10 on the grid = top 10 finishers, measure its performance, compare with your model. Monaco is very grid dependent (many GPs might be similar), and some teams simply have better cars."
→ Every result is compared with the grid rule; car families (PCA + k-means) answer "better cars".

## Website (app/)
Flask: `/` page, `/api/race`, `/api/duel`, `/api/ask` (OpenAI, grounded only on model.json numbers, key via OPENAI_API_KEY env var, model via OPENAI_MODEL, default gpt-4o-mini). Model exported as plain numbers in app/model.json by `python -m f1duel.forecast 2026 <round>`. Deployment: app/DEPLOY.md (PythonAnywhere). Design: track grey, ink, kerb red/white stripes, team colours, Titillium Web; light + dark themes.
A static demo was also published as a claude.ai artifact (private).

## Status (22 Oct 2026) and next steps
1. Deploy on PythonAnywhere (user's account, e.g. paulchabert.pythonanywhere.com) and add the OpenAI key there (never in Git).
2. Saturday 24 Oct after qualifying (US GP, Austin, round 18): copy data/upcoming/TEMPLATE_2026_r18.csv → 2026_r18.csv, fill grid + best qualifying and practice lap times (seconds) from formula1.com, run `python -m f1duel.forecast 2026 18`, commit forecasts/ + app/model.json BEFORE the race (timestamp = proof).
3. After each race: download the new F1DB release (github.com/f1db/f1db/releases/latest, f1db-csv.zip) into data/raw/, update the sha256 file, `make all`.
4. Next races: Mexico City (r19, 1 Nov), São Paulo (r20, 8 Nov), Las Vegas (r21, 22 Nov), Qatar (r22, 29 Nov), Abu Dhabi (r23, 6 Dec).
5. Create the public GitHub repo and commit in several steps, each member under their own name (brief requires credible history).
6. Progress report due Sun 25 Oct (2 pages): reports/1_progress_report_25oct/main.tex (structure + generated tables + yellow fact boxes; the team writes the prose). Master checklist: TODO.md. Guide: GUIDE.md. Plans: reports/2_…, 3_…, 4_presentation/.
7. Upcoming deliverables: sentence outline (15 Nov, 2 pages), final report (25 Nov, 10–12 pages, official BAM LaTeX template, two columns, MLA, GitHub link), presentation (ML sessions 27–28, 10 min + 10 min Q&A).

## Rules to respect
- Course brief §5.1: AI allowed for code, but **all AI use must be declared in every submission**, and **AI may not write the submitted prose**. Help structure and explain; the team writes report text.
- Never use information from a race to predict that race. Keep the grid rule as the baseline everywhere.
- Keep explanations simple. Paul asked (22 Oct) for EVERYTHING in English, including chat and docs.

## Commands
`pip install -r requirements.txt` · `make all` (data → evaluate → notebooks → tables → forecast; uses .venv/ if present) · `make site` (local website on http://127.0.0.1:5050) · `python notebooks/build_notebooks.py` (rebuild notebooks).
Older, superseded work (proposal and progress report on a different framing) lives in a separate folder f1-points, not needed here.
