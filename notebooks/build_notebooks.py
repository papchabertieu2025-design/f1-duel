"""Writes the 6 project notebooks (one per part) and executes them so every output is saved.
Usage (from the project root):  python notebooks/build_notebooks.py
"""
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

HERE = Path(__file__).parent
SETUP = """import sys, warnings; sys.path.insert(0, '..'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from f1duel.data import build, TEAM_STATS
from f1duel.plots import style, title, save, BLUE, ORANGE, AQUA, GREY, INK, INK2, MUTED, TIER_COLORS
style(); pd.set_option('display.width', 140); pd.set_option('display.max_columns', 20)
d = build()"""


def md(t):
    return nbf.v4.new_markdown_cell(t.strip())


def code(t):
    return nbf.v4.new_code_cell(t.strip())


NB = {}

# ----------------------------------------------------------------------------- Part I
NB["01_data"] = [
    md("""
# Part I — The data
**Question of the project:** *before a Grand Prix starts, can we predict who will finish ahead of whom better than the starting grid does, and will our model be right on the real races still to come?*

**This notebook:** where the data comes from, what one row is, and the one rule that keeps the project honest.

| | |
|---|---|
| **Source** | Official Formula 1 race results, as stored in the open F1DB database (github.com/f1db/f1db, CC BY 4.0). One single source for every number. |
| **Snapshot** | `data/raw/f1db-csv.zip`, downloaded 21 Oct 2026 (after the Singapore GP), SHA-256 checksum in `data/raw/` |
| **Period** | 2010–2026: the same points system throughout (top 10 score), covering the two engine resets of 2014 and 2026 |
| **One row** | one driver who started one Grand Prix |
| **Golden rule** | a feature for race *r* only uses races **before** *r* (grid, qualifying and practice are allowed: they are known on Saturday, before the race) |
"""),
    code(SETUP + """
print(f"{len(d):,} driver starts | {d.raceId.nunique()} races | {d.driverId.nunique()} drivers | {d.year.min()}–{d.year.max()}")
d[['year','gp_name','driver_name','team_name','grid','quali_gap','practice_gap','finish_pos','top10']].tail(5)"""),
    md("""
### What we predict, and what we use

| Target (what happened) | Meaning |
|---|---|
| `finish_pos` | finishing position (empty = did not finish) |
| `top10` | 1 if the driver finished in the top 10 = scored points |
| `win` | 1 if the driver won |

| Feature | Known when | Meaning |
|---|---|---|
| `grid` | Saturday | starting position after penalties |
| `quali_gap` | Saturday | gap to pole position in qualifying (%) |
| `practice_gap` | Friday/Saturday | gap to the fastest practice lap (%) |
| 7 team statistics | Monday | team points, top-10 rate, podium rate, average finish, average grid, qualifying gap (last 5 races) and mechanical failures (last 10) |
| `drv_avg_finish`, `drv_top10` | Monday | driver's average finish and top-10 rate over his last 5 starts |
"""),
    code("""
d[TEAM_STATS + ['drv_avg_finish','drv_top10']].describe().round(2).T[['mean','std','min','max']]"""),
    md("### The starting point: the grid already says a lot"),
    code("""
c = d[d.finish_pos.notna()]
by_grid = c.groupby(c.grid.clip(upper=20)).finish_pos.mean()
top10_rate = d.groupby(d.grid.clip(upper=20)).top10.mean()
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot(by_grid.index, by_grid.values, 'o-', color=BLUE); ax[0].plot([1, 20], [1, 20], ls=':', color=MUTED)
ax[0].set_xlabel('Starting position'); ax[0].set_ylabel('Average finishing position'); title(ax[0], '(a) Start vs finish (classified cars)')
ax[1].bar(top10_rate.index, top10_rate.values, color=[BLUE if g <= 10 else GREY for g in top10_rate.index])
ax[1].set_xlabel('Starting position'); ax[1].set_ylabel('Share who scored points'); ax[1].yaxis.set_major_formatter(lambda v, _: f'{v:.0%}')
title(ax[1], '(b) Chance of scoring by starting position')
fig.tight_layout(); save(fig, '01_grid'); plt.show()
print(f"Pole sitter scores {top10_rate[1]:.0%} of the time; 10th {top10_rate[10]:.0%}; 11th {top10_rate[11]:.0%}; 20th {top10_rate[20]:.0%}")"""),
    md("""
**Reading:** the grid is a strong guide, but not a perfect one. Starting 10th or 11th is close to a coin flip, and even the pole sitter misses the points sometimes. This is the **baseline every model must beat** (the professor's request): *"the 10 drivers who start in the top 10 are the 10 who score"*.
"""),
    code("""
races = d.groupby('raceId')
hits = races.apply(lambda g: int(g.top10[g.grid.rank(method='first') <= 10].sum()), include_groups=False)
print(f"Grid rule: on average {hits.mean():.2f} of the 10 drivers starting in the top 10 score points (all {len(hits)} races, 2010-2026).")
print("(The test seasons used later, 2020-2025, give 7.71: that is the number the models are compared with.)")"""),
]

# ----------------------------------------------------------------------------- Part II
NB["02_linear_regression"] = [
    md("""
# Part II — Linear regression: how many places is a grid slot worth?
**Method (sessions 2–5):** linear regression, least squares, Ridge and Lasso, cross-validation.
**Why first:** it is the simplest model and every coefficient is a sentence: *"one place further back on the grid costs X places at the finish"*.
**Target:** finishing position of classified drivers. **Train:** 2010–2024. **Test:** 2025 and 2026 (never seen).
"""),
    code(SETUP + """
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
c = d[d.finish_pos.notna()].copy(); c['log_grid'] = np.log(c.grid)
train, tests = c[c.year <= 2024], {y: c[c.year == y] for y in (2025, 2026)}"""),
    md("### Step 1 — one variable: the grid"),
    code("""
m = LinearRegression().fit(train[['grid']], train.finish_pos)
print(f"finish = {m.intercept_:.2f} + {m.coef_[0]:.2f} x grid")
print(f"=> one place further back on the grid costs {m.coef_[0]:.2f} places at the finish, on average.")"""),
    md("### Step 2 — add what we know about the car and the driver, and regularise (Ridge / Lasso, alpha chosen by 5-fold cross-validation grouped by season)"),
    code("""
FEATS = ['grid', 'log_grid', 'quali_gap', 'practice_gap', 'team_points', 'team_avg_finish', 'team_mech_dnf', 'drv_avg_finish']
cv = list(GroupKFold(5).split(train, groups=train.year))
models = {
    'Baseline: finish = grid': None,
    'Linear, grid only': make_pipeline(StandardScaler(), LinearRegression()),
    'Linear, extended (8 variables)': make_pipeline(StandardScaler(), LinearRegression()),
    'Ridge, extended': make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25), cv=cv)),
    'Lasso, extended': make_pipeline(StandardScaler(), LassoCV(alphas=np.logspace(-3, 0, 25), cv=cv, random_state=0)),
}
rows = []
for name, m in models.items():
    f = ['grid'] if name == 'Linear, grid only' else FEATS
    if m is not None: m.fit(train[f], train.finish_pos)
    for y, t in tests.items():
        p = t.grid.values if m is None else m.predict(t[f])
        rows.append({'Model': name, 'Test': y, 'MAE': mean_absolute_error(t.finish_pos, p),
                     'RMSE': mean_squared_error(t.finish_pos, p) ** 0.5, 'R2': r2_score(t.finish_pos, p)})
res = pd.DataFrame(rows).pivot(index='Model', columns='Test', values=['MAE', 'R2']).round(3)
res.loc[list(models)]"""),
    code("""
ridge = models['Ridge, extended']; lasso = models['Lasso, extended']
coef = pd.DataFrame({'Ridge': ridge[-1].coef_, 'Lasso': lasso[-1].coef_}, index=FEATS).round(2)
print('Chosen alpha  Ridge:', round(ridge[-1].alpha_, 3), '| Lasso:', round(lasso[-1].alpha_, 4))
coef.sort_values('Ridge', key=abs, ascending=False)"""),
    md("### Step 3 — the grid effect is not a straight line, especially at the front"),
    code("""
win_rate = d[d.year <= 2024].groupby(d.grid.clip(upper=20)).win.mean()
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.bar(win_rate.index[:10], win_rate.values[:10], color=BLUE)
for g, v in win_rate.head(4).items(): ax.text(g, v + 0.01, f"{v:.0%}", ha='center', color=INK)
ax.set_xlabel('Starting position'); ax.set_ylabel('Share of races won'); ax.yaxis.set_major_formatter(lambda v, _: f'{v:.0%}')
title(ax, 'Chance of winning halves with each place at the front'); save(fig, '02_win_by_grid'); plt.show()
for f in (['grid'], ['grid', 'log_grid']):
    print(f"R² on train with {f}: {LinearRegression().fit(train[f], train.finish_pos).score(train[f], train.finish_pos):.3f}")
    """),
    md("""
### What we learn (and carry to the next parts)
1. **One grid place is worth about 0.65 finishing places.** The grid alone explains more than half of the finishing order (R² ≈ 0.55 on the test seasons).
2. **Adding car and qualifying information lowers the error** below the "finish = grid" baseline on both test seasons.
3. **Ridge and Lasso keep almost the same coefficients:** with 8 features and thousands of rows, there is little overfitting to fix.
4. **The grid effect bends:** pole wins 53% of races, 2nd 25%, 3rd 10%. Each place at the front halves the chance of winning, which no straight line can follow. Adding `log(grid)` captures this curve in a linear regression (R² rises a little). In the logistic regression of Part IV, the sigmoid already bends, and `log(grid)` adds nothing (tested: cross-entropy 0.481 with grid alone, 0.481 with both), so the final model keeps **grid only**.
5. **The car matters:** team strength features are among the largest coefficients. Part III turns them into car families.
"""),
]

# ----------------------------------------------------------------------------- Part III
NB["03_kmeans"] = [
    md("""
# Part III — k-means: the three families of cars
**Method (sessions 10–11):** k-means clustering, elbow plot and silhouette score.
**Why:** the professor noted that *"some teams simply have better cars"*. We describe each car with two simple team statistics:
- **car performance** = the team's points per race over its last 5 races,
- **reliability** = minus the team's mechanical-failure rate over its last 10 races,

both standardised (mean 0, standard deviation 1). On this map, cars form groups; k-means finds them automatically, and the group becomes a feature: **top car / midfield car / unreliable car**.
"""),
    code(SETUP + """
from f1duel.model import CarProfile
prof = CarProfile().fit(d[d.year <= 2024])
dd = prof.transform(d)
season = dd[dd.year == 2026].groupby('team_name')[['car_perf', 'car_rel']].mean()
fig, ax = plt.subplots(figsize=(7, 4.4))
ax.scatter(season.car_perf, season.car_rel, s=60, color=BLUE)
for t, r in season.iterrows(): ax.annotate(t, (r.car_perf, r.car_rel), xytext=(5, 3), textcoords='offset points', fontsize=10)
ax.axhline(0, color=MUTED, lw=0.6); ax.axvline(0, color=MUTED, lw=0.6)
ax.set_xlabel('car performance: team points per race (standardised, → faster)'); ax.set_ylabel('reliability (standardised, ↑ fewer failures)')
title(ax, 'The 2026 teams (season average)'); save(fig, '03_map_2026'); plt.show()"""),
    code("""
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
train = d[d.year <= 2024]
prof = CarProfile().fit(train)
Z = prof._z(train.drop_duplicates(['raceId', 'team']))
ks = range(2, 9); inertia, sil = [], []
for k in ks:
    km = KMeans(k, n_init=20, random_state=2026).fit(Z); inertia.append(km.inertia_); sil.append(silhouette_score(Z, km.labels_))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
ax[0].plot(ks, inertia, 'o-', color=BLUE); ax[0].set_xlabel('k'); ax[0].set_ylabel('Inertia'); title(ax[0], '(a) Elbow plot')
ax[1].plot(ks, sil, 'o-', color=BLUE); ax[1].set_xlabel('k'); ax[1].set_ylabel('Silhouette'); title(ax[1], '(b) Silhouette score')
for a in ax: a.axvline(3, color=ORANGE, ls='--', lw=1)
fig.tight_layout(); save(fig, '03_choose_k'); plt.show()
pd.DataFrame({'k': list(ks), 'inertia': np.round(inertia), 'silhouette': np.round(sil, 3)}).set_index('k').T"""),
    md("""
**Choice: k = 3.** The elbow bends at 3 and the silhouette score is highest at k = 3: both criteria agree.
"""),
    code("""
dd = prof.transform(d)
summary = dd.groupby('tier').agg(starts=('top10', 'size'), scored=('top10', 'mean'), podium=('podium', 'mean'),
                                 avg_grid=('grid', 'mean'), car_perf=('car_perf', 'mean'))
summary.index = ['Top car', 'Midfield car', 'Unreliable car']; summary.round(2)"""),
    code("""
pts = dd[dd.year <= 2024].drop_duplicates(['raceId', 'team'])
fig, ax = plt.subplots(figsize=(7, 4.2))
for t, name in enumerate(['Top car', 'Midfield car', 'Unreliable car']):
    p = pts[pts.tier == t]; ax.scatter(p.car_perf, p.car_rel, s=8, alpha=0.5, color=TIER_COLORS[t], label=name)
ax.set_xlabel('car performance (standardised team points)'); ax.set_ylabel('reliability (standardised)'); ax.legend(markerscale=3)
title(ax, 'k-means on the two team statistics (each dot = one team at one race)'); save(fig, '03_clusters'); plt.show()"""),
    md("""
**Reading:** the three families are not simply fast / medium / slow. A **top car** (high points, few failures) scores in about 83% of its starts. The two others score about as many points per race, but differ in reliability: **midfield cars** break down in about 5% of races and score in 37% of their starts, **unreliable cars** break down in about 22% and score in 28%. In 2026, Aston Martin (new Honda engine) and Cadillac (new team) are unreliable cars. In Part IV, these groups help the model see a fast car stuck at the back of the grid, or a car likely to break down.
"""),
]

# ----------------------------------------------------------------------------- Part IV
NB["04_logistic_regression"] = [
    md("""
# Part IV — Logistic regression: the main model
**Method (sessions 6–8):** logistic regression (sigmoid, cross-entropy, ROC/AUC), and its multiclass version, softmax.
**Why:** our questions are probabilities. One family of models answers all three:

| Question | Model | Output |
|---|---|---|
| Will this driver score points? | binary logistic regression | P(top 10) |
| Will A finish ahead of B? | logistic regression on differences x_A − x_B | P(A ahead of B) → **the duel of the website** |
| Who wins? | softmax over the 20 drivers of the race | P(win) for each driver |

**Features (6, known on Saturday night):** grid, qualifying gap, practice gap, car performance (team points), top-car and unreliable-car flags (k-means).
**Honest test:** each season 2020–2025 is predicted by a model trained only on earlier seasons; 2026 (new rules) is re-trained before every race. All results come from `python -m f1duel.evaluate`.
"""),
    code(SETUP + """
import json
from f1duel.model import CarProfile, DuelModel, WinnerModel, fit_top10, AFTER_QUALI, BEFORE_WEEKEND, scorers_named
S = json.load(open('../results/summary.json')); P = pd.read_csv('../results/predictions.csv')
print('Features:', AFTER_QUALI)"""),
    md("### Question 1 — Who scores points? Against the professor's baseline (the grid top 10)"),
    code("""
rows = []
for test in ['2020-2025', '2026']:
    s = S[test]
    for m in ['Grid rule', 'Logistic regression', 'Logistic, before the weekend']:
        g = s['gain_vs_grid'].get(m)
        rows.append({'Test': f"{test} ({s['races']} races)", 'Method': m, 'Scorers named (of 10)': round(s['scorers_named'][m], 2),
                     'Gain vs grid [95% CI]': '—' if g is None else f"{g[0]:+.2f} [{g[1]:+.2f}, {g[2]:+.2f}]",
                     'ROC-AUC': round(s['auc'].get(m, float('nan')), 3)})
pd.DataFrame(rows).set_index(['Test', 'Method'])"""),
    md("""
**Reading:** with qualifying known, logistic regression names **0.16 more real scorers per race** than the grid over 2020–2025, and the 95% interval stays above zero. Before qualifying (Monday, no grid) it is clearly worse than the grid: qualifying is the single most important piece of information. In 2026 the gain is smaller and not yet certain with 17 races.
"""),
    code("""
from sklearn.metrics import roc_curve
fig, ax = plt.subplots(figsize=(5, 4))
for col, lab, c in [('p_top10_logit', 'Logistic, after qualifying', BLUE), ('p_top10_logit_before', 'Logistic, before the weekend', ORANGE)]:
    q = P[P.test == '2020-2025']; fpr, tpr, _ = roc_curve(q.top10, q[col]); ax.plot(fpr, tpr, color=c, label=lab)
q = P[P.test == '2020-2025']; fpr, tpr, _ = roc_curve(q.top10, -q.grid); ax.plot(fpr, tpr, color=GREY, ls='--', label='Grid order')
ax.plot([0, 1], [0, 1], ':', color=MUTED); ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate'); ax.legend(loc='lower right')
title(ax, 'ROC curves, 2020–2025'); save(fig, '04_roc'); plt.show()"""),
    md("### Question 2 — The duel: will A finish ahead of B?"),
    code("""
rows = []
for test in ['2020-2025', '2026']:
    for m, lab in [('after', 'After qualifying'), ('before', 'Before the weekend')]:
        x = S[test]['duel'][m]
        rows.append({'Test': test, 'Model': lab, 'Pairs': x['pairs'], 'Model accuracy': f"{x['accuracy']:.1%}",
                     'Rule "who starts ahead finishes ahead"': f"{x['grid_rule_accuracy']:.1%}",
                     'Gain [95% CI, bootstrap over races]': f"{(x['accuracy']-x['grid_rule_accuracy'])*100:+.1f} pts [{x['gain_ci'][0]*100:+.1f}, {x['gain_ci'][1]*100:+.1f}]"})
pd.DataFrame(rows).set_index(['Test', 'Model'])"""),
    md("""
**Reading:** after qualifying, the duel model calls **77.4%** of head-to-heads right (2020–2025, 24,134 pairs) against 75.5% for "whoever starts ahead finishes ahead": +1.9 points, and the 95% interval (resampling whole races) stays above zero. In 2026: **81.2% vs 78.9%**, also significant. Before qualifying the model is *below* the grid rule. This is the model inside the website's duel.
"""),
    md("### What the model learned (trained on 2010 → Singapore 2026)"),
    code("""
live = d[d.upcoming == 0]; prof = CarProfile().fit(live); tr = prof.transform(live)
duel = DuelModel().fit(tr, AFTER_QUALI)
w = pd.Series(duel.lr.coef_[0], index=AFTER_QUALI).sort_values()
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.barh(w.index, w.values, color=[ORANGE if v < 0 else BLUE for v in w.values]); ax.axvline(0, color=MUTED, lw=0.6)
ax.set_xlabel('Weight per standard deviation (positive = helps finish ahead)'); title(ax, 'Duel model: what makes a driver finish ahead')
save(fig, '04_duel_weights'); plt.show(); w.round(3)"""),
    md("""
**How to read a weight:** P(A ahead of B) = sigmoid(w · (x_A − x_B)). A positive weight means more of that feature helps. Grid is negative (starting further back hurts); car performance is positive.
"""),
    md("### Question 3 — Who wins? (softmax over the drivers of a race)"),
    code("""
pd.DataFrame({t: {'Races': S[t]['winner']['races'], "Model's favourite won": f"{S[t]['winner']['favourite_won']:.0%}",
                  'Pole sitter won': f"{S[t]['winner']['pole_won']:.0%}",
                  "Favourite = pole sitter": f"{S[t]['winner']['favourite_is_pole']:.0%}",
                  'Log-loss, our model': round(S[t]['winner']['logloss'], 3),
                  'Log-loss, win rate of each grid slot': round(S[t]['winner']['logloss_gridrate_baseline'], 3)}
              for t in ['2020-2025', '2026']})"""),
    md("### Are the probabilities honest? Calibration, 2020–2025"),
    code("""
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for a, key, name in [(ax[0], 'p_win', 'P(win)'), (ax[1], 'p_top10_logit', 'P(top 10)')]:
    c = pd.DataFrame(S['2020-2025']['calibration'][key])
    a.plot([0, 1], [0, 1], ':', color=MUTED); a.plot(c.predicted, c.observed, 'o-', color=BLUE)
    for _, r in c.iterrows(): a.annotate(f"n={int(r.n)}", (r.predicted, r.observed), xytext=(4, -12), textcoords='offset points', fontsize=9, color=INK2)
    a.set_xlabel(f'Predicted {name}'); a.set_ylabel('Observed frequency'); title(a, f'Calibration of {name}')
fig.tight_layout(); save(fig, '04_calibration'); plt.show()
pd.DataFrame(S['2020-2025']['calibration']['p_win']).round(3)"""),
    md("""
**Reading:** the model's favourite *is* the pole sitter in 92% of races, so it wins about as often (54% vs 55% for pole): we do not beat the obvious answer on *who* wins. What the model adds is **how likely** each driver is to win. Its probabilities beat a fair baseline (the historical win rate of each grid slot: log-loss 1.38 vs 1.56), and they are **calibrated**: when it says 24%, the driver wins 25% of the time; 66% → 68%. The top-10 probabilities are too high for outsiders (20% predicted vs 11% observed), a limitation we report.

The softmax is fitted by maximum likelihood; its gradient, used by the optimiser, is
$$\\frac{\\partial(-\\log p_k)}{\\partial w} = \\sum_j p_j\\,x_j - x_k,$$
the expected features minus the winner's features (see `WinnerModel` in `f1duel/model.py`).
"""),
]

# ----------------------------------------------------------------------------- Part V
NB["05_verdict"] = [
    md("""
# Part V — The verdict: model against reality
**Question:** *could we have predicted the 2026 season, race by race?* Before each 2026 race, every model is re-trained on everything that happened before it (car profile and k-means included), exactly like a live forecast. Then we compare with what really happened.
"""),
    code(SETUP + """
import json
S = json.load(open('../results/summary.json')); P = pd.read_csv('../results/predictions.csv')
from f1duel.model import scorers_named, pick10
Q = P[P.test == '2026']
rows = []
for (r, gp), g in Q.groupby(['round', 'gp_name']):
    fav = g.loc[g.p_win.idxmax()]; win = g[g.win == 1]
    rows.append({'Round': r, 'Grand Prix': gp,
                 'Grid rule': int(g.top10[pick10(g, 'grid', True) == 1].sum()),
                 'Model': int(g.top10[pick10(g, 'p_top10_logit') == 1].sum()),
                 'Model favourite': f"{fav.driver_name} ({fav.p_win:.0%})",
                 'Real winner': win.driver_name.iloc[0] if len(win) else '—'})
T = pd.DataFrame(rows).set_index('Round'); T"""),
    code("""
fig, ax = plt.subplots(figsize=(9, 3.4))
ax.plot(T.index, T['Grid rule'], 'o-', color=GREY, label=f"Grid rule (avg {T['Grid rule'].mean():.2f})")
ax.plot(T.index, T['Model'], 'o-', color=BLUE, label=f"Our model (avg {T['Model'].mean():.2f})")
ax.set_xticks(T.index); ax.set_xlabel('2026 round'); ax.set_ylabel('Real scorers among the 10 picked'); ax.set_ylim(4, 10.5); ax.legend(loc='lower left')
title(ax, '2026, race by race: how many of the 10 points scorers were predicted?'); save(fig, '05_2026'); plt.show()
print(f"Model better in {(T.Model > T['Grid rule']).sum()} races, equal in {(T.Model == T['Grid rule']).sum()}, worse in {(T.Model < T['Grid rule']).sum()}.")"""),
    md("""
### Monaco and other grid-dependent tracks (the professor's question)
*"Monaco is very grid dependent, and many GPs might be similar."* We group the 2020–2025 test races by track. A track's **grid dependence** is the share of its top-10 finishers who started in the top 10, measured on **2010–2019 only**, so the groups are fixed before looking at the test races.
"""),
    code("""
TR = S['tracks']
G = pd.DataFrame({k: v for k, v in TR.items() if isinstance(v, dict)}).T
G['gain_scorers'] = G.scorers_model - G.scorers_grid; G['gain_duels_pts'] = 100 * (G.duel_model - G.duel_grid)
display(G.round(3))
print('Most grid-dependent tracks (2010-2019):', ', '.join(TR['grid_dependent_tracks']))
print(f"Monaco 2010-2019: {TR['monaco_grid_dependence_2010_2019']:.0%} of top-10 finishers started top 10 (median track: {TR['grid_dependence_median_2010_2019']:.0%})")
fig, ax = plt.subplots(figsize=(8, 2.8)); y = np.arange(len(G))[::-1]
ax.barh(y + 0.2, G.duel_grid * 100, 0.4, color=GREY, label='Grid rule'); ax.barh(y - 0.2, G.duel_model * 100, 0.4, color=BLUE, label='Our model')
ax.set_yticks(y); ax.set_yticklabels([f"{n} ({int(r)})" for n, r in zip(G.index, G.races)]); ax.set_xlim(70, 88)
ax.set_xlabel('Duels predicted correctly, 2020-2025 (%)'); ax.legend(loc='upper right')
title(ax, 'The model beats the grid on grid-dependent tracks, but not at Monaco'); save(fig, '05_tracks'); plt.show()"""),
    md("""
**Reading.** At **Monaco the grid wins** (5 races): it names 9.2 of the 10 scorers against 8.0 for the model. Overtaking is almost impossible there, so any change the model makes to the grid order is a mistake. But Monaco is an exception, not a family: on the **third of tracks that depended most on the grid in 2010–2019** and on **street circuits**, the model still beats the grid. Monaco itself was *not* unusually grid-dependent in 2010–2019; it became so after 2021 (bigger cars), which is why a rule learnt from the past could not anticipate it.
"""),
    md("""
### One model or two? (normal tracks vs hard-to-overtake tracks)
Instead of one logistic regression for every track, we train **two**: one on normal tracks and one on the tracks where overtaking is hardest (Monaco, Zandvoort, Hungaroring, Marina Bay, Madring). Each has its own weights $\\theta$. For comparison, a second split defined from the data (top third of circuits by share of top-10 finishers who started in the top 10) gives the same conclusion. Code: `f1duel/tracks.py`.
"""),
    code("""
T = json.load(open('../results/tracks_summary.json'))
th = pd.DataFrame(T['final']['thetas']); display(th.round(2))
rows = [{'Test': t, 'Races': g, 'n': v['races'], 'Grid rule': v['named_grid'], 'One model': v['named_one'], 'Two models': v['named_two'],
         'Two models (data split)': v['named_two_data'], 'Cross-entropy one': v['ce_one'], 'Cross-entropy two': v['ce_two']}
        for t in ('2020-2025', '2026') for g, v in T[t].items()]
display(pd.DataFrame(rows).set_index(['Test', 'Races']).round(3))
print('Hard-to-overtake tracks:', ', '.join(x['name'] for x in T['final']['hard_tracks']))
from IPython.display import Image; display(Image('../figures/05_two_models.png', width=650))"""),
    md("""
**Reading.** The weights differ, but the hard-to-overtake model learns from only about 50 races at 5 circuits, so its weights are noisy (bias–variance trade-off, session 4): on the 20 hard-to-overtake test races it names fewer points scorers than the single model (7.85 vs 7.95). Two models do not predict better than one, so we keep one model for the forecasts.
"""),
    md("### The answer to our question"),
    code("""
a, b = S['2020-2025'], S['2026']
print(f\"\"\"Could we predict who finishes ahead, better than the grid?
  * Points (top 10):  YES on 2020-2025: {a['scorers_named']['Logistic regression']:.2f} vs {a['scorers_named']['Grid rule']:.2f} scorers per race (gain +{a['gain_vs_grid']['Logistic regression'][0]:.2f}, 95% CI {a['gain_vs_grid']['Logistic regression'][1]:+.2f} to {a['gain_vs_grid']['Logistic regression'][2]:+.2f}).
                      2026: {b['scorers_named']['Logistic regression']:.2f} vs {b['scorers_named']['Grid rule']:.2f} (gain {b['gain_vs_grid']['Logistic regression'][0]:+.2f}, 95% CI {b['gain_vs_grid']['Logistic regression'][1]:+.2f} to {b['gain_vs_grid']['Logistic regression'][2]:+.2f}, 17 races).
  * Duels (A vs B):   YES: {a['duel']['after']['accuracy']:.1%} vs {a['duel']['after']['grid_rule_accuracy']:.1%} (2020-2025) and {b['duel']['after']['accuracy']:.1%} vs {b['duel']['after']['grid_rule_accuracy']:.1%} (2026); both gains significant.
  * Winner:           the favourite is usually the pole sitter ({a['winner']['favourite_is_pole']:.0%} of races), so it wins as often ({a['winner']['favourite_won']:.0%}); the gain is in calibrated chances (log-loss {a['winner']['logloss']:.2f} vs {a['winner']['logloss_gridrate_baseline']:.2f}).
  * Before qualifying: worse than the grid. Saturday matters most.\"\"\")"""),
    md("""
### Next: the live test
The real test is the races that have not happened yet: **United States (25 Oct), Mexico City (1 Nov), São Paulo (8 Nov), Las Vegas (22 Nov), Qatar (29 Nov), Abu Dhabi (6 Dec)**. Before each one, `python -m f1duel.forecast` publishes the predictions (committed to GitHub before the start, so the timestamp proves it), and the website shows *model vs reality* after each race.
"""),
]


# ----------------------------------------------------------------------------- Part VI
NB["06_reset"] = [
    md("""
# Part VI — The reset: past results or this weekend's pace?
**Proposal question:** *can a pre-race model predict which drivers score points (top 10) through a regulation reset, and should it rely on past results or on the current weekend's pace?*

- **H1** after a reset, weekend pace (practice, qualifying, grid) predicts better than past results
- **H2** past results regain their value as the new season progresses
- **H3** live forecasts for the remaining 2026 races are as accurate as the simulation (tested after the races)

The same logistic regression is trained with different sets of variables. Each 2014 and 2026 race is predicted by a model re-trained on everything before it (car profile and k-means included). **New variables for the reset (the 2026 power units are about 50% electric):** `pu_works` (works or lead-partner team of the engine maker) and `pu_change` (the team changed engine supplier this season).
Code: `f1duel/reset.py`.
"""),
    code(SETUP + """
import json
from f1duel.reset import MODELS
R = json.load(open('../results/reset_summary.json'))
F = R['facts']
print(f"Points that changed hands over the winter: 2014 {F['points_changing_hands']['2014']:.0%}, 2026 {F['points_changing_hands']['2026']:.0%} (median of other years {np.median([v for y, v in F['points_changing_hands'].items() if y not in ('2014', '2026')]):.0%})")
print(f"Mechanical retirements: 2025 {F['mechanical_dnf_rate']['2025']:.0%} -> 2026 {F['mechanical_dnf_rate']['2026']:.0%}; 2013 {F['mechanical_dnf_rate']['2013']:.0%} -> 2014 {F['mechanical_dnf_rate']['2014']:.0%}")
from IPython.display import Image; display(Image('../figures/06_reset_facts.png', width=800))"""),
    code("""
rows = []
for test in ['Reset 2026', 'Reset 2014', 'Normal seasons 2020-2025']:
    for n, v in R[test]['models'].items():
        rows.append({'Test': test, 'Model': n, 'Cross-entropy rounds 1-5': v['logloss_early'], 'Cross-entropy rounds 6+': v['logloss_late'],
                     'Cross-entropy all': v['logloss'], 'AUC': v['auc'], 'Precision = recall (of 10)': v['named'] / 10, 'Gain vs grid': v['gain_vs_grid'][0]})
T = pd.DataFrame(rows).set_index(['Test', 'Model']).round(4); T"""),
    code("""
for test in ['Reset 2026', 'Reset 2014']:
    m = R[test]['models']
    e = m['Past results only']['logloss_early'] - m['Weekend pace only']['logloss_early']
    l = m['Past results only']['logloss_late'] - m['Weekend pace only']['logloss_late']
    pu = m['Full model']['logloss'] - m['Full model (no power unit)']['logloss']
    print(f"{test}: H1 weekend beats past early by {e:+.4f} cross-entropy; H2 gap later {l:+.4f} (positive = weekend better); power unit changes the full model by {pu:+.4f}")
from IPython.display import Image; display(Image('../figures/06_reset_2026.png', width=700))"""),
    md("""
**Reading.** H1 holds in both resets: in the first five races, the weekend's pace predicts the points scorers better than past results. H2 holds too: the gap shrinks as the season goes on (in 2014, past results even become the better source after round 5). The full model, which combines both, is best overall. The power-unit variables do **not** improve the full model: the engine's past results are themselves "past results", which a reset makes stale; what matters is visible on Saturday.
"""),
]


def main():
    ep = ExecutePreprocessor(timeout=1800, kernel_name="python3")
    for name, cells in NB.items():
        nb = nbf.v4.new_notebook(); nb.cells = cells
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        ep.preprocess(nb, {"metadata": {"path": str(HERE)}})
        nbf.write(nb, HERE / f"{name}.ipynb")
        print("ok", name)


if __name__ == "__main__":
    main()
