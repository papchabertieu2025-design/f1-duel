"""The proposal's question: can a pre-race model predict who scores points (top 10) through a
regulation reset, and should it rely on past results or on the current weekend's pace?

  H1  after a reset, weekend pace predicts better than past results
  H2  past results regain their value as the new season progresses
  H3  live forecasts for the remaining 2026 races are as accurate as the simulation

Tests (every model is trained only on races before the one it predicts):
  * reset 2026 and reset 2014 (the two engine rule changes): race by race, re-trained before each race
  * normal seasons 2020-2025: each season predicted from all earlier seasons
New variables for the reset: the power unit (pu_form, pu_works, pu_change, prev_season_fade).

Usage: python -m f1duel.reset  -> results/reset_summary.json, results/reset_predictions.csv, figures/06_reset_*.png
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .data import PU_FEATURES, ROOT, build
from .model import CarProfile, fit_top10, scorers_named

PAST = ["drv_avg_finish", "drv_top10", "car_perf", "car_rel", "tier_top", "tier_unrel"]
GRID = ["grid", "log_grid"]
WEEKEND = GRID + ["quali_gap", "practice_gap"]
MODELS = {                                         # name -> features (logistic regression)
    "Past results only": PAST,
    "Past results + power unit": PAST + PU_FEATURES,
    "Grid position only": GRID,
    "Weekend pace only": WEEKEND,
    "Full model (no power unit)": WEEKEND + PAST,
    "Full model": WEEKEND + PAST + PU_FEATURES,
}
EARLY = 5                                           # "early season" = rounds 1-5


def predict(train_raw, test_raw):
    prof = CarProfile().fit(train_raw)
    tr, te = prof.transform(train_raw), prof.transform(test_raw).copy()
    for name, feats in MODELS.items():
        te[name] = fit_top10(tr, feats).predict_proba(te[feats])[:, 1]
    return te


def run():
    d = build()
    out = []
    for year in (2014, 2026):                                         # the two engine resets, race by race
        for r in sorted(d[d.year == year]["round"].unique()):
            past = d[(d.year < year) | ((d.year == year) & (d["round"] < r))]
            out.append(predict(past, d[(d.year == year) & (d["round"] == r)]).assign(test=f"Reset {year}"))
    for year in range(2020, 2026):                                    # normal seasons
        out.append(predict(d[d.year < year], d[d.year == year]).assign(test="Normal seasons 2020-2025"))
    return d, pd.concat(out)


def summarise(P):
    names = list(MODELS)
    rng = np.random.default_rng(2026)
    S = {}
    for test, Q in P.groupby("test"):
        early, late = Q[Q["round"] <= EARLY], Q[Q["round"] > EARLY]
        grid_named = scorers_named(Q, "grid", ascending=True)
        s = {"races": int(Q.raceId.nunique()), "starts": int(len(Q)),
             "grid_rule_named": float(grid_named.mean()), "models": {}}
        for n in names:
            named = scorers_named(Q, n)
            diff = (named - grid_named).to_numpy()
            boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(5000)]
            s["models"][n] = {"logloss": log_loss(Q.top10, Q[n]),
                              "logloss_early": log_loss(early.top10, early[n]),
                              "logloss_late": log_loss(late.top10, late[n]),
                              "brier": brier_score_loss(Q.top10, Q[n]),
                              "brier_early": brier_score_loss(early.top10, early[n]),
                              "brier_late": brier_score_loss(late.top10, late[n]),
                              "auc": roc_auc_score(Q.top10, Q[n]),
                              "named": float(named.mean()),
                              "gain_vs_grid": [float(diff.mean()), float(np.percentile(boots, 2.5)),
                                               float(np.percentile(boots, 97.5))]}
        S[test] = s
    return S


def reset_facts(d):
    """The two facts of the proposal's Figure 1: points changing hands over the winter, mechanical retirements."""
    tp = d.groupby(["year", "round", "team"]).points_scored.sum().reset_index()
    shift = {}
    for y in range(2011, 2027):
        a = tp[tp.year == y - 1]; a = a[a["round"] > a["round"].max() - 5].groupby("team").points_scored.sum()
        b = tp[(tp.year == y) & (tp["round"] <= 5)].groupby("team").points_scored.sum()
        a, b = a / a.sum(), b / b.sum()
        teams = a.index.union(b.index)
        shift[y] = float(0.5 * (a.reindex(teams, fill_value=0) - b.reindex(teams, fill_value=0)).abs().sum())
    mech = d.groupby("year").dnf_mechanical.mean().astype(float).to_dict()
    return {"points_changing_hands": shift, "mechanical_dnf_rate": {int(k): v for k, v in mech.items()}}


def figures(S, F):
    import matplotlib.pyplot as plt
    from .plots import BLUE, GREY, ORANGE, style, title, save
    style()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    ys = sorted(F["points_changing_hands"])
    c = [ORANGE if y in (2014, 2026) else GREY for y in ys]
    ax[0].bar(ys, [100 * F["points_changing_hands"][y] for y in ys], color=c)
    ax[0].set_ylabel("% of team points"); ax[0].set_xticks([2011, 2014, 2017, 2020, 2023, 2026]); title(ax[0], "(a) Points that change hands over the winter")
    ym = sorted(F["mechanical_dnf_rate"])
    ax[1].bar(ym, [100 * F["mechanical_dnf_rate"][y] for y in ym], color=[ORANGE if y in (2014, 2026) else GREY for y in ym])
    ax[1].set_ylabel("% of starters"); ax[1].set_xticks([2010, 2014, 2018, 2022, 2026]); title(ax[1], "(b) Mechanical retirements")
    save(fig, "06_reset_facts"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    show = ["Past results only", "Past results + power unit", "Grid position only", "Weekend pace only", "Full model"]
    x = np.arange(len(show)); w = 0.38
    m = S["Reset 2026"]["models"]
    ax.bar(x - w / 2, [m[n]["logloss_early"] for n in show], w, color=ORANGE, label="Rounds 1-5")
    ax.bar(x + w / 2, [m[n]["logloss_late"] for n in show], w, color=BLUE, label="Rounds 6-17")
    ax.set_xticks(x); ax.set_xticklabels([n.replace(" + ", "\n+ ").replace(" only", "\nonly") for n in show], fontsize=9)
    ax.set_ylabel("Cross-entropy (lower = better)"); ax.set_ylim(0.40, 0.58); ax.legend()
    title(ax, "2026 reset: which information predicts the points scorers?")
    save(fig, "06_reset_2026"); plt.close(fig)


def main():
    d, P = run()
    S = summarise(P)
    S["facts"] = reset_facts(d)
    (ROOT / "results" / "reset_summary.json").write_text(json.dumps(S, indent=1))
    P.to_csv(ROOT / "results" / "reset_predictions.csv", index=False)
    figures(S, S["facts"])
    for test, s in S.items():
        if test == "facts":
            continue
        print(f"\n{test}: {s['races']} races, grid rule names {s['grid_rule_named']:.2f}/10")
        for n, v in s["models"].items():
            print(f"  {n:28s} CE {v['logloss']:.4f} (early {v['logloss_early']:.4f}, late {v['logloss_late']:.4f})"
                  f"  AUC {v['auc']:.3f}  named {v['named']:.2f}  gain {v['gain_vs_grid'][0]:+.2f}"
                  f" [{v['gain_vs_grid'][1]:+.2f}, {v['gain_vs_grid'][2]:+.2f}]")
    F = S["facts"]
    print("\npoints changing hands:", {y: round(v, 2) for y, v in F["points_changing_hands"].items()})
    print("mechanical DNF:", {y: round(v, 3) for y, v in F["mechanical_dnf_rate"].items()})


if __name__ == "__main__":
    main()
