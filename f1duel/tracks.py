"""Two models instead of one: one for normal tracks, one for tracks where overtaking is hardest (Monaco-like).

Same logistic regression P(y=1|x) = sigma(theta . x), same 8 predictors, but each track type gets its own
weights theta: on a track where overtaking is hard, the grid should weigh more.

Track type (main definition): the 2026 circuits where overtaking is hardest, from the team's list
(Monaco, Zandvoort, Hungaroring, Marina Bay, Madring); every other circuit is a normal track.
Comparison definition, from PAST races only: grid dependence of a circuit = share of its top-10 finishers who
started in the top 10; circuits in the top third are "grid-dependent" (at least 3 races).

Test (never uses the future): seasons 2020-2025 predicted from earlier seasons; 2026 race by race.
Compared with the single model and with the grid rule.

Usage: python -m f1duel.tracks  -> results/tracks_summary.json, figures/06_two_models.png
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from .data import ROOT, build
from .model import AFTER_QUALI, CarProfile, fit_top10, scorers_named

TYPES = ["Normal track", "Hard-to-overtake track"]
HARD_TO_OVERTAKE = {"monaco": "Monaco", "zandvoort": "Zandvoort (Netherlands)", "hungaroring": "Hungaroring (Hungary)",
                    "marina-bay": "Marina Bay (Singapore)", "madring": "Madring (Madrid, Spain)"}
MIN_RACES = 3


def grid_dependence(hist):
    """Per circuit: average share of top-10 finishers who started in the top 10, and number of races."""
    t = hist[hist.top10 == 1].assign(g10=lambda x: x.grid <= 10)
    per_race = t.groupby(["circuitId", "raceId"]).g10.mean()
    return per_race.groupby("circuitId").agg(["mean", "size"]).rename(columns={"mean": "dependence", "size": "races"})


def track_types(hist):
    """circuitId -> 1 if grid-dependent (top third of circuits with enough races), else 0."""
    dep = grid_dependence(hist)
    ok = dep[dep.races >= MIN_RACES]
    cut = ok.dependence.quantile(2 / 3)
    return {c: int(r.races >= MIN_RACES and r.dependence >= cut) for c, r in dep.iterrows()}, float(cut)


class TwoModels:
    """One logistic regression per track type. definition = "list" (hard-to-overtake list) or "data" (grid dependence)."""

    def __init__(self, definition="list"):
        self.definition = definition

    def fit(self, train):
        if self.definition == "list":
            self.types, self.cut = {c: int(c in HARD_TO_OVERTAKE) for c in train.circuitId.unique()} | {c: 1 for c in HARD_TO_OVERTAKE}, None
        else:
            self.types, self.cut = track_types(train)
        tr = train.assign(gd=train.circuitId.map(self.types).fillna(0).astype(int))
        self.models = {g: fit_top10(tr[tr.gd == g], AFTER_QUALI) for g in (0, 1)}
        return self

    def kind(self, df):
        return df.circuitId.map(self.types).fillna(0).astype(int)

    def predict(self, df):
        g = self.kind(df).to_numpy()
        p = np.empty(len(df))
        for k in (0, 1):
            if (g == k).any():
                p[g == k] = self.models[k].predict_proba(df.loc[g == k, AFTER_QUALI])[:, 1]
        return p

    def thetas(self):
        """Standardised weights of each model (one row per predictor) + intercept."""
        out = {}
        for k, m in self.models.items():
            lr = m[-1]
            out[TYPES[k]] = dict(zip(AFTER_QUALI, lr.coef_[0].round(3).tolist()), intercept=round(float(lr.intercept_[0]), 3))
        return out


INTER = AFTER_QUALI + ["gd", "grid_x_gd"]          # one model, grid allowed to weigh more on grid-dependent tracks


def add_interaction(df, kind):
    return df.assign(gd=kind, grid_x_gd=df.grid * kind)


def predict_block(train_raw, test_raw):
    prof = CarProfile().fit(train_raw)
    tr, te = prof.transform(train_raw), prof.transform(test_raw).copy()
    te["p_one"] = fit_top10(tr, AFTER_QUALI).predict_proba(te[AFTER_QUALI])[:, 1]
    two = TwoModels("list").fit(tr)
    te["p_two"] = two.predict(te)
    te["track_type"] = two.kind(te)
    te["p_two_data"] = TwoModels("data").fit(tr).predict(te)
    tri = add_interaction(tr, two.kind(tr))
    tei = add_interaction(te, te.track_type)
    te["p_inter"] = fit_top10(tri, INTER).predict_proba(tei[INTER])[:, 1]
    return te


def run():
    d = build()
    out = []
    for y in range(2020, 2026):
        out.append(predict_block(d[d.year < y], d[d.year == y]).assign(test="2020-2025"))
    for r in sorted(d[d.year == 2026]["round"].unique()):
        past = d[(d.year < 2026) | ((d.year == 2026) & (d["round"] < r))]
        out.append(predict_block(past, d[(d.year == 2026) & (d["round"] == r)]).assign(test="2026"))
    return d, pd.concat(out)


def summarise(P):
    rng = np.random.default_rng(2026)
    S = {}
    for test, Q in P.groupby("test"):
        s = {}
        for name, R in [("All races", Q)] + [(TYPES[k], Q[Q.track_type == k]) for k in (0, 1)]:
            if R.empty:
                continue
            grid = scorers_named(R, "grid", ascending=True)
            one, two, inter = scorers_named(R, "p_one"), scorers_named(R, "p_two"), scorers_named(R, "p_inter")
            two_data = scorers_named(R, "p_two_data")
            diff = (two - one).to_numpy()
            boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(5000)]
            s[name] = {"races": int(R.raceId.nunique()),
                       "named_grid": float(grid.mean()), "named_one": float(one.mean()), "named_two": float(two.mean()),
                       "named_inter": float(inter.mean()), "ce_inter": float(log_loss(R.top10, R.p_inter)),
                       "named_two_data": float(two_data.mean()), "ce_two_data": float(log_loss(R.top10, R.p_two_data)),
                       "ce_one": float(log_loss(R.top10, R.p_one)), "ce_two": float(log_loss(R.top10, R.p_two)),
                       "gain_two_vs_one": [float(diff.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
        S[test] = s
    return S


def figure(F):
    import matplotlib.pyplot as plt
    from .plots import BLUE, ORANGE, style, title, save
    style()
    feats = AFTER_QUALI
    a = [F["thetas"]["Normal track"][f] for f in feats]
    b = [F["thetas"]["Hard-to-overtake track"][f] for f in feats]
    y = np.arange(len(feats))[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.barh(y + 0.2, a, 0.4, color=BLUE, label="Normal tracks")
    ax.barh(y - 0.2, b, 0.4, color=ORANGE, label="Hard-to-overtake tracks")
    ax.set_yticks(y); ax.set_yticklabels(feats); ax.axvline(0, color="#888", lw=0.6)
    ax.set_xlabel(r"weight $\theta$ per standard deviation (positive = more likely to score)"); ax.legend(loc="lower right")
    title(ax, r"Two models, two sets of weights $\theta$")
    save(fig, "06_two_models"); plt.close(fig)


def final_fit(d):
    """Fit on all races run so far: the weights shown in the report and on the website."""
    past = d[d.upcoming == 0] if "upcoming" in d else d
    prof = CarProfile().fit(past)
    two = TwoModels("list").fit(prof.transform(past))
    dep = grid_dependence(past)
    data_types, cut = track_types(past)
    names = past.drop_duplicates("circuitId").set_index("circuitId").gp_name
    return {"thetas": two.thetas(), "cutoff": cut,
            "hard_tracks": [{"circuit": c, "name": n, "races_2010_2026": int(dep.loc[c, "races"]) if c in dep.index else 0}
                            for c, n in HARD_TO_OVERTAKE.items()],
            "grid_dependent_tracks_data": [names.get(c, c) for c, v in sorted(data_types.items()) if v]}


def main():
    d, P = run()
    S = summarise(P)
    S["final"] = final_fit(d)
    P[["test", "raceId", "year", "round", "gp_name", "circuitId", "driverId", "driver_name", "grid", "top10",
       "track_type", "p_one", "p_two"]].to_csv(ROOT / "results" / "tracks_predictions.csv", index=False)
    (ROOT / "results" / "tracks_summary.json").write_text(json.dumps(S, indent=1))
    figure(S["final"])
    for test in ("2020-2025", "2026"):
        print(f"\n{test}")
        for name, v in S[test].items():
            g = v["gain_two_vs_one"]
            print(f"  {name:22s} {v['races']:3d} races | named: grid {v['named_grid']:.2f}  one model {v['named_one']:.2f}  "
                  f"two models {v['named_two']:.2f} (two-one {g[0]:+.2f} [{g[1]:+.2f}, {g[2]:+.2f}]) | "
                  f"CE one {v['ce_one']:.4f} two {v['ce_two']:.4f} | interaction {v['named_inter']:.2f} | two (data def.) {v['named_two_data']:.2f}")
    print("\nhard-to-overtake tracks:", [t["name"] for t in S["final"]["hard_tracks"]])
    print(json.dumps(S["final"]["thetas"], indent=1))


if __name__ == "__main__":
    main()
