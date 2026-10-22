"""Part VII - the verdict: every model against the grid and against reality.

Test 1 (2020-2025): each season is predicted by models trained on all EARLIER seasons.
Test 2 (2026, new rules): each race is predicted by models re-trained on everything before it,
                          exactly as a live forecast would be made.
KNN's k is chosen once, on seasons 2016-2019, and never on the test seasons.

Usage: python -m f1duel.evaluate   -> results/predictions.csv, results/summary.json
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from .data import ROOT, build
from .model import (AFTER_QUALI, BEFORE_WEEKEND, CarProfile, DuelModel, WinnerModel, fit_knn,
                    fit_top10, race_pairs, scorers_named)

RES = ROOT / "results"
K_GRID = [5, 15, 25, 51, 101, 201, 301]


def choose_k(d):
    """Season-ahead on 2016-2019: pick the k with the best average log-loss."""
    scores = {}
    for k in K_GRID:
        ll = []
        for y in range(2016, 2020):
            prof = CarProfile().fit(d[d.year < y])
            tr, te = prof.transform(d[d.year < y]), prof.transform(d[d.year == y])
            p = np.clip(fit_knn(tr, AFTER_QUALI, k).predict_proba(te[AFTER_QUALI])[:, 1], 1e-3, 1 - 1e-3)
            ll.append(log_loss(te.top10, p))
        scores[k] = float(np.mean(ll))
    return min(scores, key=scores.get), scores


def predict_block(train_raw, test_raw, k):
    """Fit everything on train_raw, predict test_raw. Returns test rows with predictions + fitted duel models."""
    prof = CarProfile().fit(train_raw)
    tr, te = prof.transform(train_raw), prof.transform(test_raw).copy()
    te["p_top10_logit"] = fit_top10(tr, AFTER_QUALI).predict_proba(te[AFTER_QUALI])[:, 1]
    te["p_top10_logit_before"] = fit_top10(tr, BEFORE_WEEKEND).predict_proba(te[BEFORE_WEEKEND])[:, 1]
    te["p_top10_knn"] = fit_knn(tr, AFTER_QUALI, k).predict_proba(te[AFTER_QUALI])[:, 1]
    duel = DuelModel().fit(tr, AFTER_QUALI)
    duel_b = DuelModel().fit(tr, BEFORE_WEEKEND)
    te["race_score"] = duel.score(te)
    win = WinnerModel().fit(tr, AFTER_QUALI)
    te = pd.concat([g for _, g in te.groupby("raceId", sort=False)])       # rows grouped by race
    te["p_win"] = np.concatenate([win.prob(g) for _, g in te.groupby("raceId", sort=False)])
    # simpler baseline for the winner: historical win rate of each grid slot, normalised within the race
    rate = tr.groupby(tr.grid.clip(upper=20)).win.mean()
    te["p_win_gridrate"] = te.grid.clip(upper=20).map(rate).fillna(rate.min()).clip(lower=1e-4)
    te["p_win_gridrate"] /= te.groupby("raceId").p_win_gridrate.transform("sum")
    duel_stats = []
    for name, m in [("after", duel), ("before", duel_b)]:
        X, y, grid_says, race = race_pairs(te, m.features, m.scaler, return_race=True)
        p = m.lr.predict_proba(X)[:, 1]
        for rid in np.unique(race):
            k = race == rid
            duel_stats.append({"model": name, "raceId": int(rid), "pairs": int(k.sum()),
                               "correct": int(((p[k] > 0.5) == y[k]).sum()), "grid_correct": int((grid_says[k] == y[k]).sum()),
                               "logloss_sum": float(-(y[k] * np.log(p[k]) + (1 - y[k]) * np.log(1 - p[k])).sum())})
    return te, duel_stats


def track_analysis(d, P, D):
    """Teacher's question: Monaco (and similar tracks) depend on the grid. Does the model still help there?
    Grid dependence of a circuit = share of its top-10 finishers who started in the top 10, measured on
    2010-2019 only (before the test seasons), so the groups are fixed without looking at the test races."""
    hist = d[d.year < 2020]
    dep = (hist[hist.top10 == 1].assign(g10=lambda x: x.grid <= 10)
           .groupby(["circuitId", "raceId"]).g10.mean().groupby("circuitId").mean())
    top_third = set(dep[dep >= dep.quantile(2 / 3)].index)
    Q = P[P.test == "2020-2025"]
    races = Q.groupby("raceId").agg(circuitId=("circuitId", "first"), circuit_type=("circuit_type", "first"))
    races["grid_named"] = scorers_named(Q, "grid", ascending=True)
    races["model_named"] = scorers_named(Q, "p_top10_logit")
    races = races.join(D[(D.test == "2020-2025") & (D.model == "after")].set_index("raceId")[["pairs", "correct", "grid_correct"]])
    groups = {"All races": races.index == races.index,
              "Street circuits": races.circuit_type == "STREET",
              "Most grid-dependent third of tracks": races.circuitId.isin(top_third),
              "Monaco": races.circuitId == "monaco"}
    out = {}
    for name, m in groups.items():
        r = races[m]
        out[name] = {"races": int(len(r)), "scorers_grid": float(r.grid_named.mean()), "scorers_model": float(r.model_named.mean()),
                     "duel_grid": float(r.grid_correct.sum() / r.pairs.sum()), "duel_model": float(r.correct.sum() / r.pairs.sum())}
    out["grid_dependent_tracks"] = sorted(top_third)
    out["monaco_grid_dependence_2010_2019"] = float(dep.get("monaco", np.nan))
    out["grid_dependence_median_2010_2019"] = float(dep.median())
    return out


def main():
    t0 = time.time()
    d = build()
    k, k_scores = choose_k(d)
    print("chosen k =", k, {kk: round(v, 4) for kk, v in k_scores.items()})
    preds, duels = [], []
    for y in range(2020, 2026):                                       # Test 1
        te, ds = predict_block(d[d.year < y], d[d.year == y], k)
        preds.append(te.assign(test="2020-2025")); duels += [dict(x, test="2020-2025", year=y) for x in ds]
        print(y, f"{time.time() - t0:.0f}s")
    for r in sorted(d[d.year == 2026]["round"].unique()):             # Test 2
        past = d[(d.year < 2026) | ((d.year == 2026) & (d["round"] < r))]
        te, ds = predict_block(past, d[(d.year == 2026) & (d["round"] == r)], k)
        preds.append(te.assign(test="2026")); duels += [dict(x, test="2026", round=int(r)) for x in ds]
    P = pd.concat(preds)
    P.to_csv(RES / "predictions.csv", index=False)
    D = pd.DataFrame(duels)
    D.to_csv(RES / "duels.csv", index=False)

    summary = {"k": k, "k_cv_logloss": k_scores}
    for test, Q in P.groupby("test"):
        races = Q.raceId.nunique()
        s = {"races": races, "starts": len(Q)}
        named = {"Grid rule": scorers_named(Q, "grid", ascending=True)}
        for col, name in [("p_top10_logit", "Logistic regression"), ("p_top10_knn", "KNN"),
                          ("p_top10_logit_before", "Logistic, before the weekend")]:
            named[name] = scorers_named(Q, col)
            s.setdefault("auc", {})[name] = roc_auc_score(Q.top10, Q[col])
        s["scorers_named"] = {n: float(v.mean()) for n, v in named.items()}
        rng = np.random.default_rng(2026)
        for n in ["Logistic regression", "KNN", "Logistic, before the weekend"]:
            diff = (named[n] - named["Grid rule"]).to_numpy()
            boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(5000)]
            s.setdefault("gain_vs_grid", {})[n] = [float(diff.mean()), float(np.percentile(boots, 2.5)),
                                                   float(np.percentile(boots, 97.5))]
        g = D[D.test == test].groupby("model")[["pairs", "correct", "grid_correct", "logloss_sum"]].sum()
        s["duel"] = {m: {"pairs": int(r.pairs), "accuracy": r.correct / r.pairs,
                         "grid_rule_accuracy": r.grid_correct / r.pairs, "logloss": r.logloss_sum / r.pairs}
                     for m, r in g.iterrows()}
        for m in ("after", "before"):                      # bootstrap over races for the duel gain
            rr = D[(D.test == test) & (D.model == m)]
            c, gc, n = rr.correct.to_numpy(), rr.grid_correct.to_numpy(), rr.pairs.to_numpy()
            bs = []
            for _ in range(5000):
                ix = rng.integers(0, len(rr), len(rr))
                bs.append((c[ix].sum() - gc[ix].sum()) / n[ix].sum())
            s["duel"][m]["gain_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        w = []
        for _, g2 in Q.groupby("raceId"):
            if g2.win.sum() != 1:
                continue
            w.append({"fav_won": int(g2.p_win.idxmax() == g2.win.idxmax()), "pole_won": int(g2.loc[g2.win.idxmax(), "grid"] == 1),
                      "fav_is_pole": int(g2.loc[g2.p_win.idxmax(), "grid"] == 1),
                      "p_winner": float(g2.loc[g2.win.idxmax(), "p_win"]),
                      "p_winner_gridrate": float(g2.loc[g2.win.idxmax(), "p_win_gridrate"])})
        w = pd.DataFrame(w)
        s["winner"] = {"races": len(w), "favourite_won": float(w.fav_won.mean()), "pole_won": float(w.pole_won.mean()),
                       "favourite_is_pole": float(w.fav_is_pole.mean()),
                       "logloss": float(-np.log(w.p_winner).mean()),
                       "logloss_gridrate_baseline": float(-np.log(w.p_winner_gridrate).mean())}
        # calibration: predicted vs observed, in bins
        cal = {}
        for col, tgt, edges in [("p_win", "win", [0, .02, .05, .15, .35, .6, 1]),
                                ("p_top10_logit", "top10", [0, .1, .3, .5, .7, .9, 1])]:
            b = pd.cut(Q[col], edges, include_lowest=True)
            t = Q.groupby(b, observed=True).agg(predicted=(col, "mean"), observed=(tgt, "mean"), n=(tgt, "size"))
            cal[col] = [{"bin": str(i), **{k: float(v) for k, v in r.items()}} for i, r in t.iterrows()]
        s["calibration"] = cal
        summary[test] = s
    summary["tracks"] = track_analysis(d, P, D)
    (RES / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    RES.mkdir(exist_ok=True)
    main()
