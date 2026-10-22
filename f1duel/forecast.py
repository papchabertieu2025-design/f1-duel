"""Live forecast for the next Grand Prix + the file the website reads (app/model.json).

Two models (decision of the progress report): the next race is forecast by the model trained only on past
races of the same track type (hard-to-overtake: Monaco, Zandvoort, Hungaroring, Marina Bay, Madring; or normal).
The single model trained on every race is kept alongside, so both are compared on the live races.

Two moments:
  * BEFORE THE WEEKEND (Monday): only history is known -> car profile (PCA + k-means) and driver form.
  * AFTER QUALIFYING (Saturday night): fill data/upcoming/<year>_r<round>.csv with
    driverId, grid, quali_time_s, practice_time_s  -> the full model (8 features) is used.

Usage:  python -m f1duel.forecast 2026 18      (United States GP)
Then commit forecasts/ and app/model.json BEFORE the race: the commit time proves the forecast.
"""
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .data import ROOT, build
from .model import AFTER_QUALI, BEFORE_WEEKEND, TIERS, CarProfile, DuelModel, WinnerModel, fit_top10
from .tracks import HARD_TO_OVERTAKE

LABELS = {"grid": "starting position", "log_grid": "front-row advantage (log grid)",
          "quali_gap": "qualifying gap to pole", "practice_gap": "practice gap to fastest",
          "car_perf": "car performance (PCA axis 1)", "car_rel": "car reliability (PCA axis 2)",
          "tier_top": "top-car family (k-means)", "tier_back": "back-of-grid family (k-means)",
          "drv_avg_finish": "driver's average finish (last 5)", "drv_top10": "driver's top-10 rate (last 5)"}


def export_linear(model, features):
    """A logistic/duel/softmax model as plain numbers: standardise with (mean, scale), then dot with w."""
    sc = model.scaler if hasattr(model, "scaler") else model[0]
    if hasattr(model, "lr"):
        w, b = model.lr.coef_[0], 0.0
    elif hasattr(model, "w"):
        w, b = model.w, 0.0
    else:
        w, b = model[-1].coef_[0], float(model[-1].intercept_[0])
    return {"features": features, "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "w": w.tolist(), "b": b}


def scoreboard():
    """Model vs reality for every 2026 race already run (from results/predictions.csv)."""
    path = ROOT / "results" / "predictions.csv"
    if not path.exists():
        return []
    P = pd.read_csv(path)
    tp = ROOT / "results" / "tracks_predictions.csv"
    T = pd.read_csv(tp) if tp.exists() else None
    out = []
    for (r, gp), g in P[P.test == "2026"].groupby(["round", "gp_name"]):
        top_model = set(g.nlargest(10, "p_top10_logit").driverId)
        top_grid = set(g.nsmallest(10, "grid").driverId)
        fav = g.loc[g.p_win.idxmax()]
        win = g[g.win == 1]
        two_hits = None
        if T is not None:
            t = T[(T.test == "2026") & (T["round"] == r)]
            if len(t):
                two_hits = int(t[t.driverId.isin(set(t.nlargest(10, "p_two").driverId))].top10.sum())
        out.append({"round": int(r), "gp": gp, "model_hits": int(g[g.driverId.isin(top_model)].top10.sum()),
                    "two_model_hits": two_hits, "hard_track": bool(g.circuitId.iloc[0] in HARD_TO_OVERTAKE),
                    "grid_hits": int(g[g.driverId.isin(top_grid)].top10.sum()),
                    "favourite": fav.driver_name, "favourite_p": round(float(fav.p_win), 3),
                    "winner": win.driver_name.iloc[0] if len(win) else None})
    return out


def main(year, rnd):
    d = build(upcoming=(year, rnd))
    past = d[d.upcoming == 0]
    prof = CarProfile().fit(past)
    tr, nxt = prof.transform(past), prof.transform(d[d.upcoming == 1]).copy()
    after = nxt.grid.notna().all()
    feats = AFTER_QUALI if after else BEFORE_WEEKEND

    hard = nxt.circuitId.iloc[0] in HARD_TO_OVERTAKE              # which of the two models?
    tr_type = tr[tr.circuitId.isin(HARD_TO_OVERTAKE) == hard]
    top10 = fit_top10(tr_type, feats)
    duel = DuelModel().fit(tr_type, feats)
    win = WinnerModel().fit(tr_type, feats)
    nxt["p_top10"] = top10.predict_proba(nxt[feats])[:, 1]
    nxt["p_top10_one"] = fit_top10(tr, feats).predict_proba(nxt[feats])[:, 1]   # single model, for comparison
    nxt["race_score"] = duel.score(nxt)
    nxt["p_win"] = win.prob(nxt)
    nxt["tier_name"] = [TIERS[t] for t in nxt.tier]
    nxt = nxt.sort_values("p_win", ascending=False)

    race = nxt.iloc[0]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    out_csv = ROOT / "forecasts" / f"{year}_r{rnd:02d}_{'after_quali' if after else 'before_weekend'}.csv"
    out_csv.parent.mkdir(exist_ok=True)
    nxt[["driver_name", "team_name", "grid", "tier_name", "p_top10", "p_top10_one", "p_win"]].round(3).to_csv(out_csv, index=False)

    S = json.loads((ROOT / "results" / "summary.json").read_text())
    tracks_path = ROOT / "results" / "tracks_summary.json"
    tracks = None
    if tracks_path.exists():                              # one model or two? (f1duel.tracks)
        T = json.loads(tracks_path.read_text())
        tracks = {"next_race_type": "Hard-to-overtake track" if race.circuitId in HARD_TO_OVERTAKE else "Normal track",
                  "thetas": T["final"]["thetas"], "hard_tracks": [t["name"] for t in T["final"]["hard_tracks"]],
                  "test": {key: {m: T["2020-2025"][group][m] for m in ("races", "named_grid", "named_one", "named_two")}
                           for key, group in [("all", "All races"), ("hard", "Hard-to-overtake track")]}}
    model = {
        "generated": stamp, "race": {"year": year, "round": rnd, "name": race.gp_name,
                                     "date": str(race.date)[:10], "circuit": race.circuitId},
        "mode": "after_quali" if after else "before_weekend",
        "model_used": "Hard-to-overtake track" if hard else "Normal track",
        "training_races": int(tr_type.raceId.nunique()),
        "labels": LABELS,
        "drivers": [{"id": r.driverId, "name": r.driver_name, "code": r.driver_code, "team": r.team_name,
                     "tier": r.tier_name, "grid": None if pd.isna(r.grid) else int(r.grid),
                     "x": {f: float(getattr(r, f)) for f in feats}, "p_top10": round(float(r.p_top10), 4),
                     "p_top10_one": round(float(r.p_top10_one), 4),
                     "p_win": round(float(r.p_win), 4)} for r in nxt.itertuples()],
        "duel": export_linear(duel, feats),
        "top10": export_linear(top10, feats),
        "winner": export_linear(win, feats),
        "evidence": {t: {"duel_accuracy": S[t]["duel"]["after"]["accuracy"],
                         "duel_grid_rule": S[t]["duel"]["after"]["grid_rule_accuracy"],
                         "duel_accuracy_before": S[t]["duel"]["before"]["accuracy"],
                         "scorers_model": S[t]["scorers_named"]["Logistic regression"],
                         "scorers_grid": S[t]["scorers_named"]["Grid rule"],
                         "races": S[t]["races"]} for t in ("2020-2025", "2026")},
        "scoreboard": scoreboard(),
        "tracks": tracks,
        "source": "Official Formula 1 race results via F1DB (github.com/f1db/f1db), snapshot "
                  + (ROOT / "data" / "raw" / "RETRIEVED_AT.txt").read_text().strip(),
    }
    (ROOT / "app" / "model.json").write_text(json.dumps(model, indent=1))
    print(f"{race.gp_name} {year} (round {rnd}) - mode: {model['mode']} - {stamp}")
    print(nxt[["driver_name", "team_name", "tier_name", "p_top10", "p_win"]].round(3).head(12).to_string(index=False))
    print(f"-> {out_csv.relative_to(ROOT)} and app/model.json")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
