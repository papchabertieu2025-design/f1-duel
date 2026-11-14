"""Replay any Grand Prix on the website: model A's forecast (made with earlier races only) and the grid rule
(model B), against the real result, for every race since 2011.

  2011-2025  model A re-trained before each race on all earlier races and simulated 10,000 times
             (same code and seeds as f1duel.choose, so the scores match results/choose/backtest).
  2026       the forecasts of f1duel.montecarlo (results/montecarlo/2026_rXX_*.csv), as in the report.

Each driver also gets model A's score s; on the website, P(i ahead of j | both finish) = 1 / (1 + exp(-(s_i - s_j))).
Usage:  python -m f1duel.site_past 160      (repeat until it prints 'all races done'; resumes from the cache)
        python -m f1duel.site_past build    (writes app/past.json)
"""
import json
import sys
import time

import numpy as np
import pandas as pd

from .choose import predict, score
from .data import ROOT, build
from .model import CarProfile
from .montecarlo import FEAT_A, OrderModel, summarise_sims

CACHE = ROOT / "results" / "site_past"
YEAR = 2026


def drivers_table(te, s, summ, finish):
    rank = pd.Series(summ["exp_pos"]).rank(method="first").astype(int).to_numpy()
    rows = []
    for i, r in te.reset_index(drop=True).iterrows():
        f = finish[i]
        rows.append([r.driver_name, r.team_name, int(r.grid), round(float(s[i]), 3), round(float(summ["exp_pos"][i]), 1),
                     int(rank[i]), round(float(summ["p_win"][i]), 3), round(float(summ["p_top10"][i]), 3),
                     round(float(summ["p_dnf"][i]), 3), None if f is None or (isinstance(f, float) and np.isnan(f)) else int(f)])
    return rows


def run(budget_s=160):
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d = build()
    races = d[(d.year >= 2011) & (d.year < YEAR)].drop_duplicates("raceId").sort_values(["date", "raceId"])
    todo = [r for _, r in races.iterrows() if not (CACHE / f"{int(r.raceId)}.json").exists()]
    for r in todo:
        if time.time() - t0 > budget_s:
            print(f"paused: {len(todo)} races were left at start; run again", flush=True)
            return
        past = d[d.date < r.date]
        now = d[d.raceId == r.raceId]
        te, sim, A, crash = predict(past, now, seed=int(r.raceId))
        sm = summarise_sims(sim)
        sc = score(te, sim)
        row = {"id": int(r.raceId), "year": int(r.year), "round": int(r["round"]), "date": str(r.date.date()),
               "gp": r.gp_name, "circuit": r.circuitId, "training_races": int(past.raceId.nunique()),
               "A": round(100 * sc["duels_A"], 1), "B": round(100 * sc["duels_B"], 1), "pairs": sc["pairs"],
               "right_A": sc["right_A"], "right_B": sc["right_B"], "scorers_A": sc["scorers_A"], "scorers_B": sc["scorers_B"],
               "drivers": drivers_table(te, A.strength(te), {k: sm[k].to_numpy() for k in sm.columns},
                                        list(te.finish_pos.to_numpy()))}
        json.dump(row, open(CACHE / f"{int(r.raceId)}.json", "w"), ensure_ascii=False)
    print("all races done", flush=True)


def races_2026(d):
    mc = {r["round"]: r for r in json.load(open(ROOT / "results" / "montecarlo" / "summary.json"))["races"]}
    out = []
    for rnd, m in sorted(mc.items()):
        past = d[(d.year < YEAR) | ((d.year == YEAR) & (d["round"] < rnd))]
        now = d[(d.year == YEAR) & (d["round"] == rnd)]
        prof = CarProfile().fit(past)
        te = prof.transform(now).sort_values("grid").reset_index(drop=True)
        A = OrderModel().fit(prof.transform(past), FEAT_A)
        s = A.strength(te)
        f = next((ROOT / "results" / "montecarlo").glob(f"2026_r{rnd:02d}_*.csv"))
        C = pd.read_csv(f).set_index("driver")
        summ = {"exp_pos": C.loc[te.driver_name, "A_exp_pos"].to_numpy(), "p_win": C.loc[te.driver_name, "A_p_win"].to_numpy(),
                "p_top10": C.loc[te.driver_name, "A_p_top10"].to_numpy(), "p_dnf": C.loc[te.driver_name, "A_p_dnf"].to_numpy()}
        out.append({"id": int(now.raceId.iloc[0]), "year": YEAR, "round": rnd, "date": str(now.date.iloc[0].date()),
                    "gp": now.gp_name.iloc[0], "circuit": m["circuit"], "training_races": m["training_races"],
                    "A": round(100 * m["duels_A"], 1), "B": round(100 * m["duels_B"], 1), "pairs": m["pairs"],
                    "right_A": round(m["duels_A"] * m["pairs"]), "right_B": round(m["duels_B"] * m["pairs"]),
                    "scorers_A": m["scorers_A"], "scorers_B": m["scorers_B"],
                    "drivers": drivers_table(te, s, summ, list(te.finish_pos.to_numpy()))})
    return out


def build_json():
    d = build()
    races = [json.load(open(f)) for f in sorted(CACHE.glob("*.json"))] + races_2026(d)
    T = pd.read_csv(ROOT / "results" / "grid_dependence" / "tracks.csv").set_index("circuitId")
    season = json.load(open(ROOT / "app" / "season.json"))
    cal = {c["circuit"]: c for c in season["calendar"]}
    tracks = {}
    for c in sorted({r["circuit"] for r in races}):
        if c in cal:                                   # 2026 calendar: the rule used for every 2026 Grand Prix
            k, m, why = cal[c]["class"], cal[c]["model"], cal[c]["why"]
        elif c in T.index:
            k = T.loc[c, "class"]
            m = "B" if k == "grid-dependent" else "A"
            why = (f"{k.capitalize()}: in {int(T.loc[c, 'races'])} races (2011-2025) model A was better "
                   f"{int(T.loc[c, 'A_better'])} times, the grid {int(T.loc[c, 'B_better'])} times.")
        else:
            k, m, why = "too few races", "A", "Fewer than three races here: model A by default."
        name = T.loc[c, "track"] if c in T.index else c.replace("-", " ").title()
        tracks[c] = {"name": name, "class": k, "model": m, "why": why}
    races.sort(key=lambda r: (r["year"], r["round"]))
    out = {"generated": time.strftime("%Y-%m-%d"), "columns": ["driver", "team", "grid", "score", "exp_pos", "rank_A",
                                                               "p_win", "p_top10", "p_retire", "finish"],
           "tracks": tracks, "races": races}
    (ROOT / "app" / "past.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"app/past.json: {len(races)} races, {(ROOT / 'app' / 'past.json').stat().st_size // 1024} KB")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "160"
    build_json() if arg == "build" else run(float(arg))
