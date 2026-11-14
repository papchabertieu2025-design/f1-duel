"""Writes app/season.json for the website: model A vs model B race by race in 2026, the class of every track,
and the model recommended for the next race (with its forecast once the Saturday file has been run).

Usage: python -m f1duel.site_season   (after montecarlo, choose and grid_dependence)
"""
import json
from datetime import datetime, timezone

import pandas as pd

from .data import ROOT

NAMES = {1: "Australia", 2: "China", 3: "Japan", 4: "Miami", 5: "Canada", 6: "Monaco", 7: "Barcelona", 8: "Austria",
         9: "Britain", 10: "Belgium", 11: "Hungary", 12: "Netherlands", 13: "Italy", 14: "Madrid", 15: "Azerbaijan",
         16: "Bahrain GP (Sepang)", 17: "Singapore", 18: "United States", 19: "Mexico", 20: "São Paulo",
         21: "Las Vegas", 22: "Qatar", 23: "Abu Dhabi"}
HARD = {"monaco", "zandvoort", "hungaroring", "marina-bay", "madring"}


def run():
    mc = json.load(open(ROOT / "results" / "montecarlo" / "summary.json"))
    o = mc["overall"]
    races = [{"round": r["round"], "name": NAMES.get(r["round"], r["grand_prix"]), "circuit": r["circuit"],
              "hard": r["circuit"] in HARD, "A": round(100 * r["duels_A"], 1), "B": round(100 * r["duels_B"], 1),
              "better": r["better"], "clear": bool(r["mcnemar_p"] < 0.05),
              "only_A": r["pairs_only_A_right"], "only_B": r["pairs_only_B_right"], "winner": r["winner"]}
             for r in mc["races"]]
    T = pd.read_csv(ROOT / "results" / "grid_dependence" / "tracks.csv")
    cls = {r.circuitId: {"class": r["class"], "gain": float(r.gain), "ci": [float(r.ci_low), float(r.ci_high)],
                         "races": int(r.races), "rho": float(r.rho)} for _, r in T.iterrows()}
    cal = pd.read_csv(ROOT / "data" / "raw" / "f1db" / "f1db-races.csv")
    cal = cal[cal.year == 2026].sort_values("round")
    done = {r["round"] for r in mc["races"]}
    calendar = []
    res26 = {r["circuit"]: r for r in races}
    for _, r in cal.iterrows():
        c = cls.get(r.circuitId)
        row = T[T.circuitId == r.circuitId]
        this = res26.get(r.circuitId)
        # Rule: grid-dependent track -> B; track without enough history -> the model that won its only race; else A.
        if c and c["class"] == "grid-dependent":
            model, why = "B", f"Grid-dependent: in {c['races']} past races the grid beat the model {int(row.B_better.iloc[0])} times."
        elif not c and this:
            model = this["better"] if this["better"] in ("A", "B") else "A"
            why = f"New track: its only race (2026) was won by model {this['better']} ({this['A']}% vs {this['B']}%)."
        elif not c:
            model, why = "A", "No past race here: the full model by default."
        else:
            model = "A"
            why = (f"{c['class'].capitalize()}: in {c['races']} past races the full model beat the grid "
                   f"{int(row.A_better.iloc[0])} times (grid: {int(row.B_better.iloc[0])}).")
        calendar.append({"round": int(r["round"]), "name": NAMES.get(int(r["round"]), r.grandPrixId), "circuit": r.circuitId,
                         "date": r.date, "done": int(r["round"]) in done, "hard": r.circuitId in HARD,
                         "class": c["class"] if c else "new track", "gain": c["gain"] if c else None,
                         "past_races": c["races"] if c else 0,
                         "past_A": int(row.A_better.iloc[0]) if c else 0, "past_B": int(row.B_better.iloc[0]) if c else 0,
                         "result_2026": {k: this[k] for k in ("A", "B", "better", "winner")} if this else None,
                         "model": model, "why": why})
    nxt = next((c for c in calendar if not c["done"]), None)
    forecast = None
    if nxt:
        f = ROOT / "forecasts" / f"2026_r{nxt['round']:02d}_choice_montecarlo.csv"
        if f.exists() and not f.read_text().startswith("#"):
            F = pd.read_csv(f)
            forecast = F[["driver", "team", "grid", "A_p_win", "A_p_top10", "A_p_retire", "predicted_order"]].to_dict("records")
    gaps = sorted((r["A"] - r["B"], r["name"]) for r in races)
    ml = None
    fml = ROOT / "results" / "ml" / "summary.json"
    if fml.exists():                       # the benchmark of the eleven course models (f1duel.ml_benchmark)
        ml = [{"model": m["model"], "family": m["family"], "t1": round(100 * m["2020-2025"]["pair_acc"], 1),
               "t2": round(100 * m["2026"]["pair_acc"], 1)} for m in json.load(open(fml))]
    out = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
           "season": {"races": o["races"], "A": round(100 * o["duels_A"], 1), "B": round(100 * o["duels_B"], 1),
                      "gain": round(o["gain_A_minus_B_pts"][0], 1), "A_better": o["A_better"], "B_better": o["B_better"],
                      "ties": o["ties"], "A_clear": o["A_clearly_better"], "B_clear": o["B_clearly_better"],
                      "scorers_A": o["scorers_A"], "scorers_B": o["scorers_B"],
                      "best_A": {"gain": round(gaps[-1][0], 1), "name": gaps[-1][1]},
                      "best_B": {"gain": round(-gaps[0][0], 1), "name": gaps[0][1]}},
           "races": races, "calendar": calendar, "next": nxt, "forecast": forecast, "ml": ml}
    (ROOT / "app" / "season.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"app/season.json written: {len(races)} races, next = {nxt['name'] if nxt else None}")


if __name__ == "__main__":
    run()
