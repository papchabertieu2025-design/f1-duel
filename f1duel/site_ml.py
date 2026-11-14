"""Writes app/ml.json: everything the website needs to explain the machine-learning part of the project step by step
(the grid as a forecast, PCA, k-means, model A's parameters and worked example, validation curves of the ten check
models, summary table, confusion matrices and ROC curves, feature importance, decision tree, grid dependence of tracks).

Every number is read from results/ (no number typed by hand).
Usage: python -m f1duel.site_ml        (after ml_benchmark, ml_importance, report_params, grid_dependence, choose)
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

try:
    from .data import ROOT
except ImportError:                                   # run as a plain script: python site_ml.py <repo>
    ROOT = Path(sys.argv[1])
R = ROOT / "results"
ML = R / "ml"
SEED = 2026
slug = lambda n: n.lower().replace(" ", "_").replace("(", "").replace(")", "")
LAB12 = {"grid": "Grid position", "quali_gap": "Qualifying gap", "practice_gap": "Practice gap", "team_points": "Team points",
         "team_top10": "Team top-10 rate", "team_podium": "Team podium rate", "team_avg_finish": "Team avg. finish",
         "team_avg_grid": "Team avg. grid", "team_quali_gap": "Team quali gap", "team_mech_dnf": "Team mech. failures",
         "drv_avg_finish": "Driver avg. finish", "drv_top10": "Driver top-10 rate"}
PRETTY = {"monaco": "Monaco", "yeongam": "Yeongam", "nurburgring": "Nürburgring", "shanghai": "Shanghai", "silverstone": "Silverstone",
          "melbourne": "Melbourne", "marina-bay": "Singapore", "mexico-city": "Mexico City", "baku": "Baku", "spielberg": "Austria",
          "montreal": "Montréal", "lusail": "Qatar", "imola": "Imola", "interlagos": "São Paulo", "yas-marina": "Abu Dhabi",
          "buddh": "India", "paul-ricard": "Paul Ricard", "suzuka": "Suzuka", "sepang": "Sepang", "bahrain": "Bahrain",
          "hockenheimring": "Hockenheim", "jeddah": "Jeddah", "catalunya": "Barcelona", "miami": "Miami", "monza": "Monza",
          "hungaroring": "Hungary", "spa-francorchamps": "Spa", "austin": "Austin", "zandvoort": "Zandvoort", "sirius": "Sochi",
          "istanbul": "Istanbul", "las-vegas": "Las Vegas"}
r1 = lambda v: round(100 * float(v), 1)


def model_file(name):
    return json.load(open(ML / f"model_{slug(name)}.json"))


def run():
    S = {r["model"]: r for r in json.load(open(ML / "summary.json"))}
    E = json.load(open(ML / "eda.json")); X = json.load(open(ML / "extras.json")); I = json.load(open(ML / "importance.json"))
    PAR = json.load(open(ML / "parameters.json"))
    out = {}

    # ---------------------------------------------------------------- 1. data and the grid as a forecast
    out["data"] = {"starts": E["n_starts"], "races": E["n_races"], "drivers": E["n_drivers"], "retired_pct": r1(E["retired_share"]),
                   "by_grid": [{"grid": int(g["grid"]), "win": r1(g["win"]), "top10": r1(g["top10"]), "finish": round(g["mean_finish"], 2)}
                               for g in E["by_grid"]],
                   "lr": {k: round(float(v), 2) if k != "n" else int(v) for k, v in E["lr"].items()},
                   "corr_grid_quali": round(E["corr"]["grid"]["quali_gap"], 2)}

    # ---------------------------------------------------------------- 2. PCA and k-means
    P = X["pca"]
    out["pca"] = {"n": P["n"], "features": [LAB12[f].replace("Team ", "").capitalize() for f in P["features"]],
                  "explained": [r1(v) for v in P["explained"]],
                  "pc1": [round(-v, 2) for v in P["loadings_pc1"]], "pc2": [round(v, 2) for v in P["loadings_pc2"]]}
    A1 = PAR["model_A_before_sao_paulo"]
    KP = pd.read_csv(ML / "kmeans_before_sao_paulo.csv")
    pts = KP.groupby(["team_points", "team_mech_dnf", "family"]).size().reset_index(name="n")       # unique dots only
    out["kmeans"] = {"curve": [{"k": k["k"], "inertia": round(k["inertia"], 1), "silhouette": round(k["silhouette"], 3)} for k in X["kmeans"]],
                     "points": [[round(float(r.team_points), 1), r1(r.team_mech_dnf), r.family[0]] for r in pts.itertuples()],
                     "families": [{"family": f["family"], "points": round(f["team_points"], 1), "fail_pct": round(100 * f["mech_failure_rate"]),
                                   "top10_pct": round(100 * f["top10_rate"]), "team_races": f["team_races"]} for f in A1["families"]]}
    try:                                              # PCA scores of every team-race, coloured by car family (needs the data)
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from .data import TEAM_STATS
        from .ml_benchmark import load
        from .model import CarProfile
        d = load(); hist = d[d.year <= 2025]
        teams = hist.drop_duplicates(["raceId", "team"]).reset_index(drop=True)
        Z = PCA(2).fit_transform(StandardScaler().fit_transform(teams[TEAM_STATS]))
        prof = CarProfile().fit(d[(d.year < 2026) | ((d.year == 2026) & (d["round"] < 20))])   # the families used for São Paulo 2026
        tier = prof.transform(teams).tier.map({0: "top", 1: "midfield", 2: "unreliable"}).to_numpy()
        sgn = -1 if np.corrcoef(Z[:, 0], teams["team_points"])[0, 1] < 0 else 1       # PC1 pointing towards stronger cars
        take = np.random.default_rng(SEED).choice(len(Z), size=min(1200, len(Z)), replace=False)
        out["pca"]["scores"] = [[round(float(sgn * Z[i, 0]), 2), round(float(Z[i, 1]), 2), str(tier[i])[0]] for i in take]
    except Exception as e:                            # the page then simply hides that chart
        print("PCA scores skipped:", type(e).__name__, e)

    # ---------------------------------------------------------------- 3. model A: parameters and worked example
    keys = [("grid", "grid", "Grid position"), ("quali_gap", "quali_gap", "Qualifying gap (%)"), ("practice_gap", "practice_gap", "Practice gap (%)"),
            ("car_perf", "team_points", "Team points per race"), ("tier_top", "tier_top", "Top car (1 = yes)"),
            ("tier_unrel", "tier_unrel", "Unreliable car (1 = yes)")]
    out["modelA"] = {"races": A1["races"], "lambda": A1["l2"], "crash_pct": r1(A1["crash_rate"]),
                     "params": [{"key": k, "label": lab, "theta": round(A1["theta"][k], 2), "mean": round(A1["mean_sd_original_units"][u][0], 2),
                                 "sd": round(A1["mean_sd_original_units"][u][1], 2)} for k, u, lab in keys],
                     "lambda_cv": [{"lambda": c["params"]["l2"], "cv": r1(c["cv_pair_acc"])} for c in model_file("Softmax ranking (model A)")["cv"]]}
    sp = PAR["sao_paulo_2026"]
    rank = {r["driver"]: k + 1 for k, r in enumerate(sorted(sp["drivers"], key=lambda r: r["exp_pos"]))}
    out["example"] = {"race": "São Paulo 2026", "drivers": [
        {"driver": r["driver"], "team": r["team"], "family": r["family"], "grid": r["grid"], "quali_gap": round(r["quali_gap"], 2),
         "practice_gap": round(r["practice_gap"], 2), "team_points": round(r["team_points"], 1),
         "contrib": {k: round(v, 2) for k, v in r["contrib"].items()}, "score": round(r["score"], 2), "p_retire": r1(r["p_retire"]),
         "p_win": r1(r["p_win"]), "p_top10": r1(r["p_top10"]), "rank": rank[r["driver"]], "finish": r["finish"]}
        for r in sorted(sp["drivers"], key=lambda r: r["grid"])]}

    # ---------------------------------------------------------------- 4. validation curves of the check models (labs: tune and compare)
    def cv_of(name):
        return model_file(name)["cv"], model_file(name)["best"]
    T = {}
    cv, b = cv_of("Decision tree")
    T["tree"] = {"x": [c["params"]["depth"] for c in cv], "train": [r1(c["train_pair_acc"]) for c in cv], "cv": [r1(c["cv_pair_acc"]) for c in cv], "best": b["depth"]}
    cv, b = cv_of("KNN"); cv2 = [c for c in cv if c["params"]["p"] == 2]
    T["knn"] = {"x": [c["params"]["k"] for c in cv2], "train": [r1(c["train_pair_acc"]) for c in cv2], "cv": [r1(c["cv_pair_acc"]) for c in cv2], "best": b["k"]}
    bc = PAR["boosting_curve_used"]
    T["boosting"] = {"x": [c["n"] for c in bc], "train": [r1(c["train"]) for c in bc], "cv": [r1(c["cv"]) for c in bc], "best": 100}
    cv, b = cv_of("Ridge"); cvl, bl = cv_of("Lasso")
    T["ridge"] = {"x": [c["params"]["alpha"] for c in cv], "cv": [r1(c["cv_pair_acc"]) for c in cv], "best": b["alpha"]}
    T["lasso"] = {"x": [c["params"]["alpha"] for c in cvl], "cv": [r1(c["cv_pair_acc"]) for c in cvl], "best": bl["alpha"]}
    cv, b = cv_of("Kernel ridge (RBF)")
    T["kernel"] = {"cells": [{"lambda": c["params"]["alpha"], "gamma": c["params"]["gamma"], "cv": r1(c["cv_pair_acc"])} for c in cv], "best": b}
    cv, b = cv_of("Random forest")
    T["forest"] = {"cells": [{"leaf": c["params"]["leaf"], "mf": c["params"]["mf"], "cv": r1(c["cv_pair_acc"])} for c in cv], "best": b}
    out["tuning"] = T

    # ---------------------------------------------------------------- 5. summary table
    BF = PAR["benchmark_fitted_before_2026"]["fitted"]
    INFO = [("Grid rule", "Model B: grid rule", "–", "finishing order = starting order", "0"),
            ("Softmax ranking (model A)", "Model A: softmax ranking", "S7, S10", "6 features, penalty λ = 1, 10,000 simulations", "6 weights"),
            ("Linear regression", "Linear regression", "S2–3", "least squares", "13 numbers"),
            ("Ridge", "Ridge", "S4", "λ = 3.2", "13 numbers"),
            ("Lasso", "Lasso", "S4", "λ = 0.01", f"13 numbers ({BF['Lasso']['nonzero'] + 1} not zero)"),
            ("Logistic regression", "Logistic regression", "S6–7", "C = 1/λ = 0.03", "13 numbers"),
            ("KNN", "k-nearest neighbours", "S14–15", "k = 200, Euclidean distance", "nothing: it stores the starts"),
            ("Decision tree", "Decision tree", "S16", "depth 4, leaves ≥ 20 starts", f"{BF['Decision tree']['learned']} nodes"),
            ("Random forest", "Random forest", "S17", "300 trees, leaves ≥ 50, ⅓ of features per split", f"{BF['Random forest']['learned']:,} nodes"),
            ("Gradient boosting", "Gradient boosting", "S17", "100 trees of depth 4, learning rate 0.03", f"{BF['Gradient boosting']['learned']:,} nodes"),
            ("Kernel ridge (RBF)", "Kernel ridge (RBF)", "S19–21", "λ = 1, γ = 0.03", f"{BF['Kernel ridge (RBF)']['learned']:,} weights"),
            ("Neural network (MLP)", "Neural network (MLP)", "S22–26", "64 + 32 ReLU units, early stopping", f"{BF['Neural network (MLP)']['learned']:,} weights")]

    def cvbest(name):
        M = model_file(name)
        return S[name].get("cv_pair_acc") if not M["best"] else [c for c in M["cv"] if c["params"] == M["best"]][0]["cv_pair_acc"]
    out["models"] = [{"key": k, "name": nm, "session": ses, "setting": setting, "learned": learned, "cv": r1(cvbest(k)),
                      "t1": r1(S[k]["2020-2025"]["pair_acc"]), "t2": r1(S[k]["2026"]["pair_acc"]),
                      "scorers1": round(S[k]["2020-2025"]["scorers"], 2), "auc": round(S[k]["2020-2025"]["auc_top10"], 3),
                      "r2": round(S[k]["2020-2025"]["r2"], 2) if "r2" in S[k]["2020-2025"] else None,
                      "races_won": None if k == "Grid rule" else S[k]["2020-2025"]["races_beating_grid"]}
                     for k, nm, ses, setting, learned in INFO]
    out["test1_races"] = S["Grid rule"]["2020-2025"]["races"]

    # ---------------------------------------------------------------- 6. classification view: confusion matrices and ROC curves
    cls = {}
    for key, fn, lab in (("grid", "pred_grid_rule.csv", "Model B: top ten on the grid"), ("logistic", "pred_logistic_regression.csv", "Logistic regression"),
                         ("A", "pred_softmax_ranking_model_a.csv", "Model A")):
        p = pd.read_csv(ML / fn); p = p[p.test == "2020-2025"]
        y = p.top10.to_numpy()
        yh = (p.grid <= 10).astype(int).to_numpy() if key == "grid" else (p.prob.to_numpy() >= 0.5).astype(int)
        tp, fp = int(((yh == 1) & (y == 1)).sum()), int(((yh == 1) & (y == 0)).sum())
        fn_, tn = int(((yh == 0) & (y == 1)).sum()), int(((yh == 0) & (y == 0)).sum())
        fpr, tpr, _ = roc_curve(y, -p.score)
        idx = np.unique(np.round(np.linspace(0, len(fpr) - 1, 60)).astype(int))
        cls[key] = {"label": lab, "tp": tp, "fp": fp, "fn": fn_, "tn": tn, "accuracy": r1((tp + tn) / len(y)), "precision": r1(tp / (tp + fp)),
                    "recall": r1(tp / (tp + fn_)), "f1": r1(2 * tp / (2 * tp + fp + fn_)), "auc": round(float(roc_auc_score(y, -p.score)), 3),
                    "roc": [[round(float(fpr[i]), 3), round(float(tpr[i]), 3)] for i in idx], "n": int(len(y))}
    out["classification"] = cls

    # ---------------------------------------------------------------- 7. what matters: permutation importance and the tree
    imp = sorted(I["rf_permutation_pts"].items(), key=lambda kv: -kv[1])
    out["importance"] = [{"feature": LAB12[k], "pp": round(v, 2)} for k, v in imp]
    try:
        tr = pickle.load(open(R / "forest" / "tree_depth3.pkl", "rb")); t = tr.tree_; names = list(tr.feature_names_in_)

        def node(i):
            if t.children_left[i] == -1:
                return {"leaf": True, "finish": round(float(t.value[i][0][0]), 1), "n": int(t.n_node_samples[i])}
            f = names[t.feature[i]]
            return {"leaf": False, "feature": LAB12.get(f, f), "key": f, "thr": round(float(t.threshold[i]), 2), "n": int(t.n_node_samples[i]),
                    "yes": node(t.children_left[i]), "no": node(t.children_right[i])}
        out["tree"] = node(0)
    except Exception as e:
        print("tree skipped:", type(e).__name__, e)

    # ---------------------------------------------------------------- 8. 2026: top cars out of position
    MC = json.load(open(R / "montecarlo" / "summary.json"))
    rows = []
    for r in MC["races"]:
        t_ = pd.read_csv(sorted((R / "montecarlo").glob(f"2026_r{r['round']:02d}_*.csv"))[0])
        rows.append({"round": r["round"], "gp": r["grand_prix"], "gain": round(100 * (r["duels_A"] - r["duels_B"]), 1),
                     "top_back": int(((t_.car_family == "top") & (t_.grid > 10)).sum())})
    G = pd.DataFrame(rows); grp = G.top_back.clip(upper=2)
    out["top_back"] = {"races": rows, "groups": [{"label": lab, "races": int((grp == k).sum()), "gain": round(float(G.gain[grp == k].mean()), 1)}
                                                 for k, lab in enumerate(["none", "one", "two or more"])]}

    # ---------------------------------------------------------------- 9. grid dependence of tracks
    TR = pd.read_csv(R / "grid_dependence" / "tracks.csv")
    GD = json.load(open(R / "grid_dependence" / "summary.json")); rg = GD["regression_tracks"]
    C = json.load(open(R / "choose" / "summary.json"))
    b0 = float((TR.gain - rg["slope"] * TR.rho).mean())
    out["tracks"] = {"list": [{"name": PRETTY.get(r.circuitId, r.track), "races": int(r.races), "rho": round(float(r.rho), 3), "gain": round(float(r.gain), 1),
                               "A": int(r.A_better), "B": int(r.B_better), "class": r["class"], "cal26": bool(r.on_2026_calendar)} for _, r in TR.iterrows()],
                     "classes": GD["classes"], "slope": round(rg["slope"], 1), "intercept": round(b0, 1), "r2": round(rg["r2"], 2),
                     "replay": {"races": C["overall_2011_2025"]["races"], "A": r1(C["overall_2011_2025"]["duels_A"]), "B": r1(C["overall_2011_2025"]["duels_B"]),
                                "A_better": C["overall_2011_2025"]["races_A_better"], "B_better": C["overall_2011_2025"]["races_B_better"]},
                     "rule_check": {"rule": r1(C["checks"]["past"]["chosen"]), "A": r1(C["checks"]["past"]["always_A"]), "B": r1(C["checks"]["past"]["always_B"])}}

    f = ROOT / "app" / "ml.json"
    f.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print("wrote", f, f"{f.stat().st_size / 1024:.0f} KB", "| sections:", ", ".join(out))


if __name__ == "__main__":
    run()
