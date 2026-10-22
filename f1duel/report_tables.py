"""LaTeX tables for the reports, written from results/summary.json (never typed by hand).

Usage: python -m f1duel.report_tables   -> reports/tables/*.tex  (\\input them in Overleaf)
"""
import json

from .data import ROOT

OUT = ROOT / "reports" / "tables"


def pct(x):
    return f"{100 * x:.1f}\\%"


def ci(lo, hi, scale=1, unit=""):
    return f"{scale * lo:+.2f}{unit} to {scale * hi:+.2f}{unit}" if scale == 1 else f"{scale * lo:+.1f} to {scale * hi:+.1f}"


def features():
    rows = [("Weekend pace", "grid position after penalties and $\\log(\\text{grid})$; gap to pole lap (\\%); gap to fastest practice lap (\\%)", "Grid, qualifying, practice"),
            ("Past results", "car performance and reliability (PCA of 7 team statistics, last 5 races); car family (k-means, $k=3$)", "Race results"),
            ("", "driver: average finish and top-10 rate, last 5 starts", "Race results"),
            ("Power unit (new)", "engine's points per car (last 5 races); works team; changed engine supplier; team's fade at the end of last season", "Results, engines")]
    body = "\n".join(" & ".join(r) + " \\\\" for r in rows)
    return ("\\begin{tabularx}{\\linewidth}{@{}l>{\\raggedright\\arraybackslash}Xl@{}}\n\\hline\nSet & Predictors & F1DB source \\\\\n\\hline\n"
            + body + "\n\\hline\n\\end{tabularx}\n")


def results(S):
    a, b = S["2020-2025"], S["2026"]
    da, db = a["duel"], b["duel"]
    g = a["gain_vs_grid"]
    rows = [
        ("Duels: who finishes ahead", "Grid rule", pct(da["after"]["grid_rule_accuracy"]), pct(db["after"]["grid_rule_accuracy"]), "--"),
        ("", "Logistic (after quali)", pct(da["after"]["accuracy"]), pct(db["after"]["accuracy"]),
         f"{100 * (da['after']['accuracy'] - da['after']['grid_rule_accuracy']):+.1f} pts ({ci(*da['after']['gain_ci'], scale=100)})"),
        ("", "Logistic (before quali)", pct(da["before"]["accuracy"]), pct(db["before"]["accuracy"]),
         f"{100 * (da['before']['accuracy'] - da['before']['grid_rule_accuracy']):+.1f} pts ({ci(*da['before']['gain_ci'], scale=100)})"),
        ("Points: scorers named of 10", "Grid rule (top 10 on grid)", f"{a['scorers_named']['Grid rule']:.2f}", f"{b['scorers_named']['Grid rule']:.2f}", "--"),
        ("", "Logistic (after quali)", f"{a['scorers_named']['Logistic regression']:.2f}", f"{b['scorers_named']['Logistic regression']:.2f}",
         f"{g['Logistic regression'][0]:+.2f} ({ci(*g['Logistic regression'][1:])})"),
        ("", f"KNN ($k={S['k']}$)", f"{a['scorers_named']['KNN']:.2f}", f"{b['scorers_named']['KNN']:.2f}",
         f"{g['KNN'][0]:+.2f} ({ci(*g['KNN'][1:])})"),
        ("Winner: log-loss (lower is better)", "Win rate of each grid slot", f"{a['winner']['logloss_gridrate_baseline']:.2f}", f"{b['winner']['logloss_gridrate_baseline']:.2f}", "--"),
        ("", "Softmax (after quali)", f"{a['winner']['logloss']:.2f}", f"{b['winner']['logloss']:.2f}",
         f"favourite = pole in {100 * a['winner']['favourite_is_pole']:.0f}\\% of races"),
    ]
    body = "\n".join(" & ".join(r) + " \\\\" for r in rows)
    return ("\\begin{tabular}{lllll}\n\\hline\nQuestion & Method & 2020--2025 & 2026 & Gain vs grid, 2020--2025 (95\\% CI) \\\\\n"
            f" & & ({a['races']} races) & ({b['races']} races) & \\\\\n\\hline\n" + body + "\n\\hline\n\\end{tabular}\n")


def tracks(S):
    T = S["tracks"]
    rows = []
    for name in ["All races", "Most grid-dependent third of tracks", "Street circuits", "Monaco"]:
        r = T[name]
        rows.append(f"{name} & {r['races']} & {pct(r['duel_grid'])} & {pct(r['duel_model'])} & "
                    f"{r['scorers_grid']:.2f} & {r['scorers_model']:.2f} \\\\")
    return ("\\begin{tabular}{lrrrrr}\n\\hline\n & & \\multicolumn{2}{c}{Duels correct} & \\multicolumn{2}{c}{Scorers named} \\\\\n"
            "Races, 2020--2025 & $n$ & Grid & Model & Grid & Model \\\\\n\\hline\n" + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")


def reset_table(R):
    """H1/H2: cross-entropy (log-loss) early (rounds 1-5) and later in the two reset seasons."""
    show = ["Past results only", "Past results + power unit", "Grid position only", "Weekend pace only",
            "Full model (no power unit)", "Full model"]
    rows = []
    for n in show:
        cells = []
        for t in ("Reset 2014", "Reset 2026"):
            m = R[t]["models"][n]
            cells += [f"{m['logloss_early']:.3f}", f"{m['logloss_late']:.3f}"]
        rows.append(n + " & " + " & ".join(cells) + " \\\\")
    return ("\\begin{tabular}{lcccc}\n\\hline\n & \\multicolumn{2}{c}{2014 reset} & \\multicolumn{2}{c}{2026 reset} \\\\\n"
            "$P(y=1\\mid x)=\\sigma(\\theta^\\top x)$ with $x=$ & rounds 1--5 & 6--19 & rounds 1--5 & 6--17 \\\\\n\\hline\n" + "\n".join(rows)
            + "\n\\hline\n\\end{tabular}\n")


def grid_table(R):
    """The professor's baseline: how many of the 10 scorers each method names, against the grid rule."""
    show = ["Past results only", "Weekend pace only", "Full model", "KNN (full model)"]
    n, a, b = R["Normal seasons 2020-2025"], R["Reset 2014"], R["Reset 2026"]
    rows = [f"Grid rule (top 10 on the grid) & {n['grid_rule_named']:.2f} & -- & {a['grid_rule_named']:.2f} & {b['grid_rule_named']:.2f} \\\\"]
    for m in show:
        g = n["models"][m]["gain_vs_grid"]
        rows.append(f"{m.replace('KNN (full model)', 'KNN, full model')} & {n['models'][m]['named']:.2f} & {g[0]:+.2f} ({g[1]:+.2f} to {g[2]:+.2f}) & "
                    f"{a['models'][m]['named']:.2f} & {b['models'][m]['named']:.2f} \\\\")
    return ("\\begin{tabular}{lcccc}\n\\hline\n & \\multicolumn{2}{c}{2020--2025 (" + str(n["races"]) + " races)} & 2014 reset & 2026 reset \\\\\n"
            "Method & named & gain vs grid (95\\% CI) & (" + str(a["races"]) + " races) & (" + str(b["races"]) + " races) \\\\\n\\hline\n"
            + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")


def main():
    S = json.loads((ROOT / "results" / "summary.json").read_text())
    R = json.loads((ROOT / "results" / "reset_summary.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    for name, tex in [("features", features()), ("results", results(S)), ("tracks", tracks(S)),
                      ("reset", reset_table(R)), ("grid", grid_table(R))]:
        (OUT / f"{name}.tex").write_text(f"% generated by python -m f1duel.report_tables from results/summary.json\n{tex}")
        print("->", OUT / f"{name}.tex")


if __name__ == "__main__":
    main()
