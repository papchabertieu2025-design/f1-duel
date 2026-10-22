"""Part I - The data.

One source: the official Formula 1 race results, as stored in the open F1DB database
(github.com/f1db/f1db, snapshot in data/raw/ with its SHA-256 checksum).

build() returns one row per driver per Grand Prix (2014 onwards) with:
  * what happened in the race  -> finish_pos, top10, points, dnf_mechanical
  * what was known BEFORE the race -> features (past form, team strength, grid, qualifying gap)

Rule used everywhere: a feature for race r only uses races strictly before r
(except grid and qualifying, which are known on Saturday, before the race).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "f1db"
FIRST_YEAR = 2014          # start of the turbo-hybrid era: same points system ever since
DNF_POSITION = 20          # a driver who does not finish counts as 20th in average-position features
INCIDENT = ("collision", "accident", "spun", "crash", "damage", "contact", "puncture",
            "debris", "withdrew", "illness", "injury")

# Team statistics summarised by PCA in Part III (rolling, previous 5 races, both cars)
TEAM_STATS = ["team_points", "team_top10", "team_podium", "team_avg_finish",
              "team_avg_grid", "team_quali_gap", "team_mech_dnf"]


def _read(name):
    return pd.read_csv(RAW / f"f1db-{name}.csv", low_memory=False)


def _lineage():
    """Map (constructorId, year) -> one name per team lineage (e.g. Toro Rosso -> Racing Bulls)."""
    chron = _read("constructors-chronology")
    out = {}
    for _, chain in chron.groupby("parentConstructorId"):
        chain = chain.sort_values("positionDisplayOrder")
        last = chain.iloc[-1]["constructorId"]
        for _, r in chain.iterrows():
            y1 = int(r.yearTo) if pd.notna(r.yearTo) else 2030
            for y in range(max(int(r.yearFrom), FIRST_YEAR), y1 + 1):
                out.setdefault((r.constructorId, y), last)
    return out


def _prev_mean(s, groups, window):
    """Average of the previous `window` values within each group (the current race is excluded)."""
    return s.groupby(groups).transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())


def upcoming_entries(year, rnd):
    """Drivers expected at race (year, rnd): the line-up of the latest race run, plus the
    weekend data if a file data/upcoming/{year}_r{rnd:02d}.csv exists (filled in on Saturday night):
    columns driverId, grid, quali_time_s, practice_time_s (any can be left empty)."""
    res = _read("races-race-results")
    last = res[res.year == res.year.max()]
    last = last[last["round"] == last["round"].max()][["driverId", "constructorId"]]
    path = ROOT / "data" / "upcoming" / f"{year}_r{rnd:02d}.csv"
    if path.exists():
        extra = pd.read_csv(path)
        last = extra.merge(last, on="driverId", how="left") if "constructorId" not in extra else extra
    return last


def build(upcoming=None):
    """upcoming = (year, round) of a race not run yet -> its rows are added with empty results."""
    races = _read("races")
    races = races[races.year >= FIRST_YEAR].copy()
    races["date"] = pd.to_datetime(races.date)
    gp = _read("grands-prix").set_index("id")
    races["gp_name"] = races.grandPrixId.map(gp["name"])
    circ = _read("circuits").set_index("id")
    races["circuit_type"] = races.circuitId.map(circ["type"])
    calendar = races.groupby("year")["round"].max()                   # published calendar length

    res = _read("races-race-results")
    res = res[(res.year >= FIRST_YEAR) & res.gridPositionText.notna()
              & ~res.positionText.isin(["DNS", "DNP", "DNQ", "DNPQ", "EX"])].copy()
    if upcoming is not None:
        y, r = upcoming
        race = races[(races.year == y) & (races["round"] == r)].iloc[0]
        ent = upcoming_entries(y, r)
        add = pd.DataFrame({"raceId": race["id"], "year": y, "round": r, "driverId": ent.driverId,
                            "constructorId": ent.constructorId, "gridPositionText": "?"})
        for col in ("grid", "quali_time_s", "practice_time_s"):
            add["up_" + col] = ent[col].values if col in ent else np.nan
        res = pd.concat([res, add], ignore_index=True)
    df = res.merge(races[["id", "date", "gp_name", "circuitId", "circuit_type"]],
                   left_on="raceId", right_on="id").drop(columns="id")
    df["upcoming"] = df.positionText.isna().astype(int) if upcoming is not None else 0

    # --- what happened -------------------------------------------------------
    df["finish_pos"] = pd.to_numeric(df.positionNumber, errors="coerce")      # NaN = not classified
    df["top10"] = (df.finish_pos <= 10).astype(int)          # (rows of an upcoming race get 0, never used)
    df["podium"] = (df.finish_pos <= 3).astype(int)
    df["win"] = (df.finish_pos == 1).astype(int)
    df["points_scored"] = pd.to_numeric(df.points, errors="coerce").fillna(0)
    reason = df.reasonRetired.fillna("").str.lower()
    retired = df.positionText.isin(["DNF", "NC"])
    df["dnf_mechanical"] = (retired & ~reason.str.contains("|".join(INCIDENT))).astype(int)

    # --- known on Saturday (after qualifying) ---------------------------------
    df["n_starters"] = df.groupby("raceId").driverId.transform("size")
    df["grid"] = pd.to_numeric(df.gridPositionNumber, errors="coerce")
    df["grid"] = df.grid.fillna(df.n_starters)                                 # pit-lane start = back
    if upcoming is not None:
        df.loc[df.upcoming == 1, "grid"] = df.loc[df.upcoming == 1, "up_grid"]          # NaN until Saturday
    q = _read("races-qualifying-results")
    q = q[q.year >= FIRST_YEAR].copy()
    q["best"] = q[["q1Millis", "q2Millis", "q3Millis", "timeMillis"]].min(axis=1)
    q["quali_gap"] = 100 * (q.best / q.groupby("raceId").best.transform("min") - 1)
    df = df.merge(q[["raceId", "driverId", "quali_gap"]], on=["raceId", "driverId"], how="left")
    if upcoming is not None:
        u = df.upcoming == 1
        df.loc[u, "quali_gap"] = 100 * (df.loc[u, "up_quali_time_s"] / df.loc[u, "up_quali_time_s"].min() - 1)
    df["quali_gap"] = df.quali_gap.fillna(df.groupby("raceId").quali_gap.transform("max")).clip(upper=5)

    # best practice lap (FP1-3), % gap to the fastest car of the weekend (known before qualifying)
    fps = pd.concat([_read(f"races-free-practice-{n}-results")[["raceId", "driverId", "timeMillis"]] for n in (1, 2, 3)])
    fp = fps.groupby(["raceId", "driverId"]).timeMillis.min().rename("fp_best").reset_index()
    fp["practice_gap"] = 100 * (fp.fp_best / fp.groupby("raceId").fp_best.transform("min") - 1)
    df = df.merge(fp[["raceId", "driverId", "practice_gap"]], on=["raceId", "driverId"], how="left")
    if upcoming is not None:
        u = df.upcoming == 1
        df.loc[u, "practice_gap"] = 100 * (df.loc[u, "up_practice_time_s"] / df.loc[u, "up_practice_time_s"].min() - 1)
    df["practice_gap"] = df.practice_gap.fillna(df.groupby("raceId").practice_gap.transform("max")).fillna(3).clip(upper=5)

    # --- names -----------------------------------------------------------------
    lin = _lineage()
    df["team"] = [lin.get((c, y), c) for c, y in zip(df.constructorId, df.year)]
    cons = _read("constructors").set_index("id")["name"]
    df["team_name"] = df.constructorId.map(cons)
    drv = _read("drivers").set_index("id")
    df["driver_name"] = df.driverId.map(drv["name"])
    df["driver_code"] = df.driverId.map(drv["abbreviation"])

    # --- known before the weekend (history only) --------------------------------
    df = df.sort_values(["date", "raceId", "grid"]).reset_index(drop=True)
    order = {r: i for i, r in enumerate(df.drop_duplicates("raceId").raceId)}
    df["race_idx"] = df.raceId.map(order)
    fin = df.finish_pos.fillna(DNF_POSITION)
    df["drv_avg_finish"] = _prev_mean(fin, df.driverId, 5)
    df["drv_top10"] = _prev_mean(df.top10, df.driverId, 5)

    team_race = (df.assign(fin=fin).groupby(["team", "race_idx"])
                 .agg(points=("points_scored", "sum"), top10=("top10", "mean"), podium=("podium", "mean"),
                      avg_finish=("fin", "mean"), avg_grid=("grid", "mean"),
                      quali_gap=("quali_gap", "min"), mech=("dnf_mechanical", "mean"))
                 .reset_index().sort_values("race_idx"))
    g = team_race.team
    for col, name, w in [("points", "team_points", 5), ("top10", "team_top10", 5), ("podium", "team_podium", 5),
                         ("avg_finish", "team_avg_finish", 5), ("avg_grid", "team_avg_grid", 5),
                         ("quali_gap", "team_quali_gap", 5), ("mech", "team_mech_dnf", 10)]:
        team_race[name] = _prev_mean(team_race[col], g, w)
    df = df.merge(team_race[["team", "race_idx"] + TEAM_STATS], on=["team", "race_idx"], how="left")

    # newcomers (no history): neutral values from the 2014-2019 seasons, flagged
    df["new_entry"] = df.drv_avg_finish.isna().astype(int)
    ref = df[df.year <= 2019]
    for col in ["drv_avg_finish", "drv_top10"] + TEAM_STATS:
        df[col] = df[col].fillna(ref[col].median())
    df["season_progress"] = df["round"] / df.year.map(calendar)

    keep = ["raceId", "year", "round", "date", "gp_name", "circuitId", "circuit_type", "driverId",
            "driver_name", "driver_code", "team", "team_name", "n_starters", "grid", "quali_gap", "practice_gap",
            "drv_avg_finish", "drv_top10", "new_entry", "season_progress", *TEAM_STATS,
            "finish_pos", "top10", "podium", "win", "points_scored", "dnf_mechanical", "upcoming"]
    return df[keep]


if __name__ == "__main__":
    out = build()
    path = ROOT / "data" / "processed" / "starts.csv"
    out.to_csv(path, index=False)
    print(f"{len(out)} driver starts, {out.raceId.nunique()} races, {out.year.min()}-{out.year.max()} -> {path}")
