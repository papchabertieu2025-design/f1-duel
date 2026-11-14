"""F1 Duel - Grid or model? Flask website (Technology with Impact: ML model deployed as an API + OpenAI).

Routes
  GET  /                the page
  GET  /api/season      2026: model A vs grid rule race by race, model used on every track, next race, 11-model check
  GET  /api/ml          the numbers behind every chart of the page (PCA, k-means, model A, validation curves, ...)
  GET  /api/past        list of the 330 replayable races (2011-2026) with both scores
  GET  /api/past/<id>   one race: model A's forecast (made with earlier races only), the grid, the real result
  POST /api/ask         {"question": "...", "race": id} -> answer grounded only in the project's numbers (OpenAI)
  GET  /api/race, POST /api/duel   (older forecast endpoints, kept for compatibility; they need model.json)

Data files written by the project code:  season.json (python -m f1duel.site_season)
                                         past.json   (python -m f1duel.site_past build)
                                         ml.json     (python -m f1duel.site_ml)
Set OPENAI_API_KEY in the PythonAnywhere WSGI file (never in Git).
"""
import json
import math
import os
from functools import lru_cache
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request

HERE = Path(__file__).parent
app = Flask(__name__)
REPORT_URL = "https://github.com/papchabertieu2025-design/f1-duel"


def _read(name):
    f = HERE / name
    return json.loads(f.read_text()) if f.exists() else None


def load_season():
    return _read("season.json")


@lru_cache(maxsize=1)
def load_past():
    P = _read("past.json")
    if not P:
        return None
    P["by_id"] = {r["id"]: r for r in P["races"]}
    return P


def past_index():
    P = load_past()
    if not P:
        return []
    return [{"id": r["id"], "year": r["year"], "round": r["round"], "gp": r["gp"], "circuit": r["circuit"],
             "A": r["A"], "B": r["B"], "pairs": r["pairs"], "right_A": r["right_A"], "right_B": r["right_B"]}
            for r in P["races"]]


def past_race(race_id):
    P = load_past()
    if not P or race_id not in P["by_id"]:
        return None
    r = dict(P["by_id"][race_id])
    r["track"] = P["tracks"].get(r["circuit"], {"name": r["circuit"], "class": "", "model": "A", "why": ""})
    r["columns"] = P["columns"]
    return r


@app.route("/")
def index():
    P = load_past()
    return render_template("index.html", season=load_season(), past=past_index(), ml=_read("ml.json"),
                           tracks=P["tracks"] if P else {}, report_url=REPORT_URL)


@app.route("/api/ml")
def api_ml():
    m = _read("ml.json")
    return (jsonify(m), 200) if m else (jsonify({"error": "ml.json missing: run python -m f1duel.site_ml"}), 404)


@app.route("/api/season")
def api_season():
    s = load_season()
    return (jsonify(s), 200) if s else (jsonify({"error": "season.json missing: run python -m f1duel.site_season"}), 404)


@app.route("/api/past")
def api_past():
    return jsonify(past_index())


@app.route("/api/past/<int:race_id>")
def api_past_race(race_id):
    r = past_race(race_id)
    if r is None:
        abort(404)
    return jsonify(r)


def race_summary(r):
    """A short, factual summary of one replayed race for the AI box."""
    cols = r["columns"]
    rows = [dict(zip(cols, d)) for d in r["drivers"]]
    top = sorted(rows, key=lambda d: d["rank_A"])[:5]
    real = sorted([d for d in rows if d["finish"]], key=lambda d: d["finish"])[:5]
    return {"race": f"{r['year']} {r['gp']} Grand Prix (round {r['round']})", "track": r["track"],
            "pairs_right_model_A_pct": r["A"], "pairs_right_grid_pct": r["B"],
            "points_scorers_found_model_A": r["scorers_A"], "points_scorers_found_grid": r["scorers_B"],
            "model_A_top5": [{"driver": d["driver"], "grid": d["grid"], "p_win": d["p_win"], "p_top10": d["p_top10"]} for d in top],
            "real_top5": [{"driver": d["driver"], "grid": d["grid"], "finish": d["finish"]} for d in real]}


@app.route("/api/ask", methods=["POST"])
def api_ask():
    body = request.get_json(force=True, silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Send a JSON object with a question."}), 400
    question = str(body.get("question") or "").strip()[:500]
    if not question:
        return jsonify({"error": "Type a question first."}), 400
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key.startswith("sk-"):                                # missing or placeholder key
        return jsonify({"error": "The AI box is off: add OPENAI_API_KEY to the server settings."}), 503
    context = {}
    season = load_season()
    if season:
        context["season_2026"] = {"summary": season["season"], "next_race": season["next"],
                                  "races": [{k: r[k] for k in ("name", "A", "B", "better", "winner")} for r in season["races"]],
                                  "model_per_grand_prix": [{k: c[k] for k in ("round", "name", "model", "class")}
                                                           for c in season["calendar"]],
                                  "eleven_models_check": season.get("ml")}
    ml = _read("ml.json")
    if ml:                                                       # the method, in numbers
        context["method"] = {"model_A_weights": ml["modelA"]["params"], "car_families_kmeans": ml["kmeans"]["families"],
                             "pca_variance_explained_pct": ml["pca"]["explained"][:2],
                             "random_forest_permutation_importance_pp": ml["importance"],
                             "tracks_2011_2025": {"classes": ml["tracks"]["classes"], "replay": ml["tracks"]["replay"],
                                                  "rule_check_2020_2025": ml["tracks"]["rule_check"]}}
    try:
        r = past_race(int(body.get("race")))
    except (TypeError, ValueError):
        r = None
    if r:
        context["selected_race"] = race_summary(r)
    from openai import OpenAI, OpenAIError                       # as in the GenAI tutorial
    # gzip only: brotli/zstd decoding crashes with the httpx version installed on PythonAnywhere
    client = OpenAI(api_key=key, default_headers={"Accept-Encoding": "gzip"})
    try:
        reply = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "system", "content":
                       "You are the race engineer of a student machine-learning project that compares two forecasts of "
                       "Formula 1 races: model A (a softmax ranking model trained on earlier races, simulated 10,000 times) "
                       "and model B (the starting grid). 'Pairs in the right order' is the share of driver pairs that a "
                       "forecast orders as they finished. Answer in the user's language, in at most 120 words, for a general "
                       "audience. Use ONLY the numbers in this JSON; never invent statistics. If the data cannot answer, "
                       "say so.\n" + json.dumps(context)},
                      {"role": "user", "content": question}],
            temperature=0.3, max_tokens=300)
    except (OpenAIError, TypeError, ValueError):
        return jsonify({"error": "The AI box is unavailable right now (check the OpenAI key on the server)."}), 503
    return jsonify({"answer": reply.choices[0].message.content})


# ----------------------------------------------------------------------------- older endpoints (model.json)
def standardise(block, x):
    return [(x[f] - m) / s for f, m, s in zip(block["features"], block["mean"], block["scale"])]


@app.route("/api/race")
def api_race():
    m = _read("model.json")
    if not m:
        abort(404)
    return jsonify({"race": m["race"], "mode": m["mode"],
                    "drivers": [{k: d[k] for k in ("id", "name", "team", "tier", "grid", "p_win", "p_top10")}
                                for d in m["drivers"]]})


@app.route("/api/duel", methods=["POST"])
def api_duel():
    m = _read("model.json")
    body = request.get_json(force=True, silent=True)
    try:
        drivers = {d["id"]: d for d in m["drivers"]}
        a, b, blk = drivers[body["a"]], drivers[body["b"]], m["duel"]
        s = sum(w * (xa - xb) for w, xa, xb in zip(blk["w"], standardise(blk, a["x"]), standardise(blk, b["x"])))
        p_a = 1 / (1 + math.exp(-s))
        return jsonify({"a": a["name"], "b": b["name"], "p_a": p_a, "p_b": 1 - p_a})
    except (KeyError, TypeError):
        return jsonify({"error": "Pick two drivers from the list."}), 400


if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 5050)))   # 5000 is taken by AirPlay on macOS
