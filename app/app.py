"""F1 Duel - Flask website (Technology with Impact: ML model deployed as an API + OpenAI).

Routes
  GET  /             the page
  GET  /api/race     next race: every driver's chance to win and to score points
  POST /api/duel     {"a": driverId, "b": driverId} -> chance each finishes ahead + the 3 main reasons
  POST /api/ask      {"question": "...", "a": .., "b": ..} -> ChatGPT answer grounded in the model's numbers

The model is plain numbers in model.json (written by `python -m f1duel.forecast`), so the site
only needs Flask (and `openai` for the AI box). Set OPENAI_API_KEY in the PythonAnywhere WSGI file.
"""
import json
import math
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

HERE = Path(__file__).parent
app = Flask(__name__)


def load_model():
    return json.loads((HERE / "model.json").read_text())


def standardise(block, x):
    return [(x[f] - m) / s for f, m, s in zip(block["features"], block["mean"], block["scale"])]


def duel(model, a_id, b_id):
    drivers = {d["id"]: d for d in model["drivers"]}
    a, b = drivers[a_id], drivers[b_id]
    blk = model["duel"]
    za, zb = standardise(blk, a["x"]), standardise(blk, b["x"])
    parts = [w * (xa - xb) for w, xa, xb in zip(blk["w"], za, zb)]
    s = sum(parts)
    p_a = 1 / (1 + math.exp(-s))
    reasons = sorted(zip(blk["features"], parts), key=lambda t: -abs(t[1]))[:3]
    return {"a": a["name"], "b": b["name"], "p_a": p_a, "p_b": 1 - p_a,
            "reasons": [{"feature": model["labels"][f], "favours": a["name"] if v > 0 else b["name"],
                         "weight": abs(v)} for f, v in reasons if abs(v) > 0.05]}


@app.route("/")
def index():
    return render_template("index.html", model=load_model())


@app.route("/api/race")
def api_race():
    m = load_model()
    return jsonify({"race": m["race"], "mode": m["mode"],
                    "drivers": [{k: d[k] for k in ("id", "name", "team", "tier", "grid", "p_win", "p_top10")}
                                for d in m["drivers"]]})


@app.route("/api/duel", methods=["POST"])
def api_duel():
    body = request.get_json(force=True, silent=True)
    try:
        return jsonify(duel(load_model(), body["a"], body["b"]))
    except (KeyError, TypeError):
        return jsonify({"error": "Pick two drivers from the list."}), 400


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
    m = load_model()
    context = {"race": m["race"], "mode": m["mode"],
               "drivers": [{k: d[k] for k in ("name", "team", "tier", "grid", "p_win", "p_top10")} for d in m["drivers"]],
               "evidence": m["evidence"], "scoreboard": m["scoreboard"][-5:]}
    ids = {d["id"] for d in m["drivers"]}
    if body.get("a") in ids and body.get("b") in ids and body["a"] != body["b"]:
        context["current_duel"] = duel(m, body["a"], body["b"])
    from openai import OpenAI, OpenAIError                       # as in the GenAI tutorial
    client = OpenAI(api_key=key)
    try:
        reply = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "system", "content":
                       "You are the race engineer of a student machine-learning project that predicts Formula 1 results. "
                       "Answer in the user's language, in at most 120 words, for a general audience. "
                       "Use ONLY the numbers in this JSON (model outputs and test results); never invent statistics. "
                       "If the data cannot answer, say so. Probabilities are estimates, not certainties.\n"
                       + json.dumps(context)},
                      {"role": "user", "content": question}],
            temperature=0.3, max_tokens=300)
    except OpenAIError:
        return jsonify({"error": "The AI box is unavailable right now (check the OpenAI key on the server)."}), 503
    return jsonify({"answer": reply.choices[0].message.content})


if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 5050)))   # 5000 is taken by AirPlay on macOS
