"""
app.py — Flask Prediction API
------------------------------
Serves the trained income prediction model via a REST API.

Endpoints:
    GET  /health    → liveness check
    GET  /info      → model metadata
    POST /predict   → predict median household income

Example request:
    curl -X POST http://localhost:5000/predict \
         -H "Content-Type: application/json" \
         -d '{
               "median_age": 38.5,
               "total_population": 4200000,
               "unemployment_rate": 0.048,
               "housing_occupancy_rate": 0.35
             }'
"""

import os
import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load model at startup
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "model/income_model.pkl")

FEATURES = [
    "median_age",
    "total_population",
    "unemployment_rate",
    "housing_occupancy_rate",
]

try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"WARNING: Could not load model — {e}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    status = "ok" if model is not None else "degraded"
    return jsonify({"status": status}), 200


@app.route("/info", methods=["GET"])
def info():
    return jsonify(
        {
            "model": "Random Forest Regressor",
            "target": "median_household_income (USD)",
            "features": FEATURES,
            "data_source": "US Census ACS 5-Year Estimates",
        }
    ), 200


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing features: {missing}"}), 400

    try:
        X = np.array([[float(data[f]) for f in FEATURES]])
        prediction = model.predict(X)[0]
        return jsonify(
            {
                "predicted_median_household_income_usd": round(float(prediction), 2),
                "input": {f: data[f] for f in FEATURES},
            }
        ), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
