"""
tests/test_pipeline.py
-----------------------
Pytest suite for the ACS Income Predictor pipeline.

Covers:
  - Data file existence and schema validation
  - Data quality (no negative incomes, no nulls in required columns)
  - Model file existence and correct load
  - Model produces a sensible prediction
  - Flask API endpoints (/health, /info, /predict, error cases)
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
DATA_PATH  = "data/acs_data.csv"
MODEL_PATH = "model/income_model.pkl"

REQUIRED_DATA_COLS = [
    "median_household_income",
    "median_age",
    "total_population",
    "unemployment_rate",
    "housing_occupancy_rate",
]

SAMPLE_INPUT = {
    "median_age": 38.5,
    "total_population": 4_200_000,
    "unemployment_rate": 0.048,
    "housing_occupancy_rate": 0.35,
}


# ===========================================================================
# Data tests
# ===========================================================================

def test_data_file_exists():
    assert os.path.exists(DATA_PATH), f"Data file not found: {DATA_PATH}"


def test_data_has_required_columns():
    df = pd.read_csv(DATA_PATH)
    for col in REQUIRED_DATA_COLS:
        assert col in df.columns, f"Missing column: {col}"


def test_data_has_rows():
    df = pd.read_csv(DATA_PATH)
    assert len(df) >= 50, f"Expected at least 50 state rows, got {len(df)}"


def test_data_no_negative_income():
    df = pd.read_csv(DATA_PATH)
    assert (df["median_household_income"] > 0).all(), (
        "Found non-positive income values — ACS sentinel values not cleaned"
    )


def test_data_no_nulls_in_features():
    df = pd.read_csv(DATA_PATH)
    nulls = df[REQUIRED_DATA_COLS].isnull().sum()
    assert nulls.sum() == 0, f"Null values found:\n{nulls[nulls > 0]}"


def test_unemployment_rate_range():
    df = pd.read_csv(DATA_PATH)
    assert df["unemployment_rate"].between(0, 1).all(), (
        "Unemployment rate out of [0, 1] range"
    )


# ===========================================================================
# Model tests
# ===========================================================================

def test_model_file_exists():
    assert os.path.exists(MODEL_PATH), (
        f"Model file not found: {MODEL_PATH}. "
        "Run `python bootstrap.py` locally before pushing."
    )


def test_model_loads():
    import joblib
    model = joblib.load(MODEL_PATH)
    assert model is not None


def test_model_predicts_positive_income():
    import joblib
    model = joblib.load(MODEL_PATH)
    features = ["median_age", "total_population", "unemployment_rate", "housing_occupancy_rate"]
    X = np.array([[SAMPLE_INPUT[f] for f in features]])
    pred = model.predict(X)
    assert len(pred) == 1
    assert pred[0] > 0, f"Predicted a non-positive income: {pred[0]}"


def test_model_prediction_in_plausible_range():
    """Income should be between $30k and $200k for US states."""
    import joblib
    model = joblib.load(MODEL_PATH)
    features = ["median_age", "total_population", "unemployment_rate", "housing_occupancy_rate"]
    X = np.array([[SAMPLE_INPUT[f] for f in features]])
    pred = model.predict(X)[0]
    assert 30_000 < pred < 200_000, f"Prediction ${pred:,.0f} outside plausible range"


# ===========================================================================
# Flask API tests
# ===========================================================================

@pytest.fixture(scope="module")
def client():
    os.environ["MODEL_PATH"] = MODEL_PATH
    # Import app only after setting env var
    import importlib
    import app as app_module
    importlib.reload(app_module)
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_status_ok(client):
    data = resp = client.get("/health").get_json()
    assert data["status"] in ("ok", "degraded")


def test_info_returns_features(client):
    resp = client.get("/info")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "features" in data
    assert "median_age" in data["features"]


def test_predict_valid_input(client):
    resp = client.post("/predict", json=SAMPLE_INPUT)
    assert resp.status_code == 200
    data = resp.get_json()
    assert "predicted_median_household_income_usd" in data
    assert data["predicted_median_household_income_usd"] > 0


def test_predict_missing_feature_returns_400(client):
    bad_input = {"median_age": 38.5}   # missing 3 features
    resp = client.post("/predict", json=bad_input)
    assert resp.status_code == 400
    data = resp.get_json()
    assert "error" in data


def test_predict_no_body_returns_400(client):
    resp = client.post("/predict", data="not-json", content_type="text/plain")
    assert resp.status_code == 400
