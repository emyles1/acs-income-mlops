"""
train.py — Model Training & Testing
------------------------------------
Trains a Random Forest regression model on ACS 5-Year state-level data
to predict median household income. Logs parameters and metrics to MLflow.

Inputs  : data/acs_data.csv  (produced by ingest.py)
Outputs : model/income_model.pkl  (committed to repo, built into Docker image)

Usage:
    MLFLOW_TRACKING_URI=http://localhost:5555 python train.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FEATURES = [
    "median_age",
    "total_population",
    "unemployment_rate",
    "housing_occupancy_rate",
]
TARGET = "median_household_income"
DATA_PATH = os.path.join("data", "acs_data.csv")
MODEL_PATH = os.path.join("model", "income_model.pkl")

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "")

# Hyperparameters
N_ESTIMATORS = 100
MAX_DEPTH = 5
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# MLflow logging (optional — skipped if server unreachable)
# ---------------------------------------------------------------------------
def try_log_mlflow(params: dict, metrics: dict, model):
    if not MLFLOW_URI:
        print("MLFLOW_TRACKING_URI not set — skipping MLflow logging.")
        return
    try:
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("acs-income-predictor")
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
            print(f"Logged to MLflow at {MLFLOW_URI}")
    except Exception as e:
        print(f"MLflow logging skipped: {e}")


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=FEATURES + [TARGET])

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=N_ESTIMATORS,
                    max_depth=MAX_DEPTH,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    print(f"RMSE : ${rmse:,.0f}")
    print(f"R²   : {r2:.4f}")

    params = {
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "features": str(FEATURES),
        "acs_year": int(df["acs_year"].iloc[0]) if "acs_year" in df.columns else "unknown",
    }
    metrics = {"rmse": rmse, "r2": r2}

    try_log_mlflow(params, metrics, model)

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    return rmse, r2


if __name__ == "__main__":
    train()
