"""
ingest.py — Data Acquisition & Preprocessing
---------------------------------------------
Loads and validates the ACS 5-Year state-level dataset before training.

The source dataset (data/acs_data.csv) contains ACS 2024 5-Year estimates
for all 50 US states and DC, covering income, age, population, employment,
and housing variables published by the US Census Bureau.

In a production environment this script would fetch live data directly from
the Census Bureau API (api.census.gov/data/{year}/acs/acs5) each December
when new estimates are released. For this pipeline the validated CSV is
committed to the repository and used as the stable data source.

Outputs : data/acs_data.csv  (validated and ready for training)

Usage:
    python ingest.py
"""

import os
import sys
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join("data", "acs_data.csv")

REQUIRED_COLS = [
    "median_household_income",
    "median_age",
    "total_population",
    "unemployment_rate",
    "housing_occupancy_rate",
]


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
def validate() -> pd.DataFrame:
    print(f"Loading data from {DATA_PATH}...")

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

    # Required columns present
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        sys.exit(1)

    # Minimum row count (50 states + DC)
    if len(df) < 50:
        print(f"ERROR: Expected at least 50 rows, got {len(df)}")
        sys.exit(1)

    # No nulls in required columns
    nulls = df[REQUIRED_COLS].isnull().sum()
    if nulls.sum() > 0:
        print(f"ERROR: Null values found:\n{nulls[nulls > 0]}")
        sys.exit(1)

    # No negative or zero income values
    if not (df["median_household_income"] > 0).all():
        print("ERROR: Non-positive income values found — check for sentinel values.")
        sys.exit(1)

    # Unemployment rate in valid range
    if not df["unemployment_rate"].between(0, 1).all():
        print("ERROR: Unemployment rate out of [0, 1] range.")
        sys.exit(1)

    print("All validation checks passed.")
    print(df[REQUIRED_COLS].describe().to_string())
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    validate()
