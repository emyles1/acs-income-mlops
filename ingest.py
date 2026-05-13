"""
ingest.py — ACS Data Acquisition & Preprocessing
-------------------------------------------------
Fetches ACS 5-Year estimates from the Census Bureau API for all US states.
Target variable  : B19013_001E — Median Household Income
Feature variables: median age, total population, employment, housing occupancy

The ACS releases new 5-year estimates each December. The YEAR env var controls
which vintage is pulled, making this script the trigger for Continuous Training.

Usage:
    ACS_YEAR=2024 CENSUS_API_KEY=your_key python ingest.py
"""

import os
import sys
import requests
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("CENSUS_API_KEY", "")
YEAR = os.environ.get("ACS_YEAR", "2024")
BASE_URL = f"https://api.census.gov/data/{YEAR}/acs/acs5"

# ACS variable codes → friendly column names
VARIABLES = {
    "B19013_001E": "median_household_income",  # TARGET
    "B01002_001E": "median_age",
    "B01003_001E": "total_population",
    "B23025_004E": "employed",
    "B23025_005E": "unemployed",
    "B25002_002E": "occupied_housing_units",
}

OUTPUT_PATH = os.path.join("data", "acs_data.csv")


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
def fetch_acs_data(year: str = YEAR) -> pd.DataFrame:
    var_string = ",".join(VARIABLES.keys())
    url = f"https://api.census.gov/data/{year}/acs/acs5?get=NAME,{var_string}&for=state:*"
    if API_KEY:
        url += f"&key={API_KEY}"

    print(f"Fetching ACS {year} 5-Year data from Census API...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    data = response.json()
    headers = data[0]
    rows = data[1:]

    df = pd.DataFrame(rows, columns=headers)
    return df


# ---------------------------------------------------------------------------
# Preprocess
# ---------------------------------------------------------------------------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=VARIABLES)

    # Cast to numeric — ACS uses -666666666 / -888888888 for N/A
    numeric_cols = list(VARIABLES.values())
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop missing target
    df = df.dropna(subset=["median_household_income"])

    # Remove sentinel negative values
    df = df[df["median_household_income"] > 0]
    df = df[df["employed"] > 0]

    # Derived features
    df["unemployment_rate"] = df["unemployed"] / (df["employed"] + df["unemployed"])
    df["housing_occupancy_rate"] = df["occupied_housing_units"] / df["total_population"]

    # Tag with vintage year
    df["acs_year"] = int(YEAR)

    # Keep only the columns we need
    keep = [
        "NAME",
        "state",
        "acs_year",
        "median_household_income",
        "median_age",
        "total_population",
        "unemployment_rate",
        "housing_occupancy_rate",
    ]
    df = df[keep].reset_index(drop=True)

    print(f"Preprocessed {len(df)} state rows.")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs("data", exist_ok=True)

    df = fetch_acs_data(YEAR)
    df = preprocess(df)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")
    print(df.describe())


if __name__ == "__main__":
    main()
