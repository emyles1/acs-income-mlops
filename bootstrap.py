"""
bootstrap.py — One-time local setup
-------------------------------------
Run this once on your local machine before your first git push.
It trains the baseline model on the sample ACS data and saves
model/income_model.pkl so the Dockerfile and CI tests have a
model to work with.

Usage:
    pip install -r requirements.txt
    python bootstrap.py
"""

from train import train

if __name__ == "__main__":
    print("Running initial model training on sample ACS data...")
    rmse, r2 = train()
    print(f"\nBootstrap complete — model saved to model/income_model.pkl")
    print(f"Baseline RMSE: ${rmse:,.0f}  |  R²: {r2:.4f}")
    print("\nYou can now commit everything and push to GitHub.")
