# ============================================================
# Dockerfile — ACS Income Predictor Flask API
# ============================================================
# Multi-stage is overkill for this model size — single stage
# keeps the demo clear and the image small enough.
#
# Build context: repo root (so model/ and app.py are in scope)
# ============================================================

FROM python:3.10-slim

WORKDIR /app

# Install only the runtime dependencies (no dev/test tools)
RUN pip install --no-cache-dir \
    flask==3.0.3 \
    scikit-learn==1.4.0 \
    joblib==1.3.2 \
    numpy==1.26.4

# Copy the trained model
COPY model/income_model.pkl model/income_model.pkl

# Copy the Flask application
COPY app.py .

EXPOSE 5000

CMD ["python", "app.py"]
