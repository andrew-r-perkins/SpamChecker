"""
api.py — Flask REST API for the spam classifier.

This is the bridge between the React frontend and the trained ML model.
It loads the model once at startup (expensive), then serves fast predictions
on each incoming request.

Endpoints:
  POST /predict  — accepts { "text": "..." }, returns { "spam_probability": 0.97 }
  GET  /health   — simple liveness check, returns { "status": "ok" }

Usage:
  python api.py            # WARNING level — only errors shown
  python api.py -v         # INFO    — startup messages + per-request results
  python api.py -vv        # DEBUG   — adds text length, vector shape per request
  python api.py -vvv       # TRACE   — adds top TF-IDF features per request

Must be run from the project root so relative model paths resolve correctly.
"""

import argparse
import logging

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import joblib
import numpy as np

from src.log_config import add_verbosity_args, setup_logging, TRACE

# ---------------------------------------------------------------------------
# CLI — parse -v / -vv / -vvv before Flask starts
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Spam Checker API server")
add_verbosity_args(parser)
args = parser.parse_args()
setup_logging(args.verbose)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask app setup
# CORS allows the Vite dev server (localhost:5173) to call this API.
# In production restrict origins to your actual domain.
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Model paths (relative to project root)
# ---------------------------------------------------------------------------
MODEL_PATH      = "./models/tfidf_nn/spam_model.keras"
VECTORIZER_PATH = "./models/tfidf_nn/vectorizer.joblib"

# ---------------------------------------------------------------------------
# Load model artifacts at startup.
# We load once and keep in memory — loading on every request would be ~10 s.
# ---------------------------------------------------------------------------
logger.info("Loading model from %s", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

logger.info("Loading vectorizer from %s", VECTORIZER_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

logger.info("Model and vectorizer ready — API is live on http://localhost:5000")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a JSON body: { "text": "<email or message>" }
    Returns:            { "spam_probability": <float 0-1> }

    Pipeline:
      1. Parse + validate the request body
      2. Vectorize text with the pre-fitted TF-IDF vectorizer
      3. Run a forward pass through the neural network
      4. Return the spam probability as JSON

    Log output by verbosity:
      -v   : text length + final probability
      -vv  : feature vector shape
      -vvv : top 10 TF-IDF features that fired for this text
    """
    data = request.get_json()
    text = data.get("text", "")

    # Reject empty input early — the model never sees a zero-length string
    if not text.strip():
        logger.warning("POST /predict — rejected empty text")
        return jsonify({"error": "No text provided"}), 400

    # -vvv: show the first 200 chars of the raw input
    logger.trace("POST /predict — raw input (first 200 chars): %.200s", text)

    # Transform raw text → TF-IDF feature vector (shape: 1 × max_features)
    # .toarray() converts the sparse matrix to dense for Keras
    vec = vectorizer.transform([text]).toarray()
    logger.debug("POST /predict — feature vector shape: %s", vec.shape)

    # -vvv: show which TF-IDF features actually fired (non-zero weights)
    if logger.isEnabledFor(TRACE):
        feature_names   = vectorizer.get_feature_names_out()
        non_zero_idx    = vec[0].nonzero()[0]
        top_features    = sorted(
            [(feature_names[i], round(float(vec[0][i]), 4)) for i in non_zero_idx],
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        logger.trace(
            "POST /predict — %d/%d features fired | top 10: %s",
            len(non_zero_idx),
            vec.shape[1],
            top_features,
        )

    # model.predict returns shape (1, 1); extract the scalar probability
    prob = float(model.predict(vec)[0][0])

    logger.info(
        "POST /predict — %d chars → spam probability: %.4f",
        len(text), prob,
    )

    return jsonify({"spam_probability": prob})


@app.route("/health", methods=["GET"])
def health():
    """
    Simple liveness probe.
    Returns 200 OK with { "status": "ok" } when the server is up.
    """
    logger.debug("GET /health — OK")
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting Flask server on http://localhost:5000")
    app.run(port=5000, debug=False)
