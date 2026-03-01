"""
api.py — Flask REST API for the spam classifier.

This is the bridge between the React frontend and both trained ML models.
It loads both models once at startup (expensive), then serves fast predictions
on each incoming request.

Endpoints:
  POST /predict  — accepts { "text": "..." }, returns { "tfidf": 0.97, "mbert": 0.95 }
  GET  /health   — simple liveness check, returns { "status": "ok" }

Usage:
  python api.py            # WARNING level — only errors shown
  python api.py -v         # INFO    — startup messages + per-request results
  python api.py -vv        # DEBUG   — adds text length, vector shape per request
  python api.py -vvv       # TRACE   — adds top TF-IDF features + mBERT tokens

Must be run from the project root so relative model paths resolve correctly.
"""

import argparse
import logging

from flask import Flask, request, jsonify
from flask_cors import CORS

from src.log_config import add_verbosity_args, setup_logging
from src.tfidf_model.predict import load_model_and_vectorizer, score_email as tfidf_score
from src.mbert_model.predict import load_model_and_tokenizer, score_email as mbert_score

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
# Load both model artifacts at startup.
# We load once and keep in memory — loading on every request would be too slow.
# ---------------------------------------------------------------------------
logger.info("Loading TF-IDF model...")
tfidf_model, vectorizer = load_model_and_vectorizer()

logger.info("Loading mBERT model...")
mbert_model, tokenizer = load_model_and_tokenizer()

logger.info("Both models ready — API is live on http://localhost:5000")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a JSON body: { "text": "<email or message>" }
    Returns:            { "tfidf": <float 0-1>, "mbert": <float 0-1> }

    Runs both models on every request and returns both scores.

    Log output by verbosity:
      -v   : text length + both probabilities
      -vv  : feature vector shape (TF-IDF), token count (mBERT)
      -vvv : top 10 TF-IDF features + mBERT token list
    """
    data = request.get_json()
    text = data.get("text", "")

    if not text.strip():
        logger.warning("POST /predict — rejected empty text")
        return jsonify({"error": "No text provided"}), 400

    tfidf_prob = float(tfidf_score(tfidf_model, vectorizer, text))
    mbert_prob = float(mbert_score(mbert_model, tokenizer, text))

    logger.info(
        "POST /predict — %d chars → tfidf: %.4f | mbert: %.4f",
        len(text), tfidf_prob, mbert_prob,
    )

    return jsonify({"tfidf": tfidf_prob, "mbert": mbert_prob})


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
