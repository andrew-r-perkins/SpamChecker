"""
api.py — Flask REST API for the spam classifier.

This is the bridge between the React frontend and the trained ML models.
Models are loaded once at startup (expensive), then serve fast predictions.

Which models to load is controlled by --models (default: all three).
This lets the app run on smaller instances by loading only a subset:

  python api.py                              # load all three models
  python api.py --models tfidf              # TF-IDF only  (t2.micro-friendly)
  python api.py --models tfidf distilbert   # skip mBERT   (t3.medium-friendly)
  python api.py --models tfidf mbert -v     # two models + INFO logging

Endpoints:
  GET  /config   — returns list of active models (used by frontend on mount)
  POST /predict  — accepts { "text": "..." }, returns scores for active models
  GET  /health   — simple liveness check, returns { "status": "ok" }

Verbosity flags:
  python api.py -v    # INFO  — startup messages + per-request scores
  python api.py -vv   # DEBUG — text length, vector/token shapes per request
  python api.py -vvv  # TRACE — top TF-IDF features + transformer tokens

Must be run from the project root so relative model paths resolve correctly.
"""

import argparse
import logging

from flask import Flask, request, jsonify
from flask_cors import CORS

from src.log_config import add_verbosity_args, setup_logging
from src.tfidf_model.predict   import (load_model_and_vectorizer,
                                        score_email as tfidf_score)
from src.mbert_model.predict   import (load_model_and_tokenizer  as load_mbert,
                                        score_email               as mbert_score)
from src.distilbert_model.predict import (load_model_and_tokenizer as load_distilbert,
                                           score_email              as distilbert_score)

# ---------------------------------------------------------------------------
# CLI — parse flags before Flask starts
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Spam Checker API server")
add_verbosity_args(parser)
parser.add_argument(
    "--models",
    nargs="+",
    choices=["tfidf", "mbert", "distilbert"],
    default=["tfidf", "mbert", "distilbert"],
    metavar="MODEL",
    help=(
        "Which models to load at startup. "
        "Choices: tfidf, mbert, distilbert. "
        "Default: all three. "
        "Example: --models tfidf distilbert"
    ),
)
args = parser.parse_args()
setup_logging(args.verbose)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry
# Each entry defines how to load the model and how to score a text with it.
# The loader returns a tuple (obj_a, obj_b) which the scorer unpacks.
#   TF-IDF   → (keras_model, vectorizer)
#   mBERT    → (torch_model, tokenizer)
#   DistilBERT → (torch_model, tokenizer)
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "tfidf": {
        "display": "TF-IDF + NN",
        "loader":  load_model_and_vectorizer,
        "scorer":  lambda obj, text: tfidf_score(obj[0], obj[1], text),
    },
    "mbert": {
        "display": "mBERT",
        "loader":  load_mbert,
        "scorer":  lambda obj, text: mbert_score(obj[0], obj[1], text),
    },
    "distilbert": {
        "display": "DistilBERT",
        "loader":  load_distilbert,
        "scorer":  lambda obj, text: distilbert_score(obj[0], obj[1], text),
    },
}

# ---------------------------------------------------------------------------
# Flask app setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)   # allow Vite dev server (localhost:5173); restrict in production

# ---------------------------------------------------------------------------
# Load requested models at startup — kept in memory for fast inference.
# LOADED maps key → (obj_a, obj_b) tuple returned by each loader.
# ---------------------------------------------------------------------------
LOADED = {}

for key in args.models:
    display = MODEL_REGISTRY[key]["display"]
    logger.info("Loading %s...", display)
    LOADED[key] = MODEL_REGISTRY[key]["loader"]()
    logger.info("%s ready.", display)

active = ", ".join(MODEL_REGISTRY[k]["display"] for k in LOADED)
logger.info("Models active: %s — API is live on http://localhost:5000", active)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/config", methods=["GET"])
def config():
    """
    Returns the list of models currently loaded.
    The frontend fetches this once on mount to know how many panels to render.

    Response: { "models": [{"key": "tfidf", "name": "TF-IDF + NN"}, ...] }
    """
    logger.debug("GET /config — returning %d active models", len(LOADED))
    return jsonify({
        "models": [
            {"key": k, "name": MODEL_REGISTRY[k]["display"]}
            for k in LOADED
        ]
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a JSON body: { "text": "<email or message>" }
    Returns scores for all active models, e.g.:
      { "tfidf": 0.97, "mbert": 0.95, "distilbert": 0.94 }

    Only the models that were loaded at startup are included in the response.
    """
    data = request.get_json()
    text = data.get("text", "")

    if not text.strip():
        logger.warning("POST /predict — rejected empty text")
        return jsonify({"error": "No text provided"}), 400

    scores = {}
    for key, obj in LOADED.items():
        scores[key] = float(MODEL_REGISTRY[key]["scorer"](obj, text))

    score_str = " | ".join(f"{k}: {v:.4f}" for k, v in scores.items())
    logger.info("POST /predict — %d chars → %s", len(text), score_str)

    return jsonify(scores)


@app.route("/health", methods=["GET"])
def health():
    """Simple liveness probe — returns 200 OK when the server is up."""
    logger.debug("GET /health — OK")
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting Flask server on http://localhost:5000")
    app.run(port=5000, debug=False)
