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

Security controls:
  - MAX_CONTENT_LENGTH: 64 KB hard limit on request body (Flask built-in)
  - MAX_TEXT_CHARS: 10,000 character limit on the text field
  - Rate limiting via Flask-Limiter (per IP):
      /predict → 30 requests/minute
      /config  → 10 requests/minute
      /health  → 60 requests/minute

Verbosity flags (when run directly):
  python api.py -v    # INFO  — startup messages + per-request scores
  python api.py -vv   # DEBUG — text length, vector/token shapes per request
  python api.py -vvv  # TRACE — top TF-IDF features + transformer tokens

Must be run from the project root so relative model paths resolve correctly.

Testing:
  The create_app() factory accepts a preloaded dict and optional registry
  override so tests can inject mock models without loading real files.
  See tests/test_api.py.
"""

import argparse
import logging
import os

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from src.log_config import add_verbosity_args, setup_logging
from src.tfidf_model.predict      import (load_model_and_vectorizer,
                                           score_email as tfidf_score)
from src.mbert_model.predict      import (load_model_and_tokenizer  as load_mbert,
                                           score_email               as mbert_score)
from src.distilbert_model.predict import (load_model_and_tokenizer  as load_distilbert,
                                           score_email               as distilbert_score)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry
# Each entry defines how to load the model and how to score a text with it.
# The loader returns a tuple (obj_a, obj_b) which the scorer unpacks.
#   TF-IDF     → (keras_model, vectorizer)
#   mBERT      → (torch_model, tokenizer)
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

# Soft limit on the text field (checked inside /predict).
# Also exported so tests can import this constant directly.
MAX_TEXT_CHARS = 10_000


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(preloaded, registry=None):
    """Create and configure the Flask app.

    Args:
        preloaded: dict mapping model key → loaded model objects.
                   In production: pass the LOADED dict built at startup.
                   In tests: pass a mock dict (no real model files needed).
        registry:  optional override for MODEL_REGISTRY (used in tests to
                   inject mock scorer functions).
    Returns:
        Configured Flask application instance.
    """
    if registry is None:
        registry = MODEL_REGISTRY

    app = Flask(__name__)

    # Restrict CORS to specific origins.
    # Dev default: Vite dev server on localhost:5173.
    # Production: set ALLOWED_ORIGINS=https://yoursite.com before starting.
    # Multiple origins: comma-separated, e.g. "https://a.com,https://b.com".
    _raw_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173")
    _allowed_origins = [o.strip() for o in _raw_origins.split(",")]
    CORS(app, origins=_allowed_origins)

    # Hard limit on incoming request body size — Flask returns 413
    # automatically if Content-Length exceeds this before the route runs.
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024  # 64 KB

    # Rate limiting — per IP, in-memory (suitable for single-worker).
    # Swap storage_uri to Redis for multi-worker setups.
    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=[],       # no blanket default — set per-route below
        storage_uri="memory://",
    )

    # ── Security headers ────────────────────────────────────────────────────

    @app.after_request
    def add_security_headers(response):
        # Prevent browsers from MIME-sniffing the content-type
        response.headers["X-Content-Type-Options"] = "nosniff"
        # Prevent responses being embedded in iframes (clickjacking)
        response.headers["X-Frame-Options"] = "DENY"
        # Don't leak the URL in the Referer header on outbound links
        response.headers["Referrer-Policy"] = "no-referrer"
        # This API only serves JSON — no scripts, styles, or media needed
        response.headers["Content-Security-Policy"] = "default-src 'none'"
        # Disable browser features this API has no need for
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=()"
        )
        return response

    # ── Error handlers ──────────────────────────────────────────────────────

    @app.errorhandler(413)
    def request_too_large(e):
        logger.warning("413 — request body exceeded 64 KB limit")
        return jsonify({"error": "Request too large (max 64 KB)"}), 413

    @app.errorhandler(429)
    def rate_limit_exceeded(e):
        logger.warning("429 — rate limit exceeded: %s", e.description)
        return jsonify({"error": "Too many requests — please slow down"}), 429

    # ── Routes ──────────────────────────────────────────────────────────────

    @app.route("/config", methods=["GET"])
    @limiter.limit("10 per minute")
    def config():
        """
        Returns the list of models currently loaded.
        The frontend fetches this once on mount to know how many panels to render.

        Response: { "models": [{"key": "tfidf", "name": "TF-IDF + NN"}, ...] }
        """
        logger.debug("GET /config — returning %d active models", len(preloaded))
        return jsonify({
            "models": [
                {"key": k, "name": registry[k]["display"]}
                for k in preloaded
            ]
        })

    @app.route("/predict", methods=["POST"])
    @limiter.limit("30 per minute")
    def predict():
        """
        Accepts a JSON body: { "text": "<email or message>" }
        Returns scores for all active models, e.g.:
          { "tfidf": 0.97, "mbert": 0.95, "distilbert": 0.94 }

        Rejects with 400 if body is not JSON, text is missing, or text is empty.
        Rejects with 413 if text exceeds MAX_TEXT_CHARS characters.
        Rejects with 429 if the caller exceeds 30 requests per minute.
        """
        # silent=True returns None instead of raising on malformed/missing JSON;
        # "or {}" ensures data.get() never throws AttributeError.
        data = request.get_json(silent=True) or {}
        text = data.get("text", "")

        if not text.strip():
            logger.warning("POST /predict — rejected empty text")
            return jsonify({"error": "No text provided"}), 400

        if len(text) > MAX_TEXT_CHARS:
            logger.warning(
                "POST /predict — rejected oversized text (%d chars)", len(text)
            )
            return jsonify({
                "error": f"Text too long — max {MAX_TEXT_CHARS:,} characters"
            }), 413

        scores = {}
        for key, obj in preloaded.items():
            try:
                scores[key] = float(registry[key]["scorer"](obj, text))
            except Exception as e:
                logger.error("Inference error for %s: %s", key, e)
                return jsonify({"error": "Prediction failed — please try again"}), 500

        score_str = " | ".join(f"{k}: {v:.4f}" for k, v in scores.items())
        logger.info("POST /predict — %d chars → %s", len(text), score_str)

        return jsonify(scores)

    @app.route("/health", methods=["GET"])
    @limiter.limit("60 per minute")
    def health():
        """Simple liveness probe — returns 200 OK when the server is up."""
        logger.debug("GET /health — OK")
        return jsonify({"status": "ok"})

    return app


# ---------------------------------------------------------------------------
# CLI entry point — only executed when run directly, not when imported.
# Keeping model loading here means `import api` in tests never touches
# the filesystem or loads any ML libraries unnecessarily.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
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

    LOADED = {}
    for key in args.models:
        display = MODEL_REGISTRY[key]["display"]
        logger.info("Loading %s...", display)
        try:
            LOADED[key] = MODEL_REGISTRY[key]["loader"]()
            logger.info("%s ready.", display)
        except FileNotFoundError:
            logger.warning("%s — model files not found, skipping.", display)
        except Exception as e:
            logger.warning("%s — failed to load (%s: %s), skipping.",
                           display, type(e).__name__, e)

    if not LOADED:
        logger.critical(
            "No models loaded — cannot start. "
            "Check that model files exist in ./models/"
        )
        raise SystemExit(1)

    active = ", ".join(MODEL_REGISTRY[k]["display"] for k in LOADED)
    logger.info("Models active: %s — API is live on http://localhost:5000", active)

    app = create_app(LOADED)
    app.run(port=5000, debug=False)
