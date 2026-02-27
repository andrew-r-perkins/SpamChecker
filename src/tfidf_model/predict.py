"""
predict.py — Loads the trained model and scores individual messages.

Used in two ways:
  1. Imported by the Flask API (api.py) — logging configured by the server.
  2. Run directly as a CLI tool for quick manual testing.

Usage (CLI, run from project root):
  python -m src.tfidf_model.predict          "Your message here"   # WARNING
  python -m src.tfidf_model.predict -v       "Your message here"   # INFO
  python -m src.tfidf_model.predict -vv      "Your message here"   # DEBUG
  python -m src.tfidf_model.predict -vvv     "Your message here"   # TRACE

Score interpretation:
  0.0  = certainly ham (legitimate mail)
  0.5  = model is uncertain
  1.0  = certainly spam
"""

import argparse
import logging

import tensorflow as tf
import joblib

from src.log_config import add_verbosity_args, setup_logging, TRACE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model paths (relative to project root)
# ---------------------------------------------------------------------------
MODEL_PATH      = "./models/tfidf_nn/spam_model.keras"
VECTORIZER_PATH = "./models/tfidf_nn/vectorizer.joblib"


def load_model_and_vectorizer():
    """
    Loads the saved Keras model and TF-IDF vectorizer from disk.

    Both artifacts are produced together by train.py and MUST be loaded
    as a matched pair — the vectorizer's vocabulary defines the feature
    columns the model was trained on.

    Log output by verbosity:
      -vv  : confirms paths being loaded
      -vvv : (nothing extra here — detail is in score_email)

    Returns:
        (model, vectorizer) tuple
    """
    logger.debug("Loading Keras model from %s", MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)

    logger.debug("Loading TF-IDF vectorizer from %s", VECTORIZER_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    logger.info("Model and vectorizer loaded successfully")
    return model, vectorizer


def score_email(model, vectorizer, text):
    """
    Scores a single piece of text and returns a spam probability.

    Pipeline:
      1. Wrap text in a list (sklearn vectorizers expect an iterable)
      2. Transform with the fitted TF-IDF vectorizer → sparse matrix
      3. Convert to dense array (.toarray()) for Keras
      4. Forward pass through the neural network
      5. Extract the scalar probability from the (1, 1) output tensor

    Log output by verbosity:
      -v   : nothing (result printed by main())
      -vv  : feature vector shape, raw model output
      -vvv : number of non-zero features + top 10 TF-IDF terms that fired

    Args:
        model:      loaded Keras Sequential model
        vectorizer: fitted TF-IDF vectorizer (must match the training one)
        text (str): raw email / message text

    Returns:
        float: spam probability in [0, 1]
    """
    logger.debug("Scoring text (%d chars)", len(text))

    # Transform text → TF-IDF feature vector (shape: 1 × max_features)
    vec = vectorizer.transform([text]).toarray()
    logger.debug("Feature vector shape: %s", vec.shape)

    # -vvv: show which words in the vocabulary actually matched the text
    if logger.isEnabledFor(TRACE):
        feature_names = vectorizer.get_feature_names_out()
        non_zero_idx  = vec[0].nonzero()[0]
        top_features  = sorted(
            [(feature_names[i], round(float(vec[0][i]), 4)) for i in non_zero_idx],
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        logger.trace(
            "%d/%d features fired | top 10 terms: %s",
            len(non_zero_idx),
            vec.shape[1],
            top_features,
        )

    # model.predict returns shape (1, 1); [0][0] extracts the scalar float
    prob = model.predict(vec)[0][0]
    logger.debug("Raw model output: %.6f", prob)

    return prob


def main():
    """CLI entry point — score a message passed as a command-line argument."""

    parser = argparse.ArgumentParser(
        description="Score a single message for spam likelihood."
    )
    add_verbosity_args(parser)
    parser.add_argument(
        "text",
        nargs="?",
        help="The email or message text to score.",
    )
    args = parser.parse_args()
    setup_logging(args.verbose)

    if not args.text:
        parser.print_help()
        return

    logger.info(
        "Input text (%d chars): %s",
        len(args.text),
        args.text[:80] + ("…" if len(args.text) > 80 else ""),
    )

    model, vectorizer = load_model_and_vectorizer()
    score = score_email(model, vectorizer, args.text)

    verdict = "SPAM" if score >= 0.5 else "HAM"
    logger.info("Result: %s (%.2f%% spam likelihood)", verdict, score * 100)

    # Always print the result regardless of verbosity
    print(f"Spam likelihood: {score:.2%}  [{verdict}]")


if __name__ == "__main__":
    main()
