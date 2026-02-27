"""
train.py — Trains the TF-IDF + Dense Neural Network spam classifier.

Full pipeline:
  1. Load raw labelled email data from CSV
  2. Split into 80% train / 20% test
  3. Vectorize text with TF-IDF (vocabulary learned from training set only)
  4. Build a 3-layer dense neural network
  5. Train for 5 epochs with binary cross-entropy loss
  6. Evaluate accuracy on the held-out test set
  7. Save model weights (.keras) and the fitted vectorizer (.joblib)

Usage (run from project root):
  python -m src.tfidf_model.train          # WARNING level
  python -m src.tfidf_model.train -v       # INFO  — step-by-step milestones
  python -m src.tfidf_model.train -vv      # DEBUG — shapes, sizes, class split
  python -m src.tfidf_model.train -vvv     # TRACE — sample rows, vocabulary, model summary
"""

import argparse
import logging

import tensorflow as tf
from tensorflow.keras import layers
import joblib

from src.log_config import add_verbosity_args, setup_logging, TRACE
from src.data_utils import load_and_prepare_data, split_data, vectorize_text

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH       = "./data/spam_ham_dataset.csv"
MODEL_PATH      = "./models/tfidf_nn/spam_model.keras"
VECTORIZER_PATH = "./models/tfidf_nn/vectorizer.joblib"

logger = logging.getLogger(__name__)


def build_model(input_dim):
    """
    Constructs and compiles the neural network.

    Architecture:
      Input → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Sigmoid)

    Why this shape:
      Two hidden layers provide enough capacity to learn spam patterns
      without overfitting on ~100K samples.  Sigmoid output maps to a
      probability in [0, 1]: 0 = ham, 1 = spam.

    Loss:      binary_crossentropy — standard for binary classification.
    Optimiser: Adam — adaptive learning rate, robust default.

    Args:
        input_dim (int): number of TF-IDF features (vocabulary size cap).

    Returns:
        Compiled tf.keras.Sequential model.
    """
    logger.debug("Building model — input_dim=%d", input_dim)

    model = tf.keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),   # first hidden layer
        layers.Dense(32, activation="relu"),   # second hidden layer
        layers.Dense(1, activation="sigmoid"), # output: spam probability [0,1]
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # -vvv: print layer-by-layer model summary
    if logger.isEnabledFor(TRACE):
        model.summary(print_fn=lambda line: logger.trace("  %s", line))

    return model


def main():
    # ── Parse flags ──────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Train the TF-IDF spam classifier")
    add_verbosity_args(parser)
    args = parser.parse_args()
    setup_logging(args.verbose)

    # ── Step 1: Load data ────────────────────────────────────────────────────
    logger.info("Loading data from %s", DATA_PATH)
    df = load_and_prepare_data(DATA_PATH)
    logger.info(
        "Dataset loaded — %d rows | spam: %d | ham: %d",
        len(df), df["label"].sum(), (df["label"] == 0).sum(),
    )

    # -vvv: show the first few rows so we can sanity-check column formats
    if logger.isEnabledFor(TRACE):
        logger.trace("First 3 rows:\n%s", df.head(3).to_string())

    # ── Step 2: Train/test split ─────────────────────────────────────────────
    logger.info("Splitting data — 80%% train / 20%% test (random_state=42)")
    X_train, X_test, y_train, y_test = split_data(df)
    logger.debug("Train: %d rows | Test: %d rows", len(X_train), len(X_test))

    # ── Step 3: Vectorize ────────────────────────────────────────────────────
    # IMPORTANT: fit_transform on train ONLY to prevent data leakage.
    logger.info("Vectorizing text with TF-IDF")
    X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)
    logger.debug(
        "Feature matrix — train: %s | test: %s",
        X_train_vec.shape, X_test_vec.shape,
    )

    # -vvv: show a sample of the vocabulary that was learned
    if logger.isEnabledFor(TRACE):
        sample_vocab = list(vectorizer.vocabulary_.keys())[:20]
        logger.trace("Vocabulary sample (first 20 terms): %s", sample_vocab)

    # ── Step 4: Build model ──────────────────────────────────────────────────
    logger.info("Building neural network")
    model = build_model(X_train_vec.shape[1])

    # ── Step 5: Train ────────────────────────────────────────────────────────
    # validation_split=0.1 reserves 10% of training data to monitor val_loss
    # each epoch so we can detect overfitting early.
    logger.info("Training — epochs=5, batch_size=32, validation_split=0.1")
    model.fit(
        X_train_vec,
        y_train,
        validation_split=0.1,
        epochs=5,
        batch_size=32,
    )

    # ── Step 6: Evaluate ─────────────────────────────────────────────────────
    logger.info("Evaluating on held-out test set")
    loss, acc = model.evaluate(X_test_vec, y_test)
    logger.info("Test loss: %.4f | Test accuracy: %.2f%%", loss, acc * 100)

    # ── Step 7: Save artifacts ───────────────────────────────────────────────
    # Both artifacts must be saved together and reloaded as a matched pair.
    logger.info("Saving model to %s", MODEL_PATH)
    model.save(MODEL_PATH)

    logger.info("Saving vectorizer to %s", VECTORIZER_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    logger.info("Training complete")


if __name__ == "__main__":
    main()
