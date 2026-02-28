"""
predict.py — Loads the fine-tuned mBERT model and scores individual messages.

Unlike the TF-IDF model, mBERT understands word context and order — it reads
the full sequence as a transformer, not a bag of words. This means it can
theoretically catch subtle spam patterns that TF-IDF misses (e.g. the Nigerian
prince case where no single word screams spam).

The model outputs two logits (one per class). We apply softmax to convert them
to probabilities and return the spam class probability (index 1).

Usage (CLI, run from project root):
  python -m src.mbert_model.predict          "Your message here"   # WARNING
  python -m src.mbert_model.predict -v       "Your message here"   # INFO
  python -m src.mbert_model.predict -vv      "Your message here"   # DEBUG
  python -m src.mbert_model.predict -vvv     "Your message here"   # TRACE

⚠️  Requires the model to have been trained first:
      python -m src.mbert_model.train -v

Score interpretation:
  0.0 = certainly ham
  0.5 = model is uncertain
  1.0 = certainly spam
"""

import argparse
import logging

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.log_config import add_verbosity_args, setup_logging, TRACE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SAVE_PATH  = "./models/mbert/"
MAX_LENGTH = 128  # must match the value used during training


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_model_and_tokenizer():
    """
    Loads the fine-tuned mBERT model and tokenizer from disk.

    Both are loaded from the same SAVE_PATH directory — HuggingFace's
    save_pretrained() writes config, weights, and vocab there together.

    model.eval() switches off dropout layers, which are only needed during
    training. Without this, predictions would be non-deterministic.

    Log output by verbosity:
      -vv  : confirms paths being loaded
      -vvv : model config (num layers, hidden size, etc.)

    Returns:
        (model, tokenizer) tuple, model in eval mode
    """
    logger.debug("Loading tokenizer from %s", SAVE_PATH)
    tokenizer = AutoTokenizer.from_pretrained(SAVE_PATH)

    logger.debug("Loading model from %s", SAVE_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(SAVE_PATH)

    # Switch to inference mode — disables dropout for deterministic output
    model.eval()

    if logger.isEnabledFor(TRACE):
        cfg = model.config
        logger.trace(
            "Model config — layers: %d | hidden_size: %d | num_labels: %d",
            cfg.num_hidden_layers,
            cfg.hidden_size,
            cfg.num_labels,
        )

    logger.info("Model and tokenizer loaded from %s", SAVE_PATH)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------

def score_email(model, tokenizer, text):
    """
    Scores a single piece of text and returns a spam probability.

    Pipeline:
      1. Tokenize the text — convert to token IDs + attention mask
      2. Forward pass through the model (no gradient tracking needed)
      3. Apply softmax to logits → probabilities for [ham, spam]
      4. Return the spam probability (index 1)

    Log output by verbosity:
      -vv  : token count, raw logits
      -vvv : full token list, per-class probabilities

    Args:
        model:     fine-tuned AutoModelForSequenceClassification (eval mode)
        tokenizer: matching AutoTokenizer
        text (str): raw email / message text

    Returns:
        float: spam probability in [0, 1]
    """
    logger.debug("Scoring text (%d chars)", len(text))

    # Tokenize — same settings as training so the model sees familiar input shapes
    inputs = tokenizer(
        text,
        return_tensors="pt",  # return PyTorch tensors
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )

    logger.debug("Token count: %d", inputs["input_ids"].shape[1])

    # -vvv: show the actual tokens the model will see — useful for understanding
    # why a borderline message was classified one way or the other
    if logger.isEnabledFor(TRACE):
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        logger.trace("Tokens: %s", tokens)

    # torch.no_grad() skips gradient tracking — reduces memory and speeds up inference
    with torch.no_grad():
        outputs = model(**inputs)

    logger.debug("Raw logits: %s", outputs.logits.tolist())

    # Softmax converts raw logits → probabilities that sum to 1.0
    # Shape: (1, 2) — [[ham_prob, spam_prob]]
    probs     = torch.softmax(outputs.logits, dim=-1)
    spam_prob = probs[0][1].item()  # index 1 = spam class

    if logger.isEnabledFor(TRACE):
        logger.trace(
            "Softmax probabilities — ham: %.4f | spam: %.4f",
            probs[0][0].item(),
            probs[0][1].item(),
        )

    return spam_prob


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Scores a message passed as a command-line argument."""
    parser = argparse.ArgumentParser(
        description="Score a message for spam likelihood using fine-tuned mBERT."
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

    model, tokenizer = load_model_and_tokenizer()
    score = score_email(model, tokenizer, args.text)

    verdict = "SPAM" if score >= 0.5 else "HAM"
    logger.info("Result: %s (%.2f%% spam likelihood)", verdict, score * 100)

    # Always print the result regardless of verbosity level
    print(f"Spam likelihood: {score:.2%}  [{verdict}]")


if __name__ == "__main__":
    main()
