"""
predict.py — Loads the fine-tuned DistilBERT model and scores individual messages.

DistilBERT (distilbert-base-multilingual-cased) is a distilled version of mBERT:
  - 40% fewer parameters (66M vs 179M)
  - ~2× faster inference
  - Retains ~97% of mBERT accuracy on downstream tasks
  - Same 104-language coverage, same tokenizer approach

The model outputs two logits (one per class). We apply softmax to convert them
to probabilities and return the spam class probability (index 1).

Usage (CLI, run from project root):
  python -m src.distilbert_model.predict          "Your message here"
  python -m src.distilbert_model.predict -v       "Your message here"   # INFO
  python -m src.distilbert_model.predict -vv      "Your message here"   # DEBUG
  python -m src.distilbert_model.predict -vvv     "Your message here"   # TRACE

Requires the model to have been trained first:
  python -m src.distilbert_model.train -v

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
SAVE_PATH   = "./models/distilbert/"
MAX_LENGTH  = 128   # must match the value used during training
MODEL_LABEL = "DistilBERT"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_model_and_tokenizer():
    """
    Loads the fine-tuned DistilBERT model and tokenizer from disk.

    Both are loaded from the same SAVE_PATH directory — HuggingFace's
    save_pretrained() writes config, weights, and vocab there together.

    model.eval() switches off dropout layers for deterministic inference.

    Returns:
        (model, tokenizer) tuple, model in eval mode
    """
    logger.debug("Loading %s tokenizer from %s", MODEL_LABEL, SAVE_PATH)
    tokenizer = AutoTokenizer.from_pretrained(SAVE_PATH)

    logger.debug("Loading %s model from %s", MODEL_LABEL, SAVE_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(SAVE_PATH)

    model.eval()

    if logger.isEnabledFor(TRACE):
        cfg = model.config
        logger.trace(
            "%s config — layers: %d | hidden_size: %d | num_labels: %d",
            MODEL_LABEL,
            cfg.num_hidden_layers,
            cfg.hidden_size,
            cfg.num_labels,
        )

    logger.info("%s model and tokenizer loaded from %s", MODEL_LABEL, SAVE_PATH)
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

    Args:
        model:     fine-tuned AutoModelForSequenceClassification (eval mode)
        tokenizer: matching AutoTokenizer
        text (str): raw email / message text

    Returns:
        float: spam probability in [0, 1]
    """
    logger.debug("Scoring text (%d chars) with %s", len(text), MODEL_LABEL)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )

    logger.debug("Token count: %d", inputs["input_ids"].shape[1])

    if logger.isEnabledFor(TRACE):
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        logger.trace("Tokens: %s", tokens)

    with torch.no_grad():
        outputs = model(**inputs)

    logger.debug("Raw logits: %s", outputs.logits.tolist())

    probs     = torch.softmax(outputs.logits, dim=-1)
    spam_prob = probs[0][1].item()

    if logger.isEnabledFor(TRACE):
        logger.trace(
            "%s softmax — ham: %.4f | spam: %.4f",
            MODEL_LABEL,
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
        description=f"Score a message for spam likelihood using fine-tuned {MODEL_LABEL}."
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

    print(f"Spam likelihood: {score:.2%}  [{verdict}]")


if __name__ == "__main__":
    main()
