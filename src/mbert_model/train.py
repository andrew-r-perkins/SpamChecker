"""
train.py — Fine-tunes mBERT for spam/ham classification using PyTorch.

mBERT (bert-base-multilingual-cased) is a 179M parameter transformer model
pre-trained on 104 languages. Fine-tuning it on our email dataset adapts its
general language understanding specifically to spam detection.

Full pipeline:
  1. Load and split the labelled email dataset
  2. Tokenize both train and test splits
  3. Load the pre-trained mBERT model
  4. Set up AdamW optimiser + linear warmup/decay learning rate scheduler
  5. Train for N epochs, logging average loss per epoch
  6. Evaluate on the test set after each epoch (accuracy + loss)
  7. Save the fine-tuned model and tokenizer to disk

Usage (run from project root):
  python -m src.mbert_model.train          # WARNING — only problems
  python -m src.mbert_model.train -v       # INFO    — milestones + accuracy
  python -m src.mbert_model.train -vv      # DEBUG   — shapes, batch counts, LR
  python -m src.mbert_model.train -vvv     # TRACE   — per-500-batch loss + LR

⚠️  Training time warning:
  CPU only : ~2–5 days for 2 epochs on 101K emails
  GPU      : ~2–4 hours
  Recommend running on Google Colab (free GPU) or a cloud instance.
"""

import argparse
import logging

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from src.log_config import add_verbosity_args, setup_logging, TRACE
from src.data_utils import load_and_prepare_data, split_data

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME      = "bert-base-multilingual-cased"
DATA_PATH       = "./data/spam_ham_dataset.csv"
SAVE_PATH       = "./models/mbert/"

NUM_EPOCHS      = 2      # 2 epochs is typically enough for BERT fine-tuning
BATCH_SIZE      = 16     # reduce to 8 if you hit GPU out-of-memory errors
LEARNING_RATE   = 5e-5   # standard starting LR for BERT fine-tuning
MAX_LENGTH      = 128    # truncate emails beyond 128 tokens
WARMUP_FRACTION = 0.1    # fraction of training steps used for LR warmup
GRAD_CLIP_NORM  = 1.0    # max gradient norm — prevents exploding gradients

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EmailDataset(Dataset):
    """
    Wraps HuggingFace tokenizer encodings and labels for use with PyTorch
    DataLoader.

    HuggingFace tokenizers return a dict of lists (not tensors), so
    __getitem__ converts each value to a tensor on the fly.
    """

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = list(labels)  # convert pandas Series → plain list

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Build a dict of tensors for this single example
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Labels must be torch.long (int64) for CrossEntropyLoss
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, loader, device):
    """
    Runs the model over a DataLoader in inference mode and returns
    accuracy and average loss.

    Uses torch.no_grad() to skip gradient computation — saves memory
    and speeds up evaluation significantly.

    Args:
        model:  fine-tuned AutoModelForSequenceClassification
        loader: DataLoader wrapping the test EmailDataset
        device: torch.device (cpu or cuda)

    Returns:
        (accuracy: float, avg_loss: float)
    """
    model.eval()
    correct    = 0
    total      = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            # Passing labels= causes the model to compute loss internally
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_loss += outputs.loss.item()

            # Pick the class with the highest logit as the prediction
            predictions = outputs.logits.argmax(dim=-1)
            correct     += (predictions == labels).sum().item()
            total       += labels.size(0)

    accuracy = correct / total
    avg_loss = total_loss / len(loader)

    logger.debug(
        "Evaluation complete — correct: %d/%d | avg_loss: %.4f",
        correct, total, avg_loss,
    )
    return accuracy, avg_loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ── Parse flags ──────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Fine-tune mBERT for spam classification"
    )
    add_verbosity_args(parser)
    args = parser.parse_args()
    setup_logging(args.verbose)

    # ── Step 1: Load + split data ─────────────────────────────────────────────
    logger.info("Loading data from %s", DATA_PATH)
    df = load_and_prepare_data(DATA_PATH)
    logger.info(
        "Dataset loaded — %d rows | spam: %d | ham: %d",
        len(df), df["label"].sum(), (df["label"] == 0).sum(),
    )

    logger.info("Splitting — 80%% train / 20%% test")
    X_train, X_test, y_train, y_test = split_data(df)
    logger.debug("Train: %d rows | Test: %d rows", len(X_train), len(X_test))

    # ── Step 2: Tokenize ─────────────────────────────────────────────────────
    # The tokenizer converts raw strings into token IDs and attention masks.
    # We use the same max_length, truncation, and padding for train and test
    # so the model always receives tensors of the same shape.
    logger.info("Loading tokenizer: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    logger.info("Tokenizing training set (%d examples)...", len(X_train))
    train_encodings = tokenizer(
        X_train.tolist(),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )

    logger.info("Tokenizing test set (%d examples)...", len(X_test))
    test_encodings = tokenizer(
        X_test.tolist(),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )

    # ── Step 3: Build DataLoaders ─────────────────────────────────────────────
    train_dataset = EmailDataset(train_encodings, y_train)
    test_dataset  = EmailDataset(test_encodings,  y_test)

    # shuffle=True for training — helps prevent the model memorising order
    # test_loader uses 2× batch size since no gradients are computed
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,     shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE * 2, shuffle=False)

    logger.debug(
        "DataLoaders ready — train batches: %d | test batches: %d",
        len(train_loader), len(test_loader),
    )

    # ── Step 4: Load model ───────────────────────────────────────────────────
    logger.info("Loading model: %s (num_labels=2)", MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,  # 0 = ham, 1 = spam
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    model.to(device)

    # ── Step 5: Optimiser + LR scheduler ─────────────────────────────────────
    # AdamW is Adam with decoupled weight decay — standard for transformer fine-tuning.
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # Linear warmup + linear decay:
    #   - LR ramps from 0 → LEARNING_RATE over the first 10% of steps (warmup)
    #   - LR decays linearly from LEARNING_RATE → 0 for the remaining steps
    # This prevents large weight updates early in training when the model is
    # still adapting its pre-trained weights to the new task.
    num_training_steps = NUM_EPOCHS * len(train_loader)
    num_warmup_steps   = int(num_training_steps * WARMUP_FRACTION)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    logger.debug(
        "Scheduler — total_steps: %d | warmup_steps: %d",
        num_training_steps, num_warmup_steps,
    )

    # ── Step 6: Training loop ────────────────────────────────────────────────
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0

        logger.info("Epoch %d/%d — training...", epoch + 1, NUM_EPOCHS)

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            # Forward pass — loss is computed internally when labels are provided
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            # Backward pass
            loss.backward()

            # Gradient clipping — caps the gradient norm to prevent large,
            # destabilising weight updates (common issue with deep transformers)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)

            optimizer.step()
            scheduler.step()  # advance the LR schedule

            epoch_loss += loss.item()

            # -vvv: log every 500 batches so training progress is visible
            # without flooding the console
            if logger.isEnabledFor(TRACE) and batch_idx % 500 == 0:
                logger.trace(
                    "Epoch %d | batch %d/%d | loss: %.4f | lr: %.2e",
                    epoch + 1,
                    batch_idx,
                    len(train_loader),
                    loss.item(),
                    scheduler.get_last_lr()[0],
                )

        avg_train_loss = epoch_loss / len(train_loader)
        logger.info(
            "Epoch %d/%d complete — avg train loss: %.4f",
            epoch + 1, NUM_EPOCHS, avg_train_loss,
        )

        # ── Evaluate after each epoch ─────────────────────────────────────────
        logger.info("Epoch %d/%d — evaluating on test set...", epoch + 1, NUM_EPOCHS)
        acc, eval_loss = evaluate(model, test_loader, device)
        logger.info(
            "Epoch %d/%d — test accuracy: %.2f%% | test loss: %.4f",
            epoch + 1, NUM_EPOCHS, acc * 100, eval_loss,
        )

    # ── Step 7: Save ─────────────────────────────────────────────────────────
    # save_pretrained() writes config.json, model weights, and vocab files.
    # The tokenizer must be saved alongside the model — it defines how text
    # is converted to token IDs at prediction time.
    logger.info("Saving model to %s", SAVE_PATH)
    model.save_pretrained(SAVE_PATH)

    logger.info("Saving tokenizer to %s", SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)

    logger.info("Training complete")


if __name__ == "__main__":
    main()
