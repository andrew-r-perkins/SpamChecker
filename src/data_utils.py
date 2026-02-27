"""
data_utils.py — Shared data loading and preprocessing utilities.

Used exclusively during MODEL TRAINING (train.py).
Not needed at prediction time — by then the vectorizer is already fitted
and saved to disk.

High-level pipeline:
  1. load_and_prepare_data() — read CSV, normalise labels to 0/1
  2. split_data()            — 80/20 train/test split
  3. vectorize_text()        — fit TF-IDF on train, transform both splits

Why this file exists:
  - Keeps train.py clean and readable
  - Makes each data step independently testable
  - Separates "data work" from "model work"

IMPORTANT:
  The SAME vectorizer produced by vectorize_text() during training must be
  saved and reused at prediction time.  A new vectorizer fitted on different
  data would produce misaligned feature columns, giving meaningless results.

Log output by verbosity (set in the calling entry point — train.py):
  -v   : row count, label distribution
  -vv  : column list, shapes, vocabulary size
  -vvv : sample rows, first vocabulary terms, label value counts
"""

import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Importing log_config registers the TRACE level and .trace() method
# on logging.Logger — this must happen before any .trace() calls below.
from src.log_config import TRACE  # noqa: F401  (side-effect import)

logger = logging.getLogger(__name__)


def load_and_prepare_data(filepath):
    """
    Loads the raw dataset from disk and prepares it for training.

    Reads a CSV with at least two columns:
      - 'text'  : the raw email/message string
      - 'label' : 'ham' or 'spam'

    Steps:
      1. Read CSV into a DataFrame
      2. Drop the auto-generated 'Unnamed: 0' index column if present
      3. Map string labels → integers: ham → 0, spam → 1

    Why integer labels?
      Neural networks and sklearn metrics expect numeric targets, not strings.

    Log output by verbosity:
      -v   : row count, spam/ham totals
      -vv  : raw shape, column names
      -vvv : first 3 rows, label value counts

    Args:
        filepath (str): path to the CSV file (relative to project root)

    Returns:
        pd.DataFrame with columns ['text', 'label'] where label ∈ {0, 1}
    """
    logger.info("Reading dataset from %s", filepath)
    df = pd.read_csv(filepath)

    logger.debug("Raw shape: %s | Columns: %s", df.shape, list(df.columns))

    # Drop the auto-index column pandas sometimes writes during to_csv()
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    # Convert string labels to integers:
    #   ham  → 0  (legitimate mail)
    #   spam → 1  (unwanted mail)
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    # Warn if any rows have unmapped labels (they become NaN)
    null_count = df["label"].isna().sum()
    if null_count:
        logger.warning(
            "%d rows have unrecognised labels and will be dropped", null_count
        )
        df = df.dropna(subset=["label"])

    logger.info(
        "Dataset ready — %d rows | ham: %d | spam: %d",
        len(df),
        (df["label"] == 0).sum(),
        (df["label"] == 1).sum(),
    )

    # -vvv: print the first 3 rows so we can sanity-check the raw content
    if logger.isEnabledFor(TRACE):
        logger.trace("Label value counts:\n%s", df["label"].value_counts().to_string())
        logger.trace("First 3 rows:\n%s", df.head(3).to_string())

    return df


def split_data(df):
    """
    Splits the DataFrame into training and testing sets (80/20).

    Why split at all?
      We must evaluate the model on data it has NEVER seen during training.
      This detects overfitting — a model that has memorised the training set
      rather than genuinely learned to detect spam.

    random_state=42 makes the split reproducible across runs.

    Log output by verbosity:
      -v   : split ratio confirmation
      -vv  : exact train/test row counts
      -vvv : label distribution within each split

    Args:
        df (pd.DataFrame): DataFrame with 'text' and 'label' columns

    Returns:
        X_train, X_test   — email text (pandas Series)
        y_train, y_test   — integer labels (pandas Series)
    """
    logger.info("Splitting data — 80%% train / 20%% test (random_state=42)")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,  # fixed seed → same split every run
    )

    logger.debug(
        "Split complete — train: %d rows | test: %d rows",
        len(X_train), len(X_test),
    )

    # -vvv: confirm spam/ham balance isn't skewed after splitting
    if logger.isEnabledFor(TRACE):
        logger.trace(
            "Train label distribution: spam=%d ham=%d",
            y_train.sum(), (y_train == 0).sum(),
        )
        logger.trace(
            "Test  label distribution: spam=%d ham=%d",
            y_test.sum(), (y_test == 0).sum(),
        )

    return X_train, X_test, y_train, y_test


def vectorize_text(X_train, X_test):
    """
    Converts raw email strings into fixed-length numeric feature vectors
    using TF-IDF (Term Frequency – Inverse Document Frequency).

    What TF-IDF does:
      For each word in the vocabulary it computes a score reflecting:
        - How often the word appears in THIS email (term frequency)
        - How rare the word is across ALL emails (inverse document frequency)
      Common English function words ("the", "a", "is") are filtered out
      because they carry no spam/ham signal.

    Key parameters:
      max_features=5000    — keep only the 5,000 most informative words
      stop_words="english" — skip English function words

    CRITICAL — fit on train only:
      vectorizer.fit_transform(X_train) ← learns vocabulary from training data
      vectorizer.transform(X_test)      ← applies that vocabulary to test data
      This prevents data leakage.

    Log output by verbosity:
      -v   : confirms vectorization complete
      -vv  : vocabulary size, feature matrix shapes
      -vvv : first 20 vocabulary terms learned

    Args:
        X_train: training email strings (pandas Series or list)
        X_test:  test email strings     (pandas Series or list)

    Returns:
        X_train_vec (np.ndarray): dense feature matrix for training
        X_test_vec  (np.ndarray): dense feature matrix for testing
        vectorizer:               the fitted TfidfVectorizer (MUST be saved)
    """
    logger.info("Fitting TF-IDF vectorizer on training data (max_features=5000)")

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
    )

    # fit_transform: learn vocabulary AND convert training emails to vectors
    X_train_vec = vectorizer.fit_transform(X_train).toarray()

    # transform only: apply the already-learned vocabulary to test emails
    X_test_vec = vectorizer.transform(X_test).toarray()

    logger.debug(
        "Vocabulary size: %d | train matrix: %s | test matrix: %s",
        len(vectorizer.vocabulary_),
        X_train_vec.shape,
        X_test_vec.shape,
    )

    # -vvv: show a sample of the vocabulary terms that were retained
    if logger.isEnabledFor(TRACE):
        sample_terms = list(vectorizer.vocabulary_.keys())[:20]
        logger.trace("Vocabulary sample (first 20 terms): %s", sample_terms)

    logger.info("Vectorization complete")
    return X_train_vec, X_test_vec, vectorizer
