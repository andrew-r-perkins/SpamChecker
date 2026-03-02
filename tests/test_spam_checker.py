"""
Tests for all three spam classifiers: TF-IDF + NN, mBERT, and DistilBERT.

Run from the project root:
    pytest tests/ -v                         # all tests (slow — loads all 3 models)
    pytest tests/ -v -k tfidf               # TF-IDF only (fast)
    pytest tests/ -v -k "mbert or distilbert"  # transformer tests only
    pytest tests/ -v -k "not slow"          # skip known-slow transformer tests

Each model is loaded once per session via module-scoped fixtures.
Transformer forward passes take ~200 ms each on CPU, so the full suite
runs in roughly 2–3 minutes.

Thresholds
----------
  SPAM_THRESHOLD = 0.70  — score must be >= this to count as spam detected
  HAM_THRESHOLD  = 0.30  — score must be <= this to count as ham detected

These are intentionally conservative (not 0.5) to catch cases where a model
is merely uncertain rather than confidently correct.
"""

import pytest

from src.tfidf_model.predict     import (load_model_and_vectorizer,
                                          score_email as tfidf_score)
from src.mbert_model.predict     import (load_model_and_tokenizer  as load_mbert,
                                          score_email               as mbert_score)
from src.distilbert_model.predict import (load_model_and_tokenizer as load_distilbert,
                                           score_email              as distilbert_score)

# ---------------------------------------------------------------------------
# Shared thresholds
# ---------------------------------------------------------------------------
SPAM_THRESHOLD = 0.70
HAM_THRESHOLD  = 0.30

# ---------------------------------------------------------------------------
# Shared test cases — reused across all three models
# ---------------------------------------------------------------------------

SPAM_CASES = [
    (
        "prize_winner",
        "WINNER! You have been selected for a FREE iPhone! Click here NOW to claim your prize before it expires!!!",
    ),
    (
        "lottery_win",
        "Congratulations! You have won £1,000,000 in our international lottery! Send your bank details to claim.",
    ),
    (
        "work_from_home",
        "Make $5000 a day working from home! No experience needed. Guaranteed income. Click here to start!",
    ),
    (
        "account_suspended",
        "URGENT: Your account has been suspended. Verify your personal details immediately or lose access forever.",
    ),
    (
        "online_pharmacy",
        "Buy Viagra online! No prescription needed. Lowest prices guaranteed. Discreet shipping. Order now!",
    ),
    (
        "free_gift_card",
        "FREE $500 gift card! You have been specially selected. Claim your reward NOW before it expires. Click here!!!",
    ),
    (
        "weight_loss",
        "LOSE 30 POUNDS IN 30 DAYS! Doctors hate this secret trick. Click now for your FREE trial bottle!!!",
    ),
]

HAM_CASES = [
    (
        "meeting_request",
        "Hi, just checking in about our meeting tomorrow at 3pm. Does the conference room still work for you?",
    ),
    (
        "work_feedback",
        "Can you please review the attached report and send me your feedback before the end of the week?",
    ),
    (
        "casual_catchup",
        "Thanks for dinner last night, it was really great catching up. We should do it again soon!",
    ),
    (
        "deadline_update",
        "Just a heads up that the project deadline has been moved to next Friday due to the bank holiday.",
    ),
    (
        "invoice",
        "Please find attached the invoice for last month's consultancy services. Payment terms are 30 days.",
    ),
    (
        "interview_confirmation",
        "Thank you for applying. We would like to invite you to an interview on Thursday at 10am.",
    ),
    (
        "password_reset",
        "You recently requested a password reset for your account. Click the link below to set a new password.",
    ),
]

# ---------------------------------------------------------------------------
# Fixtures — each model loaded once per test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tfidf():
    return load_model_and_vectorizer()


@pytest.fixture(scope="module")
def mbert():
    return load_mbert()


@pytest.fixture(scope="module")
def distilbert():
    return load_distilbert()


# ===========================================================================
# TF-IDF + NN
# ===========================================================================

@pytest.mark.parametrize("name,text", SPAM_CASES, ids=[c[0] for c in SPAM_CASES])
def test_tfidf_spam_detected(tfidf, name, text):
    model, vectorizer = tfidf
    score = tfidf_score(model, vectorizer, text)
    assert score >= SPAM_THRESHOLD, (
        f"[tfidf/{name}] Expected spam score >= {SPAM_THRESHOLD:.0%}, got {score:.2%}"
    )


@pytest.mark.parametrize("name,text", HAM_CASES, ids=[c[0] for c in HAM_CASES])
def test_tfidf_ham_detected(tfidf, name, text):
    model, vectorizer = tfidf
    score = tfidf_score(model, vectorizer, text)
    assert score <= HAM_THRESHOLD, (
        f"[tfidf/{name}] Expected ham score <= {HAM_THRESHOLD:.0%}, got {score:.2%}"
    )


@pytest.mark.xfail(reason=(
    "TF-IDF scores 'Nigerian prince' style emails as ham (~10%). "
    "Natural-sounding social engineering lacks the keyword triggers the model relies on."
))
def test_tfidf_known_limitation_nigerian_prince(tfidf):
    model, vectorizer = tfidf
    text = "Dear Friend, I am a Nigerian prince and I need your help transferring $15,000,000. You will receive 30%."
    score = tfidf_score(model, vectorizer, text)
    assert score >= SPAM_THRESHOLD


@pytest.mark.xfail(reason=(
    "TF-IDF over-triggers on legitimate job offer emails (~99.8% spam). "
    "Words like 'click', '48 hours', 'congratulations' fire regardless of context."
))
def test_tfidf_known_limitation_job_offer(tfidf):
    model, vectorizer = tfidf
    text = (
        "Hi Alex, Congratulations! We are pleased to formally offer you the Software Engineer "
        "position. This offer is time-sensitive — we need your acceptance within 48 hours. "
        "Please click the link below to sign your offer letter. Best regards, Claire Sutton"
    )
    score = tfidf_score(model, vectorizer, text)
    assert score <= HAM_THRESHOLD


# ===========================================================================
# mBERT
# ===========================================================================

@pytest.mark.parametrize("name,text", SPAM_CASES, ids=[c[0] for c in SPAM_CASES])
def test_mbert_spam_detected(mbert, name, text):
    model, tokenizer = mbert
    score = mbert_score(model, tokenizer, text)
    assert score >= SPAM_THRESHOLD, (
        f"[mbert/{name}] Expected spam score >= {SPAM_THRESHOLD:.0%}, got {score:.2%}"
    )


@pytest.mark.parametrize("name,text", HAM_CASES, ids=[c[0] for c in HAM_CASES])
def test_mbert_ham_detected(mbert, name, text):
    model, tokenizer = mbert
    score = mbert_score(model, tokenizer, text)
    assert score <= HAM_THRESHOLD, (
        f"[mbert/{name}] Expected ham score <= {HAM_THRESHOLD:.0%}, got {score:.2%}"
    )


@pytest.mark.xfail(reason=(
    "mBERT is fooled by social engineering written in natural language. "
    "The grandparent scam reads like a genuine personal message and scores ~1% spam. "
    "mBERT learned that friendly, conversational tone = ham."
))
def test_mbert_known_limitation_grandparent_scam(mbert):
    model, tokenizer = mbert
    text = (
        "Hi Nan, it is me, your grandson Jake. I am in a bit of trouble and too embarrassed "
        "to call Mum. I had an accident in the car abroad and I need some money to pay the "
        "local garage. It is only 400 pounds. Can you send it to my friend's account? "
        "Please do not tell anyone just yet. Love, Jake"
    )
    score = mbert_score(model, tokenizer, text)
    assert score >= SPAM_THRESHOLD


def test_mbert_nigerian_prince(mbert):
    """mBERT correctly catches Nigerian prince emails — unlike TF-IDF which misses them."""
    model, tokenizer = mbert
    text = "Dear Friend, I am a Nigerian prince and I need your help transferring $15,000,000. You will receive 30%."
    score = mbert_score(model, tokenizer, text)
    assert score >= SPAM_THRESHOLD, (
        f"[mbert/nigerian_prince] Expected spam score >= {SPAM_THRESHOLD:.0%}, got {score:.2%}"
    )


# ===========================================================================
# DistilBERT
# ===========================================================================

@pytest.mark.parametrize("name,text", SPAM_CASES, ids=[c[0] for c in SPAM_CASES])
def test_distilbert_spam_detected(distilbert, name, text):
    model, tokenizer = distilbert
    score = distilbert_score(model, tokenizer, text)
    assert score >= SPAM_THRESHOLD, (
        f"[distilbert/{name}] Expected spam score >= {SPAM_THRESHOLD:.0%}, got {score:.2%}"
    )


@pytest.mark.parametrize("name,text", HAM_CASES, ids=[c[0] for c in HAM_CASES])
def test_distilbert_ham_detected(distilbert, name, text):
    model, tokenizer = distilbert
    score = distilbert_score(model, tokenizer, text)
    assert score <= HAM_THRESHOLD, (
        f"[distilbert/{name}] Expected ham score <= {HAM_THRESHOLD:.0%}, got {score:.2%}"
    )


def test_distilbert_grandparent_scam(distilbert):
    """DistilBERT catches the grandparent scam — outperforming mBERT on this case."""
    model, tokenizer = distilbert
    text = (
        "Hi Nan, it is me, your grandson Jake. I am in a bit of trouble and too embarrassed "
        "to call Mum. I had an accident in the car abroad and I need some money to pay the "
        "local garage. It is only 400 pounds. Can you send it to my friend's account? "
        "Please do not tell anyone just yet. Love, Jake"
    )
    score = distilbert_score(model, tokenizer, text)
    assert score >= SPAM_THRESHOLD, (
        f"[distilbert/grandparent_scam] Expected spam score >= {SPAM_THRESHOLD:.0%}, got {score:.2%}"
    )


def test_distilbert_nigerian_prince(distilbert):
    """DistilBERT correctly catches Nigerian prince emails."""
    model, tokenizer = distilbert
    text = "Dear Friend, I am a Nigerian prince and I need your help transferring $15,000,000. You will receive 30%."
    score = distilbert_score(model, tokenizer, text)
    assert score >= SPAM_THRESHOLD, (
        f"[distilbert/nigerian_prince] Expected spam score >= {SPAM_THRESHOLD:.0%}, got {score:.2%}"
    )
