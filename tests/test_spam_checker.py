"""
Standard test cases for the TF-IDF spam classifier.

Run from the project root:
    pytest tests/ -v
"""

import pytest
from src.tfidf_model.predict import load_model_and_vectorizer, score_email

# ---------------------------------------------------------------------------
# Model fixture — loaded once for the entire test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model_and_vectorizer():
    return load_model_and_vectorizer()


# ---------------------------------------------------------------------------
# Test cases
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
# Tests
# ---------------------------------------------------------------------------

SPAM_THRESHOLD = 0.70   # score must be at or above this to pass
HAM_THRESHOLD  = 0.30   # score must be at or below this to pass


@pytest.mark.parametrize("name,text", SPAM_CASES, ids=[c[0] for c in SPAM_CASES])
def test_spam_detected(model_and_vectorizer, name, text):
    model, vectorizer = model_and_vectorizer
    score = score_email(model, vectorizer, text)
    assert score >= SPAM_THRESHOLD, (
        f"[{name}] Expected spam score >= {SPAM_THRESHOLD:.0%}, got {score:.2%}"
    )


@pytest.mark.parametrize("name,text", HAM_CASES, ids=[c[0] for c in HAM_CASES])
def test_ham_detected(model_and_vectorizer, name, text):
    model, vectorizer = model_and_vectorizer
    score = score_email(model, vectorizer, text)
    assert score <= HAM_THRESHOLD, (
        f"[{name}] Expected ham score <= {HAM_THRESHOLD:.0%}, got {score:.2%}"
    )


# ---------------------------------------------------------------------------
# Known model limitations (documented but not enforced)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="Model scores 'Nigerian prince' style emails as ham (~10%). "
                           "Likely underrepresented in training data.")
def test_known_limitation_nigerian_prince(model_and_vectorizer):
    model, vectorizer = model_and_vectorizer
    text = "Dear Friend, I am a Nigerian prince and I need your help transferring $15,000,000. You will receive 30%."
    score = score_email(model, vectorizer, text)
    assert score >= SPAM_THRESHOLD
