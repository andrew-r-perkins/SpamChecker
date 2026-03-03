"""
tests/test_api.py — Integration tests for the Flask API layer.

Uses Flask's built-in test client with a mock model registry so no real
model files are needed and the suite runs in < 1 second.

What is tested:
  - /health  — status code and body
  - /config  — model list shape
  - Security headers — all 5 headers present and correct on every response
  - /predict — valid input, score type/range, determinism
  - /predict — invalid inputs: empty, whitespace, missing key, non-JSON, too long
  - 404 for unknown routes

Run with:
    pytest tests/test_api.py -v
    pytest tests/test_api.py -v -k headers
    pytest tests/ -v           # run alongside model tests (model tests are slow)
"""

import pytest

from api import create_app, MAX_TEXT_CHARS

# ---------------------------------------------------------------------------
# Mock registry — replaces real scorers with fast lambdas.
# The sentinel object stands in for the loaded model tuple; the scorer
# ignores it and always returns a fixed float.
# ---------------------------------------------------------------------------
MOCK_REGISTRY = {
    "tfidf": {
        "display": "TF-IDF + NN",
        "loader":  None,                          # never called in tests
        "scorer":  lambda obj, text: 0.9,         # always spam-ish
    },
}

MOCK_LOADED = {"tfidf": object()}   # opaque sentinel — scorer ignores it


# ---------------------------------------------------------------------------
# Fixture — one test client shared across all tests in this module.
# Module scope keeps setup cost to a single Flask app creation (~10 ms).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    app = create_app(preloaded=MOCK_LOADED, registry=MOCK_REGISTRY)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ===========================================================================
# /health
# ===========================================================================

def test_health_returns_200(client):
    res = client.get("/health")
    assert res.status_code == 200


def test_health_body(client):
    res = client.get("/health")
    assert res.get_json() == {"status": "ok"}


# ===========================================================================
# /config
# ===========================================================================

def test_config_returns_200(client):
    res = client.get("/config")
    assert res.status_code == 200


def test_config_returns_model_list(client):
    data = client.get("/config").get_json()
    assert "models" in data
    assert isinstance(data["models"], list)
    assert len(data["models"]) == 1
    model = data["models"][0]
    assert model["key"] == "tfidf"
    assert model["name"] == "TF-IDF + NN"


# ===========================================================================
# Security headers — checked on /health (applied to every route via
# @app.after_request, so testing one route is sufficient)
# ===========================================================================

EXPECTED_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options":        "DENY",
    "Referrer-Policy":        "no-referrer",
    "Content-Security-Policy": "default-src 'none'",
    "Permissions-Policy":     "geolocation=(), microphone=(), camera=()",
}


@pytest.mark.parametrize("header,expected", EXPECTED_SECURITY_HEADERS.items(),
                         ids=list(EXPECTED_SECURITY_HEADERS.keys()))
def test_security_header_present(client, header, expected):
    res = client.get("/health")
    actual = res.headers.get(header)
    assert actual == expected, (
        f"Header '{header}': expected {expected!r}, got {actual!r}"
    )


# ===========================================================================
# /predict — valid input
# ===========================================================================

def test_predict_valid_text_returns_200(client):
    res = client.post("/predict", json={"text": "Hello, this is a normal email."})
    assert res.status_code == 200


def test_predict_score_is_float_in_range(client):
    data = client.post("/predict",
                       json={"text": "Hello, this is a normal email."}).get_json()
    assert "tfidf" in data
    score = data["tfidf"]
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_predict_score_is_deterministic(client):
    """Scoring the same text twice should return exactly the same float."""
    text = "Identical text scored twice must give the same result."
    s1 = client.post("/predict", json={"text": text}).get_json()["tfidf"]
    s2 = client.post("/predict", json={"text": text}).get_json()["tfidf"]
    assert s1 == s2


# ===========================================================================
# /predict — invalid inputs
# ===========================================================================

def test_predict_empty_text_returns_400(client):
    res = client.post("/predict", json={"text": ""})
    assert res.status_code == 400


def test_predict_whitespace_only_returns_400(client):
    res = client.post("/predict", json={"text": "   \n\t  "})
    assert res.status_code == 400


def test_predict_missing_text_key_returns_400(client):
    """Empty JSON object should return 400, not 500.

    Previously this would crash with AttributeError because
    request.get_json() could return None and None.get() throws.
    Fixed by: data = request.get_json(silent=True) or {}
    """
    res = client.post("/predict", json={})
    assert res.status_code == 400


def test_predict_non_json_body_returns_400(client):
    """Plain-text body with wrong Content-Type should return 400, not 500.

    Same root fix as test_predict_missing_text_key_returns_400.
    """
    res = client.post("/predict", data="not json", content_type="text/plain")
    assert res.status_code == 400


def test_predict_text_too_long_returns_413(client):
    """Text exceeding MAX_TEXT_CHARS should be rejected with 413."""
    res = client.post("/predict", json={"text": "x" * (MAX_TEXT_CHARS + 1)})
    assert res.status_code == 413


def test_predict_413_body_has_error_key(client):
    """413 response should include a human-readable error message."""
    data = client.post("/predict",
                       json={"text": "x" * (MAX_TEXT_CHARS + 1)}).get_json()
    assert "error" in data


# ===========================================================================
# 404
# ===========================================================================

def test_unknown_route_returns_404(client):
    res = client.get("/nonexistent")
    assert res.status_code == 404
