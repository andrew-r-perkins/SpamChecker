# Ham / Spam Checker

A spam classifier that runs three machine-learning models side-by-side and scores
emails in real time. Paste any message and watch TF-IDF, mBERT, and DistilBERT
give their verdicts simultaneously.

**Stack:** Flask REST API · React + Vite frontend · TensorFlow · PyTorch · HuggingFace

<img width="720" height="873" alt="image" src="https://github.com/user-attachments/assets/8fc90ee5-7ed2-468d-a8a6-a9f1828b6448" />

---

## Models

| Model | Architecture | Test Accuracy | Speed |
|---|---|---|---|
| TF-IDF + NN | TF-IDF (5,000 features) → 64 → 32 → Sigmoid | ~98% | ~1 ms |
| mBERT | bert-base-multilingual-cased, 12-layer, 179M params | 98.84% | ~200 ms (CPU) |
| DistilBERT | distilbert-base-multilingual-cased, 6-layer, 66M params | 98.84% | ~2× faster than mBERT |

All three models were trained on a labelled dataset of 5,171 emails (71% ham,
29% spam). The BERT models were fine-tuned for 2 epochs on a Colab T4 GPU.

Dataset: [Spam Mails Dataset](https://www.kaggle.com/datasets/venky73/spam-mails-dataset) (Kaggle, CC0 Public Domain)

---

## Project Structure

```
SpamChecker/
├── data/
│   └── spam_ham_dataset.csv       # 5,171 labelled emails
├── models/
│   ├── tfidf_nn/                  # Keras model + TF-IDF vectorizer
│   ├── mbert/                     # Fine-tuned mBERT (gitignored)
│   └── distilbert/                # Fine-tuned DistilBERT (gitignored)
├── src/
│   ├── data_utils.py              # Shared data loading / splitting
│   ├── log_config.py              # Centralised logging (-v/-vv/-vvv)
│   ├── tfidf_model/               # train.py + predict.py
│   ├── mbert_model/               # train.py + predict.py
│   └── distilbert_model/          # train.py + predict.py
├── tests/
│   ├── test_spam_checker.py       # Model quality tests (48 tests)
│   └── test_api.py                # Flask integration tests (19 tests)
├── frontend/                      # React + Vite UI
└── api.py                         # Flask REST API
```

---

## Setup

**Requirements:** Python 3.10+, Node 18+

```bash
# Python dependencies
pip install -r requirements.txt

# Frontend dependencies
cd frontend && npm install
```

> **Note:** The BERT model binaries are excluded from the repository (large files).
> Train them yourself (see [Training](#training)) or load only the TF-IDF model.

---

## Running the App

```bash
# Start the API (all three models)
python api.py

# Start the API with a subset of models (useful on lower-spec machines)
python api.py --models tfidf                   # TF-IDF only
python api.py --models tfidf distilbert        # skip mBERT

# Start the frontend (in a separate terminal)
cd frontend && npm run dev
```

Open http://localhost:5173 in your browser. The UI scores text automatically
as you type (600 ms debounce).

---

## Training

All scripts are run from the project root as modules:

```bash
# TF-IDF + NN (fast, CPU-friendly)
python -m src.tfidf_model.train

# mBERT — recommended on GPU (Colab T4 trains in ~4 min)
python -m src.mbert_model.train

# DistilBERT — recommended on GPU (~2× faster than mBERT)
python -m src.distilbert_model.train
```

Verbosity flags are available on all scripts: `-v` INFO · `-vv` DEBUG · `-vvv` TRACE

---

## API Reference

Base URL: `http://localhost:5000`

### `GET /health`
Simple liveness check.
```json
{ "status": "ok" }
```

### `GET /config`
Returns the list of models currently loaded.
```json
{ "models": [{ "key": "tfidf", "name": "TF-IDF + NN" }, ...] }
```

### `POST /predict`
Scores a message against all active models.

**Request:**
```json
{ "text": "Congratulations! You have won a prize..." }
```

**Response:**
```json
{ "tfidf": 0.9821, "mbert": 0.9745, "distilbert": 0.9812 }
```

Values are spam probability floats in [0, 1]. A score ≥ 0.5 is classified as spam.

**Errors:**
| Status | Reason |
|---|---|
| 400 | Missing or empty `text` field |
| 413 | Text exceeds 10,000 characters, or request body exceeds 64 KB |
| 429 | Rate limit exceeded (30 requests/minute per IP) |
| 500 | Inference error |

---

## Security

- **Payload limits:** 64 KB hard limit (Flask) + 10,000 character soft limit on text
- **Rate limiting:** Flask-Limiter per IP — `/predict` 30/min, `/config` 10/min, `/health` 60/min
- **CORS:** restricted via `ALLOWED_ORIGINS` env var (default: `http://localhost:5173`)
- **HTTP security headers:** `X-Content-Type-Options`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, `Content-Security-Policy`, `Permissions-Policy`
- **Pinned dependencies:** all packages use exact versions (`==`) in both `requirements.txt` and `package.json`

To allow a different frontend origin:
```bash
ALLOWED_ORIGINS=https://yoursite.com python api.py
```

---

## Tests

```bash
# All tests (~5 min — loads all three models)
pytest tests/ -v

# API integration tests only (~2 min, no model files needed)
pytest tests/test_api.py -v

# Filter by model
pytest -k tfidf
pytest -k "mbert or distilbert"
```

The test suite has 67 tests total: 48 model quality tests and 19 API integration tests.
Three tests are marked `xfail` — known model limitations:

| Model | Case | Why it fails |
|---|---|---|
| TF-IDF | Nigerian prince scam | Natural language bypasses keyword triggers |
| TF-IDF | Legitimate job offer | "congratulations / click / 48h" triggers false positive |
| mBERT | Grandparent scam | Friendly tone scores ~1% spam despite being a scam |

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| Flask | 3.0.3 | REST API |
| flask-cors | 6.0.2 | CORS handling |
| flask-limiter | 4.1.1 | Rate limiting |
| TensorFlow | 2.20.0 | TF-IDF + NN model |
| PyTorch | 2.10.0 | BERT models |
| Transformers | 5.1.0 | HuggingFace model loading |
| scikit-learn | 1.8.0 | TF-IDF vectorizer |
| React | 18.3.1 | Frontend UI |
| Vite | 5.4.21 | Frontend build + dev server |
