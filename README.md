# JobShield AI — Fake Job Posting Detector

**30-second pitch:** Fresh graduates and students are getting scammed by fake job postings — "jobs" that ask for a registration fee, move contact to WhatsApp only, or promise $3,000/week with no experience. JobShield AI lets you paste a job posting and get back an ML fraud-risk score **plus the exact red flags in the text**, in under a second, from a model smaller than 1 MB.

## The problem

Online recruitment fraud targets the people least able to afford it: students and first-time job seekers. The warning signs are usually *in the posting itself* — fee requests, free-email recruiters, unrealistic pay, requests for bank/passport details — but a desperate job seeker reads past them. JobShield makes those signs impossible to miss, and backs them with a trained classifier.

## How it works

Two clearly separate layers:

1. **ML model (trained classifier)** — TF-IDF + Logistic Regression trained on 17,880 real job postings (the EMSCAD dataset). It returns a fraud probability for the pasted text.
2. **Rule-based red flags (transparent checks, NOT ML)** — plain pattern checks in `red_flags.py`: fee/payment requests, WhatsApp/Telegram-only contact, recruiter on a free email domain (Gmail/Yahoo), unrealistic salary / easy-money promises, requests for banking/personal documents, money-handling schemes, urgency pressure, and very short/vague postings. Each flag shows the exact snippet that triggered it and why it matters.

Keeping the layers separate is the point: the user can *verify* every red flag by reading the posting, instead of trusting a black-box score.

## Why this model?

| Choice | Reason |
|---|---|
| TF-IDF + Logistic Regression | Small, fast, interpretable (you can read the words driving the score), and strong on text classification with limited data |
| Model size **0.69 MB** | Deploys free almost anywhere; no GPU, no 200 MB downloads, no multi-minute phone inference |
| Inference **~1 ms** per posting (measured locally: 0.7 ms mean over 20 runs) | Instant result in the browser demo |
| `class_weight="balanced"` | The dataset is only 4.84% fraud — a naive model could score 95% accuracy by calling everything real |

## Results (held-out stratified test split — real numbers from `train.py`)

Dataset: 17,880 postings — 17,014 real, 866 fraudulent. Test split: 3,576 postings (173 fraudulent).

| Metric (fraud class) | Value |
|---|---|
| Precision | 0.729 |
| Recall | 0.919 |
| F1 | 0.813 |
| PR-AUC | 0.930 |
| ROC-AUC | 0.991 |
| Accuracy (reference only) | 0.980 |

Confusion matrix: TN 3,344 · FP 59 · FN 14 · TP 159. Full report: [`eval_report.md`](eval_report.md), machine-readable: `metrics.json`, raw run output: `train_run_output.txt`.

**Demo samples** (in `samples/`, also built into the web UI):
- Scam example (fee + WhatsApp + Gmail recruiter + bank details): **ML risk 94.6% — high, 6 red flags**
- Legit example (named company, real requirements, official careers page): **ML risk 26.2% — low, 0 red flags**

## Run it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# (re)train - needs the dataset in data/ (see below)
python train.py

# run the app
uvicorn app:app --port 8000
# open http://127.0.0.1:8000  - or POST /predict {"text": "<posting>"}

# tests
pytest -q
```

Dataset download (48 MB CSV, CC0 mirror of the EMSCAD / Kaggle dataset):

```bash
mkdir -p data
curl -L -o data/fake_job_postings.csv \
  https://huggingface.co/datasets/victor/real-or-fake-fake-jobposting-prediction/resolve/main/fake_job_postings.csv
```

Source page: https://huggingface.co/datasets/victor/real-or-fake-fake-jobposting-prediction

## API

`POST /predict`

```json
{"text": "URGENT HIRING!!! Earn $3000 per week..."}
```

returns `ml_fraud_probability`, `risk_score_percent`, `risk_level` (low/medium/high), a plain-language `verdict`, and `red_flags` (each with evidence + why it matters). Empty input returns HTTP 400.

## Honest limitations

- **English postings only**, and the training data is old (EMSCAD, collected for a 2017 study, mostly US/UK/AU). Modern WhatsApp-first scams in South Asia are under-represented in it — the rule layer exists partly to cover that gap, but neither layer is complete.
- **Text-only.** The original dataset's structured signals (company logo, screening questions) can't be used, because a pasted posting doesn't carry them.
- **A score is a signal, not proof.** In testing, a detailed legitimate posting still scored 26% — always verify an employer through its official website before applying, and never pay a fee or send ID/bank documents to get a job.

## Authorship

JobShield AI was built with AI assistance (code generation and debugging), and was directed, tested and validated by **Amin Khan** — dataset and model choices, red-flag definitions, evaluation design, and end-to-end verification of the app and its reported metrics are his responsibility.

## Project structure

```
app.py          FastAPI app + single-page UI (GET /, POST /predict, GET /health)
train.py        Training + evaluation; writes model.joblib, metrics.json, eval_report.md
red_flags.py    Transparent rule-based red-flag layer (separate from the ML model)
model.joblib    Trained pipeline (0.69 MB)
samples/        scammy.txt / legit.txt demo postings
tests/          pytest suite (7 tests)
data/           dataset CSV goes here (not committed - see download above)
```
