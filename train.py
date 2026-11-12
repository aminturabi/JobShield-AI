"""
JobShield AI - training script.

Trains a TF-IDF + Logistic Regression pipeline on the real EMSCAD /
fake_job_postings dataset and writes:
  - model.joblib   (the sklearn pipeline, small on purpose)
  - metrics.json   (machine-readable results)
  - eval_report.md (human-readable evaluation)

Run:
    python train.py [--data data/fake_job_postings.csv]

Dataset (downloaded separately, NOT fabricated):
    Hugging Face mirror of the classic Kaggle "Real or Fake Job Posting
    Prediction" / EMSCAD dataset:
    https://huggingface.co/datasets/victor/real-or-fake-fake-jobposting-prediction
    File: fake_job_postings.csv  (17,880 rows, `fraudulent` label)

Design choice: the app takes ONE pasted text blob at inference time, so the
model is trained on one concatenated text field per posting
(title + company_profile + description + requirements + benefits + ...).
Structured metadata (has_company_logo etc.) is deliberately not used,
because a user pasting a posting does not have it.
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).parent
TEXT_COLS = [
    "title",
    "location",
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
]
DATASET_URL = (
    "https://huggingface.co/datasets/victor/real-or-fake-fake-jobposting-prediction"
    "/resolve/main/fake_job_postings.csv"
)
DATASET_PAGE = "https://huggingface.co/datasets/victor/real-or-fake-fake-jobposting-prediction"


def build_text(df: pd.DataFrame) -> pd.Series:
    parts = []
    work = df.copy()
    for col in TEXT_COLS:
        work[col] = work[col].fillna("").astype(str)
    text = work[TEXT_COLS].agg(" ".join, axis=1)
    return text.str.replace(r"\s+", " ", regex=True).str.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "fake_job_postings.csv"))
    ap.add_argument("--out", default=str(ROOT / "model.joblib"))
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    assert "fraudulent" in df.columns, "dataset has no `fraudulent` column"
    n = len(df)
    counts = df["fraudulent"].value_counts().sort_index()
    n_fraud = int(counts.get(1, 0))
    n_real = int(counts.get(0, 0))

    X = build_text(df)
    y = df["fraudulent"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    max_features=15000,
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced", max_iter=1000, C=1.0, solver="lbfgs"
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    acc = float(accuracy_score(y_test, pred))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="binary", pos_label=1, zero_division=0
    )
    pr_auc = float(average_precision_score(y_test, proba))
    roc_auc = float(roc_auc_score(y_test, proba))
    cm = confusion_matrix(y_test, pred).tolist()  # [[TN, FP], [FN, TP]]
    report = classification_report(y_test, pred, digits=4, zero_division=0)

    joblib.dump(pipe, args.out)
    size_mb = Path(args.out).stat().st_size / (1024 * 1024)

    # Top fraud-indicative words (interpretability, from LR coefficients)
    vec = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    feats = vec.get_feature_names_out()
    coefs = clf.coef_[0]
    top_idx = coefs.argsort()[::-1][:20]
    bottom_idx = coefs.argsort()[:20]
    top_fraud = [str(feats[i]) for i in top_idx]
    top_real = [str(feats[i]) for i in bottom_idx]

    metrics = {
        "dataset_source": DATASET_PAGE,
        "dataset_file_url": DATASET_URL,
        "dataset_rows": n,
        "dataset_real_0": n_real,
        "dataset_fraud_1": n_fraud,
        "dataset_fraud_rate": round(n_fraud / n, 4),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_fraud_rows": int((y_test == 1).sum()),
        "accuracy": round(acc, 4),
        "fraud_precision": round(float(prec), 4),
        "fraud_recall": round(float(rec), 4),
        "fraud_f1": round(float(f1), 4),
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix_TN_FP_FN_TP": [cm[0][0], cm[0][1], cm[1][0], cm[1][1]],
        "model_file_size_mb": round(size_mb, 2),
    }
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    eval_md = f"""# JobShield AI — Evaluation Report

All numbers below were produced by actually running `train.py` in this repo.
Nothing is copied from other papers or projects.

## Dataset (real, public — not fabricated)

- **Name:** Employment Scam Aegean Dataset (EMSCAD) — the classic
  "Real or Fake Job Posting Prediction" dataset (17,880 real-life postings).
- **Downloaded from:** {DATASET_PAGE}
- **Direct file:** {DATASET_URL}
- **Rows:** {n:,}
- **Label balance:** real (0) = {n_real:,}, fraudulent (1) = {n_fraud:,} ({n_fraud / n:.2%} fraud)
- **Split:** stratified 80/20, `random_state=42` → train {len(X_train):,} rows, test {len(X_test):,} rows ({int((y_test == 1).sum())} fraudulent in test)

## Model

- Pipeline: TF-IDF (unigrams+bigrams, max 15,000 features, sublinear TF)
  + Logistic Regression (`class_weight="balanced"` to handle the imbalance).
- Input: one concatenated text field per posting (title, company profile,
  description, requirements, benefits, employment type, experience,
  education, industry, function, location) — the same single-text input
  the app receives when a user pastes a posting.
- Model file: `model.joblib` — **{size_mb:.2f} MB** (deliberately small: instant
  inference, deploys on free tiers).

## Held-out test results (fraud class = the class that matters)

| Metric | Value |
|---|---|
| Fraud precision | {float(prec):.4f} |
| Fraud recall | {float(rec):.4f} |
| Fraud F1 | {float(f1):.4f} |
| PR-AUC (average precision) | {pr_auc:.4f} |
| ROC-AUC | {roc_auc:.4f} |
| Accuracy (for reference only — misleading on imbalanced data) | {acc:.4f} |

Confusion matrix on the test split: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}.

Full classification report:

```
{report}
```

## Most fraud-indicative words learned by the model

{", ".join(top_fraud)}

## Most real-indicative words learned by the model

{", ".join(top_real)}

## Honest limitations

- The dataset is old (collected for the 2017 EMSCAD study, mostly US/UK/AU
  English postings). Modern scams (e.g. WhatsApp-first scams in Pakistan /
  South Asia) use patterns under-represented in it — which is exactly why
  the app adds a separate, transparent rule-based red-flag layer on top.
- Text-only: the app does not use structured signals from the original
  dataset (company logo, screening questions), because a pasted posting
  does not carry them.
- A score is a risk signal, not proof. A legitimate small company posting
  can score medium; a polished scam can score low. Always verify through
  the company's official website.
"""
    (ROOT / "eval_report.md").write_text(eval_md)

    print(json.dumps(metrics, indent=2))
    print("\nClassification report:\n", report)
    print(f"Saved model to {args.out} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
