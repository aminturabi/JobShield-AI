# JobShield AI — Evaluation Report

All numbers below were produced by actually running `train.py` in this repo.
Nothing is copied from other papers or projects.

## Dataset (real, public — not fabricated)

- **Name:** Employment Scam Aegean Dataset (EMSCAD) — the classic
  "Real or Fake Job Posting Prediction" dataset (17,880 real-life postings).
- **Downloaded from:** https://huggingface.co/datasets/victor/real-or-fake-fake-jobposting-prediction
- **Direct file:** https://huggingface.co/datasets/victor/real-or-fake-fake-jobposting-prediction/resolve/main/fake_job_postings.csv
- **Rows:** 17,880
- **Label balance:** real (0) = 17,014, fraudulent (1) = 866 (4.84% fraud)
- **Split:** stratified 80/20, `random_state=42` → train 14,304 rows, test 3,576 rows (173 fraudulent in test)

## Model

- Pipeline: TF-IDF (unigrams+bigrams, max 15,000 features, sublinear TF)
  + Logistic Regression (`class_weight="balanced"` to handle the imbalance).
- Input: one concatenated text field per posting (title, company profile,
  description, requirements, benefits, employment type, experience,
  education, industry, function, location) — the same single-text input
  the app receives when a user pastes a posting.
- Model file: `model.joblib` — **0.69 MB** (deliberately small: instant
  inference, deploys on free tiers).

## Held-out test results (fraud class = the class that matters)

| Metric | Value |
|---|---|
| Fraud precision | 0.7294 |
| Fraud recall | 0.9191 |
| Fraud F1 | 0.8133 |
| PR-AUC (average precision) | 0.9297 |
| ROC-AUC | 0.9907 |
| Accuracy (for reference only — misleading on imbalanced data) | 0.9796 |

Confusion matrix on the test split: TN=3344, FP=59, FN=14, TP=159.

Full classification report:

```
              precision    recall  f1-score   support

           0     0.9958    0.9827    0.9892      3403
           1     0.7294    0.9191    0.8133       173

    accuracy                         0.9796      3576
   macro avg     0.8626    0.9509    0.9013      3576
weighted avg     0.9829    0.9796    0.9807      3576

```

## Most fraud-indicative words learned by the model

link, administrative, data entry, hospital, accounting, earn, aptitude staffing, oil, high school, 000, assistant, money, entry, customer service, below, clerk, administrative assistant, aptitude, part time, tx houston

## Most real-indicative words learned by the model

our, software, companies, recruitment, web, digital, growing, our clients, team, of our, client, english, associate, it, clients, social, by, fun, an, about

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
