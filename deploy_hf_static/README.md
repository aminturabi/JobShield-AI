---
title: JobShield AI
emoji: 🛡️
colorFrom: red
colorTo: green
sdk: static
app_file: index.html
pinned: false
short_description: Fake job posting detector - ML risk score + red flags
---

# JobShield AI — Fake Job Posting Detector

Paste a job posting → get an **ML fraud-risk score** plus the **specific red flags** found in the text.

**Why:** fresh graduates and students are scammed by fake job postings — fee requests,
WhatsApp-only contact, free-email recruiters, unrealistic salary promises.

## How it works (two separate layers)

1. **ML model** — TF-IDF (1–2 grams, 15k features) + Logistic Regression (`class_weight="balanced"`),
   trained on the real EMSCAD dataset (17,880 postings, 866 fraudulent / 4.84%).
   Held-out test: fraud precision 0.7294, recall 0.9191, **F1 0.8133**, PR-AUC 0.9297, ROC-AUC 0.9907.
   The trained weights are exported verbatim to `model.json` and the TF-IDF transform is
   re-implemented in JavaScript (`jobshield.js`), so the **same model runs 100% in the
   visitor's browser** — instant inference, nothing is uploaded. Parity with the Python
   model was verified on held-out postings before deployment.
2. **Red-flag rules** — transparent pattern checks (fees, WhatsApp/Telegram-only contact,
   free email domains, pay promises, requests for banking/personal details, urgency).
   A separate rule layer (port of `red_flags.py`), never ML output.

> A Gradio version of this app (server-side, using the original `model.joblib`) also exists;
> this account's free tier requires a PRO subscription for Gradio/Docker Spaces, so the
> live free demo is this static version — same model, same outputs.

## Honest limitations

The training data is from the 2017 EMSCAD study (mostly US/UK/AU English postings);
modern scams use patterns under-represented in it — one reason for the extra rule layer.
A score is a risk signal, not proof. Always verify through the company's official website.

Built with AI assistance; directed, tested and validated by Amin Khan.
Portfolio: https://aminkhanofficial.vercel.app/
