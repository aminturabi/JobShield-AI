"""
JobShield AI - FastAPI app.

Serves:
  GET  /         -> single-page web UI (paste a job posting, get a result)
  POST /predict  -> JSON: ML fraud probability + separate rule-based red flags
  GET  /health   -> {"status": "ok", "model_loaded": bool}

Run:
    uvicorn app:app --reload --port 8000

The ML score and the rule-based red flags are two SEPARATE layers:
  - ml_fraud_probability comes from the trained TF-IDF + Logistic
    Regression pipeline (model.joblib, see train.py / eval_report.md).
  - red_flags come from red_flags.py - plain pattern checks a human
    can verify by reading the posting. They are never ML output.
"""
from pathlib import Path

import joblib
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from red_flags import detect_red_flags

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model.joblib"

app = FastAPI(title="JobShield AI - Fake Job Posting Detector", version="0.1.0")

_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


class PredictRequest(BaseModel):
    text: str


def risk_level(prob: float) -> str:
    if prob >= 0.65:
        return "high"
    if prob >= 0.35:
        return "medium"
    return "low"


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict")
def predict(req: PredictRequest):
    text = (req.text or "").strip()
    if not text:
        return JSONResponse(
            status_code=400,
            content={"error": "Empty input: paste a job posting to analyse."},
        )
    if len(text.split()) < 5:
        return JSONResponse(
            status_code=400,
            content={"error": "Input too short: paste the full job posting (at least a few sentences)."},
        )
    model = get_model()
    prob = float(model.predict_proba([text])[0][1])
    flags = detect_red_flags(text)
    return {
        "ml_fraud_probability": round(prob, 4),
        "risk_score_percent": round(prob * 100, 1),
        "risk_level": risk_level(prob),
        "verdict": (
            "Likely fraudulent - do not pay anything or share personal documents."
            if prob >= 0.65
            else "Suspicious - verify the company through its official website before applying."
            if prob >= 0.35
            else "Looks like a normal posting on the ML model - still verify the employer independently."
        ),
        "red_flags": flags,
        "red_flag_count": len(flags),
        "note": "red_flags are from a separate transparent rule-based check, not from the ML model. A risk score is a signal, not proof of fraud.",
    }


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>JobShield AI - Fake Job Posting Detector</title>
<style>
body{font-family:system-ui,Arial,sans-serif;max-width:820px;margin:24px auto;padding:0 16px;color:#1a1a1a}
textarea{width:100%;min-height:220px;padding:12px;font-size:15px}
button{padding:10px 18px;font-size:15px;cursor:pointer;margin:6px 4px 6px 0}
.bar{height:22px;background:#eee;border-radius:6px;overflow:hidden}
.fill{height:100%}
.flag{background:#fff3f3;border:1px solid #e5b4b4;border-radius:6px;padding:8px 12px;margin:8px 0}
.small{color:#555;font-size:13px}
</style></head><body>
<h1>🛡️ JobShield AI</h1>
<p>Paste a job posting. Get an ML fraud-risk score <b>plus</b> the specific red flags found in the text.</p>
<textarea id="txt" placeholder="Paste the full job posting here..."></textarea><br>
<button onclick="run()">Check this posting</button>
<button onclick="loadSample('scam')">Load scam example</button>
<button onclick="loadSample('legit')">Load legit example</button>
<div id="out"></div>
<p class="small">JobShield gives a risk signal, not proof. Never pay a fee to get a job, and always verify the employer on its official website.</p>
<script>
const SAMPLES = {
scam: `URGENT HIRING!!! Data Entry Operator - WORK FROM HOME. Earn $3000 per week, easy money, no experience needed! Guaranteed income. To start, pay a small registration fee of $50 by gift card. Contact ONLY on WhatsApp. Send your bank account details and passport copy today. Recruiter: hr.jobs247@gmail.com - apply immediately, limited slots, no interview!`,
legit: `Software Engineer (Backend) - TechNova Solutions, Lahore. TechNova Solutions is a product company building logistics software for clients in three countries. We are hiring a backend engineer to design and maintain REST APIs in Python and PostgreSQL, write unit tests, and take part in code reviews. Requirements: BS in Computer Science or equivalent, 2+ years of professional experience, solid understanding of databases and Git. We offer a market salary, health insurance, and annual leave. Apply through the careers page on our official website. TechNova Solutions is an equal-opportunity employer.`
};
function loadSample(k){document.getElementById('txt').value = SAMPLES[k];}
async function run(){
  const text = document.getElementById('txt').value;
  const out = document.getElementById('out');
  out.innerHTML = 'Checking...';
  const r = await fetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({text})});
  const d = await r.json();
  if(!r.ok){ out.innerHTML = '<p style="color:red">' + (d.error || 'Error') + '</p>'; return; }
  const color = d.risk_level === 'high' ? '#c0392b' : d.risk_level === 'medium' ? '#e67e22' : '#27ae60';
  let h = '<h2>Fraud risk: ' + d.risk_score_percent + '% (' + d.risk_level.toUpperCase() + ')</h2>';
  h += '<div class="bar"><div class="fill" style="width:' + d.risk_score_percent + '%;background:' + color + '"></div></div>';
  h += '<p>' + d.verdict + '</p>';
  h += '<h3>Red flags found by rule checks (' + d.red_flag_count + ')</h3>';
  if(d.red_flags.length === 0){ h += '<p>No rule-based red flags found.</p>'; }
  d.red_flags.forEach(f => { h += '<div class="flag"><b>🚩 ' + f.flag + '</b><br><span class="small">' + f.evidence + '</span><br>' + f.why_it_matters + '</div>'; });
  h += '<p class="small">' + d.note + '</p>';
  out.innerHTML = h;
}
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
