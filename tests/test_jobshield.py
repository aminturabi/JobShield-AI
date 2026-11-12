"""JobShield AI tests. Run: pytest -q (with model.joblib present)."""
import sys
from pathlib import Path

import joblib
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app import app  # noqa: E402
from red_flags import detect_red_flags  # noqa: E402

client = TestClient(app)
SCAMMY = (ROOT / "samples" / "scammy.txt").read_text()
LEGIT = (ROOT / "samples" / "legit.txt").read_text()


def test_model_artifact_loads_and_predicts():
    model = joblib.load(ROOT / "model.joblib")
    proba = model.predict_proba([LEGIT])[0]
    assert len(proba) == 2
    assert 0.0 <= float(proba[1]) <= 1.0


def test_scammy_scores_higher_than_legit():
    scam = client.post("/predict", json={"text": SCAMMY}).json()
    legit = client.post("/predict", json={"text": LEGIT}).json()
    assert scam["ml_fraud_probability"] > legit["ml_fraud_probability"], (
        f"scammy={scam['ml_fraud_probability']} legit={legit['ml_fraud_probability']}"
    )


def test_red_flags_detect_fee_and_whatsapp():
    ids = {f["id"] for f in detect_red_flags(SCAMMY)}
    assert "fee_or_payment_request" in ids
    assert "messaging_app_only_contact" in ids
    assert "free_email_recruiter" in ids
    assert "personal_or_banking_info_request" in ids


def test_legit_has_fewer_flags_than_scammy():
    assert len(detect_red_flags(LEGIT)) < len(detect_red_flags(SCAMMY))


def test_empty_input_is_rejected():
    r = client.post("/predict", json={"text": "   "})
    assert r.status_code == 400
    assert "error" in r.json()


def test_home_page_serves_ui():
    r = client.get("/")
    assert r.status_code == 200
    assert "JobShield" in r.text


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True
