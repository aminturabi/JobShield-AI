"""
JobShield AI - transparent rule-based red-flag layer.

This module is SEPARATE from the machine-learning model in train.py / app.py.
Every flag here is a plain, human-readable pattern check on the posting text.
Rule hits are never presented as ML output - the API returns them in their
own field so a user can see exactly *why* something was flagged.
"""
import re

# Each rule: (id, label, regex, explanation)
_RULES = [
    (
        "fee_or_payment_request",
        "Asks for a fee or payment",
        r"(registration|training|processing|application|joining|security)\s+fee|pay\s+(a\s+)?fee|upfront\s+payment|wire\s+transfer|gift\s+cards?|send\s+money|bank\s+transfer.*(before|upfront|first)",
        "Legitimate employers do not charge candidates fees to apply, register, or start a job.",
    ),
    (
        "messaging_app_only_contact",
        "WhatsApp / Telegram-only contact",
        r"whatsapp|telegram",
        "Scam postings often move contact off official channels to WhatsApp or Telegram only.",
    ),
    (
        "free_email_recruiter",
        "Recruiter uses a free email domain",
        r"[\w.\-]+@(gmail|yahoo|hotmail|outlook|aol)\.com",
        "A real company's recruiter normally emails from a company domain, not Gmail/Yahoo/Hotmail.",
    ),
    (
        "unrealistic_pay_promise",
        "Unrealistic salary / easy-money promise",
        r"earn\s+\$?[\d,]+|make\s+\$[\d,]+\s*(per|a|\/)\s*(day|week|hour)|easy\s+money|guaranteed\s+income|no\s+experience\s+(needed|required|necessary)",
        "Very high pay with no experience or guaranteed income is a classic scam lure.",
    ),
    (
        "personal_or_banking_info_request",
        "Asks for personal or banking details up front",
        r"bank\s+account\s+(details|number|information)|social\s+security|passport\s+(copy|number)|credit\s+card\s+(details|number)|date\s+of\s+birth",
        "Bank details, SSN, or passport copies are never needed just to apply for a job.",
    ),
    (
        "money_handling_scheme",
        "Money / package handling scheme",
        r"cash\s+(checks|cheques)|reship|re-?shipping|package\s+handler.*home|money\s+mule|process\s+payments\s+from\s+home",
        "Cheque-cashing and package-reshipping 'jobs' are common money-mule scams.",
    ),
    (
        "pressure_urgency",
        "Pressure / urgency language",
        r"apply\s+immediately|urgent(ly)?\s+hiring|limited\s+(slots|positions|seats)|start\s+today|immediate\s+start\s+no\s+interview|no\s+interview",
        "Pressure to act fast, or a job with no interview at all, is used to stop victims thinking it through.",
    ),
]

_COMPILED = [(rid, label, re.compile(pat, re.IGNORECASE), why) for rid, label, pat, why in _RULES]

MIN_WORDS = 40


def detect_red_flags(text: str) -> list[dict]:
    """Return a list of rule-based red flags found in `text`.

    Each item: {"id", "flag", "evidence", "why_it_matters"}.
    This is a rule layer only - it does not use the ML model.
    """
    flags: list[dict] = []
    if not text:
        return flags
    for rid, label, pattern, why in _COMPILED:
        m = pattern.search(text)
        if m:
            start = max(0, m.start() - 30)
            end = min(len(text), m.end() + 30)
            evidence = re.sub(r"\s+", " ", text[start:end]).strip()
            flags.append(
                {"id": rid, "flag": label, "evidence": f"...{evidence}...", "why_it_matters": why}
            )
    words = len(text.split())
    if 0 < words < MIN_WORDS:
        flags.append(
            {
                "id": "very_short_vague_posting",
                "flag": "Very short / vague posting",
                "evidence": f"Posting is only {words} words.",
                "why_it_matters": "Real postings usually describe the company, role and requirements; scam ads tend to be short and sketchy (a known pattern in the EMSCAD research).",
            }
        )
    return flags
