"use strict";
/*
 * JobShield AI - browser-side inference.
 *
 * ML layer: a faithful JS re-implementation of the trained scikit-learn
 * TF-IDF + Logistic Regression pipeline. The weights (vocabulary, idf,
 * coefficients, intercept) are exported verbatim from model.joblib into
 * model.json - it is the SAME model, not a retrain.
 * sklearn transform replicated: lowercase, token_pattern \b\w\w+\b,
 * 1-2 grams, sublinear tf (1 + ln tf), tf * idf, L2 normalisation,
 * then sigmoid(X . coef + intercept).
 *
 * Red-flag layer: direct port of red_flags.py - transparent rule checks,
 * kept separate from the ML output, never presented as ML.
 */

function tokenize(text) {
  const runs = text.toLowerCase().match(/[\p{L}\p{N}_]+/gu) || [];
  return runs.filter((t) => [...t].length >= 2);
}

function predictML(text, model) {
  const vocab = model._vocabMap || (model._vocabMap = new Map(model.terms.map((t, i) => [t, i])));
  const tokens = tokenize(text);
  const counts = new Map();
  const bump = (feat) => {
    const idx = vocab.get(feat);
    if (idx !== undefined) counts.set(idx, (counts.get(idx) || 0) + 1);
  };
  for (let i = 0; i < tokens.length; i++) {
    bump(tokens[i]);
    if (i + 1 < tokens.length) bump(tokens[i] + " " + tokens[i + 1]);
  }
  let normSq = 0;
  const vals = new Map();
  for (const [idx, c] of counts) {
    const v = (1 + Math.log(c)) * model.idf[idx];
    vals.set(idx, v);
    normSq += v * v;
  }
  const norm = Math.sqrt(normSq);
  let score = model.intercept;
  if (norm > 0) {
    for (const [idx, v] of vals) score += (v / norm) * model.coef[idx];
  }
  return 1 / (1 + Math.exp(-score));
}

function riskLevel(prob) {
  if (prob >= 0.65) return "high";
  if (prob >= 0.35) return "medium";
  return "low";
}

const RULES = [
  ["Asks for a fee or payment",
   /(registration|training|processing|application|joining|security)\s+fee|pay\s+(a\s+)?fee|upfront\s+payment|wire\s+transfer|gift\s+cards?|send\s+money|bank\s+transfer.*(before|upfront|first)/i,
   "Legitimate employers do not charge candidates fees to apply, register, or start a job."],
  ["WhatsApp / Telegram-only contact",
   /whatsapp|telegram/i,
   "Scam postings often move contact off official channels to WhatsApp or Telegram only."],
  ["Recruiter uses a free email domain",
   /[\w.\-]+@(gmail|yahoo|hotmail|outlook|aol)\.com/i,
   "A real company's recruiter normally emails from a company domain, not Gmail/Yahoo/Hotmail."],
  ["Unrealistic salary / easy-money promise",
   /earn\s+\$?[\d,]+|make\s+\$[\d,]+\s*(per|a|\/)\s*(day|week|hour)|easy\s+money|guaranteed\s+income|no\s+experience\s+(needed|required|necessary)/i,
   "Very high pay with no experience or guaranteed income is a classic scam lure."],
  ["Asks for personal or banking details up front",
   /bank\s+account\s+(details|number|information)|social\s+security|passport\s+(copy|number)|credit\s+card\s+(details|number)|date\s+of\s+birth/i,
   "Bank details, SSN, or passport copies are never needed just to apply for a job."],
  ["Money / package handling scheme",
   /cash\s+(checks|cheques)|reship|re-?shipping|package\s+handler.*home|money\s+mule|process\s+payments\s+from\s+home/i,
   "Cheque-cashing and package-reshipping 'jobs' are common money-mule scams."],
  ["Pressure / urgency language",
   /apply\s+immediately|urgent(ly)?\s+hiring|limited\s+(slots|positions|seats)|start\s+today|immediate\s+start\s+no\s+interview|no\s+interview/i,
   "Pressure to act fast, or a job with no interview at all, is used to stop victims thinking it through."],
];

function detectRedFlags(text) {
  const flags = [];
  if (!text) return flags;
  for (const [label, pattern, why] of RULES) {
    const m = pattern.exec(text);
    if (m) {
      const start = Math.max(0, m.index - 30);
      const end = Math.min(text.length, m.index + m[0].length + 30);
      const evidence = text.slice(start, end).replace(/\s+/g, " ").trim();
      flags.push({ flag: label, evidence: "..." + evidence + "...", why_it_matters: why });
    }
  }
  const words = text.split(/\s+/).filter(Boolean).length;
  if (words > 0 && words < 40) {
    flags.push({
      flag: "Very short / vague posting",
      evidence: "Posting is only " + words + " words.",
      why_it_matters: "Real postings usually describe the company, role and requirements; scam ads tend to be short and sketchy (a known pattern in the EMSCAD research).",
    });
  }
  return flags;
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { tokenize, predictML, riskLevel, detectRedFlags };
}
