"""Export the sklearn TF-IDF + Logistic Regression pipeline to a plain JSON
file so the *same* trained model can run in the browser (static HF Space,
since Gradio Spaces need a PRO subscription on this account).

Exports: ordered feature terms, idf vector, LR coefficients, intercept.
The JS in index.html replicates sklearn's TF-IDF transform:
token_pattern (?u)\\b\\w\\w+\\b, lowercase, 1-2 grams, sublinear tf
(1 + ln tf), tf * idf, L2 normalisation; then sigmoid(X . coef + intercept).
"""
import json
from pathlib import Path
import joblib

ROOT = Path(__file__).parent.parent
pipe = joblib.load(ROOT / "model.joblib")
vec = pipe.named_steps["tfidf"]
clf = pipe.named_steps["clf"]
terms = [str(t) for t in vec.get_feature_names_out()]
data = {
    "terms": terms,
    "idf": [float(x) for x in vec.idf_],
    "coef": [float(x) for x in clf.coef_[0]],
    "intercept": float(clf.intercept_[0]),
    "note": "Exported verbatim from model.joblib (scikit-learn 1.9.1 pipeline). Same weights, JS re-implementation of the TF-IDF transform.",
}
out = Path(__file__).parent / "model.json"
out.write_text(json.dumps(data))
print(f"features={len(terms)} size={out.stat().st_size/1024:.0f} KB -> {out}")
# reference probabilities for parity testing
for name in ("scammy.txt", "legit.txt"):
    txt = (ROOT / "samples" / name).read_text()
    print(name, "python_prob=", float(pipe.predict_proba([txt])[0][1]))
