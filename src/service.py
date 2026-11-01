"""Backend service layer for the demo: runs all models on a comment and packages the result.

No UI code here, so the same functions could sit behind a REST API (e.g. FastAPI) unchanged.

    from service import Service
    svc = Service()
    out = svc.analyse("yaar ye gana bahut mast hai")
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from predict import CLASSES, LABELS, Predictor, TfidfPredictor

ROOT = Path(__file__).resolve().parents[1]
MAX_CHARS = 1000
MAX_BATCH = 200
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
HAS_LETTER = re.compile(r"[A-Za-zऀ-ॿ]")

# display order = by test macro-F1 (best first)
ORDER = ["tfidf_lr", "charcnn", "bilstm_attn", "textcnn", "transformer"]
NAMES = {
    "tfidf_lr": "TF-IDF + LogReg",
    "charcnn": "CharCNN",
    "bilstm_attn": "BiLSTM + attention",
    "textcnn": "TextCNN (word)",
    "transformer": "Transformer (scratch)",
}


def test_scores() -> dict[str, float]:
    """Test macro-F1 per model, read from the saved result files (never hard-coded)."""
    out = {}
    try:
        base = json.loads((ROOT / "results" / "baseline_results.json").read_text())
        out["tfidf_lr"] = base[base["headline"]]["macro_f1"]
        for k, v in json.loads((ROOT / "results" / "dl_results.json").read_text()).items():
            out[k] = v["macro_f1_mean"]
    except FileNotFoundError:
        pass
    return out


class Service:
    def __init__(self):
        self._cache: dict = {}
        self.scores = test_scores()

    def _get(self, key: str):
        if key not in self._cache:
            self._cache[key] = TfidfPredictor() if key == "tfidf_lr" else Predictor(key)
        return self._cache[key]

    @staticmethod
    def _confidence(probs: dict) -> tuple[float, float]:
        top = sorted(probs.values(), reverse=True)
        return top[0], top[0] - top[1]

    def analyse(self, text: str, primary: str = "tfidf_lr") -> dict:
        text = (text or "").strip()
        if not text:
            return dict(empty=True)
        warnings = []
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS]
            warnings.append(f"The comment was cut to the first {MAX_CHARS} characters.")
        if DEVANAGARI.search(text):
            warnings.append("This comment contains Devanagari script. The models were trained only on "
                            "Roman-script Hinglish, so Devanagari words are unknown to them.")
        if not HAS_LETTER.search(text):
            warnings.append("No letters found, so the models have almost nothing to go on.")

        results = {k: self._get(k)(text) for k in ORDER}
        main = results[primary]
        top, margin = self._confidence(main["probs"])
        unsure = top < 0.50 or margin < 0.15

        n_tok = len(main["tokens"]) if main["tokens"] != ["<empty>"] else 0
        if n_tok > 50:
            warnings.append("Only the first 50 words are used by the word-level neural models.")
        if results["tfidf_lr"]["unk_ratio"] >= 0.5:
            unsure = True
        if n_tok >= 1 and results["tfidf_lr"]["unk_ratio"] >= 0.5:
            warnings.append("More than half of the words were never seen in training, so every model "
                            "is largely guessing.")

        votes = Counter(r["label"] for r in results.values())
        best_n = max(votes.values())
        tied = [lab for lab, n in votes.items() if n == best_n]
        majority = main["label"] if main["label"] in tied else tied[0]

        # explanation: exact word evidence for the linear model, attention for the BiLSTM
        if primary == "tfidf_lr":
            ev = main["evidence"]
            expl = dict(kind="evidence", tokens=main["tokens"],
                        toward=[(w, c) for w, c in ev if c > 0][:6],
                        against=[(w, c) for w, c in ev if c < 0][:4],
                        note="Contribution of each word to the predicted class, computed exactly from the "
                             "linear model.")
        else:
            att = results["bilstm_attn"]
            expl = dict(kind="attention", tokens=att["tokens"], weights=att["attention"],
                        note=("The BiLSTM's attention weights. " if primary == "bilstm_attn" else
                              f"{NAMES[primary]} has no built-in explanation, so the BiLSTM's attention is shown. ")
                             + "Attention shows where the model looked, not a guaranteed reason.")
        return dict(
            empty=False, text=text, primary=primary, label=main["label"], probs=main["probs"],
            confidence=top, margin=margin, unsure=unsure, warnings=warnings, explanation=expl,
            agreement=dict(label=majority, count=votes[majority], total=len(results)),
            models=[dict(key=k, name=NAMES[k], label=r["label"], probs=r["probs"], test_f1=self.scores.get(k))
                    for k, r in results.items()],
        )

    def analyse_many(self, texts: list[str], primary: str = "tfidf_lr") -> list[dict]:
        rows = []
        for t in texts[:MAX_BATCH]:
            o = self.analyse(t, primary)
            if o["empty"]:
                continue
            rows.append(dict(comment=o["text"], prediction=o["label"], confidence=round(o["confidence"], 3),
                             unsure="yes" if o["unsure"] else "no",
                             models_agreeing=f"{o['agreement']['count']}/{o['agreement']['total']} say {o['agreement']['label']}"))
        return rows


if __name__ == "__main__":
    import sys

    svc = Service()
    r = svc.analyse(" ".join(sys.argv[1:]) or "yaar ye gana bahut mast hai")
    print(json.dumps({k: v for k, v in r.items() if k != "models"}, indent=2, default=str)[:1500])
    for m in r["models"]:
        print(f"{m['name']:24s} {m['label']:9s} " + " ".join(f"{p:.2f}" for p in m["probs"].values()))
