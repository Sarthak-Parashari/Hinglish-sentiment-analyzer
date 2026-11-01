"""Inference helper shared by the CLI and the demo app.

    python src/predict.py "yaar ye gana bahut mast hai"
    python src/predict.py --model textcnn "ye bakwas hai"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from data import clean_text, tokenize
from models import BiLSTMAttn, TextCNN, TransformerCls

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["negative", "neutral", "positive"]
CLASSES = {"bilstm_attn": BiLSTMAttn, "textcnn": TextCNN, "charcnn": TextCNN, "transformer": TransformerCls}
PAD, UNK = 0, 1


class Predictor:
    def __init__(self, name: str = "bilstm_attn", models_dir: Path | None = None):
        path = (models_dir or ROOT / "models") / f"{name}.pt"
        ck = torch.load(path, map_location="cpu", weights_only=True)   # only tensors / plain containers
        self.name, self.level, self.max_len = name, ck["level"], ck["max_len"]
        self.vocab = {u: i for i, u in enumerate(ck["itos"])}
        self.model = CLASSES[name](vocab_size=len(self.vocab), num_classes=3, **ck["cand"]["model"])
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def __call__(self, text: str) -> dict:
        clean = clean_text(text)
        units = (tokenize(clean) if self.level == "word" else list(clean))[: self.max_len]
        ids = [self.vocab.get(u, UNK) for u in units] or [UNK]        # empty input -> one <unk>
        x = torch.zeros(1, self.max_len, dtype=torch.long)             # always pad to training length
        x[0, : len(ids)] = torch.tensor(ids)
        attention = None
        if hasattr(self.model, "attn_vec"):                            # BiLSTM + attention
            logits, alpha = self.model(x, return_attn=True)
            attention = alpha[0, : len(ids)].tolist()
        else:
            logits = self.model(x)
        probs = F.softmax(logits, dim=-1)[0].tolist()
        best = max(range(3), key=probs.__getitem__)
        unk_ratio = sum(u not in self.vocab for u in units) / len(units) if units else 0.0
        return dict(clean=clean, tokens=units if units else ["<empty>"], label=LABELS[best],
                    probs=dict(zip(LABELS, probs)), attention=attention, unk_ratio=unk_ratio)


class TfidfPredictor:
    """Serves the evaluated TF-IDF + LogReg baseline (see save_tfidf.py).

    Because the model is linear, we can show exactly which words pushed the prediction:
    contribution of a word feature = (w_pred - mean(w_others)) * tfidf value.
    """

    name = "tfidf_lr"
    level = "word"

    def __init__(self, models_dir: Path | None = None):
        import joblib   # only needed when the TF-IDF model is used

        d = joblib.load((models_dir or ROOT / "models") / "tfidf_lr.joblib")   # our own file
        self.word, self.char, self.clf, self.n_word = d["word"], d["char"], d["clf"], d["n_word"]
        self.word_names = self.word.get_feature_names_out()

    def __call__(self, text: str) -> dict:
        import numpy as np
        from scipy.sparse import hstack

        clean = clean_text(text)
        Xw = self.word.transform([clean])
        X = hstack([Xw, self.char.transform([clean])]).tocsr()
        probs = self.clf.predict_proba(X)[0]
        best = int(np.argmax(probs))
        w = self.clf.coef_
        delta = w[best, : self.n_word] - np.delete(w, best, axis=0)[:, : self.n_word].mean(axis=0)
        row = Xw.tocoo()
        contrib = sorted(((self.word_names[j], float(v * delta[j])) for j, v in zip(row.col, row.data)),
                         key=lambda t: -abs(t[1]))
        tokens = tokenize(clean)
        known = set(self.word.vocabulary_)
        unk_ratio = sum(t not in known for t in tokens) / len(tokens) if tokens else 0.0
        return dict(clean=clean, tokens=tokens or ["<empty>"], label=LABELS[best],
                    probs=dict(zip(LABELS, probs.tolist())), attention=None,
                    evidence=contrib[:12], unk_ratio=unk_ratio)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--model", default="bilstm_attn", choices=list(CLASSES))
    a = ap.parse_args()
    out = Predictor(a.model)(a.text)
    print(f"{out['label']}  " + "  ".join(f"{k}={v:.2f}" for k, v in out["probs"].items()))
    if out["attention"]:
        top = sorted(zip(out["tokens"], out["attention"]), key=lambda t: -t[1])[:5]
        print("most attended:", ", ".join(f"{t} ({w:.2f})" for t, w in top))
