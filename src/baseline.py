"""TF-IDF + Logistic Regression baseline, with a normalisation ablation.

Why this exists: before claiming deep learning helps, we need to know how far a
simple linear model on n-gram features goes. Character n-grams are included on
purpose - Hinglish has no fixed spelling ("accha"/"acha"/"achha"), and char
n-grams are robust to that, so this is a *strong* baseline, not a strawman.

Usage:  python src/baseline.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from data import load_splits
from metrics import bootstrap_ci, compute_metrics, plot_confusion, text_report

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"
SEED = 42


def fit_eval(mode: str, features: str, C_grid=(0.3, 1, 3, 10)):
    """Pick C on the validation set, then report on the untouched test set."""
    train, val, test = load_splits(mode)
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)

    def build(fit_df=None):
        mats = []
        if "word" in features:
            mats.append(word.fit_transform(train["text"]) if fit_df is not None else None)
        if "char" in features:
            mats.append(char.fit_transform(train["text"]) if fit_df is not None else None)
        return mats

    def transform(df):
        mats = []
        if "word" in features:
            mats.append(word.transform(df["text"]))
        if "char" in features:
            mats.append(char.transform(df["text"]))
        return hstack(mats).tocsr()

    Xtr = hstack(build(train)).tocsr()
    Xva, Xte = transform(val), transform(test)

    best = None
    for C in C_grid:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=SEED)
        clf.fit(Xtr, train["label"])
        f1 = compute_metrics(val["label"], clf.predict(Xva))["macro_f1"]
        if best is None or f1 > best[0]:
            best = (f1, C, clf)
    val_f1, C, clf = best
    pred = clf.predict(Xte)
    m = compute_metrics(test["label"], pred)
    lo, hi = bootstrap_ci(test["label"], pred)
    m.update({"val_macro_f1": val_f1, "best_C": C, "macro_f1_ci95": [lo, hi],
              "n_features": int(Xtr.shape[1])})
    return m, pred, test


if __name__ == "__main__":
    RES.mkdir(exist_ok=True)
    results = {}
    t0 = time.time()
    # Ablation grid: normalisation (minimal vs full) x feature set.
    for mode in ("minimal", "full"):
        for feats in ("word", "char", "word+char"):
            m, pred, test = fit_eval(mode, feats)
            key = f"tfidf_lr[{feats}|{mode}]"
            results[key] = m
            print(f"{key:28s} test macro-F1={m['macro_f1']:.3f} "
                  f"(95% CI {m['macro_f1_ci95'][0]:.3f}-{m['macro_f1_ci95'][1]:.3f})  "
                  f"acc={m['accuracy']:.3f}  val F1={m['val_macro_f1']:.3f}  C={m['best_C']}")

    # Headline baseline = word+char with full normalisation.
    m, pred, test = fit_eval("full", "word+char")
    print("\nHeadline baseline (word+char, full normalisation)")
    print(text_report(test["label"], pred))
    plot_confusion(test["label"], pred, "TF-IDF + LR (word+char)", FIG / "cm_tfidf_lr.png")
    test.assign(pred=pred).to_csv(RES / "preds_tfidf_lr.csv", index=False)
    results["headline"] = "tfidf_lr[word+char|full]"
    (RES / "baseline_results.json").write_text(json.dumps(results, indent=2))
    print(f"done in {time.time() - t0:.1f}s -> results/baseline_results.json")
