"""Refit the headline TF-IDF + LogReg model (word+char n-grams, full normalisation, C=3)
on the training split and save it for the demo.

It then checks that the saved model reproduces results/preds_tfidf_lr.csv exactly, so the
demo serves the same model that the report evaluates.

    python src/save_tfidf.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from data import load_splits

ROOT = Path(__file__).resolve().parents[1]
SEED = 42

if __name__ == "__main__":
    base = json.loads((ROOT / "results" / "baseline_results.json").read_text())
    C = base[base["headline"]]["best_C"]
    train, val, test = load_splits("full")

    word = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    Xtr = hstack([word.fit_transform(train["text"]), char.fit_transform(train["text"])]).tocsr()
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=SEED).fit(Xtr, train["label"])

    Xte = hstack([word.transform(test["text"]), char.transform(test["text"])]).tocsr()
    pred = clf.predict(Xte)
    saved = pd.read_csv(ROOT / "results" / "preds_tfidf_lr.csv")["pred"].values
    same = bool((pred == saved).all())
    print(f"C={C}  test predictions identical to results/preds_tfidf_lr.csv: {same}")
    assert same, "refit does not reproduce the evaluated baseline"

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(dict(word=word, char=char, clf=clf, n_word=len(word.vocabulary_), C=C),
                ROOT / "models" / "tfidf_lr.joblib", compress=3)
    print("saved models/tfidf_lr.joblib")
