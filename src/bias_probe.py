"""Exploratory bias probe: do the trained models treat community words as a sentiment shortcut?

We fill neutral templates with different community words and compare the mean P(negative).
In the training corpus, comments mentioning a form of "musalman" are ~81% negative (n=57) versus
~15% negative overall, so a model can learn the word as a shortcut. This is a small exploratory
audit (5 templates x 4 words), NOT a complete fairness evaluation.

    python src/bias_probe.py
"""
from pathlib import Path

import pandas as pd

from predict import Predictor

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ["mera dost {} hai", "wo {} hai", "{} ke bare me video dekha", "kal ek {} se mila", "ye {} log kahan rehte hain"]
GROUPS = ["musalman", "hindu", "sikh", "christian"]
MODELS = ["charcnn", "bilstm_attn", "textcnn", "transformer"]

if __name__ == "__main__":
    rows = []
    for m in MODELS:
        p = Predictor(m)
        for g in GROUPS:
            probs = [p(t.format(g))["probs"]["negative"] for t in TEMPLATES]
            rows.append(dict(model=m, word=g, mean_p_negative=round(sum(probs) / len(probs), 3)))
    df = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / "bias_probe.csv", index=False)
    print(df.pivot(index="model", columns="word", values="mean_p_negative").to_string())
