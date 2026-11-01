"""Data loading, text normalisation and train/val/test splitting.

Dataset: IIITH Hindi-English code-mixed sentiment corpus
(Prabhu, Joshi, Shrivastava, Varma - COLING 2016, arXiv:1611.00472).
Line format:  id <TAB> text <TAB> label <TAB> label
(the two label columns are identical in the released file).

Usage:
    python src/data.py            # builds data/processed/{train,val,test}.csv
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "IIITH_Codemixed.txt"
PROC_DIR = ROOT / "data" / "processed"

# 0/1/2 -> class names. Inferred from samples + class proportions; checked
# again in `label_sanity_check` below (top discriminative tokens per class).
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}
NUM_CLASSES = 3
SEED = 42

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_HASHTAG = re.compile(r"#(\w+)")
_REPEAT = re.compile(r"(.)\1{2,}")          # 3+ same char -> 2  ("baaaki" -> "baaki")
_SPACES = re.compile(r"\s+")
_TOKEN = re.compile(r"\w+|[^\w\s]")          # words/numbers, and each punctuation mark / emoji


def clean_text(text: str, mode: str = "full") -> str:
    """Normalise a raw comment.

    mode="minimal": only unicode-normalise + lowercase (ablation baseline).
    mode="full"   : also map URLs/@mentions to placeholders, keep hashtag words,
                    squeeze elongated characters ("soooo" -> "soo"), collapse spaces.
    """
    t = unicodedata.normalize("NFKC", str(text)).lower()
    if mode == "minimal":
        return _SPACES.sub(" ", t).strip()
    t = _URL.sub(" urltok ", t)
    t = _MENTION.sub(" usertok ", t)
    t = _HASHTAG.sub(r" \1 ", t)
    t = _REPEAT.sub(r"\1\1", t)
    return _SPACES.sub(" ", t).strip()


def tokenize(text: str) -> list[str]:
    """Word-level tokens. Punctuation such as '!' and '?' is kept (sentiment cue)."""
    return _TOKEN.findall(text)


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            rows.append((int(parts[0]), parts[1], int(parts[2])))
    return pd.DataFrame(rows, columns=["id", "text_raw", "label"])


def make_splits(df: pd.DataFrame, seed: int = SEED):
    """Stratified 80/10/10 split."""
    train, rest = train_test_split(
        df, test_size=0.20, stratify=df["label"], random_state=seed
    )
    val, test = train_test_split(
        rest, test_size=0.50, stratify=rest["label"], random_state=seed
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def load_splits(mode: str = "full"):
    """Load saved splits and (re)compute the cleaned text for the chosen mode."""
    out = []
    for name in ("train", "val", "test"):
        d = pd.read_csv(PROC_DIR / f"{name}.csv")
        d["text"] = d["text_raw"].map(lambda s: clean_text(s, mode))
        out.append(d)
    return tuple(out)


def label_sanity_check(train: pd.DataFrame, top_k: int = 12) -> pd.DataFrame:
    """Top tokens per class by smoothed log-odds vs. the other classes.

    Used to verify that label 0/1/2 really behave like negative/neutral/positive.
    """
    import math
    from collections import Counter

    counts = {c: Counter() for c in LABEL_NAMES}
    for text, lab in zip(train["text_raw"], train["label"]):
        counts[lab].update(set(tokenize(clean_text(text))))
    vocab = set().union(*counts.values())
    rows = []
    for c in LABEL_NAMES:
        other = Counter()
        for o in LABEL_NAMES:
            if o != c:
                other.update(counts[o])
        n_c, n_o = sum(train["label"] == c), sum(train["label"] != c)
        scored = []
        for w in vocab:
            if counts[c][w] + other[w] < 8 or not w.isalpha():
                continue
            lo = math.log((counts[c][w] + 1) / (n_c + 2)) - math.log((other[w] + 1) / (n_o + 2))
            scored.append((lo, w, counts[c][w]))
        scored.sort(reverse=True)
        for lo, w, k in scored[:top_k]:
            rows.append((c, LABEL_NAMES[c], w, k, round(lo, 2)))
    return pd.DataFrame(rows, columns=["label", "name", "token", "count_in_class", "log_odds"])


if __name__ == "__main__":
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw()
    train, val, test = make_splits(df)
    for name, d in (("train", train), ("val", val), ("test", test)):
        d.to_csv(PROC_DIR / f"{name}.csv", index=False)
        dist = d["label"].value_counts().sort_index().to_dict()
        print(f"{name:5s} n={len(d):4d}  class counts={dist}")
    chk = label_sanity_check(train)
    (ROOT / "results").mkdir(exist_ok=True)
    chk.to_csv(ROOT / "results" / "label_sanity_top_tokens.csv", index=False)
    for c in LABEL_NAMES:
        toks = ", ".join(chk[chk.label == c]["token"].tolist())
        print(f"[{c} = {LABEL_NAMES[c]}] {toks}")
