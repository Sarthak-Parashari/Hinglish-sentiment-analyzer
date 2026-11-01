"""Turn raw experiment outputs into the tables and figures used in the report.

Run after baseline.py and train_dl.py (and optionally after copying the Colab outputs into results/):
    python src/make_report_assets.py

Produces
  results/comparison_table.csv|md   main results table
  results/mcnemar.csv               paired significance test: each neural model vs the TF-IDF baseline
  results/error_slices.csv          accuracy by text length / negation presence
  results/error_examples.csv        misclassified test sentences for manual reading
  figures/model_comparison.png, learning_curves.png, attention_examples.png
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from data import clean_text, load_splits, tokenize
from metrics import bootstrap_ci

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
NAMES = {
    "tfidf_lr": "TF-IDF + LogReg (word+char)",
    "bilstm_attn": "BiLSTM + attention",
    "textcnn": "TextCNN (word)",
    "charcnn": "CharCNN",
    "transformer": "Transformer (from scratch)",
}
FAMILY_COLOR = {"classical": "#8c8c8c", "scratch": "#2a6fbb", "pretrained": "#d1782b"}
NEGATORS = {"nahi", "nahin", "nhi", "nhin", "nai", "mat", "not", "no", "never", "nahii", "nehi"}


# --------------------------------------------------------------------------- #
def load_rows() -> pd.DataFrame:
    rows = []
    base = json.loads((RES / "baseline_results.json").read_text())
    b = base[base["headline"]]
    rows.append(dict(model=NAMES["tfidf_lr"], family="classical", macro_f1=b["macro_f1"], std=np.nan,
                     accuracy=b["accuracy"], f1_neg=b["f1_negative"], f1_neu=b["f1_neutral"],
                     f1_pos=b["f1_positive"], params=b["n_features"] * 3 + 3, train_s=np.nan, hardware="CPU"))
    dl = json.loads((RES / "dl_results.json").read_text())
    for key, r in dl.items():
        rows.append(dict(model=NAMES[key], family="scratch", macro_f1=r["macro_f1_mean"], std=r["macro_f1_std"],
                         accuracy=r["accuracy_mean"], f1_neg=r["f1_negative_mean"], f1_neu=r["f1_neutral_mean"],
                         f1_pos=r["f1_positive_mean"], params=r["n_params"], train_s=r["mean_train_seconds"],
                         hardware="CPU"))
    colab = RES / "colab_results.csv"
    if colab.exists():
        for _, r in pd.read_csv(colab).iterrows():
            rows.append(dict(model=f"{r['model']} (pretrained, fine-tuned)", family="pretrained",
                             macro_f1=r["macro_f1_mean"], std=r["macro_f1_std"], accuracy=r["accuracy_mean"],
                             f1_neg=r["f1_negative_mean"], f1_neu=r["f1_neutral_mean"], f1_pos=r["f1_positive_mean"],
                             params=r["n_params"], train_s=r["mean_train_seconds"], hardware="GPU (Colab)"))
    return pd.DataFrame(rows)


def write_table(df: pd.DataFrame):
    out = df.copy()
    out.to_csv(RES / "comparison_table.csv", index=False)
    lines = ["| Model | Macro-F1 | Accuracy | F1 neg | F1 neu | F1 pos | Params | Train s/run | HW |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in out.iterrows():
        f1 = f"{r.macro_f1:.3f}" + (f" ± {r['std']:.3f}" if not np.isnan(r["std"]) else " (deterministic)")
        ts = "-" if np.isnan(r.train_s) else f"{r.train_s:.0f}"
        lines.append(f"| {r.model} | {f1} | {r.accuracy:.3f} | {r.f1_neg:.3f} | {r.f1_neu:.3f} | {r.f1_pos:.3f} "
                     f"| {int(r.params):,} | {ts} | {r.hardware} |")
    (RES / "comparison_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot_comparison(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.6, 0.55 * len(df) + 1.2))
    y = np.arange(len(df))[::-1]
    ax.barh(y, df["macro_f1"], xerr=df["std"].fillna(0), color=[FAMILY_COLOR[f] for f in df["family"]],
            capsize=3, height=0.62)
    for yi, (_, r) in zip(y, df.iterrows()):
        ax.text(r.macro_f1 + (0 if np.isnan(r["std"]) else r["std"]) + 0.006, yi, f"{r.macro_f1:.3f}", va="center", fontsize=8)
    ax.set_yticks(y, df["model"], fontsize=8)
    ax.set_xlim(0.4, max(0.85, df["macro_f1"].max() + 0.08))
    ax.set_xlabel("Test macro-F1  (bars: mean over seeds, whiskers: ± 1 std; test n = 388)")
    ax.grid(axis="x", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLOR.values()]
    ax.legend(handles, FAMILY_COLOR.keys(), fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "model_comparison.png", dpi=170)
    plt.close(fig)


def plot_curves():
    curves = json.loads((RES / "dl_curves.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4))
    for key, hist in curves.items():
        ep = [h["epoch"] for h in hist]
        axes[0].plot(ep, [h["train_loss"] for h in hist], label=NAMES[key])
        axes[1].plot(ep, [h["val_loss"] for h in hist], label=NAMES[key])
        axes[2].plot(ep, [h["val_macro_f1"] for h in hist], label=NAMES[key])
    axes[0].set_title("Train loss (class-weighted, label-smoothed)", fontsize=9)
    axes[1].set_title("Validation loss (plain cross-entropy)", fontsize=9)
    axes[2].set_title("Validation macro-F1", fontsize=9)
    for a in axes:
        a.set_xlabel("epoch"); a.grid(alpha=0.3)
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "learning_curves.png", dpi=170)
    plt.close(fig)


def mcnemar_vs_baseline(test: pd.DataFrame):
    base = pd.read_csv(RES / "preds_tfidf_lr.csv")["pred"].values
    y = test["label"].values
    rows = []
    for key in NAMES:
        if key == "tfidf_lr":
            continue
        f = RES / f"preds_{key}_s42.csv"
        if not f.exists():
            continue
        p = pd.read_csv(f)["pred"].values
        b = int(((base == y) & (p != y)).sum())      # baseline right, neural wrong
        c = int(((base != y) & (p == y)).sum())      # baseline wrong, neural right
        pval = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0   # exact McNemar
        rows.append(dict(model=NAMES[key], baseline_only_correct=b, model_only_correct=c, p_value=round(pval, 4)))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "mcnemar.csv", index=False)
    print("\nMcNemar (exact) vs TF-IDF baseline, seed 42:\n", out.to_string(index=False))


def error_slices(test: pd.DataFrame):
    toks = test["text_raw"].map(lambda s: tokenize(clean_text(s)))
    n_words = toks.map(lambda t: sum(w.isalnum() for w in t))
    has_neg = toks.map(lambda t: any(w in NEGATORS for w in t))
    slices = {
        "all": np.ones(len(test), bool),
        "<=3 words": (n_words <= 3).values,
        "4-8 words": ((n_words >= 4) & (n_words <= 8)).values,
        "9-15 words": ((n_words >= 9) & (n_words <= 15)).values,
        ">15 words": (n_words > 15).values,
        "has negator": has_neg.values,
        "no negator": (~has_neg).values,
    }
    rows = []
    y = test["label"].values
    for key in NAMES:
        f = RES / ("preds_tfidf_lr.csv" if key == "tfidf_lr" else f"preds_{key}_s42.csv")
        if not f.exists():
            continue
        p = pd.read_csv(f)["pred"].values
        for sname, m in slices.items():
            rows.append(dict(model=NAMES[key], slice=sname, n=int(m.sum()), accuracy=round(float((p[m] == y[m]).mean()), 3)))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "error_slices.csv", index=False)
    print("\nAccuracy by slice (seed 42):\n", out.pivot(index="slice", columns="model", values="accuracy").to_string())
    print("\nSlice sizes:", {k: int(v.sum()) for k, v in slices.items()})


def error_examples(test: pd.DataFrame, best_key: str):
    y = test["label"].values
    base = pd.read_csv(RES / "preds_tfidf_lr.csv")["pred"].values
    best = pd.read_csv(RES / f"preds_{best_key}_s42.csv")["pred"].values
    names = ["negative", "neutral", "positive"]
    df = pd.DataFrame(dict(text=test["text_raw"], true=[names[i] for i in y],
                           tfidf=[names[i] for i in base], neural=[names[i] for i in best]))
    df["both_wrong"] = (base != y) & (best != y)
    df["any_wrong"] = (base != y) | (best != y)
    df[df.any_wrong].to_csv(RES / "error_examples.csv", index=False)
    print(f"\nTest errors: baseline {int((base != y).sum())}, {best_key} {int((best != y).sum())}, "
          f"shared by both {int(((base != y) & (best != y)).sum())} of {len(y)}")


def attention_figure(test: pd.DataFrame):
    from predict import Predictor
    pred = Predictor("bilstm_attn")
    y = test["label"].values
    p_file = pd.read_csv(RES / "preds_bilstm_attn_s42.csv")["pred"].values
    names = ["negative", "neutral", "positive"]
    # keep the report presentable: skip comments with abuse / community-targeting words
    skip = {"madarchod", "harami", "kutte", "kutta", "sala", "sale", "kafir", "musalman", "musalmaan",
            "musalmano", "muslman", "muslim", "muslims", "islam", "hindu", "hindus", "lanat", "bhosd", "chutiya", "gandu", "bc", "mc", "randi"}
    ok = lambda i: 5 <= len(test["text_raw"][i].split()) <= 14 and not (set(tokenize(clean_text(test["text_raw"][i]))) & skip)
    picks = []
    for c in (0, 1, 2):                                   # two correctly classified examples per class
        idx = [i for i in range(len(y)) if y[i] == c and p_file[i] == c and ok(i)]
        picks += idx[:2]
    wrong = [i for i in range(len(y)) if y[i] != p_file[i] and ok(i)]
    picks += wrong[:2]                                    # plus two mistakes
    fig, ax = plt.subplots(figsize=(10, 0.72 * len(picks) + 0.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, len(picks)); ax.axis("off")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    cmap = plt.get_cmap("Oranges")
    for row, i in enumerate(picks):
        out = pred(test["text_raw"][i])
        ytop = len(picks) - row - 0.5
        header = f"true: {names[y[i]]} | predicted: {out['label']} ({out['probs'][out['label']]:.2f})"
        ax.text(0.0, ytop + 0.30, header, fontsize=7.5, color="#444", va="center")
        att = np.array(out["attention"]); att = att / att.max()
        x, line_y = 0.0, ytop - 0.06
        for tok, w in zip(out["tokens"], att):
            t = ax.text(x, line_y, tok, fontsize=9, va="center",
                        bbox=dict(facecolor=cmap(0.08 + 0.85 * w), edgecolor="none", pad=1.6))
            bb = t.get_window_extent(renderer).transformed(ax.transData.inverted())
            x = bb.x1 + 0.012
    fig.tight_layout()
    fig.savefig(FIG / "attention_examples.png", dpi=170)
    plt.close(fig)


def latency():
    from predict import Predictor
    _, _, test = load_splits("full")
    texts = list(test["text_raw"][:150])
    rows = []
    for key in ("bilstm_attn", "textcnn", "charcnn", "transformer"):
        if not (ROOT / "models" / f"{key}.pt").exists():
            continue
        p = Predictor(key)
        p(texts[0])
        t0 = time.perf_counter()
        for t in texts:
            p(t)
        rows.append(dict(model=NAMES[key], ms_per_sentence=round(1000 * (time.perf_counter() - t0) / len(texts), 2)))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "latency_cpu.csv", index=False)
    print("\nCPU inference latency (batch=1, incl. preprocessing):\n", out.to_string(index=False))


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    _, _, test = load_splits("full")
    df = load_rows()
    write_table(df)
    plot_comparison(df)
    plot_curves()
    mcnemar_vs_baseline(test)
    error_slices(test)
    scratch = df[df.family == "scratch"].sort_values("macro_f1", ascending=False)
    inv = {v: k for k, v in NAMES.items()}
    best_key = inv[scratch.iloc[0]["model"]]
    print(f"\nBest from-scratch neural model by mean macro-F1: {best_key}")
    error_examples(test, best_key)
    attention_figure(test)
    latency()
