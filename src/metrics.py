"""Shared evaluation helpers: metrics, bootstrap confidence interval, confusion-matrix plot."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

LABEL_NAMES = ["negative", "neutral", "positive"]


def compute_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_negative": float(per_class[0]),
        "f1_neutral": float(per_class[1]),
        "f1_positive": float(per_class[2]),
    }


def bootstrap_ci(y_true, y_pred, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """95% percentile bootstrap CI for macro-F1 (test set is small, so this matters)."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        scores.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
    lo, hi = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def text_report(y_true, y_pred) -> str:
    return classification_report(
        y_true, y_pred, labels=[0, 1, 2], target_names=LABEL_NAMES, digits=3, zero_division=0
    )


def plot_confusion(y_true, y_pred, title: str, out_path: Path, normalize: bool = True):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    shown = cm.astype(float) / cm.sum(axis=1, keepdims=True) if normalize else cm
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(3), LABEL_NAMES, rotation=20)
    ax.set_yticks(range(3), LABEL_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    for i in range(3):
        for j in range(3):
            txt = f"{shown[i, j]:.2f}\n(n={cm[i, j]})" if normalize else f"{cm[i, j]}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if shown[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
