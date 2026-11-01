"""Train and compare BiLSTM+attention, TextCNN, CharCNN and a from-scratch Transformer.

Protocol (identical for every model, so the comparison is fair):
  * splits: fixed stratified 80/10/10 (data/processed/*.csv)
  * hyper-parameters: small grid, chosen on the VALIDATION set (seed 42)
  * early stopping + checkpoint selection on validation macro-F1
  * the TEST set is touched exactly once per run, at the selected checkpoint
  * best config is re-run with 3 seeds -> mean +/- std (test set has only 388 rows,
    so single-run differences of 1-2 F1 points are noise)

Usage:
    python src/train_dl.py                     # full experiment (CPU, several minutes)
    python src/train_dl.py --quick             # 1-epoch smoke test, writes nothing important
    python src/train_dl.py --models bilstm_attn textcnn
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset

from data import load_splits, tokenize
from metrics import bootstrap_ci, compute_metrics, plot_confusion
from models import BiLSTMAttn, TextCNN, TransformerCls, count_params

ROOT = Path(__file__).resolve().parents[1]
RES, FIG, MOD = ROOT / "results", ROOT / "figures", ROOT / "models"
torch.set_num_threads(2)

PAD, UNK = 0, 1

# Each spec: input level, max length, class, and a small hyper-parameter grid.
SPECS = {
    "bilstm_attn": dict(
        level="word", max_len=50, min_freq=2, cls=BiLSTMAttn,
        grid=[
            dict(model=dict(emb_dim=128, hidden=128, dropout=0.4), lr=2e-3),
            dict(model=dict(emb_dim=128, hidden=128, dropout=0.5), lr=1e-3),
            dict(model=dict(emb_dim=100, hidden=64, dropout=0.3), lr=2e-3),
        ],
    ),
    "textcnn": dict(
        level="word", max_len=50, min_freq=2, cls=TextCNN,
        grid=[
            dict(model=dict(emb_dim=128, num_filters=100, kernel_sizes=(2, 3, 4), dropout=0.5), lr=2e-3),
            dict(model=dict(emb_dim=128, num_filters=64, kernel_sizes=(1, 2, 3), dropout=0.5), lr=2e-3),
            dict(model=dict(emb_dim=100, num_filters=128, kernel_sizes=(2, 3, 4, 5), dropout=0.6), lr=1e-3),
        ],
    ),
    "charcnn": dict(
        level="char", max_len=200, min_freq=2, cls=TextCNN,
        grid=[
            dict(model=dict(emb_dim=48, num_filters=128, kernel_sizes=(3, 5, 7), dropout=0.5), lr=2e-3),
            dict(model=dict(emb_dim=48, num_filters=96, kernel_sizes=(2, 4, 6), dropout=0.5), lr=2e-3),
            dict(model=dict(emb_dim=64, num_filters=128, kernel_sizes=(3, 5, 7), dropout=0.6), lr=1e-3),
        ],
    ),
    "transformer": dict(
        level="word", max_len=50, min_freq=2, cls=TransformerCls,
        grid=[
            dict(model=dict(d_model=128, n_heads=4, n_layers=2, d_ff=256, dropout=0.2), lr=1e-3),
            dict(model=dict(d_model=128, n_heads=4, n_layers=2, d_ff=256, dropout=0.4), lr=5e-4),
            dict(model=dict(d_model=64, n_heads=4, n_layers=1, d_ff=128, dropout=0.3), lr=1e-3),
        ],
    ),
}


# --------------------------------------------------------------------------- #
# Data -> tensors
# --------------------------------------------------------------------------- #
def to_units(text: str, level: str) -> list[str]:
    return tokenize(text) if level == "word" else list(text)


def build_vocab(texts, level, min_freq):
    c = Counter(u for t in texts for u in to_units(t, level))
    itos = ["<pad>", "<unk>"] + [u for u, n in c.most_common() if n >= min_freq]
    return {u: i for i, u in enumerate(itos)}, itos


def encode(texts, vocab, level, max_len):
    out = np.zeros((len(texts), max_len), dtype=np.int64)
    truncated = 0
    for i, t in enumerate(texts):
        units = to_units(t, level)
        truncated += len(units) > max_len
        ids = [vocab.get(u, UNK) for u in units][:max_len]
        out[i, : len(ids)] = ids
    return torch.from_numpy(out), truncated / max(len(texts), 1)


class Data:
    """Everything a run needs for one input level (word or char)."""

    def __init__(self, level: str, max_len: int, min_freq: int):
        train, val, test = load_splits("full")
        self.vocab, self.itos = build_vocab(train["text"], level, min_freq)
        self.level, self.max_len = level, max_len
        self.Xtr, self.trunc_train = encode(train["text"], self.vocab, level, max_len)
        self.Xva, _ = encode(val["text"], self.vocab, level, max_len)
        self.Xte, _ = encode(test["text"], self.vocab, level, max_len)
        self.ytr = torch.tensor(train["label"].values)
        self.yva = torch.tensor(val["label"].values)
        self.yte = torch.tensor(test["label"].values)
        self.test_df = test
        # share of validation units that are out-of-vocabulary (word level: spelling variation hurts)
        units = [u for t in val["text"] for u in to_units(t, level)]
        self.oov_val = float(np.mean([u not in self.vocab for u in units]))
        counts = np.bincount(train["label"].values, minlength=3)
        self.class_weights = torch.tensor(len(train) / (3 * counts), dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model, X, batch_size=256):
    model.eval()
    logits = torch.cat([model(X[i: i + batch_size]) for i in range(0, len(X), batch_size)])
    return logits


def run_once(name, cand, seed, D: Data, epochs, patience, label_smoothing=0.05, batch_size=64):
    spec = SPECS[name]
    set_seed(seed)
    model = spec["cls"](vocab_size=len(D.vocab), num_classes=3, **cand["model"])
    opt = AdamW(model.parameters(), lr=cand["lr"], weight_decay=1e-2)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(D.Xtr, D.ytr), batch_size=batch_size, shuffle=True, generator=g)
    total, warm = epochs * len(loader), max(1, 2 * len(loader))
    sched = LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    loss_fn = nn.CrossEntropyLoss(weight=D.class_weights, label_smoothing=label_smoothing)
    plain_ce = nn.CrossEntropyLoss()

    best = dict(f1=-1.0, loss=1e9, epoch=0, state=None)
    history, bad, t0 = [], 0, time.time()
    for ep in range(1, epochs + 1):
        model.train()
        run_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # keeps LSTM gradients stable
            opt.step()
            sched.step()
            run_loss += loss.item() * len(xb)
        vlog = predict(model, D.Xva)
        v_f1 = compute_metrics(D.yva.numpy(), vlog.argmax(1).numpy())["macro_f1"]
        v_loss = plain_ce(vlog, D.yva).item()
        history.append(dict(epoch=ep, train_loss=run_loss / len(D.Xtr), val_loss=v_loss, val_macro_f1=v_f1))
        if v_f1 > best["f1"] or (v_f1 == best["f1"] and v_loss < best["loss"]):
            best = dict(f1=v_f1, loss=v_loss, epoch=ep, state=copy.deepcopy(model.state_dict()))
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best["state"])
    pred = predict(model, D.Xte).argmax(1).numpy()
    metrics = compute_metrics(D.yte.numpy(), pred)
    return dict(metrics=metrics, val_macro_f1=best["f1"], best_epoch=best["epoch"],
                epochs_run=len(history), train_seconds=time.time() - t0,
                n_params=count_params(model), history=history, pred=pred,
                state=best["state"])


def save_checkpoint(name, cand, D: Data, state):
    MOD.mkdir(exist_ok=True)
    torch.save(dict(name=name, cand=cand, level=D.level, max_len=D.max_len,
                    itos=D.itos, state_dict=state), MOD / f"{name}.pt")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(SPECS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--quick", action="store_true", help="1 epoch, 1 seed, first grid entry only")
    args = ap.parse_args()
    if args.quick:
        args.epochs, args.seeds = 1, args.seeds[:1]
    tag = "_smoke" if args.quick else ""
    RES.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)

    summary, curves = {}, {}
    for name in args.models:
        spec = SPECS[name]
        D = Data(spec["level"], spec["max_len"], spec["min_freq"])
        print(f"\n=== {name}  (level={D.level}, vocab={len(D.vocab)}, "
              f"val OOV={D.oov_val:.1%}, truncated train={D.trunc_train:.1%}) ===")

        # 1) hyper-parameter selection on validation macro-F1 (seed 42)
        grid = spec["grid"][:1] if args.quick else spec["grid"]
        tuned = []
        for i, cand in enumerate(grid):
            r = run_once(name, cand, args.seeds[0], D, args.epochs, args.patience)
            tuned.append((r["val_macro_f1"], i, cand, r))
            print(f"  cand {i}: val macro-F1={r['val_macro_f1']:.3f} (epoch {r['best_epoch']}, "
                  f"{r['train_seconds']:.0f}s)  {cand}")
        _, bi, best_cand, first = max(tuned, key=lambda t: (t[0], -t[1]))

        # 2) re-run best config with several seeds (seed 1 result is reused from the grid)
        runs = [first]
        for s in args.seeds[1:]:
            runs.append(run_once(name, best_cand, s, D, args.epochs, args.patience))

        # 3) persist
        for s, r in zip(args.seeds, runs):
            D.test_df.assign(pred=r["pred"]).to_csv(RES / f"preds_{name}_s{s}{tag}.csv", index=False)
        if not args.quick:
            save_checkpoint(name, best_cand, D, first["state"])
            plot_confusion(D.yte.numpy(), first["pred"], name, FIG / f"cm_{name}.png")
        curves[name] = first["history"]

        f1s = np.array([r["metrics"]["macro_f1"] for r in runs])
        accs = np.array([r["metrics"]["accuracy"] for r in runs])
        lo, hi = bootstrap_ci(D.yte.numpy(), first["pred"])
        summary[name] = dict(
            level=D.level, vocab_size=len(D.vocab), val_oov_rate=D.oov_val,
            n_params=first["n_params"], best_config=best_cand,
            grid_val_f1=[round(t[0], 4) for t in tuned],
            seeds=args.seeds, macro_f1_per_seed=f1s.tolist(),
            macro_f1_mean=float(f1s.mean()), macro_f1_std=float(f1s.std(ddof=0)),
            accuracy_mean=float(accs.mean()), accuracy_std=float(accs.std(ddof=0)),
            f1_negative_mean=float(np.mean([r["metrics"]["f1_negative"] for r in runs])),
            f1_neutral_mean=float(np.mean([r["metrics"]["f1_neutral"] for r in runs])),
            f1_positive_mean=float(np.mean([r["metrics"]["f1_positive"] for r in runs])),
            seed42_macro_f1_ci95=[lo, hi],
            mean_train_seconds=float(np.mean([r["train_seconds"] for r in runs])),
            mean_best_epoch=float(np.mean([r["best_epoch"] for r in runs])),
        )
        print(f"  -> TEST macro-F1 {f1s.mean():.3f} +/- {f1s.std():.3f} over seeds {args.seeds}  "
              f"(params={first['n_params']:,})")

    (RES / f"dl_results{tag}.json").write_text(json.dumps(summary, indent=2))
    (RES / f"dl_curves{tag}.json").write_text(json.dumps(curves))
    print(f"\nsaved results/dl_results{tag}.json")


if __name__ == "__main__":
    main()
