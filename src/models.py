"""Neural architectures for Hinglish sentiment classification (PyTorch).

All models take a LongTensor of token ids (B, T) with 0 = <pad> and return
logits (B, num_classes).

1. BiLSTMAttn     - embedding -> BiLSTM -> additive attention pooling -> linear
2. TextCNN        - Kim (2014): parallel 1-D convs of several widths + max-over-time pooling.
                    Used at word level AND at character level (CharCNN).
3. TransformerCls - small Transformer encoder written from scratch
                    (sinusoidal positions, multi-head scaled dot-product self-attention,
                    pre-LayerNorm blocks, masked mean pooling).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD = 0


# --------------------------------------------------------------------------- #
# 1. BiLSTM + additive attention
# --------------------------------------------------------------------------- #
class BiLSTMAttn(nn.Module):
    """Why BiLSTM: reads the sentence in both directions, so "nahi" can modify a
    word that comes after it ("nahi accha") as well as before it.
    Why attention pooling instead of the last hidden state: the model learns
    *which* words carry the sentiment, and the weights are inspectable."""

    def __init__(self, vocab_size, emb_dim=128, hidden=128, num_classes=3, dropout=0.4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.emb_drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.attn_proj = nn.Linear(2 * hidden, 2 * hidden)
        self.attn_vec = nn.Linear(2 * hidden, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden, num_classes)

    def forward(self, x, return_attn: bool = False):
        mask = x.ne(PAD)                                  # (B, T) True = real token
        lengths = mask.sum(1).clamp(min=1)
        e = self.emb_drop(self.emb(x))
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        # additive (Bahdanau-style) attention: score_t = v^T tanh(W h_t)
        scores = self.attn_vec(torch.tanh(self.attn_proj(out))).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)          # never attend to padding
        alpha = F.softmax(scores, dim=1)                  # (B, T), sums to 1
        ctx = torch.bmm(alpha.unsqueeze(1), out).squeeze(1)
        logits = self.fc(self.drop(ctx))
        return (logits, alpha) if return_attn else logits


# --------------------------------------------------------------------------- #
# 2. TextCNN (word- or character-level)
# --------------------------------------------------------------------------- #
class TextCNN(nn.Module):
    """Each kernel width k acts as a learned k-gram detector. Max-over-time pooling
    keeps the strongest match anywhere in the text, so it is position invariant.
    At character level the kernels become learned character n-grams, which makes
    the model tolerant to Hinglish spelling variation ("accha"/"acha"/"achha")."""

    def __init__(self, vocab_size, emb_dim=128, num_filters=100, kernel_sizes=(2, 3, 4),
                 num_classes=3, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, num_filters, k) for k in kernel_sizes])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        lengths = x.ne(PAD).sum(1)                        # (B,)
        e = self.emb(x).transpose(1, 2)                   # (B, emb, T)
        feats = []
        for conv in self.convs:
            h = F.relu(conv(e))                           # (B, F, T-k+1)
            pos = torch.arange(h.size(2), device=x.device)[None, :]
            valid = pos < lengths[:, None].clamp(min=1)   # window must start inside the text
            h = h.masked_fill(~valid[:, None, :], 0.0)    # ReLU >= 0, so 0 never wins by accident
            feats.append(h.max(dim=2).values)
        return self.fc(self.drop(torch.cat(feats, dim=1)))


# --------------------------------------------------------------------------- #
# 3. Small Transformer encoder, from scratch
# --------------------------------------------------------------------------- #
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.h, self.dk = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_valid):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)  # (B,h,T,dk)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dk)     # (B,h,T,T) scaled dot-product
        scores = scores.masked_fill(~key_valid[:, None, None, :], -1e9)
        attn = self.drop(F.softmax(scores, dim=-1))
        ctx = (attn @ v).transpose(1, 2).reshape(B, T, D)         # concat heads
        return self.out(ctx)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_valid):
        x = x + self.drop(self.attn(self.ln1(x), key_valid))      # pre-LN residual
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


def sinusoidal_positions(max_len: int, d_model: int) -> torch.Tensor:
    pos = torch.arange(max_len).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class TransformerCls(nn.Module):
    """Honest expectation: with ~3k training sentences and no pretraining, a
    Transformer usually does NOT beat BiLSTM/CNN. It is included to show that
    (and to contrast with a *pretrained* transformer in the Colab notebook)."""

    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, d_ff=256,
                 num_classes=3, dropout=0.2, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.register_buffer("pe", sinusoidal_positions(max_len, d_model), persistent=False)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        valid = x.ne(PAD)
        h = self.emb(x) * math.sqrt(self.d_model) + self.pe[: x.size(1)]
        h = self.drop(h)
        for blk in self.blocks:
            h = blk(h, valid)
        h = self.ln(h)
        m = valid.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)        # masked mean pooling
        return self.fc(pooled)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
