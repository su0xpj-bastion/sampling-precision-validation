"""A small next-token model to measure precision loss on: real matmuls, real nonlinearities.

Shape: embed(last CONTEXT tokens, mean) → N residual MLP blocks → layernorm → tied
unembedding. Small enough to run in CI, big enough that quantization error compounds
through depth the way it does in an LLM.

Layer OUTLIER_LAYER is built with activation outliers: a few hidden channels run
~50x hotter than the rest, which is the well-documented LLM behaviour that breaks
per-tensor INT8 / FP8 activation scaling (one huge channel sets the scale and
crushes every other channel to a handful of levels). The function is unchanged in
fp32: those channels' weights in the next matmul are scaled down to compensate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from spv.formats import SCALED, quantize

VOCAB, DIM, HIDDEN, LAYERS, CONTEXT = 256, 64, 256, 6, 4
OUTLIER_LAYER, OUTLIER_CHANNELS, OUTLIER_GAIN = 3, (7, 91, 200), 50.0
LOGIT_SCALE = 0.25   # keeps next-token entropy LLM-like; unscaled tied logits are unrealistically peaky


@dataclass
class QuantConfig:
    fmt: str = "fp32"
    weight_axis: int | None = 0          # 0 = per-output-channel weight scales; None = per-tensor
    act_axis: int | None = None          # None = per-tensor activation scale; 1 = per-token
    only_layers: set[int] | None = None  # quantize just these layers (localization); None = all
    keep_high: set[int] = field(default_factory=set)   # layers kept in bf16 (mixed precision)
    smooth_alpha: float = 0.0            # SmoothQuant migration strength; 0 = off

    def fmt_for(self, layer: int) -> str:
        if self.only_layers is not None and layer not in self.only_layers:
            return "fp32"
        return "bf16" if layer in self.keep_high else self.fmt


@dataclass
class Model:
    embed: np.ndarray                    # (VOCAB, DIM)
    w1: list[np.ndarray]                 # (DIM, HIDDEN)
    w2: list[np.ndarray]                 # (HIDDEN, DIM)
    act_absmax: list[np.ndarray] | None = None   # per-channel calibration for SmoothQuant


def build(seed: int = 0) -> Model:
    rng = np.random.default_rng(seed)
    embed = rng.standard_normal((VOCAB, DIM)) * 1.0
    w1 = [rng.standard_normal((DIM, HIDDEN)) * DIM ** -0.5 for _ in range(LAYERS)]
    w2 = [rng.standard_normal((HIDDEN, DIM)) * HIDDEN ** -0.5 * 0.8 for _ in range(LAYERS)]
    for c in OUTLIER_CHANNELS:
        w1[OUTLIER_LAYER][:, c] = np.abs(w1[OUTLIER_LAYER][:, c]) * OUTLIER_GAIN   # hot, mostly positive
        w2[OUTLIER_LAYER][c, :] /= OUTLIER_GAIN
    return Model(embed, w1, w2)


def _ln(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def _qmatmul(a: np.ndarray, w: np.ndarray, fmt: str, qc: QuantConfig) -> np.ndarray:
    if fmt == "fp32":
        return a @ w
    if fmt in SCALED:
        return quantize(a, fmt, qc.act_axis) @ quantize(w, fmt, qc.weight_axis)
    return quantize(a, fmt) @ quantize(w, fmt)


def forward(m: Model, ctx: np.ndarray, qc: QuantConfig | None = None, record: list | None = None) -> np.ndarray:
    """ctx: (N, CONTEXT) token ids → (N, VOCAB) logits (float64)."""
    qc = qc or QuantConfig()
    h = m.embed[ctx].mean(axis=1)
    for i in range(LAYERS):
        fmt = qc.fmt_for(i)
        a = _gelu(_qmatmul(_ln(h), m.w1[i], fmt, qc))
        if record is not None:
            record.append(np.abs(a).max(axis=0))
        w2 = m.w2[i]
        if qc.smooth_alpha and fmt in SCALED and m.act_absmax is not None:
            # SmoothQuant: divide activation channel j by s_j, multiply weight row j by s_j.
            # Mathematically identical in fp32; moves outlier magnitude into the weights,
            # which are quantized per-output-channel and absorb it far better.
            wmax = np.abs(w2).max(axis=1)
            s = np.maximum(m.act_absmax[i] ** qc.smooth_alpha / np.maximum(wmax, 1e-8) ** (1 - qc.smooth_alpha), 1e-5)
            a, w2 = a / s, w2 * s[:, None]
        h = h + _qmatmul(a, w2, fmt, qc)
    return (_ln(h) @ m.embed.T) * LOGIT_SCALE


def calibrate(m: Model, ctx: np.ndarray) -> Model:
    """Record per-channel activation absmax on calibration data (needed for SmoothQuant)."""
    rec: list[np.ndarray] = []
    forward(m, ctx, record=rec)
    m.act_absmax = rec
    return m
