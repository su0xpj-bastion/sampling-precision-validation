"""Low-precision accuracy gate: compare a quantized model to its fp32 reference.

Metrics, per format, on a held-out evaluation set:
  top1_agreement   share of DECISIVE positions (fp32 top-1 beats top-2 by >= DECISIVE_MARGIN)
                   where the argmax is unchanged. Near-tie positions flip on any noise and
                   would make the gate flaky; raw agreement is reported but not gated
  top1_raw         agreement over all positions, for information only
  kl_mean          mean KL(p_fp32 || p_quant) over next-token distributions (nats)
  ppl_delta_pct    perplexity change on reference targets, in percent

Localization: quantize ONE layer at a time (everything else fp32) and attribute
KL to each layer. That answers "which layer broke under INT8 or FP8" without bisecting by hand.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from spv.model import CONTEXT, LAYERS, VOCAB, Model, QuantConfig, build, calibrate, forward

ENCODING = "utf-8"
GATES_FILE = Path(__file__).resolve().parents[1] / "gates.json"
EVAL_SEED, CALIB_SEED, N_EVAL, N_CALIB = 1234, 99, 2048, 512
DECISIVE_MARGIN = 0.05


@dataclass
class Metrics:
    fmt: str
    top1_agreement: float
    top1_raw: float
    kl_mean: float
    ppl_delta_pct: float


def _logsoftmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def dataset(m: Model, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Contexts + targets sampled from the fp32 model itself, so perplexity is meaningful."""
    rng = np.random.default_rng(seed)
    ctx = rng.integers(0, VOCAB, (n, CONTEXT))
    p = np.exp(_logsoftmax(forward(m, ctx)))
    u = rng.random(n)[:, None]
    targets = (np.cumsum(p, -1) < u).sum(-1).clip(0, VOCAB - 1)
    return ctx, targets


def compare(m: Model, ctx: np.ndarray, targets: np.ndarray, qc: QuantConfig) -> Metrics:
    ref, q = _logsoftmax(forward(m, ctx)), _logsoftmax(forward(m, ctx, qc))
    pr = np.exp(ref)
    rows = np.arange(len(targets))
    ppl_ref, ppl_q = np.exp(-ref[rows, targets].mean()), np.exp(-q[rows, targets].mean())
    top2 = np.sort(pr, -1)[:, -2:]
    decisive = (top2[:, 1] - top2[:, 0]) >= DECISIVE_MARGIN
    same = ref.argmax(-1) == q.argmax(-1)
    return Metrics(
        fmt=qc.fmt,
        top1_agreement=float(same[decisive].mean()),
        top1_raw=float(same.mean()),
        kl_mean=float((pr * (ref - q)).sum(-1).mean()),
        ppl_delta_pct=float((ppl_q / ppl_ref - 1) * 100),
    )


def localize(m: Model, ctx: np.ndarray, targets: np.ndarray, qc: QuantConfig) -> list[dict]:
    """KL when only layer i is quantized, and each layer's share of the summed per-layer KL."""
    per = []
    for i in range(LAYERS):
        single = QuantConfig(**{**asdict(qc), "only_layers": {i}})
        per.append(compare(m, ctx, targets, single).kl_mean)
    total = sum(per) or 1.0
    return [{"layer": i, "kl": kl, "share": kl / total} for i, kl in enumerate(per)]


def load_gates(path: Path = GATES_FILE) -> dict:
    with path.open("r", encoding=ENCODING) as fh:
        return json.load(fh)


def check(metrics: Metrics, gate: dict) -> list[str]:
    out = []
    if metrics.top1_agreement < gate["top1_agreement_min"]:
        out.append(f"top1 agreement {metrics.top1_agreement:.4f} < {gate['top1_agreement_min']}")
    if metrics.kl_mean > gate["kl_max"]:
        out.append(f"KL {metrics.kl_mean:.2e} > {gate['kl_max']:.1e}")
    if abs(metrics.ppl_delta_pct) > gate["ppl_delta_pct_max"]:
        out.append(f"|ppl Δ| {abs(metrics.ppl_delta_pct):.2f}% > {gate['ppl_delta_pct_max']}%")
    return out


def setup(seed: int = 0) -> tuple[Model, np.ndarray, np.ndarray]:
    m = build(seed)
    calib_ctx, _ = dataset(m, N_CALIB, CALIB_SEED)
    calibrate(m, calib_ctx)
    ctx, targets = dataset(m, N_EVAL, EVAL_SEED)
    return m, ctx, targets
