"""Vectorized LLM token sampler: the system under test.

Processing order matches common serving engines:
    repetition penalty → temperature → softmax → top-k → top-p → min-p → draw

Determinism contract: each request draws from its own generator seeded by
(seed, step), so a request's token never depends on batch size, batch
position, or its neighbours (batch invariance).

`bugs` turns on realistic defects seen in real sampler code, so the audit can
prove it would catch them:
  topp_boundary     keeps tokens with cumsum < p, which drops the token that
                    crosses p. When the top token alone is >= p, nothing is
                    kept and the distribution is NaN.
  rep_penalty_sign  divides every repeated logit by the penalty, so a NEGATIVE
                    logit moves toward 0 and the token gets MORE likely
  shared_rng        one generator per batch: tokens depend on batch composition
  temp_after_topp   applies temperature after the nucleus is chosen, so the wrong
                    token set survives
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KNOWN_BUGS = ("topp_boundary", "rep_penalty_sign", "shared_rng", "temp_after_topp")


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 1.0
    top_k: int = 0             # 0 = off
    top_p: float = 1.0         # 1.0 = off
    min_p: float = 0.0         # 0.0 = off
    repetition_penalty: float = 1.0

    def __post_init__(self) -> None:
        if self.temperature < 0 or not 0 < self.top_p <= 1 or not 0 <= self.min_p <= 1:
            raise ValueError(f"invalid sampling params: {self}")
        if self.top_k < 0 or self.repetition_penalty <= 0:
            raise ValueError(f"invalid sampling params: {self}")


def _renorm(p: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return p / p.sum(axis=-1, keepdims=True)


def _softmax(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max(axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


def process(logits: np.ndarray, params: SamplingParams, prev_mask: np.ndarray | None = None,
            bugs: frozenset[str] = frozenset()) -> np.ndarray:
    """(B, V) logits → (B, V) sampling distribution. Rows sum to 1 (or NaN under some bugs)."""
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("logits must be (batch, vocab)")
    if np.isnan(x).any():
        raise ValueError("NaN in logits: refusing to sample (upstream numerics bug)")
    x = x.copy()

    rp = params.repetition_penalty
    if rp != 1.0 and prev_mask is not None:
        if "rep_penalty_sign" in bugs:
            x = np.where(prev_mask, x / rp, x)
        else:
            x = np.where(prev_mask, np.where(x > 0, x / rp, x * rp), x)

    if params.temperature == 0.0:                       # greedy; ties → lowest token id
        out = np.zeros_like(x)
        out[np.arange(len(x)), np.argmax(x, axis=-1)] = 1.0
        return out

    if "temp_after_topp" not in bugs:
        x = x / params.temperature
    probs = _softmax(x)

    if params.top_k and params.top_k < probs.shape[1]:
        # O(V) selection instead of a full sort: the hot path at 128k-token vocabularies
        kth = -np.partition(-probs, params.top_k - 1, axis=-1)[:, params.top_k - 1: params.top_k]
        probs = _renorm(np.where(probs >= kth, probs, 0.0))   # ties at the k-th value are kept

    if params.top_p < 1.0:
        order = np.argsort(-probs, axis=-1, kind="stable")
        sorted_p = np.take_along_axis(probs, order, axis=-1)
        cum = np.cumsum(sorted_p, axis=-1)
        keep_sorted = (cum < params.top_p) if "topp_boundary" in bugs else (cum - sorted_p < params.top_p)
        keep = np.zeros_like(keep_sorted)
        np.put_along_axis(keep, order, keep_sorted, axis=-1)
        probs = _renorm(np.where(keep, probs, 0.0))

    if "temp_after_topp" in bugs:
        with np.errstate(divide="ignore"):
            probs = _renorm(np.where(probs > 0, probs ** (1.0 / params.temperature), 0.0))

    if params.min_p > 0.0:
        floor = params.min_p * probs.max(axis=-1, keepdims=True)
        probs = _renorm(np.where(probs >= floor, probs, 0.0))
    return probs


def draw(probs: np.ndarray, seeds: list[int], step: int, bugs: frozenset[str] = frozenset()) -> np.ndarray:
    """Inverse-CDF draw, one uniform per request."""
    if "shared_rng" in bugs:
        u = np.random.default_rng([seeds[0], step]).random(len(seeds))
    else:
        u = np.array([np.random.default_rng([s, step]).random() for s in seeds])
    cdf = np.cumsum(probs, axis=-1)
    tok = (cdf < (u * cdf[:, -1])[:, None]).sum(axis=-1)
    return np.minimum(tok, probs.shape[1] - 1)


def sample(logits: np.ndarray, params: SamplingParams, seeds: list[int], step: int = 0,
           prev_mask: np.ndarray | None = None, bugs: frozenset[str] = frozenset()) -> np.ndarray:
    probs = process(logits, params, prev_mask, bugs)
    if np.isnan(probs).any():
        raise FloatingPointError("sampling distribution is NaN (empty candidate set?)")
    return draw(probs, seeds, step, bugs)
