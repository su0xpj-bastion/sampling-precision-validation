"""Sampler audit: named checks that together decide whether a sampler build is correct.

  golden_equivalence  vectorized distribution == plain-Python golden, over a param grid
  distribution_fit    20k seeded draws fit the golden distribution (chi-square),
                      and never land on a zero-probability token (support check)
  batch_invariance    a request's token is identical alone, or anywhere in a batch of 32
  seed_determinism    the same seeds and step give the same tokens on every run
  edge_cases          T=0 is argmax · top_k=1 equals greedy · a tiny top_p keeps exactly
                      the top token · -inf logits are never drawn · NaN logits are rejected ·
                      repetition penalty lowers the repeated token's probability for
                      positive AND negative logits

Statistical checks use fixed seeds, so they are deterministic: a pass/fail never flakes.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.stats import chisquare

from spv import reference
from spv.sampler import SamplingParams, draw, process, sample

P_VALUE_FLOOR = 1e-4        # chi-square: below this, the draws do not come from the golden distribution
N_DRAWS = 20_000
GOLDEN_ATOL = 1e-9


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def _logits(seed: int, vocab: int = 32, scale: float = 2.5) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((1, vocab)) * scale


PARAM_GRID = [
    SamplingParams(),
    SamplingParams(temperature=0.7),
    SamplingParams(temperature=1.3, top_k=5),
    SamplingParams(top_p=0.9),
    SamplingParams(temperature=0.6, top_p=0.8),
    SamplingParams(temperature=1.5, top_k=10, top_p=0.95),
    SamplingParams(min_p=0.1),
    SamplingParams(temperature=0.8, repetition_penalty=1.3),
    SamplingParams(temperature=0.9, top_p=0.7, min_p=0.05, repetition_penalty=1.2),
]


def golden_equivalence(bugs: frozenset[str]) -> Check:
    worst, where = 0.0, ""
    for seed, params in itertools.product(range(12), PARAM_GRID):
        x = _logits(seed)
        prev = {0, 3, 7, int(np.argmin(x[0]))}           # includes a negative logit on purpose
        mask = np.zeros_like(x, dtype=bool)
        mask[0, list(prev)] = True
        got = process(x, params, mask, bugs)[0]
        want = np.array(reference.distribution(list(x[0]), params, prev))
        err = np.inf if np.isnan(got).any() else float(np.abs(got - want).max())
        if err > worst:
            worst, where = err, f"seed={seed} {params}"
    return Check("golden_equivalence", worst <= GOLDEN_ATOL, f"max |Δp| {worst:.2e}" + (f" at {where}" if where else ""))


FIT_CASES = [
    (101, SamplingParams(temperature=0.8)),
    (102, SamplingParams(top_p=0.85)),
    (103, SamplingParams(temperature=1.2, top_k=6)),
    (104, SamplingParams(min_p=0.08, temperature=0.9)),
]


def distribution_fit(bugs: frozenset[str]) -> Check:
    worst_p, msg = 1.0, ""
    for seed, params in FIT_CASES:
        x = _logits(seed, vocab=16, scale=1.5)
        want = np.array(reference.distribution(list(x[0]), params))
        probs = process(np.repeat(x, N_DRAWS, axis=0), params, bugs=bugs)
        if np.isnan(probs).any():
            return Check("distribution_fit", False, f"NaN distribution for {params}")
        counts = np.bincount(draw(probs, list(range(N_DRAWS)), step=0, bugs=bugs), minlength=x.shape[1])
        if (counts[want == 0] > 0).any():
            return Check("distribution_fit", False, f"drew a zero-probability token under {params}")
        support = want > 0
        p = float(chisquare(counts[support], want[support] * N_DRAWS).pvalue)
        if p < worst_p:
            worst_p, msg = p, f"{params}"
    return Check("distribution_fit", worst_p >= P_VALUE_FLOOR, f"min chi-square p={worst_p:.2e} ({msg})")


def batch_invariance(bugs: frozenset[str]) -> Check:
    params = SamplingParams(temperature=0.9, top_p=0.9)
    rng = np.random.default_rng(7)
    logits = rng.standard_normal((32, 64)) * 2
    seeds = [int(s) for s in rng.integers(0, 2**31, 32)]
    batched = sample(logits, params, seeds, step=3, bugs=bugs)
    alone = np.array([sample(logits[i:i + 1], params, [seeds[i]], step=3, bugs=bugs)[0] for i in range(32)])
    diff = int((batched != alone).sum())
    return Check("batch_invariance", diff == 0, f"{diff}/32 requests changed token with batch composition")


def seed_determinism(bugs: frozenset[str]) -> Check:
    params = SamplingParams(temperature=1.1, top_k=20)
    logits = np.random.default_rng(9).standard_normal((16, 64))
    a = sample(logits, params, list(range(16)), step=5, bugs=bugs)
    b = sample(logits, params, list(range(16)), step=5, bugs=bugs)
    return Check("seed_determinism", bool((a == b).all()), "identical across runs" if (a == b).all() else "differs")


def edge_cases(bugs: frozenset[str]) -> Check:
    failures: list[str] = []
    x = _logits(11)
    top = int(np.argmax(x[0]))
    try:
        if int(sample(x, SamplingParams(temperature=0.0), [1], bugs=bugs)[0]) != top:
            failures.append("T=0 is not argmax")
        if int(sample(x, SamplingParams(top_k=1), [2], bugs=bugs)[0]) != top:
            failures.append("top_k=1 is not greedy")
        tiny = process(x, SamplingParams(top_p=1e-6), bugs=bugs)[0]
        if np.isnan(tiny).any() or int(np.argmax(tiny)) != top or tiny[top] != 1.0:
            failures.append("tiny top_p does not keep exactly the top token")
    except (FloatingPointError, ValueError) as exc:
        failures.append(f"crash on a valid edge case: {exc}")
    masked = x.copy()
    masked[0, top] = -np.inf
    toks = sample(np.repeat(masked, 500, axis=0), SamplingParams(), list(range(500)), bugs=bugs)
    if (toks == top).any():
        failures.append("-inf logit was drawn")
    try:
        bad = x.copy()
        bad[0, 0] = np.nan
        process(bad, SamplingParams(), bugs=bugs)
        failures.append("NaN logits accepted")
    except ValueError:
        pass
    for sign in (1.0, -1.0):
        y = np.array([[sign * 1.5, 0.2, -0.3, 0.9]])
        mask = np.array([[True, False, False, False]])
        before = process(y, SamplingParams(), bugs=bugs)[0, 0]
        after = process(y, SamplingParams(repetition_penalty=1.5), mask, bugs)[0, 0]
        if not after < before:
            failures.append(f"repetition penalty raised a repeated token's probability (logit sign {sign:+})")
    return Check("edge_cases", not failures, "; ".join(failures) or "all edge cases hold")


CHECKS = (golden_equivalence, distribution_fit, batch_invariance, seed_determinism, edge_cases)


def audit(bugs: frozenset[str] = frozenset()) -> list[Check]:
    return [c(bugs) for c in CHECKS]
