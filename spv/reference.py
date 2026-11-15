"""Golden sampler: plain Python, one row at a time, written from the spec.

It shares no code with spv/sampler.py on purpose. A golden that reuses the
vectorized code can only agree with it.
"""
from __future__ import annotations

import math

from spv.sampler import SamplingParams


def distribution(logits: list[float], params: SamplingParams, prev: set[int] = frozenset()) -> list[float]:
    x = list(logits)
    if params.repetition_penalty != 1.0:
        for t in prev:
            x[t] = x[t] / params.repetition_penalty if x[t] > 0 else x[t] * params.repetition_penalty
    n = len(x)
    if params.temperature == 0.0:
        best = max(range(n), key=lambda i: (x[i], -i))     # highest logit, lowest id on ties
        return [1.0 if i == best else 0.0 for i in range(n)]

    scaled = [v / params.temperature for v in x]
    m = max(scaled)
    e = [math.exp(v - m) if v != -math.inf else 0.0 for v in scaled]
    s = sum(e)
    p = [v / s for v in e]

    if params.top_k and params.top_k < n:
        kth = sorted(p, reverse=True)[params.top_k - 1]
        p = [v if v >= kth else 0.0 for v in p]
        s = sum(p)
        p = [v / s for v in p]

    if params.top_p < 1.0:
        order = sorted(range(n), key=lambda i: (-p[i], i))
        keep, cum = set(), 0.0
        for i in order:
            if cum >= params.top_p:        # the token that crosses top_p is kept
                break
            keep.add(i)
            cum += p[i]
        p = [v if i in keep else 0.0 for i, v in enumerate(p)]
        s = sum(p)
        p = [v / s for v in p]

    if params.min_p > 0.0:
        floor = params.min_p * max(p)
        p = [v if v >= floor else 0.0 for v in p]
        s = sum(p)
        p = [v / s for v in p]
    return p
