# Validation strategy: sampling and low precision

## Why example tests fail here
A sampler is random by design, and a quantized model is wrong by design (by a small amount). "Output equals expected string" tests either flake or can't tell a 0.1% shift from a 10% one. Both need reference-based, statistical checks with explicit tolerances.

## Sampler: what a release must prove
| Property | Check | Bug class it exists for |
|---|---|---|
| Distribution is right | golden equivalence (exact) + chi-square on seeded draws | boundary errors, wrong processing order, sign errors |
| Support is right | never draw a zero-probability token | masks that leak (top-p / min-p / -inf) |
| Batch invariance | token identical alone or in any batch position | shared RNG, cross-request state |
| Determinism | same seed + step → same token | RNG seeded from time or thread state |
| Edge cases | T=0, top_k=1, tiny top_p, -inf, NaN, penalty sign | crashes and silent NaN at the extremes |

Fix the seeds for statistical checks. A chi-square test with random seeds fails 1 in 10,000 runs at p=1e-4. With fixed seeds it either always passes or always fails, which is the only kind of statistical test CI should gate on.

## Precision: which signals gate a release
| Signal | Gate? | Why |
|---|---|---|
| Top-1 agreement on **decisive** positions | yes | flips only when the model's answer really changes |
| Raw top-1 agreement | report only | near-ties flip on any rounding noise |
| Mean KL vs fp32 | yes | catches distribution drift that argmax can't see |
| Perplexity Δ on reference targets | yes | the user-visible quality proxy |
| Per-layer KL share | localization, not gating | tells the owner which layer and which mitigation to try |

Set thresholds *between* the observed passing and failing populations, never at the edge of the passing one, and re-derive them when the model changes.

## When a format fails the gate
1. Localize: quantize one layer at a time. A single dominant layer means outliers or a sensitive op. Diffuse error means the format itself is too coarse (E5M2 here).
2. Mitigate in order of cost: keep that layer in higher precision, then SmoothQuant or per-channel scales, then a different format.
3. Re-run the full gate, not just the failing metric.
