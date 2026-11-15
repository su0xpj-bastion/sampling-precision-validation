"""spv — sampling & precision validation.

  python -m spv audit [--bug topp_boundary]          sampler audit (exit 1 on any failed check)
  python -m spv precision --fmt int8 [--keep-high 3] [--smooth 0.5] [--act-scale token]
                                                     accuracy gate + per-layer localization
  python -m spv report --out reports/                full matrix → markdown (CI job summary)
  python -m spv bench                                sampler throughput (fast path), informational
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from spv.audit import audit
from spv.model import OUTLIER_LAYER, QuantConfig
from spv.precision import check, compare, load_gates, localize, setup
from spv.sampler import KNOWN_BUGS, SamplingParams, sample

ENCODING = "utf-8"

PRECISION_MATRIX = [
    ("fp16", QuantConfig("fp16"), "expect pass"),
    ("bf16", QuantConfig("bf16"), "expect pass"),
    ("fp8_e4m3, per-tensor activations", QuantConfig("fp8_e4m3"), "expect pass: exponent absorbs outliers"),
    ("fp8_e5m2, per-tensor activations", QuantConfig("fp8_e5m2"), "expect FAIL: 2-bit mantissa too coarse for inference"),
    ("int8, per-tensor activations", QuantConfig("int8"), "expect FAIL: outlier layer crushes other channels"),
    (f"int8 + layer {OUTLIER_LAYER} kept in bf16", QuantConfig("int8", keep_high={OUTLIER_LAYER}), "expect pass"),
    ("int8 + SmoothQuant (alpha 0.5)", QuantConfig("int8", smooth_alpha=0.5), "expect pass"),
]


def _audit_rows(bugs: frozenset[str]) -> tuple[bool, list[str]]:
    checks = audit(bugs)
    return all(c.passed for c in checks), [f"| {c.name} | {'✅' if c.passed else '❌'} | {c.detail} |" for c in checks]


def cmd_audit(args: argparse.Namespace) -> int:
    ok, rows = _audit_rows(frozenset(args.bug or []))
    print("| Check | Result | Detail |\n|---|---|---|\n" + "\n".join(rows))
    return 0 if ok else 1


def _qc(args: argparse.Namespace) -> QuantConfig:
    return QuantConfig(args.fmt, act_axis=1 if args.act_scale == "token" else None,
                       keep_high=set(args.keep_high or []), smooth_alpha=args.smooth)


def cmd_precision(args: argparse.Namespace) -> int:
    m, ctx, targets = setup()
    qc = _qc(args)
    met = compare(m, ctx, targets, qc)
    violations = check(met, load_gates()[args.fmt])
    print(json.dumps({"metrics": asdict(met), "violations": violations}, indent=2))
    print("\n| Layer | KL if only this layer is quantized | Share |\n|---|---|---|")
    for row in localize(m, ctx, targets, qc):
        print(f"| {row['layer']} | {row['kl']:.2e} | {row['share']:.0%} |")
    return 1 if violations else 0


def cmd_report(args: argparse.Namespace) -> int:
    lines = ["## Sampler audit", "", "| Build | Verdict | Failed checks |", "|---|---|---|"]
    story_ok = True
    for bugs in [frozenset()] + [frozenset([b]) for b in KNOWN_BUGS]:
        checks = audit(bugs)
        failed = [c.name for c in checks if not c.passed]
        name = "correct" if not bugs else f"seeded bug `{next(iter(bugs))}`"
        caught = (not failed) if not bugs else bool(failed)
        story_ok &= caught
        verdict = ("✅ passes" if not failed else "❌ caught") if caught else "⚠️ MISSED"
        lines.append(f"| {name} | {verdict} | {', '.join(failed) or '-'} |")

    m, ctx, targets = setup()
    gates = load_gates()
    lines += ["", "## Low-precision accuracy gate (vs fp32)", "",
              "| Config | Top-1 (decisive) | KL (nats) | PPL Δ | Gate | Expectation |", "|---|---|---|---|---|---|"]
    for name, qc, expect in PRECISION_MATRIX:
        met = compare(m, ctx, targets, qc)
        v = check(met, gates[qc.fmt])
        story_ok &= (not v) == ("pass" in expect.split(":")[0])
        lines.append(f"| {name} | {met.top1_agreement:.4f} | {met.kl_mean:.2e} | {met.ppl_delta_pct:+.2f}% | "
                     f"{'✅' if not v else '❌ ' + '; '.join(v)} | {expect} |")

    loc = localize(m, ctx, targets, QuantConfig("int8"))
    lines += ["", "## Which layer broke under INT8? (one layer quantized at a time)", "",
              "| Layer | KL | Share |", "|---|---|---|"]
    lines += [f"| {r['layer']}{' (planted outliers)' if r['layer'] == OUTLIER_LAYER else ''} | {r['kl']:.2e} | {r['share']:.0%} |" for r in loc]
    worst = max(loc, key=lambda r: r["share"])
    story_ok &= worst["layer"] == OUTLIER_LAYER
    text = "\n".join(lines)
    print(text)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.md").write_text(text, encoding=ENCODING)
    print("\nreport:", "every expectation holds" if story_ok else "EXPECTATION BROKEN", file=sys.stderr)
    return 0 if story_ok else 1


def cmd_bench(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((args.batch, args.vocab)).astype(np.float32) * 3
    params = SamplingParams(temperature=0.8, top_p=0.9, top_k=50)
    seeds = list(range(args.batch))
    sample(logits, params, seeds)                        # warm-up
    t0 = time.perf_counter()
    for step in range(args.steps):
        sample(logits, params, seeds, step=step)
    dt = time.perf_counter() - t0
    print(f"batch {args.batch} × vocab {args.vocab}: {args.batch * args.steps / dt:,.0f} tokens/s "
          f"({dt / args.steps * 1000:.2f} ms/step, numpy reference implementation; not a kernel benchmark)")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="spv")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit")
    a.add_argument("--bug", action="append", choices=KNOWN_BUGS)
    a.set_defaults(fn=cmd_audit)
    pr = sub.add_parser("precision")
    pr.add_argument("--fmt", required=True, choices=sorted(load_gates().keys() - {"_comment"}))
    pr.add_argument("--keep-high", type=int, action="append")
    pr.add_argument("--smooth", type=float, default=0.0)
    pr.add_argument("--act-scale", choices=("tensor", "token"), default="tensor")
    pr.set_defaults(fn=cmd_precision)
    r = sub.add_parser("report")
    r.add_argument("--out")
    r.set_defaults(fn=cmd_report)
    b = sub.add_parser("bench")
    b.add_argument("--batch", type=int, default=256)
    b.add_argument("--vocab", type=int, default=32000)
    b.add_argument("--steps", type=int, default=20)
    b.set_defaults(fn=cmd_bench)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
