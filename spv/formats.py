"""Fake quantization: round float values onto the grid of a low-precision format.

The rounding is exact to the format's grid (round-to-nearest-even, subnormals,
saturation) but compute stays in float64. This is the standard way to study
accuracy before kernels exist ("fake quant" / QAT-style emulation). It says
nothing about speed.

Formats
  fp32        identity
  fp16        IEEE half (numpy native)
  bf16        8-bit exponent, 7-bit mantissa
  fp8_e4m3    OCP FP8 E4M3 (max 448, no inf): weights / activations
  fp8_e5m2    OCP FP8 E5M2 (max 57344): wider range, less precision
  int8        symmetric, scale = amax / 127
"""
from __future__ import annotations

import numpy as np

# name: (mantissa bits, min normal exponent, max finite value)
FLOAT_GRIDS = {
    "bf16": (7, -126, 3.3895313892515355e38),
    "fp8_e4m3": (3, -6, 448.0),
    "fp8_e5m2": (2, -14, 57344.0),
}
FORMATS = ("fp32", "fp16", "bf16", "fp8_e4m3", "fp8_e5m2", "int8")
SCALED = {"fp8_e4m3", "fp8_e5m2", "int8"}        # need a per-tensor / per-channel scale


def round_to_grid(x: np.ndarray, mantissa_bits: int, min_exp: int, max_val: float) -> np.ndarray:
    """Round to nearest representable value (ties to even), with subnormals and saturation."""
    x = np.asarray(x, dtype=np.float64)
    ax = np.abs(x)
    _, e = np.frexp(ax)                                   # ax = m * 2**e, m in [0.5, 1)
    e = np.maximum(e - 1, min_exp)                        # exponent of the leading bit, floored at min normal
    spacing = np.ldexp(1.0, (e - mantissa_bits).astype(np.int64))
    q = np.round(ax / spacing) * spacing                  # np.round is round-half-to-even
    return np.sign(x) * np.minimum(q, max_val)


def _amax(x: np.ndarray, axis: int | None) -> np.ndarray:
    a = np.abs(x).max(axis=axis, keepdims=axis is not None)
    return np.where(a == 0, 1.0, a)


def quantize(x: np.ndarray, fmt: str, axis: int | None = None) -> np.ndarray:
    """Fake-quantize x. `axis` is the reduction axis for the scale: None = per-tensor;
    for a (in, out) weight, axis=0 gives one scale per output channel; for (tokens, ch)
    activations, axis=1 gives one scale per token."""
    if fmt == "fp32":
        return np.asarray(x, dtype=np.float64)
    if fmt == "fp16":
        return np.asarray(x).astype(np.float16).astype(np.float64)
    if fmt == "bf16":
        return round_to_grid(x, *FLOAT_GRIDS["bf16"])
    if fmt == "int8":
        scale = _amax(x, axis) / 127.0
        return np.clip(np.round(x / scale), -127, 127) * scale
    if fmt in FLOAT_GRIDS:
        mbits, emin, vmax = FLOAT_GRIDS[fmt]
        scale = _amax(x, axis) / vmax
        return round_to_grid(x / scale, mbits, emin, vmax) * scale
    raise ValueError(f"unknown format {fmt!r}; expected one of {FORMATS}")
