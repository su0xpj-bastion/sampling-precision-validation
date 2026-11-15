"""Formats are bit-exact; the gate passes and fails the right configs; localization names the layer."""
from __future__ import annotations

import numpy as np
import pytest

from spv import cli
from spv.formats import FLOAT_GRIDS, quantize, round_to_grid
from spv.model import OUTLIER_LAYER, QuantConfig
from spv.precision import check, compare, load_gates, localize, setup


@pytest.fixture(scope="module")
def data():
    return setup()


class TestFormats:
    def test_bf16_is_bit_exact_round_to_nearest_even(self):
        rng = np.random.default_rng(0)
        x = (rng.standard_normal(50_000) * np.exp(rng.uniform(-20, 20, 50_000))).astype(np.float32)
        u = x.view(np.uint32).astype(np.uint64)
        want = ((u + ((u >> 16) & 1) + 0x7FFF) >> 16 << 16).astype(np.uint32).view(np.float32).astype(np.float64)
        np.testing.assert_array_equal(quantize(x, "bf16"), want)

    def test_fp16_grid_matches_ieee_half_in_range(self):
        y = np.random.default_rng(1).standard_normal(50_000) * 100
        np.testing.assert_array_equal(round_to_grid(y, 10, -14, 65504.0), y.astype(np.float16).astype(np.float64))

    def test_e4m3_has_127_nonnegative_values_and_saturates_at_448(self):
        grid = round_to_grid(np.linspace(0, 448, 2_000_000), *FLOAT_GRIDS["fp8_e4m3"])
        assert len(np.unique(grid)) == 127
        assert round_to_grid(np.array([1e6, -1e6]), *FLOAT_GRIDS["fp8_e4m3"]).tolist() == [448.0, -448.0]

    @pytest.mark.parametrize("x,want", [(1.0625, 1.0), (1.1875, 1.25), (2 ** -9, 2 ** -9), (2 ** -10, 0.0)])
    def test_e4m3_ties_to_even_and_subnormals(self, x, want):
        assert round_to_grid(np.array([x]), *FLOAT_GRIDS["fp8_e4m3"])[0] == want

    def test_int8_uses_at_most_255_levels(self):
        q = quantize(np.random.default_rng(2).standard_normal(10_000), "int8")
        assert len(np.unique(q)) <= 255

    def test_unknown_format_rejected(self):
        with pytest.raises(ValueError):
            quantize(np.ones(3), "fp4")


EXPECT = [
    ("fp16", QuantConfig("fp16"), True),
    ("bf16", QuantConfig("bf16"), True),
    ("fp8_e4m3", QuantConfig("fp8_e4m3"), True),
    ("fp8_e5m2", QuantConfig("fp8_e5m2"), False),
    ("int8", QuantConfig("int8"), False),
    ("int8 + keep outlier layer bf16", QuantConfig("int8", keep_high={OUTLIER_LAYER}), True),
    ("int8 + SmoothQuant", QuantConfig("int8", smooth_alpha=0.5), True),
]


@pytest.mark.parametrize("name,qc,passes", EXPECT, ids=[e[0] for e in EXPECT])
def test_gate_outcome(data, name, qc, passes):
    violations = check(compare(*data, qc), load_gates()[qc.fmt])
    assert (not violations) == passes, violations


def test_fp32_config_is_exactly_the_reference(data):
    m = compare(*data, QuantConfig("fp32"))
    assert m.kl_mean == 0.0 and m.top1_agreement == 1.0


def test_localization_names_the_outlier_layer(data):
    loc = localize(*data, QuantConfig("int8"))
    worst = max(loc, key=lambda r: r["share"])
    assert worst["layer"] == OUTLIER_LAYER and worst["share"] >= 0.9


def test_fp8_e4m3_is_not_dominated_by_the_outlier_layer(data):
    """The format finding: FP8's exponent absorbs activation outliers that break per-tensor INT8."""
    share = {r["layer"]: r["share"] for r in localize(*data, QuantConfig("fp8_e4m3"))}
    assert share[OUTLIER_LAYER] < 0.5


def test_report_holds_every_expectation(tmp_path):
    assert cli.main(["report", "--out", str(tmp_path)]) == 0
    text = (tmp_path / "report.md").read_text()
    row = next(line for line in text.splitlines() if "(planted outliers)" in line)
    assert int(row.rsplit("|", 2)[-2].strip().rstrip("%")) >= 90
