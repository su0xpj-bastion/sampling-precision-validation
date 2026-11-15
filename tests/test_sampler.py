"""Sampler: the audit passes a correct build and catches every seeded defect.

The parametrized self-test is the important one: if a refactor of the audit stops
catching a known bug class, CI goes red. A test suite that silently loses the ability
to see a bug is worse than no test.
"""
from __future__ import annotations

import numpy as np
import pytest

from spv import reference
from spv.audit import audit
from spv.sampler import KNOWN_BUGS, SamplingParams, process, sample

EXPECTED_CATCH = {
    "topp_boundary": {"golden_equivalence", "distribution_fit", "edge_cases"},
    "rep_penalty_sign": {"golden_equivalence", "edge_cases"},
    "shared_rng": {"batch_invariance"},
    "temp_after_topp": {"golden_equivalence"},
}


def test_correct_build_passes_every_check():
    failed = [c for c in audit() if not c.passed]
    assert not failed, failed


@pytest.mark.parametrize("bug", KNOWN_BUGS)
def test_audit_catches_seeded_bug(bug):
    failed = {c.name for c in audit(frozenset([bug])) if not c.passed}
    assert failed == EXPECTED_CATCH[bug]


def test_every_known_bug_has_an_expectation():
    assert set(EXPECTED_CATCH) == set(KNOWN_BUGS)


def test_fast_path_matches_golden_on_random_fuzz():
    rng = np.random.default_rng(2024)
    for _ in range(200):
        v = int(rng.integers(2, 40))
        x = rng.standard_normal((1, v)) * rng.uniform(0.1, 5)
        params = SamplingParams(temperature=float(rng.choice([0.0, 0.3, 1.0, 1.7])),
                                top_k=int(rng.integers(0, v + 1)), top_p=float(rng.uniform(0.05, 1.0)),
                                min_p=float(rng.choice([0.0, 0.02, 0.2])))
        got = process(x, params)[0]
        want = np.array(reference.distribution(list(x[0]), params))
        np.testing.assert_allclose(got, want, atol=1e-9)


def test_tokens_stay_in_vocab_and_follow_support():
    x = np.full((1000, 8), -np.inf)
    x[:, 2] = 0.0
    x[:, 5] = 0.0
    toks = sample(x, SamplingParams(), list(range(1000)))
    assert set(toks.tolist()) == {2, 5}


@pytest.mark.parametrize("kwargs", [{"temperature": -1}, {"top_p": 0}, {"top_p": 1.5}, {"top_k": -1},
                                    {"min_p": 2}, {"repetition_penalty": 0}])
def test_invalid_params_rejected(kwargs):
    with pytest.raises(ValueError):
        SamplingParams(**kwargs)
