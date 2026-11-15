# sampling-precision-validation — agent guardrails

- `spv/reference.py` (golden) must never import from `spv/sampler.py` beyond `SamplingParams`.
- Seeded bugs in `KNOWN_BUGS` are intentional. A new bug class needs an entry in
  `EXPECTED_CATCH` (tests/test_sampler.py) proving the audit catches it.
- Statistical checks use fixed seeds. Never randomize seeds in CI.
- Gate thresholds live in `gates.json` and sit between observed pass/fail populations.
  If you change the model or formats, re-derive them and update the README table.
- Python: `from __future__ import annotations`, module-level constants, `Path.open(..., encoding=)` + `json.load`.
- Done means: `python -m pytest` green AND `python -m spv report` exits 0.
