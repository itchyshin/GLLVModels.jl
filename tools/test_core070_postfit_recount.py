#!/usr/bin/env python3
"""Regression check for extending postfit receipt tiers."""

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from core070_postfit_p1_receipts import recount


def main():
    case_map = {
        "counts": {"numeric_pass": 0, "partial_non_numeric_case": 0},
        "rows": [
            {"evidence_tier": "numeric"},
            {"evidence_tier": "behavioural"},
            {"evidence_tier": "partial_non_numeric_case"},
        ],
    }
    expected = {
        "numeric_pass": 1,
        "partial_non_numeric_case": 1,
        "behavioural": 1,
    }
    actual = recount(case_map)
    if actual != expected:
        raise AssertionError(f"recount mismatch: {actual!r} != {expected!r}")
    print("POSTFIT_RECOUNT_BEHAVIOURAL_PASS")


if __name__ == "__main__":
    main()
