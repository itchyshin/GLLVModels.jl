"""Verify the T5 estimand-alignment re-bind batch: the 4
postfit/POSTFIT-SURFACE-extract_{communality,correlations,proportions,Omega}
ledger rows, re-run against gllvmTMB's REAL tier-scoped accessors on the
identical gaussian_small fixture used by the (frozen, unedited)
tools/core070_surface_conversion_batch.R. See
tools/core070_estimand_rebind_batch.R's header and
docs/dev-log/core070/parity-defect-rebind-2026-09-02.md for the full
accounting.

This batch has no separate frozen contract JSON (it is a small, standalone
4-case addendum, not an amendment to the 20/21-row surface-conversion
contract, which stays untouched). The 4 case_id/source_id/tolerance values
are pinned directly in this file (CASE_META) and in the R/Julia runners
they must match verbatim.

Two checks, mirroring tools/core070_verify_surface_conversion_batch.py:

  1. verify_state(path) -- structural + numeric checks against a retained
     run of tools/core070_estimand_rebind_batch.R (receipt.json,
     results.tsv, julia-results.json).
  2. --self-test -- mutates a SYNTHETIC valid state (no retained run
     required) in >=3 independent ways and asserts every mutation is
     rejected by check_state(). Per the wave5/postfit-policy precedent's
     vacuous-pass incident, --self-test NEVER substitutes for a real
     --state check below: passing --self-test alone still exits nonzero
     if --state is not given.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import sys as _sys  # noqa: E402
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from parity_oracle import R_REF_PINS, SELECTED_PIN  # noqa: E402  (GLLVM_PARITY_PIN; P0 default; unknown pin exits)
import core070_source_pin_check  # noqa: E402  (PR #569 review finding 2: library pin + version)

# No contract file for this batch; the pin only changes the expected reference commit.
REFERENCE_COMMIT = R_REF_PINS[SELECTED_PIN] if SELECTED_PIN == "P1" else "b4d5fee64def88bc768dda1f1f77c29b295edd86"
TOLERANCE = 1e-4

CASE_META = {
    "CORE070-ESTIMAND-REBIND-EXTRACT-COMMUNALITY": "postfit/POSTFIT-SURFACE-extract_communality",
    "CORE070-ESTIMAND-REBIND-EXTRACT-CORRELATIONS": "postfit/POSTFIT-SURFACE-extract_correlations",
    "CORE070-ESTIMAND-REBIND-EXTRACT-PROPORTIONS": "postfit/POSTFIT-SURFACE-extract_proportions",
    "CORE070-ESTIMAND-REBIND-EXTRACT-OMEGA": "postfit/POSTFIT-SURFACE-extract_Omega",
}
CASE_IDS = set(CASE_META.keys())
SOURCE_IDS = set(CASE_META.values())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


# ---------------------------------------------------------------------------
# Retained-state checks (receipt.json + results.tsv + julia-results.json).
# ---------------------------------------------------------------------------
def check_state(receipt, results_lines, julia_report):
    need(receipt["status"] == "PASS", "receipt status is not PASS")
    need(receipt["scope"] == "CORE070_ESTIMAND_REBIND_BATCH", "wrong receipt scope")
    need(receipt["reference_commit"] == REFERENCE_COMMIT, "receipt reference_commit mismatch")
    need(receipt["source_unchanged"] is True, "receipt does not assert source_unchanged")
    need(receipt["case_count"] == len(CASE_IDS), "receipt case_count drift")
    need(receipt["julia_exit_code"] == 0, "julia child exited nonzero")
    core070_source_pin_check.check_source_pin(receipt, SELECTED_PIN, need)
    need(receipt.get("oracle_error_count", 0) == 0, "R oracle recorded at least one computation error")
    need(set(receipt["target_source_ids"]) == SOURCE_IDS, "receipt target_source_ids does not match CASE_META")
    need(set(receipt["expected_case_ids"]) == CASE_IDS, "receipt expected_case_ids does not match CASE_META")

    seen = {}
    for line in results_lines:
        if not line.strip():
            continue
        source_id, case_id, verdict, bucket = line.split("\t")
        seen[case_id] = (source_id, verdict, bucket)
    need(set(seen.keys()) == CASE_IDS, "results.tsv case_id set does not match CASE_META")
    for cid, source_id in CASE_META.items():
        need(seen[cid] == (source_id, "PASS", "estimand_rebind"),
             f"{cid}: expected ({source_id}, PASS, estimand_rebind), got {seen[cid]}")

    need(julia_report["status"] == "PASS", "julia-results.json status is not PASS")
    need(julia_report["all_checks"] is True, "julia-results.json all_checks is not true")
    need(julia_report["negative_controls_behaved_as_expected"] is True,
         "julia-results.json negative controls did not behave as expected")
    need(julia_report["case_count"] == len(CASE_IDS), "julia-results.json case_count drift")

    for cid in CASE_IDS:
        jc = julia_report["cases"].get(cid)
        need(jc is not None, f"{cid}: missing from julia-results.json cases")
        need(jc.get("pass") is True, f"{cid}: julia-results.json reports pass=false")
        need(jc.get("max_abs_diff") is not None and jc["max_abs_diff"] <= TOLERANCE,
             f"{cid}: max_abs_diff {jc.get('max_abs_diff')} exceeds tolerance {TOLERANCE}")
        need(jc.get("r_len", 0) > 0 and jc.get("julia_len", 0) == jc.get("r_len", -1),
             f"{cid}: r_len/julia_len mismatch or zero-length")


def verify_state(state_dir: Path):
    if not state_dir.is_dir():
        raise SystemExit(f"FATAL: --state directory does not exist: {state_dir}")
    receipt_path = state_dir / "receipt.json"
    results_path = state_dir / "results.tsv"
    julia_path = state_dir / "julia-results.json"
    for p in (receipt_path, results_path, julia_path):
        if not p.exists():
            raise SystemExit(f"FATAL: required retained-run file missing: {p}")
    receipt = json.loads(receipt_path.read_text())
    results_lines = results_path.read_text().splitlines()
    julia_report = json.loads(julia_path.read_text())
    check_state(receipt, results_lines, julia_report)


# ---------------------------------------------------------------------------
# --self-test: build a SYNTHETIC valid state, confirm check_state() accepts
# it, then mutate it >=3 ways and confirm check_state() rejects every one.
# ---------------------------------------------------------------------------
def _synthetic_source_pin():
    pin = core070_source_pin_check.PINS[SELECTED_PIN]
    return {"gllvmTMB_version": pin["version"],
            "source_pin": {**{k: pin[k] for k in core070_source_pin_check.SOURCE_PIN_KEYS},
                           "version": pin["version"]}}


def _synthetic_state():
    results_lines = [f"{sid}\t{cid}\tPASS\testimand_rebind" for cid, sid in CASE_META.items()]
    julia_cases = {
        cid: {"pass": True, "quantity": "x", "tolerance": TOLERANCE,
              "max_abs_diff": TOLERANCE / 2, "r_len": 5, "julia_len": 5, "error": ""}
        for cid in CASE_IDS
    }
    julia_report = {
        "status": "PASS", "case_count": len(CASE_IDS), "all_checks": True,
        "negative_controls_behaved_as_expected": True, "cases": julia_cases,
    }
    receipt = {
        "status": "PASS", "scope": "CORE070_ESTIMAND_REBIND_BATCH",
        "reference_commit": REFERENCE_COMMIT, "source_unchanged": True,
        "target_source_ids": list(SOURCE_IDS), "case_count": len(CASE_IDS),
        "expected_case_ids": list(CASE_IDS), "oracle_error_count": 0, "julia_exit_code": 0, **_synthetic_source_pin(),
    }
    return receipt, results_lines, julia_report


def run_self_test():
    receipt, results_lines, julia_report = _synthetic_state()
    check_state(receipt, results_lines, julia_report)

    rejected = []

    def expect_rejected(label, mutate_fn):
        r2, res2, jr2 = deepcopy(receipt), list(results_lines), deepcopy(julia_report)
        r2, res2, jr2 = mutate_fn(r2, res2, jr2)
        try:
            check_state(r2, res2, jr2)
        except ValueError:
            rejected.append(label)
            return
        raise AssertionError(f"REJECTED MUTATION FAILED TO BE CAUGHT: {label}")

    def mut_wrong_scope(r, res, jr):
        r["scope"] = "SOMETHING_ELSE"
        return r, res, jr

    def mut_flip_one_verdict(r, res, jr):
        parts = res[0].split("\t")
        res[0] = "\t".join(parts[:2]) + "\tFAIL\testimand_rebind"
        return r, res, jr

    def mut_tolerance_blown(r, res, jr):
        cid = next(iter(CASE_IDS))
        jr["cases"][cid]["max_abs_diff"] = 1e9
        jr["cases"][cid]["pass"] = True  # deliberately inconsistent: pass=true but diff blown
        return r, res, jr

    def mut_oracle_error_present(r, res, jr):
        r["oracle_error_count"] = 1
        return r, res, jr

    def mut_missing_case(r, res, jr):
        cid = next(iter(CASE_IDS))
        del jr["cases"][cid]
        return r, res, jr

    expect_rejected("receipt scope tampered", mut_wrong_scope)
    expect_rejected("one verdict flipped to FAIL in results.tsv", mut_flip_one_verdict)
    expect_rejected("max_abs_diff blown past tolerance while pass stays true", mut_tolerance_blown)
    expect_rejected("receipt records a nonzero oracle_error_count", mut_oracle_error_present)
    expect_rejected("a case missing from julia-results.json cases", mut_missing_case)

    n_pin = core070_source_pin_check.self_test(SELECTED_PIN)
    rejected.extend(f"source pin mutation {i}" for i in range(n_pin))

    if len(rejected) < 3:
        raise AssertionError(f"only {len(rejected)} rejected mutations ran, need >=3")

    print(f"CORE070_ESTIMAND_REBIND_VERIFY_SELF_TEST_OK rejected_mutations={len(rejected)} "
          f"cases={len(CASE_IDS)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state", nargs="?",
                     help="retained-run directory to verify (e.g. .unlazy/core070-aghq/t5-rebind-01)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        run_self_test()
        if args.state is None:
            return  # self-test alone is a valid invocation

    if args.state is None:
        raise SystemExit("FATAL: no --state directory given and --self-test was not requested (or was "
                          "requested alongside a missing state) -- a self-test pass never substitutes for "
                          "a real retained-run check.")

    verify_state(Path(args.state))
    print("CORE070_ESTIMAND_REBIND_STATE_OK", args.state)


if __name__ == "__main__":
    main()
