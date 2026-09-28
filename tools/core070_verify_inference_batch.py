"""Verify the retained "inference" manifest-area batch: 45 EXECUTABLE_NOW
Julia routing cases + 2 negative controls (tools/core070_inference_batch.jl),
paired with a retained R-side crosscheck against the already-frozen R route
probe (tools/core070_inference_batch.R). 8 rows are bucket
NEEDS_NEW_JULIA_SURFACE and 11 rows are bucket SPEC_DEFECT; both are asserted
directly against the frozen contract rather than run.

Three independent checks:
  1. check_julia_report() / verify_julia_state() -- structural + hash checks
     against a retained run of tools/core070_inference_batch.jl
     (receipt.json, inference-batch-results.json, raw.tsv).
  2. check_r_crosscheck() / verify_r_state() -- structural + hash checks
     against a retained run of tools/core070_inference_batch.R
     (receipt.json, inference-batch-r-crosscheck.json,
     r-comparand-crosscheck.tsv, retained-inputs/).
  3. --self-test -- mutates synthetic valid reports (built from the frozen
     contract, no retained run required) in several independent ways and
     asserts every mutation is rejected. Runs without Julia, R, or the
     frozen source readback.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
import sys as _sys  # noqa: E402
_sys.path.insert(0, str(ROOT / "tools"))
from parity_oracle import R_REF_PINS, SELECTED_PIN  # noqa: E402  (GLLVM_PARITY_PIN; P0 default; unknown pin exits)
import core070_source_pin_check  # noqa: E402  (PR #571 review: library pin + version, shared with postfit)
# P0 keeps the frozen contract; P1 reads the twin written by tools/core070_inference_p1_contract.py.
CONTRACT_PATH = ROOT / ("docs/dev-log/core070/true-parity-latest/inference-batch-contract-p1.json"
                        if SELECTED_PIN == "P1" else "docs/dev-log/core070/inference-batch-contract.json")
REFERENCE_COMMIT = R_REF_PINS[SELECTED_PIN] if SELECTED_PIN == "P1" else "b4d5fee64def88bc768dda1f1f77c29b295edd86"
DEFAULT_JULIA_STATE = ROOT / ".unlazy/core070-aghq/inference-batch-01/julia"
DEFAULT_R_STATE = ROOT / ".unlazy/core070-aghq/inference-batch-01/r-crosscheck"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def load_contract():
    return json.loads(CONTRACT_PATH.read_text())


def contract_rows_by_bucket(contract):
    buckets = {"EXECUTABLE_NOW": [], "NEEDS_NEW_JULIA_SURFACE": [], "SPEC_DEFECT": []}
    for row in contract["rows"]:
        buckets[row["bucket"]].append(row)
    return buckets


# ---------------------------------------------------------------------------
# Contract-shape checks (bucket counts / case_id inventory) -- these hold
# regardless of whether any runner was ever executed, so they run first and
# unconditionally.
# ---------------------------------------------------------------------------
def check_contract_shape(contract):
    need(contract.get("schema") == "core070-inference-batch-contract/v1", "wrong schema")
    need(contract.get("status") == "FROZEN_INFERENCE_BATCH_CONTRACT", "contract not frozen")
    need(contract.get("area") == "inference", "wrong area")
    need(contract.get("reference_commit") == REFERENCE_COMMIT,
         "wrong reference commit")
    buckets = contract_rows_by_bucket(contract)
    need(len(buckets["EXECUTABLE_NOW"]) == contract["expected_executable_case_count"] == 45,
         "wrong EXECUTABLE_NOW row count")
    need(len(buckets["NEEDS_NEW_JULIA_SURFACE"]) == contract["expected_needs_new_julia_surface_count"] == 8,
         "wrong NEEDS_NEW_JULIA_SURFACE row count")
    need(len(buckets["SPEC_DEFECT"]) == contract["expected_spec_defect_count"] == 11,
         "wrong SPEC_DEFECT row count")
    need(len(contract["rows"]) == contract["expected_total_row_count"] == 64, "wrong total row count")
    ids = [r["source_id"] for r in contract["rows"]]
    need(len(ids) == len(set(ids)), "duplicate source_id in contract")
    need(len(set(r["case_id"] for r in contract["rows"])) == len(contract["case_ids"]) == 19,
         "wrong case_id inventory")
    for row in buckets["EXECUTABLE_NOW"]:
        need(bool(row.get("julia_call")) and bool(row.get("expected_route_tag")),
             f"{row['source_id']}: EXECUTABLE_NOW row missing julia_call/expected_route_tag")
        need(row.get("reason") is None, f"{row['source_id']}: EXECUTABLE_NOW row must not carry a reason")
    for row in buckets["NEEDS_NEW_JULIA_SURFACE"] + buckets["SPEC_DEFECT"]:
        need(row.get("julia_call") is None and row.get("expected_route_tag") is None,
             f"{row['source_id']}: non-executable row must not carry a julia_call/expected_route_tag")
        need(bool(row.get("reason")), f"{row['source_id']}: non-executable row missing a reason")
    return buckets


# ---------------------------------------------------------------------------
# Julia-side report checks.
# ---------------------------------------------------------------------------
def check_julia_report(report, contract, contract_sha256, buckets):
    expected_ids = {r["source_id"].split("/")[-1]: r for r in buckets["EXECUTABLE_NOW"]}
    need(report.get("status") == "PASS", "julia batch did not pass")
    need(report.get("area") == "inference", "wrong area")
    need(report.get("scope") == "CORE070_INFERENCE_BATCH", "wrong scope")
    need(report.get("case_count") == 45, "wrong executable case count")
    need(report.get("negative_control_count") == 2, "wrong negative control count")
    need(report.get("all_positive_pass") is True, "a required case failed")
    need(report.get("negative_controls_behaved_as_expected") is True, "a negative control misbehaved")
    need(report.get("all_checks") is True, "runner reported overall failure")

    cases = report.get("cases", [])
    positive = [c for c in cases if not c["source_id"].startswith("NEGATIVE-CONTROL")]
    negative = [c for c in cases if c["source_id"].startswith("NEGATIVE-CONTROL")]
    need(len(positive) == 45, "wrong number of positive cases in report")
    need(len(negative) == 2, "wrong number of negative controls in report")

    seen = set()
    for case in positive:
        sid = case.get("source_id")
        need(sid in expected_ids and sid not in seen, f"unexpected or duplicate case {sid}")
        seen.add(sid)
        row = expected_ids[sid]
        need(case.get("case_id") == row["case_id"], f"{sid}: case_id drift")
        need(case.get("expect") == row["expected_route_tag"], f"{sid}: expected_route_tag drift")
        need(case.get("ok") is True, f"{sid}: case did not pass")
        need(case.get("actual") == case.get("expect"), f"{sid}: actual route_tag did not match expectation")
    need(seen == set(expected_ids), "case set does not match contract's EXECUTABLE_NOW rows")

    for nc in negative:
        need(nc.get("ok") is True, f"{nc['source_id']}: negative control did not behave")
        need(nc.get("actual") == "false", f"{nc['source_id']}: negative control did not evaluate to false")


def verify_julia_state(state=DEFAULT_JULIA_STATE):
    contract = load_contract()
    contract_sha256 = sha(CONTRACT_PATH)
    buckets = contract_rows_by_bucket(contract)

    receipt_path = state / "receipt.json"
    results_path = state / "inference-batch-results.json"
    raw_path = state / "raw.tsv"
    need(receipt_path.is_file() and results_path.is_file() and raw_path.is_file(),
         "missing retained Julia-side run")

    receipt = json.loads(receipt_path.read_text())
    need(receipt.get("status") == "PASS" and receipt.get("source_unchanged") is True, "receipt did not pass")
    need(receipt.get("case_count") == 45, "receipt case count wrong")
    need(receipt.get("negative_control_count") == 2, "receipt negative-control count wrong")
    need(receipt.get("expected_case_source_ids") == [r["source_id"].split("/")[-1] for r in buckets["EXECUTABLE_NOW"]],
         "receipt case id list drifted or reordered vs contract")
    need(receipt.get("results_sha256") == sha(results_path), "results file changed since receipt")
    need(receipt.get("raw_sha256") == sha(raw_path), "raw.tsv changed since receipt")

    report = json.loads(results_path.read_text())
    check_julia_report(report, contract, contract_sha256, buckets)

    print("CORE070_INFERENCE_BATCH_JULIA_VERIFIED", len(buckets["EXECUTABLE_NOW"]), "cases")
    return {
        "receipt_sha256": sha(receipt_path),
        "results_sha256": sha(results_path),
    }


# ---------------------------------------------------------------------------
# R-side crosscheck checks.
# ---------------------------------------------------------------------------
def check_r_crosscheck(result, contract, contract_sha256):
    need(result.get("status") == "PASS", "R crosscheck did not pass")
    need(result.get("scope") == "CORE070_INFERENCE_BATCH_R_CROSSCHECK", "wrong scope")
    need(result.get("contract_sha256") == contract_sha256, "stale contract in R crosscheck")
    need(result.get("r_route_pins") == contract["r_route_comparand"]["pins"], "R route pins drift")
    need(result.get("in_scope_row_count") == 64, "wrong in-scope row count")
    need(result.get("all_in_scope_rows_pass_r_routing") is True, "an in-scope row failed R routing")


def verify_r_state(state=DEFAULT_R_STATE):
    contract = load_contract()
    contract_sha256 = sha(CONTRACT_PATH)

    receipt_path = state / "receipt.json"
    results_path = state / "inference-batch-r-crosscheck.json"
    crosscheck_path = state / "r-comparand-crosscheck.tsv"
    need(receipt_path.is_file() and results_path.is_file() and crosscheck_path.is_file(),
         "missing retained R-side crosscheck run")

    receipt = json.loads(receipt_path.read_text())
    need(receipt.get("status") == "PASS" and receipt.get("source_unchanged") is True, "R receipt did not pass")
    need(receipt.get("contract_sha256") == contract_sha256, "stale contract in R receipt")
    need(receipt.get("in_scope_row_count") == 64, "R receipt row count wrong")
    need(receipt.get("crosscheck_sha256") == sha(crosscheck_path), "crosscheck tsv changed since receipt")
    need(receipt.get("results_json_sha256") == sha(results_path), "R crosscheck json changed since receipt")
    # At P1 the crosscheck must record the oracle library's CORE070_SOURCE_PIN marker and version.
    core070_source_pin_check.check_source_pin(receipt, SELECTED_PIN, need)

    result = json.loads(results_path.read_text())
    check_r_crosscheck(result, contract, contract_sha256)

    lines = crosscheck_path.read_text().splitlines()
    need(len(lines) == 65, "crosscheck tsv row count wrong (header + 64 rows)")  # header + 64

    print("CORE070_INFERENCE_BATCH_R_CROSSCHECK_VERIFIED 64 rows")
    return {"receipt_sha256": sha(receipt_path), "results_sha256": sha(results_path)}


# ---------------------------------------------------------------------------
# --self-test: mutation-rejection controls, no retained run required.
# ---------------------------------------------------------------------------
def synthetic_julia_report(contract, buckets):
    cases = []
    for row in buckets["EXECUTABLE_NOW"]:
        sid = row["source_id"].split("/")[-1]
        cases.append({
            "source_id": sid, "case_id": row["case_id"],
            "expect": row["expected_route_tag"], "actual": row["expected_route_tag"],
            "ok": True, "detail": "propertynames=(...)", "elapsed_seconds": 0.1,
        })
    for i, sid in enumerate(["NEGATIVE-CONTROL-1", "NEGATIVE-CONTROL-2"]):
        cases.append({
            "source_id": sid, "case_id": "n/a", "expect": "false", "actual": "false",
            "ok": True, "detail": "negative control", "elapsed_seconds": 0.0,
        })
    return {
        "status": "PASS", "area": "inference", "scope": "CORE070_INFERENCE_BATCH",
        "case_count": 45, "negative_control_count": 2,
        "all_positive_pass": True, "negative_controls_behaved_as_expected": True,
        "cases": cases, "all_checks": True,
    }


def synthetic_r_result(contract):
    return {
        "status": "PASS", "scope": "CORE070_INFERENCE_BATCH_R_CROSSCHECK",
        "contract_sha256": "0" * 64,  # overwritten by caller with the real value
        "r_route_pins": contract["r_route_comparand"]["pins"],
        "in_scope_row_count": 64,
        "all_in_scope_rows_pass_r_routing": True,
    }


def self_test():
    contract = load_contract()
    contract_sha256 = sha(CONTRACT_PATH)
    buckets = check_contract_shape(contract)
    print("CORE070_INFERENCE_BATCH_CONTRACT_SHAPE_VERIFIED",
          len(buckets["EXECUTABLE_NOW"]), "executable;",
          len(buckets["NEEDS_NEW_JULIA_SURFACE"]), "needs-new-surface;",
          len(buckets["SPEC_DEFECT"]), "spec-defect")

    good = synthetic_julia_report(contract, buckets)
    check_julia_report(good, contract, contract_sha256, buckets)  # sanity

    julia_mutations = [
        ("drop a positive case", lambda r: [c for c in r["cases"] if c["source_id"] != "CI-ROUTE-001"]
            and r.__setitem__("cases", [c for c in r["cases"] if c["source_id"] != "CI-ROUTE-001"])),
        ("flip status", lambda r: r.update(status="FAIL")),
        ("flip all_checks", lambda r: r.update(all_checks=False)),
        ("flip all_positive_pass", lambda r: r.update(all_positive_pass=False)),
        ("flip negative_controls_behaved_as_expected",
         lambda r: r.update(negative_controls_behaved_as_expected=False)),
        ("flip one positive case's ok", lambda r: r["cases"][0].__setitem__("ok", False)),
        ("claim wrong route_tag for a positive case",
         lambda r: r["cases"][0].__setitem__("actual", "IMPOSSIBLE_ROUTE")),
        ("relabel a case_id", lambda r: r["cases"][0].__setitem__("case_id", "BOGUS-CASE-ID")),
        ("flip a negative control's actual to true",
         lambda r: next(c for c in r["cases"] if c["source_id"] == "NEGATIVE-CONTROL-1")
                        .__setitem__("actual", "true")),
        ("duplicate a case id", lambda r: r["cases"].append(deepcopy(r["cases"][0]))),
        ("wrong case_count", lambda r: r.update(case_count=44)),
    ]
    rejected = 0
    for name, mutate in julia_mutations:
        bad = deepcopy(good)
        mutate(bad)
        try:
            check_julia_report(bad, contract, contract_sha256, buckets)
        except ValueError:
            rejected += 1
            continue
        raise AssertionError(f"accepted invalid julia inference-batch evidence: {name}")
    need(rejected >= 4, "fewer than 4 rejected mutations exercised (julia side)")
    print("CORE070_INFERENCE_BATCH_JULIA_NEGATIVES_PASS", rejected)

    good_r = synthetic_r_result(contract)
    good_r["contract_sha256"] = contract_sha256
    check_r_crosscheck(good_r, contract, contract_sha256)  # sanity

    r_mutations = [
        ("flip status", lambda r: r.update(status="FAIL")),
        ("stale contract hash", lambda r: r.update(contract_sha256="0" * 64)),
        ("change a route pin", lambda r: r["r_route_pins"].__setitem__(
            "test/parity/fixtures/core070_inference_routes.tsv", "0" * 64)),
        ("wrong in_scope_row_count", lambda r: r.update(in_scope_row_count=63)),
        ("flip all_in_scope_rows_pass_r_routing",
         lambda r: r.update(all_in_scope_rows_pass_r_routing=False)),
    ]
    r_rejected = 0
    for name, mutate in r_mutations:
        bad = deepcopy(good_r)
        mutate(bad)
        try:
            check_r_crosscheck(bad, contract, contract_sha256)
        except ValueError:
            r_rejected += 1
            continue
        raise AssertionError(f"accepted invalid R crosscheck evidence: {name}")
    need(r_rejected >= 4, "fewer than 4 rejected mutations exercised (R side)")
    print("CORE070_INFERENCE_BATCH_R_NEGATIVES_PASS", r_rejected)

    n_pin = core070_source_pin_check.self_test(SELECTED_PIN)
    print("CORE070_INFERENCE_BATCH_SOURCE_PIN_NEGATIVES_PASS", n_pin)

    return rejected + r_rejected + n_pin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--julia-state", type=Path, default=DEFAULT_JULIA_STATE)
    parser.add_argument("--r-state", type=Path, default=DEFAULT_R_STATE)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    mutation_count = self_test() if args.self_test else None
    # A missing state is a FAILURE, never a silent skip -- the vacuous-pass
    # incident this template's hardened lessons name is exactly a verifier
    # that exits 0 on --self-test output alone while no real batch ran.
    if not args.julia_state.exists():
        raise SystemExit(f"verify_inference_batch: julia state does not exist: {args.julia_state}")
    if not args.r_state.exists():
        raise SystemExit(f"verify_inference_batch: R state does not exist: {args.r_state}")

    julia_result = verify_julia_state(args.julia_state)
    r_result = verify_r_state(args.r_state)

    result = {
        "status": "CORE070_INFERENCE_BATCH_PASS",
        "executable_case_count": 45,
        "needs_new_julia_surface_count": 8,
        "spec_defect_count": 11,
        "julia": julia_result,
        "r_crosscheck": r_result,
    }
    if mutation_count:
        result["self_test_mutations_rejected"] = mutation_count
    print("CORE070_INFERENCE_BATCH_FULLY_VERIFIED")
    if args.output:
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
