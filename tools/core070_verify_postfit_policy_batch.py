"""Verify the retained "postfit-policy" manifest-area batch (15 executable
cases + 2 negative controls; 8 needs-new-Julia-surface deferrals + 1
zero-case spec-defect row are recorded in the contract but never executed).

Two independent checks, mirroring tools/core070_verify_data_batch.py and
tools/core070_verify_masks_known.py:

  1. verify_contract() -- structural checks on the frozen contract itself
     (bucket counts sum to the manifest's 24 rows, no case id appears in more
     than one bucket, source pins are well-formed). Needs no retained run.
  2. verify_state() / check_report() -- structural + hash checks against a
     retained run of tools/core070_postfit_policy_batch.R (receipt.json,
     julia-results.json, results.tsv, diagnostics.log).
  3. --self-test -- mutates a *synthetic* valid report (built from the frozen
     contract, no retained run required) in several independent ways and
     asserts every mutation is rejected by check_report(). Runs without R,
     Julia, or the frozen library. Per the 2026-09-01 vacuous-pass incident,
     --self-test NEVER substitutes for the real --state check below: passing
     --self-test alone still exits nonzero if --state does not exist.
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
CONTRACT_PATH = ROOT / ("docs/dev-log/core070/true-parity-latest/postfit-policy-batch-contract-p1.json"
                        if SELECTED_PIN == "P1" else "docs/dev-log/core070/postfit-policy-batch-contract.json")
REFERENCE_COMMIT = R_REF_PINS[SELECTED_PIN] if SELECTED_PIN == "P1" else "b4d5fee64def88bc768dda1f1f77c29b295edd86"
DEFAULT_STATE = ROOT / ".unlazy/core070-aghq/postfit-policy-batch-01"

MANIFEST_ROW_COUNT = 24


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def load_contract():
    return json.loads(CONTRACT_PATH.read_text())


# ---------------------------------------------------------------------------
# 1. Contract-only structural checks (no retained run required).
# ---------------------------------------------------------------------------
def verify_contract(contract=None):
    c = contract or load_contract()
    need(c["status"] == "FROZEN_POSTFIT_POLICY_BATCH_CONTRACT", "wrong contract status")
    need(c["area"] == "postfit-policy", "wrong area")
    need(c["reference_commit"] == REFERENCE_COMMIT, "wrong reference commit")
    need(c["manifest_row_count"] == MANIFEST_ROW_COUNT, "manifest row count drift")

    executable_ids = {x["case_id"] for x in c["cases"]}
    needs_surface_ids = {x["source_id"] for x in c["needs_new_julia_surface"]}
    spec_defect_ids = {x["source_id"] for x in c["spec_defect_notes"]}

    need(len(executable_ids) == len(c["cases"]) == c["expected_case_count"] == 15,
         "executable case id set has duplicates or wrong count")
    need(len(needs_surface_ids) == len(c["needs_new_julia_surface"]) ==
         c["needs_new_julia_surface_count"] == 8,
         "needs-new-surface id set has duplicates or wrong count")
    need(len(spec_defect_ids) == len(c["spec_defect_notes"]) == c["spec_defect_note_count"] == 1,
         "spec-defect note set has duplicates or wrong count")

    executable_source_ids = {x["source_id"] for x in c["cases"]}
    need(len(executable_source_ids & needs_surface_ids) == 0,
         "a source_id appears in both the executable and needs-new-surface buckets")
    need(len(executable_source_ids & spec_defect_ids) == 0,
         "a source_id appears in both the executable and spec-defect buckets")
    need(len(needs_surface_ids & spec_defect_ids) == 0,
         "a source_id appears in both the needs-new-surface and spec-defect buckets")

    total_rows = len(executable_source_ids) + len(needs_surface_ids) + len(spec_defect_ids)
    need(total_rows == MANIFEST_ROW_COUNT,
         f"bucket accounting does not sum to the manifest's {MANIFEST_ROW_COUNT} rows (got {total_rows})")

    for x in c["cases"]:
        need(x["case_id"].startswith("CORE070-POSTFIT-") and x["case_id"].endswith("-NATIVE"),
             f"malformed case_id {x['case_id']}")
        need(x["source_id"].startswith("postfit-policy/POST-"), f"malformed source_id on {x['case_id']}")
        for key in ("julia_surface", "r_call", "comparand", "check"):
            need(isinstance(x.get(key), str) and len(x[key]) > 0, f"{x['case_id']} missing {key}")

    for x in c["needs_new_julia_surface"]:
        for key in ("missing_surface", "reason"):
            need(isinstance(x.get(key), str) and len(x[key]) > 0, f"{x['source_id']} missing {key}")

    need(len(c["negative_controls"]) >= 2, "contract itself must name >=2 negative controls")
    for nc in c["negative_controls"]:
        for key in ("control_id", "description", "check"):
            need(isinstance(nc.get(key), str) and len(nc[key]) > 0, f"{nc.get('control_id')} missing {key}")

    need(isinstance(c["source_pins"], dict) and len(c["source_pins"]) == 4, "wrong source_pins count")
    for path_str, digest in c["source_pins"].items():
        need(len(digest) == 64 and all(ch in "0123456789abcdef" for ch in digest),
             f"malformed sha256 for {path_str}")

    print("CORE070_POSTFIT_POLICY_CONTRACT_VERIFIED",
          "executable=", len(executable_ids), "needs_surface=", len(needs_surface_ids),
          "spec_defect=", len(spec_defect_ids), "total=", total_rows)
    return c


# ---------------------------------------------------------------------------
# 2. Retained-run checks.
# ---------------------------------------------------------------------------
def check_report(report, contract, contract_sha256):
    contract_cases = {c["case_id"]: c for c in contract["cases"]}
    contract_neg = {c["control_id"]: c for c in contract["negative_controls"]}

    need(report.get("status") == "PASS", "batch did not pass")
    need(report.get("area") == "postfit-policy", "wrong area")
    need(report.get("scope") == "CORE070_POSTFIT_POLICY_BATCH", "wrong scope")
    need(report.get("case_count") == len(contract_cases), "wrong case count")
    need(report.get("negative_control_count") == len(contract_neg), "wrong negative-control count")
    need(report.get("all_positive_pass") is True, "a required case failed")
    need(report.get("negative_controls_behaved_as_expected") is True, "a negative control misbehaved")
    need(report.get("all_checks") is True, "runner reported an overall failure")

    cases = report.get("cases", {})
    need(set(cases) == set(contract_cases), "case id set does not match contract")
    for cid, case in cases.items():
        need(case.get("pass") is True, f"{cid} did not pass")

    negatives = report.get("negative_controls", {})
    need(set(negatives) == set(contract_neg), "negative-control id set does not match contract")
    for cid, nc in negatives.items():
        need(nc.get("behaved") is True, f"negative control {cid} did not behave as required")


def verify_state(state=DEFAULT_STATE):
    contract = load_contract()
    verify_contract(contract)
    contract_sha256 = sha(CONTRACT_PATH)

    receipt_path = state / "receipt.json"
    julia_path = state / "julia-results.json"
    raw_path = state / "results.tsv"
    diag_path = state / "diagnostics.log"
    need(receipt_path.is_file() and julia_path.is_file() and raw_path.is_file() and diag_path.is_file(),
         "missing retained R-side run")

    receipt = json.loads(receipt_path.read_text())
    need(receipt.get("status") == "PASS" and receipt.get("source_unchanged") is True,
         "receipt did not pass")
    need(receipt.get("reference_commit") == contract["reference_commit"], "wrong reference commit")
    need(receipt.get("contract_sha256") == contract_sha256, "stale contract in receipt")
    need(receipt.get("source_pins") == contract["source_pins"], "receipt source pins changed")
    need(receipt.get("case_count") == contract["expected_case_count"], "receipt case count wrong")
    need(receipt.get("negative_control_count") == len(contract["negative_controls"]),
         "receipt negative-control count wrong")
    need(sorted(receipt.get("expected_case_ids", [])) ==
         sorted(c["case_id"] for c in contract["cases"]), "receipt case id list drifted")
    need(sorted(receipt.get("negative_control_case_ids", [])) ==
         sorted(c["control_id"] for c in contract["negative_controls"]),
         "receipt negative-control id list drifted")
    need(receipt.get("julia_exit_code") == 0, "julia child exited nonzero")
    need(receipt.get("julia_results_sha256") == sha(julia_path), "julia-results.json changed since receipt")
    need(receipt.get("raw_sha256") == sha(raw_path), "results.tsv changed since receipt")
    need(receipt.get("diagnostics_sha256") == sha(diag_path), "diagnostics.log changed since receipt")

    report = json.loads(julia_path.read_text())
    check_report(report, contract, contract_sha256)

    diag_text = diag_path.read_text().strip()
    need(diag_text == "", "diagnostics.log is non-empty on an all-PASS run")

    print("CORE070_POSTFIT_POLICY_BATCH_VERIFIED", len(contract["cases"]),
          "negative_controls", len(contract["negative_controls"]))
    return {
        "status": "CORE070_POSTFIT_POLICY_BATCH_PASS",
        "case_count": contract["expected_case_count"],
        "negative_control_count": len(contract["negative_controls"]),
        "receipt_sha256": sha(receipt_path),
        "julia_results_sha256": sha(julia_path),
    }


# ---------------------------------------------------------------------------
# 3. Self-test: synthetic report + mutation battery (no retained run needed).
# ---------------------------------------------------------------------------
def synthetic_report(contract):
    cases = {
        c["case_id"]: {"pass": True}
        for c in contract["cases"]
    }
    negatives = {
        nc["control_id"]: {"behaved": True}
        for nc in contract["negative_controls"]
    }
    return {
        "status": "PASS",
        "area": "postfit-policy",
        "scope": "CORE070_POSTFIT_POLICY_BATCH",
        "case_count": len(cases),
        "negative_control_count": len(negatives),
        "all_positive_pass": True,
        "negative_controls_behaved_as_expected": True,
        "all_checks": True,
        "cases": cases,
        "negative_controls": negatives,
    }


def self_test():
    contract = load_contract()
    verify_contract(contract)  # sanity: the frozen contract itself is well-formed
    contract_sha256 = sha(CONTRACT_PATH)
    good = synthetic_report(contract)
    check_report(good, contract, contract_sha256)  # sanity: synthetic report is itself valid

    first_case_id = contract["cases"][0]["case_id"]
    first_neg_id = contract["negative_controls"][0]["control_id"]

    mutations = [
        ("drop a case", lambda r: r["cases"].pop(first_case_id)),
        ("flip status", lambda r: r.update(status="FAIL")),
        ("flip all_checks", lambda r: r.update(all_checks=False)),
        ("flip all_positive_pass", lambda r: r.update(all_positive_pass=False)),
        ("flip negative_controls_behaved_as_expected",
         lambda r: r.update(negative_controls_behaved_as_expected=False)),
        ("flip one case's pass", lambda r: r["cases"][first_case_id].__setitem__("pass", False)),
        ("flip one negative control's behaved",
         lambda r: r["negative_controls"][first_neg_id].__setitem__("behaved", False)),
        ("drop a negative control", lambda r: r["negative_controls"].pop(first_neg_id)),
        ("wrong case count", lambda r: r.update(case_count=99)),
        ("duplicate-shaped extra case",
         lambda r: r["cases"].__setitem__("CORE070-POSTFIT-BOGUS-NATIVE", {"pass": True})),
    ]
    for name, mutate in mutations:
        bad = deepcopy(good)
        mutate(bad)
        try:
            check_report(bad, contract, contract_sha256)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid postfit-policy-batch evidence: {name}")
    print("CORE070_POSTFIT_POLICY_BATCH_NEGATIVES_PASS", len(mutations))

    # Contract-level mutations: a tampered bucket accounting must also be caught.
    contract_mutations = [
        ("shrink executable cases without updating expected_case_count",
         lambda c: c["cases"].pop()),
        ("duplicate a source_id across buckets",
         lambda c: c["needs_new_julia_surface"].append(
             {**c["cases"][0], "source_id": c["cases"][0]["source_id"],
              "missing_surface": "x", "reason": "x"})),
        ("wrong manifest_row_count", lambda c: c.update(manifest_row_count=23)),
        ("only one negative control",
         lambda c: c.__setitem__("negative_controls", c["negative_controls"][:1])),
    ]
    for name, mutate in contract_mutations:
        bad = deepcopy(contract)
        mutate(bad)
        try:
            verify_contract(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid postfit-policy contract: {name}")
    print("CORE070_POSTFIT_POLICY_CONTRACT_NEGATIVES_PASS", len(contract_mutations))

    return len(mutations) + len(contract_mutations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    # --self-test never substitutes for the real --state check below --
    # there is deliberately no flag that skips it (the vacuous-pass
    # incident: verifier exit 0 with only self-test output while the real
    # batch had produced nothing).
    mutation_count = self_test() if args.self_test else None

    if not args.state.exists():
        raise SystemExit(f"verify_postfit_policy_batch: state does not exist: {args.state}")
    result = verify_state(args.state)
    if result and mutation_count:
        result["self_test_mutations_rejected"] = mutation_count
    if args.output and result:
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
