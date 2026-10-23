"""Verify the retained "aghq control" M2 batch (16 executable paired-control
cases + 3 negative controls; the 22 needs-new-Julia-surface deferrals from
docs/dev-log/core070/aghq-required-case-plan.json's AGHQ-AUTO-K-*/AGHQ-POLICY-*
rows are recorded in the contract but never executed).

Three independent checks, mirroring tools/core070_verify_namespace_2_batch.py:

  1. verify_contract() -- structural checks on the frozen contract itself
     (bucket counts sum to the 29-row required-source-case-map.json scope
     plus the 9 out-of-scope AGHQ-INVALID-* rows, no case id appears in more
     than one bucket, source pins are well-formed). Needs no retained run.
  2. verify_state() / check_report() -- structural + hash checks against a
     retained run of tools/core070_aghq_batch.R (receipt.json,
     julia-results.json, results.tsv, diagnostics.log).
  3. --self-test -- mutates a *synthetic* valid report (built from the frozen
     contract, no retained run required) in several independent ways and
     asserts every mutation is rejected by check_report(). Runs without R,
     Julia, or the frozen library. Per the 2026-09-01 postfit-policy
     vacuous-pass incident, --self-test NEVER substitutes for the real
     --state check below: passing --self-test alone still exits nonzero if
     --state does not exist.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from parity_oracle import R_REF_PINS, SELECTED_PIN  # noqa: E402  (GLLVM_PARITY_PIN; P0 default; unknown pin exits)
import core070_source_pin_check  # noqa: E402  (library pin + version, shared with the other P1 batches)
# P0 keeps the frozen contract; P1 reads the twin written by tools/core070_aghq_p1_contract.py.
CONTRACT_PATH = ROOT / ("docs/dev-log/core070/true-parity-latest/aghq-batch-contract-p1.json"
                        if SELECTED_PIN == "P1" else "docs/dev-log/core070/aghq-batch-contract.json")
REFERENCE_COMMIT = R_REF_PINS[SELECTED_PIN] if SELECTED_PIN == "P1" else "b4d5fee64def88bc768dda1f1f77c29b295edd86"
DEFAULT_STATE = ROOT / ".unlazy/core070-aghq/aghq-batch-01"

# 16 paired-control cases + 22 needs-new-surface rows = 38 accounted rows.
# 7 of the 16 cases and all 22 needs-new-surface rows come from the 29-row
# required_core/compatibility_adapter scope in required-source-case-map.json
# (29 = 7 + 22); the remaining 9 cases (AGHQ-INVALID-*) are executed from the
# frozen plan's own paired_control_cases even though their source rows carry
# classification="rejected" there (out of the 29-row scope) -- see each such
# case's source_classification_note in the contract.
SCOPE_ROW_COUNT = 29
OUT_OF_SCOPE_EXECUTED_COUNT = 9  # the 9 AGHQ-INVALID-* paired controls


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
    need(c["status"] == "FROZEN_AGHQ_CONTROL_BATCH_CONTRACT", "wrong contract status")
    need(c["area"] == "aghq-control", "wrong area")
    need(c["reference_commit"] == REFERENCE_COMMIT, "wrong reference commit")
    need(c["source_row_count"] == SCOPE_ROW_COUNT, "source row count drift")

    case_ids = {x["case_id"] for x in c["cases"]}
    needs_surface_ids = {x["source_id"] for x in c["needs_new_julia_surface"]}
    reclass_ids = {x.get("source_id") for x in c.get("reclassify_rows", [])}
    spec_defect_ids = {x.get("source_id") for x in c.get("spec_defect_notes", [])}

    need(len(case_ids) == len(c["cases"]) == c["expected_case_count"] == 16,
         "executable case id set has duplicates or wrong count")
    need(len(needs_surface_ids) == len(c["needs_new_julia_surface"]) ==
         c["needs_new_julia_surface_count"] == 22,
         "needs-new-surface id set has duplicates or wrong count")
    need(len(c.get("reclassify_rows", [])) == c.get("reclassify_row_count", 0) == 0,
         "reclassify-row count drift (expected 0 for this batch)")
    need(len(c.get("spec_defect_notes", [])) == c.get("spec_defect_note_count", 0) == 0,
         "spec-defect note count drift")

    case_source_ids = {x["source_id"] for x in c["cases"]}
    need(len(case_source_ids & needs_surface_ids) == 0,
         "a source_id appears in both the executable and needs-new-surface buckets")

    in_scope_case_source_ids = {
        x["source_id"] for x in c["cases"]
        if x.get("source_classification_note") != "rejected_in_required_source_case_map_out_of_29_row_scope"
    }
    out_of_scope_case_source_ids = case_source_ids - in_scope_case_source_ids
    need(len(out_of_scope_case_source_ids) == OUT_OF_SCOPE_EXECUTED_COUNT,
         f"expected exactly {OUT_OF_SCOPE_EXECUTED_COUNT} out-of-scope (AGHQ-INVALID-*) executed cases")

    total_in_scope_rows = len(in_scope_case_source_ids) + len(needs_surface_ids) + len(reclass_ids) + len(spec_defect_ids)
    need(total_in_scope_rows == SCOPE_ROW_COUNT,
         f"bucket accounting does not sum to the {SCOPE_ROW_COUNT}-row required-source-case-map scope (got {total_in_scope_rows})")

    for x in c["cases"]:
        need(x["case_id"].startswith("CORE070-AGHQ-") and x["case_id"].endswith("-PAIRED-CONTROL"),
             f"malformed case_id {x['case_id']}")
        need(isinstance(x.get("source_id"), str) and x["source_id"].startswith("aghq/"),
             f"malformed source_id on {x['case_id']} (must carry the verbatim aghq/ prefix)")
        for key in ("evidence_kind", "r_call", "r_assertion", "julia_call", "expected",
                    "acceptance_rule", "fixture", "fixture_sha256"):
            need(isinstance(x.get(key), str) and len(x[key]) > 0, f"{x['case_id']} missing {key}")
        need(len(x["fixture_sha256"]) == 64, f"{x['case_id']} malformed fixture_sha256")

    for x in c["needs_new_julia_surface"]:
        need(isinstance(x.get("source_id"), str) and x["source_id"].startswith("aghq/"),
             f"malformed needs-surface source_id {x.get('source_id')} (must carry the verbatim aghq/ prefix)")
        for key in ("missing_surface", "reason"):
            need(isinstance(x.get(key), str) and len(x[key]) > 0, f"{x['source_id']} missing {key}")

    need(len(c["negative_controls"]) >= 3, "contract itself must name >=3 negative controls")
    for nc in c["negative_controls"]:
        for key in ("control_id", "description", "check"):
            need(isinstance(nc.get(key), str) and len(nc[key]) > 0, f"{nc.get('control_id')} missing {key}")

    need(isinstance(c["source_pins"], dict) and len(c["source_pins"]) == 4, "wrong source_pins count")
    for path_str, digest in c["source_pins"].items():
        need(len(digest) == 64 and all(ch in "0123456789abcdef" for ch in digest),
             f"malformed sha256 for {path_str}")

    print("CORE070_AGHQ_CONTRACT_VERIFIED",
          "executable=", len(case_ids), "needs_surface=", len(needs_surface_ids),
          "in_scope_total=", total_in_scope_rows, "out_of_scope_executed=", len(out_of_scope_case_source_ids))
    return c


# ---------------------------------------------------------------------------
# 2. Retained-run checks.
# ---------------------------------------------------------------------------
def check_report(report, contract, contract_sha256):
    contract_cases = {c["case_id"]: c for c in contract["cases"]}
    contract_neg = {c["control_id"]: c for c in contract["negative_controls"]}

    need(report.get("status") == "PASS", "batch did not pass")
    need(report.get("scope") == "CORE070_AGHQ_CONTROL_BATCH", "wrong scope")
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
    core070_source_pin_check.check_source_pin(receipt, SELECTED_PIN, need)
    need(receipt.get("julia_results_sha256") == sha(julia_path), "julia-results.json changed since receipt")
    need(receipt.get("raw_sha256") == sha(raw_path), "results.tsv changed since receipt")
    need(receipt.get("diagnostics_sha256") == sha(diag_path), "diagnostics.log changed since receipt")

    # All 16 cases and all 3 negative controls are decided entirely inside the
    # Julia child in this batch (unlike namespace-2, nothing is decided
    # R-only), so julia-results.json is checked directly against the full
    # contract.
    report = json.loads(julia_path.read_text())
    check_report(report, contract, contract_sha256)

    raw_lines = [ln for ln in raw_path.read_text().splitlines() if ln.strip()]
    raw_positive = {ln.split("\t")[0]: ln.split("\t")[1] for ln in raw_lines if ln.endswith("positive")}
    need(set(raw_positive) == {c["case_id"] for c in contract["cases"]},
         "results.tsv positive-case id set does not match contract (all 16 expected)")
    need(all(v == "PASS" for v in raw_positive.values()), "a case in results.tsv did not PASS")

    diag_text = diag_path.read_text().strip()
    need(diag_text == "", "diagnostics.log is non-empty on an all-PASS run")

    print("CORE070_AGHQ_CONTROL_BATCH_VERIFIED", len(contract["cases"]),
          "negative_controls", len(contract["negative_controls"]))
    return {
        "status": "CORE070_AGHQ_CONTROL_BATCH_PASS",
        "case_count": contract["expected_case_count"],
        "negative_control_count": len(contract["negative_controls"]),
        "receipt_sha256": sha(receipt_path),
        "julia_results_sha256": sha(julia_path),
    }


# ---------------------------------------------------------------------------
# 3. Self-test: synthetic report + mutation battery (no retained run needed).
# ---------------------------------------------------------------------------
def synthetic_report(contract):
    cases = {c["case_id"]: {"pass": True} for c in contract["cases"]}
    negatives = {
        nc["control_id"]: {"behaved": True}
        for nc in contract["negative_controls"]
    }
    return {
        "status": "PASS",
        "area": "aghq-control",
        "scope": "CORE070_AGHQ_CONTROL_BATCH",
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

    first_case_id = next(iter(good["cases"]))
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
         lambda r: r["cases"].__setitem__("CORE070-AGHQ-BOGUS-PAIRED-CONTROL", {"pass": True})),
    ]
    for name, mutate in mutations:
        bad = deepcopy(good)
        mutate(bad)
        try:
            check_report(bad, contract, contract_sha256)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid aghq-batch evidence: {name}")
    print("CORE070_AGHQ_BATCH_NEGATIVES_PASS", len(mutations))

    # Contract-level mutations: a tampered bucket accounting must also be caught.
    contract_mutations = [
        ("shrink executable cases without updating expected_case_count",
         lambda c: c["cases"].pop()),
        ("duplicate a source_id across buckets",
         lambda c: c["needs_new_julia_surface"].append(
             {**{k: v for k, v in c["cases"][0].items() if k != "source_classification_note"},
              "missing_surface": "x", "reason": "x"})),
        ("wrong source_row_count", lambda c: c.update(source_row_count=28)),
        ("only two negative controls",
         lambda c: c.__setitem__("negative_controls", c["negative_controls"][:2])),
        ("mislabel an out-of-scope case as in-scope",
         lambda c: c["cases"][7].__setitem__("source_classification_note", "required_core_or_compatibility_adapter")),
    ]
    for name, mutate in contract_mutations:
        bad = deepcopy(contract)
        mutate(bad)
        try:
            verify_contract(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid aghq contract: {name}")
    print("CORE070_AGHQ_CONTRACT_NEGATIVES_PASS", len(contract_mutations))

    n_pin = core070_source_pin_check.self_test(SELECTED_PIN)
    print("CORE070_AGHQ_BATCH_SOURCE_PIN_NEGATIVES_PASS", n_pin)
    return len(mutations) + len(contract_mutations) + n_pin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    # --self-test never substitutes for the real --state check below --
    # there is deliberately no flag that skips it (the postfit-policy
    # vacuous-pass incident: verifier exit 0 with only self-test output
    # while the real batch had produced nothing).
    mutation_count = self_test() if args.self_test else None

    if not args.state.exists():
        raise SystemExit(f"verify_aghq_batch: state does not exist: {args.state}")
    result = verify_state(args.state)
    if result and mutation_count:
        result["self_test_mutations_rejected"] = mutation_count
    if args.output and result:
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
