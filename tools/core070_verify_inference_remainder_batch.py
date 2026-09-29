"""Verify the retained "inference remainder" batch (M2): the 19 unbound
`inference/` rows in docs/dev-log/core070/required-source-case-map.json
(CI-ROUTE-005, 008-014, 019-021, 026-028, 033, 035, 040-042) -- 5 executable
UNSUPPORTED-METHOD-REJECT cases + 2 needs-new-Julia-surface deferrals
(CI-ROUTE-005 standardized-loading Wald, CI-ROUTE-008..011 two-level
repeatability CI); see docs/dev-log/core070/inference-remainder-batch-contract.json
for the full case list and the ground-truth R source citations for each
estimand's method-dispatch behaviour.

REPAIR (2026-09-01): CORE070-INFERENCE-ICC-CI-METHOD-ROUTE was originally an
executable case here but crashed on Totoro -- R's route needs a genuine
two-tier fit (vB>0, vW>0) and Julia's twolevel.jl has no CI machinery for the
repeatability estimand at all (point-estimate only). Moved to
needs_new_julia_surface; see the contract's `notes` and `totoro_failed_attempts`.

Three independent checks, mirroring tools/core070_verify_namespace_2_batch.py:

  1. verify_contract() -- structural checks on the frozen contract itself
     (case/needs-surface bucket accounting sums to the 19 covered rows, no
     source_id appears in more than one bucket). Needs no retained run.
  2. verify_state() / check_report() -- structural + hash checks against a
     retained run of tools/core070_inference_remainder_batch.R (receipt.json,
     julia-results.json, results.tsv, diagnostics.log).
  3. --self-test -- mutates a *synthetic* valid report (built from the frozen
     contract, no retained run required) in several independent ways and
     asserts every mutation is rejected by check_report(). Runs without R,
     Julia, or the frozen library. --self-test NEVER substitutes for the real
     --state check below: passing --self-test alone still exits nonzero if
     --state does not exist.
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
CONTRACT_PATH = ROOT / ("docs/dev-log/core070/true-parity-latest/inference-remainder-batch-contract-p1.json"
                        if SELECTED_PIN == "P1" else "docs/dev-log/core070/inference-remainder-batch-contract.json")
REFERENCE_COMMIT = R_REF_PINS[SELECTED_PIN] if SELECTED_PIN == "P1" else "b4d5fee64def88bc768dda1f1f77c29b295edd86"
DEFAULT_STATE = ROOT / ".unlazy/core070-aghq/inference-remainder-batch-01"

MANIFEST_ROW_COUNT = 19


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
    need(c["status"] == "FROZEN_INFERENCE_REMAINDER_BATCH_CONTRACT", "wrong contract status")
    need(c["area"] == "inference-remainder", "wrong area")
    need(c["reference_commit"] == REFERENCE_COMMIT, "wrong reference commit")
    need(c["manifest_row_count"] == MANIFEST_ROW_COUNT, "manifest row count drift")

    covered = c["covered_source_ids"]
    need(len(covered) == len(set(covered)) == MANIFEST_ROW_COUNT,
         "covered_source_ids has duplicates or wrong count")
    for sid in covered:
        need(sid.startswith("inference/CI-ROUTE-"), f"malformed source_id {sid}")

    case_source_ids = set()
    for x in c["cases"]:
        need(x["case_id"].startswith("CORE070-INFERENCE-"), f"malformed case_id {x['case_id']}")
        need(isinstance(x.get("source_ids"), list) and len(x["source_ids"]) > 0,
             f"{x['case_id']} missing source_ids")
        for sid in x["source_ids"]:
            need(sid.startswith("inference/CI-ROUTE-"), f"malformed source_id {sid} on {x['case_id']}")
            case_source_ids.add(sid)
        for key in ("evidence_kind", "r_call", "julia_surface", "check", "fixture"):
            need(isinstance(x.get(key), str) and len(x[key]) > 0, f"{x['case_id']} missing {key}")
        need(isinstance(x.get("tolerance"), dict), f"{x['case_id']} missing tolerance dict")

    needs_surface_ids = set()
    for x in c["needs_new_julia_surface"]:
        need(isinstance(x.get("source_ids"), list) and len(x["source_ids"]) > 0,
             f"{x.get('case_id')} missing source_ids")
        for sid in x["source_ids"]:
            need(sid.startswith("inference/CI-ROUTE-"), f"malformed needs-surface source_id {sid}")
            needs_surface_ids.add(sid)
        for key in ("case_id", "missing_surface", "reason"):
            need(isinstance(x.get(key), str) and len(x[key]) > 0, f"{x.get('case_id')} missing {key}")

    need(len(case_source_ids & needs_surface_ids) == 0,
         "a source_id appears in both the executable and needs-new-surface buckets")

    total_rows = len(case_source_ids) + len(needs_surface_ids)
    need(total_rows == MANIFEST_ROW_COUNT,
         f"bucket accounting does not sum to the manifest's {MANIFEST_ROW_COUNT} rows (got {total_rows})")
    need(case_source_ids | needs_surface_ids == set(covered),
         "case + needs-surface source_ids do not exactly match covered_source_ids")

    need(len(c["cases"]) == c["expected_case_count"] == 5,
         "wrong executable case count")
    need(len(c["needs_new_julia_surface"]) == c["needs_new_julia_surface_count"] == 2,
         "wrong needs-new-surface count")
    need(len(c.get("reclassify_rows", [])) == c.get("reclassify_row_count", 0) == 0,
         "reclassify row count drift")
    need(len(c.get("spec_defect_notes", [])) == c.get("spec_defect_note_count", 0) == 0,
         "spec-defect note count drift")

    need(len(c["negative_controls"]) >= 2, "contract itself must name >=2 negative controls")
    for nc in c["negative_controls"]:
        for key in ("control_id", "description", "check"):
            need(isinstance(nc.get(key), str) and len(nc[key]) > 0, f"{nc.get('control_id')} missing {key}")

    need(isinstance(c["source_pins"], dict) and len(c["source_pins"]) >= 1, "wrong source_pins count")
    for path_str, digest in c["source_pins"].items():
        need(len(digest) == 64 and all(ch in "0123456789abcdef" for ch in digest),
             f"malformed sha256 for {path_str}")

    print("CORE070_INFERENCE_REMAINDER_CONTRACT_VERIFIED",
          "executable=", len(case_source_ids), "needs_surface=", len(needs_surface_ids),
          "total=", total_rows)
    return c


# ---------------------------------------------------------------------------
# 2. Retained-run checks.
# ---------------------------------------------------------------------------
def check_report(report, contract, contract_sha256):
    contract_cases = {c["case_id"]: c for c in contract["cases"]}
    contract_neg = {c["control_id"]: c for c in contract["negative_controls"]}

    need(report.get("status") == "PASS", "batch did not pass")
    need(report.get("scope") == "CORE070_INFERENCE_REMAINDER_BATCH", "wrong scope")
    need(report.get("case_count") == len(contract_cases), "wrong case count")
    need(report.get("negative_control_count") == len(contract_neg), "wrong negative-control count")
    need(report.get("all_positive_pass") is True, "a required case failed")
    need(report.get("negative_controls_behaved_as_expected") is True, "a negative control misbehaved")
    need(report.get("all_checks") is True, "runner reported an overall failure")

    cases = report.get("cases", {})
    need(set(cases) == set(contract_cases), "case id set does not match contract")
    for cid, case in cases.items():
        need(case.get("pass") is True, f"{cid} did not pass")
        methods = case.get("methods")
        if methods is not None:
            for m, entry in methods.items():
                need(entry.get("pass") is True, f"{cid}/{m} did not pass")
                need(entry.get("julia_error_kind") == "MethodError",
                     f"{cid}/{m} julia rejection did not match MethodError")

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
    need(sorted(receipt.get("expected_case_ids", [])) ==
         sorted(c["case_id"] for c in contract["cases"]), "receipt case id list drifted")
    need(sorted(receipt.get("covered_source_ids", [])) == sorted(contract["covered_source_ids"]),
         "receipt covered_source_ids drifted")
    need(receipt.get("julia_exit_code") == 0, "julia child exited nonzero")
    need(receipt.get("julia_results_sha256") == sha(julia_path), "julia-results.json changed since receipt")
    need(receipt.get("raw_sha256") == sha(raw_path), "results.tsv changed since receipt")
    need(receipt.get("diagnostics_sha256") == sha(diag_path), "diagnostics.log changed since receipt")
    core070_source_pin_check.check_source_pin(receipt, SELECTED_PIN, need)

    report = json.loads(julia_path.read_text())
    check_report(report, contract, contract_sha256)

    raw_lines = [ln for ln in raw_path.read_text().splitlines() if ln.strip()]
    raw_positive = {ln.split("\t")[0]: ln.split("\t")[1] for ln in raw_lines if ln.endswith("positive")}
    need(set(raw_positive) == {c["case_id"] for c in contract["cases"]},
         "results.tsv positive-case id set does not match contract")
    need(all(v == "PASS" for v in raw_positive.values()), "a case in results.tsv did not PASS")

    diag_text = diag_path.read_text().strip()
    need(diag_text == "", "diagnostics.log is non-empty on an all-PASS run")

    print("CORE070_INFERENCE_REMAINDER_BATCH_VERIFIED", len(contract["cases"]),
          "negative_controls", len(contract["negative_controls"]))
    return {
        "status": "CORE070_INFERENCE_REMAINDER_BATCH_PASS",
        "case_count": contract["expected_case_count"],
        "negative_control_count": len(contract["negative_controls"]),
        "receipt_sha256": sha(receipt_path),
        "julia_results_sha256": sha(julia_path),
    }


# ---------------------------------------------------------------------------
# 3. Self-test: synthetic report + mutation battery (no retained run needed).
# ---------------------------------------------------------------------------
def synthetic_report(contract):
    cases = {}
    for c in contract["cases"]:
        entry = {"pass": True}
        if "UNSUPPORTED-METHOD-REJECT" in c["case_id"]:
            methods = {}
            for m in c["source_ids"]:
                pass  # source_ids don't name methods directly; use a fixed synthetic set below
            method_names = ["wald_asym", "bogus"] if "RHO" in c["case_id"] else ["wald_asym", "fisher-z", "bogus"]
            for m in method_names:
                methods[m] = {"pass": True, "julia_error_kind": "MethodError",
                               "r_raised": True, "r_matches": True, "julia_ok": True}
            entry["methods"] = methods
        cases[c["case_id"]] = entry
    negatives = {
        nc["control_id"]: {"behaved": True}
        for nc in contract["negative_controls"]
    }
    return {
        "status": "PASS",
        "area": "inference-remainder",
        "scope": "CORE070_INFERENCE_REMAINDER_BATCH",
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
    reject_case_id = next(cid for cid in good["cases"] if "methods" in good["cases"][cid])
    reject_method = next(iter(good["cases"][reject_case_id]["methods"]))

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
         lambda r: r["cases"].__setitem__("CORE070-INFERENCE-BOGUS-ROUTE", {"pass": True})),
        ("reject method flips to non-MethodError",
         lambda r: r["cases"][reject_case_id]["methods"][reject_method].__setitem__(
             "julia_error_kind", "none")),
        ("reject method flips pass without fixing julia_error_kind",
         lambda r: r["cases"][reject_case_id]["methods"][reject_method].__setitem__("pass", True) or
                   r["cases"][reject_case_id]["methods"][reject_method].__setitem__(
                       "julia_error_kind", "ArgumentError")),
    ]
    for name, mutate in mutations:
        bad = deepcopy(good)
        mutate(bad)
        try:
            check_report(bad, contract, contract_sha256)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid inference-remainder-batch evidence: {name}")
    print("CORE070_INFERENCE_REMAINDER_BATCH_NEGATIVES_PASS", len(mutations))

    # Contract-level mutations: a tampered bucket accounting must also be caught.
    contract_mutations = [
        ("shrink executable cases without updating expected_case_count",
         lambda c: c["cases"].pop()),
        ("duplicate a source_id across buckets",
         lambda c: c["needs_new_julia_surface"].append(
             {**c["needs_new_julia_surface"][0], "source_ids": [c["cases"][0]["source_ids"][0]]})),
        ("wrong manifest_row_count", lambda c: c.update(manifest_row_count=18)),
        ("only one negative control",
         lambda c: c.__setitem__("negative_controls", c["negative_controls"][:1])),
        ("covered_source_ids drifts from case+needs-surface union",
         lambda c: c["covered_source_ids"].append("inference/CI-ROUTE-999")),
    ]
    for name, mutate in contract_mutations:
        bad = deepcopy(contract)
        mutate(bad)
        try:
            verify_contract(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid inference-remainder contract: {name}")
    print("CORE070_INFERENCE_REMAINDER_CONTRACT_NEGATIVES_PASS", len(contract_mutations))

    n_pin = core070_source_pin_check.self_test(SELECTED_PIN)
    print("CORE070_INFERENCE_REMAINDER_SOURCE_PIN_NEGATIVES_PASS", n_pin)

    return len(mutations) + len(contract_mutations) + n_pin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    # --self-test never substitutes for the real --state check below.
    mutation_count = self_test() if args.self_test else None

    if not args.state.exists():
        raise SystemExit(f"verify_inference_remainder_batch: state does not exist: {args.state}")
    result = verify_state(args.state)
    if result and mutation_count:
        result["self_test_mutations_rejected"] = mutation_count
    if args.output and result:
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
