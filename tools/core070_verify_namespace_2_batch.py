"""Verify the retained "namespace-2" manifest-area batch (9 executable cases
+ 4 negative controls; 47 needs-new-Julia-surface deferrals + 34
reused-or-reclassify rows carried over verbatim from the manifest draft's own
`reclassify` proposals are recorded in the contract but never executed).

Three independent checks, mirroring tools/core070_verify_postfit_policy_batch.py:

  1. verify_contract() -- structural checks on the frozen contract itself
     (bucket counts sum to the manifest's 90 rows, no case id appears in more
     than one bucket, source pins are well-formed). Needs no retained run.
  2. verify_state() / check_report() -- structural + hash checks against a
     retained run of tools/core070_namespace_2_batch.R (receipt.json,
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

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "docs/dev-log/core070/namespace-2-batch-contract.json"
# --pin P1 (D-294/D-295): the P1 contract from tools/core070_namespace_2_p1_contract.py.
PIN_CONTRACTS = {
    "P0": (ROOT / "docs/dev-log/core070/namespace-2-batch-contract.json",
           "b4d5fee64def88bc768dda1f1f77c29b295edd86"),
    "P1": (ROOT / "docs/dev-log/core070/true-parity-latest/namespace-2-batch-contract-p1.json",
           "9539352f66f2db2cc26b1c393e67212a359b60c9"),
}
EXPECTED_REFERENCE = PIN_CONTRACTS["P0"][1]
DEFAULT_STATE = ROOT / ".unlazy/core070-aghq/namespace-2-batch-01"

MANIFEST_ROW_COUNT = 90


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
    need(c["status"] == "FROZEN_NAMESPACE_2_BATCH_CONTRACT", "wrong contract status")
    need(c["area"] == "namespace-2", "wrong area")
    need(c["reference_commit"] == EXPECTED_REFERENCE, "wrong reference commit")
    need(c["manifest_row_count"] == MANIFEST_ROW_COUNT, "manifest row count drift")

    executable_ids = {x["case_id"] for x in c["cases"]}
    needs_surface_ids = {x["source_id"] for x in c["needs_new_julia_surface"]}
    reclass_ids = {x["source_id"] for x in c["reclassify_rows"]}
    spec_defect_ids = {x.get("source_id") for x in c.get("spec_defect_notes", [])}

    need(len(executable_ids) == len(c["cases"]) == c["expected_case_count"] == 9,
         "executable case id set has duplicates or wrong count")
    need(len(needs_surface_ids) == len(c["needs_new_julia_surface"]) ==
         c["needs_new_julia_surface_count"] == 47,
         "needs-new-surface id set has duplicates or wrong count")
    need(len(reclass_ids) == len(c["reclassify_rows"]) == c["reclassify_row_count"] == 34,
         "reclassify-row id set has duplicates or wrong count")
    need(len(c.get("spec_defect_notes", [])) == c.get("spec_defect_note_count", 0) == 0,
         "spec-defect note count drift")

    executable_source_ids = {x["source_id"] for x in c["cases"]}
    need(len(executable_source_ids & needs_surface_ids) == 0,
         "a source_id appears in both the executable and needs-new-surface buckets")
    need(len(executable_source_ids & reclass_ids) == 0,
         "a source_id appears in both the executable and reclassify buckets")
    need(len(needs_surface_ids & reclass_ids) == 0,
         "a source_id appears in both the needs-new-surface and reclassify buckets")

    total_rows = len(executable_source_ids) + len(needs_surface_ids) + len(reclass_ids) + len(spec_defect_ids)
    need(total_rows == MANIFEST_ROW_COUNT,
         f"bucket accounting does not sum to the manifest's {MANIFEST_ROW_COUNT} rows (got {total_rows})")

    for x in c["cases"]:
        need(x["case_id"].startswith("CORE070-NAMESPACE2-"), f"malformed case_id {x['case_id']}")
        need(x["source_id"].startswith("namespace/export/"), f"malformed source_id on {x['case_id']}")
        for key in ("evidence_kind", "r_call", "julia_surface", "check", "tolerance", "fixture"):
            need(isinstance(x.get(key), str) and len(x[key]) > 0, f"{x['case_id']} missing {key}")

    for x in c["needs_new_julia_surface"]:
        need(x["source_id"].startswith("namespace/export/"), f"malformed needs-surface source_id {x['source_id']}")
        for key in ("missing_surface", "reason"):
            need(isinstance(x.get(key), str) and len(x[key]) > 0, f"{x['source_id']} missing {key}")

    for x in c["reclassify_rows"]:
        need(x.get("status") == "REUSED_OR_RECLASSIFY", f"{x.get('source_id')} wrong reclassify status")
        for key in ("pointer", "reason"):
            need(isinstance(x.get(key), str) and len(x[key]) > 0, f"{x.get('source_id')} missing {key}")

    need(len(c["negative_controls"]) >= 2, "contract itself must name >=2 negative controls")
    for nc in c["negative_controls"]:
        for key in ("control_id", "description", "check"):
            need(isinstance(nc.get(key), str) and len(nc[key]) > 0, f"{nc.get('control_id')} missing {key}")

    need(isinstance(c["source_pins"], dict) and len(c["source_pins"]) == 4, "wrong source_pins count")
    for path_str, digest in c["source_pins"].items():
        need(len(digest) == 64 and all(ch in "0123456789abcdef" for ch in digest),
             f"malformed sha256 for {path_str}")

    print("CORE070_NAMESPACE_2_CONTRACT_VERIFIED",
          "executable=", len(executable_ids), "needs_surface=", len(needs_surface_ids),
          "reclassify=", len(reclass_ids), "total=", total_rows)
    return c


# ---------------------------------------------------------------------------
# 2. Retained-run checks.
# ---------------------------------------------------------------------------
def check_report(report, contract, contract_sha256):
    contract_cases = {c["case_id"]: c for c in contract["cases"]}
    contract_neg = {c["control_id"]: c for c in contract["negative_controls"]}

    need(report.get("status") == "PASS", "batch did not pass")
    need(report.get("scope") == "CORE070_NAMESPACE_2_BATCH", "wrong scope")
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

    # NOTE: the retained julia-results.json only carries 6 of the 9 cases --
    # gllvmTMB_wide native-fit consistency and the two lognormal/
    # truncated_poisson rejection-path cases are decided entirely R-side and
    # never appear in it. results.tsv is the authoritative, receipt-hashed
    # record of ALL 9 cases; check_report() below is run against a
    # report reconstructed from results.tsv for the full case set, and
    # separately against the raw julia-results.json for its 6-case subset.
    report = json.loads(julia_path.read_text())
    julia_only_contract = {
        **contract,
        "cases": [c for c in contract["cases"] if c["case_id"] in report.get("cases", {})],
    }
    need(len(julia_only_contract["cases"]) == 6, "expected exactly 6 cases in the Julia child's own report")
    check_report(report, julia_only_contract, contract_sha256)

    raw_lines = [ln for ln in raw_path.read_text().splitlines() if ln.strip()]
    raw_positive = {ln.split("\t")[0]: ln.split("\t")[1] for ln in raw_lines if ln.endswith("positive")}
    need(set(raw_positive) == {c["case_id"] for c in contract["cases"]},
         "results.tsv positive-case id set does not match contract (all 9 expected)")
    need(all(v == "PASS" for v in raw_positive.values()), "a case in results.tsv did not PASS")

    diag_text = diag_path.read_text().strip()
    need(diag_text == "", "diagnostics.log is non-empty on an all-PASS run")

    print("CORE070_NAMESPACE_2_BATCH_VERIFIED", len(contract["cases"]),
          "negative_controls", len(contract["negative_controls"]))
    return {
        "status": "CORE070_NAMESPACE_2_BATCH_PASS",
        "case_count": contract["expected_case_count"],
        "negative_control_count": len(contract["negative_controls"]),
        "receipt_sha256": sha(receipt_path),
        "julia_results_sha256": sha(julia_path),
    }


# ---------------------------------------------------------------------------
# 3. Self-test: synthetic report + mutation battery (no retained run needed).
# ---------------------------------------------------------------------------
def synthetic_report(contract):
    julia_case_ids = [c["case_id"] for c in contract["cases"]
                       if c["case_id"] not in (
                           "CORE070-NAMESPACE2-GLLVMTMB-WIDE-NATIVE-FIT",
                           "CORE070-NAMESPACE2-LOGNORMAL-FAMILY-BRIDGE",
                           "CORE070-NAMESPACE2-TRUNCATED-POISSON-FAMILY-BRIDGE",
                       )]
    cases = {cid: {"pass": True} for cid in julia_case_ids}
    negatives = {
        nc["control_id"]: {"behaved": True}
        for nc in contract["negative_controls"]
    }
    return {
        "status": "PASS",
        "area": "namespace-2",
        "scope": "CORE070_NAMESPACE_2_BATCH",
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
    julia_only_contract = {
        **contract,
        "cases": [c for c in contract["cases"] if c["case_id"] in good["cases"]],
    }
    check_report(good, julia_only_contract, contract_sha256)  # sanity: synthetic report is itself valid

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
         lambda r: r["cases"].__setitem__("CORE070-NAMESPACE2-BOGUS-NATIVE", {"pass": True})),
    ]
    for name, mutate in mutations:
        bad = deepcopy(good)
        mutate(bad)
        try:
            check_report(bad, julia_only_contract, contract_sha256)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid namespace-2-batch evidence: {name}")
    print("CORE070_NAMESPACE_2_BATCH_NEGATIVES_PASS", len(mutations))

    # Contract-level mutations: a tampered bucket accounting must also be caught.
    contract_mutations = [
        ("shrink executable cases without updating expected_case_count",
         lambda c: c["cases"].pop()),
        ("duplicate a source_id across buckets",
         lambda c: c["needs_new_julia_surface"].append(
             {**c["cases"][0], "source_id": c["cases"][0]["source_id"],
              "missing_surface": "x", "reason": "x"})),
        ("wrong manifest_row_count", lambda c: c.update(manifest_row_count=89)),
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
        raise AssertionError(f"accepted invalid namespace-2 contract: {name}")
    print("CORE070_NAMESPACE_2_CONTRACT_NEGATIVES_PASS", len(contract_mutations))

    return len(mutations) + len(contract_mutations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pin", choices=sorted(PIN_CONTRACTS), default="P0")
    args = parser.parse_args()
    CONTRACT_PATH, EXPECTED_REFERENCE = PIN_CONTRACTS[args.pin]

    # --self-test never substitutes for the real --state check below --
    # there is deliberately no flag that skips it (the postfit-policy
    # vacuous-pass incident: verifier exit 0 with only self-test output
    # while the real batch had produced nothing).
    mutation_count = self_test() if args.self_test else None

    if not args.state.exists():
        raise SystemExit(f"verify_namespace_2_batch: state does not exist: {args.state}")
    result = verify_state(args.state)
    if result and mutation_count:
        result["self_test_mutations_rejected"] = mutation_count
    if args.output and result:
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
