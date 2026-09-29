"""Verify the retained "data" manifest-area batch (28 planned cases + 2
negative controls, all verdict SPEC_DEFECT on the Julia side).

Three independent checks:
  1. check_report() / verify_state() -- structural + hash checks against a
     retained run of tools/core070_data_batch.R (receipt.json,
     data-batch-results.json, raw.tsv, diagnostics.log).
  2. check_julia_introspection() -- structural checks against a retained run
     of tools/core070_data_batch.jl (the JSON receipt naming which candidate
     Julia symbols/keywords were checked and confirmed absent).
  3. --self-test -- mutates a *synthetic* valid report (built from the frozen
     contract, no retained run required) in several independent ways and
     asserts every mutation is rejected by check_report(). This runs without
     R, Julia, or the frozen source readback.
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
import core070_source_pin_check  # noqa: E402  (library pin + version, shared with postfit / inference)
# P0 keeps the frozen contract; P1 reads the twin written by tools/core070_data_p1_contract.py.
CONTRACT_PATH = ROOT / ("docs/dev-log/core070/true-parity-latest/data-batch-contract-p1.json"
                        if SELECTED_PIN == "P1" else "docs/dev-log/core070/data-batch-contract.json")
REFERENCE_COMMIT = R_REF_PINS[SELECTED_PIN] if SELECTED_PIN == "P1" else "b4d5fee64def88bc768dda1f1f77c29b295edd86"
DEFAULT_STATE = ROOT / ".unlazy/core070-aghq/data-batch-01"
DEFAULT_JULIA_RECEIPT = ROOT / ".unlazy/core070-aghq/data-batch-01/data-batch-julia-introspection.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def load_contract():
    contract = json.loads(CONTRACT_PATH.read_text())
    need(contract["reference_commit"] == REFERENCE_COMMIT, "contract is not pinned at the selected pin")
    return contract


def check_report(report, contract, contract_sha256):
    contract_cases = {c["manifest_case_id"]: c for c in contract["cases"]}
    positive_ids = {cid for cid, c in contract_cases.items() if not c["negative_control"]}
    negative_ids = {cid for cid, c in contract_cases.items() if c["negative_control"]}
    need(len(positive_ids) == contract["expected_case_count"] == 28, "wrong positive case count")
    need(len(negative_ids) >= 2, "contract itself must name >=2 negative controls")
    need(negative_ids == set(contract["negative_control_case_ids"]), "negative-control id set drift")

    need(report.get("status") == "PASS", "batch did not pass")
    need(report.get("area") == "data", "wrong area")
    need(report.get("scope") == "CORE070_DATA_BATCH", "wrong scope")
    need(report.get("reference_commit") == contract["reference_commit"], "wrong reference commit")
    need(report.get("contract_sha256") == contract_sha256, "stale contract")
    need(report.get("source_pins") == contract["source_pins"], "source pins changed")
    need(report.get("case_count") == contract["expected_case_count"], "wrong positive case count")
    need(report.get("negative_control_count") == len(negative_ids), "wrong negative-control count")
    need(report.get("all_positive_pass") is True, "a required case failed")
    need(report.get("negative_controls_behaved_as_expected") is True, "a negative control misbehaved")
    need(report.get("all_checks") is True, "runner reported an overall failure")

    cases = report.get("cases", [])
    need(len(cases) == len(contract_cases), "missing or extra case")
    seen = set()
    for case in cases:
        cid = case.get("manifest_case_id")
        need(cid in contract_cases and cid not in seen, f"unexpected or duplicate case {cid}")
        seen.add(cid)
        expected = contract_cases[cid]
        need(case.get("source_id") == expected["source_id"], f"{cid} source_id drift")
        need(case.get("fixture_case_id") == expected["fixture_case_id"], f"{cid} fixture_case_id drift")
        need(case.get("evidence_kind") == expected["evidence_kind"], f"{cid} evidence_kind drift")
        need(case.get("expression") == expected["expression"], f"{cid} expression drift")
        need(case.get("expected") == expected["expected"] == "TRUE", f"{cid} expected-value drift")
        need(case.get("negative_control") == expected["negative_control"], f"{cid} negative_control drift")
        need(case.get("is_error") is False, f"{cid} raised an error instead of evaluating")
        need(case.get("ok") is True, f"{cid} did not behave as required")
        if case.get("negative_control"):
            # A negative control's expression must have evaluated to FALSE --
            # ok==True here means "behaved as required" (i.e. correctly FALSE).
            need(case.get("actual") is False, f"{cid} negative control did not evaluate to FALSE")
            need(case.get("julia_verdict") == "n/a", f"{cid} negative control must not carry a Julia verdict")
        else:
            need(case.get("actual") is True, f"{cid} did not evaluate to TRUE")
            need(case.get("julia_verdict") == "SPEC_DEFECT", f"{cid} must be verdict SPEC_DEFECT")
            need(case.get("julia_surface") == expected["julia_surface"], f"{cid} julia_surface drift")
    need(seen == set(contract_cases), "case set does not match contract")


def check_julia_introspection(receipt, contract):
    need(receipt.get("reference_commit") == contract["reference_commit"], "julia receipt: wrong reference commit")
    need(receipt.get("all_planned_surfaces_absent") is True,
         "julia receipt claims a planned surface now exists -- SPEC_DEFECT verdicts in the contract are stale")
    surfaces = receipt.get("surfaces", {})
    contract_groups = set(contract["julia_planned_surfaces"])
    need(set(surfaces) == contract_groups, "julia receipt surface-group set does not match contract")
    for group, spec in contract["julia_planned_surfaces"].items():
        entry = surfaces.get(group, {})
        need(entry.get("surface_absent") is True, f"julia receipt: {group} not absent")
        need(entry.get("found_symbols") == [] and entry.get("found_kwargs") == [],
             f"julia receipt: {group} unexpectedly found a candidate")
        need(set(entry.get("candidate_exported_symbols", [])) == set(spec["candidate_exported_symbols"]),
             f"julia receipt: {group} candidate symbol list drift")
        need(set(entry.get("candidate_kwargs", [])) == set(spec["candidate_kwargs"]),
             f"julia receipt: {group} candidate kwarg list drift")
    need(isinstance(receipt.get("gllvm_exported_symbol_count"), int) and
         receipt["gllvm_exported_symbol_count"] > 0, "julia receipt: implausible exported-symbol count")


def verify_state(state=DEFAULT_STATE, julia_receipt_path=DEFAULT_JULIA_RECEIPT):
    contract = load_contract()
    contract_sha256 = sha(CONTRACT_PATH)
    receipt_path = state / "receipt.json"
    results_path = state / "data-batch-results.json"
    raw_path = state / "raw.tsv"
    diag_path = state / "diagnostics.log"
    need(receipt_path.is_file() and results_path.is_file() and raw_path.is_file()
         and diag_path.is_file(), "missing retained R-side run")

    receipt = json.loads(receipt_path.read_text())
    need(receipt.get("status") == "PASS" and receipt.get("source_unchanged") is True,
         "receipt did not pass")
    need(receipt.get("reference_commit") == contract["reference_commit"], "wrong reference commit")
    need(receipt.get("contract_sha256") == contract_sha256, "stale contract in receipt")
    need(receipt.get("source_pins") == contract["source_pins"], "receipt source pins changed")
    need(receipt.get("case_count") == contract["expected_case_count"], "receipt case count wrong")
    need(receipt.get("negative_control_count") == len(contract["negative_control_case_ids"]),
         "receipt negative-control count wrong")
    need(receipt.get("expected_case_ids") ==
         [c["manifest_case_id"] for c in contract["cases"] if not c["negative_control"]],
         "receipt positive-case id list drifted or reordered")
    need(receipt.get("negative_control_case_ids") == contract["negative_control_case_ids"],
         "receipt negative-control id list drifted or reordered")
    need(receipt.get("results_sha256") == sha(results_path), "results file changed since receipt")
    need(receipt.get("raw_sha256") == sha(raw_path), "raw.tsv changed since receipt")
    need(receipt.get("diagnostics_sha256") == sha(diag_path), "diagnostics.log changed since receipt")
    core070_source_pin_check.check_source_pin(receipt, SELECTED_PIN, need)

    # Live source pins are checked only if the readback source is present at the
    # conventional path; contract-side pinning already guards the content, this
    # re-derives it when the tree is available.
    readback_root = ROOT / ".unlazy/core070-aghq/oracle-source/readback"
    if readback_root.is_dir():
        for rel, digest in contract["source_pins"].items():
            local = readback_root / rel
            if local.is_file():
                need(sha(local) == digest, f"changed pinned source: {rel}")

    report = json.loads(results_path.read_text())
    check_report(report, contract, contract_sha256)

    raw_lines = raw_path.read_text().splitlines()
    expected_raw = [
        c["manifest_case_id"] + "\tPASS\t" + ("negative_control" if c["negative_control"] else "positive")
        for c in contract["cases"]
    ]
    need(raw_lines == expected_raw, "raw.tsv does not match expected PASS lines")
    need(diag_path.read_text().strip() == "", "diagnostics.log is non-empty on an all-PASS run")

    julia_ok = False
    julia_receipt_sha = None
    if julia_receipt_path.is_file():
        julia_receipt = json.loads(julia_receipt_path.read_text())
        check_julia_introspection(julia_receipt, contract)
        julia_ok = True
        julia_receipt_sha = sha(julia_receipt_path)

    print("CORE070_DATA_BATCH_VERIFIED", len(contract["cases"]), "julia_checked", julia_ok)
    return {
        "status": "CORE070_DATA_BATCH_PASS",
        "case_count": contract["expected_case_count"],
        "negative_control_count": len(contract["negative_control_case_ids"]),
        "receipt_sha256": sha(receipt_path),
        "results_sha256": sha(results_path),
        "julia_introspection_checked": julia_ok,
        "julia_introspection_sha256": julia_receipt_sha,
    }


def synthetic_report(contract, contract_sha256):
    """A minimally valid report matching the contract, for mutation self-tests."""
    cases = []
    for c in contract["cases"]:
        cases.append({
            "manifest_case_id": c["manifest_case_id"],
            "source_id": c["source_id"],
            "fixture_case_id": c["fixture_case_id"],
            "evidence_kind": c["evidence_kind"],
            "negative_control": c["negative_control"],
            "julia_surface": c["julia_surface"],
            "julia_verdict": "n/a" if c["negative_control"] else "SPEC_DEFECT",
            "expression": c["expression"],
            "expected": c["expected"],
            "actual": False if c["negative_control"] else True,
            "is_error": False,
            "ok": True,
        })
    n_negative = sum(1 for c in contract["cases"] if c["negative_control"])
    return {
        "status": "PASS",
        "area": "data",
        "scope": "CORE070_DATA_BATCH",
        "reference_commit": contract["reference_commit"],
        "contract_sha256": contract_sha256,
        "source_pins": contract["source_pins"],
        "case_count": contract["expected_case_count"],
        "negative_control_count": n_negative,
        "all_positive_pass": True,
        "negative_controls_behaved_as_expected": True,
        "cases": cases,
        "all_checks": True,
    }


def self_test():
    contract = load_contract()
    contract_sha256 = sha(CONTRACT_PATH)
    good = synthetic_report(contract, contract_sha256)
    check_report(good, contract, contract_sha256)  # sanity: the synthetic report is itself valid

    first_negative_id = contract["negative_control_case_ids"][0]

    mutations = [
        ("drop a case", lambda r: r["cases"].pop()),
        ("flip status", lambda r: r.update(status="FAIL")),
        ("flip all_checks", lambda r: r.update(all_checks=False)),
        ("flip all_positive_pass", lambda r: r.update(all_positive_pass=False)),
        ("flip negative_controls_behaved_as_expected",
         lambda r: r.update(negative_controls_behaved_as_expected=False)),
        ("flip one positive case's ok", lambda r: r["cases"][0].__setitem__("ok", False)),
        ("flip one positive case's actual to FALSE",
         lambda r: r["cases"][0].__setitem__("actual", False)),
        ("flip a negative control's actual to TRUE (should have been FALSE)",
         lambda r: next(c for c in r["cases"] if c["manifest_case_id"] == first_negative_id)
                        .__setitem__("actual", True)),
        ("relabel a negative control as positive",
         lambda r: next(c for c in r["cases"] if c["manifest_case_id"] == first_negative_id)
                        .__setitem__("negative_control", False)),
        ("claim a Julia surface exists for a positive case",
         lambda r: next(c for c in r["cases"] if not c["negative_control"])
                        .__setitem__("julia_verdict", "IMPLEMENTED")),
        ("tamper with an expression", lambda r: r["cases"][1].__setitem__("expression", "TRUE")),
        ("stale contract hash", lambda r: r.update(contract_sha256="0" * 64)),
        ("change a source pin", lambda r: r["source_pins"].__setitem__(
            "R/weights-shape.R", "0" * 64)),
        ("duplicate a case id", lambda r: r["cases"].append(deepcopy(r["cases"][0]))),
        ("mark an error case ok", lambda r: (r["cases"][0].__setitem__("is_error", True))),
    ]
    for name, mutate in mutations:
        bad = deepcopy(good)
        mutate(bad)
        try:
            check_report(bad, contract, contract_sha256)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid data-batch evidence: {name}")
    print("CORE070_DATA_BATCH_NEGATIVES_PASS", len(mutations))

    # Julia-introspection mutation checks, independent of the R-side report.
    julia_good = {
        "reference_commit": contract["reference_commit"],
        "all_planned_surfaces_absent": True,
        "gllvm_exported_symbol_count": 368,
        "surfaces": {
            group: {
                "candidate_exported_symbols": spec["candidate_exported_symbols"],
                "candidate_kwargs": spec["candidate_kwargs"],
                "found_symbols": [],
                "found_kwargs": [],
                "surface_absent": True,
            }
            for group, spec in contract["julia_planned_surfaces"].items()
        },
    }
    check_julia_introspection(julia_good, contract)  # sanity
    julia_mutations = [
        ("flip all_planned_surfaces_absent", lambda r: r.update(all_planned_surfaces_absent=False)),
        ("claim a found symbol while surface_absent stays true",
         lambda r: next(iter(r["surfaces"].values())).__setitem__("found_symbols", ["miss_control"])),
        ("flip one surface's surface_absent",
         lambda r: next(iter(r["surfaces"].values())).__setitem__("surface_absent", False)),
        ("drop a surface group", lambda r: r["surfaces"].popitem()),
        ("wrong reference commit", lambda r: r.update(reference_commit="0" * 40)),
    ]
    for name, mutate in julia_mutations:
        bad = deepcopy(julia_good)
        mutate(bad)
        try:
            check_julia_introspection(bad, contract)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid julia-introspection evidence: {name}")
    print("CORE070_DATA_BATCH_JULIA_NEGATIVES_PASS", len(julia_mutations))
    n_pin = core070_source_pin_check.self_test(SELECTED_PIN)
    print("CORE070_DATA_BATCH_SOURCE_PIN_NEGATIVES_PASS", n_pin)
    return len(mutations) + len(julia_mutations) + n_pin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--julia-receipt", type=Path, default=DEFAULT_JULIA_RECEIPT)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    mutation_count = self_test() if args.self_test else None
    # A missing state is a FAILURE, never a silent skip: the vacuous-pass
    # incident of 2026-09-01 (verifier exit 0 with only self-test output while
    # the real batch had produced nothing) is exactly what this guards.
    if not args.state.exists():
        raise SystemExit(f"verify_data_batch: state does not exist: {args.state}")
    result = verify_state(args.state, args.julia_receipt)
    if result and mutation_count:
        result["self_test_mutations_rejected"] = mutation_count
    if args.output and result:
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
