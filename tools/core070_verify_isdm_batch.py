"""Verify the retained isdm manifest-area admission batch (20 cases).

Two independent checks:
  1. check_report() / verify_state() -- structural + hash checks against a
     retained run of tools/core070_isdm_batch.R (receipt.json,
     isdm-batch-results.json, raw.tsv, diagnostics.log).
  2. --self-test -- mutates a *synthetic* valid report (built from the frozen
     contract, no retained run required) in several ways and asserts every
     mutation is rejected by check_report(). This runs without R or the
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
import core070_source_pin_check  # noqa: E402  (library pin + version, shared with data / postfit / inference)
# P0 keeps the frozen contract; P1 reads the twin written by tools/core070_isdm_p1_contract.py.
CONTRACT_PATH = ROOT / ("docs/dev-log/core070/true-parity-latest/isdm-batch-contract-p1.json"
                        if SELECTED_PIN == "P1" else "docs/dev-log/core070/isdm-batch-contract.json")
REFERENCE_COMMIT = R_REF_PINS[SELECTED_PIN] if SELECTED_PIN == "P1" else "b4d5fee64def88bc768dda1f1f77c29b295edd86"
DEFAULT_STATE = ROOT / ".unlazy/core070-aghq/isdm-batch-01"


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
    negative_ids = set(contract["negative_control_case_ids"])
    need(len(negative_ids) >= 2, "contract itself must name >=2 negative controls")

    need(report.get("status") == "PASS", "batch did not pass")
    need(report.get("area") == "isdm", "wrong area")
    need(report.get("scope") == "CORE070_ISDM_BATCH", "wrong scope")
    need(report.get("reference_commit") == contract["reference_commit"], "wrong reference commit")
    need(report.get("contract_sha256") == contract_sha256, "stale contract")
    need(report.get("source_pins") == contract["source_pins"], "source pins changed")
    need(report.get("fixture_sha256") == contract["fixture_sha256"], "stale fixture")
    need(report.get("case_count") == contract["expected_case_count"] == len(contract_cases),
         "wrong case count")
    need(report.get("all_checks") is True, "runner reported a failing case")

    cases = report.get("cases", [])
    need(len(cases) == len(contract_cases), "missing or extra case")
    seen = set()
    negatives_checked = set()
    for case in cases:
        cid = case.get("manifest_case_id")
        need(cid in contract_cases and cid not in seen, f"unexpected or duplicate case {cid}")
        seen.add(cid)
        expected = contract_cases[cid]
        need(case.get("admission_case_id") == expected["admission_case_id"], f"{cid} id drift")
        need(case.get("source_id") == expected["source_id"], f"{cid} source_id drift")
        need(case.get("expression") == expected["expression"], f"{cid} expression drift")
        need(case.get("expected") == expected["expected"], f"{cid} expected-value drift")
        need(case.get("error_contains") == expected["error_contains"], f"{cid} error_contains drift")
        need(case.get("negative_control") == expected["negative_control"], f"{cid} negative_control drift")
        need(case.get("ok") is True, f"{cid} did not pass")
        if expected["expected"] == "TRUE":
            need(case.get("actual") is True, f"{cid} actual is not TRUE")
        else:
            need(isinstance(case.get("actual"), str) and expected["error_contains"] in case["actual"],
                 f"{cid} actual error text drift")
        if case.get("negative_control"):
            negatives_checked.add(cid.rsplit("-PAIRED-CONTROL", 1)[0])
    need(seen == set(contract_cases), "case set does not match contract")
    admission_negative_ids = {contract_cases[cid]["admission_case_id"]
                               for cid in seen if contract_cases[cid]["negative_control"]}
    need(admission_negative_ids == negative_ids, "negative-control set does not match contract")


def verify_state(state=DEFAULT_STATE):
    contract = load_contract()
    contract_sha256 = sha(CONTRACT_PATH)
    receipt_path = state / "receipt.json"
    results_path = state / "isdm-batch-results.json"
    raw_path = state / "raw.tsv"
    diag_path = state / "diagnostics.log"
    need(receipt_path.is_file() and results_path.is_file() and raw_path.is_file()
         and diag_path.is_file(), "missing retained run")

    receipt = json.loads(receipt_path.read_text())
    need(receipt.get("status") == "PASS" and receipt.get("source_unchanged") is True,
         "receipt did not pass")
    need(receipt.get("reference_commit") == contract["reference_commit"], "wrong reference commit")
    need(receipt.get("contract_sha256") == contract_sha256, "stale contract in receipt")
    need(receipt.get("source_pins") == contract["source_pins"], "receipt source pins changed")
    need(receipt.get("fixture_sha256") == contract["fixture_sha256"], "receipt fixture stale")
    need(receipt.get("case_count") == contract["expected_case_count"], "receipt case count wrong")
    need(receipt.get("expected_case_ids") == [c["manifest_case_id"] for c in contract["cases"]],
         "receipt case id list drifted or reordered")
    need(receipt.get("results_sha256") == sha(results_path), "results file changed since receipt")
    need(receipt.get("raw_sha256") == sha(raw_path), "raw.tsv changed since receipt")
    need(receipt.get("diagnostics_sha256") == sha(diag_path), "diagnostics.log changed since receipt")
    core070_source_pin_check.check_source_pin(receipt, SELECTED_PIN, need)

    # Live source pins are checked only if the readback source is present at the
    # conventional path recorded for this run; contract-side pinning already
    # guards the *content*, this re-derives it when the tree is available.
    # The readback is the P0 tree, so this re-derivation applies at P0 only.
    readback_root = ROOT / ".unlazy/core070-aghq/oracle-source/readback"
    if SELECTED_PIN == "P0" and readback_root.is_dir():
        for rel, digest in contract["source_pins"].items():
            local = readback_root / rel
            if local.is_file():
                need(sha(local) == digest, f"changed pinned source: {rel}")

    report = json.loads(results_path.read_text())
    check_report(report, contract, contract_sha256)

    raw_lines = raw_path.read_text().splitlines()
    expected_raw = [c["manifest_case_id"] + "\tPASS" for c in contract["cases"]]
    need(raw_lines == expected_raw, "raw.tsv does not match expected PASS lines")

    print("CORE070_ISDM_BATCH_VERIFIED", len(contract["cases"]))
    return {
        "status": "CORE070_ISDM_BATCH_PASS",
        "case_count": len(contract["cases"]),
        "receipt_sha256": sha(receipt_path),
        "results_sha256": sha(results_path),
    }


def synthetic_report(contract, contract_sha256):
    """A minimally valid report matching the contract, for mutation self-tests."""
    cases = []
    for c in contract["cases"]:
        cases.append({
            "manifest_case_id": c["manifest_case_id"],
            "admission_case_id": c["admission_case_id"],
            "source_id": c["source_id"],
            "expression": c["expression"],
            "expected": c["expected"],
            "error_contains": c["error_contains"],
            "negative_control": c["negative_control"],
            "actual": True if c["expected"] == "TRUE" else ("x" + c["error_contains"]),
            "ok": True,
        })
    return {
        "status": "PASS",
        "area": "isdm",
        "scope": "CORE070_ISDM_BATCH",
        "reference_commit": contract["reference_commit"],
        "contract_sha256": contract_sha256,
        "source_pins": contract["source_pins"],
        "fixture_sha256": contract["fixture_sha256"],
        "case_count": contract["expected_case_count"],
        "cases": cases,
        "all_checks": True,
    }


def self_test():
    contract = load_contract()
    contract_sha256 = sha(CONTRACT_PATH)
    good = synthetic_report(contract, contract_sha256)
    check_report(good, contract, contract_sha256)  # sanity: the synthetic report is itself valid

    first_negative = contract["negative_control_case_ids"][0]
    first_negative_manifest_id = next(
        c["manifest_case_id"] for c in contract["cases"]
        if c["admission_case_id"] == first_negative)
    first_case_id = good["cases"][0]["manifest_case_id"]

    mutations = [
        ("drop a case", lambda r: r["cases"].pop()),
        ("flip status", lambda r: r.update(status="FAIL")),
        ("flip all_checks", lambda r: r.update(all_checks=False)),
        ("flip one case's ok", lambda r: r["cases"][0].__setitem__("ok", False)),
        ("flip one case's actual", lambda r: r["cases"][0].__setitem__("actual", False)),
        ("relabel a negative control as positive",
         lambda r: next(c for c in r["cases"]
                         if c["manifest_case_id"] == first_negative_manifest_id)
                         .__setitem__("negative_control", False)),
        ("tamper with an expression", lambda r: r["cases"][1].__setitem__("expression", "TRUE")),
        ("stale contract hash", lambda r: r.update(contract_sha256="0" * 64)),
        ("change a source pin", lambda r: r["source_pins"].__setitem__(
            "R/isdm-sources.R", "0" * 64)),
        ("duplicate a case id",
         lambda r: r["cases"].append(deepcopy(r["cases"][0]))),
    ]
    for name, mutate in mutations:
        bad = deepcopy(good)
        mutate(bad)
        try:
            check_report(bad, contract, contract_sha256)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid isdm-batch evidence: {name}")
    print("CORE070_ISDM_BATCH_NEGATIVES_PASS", len(mutations))
    n_pin = core070_source_pin_check.self_test(SELECTED_PIN)
    print("CORE070_ISDM_BATCH_SOURCE_PIN_NEGATIVES_PASS", n_pin)
    return len(mutations) + n_pin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    mutation_count = self_test() if args.self_test else None
    result = None
    if args.state.exists() or not args.self_test:
        result = verify_state(args.state)
    if result and mutation_count:
        result["self_test_mutations_rejected"] = mutation_count
    if args.output and result:
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
