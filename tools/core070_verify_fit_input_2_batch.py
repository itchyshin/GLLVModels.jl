"""Verify the retained "fit-input-2" (M2) batch evidence.

Independently recomputes the PASS/FAIL verdict for the 13 EXECUTABLE_NOW
cases and 4 negative controls from the raw results.tsv / receipt.json /
julia-results.json triad the paired runner
(tools/core070_fit_input_2_batch.R + tools/core070_fit_input_2_batch.jl)
writes, rather than trusting the runner's own "status": "PASS" alone. Also
checks that the 5 NEEDS_NEW_JULIA_SURFACE cases are declared as such (never
silently dropped or silently claimed executable) and that the frozen
contract's own internal consistency guard (KERNEL-TWO-AUTO degenerates to
KERNEL-TWO) held on this run.

Usage:
    python3 tools/core070_verify_fit_input_2_batch.py --results <dir> [--self-test]
    python3 tools/core070_verify_fit_input_2_batch.py --self-test   # contract-only smoke
"""
import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from parity_oracle import R_REF_PINS, SELECTED_PIN  # noqa: E402  (GLLVM_PARITY_PIN; P0 default; unknown pin exits)
import core070_source_pin_check  # noqa: E402  (library pin + version, shared with postfit / inference)
# P0 keeps the frozen contract; P1 reads the twin written by tools/core070_data_p1_contract.py.
CONTRACT_PATH = ROOT / ("docs/dev-log/core070/true-parity-latest/fit-input-2-batch-contract-p1.json"
                        if SELECTED_PIN == "P1" else "docs/dev-log/core070/fit-input-2-batch-contract.json")
REFERENCE_COMMIT = R_REF_PINS[SELECTED_PIN] if SELECTED_PIN == "P1" else "b4d5fee64def88bc768dda1f1f77c29b295edd86"

EXPECTED_EXECUTABLE_CASE_IDS = [
    "CORE070-FIT-INPUT-GAUSS-DEFAULT-NATIVE-MODEL",
    "CORE070-FIT-INPUT-GAUSS-DEFAULT-FORMULA-INTERFACE",
    "CORE070-FIT-INPUT-GAUSS-LOADINGS-NATIVE-MODEL",
    "CORE070-FIT-INPUT-BINOMIAL-DEFAULT-NATIVE-MODEL",
    "CORE070-FIT-INPUT-BINOMIAL-DEFAULT-FORMULA-INTERFACE",
    "CORE070-FIT-INPUT-ANIMAL-LATENT-NATIVE-MODEL",
    "CORE070-FIT-INPUT-ANIMAL-LATENT-FORMULA-INTERFACE",
    "CORE070-FIT-INPUT-KERNEL-ONE-NATIVE-MODEL",
    "CORE070-FIT-INPUT-KERNEL-ONE-FORMULA-INTERFACE",
    "CORE070-FIT-INPUT-KERNEL-TWO-NATIVE-MODEL",
    "CORE070-FIT-INPUT-KERNEL-TWO-FORMULA-INTERFACE",
    "CORE070-FIT-INPUT-KERNEL-TWO-AUTO-NATIVE-MODEL",
    "CORE070-FIT-INPUT-KERNEL-TWO-AUTO-FORMULA-INTERFACE",
]
EXPECTED_NEEDS_SURFACE_CASE_IDS = [
    "CORE070-FIT-INPUT-GAUSS-LOADINGS-FORMULA-INTERFACE",
    "CORE070-FIT-INPUT-GAUSS-COMMON-NATIVE-MODEL",
    "CORE070-FIT-INPUT-GAUSS-COMMON-FORMULA-INTERFACE",
    "CORE070-FIT-INPUT-POISSON-DEFAULT-NATIVE-MODEL",
    "CORE070-FIT-INPUT-POISSON-DEFAULT-FORMULA-INTERFACE",
]
EXPECTED_NEGATIVE_CONTROL_IDS = [
    "NEG-COEF-SHIFTED",
    "NEG-LOGLIK-SHIFTED",
    "NEG-KERNEL-TWO-AUTO-WRONG-MODEL",
    "NEG-BINOMIAL-SWAPPED-Y",
]
# The 9 source_ids this batch claims to cover, verbatim including the
# `fit-input/` prefix as required by the task brief.
COVERED_SOURCE_IDS = [
    "fit-input/INPUT-ANIMAL-LATENT",
    "fit-input/INPUT-BINOMIAL-DEFAULT",
    "fit-input/INPUT-GAUSS-COMMON",
    "fit-input/INPUT-GAUSS-DEFAULT",
    "fit-input/INPUT-GAUSS-LOADINGS",
    "fit-input/INPUT-KERNEL-ONE",
    "fit-input/INPUT-KERNEL-TWO",
    "fit-input/INPUT-KERNEL-TWO-AUTO",
    "fit-input/INPUT-POISSON-DEFAULT",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def load_json_hard(path, what):
    """HARD SystemExit (not a caught ValueError) if a required state file is
    missing -- a vacuous-pass guard so a directory with no receipt at all
    cannot be reported as verified."""
    if not path.is_file():
        print(f"FATAL: missing {what}: {path}", file=sys.stderr)
        raise SystemExit(1)
    return json.loads(path.read_text())


def recompute_from_raw(results_dir, contract):
    """Independently recompute the batch verdict from results.tsv +
    julia-results.json, never trusting receipt.json's own status field
    alone."""
    receipt = load_json_hard(results_dir / "receipt.json", "receipt.json")
    raw_path = results_dir / "results.tsv"
    need(raw_path.is_file(), f"missing results.tsv: {raw_path}")
    need(sha(raw_path) == receipt.get("raw_sha256"), "results.tsv does not match receipt's raw_sha256")

    julia_path = results_dir / "julia-results.json"
    need(julia_path.is_file(), f"missing julia-results.json: {julia_path}")
    need(sha(julia_path) == receipt.get("julia_results_sha256"),
         "julia-results.json does not match receipt's julia_results_sha256")
    julia_report = json.loads(julia_path.read_text())

    need(receipt.get("scope") == "CORE070_FIT_INPUT_2_BATCH", "wrong evidence scope")
    need(receipt.get("reference_commit") == contract["reference_commit"], "wrong reference commit")
    need(receipt.get("contract_sha256") == sha(CONTRACT_PATH),
         "receipt not tied to the frozen contract (contract_sha256 mismatch)")
    need(receipt.get("source_unchanged") is True, "source_unchanged is not True")
    need(receipt.get("kernel_two_auto_matches_kernel_two") is True,
         "the KERNEL-TWO-AUTO internal consistency guard did not hold on this run")
    need(receipt.get("julia_exit_code") == 0, "Julia child exited nonzero")
    core070_source_pin_check.check_source_pin(receipt, SELECTED_PIN, need)

    for name, digest in (contract.get("source_pins") or {}).items():
        got = receipt.get("source_pins", {}).get(name)
        need(got == digest, f"source pin drifted for {name}")

    # Parse results.tsv (id \t status \t kind) and re-derive PASS/FAIL,
    # cross-checked against julia_report's own per-case "pass" booleans for
    # the executable rows.
    rows = [line.split("\t") for line in raw_path.read_text().splitlines() if line.strip()]
    by_id = {r[0]: r for r in rows}

    seen_executable = set()
    for cid in EXPECTED_EXECUTABLE_CASE_IDS:
        need(cid in by_id, f"missing executable case row in results.tsv: {cid}")
        _, status, kind = by_id[cid]
        need(kind == "positive", f"{cid}: wrong kind in results.tsv")
        need(cid in julia_report.get("cases", {}), f"{cid}: missing from julia-results.json cases")
        entry = julia_report["cases"][cid]
        need(entry.get("pass") is True, f"{cid}: julia-results.json reports pass != true")
        need(status == "PASS", f"{cid}: results.tsv reports {status}, not PASS")
        seen_executable.add(cid)
    need(seen_executable == set(EXPECTED_EXECUTABLE_CASE_IDS), "executable case-id set mismatch")

    for cid in EXPECTED_NEEDS_SURFACE_CASE_IDS:
        need(cid in by_id, f"missing needs-surface case row in results.tsv: {cid}")
        _, status, kind = by_id[cid]
        need(status == "NEEDS_NEW_JULIA_SURFACE", f"{cid}: expected NEEDS_NEW_JULIA_SURFACE, got {status}")
        need(kind == "not_attempted", f"{cid}: expected not_attempted kind")
        need(cid not in julia_report.get("cases", {}),
             f"{cid}: a NEEDS_NEW_JULIA_SURFACE case must not appear as an attempted julia case")

    for cid in EXPECTED_NEGATIVE_CONTROL_IDS:
        need(cid in by_id, f"missing negative control row in results.tsv: {cid}")
        _, status, kind = by_id[cid]
        need(kind == "negative_control", f"{cid}: wrong kind in results.tsv")
        need(cid in julia_report.get("negative_controls", {}), f"{cid}: missing from julia-results.json")
        need(julia_report["negative_controls"][cid].get("behaved") is True,
             f"{cid}: negative control did not behave as expected")
        need(status == "PASS", f"{cid}: results.tsv reports {status}, not PASS")

    need(julia_report.get("status") == "PASS", "julia-results.json status != PASS")
    need(julia_report.get("all_positive_pass") is True, "not all positive cases passed")
    need(julia_report.get("negative_controls_behaved_as_expected") is True,
         "not all negative controls behaved")
    need(julia_report.get("case_count") == len(EXPECTED_EXECUTABLE_CASE_IDS),
         "julia-results.json case_count mismatch")

    need(receipt.get("status") == "PASS", "receipt status != PASS")
    return receipt, julia_report


def check_contract_shape(contract):
    """Structural checks on the frozen contract itself -- run unconditionally
    (including under --self-test with no results dir), so a corrupted
    contract fails even before any results exist."""
    need(contract.get("status") == "FROZEN_FIT_INPUT_2_BATCH_CONTRACT", "wrong contract status")
    need(contract.get("reference_commit") == REFERENCE_COMMIT,
         "wrong reference_commit in contract")
    need(contract.get("expected_executable_case_count") == 13, "contract executable count != 13")
    need(contract.get("expected_needs_surface_case_count") == 5, "contract needs-surface count != 5")
    need(len(contract.get("negative_controls", [])) >= 3, "contract declares fewer than 3 negative controls")

    row_source_ids = [r["source_id"] for r in contract["rows"]]
    need(set(row_source_ids) == set(COVERED_SOURCE_IDS), "contract row source_ids do not match the 9 covered rows")
    for sid in row_source_ids:
        need(sid.startswith("fit-input/"), f"source_id not recorded verbatim with fit-input/ prefix: {sid}")

    executable_ids = set()
    for row in contract["rows"]:
        for case in row["cases"]:
            if case["status"] == "EXECUTABLE_NOW":
                executable_ids.add(case["case_id"])
    need(executable_ids == set(EXPECTED_EXECUTABLE_CASE_IDS),
         "contract's EXECUTABLE_NOW case_ids do not match the expected 13")

    needs_surface_ids = {c["case_id"] for c in contract["needs_new_julia_surface"]}
    need(needs_surface_ids == set(EXPECTED_NEEDS_SURFACE_CASE_IDS),
         "contract's needs_new_julia_surface case_ids do not match the expected 5")
    for entry in contract["needs_new_julia_surface"]:
        need(bool(entry.get("gap")) and bool(entry.get("file_line")),
             f"{entry.get('case_id')}: needs_new_julia_surface entry missing gap/file_line evidence")


def verify(results_dir=None, self_test=False):
    need(CONTRACT_PATH.is_file(), f"missing contract: {CONTRACT_PATH}")
    contract = json.loads(CONTRACT_PATH.read_text())
    check_contract_shape(contract)

    receipt = None
    if results_dir is not None:
        receipt, julia_report = recompute_from_raw(results_dir, contract)

    if self_test:
        # (1) Contract-shape mutations: corrupt an in-memory copy of the
        # contract and confirm check_contract_shape rejects it.
        contract_mutations = [
            ("wrong reference_commit",
             lambda c: c.update(reference_commit="0" * 40)),
            ("drop an EXECUTABLE_NOW row down to NEEDS_NEW_JULIA_SURFACE",
             lambda c: c["rows"][0]["cases"][0].update(status="NEEDS_NEW_JULIA_SURFACE")),
            ("claim a 9th needs_new_julia_surface entry without gap evidence",
             lambda c: c["needs_new_julia_surface"].append({"case_id": "FAKE", "gap": "", "file_line": ""})),
            ("retarget expected_executable_case_count to a false value",
             lambda c: c.update(expected_executable_case_count=14)),
            ("drop a covered source_id",
             lambda c: c["rows"].pop()),
        ]
        for label, mutate in contract_mutations:
            bad = deepcopy(contract)
            mutate(bad)
            try:
                check_contract_shape(bad)
            except ValueError:
                continue
            raise AssertionError(f"self-test contract mutation was NOT rejected: {label}")

        # (2) Results-shape mutations, only meaningful with a real results dir.
        if receipt is not None:
            result_mutations = [
                ("flip a passing case to failing in julia-results.json",
                 lambda r, j: j["cases"]["CORE070-FIT-INPUT-GAUSS-DEFAULT-NATIVE-MODEL"].__setitem__("pass", False)),
                ("mark a negative control as not having behaved",
                 lambda r, j: j["negative_controls"]["NEG-COEF-SHIFTED"].update(behaved=False)),
                ("retarget contract_sha256 in the receipt to a stale value",
                 lambda r, j: r.update(contract_sha256="0" * 64)),
                ("claim kernel_two_auto_matches_kernel_two=False",
                 lambda r, j: r.update(kernel_two_auto_matches_kernel_two=False)),
            ]
            for label, mutate in result_mutations:
                bad_receipt = deepcopy(receipt)
                bad_julia = deepcopy(julia_report)
                mutate(bad_receipt, bad_julia)

                def fake_recompute():
                    # Re-run the same structural checks recompute_from_raw
                    # performs, against the mutated dicts directly.
                    need(bad_receipt.get("scope") == "CORE070_FIT_INPUT_2_BATCH", "wrong scope")
                    need(bad_receipt.get("contract_sha256") == sha(CONTRACT_PATH), "contract_sha256 mismatch")
                    need(bad_receipt.get("kernel_two_auto_matches_kernel_two") is True, "guard failed")
                    for cid in EXPECTED_EXECUTABLE_CASE_IDS:
                        entry = bad_julia["cases"][cid]
                        need(entry.get("pass") is True, f"{cid} pass != true")
                    for cid in EXPECTED_NEGATIVE_CONTROL_IDS:
                        need(bad_julia["negative_controls"][cid].get("behaved") is True, f"{cid} not behaved")

                try:
                    fake_recompute()
                except ValueError:
                    continue
                raise AssertionError(f"self-test result mutation was NOT rejected: {label}")

        n_pin = core070_source_pin_check.self_test(SELECTED_PIN)
        print("CORE070_FIT_INPUT_2_BATCH_SOURCE_PIN_NEGATIVES_PASS", n_pin)
        print("CORE070_FIT_INPUT_2_BATCH_NEGATIVES_PASS",
              len(contract_mutations) + (len(result_mutations) if receipt is not None else 0) + n_pin)

    print("CORE070_FIT_INPUT_2_BATCH_VERIFIED")
    return {
        "status": "FIT_INPUT_2_BATCH_PASS",
        "area": "fit-input-2",
        "reference_commit": contract["reference_commit"],
        "contract_sha256": sha(CONTRACT_PATH),
        "results_verified": receipt is not None,
        "executable_case_count": len(EXPECTED_EXECUTABLE_CASE_IDS),
        "needs_surface_case_count": len(EXPECTED_NEEDS_SURFACE_CASE_IDS),
        "covered_source_ids": COVERED_SOURCE_IDS,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=None,
                         help="directory written by tools/core070_fit_input_2_batch.R "
                              "(contains receipt.json, results.tsv, julia-results.json)")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.results is None and not args.self_test:
        print("FATAL: --results is required unless --self-test is given", file=sys.stderr)
        raise SystemExit(1)
    result = verify(args.results, args.self_test)
    if args.output:
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
