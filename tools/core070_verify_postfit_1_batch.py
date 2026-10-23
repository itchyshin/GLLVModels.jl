"""Verify the retained "postfit-1" manifest-area batch: 51 planned cases
triaged into 1 EXECUTABLE_NOW (CORE070-POSTFIT-COEF-MULTI-READBACK) and 50
NEEDS_NEW_JULIA_SURFACE, per docs/dev-log/core070/postfit-1-batch-contract.json.

Three independent checks:
  1. check_triage() -- structural check that the contract carries all 51
     cases with a valid bucket, and that the bucket counts match.
  2. check_pair() -- numeric-tolerance comparison of the retained R-side run
     (tools/core070_postfit_1_batch.R, Totoro-only, needs the frozen
     library) against the retained Julia-side run
     (tools/core070_postfit_1_batch.jl, locally runnable) for the one
     EXECUTABLE_NOW case.
  3. --self-test -- mutates synthetic valid contract/pair state in several
     independent ways (>=4 rejected mutations) plus >=2 negative controls
     (a mismatched R-vs-Julia pair; a case relabelled into the wrong
     bucket) and asserts every one is rejected. Runs without R, Julia, or
     any retained state.

Per the vacuous-pass incident (2026-09-01): a missing retained state is a
FAILURE, never a silent skip. verify_state() raises SystemExit if either
side's retained run is absent -- it never falls back to self-test output.
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
import core070_source_pin_check  # noqa: E402  (PR #569 review finding 2: library pin + version)
CONTRACT_PATH = ROOT / ("docs/dev-log/core070/true-parity-latest/postfit-1-batch-contract-p1.json"
                        if SELECTED_PIN == "P1" else "docs/dev-log/core070/postfit-1-batch-contract.json")
DEFAULT_R_STATE = ROOT / ".unlazy/core070-aghq/postfit-1-batch-r-01"
DEFAULT_JULIA_STATE = ROOT / ".unlazy/core070-aghq/postfit-1-batch-julia-01"

VALID_BUCKETS = {"EXECUTABLE_NOW", "NEEDS_NEW_JULIA_SURFACE", "SPEC_DEFECT"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def load_contract():
    return json.loads(CONTRACT_PATH.read_text())


# ---------------------------------------------------------------------------
# Check 1: triage structure (self-contained -- only needs the frozen contract)
# ---------------------------------------------------------------------------
def check_triage(contract):
    cases = contract.get("cases", [])
    need(contract.get("total_case_count") == 51 == len(cases), "wrong total case count")
    seen = set()
    counts = {"EXECUTABLE_NOW": 0, "NEEDS_NEW_JULIA_SURFACE": 0, "SPEC_DEFECT": 0}
    executable_ids = []
    for c in cases:
        cid = c.get("case_id")
        need(isinstance(cid, str) and cid not in seen, f"missing/duplicate case_id {cid}")
        seen.add(cid)
        bucket = c.get("bucket")
        need(bucket in VALID_BUCKETS, f"{cid}: invalid bucket {bucket}")
        counts[bucket] += 1
        need(isinstance(c.get("triage_note"), str) and len(c["triage_note"]) > 0,
             f"{cid}: empty triage_note")
        need(isinstance(c.get("r_call"), str) and len(c["r_call"]) > 0, f"{cid}: empty r_call")
        need(isinstance(c.get("julia_surface_planned"), str) and len(c["julia_surface_planned"]) > 0,
             f"{cid}: empty julia_surface_planned")
        if bucket == "EXECUTABLE_NOW":
            executable_ids.append(cid)
    need(counts == contract.get("bucket_counts"), "bucket_counts drift from actual case list")
    need(executable_ids == ["CORE070-POSTFIT-COEF-MULTI-READBACK"],
         f"unexpected EXECUTABLE_NOW set: {executable_ids}")
    return counts


def self_test_triage(contract):
    good = deepcopy(contract)
    check_triage(good)  # sanity

    mutations = [
        ("drop a case", lambda c: c["cases"].pop()),
        ("duplicate a case id", lambda c: c["cases"].append(deepcopy(c["cases"][1]))),
        ("flip a NEEDS_NEW case to EXECUTABLE_NOW without evidence",
         lambda c: next(x for x in c["cases"] if x["bucket"] == "NEEDS_NEW_JULIA_SURFACE")
                   .__setitem__("bucket", "EXECUTABLE_NOW")),
        ("blank a triage_note", lambda c: c["cases"][2].__setitem__("triage_note", "")),
        ("invalid bucket string", lambda c: c["cases"][3].__setitem__("bucket", "MAYBE")),
        ("wrong bucket_counts", lambda c: c["bucket_counts"].__setitem__("SPEC_DEFECT", 1)),
        ("wrong total_case_count", lambda c: c.update(total_case_count=50)),
        ("relabel the executable case away", lambda c: next(
            x for x in c["cases"] if x["case_id"] == "CORE070-POSTFIT-COEF-MULTI-READBACK"
        ).__setitem__("bucket", "NEEDS_NEW_JULIA_SURFACE")),
    ]
    for name, mutate in mutations:
        bad = deepcopy(good)
        mutate(bad)
        try:
            check_triage(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid triage: {name}")
    print("CORE070_POSTFIT_1_TRIAGE_NEGATIVES_PASS", len(mutations))
    return len(mutations)


# ---------------------------------------------------------------------------
# Check 2: R vs Julia numeric pair comparison for the one executable case
# ---------------------------------------------------------------------------
def check_pair(r_result, julia_result, contract):
    eb = contract["executable_batch"]
    need(r_result.get("case_id") == eb["case_id"] == julia_result.get("case_id"),
         "case_id mismatch across contract/R/Julia")
    need(r_result.get("reference_commit") == contract["reference_commit"] ==
         julia_result.get("reference_commit"), "reference_commit mismatch")
    need(r_result.get("p") == julia_result.get("p") == eb["fixture"]["p"], "p mismatch")
    need(r_result.get("n") == julia_result.get("n") == eb["fixture"]["n"], "n mismatch")
    need(r_result.get("K") == julia_result.get("K") == eb["fixture"]["K"], "K mismatch")
    need(r_result.get("all_checks") is True, "R side reported a failure")
    need(julia_result.get("all_checks") is True, "Julia side reported a failure")
    need(bool(julia_result.get("converged")), "Julia fit did not converge")

    r_coef = r_result.get("coef")
    j_coef = julia_result.get("coef")
    need(isinstance(r_coef, list) and isinstance(j_coef, list), "coef vectors missing")
    p = eb["fixture"]["p"]
    need(len(r_coef) == len(j_coef) == p, "coef length mismatch")

    tol = eb["tolerance"]["max_abs_diff"]
    diffs = [abs(a - b) for a, b in zip(r_coef, j_coef)]
    need(all(d <= tol for d in diffs), f"coef diff exceeds tolerance {tol}: {diffs}")
    return diffs


def synthetic_pair(contract, *, break_agreement=False, wrong_case=False):
    eb = contract["executable_batch"]
    j_coef = [-1.00580175, -0.2481022083333333, 0.35365920833333336, 0.9301414166666667]
    r_coef = list(j_coef) if not break_agreement else [x + 5.0 for x in j_coef]
    r_result = {
        "case_id": eb["case_id"] if not wrong_case else "SOME-OTHER-CASE",
        "reference_commit": contract["reference_commit"],
        "p": eb["fixture"]["p"], "n": eb["fixture"]["n"], "K": eb["fixture"]["K"],
        "coef": r_coef,
        "all_checks": True,
    }
    julia_result = {
        "case_id": eb["case_id"],
        "reference_commit": contract["reference_commit"],
        "p": eb["fixture"]["p"], "n": eb["fixture"]["n"], "K": eb["fixture"]["K"],
        "coef": j_coef,
        "converged": True,
        "all_checks": True,
    }
    return r_result, julia_result


def self_test_pair(contract):
    good_r, good_j = synthetic_pair(contract)
    check_pair(good_r, good_j, contract)  # sanity

    mutations = [
        ("flip R all_checks", lambda r, j: r.update(all_checks=False)),
        ("flip Julia all_checks", lambda r, j: j.update(all_checks=False)),
        ("Julia did not converge", lambda r, j: j.update(converged=False)),
        ("drop a coef entry on R side", lambda r, j: r["coef"].pop()),
        ("wrong reference_commit on R side", lambda r, j: r.update(reference_commit="0" * 40)),
        ("wrong p on Julia side", lambda r, j: j.update(p=999)),
    ]
    for name, mutate in mutations:
        bad_r, bad_j = deepcopy(good_r), deepcopy(good_j)
        mutate(bad_r, bad_j)
        try:
            check_pair(bad_r, bad_j, contract)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid pair evidence: {name}")

    # --- >=2 genuine negative controls: a live, discriminating comparison
    #     that MUST be rejected, not just a flipped boolean flag. ------------
    neg_r, neg_j = synthetic_pair(contract, break_agreement=True)
    try:
        check_pair(neg_r, neg_j, contract)
        raise AssertionError("negative control 1 (R/Julia coef disagree by 5.0) was accepted")
    except ValueError:
        pass

    neg_r2, neg_j2 = synthetic_pair(contract, wrong_case=True)
    try:
        check_pair(neg_r2, neg_j2, contract)
        raise AssertionError("negative control 2 (mismatched case_id) was accepted")
    except ValueError:
        pass

    print("CORE070_POSTFIT_1_PAIR_NEGATIVES_PASS", len(mutations) + 2)
    return len(mutations) + 2


# ---------------------------------------------------------------------------
# verify_state -- retained-run mode
# ---------------------------------------------------------------------------
def verify_state(r_state=DEFAULT_R_STATE, julia_state=DEFAULT_JULIA_STATE):
    contract = load_contract()
    triage_counts = check_triage(contract)

    r_receipt_path = r_state / "receipt.json"
    r_results_path = r_state / "postfit-1-r-results.json"
    j_results_path = julia_state / "postfit-1-julia-results.json"

    # Hardened lesson: missing state is a FAILURE, never a silent skip.
    if not r_receipt_path.is_file() or not r_results_path.is_file():
        raise SystemExit(
            f"verify_postfit_1_batch: missing retained R-side run under {r_state} "
            "(expected receipt.json + postfit-1-r-results.json from a Totoro run "
            "of tools/core070_postfit_1_batch.R against the frozen library)"
        )
    if not j_results_path.is_file():
        raise SystemExit(
            f"verify_postfit_1_batch: missing retained Julia-side run under {julia_state} "
            "(expected postfit-1-julia-results.json from tools/core070_postfit_1_batch.jl)"
        )

    r_receipt = json.loads(r_receipt_path.read_text())
    need(r_receipt.get("status") == "PASS", "R receipt did not pass")
    need(r_receipt.get("reference_commit") == contract["reference_commit"], "R receipt: wrong reference_commit")
    need(r_receipt.get("results_sha256") == sha(r_results_path), "R results.json changed since receipt")
    core070_source_pin_check.check_source_pin(r_receipt, SELECTED_PIN, need)

    r_result = json.loads(r_results_path.read_text())
    julia_result = json.loads(j_results_path.read_text())
    diffs = check_pair(r_result, julia_result, contract)

    print("CORE070_POSTFIT_1_BATCH_VERIFIED",
          "executable_now", triage_counts["EXECUTABLE_NOW"],
          "needs_new_julia_surface", triage_counts["NEEDS_NEW_JULIA_SURFACE"],
          "spec_defect", triage_counts["SPEC_DEFECT"],
          "max_coef_diff", max(diffs))
    return {
        "status": "CORE070_POSTFIT_1_BATCH_PASS",
        "triage_counts": triage_counts,
        "max_coef_diff": max(diffs),
        "r_receipt_sha256": sha(r_receipt_path),
        "julia_results_sha256": sha(j_results_path),
    }


def self_test():
    contract = load_contract()
    n1 = self_test_triage(contract)
    n2 = self_test_pair(contract)
    n3 = core070_source_pin_check.self_test(SELECTED_PIN)
    print("CORE070_POSTFIT_1_SOURCE_PIN_NEGATIVES_PASS", n3)
    total = n1 + n2 + n3
    print("CORE070_POSTFIT_1_BATCH_SELF_TEST_MUTATIONS_REJECTED", total)
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r-state", type=Path, default=DEFAULT_R_STATE)
    parser.add_argument("--julia-state", type=Path, default=DEFAULT_JULIA_STATE)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    mutation_count = self_test() if args.self_test else None
    # --self-test alone never substitutes for a real verification: fall
    # through to verify_state() regardless, which fails loudly if the
    # retained runs are absent (the vacuous-pass incident this guards).
    result = verify_state(args.r_state, args.julia_state)
    if result and mutation_count:
        result["self_test_mutations_rejected"] = mutation_count
    if args.output and result:
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
