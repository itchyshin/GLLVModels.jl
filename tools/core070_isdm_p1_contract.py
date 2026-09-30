"""Regenerate the Core070 isdm admission batch contract at gllvmTMB pin P1.

In scope: the 20 required isdm rows the P1 carry scan lists as DANGLING. All
20 are paid by one P0 batch, an R admission-predicate replay with no fit and no
Julia call:

  P0 contract (unchanged)       P1 twin (written here)
  isdm-batch-contract.json      true-parity-latest/isdm-batch-contract-p1.json

The P0 file is read and never written; it stays as history.

What changes, recorded in the output's `regeneration_log`:

  * reference_commit -> P1 (from tools/core070_oracle_pins.toml, not a literal);
  * source_pins: the three R/<file> sha256 values, checked against the P0 bytes
    and recomputed from the P1 bytes (`git -C $GLLVMTMB_DIR show <P1>:<path>`).
    At P1 the runner checks them under the P1 source tree the oracle was built
    from (its <source-root> argument);
  * `p0_to_p1_r_function_diff`: for each R function the runner loads, whether
    its body is byte-identical at P0 and P1. A record, not a gate;
  * `julia_surface_status_p1_note`: what the carried P0 julia_surface_status
    means at P1, given that draft PRs build an iSDM surface elsewhere.

Carried verbatim: status, area, cases (expression, expected, error_contains,
negative_control, classification, rationale), expected_case_count,
loaded_functions, fixture and fixture_sha256, tolerances, health_checks,
negative_control_case_ids, failure_action, claim_boundary. No case,
expectation or tolerance is edited; a P1 mismatch is recorded by the batch as a
failure.

Usage:
  GLLVMTMB_DIR=/path/to/gllvmTMB python3 tools/core070_isdm_p1_contract.py [--check]

--check regenerates in memory and exits nonzero if the tracked P1 contract
differs, so a stale contract is caught without rewriting it.
"""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core070_data_p1_contract import (  # noqa: E402  (same helpers as the data/fit-input/family twins)
    OUT_DIR, P0_DIR, ROOT, function_body, git, pins, sha_bytes)

GENERATOR = "tools/core070_isdm_p1_contract.py"
STEM = "isdm-batch-contract"

SURFACE_NOTE = (
    "julia_surface_status is carried verbatim from P0. At the commit this twin was generated on, GLLVModels' src/ "
    "has no iSDM code (no isdm_sources, isdm_source or integrated-source fitter; the receipt tool records a git "
    "grep of src/ at the run commit). Draft PRs #546 (isdm_sources/isdm_source, isdm_table, fit_isdm_gllvm, "
    "predict) and #558 (unique = TRUE) build that surface on another branch, and #546's "
    "test/parity/isdm_cases.jl runs 19 of these 20 predicates natively in Julia against an R replay at P1 "
    "(ISDM-LEGACY is R-only by decision D-296). Whether those twins should pay these rows is a maintainer "
    "decision; this batch re-measures the R side only, as at P0.")


def function_diff(repo, loaded, p0_sha, p1_sha):
    rows = []
    for path, fns in loaded.items():
        t0 = git(repo, "show", f"{p0_sha}:{path}").decode()
        t1 = git(repo, "show", f"{p1_sha}:{path}").decode()
        for fn in fns:
            b0, b1 = function_body(t0, fn), function_body(t1, fn)
            if b0 is None or b1 is None:
                raise SystemExit(f"{fn} not found in {path} at {'P0' if b0 is None else 'P1'}")
            rows.append({"function": fn, "file": path,
                         "body_identical_p0_p1": b0 == b1,
                         "body_sha256_p0": sha_bytes(b0.encode()),
                         "body_sha256_p1": sha_bytes(b1.encode())})
    return rows


def build(repo):
    p0, p1 = pins()
    src = P0_DIR / f"{STEM}.json"
    contract = json.loads(src.read_text())
    if contract["reference_commit"] != p0["reference_commit"]:
        raise SystemExit(f"{src.name} is not pinned at P0")
    log = [{"field": "reference_commit", "p0": contract["reference_commit"], "p1": p1["reference_commit"]}]
    contract["reference_commit"] = p1["reference_commit"]
    new = {}
    for rel, old in contract["source_pins"].items():
        if not rel.startswith("R/"):
            raise SystemExit(f"unexpected source pin path {rel}")
        if sha_bytes(git(repo, "show", f"{p0['reference_commit']}:{rel}")) != old:
            raise SystemExit(f"{src.name} source pin {rel} does not match the P0 bytes; refuse")
        new[rel] = sha_bytes(git(repo, "show", f"{p1['reference_commit']}:{rel}"))
        log.append({"field": f"source_pins[{rel}]", "p0": old, "p1": new[rel], "changed": old != new[rel]})
    contract["source_pins"] = new
    contract["p0_contract"] = str(src.relative_to(ROOT))
    contract["p0_contract_sha256"] = sha_bytes(src.read_bytes())
    contract["p0_to_p1_r_function_diff"] = function_diff(repo, contract["loaded_functions"],
                                                         p0["reference_commit"], p1["reference_commit"])
    contract["julia_surface_status_p1_note"] = SURFACE_NOTE
    log.append({"field": "julia_surface_status_p1_note", "added": True,
                "why": "the carried P0 status predates the draft iSDM port (#546, #558)"})
    contract["regeneration_log"] = {
        "generator": GENERATOR,
        "changes": log,
        "carried_verbatim": "status, area, cases (expression, expected, error_contains, negative_control, "
                            "classification, rationale), expected_case_count, loaded_functions, fixture and "
                            "fixture_sha256, tolerances, health_checks, negative_control_case_ids, failure_action, "
                            "claim_boundary. No case, expectation or tolerance was edited; a P1 mismatch is "
                            "recorded by the batch as a failure.",
    }
    return json.dumps(contract, indent=2) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    repo = os.environ.get("GLLVMTMB_DIR")
    if not repo:
        raise SystemExit("set GLLVMTMB_DIR to a gllvmTMB clone that contains the P1 commit")
    out = OUT_DIR / f"{STEM}-p1.json"
    text = build(repo)
    if args.check:
        if not out.exists() or out.read_text() != text:
            print("STALE " + str(out.relative_to(ROOT)))
            sys.exit(1)
        print("CORE070_ISDM_P1_CONTRACT_CURRENT")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
