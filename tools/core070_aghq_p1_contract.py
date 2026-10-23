"""Regenerate the Core070 aghq contracts at gllvmTMB pin P1.

In scope: the 21 required aghq rows (19 required_core, 2 compatibility_adapter)
the P1 carry scan lists as DANGLING (7) or PARTIAL_STALE_AT_P1 (14). They were
paid at P0 by two batches, neither of which compares an R number with a Julia
number:

  P0 source (unchanged)                          P1 twin (written here)
  aghq-batch-contract.json                       aghq-batch-contract-p1.json
    (7 AGHQ-CTRL-* rows: paired control of R's .gllvmTMB_normalize_aghq and
     GLLVModels._aghq_request; categorical labels, no fit)
  aghq-public-policy-bind-receipt-2026-09-04.json aghq-public-policy-contract-p1.json
    + tools/core070_aghq_public_policy_bind.R
    (14 AUTO-K / DEFAULT-OFF / POLICY rows: public gllvmTMB() fits read for
     fit$aghq; R only, no Julia call is part of the case)

The P0 files are read and never written; they stay as history.

aghq-batch-contract-p1.json, recorded in its `regeneration_log`:

  * reference_commit -> P1 (from tools/core070_oracle_pins.toml, not a literal);
  * source_pins recomputed from the P1 bytes (`git -C $GLLVMTMB_DIR show
    <P1>:<path>`), after checking each P0 pin against the P0 bytes; at P1 the
    runner checks them under CORE070_P1_R_SOURCE_ROOT;
  * `p0_to_p1_r_function_diff` for .gllvmTMB_normalize_aghq (a record, not a gate);
  * `needs_new_julia_surface_p1_note`: 14 of the 22 carried deferrals were later
    bound at P0 by the R-only public policy bind; where their P1 re-measure lives.

aghq-public-policy-contract-p1.json has no P0 contract to twin: the P0 bind kept
its cases in the runner. It records the 14 row ids and their case ids (from
tools/core070_aghq_public_policy_bind_apply.py), each row's public call, fixture
and expected k carried verbatim from the P0 receipt, the P0 observation for
reference, the runner's sha256 (so a runner edit makes this contract stale), and
`p0_to_p1_r_function_diff` for the R functions the public fit routes through.
The assertions themselves are the runner's; nothing is re-derived.

No case, expectation or tolerance is edited; a P1 mismatch is recorded by the
batch as a failure.

Usage:
  GLLVMTMB_DIR=/path/to/gllvmTMB python3 tools/core070_aghq_p1_contract.py [--check]

--check regenerates in memory and exits nonzero if either tracked P1 contract
differs, so a stale contract is caught without rewriting it.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core070_data_p1_contract import (  # noqa: E402  (same helpers as the data/fit-input/family twins)
    OUT_DIR, P0_DIR, ROOT, function_body, git, pins, sha_bytes)
from core070_aghq_public_policy_bind_apply import ROW_TO_CASE  # noqa: E402

GENERATOR = "tools/core070_aghq_p1_contract.py"
POLICY_RUNNER = "tools/core070_aghq_public_policy_bind.R"
POLICY_P0_RECEIPT = "docs/dev-log/core070/aghq-public-policy-bind-receipt-2026-09-04.json"

CONTROL_FUNCTIONS = [(".gllvmTMB_normalize_aghq", "R/gllvmTMB.R")]
POLICY_FUNCTIONS = [
    ("gllvmTMB", "R/gllvmTMB.R"),
    ("gllvmTMBcontrol", "R/gllvmTMB.R"),
    (".gllvmTMB_normalize_aghq", "R/gllvmTMB.R"),
    (".gllvmTMB_aghq_k", "R/fit-multi.R"),
    (".aghq_family_label", "R/aghq-control.R"),
    (".aghq_resolve", "R/aghq-control.R"),
    (".aghq_auto_gate", "R/aghq-control.R"),
    (".aghq_auto_decide", "R/aghq-control.R"),
    (".aghq_gate", "R/aghq-gate.R"),
]

CONTROL_SURFACE_NOTE = (
    "needs_new_julia_surface is carried verbatim from P0. Its 22 entries say the frozen case plan names no "
    "r_call/r_assertion/julia_call for those rows; that is a statement about the plan, not a probe of GLLVModels. "
    "Since this contract was frozen, 14 of the 22 (the AUTO-K, DEFAULT-OFF and POLICY-OFF/-EXPLICIT/-EXPLICIT-"
    "BYPASS-CUTOFF/-AUTO-ENFORCE-CUTOFF/-TRAITS19/-TRAITS20 rows) were bound at P0 by the R-only public policy "
    "bind (docs/dev-log/core070/aghq-public-policy-bind-receipt-2026-09-04.json); their P1 re-measure is "
    "aghq-public-policy-contract-p1.json. The other 8 are intentionally_excluded in the P0 case map.")


def function_diff(repo, functions, p0_sha, p1_sha):
    rows = []
    for fn, path in functions:
        b0 = function_body(git(repo, "show", f"{p0_sha}:{path}").decode(), fn)
        b1 = function_body(git(repo, "show", f"{p1_sha}:{path}").decode(), fn)
        if b0 is None or b1 is None:
            raise SystemExit(f"{fn} not found in {path} at {'P0' if b0 is None else 'P1'}")
        rows.append({"function": fn, "file": path,
                     "body_identical_p0_p1": b0 == b1,
                     "body_sha256_p0": sha_bytes(b0.encode()),
                     "body_sha256_p1": sha_bytes(b1.encode())})
    return rows


def build_control(repo):
    p0, p1 = pins()
    src = P0_DIR / "aghq-batch-contract.json"
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
    contract["p0_to_p1_r_function_diff"] = function_diff(repo, CONTROL_FUNCTIONS, p0["reference_commit"],
                                                         p1["reference_commit"])
    contract["needs_new_julia_surface_p1_note"] = CONTROL_SURFACE_NOTE
    log.append({"field": "needs_new_julia_surface_p1_note", "added": True,
                "why": "14 of the carried deferrals were bound at P0 by the R-only public policy bind"})
    contract["regeneration_log"] = {
        "generator": GENERATOR,
        "changes": log,
        "carried_verbatim": "status, cases (r_call, r_assertion, julia_call, expected, acceptance_rule, fixture), "
                            "needs_new_julia_surface, negative controls, fixture specification, rationale notes, "
                            "runner block. No case or expectation was edited; a P1 mismatch is recorded by the "
                            "batch as a failure.",
    }
    return json.dumps(contract, indent=2) + "\n"


def build_policy(repo):
    p0, p1 = pins()
    src = ROOT / POLICY_P0_RECEIPT
    receipt = json.loads(src.read_text())
    if receipt.get("status") != "PASS" or receipt.get("expected_count") != 14:
        raise SystemExit(f"{src.name} is not the 14-row PASS receipt")
    if sorted(receipt["bound_row_ids"]) != sorted(ROW_TO_CASE):
        raise SystemExit("P0 receipt row ids differ from the ledger-apply tool's ROW_TO_CASE")
    cases = []
    for rid in receipt["bound_row_ids"]:
        c = receipt["cases"][rid]
        cases.append({
            "row_id": rid,
            "source_id": f"aghq/{rid}",
            "case_id": ROW_TO_CASE[rid],
            "public_call": c["public_call"],
            "fixture": c.get("fixture"),
            "expected_k": c.get("expected_k"),
            "p0_detail": c["detail"],
            "p0_observed": c["observed"],
        })
    contract = {
        "schema": "core070-aghq-public-policy-contract/v1",
        "status": "P1_AGHQ_PUBLIC_POLICY_CONTRACT",
        "area": "aghq-public-policy",
        "reference_commit": p1["reference_commit"],
        "evidence_kind": "r_only_policy_observation",
        "twin_note": "Each case runs a public gllvmTMB() fit (or reads formals(gllvmTMBcontrol)) and checks R's own "
                     "fit$aghq record against the runner's assertion. No Julia call is part of any case, so no "
                     "row here compares R with Julia; a P1 pass shows the R contract still holds at P1, not parity.",
        "runner": POLICY_RUNNER,
        "runner_sha256": sha_bytes((ROOT / POLICY_RUNNER).read_bytes()),
        "runner_argv_p1": "GLLVM_PARITY_PIN=P1 Rscript --vanilla tools/core070_aghq_public_policy_bind.R "
                          "<frozen-library> <destination>",
        "assertions": "as coded in the runner (unchanged from P0): AUTO-K rows used && k == expected_k && finite "
                      "objective; DEFAULT-OFF formals(gllvmTMBcontrol)$aghq identical to FALSE; POLICY-OFF !used; "
                      "POLICY-EXPLICIT used && k == 3; EXPLICIT-BYPASS-CUTOFF used && k == 9 at 20 traits; "
                      "AUTO-ENFORCE-CUTOFF and TRAITS20 !used && reason mentions the cutoff at 20 traits; "
                      "TRAITS19 used && finite objective at 19 traits",
        "same_observation": {"AGHQ-POLICY-TRAITS20": "AGHQ-POLICY-AUTO-ENFORCE-CUTOFF"},
        "p0_receipt": POLICY_P0_RECEIPT,
        "p0_receipt_sha256": sha_bytes(src.read_bytes()),
        "p0_r_engine": {k: receipt["r_engine"].get(k) for k in ("git_head", "gllvmTMB_version", "load_method")},
        "p0_r_engine_note": "The P0 bind ran devtools::load_all on a gllvmTMB twin branch at the git_head above, "
                            "which is neither pin: it is not an ancestor of P1, and the P0 oracle is b4d5fee64.",
        "cases": cases,
        "p0_to_p1_r_function_diff": function_diff(repo, POLICY_FUNCTIONS, p0["reference_commit"],
                                                  p1["reference_commit"]),
        "regeneration_log": {
            "generator": GENERATOR,
            "source": "row ids, public calls, fixtures, expected k and P0 observations from the P0 receipt; case "
                      "ids from tools/core070_aghq_public_policy_bind_apply.py ROW_TO_CASE",
            "carried_verbatim": "public_call, fixture, expected_k, the runner's assertions. No case or expectation "
                                "was edited; a P1 mismatch is recorded by the batch as a failure.",
        },
    }
    return json.dumps(contract, indent=2) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    repo = os.environ.get("GLLVMTMB_DIR")
    if not repo:
        raise SystemExit("set GLLVMTMB_DIR to a gllvmTMB clone that contains the P1 commit")
    outputs = {OUT_DIR / "aghq-batch-contract-p1.json": build_control(repo),
               OUT_DIR / "aghq-public-policy-contract-p1.json": build_policy(repo)}
    if args.check:
        stale = [str(p.relative_to(ROOT)) for p, t in outputs.items() if not p.exists() or p.read_text() != t]
        if stale:
            print("STALE " + " ".join(stale))
            sys.exit(1)
        print("CORE070_AGHQ_P1_CONTRACTS_CURRENT")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p, t in outputs.items():
        p.write_text(t)
        print(f"wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
