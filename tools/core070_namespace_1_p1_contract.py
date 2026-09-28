"""Regenerate the namespace-1 Tier 0 batch contract at gllvmTMB pin P1.

Reads the frozen P0 contract (docs/dev-log/core070/namespace-1-batch-contract.json,
left untouched as history) and writes the P1 contract
(docs/dev-log/core070/true-parity-latest/namespace-1-batch-contract-p1.json).

R side: every NAMESPACE / R source byte is read with
`git -C $GLLVMTMB_DIR show <P1>:<path>` (the gllvmTMB clone is never checked
out or edited). The P1 commit and NAMESPACE hash come from the shared pin
source, tools/core070_oracle_pins.toml, not from a literal in this file.

Julia side: the rows whose Julia surface changed since the P0 triage are
listed explicitly in JULIA_AT_HEAD below, each with the src/ location that
justifies the change. Everything else is carried verbatim from P0.

Changes applied, all recorded in the output's `regeneration_log`:
  * rows whose export left the P1 NAMESPACE move to `retired_at_p1`
    (expected_r_registered = false; the check confirms the removal);
  * NEEDS_NEW_JULIA_SURFACE rows whose same-named Julia surface now exists
    and is exported at the branch head move to `cases`;
  * definition line hints and source pins are recomputed at P1;
  * negative controls are re-anchored, because the two P0 controls
    (deviance, tidy absent in Julia) are no longer true at the branch head.

Usage:
  GLLVMTMB_DIR=/path/to/gllvmTMB python3 tools/core070_namespace_1_p1_contract.py [--check]

--check regenerates in memory and exits nonzero if the tracked P1 contract
differs, so a stale contract is caught without rewriting it.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
P0_CONTRACT = ROOT / "docs/dev-log/core070/namespace-1-batch-contract.json"
P1_CONTRACT = ROOT / "docs/dev-log/core070/true-parity-latest/namespace-1-batch-contract-p1.json"
PINS_FILE = ROOT / "tools/core070_oracle_pins.toml"
PIN = "P1"

# NEEDS_NEW_JULIA_SURFACE rows (P0 triage) whose same-named, exported Julia
# surface exists at the branch head. Located by grep on src/ and confirmed with
# isdefined + Base.isexported on the loaded module (2026-09-27).
JULIA_AT_HEAD = {
    "CORE070-NAMESPACE-DEVIANCE-MULTI-NATIVE": (
        "deviance",
        "StatsAPI.deviance(fit::AnyGllvmFit) = -2 * loglikelihood(fit), src/postfit_tables.jl:21 "
        "(added a9464034d, core070 section 1.1); exported.",
    ),
    "CORE070-NAMESPACE-TIDY-MULTI-NATIVE": (
        "tidy",
        "tidy(fit::GllvmFit, Y; ...) src/postfit_tables.jl:898 (added df3e61b55, core070 section 1.13); exported.",
    ),
    "CORE070-NAMESPACE-CHECK-GLLVMTMB-DIAGNOSTIC": (
        "check_gllvmTMB",
        "check_gllvmTMB(fit; ...) src/diagnostics.jl:344 (added 0fc22cddf, core070 diagnostics cluster); exported.",
    ),
    "CORE070-NAMESPACE-CONFINT-INSPECT-NATIVE": (
        "confint_inspect",
        "confint_inspect(fit::GllvmFit, y; ...) src/diagnostics.jl:702 (added 0fc22cddf, core070 diagnostics cluster); exported.",
    ),
}

NEGATIVE_CONTROLS = [
    {
        "control_id": "NEG-DEP-ABSENT",
        "case_id": "CORE070-NAMESPACE-DEP-FORMULA-KEYWORD",
        "kind": "julia_absent",
        "description": "dep is a NEEDS_NEW_JULIA_SURFACE row whose Julia symbol is genuinely absent at the branch head; "
                       "the Julia facts must report exists=false and the row must not pass as if a surface existed.",
        "check": "julia_facts['dep']['exists'] == false AND expected_julia_symbol_exists == false for this case_id",
    },
    {
        "control_id": "NEG-RETIRED-EXPORT-UNREGISTERED",
        "case_id": "CORE070-NAMESPACE-PROPORTIONS-WALD-CI",
        "kind": "r_unregistered",
        "description": "export(.proportions_wald_ci) left the NAMESPACE between P0 and P1 while the function body stayed in "
                       "R/proportions-ci.R; the R scan must report registered=false, defined=true, proving the "
                       "registration check discriminates and is not reading the definition as registration.",
        "check": "r_facts[case_id].registered == false AND r_facts[case_id].defined == true",
    },
    {
        "control_id": "NEG-NEVER-EXISTED-SYMBOL",
        "case_id": None,
        "kind": "julia_absent_synthetic",
        "description": "A symbol that has never existed in GLLVModels.jl (gllvmTMB_julia_bridge_nonexistent_surface_zzz) "
                       "must resolve to exists=false, proving the Julia introspection path is not an always-true stub.",
        "check": "isdefined(GLLVModels, :gllvmTMB_julia_bridge_nonexistent_surface_zzz) == false",
    },
]


def load_pin():
    table = tomllib.loads(PINS_FILE.read_text())
    return table[PIN]


def gllvmtmb_dir():
    d = os.environ.get("GLLVMTMB_DIR")
    if not d:
        raise SystemExit("FATAL: set GLLVMTMB_DIR to a local gllvmTMB clone (read-only; git show only).")
    return d


def git_show_bytes(repo, commit, path):
    return subprocess.run(["git", "-C", repo, "show", f"{commit}:{path}"], check=True, capture_output=True).stdout


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def line_hint(text, pattern):
    for i, line in enumerate(text.splitlines(), start=1):
        if re.search(pattern, line):
            return i
    return None


def build():
    pin = load_pin()
    commit = pin["reference_commit"]
    repo = gllvmtmb_dir()
    resolved = subprocess.run(["git", "-C", repo, "rev-parse", f"{commit}^{{commit}}"],
                              check=True, capture_output=True, text=True).stdout.strip()
    if resolved != commit:
        raise SystemExit(f"FATAL: {repo} resolves {commit} to {resolved}")

    p0_bytes = P0_CONTRACT.read_bytes()
    p0 = json.loads(p0_bytes)

    ns_bytes = git_show_bytes(repo, commit, "NAMESPACE")
    ns_sha = sha256(ns_bytes)
    if ns_sha != pin["namespace_sha256"]:
        raise SystemExit(f"FATAL: NAMESPACE at {commit} hashes to {ns_sha}, pins file says {pin['namespace_sha256']}")
    ns_lines = {l.strip() for l in ns_bytes.decode().splitlines()}

    file_text = {}
    source_pins = {}
    pin_changes = []
    for rel, p0_digest in p0["source_pins"].items():
        b = git_show_bytes(repo, commit, rel)
        file_text[rel] = b.decode()
        source_pins[rel] = sha256(b)
        pin_changes.append({"path": rel, "sha256_at_p0": p0_digest, "sha256_at_p1": source_pins[rel],
                            "changed": p0_digest != source_pins[rel]})

    log = []
    cases, needs, retired = [], [], []

    def refresh(row):
        new = dict(row)
        hint = line_hint(file_text[row["r_file"]], row["r_definition_pattern"])
        if hint is None:
            raise SystemExit(f"FATAL: {row['case_id']}: definition pattern not found in {row['r_file']} at P1; "
                             "relocate by hand before regenerating")
        if hint != row["r_definition_line_hint"]:
            log.append({"case_id": row["case_id"], "change": "r_definition_line_hint",
                        "p0": row["r_definition_line_hint"], "p1": hint})
        new["r_definition_line_hint"] = hint
        return new

    for row in p0["cases"]:
        new = refresh(row)
        if row["r_namespace_line"] not in ns_lines:
            new["expected_r_registered"] = False
            new["retired_note"] = (f"{row['r_namespace_line']} is absent from the P1 NAMESPACE; the definition is still "
                                   f"present at {row['r_file']}:{new['r_definition_line_hint']} (internal, unexported).")
            retired.append(new)
            log.append({"case_id": row["case_id"], "change": "retired_at_p1", "reason": new["retired_note"]})
        else:
            cases.append(new)

    for row in p0["needs_new_julia_surface"]:
        new = refresh(row)
        if row["r_namespace_line"] not in ns_lines:
            raise SystemExit(f"FATAL: needs row {row['case_id']} left the P1 NAMESPACE; handle by hand")
        if row["case_id"] in JULIA_AT_HEAD:
            sym, note = JULIA_AT_HEAD[row["case_id"]]
            log.append({"case_id": row["case_id"], "change": "promoted_to_cases",
                        "julia_symbol_p0": row["julia_symbol"], "julia_symbol_p1": sym,
                        "expected_julia_symbol_exists_p0": row["expected_julia_symbol_exists"],
                        "reason": note})
            new["julia_symbol"] = sym
            new["julia_note"] = note
            new["expected_julia_symbol_exists"] = True
            cases.append(new)
        else:
            needs.append(new)

    out = {
        "schema": p0["schema"],
        "status": "FROZEN_NAMESPACE_1_BATCH_CONTRACT",
        "area": "namespace-1",
        "pin": PIN,
        "reference_commit": commit,
        "regenerated_from": {"path": str(P0_CONTRACT.relative_to(ROOT)), "sha256": sha256(p0_bytes),
                             "reference_commit": p0["reference_commit"]},
        "regenerated_by": "tools/core070_namespace_1_p1_contract.py",
        "manifest_source": p0["manifest_source"],
        "case_plan_annex": p0["case_plan_annex"],
        "manifest_row_count": p0["manifest_row_count"],
        "triage_note": ("Tier 0 at P1: existence/registration parity only, same tier as the P0 batch. R side reads "
                        "NAMESPACE and the cited R files with `git show <P1>:<path>` from a local gllvmTMB clone "
                        "(GLLVMTMB_DIR), no installed gllvmTMB, no RCall. Julia side is isdefined() on the loaded "
                        "GLLVModels module at the branch head. Not a numeric-output comparison (see P0 "
                        "runner.tier1_followup, still unbuilt)."),
        "expected_case_count": len(cases),
        "needs_new_julia_surface_count": len(needs),
        "retired_at_p1_count": len(retired),
        "spec_defect_count": 0,
        "reused_or_reclassify_count": len(p0["reused_or_reclassify"]),
        "source_access": "git-show",
        "namespace_pin": "NAMESPACE",
        "namespace_sha256": ns_sha,
        "source_pins": source_pins,
        "source_pin_changes_since_p0": pin_changes,
        "julia_source_root": "src",
        "cases": cases,
        "needs_new_julia_surface": needs,
        "retired_at_p1": retired,
        "spec_defect_notes": [],
        "reused_or_reclassify": p0["reused_or_reclassify"],
        "negative_controls": NEGATIVE_CONTROLS,
        "regeneration_log": log,
        "runner": {
            "outer": "tools/core070_namespace_1_batch.R",
            "inner": "tools/core070_namespace_1_batch.jl",
            "outer_usage": "GLLVM_PARITY_PIN=P1 GLLVMTMB_DIR=<gllvmTMB clone> Rscript --vanilla tools/core070_namespace_1_batch.R <destination>",
            "inner_invocation": "GLLVM_PARITY_PIN=P1 julia --project=. tools/core070_namespace_1_batch.jl <destination>/julia-facts.json",
            "needs_frozen_library": False,
        },
        "verifier": {
            "path": "tools/core070_verify_namespace_1_batch.py",
            "invocation": "GLLVM_PARITY_PIN=P1 python3 tools/core070_verify_namespace_1_batch.py <destination> --write-case-receipts <dir>",
            "self_test": "GLLVM_PARITY_PIN=P1 python3 tools/core070_verify_namespace_1_batch.py --self-test",
            "rejected_mutations_required": 4,
            "negative_controls_required": 2,
        },
    }
    return json.dumps(out, indent=1, ensure_ascii=False) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    text = build()
    if args.check:
        if not P1_CONTRACT.exists() or P1_CONTRACT.read_text() != text:
            print(f"STALE: {P1_CONTRACT.relative_to(ROOT)} differs from a fresh regeneration")
            sys.exit(1)
        print("CORE070_NAMESPACE_1_P1_CONTRACT_CURRENT")
        return
    P1_CONTRACT.parent.mkdir(parents=True, exist_ok=True)
    P1_CONTRACT.write_text(text)
    print(f"CORE070_NAMESPACE_1_P1_CONTRACT_WRITTEN {P1_CONTRACT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
