"""Regenerate the namespace-2 batch contract at gllvmTMB pin P1.

Reads the frozen P0 contract (docs/dev-log/core070/namespace-2-batch-contract.json,
left untouched as history) and writes the P1 contract
(docs/dev-log/core070/true-parity-latest/namespace-2-batch-contract-p1.json).

Exactly four top-level fields differ from the P0 contract:
  * reference_commit becomes the P1 commit from tools/core070_oracle_pins.toml;
  * source_pins are re-hashed at P1, reading every byte with
    `git -C $GLLVMTMB_DIR show <P1>:<path>` (the clone is never checked out or edited);
  * source_access = "git-show" (new field), so tools/core070_namespace_2_batch.R reads
    the R sources the same way instead of from the P0 readback tree under .unlazy/;
  * regeneration_log (new field) records the three changes above.

Cases, tolerances, buckets and negative controls are carried verbatim; no case is
reclassified and no admission set changes.

Usage:
  GLLVMTMB_DIR=/path/to/gllvmTMB python3 tools/core070_namespace_2_p1_contract.py [--check]

--check regenerates in memory and exits nonzero if the tracked P1 contract differs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
P0_CONTRACT = ROOT / "docs/dev-log/core070/namespace-2-batch-contract.json"
P1_CONTRACT = ROOT / "docs/dev-log/core070/true-parity-latest/namespace-2-batch-contract-p1.json"
PINS_FILE = ROOT / "tools/core070_oracle_pins.toml"
PIN = "P1"


def git_show(gdir, commit, rel):
    return subprocess.run(["git", "-C", gdir, "show", f"{commit}:{rel}"],
                          check=True, capture_output=True).stdout


def build():
    gdir = os.environ.get("GLLVMTMB_DIR")
    if not gdir:
        sys.exit("GLLVMTMB_DIR must name a local gllvmTMB clone (read with git show only)")
    pin = tomllib.loads(PINS_FILE.read_text())[PIN]
    commit = pin["reference_commit"]
    p0 = json.loads(P0_CONTRACT.read_text())
    out = dict(p0)
    out["reference_commit"] = commit
    out["source_access"] = "git-show"
    out["source_pins"] = {rel: hashlib.sha256(git_show(gdir, commit, rel)).hexdigest()
                          for rel in p0["source_pins"]}
    out["regeneration_log"] = {
        "generator": "tools/core070_namespace_2_p1_contract.py",
        "p0_contract": str(P0_CONTRACT.relative_to(ROOT)),
        "p0_contract_sha256": hashlib.sha256(P0_CONTRACT.read_bytes()).hexdigest(),
        "p0_reference_commit": p0["reference_commit"],
        "pin": PIN,
        "gllvmtmb_version": pin["version"],
        "changes": [
            "reference_commit set to the P1 commit from tools/core070_oracle_pins.toml",
            "source_pins re-hashed at P1 with git show (same four files as P0)",
            "source_access = git-show (R sources read from the P1 commit, not the P0 readback tree)",
        ],
        "p0_source_pins": p0["source_pins"],
        "unchanged": "cases, tolerances, buckets (needs_new_julia_surface, reclassify_rows), "
                     "negative controls and fixtures are carried verbatim from P0",
    }
    return json.dumps(out, indent=1, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    text = build()
    if args.check:
        if not P1_CONTRACT.exists() or P1_CONTRACT.read_text() != text:
            sys.exit("namespace-2 P1 contract is stale or missing")
        print("CORE070_NAMESPACE_2_P1_CONTRACT_CURRENT")
    else:
        P1_CONTRACT.write_text(text)
        print("wrote", P1_CONTRACT.relative_to(ROOT))
