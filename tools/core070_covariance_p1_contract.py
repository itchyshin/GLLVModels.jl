"""Regenerate the Core070 covariance contracts at gllvmTMB pin P1.

Three P0 files are read and left untouched as history:

  * docs/dev-log/core070/frozen-r070-contract.toml -- the manifest the
    test/parity runner (runparity.jl) validates before any required case runs;
  * docs/dev-log/core070/covariance-batch-contract.json -- the R-only
    formula-grammar batch (tools/core070_covariance_batch.R);
  * docs/dev-log/core070/wave6-conversion-batch-contract.json -- the paired
    R/Julia structured-term batch whose two kernel_latent cases pay
    covariance/COV-KERNEL-FOLDED-UNIQUE and covariance/COV-KERNEL-LATENT.

and three P1 files are written under docs/dev-log/core070/true-parity-latest/:

  * frozen-r070-contract-p1.toml
  * covariance-batch-contract-p1.json
  * wave6-conversion-batch-contract-p1.json

R side: every gllvmTMB byte is read with `git -C $GLLVMTMB_DIR show <P1>:<path>`
(GIT_OPTIONAL_LOCKS=0; the clone is never checked out or edited). The P1 commit
and its archive / NAMESPACE / source-tree hashes come from the shared pin source,
tools/core070_oracle_pins.toml, not from a literal in this file.

What changes, all recorded in each output's regeneration log:

  frozen-r070-contract-p1.toml
    * the provenance header (reference_commit, NAMESPACE blob + sha256,
      R/fit-multi.R and R/families.R blobs, archive and source-tree sha256,
      source inventory path, `source` line) is recomputed at P1;
    * everything else (case-id registries, family rows, obligations, and their
      free-text `R/<file>:<lines> @ b4d5fee64...` citations) is carried verbatim
      from P0. Those citations are NOT re-anchored; they describe where P0 R
      source said something and are history, not P1 evidence.

  covariance-batch-contract-p1.json
    * reference_commit and the four source_pins sha256 are recomputed at P1;
    * cases, expectations and negative controls are carried verbatim: no
      expectation is edited to make a P1 run pass. A P1 failure is recorded as
      a failure by the batch, never absorbed here.

  wave6-conversion-batch-contract-p1.json
    * reference_commit is set to P1; cases, r_call strings, tolerances,
      deferred rows, rejection cases and negative controls are carried
      verbatim. The whole 10-case batch runs, not only its two covariance
      cases, because the runner and verifier check the batch as one unit.

Usage:
  GLLVMTMB_DIR=/path/to/gllvmTMB python3 tools/core070_covariance_p1_contract.py [--check]

--check regenerates in memory and exits nonzero if any tracked P1 contract
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
P0_TOML = ROOT / "docs/dev-log/core070/frozen-r070-contract.toml"
P0_BATCH = ROOT / "docs/dev-log/core070/covariance-batch-contract.json"
OUT_DIR = ROOT / "docs/dev-log/core070/true-parity-latest"
P1_TOML = OUT_DIR / "frozen-r070-contract-p1.toml"
P1_BATCH = OUT_DIR / "covariance-batch-contract-p1.json"
P0_WAVE6 = ROOT / "docs/dev-log/core070/wave6-conversion-batch-contract.json"
P1_WAVE6 = OUT_DIR / "wave6-conversion-batch-contract-p1.json"
PINS_FILE = ROOT / "tools/core070_oracle_pins.toml"
PIN = "P1"
P1_SOURCE_INVENTORY = "docs/dev-log/core070/true-parity-latest/receipts/covariance/oracle/source.json"


def git_show(repo, sha, path):
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    return subprocess.run(["git", "-C", str(repo), "show", f"{sha}:{path}"], check=True,
                          capture_output=True, env=env).stdout


def git_blob(repo, sha, path):
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    return subprocess.run(["git", "-C", str(repo), "rev-parse", f"{sha}:{path}"], check=True,
                          capture_output=True, text=True, env=env).stdout.strip()


def pin_entry():
    table = tomllib.loads(PINS_FILE.read_text())
    return table["P0"], table[PIN]


def replace_line(text, key, value, log):
    pattern = re.compile(rf'^{re.escape(key)} = "([^"]*)"$', re.M)
    m = pattern.search(text)
    if m is None:
        raise SystemExit(f"P0 contract has no top-level `{key}` line")
    log.append({"field": key, "p0": m.group(1), "p1": value})
    return text[:m.start()] + f'{key} = "{value}"' + text[m.end():]


def build_toml(repo):
    p0, p1 = pin_entry()
    sha = p1["reference_commit"]
    text = P0_TOML.read_text()
    head_end = text.index("\nfamily_smoke_case_ids")
    head, rest = text[:head_end], text[head_end:]
    log = []
    ns_bytes = git_show(repo, sha, "NAMESPACE")
    ns_sha = hashlib.sha256(ns_bytes).hexdigest()
    if ns_sha != p1["namespace_sha256"]:
        raise SystemExit(f"NAMESPACE at {sha} hashes to {ns_sha}, pins file says {p1['namespace_sha256']}")
    for key, value in (
        ("reference_commit", sha),
        ("reference_namespace_blob", git_blob(repo, sha, "NAMESPACE")),
        ("reference_namespace_sha256", ns_sha),
        ("reference_fit_multi_blob", git_blob(repo, sha, "R/fit-multi.R")),
        ("reference_families_blob", git_blob(repo, sha, "R/families.R")),
        ("reference_archive_sha256", p1["archive_sha256"]),
        ("reference_source_tree_sha256", p1["source_tree_sha256"]),
        ("reference_source_inventory", P1_SOURCE_INVENTORY),
        ("source", f"git show gllvmTMB:{sha}"),
    ):
        head = replace_line(head, key, value, log)
    carried = len(re.findall(p0["reference_commit"], rest))
    note = [
        "",
        "# P1 regeneration (tools/core070_covariance_p1_contract.py). Only the provenance header",
        "# above was recomputed at P1; every line below is carried verbatim from",
        "# docs/dev-log/core070/frozen-r070-contract.toml. Free-text citations of the form",
        f"# `R/<file>:<lines> @ {p0['reference_commit']}` below ({carried} occurrences) are P0 history,",
        "# not re-anchored at P1. Header changes:",
    ]
    note += [f"#   {row['field']}: {row['p0']} -> {row['p1']}" for row in log]
    return head + "\n".join(note) + rest


def build_batch(repo):
    p0, p1 = pin_entry()
    sha = p1["reference_commit"]
    contract = json.loads(P0_BATCH.read_text())
    if contract["reference_commit"] != p0["reference_commit"]:
        raise SystemExit("P0 covariance batch contract is not pinned at P0")
    log = [{"field": "reference_commit", "p0": contract["reference_commit"], "p1": sha}]
    contract["reference_commit"] = sha
    pins = {}
    for rel, old in contract["source_pins"].items():
        new = hashlib.sha256(git_show(repo, sha, rel)).hexdigest()
        pins[rel] = new
        log.append({"field": f"source_pins[{rel}]", "p0": old, "p1": new, "changed": old != new})
    contract["source_pins"] = pins
    contract["status"] = "REGENERATED_AT_P1_BEFORE_RUN"
    contract["p0_contract"] = "docs/dev-log/core070/covariance-batch-contract.json"
    contract["p0_contract_sha256"] = hashlib.sha256(P0_BATCH.read_bytes()).hexdigest()
    contract["regeneration_log"] = {
        "generator": "tools/core070_covariance_p1_contract.py",
        "changes": log,
        "carried_verbatim": "cases, expected_covstructs, negative_controls, fixture, required_health_checks, "
                            "claim_boundary. No expectation was edited; a P1 mismatch is recorded by the batch as a failure.",
    }
    return json.dumps(contract, indent=2) + "\n"


def build_wave6():
    p0, p1 = pin_entry()
    contract = json.loads(P0_WAVE6.read_text())
    if contract["reference_commit"] != p0["reference_commit"]:
        raise SystemExit("P0 wave6 contract is not pinned at P0")
    contract["reference_commit"] = p1["reference_commit"]
    contract["p0_contract"] = "docs/dev-log/core070/wave6-conversion-batch-contract.json"
    contract["p0_contract_sha256"] = hashlib.sha256(P0_WAVE6.read_bytes()).hexdigest()
    contract["regeneration_log"] = {
        "generator": "tools/core070_covariance_p1_contract.py",
        "changes": [{"field": "reference_commit", "p0": p0["reference_commit"], "p1": p1["reference_commit"]}],
        "carried_verbatim": "status, cases (r_call, term_expr, quantity, tolerance), deferred, rejection_cases, "
                            "negative_controls, fixtures, runner. No tolerance or expectation was edited.",
    }
    return json.dumps(contract, indent=2) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    repo = os.environ.get("GLLVMTMB_DIR")
    if not repo:
        raise SystemExit("set GLLVMTMB_DIR to a gllvmTMB clone that contains the P1 commit")
    outputs = {P1_TOML: build_toml(repo), P1_BATCH: build_batch(repo), P1_WAVE6: build_wave6()}
    tomllib.loads(outputs[P1_TOML])  # the regenerated manifest must still parse
    if args.check:
        stale = [str(p.relative_to(ROOT)) for p, t in outputs.items() if not p.exists() or p.read_text() != t]
        if stale:
            print("STALE " + " ".join(stale))
            sys.exit(1)
        print("CORE070_COVARIANCE_P1_CONTRACTS_CURRENT")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p, t in outputs.items():
        p.write_text(t)
        print(f"wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
