"""Regenerate the Core070 postfit batch contracts at gllvmTMB pin P1.

The postfit family's 36 required rows (34 DANGLING + 2 RETIRED in the P1
carry scan) and the 16 DANGLING postfit-policy rows are paid by seven P0
batches. Five of them read a frozen contract JSON, which this tool copies
to a P1 twin under docs/dev-log/core070/true-parity-latest/. The P0 files
are read and never written; they stay as history.

  P0 contract (unchanged)                    P1 twin (written here)
  surface-conversion-batch-contract.json     surface-conversion-batch-contract-p1.json
  wave7-conversion-batch-contract.json       wave7-conversion-batch-contract-p1.json
  wave8-conversion-batch-contract.json       wave8-conversion-batch-contract-p1.json
  postfit-1-batch-contract.json              postfit-1-batch-contract-p1.json
  postfit-policy-batch-contract.json         postfit-policy-batch-contract-p1.json

The other two batches need no new contract: wave6 already has its P1 twin
(tools/core070_covariance_p1_contract.py, PR #567), and the estimand-rebind
batch has no contract file (its four case ids and tolerance live in the
runner and verifier, which read GLLVM_PARITY_PIN).

What changes, recorded in each output's `regeneration_log`:

  * reference_commit -> P1 (from tools/core070_oracle_pins.toml, not a literal);
  * postfit-policy only: the four `source_pins` sha256 are recomputed from the
    P1 bytes (`git -C $GLLVMTMB_DIR show <P1>:<path>`), because the runner
    checks them against the R source tree before it fits anything;
  * postfit-1 only: `executable_batch.r_source_sha256_at_p1` is added next to
    the P0 `r_source_sha256_at_readback`, which is kept (history);
  * every contract gains `p0_to_p1_accessor_diff`: for each R accessor its
    cases call, whether the function body is byte-identical at P0 and P1.
    This is a record, not a gate; a changed body is not itself a failure.

Carried verbatim: status, cases (r_call, quantity, tolerance, check,
comparand), deferred rows, rejection cases, negative controls, fixtures,
runner blocks. No case, expectation or tolerance is edited; a P1 mismatch is
recorded by the batch as a failure, never absorbed here.

Usage:
  GLLVMTMB_DIR=/path/to/gllvmTMB python3 tools/core070_postfit_p1_contract.py [--check]

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
P0_DIR = ROOT / "docs/dev-log/core070"
OUT_DIR = P0_DIR / "true-parity-latest"
PINS_FILE = ROOT / "tools/core070_oracle_pins.toml"
GENERATOR = "tools/core070_postfit_p1_contract.py"

CONTRACTS = ["surface-conversion", "wave7-conversion", "wave8-conversion", "postfit-1", "postfit-policy"]

# R accessor(s) each batch's postfit cases call -> the R file that defines them.
ACCESSORS = {
    "surface-conversion": [
        ("getLoadings", "R/output-methods.R"), ("extract_loadings", "R/output-methods.R"),
        ("getLV", "R/output-methods.R"), ("extract_Sigma", "R/extract-sigma.R"),
        ("extract_Sigma_table", "R/extract-sigma-table.R"),
        ("extract_residual_cov", "R/output-methods.R"), ("extract_residual_cor", "R/output-methods.R"),
        ("getResidualCov", "R/output-methods.R"), ("getResidualCor", "R/output-methods.R"),
        ("extract_ordination", "R/extractors.R"), ("extract_ICC_site", "R/extractors.R"),
        ("extract_repeatability", "R/extract-repeatability.R"), ("extract_cutpoints", "R/extract-cutpoints.R"),
    ],
    "wave7-conversion": [
        ("check_auto_residual", "R/check-auto-residual.R"), ("sanity_multi", "R/methods-gllvmTMB.R"),
        ("compare_loadings", "R/rotate-loadings.R"), ("fitted.gllvmTMB_multi", "R/methods-gllvmTMB.R"),
        ("predict.gllvmTMB_multi", "R/methods-gllvmTMB.R"),
        ("residuals.gllvmTMB_multi", "R/predictive-diagnostics.R"),
    ],
    "wave8-conversion": [
        ("deviance.gllvmTMB_multi", "R/methods-gllvmTMB.R"), ("tidy.gllvmTMB_multi", "R/methods-gllvmTMB.R"),
        ("summary.gllvmTMB_multi", "R/methods-gllvmTMB.R"), ("rotate_loadings", "R/rotate-loadings.R"),
        ("extract_rotated_loadings_table", "R/rotate-loadings.R"),
        ("predict_missing", "R/methods-gllvmTMB.R"), ("simulate_unit_trait", "R/simulate-unit-trait.R"),
    ],
    "postfit-1": [("coef.gllvmTMB_multi", "R/vcov-coef.R")],
    "postfit-policy": [
        ("coef.gllvmTMB_multi", "R/vcov-coef.R"), ("confint.gllvmTMB_multi", "R/z-confint-gllvmTMB.R"),
        ("fitted.gllvmTMB_multi", "R/methods-gllvmTMB.R"), ("logLik.gllvmTMB_multi", "R/methods-gllvmTMB.R"),
        ("nobs.gllvmTMB_multi", "R/methods-gllvmTMB.R"), ("predict.gllvmTMB_multi", "R/methods-gllvmTMB.R"),
        ("residuals.gllvmTMB_multi", "R/predictive-diagnostics.R"),
        ("simulate.gllvmTMB_multi", "R/methods-gllvmTMB.R"),
    ],
}


def git(repo, *argv):
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    return subprocess.run(["git", "-C", str(repo), *argv], check=True, capture_output=True, env=env).stdout


def pins():
    table = tomllib.loads(PINS_FILE.read_text())
    return table["P0"], table["P1"]


def function_body(text, fn):
    m = re.search(rf"^`?{re.escape(fn)}`?\s*(<-|=)\s*function", text, re.M)
    if m is None:
        return None
    out = []
    for i, line in enumerate(text[m.start():].split("\n")):
        out.append(line)
        if i > 0 and line.startswith("}"):
            break
    return "\n".join(out)


def accessor_diff(repo, name, p0_sha, p1_sha):
    rows = []
    for fn, path in ACCESSORS[name]:
        b0 = function_body(git(repo, "show", f"{p0_sha}:{path}").decode(), fn)
        b1 = function_body(git(repo, "show", f"{p1_sha}:{path}").decode(), fn)
        if b0 is None or b1 is None:
            raise SystemExit(f"{fn} not found in {path} at {'P0' if b0 is None else 'P1'}")
        rows.append({"function": fn, "file": path,
                     "body_identical_p0_p1": b0 == b1,
                     "body_sha256_p0": hashlib.sha256(b0.encode()).hexdigest(),
                     "body_sha256_p1": hashlib.sha256(b1.encode()).hexdigest()})
    return rows


def build(repo, name):
    p0, p1 = pins()
    src = P0_DIR / f"{name}-batch-contract.json"
    contract = json.loads(src.read_text())
    if contract["reference_commit"] != p0["reference_commit"]:
        raise SystemExit(f"{src.name} is not pinned at P0")
    log = [{"field": "reference_commit", "p0": contract["reference_commit"], "p1": p1["reference_commit"]}]
    contract["reference_commit"] = p1["reference_commit"]
    if "source_pins" in contract:
        new = {}
        for rel, old in contract["source_pins"].items():
            new[rel] = hashlib.sha256(git(repo, "show", f"{p1['reference_commit']}:{rel}")).hexdigest()
            log.append({"field": f"source_pins[{rel}]", "p0": old, "p1": new[rel], "changed": old != new[rel]})
        contract["source_pins"] = new
    if name == "postfit-1":
        eb = contract["executable_batch"]
        eb["r_source_sha256_at_p1"] = hashlib.sha256(
            git(repo, "show", f"{p1['reference_commit']}:{eb['r_source_file']}")).hexdigest()
        log.append({"field": "executable_batch.r_source_sha256_at_p1 (added)",
                    "p0_r_source_sha256_at_readback": eb["r_source_sha256_at_readback"],
                    "p1": eb["r_source_sha256_at_p1"],
                    "changed": eb["r_source_sha256_at_readback"] != eb["r_source_sha256_at_p1"]})
    contract["p0_contract"] = str(src.relative_to(ROOT))
    contract["p0_contract_sha256"] = hashlib.sha256(src.read_bytes()).hexdigest()
    contract["p0_to_p1_accessor_diff"] = accessor_diff(repo, name, p0["reference_commit"], p1["reference_commit"])
    contract["regeneration_log"] = {
        "generator": GENERATOR,
        "changes": log,
        "carried_verbatim": "status, cases (r_call, quantity, tolerance, check, comparand), deferred, "
                            "rejection_cases, negative_controls, fixtures, runner. No case, expectation or "
                            "tolerance was edited; a P1 mismatch is recorded by the batch as a failure.",
    }
    return json.dumps(contract, indent=2) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    repo = os.environ.get("GLLVMTMB_DIR")
    if not repo:
        raise SystemExit("set GLLVMTMB_DIR to a gllvmTMB clone that contains the P1 commit")
    outputs = {OUT_DIR / f"{n}-batch-contract-p1.json": build(repo, n) for n in CONTRACTS}
    if args.check:
        stale = [str(p.relative_to(ROOT)) for p, t in outputs.items() if not p.exists() or p.read_text() != t]
        if stale:
            print("STALE " + " ".join(stale))
            sys.exit(1)
        print("CORE070_POSTFIT_P1_CONTRACTS_CURRENT")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p, t in outputs.items():
        p.write_text(t)
        print(f"wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
