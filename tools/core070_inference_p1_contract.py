"""Regenerate the Core070 inference batch contracts at gllvmTMB pin P1.

The inference family's 63 required rows (all DANGLING in the P1 carry scan)
are paid by three P0 batches:

  P0 contract (unchanged)                  P1 twin (written here)
  inference-batch-contract.json            inference-batch-contract-p1.json            (45 rows, wave2)
  inference-remainder-batch-contract.json  inference-remainder-batch-contract-p1.json  (14 rows, wave4)
  surface-conversion-batch-contract.json   (twin already written by tools/core070_postfit_p1_contract.py;
                                            4 rows, CI-ROUTE-008..011)

The P0 files are read and never written; they stay as history.

What changes, recorded in each output's `regeneration_log`:

  * reference_commit -> P1 (from tools/core070_oracle_pins.toml, not a literal);
  * inference-batch: the P0 R comparand is a retained run of the frozen-source
    route probe (docs/dev-log/core070/inference-routing-subset.json, which
    points into .unlazy/ and does not exist at P1). The twin replaces
    `r_route_comparand.pins` with the fixture and the P1 probe
    (tools/core070_inference_routes_p1.R; one added parsed file, R/temporal.R)
    and adds `p1_r_source_pins`, the P1 sha256 of every R file that probe
    parses, so the runner re-runs the probe live at P1 and checks its inputs;
  * inference-remainder: `source_pins` re-keyed from the P0 readback path
    (.unlazy/core070-aghq/oracle-source/readback/R/<file>) to R/<file> and
    recomputed from the P1 bytes (`git -C $GLLVMTMB_DIR show <P1>:<path>`);
    at P1 the runner checks them under CORE070_P1_R_SOURCE_ROOT;
  * both twins gain `p0_to_p1_r_function_diff`: for each R function the cases
    route through, whether the body is byte-identical at P0 and P1. This is a
    record, not a gate.

Carried verbatim: status, rows / cases (julia_call, expected_route_tag, check,
r_call, tolerance), buckets, needs_new_julia_surface, spec-defect rows,
negative controls, fixtures, runner blocks. No case, expectation or tolerance
is edited; a P1 mismatch is recorded by the batch as a failure.

Usage:
  GLLVMTMB_DIR=/path/to/gllvmTMB python3 tools/core070_inference_p1_contract.py [--check]

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
GENERATOR = "tools/core070_inference_p1_contract.py"

CONTRACTS = ["inference", "inference-remainder"]
P0_READBACK = ".unlazy/core070-aghq/oracle-source/readback/"
FIXTURE = "test/parity/fixtures/core070_inference_routes.tsv"
P1_PROBE = "tools/core070_inference_routes_p1.R"
P1_PROBE_FILES = ["R/mspl.R", "R/fit-multi.R", "R/z-confint-gllvmTMB.R", "R/temporal.R"]

# R functions the cases route through -> defining file. Same set for both
# batches: the top-level dispatcher plus each estimand's internal helper.
FUNCTIONS = [
    ("confint.gllvmTMB_multi", "R/z-confint-gllvmTMB.R"),
    (".confint_lambda", "R/z-confint-gllvmTMB.R"),
    (".confint_icc", "R/z-confint-gllvmTMB.R"),
    (".confint_phylo_signal", "R/z-confint-gllvmTMB.R"),
    (".confint_communality", "R/z-confint-gllvmTMB.R"),
    (".confint_rho", "R/z-confint-gllvmTMB.R"),
    (".confint_proportion", "R/z-confint-gllvmTMB.R"),
    (".confint_sigma", "R/z-confint-gllvmTMB.R"),
    (".gllvmTMB_mspl_assert_inference", "R/mspl.R"),
    (".gllvmTMB_require_unweighted_inference", "R/fit-multi.R"),
]


def git(repo, *argv):
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    return subprocess.run(["git", "-C", str(repo), *argv], check=True, capture_output=True, env=env).stdout


def pins():
    table = tomllib.loads(PINS_FILE.read_text())
    return table["P0"], table["P1"]


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


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


def function_diff(repo, p0_sha, p1_sha):
    rows = []
    for fn, path in FUNCTIONS:
        b0 = function_body(git(repo, "show", f"{p0_sha}:{path}").decode(), fn)
        b1 = function_body(git(repo, "show", f"{p1_sha}:{path}").decode(), fn)
        if b0 is None or b1 is None:
            raise SystemExit(f"{fn} not found in {path} at {'P0' if b0 is None else 'P1'}")
        rows.append({"function": fn, "file": path,
                     "body_identical_p0_p1": b0 == b1,
                     "body_sha256_p0": sha_bytes(b0.encode()),
                     "body_sha256_p1": sha_bytes(b1.encode())})
    return rows


def build(repo, name):
    p0, p1 = pins()
    src = P0_DIR / f"{name}-batch-contract.json"
    contract = json.loads(src.read_text())
    if contract["reference_commit"] != p0["reference_commit"]:
        raise SystemExit(f"{src.name} is not pinned at P0")
    log = [{"field": "reference_commit", "p0": contract["reference_commit"], "p1": p1["reference_commit"]}]
    contract["reference_commit"] = p1["reference_commit"]
    if name == "inference":
        comp = contract["r_route_comparand"]
        old_pins = comp["pins"]
        new_pins = {FIXTURE: sha_bytes((ROOT / FIXTURE).read_bytes()),
                    P1_PROBE: sha_bytes((ROOT / P1_PROBE).read_bytes())}
        if new_pins[FIXTURE] != old_pins[FIXTURE]:
            raise SystemExit("route fixture changed since P0; cases are carried verbatim, refuse")
        src_pins = {rel: sha_bytes(git(repo, "show", f"{p1['reference_commit']}:{rel}")) for rel in P1_PROBE_FILES}
        comp["p0_pins"] = old_pins
        comp["pins"] = new_pins
        comp["p1_probe"] = {"script": P1_PROBE, "fixture": FIXTURE,
                            "parsed_files": P1_PROBE_FILES,
                            "differs_from_p0_probe": "adds R/temporal.R to the parsed file list: at P1 "
                            "confint.gllvmTMB_multi first calls .temporal_assert_no_iid_inference(), "
                            "defined there; with the P0 file list all 98 probe rows error "
                            "'could not find function'."}
        comp["p1_r_source_pins"] = src_pins
        comp["p1_note"] = ("At P1 the runner re-runs the route probe live against CORE070_P1_R_SOURCE_ROOT "
                           "(the P1 source tree the oracle was built from) instead of reading the retained "
                           "P0 run named by docs/dev-log/core070/inference-routing-subset.json.")
        log.append({"field": "r_route_comparand.pins", "p0": old_pins, "p1": new_pins})
        log.append({"field": "r_route_comparand.p1_r_source_pins (added)", "p1": src_pins})
    if name == "inference-remainder":
        new = {}
        for rel, old in contract["source_pins"].items():
            if not rel.startswith(P0_READBACK):
                raise SystemExit(f"unexpected source pin path {rel}")
            key = rel[len(P0_READBACK):]
            new[key] = sha_bytes(git(repo, "show", f"{p1['reference_commit']}:{key}"))
            log.append({"field": f"source_pins[{rel}] -> source_pins[{key}]", "p0": old, "p1": new[key],
                        "changed": old != new[key]})
        contract["source_pins"] = new
    contract["p0_contract"] = str(src.relative_to(ROOT))
    contract["p0_contract_sha256"] = sha_bytes(src.read_bytes())
    contract["p0_to_p1_r_function_diff"] = function_diff(repo, p0["reference_commit"], p1["reference_commit"])
    contract["regeneration_log"] = {
        "generator": GENERATOR,
        "changes": log,
        "carried_verbatim": "status, rows/cases (julia_call, expected_route_tag, r_call, check, tolerance), "
                            "buckets, needs_new_julia_surface, spec-defect rows, negative_controls, fixtures, "
                            "runner. No case, expectation or tolerance was edited; a P1 mismatch is recorded "
                            "by the batch as a failure.",
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
        print("CORE070_INFERENCE_P1_CONTRACTS_CURRENT")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p, t in outputs.items():
        p.write_text(t)
        print(f"wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
