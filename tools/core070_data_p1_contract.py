"""Regenerate the Core070 data and fit-input batch contracts at gllvmTMB pin P1.

In scope: the 28 required data rows and the 6 required fit-input rows the P1
carry scan lists as DANGLING. They are paid by two P0 batches:

  P0 contract (unchanged)            P1 twin (written here)
  data-batch-contract.json           data-batch-contract-p1.json         (28 data rows, wave1)
  fit-input-2-batch-contract.json    fit-input-2-batch-contract-p1.json  (6 fit-input rows, wave4)

The P0 files are read and never written; they stay as history.

What changes, recorded in each output's `regeneration_log`:

  * reference_commit -> P1 (from tools/core070_oracle_pins.toml, not a literal);
  * source_pins: both P0 contracts pin R/<file> sha256 values checked against the
    P0 readback tree. The twin recomputes each from the P1 bytes
    (`git -C $GLLVMTMB_DIR show <P1>:<path>`); at P1 the runners check them under
    the P1 source tree the oracle was built from (the data runner's <source-root>
    argument, CORE070_P1_R_SOURCE_ROOT for fit-input-2);
  * both twins gain `p0_to_p1_r_function_diff`: for each R function the cases
    call (data) or that the fit path is keyed on (fit-input-2), whether its body
    is byte-identical at P0 and P1. This is a record, not a gate;
  * the data twin gains `julia_surface_status_p1_note`: the carried P0
    julia_surface_status leans on Base.kwarg_decl, which cannot see keywords
    forwarded through `kwargs...`; the note says what the surface probe found.

Carried verbatim: status, cases / rows (expression, expected, r_call, julia_call,
acceptance), tolerances, julia_planned_surfaces, needs_new_julia_surface,
negative controls, runner blocks. No case, expectation or tolerance is edited;
a P1 mismatch is recorded by the batch as a failure.

Usage:
  GLLVMTMB_DIR=/path/to/gllvmTMB python3 tools/core070_data_p1_contract.py [--check]

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
GENERATOR = "tools/core070_data_p1_contract.py"

CONTRACTS = ["data", "fit-input-2"]

# R functions each batch's cases route through -> defining file.
FUNCTIONS = {
    "data": [
        ("normalise_weights", "R/weights-shape.R"),
        ("gll_prepare_offset", "R/offset.R"),
        (".gllvmTMB_offset_vec", "R/offset.R"),
        (".gllvmTMB_offset_newdata", "R/offset.R"),
        ("miss_control", "R/gllvmTMB.R"),
    ],
    "fit-input-2": [
        ("gllvmTMB", "R/gllvmTMB.R"),
        ("animal_latent", "R/animal-keyword.R"),
        ("kernel_latent", "R/kernel-keywords.R"),
    ],
}


DATA_SURFACE_NOTE = (
    "Read julia_surface_status (carried verbatim from P0) as a statement about helper-equivalent surfaces only. "
    "Its parenthetical 'Base.kwarg_decl() over gllvm()/fit_gllvm() to rule out an undocumented keyword' does not "
    "hold: both entry points end in kwargs..., so kwarg_decl lists only the dispatcher's own keywords, not what it "
    "forwards. GLLVModels does have fit-time offset= and mask= / missing-in-Y surfaces on the non-Gaussian fitters "
    "(and offset= / mask= on the default Gaussian path), reachable through fit_gllvm and gllvm; it has no weights "
    "surface. What it lacks is the helper layer these cases replay: a formula-offset evaluator, a stored or "
    "predict-time offset accessor, a miss_control constructor, and a normalise_weights adapter. The behavioural "
    "evidence is tools/core070_data_surface_probe.jl and its tracked receipt next to the data batch.")


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


def function_diff(repo, name, p0_sha, p1_sha):
    rows = []
    for fn, path in FUNCTIONS[name]:
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
    new = {}
    for rel, old in contract["source_pins"].items():
        if not rel.startswith("R/"):
            raise SystemExit(f"unexpected source pin path {rel}")
        new[rel] = sha_bytes(git(repo, "show", f"{p1['reference_commit']}:{rel}"))
        p0_bytes = sha_bytes(git(repo, "show", f"{p0['reference_commit']}:{rel}"))
        if p0_bytes != old:
            raise SystemExit(f"{src.name} source pin {rel} does not match the P0 bytes; refuse")
        log.append({"field": f"source_pins[{rel}]", "p0": old, "p1": new[rel], "changed": old != new[rel]})
    contract["source_pins"] = new
    contract["p0_contract"] = str(src.relative_to(ROOT))
    contract["p0_contract_sha256"] = sha_bytes(src.read_bytes())
    contract["p0_to_p1_r_function_diff"] = function_diff(repo, name, p0["reference_commit"], p1["reference_commit"])
    if name == "data":
        # The P0 julia_surface_status is carried verbatim; this adds a P1 reading note, it edits nothing.
        contract["julia_surface_status_p1_note"] = DATA_SURFACE_NOTE
        log.append({"field": "julia_surface_status_p1_note", "added": True,
                    "why": "Base.kwarg_decl on a kwargs... dispatcher is not a keyword census"})
    contract["regeneration_log"] = {
        "generator": GENERATOR,
        "changes": log,
        "carried_verbatim": "status, cases/rows (expression, expected, r_call, julia_call, acceptance), "
                            "tolerances, julia_planned_surfaces, needs_new_julia_surface, negative controls, "
                            "runner blocks. No case, expectation or tolerance was edited; a P1 mismatch is "
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
    outputs = {OUT_DIR / f"{n}-batch-contract-p1.json": build(repo, n) for n in CONTRACTS}
    if args.check:
        stale = [str(p.relative_to(ROOT)) for p, t in outputs.items() if not p.exists() or p.read_text() != t]
        if stale:
            print("STALE " + " ".join(stale))
            sys.exit(1)
        print("CORE070_DATA_P1_CONTRACTS_CURRENT")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p, t in outputs.items():
        p.write_text(t)
        print(f"wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
