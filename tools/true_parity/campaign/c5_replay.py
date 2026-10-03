#!/usr/bin/env python3
"""Replay the four merged PR #593 grouping receipts (C5) at the current commit.

Signed itchyshin/GLLVModels.jl#684 item 4. Plan section 1.4: the receipts for unit, unit_obs, cluster
and cluster2 already exist; the work is to turn them into rows, after re-verifying their hashes and
replaying both engines. This script does exactly that and nothing else:

  1. re-hashes the four fixture CSVs against fixtures/manifest.json and against each receipt's
     recorded fixture sha256, and re-hashes every file the receipt lists under `read_from`;
  2. re-runs the R side (tools/core070_grouping_p1_r.R, the P1 oracle library, refuses unless the
     oracle's build.json names P1) and the Julia side (tools/core070_grouping_p1_julia.jl, at the
     current commit) into a scratch directory, leaving the tracked receipts untouched;
  3. writes one replay record per level, c5_replay_<level>.json, for write_receipts.py:
     fixture hashes verified, name parity re-measured (keyword accepted, misspelt keyword rejected on
     both engines), the replayed R and Julia logLik, their difference, and their distance from the
     logLik values recorded in the receipt.

Usage (from the repo root, with the same thread caps as the receipts' own run):
  OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=4 python3 tools/true_parity/campaign/c5_replay.py <oracle_build_dir> <out_dir>
"""
import hashlib, json, os, subprocess, sys, tomllib, time, platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REC = ROOT / "docs/dev-log/core070/true-parity-latest/receipts/grouping"
LEVELS = ("unit", "unit_obs", "cluster", "cluster2")
JULIA = os.environ.get("JULIA", os.path.expanduser("~/.juliaup/bin/julia"))


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    oracle, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((REC / "fixtures/manifest.json").read_text())
    verified = {}
    for lvl in LEVELS:
        rc = json.loads((REC / f"{lvl}.json").read_text())
        want = manifest["files"][f"{lvl}.csv"]
        ok = sha(REC / f"fixtures/{lvl}.csv") == want == rc["fixture"]["sha256"]
        for rel, h in rc["read_from"].items():
            ok = ok and sha(ROOT / rel) == h
        verified[lvl] = ok
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    t0 = time.time()
    r_json, j_toml = out / "replay-r.json", out / "replay-julia.toml"
    env = dict(os.environ)
    subprocess.run(["Rscript", "tools/core070_grouping_p1_r.R", str(oracle), str(r_json)], cwd=ROOT, check=True, env=env)
    t_r = time.time() - t0; t1 = time.time()
    subprocess.run([JULIA, "--project=.", "tools/core070_grouping_p1_julia.jl", str(j_toml)], cwd=ROOT, check=True, env=env)
    t_j = time.time() - t1
    r = json.loads(r_json.read_text()); j = tomllib.loads(j_toml.read_text())
    for lvl in LEVELS:
        rc = json.loads((REC / f"{lvl}.json").read_text())
        rl, jl = r["levels"][lvl], j["levels"][lvl]
        ll = [c for c in rc["comparison"]["cases"] if c["case_id"].endswith("/logLik")][0]
        name_ok = bool(rl.get("keyword_accepted") and jl.get("keyword_accepted") and rl["negative_control"]["rejected"]
                       and jl["negative_control"]["rejected"])
        rec = {
            "level": lvl, "glvmodels_commit_of_replay": head, "oracle_reference_commit": r.get("reference_commit"),
            "gllvmTMB_version": r.get("gllvmTMB_version"), "R_version": r.get("R_version"), "julia_version": str(j.get("julia_version", "")),
            "host": platform.node(), "fixture_sha256_verified": verified[lvl],
            "name_parity_pass": name_ok,
            "r_convergence": rl.get("convergence"), "julia_converged": jl.get("converged"),
            "replay_r_logLik": rl["logLik"], "replay_julia_logLik": jl["logLik"],
            "replay_abs_logLik_r_vs_julia": abs(rl["logLik"] - jl["logLik"]),
            "receipt_r_logLik": ll["r_value"], "receipt_julia_logLik": ll["julia_value"],
            "max_abs_logLik_vs_receipt": max(abs(rl["logLik"] - ll["r_value"]), abs(jl["logLik"] - ll["julia_value"])),
            "wall_seconds": {"r_all_levels": t_r, "julia_all_levels_including_compile": t_j},
            "threads": {"OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"), "JULIA_NUM_THREADS": os.environ.get("JULIA_NUM_THREADS")},
        }
        (out / f"c5_replay_{lvl}.json").write_text(json.dumps(rec, indent=1) + "\n")
        print(lvl, "hashes", verified[lvl], "name parity", name_ok, "dLL(R,J)", rec["replay_abs_logLik_r_vs_julia"], "vs receipt", rec["max_abs_logLik_vs_receipt"])


if __name__ == "__main__":
    main()
