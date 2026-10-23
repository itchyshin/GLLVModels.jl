#!/usr/bin/env python3
"""Paired Gaussian R-vs-Julia receipts for the four grouping levels at gllvmTMB pin P1.

Levels: unit, unit_obs, cluster, cluster2. For each level the receipt records
  (a) name parity measured by calling: the keyword is passed to R's gllvmTMB()
      and to Julia's fit_gllvm() in the paired fit (a successful fit of the
      requested term is the positive result), a misspelt keyword is rejected by
      both calls (negative control), and moving one observation to another group
      of the level changes logLik on both engines (membership control: the
      keyword's labels are used, not accepted and ignored);
  (b) a paired Gaussian ML fit of the same literal, sha256-guarded fixture, fitted
      independently by each engine (Julia uses its default start, never R's
      coordinates), comparing logLik, the two trait intercepts, the level's
      diagonal trait covariance and sigma_eps at declared tolerances; the
      membership-control refits are paired too.

These are receipts only. They create no case-map or scoreboard row and classify
nothing: the row naming is an open maintainer decision (Packet 2, row A3).

Usage (from the repo root):
  python3 tools/core070_grouping_p1_receipts.py run [--oracle DIR]
      Refuses unless HEAD is clean and the runners and fixtures are tracked.
      Writes run/run-commit.json, runs both engines, the batch verifier, and
      writes one receipt per level.
  python3 tools/core070_grouping_p1_receipts.py --check
      Re-checks the fixtures against the generator, re-derives every receipt
      from the tracked raw outputs, re-hashes read_from, and checks that each
      receipt's glvmodels_commit is the clean run commit, an ancestor of HEAD,
      with the runners unchanged since.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REC_REL = "docs/dev-log/core070/true-parity-latest/receipts/grouping"
REC = ROOT / REC_REL
RUN_REL = f"{REC_REL}/run"
FIX_REL = f"{REC_REL}/fixtures"
P1_SHA = "9539352f66f2db2cc26b1c393e67212a359b60c9"
DEFAULT_ORACLE = "/Users/z3437171/local-scratch/a3cov-oracle/build"
JULIA = os.path.expanduser("~/.juliaup/bin/julialauncher")
LEVELS = ("unit", "unit_obs", "cluster", "cluster2")
RUNNERS = ("tools/core070_grouping_p1_fixtures.py", "tools/core070_grouping_p1_r.R",
           "tools/core070_grouping_p1_julia.jl", "tools/core070_grouping_p1_receipts.py")
FIXTURES = tuple(f"{FIX_REL}/{n}" for n in ("manifest.json", *(f"{lvl}.csv" for lvl in LEVELS)))
RAW = tuple(f"{RUN_REL}/{n}" for n in ("run-commit.json", "r-results.json", "julia-results.toml", "verify.txt"))

# Declared before comparison. logLik: 1e-6 absolute. Parameters: 1e-5 absolute,
# the optimizer-convergence scale (R's final max |gradient| is up to ~1e-4 here).
TOL = {"logLik": 1e-6, "beta": 1e-5, "Sigma_diag": 1e-5, "sigma_eps": 1e-5, "membership_control_logLik": 1e-6}
TOL_RULE = {
    "logLik": "1e-6 absolute on the marginal log-likelihood",
    "beta": "1e-5 absolute per trait intercept (optimizer-convergence scale)",
    "Sigma_diag": "1e-5 absolute per diagonal variance of the level's trait covariance (optimizer-convergence scale)",
    "sigma_eps": "1e-5 absolute on the residual SD (optimizer-convergence scale)",
    "membership_control_logLik": "1e-6 absolute on the refit after moving observation 1 to another group",
}
MEMBERSHIP_MIN_SHIFT = 1e-3
NOT_COVERED = [
    "Gaussian only; non-Gaussian pairing of grouping levels is out of scope here.",
    "Diagonal (indep) trait covariance only; latent (rank-reduced) and full (dep) forms are not paired.",
    "One grouping term per fit; joint terms (e.g. unit + unit_obs, cluster + cluster2) are not paired.",
    "Two traits, one small toy fixture per level; no realistic-size, recovery, interval or coverage evidence.",
    "Point estimates and logLik only; no standard errors, confidence intervals or REML.",
    "Direct engines only (R TMB engine vs Julia fit_gllvm); the engine = 'julia' R bridge is not exercised.",
]
ROW_STATUS = ("receipt only: no case-map or scoreboard row is created and nothing is classified; the row naming "
              "(GRP- versus RD- prefix, Packet 2 row A3) is an open maintainer decision")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def git(*argv, check=True):
    return subprocess.run(["git", "-C", str(ROOT), *argv], check=check, capture_output=True, text=True)


def read_from(*rels):
    return {rel: sha(ROOT / rel) for rel in rels}


def dirty_paths():
    own = f"{REC_REL}/"
    fixtures = f"{FIX_REL}/"
    out = []
    for line in git("status", "--porcelain", "--untracked-files=no").stdout.splitlines():
        path = line[3:]
        if path.startswith(own) and not path.startswith(fixtures):
            continue
        out.append(path)
    return out


# ---------------------------------------------------------------------------
# Batch verifier: health of the run, independent of the paired numbers.
# ---------------------------------------------------------------------------
def verify(r, j, manifest):
    checks = []
    checks.append(("R pinned at P1", r.get("pin") == "P1" and r.get("reference_commit") == P1_SHA))
    checks.append(("R gllvmTMB version 0.7.1 from the oracle library",
                   r.get("gllvmTMB_version") == "0.7.1" and "a3cov-oracle/build/library" in r.get("loaded_path", "")))
    for lvl in LEVELS:
        rl, jl = r["levels"].get(lvl, {}), j["levels"].get(lvl, {})
        want = manifest["files"][f"{lvl}.csv"]
        checks.append((f"{lvl}: both engines read the manifest fixture",
                       rl.get("fixture_sha256") == want and jl.get("fixture_sha256") == want))
        checks.append((f"{lvl}: R accepted the keyword and converged (code 0)",
                       rl.get("keyword_accepted") is True and rl.get("convergence") == 0))
        checks.append((f"{lvl}: Julia accepted the keyword, fitted term {lvl!r}, converged",
                       jl.get("keyword_accepted") is True and jl.get("term_names") == [lvl]
                       and jl.get("converged") is True))
        checks.append((f"{lvl}: misspelt keyword rejected by both calls",
                       (rl.get("negative_control") or {}).get("rejected") is True
                       and (jl.get("negative_control") or {}).get("rejected") is True))
        rm, jm = rl.get("membership_control") or {}, jl.get("membership_control") or {}
        checks.append((f"{lvl}: membership control moves logLik by > {MEMBERSHIP_MIN_SHIFT:g} on both engines",
                       "logLik" in rm and "logLik" in jm
                       and abs(rm["logLik"] - rl.get("logLik", rm["logLik"])) > MEMBERSHIP_MIN_SHIFT
                       and abs(jm["logLik"] - jl.get("logLik", jm["logLik"])) > MEMBERSHIP_MIN_SHIFT
                       and rm.get("to_group") == jm.get("to_group")))
    lines = [f"{'PASS' if ok else 'FAIL'}  {name}" for name, ok in checks]
    status = "PASS" if all(ok for _, ok in checks) else "FAIL"
    return status, "\n".join([f"grouping P1 batch verifier: {status}", *lines]) + "\n"


# ---------------------------------------------------------------------------
# Derivation: one receipt per level from the tracked raw outputs only.
# ---------------------------------------------------------------------------
def entry(lvl, quantity, rv, jv):
    rl = rv if isinstance(rv, list) else [rv]
    jl = jv if isinstance(jv, list) else [jv]
    if len(rl) != len(jl):
        raise SystemExit(f"{lvl} {quantity}: R length {len(rl)} != Julia length {len(jl)}")
    diff = max(abs(a - b) for a, b in zip(rl, jl))
    tol = TOL[quantity]
    return {"case_id": f"grouping-p1/{lvl}/{quantity}", "quantity": quantity, "r_value": rv, "julia_value": jv,
            "max_abs_diff": diff, "tolerance": tol, "tolerance_rule": TOL_RULE[quantity], "n_values": len(rl),
            "within_tolerance": diff <= tol, "diff_source": "recomputed from the saved R and Julia values"}


def derive(lvl, r, j, run_commit, verifier_status):
    rl, jl = r["levels"][lvl], j["levels"][lvl]
    cases = [entry(lvl, "logLik", rl["logLik"], jl["logLik"]),
             entry(lvl, "beta", rl["beta"], jl["beta"]),
             entry(lvl, "Sigma_diag", rl["Sigma_diag"], jl["Sigma_diag"]),
             entry(lvl, "sigma_eps", rl["sigma_eps"], jl["sigma_eps"]),
             entry(lvl, "membership_control_logLik", rl["membership_control"]["logLik"],
                   jl["membership_control"]["logLik"])]
    for c in cases:
        if c["tolerance"] <= 0:
            raise SystemExit(f"{c['case_id']}: tolerance must be > 0")
    name_ok = (rl["keyword_accepted"] and jl["keyword_accepted"] and rl["negative_control"]["rejected"]
               and jl["negative_control"]["rejected"])
    within = all(c["within_tolerance"] for c in cases)
    verdict = "PASS" if name_ok and within and verifier_status == "PASS" else "FAIL"
    return {
        "level": lvl, "pin": "P1", "reference_commit": P1_SHA, "glvmodels_commit": run_commit,
        "evidence_kind": "name_parity_by_call_plus_paired_gaussian_fit", "verdict": verdict,
        "row_status": ROW_STATUS,
        "name_parity": {
            "keyword": lvl, "verdict": "PASS" if name_ok else "FAIL",
            "method": "measured by calling each engine with the keyword (the paired fit) and with a misspelt "
                      "keyword (negative control); signatures were not read",
            "r": {"function": "gllvmTMB::gllvmTMB", "keyword_accepted": rl["keyword_accepted"], "call": rl["call"],
                  "negative_control": rl["negative_control"]},
            "julia": {"function": "GLLVModels.fit_gllvm", "keyword_accepted": jl["keyword_accepted"],
                      "call": jl["call"], "fit_type": jl["fit_type"], "term_names": jl["term_names"],
                      "negative_control": jl["negative_control"]},
            "membership_control": {"r": rl["membership_control"], "julia": jl["membership_control"],
                                   "rule": "observation 1 moved to the group of the first later observation "
                                           "whose group differs; refit on each engine"},
        },
        "fixture": {"path": rl["fixture"], "sha256": rl["fixture_sha256"], "n_long_rows": rl["n_rows"],
                    "n_groups": rl["n_groups"], "generator": RUNNERS[0]},
        "fits": {
            "model": "value_ti = beta_t + u_{t,g(i)} + eps_ti, u_{t,g} ~ N(0, sd_t^2) diagonal (indep), "
                     "eps ~ N(0, sigma_eps^2), ML",
            "r": {"call": rl["call"], "convergence": rl["convergence"],
                  "optimizer_message": rl["optimizer_message"], "max_abs_gradient": rl["max_abs_gradient"],
                  "fit_seconds": rl["fit_seconds"]},
            "julia": {"call": jl["call"], "converged": jl["converged"], "iterations": jl["iterations"],
                      "fit_seconds": jl["fit_seconds"], "start": "fitter default (independent of R)"},
            "structural_zero": {"Sigma_offdiag_r": rl["Sigma_offdiag"], "Sigma_offdiag_julia": jl["Sigma_offdiag"],
                                "note": "diagonal by construction on both engines; recorded, not compared"},
        },
        "comparison": {"pin": "P1", "cases": cases},
        "batch_verifier": {"tool": RUNNERS[3], "status": verifier_status, "log": f"{RUN_REL}/verify.txt"},
        "not_covered": NOT_COVERED,
        "read_from": read_from(*RAW, *FIXTURES),
    }


def derive_all():
    r = load(ROOT / RUN_REL / "r-results.json")
    j = tomllib.loads((ROOT / RUN_REL / "julia-results.toml").read_text())
    manifest = load(ROOT / FIX_REL / "manifest.json")
    rc = load(ROOT / RUN_REL / "run-commit.json")
    status, text = verify(r, j, manifest)
    return {lvl: derive(lvl, r, j, rc["glvmodels_commit"], status) for lvl in LEVELS}, status, text


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def run(oracle):
    for rel in (*RUNNERS, *FIXTURES):
        if git("ls-files", "--error-unmatch", rel, check=False).returncode != 0:
            raise SystemExit(f"{rel} is not tracked; commit the runners and fixtures before running")
    dirty = dirty_paths()
    head = git("rev-parse", "HEAD").stdout.strip()
    if dirty:
        raise SystemExit(f"refusing to run: tree is dirty at {head}: {dirty}")
    subprocess.run([sys.executable, str(ROOT / RUNNERS[0]), "--check"], check=True, cwd=ROOT)
    run_dir = ROOT / RUN_REL
    run_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", JULIA_NUM_THREADS="4")
    t0 = time.time()
    with open(run_dir / "r.log", "w") as log:
        subprocess.run(["Rscript", RUNNERS[1], oracle, f"{RUN_REL}/r-results.json"], check=True, cwd=ROOT,
                       env=env, stdout=log, stderr=subprocess.STDOUT)
    t_r = time.time() - t0
    with open(run_dir / "julia.log", "w") as log:
        subprocess.run([JULIA, "--project=.", RUNNERS[2], f"{RUN_REL}/julia-results.toml"], check=True, cwd=ROOT,
                       env=env, stdout=log, stderr=subprocess.STDOUT)
    t_j = time.time() - t0 - t_r
    write_json(run_dir / "run-commit.json", {
        "glvmodels_commit": head, "dirty": [], "host": platform.node(), "oracle_build_dir": oracle,
        "oracle_build_json_sha256": sha(Path(oracle) / "build.json"),
        "threads": {"OPENBLAS_NUM_THREADS": 1, "JULIA_NUM_THREADS": 4},
        "wall_seconds": {"r": round(t_r, 1), "julia_including_compile": round(t_j, 1)},
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    r = load(run_dir / "r-results.json")
    j = tomllib.loads((run_dir / "julia-results.toml").read_text())
    status, text = verify(r, j, load(ROOT / FIX_REL / "manifest.json"))
    (run_dir / "verify.txt").write_text(text)
    receipts, _, _ = derive_all()
    for lvl, rec in receipts.items():
        write_json(REC / f"{lvl}.json", rec)
    print(text, end="")
    for lvl, rec in receipts.items():
        worst = ", ".join(f"{c['quantity']} {c['max_abs_diff']:.2e}" for c in rec["comparison"]["cases"])
        print(f"{lvl}: {rec['verdict']}  ({worst})")


def check():
    problems = []
    fx = subprocess.run([sys.executable, str(ROOT / RUNNERS[0]), "--check"], cwd=ROOT, capture_output=True, text=True)
    if fx.returncode != 0:
        problems.append(f"fixtures: {fx.stdout.strip()} {fx.stderr.strip()}")
    rc = load(ROOT / RUN_REL / "run-commit.json")
    commit = rc.get("glvmodels_commit")
    if rc.get("dirty") != []:
        problems.append(f"run-commit.json records a dirty tree: {rc.get('dirty')}")
    if git("merge-base", "--is-ancestor", commit or "", "HEAD", check=False).returncode != 0:
        problems.append(f"run commit {commit} is not an ancestor of HEAD")
    else:
        for rel in (*RUNNERS[:3], *FIXTURES):
            if git("diff", "--quiet", commit, "HEAD", "--", rel, check=False).returncode != 0:
                problems.append(f"{rel} changed since the run commit {commit}; re-run")
    fresh, status, text = derive_all()
    if (ROOT / RUN_REL / "verify.txt").read_text() != text:
        problems.append("verify.txt differs from the re-derived batch verifier output")
    for lvl, rec in fresh.items():
        path = REC / f"{lvl}.json"
        if not path.is_file():
            problems.append(f"{lvl}: receipt missing")
            continue
        stored = load(path)
        if stored != json.loads(json.dumps(rec)):
            problems.append(f"{lvl}: receipt differs from the re-derivation")
        if stored.get("glvmodels_commit") != commit:
            problems.append(f"{lvl}: glvmodels_commit differs from run-commit.json")
        if stored.get("pin") != "P1" or stored.get("reference_commit") != P1_SHA:
            problems.append(f"{lvl}: not pinned at P1")
        for rel, digest in stored.get("read_from", {}).items():
            if sha(ROOT / rel) != digest:
                problems.append(f"{lvl}: read_from hash mismatch for {rel}")
        if any(c["tolerance"] <= 0 for c in stored["comparison"]["cases"]):
            problems.append(f"{lvl}: non-positive tolerance")
    if problems:
        raise SystemExit("grouping P1 receipts --check FAILED:\n  " + "\n  ".join(problems))
    print(f"grouping P1 receipts --check OK (batch verifier {status}; run commit {commit[:12]}):")
    for lvl, rec in fresh.items():
        print(f"  {lvl}: {rec['verdict']}")


def main(argv):
    if argv[:1] == ["--check"]:
        check()
    elif argv[:1] == ["run"]:
        oracle = argv[argv.index("--oracle") + 1] if "--oracle" in argv else DEFAULT_ORACLE
        run(oracle)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
