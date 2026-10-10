"""Write tracked P1 receipts and case-map rows for the family family.

In scope: the 21 required family rows the P1 carry scan lists as DANGLING or
NO_R_PINS (20 required_core rows and the compatibility_adapter row
FAMILY-BETA-ALIAS, which tools/true_parity_check.mjs counts as required). The
one NOT_BOUND_AT_P0 required row (FAMILY-16-LOGIT, empty case list,
PARTIAL_PENDING_DECISION_OPEN_QUESTION) and the 47 rejected or excluded rows
are out of scope. Three harnesses pay the rows, all run at gllvmTMB pin P1
(GLLVM_PARITY_PIN=P1):

  runparity (14 runs)   test/parity/runparity.jl, CORE070_PARITY_REQUIRED=1, one
                        run per fixture scope the case registry enforces (a
                        formula case shares a run with its native case). Each
                        cell fits one toy fixture in GLLVModels and in the P1
                        oracle and writes the R and Julia numbers it compares
                        to values-<case>.toml (core070_record_values!).
  family-links          tools/core070_family_links_batch.R + .jl: Bernoulli probit
                        and cloglog, logLik and trait intercepts at 1e-4.
  a6                    tools/core070_a6_studentt_fixture.jl + .R: Student-t with df
                        pinned at 6 on both engines; logLik, beta, sigma and
                        sign-aligned loadings at 1e-4 (the fixture's nu check can
                        only echo the pin, so it is recorded, not compared).

The executable case ids CORE070-FAMILY-<nn>-...-NATIVE-MODEL of the P0 case map
are paid by harness cells under NATIVE-<nn> ids through the maintainer's P0
registration decision (family-reconciliation-2026-09-01.json names the cell
for each); REGISTRATION below carries that mapping and --check verifies it
against the reconciliation record.

  bridge-p1             tools/core070_family_bridge_p1.R: the P1 twin of the frozen
                        three-case public R bridge batch (02 Poisson, 07 Beta, 05 NB2)
                        plus the FAMILY-BETA-ALIAS adapter case, live through JuliaCall
                        on the data the same-run native cells fit. Each public route
                        (gllvm_julia_fit, gllvmTMB(engine = "julia")) against a fresh
                        native GLLVModels fit at the frozen 1e-8 rule, and against the
                        same-run P1 R fit of the native cell (logLik, rtol 1e-6). The
                        FAMILY-05 case uses the stored NB2 smoke draw (decision
                        2026-09-28), a recorded deviation from the frozen contract's data.
                        Only the R-vs-Julia entries (route logLik vs the same-run R fit;
                        alias R fit vs native Julia) sit in a receipt's `comparison`
                        block. Route-vs-native entries (Julia vs Julia) go to
                        `route_consistency` and alias-vs-Beta entries (R vs R) to
                        `alias_identity_r`: they gate the case verdict under the frozen
                        rule but are never read as R-vs-Julia evidence or for the tier.

Not executed at P1 (receipt with evidence_kind not_executed): the 00 and 11
*-PUBLIC-R-BRIDGE cases, which are outside the frozen three-case bridge
sub-contract.

Shared gates (PR #567 / #569 / #571 / #579):

  * Batch verifier. family-links: tools/core070_verify_family_links_batch.py
    --self-test. a6: the fixture's own --self-test plus this tool's checks of the
    results file (verdict, pin, contract hash, R source-pin record). runparity:
    this tool's checks of the tracked run receipts (status, requested ==
    completed, every cell success, pins, oracle receipts, contract hash, every
    cell wrote values, harness files at the run commit). Each keeps a tracked
    verify.txt. A numeric row whose batch verifier did not pass is held back
    (numeric_held_batch_verifier_failed). There is no exception path.
  * Degenerate comparison: tools/core070_postfit_p1_receipts.py's mark_degenerate.
  * Provenance. Every receipt records glvmodels_commit = HEAD; the tool refuses a
    dirty tree (unless --allow-dirty, recorded) and refuses unless each run
    directory's run-commit.json names HEAD with an empty dirty list. --check
    verifies every receipt's glvmodels_commit against its batch's tracked
    run-commit.json.
  * Read-file hashes. Every case receipt records `read_from`; --check re-hashes
    them, re-derives every case receipt, every verify.txt it owns, and every
    case-map row, and exits nonzero on any difference.

Usage:
  python3 tools/core070_family_p1_receipts.py --runs DIR --runtimes JSON [--allow-dirty]
  python3 tools/core070_family_p1_receipts.py --rederive
  python3 tools/core070_family_p1_receipts.py --check
where DIR holds runparity-<name>/, family-links-p1/, a6-p1/ and bridge-p1/ (each with
run-commit.json) and carry-scan-p1.json. Rows another writer owns in case-map-family.json
(the C3 campaign rows, which carry a `clause`) and receipts/family/campaign/ are left
untouched.
"""
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from core070_postfit_p1_receipts import mark_degenerate  # noqa: E402  (PR #569's degenerate-comparison rule)
import core070_source_pin_check  # noqa: E402

OUT_REL = "docs/dev-log/core070/true-parity-latest"
REC_REL = f"{OUT_REL}/receipts/family"
CASEMAP_REL = f"{OUT_REL}/case-map-family.json"
P0_CASEMAP = ROOT / "docs/dev-log/core070/required-source-case-map.json"
RECONCILIATION = "docs/dev-log/core070/family-reconciliation-2026-09-01.json"
PINS = tomllib.loads((ROOT / "tools/core070_oracle_pins.toml").read_text())
P1_SHA = PINS["P1"]["reference_commit"]
P0_SHA = PINS["P0"]["reference_commit"]
ORACLE_BUILD = f"{OUT_REL}/receipts/covariance/oracle/build.json"
# The P1 oracle may be the default (Mac) build or the registered Totoro build of the same
# source archive (test/parity/core070_pin.jl _CORE070_ORACLE_BUILD_HOSTS); the run records
# which build receipt it used, and the receipts name it. The Mac string is fixed text; the
# Totoro string is filled from the versions the runs record (see host_string).
ORACLE_BUILD_TOTORO = f"{OUT_REL}/receipts/covariance/oracle/build-totoro.json"
ORACLE_BUILDS = {ORACLE_BUILD: "local Mac (M1 Ultra), OPENBLAS/OMP threads 1, JULIA_NUM_THREADS=4",
                 ORACLE_BUILD_TOTORO: "Totoro (Linux x86_64, R {r}, TMB {tmb}, Matrix {matrix}, Julia {julia}), "
                                      "OPENBLAS/OMP threads {blas}, JULIA_NUM_THREADS={jthreads}"}
ORACLE_SOURCE = f"{OUT_REL}/receipts/covariance/oracle/source.json"
MANIFEST = f"{OUT_REL}/frozen-r070-contract-p1.toml"
FL_CONTRACT = f"{OUT_REL}/family-links-batch-contract-p1.json"
A6_CONTRACT = f"{OUT_REL}/a6-studentt-contract-p1.json"

BRIDGE_BATCH = "bridge-p1"
BRIDGE_TOOL = "tools/core070_family_bridge_p1.R"
BRIDGE_CONTRACT = "docs/dev-log/core070/public-bridge-required-cases.json"
# bridge case -> (family key in the results, runparity batch whose same-run R fit it reads, health file)
BRIDGE_CASES = {
    "CORE070-FAMILY-02-LOG-PUBLIC-R-BRIDGE": ("poisson", "runparity-poisson", "poisson-health.toml"),
    "CORE070-FAMILY-07-LOGIT-PUBLIC-R-BRIDGE": ("beta", "runparity-beta", "beta-health.toml"),
    "CORE070-FAMILY-05-LOG-PUBLIC-R-BRIDGE": ("nb2", "runparity-nb2", "nb2-health.toml"),
}
ALIAS_CASE = "CORE070-FAMILY-BETA-ALIAS-COMPATIBILITY-ADAPTER"
NB2_SMOKE_SHA = "2bf2d819802a66e9836600caefec6e50802cac047db5d1a14611b4152ff1837c"

# runparity run -> the case ids it requests (whole fixture scopes; a formula case
# rides with its native case, as Core070CaseRegistry.requested_ids requires).
RUNPARITY_RUNS = {
    "runparity-binomial": ["NATIVE-02-BINOMIAL"],
    "runparity-poisson": ["NATIVE-03-POISSON", "CORE070-FAMILY-02-LOG-FORMULA-INTERFACE"],
    "runparity-lognormal": ["NATIVE-04-LOGNORMAL"],
    "runparity-dispersion": ["NATIVE-05-GAMMA", "NATIVE-16-NB1", "NATIVE-09-BETABINOMIAL"],
    "runparity-nb2": ["NATIVE-06-NB2"],
    "runparity-nb2-formula": ["CORE070-FAMILY-05-LOG-FORMULA-INTERFACE"],
    "runparity-tweedie": ["NATIVE-07-TWEEDIE"],
    "runparity-beta": ["NATIVE-08-BETA", "CORE070-FAMILY-07-LOGIT-FORMULA-INTERFACE"],
    "runparity-truncated-poisson": ["NATIVE-11-TRUNCATED-POISSON"],
    "runparity-truncated-nb2": ["NATIVE-12-TRUNCATED-NB2", "CORE070-FAMILY-11-LOG-FORMULA-INTERFACE"],
    "runparity-delta-lognormal": ["NATIVE-13-DELTA-LOGNORMAL"],
    "runparity-delta-gamma": ["NATIVE-14-DELTA-GAMMA"],
    "runparity-ordinal-probit": ["NATIVE-15-ORDINAL-PROBIT"],
    "runparity-gaussian": ["CORE070-FAMILY-00-IDENTITY-NATIVE-MODEL", "CORE070-FAMILY-00-IDENTITY-FORMULA-INTERFACE"],
}
# Supporting reports the fixtures already wrote into the receipt directory (TOML only).
RUNPARITY_REPORTS = ["poisson-fixture.toml", "poisson-health.toml", "beta-fixture.toml", "beta-health.toml",
                     "nb2-health.toml", "formula-nb2-health.toml", "nb2-formula.toml", "truncnb2-policy.toml",
                     "poisson-formula.toml", "beta-formula.toml", "truncnb2-formula.toml",
                     "gaussian-native.toml", "gaussian-formula.toml", "gaussian-long.toml"]

# P0 executable case id -> (batch, harness cell). CORE070-FAMILY-<nn>-...-NATIVE-MODEL ids
# named in family-reconciliation-2026-09-01.json are checked against that record.
REGISTRATION = {
    "CORE070-FAMILY-00-IDENTITY-NATIVE-MODEL": ("runparity-gaussian", "CORE070-FAMILY-00-IDENTITY-NATIVE-MODEL"),
    "CORE070-FAMILY-00-IDENTITY-FORMULA-INTERFACE": ("runparity-gaussian", "CORE070-FAMILY-00-IDENTITY-FORMULA-INTERFACE"),
    "CORE070-FAMILY-01-LOGIT-NATIVE-MODEL": ("runparity-binomial", "NATIVE-02-BINOMIAL"),
    "CORE070-FAMILY-01-PROBIT-NATIVE-MODEL": ("family-links-p1", "CORE070-FAMILY-01-PROBIT-NATIVE-MODEL"),
    "CORE070-FAMILY-01-CLOGLOG-NATIVE-MODEL": ("family-links-p1", "CORE070-FAMILY-01-CLOGLOG-NATIVE-MODEL"),
    "NATIVE-03-POISSON": ("runparity-poisson", "NATIVE-03-POISSON"),
    "CORE070-FAMILY-02-LOG-FORMULA-INTERFACE": ("runparity-poisson", "CORE070-FAMILY-02-LOG-FORMULA-INTERFACE"),
    "CORE070-FAMILY-03-LOG-NATIVE-MODEL": ("runparity-lognormal", "NATIVE-04-LOGNORMAL"),
    "CORE070-FAMILY-04-LOG-NATIVE-MODEL": ("runparity-dispersion", "NATIVE-05-GAMMA"),
    "NATIVE-06-NB2": ("runparity-nb2", "NATIVE-06-NB2"),
    "CORE070-FAMILY-05-LOG-FORMULA-INTERFACE": ("runparity-nb2-formula", "CORE070-FAMILY-05-LOG-FORMULA-INTERFACE"),
    "CORE070-FAMILY-06-LOG-NATIVE-MODEL": ("runparity-tweedie", "NATIVE-07-TWEEDIE"),
    "CORE070-FAMILY-06-FIXED-SHAPE-NATIVE-MODEL": ("runparity-tweedie", "NATIVE-07-TWEEDIE"),
    "NATIVE-08-BETA": ("runparity-beta", "NATIVE-08-BETA"),
    "CORE070-FAMILY-07-LOGIT-FORMULA-INTERFACE": ("runparity-beta", "CORE070-FAMILY-07-LOGIT-FORMULA-INTERFACE"),
    "CORE070-FAMILY-08-LOGIT-NATIVE-MODEL": ("runparity-dispersion", "NATIVE-09-BETABINOMIAL"),
    "CORE070-A6-STUDENTT-FIXED-DF-PAIRED": ("a6-p1", "fixed"),
    "CORE070-FAMILY-10-LOG-NATIVE-MODEL": ("runparity-truncated-poisson", "NATIVE-11-TRUNCATED-POISSON"),
    "NATIVE-12-TRUNCATED-NB2": ("runparity-truncated-nb2", "NATIVE-12-TRUNCATED-NB2"),
    "CORE070-FAMILY-11-LOG-FORMULA-INTERFACE": ("runparity-truncated-nb2", "CORE070-FAMILY-11-LOG-FORMULA-INTERFACE"),
    "CORE070-FAMILY-12-LOGIT-LOG-NATIVE-MODEL": ("runparity-delta-lognormal", "NATIVE-13-DELTA-LOGNORMAL"),
    "CORE070-FAMILY-13-LOGIT-LOG-NATIVE-MODEL": ("runparity-delta-gamma", "NATIVE-14-DELTA-GAMMA"),
    "CORE070-FAMILY-14-PROBIT-NATIVE-MODEL": ("runparity-ordinal-probit", "NATIVE-15-ORDINAL-PROBIT"),
    "CORE070-FAMILY-15-LOG-NATIVE-MODEL": ("runparity-dispersion", "NATIVE-16-NB1"),
    **{cid: (BRIDGE_BATCH, cid) for cid in BRIDGE_CASES},
    ALIAS_CASE: (BRIDGE_BATCH, ALIAS_CASE),
}
NOT_EXECUTED = {
    **{f"CORE070-FAMILY-{t}-PUBLIC-R-BRIDGE": (
        "Public R bridge case: gllvmTMB(engine = 'julia') and gllvm_julia_fit() calling GLLVModels through "
        "JuliaCall. Not in the frozen three-case bridge sub-contract (docs/dev-log/core070/public-bridge-required-"
        "cases.json covers 02, 05 and 07 only), so the P1 bridge twin (tools/core070_family_bridge_p1.R) does not "
        "run it. Not run at P1 here.")
       for t in ("00-IDENTITY",)},
}
FAMILY11_BOUNDARY_CASE = "CORE070-FAMILY-11-LOG-PUBLIC-R-BRIDGE"
FAMILY11_BOUNDARY_RAW = f"{REC_REL}/first-seven-boundary/r-public-bridge.json"
FAMILY11_BOUNDARY_RECEIPT = f"{REC_REL}/first-seven-boundary/{FAMILY11_BOUNDARY_CASE}.json"
FAMILY11_DATA_SHA = "ecbcf9f501c7e618131f2c3f1f0d213bb0e92364a72c0519095c52ef30930948"
FAMILY11_GLVMODELS_COMMIT = "4b78fa01245381f6c0ab8a8b9ea0bc581ade825a"

# What each harness cell fits: toy fixtures, likelihood-level agreement.
MEASURES = {
    "NATIVE-02-BINOMIAL": "Bernoulli logit, p=5 traits, n=60 sites, K=2, seed 43",
    "NATIVE-03-POISSON": "Poisson log, p=5, n=60, K=2, seed 44",
    "CORE070-FAMILY-02-LOG-FORMULA-INTERFACE": "the NATIVE-03-POISSON data refit through gllvm(@formula) wide and long, "
                                               "each compared with that cell's R fit (no separate R formula fit)",
    "NATIVE-04-LOGNORMAL": "lognormal, p=5, n=60, K=2, seed 52",
    "NATIVE-05-GAMMA": "Gamma log, per-trait dispersion, p=5, n=120, K=1, seed 54",
    "NATIVE-16-NB1": "NB1 log, per-trait dispersion, p=5, n=120, K=1, seed 55",
    "NATIVE-09-BETABINOMIAL": "beta-binomial logit, N=8 trials, per-trait dispersion, p=5, n=120, K=1, seed 56",
    "NATIVE-06-NB2": "NB2 log, per-trait dispersion, p=5, n=80, K=2, seed 45",
    "CORE070-FAMILY-05-LOG-FORMULA-INTERFACE": "the NATIVE-06-NB2 data: the R-vs-Julia number is the native route "
                                               "against a fresh R fit; the wide and long formula routes are checked "
                                               "against the native route in Julia only (atol 1e-10)",
    "NATIVE-07-TWEEDIE": "Tweedie log, p=5, n=150, K=1, seed 82; three models: fixed common power 1.5, estimated "
                         "shared power, estimated per-species power (all three must agree)",
    "NATIVE-08-BETA": "Beta logit, p=5, n=60, K=1, seed 45",
    "CORE070-FAMILY-07-LOGIT-FORMULA-INTERFACE": "the NATIVE-08-BETA data refit through gllvm(@formula) wide and "
                                                 "long, each compared with that cell's R fit",
    "NATIVE-11-TRUNCATED-POISSON": "zero-truncated Poisson, p=5, n=60, K=2, seed 53",
    "NATIVE-12-TRUNCATED-NB2": "zero-truncated NB2, per-trait dispersion, p=5, n=120, K=1, seed 58",
    "CORE070-FAMILY-11-LOG-FORMULA-INTERFACE": "the NATIVE-12-TRUNCATED-NB2 data refit through gllvm(@formula) wide "
                                               "and long, each compared with that cell's R fit",
    "NATIVE-13-DELTA-LOGNORMAL": "delta-lognormal, per-trait dispersion, p=5, n=130, K=1, seed 61",
    "NATIVE-14-DELTA-GAMMA": "delta-Gamma, per-trait dispersion, p=5, n=130, K=1, seed 62",
    "NATIVE-15-ORDINAL-PROBIT": "ordinal probit, 3 categories, p=5, n=60, K=1, seed 46",
    "CORE070-FAMILY-00-IDENTITY-NATIVE-MODEL": "Gaussian identity, default unique, p=4, n=120, K=1, seed 81031 "
                                               "(fixture core070_gaussian_original.toml)",
    "CORE070-FAMILY-00-IDENTITY-FORMULA-INTERFACE": "the same Gaussian data refit through gllvm(@formula) wide and "
                                                    "long, each compared with the group's R fit",
    "CORE070-FAMILY-01-PROBIT-NATIVE-MODEL": "Bernoulli probit, p=4, n=120, K=1, seed 81011",
    "CORE070-FAMILY-01-CLOGLOG-NATIVE-MODEL": "Bernoulli cloglog, p=4, n=120, K=1, seed 81012",
    "fixed": "Student-t identity, df pinned at 6 on both engines, per-trait sigma, p=5, n=250, K=1, seed 20260901",
}
SAME_MEASUREMENT = {
    "CORE070-FAMILY-06-LOG-NATIVE-MODEL": "CORE070-FAMILY-06-FIXED-SHAPE-NATIVE-MODEL",
    "CORE070-FAMILY-06-FIXED-SHAPE-NATIVE-MODEL": "CORE070-FAMILY-06-LOG-NATIVE-MODEL",
}
ROW_NOTES = {
    "family/FAMILY-06-LOG": "FAMILY-06-LOG and FAMILY-06-FIXED-SHAPE are one measurement (the NATIVE-07-TWEEDIE cell, "
                            "by the P0 registration) counted on two rows; whether it counts once or twice is for the "
                            "maintainer.",
    "family/FAMILY-06-FIXED-SHAPE": "FAMILY-06-LOG and FAMILY-06-FIXED-SHAPE are one measurement (the NATIVE-07-TWEEDIE "
                                    "cell, by the P0 registration) counted on two rows; whether it counts once or twice "
                                    "is for the maintainer.",
    "family/FAMILY-09-FIXED-SHAPE": "FAMILY-09-FIXED-SHAPE and FAMILY-09-IDENTITY share their one case id "
                                    "(CORE070-A6-STUDENTT-FIXED-DF-PAIRED): one measurement counted on two rows.",
    "family/FAMILY-09-IDENTITY": "FAMILY-09-FIXED-SHAPE and FAMILY-09-IDENTITY share their one case id "
                                 "(CORE070-A6-STUDENTT-FIXED-DF-PAIRED): one measurement counted on two rows.",
}
COUNT_KEYS = ("numeric_pass", "numeric_fail", "numeric_held_batch_verifier_failed", "numeric_non_discriminating",
              "partial_case_not_executed", "not_measured")
IN_SCOPE_STATUS = ("DANGLING", "NO_R_PINS")
A6_TOL = 1e-4


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(p):
    return json.loads(Path(p).read_text())


def load_toml(p):
    return tomllib.loads(Path(p).read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def git(*argv, check=True, text=True):
    return subprocess.run(["git", "-C", str(ROOT), *argv], check=check, capture_output=True, text=text)


def batch_rel(batch):
    return f"{REC_REL}/{batch}"


def all_batches():
    return list(RUNPARITY_RUNS) + ["family-links-p1", "a6-p1", BRIDGE_BATCH]


def git_state():
    head = git("rev-parse", "HEAD").stdout.strip()
    own = (REC_REL + "/", CASEMAP_REL)
    dirty = [line[3:] for line in git("status", "--porcelain", "--untracked-files=no").stdout.splitlines()
             if not line[3:].startswith(own)]
    return head, dirty


def check_run_commit(run_dir, head):
    p = run_dir / "run-commit.json"
    if not p.is_file():
        raise SystemExit(f"{run_dir} has no run-commit.json; re-run the batch from a clean commit")
    rc = load(p)
    if rc.get("glvmodels_commit") != head or rc.get("dirty") != []:
        raise SystemExit(f"{run_dir}: run at {rc.get('glvmodels_commit')} dirty={rc.get('dirty')}, "
                         f"not at clean HEAD {head}; re-run at HEAD")


def run_commit(batch):
    return load(ROOT / batch_rel(batch) / "run-commit.json")["glvmodels_commit"]


def host_string(build):
    """The receipts' host text for oracle build receipt `build`. For the Totoro build the
    versions and thread counts are read from what the runs record (every runparity run.toml
    [source] and bridge-p1/results.json) and must agree across them."""
    if build != ORACLE_BUILD_TOTORO:
        return ORACLE_BUILDS[build]
    seen = set()
    for b in RUNPARITY_RUNS:
        src = load_toml(ROOT / batch_rel(b) / "run.toml").get("source", {})
        seen.add((src.get("r_version"), src.get("tmb_version"), src.get("matrix_version"), src.get("julia_version"),
                  src.get("blas_threads"), src.get("julia_threads")))
    res = bridge_results()
    seen.add((res.get("r_version"), res.get("tmb_version"), res.get("matrix_version"), res.get("julia_version"),
              res.get("blas_threads"), res.get("julia_threads")))
    if len(seen) != 1 or None in next(iter(seen)):
        raise SystemExit(f"runs do not record one set of host versions: {sorted(map(str, seen))}")
    r, tmb, matrix, julia, blas, jthreads = next(iter(seen))
    return ORACLE_BUILDS[build].format(r=r.removeprefix("R version ").split()[0], tmb=tmb, matrix=matrix,
                                       julia=julia, blas=blas, jthreads=jthreads)


def read_from(*rels):
    return {rel: sha(ROOT / rel) for rel in rels}


# ---------------------------------------------------------------------------
# Batch verifiers. runparity and the a6 results checks are derived from tracked
# files only, so --check re-derives them; family-links and the a6 self-test are
# external commands whose output is kept verbatim.
# ---------------------------------------------------------------------------
def harness_drift(run, rc):
    """Execution-inventory files tracked at commit rc whose bytes differ from what the run hashed."""
    tracked = set(git("ls-tree", "-r", "--name-only", rc).stdout.splitlines())
    entries = [e for e in run.get("execution", {}).get("entries", []) if e["path"] in tracked]
    if not entries:
        return []
    proc = subprocess.run(["git", "-C", str(ROOT), "cat-file", "--batch"], check=True, capture_output=True,
                          input="".join(f"{rc}:{e['path']}\n" for e in entries).encode())
    out, pos, drift = proc.stdout, 0, []
    for e in entries:
        nl = out.index(b"\n", pos)
        size = int(out[pos:nl].split()[2])
        blob = out[nl + 1:nl + 1 + size]
        pos = nl + 1 + size + 1
        if hashlib.sha256(blob).hexdigest() != e["sha256"]:
            drift.append(e["path"])
    return drift


@functools.lru_cache(maxsize=None)
def runparity_checks(batch):
    d = ROOT / batch_rel(batch)
    run = load_toml(d / "run.toml")
    requested = RUNPARITY_RUNS[batch]
    cells = {cid: load_toml(d / f"cell-{cid}.toml") for cid in requested if (d / f"cell-{cid}.toml").is_file()}
    src = run.get("source", {})
    rc = run_commit(batch)
    drift = harness_drift(run, rc)
    checks = [
        ("run status success, exit code 0, success marker",
         run.get("status") == "success" and run.get("exit_code") == 0
         and run.get("success_marker") == "CORE070_PARITY_SUCCESS"),
        ("requested case ids are this batch's scope", sorted(run.get("requested_case_ids", [])) == sorted(requested)),
        ("completed == requested", sorted(run.get("completed_case_ids", [])) == sorted(requested)),
        ("every requested cell receipt present", sorted(cells) == sorted(requested)),
        ("every cell status success", all(c.get("status") == "success" for c in cells.values()) and bool(cells)),
        ("every cell names this run and its execution manifest",
         all(c.get("run_id") == run.get("run_id")
             and c.get("execution_manifest_sha256") == run.get("execution", {}).get("manifest_sha256")
             for c in cells.values())),
        ("source pinned at P1 (commit, archive, namespace, source tree)",
         all(src.get(k) == PINS["P1"][k] for k in ("reference_commit", "archive_sha256", "namespace_sha256",
                                                   "source_tree_sha256"))),
        ("oracle build/source receipts are tracked P1 receipts (a registered build, the pin's source receipt)",
         src.get("oracle_build_receipt_sha256") in {sha(ROOT / b) for b in ORACLE_BUILDS}
         and src.get("oracle_source_receipt_sha256") == sha(ROOT / ORACLE_SOURCE)),
        ("runner manifest is frozen-r070-contract-p1.toml", run.get("contract_sha256") == sha(ROOT / MANIFEST)),
        ("every requested cell wrote values-<case>.toml",
         all((d / f"values-{cid}.toml").is_file() and load_toml(d / f"values-{cid}.toml").get("values")
             for cid in requested)),
        (f"harness files in the execution inventory match run commit {rc[:12]}", not drift),
    ]
    return checks


@functools.lru_cache(maxsize=None)
def a6_checks():
    d = ROOT / batch_rel("a6-p1")
    res = load_toml(d / "results.toml")
    pin = res.get("r_source_pin", {})
    receipt_like = {"gllvmTMB_version": pin.get("gllvmTMB_version"), "source_pin": pin or None}
    problem = core070_source_pin_check.source_pin_problem(receipt_like, "P1")
    return [
        ("fixture verdict PASS (gating fixed case)", res.get("verdict") == "PASS"),
        ("pinned at P1", res.get("parity_pin") == "P1" and res.get("reference_commit") == P1_SHA),
        ("contract is the tracked P1 twin", res.get("contract") == A6_CONTRACT
         and res.get("contract_sha256") == sha(ROOT / A6_CONTRACT)),
        ("R readback is the tracked r-output.tsv", res.get("r_output_sha256") == sha(d / "r-output.tsv")),
        ("R source-pin record matches tools/core070_oracle_pins.toml [P1]"
         + ("" if problem is None else f" ({problem})"), problem is None),
        ("paired tolerance is the fixture's PAIRED_TOL", res.get("paired_tol") == A6_TOL),
    ]


def bridge_results():
    return load(ROOT / batch_rel(BRIDGE_BATCH) / "results.json")


def bridge_case_record(res, cid):
    hits = [c for c in res.get("cases", []) if c.get("case_id") == cid]
    if len(hits) != 1:
        raise SystemExit(f"{BRIDGE_BATCH}/results.json has {len(hits)} records for {cid}")
    return hits[0]


@functools.lru_cache(maxsize=None)
def bridge_checks():
    res = bridge_results()
    receipt_like = {"gllvmTMB_version": res.get("gllvmtmb_version"), "source_pin": res.get("source_pin")}
    problem = core070_source_pin_check.source_pin_problem(receipt_like, "P1")
    contract = load(ROOT / BRIDGE_CONTRACT)
    expected = {c["id"]: c["data_sha256"] for c in contract["cases"]}
    expected["CORE070-FAMILY-05-LOG-PUBLIC-R-BRIDGE"] = NB2_SMOKE_SHA  # recorded deviation, decision 2026-09-28
    ids = sorted(c.get("case_id") for c in res.get("cases", []))
    same_run = []
    for cid, (_fam, batch, health) in BRIDGE_CASES.items():
        rec = next((c for c in res.get("cases", []) if c.get("case_id") == cid), {})
        tracked = ROOT / batch_rel(batch) / health
        same_run.append(tracked.is_file() and (rec.get("same_run_r_fit") or {}).get("sha256") == sha(tracked))
    beta = next((c for c in res.get("cases", []) if c.get("case_id") == "CORE070-FAMILY-07-LOGIT-PUBLIC-R-BRIDGE"), {})
    alias = next((c for c in res.get("cases", []) if c.get("case_id") == ALIAS_CASE), {})
    return [
        ("schema core070-family-bridge-p1/v1", res.get("schema") == "core070-family-bridge-p1/v1"),
        ("pinned at P1 (reference commit, gllvmTMB 0.7.1)",
         res.get("reference_commit") == P1_SHA and res.get("gllvmtmb_version") == PINS["P1"]["version"]),
        ("R source-pin record matches tools/core070_oracle_pins.toml [P1]"
         + ("" if problem is None else f" ({problem})"), problem is None),
        ("bridge contract is the tracked frozen sub-contract",
         res.get("bridge_contract") == BRIDGE_CONTRACT and res.get("bridge_contract_sha256") == sha(ROOT / BRIDGE_CONTRACT)),
        ("exactly the three bridge cases and the alias case", ids == sorted([*BRIDGE_CASES, ALIAS_CASE])),
        ("each bridge case fitted its expected data bytes (NB2: the smoke draw)",
         all(bridge_case_record(res, cid).get("data_sha256") == expected[cid]
             == bridge_case_record(res, cid).get("expected_data_sha256") for cid in BRIDGE_CASES)),
        ("each bridge case read the same-run R fit tracked beside it (sha256 of the health file)", all(same_run)),
        ("the alias case fitted the Beta case's data", bool(alias) and alias.get("data_sha256") == beta.get("data_sha256")),
    ]


def render_checks(title, checks):
    lines = [f"# {title}"] + [f"{'PASS' if ok else 'FAIL'}  {name}" for name, ok in checks]
    ok = all(ok for _, ok in checks)
    return "\n".join(lines + [f"# status {'PASS' if ok else 'FAIL'}"]) + "\n", ok


def verify_text(batch):
    """The derived part of a batch's verify.txt (everything this tool computes itself)."""
    if batch in RUNPARITY_RUNS:
        return render_checks(f"runparity batch verifier (tools/core070_family_p1_receipts.py), {batch}",
                             runparity_checks(batch))
    if batch == "a6-p1":
        return render_checks("a6 results checks (tools/core070_family_p1_receipts.py)", a6_checks())
    if batch == BRIDGE_BATCH:
        return render_checks(f"bridge results checks (tools/core070_family_p1_receipts.py), {BRIDGE_BATCH}",
                             bridge_checks())
    return "", True


EXTERNAL = {
    "family-links-p1": (["python3", "tools/core070_verify_family_links_batch.py", "--state", "{state}", "--self-test"],
                        "CORE070_FAMILY_LINKS_BATCH_VERIFIED", {"GLLVM_PARITY_PIN": "P1"}),
    "a6-p1": (["julia", "--project=.", "tools/core070_a6_studentt_fixture.jl", "--self-test"],
              "CORE070_A6_STUDENTT_SELF_TEST_OK", {"GLLVM_PARITY_PIN": "P1"}),
}
SEPARATOR = "# ---- derived checks ----\n"


def run_external(batch, state):
    argv_t, marker, env = EXTERNAL[batch]
    argv = [a.format(state=str(state)) for a in argv_t]
    proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, env=dict(os.environ, **env))
    shown = " ".join(a if a != str(state) else "<raw run>" for a in argv)
    return (f"$ {' '.join(f'{k}={v}' for k, v in env.items())} {shown}\n# exit code {proc.returncode}\n"
            + proc.stdout + proc.stderr)


def verifier_block(batch):
    rel = f"{batch_rel(batch)}/verify.txt"
    text = (ROOT / rel).read_text()
    derived, derived_ok = verify_text(batch)
    external, _, tail = text.partition(SEPARATOR) if SEPARATOR in text else (text, "", "")
    ok = derived_ok and (tail == derived if derived else True)
    if batch in EXTERNAL:
        marker = EXTERNAL[batch][1]
        ok = ok and "# exit code 0\n" in external and marker in external
        tool = EXTERNAL[batch][0][1] if batch == "family-links-p1" else EXTERNAL[batch][0][2]
    elif batch == BRIDGE_BATCH:
        tool = f"tools/core070_family_p1_receipts.py ({BRIDGE_TOOL} results)"
    else:
        tool = "tools/core070_family_p1_receipts.py (runparity run receipts)"
    return {"tool": tool, "status": "PASS" if ok else "FAIL", "log": rel}


# ---------------------------------------------------------------------------
# Derivation: case receipts from tracked files only.
# ---------------------------------------------------------------------------
def entry(cid, label, r, j, tol, rule, extra=None):
    rv = r if isinstance(r, list) else [r]
    jv = j if isinstance(j, list) else [j]
    if len(rv) != len(jv):
        raise SystemExit(f"{cid} {label}: R length {len(rv)} != Julia length {len(jv)}")
    diff = max(abs(a - b) for a, b in zip(rv, jv))
    e = {"case_id": cid, "quantity": label, "max_abs_diff": diff, "tolerance": tol, "tolerance_rule": rule,
         "n_values": len(rv), "r_value": r, "julia_value": j,
         "diff_source": "recomputed from the saved R and Julia values"}
    if extra:
        e.update(extra)
    return mark_degenerate(e, rv)


def runparity_case(cid, batch, cell_id):
    d = batch_rel(batch)
    run = load_toml(ROOT / d / "run.toml")
    cell_path = f"{d}/cell-{cell_id}.toml"
    values_path = f"{d}/values-{cell_id}.toml"
    cell = load_toml(ROOT / cell_path) if (ROOT / cell_path).is_file() else None
    vals = load_toml(ROOT / values_path)
    if vals.get("case_id") != cell_id:
        raise SystemExit(f"{values_path} names {vals.get('case_id')}, not {cell_id}")
    entries = []
    for v in vals["values"]:
        tol = max(v["atol"], v["rtol"] * max(abs(v["r"]), abs(v["julia"])))
        rule = (f"the cell's own test, {v['test']} (rtol={v['rtol']:g}, atol={v['atol']:g}; tolerance = "
                f"max(atol, rtol*max(|R|,|Julia|)))")
        entries.append(entry(cid, v["label"], v["r"], v["julia"], tol, rule))
    cell_ok = cell is not None and cell.get("status") == "success"
    within = all(e["max_abs_diff"] <= e["tolerance"] for e in entries)
    verdict = "PASS" if cell_ok and within else "FAIL"
    reports = [f"{d}/{n}" for n in RUNPARITY_REPORTS if (ROOT / d / n).is_file()]
    body = {"batch": f"test/parity/runparity.jl, CORE070_PARITY_REQUIRED=1, GLLVM_PARITY_PIN=P1, run {batch}",
            "harness_cell": cell_id, "fixture": (cell or {}).get("fixture"),
            "measures": MEASURES[cell_id] + "; toy fixture, likelihood-level agreement only",
            "cell_status": (cell or {}).get("status", "no cell receipt"),
            "cell_assertions": (cell or {}).get("assertions"),
            "cell_execution_case_ids": (cell or {}).get("execution_case_ids"),
            "run_status": run.get("status"), "run_failure_reason": run.get("failure_reason"),
            "batch_verifier": verifier_block(batch),
            "read_from": read_from(f"{d}/run.toml", values_path, f"{d}/run-commit.json", f"{d}/verify.txt",
                                   *([cell_path] if cell else []), *reports),
            "raw": [values_path, *([cell_path] if cell else []), *reports]}
    if cid != cell_id:
        body["registration"] = (f"{cid} is paid by harness cell {cell_id} by the maintainer's P0 registration "
                                f"decision ({RECONCILIATION}); carried unchanged")
    if not cell_ok:
        shared = len((cell or {}).get("execution_case_ids") or []) > 1
        body["why_fail"] = ("the cell's own assertions did not all pass at P1 (see cell_assertions; the failing "
                            "assertion may be a health or structure check rather than the compared numbers)"
                            + ("; this fixture group records one shared count for all its cases, so a failure in "
                               "any of them fails every case of the group" if shared else ""))
    return verdict, body, entries


def family_links_case(cid):
    d = batch_rel("family-links-p1")
    jres, oracle, receipt = (load(ROOT / d / n) for n in ("julia-results.json", "r-oracle.json", "receipt.json"))
    contract = load(ROOT / FL_CONTRACT)
    if receipt["reference_commit"] != P1_SHA or receipt["contract_sha256"] != sha(ROOT / FL_CONTRACT):
        raise SystemExit("family-links receipt is not pinned at P1 or does not name the tracked twin")
    if receipt["julia_results_sha256"] != sha(ROOT / d / "julia-results.json"):
        raise SystemExit("family-links julia-results.json does not match its receipt")
    key = "probit" if "PROBIT" in cid else "cloglog"
    jc = jres["cases"][cid]
    if jc["r_loglik"] != oracle[key]["loglik"] or jc["r_coef"] != oracle[key]["coef"]:
        raise SystemExit(f"{cid}: Julia child's copy of the R values differs from r-oracle.json")
    rule = f"{FL_CONTRACT} case tolerance (<=1e-4 absolute, carried verbatim from P0)"
    entries = [entry(cid, "logLik", oracle[key]["loglik"], jc["julia_loglik"], 1e-4, rule),
               entry(cid, "trait intercepts", oracle[key]["coef"], jc["julia_coef"], 1e-4, rule)]
    for e, harness in zip(entries, (jc["loglik_delta"], jc["coef_delta"])):
        if abs(e["max_abs_diff"] - harness) > 1e-12 * max(1.0, abs(harness)):
            raise SystemExit(f"{cid} {e['quantity']}: recomputed {e['max_abs_diff']} != harness {harness}")
    within = all(e["max_abs_diff"] <= e["tolerance"] for e in entries)
    verdict = "PASS" if jc["pass"] and within and jc["saturated"] is False else "FAIL"
    cc = next(c for c in contract["cases"] if c["case_id"] == cid)
    body = {"batch": "tools/core070_family_links_batch.R + .jl, GLLVM_PARITY_PIN=P1",
            "measures": MEASURES[cid] + "; independent L-BFGS optimizations under the same Laplace approximation; "
                        "toy fixture",
            "r_call": cc["r_call"], "julia_surface": cc["julia_surface"], "check": cc["check"],
            "harness_pass": bool(jc["pass"]), "saturated": jc["saturated"], "batch_status": receipt["status"],
            "gllvmtmb_version": receipt["gllvmTMB_version"], "batch_verifier": verifier_block("family-links-p1"),
            "read_from": read_from(*(f"{d}/{n}" for n in ("receipt.json", "julia-results.json", "r-oracle.json",
                                                          "results.tsv", "run-commit.json", "verify.txt")),
                                   FL_CONTRACT),
            "raw": [f"{d}/r-oracle.json", f"{d}/julia-results.json"]}
    return verdict, body, entries


def a6_case(cid):
    d = batch_rel("a6-p1")
    res = load_toml(ROOT / d / "results.toml")
    j, r = res["julia_fixed"], res["r_fixed"]
    jl_load, r_load = list(j["loading"]), list(r["loading"])
    sign = -1.0 if sum(a * b for a, b in zip(jl_load, r_load)) < 0 else 1.0
    rule = "tools/core070_a6_studentt_fixture.jl PAIRED_TOL (1e-4 absolute, unchanged from P0)"
    entries = [entry(cid, "logLik", r["loglik"], j["loglik"], A6_TOL, rule),
               entry(cid, "beta", list(r["beta"]), list(j["beta"]), A6_TOL, rule),
               entry(cid, "sigma (per trait)", list(r["sigma_student"]), list(j["sigma"]), A6_TOL, rule),
               entry(cid, "loadings (K=1, sign-aligned)", r_load, [sign * x for x in jl_load], A6_TOL, rule,
                     {"julia_sign_flipped": sign < 0})]
    # The fixture also checks nu, but df is pinned at 6 on both engines, so that check can only
    # confirm the pin was passed through; it is recorded here, not counted as a comparison.
    nu_r, nu_j = list(r["df_student"]), list(j["nu"])
    pinned_nu = {"r_df_student": nu_r, "julia_nu": nu_j,
                 "max_abs_diff": max(abs(a - nu_j[0]) for a in nu_r),
                 "note": "df pinned at 6 on both engines; the fixture's nu check confirms the pin, it cannot "
                         "discriminate the fits, so it is not a comparison entry"}
    health = (j["converged"] is True and j["nu_boundary"] is False and r["healthy"] == 1.0
              and r["nu_at_boundary"] == 0.0)
    within = all(e["max_abs_diff"] <= e["tolerance"] for e in entries)
    verdict = "PASS" if res["verdict"] == "PASS" and within and health else "FAIL"
    body = {"batch": "tools/core070_a6_studentt_fixture.jl + .R, GLLVM_PARITY_PIN=P1",
            "measures": MEASURES["fixed"] + "; both-engine health required; toy fixture",
            "pinned_nu_check": pinned_nu, "fixture_verdict": res["verdict"], "fixture_messages": res["messages"], "both_engine_health": health,
            "free_nu_case": {"status": "recorded structural divergence, never gating (R per-trait df vs Julia "
                                       "shared nu)", "r_loglik": res["r_free"]["loglik"], "julia_loglik": res["julia_free"]["loglik"]},
            "gllvmtmb_version": res["r_source_pin"].get("gllvmTMB_version"),
            "batch_verifier": verifier_block("a6-p1"),
            "read_from": read_from(*(f"{d}/{n}" for n in ("results.toml", "r-output.tsv", "r-output.tsv.source-pin.tsv",
                                                          "run-commit.json", "verify.txt")), A6_CONTRACT),
            "raw": [f"{d}/results.toml", f"{d}/r-output.tsv"]}
    return verdict, body, entries


BRIDGE_RULE = (f"{BRIDGE_CONTRACT} acceptance rule, carried verbatim: public route against a fresh default native "
               "fit, absolute difference <= 1e-8 (logLik, intercepts, loading covariance), dispersion relative "
               "difference <= 1e-8 (tolerance = 1e-8 * min |native dispersion|)")
R_RULE = ("the frozen rule's R leg (R logLik rtol <= 1e-6), at P1 against the same-run P1 R fit of the native cell "
          "(the P0 batch's retained P0 R logLik is not on any host); tolerance = 1e-6 * |R logLik|")


def consistency(cid, label, a, b, tol, rule, keys):
    """A same-engine consistency record (Julia vs Julia, or R vs R). It is kept out of the
    `comparison` block, which holds only R-computed r_value against Julia-computed julia_value,
    so neither the checker nor the tier logic reads it as R-vs-Julia evidence."""
    av = a if isinstance(a, list) else [a]
    bv = b if isinstance(b, list) else [b]
    if len(av) != len(bv):
        raise SystemExit(f"{cid} {label}: {keys[0]} length {len(av)} != {keys[1]} length {len(bv)}")
    diff = max(abs(x - y) for x, y in zip(av, bv))
    return {"case_id": cid, "quantity": label, "max_abs_diff": diff, "tolerance": tol, "tolerance_rule": rule,
            "n_values": len(av), keys[0]: a, keys[1]: b, "within_tolerance": diff <= tol,
            "diff_source": "recomputed from the saved values"}


ROUTE_KEYS = ("route_value", "native_value")
ROUTE_CONSISTENCY_NOTE = ("Julia vs Julia: each public route (computed by GLLVModels through JuliaCall) against a fresh "
                          "native GLLVModels fit. Evidence that the public route reproduces the native fit; not "
                          "R-vs-Julia evidence, so not in the comparison block. Counted in the case verdict (the "
                          "frozen bridge acceptance rule), never in the evidence tier.")


def route_entries(cid, route, rec, native):
    x = rec[route]
    out = [consistency(cid, f"logLik, public route {route} vs fresh native GLLVModels", x["loglik"],
                       native["loglik"], 1e-8, BRIDGE_RULE, ROUTE_KEYS),
           consistency(cid, f"trait intercepts, public route {route} vs fresh native", x["alpha"], native["alpha"],
                       1e-8, BRIDGE_RULE, ROUTE_KEYS),
           consistency(cid, f"loading covariance LL', public route {route} vs fresh native", x["shared_covariance"],
                       native["shared_covariance"], 1e-8, BRIDGE_RULE, ROUTE_KEYS)]
    if native["dispersion"]:
        out.append(consistency(cid, f"per-trait dispersion, public route {route} vs fresh native", x["dispersion"],
                               native["dispersion"], 1e-8 * min(abs(v) for v in native["dispersion"]), BRIDGE_RULE,
                               ROUTE_KEYS))
    return out


def route_ok(rec, route, native):
    x = rec[route]
    return ("error" not in x and "gllvmTMB_julia" in (x.get("class") or []) and x.get("converged") is True
            and x.get("n_traits") == rec["p"] and x.get("n_units") == rec["n"] and x.get("d") == rec["K"]
            and x.get("df") == native.get("df"))


def native_ok(native):
    return ("error" not in native or native.get("error") is None) and native.get("converged") is True and \
        native.get("gradient_max") <= 1e-4 and native.get("fd_stability") <= 1e-4 and native.get("objective_delta") <= 1e-8


def bridge_case(cid):
    d = batch_rel(BRIDGE_BATCH)
    res = bridge_results()
    rec = bridge_case_record(res, cid)
    native = rec["native"]
    if cid == ALIAS_CASE:
        return alias_case(cid, rec, native)
    fam, batch, health = BRIDGE_CASES[cid]
    rfit = rec["same_run_r_fit"]
    entries, routes = [], []
    gates = {"native health (converged, |gradient| and FD stability <= 1e-4, objective reconstruction <= 1e-8)":
             native_ok(native),
             "same-run R fit healthy (convergence code 0) on the same data bytes":
             rfit.get("r_code") == 0 and rfit.get("r_converged") in ("true", "absent")
             and rfit.get("data_sha256") == rec["data_sha256"]}
    for route in ("matrix", "formula"):
        ok = route_ok(rec, route, native)
        gates[f"public route {route}: gllvmTMB_julia object, converged, shape and df match the native fit"] = ok
        if "error" in rec[route]:
            continue
        routes += route_entries(cid, route, rec, native)
        entries.append(entry(cid, f"logLik, public route {route} vs same-run P1 R fit (R vs Julia)",
                             rfit["r_loglik"], rec[route]["loglik"], 1e-6 * abs(rfit["r_loglik"]), R_RULE))
    within = all(e["max_abs_diff"] <= e["tolerance"] for e in entries + routes)
    verdict = "PASS" if all(gates.values()) and within and len(entries) else "FAIL"
    body = {"batch": f"{BRIDGE_TOOL}, GLLVM_PARITY_PIN=P1, run {BRIDGE_BATCH}",
            "measures": (f"{fam}, p={rec['p']}, n={rec['n']}, K={rec['K']}: the frozen public-bridge case re-run live "
                         "at P1 through JuliaCall on the data the same-run native cell fits; toy fixture, "
                         "likelihood-level agreement only"),
            "r_calls": rec["r_calls"], "data_sha256": rec["data_sha256"], "gates": gates,
            "same_run_r_fit": {k: rfit[k] for k in ("file", "sha256", "r_loglik", "r_code", "r_converged",
                                                     "r_gradient_max", "policy")},
            "julia_version": res.get("julia_version"), "gllvmtmb_version": res.get("gllvmtmb_version"),
            "route_consistency": {"note": ROUTE_CONSISTENCY_NOTE, "cases": routes},
            "batch_verifier": verifier_block(BRIDGE_BATCH),
            "read_from": read_from(f"{d}/results.json", f"{d}/run-commit.json", f"{d}/verify.txt",
                                   f"{batch_rel(batch)}/{health}", BRIDGE_CONTRACT, BRIDGE_TOOL),
            "raw": [f"{d}/results.json"]}
    if rec.get("deviation"):
        body["deviation"] = rec["deviation"]
    if verdict == "FAIL":
        body["why_fail"] = [k for k, v in gates.items() if not v] + \
            [e["quantity"] for e in entries + routes if e["max_abs_diff"] > e["tolerance"]]
    return verdict, body, entries


def validate_family11_boundary(raw):
    """Validate boundary evidence only; never turn absence or a failed probe into refusal."""
    def need(ok, message):
        if not ok:
            raise SystemExit(message)
    need(raw.get("schema") == "core070-family11-r-boundary/v1", "FAMILY-11 boundary: wrong raw schema")
    need(raw.get("case_id") == FAMILY11_BOUNDARY_CASE, "FAMILY-11 boundary: wrong case id")
    need(raw.get("pin") == "P1" and raw.get("reference_commit") == P1_SHA,
         "FAMILY-11 boundary: raw probe is not pinned at P1")
    problem = core070_source_pin_check.source_pin_problem(
        {"gllvmTMB_version": raw.get("gllvmtmb_version"), "source_pin": raw.get("source_pin")}, "P1")
    need(problem is None, f"FAMILY-11 boundary: {problem or ''}")
    expected_ns = PINS["P1"]["namespace_sha256"]
    need(raw.get("namespace_sha256") == expected_ns, "FAMILY-11 boundary: installed NAMESPACE hash is not P1")
    build = load(ROOT / ORACLE_BUILD_TOTORO)
    need(raw["source_pin"].get("installed_tree_sha256") == build["installed_tree_sha256"] and
         raw["source_pin"].get("marker_sha256") == build["marker_sha256"],
         "FAMILY-11 boundary: loaded installed build is not the registered Totoro build")
    fixture = raw.get("fixture") or {}
    need((fixture.get("p"), fixture.get("n"), fixture.get("K"), fixture.get("data_sha256")) ==
         (5, 120, 1, FAMILY11_DATA_SHA), "FAMILY-11 boundary: recreated fixture shape/hash mismatch")
    capture = raw.get("capture") or {}
    engine_commit = str(capture.get("glvmodels_commit", ""))
    need(engine_commit.startswith(FAMILY11_GLVMODELS_COMMIT),
         "FAMILY-11 boundary: GLLVModels source commit is not the registered 4b78fa012 source")
    run_commit_path = f"{batch_rel(BRIDGE_BATCH)}/run-commit.json"
    need((ROOT / run_commit_path).is_file(), "FAMILY-11 boundary: existing bridge-p1 run-commit receipt is missing")
    need(engine_commit == load(ROOT / run_commit_path).get("glvmodels_commit"),
         "FAMILY-11 boundary: probe source commit differs from the pinned bridge batch commit")
    need(capture.get("glvmodels_src_tree") == git("rev-parse", f"{engine_commit}:src").stdout.strip(),
         "FAMILY-11 boundary: executed source tree differs from the registered commit")
    need(all(capture.get(k) for k in ("r_version", "julia_version", "glvmodels_path")),
         "FAMILY-11 boundary: R/Julia source runtime provenance is incomplete")
    routes = raw.get("routes") or {}
    need(set(routes) == {"matrix", "formula"}, "FAMILY-11 boundary: expected both public routes")
    for route in ("matrix", "formula"):
        record = routes[route]
        message = str(record.get("message", ""))
        need(record.get("refused") is True and "GJL-GATE-FAMILY" in message,
             f"FAMILY-11 boundary: {route} did not capture GJL-GATE-FAMILY refusal")
        need("error" in record.get("error_class", []),
             f"FAMILY-11 boundary: {route} did not record an R error class")
        need(record.get("call"), f"FAMILY-11 boundary: {route} call string is absent")
    calls = [routes[x]["call"] for x in ("matrix", "formula")]
    need("truncated_nbinom2()" in calls[0] and "num.lv = 1" in calls[0] and
         "truncated_nbinom2()" in calls[1] and "d = 1" in calls[1] and "unique = FALSE" in calls[1] and
         "reversed long" in calls[1], "FAMILY-11 boundary: recorded calls do not match the fixture contract")
    return fixture, capture, routes, calls, expected_ns, engine_commit


def family11_boundary_case():
    """Derive the public R boundary receipt from a live, pinned, no-fit probe."""
    path = ROOT / FAMILY11_BOUNDARY_RAW
    if not path.is_file():
        raise SystemExit(f"{FAMILY11_BOUNDARY_CASE}: raw R probe is missing: {FAMILY11_BOUNDARY_RAW}")
    raw = load(path)
    fixture, capture, routes, calls, expected_ns, engine_commit = validate_family11_boundary(raw)
    run_commit_path = f"{batch_rel(BRIDGE_BATCH)}/run-commit.json"
    body = {
        "batch": "tools/core070_family_bridge_p1.R --family11-boundary (no fit; R public bridge gate capture)",
        "measures": ("Whether both pinned P1 public R bridge routes refuse truncated NB2 at GJL-GATE-FAMILY "
                     "before any numeric fit; fixture recreated from the registered seed-58 recipe."),
        "r_calls": calls,
        "data_sha256": FAMILY11_DATA_SHA,
        "fixture": fixture,
        "routes": routes,
        "r_version": capture.get("r_version"),
        "julia_version": capture.get("julia_version"),
        "gllvmodels_path": capture.get("glvmodels_path"),
        "gllvmodels_source_commit": engine_commit,
        "gllvmtmb_version": raw["gllvmtmb_version"],
        "source_pin": raw["source_pin"],
        "namespace_sha256": expected_ns,
        "read_from": read_from(FAMILY11_BOUNDARY_RAW, run_commit_path, BRIDGE_TOOL, "tools/core070_source_pin.R",
                                "tools/core070_oracle_pins.toml", "test/parity/test_truncated_nbinom2_parity.jl"),
        "raw": [FAMILY11_BOUNDARY_RAW],
    }
    return "r_public_bridge_boundary", "R_BOUNDARY_UNCHANGED", body, None


def family11_boundary_self_test():
    """Positive synthetic fixture plus fail-closed mutations; no R/Julia fit runs."""
    pin = PINS["P1"]
    good = {"schema": "core070-family11-r-boundary/v1", "case_id": FAMILY11_BOUNDARY_CASE,
            "pin": "P1", "reference_commit": P1_SHA, "gllvmtmb_version": pin["version"],
            "source_pin": {k: pin[k] for k in ("reference_commit", "source_tree_sha256", "archive_sha256",
                                                 "namespace_sha256")} | {"version": pin["version"], "installed_tree_sha256": load(ROOT / ORACLE_BUILD_TOTORO)["installed_tree_sha256"], "marker_sha256": load(ROOT / ORACLE_BUILD_TOTORO)["marker_sha256"]},
            "namespace_sha256": pin["namespace_sha256"],
            "fixture": {"p": 5, "n": 120, "K": 1, "data_sha256": FAMILY11_DATA_SHA},
            "capture": {"glvmodels_commit": FAMILY11_GLVMODELS_COMMIT, "glvmodels_src_tree": git("rev-parse", f"{FAMILY11_GLVMODELS_COMMIT}:src").stdout.strip(), "r_version": "R 4.x",
                        "julia_version": "1.10.12", "glvmodels_path": "/fixture/GLLVModels.jl"},
            "routes": {r: {"refused": True, "error_class": ["simpleError", "error", "condition"],
                           "message": "[GJL-GATE-FAMILY] unsupported family",
                       "call": ("gllvm_julia_fit(family=truncated_nbinom2(), num.lv = 1)" if r == "matrix"
                               else "gllvmTMB(data = reversed long, family=truncated_nbinom2(), d = 1, unique = FALSE)")}
                       for r in ("matrix", "formula")}}
    assert validate_family11_boundary(good), "positive synthetic P1 boundary fixture rejected"
    bad = json.loads(json.dumps(good)); bad["routes"]["matrix"]["refused"] = False
    try: validate_family11_boundary(bad)
    except SystemExit: pass
    else: raise AssertionError("NOT_EXECUTED/non-refusal negative control passed")
    bad = json.loads(json.dumps(good)); bad["reference_commit"] = "0" * 40
    try: validate_family11_boundary(bad)
    except SystemExit: pass
    else: raise AssertionError("wrong-pin negative control passed")
    bad = json.loads(json.dumps(good)); bad["fixture"]["data_sha256"] = "0" * 64
    try: validate_family11_boundary(bad)
    except SystemExit: pass
    else: raise AssertionError("mismatched-fixture negative control passed")
    bad = json.loads(json.dumps(good)); bad["routes"]["formula"]["message"] = "unsupported family"
    try: validate_family11_boundary(bad)
    except SystemExit: pass
    else: raise AssertionError("wrong-refusal negative control passed")
    bad = json.loads(json.dumps(good)); bad["capture"]["glvmodels_src_tree"] = "0" * 40
    try: validate_family11_boundary(bad)
    except SystemExit: pass
    else: raise AssertionError("wrong-source-tree negative control passed")
    bad = json.loads(json.dumps(good)); bad["source_pin"]["installed_tree_sha256"] = "0" * 64
    try: validate_family11_boundary(bad)
    except SystemExit: pass
    else: raise AssertionError("wrong-installed-build negative control passed")
    print("CORE070_FAMILY11_BOUNDARY_SELF_TEST_OK (1 positive, 6 rejected mutations)")


def family11_boundary_receipt_path():
    return ROOT / FAMILY11_BOUNDARY_RECEIPT


def alias_case(cid, rec, native):
    d = batch_rel(BRIDGE_BATCH)
    res = bridge_results()
    ra, rc = rec["r_alias"], rec["r_canonical"]
    rule_r = ("adapter equivalence in R: the alias descriptor and gllvmTMB::Beta() through the same native R call "
              "must give the same fit (absolute <= 1e-8)")
    gates = {"native health (converged, |gradient| and FD stability <= 1e-4, objective reconstruction <= 1e-8)":
             native_ok(native),
             "both R fits (alias, canonical) converged (code 0) with a positive-definite Hessian":
             "error" not in ra and "error" not in rc and ra.get("convergence") == 0 and rc.get("convergence") == 0
             and ra.get("pd_hessian") is True and rc.get("pd_hessian") is True}
    entries, alias_r, routes = [], [], []
    if "error" not in ra and "error" not in rc:
        alias_r.append(consistency(cid, "logLik, R alias descriptor vs R gllvmTMB::Beta() (both native TMB)",
                                   ra["loglik"], rc["loglik"], 1e-8, rule_r, ("r_alias", "r_beta")))
        alias_r.append(consistency(cid, "R fixed-effect and loading parameters (opt$par), alias vs gllvmTMB::Beta()",
                                   ra["par"], rc["par"], 1e-8, rule_r, ("r_alias", "r_beta")))
        entries.append(entry(cid, "logLik, R alias fit (TMB) vs fresh native GLLVModels Beta fit (R vs Julia)",
                             ra["loglik"], native["loglik"], 1e-6 * abs(ra["loglik"]),
                             "R vs Julia at the frozen bridge rule's R leg: rtol <= 1e-6 (tolerance = 1e-6 * |R logLik|)"))
    for route in ("matrix", "formula"):
        gates[f"public route {route} with the alias: gllvmTMB_julia object, converged, shape and df match"] = \
            route_ok(rec, route, native)
        if "error" not in rec[route]:
            routes += route_entries(cid, route, rec, native)
    within = all(e["max_abs_diff"] <= e["tolerance"] for e in entries + alias_r + routes)
    verdict = "PASS" if all(gates.values()) and within and entries else "FAIL"
    body = {"batch": f"{BRIDGE_TOOL}, GLLVM_PARITY_PIN=P1, run {BRIDGE_BATCH}",
            "measures": ("the R alias descriptor " + rec["descriptor"] + " on the NATIVE-08-BETA data (p=5, n=60, "
                         "K=1): native R with the alias against native R with gllvmTMB::Beta(); both public bridge "
                         "routes with the alias against a fresh native GLLVModels Beta fit; and the alias R fit "
                         "against that native fit. Toy fixture, likelihood-level agreement only"),
            "acceptance_rule_p0_case_plan": ("Verify the R alias resolves to the same family and information contract; "
                                             "do not create a duplicate Julia spelling merely to copy syntax."),
            "r_calls": rec["r_calls"], "data_sha256": rec["data_sha256"], "gates": gates,
            "r_alias_family": ra.get("family"), "r_alias_message": ra.get("message"),
            "julia_version": res.get("julia_version"), "gllvmtmb_version": res.get("gllvmtmb_version"),
            "alias_identity_r": {"note": ("R vs R: the alias descriptor and gllvmTMB::Beta() through the same native "
                                          "R call. Adapter-equivalence evidence; not R-vs-Julia evidence, so not in "
                                          "the comparison block. Counted in the case verdict, never in the evidence "
                                          "tier."), "cases": alias_r},
            "route_consistency": {"note": ROUTE_CONSISTENCY_NOTE, "cases": routes},
            "batch_verifier": verifier_block(BRIDGE_BATCH),
            "read_from": read_from(f"{d}/results.json", f"{d}/run-commit.json", f"{d}/verify.txt", BRIDGE_TOOL),
            "raw": [f"{d}/results.json"]}
    if verdict == "FAIL":
        body["why_fail"] = [k for k, v in gates.items() if not v] + \
            [e["quantity"] for e in entries + alias_r + routes if e["max_abs_diff"] > e["tolerance"]]
    return verdict, body, entries


def not_executed_case(cid):
    return "NOT_EXECUTED", {"why_not_executed": NOT_EXECUTED[cid],
                            "read_from": read_from(RECONCILIATION)}, None


def check_registration():
    rec = load(ROOT / RECONCILIATION)
    for row in rec["rows"]:
        for ev in row["evidence"]:
            cid = ev["case_id"]
            if cid in REGISTRATION and REGISTRATION[cid][0].startswith("runparity-"):
                cell = REGISTRATION[cid][1]
                if not ev["artifact"].endswith(f"/cell-{cell}.toml"):
                    raise SystemExit(f"{cid}: REGISTRATION names {cell}, reconciliation names {ev['artifact']}")


def own_rows(cm):
    """This tool's rows. Rows carrying a `clause` (the C3 campaign rows) belong to
    tools/true_parity/campaign/write_receipts.py and are never read or rewritten here."""
    return [r for r in cm["rows"] if "clause" not in r]


def foreign_rows(cm):
    return [r for r in cm["rows"] if "clause" in r]


def in_scope_case_ids():
    p0 = {r["source_id"]: r for r in load(P0_CASEMAP)["rows"]}
    cm = load(ROOT / CASEMAP_REL) if (ROOT / CASEMAP_REL).is_file() else None
    sids = [r["source_id"] for r in own_rows(cm)] if cm else []
    return sorted({cid for sid in sids for cid in p0[sid]["executable_case_ids"]})


def derive_case(cid):
    if cid == FAMILY11_BOUNDARY_CASE:
        return family11_boundary_case()
    if cid in NOT_EXECUTED:
        kind, (verdict, body, comp) = "not_executed", not_executed_case(cid)
        return kind, verdict, body, comp
    batch, cell = REGISTRATION[cid]
    if batch in RUNPARITY_RUNS:
        verdict, body, comp = runparity_case(cid, batch, cell)
    elif batch == "family-links-p1":
        verdict, body, comp = family_links_case(cid)
    elif batch == BRIDGE_BATCH:
        verdict, body, comp = bridge_case(cid)
    else:
        verdict, body, comp = a6_case(cid)
    if cid in SAME_MEASUREMENT:
        body["same_measurement_as"] = SAME_MEASUREMENT[cid]
    return "numeric_r_vs_julia", verdict, body, comp


def derive_all(case_ids):
    check_registration()
    return {cid: derive_case(cid) for cid in case_ids}


def tracked_case_receipts():
    """Return ordinary case receipts plus the explicitly owned FAMILY-11 boundary receipt."""
    paths = list((ROOT / REC_REL / "cases").glob("*.json"))
    boundary = ROOT / FAMILY11_BOUNDARY_RECEIPT
    if boundary.is_file():
        paths = [p for p in paths if p.stem != FAMILY11_BOUNDARY_CASE]
        paths.append(boundary)
    return {p.stem: (str(p.relative_to(ROOT)), load(p)) for p in sorted(paths)}


def case_receipt_path(cid):
    return ROOT / (FAMILY11_BOUNDARY_RECEIPT if cid == FAMILY11_BOUNDARY_CASE
                   else f"{REC_REL}/cases/{cid}.json")


# ---------------------------------------------------------------------------
# case-map rows
# ---------------------------------------------------------------------------
def receipt_info(path, rec):
    comp = (rec.get("comparison") or {}).get("cases") or []
    disc = all(e.get("discriminating", True) for e in comp)
    bv = (rec.get("batch_verifier") or {}).get("status", "n/a")
    return (path, rec["evidence_kind"], rec["verdict"], bv, disc)


def p0_evidence(base):
    ev = base.get("evidence") or {}
    if not ev:
        return {"recorded": False}
    return {"recorded": True, "batch": ev.get("batch"), "receipts": ev.get("receipts") or ev.get("harness_receipts"),
            "preservation_sha256": ev.get("preservation_sha256"), "raw_available_in_repo_or_on_this_host": False,
            "note": "P0 receipts live under .unlazy/ (untracked; absent on this host). Only the P0 case map's record "
                    "remains."}


# Maintainer ruling 2026-10-05 (D-319), N1: a PUBLIC-R-BRIDGE case at which R refuses before any Julia call, shown
# by a receipt (evidence_kind r_public_bridge_boundary, verdict R_BOUNDARY_UNCHANGED), is non-binding context; the
# covariance overlay applies the same test. FAMILY-00-IDENTITY and FAMILY-11-LOG are the family rows it names, but
# their bridge cases are NOT_EXECUTED (outside the frozen bridge sub-contract), not an R refusal, so neither binds
# this way at P1 and both stay partial_case_not_executed. A row binds under N1 only when its bridge case has an
# R_BOUNDARY_UNCHANGED receipt and every other case is a passing, discriminating numeric comparison from an accepted
# batch (FAMILY-00's native and formula cases also FAIL on the current receipts). The checker (boundaryContext)
# re-validates the context case.
FAMILY_N1_CONTEXT = {"family/FAMILY-00-IDENTITY", "family/FAMILY-11-LOG"}
N1_CONTEXT_KIND = ("r_public_bridge_boundary", "R_BOUNDARY_UNCHANGED")
N1_TIER = ("numeric: the native and formula-interface case receipts carry R-vs-Julia comparison blocks pinned to P1, "
           "within the harness tolerance, from a batch whose verifier passed; R refuses the PUBLIC-R-BRIDGE case at the "
           "bridge (R_BOUNDARY_UNCHANGED), which is non-binding boundary context under maintainer ruling 2026-10-05 "
           "(D-319), N1")
N1_HELD_NOTE = ("Held under maintainer ruling 2026-10-05 (D-319), N1: the PUBLIC-R-BRIDGE case was not executed, which N1 "
                "does not cover. The row binds once a P1 R probe records the bridge refusal (GJL-GATE-FAMILY) as "
                "R_BOUNDARY_UNCHANGED (tracked follow-up).")


def n1_context(sid, ids, kinds, verdicts, batch_ok, disc):
    """The boundary-context case ids when `sid` binds under N1, else None."""
    if sid not in FAMILY_N1_CONTEXT:
        return None
    ctx = [i for i in ids if kinds[i] != "numeric_r_vs_julia"]
    rest = [i for i in ids if i not in ctx]
    if not ctx or not rest:
        return None
    if not all(i.endswith("-PUBLIC-R-BRIDGE") and (kinds[i], verdicts[i]) == N1_CONTEXT_KIND for i in ctx):
        return None
    if not all(verdicts[i] == "PASS" and batch_ok[i] == "PASS" and disc[i] for i in rest):
        return None
    return ctx


def build_rows(in_scope, carry_status, receipts):
    p0 = {r["source_id"]: r for r in load(P0_CASEMAP)["rows"]}
    counts = {k: 0 for k in COUNT_KEYS}
    out_rows = []
    for sid in in_scope:
        base = p0[sid]
        ids = base["executable_case_ids"]
        row = {"source_id": sid, "classification": base["classification"], "arc": "A3",
               "carry_scan_status": carry_status[sid], "executable_case_ids": ids,
               "disposition": base.get("disposition"), "p0_batch": (base.get("evidence") or {}).get("batch"),
               "p0_evidence": p0_evidence(base)}
        have = [receipts.get(i) for i in ids]
        if not ids or any(h is None for h in have):
            raise SystemExit(f"{sid}: a case id has no receipt")
        kinds = {i: h[1] for i, h in zip(ids, have)}
        verdicts = {i: h[2] for i, h in zip(ids, have)}
        batch_ok = {i: h[3] for i, h in zip(ids, have)}
        disc = {i: h[4] for i, h in zip(ids, have)}
        paths = list(dict.fromkeys(h[0] for h in have))
        measured = [i for i in ids if kinds[i] == "numeric_r_vs_julia"]
        result = {"case_verdicts": verdicts, "batch_verifier": batch_ok, "discriminating": disc}
        if not measured:
            row.update(evidence_tier="not_measured", measured_against=None,
                       evidence={"non_binding_receipts": paths,
                                 "tier": "no case of this row was executed at P1 (see each receipt's "
                                         "why_not_executed)"})
            counts["not_measured"] += 1
        elif len(measured) < len(ids) and (ctx := n1_context(sid, ids, kinds, verdicts, batch_ok, disc)):
            by_id = {i: h[0] for i, h in zip(ids, have)}
            row["boundary_context_case_ids"] = ctx
            row.update(evidence_tier="numeric", measured_against=P1_SHA,
                       evidence={"receipt": list(dict.fromkeys(by_id[i] for i in ids if i not in ctx)),
                                 "boundary_context_receipts": [by_id[i] for i in ctx], "tier": N1_TIER},
                       measured_result={**result, "row_verdict": "PASS"})
            counts["numeric_pass"] += 1
        elif len(measured) < len(ids):
            ok = all(verdicts[i] == "PASS" for i in measured)
            row.update(evidence_tier="partial_case_not_executed", measured_against=P1_SHA,
                       evidence={"non_binding_receipts": paths,
                                 "tier": "some case ids were measured at P1 and at least one was not executed, so "
                                         "the row does not bind"},
                       measured_result={**result, "executed_cases_verdict": "PASS" if ok else "FAIL"})
            counts["partial_case_not_executed"] += 1
            if sid in FAMILY_N1_CONTEXT and ok:
                row["note"] = N1_HELD_NOTE
        elif not all(verdicts[i] == "PASS" for i in ids):
            row.update(evidence_tier="numeric_fail", measured_against=P1_SHA,
                       evidence={"non_binding_receipts": paths,
                                 "tier": "measured at P1; a case verdict is FAIL, so the row does not bind"},
                       measured_result={**result, "row_verdict": "FAIL"})
            counts["numeric_fail"] += 1
        elif not all(v == "PASS" for v in batch_ok.values()):
            row.update(evidence_tier="numeric_held_batch_verifier_failed", measured_against=P1_SHA,
                       evidence={"non_binding_receipts": paths,
                                 "tier": "comparison blocks pass, but a batch verifier rejected the run, so the "
                                         "row does not bind"},
                       measured_result=result)
            counts["numeric_held_batch_verifier_failed"] += 1
        elif not all(disc.values()):
            row.update(evidence_tier="numeric_non_discriminating", measured_against=P1_SHA,
                       evidence={"non_binding_receipts": paths,
                                 "tier": "comparison blocks pass, but at least one is flagged non-discriminating, so "
                                         "the row does not bind"},
                       measured_result=result)
            counts["numeric_non_discriminating"] += 1
        else:
            row.update(evidence_tier="numeric", measured_against=P1_SHA,
                       evidence={"receipt": paths,
                                 "tier": "numeric: every executable case receipt carries R-vs-Julia comparison "
                                         "blocks pinned to P1, within the harness tolerance, from a batch whose "
                                         "verifier passed"},
                       measured_result={**result, "row_verdict": "PASS"})
            counts["numeric_pass"] += 1
        if sid in ROW_NOTES:
            row["note"] = ROW_NOTES[sid]
        out_rows.append(row)
    return out_rows, counts


# ---------------------------------------------------------------------------
# --check
# ---------------------------------------------------------------------------
PROVENANCE_KEYS = {"pin", "reference_commit", "p0_reference_commit", "oracle_build_receipt", "oracle_source_receipt",
                   "glvmodels_commit", "glvmodels_worktree_dirty", "glvmodels_src_tree", "host", "schema", "case_id",
                   "verdict", "evidence_kind", "comparison"}


def receipt_batch(rec):
    cid = rec["case_id"]
    if cid == FAMILY11_BOUNDARY_CASE:
        return BRIDGE_BATCH
    return None if cid in NOT_EXECUTED else REGISTRATION[cid][0]


def check():
    problems = []
    tracked = tracked_case_receipts()
    for cid, (path, rec) in tracked.items():
        for rel, digest in (rec.get("read_from") or {}).items():
            if not (ROOT / rel).is_file():
                problems.append(f"{path}: read file {rel} is gone")
            elif sha(ROOT / rel) != digest:
                problems.append(f"{path}: read file {rel} changed (sha256 {sha(ROOT / rel)[:12]} != {digest[:12]})")
        if not rec.get("read_from"):
            problems.append(f"{path}: no read_from")
        batch = receipt_batch(rec)
        commits = {run_commit(b) for b in all_batches()} if batch is None else {run_commit(batch)}
        if len(commits) != 1 or rec.get("glvmodels_commit") not in commits:
            problems.append(f"{path}: glvmodels_commit {rec.get('glvmodels_commit')} is not the run commit "
                            f"{sorted(commits)} recorded in run-commit.json")
    for b in all_batches():
        derived, _ = verify_text(b)
        text = (ROOT / batch_rel(b) / "verify.txt").read_text()
        if derived and not text.endswith(derived):
            problems.append(f"{b}/verify.txt: derived checks differ from the re-derivation")
    ids = in_scope_case_ids()
    try:
        fresh = derive_all(ids)
    except (SystemExit, KeyError) as e:
        problems.append(f"re-derivation refused: {e}")
        fresh = {}
    for cid in set(tracked) - set(fresh):
        problems.append(f"{cid}: tracked receipt with no re-derived case")
    for cid, (kind, verdict, body, comparison) in fresh.items():
        if cid not in tracked:
            problems.append(f"{cid}: no tracked receipt")
            continue
        rec = tracked[cid][1]
        if (rec["evidence_kind"], rec["verdict"]) != (kind, verdict):
            problems.append(f"{cid}: kind/verdict {rec['evidence_kind']}/{rec['verdict']} != re-derived {kind}/{verdict}")
        if {k: v for k, v in rec.items() if k not in PROVENANCE_KEYS} != body:
            problems.append(f"{cid}: receipt body differs from the re-derivation")
        if (rec.get("comparison") or {}).get("cases") != comparison:
            problems.append(f"{cid}: comparison block differs from the re-derivation")
        if rec.get("reference_commit") != P1_SHA or rec.get("pin") != "P1":
            problems.append(f"{cid}: receipt not pinned at P1")
        if rec.get("oracle_build_receipt") not in ORACLE_BUILDS or rec.get("host") != host_string(rec["oracle_build_receipt"]):
            problems.append(f"{cid}: host {rec.get('host')!r} is not the host text derived from the runs")
    cm = load(ROOT / CASEMAP_REL)
    mine = own_rows(cm)
    receipts = {cid: receipt_info(path, rec) for cid, (path, rec) in tracked.items()}
    try:
        rows, counts = build_rows([r["source_id"] for r in mine],
                                  {r["source_id"]: r["carry_scan_status"] for r in mine}, receipts)
        if cm["rows"][:len(mine)] != mine:
            problems.append("case-map: this tool's rows must precede the campaign rows")
        if rows != mine:
            bad = [a["source_id"] for a, b in zip(rows, mine) if a != b] or ["row count"]
            problems.append(f"case-map rows differ from the re-derivation: {', '.join(bad)}")
        if counts != cm["counts"]:
            problems.append(f"case-map counts {cm['counts']} != re-derived {counts}")
        verifiers = {b: verifier_block(b) for b in all_batches()}
        if verifiers != cm["batch_verifiers"]:
            problems.append("case-map batch_verifiers differ from the re-derivation")
    except (SystemExit, KeyError) as e:
        problems.append(f"case-map re-derivation refused: {e}")
        rows = []
    if problems:
        print("STALE\n  " + "\n  ".join(problems))
        sys.exit(1)
    print("CORE070_FAMILY_P1_RECEIPTS_CURRENT", len(tracked), "case receipts,", len(rows), "rows")


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------
COPY = {
    "family-links-p1": ["receipt.json", "results.tsv", "julia-results.json", "r-oracle.json", "run-commit.json"],
    "a6-p1": ["results.toml", "r-output.tsv", "r-output.tsv.source-pin.tsv", "run-commit.json"],
    BRIDGE_BATCH: ["results.json", "run-commit.json"],
}
NOTE = ("Separate from case-map.json so none of its rows are touched; read by tools/true_parity_check.mjs with "
        "PARITY_CASEMAP pointing at this file. Classification and disposition are carried from "
        "docs/dev-log/core070/required-source-case-map.json unchanged; nothing is signed by an agent. Only rows whose "
        "every executable case id carries passing R-vs-Julia comparison blocks, from a batch whose verifier passed, "
        "with no flagged comparison, cite evidence.receipt. Every cell is a toy fixture (p<=5, n<=250): these rows "
        "measure likelihood-level agreement on one small data set each, not full family parity.")


COMMON_KEYS = ("pin", "reference_commit", "p0_reference_commit", "oracle_build_receipt", "oracle_source_receipt",
               "glvmodels_commit", "glvmodels_worktree_dirty", "glvmodels_src_tree", "host")


def case_receipt(cid, kind, verdict, body, comparison, common):
    rec = {"schema": "core070-family-p1-case-receipt/v1", "case_id": cid, "verdict": verdict,
           "evidence_kind": kind, **body, **common}
    if comparison is not None:
        rec["comparison"] = {"pin": "P1", "cases": comparison}
    return rec


def rederive():
    """Re-derive the case receipts and this tool's case-map rows from the tracked batch
    artifacts, for a change to the derivation only (no new run). Measurement provenance
    (glvmodels_commit = the run commit, oracle build, src tree) is carried from the tracked
    receipts; batch files and verify.txt are not touched, and no verifier is re-run."""
    head, dirty = git_state()
    if dirty:
        raise SystemExit("tracked files are modified outside this tool's outputs; commit the tool change first: "
                         + ", ".join(dirty))
    tracked = {p.stem: load(p) for p in sorted((ROOT / REC_REL / "cases").glob("*.json"))}
    ids = in_scope_case_ids()
    if sorted(tracked) != ids:
        raise SystemExit("tracked case receipts do not match the in-scope case ids; a full write (--runs) is needed")
    receipts = {}
    for cid, (kind, verdict, body, comparison) in derive_all(ids).items():
        old = tracked[cid]
        common = {k: old[k] for k in COMMON_KEYS}
        common["host"] = host_string(old["oracle_build_receipt"])
        rec = case_receipt(cid, kind, verdict, body, comparison, common)
        path = case_receipt_path(cid)
        write_json(path, rec)
        receipts[cid] = receipt_info(str(path.relative_to(ROOT)), rec)
    cm = load(ROOT / CASEMAP_REL)
    mine = own_rows(cm)
    rows, counts = build_rows([r["source_id"] for r in mine], {r["source_id"]: r["carry_scan_status"] for r in mine},
                              receipts)
    cm["counts"], cm["rows"] = counts, rows + foreign_rows(cm)
    cm["rederived"] = {"tool_commit": head,
                       "note": "case receipts and rows re-derived from the tracked batch artifacts (--rederive); "
                               "no batch was re-run, so glvmodels_commit stays the run commit"}
    write_json(ROOT / CASEMAP_REL, cm)
    print(json.dumps(counts))
    print("case receipts re-derived", len(receipts))


def copy_batch(batch, run_dir):
    dest = ROOT / batch_rel(batch)
    dest.mkdir(parents=True, exist_ok=True)
    if batch in RUNPARITY_RUNS:
        names = (["run.toml", "run-commit.json"] + [f"cell-{c}.toml" for c in RUNPARITY_RUNS[batch]]
                 + [f"values-{c}.toml" for c in RUNPARITY_RUNS[batch]] + RUNPARITY_REPORTS)
        required = {"run.toml", "run-commit.json"}
    else:
        names, required = COPY[batch], set(COPY[batch])
    out = []
    for n in names:
        p = run_dir / n
        if p.is_file():
            shutil.copyfile(p, dest / n)
            out.append(f"{batch_rel(batch)}/{n}")
        elif n in required:
            raise SystemExit(f"missing batch artifact {p}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path)
    ap.add_argument("--runtimes", type=Path)
    ap.add_argument("--allow-dirty", action="store_true",
                    help="write receipts from a checkout with modified tracked files (recorded, not hidden)")
    ap.add_argument("--check", action="store_true",
                    help="verify the tracked receipts against the files they read; write nothing")
    ap.add_argument("--rederive", action="store_true",
                    help="re-derive case receipts and rows from the tracked batch artifacts (derivation change only)")
    ap.add_argument("--family11-boundary-self-test", action="store_true",
                    help="run fixture-based positive/negative controls without R, JuliaCall, or any fit")
    args = ap.parse_args()
    if args.family11_boundary_self_test:
        family11_boundary_self_test()
        return
    if args.check:
        check()
        return
    if args.rederive:
        rederive()
        return
    if args.runs is None or args.runtimes is None:
        ap.error("--runs and --runtimes are required unless --check")
    runs, runtimes = args.runs, load(args.runtimes)
    head, dirty = git_state()
    if dirty and not args.allow_dirty:
        raise SystemExit("tracked files are modified outside this tool's outputs; commit first or pass "
                         "--allow-dirty: " + ", ".join(dirty))
    for b in all_batches():
        check_run_commit(runs / b, head)
    previous = load(ROOT / CASEMAP_REL) if (ROOT / CASEMAP_REL).is_file() else {"rows": []}
    # Stale receipts from an earlier write must not survive; receipts/family/campaign/ is not this tool's.
    for sub in ["cases", *all_batches()]:
        shutil.rmtree(ROOT / REC_REL / sub, ignore_errors=True)
    artifacts = {}
    for b in all_batches():
        artifacts[b] = copy_batch(b, runs / b)
        external = run_external(b, runs / b) if b in EXTERNAL else ""
        derived, _ = verify_text(b)
        (ROOT / batch_rel(b) / "verify.txt").write_text(external + (SEPARATOR + derived if derived else ""))
        artifacts[b].append(f"{batch_rel(b)}/verify.txt")
    carry = load(runs / "carry-scan-p1.json")
    carry_status = {r["source_id"]: r["status"] for r in carry["rows"]
                    if r["source_id"].startswith("family/") and r["status"] in IN_SCOPE_STATUS}
    p0 = {r["source_id"]: r for r in load(P0_CASEMAP)["rows"]}
    in_scope = [s for s in carry_status if p0[s]["classification"] in ("required_core", "compatibility_adapter")]
    ids = sorted({cid for sid in in_scope for cid in p0[sid]["executable_case_ids"]})
    used = {load_toml(ROOT / batch_rel(b) / "run.toml").get("source", {}).get("oracle_build_receipt_sha256")
            for b in RUNPARITY_RUNS}
    oracle_build = [b for b in ORACLE_BUILDS if {sha(ROOT / b)} == used]
    if len(oracle_build) != 1:
        raise SystemExit(f"runparity runs do not name one registered oracle build receipt: {used}")
    common = {"pin": "P1", "reference_commit": P1_SHA, "p0_reference_commit": P0_SHA,
              "oracle_build_receipt": oracle_build[0], "oracle_source_receipt": ORACLE_SOURCE,
              "glvmodels_commit": head, "glvmodels_worktree_dirty": dirty,
              "glvmodels_src_tree": git("rev-parse", f"{head}:src").stdout.strip(),
              "host": host_string(oracle_build[0])}
    receipts = {}
    for cid, (kind, verdict, body, comparison) in derive_all(ids).items():
        rec = case_receipt(cid, kind, verdict, body, comparison, common)
        path = case_receipt_path(cid)
        write_json(path, rec)
        receipts[cid] = receipt_info(str(path.relative_to(ROOT)), rec)
    rows, counts = build_rows(in_scope, carry_status, receipts)
    by_status = {s: sum(carry_status[r] == s for r in in_scope) for s in IN_SCOPE_STATUS}
    write_json(ROOT / CASEMAP_REL, {
        "schema": 1, "reference_commit": P1_SHA,
        "scope": (f"family family: the {len(in_scope)} required rows (required_core, plus the compatibility_adapter "
                  f"row FAMILY-BETA-ALIAS, which tools/true_parity_check.mjs counts as required) the P1 carry scan "
                  f"lists as {' or '.join(f'{s} ({n})' for s, n in by_status.items())}. FAMILY-16-LOGIT "
                  f"(NOT_BOUND_AT_P0, empty case list) and the rejected and excluded rows are out of scope."),
        "note": NOTE, "generator": "tools/core070_family_p1_receipts.py", "glvmodels_commit": head,
        "batch_verifiers": {b: verifier_block(b) for b in all_batches()}, "counts": counts,
        "p0_evidence_summary": {
            "rows_with_p0_evidence_record": sum(r["p0_evidence"]["recorded"] for r in rows),
            "rows_without_p0_evidence_record": sum(not r["p0_evidence"]["recorded"] for r in rows),
            "p0_raw_receipts_available_here": False},
        "runtimes_seconds": runtimes, "batch_artifacts": artifacts, "rows": rows + foreign_rows(previous)})
    print(json.dumps(counts))
    print("case receipts", len(receipts))


if __name__ == "__main__":
    main()
