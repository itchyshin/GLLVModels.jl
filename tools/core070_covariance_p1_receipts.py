"""Write tracked P1 receipts and case-map rows for the covariance family.

Reads the raw outputs of the covariance batches run at gllvmTMB pin P1
(GLLVM_PARITY_PIN=P1) and writes, under
docs/dev-log/core070/true-parity-latest/:

  receipts/covariance/<batch>/...     batch artifacts copied verbatim (text only)
  receipts/covariance/cases/<id>.json one receipt per executable case id
  case-map-covariance.json            the 17 covariance rows the P1 carry scan
                                      lists as DANGLING (10) or PARTIAL_STALE_AT_P1 (7)

Numbers in each `comparison` block are recomputed here from the raw R and
Julia values in the batch output, with the tolerance each harness itself
declares (never a wider one). A harness check written as
`isapprox(a, b; atol, rtol)` is judged as harness_norm_diff = norm(a - b) against
tolerance = max(atol, rtol * max(norm(a), norm(b))), which is exactly the
bound isapprox applies; its abs_diff records the max absolute entrywise
difference, the value tools/true_parity_check.mjs recomputes from r_value and
julia_value. A scalar `abs(a - b) <= tol` check is reported as-is.

A row is marked evidence_tier "numeric" only when every one of its
executable_case_ids has a comparison block within tolerance AND every batch
those cases came from passed its own verifier (runparity: run status success
under the tracked P1 manifest; wave6 / covariance batch: the batch verifier
script, run by this tool, exit 0). A row whose numeric cases pass but whose
batch verifier rejected the run is held back (evidence_tier
"numeric_held_batch_verifier_failed", receipts under
evidence.non_binding_receipts) unless --numeric-exceptions names a signed
exception for that row whose signed_by is the maintainer (MAINTAINER below)
and whose signed_on is a date; this tool never writes such an exception.
Rows whose case ids include an R-only or R-boundary case are marked
otherwise, and their receipts are cited under evidence.non_binding_receipts
(not evidence.receipt), so no checker version counts them as bound.

Provenance (review finding 7): each receipt records glvmodels_commit = HEAD of
this checkout and glvmodels_worktree_dirty = the tracked paths modified
outside the output directory. The tool refuses to write when that list is
non-empty (unless --allow-dirty), and refuses when any harness file listed in
the runparity run's execution inventory differs from its content at HEAD, so
the recorded commit is the one the runs used. Run the batches and this tool
from the same clean commit.

Usage (inputs are the raw run directories, e.g. under local-scratch):
  python3 tools/core070_covariance_p1_receipts.py \
      --runparity DIR --default-modes DIR --wave6 DIR --cov-batch DIR \
      --bridge-tsv FILE --oracle-dir DIR --runtimes JSON \
      [--numeric-exceptions JSON] [--allow-dirty]

Julia fit-level twins (no run directory needed; see the overlay block above main()):
  python3 tools/core070_covariance_p1_receipts.py --apply-twins   # re-derive twin rows and counts
  python3 tools/core070_covariance_p1_receipts.py --check         # tracked case map == derivation
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/dev-log/core070/true-parity-latest"
REC = OUT / "receipts/covariance"
REC_REL = "docs/dev-log/core070/true-parity-latest/receipts/covariance"
P0_CASEMAP = ROOT / "docs/dev-log/core070/required-source-case-map.json"
PINS = tomllib.loads((ROOT / "tools/core070_oracle_pins.toml").read_text())
P1_SHA = PINS["P1"]["reference_commit"]
P0_SHA = PINS["P0"]["reference_commit"]
P1_MANIFEST = OUT / "frozen-r070-contract-p1.toml"
MAINTAINER = "Shinichi Nakagawa"
HELD_NOTE = ("held pending the maintainer's ruling on the wave6 nobs expectation; the covariance cases "
             "themselves pass (7.25e-8, 1.26e-6 vs tol 1e-4)")

# The 17 rows: PARTIAL_STALE_AT_P1 (7) + DANGLING (10) in the P1 carry scan (PR #534 / branch
# claude/true-parity-p1-carry, carry-scan-p1.json), family covariance.
PARTIAL_STALE = ["COV-ANIMAL-DEP", "COV-ANIMAL-INDEP", "COV-KERNEL-DEP", "COV-KERNEL-INDEP",
                 "COV-ORD-DEP", "COV-ORD-INDEP", "COV-ORD-INDEP-COMMON"]
DANGLING = ["COV-KERNEL-FOLDED-UNIQUE", "COV-KERNEL-LATENT", "COV-META-EXACT", "COV-META-LEGACY",
            "COV-ORD-LATENT-BARE", "COV-ORD-LATENT-COMMON", "COV-ORD-LATENT-DEFAULT",
            "COV-PHYLO-A-ALIAS", "COV-PHYLO-DEP", "COV-PHYLO-FOLDED-UNIQUE"]

# Public-R-bridge case id -> the fixture id the boundary runner prints.
BRIDGE_FIXTURE_ID = {
    "MODE-ORD-INDEP-PUBLIC-R-BRIDGE": "MODE-ORD-INDEP",
    "MODE-ORD-COMMON-PUBLIC-R-BRIDGE": "MODE-ORD-COMMON",
    "FIT-MODE-ORD-DEP-PUBLIC-R-BRIDGE": "MODE-ORD-DEP",
    **{f"FIT-MODE-{s}-{m}-PUBLIC-R-BRIDGE": f"MODE-{s}-{m}"
       for s in ("ANIMAL", "KERNEL") for m in ("INDEP", "COMMON", "DEP")},
}
BRIDGE_EXPECTED_GATE = {k: ("EARLY-GENERIC-ERROR" if v == "MODE-ORD-DEP" else "GJL-GATE-STRUCTURED-TERMS")
                        for k, v in BRIDGE_FIXTURE_ID.items()}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def norm(v):
    flat = [x for row in v for x in row] if v and isinstance(v[0], list) else list(v)
    return math.sqrt(sum(x * x for x in flat))


def sub(a, b):
    if a and isinstance(a[0], list):
        return [[x - y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]
    return [x - y for x, y in zip(a, b)]


def add_diag(m, d):
    return [[v + (d if i == j else 0.0) for j, v in enumerate(row)] for i, row in enumerate(m)]


def scalar_entry(case_id, quantity, r, j, tol, rule):
    return {"case_id": case_id, "quantity": quantity, "r_value": r, "julia_value": j,
            "abs_diff": abs(r - j), "tolerance": tol, "tolerance_rule": rule}


def flat(v):
    return [x for row in v for x in row] if v and isinstance(v[0], list) else list(v)


def isapprox_entry(case_id, quantity, r, j, atol, rtol, rule):
    # abs_diff is the max absolute entrywise difference, the quantity tools/true_parity_check.mjs
    # recomputes from r_value/julia_value (as tools/core070_family_p1_receipts.py records it).
    # The harness's own test is unchanged: norm(julia - r) <= max(atol, rtol*max(norm)), kept
    # as harness_norm_diff and judged against the same tolerance. max-abs <= norm, so a case that
    # passes the harness test also passes the checker's max-abs test.
    tol = max(atol, rtol * max(norm(j), norm(r)))
    return {"case_id": case_id, "quantity": quantity, "r_value": r, "julia_value": j,
            "abs_diff": max(abs(a - b) for a, b in zip(flat(j), flat(r), strict=True)),
            "harness_norm_diff": norm(sub(j, r)), "tolerance": tol,
            "tolerance_rule": f"{rule}: isapprox(julia, r; atol={atol:g}, rtol={rtol:g}); "
                              "harness test norm(julia - r) <= tolerance = max(atol, rtol*max(norm)) "
                              "(harness_norm_diff); abs_diff = max |julia - r| entrywise"}


# ---- comparisons per harness ---------------------------------------------------------------

def modes_entries(row):
    cid, r, n = row["id"], row["r"], row["native"]
    src = "tools/core070_covariance_mode_fits.jl (tight-control)"
    out = [scalar_entry(cid, "loglik", r["loglik"], n["loglik"], 1e-6, f"{src}: abs(native.loglik - r.loglik) <= 1e-6"),
           isapprox_entry(cid, "beta", r["beta"], n["beta"], 1e-5, 1e-5, src)]
    if row["source"] == "ORD":
        out.append(isapprox_entry(cid, "total_covariance_U_plus_sigma2_I",
                                  add_diag(r["covariance"], r["residual_variance"]),
                                  add_diag(n["source_covariance"], n["residual_variance"]), 1e-5, 1e-5,
                                  src + " (ordinary source: U and sigma^2 not separately identified)"))
    else:
        out.append(isapprox_entry(cid, "source_covariance", r["covariance"], n["source_covariance"], 1e-5, 1e-5, src))
        out.append(isapprox_entry(cid, "residual_variance", [r["residual_variance"]], [n["residual_variance"]],
                                  1e-5, 1e-5, src))
    return out


def fixed_entries(row):
    cid = row["id"]
    src = "tools/core070_source_fixed_residual_pair.jl"
    rcov = [[row["r_source_sd"][i] ** 2 if i == j else 0.0 for j in range(3)] for i in range(3)]
    ncov = [[row["native_source_variance"][i] if i == j else 0.0 for j in range(3)] for i in range(3)]
    return [scalar_entry(cid, "loglik", row["r_loglik"], row["native_loglik"], 1e-6, f"{src}: abs(native.loglik - rll) <= 1e-6"),
            isapprox_entry(cid, "beta", row["r_beta"], row["native_beta"], 1e-5, 1e-5, src),
            isapprox_entry(cid, "trait_covariance", rcov, ncov, 1e-5, 1e-5, src + " (residual SD fixed by the reference rule)")]


def formula_entries(row):
    cid = row["id"]
    r = row["expected_base"]["r"]
    src = "test/parity/covariance_formula_cases.jl"
    ordinary_dep = row["source"] == "ORD" and row["mode"] == "DEP"
    out = []
    for route in ("wide", "long"):
        f = row[route]
        out.append(scalar_entry(cid, f"{route}_loglik_to_R", r["loglik"], f["loglik"], 1e-6,
                                f"{src}: abs({route}.loglik - r_loglik) <= 1e-6"))
        out.append(isapprox_entry(cid, f"{route}_beta_to_R", r["beta"], f["beta"], 1e-5, 1e-5, src))
        if ordinary_dep:
            out.append(isapprox_entry(cid, f"{route}_total_covariance_to_R",
                                      add_diag(r["source_covariance"], r["residual_sd"] ** 2),
                                      add_diag(f["source_covariance"], f["residual_sd"] ** 2), 1e-5, 1e-5,
                                      src + " (ordinary dependent: total covariance only)"))
        else:
            out.append(isapprox_entry(cid, f"{route}_source_covariance_to_R", r["source_covariance"],
                                      f["source_covariance"], 1e-5, 1e-5, src))
            if not f.get("residual_fixed"):
                out.append(isapprox_entry(cid, f"{route}_residual_variance_to_R", [r["residual_sd"] ** 2],
                                          [f["residual_sd"] ** 2], 1e-5, 1e-5, src))
    return out


def wave6_entry(case_id, julia, oracle, contract_case):
    rv = oracle["oracle_values"][case_id]
    jv = julia["cases"][case_id]["julia_values"]
    diff = max(abs(a - b) for a, b in zip(rv, jv, strict=True))
    return {"case_id": case_id, "quantity": contract_case["quantity"], "r_value": rv, "julia_value": jv,
            "max_abs_diff": diff, "tolerance": contract_case["tolerance"],
            "tolerance_rule": "wave6-conversion-batch-contract-p1.json per-case tolerance (carried verbatim from P0); "
                              "max |r - julia| over the logLik + tcrossprod(B) vector"}


# ---- writers ---------------------------------------------------------------------------------

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def copy_batch(src_dir, dest_name, patterns):
    dest = REC / dest_name
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for pat in patterns:
        for p in sorted(Path(src_dir).glob(pat)):
            if p.is_file():
                rel = p.relative_to(src_dir)
                target = dest / str(rel).replace("/", "__")
                shutil.copyfile(p, target)
                copied.append(str(target.relative_to(ROOT)))
    return copied


def git(*args, text=True):
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True, text=text).stdout


def git_state():
    """HEAD and the tracked paths modified outside the output directory (review finding 7)."""
    head = git("rev-parse", "HEAD").strip()
    out_rel = str(OUT.relative_to(ROOT)) + "/"
    dirty = [line[3:] for line in git("status", "--porcelain", "--untracked-files=no").splitlines()
             if not line[3:].startswith(out_rel)]
    return head, dirty


def harness_drift(run, head):
    """Execution-inventory files of the runparity run whose bytes differ from HEAD."""
    tracked = set(git("ls-tree", "-r", "--name-only", head).splitlines())
    drift = []
    for entry in run["execution"]["entries"]:
        if entry["path"] not in tracked:
            continue  # untracked inputs (Manifest.toml) are not part of the commit
        blob = git("show", f"{head}:{entry['path']}", text=False)
        if hashlib.sha256(blob).hexdigest() != entry["sha256"]:
            drift.append(entry["path"])
    return drift


def run_verifier(argv, log_name, marker):
    """Run a batch verifier at P1, keep its full output as the tracked verify.log."""
    proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True,
                          env=dict(os.environ, GLLVM_PARITY_PIN="P1"))
    log = REC / log_name
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(proc.stdout + proc.stderr)
    ok = proc.returncode == 0 and marker in proc.stdout
    return {"tool": " ".join(argv[1:2]), "status": "PASS" if ok else "FAIL", "exit_code": proc.returncode,
            "log": f"{REC_REL}/{log_name}"}


def load_exceptions(path):
    """Signed numeric exceptions, keyed by source_id; only the maintainer's signature counts."""
    if path is None:
        return {}
    table = json.loads(path.read_text())
    for sid, exc in table.items():
        if exc.get("signed_by") != MAINTAINER or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(exc.get("signed_on", ""))):
            raise SystemExit(f"numeric exception for {sid} is not signed by {MAINTAINER} with a signed_on date")
        if not str(exc.get("reason", "")).strip():
            raise SystemExit(f"numeric exception for {sid} has no reason")
    return table


# ---- Julia fit-level twins (overlay) -----------------------------------------------------------
# A row whose batch case is an R-only formula-grammar check is bound instead by a fit-level twin
# when one exists: R-at-P1 fits recorded in a tracked fixture against Julia fits of the same data,
# receipts written by tools/true_parity_julia_receipts.jl under receipts/julia-twins/<dir>/. The
# overlay (the pattern of tools/core070_data_p1_receipts.py) sets executable_case_ids to the twin's
# case ids, cites the twin receipt under evidence.receipt, and keeps the formula-grammar receipt
# under evidence.non_binding_receipts with its ids under evidence.batch_case_ids. Classification
# and disposition are never touched. `--apply-twins` re-derives the twin rows and the counts of
# the tracked case-map-covariance.json from the tracked receipts (no run directory needed);
# `--check` verifies that the tracked file equals that derivation. Each block below is one twin
# family; add a new family as its own block.
TWIN_ROOT_REL = "docs/dev-log/core070/true-parity-latest/receipts/julia-twins"
COV_TWINS = {}      # source_id -> (receipt path under TWIN_ROOT_REL, twin test, fixture)
COV_SCOPE_NOTES = {}  # source_id -> what the twin covers and does not

# COV-PHYLO twins (wave-plan W3-4(e)): test/test_cov_phylo_twins_p1.jl,
# fixture test/fixtures/cov_phylo_twins_p1.toml (gen_cov_phylo_twins_p1.R).
_PHYLO_TWIN = ("test/test_cov_phylo_twins_p1.jl", "test/fixtures/cov_phylo_twins_p1.toml")
COV_TWINS.update({
    "covariance/COV-PHYLO-DEP": ("covariance-twins/PHYLO-DEP.json", *_PHYLO_TWIN),
    "covariance/COV-PHYLO-A-ALIAS": ("covariance-twins/PHYLO-A-ALIAS.json", *_PHYLO_TWIN),
    "covariance/COV-PHYLO-FOLDED-UNIQUE": ("covariance-twins/PHYLO-FOLDED-UNIQUE.json", *_PHYLO_TWIN),
})
COV_SCOPE_NOTES.update({
    "covariance/COV-PHYLO-DEP": (
        "The batch case checks that phylo_dep(0 + trait | species) parses to phylo_rr(d = n_traits, .dep = TRUE); "
        "the twin is one Gaussian fit of that term (tree route, 120 tips, 4 traits) against "
        "fit_phylo_latent_gllvm(d = 4), the engine path gllvmTMB documents phylo_dep as. Julia has no phylo_dep "
        "keyword of its own on this route (fit_phylo_dep_gllvm in src/phylo_dep.jl is a different, row-phylogeny "
        "Gaussian model and is not the twin). Julia's stop reports converged = false (max |FD gradient| 4.0e-5 "
        "against g_tol 1e-5) at a Newton decrement of 2.8e-12; the twin test asserts that bound and a "
        "positive-definite Hessian instead of the flag. Gaussian only; the .dep guards (phylo_dep with "
        "phylo_latent or phylo_indep refused) are not twinned."),
    "covariance/COV-PHYLO-A-ALIAS": (
        "The batch case checks that phylo_latent(species, A = A) parses to phylo_rr(vcv = A); the twin is one "
        "Gaussian rank-1 fit with A = A (dense route) in R and Julia, and both engines give the identical result "
        "for the vcv = A spelling. Gaussian, d = 1, rho = 1 only."),
    "covariance/COV-PHYLO-FOLDED-UNIQUE": (
        "The batch case checks that phylo_latent(species, unique = TRUE) parses to phylo_rr(d = 1) plus the folded "
        ".phylo_unique/.auto_unique companion; the twin is one Gaussian fit of that term (tree route) against "
        "fit_phylo_latent_gllvm(d = 1, unique = true). R's companion is the phylo_diag block (per-trait field on "
        "the same A, sd exp(log_sd_phy_diag)) and Julia's :explicitunique has the same covariance; Julia's "
        "objective at R's optimum equals R's objective within 1.3e-11, which is the evidence that the two "
        "likelihoods are the same model. Gaussian, d = 1, rho = 1 only; the duplicate-companion and "
        "off-family guards are not twinned."),
})


def twin_tier(sid):
    _, test, fixture = COV_TWINS[sid]
    return ("numeric: Julia values recomputed by tools/true_parity_julia_receipts.jl with the same calls and settings "
            f"as {test}, against R-at-P1 values copied from {fixture} (R fits that converged with a positive-definite "
            "Hessian), each case within the tolerance asserted in that test. The formula-grammar batch case this row "
            "carried has no fit number; its receipt is kept under non_binding_receipts and its id under batch_case_ids")


def twin_overlay(row):
    """Bind `row` to its Julia twin receipt (idempotent: an already overlaid row is re-derived)."""
    sid = row["source_id"]
    rel = f"{TWIN_ROOT_REL}/{COV_TWINS[sid][0]}"
    path = ROOT / rel
    if not path.is_file():
        return
    rec = json.loads(path.read_text())
    ok = (rec.get("schema") == "true-parity-julia-twin-receipt/v1" and rec.get("source_ids") == [sid]
          and rec.get("verdict") == "PASS" and rec.get("pin") == "P1" and rec.get("reference_commit") == P1_SHA
          and rec.get("evidence_kind") == "julia_recomputed_vs_recorded_r"
          and all(c["abs_diff"] <= c["tolerance"] for c in rec["comparison"]["cases"]))
    if not ok:
        raise SystemExit(f"{rel}: not a passing P1 Julia twin receipt for {sid}")
    ev = row.get("evidence") or {}
    mr = row.get("measured_result") or {}
    prior_ids = ev.get("batch_case_ids", row["executable_case_ids"])
    if row["evidence_tier"] not in ("r_only", "numeric") or (row["evidence_tier"] == "numeric" and "batch_case_ids" not in ev):
        raise SystemExit(f"{sid}: twin overlay expects an r_only row (or one it already overlaid); got {row['evidence_tier']}")
    case_ids = [c["case_id"] for c in rec["comparison"]["cases"]]
    row["executable_case_ids"] = case_ids
    row["evidence_tier"] = "numeric"
    row["measured_against"] = P1_SHA
    row["evidence"] = {"receipt": [rel], "non_binding_receipts": ev.get("non_binding_receipts", []),
                       "batch_case_ids": prior_ids, "tier": twin_tier(sid)}
    row["measured_result"] = {"case_verdicts": mr.get("case_verdicts", {}), "twin_case_ids": case_ids,
                              "twin_verdict": rec["verdict"]}
    if sid in COV_SCOPE_NOTES:
        row["scope_note"] = COV_SCOPE_NOTES[sid]


def recount(rows):
    """counts over this tool's own rows (PARTIAL_STALE + DANGLING), from their tiers."""
    own = {f"covariance/{s}" for s in PARTIAL_STALE + DANGLING}
    counts = {"numeric_pass": 0, "numeric_fail": 0, "numeric_held_batch_verifier_failed": 0,
              "partial_numeric_bridge_boundary": 0, "r_only_needs_julia_surface": 0, "not_measured": 0}
    key = {"numeric_held_batch_verifier_failed": "numeric_held_batch_verifier_failed",
           "partial_numeric_bridge_boundary": "partial_numeric_bridge_boundary",
           "r_only": "r_only_needs_julia_surface", "not_measured": "not_measured"}
    for r in rows:
        if r["source_id"] not in own:
            continue
        if r["evidence_tier"] == "numeric":
            verdict = r["measured_result"].get("row_verdict", r["measured_result"].get("twin_verdict"))
            counts["numeric_pass" if verdict == "PASS" else "numeric_fail"] += 1
        else:
            counts[key[r["evidence_tier"]]] += 1
    return counts


def derive_twins(casemap):
    out = json.loads(json.dumps(casemap))
    for row in out["rows"]:
        if row["source_id"] in COV_TWINS:
            twin_overlay(row)
    out["counts"] = recount(out["rows"])
    return out


def apply_twins(check_only):
    path = OUT / "case-map-covariance.json"
    tracked = json.loads(path.read_text())
    derived = derive_twins(tracked)
    if check_only:
        if derived != tracked:
            raise SystemExit("case-map-covariance.json differs from the twin derivation; run --apply-twins")
        print("CORE070_COVARIANCE_TWINS_OK", json.dumps(derived["counts"]))
        return
    write_json(path, derived)
    print(json.dumps(derived["counts"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runparity", type=Path, required=True)
    ap.add_argument("--default-modes", type=Path, required=True)
    ap.add_argument("--wave6", type=Path, required=True)
    ap.add_argument("--cov-batch", type=Path, required=True)
    ap.add_argument("--bridge-tsv", type=Path, required=True)
    ap.add_argument("--oracle-dir", type=Path, required=True)
    ap.add_argument("--runtimes", type=Path, required=True)
    ap.add_argument("--numeric-exceptions", type=Path, default=None,
                    help="JSON {source_id: {signed_by, signed_on, reason}}; signed_by must be the maintainer")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="write receipts from a checkout with modified tracked files (recorded, not hidden)")
    args = ap.parse_args()
    runtimes = json.loads(args.runtimes.read_text())
    exceptions = load_exceptions(args.numeric_exceptions)
    head, dirty = git_state()
    if dirty and not args.allow_dirty:
        raise SystemExit("tracked files are modified outside the output directory; commit first or pass "
                         "--allow-dirty: " + ", ".join(dirty))
    run = tomllib.loads((args.runparity / "run.toml").read_text())
    drift = harness_drift(run, head)
    if drift:
        raise SystemExit(f"the runparity run's harness files differ from HEAD {head}; re-run at HEAD: " + ", ".join(drift))

    # batch artifacts (text only; .rds fit objects stay in the local run directory)
    oracle_files = copy_batch(args.oracle_dir / "source", "oracle", ["source.json"]) + \
        copy_batch(args.oracle_dir / "build", "oracle", ["build.json"])
    rp_files = copy_batch(args.runparity, "runparity-covariance-18", ["run.toml", "cell-*.toml", "build.json", "*/result.toml"])
    dm_files = copy_batch(args.default_modes, "mode-fits-default-control", ["result.toml"])
    w6_files = copy_batch(args.wave6, "wave6-conversion-p1",
                          ["*.json", "diagnostics.log", "julia-stderr.log", "julia-stdout.log", "*.tsv"])
    cb_files = copy_batch(args.cov_batch, "covariance-batch-p1", ["covariance-batch-results.json"])
    br_files = copy_batch(args.bridge_tsv.parent, "bridge-boundary-p1", [args.bridge_tsv.name])

    # batch verifiers (finding 1): each batch's own acceptance, recorded on every case receipt
    manifest_sha = sha(P1_MANIFEST)
    rp_ok = run.get("status") == "success" and run.get("exit_code") == 0 and run.get("contract_sha256") == manifest_sha
    verifiers = {
        "runparity": {"tool": "test/parity/runparity.jl (run.toml status, exit code, manifest sha)",
                      "status": "PASS" if rp_ok else "FAIL",
                      "detail": f"status={run.get('status')} exit_code={run.get('exit_code')} "
                                f"contract_sha256 {'==' if run.get('contract_sha256') == manifest_sha else '!='} "
                                f"sha256(frozen-r070-contract-p1.toml)"},
        "wave6": run_verifier(["python3", "tools/core070_verify_wave6_conversion_batch.py", str(args.wave6)],
                              "wave6-conversion-p1/verify.log", "CORE070_WAVE6_CONVERSION_STATE_OK"),
        "cov_batch": run_verifier(["python3", "tools/core070_verify_covariance_batch.py", "--results",
                                   str(args.cov_batch / "covariance-batch-results.json"), "--self-test"],
                                  "covariance-batch-p1/verify.log", "CORE070_COVARIANCE_BATCH_VERIFIED"),
    }
    batch_common = {"pin": "P1", "reference_commit": P1_SHA, "gllvmtmb_version": "0.7.1",
                    "oracle_build_receipt": f"{REC_REL}/oracle/build.json",
                    "oracle_source_receipt": f"{REC_REL}/oracle/source.json",
                    "glvmodels_commit": head, "glvmodels_worktree_dirty": dirty,
                    "host": "local Mac (M1 Ultra), single BLAS/OMP thread, JULIA_NUM_THREADS=4",
                    "p0_reference_commit": P0_SHA}

    receipts = {}  # case_id -> (path, kind)

    # 1. native + formula (runparity, 18 cases)
    modes = tomllib.loads((args.runparity / "covariance-modes-raw/result.toml").read_text())
    fixed = tomllib.loads((args.runparity / "covariance-fixed-raw/result.toml").read_text())
    fmodes = tomllib.loads((args.runparity / "covariance-formula-modes-raw/result.toml").read_text())
    ffixed = tomllib.loads((args.runparity / "covariance-formula-fixed-raw/result.toml").read_text())
    rows = [(r, modes_entries(r), "covariance-modes-raw/result.toml", modes["all_checks"]) for r in modes["cases"]] + \
           [(r, fixed_entries(r), "covariance-fixed-raw/result.toml", all(all(x["checks"].values()) for x in fixed["cases"])) for r in fixed["cases"]] + \
           [(r, formula_entries(r), "covariance-formula-modes-raw/result.toml", fmodes["all_checks"]) for r in fmodes["cases"]] + \
           [(r, formula_entries(r), "covariance-formula-fixed-raw/result.toml", ffixed["all_checks"]) for r in ffixed["cases"]]
    for r, entries, raw, group_ok in rows:
        cid = r["id"]
        cell = tomllib.loads((args.runparity / f"cell-{cid}.toml").read_text())
        within = all(e.get("harness_norm_diff", e["abs_diff"]) <= e["tolerance"] for e in entries)
        checks = r["checks"]
        receipt = {
            "schema": "core070-covariance-p1-case-receipt/v1", "case_id": cid,
            "verdict": "PASS" if within and all(checks.values()) and cell["status"] == "success" else "FAIL",
            "evidence_kind": "numeric_r_vs_julia",
            "batch": "test/parity/runparity.jl, CORE070_PARITY_REQUIRED=1, GLLVM_PARITY_PIN=P1, 18 covariance case ids",
            "raw": f"{REC_REL}/runparity-covariance-18/{raw.replace('/', '__')}",
            "cell": f"{REC_REL}/runparity-covariance-18/cell-{cid}.toml",
            "cell_assertions": cell.get("assertions"),
            "run_contract_sha256": run["contract_sha256"],
            "harness_checks_all_true": all(checks.values()), "harness_group_all_checks": group_ok,
            "batch_verifier": verifiers["runparity"],
            **batch_common,
            "comparison": {"pin": "P1", "cases": entries},
        }
        path = REC / "cases" / f"{cid}.json"
        write_json(path, receipt)
        receipts[cid] = (str(path.relative_to(ROOT)), "numeric", receipt["verdict"], verifiers["runparity"]["status"])

    # 2. wave6 kernel_latent cases (the two covariance ones)
    w6_contract = json.loads((OUT / "wave6-conversion-batch-contract-p1.json").read_text())
    w6_julia = json.loads((args.wave6 / "julia-results.json").read_text())
    w6_oracle = json.loads((args.wave6 / "r-oracle.json").read_text())
    w6_receipt = json.loads((args.wave6 / "receipt.json").read_text())
    for cid in ("CORE070-WAVE6-KERNEL-LATENT-SINGLE-PSI-COVARIANCE", "CORE070-WAVE6-KERNEL-LATENT-MULTI-NAMESPACE"):
        cc = next(c for c in w6_contract["cases"] if c["case_id"] == cid)
        entry = wave6_entry(cid, w6_julia, w6_oracle, cc)
        ok = entry["max_abs_diff"] <= entry["tolerance"] and w6_julia["cases"][cid]["pass"]
        receipt = {
            "schema": "core070-covariance-p1-case-receipt/v1", "case_id": cid, "source_id": cc["source_id"],
            "verdict": "PASS" if ok else "FAIL", "evidence_kind": "numeric_r_vs_julia",
            "batch": "tools/core070_wave6_conversion_batch.R + .jl, GLLVM_PARITY_PIN=P1 (whole 10-case batch)",
            "r_call": cc["r_call"],
            "raw": [f"{REC_REL}/wave6-conversion-p1/julia-results.json", f"{REC_REL}/wave6-conversion-p1/r-oracle.json"],
            "batch_status": w6_receipt["status"], "batch_verifier": verifiers["wave6"],
            "batch_status_note": ("The batch receipt reads FAIL and tools/core070_verify_wave6_conversion_batch.py rejects "
                                  "the state because of one unrelated case, CORE070-WAVE6-POSTFIT-NOBS-MULTI "
                                  "(postfit/POSTFIT-SURFACE-nobs.gllvmTMB_multi, kind own_receipt_defect): its frozen "
                                  "expectation is that Julia nobs returns n = 80 while R returns p*n = 400; at the "
                                  "GLLVModels head both return 400, so the recorded defect no longer reproduces. "
                                  "The nine point cases, including this one, pass.") if w6_receipt["status"] != "PASS" else "",
            **batch_common,
            "comparison": {"pin": "P1", "cases": [entry]},
        }
        path = REC / "cases" / f"{cid}.json"
        write_json(path, receipt)
        receipts[cid] = (str(path.relative_to(ROOT)), "numeric", receipt["verdict"], verifiers["wave6"]["status"])

    # 3. R-only formula-grammar batch (no Julia comparand)
    cb = json.loads((args.cov_batch / "covariance-batch-results.json").read_text())
    cb_contract = json.loads((OUT / "covariance-batch-contract-p1.json").read_text())
    for c in cb_contract["cases"]:
        cid = c["case_id"]
        entry = cb["cases"][cid]
        receipt = {
            "schema": "core070-covariance-p1-case-receipt/v1", "case_id": cid, "source_id": c["source_fact_id"],
            "verdict": ("SPEC_DEFECT" if c["status"] == "SPEC_DEFECT" else ("R_ONLY_PASS" if entry.get("ok") else "R_ONLY_FAIL")),
            "evidence_kind": "r_only_formula_grammar",
            "batch": "tools/core070_covariance_batch.R, GLLVM_PARITY_PIN=P1; verified by tools/core070_verify_covariance_batch.py",
            "raw": f"{REC_REL}/covariance-batch-p1/covariance-batch-results.json",
            "r_formula": c.get("r_formula"), "expected_covstructs": c.get("expected_covstructs"),
            "observed_covstructs": entry.get("covstructs"),
            "julia_surface": c.get("julia_surface") or c.get("spec_defect_reason"),
            "batch_verifier": verifiers["cov_batch"],
            "why_not_numeric": "The check is a structural fact about gllvmTMB's R formula grammar "
                               "(parse_multi_formula(desugar_brms_sugar(f))$covstructs). No R-vs-Julia numeric comparison exists "
                               "for it in this harness; needs a Julia surface before it can bind.",
            **batch_common,
        }
        path = REC / "cases" / f"{cid}.json"
        write_json(path, receipt)
        receipts[cid] = (str(path.relative_to(ROOT)), "r_only", receipt["verdict"], verifiers["cov_batch"]["status"])

    # 4. public R bridge boundary (R side only)
    tsv = {}
    for line in args.bridge_tsv.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) == 3:
            tsv[parts[0]] = (parts[1], bytes.fromhex(parts[2]).decode())
    for cid, fid in BRIDGE_FIXTURE_ID.items():
        gate, message = tsv[fid]
        receipt = {
            "schema": "core070-covariance-p1-case-receipt/v1", "case_id": cid,
            "verdict": "R_BOUNDARY_UNCHANGED" if gate == BRIDGE_EXPECTED_GATE[cid] else "R_BOUNDARY_CHANGED",
            "evidence_kind": "r_public_bridge_boundary",
            "batch": "tools/core070_covariance_bridge_boundary.R against the P1 oracle library (GLLVModels.jl path poisoned)",
            "raw": f"{REC_REL}/bridge-boundary-p1/{args.bridge_tsv.name}",
            "fixture_id": fid, "observed_gate": gate, "expected_gate_at_p0": BRIDGE_EXPECTED_GATE[cid],
            "observed_message": message,
            "why_not_numeric": "gllvmTMB(engine = 'julia') refuses this structured covariance formula before any Julia call, "
                               "at P1 as at P0. There is no bridge fit to compare; this is R-side boundary evidence only.",
            **batch_common,
        }
        path = REC / "cases" / f"{cid}.json"
        write_json(path, receipt)
        receipts[cid] = (str(path.relative_to(ROOT)), "r_boundary", receipt["verdict"], None)

    # ---- case-map rows ----
    p0 = {r["source_id"]: r for r in json.loads(P0_CASEMAP.read_text())["rows"]}
    out_rows = []
    counts = {"numeric_pass": 0, "numeric_fail": 0, "numeric_held_batch_verifier_failed": 0,
              "partial_numeric_bridge_boundary": 0, "r_only_needs_julia_surface": 0, "not_measured": 0}
    for short in PARTIAL_STALE + DANGLING:
        sid = f"covariance/{short}"
        base = p0[sid]
        ids = base["executable_case_ids"]
        have = [receipts.get(i) for i in ids]
        row = {"source_id": sid, "classification": base["classification"], "arc": "A3",
               "carry_scan_status": "PARTIAL_STALE_AT_P1" if short in PARTIAL_STALE else "DANGLING",
               "executable_case_ids": ids, "disposition": None}
        if any(h is None for h in have):
            row.update(evidence_tier="not_measured", measured_against=None, evidence={},
                       reason="Not re-measured at P1 in this PR.")
            counts["not_measured"] += 1
        else:
            kinds = {h[1] for h in have}
            verdicts = {i: h[2] for i, h in zip(ids, have)}
            paths = [h[0] for h in have]
            batch_ok = {i: h[3] for i, h in zip(ids, have)}
            if kinds == {"numeric"} and not all(v == "PASS" for v in batch_ok.values()) and sid not in exceptions:
                # finding 1: a batch whose own verifier rejected the run cannot bind a row
                row.update(evidence_tier="numeric_held_batch_verifier_failed", measured_against=P1_SHA,
                           evidence={"non_binding_receipts": paths,
                                     "tier": "numeric comparison blocks pass, but the batch verifier rejected the "
                                             "run, so the row does not bind"},
                           note=HELD_NOTE,
                           measured_result={"case_verdicts": verdicts, "batch_verifier": batch_ok})
                counts["numeric_held_batch_verifier_failed"] += 1
            elif kinds == {"numeric"}:
                ok = all(v == "PASS" for v in verdicts.values())
                row.update(evidence_tier="numeric", measured_against=P1_SHA,
                           evidence={"receipt": paths,
                                     "tier": "numeric: per-case receipts carry an R-vs-Julia comparison block pinned to P1"},
                           measured_result={"case_verdicts": verdicts, "batch_verifier": batch_ok,
                                            "row_verdict": "PASS" if ok else "FAIL"})
                if sid in exceptions:
                    # A signed exception waives only the batch verifier, never a
                    # failed comparison: the checker judges max-abs, which is weaker
                    # than the harness's norm test that the case verdict carries.
                    if not ok:
                        raise SystemExit(f"{sid}: refusing a receipt_status_exception on a row whose case "
                                         f"verdicts are not all PASS: {verdicts}")
                    row["receipt_status_exception"] = exceptions[sid]
                counts["numeric_pass" if ok else "numeric_fail"] += 1
            else:
                tier = "partial_numeric_bridge_boundary" if "numeric" in kinds else "r_only"
                row.update(evidence_tier=tier, measured_against=P1_SHA,
                           evidence={"non_binding_receipts": paths,
                                     "tier": ("native and formula cases carry numeric R-vs-Julia comparison blocks; the "
                                              "PUBLIC-R-BRIDGE case is an R-side boundary (gllvmTMB engine='julia' refuses the "
                                              "structured term), so the row as a whole has no numeric twin and does not bind")
                                     if tier != "r_only" else
                                     ("R-only formula-grammar structure check re-run at P1; no Julia comparand exists in "
                                      "this harness, so the row needs a Julia surface before it can bind")},
                           measured_result={"case_verdicts": verdicts})
                counts["partial_numeric_bridge_boundary" if tier != "r_only" else "r_only_needs_julia_surface"] += 1
        out_rows.append(row)

    for row in out_rows:  # Julia fit-level twins (see the overlay block above)
        if row["source_id"] in COV_TWINS:
            twin_overlay(row)
    counts = recount(out_rows)

    # Rows this tool does not generate (e.g. the C3 campaign rows added under #684 item 4) are
    # carried over unchanged, so regenerating the 17 rows never drops another writer's rows.
    generated = {r["source_id"] for r in out_rows}
    prior = OUT / "case-map-covariance.json"
    if prior.is_file():
        out_rows += [r for r in json.loads(prior.read_text())["rows"] if r["source_id"] not in generated]

    casemap = {
        "schema": 1, "reference_commit": P1_SHA,
        "scope": "covariance family: the 17 rows the P1 carry scan lists as PARTIAL_STALE_AT_P1 (7) or DANGLING (10); "
                 "the 22 NOT_BOUND_AT_P0 covariance rows had no P0 evidence and are out of scope here",
        "glvmodels_commit": head,
        "note": ("Separate from case-map.json so none of its rows are touched; read by tools/true_parity_check.mjs with "
                 "PARITY_CASEMAP pointing at this file. Classifications are carried from "
                 "docs/dev-log/core070/required-source-case-map.json unchanged; nothing is signed by an agent. Only rows "
                 "whose every executable case id carries a numeric R-vs-Julia comparison block cite evidence.receipt; "
                 "rows with an R-only or R-boundary case cite evidence.non_binding_receipts instead, so they read as "
                 "not bound under both the current checker and the numeric-tier rule proposed in PR #561. "
                 "A row whose numeric cases pass but whose batch verifier rejected the run is held back the same way "
                 "(evidence_tier numeric_held_batch_verifier_failed) unless a maintainer-signed receipt_status_exception is "
                 "recorded on it."),
        "generator": "tools/core070_covariance_p1_receipts.py",
        "counts": counts, "runtimes_seconds": runtimes,
        "rows": out_rows,
    }
    write_json(OUT / "case-map-covariance.json", casemap)
    print(json.dumps(counts))
    print("files", len(oracle_files + rp_files + dm_files + w6_files + cb_files + br_files), "case receipts", len(receipts))


if __name__ == "__main__":
    import sys
    if "--apply-twins" in sys.argv[1:] or "--check" in sys.argv[1:]:
        apply_twins(check_only="--check" in sys.argv[1:])
    else:
        main()
