"""Write tracked P1 receipts and case-map rows for the postfit and postfit-policy families.

Reads the raw outputs of the seven postfit batches run at gllvmTMB pin P1
(GLLVM_PARITY_PIN=P1) and writes, under docs/dev-log/core070/true-parity-latest/:

  receipts/postfit/<batch>-p1/...   batch artifacts copied verbatim (JSON/TSV), run-commit.json, verify.txt
  receipts/postfit/cases/<id>.json  one receipt per executable case id
  case-map-postfit.json             the 36 postfit rows the P1 carry scan lists as
                                    DANGLING (34) or RETIRED (2), and the 16 DANGLING
                                    postfit-policy rows

A case gets a `comparison` block only when its harness compares an R number
with a Julia number against a declared tolerance > 0. The tolerance is the
one the harness itself applies (never a wider one), and abs_diff is the
maximum absolute elementwise difference the harness bounds. Where the raw
R and Julia vectors are both in the batch output, max_abs_diff is recomputed
here and must agree with the harness figure; otherwise the harness figure is
used and marked so.

Cases that are verdicts (both engines report a boolean), own-consistency
checks (each engine checked against itself), exact-integer equalities, empty
lengths, or signature/default-policy checks carry no comparison block, and
their rows cite evidence.non_binding_receipts. A row is evidence_tier
"numeric" only when every executable case id has a comparison block, the
harness verdict is PASS, every batch those cases came from passed its own
verifier, and no comparison is degenerate.

Batch verifier gate (PR #567's rule, review finding 1 there): this tool runs
each batch's verifier at P1 (with --self-test) and keeps its full output as
the tracked <batch>/verify.txt; every case receipt carries a batch_verifier
block. A row whose numeric cases pass but whose batch verifier rejected the
run is held back: evidence_tier "numeric_held_batch_verifier_failed",
receipts under evidence.non_binding_receipts. There is no exception path.

Degenerate-comparison gate (PR #569 review finding 1): a comparison whose R
oracle values are all equal to one constant (two or more values) or all
below DEGENERATE_ABS in magnitude cannot tell a right implementation from a
constant or zero one, so it is flagged discriminating: false and its row is
evidence_tier "numeric_non_discriminating" with non-binding receipts.

Provenance: every receipt records glvmodels_commit = HEAD of this checkout.
The tool refuses to write when tracked files outside its own outputs are
modified (unless --allow-dirty, which is then recorded), and refuses unless
each run directory holds a run-commit.json (written by whoever launched the
batch: {"glvmodels_commit": <HEAD at launch>, "dirty": [<porcelain lines>]})
that names this HEAD with an empty dirty list. Run the batches and this tool
from the same clean commit.

Julia twins (overlay). Rows whose batch case was non-discriminating, a shape-only check or never
executed can be bound by a numeric twin on a non-degenerate fixture: R-at-P1 values recorded in a
tracked fixture (test/fixtures/ns_numeric_p1.toml, test/fixtures/postfit_twins_p1.toml) against Julia
fits of the same data (test/test_namespace_numeric_p1_twin.jl, test/test_postfit_twins_p1.jl), receipts
written by tools/true_parity_julia_receipts.jl under receipts/julia-twins/postfit-twins/. Where such a
receipt exists, the row gains the evidence fields it supports: the twin receipt under evidence.receipt,
executable_case_ids set to the twin's case ids (the batch case ids move to evidence.batch_case_ids and
the batch receipts stay under evidence.non_binding_receipts), evidence_tier numeric. Classification,
disposition and every other field are untouched; a row's `note` or `reason` still describes the
superseded batch case. `--apply-twins` re-applies the overlay to the tracked case-map-postfit.json
(idempotent) and `--check-twins` verifies that file is exactly that re-derivation.

Integer equality (itchyshin/GLLVModels.jl#684 item 1). The four exact-integer postfit-policy cases
(POST-LOGLIK-DF, POST-LOGLIK-NOBS, POST-NOBS-COUNT; POST-NOBS-FALLBACK stays unbound, see FALLBACK_WHY) carry a comparison block of
kind "integer_equality" with tolerance 0.5, so "within tolerance" can only mean "equal". r_value is read
from the R oracle (r-oracle.json), julia_value from the Julia results (julia-results.json); the R number
the harness itself recorded must agree with the oracle or the tool stops. Those rows are then
evidence_tier numeric. `--apply-integer-equality` re-derives this from the tracked raw files under
receipts/postfit/postfit-policy-p1/ (no run directories needed; idempotent); the full run does the same.

Usage (inputs are the raw run directories under local-scratch):
  python3 tools/core070_postfit_p1_receipts.py --runs DIR --runtimes JSON [--allow-dirty]
  python3 tools/core070_postfit_p1_receipts.py --apply-twins
  python3 tools/core070_postfit_p1_receipts.py --apply-integer-equality
  python3 tools/core070_postfit_p1_receipts.py --check-twins
where DIR holds surface-conversion-p1/, wave6-conversion-p1/, wave7-conversion-p1/,
wave8-conversion-p1/, estimand-rebind-p1/, postfit-policy-p1/, postfit-1-r-p1/,
postfit-1-julia-p1/, each with its run-commit.json.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/dev-log/core070/true-parity-latest"
REC = OUT / "receipts/postfit"
REC_REL = "docs/dev-log/core070/true-parity-latest/receipts/postfit"
P0_CASEMAP = ROOT / "docs/dev-log/core070/required-source-case-map.json"
PINS = tomllib.loads((ROOT / "tools/core070_oracle_pins.toml").read_text())
P1_SHA = PINS["P1"]["reference_commit"]
P0_SHA = PINS["P0"]["reference_commit"]
ORACLE_BUILD = "docs/dev-log/core070/true-parity-latest/receipts/covariance/oracle/build.json"
ORACLE_SOURCE = "docs/dev-log/core070/true-parity-latest/receipts/covariance/oracle/source.json"
MAX_INLINE = 25  # vectors longer than this stay in the batch raw file only
DEGENERATE_ABS = 1e-10  # every |R value| below this: the comparison cannot discriminate
ESTIMAND_ACCESSOR_RECORD = "docs/dev-log/core070/true-parity-latest/estimand-rebind-accessor-diff-p1.json"
HELD_NOTE = "held pending the maintainer's ruling on the wave6 nobs expectation"

# Rows in scope: P1 carry scan (branch claude/true-parity-p1-carry, carry-scan-p1.json).
POSTFIT_DANGLING = [
    "check_auto_residual", "coef.gllvmTMB_multi", "compare_loadings", "confint.gllvmTMB_multi",
    "deviance.gllvmTMB_multi", "extract_ICC_site", "extract_Omega", "extract_Sigma", "extract_Sigma_table",
    "extract_communality", "extract_correlations", "extract_cutpoints", "extract_loadings",
    "extract_ordination", "extract_proportions", "extract_repeatability", "extract_residual_cor",
    "extract_residual_cov", "extract_rotated_loadings_table", "fitted.gllvmTMB_multi", "getLV",
    "getLoadings", "getResidualCor", "getResidualCov", "logLik.gllvmTMB_multi", "nobs.gllvmTMB_multi",
    "predict.gllvmTMB_multi", "predict_missing", "residuals.gllvmTMB_multi", "rotate_loadings",
    "sanity_multi", "simulate_unit_trait", "summary.gllvmTMB_multi", "tidy.gllvmTMB_multi",
]
POSTFIT_RETIRED = [".proportions_bootstrap_ci", ".proportions_wald_ci"]
POLICY_DANGLING = [
    "POST-COEF-EMPTY", "POST-COEF-NAMED", "POST-CONFINT-METHODS", "POST-DEVIANCE", "POST-FITTED-DEFAULT",
    "POST-LOGLIK-DF", "POST-LOGLIK-NOBS", "POST-LOGLIK-VALUE", "POST-NOBS-COUNT", "POST-NOBS-FALLBACK",
    "POST-PREDICT-DEFAULT", "POST-RE-FORM-FULL", "POST-RESIDUAL-CONDITIONAL", "POST-RESIDUAL-SCALES",
    "POST-RESIDUAL-TYPES", "POST-SIMULATE-DEFAULT",
]

# estimand-rebind has no contract file; this is its verifier's CASE_META, verbatim.
ESTIMAND_SOURCE = {
    "CORE070-ESTIMAND-REBIND-EXTRACT-COMMUNALITY": "postfit/POSTFIT-SURFACE-extract_communality",
    "CORE070-ESTIMAND-REBIND-EXTRACT-CORRELATIONS": "postfit/POSTFIT-SURFACE-extract_correlations",
    "CORE070-ESTIMAND-REBIND-EXTRACT-PROPORTIONS": "postfit/POSTFIT-SURFACE-extract_proportions",
    "CORE070-ESTIMAND-REBIND-EXTRACT-OMEGA": "postfit/POSTFIT-SURFACE-extract_Omega",
}

WAVE6_NOTE = ("The wave6 batch receipt reads FAIL, and tools/core070_verify_wave6_conversion_batch.py rejects the "
              "state, because of one case, CORE070-WAVE6-POSTFIT-NOBS-MULTI (own_receipt_defect): its frozen "
              "expectation that Julia nobs returns n = 80 no longer holds (Julia now returns p*n = 400, as R does). "
              "The other nine cases pass; each case receipt carries its own verdict.")

# Reviewer's evidence for the rows the degenerate-comparison gate holds (PR #569 review finding 1).
NON_DISCRIMINATING_NOTES = {
    "postfit/POSTFIT-SURFACE-extract_communality": (
        "Measured on the unique = FALSE Gaussian fixture (tools/core070_estimand_rebind_batch.R), where communality "
        "is identically 1 for every trait: R and Julia both return [1, 1, 1, 1, 1]. Reviewer mutation: a Julia "
        "extract_communality that returns a constant 1.0 still PASSES the batch; 0.99 * s/t FAILS. The harness "
        "bounds |R - Julia| correctly, but this fixture cannot distinguish a wrong implementation. A non-degenerate "
        "fixture (unique = TRUE) is a new case, i.e. a contract change for the maintainer."),
    "postfit/POSTFIT-SURFACE-extract_proportions": (
        "Same unique = FALSE fixture as extract_communality: the compared proportion vectors are all 1, so a "
        "constant implementation would pass (same structure as the reviewer's communality mutation). A "
        "non-degenerate fixture is a contract change for the maintainer."),
    "postfit/POSTFIT-SURFACE-tidy.gllvmTMB_multi": (
        "Fixed-effect estimates on row-centred data: the R oracle is about 1e-14 per coefficient and the harness "
        "difference (1.4e-14) equals max |R|, i.e. Julia returns about 0. A Julia tidy that returned zeros would "
        "pass. An uncentred fixture is a contract change for the maintainer."),
    "postfit-policy/POST-COEF-NAMED": (
        "Same row-centred fixture as tidy: R coef values are about 1e-14 and the harness delta (1.359e-14) equals "
        "max |R|, so a Julia coef returning zeros would pass. An uncentred fixture is a contract change for the "
        "maintainer."),
}

# postfit-policy: case id -> the r-oracle.json key(s) holding the R values it compares (degenerate gate).
POLICY_R_KEYS = {
    "CORE070-POSTFIT-COEF-NAMED-NATIVE": ["coef"],
    "CORE070-POSTFIT-CONFINT-METHODS-WALD-NATIVE": ["ci_lower", "ci_upper"],
    "CORE070-POSTFIT-FITTED-DEFAULT-NATIVE": ["response"],
    "CORE070-POSTFIT-LOGLIK-VALUE-NATIVE": ["loglik"],
    "CORE070-POSTFIT-RE-FORM-FULL-NATIVE": ["link"],
    "CORE070-POSTFIT-RESIDUAL-CONDITIONAL-NATIVE": ["residual"],
    "CORE070-POSTFIT-RESIDUAL-SCALES-NATIVE": ["residual"],
    "CORE070-POSTFIT-RESIDUAL-TYPES-NATIVE": ["residual"],
}

# Each batch's verifier (run by this tool at P1) and the line that marks acceptance.
VERIFIERS = {
    "surface-conversion-p1": (["tools/core070_verify_surface_conversion_batch.py", "{state}", "--self-test"],
                              "CORE070_SURFACE_CONVERSION_STATE_OK"),
    "estimand-rebind-p1": (["tools/core070_verify_estimand_rebind_batch.py", "{state}", "--self-test"],
                           "CORE070_ESTIMAND_REBIND_STATE_OK"),
    "wave6-conversion-p1": (["tools/core070_verify_wave6_conversion_batch.py", "{state}"],
                            "CORE070_WAVE6_CONVERSION_STATE_OK"),
    "wave7-conversion-p1": (["tools/core070_verify_wave7_conversion_batch.py", "{state}", "--self-test"],
                            "CORE070_WAVE7_CONVERSION_STATE_OK"),
    "wave8-conversion-p1": (["tools/core070_verify_wave8_conversion_batch.py", "{state}", "--self-test"],
                            "CORE070_WAVE8_CONVERSION_STATE_OK"),
    "postfit-policy-p1": (["tools/core070_verify_postfit_policy_batch.py", "--state", "{state}", "--self-test"],
                          "CORE070_POSTFIT_POLICY_BATCH_VERIFIED"),
    "postfit-1-p1": (["tools/core070_verify_postfit_1_batch.py", "--r-state", "{r_state}", "--julia-state",
                      "{julia_state}", "--self-test"], "CORE070_POSTFIT_1_BATCH_VERIFIED"),
}

# postfit-policy: case id -> contract tolerance key (numeric cases only).
POLICY_TOL_KEY = {
    "CORE070-POSTFIT-COEF-NAMED-NATIVE": "coefficient_delta",
    "CORE070-POSTFIT-CONFINT-METHODS-WALD-NATIVE": "wald_ci_bound_delta",
    "CORE070-POSTFIT-FITTED-DEFAULT-NATIVE": "link_response_residual_delta",
    "CORE070-POSTFIT-LOGLIK-VALUE-NATIVE": "loglik_delta",
    "CORE070-POSTFIT-RE-FORM-FULL-NATIVE": "link_response_residual_delta",
    "CORE070-POSTFIT-RESIDUAL-CONDITIONAL-NATIVE": "link_response_residual_delta",
    "CORE070-POSTFIT-RESIDUAL-SCALES-NATIVE": "link_response_residual_delta",
    "CORE070-POSTFIT-RESIDUAL-TYPES-NATIVE": "link_response_residual_delta",
}
# postfit-policy cases that carry a numeric leg but whose fact is a documented default divergence.
POLICY_PARTIAL_WITH_NUMERIC_LEG = {"CORE070-POSTFIT-PREDICT-DEFAULT-NATIVE": "link_response_residual_delta"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True, text=True).stdout


def git_state():
    """HEAD and the tracked paths modified outside this tool's own outputs."""
    head = git("rev-parse", "HEAD").strip()
    own = (REC_REL + "/", "docs/dev-log/core070/true-parity-latest/case-map-postfit.json")
    dirty = [line[3:] for line in git("status", "--porcelain", "--untracked-files=no").splitlines()
             if not line[3:].startswith(own)]
    return head, dirty


def check_run_commit(run_dir, head):
    """The run directory's run-commit.json must name this HEAD and a clean tree."""
    p = run_dir / "run-commit.json"
    if not p.is_file():
        raise SystemExit(f"{run_dir} has no run-commit.json; re-run the batch from a clean commit")
    rc = load(p)
    if rc.get("glvmodels_commit") != head or rc.get("dirty") != []:
        raise SystemExit(f"{run_dir}: run at {rc.get('glvmodels_commit')} dirty={rc.get('dirty')}, "
                         f"not at clean HEAD {head}; re-run at HEAD")


def run_verifier(name, **states):
    """Run a batch verifier at P1 and keep its full output as the tracked <batch>/verify.txt."""
    argv_t, marker = VERIFIERS[name]
    argv = ["python3"] + [a.format(**{k: str(v) for k, v in states.items()}) for a in argv_t]
    proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True,
                          env=dict(os.environ, GLLVM_PARITY_PIN="P1"))
    log = REC / name / "verify.txt"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(proc.stdout + proc.stderr)
    ok = proc.returncode == 0 and marker in proc.stdout
    return {"tool": argv_t[0], "argv": " ".join(a for a in argv_t), "status": "PASS" if ok else "FAIL",
            "exit_code": proc.returncode, "accept_marker": marker, "log": f"{REC_REL}/{name}/verify.txt"}


def degenerate(values):
    """Why a comparison cannot discriminate (None when it can), judged on the R oracle values."""
    vals = [float(v) for v in values]
    if not vals:
        return "no values"
    if all(abs(v) < DEGENERATE_ABS for v in vals):
        return f"every |R value| < {DEGENERATE_ABS:g} (max {max(abs(v) for v in vals):.3g})"
    if len(vals) >= 2 and max(vals) - min(vals) <= 1e-12 * max(1.0, max(abs(v) for v in vals)):
        return f"all {len(vals)} R values equal the constant {vals[0]:.17g}"
    return None


def mark_degenerate(entry, r_values):
    why = degenerate(r_values)
    entry["discriminating"] = why is None
    if why is not None:
        entry["degenerate_reason"] = why
    return entry


def copy_batch(src_dir, dest_name, names):
    dest = REC / dest_name
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for n in names:
        p = src_dir / n
        if p.is_file():
            shutil.copyfile(p, dest / n)
            out.append(f"{REC_REL}/{dest_name}/{n}")
    return out


def vec_entry(case_id, quantity, rv, jv, tol, harness_diff, rule):
    rv = rv if isinstance(rv, list) else [rv]
    jv = jv if isinstance(jv, list) else [jv]
    if len(rv) != len(jv):
        raise SystemExit(f"{case_id}: R length {len(rv)} != Julia length {len(jv)}")
    diff = max(abs(a - b) for a, b in zip(rv, jv))
    if harness_diff is not None and abs(diff - harness_diff) > 1e-12 * max(1.0, abs(harness_diff)):
        raise SystemExit(f"{case_id}: recomputed max_abs_diff {diff} != harness {harness_diff}")
    e = {"case_id": case_id, "quantity": quantity, "max_abs_diff": diff, "tolerance": tol,
         "tolerance_rule": rule, "n_values": len(rv), "diff_source": "recomputed from raw R and Julia values"}
    if len(rv) <= MAX_INLINE:
        e["r_value"], e["julia_value"] = rv, jv
    return mark_degenerate(e, rv)


def harness_entry(case_id, quantity, diff, tol, rule, r_values):
    e = {"case_id": case_id, "quantity": quantity, "max_abs_diff": diff, "tolerance": tol,
         "tolerance_rule": rule,
         "diff_source": "harness-reported (the Julia results file records the max |R - Julia| it bounded, "
                        "not the Julia vector)"}
    if r_values is None:
        raise SystemExit(f"{case_id}: no R values to judge whether the comparison discriminates")
    return mark_degenerate(e, r_values)


TWIN_REL = "docs/dev-log/core070/true-parity-latest/receipts/julia-twins/postfit-twins"
TWIN_TIER = ("numeric: Julia values recomputed by tools/true_parity_julia_receipts.jl with the same calls and settings as "
             "the cited twin test, against R-at-P1 values copied from a tracked fixture (an R fit that converged with a "
             "positive-definite Hessian, on a non-degenerate fixture: the R values are not one constant and not ~0), "
             "each case within the tolerance asserted in that test. The batch case this row carried is superseded; its "
             "receipt is kept under non_binding_receipts and its ids under batch_case_ids")
TWIN_FILES = {  # source_id -> twin receipt stem
    "postfit/POSTFIT-SURFACE-extract_communality": "extract_communality",
    "postfit/POSTFIT-SURFACE-extract_rotated_loadings_table": "extract_rotated_loadings_table",
    "postfit/POSTFIT-SURFACE-tidy.gllvmTMB_multi": "tidy",
    "postfit-policy/POST-COEF-NAMED": "coef",
    "postfit-policy/POST-DEVIANCE": "deviance",
}


# Exact-integer postfit-policy cases (ruling 1): case id -> (r-oracle.json key, quantity).
INTEGER_EQUALITY = {
    "CORE070-POSTFIT-LOGLIK-DF-NATIVE": ("df", "attr(logLik(object), 'df')"),
    "CORE070-POSTFIT-LOGLIK-NOBS-NATIVE": ("loglik_nobs_attr", "attr(logLik(object), 'nobs')"),
    "CORE070-POSTFIT-NOBS-COUNT-NATIVE": ("nobs", "nobs(object), likelihood_rows-preferring branch"),
    # Maintainer ruling 2026-10-05 (D-319), item N2: integer_equality also covers POST-COEF-EMPTY (lengths 0 = 0).
    "CORE070-POSTFIT-COEF-EMPTY-NATIVE": ("empty_coef", "length(coef(object)) on a fit with no fixed-effect columns"),
}
COEF_EMPTY_CID = "CORE070-POSTFIT-COEF-EMPTY-NATIVE"
# Receipt text for the four cases, with line numbers read from git show 9539352f6:R/methods-gllvmTMB.R (P1), not P0.
INTEGER_TEXT = {
    "CORE070-POSTFIT-LOGLIK-DF-NATIVE": {
        "r_call": "attr(logLik(object),'df') non-REML branch (R/methods-gllvmTMB.R:1147-1148 at P1 9539352f6)",
        "julia_surface": "StatsAPI.dof(fit::AnyGllvmFit) = _nparams(fit) (src/postfit.jl:573 at the measured commit 681c4c3ca)"},
    "CORE070-POSTFIT-LOGLIK-NOBS-NATIVE": {
        "r_call": "attr(logLik(object),'nobs') (R/methods-gllvmTMB.R:1196-1202 at P1 9539352f6)",
        "julia_surface": "StatsAPI.nobs(fit::AnyGllvmFit, Y; mask) (src/postfit.jl:622 at the measured commit 681c4c3ca)"},
    "CORE070-POSTFIT-NOBS-COUNT-NATIVE": {
        "r_call": "nobs.gllvmTMB_multi, likelihood_rows-preferring branch (R/methods-gllvmTMB.R:1216-1224 at P1 9539352f6)",
        "julia_surface": "StatsAPI.nobs(fit::AnyGllvmFit, Y; mask) (src/postfit.jl:622 at the measured commit 681c4c3ca)"},
    "CORE070-POSTFIT-COEF-EMPTY-NATIVE": {
        "r_call": "coef.gllvmTMB_multi's empty-X_fix_names branch (R/vcov-coef.R:59-65 at P1 9539352f6), evaluated by "
                  "the batch on a synthetic object list(X_fix_names = character(0)), not on a fitted model",
        "julia_surface": "StatsAPI.coef(fit::GllvmFit) on a real fit with no X (src/postfit.jl:633 at the measured "
                         "commit 681c4c3ca)",
        "numeric_note": ("Integer equality on the coefficient-vector length: R 0, Julia 0. Maintainer ruling 2026-10-05 "
                         "(D-319), item N2, extends integer_equality (itchyshin/GLLVModels.jl#684 item 1) to this row. "
                         "Scope: the R length comes from coef.gllvmTMB_multi applied to a synthetic object with no "
                         "fixed-effect names; the Julia length from a real fit with no X.")},
}
# POST-NOBS-FALLBACK stays unbound (review of #688): on the fixture fit R's nobs() returns on the likelihood_rows
# branch, so the no-missing-data fallback branch this row names is never executed. Its receipt keeps no comparison.
FALLBACK_CID = "CORE070-POSTFIT-NOBS-FALLBACK-NATIVE"
FALLBACK_TEXT = {
        "r_call": "nobs.gllvmTMB_multi called on the fixture fit (R/methods-gllvmTMB.R:1216-1231 at P1 9539352f6). "
                  "The fixture fit has a non-NULL fit$missing_data$counts$likelihood_rows (400), so nobs() returns at line 1223 "
                  "and the is_y_observed / length(y) fallback (lines 1225-1230) is NOT executed by this oracle.",
        "comparand": "exact integer equality; the same measurement as NOBS-COUNT (R's likelihood_rows branch). "
                     "The R no-missing-data fallback branch is not separately executed by this evidence.",
        "julia_surface": "StatsAPI.nobs(fit::AnyGllvmFit, Y; mask) (src/postfit.jl:622 at the measured commit 681c4c3ca)"}
FALLBACK_WHY = ("Not numeric for this row: R's nobs() returned on the likelihood_rows branch (fit$missing_data$counts"
                "$likelihood_rows is 400 on this fixture), so the fallback branch POST-NOBS-FALLBACK names was not executed. "
                "The integers agree (400 = 400), but they measure NOBS-COUNT's branch. A fit where R reaches the fallback "
                "is needed before this row can bind.")
INTEGER_RULING = "itchyshin/GLLVModels.jl#684 item 1"
INTEGER_RULE = (f"integer_equality, tolerance 0.5 ({INTEGER_RULING}): both values are integers, so within 0.5 means equal; "
                "the contract's integer_exact = 0 is the same condition")
INTEGER_WHY = (f"Exact integer equality (contract tolerances.integer_exact = 0). The row now binds as numeric under "
               f"{INTEGER_RULING}: the comparison block records R and Julia integers with tolerance 0.5, which for "
               "integers means equal.")
INTEGER_TIER = (f"numeric: every executable case receipt carries an integer_equality comparison block pinned to P1 "
                f"(tolerance 0.5, so the integers must be equal), under {INTEGER_RULING}")


def integer_entry(cid, jc, po):
    """The integer_equality comparison entry for `cid`, from the raw Julia result `jc` and R oracle `po`."""
    key, quantity = INTEGER_EQUALITY[cid]
    if cid == COEF_EMPTY_CID:  # a length: R's empty coef vector against Julia's
        r, j, harness_r = len(po[key]), jc["julia_length"], jc["r_length"]
    else:
        r, j, harness_r = po[key], jc["julia"], jc["r"]
    for name, v in (("R oracle", r), ("Julia result", j)):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v != int(v):
            raise SystemExit(f"{cid}: {name} value {v!r} is not an integer")
    r, j = int(r), int(j)
    if harness_r != r:
        raise SystemExit(f"{cid}: harness-recorded R value {harness_r!r} != r-oracle.json[{key!r}] = {r}")
    if bool(jc["pass"]) != (r == j):
        raise SystemExit(f"{cid}: harness pass flag {jc['pass']!r} disagrees with R {r} vs Julia {j}")
    return {"case_id": cid, "quantity": quantity, "kind": "integer_equality", "r_value": r, "julia_value": j,
            "max_abs_diff": abs(r - j), "tolerance": 0.5, "tolerance_rule": INTEGER_RULE, "n_values": 1,
            "diff_source": (f"recomputed from len(r-oracle.json[{key!r}]) and julia-results.json cases[{cid!r}]['julia_length']"
                            if cid == COEF_EMPTY_CID else
                            f"recomputed from r-oracle.json[{key!r}] and julia-results.json cases[{cid!r}]['julia']")}


WAVE6_INTEGER_WHY = ("Exact integer equality, R nobs against Julia nobs on the gaussian_small fixture. Maintainer ruling "
                     "2026-10-05 (D-319), item N2, rewrote this wave6 case from the stale P0 own_receipt_defect expectation "
                     f"(Julia == n = 80) to integer_equality under {INTEGER_RULING}: tolerance 0.5, so the integers must be equal.")


def wave6_integer_entry(cid, jc, oracle):
    """integer_equality comparison entry for the wave6 nobs case, from the R oracle and the Julia results."""
    r, j = oracle["oracle_values"][cid], jc["julia_value"]
    for name, v in (("R oracle", r), ("Julia result", j)):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v != int(v):
            raise SystemExit(f"{cid}: {name} value {v!r} is not an integer")
    r, j = int(r), int(j)
    if jc["r_value"] != r:
        raise SystemExit(f"{cid}: harness-recorded R value {jc['r_value']!r} != r-oracle.json oracle_values = {r}")
    if bool(jc["pass"]) != (r == j):
        raise SystemExit(f"{cid}: harness pass flag {jc['pass']!r} disagrees with R {r} vs Julia {j}")
    return {"case_id": cid, "quantity": "nobs(object) on gaussian_small", "kind": "integer_equality",
            "r_value": r, "julia_value": j, "max_abs_diff": abs(r - j), "tolerance": 0.5,
            "tolerance_rule": (f"integer_equality, tolerance 0.5 ({INTEGER_RULING}): both values are integers, so within "
                               "0.5 means equal; wave6-conversion-batch-contract-p1.json gives this case tolerance 0.5"),
            "n_values": 1,
            "diff_source": f"recomputed from r-oracle.json oracle_values[{cid!r}] and julia-results.json cases[{cid!r}]['julia_value']"}


# ---- signed dispositions (maintainer ruling 2026-10-05, vault D-319) ----
# Signature source: LOOP/lanes/true-parity-latest/signed-rulings-2026-10-05.md in the true-parity lane kit, where the
# maintainer signed every "Recommend" on the rulings page (D-319). The row keeps its measured evidence (non-binding
# receipts) and gains disposition DISPOSITION-SIGNED with signer, date, the ruling reference and the reason.
RULING_SIGNED_BY = "Shinichi Nakagawa"
RULING_SIGNED_ON = "2026-10-05"
RULING_SOURCE = "maintainer ruling 2026-10-05 (D-319), LOOP/lanes/true-parity-latest/signed-rulings-2026-10-05.md"
PROPORTIONS_RETIRED = (
    "Retired at P1 by maintainer ruling 2026-10-05 (D-319), item N5. export({name}) is absent from the P1 NAMESPACE (the "
    "definition is still present, unexported, at R/proportions-ci.R:{line}), so there is no public R surface left to "
    "twin. This is the same removal the headline ledger records as retired/{name} (PR #533, signed 2026-09-30); the "
    "namespace map lists it under retired_at_p1. Classification moves from compatibility_adapter (kept in "
    "original_classification) to retired, as on the headline row.")
RULINGS = {
    "postfit/POSTFIT-SURFACE-.proportions_bootstrap_ci": {
        "item": "N5", "classification": "retired",
        "reason": PROPORTIONS_RETIRED.format(name=".proportions_bootstrap_ci", line=423)},
    "postfit/POSTFIT-SURFACE-.proportions_wald_ci": {
        "item": "N5", "classification": "retired",
        "reason": PROPORTIONS_RETIRED.format(name=".proportions_wald_ci", line=247)},
    "postfit-policy/POST-NOBS-FALLBACK": {
        "item": "POST-NOBS-FALLBACK option (a)",
        "reason": (
            "Not reachable through the public door at P1, by maintainer ruling 2026-10-05 (D-319), POST-NOBS-FALLBACK "
            "option (a). nobs.gllvmTMB_multi (R/methods-gllvmTMB.R:1216-1231 at P1 9539352f6) returns "
            "object$missing_data$counts$likelihood_rows at line 1221 whenever it is non-NULL; the is_y_observed / "
            "length(y) fallback (lines 1225-1230) runs only when that field is NULL. The only constructor of a "
            "gllvmTMB_multi object always builds missing_data with .gllvmTMB_build_missing_data, which always sets "
            "counts$likelihood_rows = as.integer(sum(is_y_observed == 1L)) (R/fit-multi.R:9784 and :9822), never NULL. "
            "Twelve probe fit types through gllvmTMB() at P1 (Gaussian and Poisson latent, missing responses dropped "
            "and masked, an all-missing trait both ways, weights, mixed family, wide traits(), no covstruct, an mi() "
            "predictor, estimator = 'mspl', and a saveRDS/readRDS round trip) all returned on the likelihood_rows "
            "branch; only a fit with fit$missing_data deleted by hand reached the fallback, and there the two branches "
            "give the same number by construction. The field and the method were added in the same commit (#334), so "
            "the fallback is defensive code for hand-built or pre-#334 objects, not a separate user-facing behaviour. "
            "The likelihood_rows branch itself is bound by POST-NOBS-COUNT. Evidence: post-nobs-fallback-evidence.md in "
            "the true-parity lane kit (probe scripts nobs_probe.R, nobs_probe2.R).")},
    "postfit-policy/POST-PREDICT-DEFAULT": {
        "item": "B (Julia default versus R default)",
        "reason": (
            "Documented default divergence accepted by maintainer ruling 2026-10-05 (D-319), item B (Julia defaults "
            "against R, row by row). R: predict.gllvmTMB_multi(object, newdata = NULL, type = c('link', 'response'), "
            "...) defaults to type = 'link' (R/methods-gllvmTMB.R:2786-2794 at P1 9539352f6). Julia: "
            "predict(fit::GllvmFit, y; type::Symbol = :response, ...) defaults to type = :response (src/postfit.jl). "
            "The batch reflects both defaults live (R 'link', Julia :response) and, called with the same explicit "
            "type = link, the two engines agree on the linear predictor (max |R - Julia| = 5.72e-06 against the "
            "contract's link_response_residual_delta 1e-4, receipt CORE070-POSTFIT-PREDICT-DEFAULT-NATIVE). A user "
            "porting a bare predict(fit) call from R gets the response scale in Julia, not the link scale; pass "
            "type = :link to match R.")},
}


def ruling_overlay(row):
    """Write the signed disposition onto a row the 2026-10-05 rulings cover (idempotent)."""
    r = RULINGS.get(row["source_id"])
    if r is None:
        return row
    if "classification" in r and row["classification"] != r["classification"]:
        row.setdefault("original_classification", row["classification"])
        row["classification"] = r["classification"]
    row["reason"] = r["reason"]
    row["disposition"] = "DISPOSITION-SIGNED"
    row["signed_by"] = RULING_SIGNED_BY
    row["signed_on"] = RULING_SIGNED_ON
    row["signature_ref"] = f"{RULING_SOURCE}, item {r['item']}"
    return row


def count_key(row):
    """The counts bucket of a row: RETIRED rows count as retired_at_p1_not_measured, others by evidence tier."""
    if row.get("carry_scan_status") == "RETIRED" and row["evidence_tier"] == "not_measured":
        return "retired_at_p1_not_measured"
    return tier_count_key(row["evidence_tier"])


def recount(cm):
    counts = {k: 0 for k in cm["counts"]}
    for r in cm["rows"]:
        counts[count_key(r)] += 1
    return counts


def bind_numeric(row, paths, verdicts, batch_ok, tier):
    """Set the evidence fields of a row that binds as numeric (shared by the full run and the integer overlay)."""
    row.update(evidence_tier="numeric", measured_against=P1_SHA,
               evidence={"receipt": paths, "tier": tier},
               measured_result={"case_verdicts": verdicts, "batch_verifier": batch_ok, "row_verdict": "PASS"})
    return row


def apply_integer_equality():
    raw = REC / "postfit-policy-p1"
    pj, po = load(raw / "julia-results.json"), load(raw / "r-oracle.json")
    recs = {}
    for cid in INTEGER_EQUALITY:
        path = REC / "cases" / f"{cid}.json"
        rec = load(path)
        if rec["harness_fields"] != pj["cases"][cid]:
            raise SystemExit(f"{cid}: receipt harness_fields {rec['harness_fields']} != raw julia-results.json "
                             f"{pj['cases'][cid]}; stop and report")
        rec["comparison"] = {"pin": "P1", "cases": [integer_entry(cid, pj["cases"][cid], po)]}
        rec.pop("why_not_numeric", None)
        rec["numeric_note"] = INTEGER_WHY
        rec.update(INTEGER_TEXT[cid])
        rec["evidence_kind"] = "numeric_r_vs_julia"
        write_json(path, rec)
        recs[cid] = (str(path.relative_to(ROOT)), rec)
    fpath = REC / "cases" / f"{FALLBACK_CID}.json"
    frec = load(fpath)
    frec.pop("comparison", None)
    frec.pop("numeric_note", None)
    frec.update(FALLBACK_TEXT)
    frec["why_not_numeric"] = FALLBACK_WHY
    frec["evidence_kind"] = "paired_policy_check"
    write_json(fpath, frec)
    cm = load(OUT / "case-map-postfit.json")
    for row in cm["rows"]:
        ids = row["executable_case_ids"]
        if ids == [FALLBACK_CID]:
            row.update(evidence_tier="partial_non_numeric_case", measured_against=P1_SHA,
                       evidence={"non_binding_receipts": [str(fpath.relative_to(ROOT))],
                                 "tier": "measured at P1 and the integers agree, but R's nobs() never reached the "
                                         "fallback branch this row names on this fixture (it returned on the "
                                         "likelihood_rows branch), so the row does not bind"},
                       measured_result={"case_verdicts": {FALLBACK_CID: frec["verdict"]},
                                        "batch_verifier": {FALLBACK_CID: frec["batch_verifier"]["status"]},
                                        "case_kinds": {FALLBACK_CID: "paired_policy_check"}})
            continue
        if len(ids) == 1 and ids[0] in INTEGER_EQUALITY:
            path, rec = recs[ids[0]]
            if rec["verdict"] != "PASS" or rec["batch_verifier"]["status"] != "PASS":
                raise SystemExit(f"{ids[0]}: receipt verdict or batch verifier is not PASS")
            bind_numeric(row, [path], {ids[0]: rec["verdict"]}, {ids[0]: rec["batch_verifier"]["status"]}, INTEGER_TIER)
    cm["rows"] = [ruling_overlay(r) for r in cm["rows"]]
    cm["counts"] = counts = recount(cm)
    write_json(OUT / "case-map-postfit.json", cm)
    print(json.dumps(counts))


def tier_count_key(tier):
    return "numeric_pass" if tier == "numeric" else tier


def twin_overlay(row):
    """Add the numeric-twin evidence fields to `row` when a Julia twin receipt exists for it (idempotent)."""
    sid = row["source_id"]
    if sid not in TWIN_FILES:
        return row
    path = ROOT / TWIN_REL / f"{TWIN_FILES[sid]}.json"
    if not path.is_file():
        return row
    rec = load(path)
    rel = str(path.relative_to(ROOT))
    ok = (rec.get("schema") == "true-parity-julia-twin-receipt/v1" and rec.get("source_ids") == [sid]
          and rec.get("verdict") == "PASS" and rec.get("pin") == "P1" and rec.get("reference_commit") == P1_SHA
          and rec.get("evidence_kind") == "julia_recomputed_vs_recorded_r")
    if not ok:
        raise SystemExit(f"{rel}: not a passing P1 Julia twin receipt for {sid}")
    case_ids = [c["case_id"] for c in rec["comparison"]["cases"]]
    ev = row.get("evidence") or {}
    prior_ids = ev.get("batch_case_ids", row["executable_case_ids"])
    row["executable_case_ids"] = case_ids
    row["evidence_tier"] = "numeric"
    row["measured_against"] = P1_SHA
    row["evidence"] = {"receipt": [rel], "non_binding_receipts": ev.get("non_binding_receipts", []),
                       "batch_case_ids": prior_ids, "tier": TWIN_TIER}
    row["measured_result"] = {**(row.get("measured_result") or {}), "twin_case_ids": case_ids,
                              "twin_verdict": rec["verdict"]}
    return row


def apply_twins():
    cm = load(OUT / "case-map-postfit.json")
    cm["rows"] = [ruling_overlay(twin_overlay(r)) for r in cm["rows"]]
    cm["counts"] = counts = recount(cm)
    write_json(OUT / "case-map-postfit.json", cm)
    print(json.dumps(counts))


def check_twins():
    cur = load(OUT / "case-map-postfit.json")
    import copy
    exp = copy.deepcopy(cur)
    exp["rows"] = [ruling_overlay(twin_overlay(r)) for r in exp["rows"]]
    exp["counts"] = counts = recount(exp)
    bad = [a["source_id"] for a, b in zip(cur["rows"], exp["rows"]) if a != b]
    if bad or cur["counts"] != counts:
        print("STALE\n  rows differ from the twin and ruling overlays: " + ", ".join(bad) +
              f"\n  counts {cur['counts']} vs {counts}")
        raise SystemExit(1)
    # Rows paid by the rerun wave6 batch must equal a fresh derivation from their tracked case receipts.
    p0 = {r["source_id"]: r for r in load(P0_CASEMAP)["rows"]}
    receipts = tracked_receipts()
    w6 = [r["source_id"] for r in cur["rows"] if set(r["executable_case_ids"]) & WAVE6_POSTFIT_CIDS]
    stale_w6 = [sid for sid in w6 for row in [next(r for r in cur["rows"] if r["source_id"] == sid)]
                if derive_row(sid, row["carry_scan_status"], p0, receipts)[0] != row]
    if stale_w6:
        print("STALE\n  wave6 rows differ from their case receipts: " + ", ".join(stale_w6))
        raise SystemExit(1)
    raw = REC / "postfit-policy-p1"
    pj, po = load(raw / "julia-results.json"), load(raw / "r-oracle.json")
    drift = []
    for cid in INTEGER_EQUALITY:
        rec = load(REC / "cases" / f"{cid}.json")
        if (rec.get("comparison") != {"pin": "P1", "cases": [integer_entry(cid, pj["cases"][cid], po)]}
                or rec["harness_fields"] != pj["cases"][cid]
                or any(rec.get(k) != v for k, v in INTEGER_TEXT[cid].items())):
            drift.append(cid)
    frec = load(REC / "cases" / f"{FALLBACK_CID}.json")
    frow = next(r for r in cur["rows"] if r["executable_case_ids"] == [FALLBACK_CID])
    if "comparison" in frec or frow["evidence_tier"] == "numeric" or any(frec.get(k) != v for k, v in FALLBACK_TEXT.items()):
        drift.append(FALLBACK_CID + " (must stay unbound)")
    if drift:
        print("STALE\n  integer-equality receipts differ from the raw re-derivation: " + ", ".join(drift))
        raise SystemExit(1)
    n = sum(1 for r in cur["rows"] if r["source_id"] in TWIN_FILES and (ROOT / TWIN_REL / f"{TWIN_FILES[r['source_id']]}.json").is_file())
    print("CORE070_POSTFIT_TWINS_CURRENT", n, "twin rows,", len(cur["rows"]), "rows")


POINT_BATCHES = [
    ("surface-conversion-p1", "tools/core070_surface_conversion_batch.R + .jl, GLLVM_PARITY_PIN=P1",
     OUT / "surface-conversion-batch-contract-p1.json"),
    ("estimand-rebind-p1", "tools/core070_estimand_rebind_batch.R + .jl, GLLVM_PARITY_PIN=P1 (no contract file)", None),
    ("wave6-conversion-p1", "tools/core070_wave6_conversion_batch.R + .jl, GLLVM_PARITY_PIN=P1 (contract from PR #567; "
                            "nobs case rewritten as integer_equality under maintainer ruling 2026-10-05, D-319, item N2)",
     OUT / "wave6-conversion-batch-contract-p1.json"),
    ("wave7-conversion-p1", "tools/core070_wave7_conversion_batch.R + .jl, GLLVM_PARITY_PIN=P1",
     OUT / "wave7-conversion-batch-contract-p1.json"),
    ("wave8-conversion-p1", "tools/core070_wave8_conversion_batch.R + .jl, GLLVM_PARITY_PIN=P1",
     OUT / "wave8-conversion-batch-contract-p1.json"),
]
WAVE6_POSTFIT_CIDS = {"CORE070-WAVE6-POSTFIT-LOGLIK-MULTI", "CORE070-WAVE6-POSTFIT-CONFINT-MULTI",
                      "CORE070-WAVE6-POSTFIT-NOBS-MULTI"}


def common_block(head, dirty, host):
    return {"pin": "P1", "reference_commit": P1_SHA, "p0_reference_commit": P0_SHA,
            "gllvmtmb_version": PINS["P1"]["version"],
            "oracle_build_receipt": ORACLE_BUILD, "oracle_source_receipt": ORACLE_SOURCE,
            "glvmodels_commit": head, "glvmodels_worktree_dirty": dirty,
            "glvmodels_src_tree": git("rev-parse", f"{head}:src").strip(),
            "host": host}


def receipt_tuple(path, rec):
    """The (path, kind, verdict, batch verifier status, discriminating) tuple derive_row reads."""
    comparison = (rec.get("comparison") or {}).get("cases")
    disc = all(e.get("discriminating", True) for e in comparison) if comparison else True
    return (str(path.relative_to(ROOT)), rec["evidence_kind"], rec["verdict"], rec["batch_verifier"]["status"], disc)


def make_emit(common, receipts):
    def emit(cid, kind, verdict, body, comparison=None):
        rec = {"schema": "core070-postfit-p1-case-receipt/v2", "case_id": cid, "verdict": verdict,
               "evidence_kind": kind, **body, **common}
        if comparison is not None:
            rec["comparison"] = {"pin": "P1", "cases": comparison}
        path = REC / "cases" / f"{cid}.json"
        write_json(path, rec)
        receipts[cid] = receipt_tuple(path, rec)
    return emit


def tracked_receipts():
    return {p.stem: receipt_tuple(p, load(p)) for p in sorted((REC / "cases").glob("*.json"))}


def apply_wave6(rd, runtimes, allow_dirty):
    """Re-ingest one rerun of the wave6 batch (maintainer ruling 2026-10-05, D-319, item N2) without the other seven
    run directories: copy its artifacts, run its verifier, rewrite its postfit case receipts, re-derive the rows they
    pay. The other rows, receipts and batches are untouched."""
    head, dirty = git_state()
    if dirty and not allow_dirty:
        raise SystemExit("tracked files are modified outside this tool's outputs; commit first or pass "
                         "--allow-dirty: " + ", ".join(dirty))
    check_run_commit(rd, head)
    d, batch, contract_path = next(b for b in POINT_BATCHES if b[0] == "wave6-conversion-p1")
    artifacts = copy_batch(rd, d, ["receipt.json", "results.tsv", "julia-results.json", "r-oracle.json", "run-commit.json"])
    verifier = run_verifier(d, state=rd)
    artifacts.append(verifier["log"])
    receipts = {}
    emit = make_emit(common_block(head, dirty, "local Mac (M1 Ultra), OPENBLAS/OMP threads 1, JULIA_NUM_THREADS=1"),
                     receipts)
    ccases = {c["case_id"]: c for c in load(contract_path)["cases"]}
    point_batch_cases(d, load(rd / "julia-results.json"), load(rd / "r-oracle.json"), load(rd / "receipt.json"),
                      ccases, batch, contract_path, verifier=verifier, emit=emit)
    if set(receipts) != WAVE6_POSTFIT_CIDS:
        raise SystemExit(f"wave6 rerun wrote receipts for {sorted(receipts)}, expected {sorted(WAVE6_POSTFIT_CIDS)}")
    cm = load(OUT / "case-map-postfit.json")
    cm["batch_verifiers"][d] = verifier
    cm["batch_artifacts"][d] = artifacts
    cm["runtimes_seconds"]["wave6_conversion_rerun_2026_10_05"] = {**runtimes, "glvmodels_commit": head}
    p0 = {r["source_id"]: r for r in load(P0_CASEMAP)["rows"]}
    for i, row in enumerate(cm["rows"]):
        if set(row["executable_case_ids"]) & WAVE6_POSTFIT_CIDS:
            cm["rows"][i] = derive_row(row["source_id"], row["carry_scan_status"], p0, receipts)[0]
    cm["counts"] = recount(cm)
    write_json(OUT / "case-map-postfit.json", cm)
    print(json.dumps(cm["counts"]))
    print("wave6 verifier", verifier["status"], {c: receipts[c][2] for c in sorted(receipts)})


def point_batch_cases(d, julia, oracle, breceipt, ccases, batch, contract_path, verifier, emit):
    """Emit one receipt per postfit case of a point-style batch (shared by the full run and --apply-wave6)."""
    for cid, jc in julia["cases"].items():
        cc = ccases.get(cid, {})
        srcs = cc.get("source_ids") or ([cc["source_id"]] if cc.get("source_id") else [])
        if d == "estimand-rebind-p1":
            srcs = [ESTIMAND_SOURCE[cid]]
        if not any(s.startswith("postfit") for s in srcs):
            continue  # namespace / inference / covariance cases in the same batch are out of scope
        kind_h = jc.get("kind") or cc.get("kind") or "point"
        passed = bool(jc.get("pass"))
        body = {"source_ids": srcs, "batch": batch, "harness_kind": kind_h, "harness_pass": passed,
                "batch_status": breceipt["status"], "batch_verifier": verifier,
                "batch_status_note": (WAVE6_NOTE if d == "wave6-conversion-p1" and breceipt["status"] != "PASS" else ""),
                "r_call": cc.get("r_call"), "julia_call": cc.get("julia_call"),
                "raw": [f"{REC_REL}/{d}/julia-results.json", f"{REC_REL}/{d}/r-oracle.json"]}
        if d == "estimand-rebind-p1":
            body["p0_to_p1_accessor_record"] = ESTIMAND_ACCESSOR_RECORD
        if kind_h in ("point", "ci") and "max_abs_diff" in jc:
            tol = jc["tolerance"]
            rule = (f"{contract_path.name if contract_path else 'tools/core070_verify_estimand_rebind_batch.py TOLERANCE'}"
                    f" per-case tolerance (carried verbatim from P0); max |R - Julia| elementwise")
            rv = oracle["oracle_values"].get(cid)
            jv = jc.get("julia_values")
            entry = (vec_entry(cid, jc.get("quantity") or cc.get("quantity"), rv, jv, tol, jc["max_abs_diff"], rule)
                     if isinstance(rv, (list, float, int)) and jv is not None
                     else harness_entry(cid, jc.get("quantity") or cc.get("quantity"), jc["max_abs_diff"], tol, rule,
                                        rv if isinstance(rv, list) else ([rv] if isinstance(rv, (float, int)) else None)))
            ok = passed and entry["max_abs_diff"] <= tol
            emit(cid, "numeric_r_vs_julia", "PASS" if ok else "FAIL", body, [entry])
        elif kind_h == "integer_equality":
            entry = wave6_integer_entry(cid, jc, oracle)
            body.update(numeric_note=WAVE6_INTEGER_WHY, ruling=cc.get("ruling"), expected=cc.get("expected"))
            emit(cid, "numeric_r_vs_julia", "PASS" if passed and entry["max_abs_diff"] == 0 else "FAIL", body, [entry])
        elif kind_h == "own_receipt_defect":
            body.update(measured={k: jc.get(k) for k in ("r_nobs", "julia_nobs", "r_expected_p_times_n", "julia_expected_n")},
                        frozen_expectation={"r": cc.get("expected_r_value_formula"), "julia": cc.get("expected_julia_value_formula")},
                        why_failing=("The frozen contract case asserts each engine against its own formula: R == p*n and "
                                     "Julia == n (a known defect pending decision). At P1 R returns p*n = 400 as at P0, but "
                                     "Julia now also returns 400, so the Julia-side expectation (n = 80) no longer holds and "
                                     "the harness reports FAIL. R and Julia agree; the expectation, not the parity, is what "
                                     "failed. Not edited here; recorded as failing."))
            emit(cid, "own_receipt_defect_expectation", "FAIL" if not passed else "PASS", body)
        else:
            verdict_fields = {k: jc.get(k) for k in ("r_verdict", "julia_verdict", "r_frobenius", "julia_frobenius", "tolerance") if k in jc}
            body.update(measured=verdict_fields,
                        why_not_numeric={"verdict": "Both engines report a boolean verdict and the harness checks they agree; "
                                                    "there is no R-vs-Julia number with a tolerance.",
                                         "own_consistency": "Each engine is checked against itself (Frobenius self-consistency); "
                                                            "the two are not compared with each other."}.get(kind_h, f"harness kind {kind_h}"))
            emit(cid, f"paired_{kind_h}", "PASS" if passed else "FAIL", body)



CARRIED_KEYS = ("uncertain", "reclassify_proposed", "original_classification")


def derive_row(sid, carry, p0, receipts):
    """One case-map row from the P0 row and the case receipts: (row, count key before overlays, count key after).

    receipts maps case_id -> (path, kind, verdict, batch verifier status, discriminating). Shared by the full run
    and the --apply-wave6 overlay, so both derive a row the same way."""
    base = p0[sid]
    ids = base["executable_case_ids"]
    row = {"source_id": sid, "classification": base["classification"], "arc": "A3", "carry_scan_status": carry,
           "executable_case_ids": ids, "disposition": base.get("disposition")}
    for k in CARRIED_KEYS:
        if k in base:
            row[k] = base[k]
    have = [receipts.get(i) for i in ids]
    if carry == "RETIRED":
        row.update(evidence_tier="not_measured", measured_against=None, evidence={},
                   reason=("The R export was removed between P0 and P1 (NAMESPACE at P1 no longer exports it); the row has "
                           "no executable case ids and keeps its P0 classification and disposition. Retiring or "
                           "re-scoping the row is a maintainer decision."))
        key = "retired_at_p1_not_measured"
    elif not ids or any(h is None for h in have):
        missing = [i for i, h in zip(ids, have) if h is None]
        if sid == "postfit-policy/POST-DEVIANCE":
            row.update(evidence_tier="needs_surface_not_executed", measured_against=None, evidence={},
                       reason=("Its executable case CORE070-POSTFIT-DEVIANCE-NATIVE was never authored into the "
                               "postfit-policy batch (the P0 contract lists POST-DEVIANCE under needs_new_julia_surface), "
                               "so nothing ran at P1. deviance.gllvmTMB_multi is measured numerically at P1 under a "
                               "different case id (CORE070-WAVE8-DEVIANCE-MULTI, row postfit/POSTFIT-SURFACE-"
                               "deviance.gllvmTMB_multi); rebinding is a maintainer decision."))
            key = "needs_surface_not_executed"
        else:
            row.update(evidence_tier="not_measured", measured_against=None, evidence={},
                       reason=f"Not re-measured at P1 in this PR (missing: {', '.join(missing)}).")
            key = "not_measured"
    else:
        kinds = {h[1] for h in have}
        verdicts = {i: h[2] for i, h in zip(ids, have)}
        paths = [h[0] for h in have]
        all_pass = all(v == "PASS" for v in verdicts.values())
        batch_ok = {i: h[3] for i, h in zip(ids, have)}
        disc = {i: h[4] for i, h in zip(ids, have)}
        if kinds == {"numeric_r_vs_julia"} and all_pass and not all(v == "PASS" for v in batch_ok.values()):
            # PR #567's gate: a batch whose own verifier rejected the run cannot bind a row
            row.update(evidence_tier="numeric_held_batch_verifier_failed", measured_against=P1_SHA,
                       evidence={"non_binding_receipts": paths,
                                 "tier": "numeric comparison blocks pass, but the batch verifier rejected the "
                                         "run, so the row does not bind"},
                       note=HELD_NOTE,
                       measured_result={"case_verdicts": verdicts, "batch_verifier": batch_ok,
                                        "discriminating": disc})
            key = "numeric_held_batch_verifier_failed"
        elif kinds == {"numeric_r_vs_julia"} and all_pass and not all(disc.values()):
            # review finding 1: a comparison that cannot discriminate does not bind
            row.update(evidence_tier="numeric_non_discriminating", measured_against=P1_SHA,
                       evidence={"non_binding_receipts": paths,
                                 "tier": "numeric comparison blocks pass, but at least one is degenerate (the R "
                                         "values are one constant or all ~0), so a constant or zero "
                                         "implementation would pass too; the row does not bind"},
                       note=NON_DISCRIMINATING_NOTES.get(sid, "flagged by the degenerate-comparison gate; see the "
                                                              "comparison blocks' degenerate_reason"),
                       measured_result={"case_verdicts": verdicts, "batch_verifier": batch_ok,
                                        "discriminating": disc})
            key = "numeric_non_discriminating"
        elif kinds == {"numeric_r_vs_julia"} and all_pass:
            bind_numeric(row, paths, verdicts, batch_ok,
                         INTEGER_TIER if all(i in INTEGER_EQUALITY for i in ids) else
                         "numeric: every executable case receipt carries an R-vs-Julia comparison "
                         "block pinned to P1, within the harness tolerance")
            key = "numeric_pass"
        elif not all_pass:
            row.update(evidence_tier="numeric_fail", measured_against=P1_SHA,
                       evidence={"non_binding_receipts": paths,
                                 "tier": "measured at P1; the harness verdict is FAIL, so the row does not bind"},
                       measured_result={"case_verdicts": verdicts, "batch_verifier": batch_ok, "row_verdict": "FAIL"})
            key = "numeric_fail"
        else:
            row.update(evidence_tier="partial_non_numeric_case", measured_against=P1_SHA,
                       evidence={"non_binding_receipts": paths,
                                 "tier": "measured at P1 and the harness passes, but at least one case is a verdict, "
                                         "own-consistency, exact-integer, empty-length or default-policy check with "
                                         "no R-vs-Julia number and tolerance, so the row does not bind"},
                       measured_result={"case_verdicts": verdicts, "batch_verifier": batch_ok,
                                        "case_kinds": {i: h[1] for i, h in zip(ids, have)}})
            key = "partial_non_numeric_case"
    before = row["evidence_tier"]
    twin_overlay(row)
    after = key if row["evidence_tier"] == before else tier_count_key(row["evidence_tier"])
    ruling_overlay(row)
    return row, key, after



def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path)
    ap.add_argument("--runtimes", type=Path)
    ap.add_argument("--apply-twins", action="store_true",
                    help="re-apply the Julia twin overlay to the tracked case-map-postfit.json")
    ap.add_argument("--check-twins", action="store_true",
                    help="verify case-map-postfit.json equals its twin-overlay re-derivation; write nothing")
    ap.add_argument("--apply-integer-equality", action="store_true",
                    help="write the integer_equality comparison blocks and bind the four integer rows (idempotent)")
    ap.add_argument("--apply-wave6", type=Path, metavar="RUN_DIR",
                    help="re-ingest one rerun of the wave6 batch (needs --runtimes; D-319 item N2)")
    ap.add_argument("--apply-rulings", action="store_true",
                    help="write the 2026-10-05 signed dispositions (D-319) onto the tracked case map (idempotent)")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="write receipts from a checkout with modified tracked files (recorded, not hidden)")
    args = ap.parse_args()
    if args.check_twins:
        check_twins()
        return
    if args.apply_twins:
        apply_twins()
        return
    if args.apply_integer_equality:
        apply_integer_equality()
        return
    if args.apply_wave6 is not None:
        if args.runtimes is None:
            ap.error("--apply-wave6 needs --runtimes")
        apply_wave6(args.apply_wave6, load(args.runtimes), args.allow_dirty)
        return
    if args.apply_rulings:
        apply_twins()
        return
    if args.runs is None or args.runtimes is None:
        ap.error("--runs and --runtimes are required unless --apply-twins or --check-twins")
    runs = args.runs
    runtimes = load(args.runtimes)
    head, dirty = git_state()
    if dirty and not args.allow_dirty:
        raise SystemExit("tracked files are modified outside this tool's outputs; commit first or pass "
                         "--allow-dirty: " + ", ".join(dirty))
    run_dirs = ["surface-conversion-p1", "estimand-rebind-p1", "wave6-conversion-p1", "wave7-conversion-p1",
                "wave8-conversion-p1", "postfit-policy-p1", "postfit-1-r-p1", "postfit-1-julia-p1"]
    for d in run_dirs:
        check_run_commit(runs / d, head)

    common = common_block(head, dirty, "local Mac (M1 Ultra), OPENBLAS/OMP threads 1, JULIA_NUM_THREADS=4")
    receipts = {}  # case_id -> (path, kind, verdict, batch verifier status, discriminating)
    artifacts = {}
    verifiers = {}
    emit = make_emit(common, receipts)

    # ---- point-style batches: surface-conversion, estimand-rebind, wave6, wave7, wave8 ----
    for d, batch, contract_path in POINT_BATCHES:
        rd = runs / d
        artifacts[d] = copy_batch(rd, d, ["receipt.json", "results.tsv", "julia-results.json", "r-oracle.json",
                                          "run-commit.json"])
        julia, oracle, breceipt = load(rd / "julia-results.json"), load(rd / "r-oracle.json"), load(rd / "receipt.json")
        ccases = {c["case_id"]: c for c in load(contract_path)["cases"]} if contract_path else {}
        verifiers[d] = run_verifier(d, state=rd)
        artifacts[d].append(verifiers[d]["log"])
        point_batch_cases(d, julia, oracle, breceipt, ccases, batch, contract_path, verifier=verifiers[d], emit=emit)

    # ---- postfit-1 (coef readback) ----
    r1, j1 = runs / "postfit-1-r-p1", runs / "postfit-1-julia-p1"
    artifacts["postfit-1-p1"] = copy_batch(r1, "postfit-1-p1", ["receipt.json", "postfit-1-r-results.json"]) + \
        copy_batch(j1, "postfit-1-p1", ["postfit-1-julia-results.json"])
    shutil.copyfile(r1 / "run-commit.json", REC / "postfit-1-p1" / "run-commit-r.json")
    shutil.copyfile(j1 / "run-commit.json", REC / "postfit-1-p1" / "run-commit-julia.json")
    verifiers["postfit-1-p1"] = run_verifier("postfit-1-p1", r_state=r1, julia_state=j1)
    artifacts["postfit-1-p1"] += [f"{REC_REL}/postfit-1-p1/run-commit-r.json", f"{REC_REL}/postfit-1-p1/run-commit-julia.json",
                                  verifiers["postfit-1-p1"]["log"]]
    rr, jr = load(r1 / "postfit-1-r-results.json"), load(j1 / "postfit-1-julia-results.json")
    c1 = load(OUT / "postfit-1-batch-contract-p1.json")["executable_batch"]
    tol = c1["tolerance"]["max_abs_diff"]
    cid = c1["case_id"]
    entry = vec_entry(cid, "coef", rr["coef"], jr["coef"], tol, None,
                      "postfit-1-batch-contract-p1.json executable_batch.tolerance.max_abs_diff (carried verbatim from P0)")
    emit(cid, "numeric_r_vs_julia", "PASS" if entry["max_abs_diff"] <= tol and rr["all_checks"] and jr["all_checks"] else "FAIL",
         {"source_ids": [c1["source_id"]], "batch": "tools/core070_postfit_1_batch.R then .jl, GLLVM_PARITY_PIN=P1",
          "batch_verifier": verifiers["postfit-1-p1"],
          "raw": [f"{REC_REL}/postfit-1-p1/postfit-1-r-results.json", f"{REC_REL}/postfit-1-p1/postfit-1-julia-results.json"]},
         [entry])

    # ---- postfit-policy ----
    rp = runs / "postfit-policy-p1"
    artifacts["postfit-policy-p1"] = copy_batch(rp, "postfit-policy-p1", ["receipt.json", "results.tsv", "julia-results.json",
                                                                         "r-oracle.json", "run-commit.json"])
    verifiers["postfit-policy-p1"] = run_verifier("postfit-policy-p1", state=rp)
    artifacts["postfit-policy-p1"].append(verifiers["postfit-policy-p1"]["log"])
    pj, po = load(rp / "julia-results.json"), load(rp / "r-oracle.json")
    pc = load(OUT / "postfit-policy-batch-contract-p1.json")
    ptol = pc["tolerances"]
    for c in pc["cases"]:
        cid = c["case_id"]
        jc = pj["cases"][cid]
        passed = bool(jc["pass"])
        body = {"source_ids": [c["source_id"]], "batch": "tools/core070_postfit_policy_batch.R + .jl, GLLVM_PARITY_PIN=P1",
                "harness_pass": passed, "batch_verifier": verifiers["postfit-policy-p1"], "r_call": c.get("r_call"),
                "julia_surface": c.get("julia_surface"), "comparand": c.get("comparand"), "check": c.get("check"),
                "harness_fields": jc,
                "raw": [f"{REC_REL}/postfit-policy-p1/julia-results.json", f"{REC_REL}/postfit-policy-p1/r-oracle.json"]}
        if cid in POLICY_TOL_KEY:
            key = POLICY_TOL_KEY[cid]
            r_vals = [v for k in POLICY_R_KEYS[cid] for v in (po[k] if isinstance(po[k], list) else [po[k]])]
            e = harness_entry(cid, c.get("comparand"), jc["delta"], ptol[key],
                              f"postfit-policy-batch-contract-p1.json tolerances.{key} (carried verbatim from P0)", r_vals)
            emit(cid, "numeric_r_vs_julia", "PASS" if passed and jc["delta"] <= ptol[key] else "FAIL", body, [e])
        else:
            if cid in POLICY_PARTIAL_WITH_NUMERIC_LEG:
                why = ("The link-scale value leg is numeric (delta recorded), but the fact this case pays is a documented "
                       "default divergence (R predict type default 'link', Julia :response). A default that differs is not "
                       "numeric parity, so the case carries no comparison block.")
            elif cid == FALLBACK_CID:
                why = FALLBACK_WHY
                body.update(FALLBACK_TEXT)
            elif cid in INTEGER_EQUALITY:
                body["numeric_note"] = INTEGER_WHY
                body.update(INTEGER_TEXT[cid])
                emit(cid, "numeric_r_vs_julia", "PASS" if passed else "FAIL", body, [integer_entry(cid, jc, po)])
                continue
            elif "length" in (c.get("comparand") or ""):
                why = "Both sides return an empty coefficient vector; there is no number to compare."
            else:
                why = "Signature-level documented-divergence check (default and keyword reflection), not a numeric comparison."
            body["why_not_numeric"] = why
            emit(cid, "paired_policy_check", "PASS" if passed else "FAIL", body)

    # ---- case-map rows ----
    p0 = {r["source_id"]: r for r in load(P0_CASEMAP)["rows"]}
    out_rows = []
    counts = {"numeric_pass": 0, "numeric_fail": 0, "numeric_held_batch_verifier_failed": 0,
              "numeric_non_discriminating": 0, "partial_non_numeric_case": 0,
              "needs_surface_not_executed": 0, "retired_at_p1_not_measured": 0, "not_measured": 0}
    in_scope = [(f"postfit/POSTFIT-SURFACE-{s}", "DANGLING") for s in POSTFIT_DANGLING] + \
               [(f"postfit/POSTFIT-SURFACE-{s}", "RETIRED") for s in POSTFIT_RETIRED] + \
               [(f"postfit-policy/{s}", "DANGLING") for s in POLICY_DANGLING]
    for sid, carry in in_scope:
        out_rows.append(derive_row(sid, carry, p0, receipts)[0])
    counts = recount({"counts": counts, "rows": out_rows})

    casemap = {
        "schema": 1, "reference_commit": P1_SHA,
        "scope": ("postfit family: the 36 rows the P1 carry scan lists as DANGLING (34) or RETIRED (2); postfit-policy "
                  "family: the 16 rows it lists as DANGLING. NOT_BOUND_AT_P0 rows (64 postfit, 5 postfit-policy) had no "
                  "P0 evidence and are out of scope here."),
        "note": ("Separate from case-map.json so none of its rows are touched; read by tools/true_parity_check.mjs with "
                 "PARITY_CASEMAP pointing at this file. Classification, disposition and the P0 `uncertain` / "
                 "`reclassify_proposed` fields are carried from docs/dev-log/core070/required-source-case-map.json "
                 "unchanged; nothing is signed by an agent. Only rows whose every executable case id carries a numeric "
                 "R-vs-Julia comparison block within tolerance cite evidence.receipt; every other measured row cites "
                 "evidence.non_binding_receipts, so it reads as not bound under both the current checker and the "
                 "numeric-tier rule proposed in PR #561. Two further holds cite non-binding receipts: a row whose "
                 "batch verifier rejected the run (numeric_held_batch_verifier_failed; no exception path) and a row "
                 "with a degenerate comparison (numeric_non_discriminating)."),
        "generator": "tools/core070_postfit_p1_receipts.py",
        "glvmodels_commit": head,
        "batch_verifiers": verifiers,
        "counts": counts, "runtimes_seconds": runtimes,
        "batch_artifacts": artifacts,
        "rows": out_rows,
    }
    write_json(OUT / "case-map-postfit.json", casemap)
    print(json.dumps(counts))
    print("case receipts", len(receipts), "artifacts", sum(len(v) for v in artifacts.values()))


if __name__ == "__main__":
    main()
