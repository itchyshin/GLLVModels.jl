#!/usr/bin/env python3
"""Write the true-parity campaign receipts (C3, C4, C5) from RAW engine outputs, and the case-map rows.

Ruling itchyshin/GLLVModels.jl#684 item 4 (never cite it as a bare number) adds the rows proposed in PR #650 and
runs the campaign. It does not quote that plan's tolerances, pass rule, row placement or licence handling: they are
carried here AS PROPOSED in PR #650 and still wait for the maintainer's confirmation. This script is the only
thing that writes the campaign receipts and the 20 campaign case-map rows: nothing is typed by hand,
and no value is invented. It reads

  <raw>/<cell>_R.json   written by tools/true_parity/campaign/run_R.R   (private P1 gllvmTMB, 0.7.1)
  <raw>/<cell>_J.toml   written by tools/true_parity/campaign/run_J.jl   (GLLVModels.jl)

for each C3/C4 cell, the tracked P1 phylo-latent receipts for COV-PHYLO-LATENT-RSZ (plus one fresh
Julia fit on the same literal fixture), and the merged PR #593 grouping receipts for C5. It writes

  receipts/<family>/campaign/<row>.json     one receipt per row, with a `comparison` block pinned to P1
  receipts/<family>/campaign/raw/*.gz       the raw outputs the receipt was built from (gzip)
  case-map-<family>.json                    the row, appended or (by source_id) replaced; nothing else touched

Pass rule (campaign plan section 1.1, as PROPOSED in PR #650): BOTH engines converged (R convergence 0
with a positive-definite Hessian; Julia `converged` true; iSDM also every cell converged) AND every
listed quantity is within its tolerance AND both engines read the same data bytes AND the ten named gllvmTMB entry points
that run_R.R calls deparse identically to their P1 source (internal gllvmTMB functions and the compiled library are not
deparse-checked: they are trusted by the recorded version 0.7.1, the library path and the sha256 of the P1 source
files). A C4 row follows the same rule: maintainer ruling 2026-10-05 (vault D-319) changed the C4 clause text to accept
direct-engine runs, so the engine = "julia" bridge leg the plan (section 1.3) asked for is no longer part of the rule
(that leg tested R's adapter, not the model). The phylo row (COV-PHYLO-LATENT-RSZ) takes its R side from the
tracked PR #547 receipt, which binds only once it records qualified = true under the maintainer's dated promotion block
(D-300 answer 9). The maintainer signed that block on 2026-10-05 (vault D-319); it is recorded in
docs/dev-log/core070/phylo-latent-p1/README.md by tools/phylo_latent/promote_p1.py. No agent signs it. C5 rows follow their own rule (see RULE_C5).
A row that meets the whole rule binds (evidence_tier "numeric", evidence.receipt). Otherwise the receipt is cited as
non-binding with the reason and the row is left unbound; no tolerance is widened and nothing is re-run to get a pass.
A row whose numbers are all inside tolerance and whose only failing leg is a required step that was not signed (the
phylo qualification) reads partial_case_not_executed, never numeric_fail.

Convergence parity (maintainer ruling 2026-10-05, D-319, N9; applied only to receipts that carry the block, as the
maintainer chose). When both raw outputs record the gradient max-abs of their own objective at the returned point
(R `max_abs_gradient`, Julia `max_abs_gradient`), the receipt gets a top-level `convergence_parity` block
({gradient_bound 1e-5, compared_point "returned", engines.R / engines.julia max_abs_gradient}) and Julia's gradient in
its engine block; the checker then binds the row only if both are <= 1e-5. No committed raw Julia output records that
field yet, so no receipt carries the block (the polished harness of PR #715 is not merged).

Signed C4 dispositions. RD-PHYLO-DISPOSITION, RD-TEMPORAL-DISPOSITION and RD-ISDM-DISPOSITION are written as
DISPOSITION-SIGNED under item F of the rulings page, signed by the maintainer on 2026-10-05 (vault D-319); an agent
records the signature, no agent signs it.

A relative tolerance is recorded as a discrepancy statistic against zero (r_value = 0, julia_value =
|J - R| / |R| per element), because the checker compares absolute differences; the r_value is therefore NOT an R
measurement. The case's quantity name says "relative difference" and `raw_values_location` says where the raw R and
Julia values are (r_raw / julia_raw in the same block, and the committed raw files). Vectors of 1000+ values (linear
predictors) carry max_abs_diff only, with the raw files committed (gzip). `--check` rebuilds every receipt's comparison
block, pass-rule legs, verdict and engine blocks, and every campaign case-map row, from those committed raw files and
fails on any difference.

Urbanisation (privacy). The urbanisation matrix is the maintainer's unpublished data, so the row's per-observation raw
outputs (linear predictors, loadings) are NOT committed. Its receipt keeps only summary values (logLik, fixed effects,
max differences, wall times, the data file's sha256 and the recorded sha256 of the two raw files). `--check` prints
that the raw outputs are kept off the public repo and checks the summary receipt only for internal consistency; give
`--local-raw DIR` (a directory holding the maintainer's retained urban_R.json and urban_J.toml) to rebuild it in full.

Usage:
  python3 write_receipts.py --raw DIR [--apply]       # default: print what would be written
  python3 write_receipts.py --check [--local-raw DIR] # rebuild every campaign receipt and row from the committed raw files; fail on drift
"""
from __future__ import annotations
import argparse, gzip, hashlib, json, math, os, subprocess, sys, tempfile, tomllib
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LEDGER = Path("docs/dev-log/core070/true-parity-latest")
P1 = "9539352f66f2db2cc26b1c393e67212a359b60c9"
RULING = "itchyshin/GLLVModels.jl#684 item 4"
SCHEMA = "true-parity-campaign-receipt/v1"
BIG = 1000  # vectors at least this long carry max_abs_diff only


# ---- small helpers ------------------------------------------------------------------------------
def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha_file(p): return sha_bytes(Path(p).read_bytes())
def fin(x): return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
def flat(x):
    if isinstance(x, list):
        out = []
        for e in x: out.extend(flat(e))
        return out
    return [x]
def maxabs(a, b): return max(abs(x - y) for x, y in zip(a, b))
def load_r(p): return json.loads(Path(p).read_text())
def load_j(p): return tomllib.loads(Path(p).read_text())
def gz_write(dst, src):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as f, gzip.GzipFile(filename="", mode="wb", fileobj=open(dst, "wb"), mtime=0) as g: g.write(f.read())
def gz_read(p): return gzip.decompress(Path(p).read_bytes())


class Case:
    """One comparison case. mode abs: judged on max |R - J|. mode rel: judged on max |J - R| / |R|."""
    def __init__(self, cid, quantity, r, j, tol, rule, tol_status, mode="abs"):
        r, j = flat(r), flat(j)
        # a relative tolerance is carried as a discrepancy statistic, so the quantity says so (r_value is not an R measurement)
        if mode == "rel": quantity = f"{quantity} relative difference"
        self.cid, self.quantity, self.tol, self.rule, self.tol_status, self.mode = cid, quantity, tol, rule, tol_status, mode
        self.problem = None
        if len(r) != len(j) or not r: self.problem = f"length mismatch or empty (R {len(r)}, Julia {len(j)})"
        elif not all(map(fin, r)) or not all(map(fin, j)): self.problem = "non-finite value on one side"
        self.r, self.j = r, j
        if self.problem: self.diff = None
        elif mode == "abs": self.diff = maxabs(r, j)
        else: self.diff = max(abs(b - a) / abs(a) if a != 0 else (0.0 if b == a else math.inf) for a, b in zip(r, j))
        self.ok = self.problem is None and self.diff is not None and math.isfinite(self.diff) and self.diff <= tol

    def block(self, raw_note):
        d = OrderedDict(case_id=self.cid, quantity=self.quantity)
        if self.mode == "abs":
            if len(self.r) < BIG:
                d["r_value"] = self.r if len(self.r) > 1 else self.r[0]
                d["julia_value"] = self.j if len(self.j) > 1 else self.j[0]
                d["diff_source"] = "recomputed by the checker from r_value and julia_value"
            elif raw_note == NOTE_OFF_REPO:
                d["diff_source"] = f"max over {len(self.r)} values, from raw files that are not committed ({NOTE_OFF_REPO}); this receipt records the maximum only"
            else:
                d["diff_source"] = f"max over {len(self.r)} values, from the committed raw files ({raw_note}); re-derived by write_receipts.py --check"
            d["max_abs_diff"] = self.diff
        else:
            d["convention"] = ("relative-difference statistic, NOT an R measurement: r_value is the target 0; julia_value is |J - R| / |R| per element "
                               "(the checker compares absolute differences, so a relative tolerance is carried this way)")
            d["raw_values_location"] = ("the raw R values are r_raw and the raw Julia values are julia_raw, in this block; the full engine outputs are "
                                        "the committed raw/<cell>_R.json.gz and raw/<cell>_J.toml.gz named in read_from")
            d["r_value"] = [0.0] * len(self.r) if len(self.r) > 1 else 0.0
            d["julia_value"] = [abs(b - a) / abs(a) if a != 0 else (0.0 if b == a else None) for a, b in zip(self.r, self.j)] if len(self.r) > 1 else (abs(self.j[0] - self.r[0]) / abs(self.r[0]) if self.r[0] != 0 else 0.0)
            d["r_raw"] = self.r if len(self.r) > 1 else self.r[0]
            d["julia_raw"] = self.j if len(self.j) > 1 else self.j[0]
            d["max_abs_diff"] = self.diff
            d["diff_source"] = "recomputed by the checker from r_value and julia_value (the relative statistic)"
        d["tolerance"] = self.tol; d["tolerance_rule"] = self.rule; d["tolerance_status"] = self.tol_status
        d["n_values"] = len(self.r); d["within_tolerance"] = bool(self.ok)
        if self.problem: d["problem"] = self.problem
        return d


def orient(v):
    """Sign convention of R/temporal.R:466-483 (the public one): first loading positive unless negligible, else the largest."""
    mx = max(abs(x) for x in v); i = 0
    if abs(v[0]) < 1e-8 * mx: i = max(range(len(v)), key=lambda k: abs(v[k]))
    return [-x for x in v] if v[i] < 0 else list(v)


# ---- row specifications -------------------------------------------------------------------------
TW = "twin: 1e-6 absolute on logLik, as the P0 worst case (2.3e-7) and every P1 twin"
PR = "PROPOSED in the campaign plan (PR #650), carried as proposed; not yet confirmed by the maintainer"
LL = ("logLik", 1e-6, "1e-6 absolute on the marginal log-likelihood", TW)
BETA = ("beta", 1e-4, "1e-4 absolute per fixed-effect estimate", PR)
SE = ("beta_se", 1e-3, "1e-3 relative per standard error of a fixed effect (R sdreport vs Julia Wald)", PR)
LLT = ("LLt", 1e-4, "1e-4 absolute per element of Lambda Lambda' (rotation and sign invariant)", PR)
DISP = ("dispersion", 1e-3, "1e-3 relative per trait dispersion (NB2 size)", PR)
ROWS = [
    # source_id, map family, cell, clause, quantities
    ("family/GAUSSIAN-IDENTITY-RSZ", "family", "gaussian", "C3", [LL, BETA, SE, LLT]),
    ("family/POISSON-LOG-RSZ", "family", "poisson", "C3", [LL, BETA, SE, LLT]),
    ("family/NB2-LOG-RSZ", "family", "nb2", "C3", [LL, BETA, SE, LLT, DISP]),
    ("family/BINOMIAL-LOGIT-RSZ", "family", "binomial", "C3", [LL, BETA, SE, LLT]),
    ("family/ORDINAL-LOGIT-RSZ", "family", "ordinal", "C3",
     [LL, ("thresholds", 1e-4, "1e-4 absolute per cutpoint (per-trait cutpoints 2..C-1; the first is fixed at 0 on both engines)", PR), LLT]),
    ("covariance/COV-PHYLO-LATENT-RSZ", "covariance", "phylo", "C3", None),
    ("covariance/COV-TEMPORAL-RSZ", "covariance", "temporal", "C3",
     [LL, ("temporal_phi", 1e-4, "1e-4 absolute on the AR(1) persistence phi", PR),
      ("temporal_loadings", 1e-4, "1e-4 absolute per loading (public sign orientation, R/temporal.R:466-483)", PR)]),
    ("isdm/ISDM-HEADLINE-RSZ", "isdm", "isdm", "C3",
     [LL, ("beta", 1e-4, "1e-4 absolute per b_fix, matched by name", PR),
      ("predict_link", 1e-6, "1e-6 absolute, fitted cells, link scale", PR), ("predict_response", 1e-6, "1e-6 absolute, fitted cells, response scale", PR),
      ("predict_newdata_offset0_link", 1e-6, "1e-6 absolute, newdata with offset 0, link scale", PR),
      ("predict_newdata_offset0_response", 1e-6, "1e-6 absolute, newdata with offset 0, response scale", PR)]),
    ("data/RD-URBANISATION-BINOMIAL", "data", "urban", "C4", [LL, BETA, LLT, ("eta", 1e-6, "1e-6 absolute, link scale on the training data", PR)]),
    ("data/RD-CRABS-GAUSSIAN", "data", "crabs", "C4", [LL, BETA, LLT, ("eta", 1e-6, "1e-6 absolute, link scale on the training data", PR)]),
    ("data/RD-SPIDER-NB2", "data", "spider", "C4", [LL, BETA, LLT, ("eta", 1e-6, "1e-6 absolute, link scale on the training data", PR)]),
    ("data/RD-BEETLE-NB2", "data", "beetle", "C4", [LL, BETA, LLT, ("eta", 1e-6, "1e-6 absolute, link scale on the training data", PR)]),
    ("data/RD-FUNGI-BINOMIAL", "data", "fungi", "C4", [LL, BETA, LLT, ("eta", 1e-6, "1e-6 absolute, link scale on the training data", PR)]),
]
DISPOSITIONS = {
    "data/RD-PHYLO-DISPOSITION": dict(
        capability="phylo-structured real data",
        text="gllvm::fungi ships a phylogeny (fungi$tree), so a real phylo workflow exists, but the campaign has no "
             "phylo-structured real-data run at P1. C4 accepts a direct-engine run (R gllvmTMB and Julia GLLVModels "
             "fitted to the same data), so one on fungi with fungi$tree would replace this disposition; revisit at P2."),
    "data/RD-TEMPORAL-DISPOSITION": dict(
        capability="temporal real data",
        text="None of gllvm, vegan, MASS, ape ships a multivariate ecological time series the plan would call a real workflow; "
             "no bridge route for temporal at P1 (D-296)."),
    "data/RD-ISDM-DISPOSITION": dict(
        capability="integrated SDM real data",
        text="No real multi-source dataset in the packages checked; no bridge route for iSDM at P1 (D-296, D-300 row 1c-5)."),
}
C5 = [("fit-input/GRP-UNIT", "unit"), ("fit-input/GRP-UNIT-OBS", "unit_obs"), ("fit-input/GRP-CLUSTER", "cluster"), ("fit-input/GRP-CLUSTER2", "cluster2")]
CAMPAIGN_DIR = "campaign"
# Cells whose per-observation raw outputs are NOT committed: the urbanisation matrix is the maintainer's unpublished data.
OFF_REPO = {"urban": "data/RD-URBANISATION-BINOMIAL"}
NOTE_OFF_REPO = "raw outputs kept off the public repo; re-derive locally with URBMAP_ROOT set"
SYNTH = ("gaussian", "poisson", "nb2", "binomial", "ordinal", "temporal", "isdm")   # synthetic cells: data committed under campaign/data/
COVS = {"spider": ["ConWate", "BareSand", "CovMoss"], "beetle": ["pH", "Moist", "Org"], "fungi": ["TEMPR", "PRECIP", "logAREA"]}


def slug(sid): return sid.split("/", 1)[1]


# ---- per-cell case builders ---------------------------------------------------------------------
def cid(clause, sid, q): return f"CAMPAIGN-{clause}-{slug(sid)}-{q.upper().replace('_', '-')}"


def cases_for(sid, cell, clause, quants, R, J):
    cs = []
    def add(q, tol, rule, ts, r, j, mode="abs", name=None):
        cs.append(Case(cid(clause, sid, q), name or q, r, j, tol, rule, ts, mode))
    nm = [q[0] for q in quants]; spec = {q[0]: q for q in quants}
    def S(q): return spec[q][1:]
    if "logLik" in spec: add("logLik", *S("logLik"), R["logLik"], J["logLik"])
    if "beta" in spec:
        rn, jn = R.get("beta_names"), J.get("beta_terms") or []
        # alignment by position after checking the names agree where both sides name the coefficients
        if cell == "crabs":
            norm = [n.replace("trait", "", 1).replace(":grp", ":") for n in rn]
            if norm != J["beta_design_names"]: raise SystemExit(f"{sid}: crabs coefficient names differ: {norm[:3]} vs {J['beta_design_names'][:3]}")
        elif cell == "isdm":
            if rn != jn: raise SystemExit(f"{sid}: iSDM b_fix names differ")
        elif cell in ("spider", "beetle", "fungi"):
            p = len(R["trait_levels"]); covs = COVS[cell]
            # Julia's fixed block is [p trait intercepts (sorted species), then the covariate slopes in formula order]
            if rn != ["trait" + t for t in R["trait_levels"]] + covs or len(rn) != len(J["beta"]):
                raise SystemExit(f"{sid}: covariate coefficient layout differs: R {rn[:2]}..{rn[-3:]}, expected covariates {covs}")
        elif cell == "temporal":
            if [n.replace("trait", "", 1) for n in rn] != [n.replace("trait: ", "", 1) for n in jn]: raise SystemExit(f"{sid}: temporal coefficient names differ")
        else:
            if len(rn) != len(J["beta"]): raise SystemExit(f"{sid}: beta length differs")
        add("beta", *S("beta"), R["beta"], J["beta"])
    if "beta_se" in spec:
        add("beta_se", *S("beta_se"), R["beta_se"] if R.get("beta_se") else [], J["beta_se"] if J.get("beta_se") else [], mode="rel")
    if "LLt" in spec: add("LLt", *S("LLt"), R["LLt"], J["LLt"])
    if "dispersion" in spec: add("dispersion", *S("dispersion"), R["dispersion_phi"], J["dispersion_phi"], mode="rel")
    if "thresholds" in spec:
        ct = R["cutpoints"]; tr = R["trait_levels"]; tau = J["tau"]
        rv, jv = [], []
        for t, i, v in zip(ct["trait"], ct["index"], ct["tau"]):
            rv.append(v); jv.append(tau[tr.index(t)][i - 1])
        add("thresholds", *S("thresholds"), rv, jv)
    if "temporal_phi" in spec: add("temporal_phi", *S("temporal_phi"), R["temporal_phi"], J["temporal_phi"])
    if "temporal_loadings" in spec:
        if R["temporal_loadings_traits"] != J["temporal_loadings_traits"]: raise SystemExit(f"{sid}: loading trait order differs")
        add("temporal_loadings", *S("temporal_loadings"), orient(R["temporal_loadings"]), orient(J["temporal_loadings"]))
    if "eta" in spec: add("eta", *S("eta"), R["eta"], J["eta"], name="linear predictor on the training data (link scale)")
    for q in ("predict_link", "predict_response", "predict_newdata_offset0_link", "predict_newdata_offset0_response"):
        if q in spec:
            # align by (cell_id, trait, source)
            rk = list(zip(R["predict_cell_id"], R["predict_trait"], R["predict_source"])); jk = list(zip(J["predict_cell_id"], J["predict_trait"], J["predict_source"]))
            rd = dict(zip(rk, R[q])); jd = dict(zip(jk, J[q]))
            if set(rd) != set(jd) or len(rd) != len(rk): raise SystemExit(f"{sid}: predict keys differ")
            keys = sorted(rd)
            add(q, *S(q), [rd[k] for k in keys], [jd[k] for k in keys])
    return cs


# How each engine's cond(H) is computed (they are different estimators in different parameter bases: recorded, never compared).
R_COND_METHOD = "kappa(solve(sdr$cov.fixed), exact = FALSE): a 1-norm condition-number ESTIMATE (LAPACK) of the inverse of TMB's sdreport covariance of all fixed parameters, in gllvmTMB's own parameter coordinates"
J_COND_METHOD = "cond(Symmetric(vcov(fit, Y))): the exact 2-norm condition number (ratio of the extreme eigenvalues) of the full inverse observed information (Wald covariance) of the Julia fit, in the Julia fit's own parameter coordinates"
J_COND_METHOD_TEMPORAL = ("exact ratio of the extreme eigenvalues of the ForwardDiff Hessian of the temporal negative log-likelihood at fit.parameters, in the optimiser's coordinates "
                          "(rebuilt in run_J.jl from the fit's own objective, not through the Wald vcov used for the other cells)")
R_COND_METHOD_PHYLO = "kappa(sd$cov.fixed, exact = TRUE) in tools/phylo_latent/r_reference_p1.R (PR #547): the exact 2-norm condition number of TMB's sdreport covariance of all fixed parameters, in gllvmTMB's own coordinates"
J_COND_METHOD_PHYLO = "cond(H) of the finite-difference Hessian of the marginal negative log-likelihood at the fit's optimum (the fit's own Hessian diagnostic, exact 2-norm), in the Julia optimiser's coordinates"


def j_cond_method(J): return J_COND_METHOD_TEMPORAL if J.get("cond_H_basis") else J_COND_METHOD


def engine_blocks(cell, R, J):
    r = OrderedDict(engine=R["engine"], gllvmTMB_version=R["gllvmTMB_version"], loaded_from_library=Path(R["gllvmTMB_loaded_from"]).parent.name + "/" + Path(R["gllvmTMB_loaded_from"]).name,
                    TMB_version=R["TMB_version"], R_version=R["R_version"], host=R["host"], formula=R["formula"],
                    convergence=R["convergence"], optimizer_message=R["message"], pdHess=R["pdHess"], max_abs_gradient=R["max_abs_gradient"],
                    n_par=R["n_par"], cond_H=R.get("cond_H"), cond_H_method=R_COND_METHOD, wall_fit_sec=R["wall_fit_sec"], wall_sdreport_sec=R["wall_sdreport_sec"], finished_utc=R["finished_utc"])
    j = OrderedDict(engine=J["engine"], julia_version=J["julia_version"], gllvmodels_commit=J["gllvmodels_commit"], host=J["host"], call=J["call"],
                    converged=J["converged"], iterations=J.get("iterations"), pd_hessian=J.get("pd_hessian", J.get("hessian_positive_definite")),
                    cond_H=J.get("cond_H"), cond_H_method=j_cond_method(J), wall_fit_sec=J["wall_fit_sec"], wall_confint_sec=J.get("wall_confint_sec"), wall_vcov_sec=J.get("wall_vcov_sec"),
                    JULIA_NUM_THREADS=J.get("JULIA_NUM_THREADS"), OPENBLAS_NUM_THREADS=J.get("OPENBLAS_NUM_THREADS"))
    if "cells_converged" in J: j["cells_converged"] = J["cells_converged"]
    if "dispersion_boundary" in J: j["dispersion_boundary"] = J["dispersion_boundary"]
    if "gradient_norm" in J: j["gradient_norm"] = J["gradient_norm"]
    if fin(J.get("max_abs_gradient")): j["max_abs_gradient"] = J["max_abs_gradient"]
    return r, j


CONVERGENCE_GRADIENT_BOUND = 1e-5


def convergence_parity(R, J):
    """The N9 block (maintainer ruling 2026-10-05, D-319), when both raw outputs record their own gradient max-abs at the
    returned point; None otherwise (the receipt then carries no block and is judged as before)."""
    rg, jg = R.get("max_abs_gradient"), J.get("max_abs_gradient")
    if not (fin(rg) and fin(jg)):
        return None
    return OrderedDict(gradient_bound=CONVERGENCE_GRADIENT_BOUND, compared_point="returned",
                       engines=OrderedDict(R=OrderedDict(max_abs_gradient=rg), julia=OrderedDict(max_abs_gradient=jg)),
                       ruling="maintainer ruling 2026-10-05 (vault D-319), N9")


def pass_rule(R, J, cases):
    """The plan's pass rule, each leg recorded."""
    legs = OrderedDict()
    legs["R_convergence_0"] = R.get("convergence") == 0
    legs["R_pdHess_true"] = R.get("pdHess") is True
    legs["Julia_converged_true"] = J.get("converged") is True and J.get("cells_converged", True) is True
    legs["same_data_sha256"] = R.get("data_sha256") == J.get("data_sha256") and bool(R.get("data_sha256"))
    legs["gllvmTMB_deparse_identical_to_P1"] = all(x.get("identical_deparse") for x in R.get("deparse_check", [])) and bool(R.get("deparse_check"))
    legs["R_version_0_7_1_from_P1_library"] = R.get("gllvmTMB_version") == "0.7.1"
    legs["every_quantity_within_tolerance"] = all(c.ok for c in cases)
    return legs


QUAL_LEG = "R_side_receipt_qualified_by_maintainer"
# Legs that say "a required step was not signed", as opposed to "a number or a convergence flag failed".
# A row whose only failing legs are these, with every number inside tolerance, is partial_case_not_executed (PARTIAL),
# never numeric_fail (FAIL): nothing disagrees, a required step is missing. (The C4 bridge leg was removed when the
# maintainer changed the C4 text to accept direct-engine runs, 2026-10-05, D-319.)
OPEN_LEGS = (QUAL_LEG,)
LEG_TEXT = {
    QUAL_LEG: ("a required leg is not signed: the R side is the tracked PR #547 receipt, which records qualified = false "
               "until the maintainer signs the dated promotion block (D-300 answer 9). No agent may sign it, so the receipt stays unqualified until then"),
    "R_convergence_0": "R did not converge (optimizer convergence code is not 0)",
    "R_pdHess_true": "R's Hessian is not positive definite (pdHess is false)",
    "Julia_converged_true": "Julia did not report converged (or a cell did not converge)",
    "same_data_sha256": "the two engines did not record the same data sha256",
    "gllvmTMB_deparse_identical_to_P1": "a named gllvmTMB entry point does not deparse identically to its P1 source",
    "R_version_0_7_1_from_P1_library": "R side is not gllvmTMB 0.7.1 from the P1 library",
}


def numbers_ok(legs):
    """Every leg except the open (not-run / not-signed) ones holds."""
    return all(v for k, v in legs.items() if k not in OPEN_LEGS)


def reasons(legs, cases):
    out = [LEG_TEXT.get(k, k) for k, v in legs.items() if not v and k != "every_quantity_within_tolerance"]
    for c in cases:
        if not c.ok:
            out.append(f"{c.quantity}: " + (c.problem if c.problem else f"{c.mode} difference {c.diff:.6g} > tolerance {c.tol:g}"))
    return out


# ---- phylo (existing P1 R receipt + one fresh Julia fit on the same literal fixture) -------------
PHY = ROOT / "docs/dev-log/core070/phylo-latent-p1"


def build_phylo(raw):
    sid = "covariance/COV-PHYLO-LATENT-RSZ"
    p = Path(raw) / "phylo_J.json"
    if not p.exists(): return None
    rr = load_r(PHY / "cov_phylo_latent_rsz/r-receipt.json"); jr = load_r(p)
    fx = load_r(PHY / "a15-fixture.json")
    cases = [
        Case(cid("C3", sid, "logLik"), "logLik", rr["loglik"], jr["loglik"], 1e-6, "1e-6 absolute on the marginal log-likelihood (A15 used relative 1e-6; absolute is stricter here)", TW),
        Case(cid("C3", sid, "Sigma_phy"), "Sigma_phy", rr["Sigma_phy"], jr["Sigma_phy"], 1e-4, "1e-4 absolute per element of Sigma_phy (A15 receipt: relative 1e-4)", PR),
    ]
    legs = OrderedDict()
    legs["R_convergence_0"] = rr.get("convergence") == 0
    legs["R_pdHess_true"] = (rr.get("hessian") or {}).get("pd_hessian") is True
    legs["Julia_converged_true"] = jr.get("converged") is True
    legs["same_data_sha256"] = rr.get("data_file_sha256") == jr.get("data_file_sha256") and bool(rr.get("data_file_sha256"))
    legs["R_version_0_7_1_from_P1_library"] = rr.get("package_version") == "0.7.1" and rr.get("source_pin") == P1
    # The R side is the PR #547 receipt. This leg reads its qualified flag, which is true only under the maintainer's dated
    # promotion block (D-300 answer 9; signed 2026-10-05, D-319; phylo-latent-p1/README.md). No agent signs it.
    legs[QUAL_LEG] = rr.get("qualified") is True
    legs["every_quantity_within_tolerance"] = all(c.ok for c in cases)
    return dict(sid=sid, cell="phylo", clause="C3", cases=cases, legs=legs, R=rr, J=jr, raw_files=[("r-receipt (tracked, PR #547)", PHY / "cov_phylo_latent_rsz/r-receipt.json"), ("julia fit on current main", p)])


# ---- C5: the merged PR #593 grouping receipts -----------------------------------------------------
def build_c5(level):
    p = ROOT / LEDGER / "receipts/grouping" / f"{level}.json"
    d = json.loads(p.read_text())
    npar = d["name_parity"]
    legs = OrderedDict()
    legs["name_parity_PASS"] = npar.get("verdict") == "PASS"
    legs["negative_control_rejected_on_both_sides"] = bool(npar["r"]["negative_control"]["rejected"]) and bool(npar["julia"]["negative_control"]["rejected"])
    legs["receipt_verdict_PASS"] = d.get("verdict") == "PASS"
    legs["R_convergence_0"] = d["fits"]["r"].get("convergence") == 0
    legs["Julia_converged_true"] = d["fits"]["julia"].get("converged") is True
    ll = [c for c in d["comparison"]["cases"] if c["case_id"].endswith("/logLik")]
    legs["paired_logLik_within_1e-6"] = len(ll) == 1 and ll[0]["tolerance"] <= 1e-6 and abs(ll[0]["r_value"] - ll[0]["julia_value"]) <= 1e-6
    return d, legs, ll


# ---- building a receipt -------------------------------------------------------------------------
def clean(x):
    """JSON has no NaN/Inf: a non-finite recorded value (e.g. cond(H) when the Hessian is not positive definite) becomes null."""
    if isinstance(x, float) and not math.isfinite(x): return None
    if isinstance(x, dict): return type(x)((k, clean(v)) for k, v in x.items())
    if isinstance(x, list): return [clean(v) for v in x]
    return x


def write_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(clean(obj), indent=1, allow_nan=False) + "\n")


# The rule actually applied, per kind of row. Each text names only legs that the receipt's `legs` record.
RULE_SCOPE_P1 = ("the ten named gllvmTMB entry points that run_R.R calls (gllvmTMB, gllvmTMBcontrol, nbinom2, ordinal_logit, isdm_sources, extract_Sigma, "
                 "extract_cutpoints, predict.gllvmTMB_multi, extract_temporal, temporal_latent) deparse identically to their P1 source; internal gllvmTMB "
                 "functions and the compiled library are not deparse-checked, they are trusted by the recorded version 0.7.1, the library path and the sha256 of the P1 source files")
RULE_C3 = ("both engines converged (R convergence 0 with a positive-definite Hessian; Julia converged true), both read the same data bytes, " + RULE_SCOPE_P1 +
           ", and every listed quantity is within its tolerance")
RULE_C4 = RULE_C3 + (" (C4 accepts this direct-engine run: maintainer ruling 2026-10-05, vault D-319, changed the clause text; "
                     "no engine = \"julia\" bridge leg is required)")
RULE_PHYLO = ("R side taken from the tracked PR #547 receipt (R was not re-run, so there is no deparse check): R convergence 0 with a positive-definite Hessian, "
              "Julia converged true on a fresh fit, the same data bytes, R recorded as gllvmTMB 0.7.1 at the P1 pin, every listed quantity within its tolerance, "
              "and the #547 R receipt promoted (qualified true) by the maintainer's own dated signature (D-300 answer 9)")
RULE_C5 = ("the rule applied to the four grouping rows, taken from the merged PR #593 receipts, which record no R Hessian leg, so none is claimed: "
           "name parity PASS with the misspelt-keyword negative control rejected by both engines; the #593 receipt's own verdict PASS; "
           "R convergence 0 and Julia converged true in the paired fit; the paired logLik within 1e-6; and a replay on current main that verifies "
           "the fixture hashes, passes name parity again and reproduces the receipt's logLik within 1e-6")


def rule_text(cell, clause):
    return RULE_C5 if clause == "C5" else RULE_PHYLO if cell == "phylo" else RULE_C4 if clause == "C4" else RULE_C3


def build_receipt(sid, cell, clause, cases, legs, eng_r, eng_j, extra, raw_refs, binds, cp=None):
    rcpt = OrderedDict()
    rcpt["schema"] = SCHEMA
    rcpt["source_id"] = sid; rcpt["clause"] = clause
    rcpt["pin"] = "P1"; rcpt["reference_commit"] = P1; rcpt["ruling"] = RULING
    # a row whose numbers all pass but whose only failing legs are required steps not run or not signed (the C4 bridge leg, the
    # phylo qualification) is recorded as measured, not as FAIL and not as PASS
    rcpt["verdict"] = "PASS" if binds else ("NUMERIC_PASS_NOT_BINDING" if numbers_ok(legs) else "FAIL")
    why = extra.pop("reasons", [])
    rcpt["row_status"] = "binds: the plan's pass rule holds on both engines" if binds else "does not bind: " + "; ".join(why)
    rcpt["pass_rule"] = OrderedDict(rule=rule_text(cell, clause), legs=legs)
    rcpt.update(extra)
    rcpt["engines"] = OrderedDict(R=eng_r, julia=eng_j)
    if cp is not None:
        rcpt["convergence_parity"] = cp
    rcpt["comparison"] = OrderedDict(pin="P1", cases=[c.block(raw_refs["note"]) for c in cases])
    rcpt["read_from"] = raw_refs["hashes"]
    return rcpt


NOT_COVERED_COMMON = [
    "One dataset and one seed per row: a spot check at this size, not a scaling study or a coverage study.",
    "Direct engines only (R TMB fit against a direct Julia fit); R's engine = 'julia' bridge route is not exercised.",
    "Julia starts from its own default start, never from R's coordinates; wall times include first-call Julia compilation.",
    "cond(H) of the two engines is recorded where computed but never compared (different parameter bases); see cond_H_statement for this receipt.",
]


def cond_statement(R, J):
    """Say plainly, per receipt, how each engine's cond(H) is computed and where it is and is not recorded."""
    rc = R.get("cond_H"); jc = J.get("cond_H")
    out = [f"R cond(H) {rc:.6g}, computed as {R_COND_METHOD}." if isinstance(rc, (int, float)) and math.isfinite(rc) else "R cond(H) not recorded."]
    if isinstance(jc, (int, float)) and math.isfinite(jc): out.append(f"Julia cond(H) {jc:.6g}, computed as {j_cond_method(J)}.")
    else:
        why = J.get("cond_H_error") or J.get("ci_skipped") or J.get("confint_error")
        if isinstance(jc, float): why = why or "the Hessian is not positive definite or not finite, so cond(H) is NaN"
        out.append("Julia cond(H) NOT recorded for this row: " + (why or "the runner did not compute it") + ".")
    out.append("The two are different estimators in different parameter bases: recorded, never compared like for like.")
    return " ".join(out)


def read_text(p):
    p = Path(p)
    return p.read_text() if p.exists() else None


def trim_log(text, limit=1500):
    """Warnings and notes from an engine log, minus progress noise, for the receipt."""
    if not text: return None
    keep = [l for l in text.splitlines() if l.strip() and not l.startswith(("DONE", "Precompil", "WROTE")) and "experimental" not in l.lower()]
    s = "\n".join(keep)
    return (s[:limit] + " ...[trimmed]") if len(s) > limit else (s or None)


def run_context(raw, cell):
    """Where and when the jobs ran, from the driver's own logs (copied into <raw>/run) or the Mac run's start file."""
    base = Path(raw); ctx = OrderedDict()
    if cell == "urban":
        st = read_text(base / "run_mac_urban" / "start.txt")
        ctx["host"] = "Mac Studio (the maintainer's machine; the urbanisation matrix is local and not copied)"
        if st: ctx["start"] = st.strip().splitlines()
        return ctx
    jl, st, en = read_text(base / "run/jobs.log"), read_text(base / "run/start.txt"), read_text(base / "run/end.txt")
    if jl is None: return None
    ctx["host"] = "totoro.biology.ualberta.ca (384 cores, shared); lane directory ~/hsq_work/true-parity-campaign-20261002"
    if st: ctx["driver_start"] = st.strip().splitlines()
    if en: ctx["driver_end"] = en.strip().splitlines()
    ctx["jobs"] = [l for l in jl.splitlines() if l.split()[1:2] == [cell]]
    rr = read_text(base / "run" / f"rerun_{cell}.txt")
    if rr: ctx["rerun"] = rr.strip().splitlines()
    ctx["caps"] = "at most 8 concurrent jobs, OPENBLAS_NUM_THREADS=1, JULIA_NUM_THREADS=2, R single-threaded; each Julia job under the timeout shown as cap= in its job line (twice its written estimate, except fungi, whose cap was 300 s against a 2 min estimate); R jobs under 600 s"
    return ctx


def data_meta(raw, cell):
    for n in (f"{cell}.meta.json", f"{cell}_wide.meta.json"):
        p = Path(raw) / "data_meta" / n
        if p.exists(): return json.loads(p.read_text())
    return None


def process(raw, out_root, apply, quiet=False):
    results = []   # (row spec, receipt path, binds, receipt)
    camp = out_root / LEDGER / "receipts"
    # --- C3 / C4 executable cells
    for sid, fam, cell, clause, quants in ROWS:
        if cell == "phylo":
            ph = build_phylo(raw)
            if ph is None:
                if not quiet: print(f"skip {sid}: no phylo_J.json")
                continue
            binds = all(ph["legs"].values())
            rdir = camp / fam / CAMPAIGN_DIR
            rr, jr = ph["R"], ph["J"]
            eng_r = OrderedDict(engine="R gllvmTMB (tracked P1 receipt of PR #547, recorded 2026-09-29)", gllvmTMB_version=rr["package_version"], source_pin=rr["source_pin"],
                                formula=rr["formula"], convergence=rr["convergence"], optimizer_message=rr["message"], gradient_max_abs=rr["gradient_max_abs"], pd_hessian=rr["hessian"]["pd_hessian"],
                                cond_H=rr["hessian"]["condition_number"], cond_H_method=R_COND_METHOD_PHYLO, wall_fit_sec=rr["elapsed_seconds"], dll_sha256=rr["dll_sha256"],
                                qualified=rr.get("qualified"))
            eng_j = OrderedDict(engine="GLLVModels.fit_phylo_latent_gllvm, fresh fit on current main", julia_version=jr["julia_version"], gllvmodels_commit=jr["git_head"],
                                converged=jr["converged"], stopping_reason=jr["stopping_reason"], gradient_norm=jr["gradient_norm"], iterations=jr["iterations"],
                                hessian_positive_definite=jr["hessian_positive_definite"], cond_H=jr["hessian_condition_number"], cond_H_method=J_COND_METHOD_PHYLO, wall_fit_sec=jr["fit_elapsed_seconds"])
            hashes = OrderedDict()
            hashes["docs/dev-log/core070/phylo-latent-p1/a15-fixture.json"] = sha_file(PHY / "a15-fixture.json")
            hashes["docs/dev-log/core070/phylo-latent-p1/cov_phylo_latent_rsz/r-receipt.json"] = sha_file(PHY / "cov_phylo_latent_rsz/r-receipt.json")
            jgz = rdir / "raw" / "phylo_J.json.gz"
            if apply: gz_write(jgz, Path(raw) / "phylo_J.json")
            hashes[str(jgz.relative_to(out_root)) + " (uncompressed sha256)"] = sha_file(Path(raw) / "phylo_J.json")
            extra = OrderedDict(
                cell=OrderedDict(shape="100 species (ape::rcoal, seed 20260928) x 5 replicates x 20 traits, rank 2", fixture="docs/dev-log/core070/phylo-latent-p1/a15-fixture.json",
                                 data_file_sha256=rr["data_file_sha256"]),
                what_this_is="R values are the tracked P1 R receipt of PR #547 (fitted once at P1 and kept). The Julia fit is NEW: fit_phylo_latent_gllvm on the same literal fixture at the commit above. "
                             "The #547 Julia receipt recorded converged = false (gradient stall); current main converges, and this receipt uses the current fit.",
                r_side_qualification=(("QUALIFIED. The R receipt (docs/dev-log/core070/phylo-latent-p1/cov_phylo_latent_rsz/r-receipt.json) records qualified = true under the maintainer's "
                                       f"dated promotion block (D-300 answer 9), signed by {rr['maintainer_promotion']['signed_by']} on {rr['maintainer_promotion']['signed_on']} (vault D-319) and kept in "
                                       "docs/dev-log/core070/phylo-latent-p1/README.md. The promotion is despite the A15 stationarity gap that README records; an agent recorded the signature, no agent signed it.")
                                      if rr.get("qualified") is True else
                                      ("NOT BINDING, on purpose. The R receipt (docs/dev-log/core070/phylo-latent-p1/cov_phylo_latent_rsz/r-receipt.json) records qualified = false, and "
                                       "docs/dev-log/core070/phylo-latent-p1/README.md says every receipt stays unqualified until the maintainer signs the dated promotion block (D-300 answer 9). "
                                       "The measurement is kept as a non-binding receipt: both numbers are inside tolerance, but the row binds only after the maintainer signs that block. No agent may sign it.")),
                julia_script="tools/phylo_latent/compare_phylo_latent_p1.jl, command: julia --project=. tools/phylo_latent/compare_phylo_latent_p1.jl fit cov_phylo_latent_rsz docs/dev-log/core070/phylo-latent-p1/a15-fixture.json <out>/phylo_J.json (output committed as raw/phylo_J.json.gz)",
                cond_H_statement=(f"R cond(H) {rr['hessian']['condition_number']:.6g}, computed as {R_COND_METHOD_PHYLO}. "
                                  f"Julia cond(H) {jr['hessian_condition_number']:.6g}, computed as {J_COND_METHOD_PHYLO}. "
                                  "The two are different estimators in different parameter bases: recorded, never compared like for like."),
                reasons=reasons(ph["legs"], ph["cases"]),
                not_covered=NOT_COVERED_COMMON[:1] + [("R was not re-run: its values are the PR #547 receipt's, promoted by the maintainer's D-300 answer 9 block (2026-10-05)." if rr.get("qualified") is True
                                                       else "R was not re-run: its values are the PR #547 receipt's, which is unqualified until the maintainer signs the D-300 answer 9 promotion block."),
                                                      "cond(H) (R 83030, Julia 82761 here) is recorded, not compared."])
            rc = build_receipt(sid, cell, clause, ph["cases"], ph["legs"], eng_r, eng_j, extra, dict(note=str(jgz.relative_to(out_root)), hashes=hashes), binds)
            results.append((sid, fam, clause, rc, binds, rdir / f"{slug(sid)}.json"))
            continue
        rp, jp = Path(raw) / f"{cell}_R.json", Path(raw) / f"{cell}_J.toml"
        if not (rp.exists() and jp.exists()):
            if not quiet: print(f"skip {sid}: raw outputs missing ({rp.name}, {jp.name})")
            continue
        R, J = load_r(rp), load_j(jp)
        if not J.get("DONE"):
            if not quiet: print(f"skip {sid}: Julia run not finished")
            continue
        cases = cases_for(sid, cell, clause, quants, R, J)
        legs = pass_rule(R, J, cases)
        cp = convergence_parity(R, J)
        # N9: a receipt with the block binds only when both engines are at gradient max-abs <= 1e-5
        if cp is not None:
            legs["convergence_parity_both_gradients_within_1e-5"] = all(
                e["max_abs_gradient"] <= CONVERGENCE_GRADIENT_BOUND for e in cp["engines"].values())
        binds = all(legs.values())
        eng_r, eng_j = engine_blocks(cell, R, J)
        rdir = camp / fam / CAMPAIGN_DIR
        hashes = OrderedDict()
        off = cell in OFF_REPO
        for src, nm in ((rp, f"{cell}_R.json"), (jp, f"{cell}_J.toml")):
            dst = rdir / "raw" / (nm + ".gz")
            if off:   # recorded, never committed: the sha256 lets the maintainer's retained copy be verified
                hashes[str(dst.relative_to(out_root)) + " (uncompressed sha256; this raw file is NOT in the repo)"] = sha_file(src)
                continue
            if apply: gz_write(dst, src)
            hashes[str(dst.relative_to(out_root)) + " (uncompressed sha256)"] = sha_file(src)
        extra = OrderedDict(
            cell=OrderedDict((k, R.get(k)) for k in ("formula",)) | OrderedDict(data_sha256=R["data_sha256"], data_file=J.get("data_file"),
                              p=J.get("p"), n=J.get("n"),
                              data_committed_copy=(f"tools/true_parity/campaign/data/{cell}.csv.gz" if cell in SYNTH else None)),
            data_meta=data_meta(raw, cell), run=run_context(raw, cell),
            engine_messages=OrderedDict(R=trim_log(read_text(Path(raw) / "logs" / f"R_{cell}.log")), julia=trim_log(read_text(Path(raw) / "logs" / f"J_{cell}.log"))),
            p1_source_sha256=R["p1_source_sha256"], gllvmTMB_deparse_check=R["deparse_check"],
            cond_H_statement=cond_statement(R, J),
            reasons=reasons(legs, cases), not_covered=NOT_COVERED_COMMON)
        if off:
            extra["raw_outputs_off_repo"] = (NOTE_OFF_REPO + ". The urbanisation matrix is the maintainer's unpublished data and its redistribution status is unconfirmed, so the "
                "per-observation raw outputs (linear predictors, loadings) are not committed. This receipt keeps summary values only: logLik, the fixed effects, the maximum "
                "differences, wall times, the data file's sha256 and the sha256 of the two raw files (read_from). write_receipts.py --check cannot re-derive this row from the "
                "repository; it prints that and checks the receipt's internal consistency. With the retained raw files, run write_receipts.py --check --local-raw DIR.")
        note = NOTE_OFF_REPO if off else f"receipts/{fam}/{CAMPAIGN_DIR}/raw/{cell}_*.gz"
        rc = build_receipt(sid, cell, clause, cases, legs, eng_r, eng_j, extra, dict(note=note, hashes=hashes), binds, cp)
        results.append((sid, fam, clause, rc, binds, rdir / f"{slug(sid)}.json"))
    # --- C5
    for sid, level in C5:
        d, legs, ll = build_c5(level)
        binds = all(legs.values())
        cases = []
        src = ROOT / LEDGER / "receipts/grouping" / f"{level}.json"
        c0 = ll[0] if ll else None
        cs = [Case(f"grouping-p1/{level}/logLik", "logLik", c0["r_value"], c0["julia_value"], 1e-6, "1e-6 absolute on the marginal log-likelihood", "twin: #576 reports 1e-8, #563 reports 9e-8") ] if c0 else []
        rdir = camp / "fit-input" / CAMPAIGN_DIR
        replay = Path(raw) / f"c5_replay_{level}.json"
        rep = json.loads(replay.read_text()) if replay.exists() else None
        if rep is not None:
            legs["replay_hashes_verified"] = rep.get("fixture_sha256_verified") is True
            legs["replay_name_parity_PASS"] = rep.get("name_parity_pass") is True
            legs["replay_logLik_within_1e-6_of_receipt"] = rep.get("max_abs_logLik_vs_receipt", 1) <= 1e-6
            binds = all(legs.values())
        else:
            legs["replay_hashes_verified"] = False; binds = False
        hashes = OrderedDict()
        hashes[str(src.relative_to(ROOT))] = sha_file(src)
        hashes[d["fixture"]["path"]] = sha_file(ROOT / d["fixture"]["path"])
        if rep is not None:
            dst = rdir / "raw" / f"c5_replay_{level}.json.gz"
            if apply: gz_write(dst, replay)
            hashes[str(dst.relative_to(out_root)) + " (uncompressed sha256)"] = sha_file(replay)
        extra = OrderedDict(
            level=level, source_receipt=str(src.relative_to(ROOT)), source_receipt_sha256=sha_file(src), name_parity=d["name_parity"]["verdict"],
            name_parity_method=d["name_parity"]["method"], paired_fit=d["fits"],
            replay=rep, reasons=reasons(legs, cs) if not binds else [],
            not_covered=d["not_covered"] + ["C5 pairing is Gaussian only: the non-Gaussian numerical pairing of grouping levels is fenced here, not measured."])
        # the comparison block carries the single logLik case the row requires; the full receipt keeps all five
        eng_r = OrderedDict(engine="R gllvmTMB 0.7.1 (P1)", **{k: d["fits"]["r"][k] for k in ("call", "convergence", "optimizer_message", "max_abs_gradient", "fit_seconds")})
        eng_j = OrderedDict(engine="GLLVModels.fit_gllvm", **{k: d["fits"]["julia"][k] for k in ("call", "converged", "iterations", "fit_seconds")}, glvmodels_commit_of_source_receipt=d["glvmodels_commit"])
        rc = build_receipt(sid, level, "C5", cs, legs, eng_r, eng_j, extra, dict(note=str(src.relative_to(ROOT)), hashes=hashes), binds)
        results.append((sid, "fit-input", "C5", rc, binds, rdir / f"{slug(sid)}.json"))
    return results


# ---- case-map rows ------------------------------------------------------------------------------
def row_for(sid, clause, rc, binds, rel_receipt):
    cases = [c["case_id"] for c in rc["comparison"]["cases"]]
    # every number inside tolerance, only a required step not run or not signed: PARTIAL, not FAIL (nothing disagrees)
    open_only = (not binds) and rc["verdict"] == "NUMERIC_PASS_NOT_BINDING"
    tier = "numeric" if binds else ("partial_case_not_executed" if open_only else "numeric_fail")
    row = OrderedDict()
    row["source_id"] = sid
    row["classification"] = "required_core"
    row["arc"] = f"campaign-{clause}"
    row["clause"] = clause
    row["executable_case_ids"] = cases
    row["disposition"] = None
    row["evidence_tier"] = tier
    row["measured_against"] = "P1"
    ev = OrderedDict()
    if binds:
        ev["receipt"] = [rel_receipt]
        ev["tier"] = "numeric: the receipt carries an R-vs-Julia comparison block pinned to P1; both engines converged and every listed quantity is within its tolerance (pass rule and tolerances as PROPOSED in the C3 to C5 campaign plan, PR #650, rows added under itchyshin/GLLVModels.jl#684 item 4; the maintainer has not yet confirmed the tolerances or the pass rule)"
    else:
        ev["non_binding_receipts"] = [rel_receipt]
        why = rc["row_status"].replace("does not bind: ", "")
        ev["tier"] = ("measured, does not bind. Every number is inside its tolerance, but " + why) if open_only else "measured, does not bind: " + why
    row["evidence"] = ev
    row["measured_result"] = OrderedDict(verdict=rc["verdict"],
        max_abs_diff_by_case={c["case_id"]: c["max_abs_diff"] for c in rc["comparison"]["cases"]},
        within_tolerance_by_case={c["case_id"]: c["within_tolerance"] for c in rc["comparison"]["cases"]})
    row["ruling"] = RULING
    return row


DISPOSITION_SIGNATURE = OrderedDict(signed_by="Shinichi Nakagawa", signed_on="2026-10-05", signature_ref="D-319")


def disposition_row(sid, d):
    row = OrderedDict()
    row["source_id"] = sid
    row["classification"] = "required_core"
    row["arc"] = "campaign-C4"; row["clause"] = "C4"
    row["executable_case_ids"] = []
    row["disposition"] = "DISPOSITION-SIGNED"
    row["evidence_tier"] = "not_measured"
    row["measured_against"] = "P1"
    row["evidence"] = OrderedDict(tier="no measurement: the row closes by the maintainer's signed disposition below")
    row["signed_by"] = DISPOSITION_SIGNATURE["signed_by"]
    row["signed_on"] = DISPOSITION_SIGNATURE["signed_on"]
    row["signature_ref"] = DISPOSITION_SIGNATURE["signature_ref"]
    row["signed_disposition"] = OrderedDict(capability=d["capability"], text=d["text"],
        status=("DISPOSITION-SIGNED: item F of the rulings page (three C4 dispositions), signed by the maintainer on 2026-10-05 "
                "(vault D-319; LOOP/lanes/true-parity-latest/signed-rulings-2026-10-05.md in the lane kit). Recorded by an agent, "
                "signed by the maintainer."))
    row["ruling"] = RULING + " (row added); disposition signed under vault D-319, item F"
    return row


def update_map(out_root, fam, rows_new):
    p = out_root / LEDGER / f"case-map-{fam}.json"
    t = p.read_text(); d = json.loads(t)
    assert json.dumps(d, indent=2) + "\n" == t, f"{p.name} does not round-trip: refusing to rewrite it"
    by = {r["source_id"]: i for i, r in enumerate(d["rows"])}
    for r in rows_new:
        if r["source_id"] in by: d["rows"][by[r["source_id"]]] = r
        else: d["rows"].append(r)
    p.write_text(json.dumps(d, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--raw"); ap.add_argument("--apply", action="store_true")
    ap.add_argument("--check", action="store_true"); ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--local-raw", help="with --check: a directory holding the maintainer's retained urban_R.json and urban_J.toml (not in the repo)")
    a = ap.parse_args()
    out_root = a.root
    if a.check:
        return check(out_root, a.local_raw)
    if not a.raw: ap.error("--raw is required")
    res = process(a.raw, out_root, a.apply)
    rows_by_fam = {}
    for sid, fam, clause, rc, binds, path in res:
        rel = str(path.relative_to(out_root))
        print(f"{'BINDS ' if binds else 'UNBOUND'} {sid:42s} " + ("" if binds else rc["row_status"][:230]))
        if a.apply:
            write_json(path, rc)
            rows_by_fam.setdefault(fam, []).append(row_for(sid, clause, rc, binds, rel))
    if a.apply:
        for sid, d in DISPOSITIONS.items(): rows_by_fam.setdefault("data", []).append(disposition_row(sid, d))
        for fam, rows in rows_by_fam.items(): update_map(out_root, fam, rows)
        print("wrote receipts and case-map rows")
    return 0


def norm(x):
    """What a value looks like after a trip through the receipt/case-map JSON files."""
    return json.loads(json.dumps(clean(x), allow_nan=False))


def summary_problems(rc, sid, clause):
    """Internal consistency of a committed receipt whose raw files are not in the repo: the recorded differences, flags, legs,
    verdict and rule text must agree with each other. This is NOT a re-derivation from raw outputs."""
    out = []
    flags = []
    for c in rc["comparison"]["cases"]:
        d = c.get("max_abs_diff")
        if "r_value" in c and "julia_value" in c and "convention" not in c:
            if d != maxabs(flat(c["r_value"]), flat(c["julia_value"])): out.append(f"{c['case_id']}: max_abs_diff is not the difference of the recorded r_value and julia_value")
        ok = d is not None and math.isfinite(d) and d <= c["tolerance"]
        if bool(c.get("within_tolerance")) != ok: out.append(f"{c['case_id']}: within_tolerance disagrees with max_abs_diff and tolerance")
        flags.append(ok)
    legs = rc["pass_rule"]["legs"]
    if legs.get("every_quantity_within_tolerance") != all(flags): out.append("leg every_quantity_within_tolerance disagrees with the case flags")
    binds = all(legs.values())
    want = "PASS" if binds else ("NUMERIC_PASS_NOT_BINDING" if numbers_ok(legs) else "FAIL")
    if rc["verdict"] != want: out.append(f"verdict {rc['verdict']} but the legs give {want}")
    if rc["pass_rule"]["rule"] != rule_text(None, clause): out.append("rule text differs from the rule applied")
    return out, binds


def check(out_root, local_raw=None):
    """Rebuild every campaign receipt and case-map row from the committed raw files and fail on any difference.

    1. every raw file's sha256 still equals the one the receipt recorded;
    2. the raw files are decompressed into a scratch directory and process() is run on them exactly as for a real run;
    3. for each result, the committed receipt's verdict, row status, pass-rule legs, comparison block (values,
       tolerances, differences, flags), engine blocks and source ids must equal the rebuilt ones, and the committed
       case-map row must equal the rebuilt row. A hand edit to a value, a tolerance or a flag therefore fails.
    The receipt text fields that come from the engines' logs and run directories (not committed as raw) are not rebuilt.

    A row whose raw outputs are kept off the public repo (OFF_REPO: the urbanisation matrix is unpublished) is NOT re-derived:
    --check says so, checks the committed summary receipt for internal consistency and its case-map row against that receipt,
    and rebuilds it in full only when --local-raw DIR supplies the retained raw files (checked against the recorded sha256)."""
    bad = 0
    recdir = out_root / LEDGER / "receipts"
    local = {}
    for cell in OFF_REPO:
        rf, jf = (Path(local_raw) / f"{cell}_R.json", Path(local_raw) / f"{cell}_J.toml") if local_raw else (None, None)
        if rf and rf.exists() and jf.exists(): local[cell] = (rf, jf)
    for p in sorted(recdir.glob("*/campaign/*.json")):
        rc = json.loads(p.read_text())
        for k, h in rc.get("read_from", {}).items():
            path = k.split(" (uncompressed sha256")[0]
            f = out_root / path
            if "NOT in the repo" in k:   # an off-repo raw file: verify the maintainer's retained copy only when it was supplied
                cell = next((c for c in OFF_REPO if Path(path).name.startswith(c + "_")), None)
                if cell in local:
                    got = sha_file(local[cell][0] if path.endswith(".json.gz") else local[cell][1])
                    if got != h: print(f"HASH DRIFT {path} (--local-raw copy) in {p.name}"); bad += 1
                continue
            if not f.exists(): print(f"MISSING {path} for {p.name}"); bad += 1; continue
            got = sha_bytes(gz_read(f)) if path.endswith(".gz") else sha_file(f)
            if got != h: print(f"HASH DRIFT {path} in {p.name}"); bad += 1
    summary_only = []
    with tempfile.TemporaryDirectory() as tmp:
        for gz in sorted(recdir.glob("*/campaign/raw/*.gz")):
            (Path(tmp) / gz.name[:-3]).write_bytes(gz_read(gz))
        for cell, (rf, jf) in local.items():
            (Path(tmp) / rf.name).write_bytes(rf.read_bytes()); (Path(tmp) / jf.name).write_bytes(jf.read_bytes())
        man = json.loads((out_root / "tools/true_parity/campaign/data/data_sha256.json").read_text())["sha256"]
        for cell, h in man.items():   # the committed synthetic data are the bytes both engines read
            f = out_root / f"tools/true_parity/campaign/data/{cell}.csv.gz"
            if not f.exists() or sha_bytes(gz_read(f)) != h: print(f"SYNTHETIC DATA DRIFT {cell}"); bad += 1
            rf = Path(tmp) / f"{cell}_R.json"
            if rf.exists() and json.loads(rf.read_text()).get("data_sha256") != h: print(f"DATA PIN DRIFT {cell}: the receipt's data sha256 is not the committed file's"); bad += 1
        res = process(tmp, out_root, False, quiet=True)
        want = {str(path.relative_to(out_root)) for _, _, _, _, _, path in res}
        have = {str(p.relative_to(out_root)) for p in recdir.glob("*/campaign/*.json")}
        off_paths = {}
        for cell, sid in OFF_REPO.items():
            if cell in local: continue
            fam = next(r[1] for r in ROWS if r[2] == cell)
            off_paths[sid] = (fam, recdir / fam / CAMPAIGN_DIR / f"{slug(sid)}.json")
        have -= {str(pp.relative_to(out_root)) for _, pp in off_paths.values()}
        if want != have: print(f"RECEIPT SET DRIFT rebuilt-only {sorted(want - have)} committed-only {sorted(have - want)}"); bad += 1
        maps = {}
        def row_of(fam, sid):
            if fam not in maps: maps[fam] = json.loads((out_root / LEDGER / f"case-map-{fam}.json").read_text())
            return next((r for r in maps[fam]["rows"] if r["source_id"] == sid), None)
        for sid, fam, clause, rc_new, binds, path in res:
            rel = str(path.relative_to(out_root))
            if not path.exists(): continue
            rc_old = json.loads(path.read_text())
            for k in ("schema", "source_id", "clause", "pin", "reference_commit", "ruling", "verdict", "row_status", "pass_rule", "comparison", "engines", "convergence_parity"):
                if rc_old.get(k) != norm(rc_new.get(k)):
                    print(f"DERIVATION DRIFT {k} in {path.name}"); bad += 1
            row_old = row_of(fam, sid); row_new = norm(row_for(sid, clause, rc_new, binds, rel))
            if row_old != row_new: print(f"CASE-MAP ROW DRIFT {sid}"); bad += 1
        for sid, (fam, path) in off_paths.items():
            print(f"NOTE {sid}: {NOTE_OFF_REPO}. Not re-derived here; checking the committed summary receipt for internal consistency only.")
            summary_only.append(sid)
            if not path.exists(): print(f"MISSING {path.relative_to(out_root)}"); bad += 1; continue
            rc_old = json.loads(path.read_text())
            probs, binds = summary_problems(rc_old, sid, rc_old.get("clause"))
            for q in probs: print(f"SUMMARY INCONSISTENT {sid}: {q}"); bad += 1
            if row_of(fam, sid) != norm(row_for(sid, rc_old.get("clause"), rc_old, binds, str(path.relative_to(out_root)))):
                print(f"CASE-MAP ROW DRIFT {sid}"); bad += 1
        for sid, d in DISPOSITIONS.items():
            if row_of("data", sid) != norm(disposition_row(sid, d)): print(f"CASE-MAP ROW DRIFT {sid}"); bad += 1
    tail = f" ({len(summary_only)} row checked for internal consistency only, not re-derived: {', '.join(summary_only)})" if summary_only else ""
    print(("CAMPAIGN_RECEIPTS_OK" + tail) if not bad else f"CAMPAIGN_RECEIPTS_BAD {bad}" + tail)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
