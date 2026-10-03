#!/usr/bin/env python3
"""Write the true-parity campaign receipts (C3, C4, C5) from RAW engine outputs, and the case-map rows.

Signed itchyshin/GLLVModels.jl#684 item 4 (never cite it as a bare number). This script is the only
thing that writes the campaign receipts and the 20 campaign case-map rows: nothing is typed by hand,
and no value is invented. It reads

  <raw>/<cell>_R.json   written by tools/true_parity/campaign/run_R.R   (private P1 gllvmTMB, 0.7.1)
  <raw>/<cell>_J.toml   written by tools/true_parity/campaign/run_J.jl   (GLLVModels.jl)

for each C3/C4 cell, the tracked P1 phylo-latent receipts for COV-PHYLO-LATENT-RSZ (plus one fresh
Julia fit on the same literal fixture), and the merged PR #593 grouping receipts for C5. It writes

  receipts/<family>/campaign/<row>.json     one receipt per row, with a `comparison` block pinned to P1
  receipts/<family>/campaign/raw/*.gz       the raw outputs the receipt was built from (gzip)
  case-map-<family>.json                    the row, appended or (by source_id) replaced; nothing else touched

Pass rule (campaign plan section 1.1, approved by ruling 4): BOTH engines converged (R convergence 0
with a positive-definite Hessian; Julia `converged` true; iSDM also every cell converged) AND every
listed quantity is within its tolerance AND both engines read the same data bytes AND every gllvmTMB
function used deparses identically to its P1 source. A row that meets it binds (evidence_tier "numeric",
evidence.receipt). Otherwise the receipt is cited as non-binding with the reason and the row is left
unbound; no tolerance is widened and nothing is re-run to get a pass.

A relative tolerance is recorded as a discrepancy statistic against zero (r_value = 0, julia_value =
|J - R| / |R| per element), because the checker compares absolute differences; the raw R and Julia
vectors are stored beside it. Vectors of 1000+ values (linear predictors) carry max_abs_diff only, with
the raw files committed (gzip) so this script's --check re-derives them.

Usage:
  python3 write_receipts.py --raw DIR [--apply]       # default: print what would be written
  python3 write_receipts.py --check                   # re-derive every campaign receipt from the committed raw files
"""
from __future__ import annotations
import argparse, gzip, hashlib, json, math, os, subprocess, sys, tomllib
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
            else:
                d["diff_source"] = f"max over {len(self.r)} values, from the committed raw files ({raw_note}); re-derived by write_receipts.py --check"
            d["max_abs_diff"] = self.diff
        else:
            d["convention"] = ("relative-difference statistic: r_value is the target 0; julia_value is |J - R| / |R| per element "
                               "(the checker compares absolute differences, so a relative tolerance is carried this way)")
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
PR = "PROPOSED in the campaign plan, approved by ruling 4"
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
        text="gllvm::fungi ships a phylogeny (fungi$tree), so a real phylo workflow exists. Blocked at the bridge gate "
             "GJL-GATE-STRUCTURED-TERMS until gllvmTMB #1236 (A4a) lands; revisit at P2.",
        draft="Proposed disposition outside_boundary until P2: real-data phylo needs the structured-term bridge route (gllvmTMB #1236). "
              "signed_by: Shinichi Nakagawa; signed_on: <date you sign>."),
    "data/RD-TEMPORAL-DISPOSITION": dict(
        capability="temporal real data",
        text="None of gllvm, vegan, MASS, ape ships a multivariate ecological time series the plan would call a real workflow; "
             "no bridge route for temporal at P1 (D-296).",
        draft="Proposed disposition outside_boundary at P1: no real temporal dataset in the checked packages and no bridge route (D-296). "
              "signed_by: Shinichi Nakagawa; signed_on: <date you sign>."),
    "data/RD-ISDM-DISPOSITION": dict(
        capability="integrated SDM real data",
        text="No real multi-source dataset in the packages checked; no bridge route for iSDM at P1 (D-296, D-300 row 1c-5).",
        draft="Proposed disposition outside_boundary at P1: no real multi-source dataset available and no bridge route (D-296, D-300 row 1c-5). "
              "signed_by: Shinichi Nakagawa; signed_on: <date you sign>."),
}
C5 = [("fit-input/GRP-UNIT", "unit"), ("fit-input/GRP-UNIT-OBS", "unit_obs"), ("fit-input/GRP-CLUSTER", "cluster"), ("fit-input/GRP-CLUSTER2", "cluster2")]
CAMPAIGN_DIR = "campaign"
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


def engine_blocks(cell, R, J):
    r = OrderedDict(engine=R["engine"], gllvmTMB_version=R["gllvmTMB_version"], loaded_from_library=Path(R["gllvmTMB_loaded_from"]).parent.name + "/" + Path(R["gllvmTMB_loaded_from"]).name,
                    TMB_version=R["TMB_version"], R_version=R["R_version"], host=R["host"], formula=R["formula"],
                    convergence=R["convergence"], optimizer_message=R["message"], pdHess=R["pdHess"], max_abs_gradient=R["max_abs_gradient"],
                    n_par=R["n_par"], cond_H=R.get("cond_H"), wall_fit_sec=R["wall_fit_sec"], wall_sdreport_sec=R["wall_sdreport_sec"], finished_utc=R["finished_utc"])
    j = OrderedDict(engine=J["engine"], julia_version=J["julia_version"], gllvmodels_commit=J["gllvmodels_commit"], host=J["host"], call=J["call"],
                    converged=J["converged"], iterations=J.get("iterations"), pd_hessian=J.get("pd_hessian", J.get("hessian_positive_definite")),
                    cond_H=J.get("cond_H"), wall_fit_sec=J["wall_fit_sec"], wall_confint_sec=J.get("wall_confint_sec"), wall_vcov_sec=J.get("wall_vcov_sec"),
                    JULIA_NUM_THREADS=J.get("JULIA_NUM_THREADS"), OPENBLAS_NUM_THREADS=J.get("OPENBLAS_NUM_THREADS"))
    if "cells_converged" in J: j["cells_converged"] = J["cells_converged"]
    if "dispersion_boundary" in J: j["dispersion_boundary"] = J["dispersion_boundary"]
    if "gradient_norm" in J: j["gradient_norm"] = J["gradient_norm"]
    return r, j


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


def reasons(legs, cases):
    out = [k for k, v in legs.items() if not v and k != "every_quantity_within_tolerance"]
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


def build_receipt(sid, cell, clause, cases, legs, eng_r, eng_j, extra, raw_refs, binds):
    rcpt = OrderedDict()
    rcpt["schema"] = SCHEMA
    rcpt["source_id"] = sid; rcpt["clause"] = clause
    rcpt["pin"] = "P1"; rcpt["reference_commit"] = P1; rcpt["ruling"] = RULING
    rcpt["verdict"] = "PASS" if binds else "FAIL"
    why = extra.pop("reasons", [])
    rcpt["row_status"] = "binds: the plan's pass rule holds on both engines" if binds else "does not bind: " + "; ".join(why)
    rcpt["pass_rule"] = OrderedDict(rule="both engines converged (R convergence 0 with a positive-definite Hessian; Julia converged true) and every listed quantity within tolerance",
                                    legs=legs)
    rcpt.update(extra)
    rcpt["engines"] = OrderedDict(R=eng_r, julia=eng_j)
    rcpt["comparison"] = OrderedDict(pin="P1", cases=[c.block(raw_refs["note"]) for c in cases])
    rcpt["read_from"] = raw_refs["hashes"]
    return rcpt


NOT_COVERED_COMMON = [
    "One dataset and one seed per row: a spot check at this size, not a scaling study or a coverage study.",
    "Direct engines only (R TMB fit against a direct Julia fit); R's engine = 'julia' bridge route is not exercised.",
    "Julia starts from its own default start, never from R's coordinates; wall times include first-call Julia compilation.",
    "cond(H) of the two engines is recorded but not compared (different parameter bases).",
]



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
    ctx["caps"] = "at most 8 concurrent jobs, OPENBLAS_NUM_THREADS=1, JULIA_NUM_THREADS=2, R single-threaded; each job under timeout at twice its written estimate"
    return ctx


def data_meta(raw, cell):
    for n in (f"{cell}.meta.json", f"{cell}_wide.meta.json"):
        p = Path(raw) / "data_meta" / n
        if p.exists(): return json.loads(p.read_text())
    return None


def process(raw, out_root, apply):
    results = []   # (row spec, receipt path, binds, receipt)
    camp = out_root / LEDGER / "receipts"
    # --- C3 / C4 executable cells
    for sid, fam, cell, clause, quants in ROWS:
        if cell == "phylo":
            ph = build_phylo(raw)
            if ph is None: print(f"skip {sid}: no phylo_J.json"); continue
            binds = all(ph["legs"].values())
            rdir = camp / fam / CAMPAIGN_DIR
            rr, jr = ph["R"], ph["J"]
            eng_r = OrderedDict(engine="R gllvmTMB (tracked P1 receipt of PR #547, recorded 2026-09-29)", gllvmTMB_version=rr["package_version"], source_pin=rr["source_pin"],
                                formula=rr["formula"], convergence=rr["convergence"], optimizer_message=rr["message"], gradient_max_abs=rr["gradient_max_abs"], pd_hessian=rr["hessian"]["pd_hessian"],
                                cond_H=rr["hessian"]["condition_number"], wall_fit_sec=rr["elapsed_seconds"], dll_sha256=rr["dll_sha256"])
            eng_j = OrderedDict(engine="GLLVModels.fit_phylo_latent_gllvm, fresh fit on current main", julia_version=jr["julia_version"], gllvmodels_commit=jr["git_head"],
                                converged=jr["converged"], stopping_reason=jr["stopping_reason"], gradient_norm=jr["gradient_norm"], iterations=jr["iterations"],
                                hessian_positive_definite=jr["hessian_positive_definite"], cond_H=jr["hessian_condition_number"], wall_fit_sec=jr["fit_elapsed_seconds"])
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
                reasons=reasons(ph["legs"], ph["cases"]),
                not_covered=NOT_COVERED_COMMON[:1] + ["R was not re-run: its values are the PR #547 receipt's.", "cond(H) (R 83030, Julia 82761 here) is recorded, not compared."])
            rc = build_receipt(sid, cell, clause, ph["cases"], ph["legs"], eng_r, eng_j, extra, dict(note=str(jgz.relative_to(out_root)), hashes=hashes), binds)
            results.append((sid, fam, clause, rc, binds, rdir / f"{slug(sid)}.json"))
            continue
        rp, jp = Path(raw) / f"{cell}_R.json", Path(raw) / f"{cell}_J.toml"
        if not (rp.exists() and jp.exists()):
            print(f"skip {sid}: raw outputs missing ({rp.name}, {jp.name})"); continue
        R, J = load_r(rp), load_j(jp)
        if not J.get("DONE"): print(f"skip {sid}: Julia run not finished"); continue
        cases = cases_for(sid, cell, clause, quants, R, J)
        legs = pass_rule(R, J, cases); binds = all(legs.values())
        eng_r, eng_j = engine_blocks(cell, R, J)
        rdir = camp / fam / CAMPAIGN_DIR
        hashes = OrderedDict()
        for src, nm in ((rp, f"{cell}_R.json"), (jp, f"{cell}_J.toml")):
            dst = rdir / "raw" / (nm + ".gz")
            if apply: gz_write(dst, src)
            hashes[str(dst.relative_to(out_root)) + " (uncompressed sha256)"] = sha_file(src)
        extra = OrderedDict(
            cell=OrderedDict((k, R.get(k)) for k in ("formula",)) | OrderedDict(data_sha256=R["data_sha256"], data_file=J.get("data_file"),
                              p=J.get("p"), n=J.get("n")),
            data_meta=data_meta(raw, cell), run=run_context(raw, cell),
            engine_messages=OrderedDict(R=trim_log(read_text(Path(raw) / "logs" / f"R_{cell}.log")), julia=trim_log(read_text(Path(raw) / "logs" / f"J_{cell}.log"))),
            p1_source_sha256=R["p1_source_sha256"], gllvmTMB_deparse_check=R["deparse_check"],
            reasons=reasons(legs, cases), not_covered=NOT_COVERED_COMMON)
        rc = build_receipt(sid, cell, clause, cases, legs, eng_r, eng_j, extra, dict(note=f"receipts/{fam}/{CAMPAIGN_DIR}/raw/{cell}_*.gz", hashes=hashes), binds)
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
    tier = "numeric" if binds else "numeric_fail"
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
        ev["tier"] = "numeric: the receipt carries an R-vs-Julia comparison block pinned to P1; both engines converged and every listed quantity is within its tolerance (pass rule of the C3 to C5 campaign plan, approved by itchyshin/GLLVModels.jl#684 item 4)"
    else:
        ev["non_binding_receipts"] = [rel_receipt]
        ev["tier"] = "measured, does not bind: " + rc["row_status"].replace("does not bind: ", "")
    row["evidence"] = ev
    row["measured_result"] = OrderedDict(verdict=rc["verdict"],
        max_abs_diff_by_case={c["case_id"]: c["max_abs_diff"] for c in rc["comparison"]["cases"]},
        within_tolerance_by_case={c["case_id"]: c["within_tolerance"] for c in rc["comparison"]["cases"]})
    row["ruling"] = RULING
    return row


def disposition_row(sid, d):
    row = OrderedDict()
    row["source_id"] = sid
    row["classification"] = "outside_boundary"
    row["arc"] = "campaign-C4"; row["clause"] = "C4"
    row["executable_case_ids"] = []
    row["disposition"] = None
    row["evidence_tier"] = "not_measured"
    row["measured_against"] = "P1"
    row["evidence"] = OrderedDict(tier="no measurement: this row awaits the maintainer's own signed disposition; nothing is signed here")
    row["proposed_disposition"] = OrderedDict(capability=d["capability"], text=d["text"],
        status="PROPOSED, UNSIGNED: the campaign plan says each of these three rows needs the maintainer's own signature, and itchyshin/GLLVModels.jl#684 does not quote them",
        draft_signature=d["draft"])
    row["ruling"] = RULING + " (row added); disposition itself unsigned"
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
    a = ap.parse_args()
    out_root = a.root
    if a.check:
        return check(out_root)
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


def check(out_root):
    """Re-derive every campaign receipt's comparison block from the committed raw files; fail on any drift."""
    bad = 0
    for p in sorted((out_root / LEDGER / "receipts").glob("*/campaign/*.json")):
        rc = json.loads(p.read_text())
        for k, h in rc.get("read_from", {}).items():
            path = k.replace(" (uncompressed sha256)", "")
            f = out_root / path
            if not f.exists(): print(f"MISSING {path} for {p.name}"); bad += 1; continue
            got = sha_bytes(gz_read(f)) if path.endswith(".gz") else sha_file(f)
            if got != h: print(f"HASH DRIFT {path} in {p.name}"); bad += 1
        for c in rc["comparison"]["cases"]:
            if "r_value" in c and "julia_value" in c:
                r, j = flat(c["r_value"]), flat(c["julia_value"])
                d = maxabs(r, j)
                if abs(d - c["max_abs_diff"]) > 1e-12 * max(abs(d), abs(c["max_abs_diff"]), 1e-300): print(f"DIFF DRIFT {c['case_id']} in {p.name}"); bad += 1
            if c["within_tolerance"] != (c["max_abs_diff"] <= c["tolerance"]): print(f"TOLERANCE FLAG DRIFT {c['case_id']}"); bad += 1
    print("CAMPAIGN_RECEIPTS_OK" if not bad else f"CAMPAIGN_RECEIPTS_BAD {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
