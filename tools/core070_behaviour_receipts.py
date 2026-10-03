#!/usr/bin/env python3
"""Behaviour blocks for the inference routing / error-class rows and the aghq control rows.

Ruling 2 of itchyshin/GLLVModels.jl#684 (the behavioural tier, GATES.md "Rulings of 2026-10-02")
closes a row whose R behaviour is a routing decision, a refusal or an error class when a receipt
shows both engines giving the same behaviour. The tier covers a frozen list of rows (the 59
inference routing and error-class rows and four named C1 rows); a row outside that list never
binds behaviourally. This tool writes the evidence. It reads only tracked raw artefacts (nothing
is re-run, nothing is typed by hand) and adds to each case receipt:

  behaviour              the S1 block: one entry per row that can bind, scoped by source_id (every
                         entry carries source_id, because the inference receipts share case ids
                         across rows), holding the label R produced and the label Julia produced
  behaviour_not_bound    one record per row that gets no entry, with the exact reason and the raw
                         evidence for it

and it writes behaviour-equivalence.json from CLASSES below. Every class lists the exact raw labels
of both engines for one route, refusal or error class, and its basis cites the P1 R function and
the Julia function that implement the same behaviour. A label in no class stands for itself, and no
entry here relies on that: both labels of every entry are listed in one class.

The 7 aghq/AGHQ-CTRL-* rows and inference/CI-ROUTE-009 are not in the frozen list. Their receipts
keep a behaviour block as non-binding evidence, their rows stay at paired_control_categorical_pass
and partial_non_numeric_case, and each carries a one-line note that binding them needs the
maintainer to confirm that ruling 2 covers them.

Raw artefacts read (all tracked under docs/dev-log/core070/true-parity-latest/receipts/):

  inference wave2 (45 route rows)    inference/inference-batch-p1/inference-batch-results.json (Julia,
                                     the solver that ran) and .../r-crosscheck/p1-route-probe-results.tsv
                                     (R, the endpoint the dispatcher chose), plus the probe fixture
                                     test/parity/fixtures/core070_inference_routes.tsv for the method
                                     each row asked for
  inference wave4 (14 error rows)    inference/inference-remainder-p1/r-oracle.json and julia-results.json (read for the not-bound record only)
  inference wave5 (CI-ROUTE-009)     postfit/surface-conversion-p1/r-oracle.json and julia-results.json
  aghq control (7 rows)              aghq/aghq-control-p1/r-oracle.json and julia-results.json

Labels. R's label is the raw string R produced (the probe's `actual`, the first line of the R error
message, or R's printed normalised value). Julia's label is `<target>:<raw>` where the target comes
from the raw case id (for example `lambda`, `sigma_B`, `communality`) and `<raw>` is the solver tag
or exception type the Julia harness recorded. The target prefix keeps one quantity's route from
matching another's.

A row gets NO entry (and so stays unbound) when its raw record shows the engines do not do the same
thing, even if the two labels look alike:

  fallback   R's message says it fell back to another method while Julia ran the method asked for
             (CI-ROUTE-067, CI-ROUTE-070: bootstrap on fixed effects and sigma_eps).
  refusal_not_observed
             The Lambda reject rows (CI-ROUTE-006, CI-ROUTE-007): R's probe stubs .confint_lambda, so
             the raw R record is a dispatch label, not a refusal. Julia refuses only on AGHQ-record fits.
  no_valid_method_control
             The 14 derived-quantity error-class rows: Julia's MethodError fires for every method value
             (the *_wald_ci functions take no method keyword), valid or not, so it is not the refusal R
             gives for a method outside its supported set. No error class is mapped; no entry is written.
  public_route_differs
             The Sigma bootstrap rows (CI-ROUTE-045, 048, 057, 060, 063): Julia's public
             confint(fit, Y; parm, method = :bootstrap) on a structured fit falls through to the
             one-argument Wald confint and ignores method, so the request does not take the route R
             takes. The harness called bootstrap_ci directly, which is not the request R's row makes.
  not_exported
             The derived-quantity profile and bootstrap rows (CI-ROUTE-016, 018, 025, 032, 039):
             Julia's side is GLLVM.profile_ci_derived / GLLVM.bootstrap_ci_derived, which are not
             exported, so a Julia user cannot make the request. Same treatment as the DEFAULT rows
             CI-ROUTE-022, 029 and 036, where the harness chose the function.
  withdrawn  R routes `profile` for communality, rho or proportion to a function that, at P1, raises
             gllvmTMB_nonlinear_profile_withdrawn (the same probe records the guard row), while Julia
             returns a profile interval.
  default    The row asked for the DEFAULT method and R's default is profile, but the Julia harness
             called an explicit profile function. Julia's own default is Wald, so Julia's default was
             not exercised and is not the same route. The same holds for a DEFAULT row whose Julia call
             is an explicit *_wald_ci function (CI-ROUTE-022, 029, 036): Julia has no default-method
             dispatcher for derived quantities, so the harness chose the function, not Julia's default.

`--check` re-derives every block and the equivalence file and exits nonzero on any difference.
`--write` writes them. Row tiers are not changed here: the case-map generators call overlay_row()
from build_rows, so `core070_inference_p1_receipts.py --check` and `core070_aghq_p1_receipts.py --check`
agree with what this tool wrote.

Usage:
  python3 tools/core070_behaviour_receipts.py --write
  python3 tools/core070_behaviour_receipts.py --check
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = "docs/dev-log/core070/true-parity-latest"
OUT = ROOT / LEDGER
EQUIV_PATH = OUT / "behaviour-equivalence.json"
INF = f"{LEDGER}/receipts/inference"
AGHQ = f"{LEDGER}/receipts/aghq"
SURF = f"{LEDGER}/receipts/postfit/surface-conversion-p1"
FIXTURE = "test/parity/fixtures/core070_inference_routes.tsv"
P1_SHA = "9539352f66f2db2cc26b1c393e67212a359b60c9"
RUN_JULIA = "61c5eda48"      # glvmodels_commit of the wave2 and wave4 inference receipts
RUN_SURF = "681c4c3ca"       # run commit of the wave5 surface-conversion receipts (PR #569)
RUN_AGHQ = "fd92b6551"       # glvmodels_commit of the aghq control receipts
RULING = "itchyshin/GLLVModels.jl#684 item 2"

TIER_TEXT = (f"behavioural, counted under {RULING}: for every executable case id both engines' raw output gives the "
             "same route, refusal or error class, compared through behaviour-equivalence.json. It is a behavioural "
             "row, not a numeric one")

# Rows a behaviour block can close, by their pre-flip evidence_tier.
OVERLAY_TIERS = {"routing_control_flow", "reject_error_class", "partial_non_numeric_case",
                 "paired_control_categorical_pass"}

# ---------------------------------------------------------------------------
# Equivalence classes. R lines: gllvmTMB P1 (9539352f6). Julia lines: at the run commit named.
# ---------------------------------------------------------------------------
RZ = "R/z-confint-gllvmTMB.R"
JC = f"src/confint.jl (at {RUN_JULIA})"


def cls(kind, canonical, r, julia, basis):
    return {"kind": kind, "canonical": canonical, "r": r, "julia": julia, "basis": basis}


CLASSES = [
    # --- Wald, packed parameters ---
    cls("route", "lambda:wald", [".confint_lambda:wald"], ["lambda:wald_packed"],
        f"R: confint.gllvmTMB_multi sends parm Lambda to .confint_lambda ({RZ}:1627-1640, wald branch :285); "
        f"Julia: confint(fit, y; parm, method=:wald) (src/confint.jl:359-368, default method=:wald at :364) reaches "
        f"_gaussian_record_confint (src/families/aghq_gaussian_fit.jl:240), the Wald interval, so a call with no method "
        f"and an explicit wald call take the Wald route in both."),
    cls("route", "sigma:wald", [".confint_sigma:wald"],
        ["sigma_B:wald_packed", "sigma_W:wald_packed", "sigma_phy:wald_packed"],
        f"R: .confint_sigma sends method wald to .confint_sigma_wald ({RZ}:1803-1810, :1891); Julia: "
        f"confint(fit, y; parm=sigma_*, method=:wald, Σ_phy) takes the one-argument Wald confint(fit) for structured "
        f"fits (src/confint.jl:246, :359-368), which is Wald whatever method is passed."),
    cls("route", "beta:wald", ["tidy:wald"], ["beta:wald_packed"],
        f"R: fixed-effect parm with method wald goes through tidy(object, \"fixed\") ({RZ}:1774-1776); Julia: "
        f"confint(fit, y; parm=\"beta[1]\", method=:wald) reaches _gaussian_record_confint "
        f"(src/families/aghq_gaussian_fit.jl:240); both give the Wald interval for a fixed effect."),
    cls("route", "sigma_eps:wald", [".confint_wald_targets:wald"], ["sigma_eps:wald_packed"],
        f"R: profile_targets() parm sigma_eps with method wald goes to .confint_wald_targets "
        f"({RZ}:1331, called at :1762); Julia: confint(fit, y; parm=\"sigma_eps\", method=:wald) reaches "
        f"_gaussian_record_confint (src/families/aghq_gaussian_fit.jl:240); both give the Wald interval."),
    # --- Wald, derived quantities (Julia picks the method by function name) ---
    cls("route", "communality:wald", [".confint_communality:wald"], ["communality:wald_derived"],
        f"R: .confint_communality wald branch ({RZ}:941, :969); Julia: communality_wald_ci "
        f"(src/confint_derived_wald.jl:375 at {RUN_JULIA}), a logit-scale Wald interval. The Julia call is an explicit "
        f"function chosen by name, so only rows that ask for wald by name use this class."),
    cls("route", "proportion:wald", [".confint_proportion:wald"], ["proportion:wald_derived"],
        f"R: .confint_proportion wald branch ({RZ}:1188, :1207); Julia: icc_wald_ci "
        f"(src/confint_derived_wald.jl:395 at {RUN_JULIA}), a logit-scale Wald interval for a proportion. The Julia call is an explicit "
        f"function chosen by name, so only rows that ask for wald by name use this class."),
    cls("route", "phylo_signal:wald", [".confint_phylo_signal:wald"], ["phylo_signal:wald_derived"],
        f"R: .confint_phylo_signal wald branch ({RZ}:894, :916); Julia: phylo_signal_wald_ci "
        f"(src/confint_derived_wald.jl:412 at {RUN_JULIA}), a logit-scale Wald interval."),
    cls("route", "rho:fisher-z", [".confint_rho:fisher-z"], ["rho:wald_derived"],
        f"R: .confint_rho with method fisher-z calls extract_correlations(method = \"fisher-z\") ({RZ}:1034, :1061); "
        f"Julia: correlation_wald_ci (src/confint_derived_wald.jl:358 at {RUN_JULIA}), a Fisher-z transformed Wald "
        f"interval. The Julia call is an explicit function chosen by name."),
    # --- profile ---
    cls("route", "lambda:profile", [".confint_lambda:profile"], ["lambda:profile"],
        f"R: .confint_lambda profile branch calls loading_profile ({RZ}:372-393); Julia: profile_ci(fit, "
        f"\"Lambda_B[1,1]\"; y) (src/confint_profile.jl:532-537, :406 at {RUN_JULIA}), the profile-likelihood interval."),
    cls("route", "beta:profile", [".confint_fixef_profile:profile"], ["beta:profile"],
        f"R: fixed-effect parm with method profile calls .confint_fixef_profile ({RZ}:1776-1777, defined :2098); "
        f"Julia: profile_ci(fit, \"beta[1]\"; y) (src/confint_profile.jl:532-537, :406 at {RUN_JULIA}); both profile the "
        f"likelihood and need no cached standard errors."),
    cls("route", "sigma_eps:profile", [".confint_profile_targets:profile"], ["sigma_eps:profile"],
        f"R: profile_targets() parm sigma_eps with method profile calls .confint_profile_targets ({RZ}:1266, called at "
        f":1746); Julia: profile_ci(fit, \"sigma_eps\"; y) (src/confint_profile.jl:532-537, :406 at {RUN_JULIA})."),
    # --- bootstrap (Lambda only: the Sigma and derived-quantity bootstrap rows are not bound, see the not_bound reasons) ---
    cls("route", "lambda:bootstrap", [".confint_lambda:bootstrap"], ["lambda:bootstrap"],
        f"R: .confint_lambda bootstrap branch calls .loading_ci_bootstrap ({RZ}:346-368); Julia: bootstrap_ci(fit; "
        f"parms=\"Lambda_B[1,1]\", y) (src/confint_bootstrap.jl:258 at {RUN_JULIA}); both refit simulated data."),
    # --- CI-ROUTE-009: profile interval for two-level repeatability is withdrawn in both (non-binding: outside the frozen scope) ---
    cls("refusal", "icc:profile-withdrawn",
        ["A profile interval for canonical full-covariance repeatability is not currently available."],
        ["A profile interval for canonical full-covariance two-level repeatability is not currently available."],
        f"R: .confint_icc raises gllvmTMB_repeatability_profile_withdrawn for method profile ({RZ}:855, :862-867); "
        f"Julia: repeatability_ci throws TwoLevelRepeatabilityProfileWithdrawn for method=:profile "
        f"(src/twolevel.jl:605-612 at {RUN_SURF}). Both refuse the same request, name the same reason (the old "
        f"profile estimated only a diagonal-companion ratio) and point to wald or bootstrap."),
    # --- aghq request normalisation (non-binding: outside the frozen scope) ---
    cls("route", "aghq:off", ["FALSE"], ["off"],
        f"R: .gllvmTMB_normalize_aghq maps NULL and FALSE to FALSE, the Laplace approximation (R/gllvmTMB.R:2492); "
        f"Julia: _aghq_request maps false and nothing to :off (src/families/aghq_fit_info.jl:38 at {RUN_AGHQ})."),
    cls("route", "aghq:auto", ["\"auto\""], ["auto"],
        f"R: .gllvmTMB_normalize_aghq maps \"auto\" and TRUE to \"auto\" (R/gllvmTMB.R:2493-2494); Julia: _aghq_request "
        f"maps true and :auto to :auto (src/families/aghq_fit_info.jl:39 at {RUN_AGHQ}). Julia rejects the string "
        f"\"auto\" itself with ArgumentError, so the Julia side is given the symbol :auto."),
    cls("route", "aghq:nodes=1", ["1L"], ["1"],
        f"R: .gllvmTMB_normalize_aghq keeps a single positive integer as integer (R/gllvmTMB.R:2495-2498); Julia: "
        f"_aghq_request keeps a positive Integer as Int (src/families/aghq_fit_info.jl:40-42 at {RUN_AGHQ}); one node."),
    cls("route", "aghq:nodes=2", ["2L"], ["2"],
        f"R: .gllvmTMB_normalize_aghq keeps a single positive integer as integer (R/gllvmTMB.R:2495-2498); Julia: "
        f"_aghq_request keeps a positive Integer as Int (src/families/aghq_fit_info.jl:40-42 at {RUN_AGHQ}); two nodes."),
    cls("route", "aghq:nodes=9", ["9L"], ["9"],
        f"R: .gllvmTMB_normalize_aghq keeps a single positive integer as integer (R/gllvmTMB.R:2495-2498); Julia: "
        f"_aghq_request keeps a positive Integer as Int (src/families/aghq_fit_info.jl:40-42 at {RUN_AGHQ}); nine nodes. Julia rejects a whole-number float such as "
        f"9.0 with ArgumentError, so the Julia side is given Int 9."),
]


def equivalence_doc():
    return {"schema": 1, "pin": "P1",
            "citation_note": (f"R file:line citations are at gllvmTMB P1 ({P1_SHA[:9]}). Julia file:line citations are at "
                              f"the commit that ran the cited receipt ({RUN_JULIA} for the inference rows, {RUN_AGHQ} "
                              "for the aghq rows, " + RUN_SURF + " for CI-ROUTE-009), not at current main: src has "
                              "changed since, so the lines may have moved."),
            "classes": CLASSES}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(p):
    return json.loads(Path(p).read_text())


def dump(value):
    return json.dumps(value, indent=2) + "\n"


def read_tsv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t", escapechar="\\", doublequote=False))


def reads(*rels):
    return {rel: sha(ROOT / rel) for rel in rels}


def first_sentence(msg):
    """R wraps messages at 80 columns, so normalise whitespace before cutting at the first full stop."""
    flat = re.sub(r"\s+", " ", msg).strip()
    m = re.match(r"(.+?\.)(\s|$)", flat)
    return m.group(1) if m else flat


SIGMA_TARGETS = ("sigma_B", "sigma_W", "sigma_phy")
DERIVED_TARGETS = ("communality", "rho", "phylo_signal", "proportion")
NOT_EXPORTED_CALLS = ("GLLVM.profile_ci_derived", "GLLVM.bootstrap_ci_derived")

TARGETS = {"LAMBDA": "lambda", "BETA": "beta", "SIGMA-EPS": "sigma_eps", "SIGMA-B": "sigma_B", "SIGMA-W": "sigma_W",
           "SIGMA-PHY": "sigma_phy", "COMMUNALITY": "communality", "RHO": "rho", "PHYLO-SIGNAL": "phylo_signal",
           "PROPORTION": "proportion", "ICC": "icc"}


def target_of(case_id):
    m = re.match(r"CORE070-INFERENCE-(.+?)-(CI-METHOD-ROUTE|CI-UNSUPPORTED-METHOD-REJECT|BOOTSTRAP-FALLBACK-DIVERGENCE"
                 r"|CI-BOOTSTRAP-FALLBACK-DIVERGENCE)$", case_id)
    if not m or m.group(1) not in TARGETS:
        raise SystemExit(f"cannot read a target from case id {case_id}")
    return TARGETS[m.group(1)]


# ---------------------------------------------------------------------------
# derivation, one function per raw source. Each returns (entries, not_bound, read_from).
# ---------------------------------------------------------------------------
def wave2(case_id, rec):
    ib = f"{INF}/inference-batch-p1"
    jres = {c["source_id"]: c for c in load(ROOT / ib / "inference-batch-results.json")["cases"]}
    rprobe = {r["id"]: r for r in read_tsv(ROOT / ib / "r-crosscheck/p1-route-probe-results.tsv")}
    fixture = {r["id"]: r for r in read_tsv(ROOT / FIXTURE)}
    target = target_of(case_id)
    entries, not_bound = [], []
    for pr in rec["per_row"]:
        sid = pr["source_id"]
        n = sid.split("/")[-1]
        r, j, fx = rprobe[n], jres[n], fixture[n]
        if j["case_id"] != case_id:
            raise SystemExit(f"{sid}: raw Julia case {j['case_id']} is not receipt case {case_id}")
        if pr["r_route_actual"] != r["actual"] or pr["julia_route_actual"] != j["actual"]:
            raise SystemExit(f"{sid}: per_row disagrees with the raw artefacts "
                             f"(per_row R {pr['r_route_actual']!r} Julia {pr['julia_route_actual']!r}; raw R "
                             f"{r['actual']!r} Julia {j['actual']!r}); fix the receipt before adding behaviour")
        if r["pass"] != "TRUE" or not j["ok"]:
            raise SystemExit(f"{sid}: raw artefact records a failed probe; not writing behaviour")
        method = fx["method"]
        if "falling back" in r["messages"]:
            not_bound.append({
                "source_id": sid, "reason": "fallback",
                "text": (f"R fell back to another method and Julia ran the method asked for. Asked for {method}. R's "
                         f"probe recorded route {r['actual']!r} with the message {r['messages']!r}; Julia's solver was "
                         f"{j['actual']!r}. Different behaviour, so no entry."),
                "evidence": {"r": f"{ib}/r-crosscheck/p1-route-probe-results.tsv#{n}", "julia": f"{ib}/inference-batch-results.json#{n}"}})
            continue
        if method == "DEFAULT" and r["actual"].endswith(":profile"):
            if target in ("phylo_signal",):
                why = ("Julia has no default-method dispatcher for derived quantities: profile_ci_derived and "
                       "bootstrap_ci_derived are not exported, and the exported entry for this quantity is "
                       "phylo_signal_wald_ci, a Wald function (src/GLLVModels.jl export list, src/confint_derived_wald.jl:412)")
            else:
                why = ("Julia's own default is Wald: confint(fit, y; parm) defaults to method=:wald "
                       "(src/confint.jl:364, :246 at " + RUN_JULIA + ")")
            not_bound.append({
                "source_id": sid, "reason": "default",
                "text": (f"The row asks for the DEFAULT method. R's default for this parm is profile (probe route "
                         f"{r['actual']!r}), but the Julia harness called an explicit profile function "
                         f"({pr['julia_call']}), so Julia's default was not exercised. {why}. Not the same default route."),
                "evidence": {"r": f"{ib}/r-crosscheck/p1-route-probe-results.tsv#{n}", "julia": f"{ib}/inference-batch-results.json#{n}",
                             "fixture": f"{FIXTURE}#{n}"}})
            continue
        if method == "DEFAULT" and not pr["julia_call"].startswith("confint("):
            not_bound.append({
                "source_id": sid, "reason": "default",
                "text": (f"The row asks for the DEFAULT method, but the Julia harness called an explicit function "
                         f"({pr['julia_call']}). Julia has no default-method dispatcher for derived quantities (the "
                         f"exported entries are the per-method *_wald_ci functions), so Julia's default was not "
                         f"exercised, the same shape as the other DEFAULT rows left unbound. Whether the exported "
                         f"Wald function counts as Julia's default is not signed under {RULING}."),
                "evidence": {"r": f"{ib}/r-crosscheck/p1-route-probe-results.tsv#{n}", "julia": f"{ib}/inference-batch-results.json#{n}",
                             "fixture": f"{FIXTURE}#{n}"}})
            continue
        if r["actual"].endswith(":profile") and target in ("communality", "rho", "proportion"):
            guards = [g for g, f in fixture.items() if f["stage"] == "guard" and f["parm"] == fx["parm"]
                      and f["method"] == "profile"]
            if len(guards) != 1 or "gllvmTMB_nonlinear_profile_withdrawn" not in rprobe[guards[0]]["actual_class"]:
                raise SystemExit(f"{sid}: expected one withdrawn guard row for parm {fx['parm']}, found {guards}")
            g = guards[0]
            not_bound.append({
                "source_id": sid, "reason": "withdrawn",
                "text": (f"R's dispatcher forwards method profile to {r['actual'].split(':')[0]}, but that function raises "
                         f"gllvmTMB_nonlinear_profile_withdrawn at P1; the same probe observed that error on guard row {g} "
                         f"(parm {fx['parm']}, method profile). Julia returned a profile interval ({pr['julia_call']}). "
                         f"R refuses, Julia computes: not the same behaviour."),
                "evidence": {"r": f"{ib}/r-crosscheck/p1-route-probe-results.tsv#{n},{g}", "julia": f"{ib}/inference-batch-results.json#{n}"}})
            continue
        if target in SIGMA_TARGETS and j["actual"] == "bootstrap":
            not_bound.append({
                "source_id": sid, "reason": "public_route_differs",
                "text": (f"R's confint(method = 'bootstrap') for a Sigma parm refits through .confint_sigma_bootstrap "
                         f"(route {r['actual']!r}). Julia's public confint(fit, y; parm, method = :bootstrap) on this "
                         f"structured fit does not refit: it falls through to the one-argument Wald confint and ignores "
                         f"method (src/confint.jl:366-369 at {RUN_JULIA}; inference-batch-contract-p1.json "
                         f"known_findings.lambda_reject_validation_boundary), so a Julia user who makes this request gets "
                         f"a Wald interval. The harness called {pr['julia_call']} directly, which is not the request R's "
                         f"row makes. Not the same route for the same request, so no entry."),
                "evidence": {"r": f"{ib}/r-crosscheck/p1-route-probe-results.tsv#{n}",
                             "julia": f"{ib}/inference-batch-results.json#{n}",
                             "contract": f"{LEDGER}/inference-batch-contract-p1.json"}})
            continue
        if target in DERIVED_TARGETS and pr["julia_call"].startswith(NOT_EXPORTED_CALLS):
            not_bound.append({
                "source_id": sid, "reason": "not_exported",
                "text": (f"R's confint(parm, method = {method!r}) serves a derived quantity (route {r['actual']!r}). "
                         f"Julia's side of this row is {pr['julia_call']}, an internal function that GLLVModels does not "
                         f"export (src/GLLVModels.jl export list), so a Julia user cannot make this request through the "
                         f"public API; the harness chose the function. The same shape as the DEFAULT rows CI-ROUTE-022, "
                         f"029 and 036, which are also left unbound. No entry."),
                "evidence": {"r": f"{ib}/r-crosscheck/p1-route-probe-results.tsv#{n}",
                             "julia": f"{ib}/inference-batch-results.json#{n}",
                             "contract": f"{LEDGER}/inference-batch-contract-p1.json"}})
            continue
        kind = "route"
        r_label, j_label = r["actual"], f"{target}:{j['actual']}"
        entry = {"case_id": case_id, "source_id": sid, "kind": kind, "r_observed": r_label, "julia_observed": j_label,
                 "requested_method": method, "julia_call": pr["julia_call"],
                 "r_source": f"{ib}/r-crosscheck/p1-route-probe-results.tsv#{n}",
                 "julia_source": f"{ib}/inference-batch-results.json#{n}"}
        if j["actual"] == "reject":
            bogus = [rid for rid, f in fixture.items() if f["stage"] == "guard" and f["parm"] == fx["parm"]
                     and f["method"] == "bogus"]
            not_bound.append({
                "source_id": sid, "reason": "refusal_not_observed",
                "text": (f"R's refusal is not recorded for this row. The probe stubs .confint_lambda, so the raw R record is "
                         f"only the dispatch label {r['actual']!r} (actual_class {r['actual_class']!r}), not a refusal; "
                         f"the one R refusal for a bad Lambda method sits on the different guard row {', '.join(bogus)} "
                         f"(method bogus). Julia refuses only on AGHQ-record Gaussian fits and returns Wald for any "
                         f"method on structured fits (inference-batch-contract-p1.json known_findings."
                         f"lambda_reject_validation_boundary). Leave unbound until .confint_lambda is observed directly "
                         f"in R with this method."),
                "evidence": {"r": f"{ib}/r-crosscheck/p1-route-probe-results.tsv#{n}", "julia": f"{ib}/inference-batch-results.json#{n}"}})
            continue
        entries.append(entry)
    rf = reads(f"{ib}/inference-batch-results.json", f"{ib}/r-crosscheck/p1-route-probe-results.tsv", FIXTURE)
    return entries, not_bound, rf


def wave4(case_id, rec):
    rb = f"{INF}/inference-remainder-p1"
    ro, rj = load(ROOT / rb / "r-oracle.json"), load(ROOT / rb / "julia-results.json")
    fixture = {r["id"]: r for r in read_tsv(ROOT / FIXTURE)}
    target = target_of(case_id)
    key = {"icc": "icc_reject", "phylo_signal": "phylo_reject", "communality": "communality_reject",
           "rho": "rho_reject", "proportion": "proportion_reject"}[target]
    jc = rj["cases"][case_id]["methods"]
    not_bound = []
    for sid in rec["source_ids"]:
        n = sid.split("/")[-1]
        method = fixture[n]["method"]
        if method not in ro[key] or method not in jc:
            raise SystemExit(f"{sid}: method {method} is not in the raw oracle for {case_id}")
        r, j = ro[key][method], jc[method]
        r_label = first_sentence(r["message"]) if r["raised"] and r["matches"] else "no-error-raised"
        j_label = j["julia_error_kind"] or "no-error-raised"
        not_bound.append({
            "source_id": sid, "reason": "no_valid_method_control",
            "text": (f"R raised its validated cli_abort ({r_label!r}) and Julia raised {j_label}, but the Julia error is "
                     f"structural, not the same refusal. The *_wald_ci function takes no method keyword, so the call "
                     f"raises MethodError for every method value, including a valid one, while R's error fires only "
                     f"for a method outside its supported set. The harness has no valid-method control, and "
                     f"inference-batch-contract-p1.json known_findings.derived_quantity_reject_spec_defect_reason "
                     f"records that no Julia call this comparand can run against exists for these quantities. A "
                     f"MethodError counting as the same error class is not signed under {RULING}."),
            "evidence": {"r": f"{rb}/r-oracle.json#{key}.{method}",
                         "julia": f"{rb}/julia-results.json#{case_id}.methods.{method}",
                         "contract": f"{LEDGER}/inference-batch-contract-p1.json"}})
    return [], not_bound, reads(f"{rb}/r-oracle.json", f"{rb}/julia-results.json", FIXTURE)


def wave5(case_id, rec):
    ro, rj = load(ROOT / SURF / "r-oracle.json"), load(ROOT / SURF / "julia-results.json")
    r, j = ro["oracle_values"][case_id], rj["cases"][case_id]
    if j["kind"] != "refusal_pair":
        raise SystemExit(f"{case_id}: not a refusal pair")
    r_label = first_sentence(r["message"]) if r["raised"] else "no-refusal"
    j_label = first_sentence(j["julia_message"]) if j["julia_raised"] else "no-refusal"
    sid = rec["source_ids"][0]
    entries = [{"case_id": case_id, "source_id": sid, "kind": "refusal", "r_observed": r_label,
                "julia_observed": j_label, "requested_method": "profile",
                "r_source": f"{SURF}/r-oracle.json#oracle_values.{case_id}",
                "julia_source": f"{SURF}/julia-results.json#cases.{case_id}"}]
    return entries, [], reads(f"{SURF}/r-oracle.json", f"{SURF}/julia-results.json")


def aghq(case_id, rec):
    d = f"{AGHQ}/aghq-control-p1"
    ro, rj = load(ROOT / d / "r-oracle.json")["cases"][case_id], load(ROOT / d / "julia-results.json")["cases"][case_id]
    m = re.match(r"CORE070-AGHQ-(CTRL-[A-Z]+)-PAIRED-CONTROL$", case_id)
    if not m:
        raise SystemExit(f"cannot read a source id from {case_id}")
    sid = f"aghq/AGHQ-{m.group(1)}"
    if rec["r_value"] != ro["r_value"] or rec["julia_label"] != rj["julia_label"]:
        raise SystemExit(f"{case_id}: receipt r_value/julia_label disagree with the raw artefacts")
    r_label = ro["r_value"] if ro["r_assertion_pass"] and not ro["r_call_errored"] else "error"
    j_label = rj["julia_label"] if rj["julia_pass"] else "error"
    entries = [{"case_id": case_id, "source_id": sid, "kind": "route", "r_observed": r_label,
                "julia_observed": j_label, "r_call": rec["r_call"], "julia_call": rec["julia_call"],
                "dialect_note": ("The two engines are given the same scalar request in their own spelling. Julia's "
                                 "_aghq_request rejects R's own literals 'auto' (a string) and 9 as a whole-number "
                                 "float with ArgumentError (tools/core070_aghq_controls_run.jl:23, the dialect "
                                 "check), so the accepted-input sets differ even where the labels agree."),
                "r_source": f"{d}/r-oracle.json#cases.{case_id}", "julia_source": f"{d}/julia-results.json#cases.{case_id}"}]
    return entries, [], reads(f"{d}/r-oracle.json", f"{d}/julia-results.json")


DERIVE = {"routing_control_flow": wave2, "reject_error_class": wave4, "paired_refusal_pair": wave5,
          "paired_control_categorical": aghq}
CASE_DIRS = [f"{INF}/cases", f"{AGHQ}/cases"]


def derive_block(case_id, rec):
    """The behaviour block and not-bound list for one case receipt, or None when the receipt has no
    behavioural evidence to give (the numeric and structural cases)."""
    fn = DERIVE.get(rec.get("evidence_kind"))
    if fn is None:
        return None
    entries, not_bound, rf = fn(case_id, rec)
    block = {"pin": "P1", "ruling": RULING, "read_from": rf, "cases": entries} if entries else None
    return block, not_bound


def tracked_receipts():
    out = {}
    for d in CASE_DIRS:
        for p in sorted((ROOT / d).glob("*.json")):
            out[p.stem] = (p, load(p))
    return out


def with_behaviour(rec, block, not_bound):
    """The receipt with `behaviour` and `behaviour_not_bound` replaced (appended when absent)."""
    new = {k: v for k, v in rec.items() if k not in ("behaviour", "behaviour_not_bound")}
    if block is not None:
        new["behaviour"] = block
    if not_bound:
        new["behaviour_not_bound"] = not_bound
    return new


# ---------------------------------------------------------------------------
# the assembler's behavioural rule, used to flip rows
# ---------------------------------------------------------------------------
_ASM = None


def assembler():
    global _ASM
    if _ASM is None:
        spec = importlib.util.spec_from_file_location("true_parity_assemble", ROOT / "tools/true_parity_assemble.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["true_parity_assemble"] = mod
        spec.loader.exec_module(mod)
        _ASM = mod
    return _ASM


_CITES = None


def citers():
    """case id -> source_ids of every row (any tier) in the ledger's case maps that lists it, as the assembler
    and the checker count them. An entry without source_id covers a case id only when one row cites it."""
    global _CITES
    if _CITES is None:
        by_id = assembler().load_maps(ROOT, Path(LEDGER), [])[0]
        _CITES = {}
        for sid, (_family, r) in by_id.items():
            for cid in r.get("executable_case_ids") or []:
                if isinstance(cid, str):
                    _CITES.setdefault(cid, set()).add(sid)
    return _CITES


OUT_OF_SCOPE_NOTE = (f"Not bound: behaviour block kept as non-binding evidence only (R and Julia give the same label), "
                     f"but this row is not in the frozen scope of {RULING}; binding it needs the maintainer to confirm "
                     f"ruling 2 covers it.")
NOT_BOUND_ROW_NOTES = {
    "public_route_differs": ("Not bound behaviourally: Julia's public confint(fit, Y; parm, method = :bootstrap) on this "
                             "structured fit returns a Wald interval, so the request does not take the same route as "
                             "R's; the harness called bootstrap_ci directly (receipt behaviour_not_bound)."),
    "not_exported": ("Not bound behaviourally: Julia's side is the unexported GLLVM.profile_ci_derived / "
                     "GLLVM.bootstrap_ci_derived, so a Julia user cannot make this request; the harness called an "
                     "internal function (receipt behaviour_not_bound)."),
}


def _add_note(row, text):
    row["note"] = " ".join(dict.fromkeys(n for n in (row.get("note"), text) if n))


def _annotate(row, paths):
    """A one-line note on a row that stays unbound for a reason this tool records: an entry that matches but sits
    outside the frozen scope, or a not-bound record whose reason has a row note."""
    asm = assembler()
    sid = row.get("source_id")
    index = asm.behaviour_equivalence(ROOT)
    for p in paths:
        rec = load(ROOT / p)
        for nb in rec.get("behaviour_not_bound") or []:
            if nb.get("source_id") == sid and nb.get("reason") in NOT_BOUND_ROW_NOTES:
                _add_note(row, NOT_BOUND_ROW_NOTES[nb["reason"]])
        if asm.behavioural_eligible_source_id(sid):
            continue
        for e in (rec.get("behaviour") or {}).get("cases") or []:
            if e.get("source_id") == sid and all(
                    asm._labels_match(index, e["kind"], a, b)
                    for a, b in zip(asm.as_list(e["r_observed"]), asm.as_list(e["julia_observed"]))):
                _add_note(row, OUT_OF_SCOPE_NOTE)


def overlay_row(row, counts):
    """Flip `row` to evidence_tier behavioural when its cited receipts' behaviour blocks bind it under the
    assembler's port of the checker's rule (frozen scope, class identity, entries scoped by source_id where a
    case id has several citers). Moves evidence.non_binding_receipts to evidence.receipt, sets the tier text,
    and nothing else. Updates `counts` (tier count down, `behavioural` up). Returns True if flipped.
    A row that is not flipped gets a one-line note when this tool records why (see _annotate)."""
    if row.get("evidence_tier") not in OVERLAY_TIERS or row.get("measured_against") != P1_SHA:
        return False
    paths = list((row.get("evidence") or {}).get("non_binding_receipts") or [])
    if not paths or not all((ROOT / p).is_file() for p in paths):
        return False
    asm = assembler()
    cand = dict(row, evidence_tier="behavioural", evidence={"receipt": paths})
    if asm.behavioural_receipt_problem(cand, ROOT, asm.behaviour_equivalence(ROOT), citers()) is not None:
        _annotate(row, paths)
        return False
    counts[row["evidence_tier"]] -= 1
    counts["behavioural"] = counts.get("behavioural", 0) + 1
    row["evidence_tier"] = "behavioural"
    row["evidence"] = {"receipt": paths, "tier": TIER_TEXT}
    return True


def row_problem(row):
    """Why a row (given its source_id and case ids) does not bind behaviourally, or None. For reporting."""
    asm = assembler()
    paths = list((row.get("evidence") or {}).get("non_binding_receipts") or (row.get("evidence") or {}).get("receipt") or [])
    cand = dict(row, evidence_tier="behavioural", evidence={"receipt": paths})
    return asm.behavioural_receipt_problem(cand, ROOT, asm.behaviour_equivalence(ROOT), citers())


# ---------------------------------------------------------------------------
# write / check
# ---------------------------------------------------------------------------
def derive_all():
    """case_id -> (path, new receipt dict) for every receipt that carries behaviour evidence."""
    out = {}
    for cid, (path, rec) in tracked_receipts().items():
        d = derive_block(cid, rec)
        if d is None:
            continue
        block, not_bound = d
        out[cid] = (path, with_behaviour(rec, block, not_bound))
    return out


def check_problems():
    problems = []
    if not EQUIV_PATH.is_file() or load(EQUIV_PATH) != equivalence_doc():
        problems.append("behaviour-equivalence.json differs from CLASSES in tools/core070_behaviour_receipts.py")
    tr = tracked_receipts()
    for cid, (path, new) in derive_all().items():
        if tr[cid][1] != new:
            problems.append(f"{path.relative_to(ROOT)}: behaviour block differs from the re-derivation")
    for cid, (path, rec) in tr.items():
        if "behaviour" in rec and rec.get("evidence_kind") not in DERIVE:
            problems.append(f"{path.relative_to(ROOT)}: behaviour block on a receipt this tool does not derive")
    return problems


def write():
    EQUIV_PATH.write_text(dump(equivalence_doc()))
    n = 0
    for cid, (path, new) in derive_all().items():
        path.write_text(dump(new))
        n += 1
    print("equivalence classes", len(CLASSES), "receipts written", n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.write == args.check:
        ap.error("pass exactly one of --write, --check")
    if args.check:
        problems = check_problems()
        if problems:
            print("STALE\n  " + "\n  ".join(problems))
            sys.exit(1)
        print("CORE070_BEHAVIOUR_RECEIPTS_CURRENT", len(derive_all()), "receipts,", len(CLASSES), "classes")
        return
    write()


if __name__ == "__main__":
    main()
