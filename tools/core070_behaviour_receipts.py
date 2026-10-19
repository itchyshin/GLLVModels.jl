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

Post-#709 (W2-1). Rows whose Julia side was an internal function, a MethodError or the Wald fall-through are
re-measured through Julia's public confint(fit, y; parm, method) in inference/inference-post709-p1 (run at
RUN_POST709, tools/core070_inference_post709_batch.jl; verifier tools/core070_verify_inference_post709_batch.py), and
the Lambda refusal has one live R observation (tools/core070_inference_lambda_reject_p1.R). Those rows now bind (the
reasons `not_exported`, `public_route_differs`, `no_valid_method_control` and `refusal_not_observed` no longer occur),
except CI-ROUTE-015 (R's default is profile, Julia's is Wald). A refusal row binds only with a valid-method control
on each side, and its class is one (target, method) pair with the labels read from the raw files.

Signed rulings of 2026-10-05 (vault D-319; LOOP/lanes/true-parity-latest/signed-rulings-2026-10-05.md in the lane kit).
The rows they touch take their Julia side from inference/inference-rulings-p1 (run at RUN_RULINGS,
tools/core070_inference_rulings_batch.jl; verifier tools/core070_verify_inference_rulings_batch.py):

  ruling C   CI-ROUTE-023, 030, 037. Julia's public confint now withdraws the profile for communality, rho and
             proportion (src/confint.jl). R's side is the dispatch row (method profile goes to .confint_<q>) plus the
             guard row of the same parm that observes gllvmTMB_nonlinear_profile_withdrawn (086, 087, 088). Each binds
             with a refusal class of its own, "<q>:profile-withdrawn" (the G7 label), never the bad-method class.
  ruling 3   CI-ROUTE-029. R's default for rho (fisher-z) and Julia's default (:wald, the Fisher-z interval) are one
             route: class "rho:fisher-z".
  alias      CI-ROUTE-034. R's explicit method fisher-z and Julia's method = :fisher_z, which returns exactly the
             :wald result; same class.
  ruling B   the default and fallback rows (015, 043, 046, 055, 058, 061, 065, 067, 068, 070) keep no behaviour
             entry: R and Julia do different things there, and the rows close by a signed disposition written by
             tools/core070_inference_p1_receipts.py, which reads the same run.

The 7 aghq/AGHQ-CTRL-* rows and inference/CI-ROUTE-009 joined the frozen list by maintainer ruling 2026-10-05
(D-319), item A (BEHAVIOURAL_EXTENDED_SOURCE_IDS in tools/true_parity_assemble.py), so their behaviour blocks now bind
their rows when the case-map generators re-derive them. OUT_OF_SCOPE_NOTE is kept for any row still outside the list.

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
  withdrawn  (before the rulings run) R routes `profile` for communality, rho or proportion to a function
             that, at P1, raises gllvmTMB_nonlinear_profile_withdrawn (the same probe records the guard
             row), while Julia returned a profile interval. Since ruling C Julia withdraws it too, and
             these rows bind through the rulings run.
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
FIRST7 = f"{LEDGER}/receipts/first-seven-behaviour/derived"
FIXTURE = "test/parity/fixtures/core070_inference_routes.tsv"
P1_SHA = "9539352f66f2db2cc26b1c393e67212a359b60c9"
RUN_JULIA = "61c5eda48"      # glvmodels_commit of the wave2 and wave4 inference receipts
RUN_SURF = "681c4c3ca"       # run commit of the wave5 surface-conversion receipts (PR #569)
RUN_AGHQ = "fd92b6551"       # glvmodels_commit of the aghq control receipts
RUN_POST709 = "f58de0eb1"    # glvmodels_commit of the post-#709 inference run (inference-post709-p1)
POST709 = f"{INF}/inference-post709-p1"
RUN_RULINGS = "1a80133a7"    # glvmodels_commit of the rulings run (inference-rulings-p1, vault D-319)
RULINGS = f"{INF}/inference-rulings-p1"
SIGNED = "signed rulings of 2026-10-05 (vault D-319)"
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
        f"(src/confint_derived_wald.jl:375 at {RUN_JULIA}), a logit-scale Wald interval. The 61c5eda48 run called that "
        f"function by name; since #709 the public confint(fit, y; parm=\"communality[t]\"), with the default method "
        f"(:wald) or method=:wald, reaches it (src/confint.jl:777-806 at {RUN_POST709}), which is how CI-ROUTE-022 is measured."),
    cls("route", "proportion:wald", [".confint_proportion:wald"], ["proportion:wald_derived"],
        f"R: .confint_proportion wald branch ({RZ}:1188, :1207); Julia: icc_wald_ci "
        f"(src/confint_derived_wald.jl:395 at {RUN_JULIA}), a logit-scale Wald interval for a proportion. The 61c5eda48 run "
        f"called that function by name; since #709 the public confint(fit, y; parm=\"proportion:<component>[t]\"), with the "
        f"default method (:wald) or method=:wald, reaches it (src/confint.jl:777-806 at {RUN_POST709}), which is how "
        f"CI-ROUTE-036 is measured."),
    cls("route", "phylo_signal:wald", [".confint_phylo_signal:wald"], ["phylo_signal:wald_derived"],
        f"R: .confint_phylo_signal wald branch ({RZ}:894, :916); Julia: phylo_signal_wald_ci "
        f"(src/confint_derived_wald.jl:412 at {RUN_JULIA}), a logit-scale Wald interval."),
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
    # --- post-#709 public confint(fit, y; parm, method) routes (W2-1) ---
    cls("route", "phylo_signal:profile", [".confint_phylo_signal:profile"], ["phylo_signal:profile"],
        f"R: .confint_phylo_signal profile branch calls profile_ci_phylo_signal ({RZ}:910); Julia: confint(fit, y; "
        f"parm=\"phylo_signal[t]\", method=:profile) calls profile_ci_phylo_signal (src/confint.jl:823 at {RUN_POST709}); "
        f"both profile the phylogenetic signal of the requested trait."),
    cls("route", "phylo_signal:bootstrap", [".confint_phylo_signal:bootstrap"], ["phylo_signal:bootstrap"],
        f"R: .confint_phylo_signal bootstrap branch calls .phylo_signal_bootstrap_ci ({RZ}:920-927); Julia: confint(fit, "
        f"y; parm=\"phylo_signal[t]\", method=:bootstrap) calls bootstrap_ci_derived (src/confint.jl:841 at {RUN_POST709}); "
        f"both refit simulated data."),
    cls("route", "communality:bootstrap", [".confint_communality:bootstrap"], ["communality:bootstrap"],
        f"R: .confint_communality bootstrap branch calls .communality_bootstrap_ci ({RZ}:994); Julia: confint(fit, y; "
        f"parm=\"communality[t]\", method=:bootstrap) calls bootstrap_ci_derived (src/confint.jl:841 at {RUN_POST709}); "
        f"both refit simulated data. The estimand is gllvmTMB's aligned extract_communality on one side and "
        f"communality(fit) on the other; the class is about the route, not the number."),
    cls("route", "rho:bootstrap", [".confint_rho:bootstrap"], ["rho:bootstrap"],
        f"R: .confint_rho with method bootstrap calls extract_correlations(method = \"bootstrap\") ({RZ}:1061-1076); "
        f"Julia: confint(fit, y; parm=\"rho[i,j]\", method=:bootstrap) calls bootstrap_ci_derived "
        f"(src/confint.jl:841 at {RUN_POST709}); both refit simulated data. The estimand is gllvmTMB's aligned "
        f"extract_correlations on one side and correlation(fit) on the other; the class is about the route."),
    cls("route", "proportion:bootstrap", [".confint_proportion:bootstrap"], ["proportion:bootstrap"],
        f"R: .confint_proportion bootstrap branch calls .proportions_bootstrap_ci ({RZ}:1213); Julia: confint(fit, y; "
        f"parm=\"proportion:<component>[t]\", method=:bootstrap) calls bootstrap_ci_derived "
        f"(src/confint.jl:841 at {RUN_POST709}); both refit simulated data. The estimand is gllvmTMB's aligned "
        f"extract_proportions on one side and proportions(fit; component) on the other; the class is about the route."),
    cls("route", "sigma:bootstrap", [".confint_sigma:bootstrap"],
        ["sigma_B:bootstrap", "sigma_W:bootstrap", "sigma_phy:bootstrap"],
        f"R: .confint_sigma sends method bootstrap to .confint_sigma_bootstrap ({RZ}:1803-1805); Julia: confint(fit, y; "
        f"parm=\"sigma_*[t]\", method=:bootstrap, Σ_phy) calls bootstrap_ci on the structured fit "
        f"(src/confint.jl:598 at {RUN_POST709}) since #709, where before it returned the Wald interval; both refit "
        f"simulated data."),
    # --- ruling 3 and the fisher_z alias (vault D-319): R's fisher-z and Julia's Fisher-z Wald are one route ---
    cls("route", "rho:fisher-z", [".confint_rho:fisher-z"], ["rho:wald_derived"],
        f"R: .confint_rho with method fisher-z, R's default for rho ({RZ}:1693), calls extract_correlations(method = "
        f"\"fisher-z\"), a Fisher-z transformed Wald interval; Julia: confint(fit, y; parm=\"rho[i,j]\") with the default "
        f"method :wald, or method = :fisher_z (an alias that returns exactly the :wald result, src/confint.jl:840 at "
        f"{RUN_RULINGS}), computes the transformed Wald interval with transform :fisher_z (src/confint.jl:858). Signed as one route under {SIGNED}: "
        f"ruling 3 for the default (CI-ROUTE-029) and the fisher_z alias for the explicit request (CI-ROUTE-034). "
        f"R's own method = \"wald\" for rho is a different route ('.confint_rho:wald') and is not in this class."),
    # --- CI-ROUTE-009: profile interval for two-level repeatability is withdrawn in both (in scope since 2026-10-05, item A) ---
    cls("refusal", "icc:profile-withdrawn",
        ["A profile interval for canonical full-covariance repeatability is not currently available."],
        ["A profile interval for canonical full-covariance two-level repeatability is not currently available."],
        f"R: .confint_icc raises gllvmTMB_repeatability_profile_withdrawn for method profile ({RZ}:855, :862-867); "
        f"Julia: repeatability_ci throws TwoLevelRepeatabilityProfileWithdrawn for method=:profile "
        f"(src/twolevel.jl:605-612 at {RUN_SURF}). Both refuse the same request, name the same reason (the old "
        f"profile estimated only a diagonal-companion ratio) and point to wald or bootstrap."),
    # --- aghq request normalisation (in scope since 2026-10-05, item A) ---
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
                              "for the aghq rows, " + RUN_SURF + " for CI-ROUTE-009, " + RUN_POST709 + " for the "
                              "post-#709 public-confint rows, " + RUN_RULINGS + " for the rows under the signed rulings of "
                              "2026-10-05), not at current main: src has "
                              "changed since, so the lines may have moved."),
            "classes": all_classes()}


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
# The post-#709 run (inference-post709-p1): Julia's PUBLIC confint(fit, y; parm, method), run at a clean commit,
# plus one live P1 R observation for the Lambda refusal. Rows it covers take their Julia side from it, not from
# the 61c5eda48 run (which called internal functions, or hit a MethodError). The R side stays what it was: the P1
# route probe and the P1 remainder oracle, plus the new Lambda oracle.
# ---------------------------------------------------------------------------
_P709 = None
R_REFUSAL_KEY = {"icc": "icc_reject", "phylo_signal": "phylo_reject", "communality": "communality_reject",
                 "rho": "rho_reject", "proportion": "proportion_reject"}
# R's valid-method control for each refusal target: the route probe row that sends a VALID method to the same
# estimand's endpoint (a stub of the endpoint, not a fit). Lambda has a live control (r-lambda-reject).
R_CONTROL_ROW = {"icc": "CI-ROUTE-010", "phylo_signal": "CI-ROUTE-017", "communality": "CI-ROUTE-024",
                 "rho": "CI-ROUTE-031", "proportion": "CI-ROUTE-038"}
R_CONTROL_ENDPOINT = {"icc": ".confint_icc", "phylo_signal": ".confint_phylo_signal",
                      "communality": ".confint_communality", "rho": ".confint_rho", "proportion": ".confint_proportion"}
def post709():
    """The tracked raw files of the post-#709 run, checked once: clean run commit, PASS, the Lambda oracle."""
    global _P709
    if _P709 is None:
        rc = load(ROOT / POST709 / "run-commit.json")
        if rc.get("dirty") != [] or not str(rc.get("glvmodels_commit", "")).startswith(RUN_POST709):
            raise SystemExit(f"{POST709}/run-commit.json is not a clean run at {RUN_POST709}: {rc}")
        jr = load(ROOT / POST709 / "inference-post709-results.json")
        ro = load(ROOT / POST709 / "r-lambda-reject/r-oracle.json")
        if jr["status"] != "PASS":
            raise SystemExit("the post-#709 Julia run is not PASS")
        _P709 = {"cases": {c["source_id"]: c for c in jr["cases"]}, "r_lambda": ro["calls"]}
    return _P709


def post709_reads():
    return reads(f"{POST709}/inference-post709-results.json", f"{POST709}/receipt.json",
                 f"{POST709}/run-commit.json", f"{POST709}/verify.txt",
                 f"{POST709}/r-lambda-reject/r-oracle.json", f"{POST709}/r-lambda-reject/receipt.json")


_RUL = None


def rulings():
    """The tracked raw files of the rulings run, checked once: clean run commit at RUN_RULINGS, PASS."""
    global _RUL
    if _RUL is None:
        rc = load(ROOT / RULINGS / "run-commit.json")
        if rc.get("dirty") != [] or not str(rc.get("glvmodels_commit", "")).startswith(RUN_RULINGS):
            raise SystemExit(f"{RULINGS}/run-commit.json is not a clean run at {RUN_RULINGS}: {rc}")
        jr = load(ROOT / RULINGS / "inference-rulings-results.json")
        if jr["status"] != "PASS":
            raise SystemExit("the rulings Julia run is not PASS")
        _RUL = {"cases": {c["source_id"]: c for c in jr["cases"]}}
    return _RUL


def rulings_reads():
    return reads(f"{RULINGS}/inference-rulings-results.json", f"{RULINGS}/receipt.json",
                 f"{RULINGS}/run-commit.json", f"{RULINGS}/verify.txt")


def rulings_entry(case_id, sid, rprobe, fixture):
    """None unless `sid` is a ruling-C, ruling-3 or alias row; else ("entry", dict)."""
    c = rulings()["cases"].get(sid)
    if c is None or c["group"] not in ("withdrawn_profile", "fisher_z_alias") and sid != "inference/CI-ROUTE-029":
        return None
    n = sid.split("/")[-1]
    method, target = fixture[n]["method"], target_of(case_id)
    if c["case_id"] != case_id or c["requested_method"] != method or c["target"] != target:
        raise SystemExit(f"{sid}: the rulings record ({c['case_id']}, {c['target']}, {c['requested_method']}) does not "
                         f"match the row ({case_id}, {target}, {method})")
    ib = f"{INF}/inference-batch-p1"
    src_j = f"{RULINGS}/inference-rulings-results.json#{n}"
    src_r = f"{ib}/r-crosscheck/p1-route-probe-results.tsv#{n}"
    r = rprobe[n]
    if r["pass"] != "TRUE":
        raise SystemExit(f"{sid}: a failed probe; not writing behaviour")
    if c["group"] == "withdrawn_profile":
        g = WITHDRAWN_GUARD[target]
        gr, ctl = rprobe[g], rprobe[WITHDRAWN_CONTROL_ROW[target]]
        if r["actual"] != f".confint_{target}:profile" or gr["pass"] != "TRUE" or \
                "gllvmTMB_nonlinear_profile_withdrawn" not in gr["actual_class"]:
            raise SystemExit(f"{sid}: R's dispatch row or guard row {g} does not show the withdrawn profile")
        if not ctl["actual"].startswith(f".confint_{target}:") or ctl["pass"] != "TRUE":
            raise SystemExit(f"{sid}: R control row {WITHDRAWN_CONTROL_ROW[target]} is not a valid method")
        jc = c["control"]
        if c["outcome"] != "error" or c["error_type"] != "ArgumentError" or "withdrawn" not in c["error_message"] or \
                jc["outcome"] != "result" or jc["finite"] is not True:
            raise SystemExit(f"{sid}: Julia did not withdraw the profile with a working :wald control")
        return "entry", {
            "case_id": case_id, "source_id": sid, "kind": "refusal", "r_observed": first_sentence(gr["actual"]),
            "julia_observed": c["error_message"], "requested_method": method, "julia_call": c["julia_call"],
            "julia_error_type": c["error_type"],
            "r_dispatch": {"route": r["actual"], "source": src_r,
                           "note": (f"the dispatcher sends method profile to .confint_{target}; guard row {g} (same parm, "
                                    f"method profile) observes that function's gllvmTMB_nonlinear_profile_withdrawn "
                                    f"error, so R refuses this request")},
            "r_control": {"kind": "route_probe_stub", "call": f"probe row {WITHDRAWN_CONTROL_ROW[target]}",
                          "result": f"routes to {ctl['actual']}",
                          "source": f"{ib}/r-crosscheck/p1-route-probe-results.tsv#{WITHDRAWN_CONTROL_ROW[target]}"},
            "julia_control": {"call": jc["call"], "result": f"{jc['route_tag']} interval", "source": src_j},
            "ruling": f"{SIGNED}, ruling C",
            "r_source": f"{ib}/r-crosscheck/p1-route-probe-results.tsv#{g}", "julia_source": src_j}
    if c["outcome"] != "result" or c["route_tag"] != "wald_derived" or c["identical_to_wald"] is not True:
        raise SystemExit(f"{sid}: Julia's rho result is not the :wald Fisher-z interval")
    if r["actual"] != ".confint_rho:fisher-z":
        raise SystemExit(f"{sid}: R's route is {r['actual']!r}, not fisher-z")
    entry = {"case_id": case_id, "source_id": sid, "kind": "route", "r_observed": r["actual"],
             "julia_observed": f"{target}:{c['route_tag']}", "requested_method": method, "julia_call": c["julia_call"],
             "julia_result_method": c["result_method"], "julia_transform": c.get("transform"),
             "julia_identical_to_wald": c["identical_to_wald"],
             "ruling": f"{SIGNED}, " + ("ruling 3 (the rho default)" if n == "CI-ROUTE-029"
                                        else "the fisher_z method alias"),
             "r_source": src_r, "julia_source": src_j}
    return "entry", entry


def r_refusal(target, method):
    """(label, source) of R's real refusal of `method` for `target`, or the Lambda match.arg refusal."""
    if target == "lambda":
        r = post709()["r_lambda"][method]
        if not r["raised"]:
            raise SystemExit(f"R did not refuse Lambda method {method!r}")
        return first_sentence(r["message"]), f"{POST709}/r-lambda-reject/r-oracle.json#calls.{method}"
    ro = load(ROOT / f"{INF}/inference-remainder-p1/r-oracle.json")
    r = ro[R_REFUSAL_KEY[target]][method]
    if not (r["raised"] and r["matches"]):
        raise SystemExit(f"R did not give its validated refusal of {method!r} for {target}")
    return first_sentence(r["message"]), f"{INF}/inference-remainder-p1/r-oracle.json#{R_REFUSAL_KEY[target]}.{method}"


def refusal_classes():
    """One refusal class per (target, method) pair measured in the post-#709 run, labels read from the raw files:
    R's first message sentence and Julia's full ArgumentError message. The Lambda rows share one class, because R's
    match.arg message does not name the method (so its two refusals carry one R label)."""
    out = []
    cases = post709()["cases"]
    derived = [(sid, c) for sid, c in sorted(cases.items()) if c["kind"] == "refusal" and c["target"] != "lambda" and c.get("reported", True)]
    basis_t = {
        "icc": (f"{RZ}:880", "icc", "wald, bootstrap"),
        "phylo_signal": (f"{RZ}:929", "phylo_signal", "profile, wald, bootstrap"),
        "communality": (f"{RZ}:1013", "communality", "wald, bootstrap"),
        "rho": (f"{RZ}:1088", "rho", "fisher-z, wald, bootstrap"),
        "proportion": (f"{RZ}:1222", "proportion", "wald, bootstrap")}
    for sid, c in derived:
        t, m = c["target"], c["requested_method"]
        line, name, avail = basis_t[t]
        rlab, _ = r_refusal(t, m)
        parm_txt = re.search(r'parm="([^"]+)"', c["julia_call"]).group(1)
        live = ("; a live R valid-method call errors on this fixture for fit-structure reasons (a single-tier fit "
                "has no two-level or phylogenetic block), which is why the control is a route-probe stub"
                if t in ("icc", "phylo_signal", "communality") else "")
        out.append(cls(
            "refusal", f"{t}:refuse:{m}", [rlab], [c["error_message"]],
            f"R: .confint_{t} aborts for method {m!r}, which is outside its supported set "
            f"({avail}), and the message names the method and the estimand ({line}); Julia: confint(fit, y; "
            f"parm=\"{parm_txt}\", method=Symbol(\"{m}\")) throws "
            f"ArgumentError from _confint_check_method for a method outside :wald, :profile, :bootstrap "
            f"(src/confint.jl:451, :781 at {RUN_POST709}), naming the method, the parm and the available methods. Both "
            f"refuse the same request; each is paired with a valid-method control (R: the route probe sends wald to the "
            f"same endpoint; Julia: the same call with :wald returns an interval). The supported sets differ outside "
            f"this request (R withdrew profile for icc, communality, rho and proportion and accepts fisher-z for rho; "
            f"Julia at {RUN_POST709} computed a profile and refused fisher-z; since the {SIGNED} it withdraws the communality, rho "
            f"and proportion profile and accepts :fisher_z for rho), and the class does not cover those. Fit differs: R refuses on a "
            f"single-tier non-phylo fit, Julia on a phylo fit; both refusals are fit-independent by dispatch order "
            f"({RZ}:1643-1712; Julia checks the method before any estimator runs){live}"))
    lam = [(sid, c) for sid, c in sorted(cases.items()) if c["kind"] == "refusal" and c["target"] == "lambda"]
    rlabs = {r_refusal("lambda", c["requested_method"])[0] for _, c in lam}
    if len(rlabs) != 1:
        raise SystemExit(f"expected one R label for the Lambda refusals, got {rlabs}")
    out.append(cls(
        "refusal", "lambda:refuse-unsupported-method", sorted(rlabs), [c["error_message"] for _, c in lam],
        f"R: confint.gllvmTMB_multi validates method with match.arg over wald, wald_asym, profile, bootstrap "
        f"({RZ}:1717), so fisher-z and bogus are refused before any estimator runs and the message lists the "
        f"supported set without naming the method (hence one R label for both rows); Julia: confint(fit, y; "
        f"parm=\"Lambda_B[1,1]\", method=Symbol(\"fisher-z\") or :bogus) throws ArgumentError from "
        f"_confint_check_method (src/confint.jl:451, :500 at {RUN_POST709}). Both refuse the same two requests. "
        f"Controls: R, a live confint(parm = \"Lambda\", method = \"wald\") on a confirmatory fit returned an interval "
        f"(P1 refuses per-entry Wald on an unconstrained fit, so the control fit pins one loading); Julia, method=:wald on "
        f"the Lambda_B term returned an interval. The supported sets differ outside these requests (R accepts "
        f"wald_asym, Julia refuses it), and the class does not cover that."))
    return out


WITHDRAWN_GUARD = {"communality": "CI-ROUTE-086", "rho": "CI-ROUTE-087", "proportion": "CI-ROUTE-088"}
WITHDRAWN_R_LINES = {"communality": f"{RZ}:964-968", "rho": f"{RZ}:1055-1060", "proportion": f"{RZ}:1202-1206"}
WITHDRAWN_CONTROL_ROW = {"communality": "CI-ROUTE-024", "rho": "CI-ROUTE-031", "proportion": "CI-ROUTE-038"}


def withdrawn_classes():
    """Ruling C (the G7 label): one refusal class per withdrawn quantity, labels read from the raw files. R: the first
    sentence of the guard row's gllvmTMB_nonlinear_profile_withdrawn message. Julia: the full ArgumentError message of
    the rulings run. Distinct from every bad-method class (Julia's message says 'withdrawn', not 'is not available')."""
    rprobe = {r["id"]: r for r in read_tsv(ROOT / f"{INF}/inference-batch-p1/r-crosscheck/p1-route-probe-results.tsv")}
    out = []
    for sid, c in sorted(rulings()["cases"].items()):
        if c["group"] != "withdrawn_profile":
            continue
        t = c["target"]
        g = rprobe[WITHDRAWN_GUARD[t]]
        out.append(cls(
            "refusal", f"{t}:profile-withdrawn", [first_sentence(g["actual"])], [c["error_message"]],
            f"R: confint.gllvmTMB_multi forwards method profile to .confint_{t}, which aborts with class "
            f"gllvmTMB_nonlinear_profile_withdrawn ({WITHDRAWN_R_LINES[t]}): the penalty-based constrained-refit "
            f"prototype is withdrawn pending an exact constraint solver. Julia: confint(fit, y; parm, method=:profile) "
            f"throws ArgumentError from _confint_profile_withdrawn (src/confint.jl:681 at {RUN_RULINGS}), saying the same "
            f"profile is withdrawn and pointing to wald or bootstrap. Withdrawn in Julia under {SIGNED}, ruling C. A "
            f"refusal class of its own (the G7 label), separate from the bad-method refusal class."))
    return out


def all_classes():
    out = CLASSES + refusal_classes() + withdrawn_classes()
    # D-319 N6/N10 permits these exact six public-door rows in the behavioural
    # tier. Build classes only from measured labels that are identical on both
    # sides; a mismatch never creates a class.
    for p in sorted((ROOT / FIRST7).glob("*.json")) if (ROOT / FIRST7).is_dir() else []:
        rec = load(p)
        if rec.get("reference_commit") != P1_SHA or rec.get("verdict") != "PASS":
            continue
        label = rec.get("r_observed")
        if not label or label != rec.get("julia_observed"):
            continue
        sid = rec.get("source_id")
        kind = "error_class" if rec.get("raw_observations", {}).get("R", {}).get("outcome") == "ERROR" else "route"
        out.append(cls(kind, label, [label], [label],
            f"Measured identical public-door behaviour for {sid}; P1 raw R and Julia outputs are preserved in "
            f"{FIRST7}/{p.name}. Included by signed D-319 N6/N10."))
    unique = {}
    for c in out:
        key = (c["kind"], c["canonical"])
        if key not in unique:
            unique[key] = c
        elif unique[key]["r"] != c["r"] or unique[key]["julia"] != c["julia"]:
            raise SystemExit(f"conflicting behaviour class {key}")
    return list(unique.values())


# ---------------------------------------------------------------------------
# derivation, one function per raw source. Each returns (entries, not_bound, read_from).
# ---------------------------------------------------------------------------
def post709_entry(case_id, sid, rprobe, fixture):
    """None when the post-#709 run does not cover `sid`; else ("entry", dict) or ("not_bound", dict)."""
    c = post709()["cases"].get(sid)
    if c is None:
        return None
    n = sid.split("/")[-1]
    fx = fixture[n]
    method, target = fx["method"], target_of(case_id)
    if c["case_id"] != case_id or c["requested_method"] != method or c["target"] != target:
        raise SystemExit(f"{sid}: the post-#709 record ({c['case_id']}, {c['target']}, {c['requested_method']}) does not "
                         f"match the row ({case_id}, {target}, {method})")
    ib = f"{INF}/inference-batch-p1"
    src_j = f"{POST709}/inference-post709-results.json#{n}"
    src_r = f"{ib}/r-crosscheck/p1-route-probe-results.tsv#{n}"
    if c["kind"] == "route":
        r = rprobe[n]
        if r["pass"] != "TRUE" or c["outcome"] != "result":
            raise SystemExit(f"{sid}: a failed probe or a Julia error; not writing behaviour")
        r_label, j_label = r["actual"], f"{target}:{c['route_tag']}"
        ev = {"r": src_r, "julia": src_j}
        if method == "DEFAULT" and r["actual"].endswith(":profile"):
            return "not_bound", {
                "source_id": sid, "reason": "default",
                "text": (f"The row asks for the DEFAULT method. R's default for this parm is profile (probe route "
                         f"{r['actual']!r}); Julia's public default is Wald (confint(fit, y; parm) with no method gave a "
                         f"{c['route_tag']} result reporting method {c['result_method']!r}, {c['julia_call']}). Not the "
                         f"same default route."),
                "evidence": ev}
        if r["actual"].endswith(":profile") and target in ("communality", "rho", "proportion"):
            guards = [g for g, f in fixture.items() if f["stage"] == "guard" and f["parm"] == fx["parm"]
                      and f["method"] == "profile"]
            if len(guards) != 1 or "gllvmTMB_nonlinear_profile_withdrawn" not in rprobe[guards[0]]["actual_class"]:
                raise SystemExit(f"{sid}: expected one withdrawn guard row for parm {fx['parm']}, found {guards}")
            g = guards[0]
            ev["r"] = f"{src_r},{g}"
            return "not_bound", {
                "source_id": sid, "reason": "withdrawn",
                "text": (f"R's dispatcher forwards method profile to {r['actual'].split(':')[0]}, but that function raises "
                         f"gllvmTMB_nonlinear_profile_withdrawn at P1; the same probe observed that error on guard row {g} "
                         f"(parm {fx['parm']}, method profile). Julia's public confint returned a profile interval "
                         f"({c['julia_call']}; src/confint.jl:826, the penalty-based route #709 documents as exploratory). "
                         f"R refuses, Julia computes: not the same behaviour."),
                "evidence": ev}
        if c["result_method"] not in ("", "wald" if method == "DEFAULT" else method):
            raise SystemExit(f"{sid}: Julia reports method {c['result_method']!r} for a request of {method!r}")
        return "entry", {"case_id": case_id, "source_id": sid, "kind": "route", "r_observed": r_label,
                         "julia_observed": j_label, "requested_method": method, "julia_call": c["julia_call"],
                         "julia_result_method": c["result_method"], "r_source": src_r, "julia_source": src_j}
    # refusal: R's real refusal, Julia's ArgumentError, and a valid-method control on each side
    if c["outcome"] != "error" or c["error_type"] != "ArgumentError" or f":{method}" not in c["error_message"]:
        raise SystemExit(f"{sid}: Julia did not refuse {method!r} with an ArgumentError that names it")
    if c.get("control_outcome") != "result" or c.get("control_finite") is not True:
        raise SystemExit(f"{sid}: no valid-method control on the Julia side")
    r_label, r_src = r_refusal(target, method)
    parm_txt = re.search(r'parm="([^"]+)"', c["julia_call"]).group(1)
    if not r_label.strip():
        raise SystemExit(f"{sid}: empty R refusal label")
    if target == "lambda":
        if not r_label.startswith("'arg' should be one of"):
            raise SystemExit(f"{sid}: R's Lambda refusal is not the match.arg error: {r_label!r}")
    elif f'"{method}"' not in r_label:
        raise SystemExit(f"{sid}: R's refusal text does not name the method {method!r}: {r_label!r}")
    if f"parm {parm_txt};" not in c["error_message"]:
        raise SystemExit(f"{sid}: Julia's refusal does not name the parm {parm_txt!r}")
    if c.get("control_route_tag") not in ("wald_derived", "wald_packed") or c.get("control_result_method") not in ("wald", ""):
        raise SystemExit(f"{sid}: Julia's control is not a Wald result")
    if target == "lambda":
        w = post709()["r_lambda"]["wald"]
        if w["raised"] or not w["finite"]:
            raise SystemExit(f"{sid}: R's valid-method control (wald) did not return an interval")
        r_control = {"kind": "live_call", "call": "confint(fit, parm = \"Lambda\", method = \"wald\")",
                     "result": f"interval, {w['n_rows']} rows", "source": f"{POST709}/r-lambda-reject/r-oracle.json#calls.wald"}
        r_call = "confint(fit, parm = \"Lambda\", level = 0.95, method = <m>)"
    else:
        row = R_CONTROL_ROW[target]
        rc = rprobe[row]
        if rc["pass"] != "TRUE" or not rc["actual"].startswith(R_CONTROL_ENDPOINT[target] + ":") or \
                not (rc["actual"].endswith(":wald") or rc["actual"].endswith(":fisher-z")):
            raise SystemExit(f"{sid}: R control row {row} is {rc['actual']!r}, not a valid method at {R_CONTROL_ENDPOINT[target]}")
        r_control = {"kind": "route_probe_stub", "call": f"probe row {row}", "result": f"routes to {rc['actual']}",
                     "source": f"{ib}/r-crosscheck/p1-route-probe-results.tsv#{row}"}
        r_call = None
    entry = {"case_id": case_id, "source_id": sid, "kind": "refusal", "r_observed": r_label,
             "julia_observed": c["error_message"], "requested_method": method, "julia_call": c["julia_call"],
             "julia_error_type": c["error_type"],
             "r_control": r_control,
             "julia_control": {"call": c["control_call"], "result": f"{c['control_route_tag']} interval",
                               "source": src_j},
             "r_source": r_src, "julia_source": src_j}
    if r_call:
        entry["r_call"] = r_call
    return "entry", entry


def wave2(case_id, rec):
    ib = f"{INF}/inference-batch-p1"
    jres = {c["source_id"]: c for c in load(ROOT / ib / "inference-batch-results.json")["cases"]}
    rprobe = {r["id"]: r for r in read_tsv(ROOT / ib / "r-crosscheck/p1-route-probe-results.tsv")}
    fixture = {r["id"]: r for r in read_tsv(ROOT / FIXTURE)}
    target = target_of(case_id)
    entries, not_bound = [], []
    used_post709 = used_rulings = False
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
        rul = rulings_entry(case_id, sid, rprobe, fixture)
        if rul is not None:
            entries.append(rul[1])
            used_rulings = True
            continue
        p709 = post709_entry(case_id, sid, rprobe, fixture)
        if p709 is not None:
            (entries if p709[0] == "entry" else not_bound).append(p709[1])
            used_post709 = True
            continue
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
    if used_post709:
        rf.update(post709_reads())
    if used_rulings:
        rf.update(rulings_reads())
    return entries, not_bound, rf


def wave4(case_id, rec):
    rb = f"{INF}/inference-remainder-p1"
    ro, rj = load(ROOT / rb / "r-oracle.json"), load(ROOT / rb / "julia-results.json")
    fixture = {r["id"]: r for r in read_tsv(ROOT / FIXTURE)}
    target = target_of(case_id)
    key = {"icc": "icc_reject", "phylo_signal": "phylo_reject", "communality": "communality_reject",
           "rho": "rho_reject", "proportion": "proportion_reject"}[target]
    jc = rj["cases"][case_id]["methods"]
    rprobe = {r["id"]: r for r in read_tsv(ROOT / f"{INF}/inference-batch-p1/r-crosscheck/p1-route-probe-results.tsv")}
    entries, not_bound, used_post709 = [], [], False
    for sid in rec["source_ids"]:
        n = sid.split("/")[-1]
        method = fixture[n]["method"]
        if method not in ro[key] or method not in jc:
            raise SystemExit(f"{sid}: method {method} is not in the raw oracle for {case_id}")
        p709 = post709_entry(case_id, sid, rprobe, fixture)
        if p709 is not None:
            (entries if p709[0] == "entry" else not_bound).append(p709[1])
            used_post709 = True
            continue
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
    rf = reads(f"{rb}/r-oracle.json", f"{rb}/julia-results.json", FIXTURE,
               f"{INF}/inference-batch-p1/r-crosscheck/p1-route-probe-results.tsv")
    if used_post709:
        rf.update(post709_reads())
    return entries, not_bound, rf


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

FIRST7_SOURCE_IDS = {
    "postfit/POSTFIT-SURFACE-check_auto_residual",
    "isdm/ISDM-COUNT", "isdm/ISDM-EXTRA-SOURCE", "isdm/ISDM-MISSING-IN-TRAIT",
    "isdm/ISDM-MISSING-SOURCE", "isdm/ISDM-WRAPPER-LAW",
}


def first7(case_id, rec):
    """Derive the signed first-seven behaviour from raw public-call labels."""
    sids = rec.get("source_ids") or ([rec["source_id"]] if rec.get("source_id") else [])
    sid = next((s for s in sids if s in FIRST7_SOURCE_IDS), None)
    if sid is None:
        return [], [], {}
    p = ROOT / FIRST7 / (sid.split("/", 1)[1] + ".json")
    if not p.is_file():
        return [], [], {}
    raw = load(p)
    if raw.get("reference_commit") != P1_SHA or raw.get("source_id") != sid:
        raise SystemExit(f"{p}: source id or P1 pin mismatch")
    import first_seven_behaviour_derive as F7
    raw_dir = Path((raw.get("provenance") or {}).get("raw_directory", ""))
    if not raw_dir.parts or raw_dir.is_absolute() or ".." in raw_dir.parts:
        raise SystemExit(f"{p}: raw provenance must name a repo-relative directory")
    d = ROOT / raw_dir
    meta = load(d / "run.json")
    if meta != raw.get("provenance"):
        raise SystemExit(f"{p}: provenance differs from the retained run metadata")
    checked_meta = dict(meta, r_output_path=str(d / "r-public.tsv"), julia_output_path=str(d / "julia-public.tsv"))
    fresh, _ = F7.derive(F7.tsv(d / "r-public.tsv", "R"), F7.tsv(d / "julia-public.tsv", "Julia"), checked_meta)
    checked_meta.pop("r_output_path", None); checked_meta.pop("julia_output_path", None)
    if fresh[sid] != raw or raw.get("case_id") != case_id:
        raise SystemExit(f"{p}: sidecar differs from raw re-derivation or frozen case id")
    rf, jf = raw.get("r_observed"), raw.get("julia_observed")
    inputs = [p, d / "run.json", d / "r-public.tsv", d / "julia-public.tsv", ROOT / "tools/first_seven_behaviour_derive.py"]
    reads = {str(x.relative_to(ROOT)): sha(x) for x in inputs}
    if raw.get("verdict") != "PASS" or rf != jf:
        return [], [{"source_id": sid, "reason": "public_route_differs",
                     "text": f"P1 public-door labels differ: R={rf!r}; Julia={jf!r}. "
                             "The exact calls and raw outputs are in the cited receipt.",
                     "evidence": {"receipt": f"{FIRST7}/{p.name}"}}], reads
    kind = "error_class" if raw.get("raw_observations", {}).get("R", {}).get("outcome") == "ERROR" else "route"
    entry = {"case_id": case_id, "source_id": sid, "kind": kind,
             "r_observed": rf, "julia_observed": jf,
             "r_source": f"{FIRST7}/{p.name}#raw_observations.R",
             "julia_source": f"{FIRST7}/{p.name}#raw_observations.Julia",
             "r_call": raw.get("raw_observations", {}).get("R", {}).get("call"),
             "julia_call": raw.get("raw_observations", {}).get("Julia", {}).get("call"),
             "ruling": "D-319 N6/N10"}
    return [entry], [], reads


def derive_block(case_id, rec):
    """The behaviour block and not-bound list for one case receipt, or None when the receipt has no
    behavioural evidence to give (the numeric and structural cases)."""
    if rec.get("evidence_kind") == "public_door_behaviour" or any(
            s in FIRST7_SOURCE_IDS for s in (rec.get("source_ids") or [rec.get("source_id")])):
        entries, not_bound, rf = first7(case_id, rec)
        block = {"pin": "P1", "ruling": RULING, "read_from": rf, "cases": entries} if entries else None
        return block, not_bound
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
    # Add only the six owned public-door case receipts, and only once the
    # independently derived raw sidecar exists. This keeps pre-measurement
    # --check identical to the existing baseline.
    for sid in sorted(FIRST7_SOURCE_IDS):
        sidecar = ROOT / FIRST7 / (sid.split("/", 1)[1] + ".json")
        if not sidecar.is_file():
            continue
        folder = "postfit" if sid.startswith("postfit/") else "isdm"
        pdir = ROOT / LEDGER / "receipts" / folder / "cases"
        for p in sorted(pdir.glob("*.json")):
            rec = load(p)
            if sid in (rec.get("source_ids") or [rec.get("source_id")]):
                out[p.stem] = (p, rec)
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
    "public_route_differs": ("Not bound behaviourally: Julia's public confint does not take the route R takes for this "
                             "request (receipt behaviour_not_bound)."),
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


ROUTE_ONLY_ROWS = {"018", "025", "032", "039", "045", "048", "057", "060", "063"}
ROUTE_ONLY_NOTE = ("route only; estimands not compared (R aligned extract_* vs Julia communality(fit) etc.; n_boot "
                   "small)")


def overlay_row(row, counts):
    """Flip `row` to evidence_tier behavioural when its cited receipts' behaviour blocks bind it under the
    assembler's port of the checker's rule (frozen scope, class identity, entries scoped by source_id where a
    case id has several citers). Moves evidence.non_binding_receipts to evidence.receipt, sets the tier text,
    and nothing else. Updates `counts` (tier count down, `behavioural` up). Returns True if flipped.
    A row that is not flipped gets a one-line note when this tool records why (see _annotate)."""
    eligible = row.get("evidence_tier") in OVERLAY_TIERS or (
        row.get("source_id") in FIRST7_SOURCE_IDS and
        row.get("evidence_tier") == "needs_surface_r_side_measured")
    if not eligible or row.get("measured_against") != P1_SHA:
        return False
    paths = list((row.get("evidence") or {}).get("non_binding_receipts") or [])
    if not paths or not all((ROOT / p).is_file() for p in paths):
        return False
    asm = assembler()
    cand = dict(row, evidence_tier="behavioural", evidence={"receipt": paths})
    if asm.behavioural_receipt_problem(cand, ROOT, asm.behaviour_equivalence(ROOT), citers()) is not None:
        _annotate(row, paths)
        return False
    if str(row.get("source_id", "")).split("-")[-1] in ROUTE_ONLY_ROWS:
        _add_note(row, ROUTE_ONLY_NOTE)
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
        if "behaviour" in rec and rec.get("evidence_kind") not in DERIVE and not any(
                sid in FIRST7_SOURCE_IDS for sid in (rec.get("source_ids") or [rec.get("source_id")])):
            problems.append(f"{path.relative_to(ROOT)}: behaviour block on a receipt this tool does not derive")
    return problems


def write():
    EQUIV_PATH.write_text(dump(equivalence_doc()))
    n = 0
    for cid, (path, new) in derive_all().items():
        path.write_text(dump(new))
        n += 1
    print("equivalence classes", len(all_classes()), "receipts written", n)


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
