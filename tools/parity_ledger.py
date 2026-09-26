#!/usr/bin/env python3
"""Reconcile GLLVModels.jl's public export surface against R's gllvmTMB, at a named git ref.

Ported from DRM.jl's tools/parity_ledger.py (the drmTMB<->DRM.jl catch-up
countdown), adapted to the gllvmTMB<->GLLVModels.jl pair. Reads gllvmTMB's
`NAMESPACE` at a named git ref and GLLVModels.jl's own `export` block in
`src/GLLVModels.jl`, then reports BOTH directions:

  FORWARD -- R exports with no Julia twin (genuinely owed)
  REVERSE -- Julia exports with no R twin (genuinely ahead)

Each unmatched name is run through written, per-name (or per-pattern) reason
tables -- ALIASES (a twin exists under a different spelling), NOT_CAPABILITY
and DELIBERATELY_NOT_PORTED (forward: R-idiom helpers or accounted absences),
RENAMED_AWAY (forward: the Julia name is deliberately NOT the R name -- see
docs/dev-log/core070/api-rename-notes.md), and AHEAD_EXPLICIT / AHEAD_PATTERNS
(reverse: Julia-only names with a written class) -- so the countdown reports
"genuinely owed"/"genuinely ahead" separately from "accounted for in writing",
rather than summing them into one number that overstates the gap.

Always reads gllvmTMB through `git show <ref>:NAMESPACE` rather than the
working tree -- a working checkout can sit hundreds of commits behind
`origin/main` (VERIFIED 2026-09-02: the frozen 0.7.0 oracle NAMESPACE has 160
export() lines; origin/main has 168).

Value-add over the DRM.jl original: every FORWARD name is cross-checked
against docs/dev-log/core070/required-source-case-map.json's
`namespace/export/<name>` rows, so a forward gap that is already a tracked
ledger row (with a disposition) is distinguished from one that is UNTRACKED.

    python3 tools/parity_ledger.py
    python3 tools/parity_ledger.py --r-ref b4d5fee64def88bc768dda1f1f77c29b295edd86
    python3 tools/parity_ledger.py --ref origin/main   # live main comparison only
    python3 tools/parity_ledger.py --self-test

``--ref`` and ``--r-ref`` both pin the R ``NAMESPACE`` read via ``git show`` (never
the working tree). Default: frozen gllvmTMB 0.7.0 ``b4d5fee6``. Capability-status
CLOSURE (``gllvmTMB/tools/parity_ledger.R``) is a separate join — see
``tools/parity_oracle.py`` and ``tools/parity_capability_closure.sh``.
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from parity_oracle import CAPABILITY_LEDGER_REF, DEFAULT_R_REF, FROZEN_GLLVMTMB_ORACLE

DEFAULT_GLLVMTMB = "/Users/z3437171/Dropbox/Github Local/gllvmTMB"
DEFAULT_REF = DEFAULT_R_REF

# ---------------------------------------------------------------------------
# FORWARD direction: R name -> Julia symbol, where the twin exists under a
# different name. Seeded from docs/src/gllvmtmb-parity.md
# "## R bridge: parameterization map" (the quantity/family-naming
# conventions the R<->Julia bridge already reconciles) plus obvious
# family-constructor renames documented in the same file.
#
# TWIN_ALIAS rows from docs/dev-log/core070/forward-export-disposition-20260915.tsv
# (after-task 2026-09-15-forward-export-disposition.md, PR #350): only entries
# where a exported Julia symbol already implements the capability.
# ---------------------------------------------------------------------------
ALIASES = {
    # parity-map row: NB2 dispersion phi (R) <-> r=1/phi (Julia); same family
    "nbinom2": "NBFit",
    "nbinom1": "NB1",
    "truncated_nbinom2": "TruncatedNegBin2",
    "truncated_poisson": "TruncatedPoisson",
    # parity-map row: Tweedie power nu (R) <-> p (Julia); same family
    "tweedie": "TweedieFit",
    # parity map + families list: R's `student()` <-> Julia's StudentT marker
    "student": "StudentT",
    "betabinomial": "BetaBinom",
    # "Ordinal and ordinal-probit bridge rows..." -- same fitter, link arg
    "ordinal_probit": "OrdinalFit",
    "cumulative_logit": "Ordinal",
    # the fitting verb
    "gllvmTMB": "fit_gllvm",
    "gllvm_julia_fit": "bridge_fit",
    # --- TWIN_ALIAS (forward-export disposition 2026-09-15) ---
    "dep": "fit_dep_gllvm",
    "Beta": "fit_beta_gllvm",  # Distributions Beta() marker not re-exported; fitter is the twin
    "kernel_indep": "fit_kernel_indep_gllvm",
    "kernel_dep": "fit_kernel_dep_gllvm",
    "kernel_latent": "fit_kernel_latent_gllvm",
    "kernel_scalar": "fit_kernel_indep_gllvm",  # scalar modifier = indep(..., common=TRUE)
    "kernel_unique": "fit_kernel_latent_gllvm",  # unique modifier on latent tier
    "animal_dep": "fit_animal_dep_gllvm",
    "animal_latent": "fit_animal_latent_gllvm",
    "extract_Sigma_B": "extract_Sigma",  # level=:unit, part=:total
    "extract_Sigma_W": "extract_Sigma",  # level=:unit_obs, part=:total
    "gllvmTMB_wide": "gllvm",  # wide-matrix @formula entry; R soft-deprecated wrapper name
    ".proportions_bootstrap_ci": "extract_proportions",  # R internal; user-layer proportions twin
    ".proportions_wald_ci": "extract_proportions",
    "flag_unreliable_loadings": "check_gllvmTMB",  # loading-runaway / Heywood flags in diagnose cluster
}

# TWIN_ALIAS rows intentionally omitted from ALIASES (#350/#355 after-task): no single Julia
# export target yet — stay in FORWARD until maintainer picks one symbol or thin wrapper.
#   animal_indep, animal_scalar — relatedness_cov + indep Gaussian; scope-limited on R side
#   extract_residual_split — split across extract_residual_cov / link_residual (structural twin)
TWIN_ALIAS_DEFERRED = frozenset({"animal_indep", "animal_scalar", "extract_residual_split"})

# FORWARD, R-idiom helpers with no meaningful Julia counterpart: control-object
# constructors, screen/meta helpers with no ported analogue, gradient plumbing.
# Mirrors DRM.jl's NOT_CAPABILITY entries "drm_control"/"gr"/"meta_known_V"
# one-for-one where gllvmTMB carries the same concept.
NOT_CAPABILITY = {
    "gr": "R gradient/generic helper; not a distinct Julia-facing capability",
    "gllvmTMBcontrol": "R control-object constructor (fit-option bag); Julia takes options as keyword args, no matching constructor export",
    "screen_control": "R control-object constructor for screen_gllvmTMB(); same shape as gllvmTMBcontrol, no Julia analogue",
    "meta_known_V": "meta-analysis known-V helper; not a GLLVModels engine capability (DRM.jl carries the identical exclusion for its own meta_known_V)",
}

# FORWARD, R name -> Julia's *deliberately different* name. Source:
# docs/dev-log/core070/api-rename-notes.md (maintainer decision, round2-3
# item #5): these 6 renames free the plain R-matching name for a future TRUE
# mirror, because the current Julia function computes a different quantity /
# has a different call shape than R's function of the same old name. A row
# here is NOT a covered twin -- it documents why the obvious alias is wrong.
RENAMED_AWAY = {
    "getREsd": ("latent_score_sd",
                "R's getREsd(fit, block=) covers auxiliary RE blocks; Julia's "
                "computes latent factor-score conditional SDs instead -- "
                "different quantity (api-rename-notes.md)"),
    "compare_Sigma_table": ("compare_fits_Sigma_table",
                "R compares a fit against a supplied ground-truth matrix; "
                "Julia's is a two-fit bridge -- different signature (api-rename-notes.md)"),
    "compare_dep_vs_two_psi": ("compare_fits_dep_vs_two_psi",
                "R refits an alternative phylogenetic two-psi model internally; "
                "Julia's is a generic two-fit bridge -- different model class (api-rename-notes.md)"),
    "compare_indep_vs_two_psi": ("compare_fits_indep_vs_two_psi",
                "indep counterpart of compare_dep_vs_two_psi; same two-psi mismatch (api-rename-notes.md)"),
    "diagnostic_table": ("fit_diagnostic_table",
                "R requires x to already carry gllvmTMB_diagnostic metadata from a "
                "prior call; Julia takes the raw fit and computes everything -- "
                "different call shape (api-rename-notes.md)"),
    "profile_targets": ("profile_curve_targets",
                "renamed alongside the diagnostics.jl round2-3 pass (api-rename-notes.md)"),
}

# FORWARD, genuinely absent but accounted for in writing (not owed, and why).
# Kept short and conservative: only claims this pass can actually ground.
DELIBERATELY_NOT_PORTED = {
    "make_mesh": "R-side geospatial mesh prep (sf, CRS) before any fit; SPDE fitters take a mesh/precision Julia already has, not this constructor",
    "get_crs": "R-side CRS/projection accessor for geospatial prep; no fitting-engine analogue",
    "add_utm_columns": "R-side coordinate-projection convenience for geospatial prep; no fitting-engine analogue",
    "impute_model": "missing-data imputation-model surface; structurally separate from GLLVModels.jl's Laplace/VA family fitters",
    "imputed": "missing-data surface; same as impute_model",
    "categorical": "an imputation family (categorical missingness), not a response family; same missing-data surface as impute_model",
    "miss_control": "missing-data control-object constructor; same missing-data surface as impute_model",
}


# ---------------------------------------------------------------------------
# REVERSE direction: Julia exports with no R twin. AHEAD_EXPLICIT carries
# hand-written per-name reasons (mirrors StatsAPI/Base generics or a small
# number of named structs, exactly as DRM.jl's AHEAD_ACCOUNTED does).
# AHEAD_PATTERNS classifies the much larger Julia-only surface (400+ exports
# vs. gllvmTMB's 160) by regex, each with a written class -- a Julia package
# this size cannot get one bespoke sentence per export within this pass, so
# the class itself is the written reason, applied uniformly and disclosed as
# a design choice in the run's markdown record.
# ---------------------------------------------------------------------------
AHEAD_EXPLICIT = {
    "predict": "mirrors Base/StatsAPI predict; gllvmTMB reaches it via S3method(predict, gllvmTMB), never export()",
    "fitted": "mirrors stats::fitted; gllvmTMB reaches it via S3method(fitted, gllvmTMB), never export()",
    "residuals": "mirrors stats::residuals; gllvmTMB reaches it via S3method(residuals, gllvmTMB), never export()",
    "aic": "mirrors stats::AIC; gllvmTMB reaches it via S3method(AIC, gllvmTMB), never export()",
    "bic": "mirrors stats::BIC; gllvmTMB reaches it via S3method(BIC, gllvmTMB), never export()",
    "simulate": "mirrors stats::simulate; gllvmTMB reaches it via S3method(simulate, gllvmTMB), never export()",
    "coef": "mirrors stats::coef; gllvmTMB reaches it via S3method(coef, gllvmTMB), never export()",
    "vcov": "mirrors stats::vcov; gllvmTMB reaches it via S3method(vcov, gllvmTMB), never export()",
    "nobs": "mirrors stats::nobs; gllvmTMB reaches it via S3method(nobs, gllvmTMB), never export()",
    "dof": "StatsAPI naming for a fixed-effect parameter count; gllvmTMB computes this internally for AIC/BIC without exposing an accessor",
    "loglikelihood": "StatsAPI naming for stats::logLik; gllvmTMB reaches it via S3method(logLik, gllvmTMB), never export()",
    "stderror": "StatsAPI generic for per-parameter SEs; gllvmTMB surfaces the same values through printed summary()/coeftable(), not a queryable function",
    "coeftable": "StatsAPI generic for the coefficient table; gllvmTMB prints the same via summary.gllvmTMB, no separate accessor",
    "deviance": "mirrors stats::deviance; gllvmTMB reaches it via S3method(deviance, gllvmTMB), never export()",
    "tidy": "broom-style generic; gllvmTMB has no broom method, this is a Julia-ecosystem convenience",
    "@formula": "StatsModels macro mirroring R's built-in ~ formula literal; base R syntax needs no export",
    "StatsAPI": "re-exported package name (Julia convention for extending a shared interface), not a gllvmTMB-comparable symbol",
}

AHEAD_PATTERNS = [
    (re.compile(r"Fit$"), "struct suffix: backs a fitted model; R represents the same as an S3 class tag, never a matching export"),
    (re.compile(r"marginal_loglik"), "internal marginal-likelihood kernel; reached only from within the fit driver, never an R-facing name"),
    (re.compile(r"^fit_"), "Julia fitting-verb entry point for one family/structure; R dispatches the same concept through gllvmTMB()'s family= argument, not a separate export per family"),
    (re.compile(r"^confint(_|$)"), "CI-machinery entry point (Wald/profile/bootstrap); gllvmTMB reaches CIs via S3method(confint, gllvmTMB), never a family of separate exports"),
    (re.compile(r"^extract_"), "named accessor for a value gllvmTMB exposes as a raw fitted-object field or printed summary() text, not a separate function"),
    (re.compile(r"Link$"), "Julia link-function marker type; R represents the same link as a string argument (e.g. link=\"logit\"), no type export"),
    (re.compile(r"_grad(!)?$"), "hand-coded analytic-gradient kernel; engine internal, never reached by an R-facing name"),
    (re.compile(r"_logpdf$|_logz$|_cdf$"), "distribution-kernel helper (log-density/normalizer/CDF); engine internal"),
    (re.compile(r"_wald_ci$|_ci$"), "named Wald/profile CI accessor for one derived quantity; gllvmTMB reaches CIs generically via confint(), not per-quantity exports"),
    (re.compile(r"^em_|_squarem"), "EM/SQUAREM alternative-solver internals; gllvmTMB's TMB path never uses this solver family"),
    (re.compile(r"^spde_|^Q_|Precision$|precision$"), "SPDE/Matern spatial substrate; a GLLVModels.jl capability gllvmTMB's TMB template does not implement (per docs/src/gllvmtmb-parity.md \"Honest gaps\")"),
    (re.compile(r"^phylo_|^augmented_|^felsenstein|Contrasts$|^EdgePhy$|^edge_|^branch_|^Branch|^clade_|^blup"), "phylogenetic engine substrate (sparse/contrasts/edge-incidence); a GLLVModels.jl capability with no gllvmTMB analogue"),
    (re.compile(r"^coevolution|^Coevo|^make_cross_kernel"), "coevolution/cross-kernel substrate; a GLLVModels.jl capability with no gllvmTMB analogue"),
]

# ---------------------------------------------------------------------------
# REVERSE, PROPOSED 2026-09-25 -- per-name classes for the 91 names that were
# 'genuinely ahead, unclassified' (clause C6 of the true-parity programme,
# docs/dev-log/core070/true-parity-decision-map.md). PROPOSED BY AN AGENT PASS,
# NOT SIGNED by the maintainer -- full class/reason table, counts, and the D8
# hand-off drafts for the three Julia-first PORT capabilities are in
# docs/dev-log/core070/reverse-gap-classes-2026-09-25.md. Each reason below
# carries a [PROPOSED ...] tag naming that doc so a reader lands on the
# banner, not a silently-adopted claim. Merged into AHEAD_EXPLICIT (not a
# separate dict) so classify_ahead()'s logic needs no change.
# ---------------------------------------------------------------------------
AHEAD_EXPLICIT.update({
    "AnBSparseSolver": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only phylogenetic engine substrate; same class as the existing ^phylo_/^edge_/^branch_/^clade_ AHEAD_PATTERNS bucket, but this name's shape doesn't match those regexes -- a GLLVModels.jl capability with no gllvmTMB analogue",
    "AugmentedPhy": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only phylogenetic engine substrate; same class as the existing ^phylo_/^edge_/^branch_/^clade_ AHEAD_PATTERNS bucket, but this name's shape doesn't match those regexes -- a GLLVModels.jl capability with no gllvmTMB analogue",
    "BetaHurdle": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only family: Beta-hurdle two-part family (Bernoulli occurrence x positive Beta, Ferrari & Cribari-Neto parameterisation) for [0,1) proportion data; gllvmTMB has no hurdle-beta family",
    "COMPoisson": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only family: Conway-Maxwell-Poisson, a two-parameter count family handling BOTH over- and under-dispersion; in-repo comment states explicitly this is 'a capability gllvmTMB does NOT offer'",
    "CVResult": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only diagnostic or extractor: result struct returned by cv_gllvm() (see below) -- paired entry",
    "GeneralizedPoisson1": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only family: Generalized-Poisson type-1 (Famoye/Consul-Jain) with signed scalar dispersion (over- or under-dispersion); gllvmTMB has no GP-1 family",
    "GllvmCoefTable": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only diagnostic or extractor: result struct returned by coef_table() (see below) -- paired entry",
    "GllvmModel": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] internal-but-exported helper: immutable dimension-bookkeeping spec (p, K, K_W, has_diag, K_phy, has_phy_unique) for a Gaussian GLLVM (fit.jl); internal dispatch/construction type, no R analogue -- candidate to unexport",
    "GllvmSummary": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only diagnostic or extractor: result struct backing summary()'s tabular output (postfit_tables.jl); gllvmTMB's summary.gllvmTMB prints the same information without a separate exported class",
    "GroupingTerm": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] internal-but-exported helper: formula-grammar term type for the unit/unit_obs/cluster/cluster2 grouping grammar (grouped_fit.jl); gllvmTMB expresses the identical grammar (its own 6x3 keyword grid) as formula tokens, not an exported class -- candidate to unexport if callers always go through fit_gllvm(..., grouping=...)",
    "HurdleNB": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only family: Hurdle-NB2 two-part family (Bernoulli occurrence x zero-truncated NB2 count, shared dispersion r); gllvmTMB has no hurdle-NB family",
    "HurdlePoisson": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only family: Hurdle-Poisson two-part family (Bernoulli occurrence x zero-truncated Poisson count); gllvmTMB has no hurdle-Poisson family",
    "LVSelection": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] R has it after 0.7.0: result struct returned by select_lv(), which R gained post-0.7.0 (export(select_lv) on origin/main, not in the frozen b4d5fee6 NAMESPACE); R exposes the same sweep information via a printed object, not a separate exported class",
    "NodePerSpecies": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only phylogenetic engine substrate; same class as the existing ^phylo_/^edge_/^branch_/^clade_ AHEAD_PATTERNS bucket, but this name's shape doesn't match those regexes -- a GLLVModels.jl capability with no gllvmTMB analogue",
    "OrderedBeta": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only family: ordered-beta family (Kubinec 2023) for [0,1] proportions with point masses at 0 and 1; gllvmTMB has no ordered-beta family",
    "PrecisionPhy": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only phylogenetic engine substrate; same class as the existing ^phylo_/^edge_/^branch_/^clade_ AHEAD_PATTERNS bucket, but this name's shape doesn't match those regexes -- a GLLVModels.jl capability with no gllvmTMB analogue",
    "SourceCovariance": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] Julia-first capability awaiting an R decision: fixed source-covariance grammar term (source_fit.jl, fit_gaussian_sources) for per-source observation-model structure; gllvmTMB has FILED this as a known gap (gllvmTMB issue #941, 'Integrated GLLVM: multisource biodiversity data (GBIF + literature) sharing one ecological latent process', which specs a source-specific observation process per (cell, species, source) alongside one shared ecological latent process), so this is a genuine Julia-first capability the R side has flagged wanting, not yet a PORT decision",
    "StudentTFamily": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] R has it under another name: `const StudentT = StudentTFamily` (families/studentt.jl) -- StudentT is already ALIASES-matched to R's student(); StudentTFamily is the same type exported under its defining name too, not an additional capability",
    "TwoLevelRepeatabilityProfileWithdrawn": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] R has it under another name: mirrors R's extract_repeatability(method=\"profile\") abort class gllvmTMB_repeatability_profile_withdrawn (twolevel.jl comment, verbatim); an intentional-refusal exception type, not a capability gap",
    "ZIB": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] R has it after 0.7.0, FLAGGED: export(zi_binomial) is on origin/main, and norm() name-matches; but bridge.jl's own comments call this route 'Julia-forward / twin-asymmetric' with 'no twin light RCall Delta' -- Julia's ZIB is a shared-z two-part Newton-scoring family (separate gamma_z/gamma_c, complete-response fixed-effect-X routes) vs R's per-trait intercept-only zero-part Laplace family. Needs maintainer confirmation whether this is a genuine twin or a documented parameterization divergence",
    "ZINegBin": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] R has it after 0.7.0, FLAGGED: export(zi_nbinom2) is on origin/main, and norm() name-matches; but bridge.jl's own comments call this route 'Julia-forward / twin-asymmetric' with 'no twin light RCall Delta' -- Julia's ZINegBin is a shared-z two-part Newton-scoring family (separate gamma_z/gamma_c, complete-response fixed-effect-X routes) vs R's per-trait intercept-only zero-part Laplace family. Needs maintainer confirmation whether this is a genuine twin or a documented parameterization divergence",
    "ZIPoisson": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] R has it after 0.7.0, FLAGGED: export(zi_poisson) is on origin/main, and norm() name-matches; but bridge.jl's own comments call this route 'Julia-forward / twin-asymmetric' with 'no twin light RCall Delta' -- Julia's ZIPoisson is a shared-z two-part Newton-scoring family (separate gamma_z/gamma_c, complete-response fixed-effect-X routes) vs R's per-trait intercept-only zero-part Laplace family. Needs maintainer confirmation whether this is a genuine twin or a documented parameterization divergence",
    "admit_phylo_precision_payload": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only phylogenetic engine substrate; same class as the existing ^phylo_/^edge_/^branch_/^clade_ AHEAD_PATTERNS bucket, but this name's shape doesn't match those regexes -- a GLLVModels.jl capability with no gllvmTMB analogue",
    "bridge_capabilities": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] internal-but-exported helper: introspection function reporting the RCall/JuliaCall bridge's OWN capability surface as a NamedTuple (bridge.jl); a bridge self-description/testing tool, not a modelling capability",
    "build_AnB_sparse": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only phylogenetic engine substrate; same class as the existing ^phylo_/^edge_/^branch_/^clade_ AHEAD_PATTERNS bucket, but this name's shape doesn't match those regexes -- a GLLVModels.jl capability with no gllvmTMB analogue",
    "build_node_perspecies": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only phylogenetic engine substrate; same class as the existing ^phylo_/^edge_/^branch_/^clade_ AHEAD_PATTERNS bucket, but this name's shape doesn't match those regexes -- a GLLVModels.jl capability with no gllvmTMB analogue",
    "chibar2_pvalue": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] R has it after 0.7.0: export(chibar2_pvalue) is on origin/main (not the frozen 0.7.0 NAMESPACE) -- the boundary-inference chi-bar-square p-value accessor now has a direct R twin; the frozen-oracle comparator this countdown runs against predates it",
    "coef_table": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only diagnostic or extractor: coefficient-table accessor (summary_table.jl); gllvmTMB prints the same table via summary.gllvmTMB, not a separately exported function of this name (same reasoning as the existing AHEAD_EXPLICIT 'coeftable' StatsAPI-generic entry, applied to this differently-named variant)",
    "communality": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only diagnostic or extractor: short-name duplicate of extract_communality(), which already has a matched R twin (export(extract_communality) both sides); this bare form is Julia-ecosystem convenience naming, not an additional capability",
    "communality_B": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only diagnostic or extractor: between/within-tier variant of communality() (twolevel.jl), which is itself a short-name duplicate of an R-twinned extract_communality(); a TwoLevelFit-specific decomposition, not an additional capability beyond the two-level model",
    "communality_W": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only diagnostic or extractor: between/within-tier variant of communality() (twolevel.jl), which is itself a short-name duplicate of an R-twinned extract_communality(); a TwoLevelFit-specific decomposition, not an additional capability beyond the two-level model",
    "compare_fits_Sigma_table": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] R has it under another name (api-rename-notes.md, RENAMED_AWAY['compare_Sigma_table']): deliberately renamed because R's compare_Sigma_table(x, truth, ...) compares a fit against a supplied ground-truth matrix, while Julia's is a two-fit bridge -- different signature, same family of capability",
    "compare_fits_dep_vs_two_psi": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] R has it under another name (api-rename-notes.md, RENAMED_AWAY['compare_dep_vs_two_psi']): R refits an alternative phylogenetic two-psi model internally; Julia's is a generic two-fit AIC/BIC/Sigma_y bridge applicable to any two fits -- different model class, same family of capability",
    "compare_fits_indep_vs_two_psi": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] R has it under another name (api-rename-notes.md, RENAMED_AWAY['compare_indep_vs_two_psi']): indep counterpart of compare_fits_dep_vs_two_psi; same two-psi naming mismatch",
    "contrast_transform": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only phylogenetic engine substrate; same class as the existing ^phylo_/^edge_/^branch_/^clade_ AHEAD_PATTERNS bucket, but this name's shape doesn't match those regexes -- a GLLVModels.jl capability with no gllvmTMB analogue",
    "correlation": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only diagnostic or extractor: short-name duplicate of extract_correlation(), which already has a matched R twin (export(extract_correlation) both sides); this bare form is Julia-ecosystem convenience naming, not an additional capability",
    "correlation_B": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only diagnostic or extractor: between/within-tier variant of correlation() (twolevel.jl), which is itself a short-name duplicate of an R-twinned extract_correlation(); a TwoLevelFit-specific decomposition, not an additional capability beyond the two-level model",
    "correlation_W": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only diagnostic or extractor: between/within-tier variant of correlation() (twolevel.jl), which is itself a short-name duplicate of an R-twinned extract_correlation(); a TwoLevelFit-specific decomposition, not an additional capability beyond the two-level model",
    "cv_gllvm": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] Julia-first capability awaiting an R decision: K-fold-style cross-validation workflow tool (cv.jl); no R export or capability-status row exists. Comparable in nature to select_lv/LVSelection above, which R has since ported -- flagged as a candidate for the same treatment",
    "estep_edge_moments": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only phylogenetic engine substrate; same class as the existing ^phylo_/^edge_/^branch_/^clade_ AHEAD_PATTERNS bucket, but this name's shape doesn't match those regexes -- a GLLVModels.jl capability with no gllvmTMB analogue",
    "excess_kurtosis": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] internal-but-exported helper: generic descriptive-statistics helper (phylo_branch_re.jl) used to validate simulated phylogenetic branch-rate distributions; not a GLLVM capability itself -- candidate to unexport",
    "find_clade_root": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only phylogenetic engine substrate; same class as the existing ^phylo_/^edge_/^branch_/^clade_ AHEAD_PATTERNS bucket, but this name's shape doesn't match those regexes -- a GLLVModels.jl capability with no gllvmTMB analogue",
    "gaussian_grouped_intercept_loglik": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only diagnostic or extractor: named log-likelihood kernel for the Gaussian grouped-intercept fit (fit_random_effects.jl); gllvmTMB reaches the same grammar via gllvmTMB()'s grouping= argument and logLik(), not a per-quantity export (same reasoning as the existing ^confint_ AHEAD_PATTERNS bucket, applied by name here since this isn't confint-prefixed)",
    "gaussian_reml_loglik": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only diagnostic or extractor: REML log-likelihood kernel (reml.jl); gllvmTMB's own non-Gaussian REML route (allow_nongaussian_reml) is an internal control knob, not an exported accessor of this shape",
    "grad_node_perspecies": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only phylogenetic engine substrate; same class as the existing ^phylo_/^edge_/^branch_/^clade_ AHEAD_PATTERNS bucket, but this name's shape doesn't match those regexes -- a GLLVModels.jl capability with no gllvmTMB analogue",
    "grouped_gaussian_intervals": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only diagnostic or extractor: named Wald/profile/bootstrap CI or profile-curve accessor for one derived grouped/phylo/precision quantity; gllvmTMB reaches CIs generically via confint()/its own grouping grammar, not per-quantity exports (same reasoning as the existing ^confint_/_wald_ci$ AHEAD_PATTERNS buckets, applied by name here since these don't match those regexes)",
    "grouped_gaussian_variance_profile": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only diagnostic or extractor: named Wald/profile/bootstrap CI or profile-curve accessor for one derived grouped/phylo/precision quantity; gllvmTMB reaches CIs generically via confint()/its own grouping grammar, not per-quantity exports (same reasoning as the existing ^confint_/_wald_ci$ AHEAD_PATTERNS buckets, applied by name here since these don't match those regexes)",
    "grouped_nongaussian_intervals": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only diagnostic or extractor: named Wald/profile/bootstrap CI or profile-curve accessor for one derived grouped/phylo/precision quantity; gllvmTMB reaches CIs generically via confint()/its own grouping grammar, not per-quantity exports (same reasoning as the existing ^confint_/_wald_ci$ AHEAD_PATTERNS buckets, applied by name here since these don't match those regexes)",
    "joint_phylo_grouped_intervals": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only diagnostic or extractor: named Wald/profile/bootstrap CI or profile-curve accessor for one derived grouped/phylo/precision quantity; gllvmTMB reaches CIs generically via confint()/its own grouping grammar, not per-quantity exports (same reasoning as the existing ^confint_/_wald_ci$ AHEAD_PATTERNS buckets, applied by name here since these don't match those regexes)",
    "latent_score_sd": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] R has it under another name (api-rename-notes.md, RENAMED_AWAY['getREsd']): R's getREsd(fit, block=) covers auxiliary RE blocks; Julia's computes latent factor-score conditional SDs instead -- a deliberately different quantity under a deliberately different name",
    "link_residual": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] R has it under another name, DEFERRED: the tool's own TWIN_ALIAS_DEFERRED comment already documents this -- R's extract_residual_cov splits across Julia's extract_residual_cov + link_residual (a structural twin), but 'no single Julia export target yet -- stay in FORWARD until maintainer picks one symbol or thin wrapper'; not a plain gap, a pending naming decision",
    "loading_profile_exploratory": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only diagnostic or extractor: named Wald/profile/bootstrap CI or profile-curve accessor for one derived grouped/phylo/precision quantity; gllvmTMB reaches CIs generically via confint()/its own grouping grammar, not per-quantity exports (same reasoning as the existing ^confint_/_wald_ci$ AHEAD_PATTERNS buckets, applied by name here since these don't match those regexes)",
    "log_det_Q": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only phylogenetic engine substrate; same class as the existing ^phylo_/^edge_/^branch_/^clade_ AHEAD_PATTERNS bucket, but this name's shape doesn't match those regexes -- a GLLVModels.jl capability with no gllvmTMB analogue",
    "lognormal_response_mean": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only diagnostic or extractor: response-scale mean accessor for the already-twinned lognormal family (lognormal.jl, twin family_id 3); gllvmTMB reports the same quantity via predict(type=\"response\"), not a separately named function",
    "lv_effects": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only diagnostic or extractor: literal one-line alias (`lv_effects(fit) = extract_lv_effects(fit)`, postfit.jl) of extract_lv_effects(), which already has a matched R twin (export(extract_lv_effects) both sides)",
    "make_phy": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only phylogenetic engine substrate; same class as the existing ^phylo_/^edge_/^branch_/^clade_ AHEAD_PATTERNS bucket, but this name's shape doesn't match those regexes -- a GLLVModels.jl capability with no gllvmTMB analogue",
    "matern_correlation": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only spatial engine substrate: same reasoning as the existing ^spde_/Precision$ AHEAD_PATTERNS bucket (SPDE/Matern spatial substrate a GLLVModels.jl capability gllvmTMB's TMB template does not implement), applied here since the name doesn't match those regexes. NOTE: gllvmTMB's own CLAUDE.md now lists spatial_indep/spatial_dep/spatial_latent/spatial_coef/spatial_slope as live keywords, so the AHEAD_PATTERNS bucket's premise may be stale -- worth a maintainer check, out of scope for this pass",
    "node_blups": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only phylogenetic engine substrate; same class as the existing ^phylo_/^edge_/^branch_/^clade_ AHEAD_PATTERNS bucket, but this name's shape doesn't match those regexes -- a GLLVModels.jl capability with no gllvmTMB analogue",
    "node_dσ_phy_only": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only phylogenetic engine substrate; same class as the existing ^phylo_/^edge_/^branch_/^clade_ AHEAD_PATTERNS bucket, but this name's shape doesn't match those regexes -- a GLLVModels.jl capability with no gllvmTMB analogue",
    "observed_mask": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] internal-but-exported helper: one-line utility building a not-missing mask from Y (`observed_mask(Y) = .!ismissing.(Y)`, laplace.jl); trivial internal helper reused across masked-likelihood routes -- candidate to unexport",
    "ordination": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only diagnostic or extractor: short-name duplicate of extract_ordination(), which already has a matched R twin (export(extract_ordination) both sides); same convenience-naming pattern as communality()/correlation() above",
    "ordination_uncertainty": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] R has it after 0.7.0: export(ordination_uncertainty) is on origin/main (not the frozen 0.7.0 NAMESPACE); R's S3 print.gllvmTMB_ordination_uncertainty landed alongside it",
    "path_membership": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only phylogenetic engine substrate; same class as the existing ^phylo_/^edge_/^branch_/^clade_ AHEAD_PATTERNS bucket, but this name's shape doesn't match those regexes -- a GLLVModels.jl capability with no gllvmTMB analogue",
    "ppca_init": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] internal-but-exported helper: probabilistic-PCA warm-start initializer for latent scores/loadings (ppca_init.jl); an internal optimizer warm-start utility, not a user-facing capability -- candidate to unexport",
    "precision_logdet_check": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only phylogenetic engine substrate; same class as the existing ^phylo_/^edge_/^branch_/^clade_ AHEAD_PATTERNS bucket, but this name's shape doesn't match those regexes -- a GLLVModels.jl capability with no gllvmTMB analogue",
    "precision_multivariate_intervals": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only diagnostic or extractor: named Wald/profile/bootstrap CI or profile-curve accessor for one derived grouped/phylo/precision quantity; gllvmTMB reaches CIs generically via confint()/its own grouping grammar, not per-quantity exports (same reasoning as the existing ^confint_/_wald_ci$ AHEAD_PATTERNS buckets, applied by name here since these don't match those regexes)",
    "predict_spatial": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only spatial engine substrate: postfit prediction for the SPDE latent-field fit (spde_latent_postfit.jl); R's spatial_* keywords route through the generic predict() S3 method (already AHEAD_EXPLICIT) rather than a separately named function of this shape",
    "profile_ci_variance": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only diagnostic or extractor: named Wald/profile/bootstrap CI or profile-curve accessor for one derived grouped/phylo/precision quantity; gllvmTMB reaches CIs generically via confint()/its own grouping grammar, not per-quantity exports (same reasoning as the existing ^confint_/_wald_ci$ AHEAD_PATTERNS buckets, applied by name here since these don't match those regexes)",
    "profile_curve_targets": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] R has it under another name (api-rename-notes.md, RENAMED_AWAY['profile_targets']): renamed alongside the diagnostics.jl round2-3 pass; same underlying profile-curve capability",
    "qq_max_dev": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] internal-but-exported helper: generic QQ-plot max-deviation statistic (phylo_branch_re.jl), same validation-utility role as excess_kurtosis -- candidate to unexport",
    "random_balanced_tree": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only phylogenetic engine substrate; same class as the existing ^phylo_/^edge_/^branch_/^clade_ AHEAD_PATTERNS bucket, but this name's shape doesn't match those regexes -- a GLLVModels.jl capability with no gllvmTMB analogue",
    "rank_sum_z": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] internal-but-exported helper: generic rank-sum-test statistic (phylo_branch_re.jl), same validation-utility role as excess_kurtosis -- candidate to unexport",
    "relatedness_cov": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only phylogenetic engine substrate; same class as the existing ^phylo_/^edge_/^branch_/^clade_ AHEAD_PATTERNS bucket, but this name's shape doesn't match those regexes -- a GLLVModels.jl capability with no gllvmTMB analogue",
    "repeatability": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only diagnostic or extractor: short-name duplicate of extract_repeatability() (cross-referenced directly in its own docstring), which already has a matched R twin (export(extract_repeatability) both sides)",
    "rotation": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only diagnostic or extractor: `rotation(fit) = _svd_rotation(_loadings(fit))` (postfit.jl), the rotation-matrix component of ordination(); gllvmTMB exposes the same information via extract_ordination()/ordiplot(), not a separately exported function of this name",
    "row_effects": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only diagnostic or extractor: BLUP accessor for the random row-effect capability (row.eff=\"random\", row_random.jl -- a gllvm-package concept gllvmTMB itself does not implement); pairs with a Julia-only capability whose fitter is already covered by the ^_marginal_loglik AHEAD_PATTERNS bucket",
    "select_lv": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] R has it after 0.7.0: export(select_lv) is on origin/main (not the frozen 0.7.0 NAMESPACE); R's S3 print.gllvmTMB_select_lv landed alongside it",
    "shrink_logrates": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only phylogenetic engine substrate; same class as the existing ^phylo_/^edge_/^branch_/^clade_ AHEAD_PATTERNS bucket, but this name's shape doesn't match those regexes -- a GLLVModels.jl capability with no gllvmTMB analogue",
    "shrinkage_factor": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only phylogenetic engine substrate; same class as the existing ^phylo_/^edge_/^branch_/^clade_ AHEAD_PATTERNS bucket, but this name's shape doesn't match those regexes -- a GLLVModels.jl capability with no gllvmTMB analogue",
    "sigma_phy_dense": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only phylogenetic engine substrate; same class as the existing ^phylo_/^edge_/^branch_/^clade_ AHEAD_PATTERNS bucket, but this name's shape doesn't match those regexes -- a GLLVModels.jl capability with no gllvmTMB analogue",
    "sigma_phy_dense_edge": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only phylogenetic engine substrate; same class as the existing ^phylo_/^edge_/^branch_/^clade_ AHEAD_PATTERNS bucket, but this name's shape doesn't match those regexes -- a GLLVModels.jl capability with no gllvmTMB analogue",
    "sigma_y_site": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only diagnostic or extractor: named accessor for the implied site-level Sigma_y (confint_derived.jl / link_residual.jl); gllvmTMB exposes the same information via summary()/predict(), not a separately exported function of this name",
    "simulate_branch_re": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only phylogenetic engine substrate; same class as the existing ^phylo_/^edge_/^branch_/^clade_ AHEAD_PATTERNS bucket, but this name's shape doesn't match those regexes -- a GLLVModels.jl capability with no gllvmTMB analogue",
    "simulate_relaxed_bm": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only phylogenetic engine substrate; same class as the existing ^phylo_/^edge_/^branch_/^clade_ AHEAD_PATTERNS bucket, but this name's shape doesn't match those regexes -- a GLLVModels.jl capability with no gllvmTMB analogue",
    "solve_AnB": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only phylogenetic engine substrate; same class as the existing ^phylo_/^edge_/^branch_/^clade_ AHEAD_PATTERNS bucket, but this name's shape doesn't match those regexes -- a GLLVModels.jl capability with no gllvmTMB analogue",
    "solve_Q": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only phylogenetic engine substrate; same class as the existing ^phylo_/^edge_/^branch_/^clade_ AHEAD_PATTERNS bucket, but this name's shape doesn't match those regexes -- a GLLVModels.jl capability with no gllvmTMB analogue",
    "spatial_cov": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only spatial engine substrate: covariance-matrix builder for the SPDE/spatial substrate (structured_cov.jl); same caveat as matern_correlation above",
    "spearman": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] internal-but-exported helper: generic Spearman-correlation utility (phylo_branch_re.jl / relaxed_clock.jl), same validation-utility role as excess_kurtosis -- candidate to unexport",
    "transformed_wald_ci_derived": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] julia-only diagnostic or extractor: named Wald/profile/bootstrap CI or profile-curve accessor for one derived grouped/phylo/precision quantity; gllvmTMB reaches CIs generically via confint()/its own grouping grammar, not per-quantity exports (same reasoning as the existing ^confint_/_wald_ci$ AHEAD_PATTERNS buckets, applied by name here since these don't match those regexes)",
    "variance_lrt": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] R has it after 0.7.0: export(variance_lrt) is on origin/main (not the frozen 0.7.0 NAMESPACE)",
    "welch_t": "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] internal-but-exported helper: generic Welch two-sample t-test utility (phylo_branch_re.jl), same validation-utility role as excess_kurtosis -- candidate to unexport",
})


def git_show(repo: Path, ref: str, path: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), "show", f"{ref}:{path}"],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        sys.exit(f"cannot read {path} at {ref} in {repo}: {out.stderr.strip()}")
    return out.stdout


def norm(s: str) -> str:
    return re.sub(r"[_.]", "", s).lower()


def r_exports(repo: Path, ref: str) -> list[str]:
    ns = git_show(repo, ref, "NAMESPACE")
    return sorted(re.findall(r"^export\((.+?)\)", ns, re.M))


def julia_exports(root: Path) -> list[str]:
    """Parse export block(s) in src/GLLVModels.jl (checked to hold them all; no
    other included src/ file carries its own top-level `export` line as of
    this port -- verified by grep across src/)."""
    names: list[str] = []
    src_files = [root / "src" / "GLLVModels.jl"]
    for f in src_files:
        text = f.read_text()
        for block in re.findall(r"^export\s+(.+?)(?=\n\s*\n|\nexport|\Z)", text, re.M | re.S):
            block = re.sub(r"#.*", "", block)
            names += [n.strip() for n in block.split(",") if n.strip()]
    return sorted(set(names))


def load_required_case_map(root: Path) -> dict:
    path = root / "docs" / "dev-log" / "core070" / "required-source-case-map.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    rows = data.get("rows", [])
    return {row["source_id"]: row for row in rows if "source_id" in row}


def classify_ahead(name: str) -> str | None:
    if name in AHEAD_EXPLICIT:
        return AHEAD_EXPLICIT[name]
    for pattern, reason in AHEAD_PATTERNS:
        if pattern.search(name):
            return reason
    return None


def reconcile(r_names: list[str], j_names: list[str]):
    j_norm = {norm(x) for x in j_names}

    def has_twin(name: str) -> bool:
        if norm(name) in j_norm:
            return True
        alias = ALIASES.get(name)
        return bool(alias and norm(alias) in j_norm)

    unmatched = [x for x in r_names if not has_twin(x)]
    not_capability = [x for x in unmatched if x in NOT_CAPABILITY]
    renamed_away = [x for x in unmatched if x in RENAMED_AWAY]
    accounted = [x for x in unmatched
                 if x in DELIBERATELY_NOT_PORTED and x not in not_capability and x not in renamed_away]
    missing = [x for x in unmatched
               if x not in NOT_CAPABILITY and x not in RENAMED_AWAY and x not in DELIBERATELY_NOT_PORTED]

    r_norm = {norm(x) for x in r_names}
    alias_j_norm = {norm(v) for v in ALIASES.values()}

    def has_r_twin(name: str) -> bool:
        n = norm(name)
        return n in r_norm or n in alias_j_norm

    ahead_unmatched = [x for x in j_names if not has_r_twin(x)]
    ahead_classified = {x: classify_ahead(x) for x in ahead_unmatched}
    ahead_accounted = [x for x in ahead_unmatched if ahead_classified[x] is not None]
    ahead_missing = [x for x in ahead_unmatched if ahead_classified[x] is None]

    return {
        "missing": missing,
        "not_capability": not_capability,
        "renamed_away": renamed_away,
        "accounted": accounted,
        "ahead_missing": ahead_missing,
        "ahead_accounted": ahead_accounted,
        "ahead_classified": ahead_classified,
    }


def run(gllvmtmb: Path, ref: str, root: Path) -> int:
    sha = subprocess.run(
        ["git", "-C", str(gllvmtmb), "rev-parse", ref],
        capture_output=True, text=True,
    ).stdout.strip()
    desc = git_show(gllvmtmb, ref, "DESCRIPTION")
    m = re.search(r"^Version:\s*(\S+)", desc, re.M)
    version = m.group(1) if m else "?"

    r_names = r_exports(gllvmtmb, ref)
    j_names = julia_exports(root)
    result = reconcile(r_names, j_names)

    case_map = load_required_case_map(root)

    print(f"gllvmTMB {version} @ {ref} ({sha[:9] if sha else '?'})")
    print(f"  R exports: {len(r_names)}   GLLVModels.jl exports: {len(j_names)}")
    print()

    print(f"FORWARD -- RENAMED AWAY ({len(result['renamed_away'])}) -- "
          "the R name is deliberately NOT the Julia name (api-rename-notes.md)")
    for name in result["renamed_away"]:
        new_name, reason = RENAMED_AWAY[name]
        print(f"  {name:<24} -> {new_name:<28} {reason}")
    print()

    print(f"FORWARD -- NOT CAPABILITY ({len(result['not_capability'])}) -- R-idiom helper, no engine analogue")
    for name in result["not_capability"]:
        print(f"  {name:<24} {NOT_CAPABILITY[name]}")
    print()

    print(f"FORWARD -- ACCOUNTED FOR IN WRITING ({len(result['accounted'])}) -- not owed, and why")
    for name in result["accounted"]:
        print(f"  {name:<24} {DELIBERATELY_NOT_PORTED[name]}")
    print()

    print(f"FORWARD -- gllvmTMB EXPORTS WITH NO GLLVModels.jl TWIN ({len(result['missing'])}) -- genuinely owed")
    for name in result["missing"]:
        row = case_map.get(f"namespace/export/{name}")
        if row is None:
            status = "UNTRACKED"
        else:
            disp = row.get("disposition")
            cls = row.get("classification", "?")
            status = disp if disp else f"no disposition (classification={cls})"
        print(f"  {name:<28} {status}")
    print()

    print(f"REVERSE -- AHEAD OF gllvmTMB, ACCOUNTED FOR IN WRITING ({len(result['ahead_accounted'])}) -- not a gap, and why")
    for name in result["ahead_accounted"]:
        print(f"  {name:<32} {result['ahead_classified'][name]}")
    print()

    print(f"REVERSE -- GLLVModels.jl EXPORTS WITH NO gllvmTMB TWIN ({len(result['ahead_missing'])}) -- genuinely ahead, unclassified")
    for name in result["ahead_missing"]:
        print(f"  {name}")
    print()

    untracked = [x for x in result["missing"]
                 if f"namespace/export/{x}" not in case_map]

    print(f"COUNTDOWN: {len(result['missing'])} export gaps genuinely owed "
          f"({len(untracked)} UNTRACKED of those) · "
          f"{len(result['renamed_away'])} renamed away · "
          f"{len(result['not_capability'])} not-capability · "
          f"{len(result['accounted'])} accounted for · "
          f"{len(result['ahead_missing'])} genuinely ahead · "
          f"{len(result['ahead_accounted'])} ahead-accounted")

    print(f"FORWARD={len(result['missing'])} REVERSE={len(result['ahead_missing'])}")
    return 0


def self_test() -> int:
    """Synthetic NAMESPACE + export block in a temp dir; assert forward/reverse
    counts, then mutate one alias and assert the count changes (negative
    control -- proves the reconciliation logic actually discriminates, rather
    than always reporting the same numbers regardless of input)."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "r_repo"
        repo.mkdir()
        subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@t.t"], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.name", "t"], check=True)
        (repo / "NAMESPACE").write_text(
            "export(fit_thing)\n"
            "export(aliased_thing)\n"
            "export(r_only_thing)\n"
            "export(gr)\n"
        )
        (repo / "DESCRIPTION").write_text("Package: test\nVersion: 9.9.9\n")
        subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "init"], check=True)

        jroot = Path(tmp) / "j_repo"
        (jroot / "src").mkdir(parents=True)
        (jroot / "src" / "GLLVModels.jl").write_text(
            "module GLLVModels\nexport fit_thing, AliasedThingJl, julia_only_thing\n\nend\n"
        )

        # local test-only aliasing (does not touch module-level ALIASES)
        old_aliases = dict(ALIASES)
        old_not_cap = dict(NOT_CAPABILITY)
        ALIASES.clear()
        ALIASES.update({"aliased_thing": "AliasedThingJl"})
        NOT_CAPABILITY.clear()
        NOT_CAPABILITY.update({"gr": "test stand-in for R-idiom helper"})
        try:
            r_names = r_exports(repo, "HEAD")
            j_names = julia_exports(jroot)
            result = reconcile(r_names, j_names)
            # forward: r_only_thing is the only genuine gap (fit_thing twinned
            # directly, aliased_thing twinned via ALIASES, gr excluded via
            # NOT_CAPABILITY)
            assert result["missing"] == ["r_only_thing"], result["missing"]
            assert result["not_capability"] == ["gr"], result["not_capability"]
            # reverse: julia_only_thing is the only genuine Julia-ahead gap
            assert result["ahead_missing"] == ["julia_only_thing"], result["ahead_missing"]

            # negative control: mutate the alias so it no longer matches the
            # Julia name -- aliased_thing must now show up as a genuine forward
            # gap, proving has_twin() actually depends on ALIASES rather than
            # always reporting the same answer.
            ALIASES["aliased_thing"] = "SomethingElseEntirely"
            result2 = reconcile(r_names, j_names)
            assert "aliased_thing" in result2["missing"], result2["missing"]
            assert result2["missing"] != result["missing"]
        finally:
            ALIASES.clear()
            ALIASES.update(old_aliases)
            NOT_CAPABILITY.clear()
            NOT_CAPABILITY.update(old_not_cap)

    assert DEFAULT_REF == FROZEN_GLLVMTMB_ORACLE, DEFAULT_REF
    print("SELFTEST_OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export-surface parity: R NAMESPACE vs GLLVModels.jl exports at a git ref.",
        epilog=(
            f"Default R pin: frozen gllvmTMB 0.7.0 {FROZEN_GLLVMTMB_ORACLE[:8]}. "
            f"Capability CLOSURE uses {CAPABILITY_LEDGER_REF} via parity_capability_closure.sh."
        ),
    )
    ap.add_argument("--gllvmtmb", default=DEFAULT_GLLVMTMB, type=Path,
                     help="path to the gllvmTMB repo")
    ap.add_argument("--ref", default=DEFAULT_REF,
                     help=f"R NAMESPACE git ref (default: frozen oracle {FROZEN_GLLVMTMB_ORACLE[:8]})")
    ap.add_argument("--r-ref", default=None,
                     help="alias for --ref on the R NAMESPACE read (P13; same default as --ref)")
    ap.add_argument("--root", default=Path(__file__).resolve().parents[1], type=Path,
                     help="path to the GLLVModels.jl repo root")
    ap.add_argument("--self-test", action="store_true",
                     help="run the synthetic self-test instead of reading real repos")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    ref = args.r_ref if args.r_ref is not None else args.ref
    return run(args.gllvmtmb, ref, args.root)


if __name__ == "__main__":
    raise SystemExit(main())
