# C2 decision packet: `namespace/export/extract_phylo_signal`

Date: 2026-11-02. Base: `origin/main` at `16a7f08df78abb49e3faa8c9811d6f371e4d981d`.
R reference: gllvmTMB at pin `9539352f66f2db2cc26b1c393e67212a359b60c9` (read only).

Status: a packet for rulings. The row stays UNBOUND. Nothing here changes a row
status, the scoreboard, a receipt or the case map. The draft code is an
unmerged proposal on a local branch and has not been pushed.

## 1. The question in plain words

R's `extract_phylo_signal()` answers one question for a fitted phylogenetic
model: for each trait, what share of the species-level variance is
phylogenetic? Our row for it is marked done only as a name check. The receipt
confirms that R registers the name and that Julia has an exported function
called `phylo_signal`. It never runs R, never compares a number, and it names
a different Julia function from the one called `extract_phylo_signal`.

Julia has three code paths that touch this question, and they do not agree
with each other or with R:

1. The `phylo_latent` twin returns R's table, but for a bare phylogenetic fit
   the answer is the constant 1, so a function that always returns 1 passes.
2. The ordinary `GllvmFit` method returns a bare vector with a different
   denominator, on a different model class.
3. The joint phylogeny-plus-grouping Gaussian fit, which is the only Julia fit
   that can hold a species-level non-phylogenetic variance next to a
   phylogenetic one, has no method at all on main.

The decision asked of Shinichi is which of these gaps to close, in what order,
and whether the row may later be measured against R.

## 2. The contract text on main

All quotes are from `origin/main` at `16a7f08df`.

Scoreboard row (`docs/dev-log/core070/true-parity-latest/scoreboard.md:384`):
`namespace-export-extract_phylo_signal`, `required_core`, status
`REGISTRATION-ONLY`, `not bound`, `evidence_tier registration`,
`measured_against 9539352f66f2db2cc26b1c393e67212a359b60c9`.

Case map row (`case-map-assembled.json`, `namespace/export/extract_phylo_signal`):
"R export(extract_phylo_signal) registered and defined at P1; Julia
`phylo_signal` is an exported Function (measured). Name match only." R
location cited: `R/extract-omega.R:468`.

Case receipt `CORE070-NAMESPACE-EXTRACT-PHYLO-SIGNAL-NATIVE.json`, field
`meaning`: "Tier 0 registration check: the R export/S3 method is registered in
the pinned NAMESPACE and defined in the cited R file, and the named Julia symbol
is defined in GLLVModels (at P1 also exported and a Function). A name match only
(evidence_tier registration): semantics and numeric output are not compared, so
a PASS is not a twin." Its `julia_check.symbol` is `phylo_signal`, first method
`src/confint_derived.jl:311`, measured at Julia commit `c29c1c0b6d8d985e1edd96426edef2621402a03f`.
On current main that method is at `src/confint_derived.jl:342`.

Gate text (`docs/dev-log/core070/true-parity-latest/GATES.md:1088-1093`): "C2: every P1 capability inside the D-295
boundary ... has a scoreboard row with a resolving P1 receipt (a row at
`EVIDENCED-BEHAVIOURAL` counts as done only inside the frozen behavioural list)".

Measured by `node tools/true_parity_check.mjs C2` at `16a7f08df`: rows 297, done
290, C2_NOT_MET. This row is one of seven NOT_DONE rows.

Earlier decision text (`docs/dev-log/plans/2026-10-19-p1-remaining-decision-packet.md:35`):
"REGISTRATION-ONLY. Approved implementation exists in multiple unlanded variants;
later variant repairs shared-residual identifiability. No paired R numerical
receipt. Reconcile under owner release, retain the later identifiability repair,
bind one source tree, and qualify the existing two-case/control proposal before
measurement and separate admission." The same packet records that the joint
phylogenetic-signal contract "has implementation approval only: explicit species
selector, point estimates, CI refusal, and shared Gaussian residual mode."

Earlier note on the model class (`docs/dev-log/core070/surface-conversion-notes.md:84-94`):
Julia's `phylo_signal(fit::GllvmFit; Σ_phy)` takes an externally supplied
`Σ_phy` on an ordinary `GllvmFit`, "but R's `extract_phylo_signal()` reads
`report$Sigma_phy` off a genuine `phylo_latent()`/`phylo_indep()` structured fit
-- a different model class."

## 3. What R does at the pin

Source tree: `~/hsq_work/true-parity-prerun-20261001/gllvmTMB-p1/` on Totoro. It
is not a git clone, so identity rests on file hashes. `R/extract-omega.R` has
sha256 `dc7e8afdc5875086f2e35f82548138198c6092baa1892a0573aaf94c9b03929b`,
which equals the hash in the case receipt's `r_sources_read`, and
the whole `R/` directory (`diff -rq`), and two test files, are byte-identical to `~/hsq_work/gllvm-w2-3/oracle-build/source/`, whose
installed library carries `CORE070_SOURCE_PIN.toml` with `reference_commit =
"9539352f66f2db2cc26b1c393e67212a359b60c9"`. Line numbers below are from that tree.

The estimand (`R/extract-omega.R`):

- 468-476: `extract_phylo_signal <- function(fit, ci = FALSE, conf_level = 0.95,
  method = c("profile", "wald", "bootstrap"), nsim = 500L, seed = NULL,
  link_residual = c("none", "auto"))`.
- 502-511: errors when the fit has no phylogenetic term.
- 526-534: `Sigma_phy` is the diagonal of `extract_Sigma(fit, level = "phy",
  part = "total")`. "Sigma_phy = Lambda_phy Lambda_phy^T + Psi_phy when phylo_diag
  is fit". The phylogenetic unique variance is inside it.
- 535-545: "H^2's denominator is the SPECIES-level latent variance, so only
  components living on the cluster/species grouping may enter it (maintainer
  ruling 2026-07-08)." `unit_is_cluster` decides which tier is species-level.
- 547-564: `Sigma_non_s` is the diagonal of the shared (loading) part of the unit
  tier, only when the unit is the cluster.
- 566-580: `Psi_diag` is the unit-tier unique part when the unit is the cluster,
  otherwise the cluster-tier `q_it` term, `sd_q^2`.
- 592: `V_eta <- Sigma_phy + Sigma_non_s + Psi_diag + (Sigma_d %||% 0)`.
  With the default `link_residual = "none"`, `Sigma_d` is NULL, so the
  observation residual is not in `V_eta`.
- 668-679 and 852-860: `H2 = Sigma_phy / V_eta`, `C2_non = Sigma_non_s / V_eta`,
  `Psi = Psi_diag / V_eta`, each NA when the total is not finite or not positive.
- 615-623: the return is a data frame `(trait, H2, C2_non, Psi, V_eta)`.
- 630-632 and 493-501: point estimate by default; `ci = TRUE` is a separate path.

The unique part is a variance, not an SD: `R/extract-sigma.R:1497` and `1721`
square the reported SD (`sd_B^2`, `sd_q^2`), and `1870-1873` returns it as `s`.

R's own tests describe the cases that matter here
(`tests/testthat/test-phylo-signal-species-level.R`): lines 44-62 fit
`phylo_unique(species) + unique(0 + trait | species)` in a crossed site by
species design with a binomial probit family and require `H2 < 1` and `Psi > 0`; lines 81-120 require
`H2 + C2_non + Psi = 1` and `Psi * V_eta` to equal the unit-tier unique part.
`tests/testthat/test-phylo-signal-categorical.R:112`, inside the test at line 96 (probit and
multinomial mock fits), asserts `H2 = 1`; the Gaussian test at lines 117-135 only asserts
that no `liability` message appears.

The Gaussian residual in R is one shared SD. `R/fit-multi.R:4288-4293` and
`5802-5804` build a length-one `log_sigma_eps`, and `src/gllvmTMB.cpp:3266-3271`
reads it as `sigma_eps(0)`. This affects how a fit is compared, not what the
extractor reports, because the residual is outside `V_eta`.

## 4. What Julia does on main

- `src/extractors.jl:673-674`: `extract_phylo_signal(fit::GllvmFit; Σ_phy)` forwards to
  `phylo_signal`. It returns a `Vector{Float64}`, not R's table. There is no `ci`
  argument and no `C2_non`, `Psi` or `V_eta`.
- `src/confint_derived.jl:342-363`: `phylo_signal` computes
  `σ²_phy / (σ²_phy + Σ_y_site[t,t])`. By its own docstring (`src/confint_derived.jl:179-185`),
  `Σ_y_site` carries `σ²_B + σ²_W + σ²_eps` and the W-tier block, so the
  observation residual is inside the denominator. R keeps it out. Also, in this model the phylogeny
  runs across the rows of `Y` (the test fixture at `test/test_confint_derived_wald.jl:118-140`
  draws one phylogenetic effect per row, shared over columns), which matches the
  "different model class" note above.
- `src/phylo_latent_postfit.jl:24-34`: `extract_phylo_signal(fit::PrecisionMultivariateFit; ci)`
  returns R's table with `H2 = 1`, `C2_non = 0`, `Psi = 0`, `V_eta = diag(Sigma_phy)`,
  and refuses `ci = true`. It matches R for a bare `phylo_latent()` fit
  (`test/test_phylo_latent_twin.jl:166-172`, citing `test-phylo-signal-categorical.R:112`).
  It cannot represent a species-level non-phylogenetic variance.
- `src/joint_phylo_grouped_postfit.jl` (main): `JointPhyloGroupedGaussianFit` has
  `extract_Sigma` and component covariances (lines 98-140) but no
  `extract_phylo_signal`. Calling it raises `MethodError: no method matching
  extract_phylo_signal(::JointPhyloGroupedGaussianFit; species_level::Symbol)`
  (observed in the red run below).
- `src/joint_phylo_grouped_fit.jl:23-45` and `:130`: the joint fit estimates one
  residual variance per trait. `PrecisionMultivariateFit` already has
  `residual_mode in (:trait, :shared)` (`src/precision_multivariate_fit.jl:19-22`);
  the joint fit has no `residual_mode`.

## 5. The gap

G1. The row's evidence is a name match against a different function
(`phylo_signal`), not against `extract_phylo_signal`. A real receipt would need
a paired numeric comparison.

G2. The joint fit has no extractor on main. The designs that can show
`H2 < 1` with a species-level `Psi > 0`, which is R's headline case in
`test-phylo-signal-species-level.R:44-62`, cannot be run in Julia today.

G3. The `GllvmFit` method returns a different shape and a different
denominator on a different model class. The size of the difference on the
repository's own derived-CI fixture (seed 21, `has_phy_unique = true`, 6 rows,
200 columns, converged): `extract_phylo_signal` gives
`[0.0003, 0.4211, 0.2556, 0.0869, 0.1032, 0.2565]`; leaving `σ²_eps` out of
the denominator gives `[0.0007, 0.5076, 0.4667, 0.1235, 0.1371, 0.5039]`. This
is an illustration of the denominator difference, not an R comparison.

G4. Fit identity. R's Gaussian route has one shared residual SD; the joint
Julia fit has one residual per trait. Equal values at a common point test the
extraction algebra only. Two independently optimised fits are different models.

G5. Identifiability. When each species is observed once, the species-level
diagonal and the residual are aliased, and the phylogenetic and species
kernels can alias. The split into `H2` and `Psi` is then not determined by the
data. I found no identifiability guard in R's `extract_phylo_signal` body
(`R/extract-omega.R:468-632`); Julia already has a diagnostic
(`_joint_covariance_identification`).

## 6. Options

Option A, narrow and local. Land a joint-fit extractor with an explicit
`species_level` selector, point estimates only, `ci = true` refused, and an
identifiability refusal. Keep the row REGISTRATION-ONLY. This closes G2 and G5
and changes no evidence. The draft in section 7 is one version of A. Codex also
holds approved variants on its branches (read only, see section 7), so A must be
reconciled into one source tree before anything lands.

Option B, A plus a paired measurement. After A, add `residual_mode = :shared`
to the joint fit (closes G4), freeze one tree, one response and one trait order,
and measure the two proposed cases against R at the pin: a common-point check
of the covariance and the marginal likelihood, the extractor outputs, the four
negative controls, then independent fits with both engines' convergence blocks,
under the existing signed tolerances. Binding the row and any re-pointing of
the receipt's Julia symbol are separate later decisions. This is the only option
that can produce a numeric receipt.

Option C, scope the row to the existing twin. Ask for a signed disposition that
treats the `PrecisionMultivariateFit` method as the row's Julia counterpart and
leaves the joint and `GllvmFit` cases as Julia extras. I recommend against it:
the bare-phylogeny answer is identically 1, so a constant function would
satisfy the evidence, and it leaves G1 to G3 untouched.

Recommendation: A now, then B, in that order. Reject C. Under A and B the row
stays UNBOUND until a paired receipt exists and Shinichi signs the admission.
For G3, recommend a separate ruling on whether `extract_phylo_signal(::GllvmFit)`
should keep its vector form as a documented Julia extra; this packet does not
change it.

Estimate for B (a guess, not a measurement): the common-point checks are
seconds each on one core; one independent R fit plus one Julia fit of the
8-species, 24-observation case is probably under 30 minutes on Totoro at one
thread, but Totoro showed a load average of about 221 when this packet was
written, so I cannot give a firm number. Under D-287 that argues for a single
pre-run of one case first, with its time reported, before the full set. The
whole of B is probably under the 3 hour line, but that is unverified.

### Drafted reply for Shinichi

> Approve A: land one joint-fit `extract_phylo_signal` (explicit `species_level`,
> point estimates, `ci = true` refused, refuse unidentified fits), reconciled with
> the Codex variants into one source tree. No row, scoreboard or receipt change.
> For B: approve adding `residual_mode = :shared` to the joint fit and a single
> one-case pre-run on Totoro (2 cores, report the time) before the full two-case
> measurement. Reject C. The row stays UNBOUND until the paired receipt exists and
> I sign the admission. On `extract_phylo_signal(::GllvmFit)`: keep as a documented
> Julia extra for now, decide later. On the receipt naming `phylo_signal`: leave
> it until B is measured, then re-point in the same change that binds the row.

## 7. The draft fix

Branch `claude/c2-extract-phylo-signal-packet-20261102`, worktree
`/private/tmp/claude-503/-Users-z3437171-Dropbox-Github-Local-GLLVM-jl/eb56923c-569d-427b-bdc5-d1f448428f52/scratchpad/c2/extract-phylo-signal/`.
Files touched by the draft:

- `src/joint_phylo_grouped_postfit.jl`: new `_joint_phylo_grouped_species_term`
  and `extract_phylo_signal(fit::JointPhyloGroupedGaussianFit; species_level, ci = false)`.
- `test/test_extract_phylo_signal_joint.jl`: new test file, added to
  `test/runtests.jl` after `test_joint_phylo_grouped_postfit.jl`.

Behaviour: `V_eta = diag(Sigma_phy) + diag(selected ordinary term)`;
`H2 = a / V_eta`; `Psi = u / V_eta`; `C2_non = 0`; NaN when `V_eta` is not
finite and positive. Returns `(trait, H2, C2_non, Psi, V_eta)`. Observation
residual variance and ordinary terms not on the species grouping are excluded,
as R excludes them. `species_level` is required and must name a term whose
observation partition equals the partition of `fit.species_id`, so a site term
is refused. `ci = true` raises `GJL-GATE-PHYLO-LATENT-SIGNAL-CI`. A fit that is
not `:identified` raises `GJL-GATE-PHYLO-LATENT-SIGNAL-IDENTIFICATION`
(or `-INVALID`). `C2_non` is zero because the joint fit takes only independent
diagonal ordinary terms, so R's shared non-phylogenetic latent part is outside
this draft.

Tests are built at a declared parameter vector (`start`, `iterations = 0`), 8
species crossed with 3 sites (24 observations), 3 traits, with the tree scaled so
tip variance is 1 (checked: all eight tip variances equal 1.0). Expected values
are algebra from the declared components, not fitted results.

- Case A (loading `(0.70, -0.45, 0.50)`, species variance `(0.12, 0.22, 0.18)`):
  `V_eta = (0.61, 0.4225, 0.43)`, `H2 = (49/61, 81/169, 25/43)`.
- Case B (adds phylogenetic unique `(0.16, 0.09, 0.12)` and a crossed site term
  `(0.10, 0.15, 0.20)`): `V_eta = (0.77, 0.5125, 0.55)`,
  `H2 = (65/77, 117/205, 37/55)`. Site variance stays out of `V_eta`.
- Controls: changing only the excluded site and residual variances leaves all
  outputs unchanged; moving `delta` from the phylogenetic unique variance to the
  species diagonal keeps `V_eta` fixed, lowers `H2` by `delta / V_eta`, raises
  `Psi` by the same amount, and changes the log-likelihood; five refusals (CI,
  missing selector, unknown term, wrong type, site term); one aliased fit refused.

Red, then green, then a mutation check (original run on the local Mac; re-run on Totoro, Julia 1.10.12, one thread, see the PR for tallies):

- Red, before the method existed: 11 tests, 3 pass, 6 fail, 2 error. The 6
  failures are the five refusal checks and the aliased-fit refusal; the 2 errors
  are Case A and Case B, all from the `MethodError` above. The 3 passes are the
  three identifiability checks, which do not call the new method.
- Green, with the method: 28 of 28 pass.
- Mutation, `V_eta` changed to include the observation residual: 18 pass,
  10 fail, so the tests do detect the wrong-denominator error. The mutation was
  reverted.
- Neighbours on the same checkout: `test_joint_phylo_grouped_postfit.jl` 29 of 29,
  `test_phylo_latent_twin.jl` 89 of 89, `test_extractors.jl` 114 of 114,
  `test_shard_selection.jl` 43 of 43. Method ambiguities involving
  `extract_phylo_signal`: 0 (three methods).

Relation to Codex work (read with `git show` only, never checked out or edited):
`2026-10-25-joint-phylo-signal-extractor` exists at commit `9a49896eb` on
`codex/true-parity-p1-current-main-20261024` and `codex/true-parity-p1-restored-20261101`;
earlier variants are `67c737f46` (2026-10-16) and `a0f1794af` (2026-10-07). Their
method has the same name, selector, return shape, CI gate and identifiability
refusal. Their branch also adds `residual_mode = :shared` to the joint fit and
changes `src/joint_covariance_identification.jl`; this draft does not. The overlap
in `src/joint_phylo_grouped_postfit.jl` will conflict, so the draft is for
comparison and for the common-point tests, which are new on main. Prefer
reconciling into the Codex tree under owner release over landing both.

## 8. What is unverified

- No R fit was run. All R statements are source reading at the pin; no number was
  compared with R.
- The pin's git object was not found in any Totoro clone I checked (three). Identity
  of the R source rests on sha256 equality with the receipt and on the pin file in
  the oracle build, not on `git cat-file`.
- Cases A and B are common-point algebra. They have not been checked in R, and the
  R formulas that would give the same covariance (R tier for the species term,
  `unit` equal to `cluster` or not, tree scaling, sign and ordering of loadings)
  are not written.
- Whether an independent R fit and an independent Julia fit reach the same optimum
  is unknown. A shared residual mode is not in this draft, so no paired fit is
  possible yet. Convergence rules, both-engine gradient and Hessian checks, and
  tolerances would apply as signed.
- The G3 numbers come from one fixture and one seed and are an illustration.
- The claim that `GllvmFit` is a different model class rests on the existing
  `surface-conversion-notes.md` text and on reading the test fixture; I did not
  build an R model for it.
- The full `Pkg.test()` suite, Aqua and JET, and the docs build were not run on this
  draft. Only the files listed in section 7 were run.
- The Totoro estimate in section 6 is a guess under a very high load.
- `slop_check` on this packet: FINDINGS 0 (5 low-level pattern hits, 2.0 per 1000 words, no em dashes).

## 9. Hard-rule check

No change to row status, scoreboard, receipt or case map. Not touched: PR #990,
`src/confint_derived_wald.jl`, any Codex branch (read with `git show` only). No
tolerance widened; no signed FAIL reopened. Staged by name, not merged, not
pushed. Totoro use: read-only `sed`, `grep` and `sha256sum` on the R tree through
the existing ControlMaster socket, no compute.
