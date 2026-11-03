# C2 decision packet: simulate for gllvmTMB_multi and POST-SIMULATE-DEFAULT

## Status block (2026-11-03): ruling adopted, what this PR does and does not do

Shinichi answered "yes" on 2026-11-03 to the recommendation in section 7: option A (correct the five wording
faults) plus option C (sign that non-temporal Julia `simulate` is unconditional only), do not build
`condition_on_RE` for the other methods, and prepare an E2 rule before any larger R run. The rows stay UNBOUND
(`postfit-policy/POST-SIMULATE-DEFAULT` PARTIAL, `namespace/S3method/simulate,gllvmTMB_multi`
REGISTRATION-ONLY). This PR changes no row status, scoreboard line, receipt, case map, GATES text or tolerance.

What was done here:

1. This packet, rebased onto current origin/main `85e47a2a9` (the cited main-side lines were re-checked on that
   tree and still hold), with the verifier citation errors fixed: the test-title line (`:28`, was `:34`), the
   commit count, the full directory paths for the receipt, scoreboard and GATES, the `rr_B`/`Sigma_W` wording for
   the stale R cite, the `GllvmCovFit` signature, and the `report$eta` equivalence claim, which is now stated as
   distributional only.
2. Section 10 below: the exact before/after text for the five wording faults (option A) and the draft disposition
   wording (option C), both written out but NOT applied.
3. Section 11 below: a draft E2 rule for the larger R-side run, for Shinichi to sign before any run.
4. The Poisson `condition_on_RE` draft is not in this PR (ruling: do not build it now). It stays on the local
   branch `claude/c2-simulate-multi-packet-20261102` (commit `c5a7a8a94`, source `src/simulate_fit.jl`, test
   `test/test_simulate_condition_on_re_draft.jl`).

Why option A is not applied as a file edit: the contract JSON strings are read by
`tools/core070_postfit_p1_contract.py`, `tools/core070_postfit_p1_receipts.py` and
`tools/core070_verify_postfit_policy_batch.py`, and the case receipt carries the same comparand text. Editing them
means regenerating a receipt, which the standing C2 ruling forbids while the rows are unbound. The text is ready
for the day a signed ruling allows it.

Totoro runs for the neighbours (one thread, `OPENBLAS_NUM_THREADS=1`, two cores, Julia 1.10.0) are recorded in
section 12.

Date: 2026-11-02. Branch: `claude/c2-simulate-multi-packet-20261102`, from origin/main `16a7f08df`.
Pin: gllvmTMB `9539352f66f2db2cc26b1c393e67212a359b60c9` (0.7.1), read only.

This packet changes no row status, scoreboard, receipt, case map or signed text. Both rows stay unbound:
`postfit-policy/POST-SIMULATE-DEFAULT` (PARTIAL) and `namespace/S3method/simulate,gllvmTMB_multi`
(REGISTRATION-ONLY). Everything below is for your ruling.

What I think you asked: "For the two simulate rows, say exactly what the contract claims, what R does at the pin,
what Julia does, where they differ, what the choices are, and prepare a fix without applying any ruling."

## 1. The short version

1. The contract sentence is wrong about Julia. It says Julia "always conditions on the fitted mode". Julia does
   the opposite: every non-temporal `simulate` method on main redraws the latent scores from N(0, I) at each
   site. That is also what R does by default. The two defaults agree.
2. R has a second mode, `condition_on_RE = TRUE` (draw at the fitted modes). The Julia methods for fitted
   scalar-mean and Gaussian fits have no such mode. One Julia method does have the keyword
   (`TemporalGaussianFit`, `src/temporal_methods.jl:358`), so the claim "accepts no such keyword at all" is also
   wrong as a statement about `simulate` in general. The policy check only passes because it probes one
   Gaussian method.
3. The 7 of 40 seed failures that keep POST-SIMULATE-DEFAULT unbound are explained by the recorded R draw, not
   by a Julia difference. Seed 77's R moments sit about one Monte-Carlo standard error above R's own long-run
   means on average, and 11 of 25 moments are more than 2 standard errors off. Against fresh R seeds the same
   rule fails 8 of 1600 times (0.5 percent). This is diagnostic evidence only. It is not a pass.
4. A draft fix adds `condition_on_RE` to the Poisson method. Red: 2 pass, 2 fail, 4 error. Green: 11 of 11. No
   status changed. The draft is not wired into `test/runtests.jl`.

## 2. The contract text on main, verbatim

File `docs/dev-log/core070/true-parity-latest/postfit-policy-batch-contract-p1.json`, lines 152 to 157
(the same text is in `docs/dev-log/core070/postfit-policy-batch-contract.json`, same lines, and in
`docs/dev-log/core070/true-parity-latest/receipts/postfit/cases/CORE070-POSTFIT-SIMULATE-DEFAULT-NATIVE.json`, lines 19 to 22):

- case_id: `CORE070-POSTFIT-SIMULATE-DEFAULT-NATIVE`, source_id `postfit-policy/POST-SIMULATE-DEFAULT`
- julia_surface: "simulate(fit::_ScalarMuFit, n::Integer; rng=...) (src/simulate_fit.jl:30) -- no
  condition_on_RE-style keyword exists"
- r_call: "formals(simulate.gllvmTMB_multi)$condition_on_RE literal default FALSE
  (R/methods-gllvmTMB.R:1371-1381)"
- comparand: "signature-level documented-divergence check: R declares an explicit condition_on_RE=FALSE (LV
  redrawn) default; Julia's simulate() accepts no such keyword at all (always conditions on the fitted mode) --
  verified live from both sources' reflection, not hardcoded"
- check: `r_simulate_condition_on_re_default == false && !julia_simulate_has_condition_on_re_kwarg`

The scoreboard rows (`docs/dev-log/core070/true-parity-latest/scoreboard.md:354` and `:458`) read REGISTRATION-ONLY and PARTIAL, both "not bound".
The PARTIAL reason is in `docs/dev-log/core070/true-parity-latest/GATES.md` section N4 (lines 967 to 982): the Monte-Carlo twin passes at its recorded
seed but 7 of 40 alternative Julia seeds fail the rule, so the pass is seed-sensitive.

Five faults in that text, each checked against source:

| # | Fault | Evidence |
|---|-------|----------|
| F1 | "always conditions on the fitted mode" is false. Julia redraws. | `src/simulate_fit.jl:132` on main (`PoissonFit` method): `z_total = Zmean[s, :] .+ randn(rng, K)`; the same line is at 83, 168, 204, 240 and 275 for the other methods. The recorded twin `test/test_mc_simulate_p1.jl:28` is titled "latent scores redrawn". |
| F2 | R line cite 1371-1381 is stale. At the pin those lines are inside `tidy()` (line 1371 is in the `rr_B` branch; line 1381 is a `Sigma_W` line, not the `rr_W` branch). The formal is at 1523 to 1530, default at 1528. | Pin file, section 3. The namespace case already uses the right line (`definition_line_hint: 1523`). |
| F3 | The julia_surface line `:30` is the generic `_ScalarMuFit` method. A `PoissonFit`, `NBFit`, `GammaFit` or `BetaFit` dispatches to its own more specific method (main lines 108, 149, 185, 221). Only `ExponentialFit` reaches line 30. The row's fixture is a `PoissonFit`. | `git show origin/main:src/simulate_fit.jl`, `grep -n "^function simulate"`. |
| F4 | "no such keyword at all" is false for `simulate(::TemporalGaussianFit; nsim, condition_on_RE=false, rng)`. | `src/temporal_methods.jl:358`. |
| F5 | The receipt field `r_condition_on_re_default` is `true` when the R default is `FALSE`. The oracle sets it with `isFALSE(...)` (`tools/core070_postfit_policy_batch.R:166`), and the Julia verdict tests `== true` (`tools/core070_postfit_policy_batch.jl:325`). The contract's check string says `== false`. A reader of the receipt sees `true` next to a check that says `false`. | Receipt `harness_fields`; the 2026-10-20 note already records this. |

Also noted: the probe at `tools/core070_postfit_policy_batch.jl:233-238` calls
`simulate(fit, 2; condition_on_RE = true)` on the Gaussian `GllvmFit` that the batch fits
(`simulate(f::GllvmFit, n; rng, X, offset)`, `src/families/aghq_gaussian_fit.jl:316`). A MethodError there means
"keyword absent". So the check answers a question about one Gaussian method, not about `simulate`.

## 3. What R does at the pin

Source read: `~/hsq_work/true-parity-prerun-20261001/gllvmTMB-p1/R/methods-gllvmTMB.R` on Totoro. That folder has
no git history, so I identified it by content: its sha256 is
`6c32924b415646697ab4e471fda8b83108e416d662ba6e607bbeba335dd21616`, the same as the pin file recorded in
`receipts/covariance/oracle/source.json`. Two other folders have the same hash
(`p1-c3-smoke-20261010/r-source`, `gllvm-w2-3/oracle-build/source`). The live R runs below used the installed
build in `~/hsq_work/gllvm-w2-3/oracle-build/library`.

- Signature, `R/methods-gllvmTMB.R:1523-1530`:
  `simulate.gllvmTMB_multi <- function(object, nsim = 1, seed = NULL, newdata = NULL, condition_on_RE = FALSE, ...)`.
- Seed: `:1531-1533`, `set.seed(seed)` when given.
- Documentation of the default, `:1457-1464`: "By default (`condition_on_RE = FALSE`) the random effects are
  redrawn from the fitted covariance and the response is drawn from the fitted family, the unconditional
  simulation appropriate for a parametric bootstrap ... Set `condition_on_RE = TRUE` for the older conditional
  behaviour, which reuses the fitted random-effect modes and only adds residual noise."
- Conditional path, `:1564-1599`: with `newdata` or `condition_on_RE = TRUE`, draw at `object$report$eta`
  (`:1571-1572`). With `newdata` it falls back to Gaussian draws on the link scale, with a warning (`:1578-1593`).
- Default path, `:1601-1643`: check the tiers can be redrawn (`.check_simulate_unconditional`, `:2164-2189`),
  rebuild the predictor from scratch with `.simulate_eta_unconditional` (`:2193-2338`), then draw the response per
  family with `.draw_y_per_family` (`:1741`). For `rr_B` the scores are `rnorm(d_B * n_sites)` times the fitted
  loadings (`:2210-2226`).
- Unredrawable tier: warn and fall back to the conditional path (`:1618-1629`), or abort for rho-enabled fits.
- Temporal fits have their own branch (`:1535-1562`) with the same keyword.
- Output: a matrix with one row per observation and `nsim` columns (`:1507-1508`).
- Tweedie rows come back `NA_real_` with a warning (`:1514-1517`, `:1734-1737`). Julia draws Tweedie. This is a
  coverage difference outside this row.
- R's own help text says `phylo_diag` falls back (`:1490-1491`, `:1609-1610`) while the code lists it as handled
  (`:2166-2167`). That is an R-side inconsistency. R is read only here, so I note it and leave it.

Live R check at the pin, same model and data as `test/fixtures/gen_mc_simulate_p1.R`
(Poisson, p = 5, n = 60, one factor): the fit converged, `pdHess` true, logLik -518.167837718671.

## 4. What Julia does

Methods of `simulate` on main (`src/simulate_fit.jl` unless stated):

| Fit type | Line | Latent scores | Keywords |
|---|---|---|---|
| `_ScalarMuFit` (generic) | 30 | redrawn | `rng` |
| `BinomialFit` | 58 | redrawn (plus `X_lv` mean) | `N, X_lv, rng, offset` |
| `PoissonFit` | 108 | redrawn (plus `X_lv` mean) | `X_lv, rng, offset` |
| `NBFit`, `GammaFit`, `BetaFit`, `OrdinalFit` | 149, 185, 221, 257 | redrawn | `X_lv, rng` |
| `StudentTFit`, `TweedieFit` | 294, 331 | redrawn | `rng` |
| `GllvmCovFit` | 350 | redrawn | takes a 3-array `X` (not `n`); keywords `N, rng` |
| `GllvmFit` (Gaussian) | `src/families/aghq_gaussian_fit.jl:316` | redrawn | `rng, X, offset` |
| `TemporalGaussianFit` | `src/temporal_methods.jl:358` | redrawn by default; fitted effect when `condition_on_RE = true` | `nsim, condition_on_RE, rng` |

So, outside the temporal method, Julia has one mode, unconditional, which is R's default mode. The return
value is `p x n` for a chosen number of sites `n`, with no `nsim`, `seed` or `newdata`.

A Julia fit does not store its latent modes. They can be recomputed from the training response:
`predict(fit, Y; type = :link)` returns `eta = beta + Lambda * zhat` at the per-site Laplace mode
(`src/postfit.jl:1132-1160`). That is the analogue of what R reads from `report$eta`; the packet supports the equivalence only distributionally (section 6G), not by a cell-level comparison.

## 5. The gap

Behaviour gaps (real):

- G1. No conditional mode for the Gaussian, scalar-mean, binomial, ordinal, Student-t, Tweedie and covariate
  methods. R offers it. The default is unaffected.
- G2. Different call shape: Julia takes `n` sites and returns `p x n`; R takes `nsim` and returns observations by
  `nsim`. Julia has no `seed` or `newdata`. This is a documented divergence of style, not of distribution.

Contract gaps (wording): F1 to F5 above.

Evidence gap: the default twin is non-binding because of seed sensitivity (section 6).

Not a gap: the default distribution. Section 6 shows no detectable difference between the engines.

## 6. Measurements (all diagnostic, none is a pass)

Julia 1.10.0, `Distributions` 0.25.132, one thread. R 4.6.1 on Totoro, one process, one thread, `nice 10`.
Estimate given before the Totoro launch: at most 15 minutes on one core. The job finished in about 2 minutes.
Totoro load average was 245 to 295 throughout.

Scripts (not committed, in the session scratchpad): `.../scratchpad/c2/work/r_diag.R`, `seedsens.jl`,
`compare.jl`; R output `r_diag.out` (also on Totoro in `~/hsq_work/c2-simulate-multi-20261102/`).

A. Same fitted model. R and Julia fits of the fixture data agree: logLik -518.167837718671 against
   -518.1678377139525 (difference 4.7e-9); trait intercepts differ by at most 9e-6; loadings by at most 7e-6;
   the first ten fitted predictors by at most 9.2e-6.

B. Today's numbers for the recorded default twin do not match the receipt. Receipt: largest ratio 0.962,
   control fails 10 of 25 moments, largest control ratio 7.98. Today, main code, same seed `Xoshiro(30_000)`:
   largest ratio 0.701, control fails 12 of 25, largest control ratio 6.69. The test still passes. The likely
   cause is package drift (`Manifest.toml` is not committed, and the receipt does not record the
   `Distributions` version). I did not confirm that cause.

C. The 7 of 40 reproduces. With Julia streams `Xoshiro(30_000 + k)`, k = 1 to 40 (my own choice of seeds, the
   review's seeds were not recorded): 7 of 40 fail (k = 15, 18, 21, 22, 24, 30, 31), largest ratio 1.247,
   median per-seed maximum 0.886.

D. Why. The recorded R draw (seed 77, B = 400) against R's own long-run means (B = 20000, seed 424242), in its
   own Monte-Carlo standard errors: mean z +1.01, 11 of 25 moments beyond 2, largest 3.19. The recorded R draw
   is an unusual sample. Every Julia seed is compared with that one draw, so failures cluster.

E. R and Julia agree in the long run. B = 20000 each, 25 moments, combined standard errors:
   default mode, largest |z| 2.59, mean z 0.03, sd 1.07, none beyond the rule's 3.54;
   conditional mode (R `condition_on_RE = TRUE` against the draft Julia mode), largest |z| 3.10, mean z 0.33,
   sd 1.08, none beyond 3.54.

F. Fresh R seeds. 40 R seeds (5001 to 5040, B = 400 each) against the 40 Julia seeds of C, same rule: 8 of 1600
   pairs fail (0.5 percent; nominal familywise level 1 percent). The failures sit on R seeds 2 (3), 9 (2), 11,
   19 and 34 (1 each). Caveat: for these runs I used the R long-run standard deviation as the replicate
   standard deviation, because I saved only the 40 means.

G. Draft conditional twin against the recorded R `condition_on_RE = TRUE` moments (fixture `cond_*`), the
   unchanged rule, `Xoshiro(40_000)`, B = 400: largest ratio 0.589, 0 of 25 fail. Control (the same Julia
   conditional draws against the recorded R default moments): 16 of 25 fail, largest ratio 8.08.

## 7. Options

All three keep the row statuses as they are until you rule.

Option A, correct the wording only. Fix F1 to F5 in the contract, both contract copies and the case receipt
comparand, and rename or document the inverted field. No code. The row stays PARTIAL.

Option B, build the missing mode. Take the draft in section 8, extend it to the other methods, and change the
case from "signature divergence" to a two-mode numeric twin (default and conditional) under the same N4 rule.
Cost: about 10 more methods to touch, each needing `Y`; the current check asserts the keyword is absent, so it
must be rewritten, and a re-run on any method that gains the keyword would flip it. Benefit: exact parity with
R's two modes and consistency with `TemporalGaussianFit`.

Option C, sign a disposition. State that non-temporal Julia `simulate` is unconditional only by design, citing
R `:1457-1464` (conditional is "the older behaviour", unsuited to bootstrap) and the Julia table in section 4.
Pair it with Option A. No code. The row's name is the default, and the default agrees.

Evidence question, separate from A, B and C. The default twin cannot bind while its R side is the seed-77 draw.
E1: leave it non-binding as is. E2: regenerate the R side with a larger R run, written down as a new rule
before any run (N4 forbids retuning after a result), then judge the default row on that.

Recommendation: A plus C, and E2. Reasons: the default behaviour is the claim the row names and the evidence
says it agrees (E, F); the conditional mode is what R itself discourages for bootstrap use; B alters a
user-visible signature across many methods for a mode no row needs. B is a fair choice if you want keyword
parity with `TemporalGaussianFit`; the draft makes it cheap to start. E2 is the only thing that can move the
default row off PARTIAL. It needs your new rule first.

Drafted reply for you to send or edit:

> simulate rows: approve option A (correct the five wording faults) and option C (sign that non-temporal Julia
> simulate is unconditional only, with R's default as the matching mode). Do not build condition_on_RE for the
> other methods now; keep the Poisson draft on its branch. For POST-SIMULATE-DEFAULT, prepare a rule for a
> larger R-side run and bring it to me before any run. Rows stay as they are until I sign.

## 8. The draft fix

Branch `claude/c2-simulate-multi-packet-20261102`. Four commits: the failing test, the fix, this packet, and the `test_confint_offset` note. Only the packet text travels in the PR that carries this file; the draft stays on the local branch (see the status block).

- `src/simulate_fit.jl`: `simulate(fit::PoissonFit, n; condition_on_RE = false, Y = nothing, ...)`. Default path
  unchanged. With `condition_on_RE = true` it needs the training `Y` (`n` must equal `size(Y, 2)`), uses
  `predict(fit, Y; type = :link)` for the fitted predictor, and draws Poisson counts. AGHQ fits throw an
  `ArgumentError`. About 33 added lines.
- `test/test_simulate_condition_on_re_draft.jl`: default path equals a verbatim copy of main's loop and equals
  `condition_on_RE = false`; argument errors; conditional draws centred on `predict(...; :response)` (4000 draws,
  300 cells, Bonferroni); the section 6G twin and its control. The N4 rule for the twin is written at the top of
  the file before the run.
- Red (before the source change): 8 tests, 2 pass, 2 fail, 4 error, all by `MethodError`
  (`unsupported keyword argument "condition_on_RE"`). Log: `.../scratchpad/c2/work/red.log`.
- Green: 11 of 11.
- Existing tests with the fix applied: `test_simulate.jl` 5 of 5; `test_mc_simulate_p1.jl` 9 of 9;
  `test_aghq_public_poisson.jl` 52 of 52 and 3 of 3; `test_bridge_lv_predictor.jl` 207 of 207.
  `test_confint_offset.jl` was started and stopped by me after about 6 minutes with no output, so it has no result.
- The batch probe is unchanged: it still returns "no keyword" for the Gaussian `GllvmFit`. Had the batch fitted
  a Poisson model, the same probe would read "keyword present" (the call raises `ArgumentError`, not
  `MethodError`) and the check would flip. That is why B needs the check rewritten.
- Not done: other families, a documentation page entry, a `runtests.jl` line, any receipt, any status.

## 9. Unverified

- The cause of the 0.962 against 0.701 difference (package drift is a guess).
- Whether the review's 40 seeds were `30_000 + k`. My 7 of 40 matches in count only.
- The R-seed rule check in 6F uses the long-run standard deviation, not each run's own.
- The pin identity of the Totoro R source rests on a sha256 match of one file, not on a commit hash. The
  installed R library was built from a source folder with that same file hash, but I did not re-check its other
  files.
- Julia 1.10.0 only. No other Julia version, no CI, no full suite. Four test files finished, not the suite.
- (Resolved 2026-11-03, section 12) `test_confint_offset.jl` finished: 324 of 324 on Totoro.
- Whether any other tool or script reads the contract's `julia_surface` or `comparand` strings. I did not search
  the verifier scripts for string matches beyond `grep` on the phrases above.
- Draft behaviour for `X_lv` and `offset` fits in conditional mode is passed straight to `predict`; it has no
  test of its own. The AGHQ refusal and the `Y`-without-`condition_on_RE` refusal are also untested.
- Nothing was run on a machine other than the Mac Studio and one Totoro process. No DRAC.

## 10. Option A and C text, written out, NOT applied

Option A, replacement text for the three copies (two contract JSON files, one case receipt):

| Field | Now | Proposed |
|---|---|---|
| `julia_surface` | `simulate(fit::_ScalarMuFit, n::Integer; rng=...) (src/simulate_fit.jl:30) -- no condition_on_RE-style keyword exists` | `simulate(fit::PoissonFit, n::Integer; X_lv, rng, offset) (src/simulate_fit.jl:108); the generic _ScalarMuFit method at :30 serves only ExponentialFit. No non-temporal method has a condition_on_RE keyword; simulate(::TemporalGaussianFit; nsim, condition_on_RE, rng) at src/temporal_methods.jl:358 does.` |
| `r_call` | `... (R/methods-gllvmTMB.R:1371-1381)` | `... (R/methods-gllvmTMB.R:1523-1530, default at 1528)` |
| `comparand` | `Julia's simulate() accepts no such keyword at all (always conditions on the fitted mode)` | `non-temporal Julia simulate() methods have no condition_on_RE keyword and always redraw the latent scores from N(0, I) at each site, which is R's default mode (condition_on_RE = FALSE)` |
| harness field `r_condition_on_re_default` | `true` (set by `isFALSE(...)`, so it means "R default is FALSE") | rename to `r_condition_on_re_default_is_false`, or add a one-line note in the contract; the check string `== false` must then read `r_condition_on_re_default_is_false == true` |

The probe at `tools/core070_postfit_policy_batch.jl:233-238` should also say in a comment that it tests one Gaussian
method (`GllvmFit`), not `simulate` in general.

Option C, draft disposition wording (for signature; it has no effect until signed):

> Non-temporal Julia `simulate` methods are unconditional only, by design. They redraw the latent scores at each
> site, which is the mode R documents as the default and as the one suited to a parametric bootstrap
> (R/methods-gllvmTMB.R:1457-1464). R's `condition_on_RE = TRUE` (the "older conditional behaviour") has no
> non-temporal Julia counterpart. The call shape differs (Julia: `n` sites and a `p x n` matrix; R: `nsim`, `seed`,
> `newdata`, observations by `nsim`); this is a documented difference of style, not of distribution.

## 11. Draft E2 rule (for signature BEFORE any run; nothing has been run under it)

Purpose: replace the seed-77, B = 400 R draw as the R side of the default twin by an R side large enough that the
N4 rule's seed sensitivity (7 of 40 Julia seeds fail) is not dominated by one unusual R sample (section 6D).

Proposed, to be written into the case plan before the run and frozen:

1. R side: `simulate.gllvmTMB_multi(fit, nsim = 20000, seed = S_R)` on the same fixture fit (Poisson, p = 5,
   n = 60, one factor), pin `9539352f66f2db2cc26b1c393e67212a359b60c9`. `S_R` is a fresh integer chosen by
   Shinichi, not 77 and not 424242 (both are already seen). The 25 moments are the existing ones in
   `test/fixtures/mc_simulate_p1.toml`; the R mean and its standard error are taken from the 20000 replicates.
2. Julia side: B = 400 replicates, stream `Xoshiro(30_000)`, unchanged. Replicate standard deviation from the Julia
   draws, as in the existing twin.
3. Rule: the existing N4 comparison rule, with the R-side standard error from the 20000 replicates. No threshold
   changes, no tolerance widening. The control (Julia conditional-mode style draws, or the existing control)
   must still fail at least as many moments as it does now.
4. Seed sensitivity is reported alongside: the same rule over Julia streams `30_000 + k`, k = 1 to 40, with the
   fraction failing. The row can only move off PARTIAL if the recorded stream passes AND the seed-sensitivity
   fraction is at or below the nominal familywise level written in N4 (1 percent of pairs is the reference level,
   section 6F). If either fails, the row stays PARTIAL and the result is reported as such.
5. Estimate before launch: R, about 2 minutes on one core (section 6: the 20000 R run finished within the 2-minute
   Totoro job); Julia, under 5 minutes. Far under the 3 hour line.
6. Whatever happens, diagnostic evidence is not a pass; a pass needs the checker to accept the new receipt.

Open choices for Shinichi: the value of `S_R`; whether point 4's "at or below nominal level" is the right
promotion test or whether the row should stay non-binding regardless.

## 12. Totoro runs on 2026-11-03 (neighbours of the draft)

Machine: Totoro, Julia 1.10.0, `JULIA_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, two cores (`taskset`), `nice 10`.
Code: the packet's draft branch (commit `c5a7a8a94`; `src/simulate_fit.jl` and the other files below are identical
to origin/main `85e47a2a9` apart from the draft). Estimate before launch: at most 2.5 hours for
`test_confint_offset.jl` (cap `timeout 9000`), a few minutes for the rest. Nothing overran.

| Test file | Result | Wall time |
|---|---|---|
| `test_simulate_condition_on_re_draft.jl` | 11 of 11 | 20 s testset |
| `test_simulate.jl` | 5 of 5 | 26 s testset |
| `test_mc_simulate_p1.jl` | 9 of 9 | 19 s testset |
| `test_aghq_public_poisson.jl` | 52 of 52 and 3 of 3 | 29 s |
| `test_bridge_lv_predictor.jl` | 207 of 207 | 77 s |
| `test_confint_offset.jl` | 324 of 324 | 5 min 13 s |

`test_confint_offset.jl` finishes. The 2026-11-02 Mac run was stopped at about 6 minutes with no output because the
file prints nothing until the testset ends and the run needs about 5 minutes on a Totoro core (the Mac was probably
slower or busier; not measured). Its pass says the draft does not disturb confint with offsets; it is not evidence
for either simulate row.
