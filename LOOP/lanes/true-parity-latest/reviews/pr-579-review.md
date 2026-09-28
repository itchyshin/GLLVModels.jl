# Review of draft PR #579 — data and fit-input families re-measured at gllvmTMB P1

Reviewer: independent adversarial review (Claude, Fable 5.1), 2026-09-27.
Range reviewed: `1b99ae6b8..201119a3e` (6 commits) on `claude/true-parity-p1-data`, stacked on #571.
Worktree: detached at `201119a3e` under `/Users/z3437171/local-scratch/lanes/GLLVM.jl-review-579` (removed after this review).
Nothing edited on the branch, nothing pushed, nothing posted to GitHub.

## Verdict: BLOCKING (one item), otherwise sound

The fit-input measurement itself is real and reproducible: I re-ran the fit-input-2 batch at P1 in my own
worktree against the P1 oracle library and got a bit-identical `r-oracle.json`, `julia-results.json` and
`results.tsv`; both batch verifiers pass; both checkers give fit-input C1_MET / C8_MET; the receipt
`--check` catches every tamper I tried on tracked values. The one blocking item is a false premise
written into the 28 data receipts, the PR body and the maintainer's decision #2: GLLVModels does have a
fit-time `offset` and missing-response (`mask` / `missing` in `Y`) surface on the non-Gaussian fitters,
reachable through `fit_gllvm(...; kwargs...)`. The runtime introspection cannot see it because it reads
`Base.kwarg_decl` on the dispatcher, which reports only `kwargs...`. The rows stay free either way
(the R side is a helper replay with no fit number), so no count changes; the text and the decision
premise do.

## Findings

### 1. BLOCKING — "every planned Julia surface absent" is wrong for offset and missing-response rows

Evidence (my own runs at `201119a3e`, Julia 1.10):

```
kwarg_decl(fit_gllvm) = [:family, :K, ..., :species_id, Symbol("kwargs...")]
fit_gllvm(Y; family=Poisson(), K=1, offset=O)   -> accepted, forwarded (no MethodError)
fit_gllvm(Y; family=Poisson(), K=1, mask=M)     -> loglik -110.546 (vs -112.539 unmasked)
fit_gllvm(Ym; family=Poisson(), K=1)  (Ym has a `missing` cell) -> -110.546, equal to the mask fit
fit_gllvm(Y; family=Poisson(), K=1, weights=W)  -> MethodError (no surface)
```

`src/families/negbin.jl:166-167` documents "pass a `mask` ... or `missing` entries in `Y`";
`src/families/binomial.jl:344-345` and `negbin.jl:177` take `mask = nothing, offset = nothing`
(`η = β + offset + Λz`, line 391/223); `src/postfit.jl` takes `offset=` on `predict`/`getLV` for the
AGHQ routes; `docs/src/api.md:617` says "`mask`/missing responses and offsets are supported".
(My offset probe used a constant offset, which the intercept absorbs, so the logLik moved only in the last
digit; the code path is unambiguous from the source.)

What the PR writes instead:
- 28 case receipts, `why_not_numeric`: "GLLVModels has no surface for it (runtime introspection found none of
  the planned symbols or keywords)". True for the 13 `W-*` rows, not for the 11 `OFF-*` and 4 `MISS-*` rows.
- PR body: "GLLVModels has no surface for any of them (no weights, offset or miss keyword on
  `gllvm()`/`fit_gllvm()`, no helper symbol)". `fit_gllvm` forwards `offset` and `mask`.
- Contract `julia_surface_status` (carried verbatim from P0, now re-stamped at P1): "`Base.kwarg_decl()` over
  `gllvm()`/`fit_gllvm()` to rule out an undocumented keyword". It cannot rule one out: both entry points end
  in `kwargs...` (`src/families/fit_gllvm.jl:164`, `src/formula.jl:186`).
- Decision #2 for the maintainer: "They cannot bind under the numeric rule until GLLVModels grows weights,
  offset and missing-data surfaces." Weights, yes. Offset and missing, no.

Why this is blocking and not cosmetic: the programme rule is that name matches never count. This is the
mirror case — name *absence* was counted as surface absence, and the resulting statement is asked to
drive a maintainer decision. The tier outcome is unaffected (no data row can bind on a helper replay), so
the fix is textual and narrow.

Suggested fix: (a) in `tools/core070_data_p1_receipts.py` and the PR body, replace "no surface" with
"no helper-equivalent surface": weights — none at all (MethodError); offset and missing — fit-time matrix
`offset=` and `mask=`/`missing` on the non-Gaussian fitters via `fit_gllvm`'s `kwargs...`, but no
formula-offset evaluator, stored-offset accessor, or `miss_control` constructor, which is what these 15
R cases replay; (b) note in the contract twin (or a regeneration-log entry) that `kwarg_decl` on a
`kwargs...` dispatcher is not a keyword census; (c) rewrite decision #2 as "offset/missing rows could get a
numeric fit-level comparison in a follow-up batch; weights cannot". Do not touch the P0 contract.

### 2. NON-BLOCKING — the coefficient comparison on the Gaussian iid rows is analytically fixed by the data

For `INPUT-GAUSS-DEFAULT` (native + formula) and the unbound `GAUSS-LOADINGS` native case, the three
"coefficients" are trait intercepts with `0 + trait` and iid sites; the Gaussian MLE of the intercept is the
trait's sample mean regardless of the latent structure. From `r-oracle.json`:

```
row means of Y   [-0.269469981520691, -0.020199070652745, -0.107674044190747]
Julia LOADINGS β [-0.269469981520691, -0.020199070652748, -0.107674044190747]   (7e-15)
Julia DEFAULT  β  same to 1e-16;  R DEFAULT β differs by 8e-10 (optimiser noise)
```

So on these rows any code that returns column means passes the coef block; only the logLik block (3e-9,
1.6e-10) discriminates the model. `mark_degenerate` (constant or ~0 R values) does not catch this, and the
receipts say `discriminating: true` on the coef entries. The rows are still discriminating via logLik, so
they bind legitimately; I would flag the coef entries on the iid Gaussian rows as "analytically
data-determined" rather than let them count as a second independent check. The binomial and
animal/kernel rows are fine: binomial coefficients differ from the logit row means (0.186 vs 0.134), and the
GLS intercepts under `A`/`K` depend on the covariance (R and Julia differ by 6e-7 to 2e-6, which is
exactly the optimiser-level disagreement an independent fit should show).

### 3. NON-BLOCKING — KERNEL-TWO-AUTO is one comparison on two rows (disclosed)

Same fixture, same data, identical R values (`kernel_two_auto == kernel_two` in `r-oracle.json`) and
identical Julia values. The receipt, case map, PR body and check-log all say so and leave the count to the
maintainer. Nothing hidden; recording it so the "6 bound" headline is read as 5 measurements.

### 4. NON-BLOCKING — provenance fields are not covered by `--check`

`--check` re-hashes every `read_from` file and re-derives bodies excluding `PROVENANCE_KEYS`. Editing
`glvmodels_commit` in a case receipt to `000...0` passes `--check` unchanged (tested). `run-commit.json`
is hashed, so the true run commit is protected; only the receipt's own copy is free-floating. Suggest
checking `rec["glvmodels_commit"] == run-commit.json` inside `check()`.

### 5. NON-BLOCKING — introspection receipt carries another worktree's absolute path

`data-batch-julia-introspection.json` records
`contract_path: /Users/z3437171/local-scratch/GLLVM.jl-a3-data/tools/../docs/.../data-batch-contract-p1.json`.
Harmless (contract sha is what binds), but a tracked receipt naming a private path on a different
worktree is noise; suggest recording the repo-relative path.

### 6. NON-BLOCKING, pre-existing — `sha256_file` in the data runner breaks on paths with spaces

Refusing a wrong source tree at P1 works (no destination created), but when the tree path contains a
space (`/Users/z3437171/Dropbox/Github Local/gllvmTMB`) the refusal is a `sha256sum ... had status 1`
error from the unquoted `system()` call rather than the pin-mismatch message. Not introduced by this PR.

## What I verified (with evidence)

- **Scope**: `git diff --stat 1b99ae6b8..201119a3e` touches only `docs/dev-log/**` and `tools/**`. No `src/`,
  `Project.toml`, `.github/`, `GATES.md`, no P0 contract, no `required-source-case-map.json`. No
  `julia-stdout.log`/`stderr.log` or other raw logs tracked. No agent `@handles` in the PR body or the diff.
- **Re-run at P1** (my worktree, `OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=4`, ~2 min total):
  `GLLVM_PARITY_PIN=P1 CORE070_P1_R_SOURCE_ROOT=.../a3cov-oracle/build/source Rscript tools/core070_fit_input_2_batch.R .../a3cov-oracle/build/library <out>` →
  `CORE070_FIT_INPUT_2_BATCH_PASS`; `r-oracle.json`, `julia-results.json`, `results.tsv` **bit-identical** to the
  tracked receipts. Data runner and Julia introspection: `CORE070_DATA_BATCH_PASS`,
  `..._ALL_SURFACES_ABSENT`; `raw.tsv` identical, `data-batch-results.json` differs only in `generated_at` and
  microsecond `elapsed_seconds`; introspection identical modulo `contract_path`.
- **R values are R at P1**: library marker `CORE070_SOURCE_PIN.toml` names P1
  `9539352f…`, version 0.7.1; the runner checks the marker and the installed package path; the three changed
  R files (`R/gllvmTMB.R`, `R/animal-keyword.R`, `R/kernel-keywords.R`) hash identically between
  `git show P1:<file>` in the gllvmTMB clone and the oracle source tree. R values differ from Julia values by
  1e-9 to 2e-6, so they are not copies. P0 raw receipts are not on this host, so I could not compare P1
  R values to P0 R values directly (the PR says the same).
- **Julia values come from GLLVModels fits, independent of R**: the child reads only `y`, `A`/`B`, dims and
  `fixed_residual_sd = sd(y)/1000` (a data constant, unchanged from P0) from the oracle; no R estimates
  seed any fit. Formula cases go through `gllvm(@formula(y ~ 1), Y, data; ...)` — a different entry path to
  the same engine (values agree with native to ≤5e-15), so they test the front door, not a second engine.
- **Tolerance**: 1e-4 for both logLik and coef in the P0 contract; the P1 twin carries it verbatim.
- **Discrimination**: +1% on one Julia coefficient (BINOMIAL, −1.636 → −1.652) → receipt `--check`
  STALE (hash), batch verifier "does not match julia_results_sha256"; this is a 0.016 gap against 1e-4, so
  it would also fail the comparison if re-derived. The batch's own negative controls (+1 shifts, wrong
  model, swapped Y) behave.
- **Checkers**: main (`880cad4c7`, identical to the branch file) and #561 (`92cf39571`), `PARITY_REF=FS`:
  data C1 required=28 bound=0 free=28 / C8 28 NOT_TWINNED_NOT_SIGNED; fit-input C1 required=6 bound=6
  (#561 bound_numeric=6, mismatches none) C1_MET, C8_MET. Matches the PR body.
- **Coverage**: every fit-input row's two `executable_case_ids` have receipts with `comparison` blocks tied to
  contract sha `90025cc7…` and run commit `10467e0cc` (clean).
- **Pin strictness / refusal**: `GLLVM_PARITY_PIN=P2` refused by both verifiers; data runner at P1 without
  `CORE070_P1_ORACLE_LIBRARY` refused with no destination created; wrong source tree refused (see #6).
  `tools/core070_data_p1_contract.py --check` → `CORE070_DATA_P1_CONTRACTS_CURRENT` against the local
  gllvmTMB clone containing P1.
- **`--check` tamper detection**: +1% Julia coef → STALE (hash); inflated `max_abs_diff` in one case
  receipt → "comparison block differs from the re-derivation"; a data row relabelled `numeric` → "case-map
  rows differ"; `glvmodels_commit` edit → NOT caught (#4).
- **masks-known "not measured"**: justified. `tools/core070_masks_known.R:149-170` replays retained inputs
  from `.unlazy/core070-aghq/masks-known-01/attempt1/out` (absent here) and stops if missing;
  `tools/core070_masks_known.jl:2-7` states it never calls GLLVModels.
- **Library bound by pin, not path**: receipts carry `source_pin` (reference commit, marker sha, source and
  installed tree shas, version); `installed_tree_sha256` is not recomputed at run time (disclosed in the PR).

## What I did not check

- P0 raw receipts (not on this host): could not confirm the P1 R values differ from or equal the P0 values.
- The `gllvmTMB` binomial/Gaussian model mappings themselves (e.g. that R `latent(0 + trait | site)`
  default is the same model as Julia `fit_gaussian_pervar_gllvm(fixed_residual_sd = sd/1000)`); the
  3e-9 logLik agreement is strong indirect evidence, and the mapping is unchanged from P0.
- The `tools/core070_source_pin.R` / `core070_source_pin_check.py` internals (shared with #567–#571,
  reviewed there).
- `test/parity/test_core070_pin.jl` and `tools/test_true_parity_check.mjs` (not re-run).
- A non-constant offset probe (my probe used a constant offset, absorbed by the intercept); the offset
  code path is read from source, not exercised numerically.
- Anything outside `1b99ae6b8..201119a3e`.
