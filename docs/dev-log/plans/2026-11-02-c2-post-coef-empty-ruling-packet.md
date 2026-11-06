# POST-COEF-EMPTY ruling packet (2026-11-02)

## Status block (2026-11-03): option B adopted, status-neutral part only

Shinichi answered "yes" on 2026-11-03 to option B (a discriminating pair, tolerance 0 on lengths `[0, 5, 6]`, two
controls, names left as the accepted gap, row stays unbound until the receipt from a real run exists, no closure
by disposition). The row `postfit-policy/POST-COEF-EMPTY` stays UNBOUND. This PR changes no row status,
scoreboard line, receipt, case map, GATES text or tolerance, and does not edit the frozen contract JSON.

Done in this PR: the packet rebased onto origin/main `85e47a2a9`, the draft R half, Julia half and R diagnostic
JSON, and the run-instruction fixes from the two verifiers (test-file header and run recipe, `R_LIBS` for the R
script, the "fails all three" wording, `bridge.jl:1085`, and the unsourced "13" count).

Not done, because each needs a signed contract edit or a receipt: moving the pair into the batch runner against
the frozen Mac oracle, editing `postfit-policy-batch-contract-p1.json`, and writing the receipt.

The new run recipe was checked on 2026-11-03 on Totoro (Julia 1.10.0, one thread, two cores, throwaway
environment, about 1 to 2 minutes per run after instantiate; estimate given before launch: at most 15 minutes): real
accessor 7 pass of 7, always-empty control 5 pass 2 fail, spurious control 4 pass 3 fail. Same tallies as section 7.
Diagnostic only, not a pass for any row.

PR #1063: see section 8 and the PR description for what it still needs.

Row: `postfit-policy/POST-COEF-EMPTY`, case `CORE070-POSTFIT-COEF-EMPTY-NATIVE`.
Pin: gllvmTMB `9539352f66f2db2cc26b1c393e67212a359b60c9`. Main at writing: `16a7f08df`.

This packet changes no row status, scoreboard line, receipt, case map or tolerance. The row stays UNBOUND (scoreboard: NON-DISCRIMINATING, not bound) until Shinichi rules. Everything below marked "measured" is diagnostic evidence and is not a pass.

## 1. The question

The only committed comparison for this row is length 0 against length 0, with a synthetic object on the R side. Should the row (A) stay as it is, (B) get a real discriminating pair, or (C) be closed by a signed disposition? Separately, what is left to decide on draft PR #1063?

## 2. Contract text on main

1. `docs/dev-log/core070/true-parity-latest/postfit-policy-batch-contract-p1.json:28`, `fixture.empty_design.description`: "POST-COEF-EMPTY only. Julia side: fit_gaussian_gllvm(Y; K=1) called with no X (a real native fit, fit.pars.β===nothing). R side: coef.gllvmTMB_multi's empty-X_fix_names branch is evaluated directly against a synthetic mock object (X_fix_names <- character(0)) by parsing just that one function out of the pinned R source and eval()-ing it in the gllvmTMB namespace ... No live gllvmTMB() formula fit with zero fixed effects is attempted; the branch is unconditional on X_fix_names length and needs no fitted TMB object to exercise honestly."
2. Same file, lines 40-46: comparand "length(coef)==0 on both sides", check `length(StatsAPI.coef(fit_no_x))==0 && length(r_empty_coef)==0`, julia_surface `src/postfit.jl:633` (stale on main, see section 5).
3. `docs/dev-log/core070/postfit-policy-required-case-plan.json:8-21`: `evidence_kind: "native"`, two comparands (length 0 in both engines; R `names(coef) == character(0)` against Julia's unnamed empty vector, "document as an accepted representational gap, not a value mismatch"), and the rationale that the branch is "callable and observable from a plain fit with no fixed effects".
4. `docs/dev-log/core070/true-parity-latest/GATES.md:393-398`: the row was added to the integer-equality list by D-319 item N2 (basis: drafted N2 text, wave-plan-2026-10-03.md line 17, which is in the lane kit and not on main). "POST-COEF-EMPTY does not bind: the shared degenerate-comparison rule ... applies to integer entries as well, so its comparison is `discriminating: false` and the row reads `numeric_non_discriminating`."
5. `docs/dev-log/core070/true-parity-latest/scoreboard.md:443`: NON-DISCRIMINATING, "not bound; cited: non-binding ...".
6. The runner: `tools/core070_postfit_policy_batch.R:175-184` parses `coef.gllvmTMB_multi` out of `R/vcov-coef.R` and calls it on `list(X_fix_names = character(0))`. `tools/core070_postfit_policy_batch.jl:245` fits `fit_gaussian_gllvm(Y; K = 1)` and `:260-263` records `length(j_empty_coef) == 0 && length(r_empty_coef) == 0`.

Tension inside main: the case plan calls the case native and says a plain no-X fit exposes the branch. The contract and runner deliberately use a mock. The plan's own wording supports a real fit; the contract text does not require one. No clause on main says "real fitted R object".

## 3. What R does at the pin

Clone: `~/hsq_work/true-parity-prerun-20261001/gllvmTMB-p1` on Totoro (a source tree without `.git`). I compared the git blob hash of all 131 files under `R/`, `src/`, `DESCRIPTION` and `NAMESPACE` with the pin's tree read from the Mac clone with `git ls-tree`. All 131 are identical, so the tree is the pin's content.

- `R/vcov-coef.R:59-65`:
  `coef.gllvmTMB_multi <- function(object, ...) { nm <- object$X_fix_names %||% character(0); if (length(nm) == 0L) { return(stats::setNames(numeric(0), character(0))) }; stats::setNames(as.numeric(.gllvmTMB_b_fix_values(object)), nm) }`
- `R/vcov-coef.R:22`: documented return, "a named numeric vector, one entry per fixed-effect term".
- `R/fit-multi.R:3332`: `X_fix <- stats::model.matrix(parsed$fixed, mf)`. `R/fit-multi.R:9275`: `X_fix_names = colnames(X_fix)`.
- `NAMESPACE:7`: `S3method(coef,gllvmTMB_multi)`.

Measured on Totoro with the installed library (version 0.7.1), through the public `gllvmTMB()` entry, p = 5, n = 80, one thread, `tools/core070_coef_empty_pair_draft.R`. Output is committed as `docs/dev-log/plans/2026-11-02-c2-post-coef-empty-r-diagnostic.json`.

| R formula (fixed part) | `fit$X_fix_names` | `length(coef(fit))` | names |
|---|---|---|---|
| `0` | NULL (design is 400 by 0) | 0 | `character(0)`, names attribute present |
| `0 + trait` | 5 names | 5 | traitt1 ... traitt5 |
| `0 + trait + x` | 6 names | 6 | traitt1 ... traitt5, x |

For the no-fixed-effect fit, `coef(fit)` is `identical()` to `setNames(numeric(0), character(0))`, the same object the synthetic mock returns. So the mock is faithful for this branch. One difference: a real fit stores `X_fix_names` as NULL where the mock sets `character(0)`. Both reach the same branch through `%||%`.

## 4. What Julia does

- `src/postfit.jl:845` on main: `StatsAPI.coef(fit::GllvmFit) = fit.pars.β === nothing ? Float64[] : copy(fit.pars.β)`. At the measured commit `681c4c3ca` the same line is `src/postfit.jl:652` (checked with `git show`).
- `src/fit.jl:137-138`: documented, "A zero-column design, or fixing every coefficient to zero, gives the same zero-mean model as `X=nothing`." `src/fit.jl:485-491` gives `Float64[]` when q is 0; `src/fit.jl:513` expands fixed zeros.
- `src/formula.jl:318-320`: `y ~ 0` is the documented zero mean (`fit_gaussian_gllvm`); `y ~ 1` goes through `_fit_gaussian_trait_intercepts` and gives one intercept per trait.
- Existing Julia test: `test/test_statsapi.jl:23`, `@test isempty(StatsAPI.coef(fit_gauss))`.

Measured (this tree, one thread, 1 Julia process per run):

| Route | coef type | length |
|---|---|---|
| `fit_gaussian_gllvm(Y; K=1)` | `Vector{Float64}` | 0 |
| same with `X = zeros(p, n, 0)` | `Vector{Float64}` | 0 |
| same with X of 2 columns | `Vector{Float64}` | 2 |
| X of 2 columns, both fixed | `Vector{Float64}` | 2 (zeros) |
| `gllvm(@formula(y ~ 0), Y, data; K=1)` | `Vector{Float64}` | 0 |
| `gllvm(@formula(y ~ 1), ...)` | `Vector{Float64}` | 5 |
| `gllvm(@formula(y ~ 1 + x), ...)` | `Vector{Float64}` | 6 |
| `gllvm(@formula(y ~ 0 + x), ...)` | `Vector{Float64}` | 1 |

`GllvmFit` has no `coefnames` method.

## 5. The gap

I found no value or engine defect. The lengths agree on every public pairing I could build (0, 5, 6). The gaps are about what the receipt can prove:

1. Evidence is not discriminating. R value 0 equals Julia value 0. `degenerate()` in `tools/core070_postfit_p1_receipts.py:256-265` returns "every |R value| < 1e-10" for `[0]` and `None` for the pair `[0, 5, 6]`, so a pair would count as discriminating under the unchanged rule. Measured control: an accessor that always returns `Float64[]` passes the committed 0 = 0 check, and fails both positive-X pairs of the draft (section 7).
2. The R side is synthetic. A real no-X fit works at the pin (section 3), so the mock was a choice, not a limit.
3. Names and type. R returns a named numeric vector, Julia an unnamed `Vector{Float64}` with no name carrier. The case plan already records this as an accepted representational gap. Nothing here changes that.
4. Stale citation. The contract cites `src/postfit.jl:633`; the line is 845 on main. PR #1063 corrects the receipt to 652, which is right at `681c4c3ca`, the measured commit.

## 6. Options

**A. Leave it.** The row stays unbound for good; #1063 lands as a label fix. No work. Cost: the row can never be evidenced, and it stays outside the bound rows (GATES.md N1 says "about 13 rows" are non-binding context; I found no exact count).

**B. Approve a discriminating pair (recommended).** New case scope: public no-fixed-effect, trait-intercept and trait-intercept-plus-covariate fits on both sides, integer equality at tolerance 0 on the lengths `[0, 5, 6]`, a spurious-coefficient control and an always-empty control, names recorded separately as the accepted gap. Then a follow-up PR runs the real measurement (R on the frozen Mac oracle, Julia on the pinned commit) and writes a receipt. Work: edit the frozen contract JSON, extend the batch pair, one Totoro or Mac run of a few minutes. The row binds only if the checker accepts the new comparison. Risk: the contract JSON is frozen, so this needs your signature; the names gap stays open.

**C. Close by signed disposition.** Record that the branch is a one-line constant-length accessor on both sides, that the measured pair agreed as diagnostic evidence, and that the names gap is accepted. The row becomes DISPOSITION-SIGNED. Cost: nothing to run. Cost to rigour: it closes without a bound comparison, when a real one takes minutes.

**Recommendation: B.** It needs no engine change, the diagnostic already shows agreement, and it turns the only hollow comparison in the postfit family into a real one. Keep A as the status until B's receipt exists.

Drafted reply, for Shinichi to send or edit:

> Option B for POST-COEF-EMPTY. Approve the new case: public no-fixed-effect, trait-intercept and trait-intercept-plus-covariate fits in both engines, lengths compared exactly (tolerance 0), with an always-empty control and a spurious-coefficient control. Names stay the accepted representational gap. Keep the row unbound until the receipt from that run exists. Do not close it by disposition.

If you prefer C, replace the last three sentences with: "Close by signed disposition, reason: constant-length accessor, measured pair agreed as diagnostic, names gap accepted."

## 7. Draft fix (no status change)

Files, all new, none wired into `runtests.jl` and none under a receipts directory:

- `tools/core070_coef_empty_pair_draft.R`: the R half. Refuses to run unless `R/vcov-coef.R` is the pin's blob (`a707d537...`). Writes the diagnostic JSON.
- `docs/dev-log/plans/2026-11-02-c2-post-coef-empty-r-diagnostic.json`: its output, `"binding": false`.
- `test/draft_coef_empty_pair.jl`: the Julia half. It fits through the public `gllvm(@formula)` route, compares each coef length with the R JSON (exact, tolerance 0), and can swap the accessor with `COEF_PAIR_ACCESSOR`. Run recipe: see the header of the test file (a throwaway environment with `Pkg.develop(path=".")` and `JSON3`; `--project=.` and `--project=test` both fail on a clean checkout). The R half needs `R_LIBS` set to the library that holds the gllvmTMB build.

No file under `src/` changed. There is no engine defect to fix, so the "red" is a mutation red, not a failing real behaviour:

| Accessor under test | Result |
|---|---|
| `always_empty` (control) | 5 pass, 2 fail of 7 (fails `y ~ 1` and `y ~ 1 + x`; passes the 0 = 0 pair) |
| `spurious` (control, one extra coefficient) | 4 pass, 3 fail of 7 (fails all three pairs) |
| `real` (`StatsAPI.coef`) | 7 pass, 0 fail of 7 |

Logs: scratchpad `c2/post-coef-empty-evidence/red_green_{always_empty,spurious,real}.log`. About 25 s per run.

To make B a receipt, the follow-up must move this logic into the batch pair against the frozen oracle, not reuse the Totoro library used here (section 9).

## 8. PR #1063 review

What it is: draft PR, head `e9ed941b9`, one commit on `3c94a76f7`. Four files: the receipt JSON (`julia_surface` line 633 to 652; `numeric_note` now opens "DIAGNOSTIC, NON-BINDING"), `tools/core070_postfit_p1_receipts.py` (same two strings), a check-log note, and `tools/test_postcoef_receipt_guard.py` (7 checks, 6 negative controls). Verdict fields are untouched: the receipt still reads `verdict: PASS`, `evidence_kind: numeric_r_vs_julia`, `harness_pass: true`.

Can it be brought current? Yes. `git merge-tree` reports no conflicts, and a real merge of `origin/main` (`16a7f08df`) into the head in a scratch worktree is clean; the result differs from main in exactly those four files. Results on the merged tree, run locally:

- `tools/test_postcoef_receipt_guard.py --root .`: 7 checks and 6 controls pass.
- `tools/core070_postfit_p1_receipts.py --check`: CORE070_POSTFIT_TWINS_CURRENT 7 twin rows, 52 rows.
- `tools/core070_postfit_p1_receipts.py --apply-integer-equality`: idempotent, `git status` clean afterwards.
- `tools/true_parity_assemble.py --check`: ASSEMBLE_OK 317 rows current.
- `tools/test_true_parity_assemble.py`: ALL PASS. `tools/test_core070_postfit_recount.py`: pass. `node tools/test_true_parity_check.mjs`: all negative controls passed.
- `tools/true_parity_fixture_receipts.py --check`: FAILS, STALE for the three `fixture-twins/zi/*.json` receipts.

The failing checks:

- negative-controls: red only at `true_parity_fixture_receipts.py --check` (the zi hashes). It fails the same way on `origin/main` `16a7f08df`, so #1063 did not cause it. It is not yet fixed on main: PRs #1046 and #1065 both refresh those hashes, both are still open, and they overlap, so one should merge and the other close. After either lands, updating #1063 should turn negative-controls green.
- Shard 2/4: `test/test_bootstrap_sigma_r.jl:981`, testset 25 "has_diag K=2 built-in refit reaches the optimum", 11234 passed and 1 failed. Main has `afabc24c8` ("testset 25 no longer gates on the fb.converged flag"), which the merge brings in. Not re-run here (a full shard is about 1.5 h).
- All other checks on the head passed (shards 1, 3, 4, both P1 twin jobs, guard, Documenter).

Findings on the PR itself:

1. The label says non-binding while the receipt's `verdict` still says PASS and `evidence_kind` still says `numeric_r_vs_julia`. The check-log note admits this. This is the open owner decision (packet step 5b). It depends on A, B or C, so it should not be settled inside #1063.
2. The guard is not wired into CI and needs `--root`. G1 passes on either wording ("real fitted r" or "synthetic mock"), so it cannot catch the contract changing. Control N6 contains the clause `or "synthetic" not in r_call`, which makes it pass vacuously once the receipt stops saying synthetic. Neither blocks the PR; both should be tightened if the guard is kept.
3. Receipt bytes change (sha256 `72fdb273...` to `2f108b0a...`). Only `docs/dev-log/audits/2026-10-19-p1-thirteen-row-source-audit.md:38` pins the old bytes; it is a historical record, correctly left alone.
4. The PR body says "Not tested in this recovery pass". The runs above now cover that.

Decision for #1063: it can be updated and is safe to merge as bookkeeping once negative-controls is green, because the label is true under A, B and C and no verdict moves. What remains for ruling: (i) A, B or C; (ii) whether the receipt's `verdict` and `evidence_kind` should change with that ruling; (iii) which of #1046 and #1065 merges first. I did not run `gh pr update-branch`, because it pushes to the remote and nobody authorised a push.

## 9. Unverified

- The N2 ruling text itself (wave-plan-2026-10-03.md line 17, lane kit) was not read. I rely on the quotation in GATES.md and on D-319 as a general approval.
- The diagnostic R fits used the Totoro library built from the clone above. The committed receipt used the Mac oracle at `/Users/z3437171/local-scratch/a3cov-oracle/build/library`. I did not show the two binaries are the same. The source tree matches the pin (131 files); the install log ends `DONE`. `R/vcov-coef.R` in the Mac oracle source (`/Users/z3437171/local-scratch/a3cov-oracle/build/source`) has the pin's blob `a707d537...`, so the function is the same in both trees; I did not hash the rest of the Mac tree or the Mac installed library.
- The R diagnostic is one run, one seed, p = 5, K = 1, Gaussian, and measures only length and names. It says nothing about `vcov`, `confint` or non-Gaussian families.
- I did not search the pin's tests exhaustively. A grep of `X_fix_names` and `length(coef(` in `tests/testthat` found no test of the empty branch.
- Julia lengths for non-Gaussian families and for `GllvmFit` objects built by the REML bridge path (`src/bridge.jl:1085`, `β = nothing`) were not run through the pair.
- Whether the checker accepts a `[0, 5, 6]` comparison as one integer entry or needs three entries is untested; I only called `degenerate()`.
- Shard 2/4 and the full Julia suite were not re-run after the merge. Only the Python and node true-parity checks were.
- Estimates were stated after the fact. Each Totoro R run took under a minute of wall time at one thread (a verifier measured 1.8 s). Julia runs took about 25 to 30 s on a quiet machine and 1 to 1.7 minutes at a Totoro load near 360, after a one-off instantiate. All were far under the 3 h line.
