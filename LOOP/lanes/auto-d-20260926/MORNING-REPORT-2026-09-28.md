# Morning report, 2026-09-28 (overnight run 00:10Z to about 11:00Z)

Goal: GOAL-2026-09-28-overnight.md. Written at the start and updated as arcs finish.

## Needs you
1. Review and merge #519 (green), then #521 (green), then #540 (green).
2. Approve the NB re-run estimate (about 5,500 core-h; nb-rerun-plan.md). Not submitted.
3. Julia vs R guard difference under the ridge (A1 below): which Hessian rule should both use?

## "All GitHub Actions green" (your 01:40Z request)
- #518: `main` merged again (CHANGELOG conflict from tonight's merges); lane tests pass; pushed, CI running.
- #521: `main` merged (picks up #530's Documenter fix); its two test files pass; pushed, CI running.
- At 03:05Z every run in the account was queued (main's and other lanes' too): GitHub's concurrent-job
  limit, not a failure. #518/#521 results will land whenever the queue drains; see the checks on each PR.
- The only other red check on every PR is the advisory "Frozen R 0.7.0 family smoke" job, which is also
  red on `main`. Its cells: NB2 (ours), Student-t and truncated NB2 (no owner tonight; the "main parity"
  lane logs them for you). **NB2 diagnosis:** on the pinned parity dataset both engines put 2 of 5 traits'
  dispersion at the Poisson boundary (Julia r of about 1e25 and 1e10, on `main` and on #521 alike), so the
  Julia fit honestly reports `converged = false` (T14 boundary rule) and gllvmTMB's gradient check fails
  (0.0024 > 1e-4). Log-likelihoods still agree. Not a code bug. Making it green needs **your decision**:
  replace the hash-pinned fixture with data that has real overdispersion, or accept a boundary fit in
  that assertion. I changed neither.
- Coordination: messaged "main parity" (owns the harness pins; asked me not to edit its pin files) and
  "Package completion planning" (works on DRModels.jl/drmTMB only; no collision).

## Arcs
- A0 (#518 CI): DONE. CI on `159631a5b`: all 8 test shards, Documenter and the P1 twin tests pass; only the advisory smoke job fails (as on `main`). Held commits pushed together with A1 (see A4).
- A1 (Julia vs R ridge on the same data): DONE. gllvmTMB fitted the 40 exact Julia datasets. With the
  ridge, recovery now agrees within one dataset in most cells (`bic`: Julia 1/8/8/4 vs R 1/7/8/3 out of
  10). **Last night's 8/10 vs 1/10 gap came from the two languages drawing different random datasets, not
  from the engines.** One real difference remains: at n = 120, p = 20, K = 3 (`bic_sites`, ridge) Julia
  recovers 9/10 and R 4/10, because R's guard rejects ridge fits whose Hessian is non-positive-definite
  ("unconverged") and Julia's guard does not check the Hessian. **Decision for you:** should the Julia
  guard adopt R's Hessian check, or R relax it under the ridge? Recorded in design/74 T7.
- A2 (Gaussian re-run for #519): DONE, and done in full rather than just planned, because Gaussian
  fits take seconds. On #518 + #519 in a throwaway tree, all 4,800 Gaussian datasets on uncentred data
  (trait means 3 + N(0, 1)): exact recovery **0.950** with `:bic_sites` (original grid 0.948) and 0.864
  with `:bic` (grid 0.865), 0 failures. So the Gaussian claim in #518 holds with trait intercepts; after
  #519 merges this only needs a confirming re-run on `main`. Recorded in design/74 (local commit
  `dd383ba22`).
- A5 (added overnight): DONE. The grid's binomial cells with the default ridge (1,200 datasets,
  weak loadings 0.8·N(0,1)): exact recovery 0.412 (`bic_sites`) against 0.437 without the ridge;
  K = 1 always right; K = 2 and 3 found almost only at n = 300; 706 too few, 0 too many. At weak signal
  the ridge buys safety (never over-selects), not accuracy; its accuracy gain is at strong loadings.
  **Worth a look:** the docs for `binary_ridge = 2` should say this plainly (design/74 T7 updated).

- Weak-signal caveat for `binary_ridge = 2`: added to the Julia `select_lv` docstring (local commit
  `02061d350`, pushes with the next #518 batch). **R side deferred:** a Codex lane
  (`cran-071-20260927`, CRAN 0.7.1 prep) holds a live lease on gllvmTMB `R/`, `man/`, `tests/`, NEWS
  and more. My first attempt ran past the refusal (a `;`-chained command); I reverted it at once, so
  nothing was committed. The one-paragraph roxygen change for `?select_lv` is ready to apply when that
  lease clears.

## Will auto-d work? (your 01:55Z question, answered with tonight's numbers)
Gaussian 0.95 and Poisson 0.999: yes. Negative binomial: promising (the fixed kernel picked the true
K where the old one picked 5) but unmeasured until the NB re-run. Binary: K = 1 reliably, higher K only
with strong signal or large n; it errs towards too few dimensions and reports every attempt. Built in
both packages as drafts (#518, #1324), CI green apart from the advisory smoke job.
- A3 (CI watch): RUNNING
- A4 (this report, checkpoint, final push): PENDING
