# F-TEMPORAL-1FO

Verdict: STILL_OPEN

Worktree: `/Users/z3437171/.codex/worktrees/p1-isdm-20261019/GLLVM.jl`
HEAD: `5cfc6754212588222779ccfd66ad7000375cf8d5` (unchanged from the 2026-10-20 note)

## Question

Does existing test code already assert every item on the F-TEMPORAL-1FO gate
line (`docs/dev-log/core070/true-parity-latest/GATES.md` line 1168): "Gaussian
rank-1 latent score, AR1/OU, first order"?

This note rereads the gate line and the three named test files and answers
item by item. It does not tick the box. No Julia process was started.

## What each file asserts

`test/test_temporal_lowrank.jl` fits `temporal_latent` at `d = 1`,
`unique = false`, looping `structure in (:ar1, :ou)` (lines 18-44). It asserts
the marginal nll value (`atol = 1e-10`, `rtol = 1e-12`) and its ForwardDiff
gradient (`atol = 1e-8`, `rtol = 1e-9`) against a dense block reference. The
file has no `getLV` call and no latent-score assertion.

`test/test_temporal_helpers.jl` testset "temporal latent scores
(ar1-methods.R:20, engine.R:119)" (lines 276-297) asserts `getLV` scores
against a dense conditional mean at `1e-10` on one fit,
`sim_ar1__latent_unique__ar1`. That cell is AR1 with `unique = true`. The file
has no `:ou` occurrence, so no OU latent score is asserted there. The same
testset asserts `getLV === nothing` for AR1 indep and dep fits.

`test/test_temporal_fit_receipts.jl` lines 137-148 assert `getLV` scores
against R's `report_z_temporal_state` at `1e-10` for every fixture cell with
`mode == "latent"`. The fixture (`test/fixtures/temporal_p1/fits.toml`) records
`sim_ou__latent__ou` and `sim_ou__latent_unique__ou` with `mode = "latent"`
and `report_z_temporal_state` present, so OU latent scores are asserted there.
The same 21-cell loop also fits indep, dep, and latent-unique cells and checks
objectives, logLik, and parameter gaps. The 2026-10-20 recheck left this mixed
loop outside the gate, and this note keeps it there: the execution contract
(`docs/dev-log/plans/2026-10-19-p1-approved-execution-contract.md` lines 59,
91-94) requires the gate's OWNS, CHECK, EXPECT, and EVIDENCE to be named
before dispatch, and adopting the loop as this gate would be a new signature.

`test/test_temporal_oracles.jl` has no `getLV` or score assertion (grep over
the file returns nothing).

## Item-by-item answer

- Gaussian. Asserted. Every temporal fit in the three files is a Gaussian
  marginal fit.
- Rank-1. Asserted where scores are checked. The low-rank file runs at
  `d = 1`; the latent fixture cells carry one factor (3 loadings over
  3 traits).
- AR1 latent score. Asserted. Helpers checks it against a dense conditional
  mean; receipts checks it against R's reported state.
- OU latent score. Not asserted in any in-scope file. It appears only inside
  the excluded receipts loop.
- First order. The first-order quantities (marginal nll and gradient) are
  asserted for both structures in the low-rank file against a dense block,
  and in the oracles file against R's `fn` and `gr`. A first-order rank-1
  score compared with R's reported conditional state exists for AR1 and OU
  only inside the excluded receipts loop.

## Verdict

Not every item on the gate line is already asserted. No in-scope file asserts
an OU rank-1 latent score, and the AR1-plus-OU first-order score comparison
with an R receipt lives only in the mixed 21-cell loop that the gate record
excludes. The box stays open.

Nothing was edited besides this note. GATES.md, the tests, the fixtures, and
the scoreboard were left untouched. Nothing was committed.
