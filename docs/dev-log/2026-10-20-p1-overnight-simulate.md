# P1 overnight, simulate rows

HEAD in `/Users/z3437171/.codex/worktrees/p1-isdm-20261019/GLLVM.jl` is `5cfc6754212588222779ccfd66ad7000375cf8d5`. `git rev-parse HEAD` returned that commit before this note was written. The commit was left there. No source file was edited. No tolerance changed. `GATES.md` was left unchanged. No Julia process was started. No full suite was launched.

## simulate gllvmTMB_multi: LEFT

`CORE070-NAMESPACE-SIMULATE-MULTI-NATIVE.json` passes as a registration check. The pinned R method `S3method(simulate,gllvmTMB_multi)` is registered and defined, and the Julia symbol `simulate` exists, is exported, and is a `Function`. The receipt category is `registration_twin`. Semantics and numeric output were not compared. The 7 of 40 seed count belongs to `POST-SIMULATE-DEFAULT`.

No judged Monte Carlo run was authorized for this row. Proposal A in the 2026-10-07 namespace plan remains unsigned, including its four-batch scope, error-budget thresholds, and conditional-control rule. N4 in `GATES.md` still allows a Monte Carlo tolerance only when the rule is stated before the run. That judged run was not started.

The Poisson method in `src/simulate_fit.jl` draws a fresh standard normal latent score and adds the fitted latent mean before sampling the counts. The recorded Monte Carlo twin for the other row uses that redraw. No existing comparison on this namespace row failed, so the source was left as it is.

## POST-SIMULATE-DEFAULT: LEFT

The row stays PARTIAL. The 7 of 40 failures stay on the record. The row was not marked passed.

`simulate_default.json` passes at its recorded seed. B is 400. There are 25 moments. The largest absolute difference divided by the pre-registered tolerance is 0.962. The discrimination control, R draws with `condition_on_RE = TRUE` against the Julia default, fails 10 of 25 moments. The four files named in that receipt have the same sha256 now as in the receipt: `test/fixtures/mc_simulate_p1.toml`, `test/fixtures/gen_mc_simulate_p1.R`, `test/test_mc_simulate_p1.jl`, and `test/mc_simulate_helpers_p1.jl`. `git log` from the receipt commit `cf289fca8ec80a968f2b8fe271f3fce57b6eaeeb` to this HEAD lists no commit on those four files or on `src/simulate_fit.jl`.

`HELD_MC_TWINS` in `tools/core070_postfit_p1_receipts.py` records that 7 of 40 alternative Julia seeds failed the same rule against one shared R realization. `GATES.md` N4 keeps that hold and leaves the rule unchanged. The 2026-10-19 source audit found no seed-level raw file for those 40 streams. Those 7 failures are the expected leftovers from that seed check. They were not treated as a defect in the existing test, and that test was not re-run.

The policy case `CORE070-POSTFIT-SIMULATE-DEFAULT-NATIVE` also stays as recorded. Its batch check passed. The R oracle stores `simulate_condition_on_re_default` as true because `tools/core070_postfit_policy_batch.R` sets that flag with `isFALSE` on the formal, and the formal default is `FALSE`. The Julia batch passes when that flag is true and `simulate` has no `condition_on_RE` keyword. The case receipt's comparand sentence says Julia always conditions on the fitted mode. The Poisson method redraws the latent score, and the Monte Carlo twin above is the comparison that uses that redraw. The sentence was left in the receipt. No source file was edited to match it.
