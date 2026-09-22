# GOAL: GLLVM.jl lane speed78-20260919 (Fable speed plan steps 7 and 8; GitHub name GLLVModels.jl)

Immutable for the run. Re-read at the top of every arc.

## Mission
Profile first, on two fixtures: the per-site Poisson Laplace gradient (p = 20/50) on the current code,
where the mode hoist, the workspace and only_fg! already landed in core070 and only_fg! bought nothing
because value and gradient each still run their own per-site Newton; and the grouped 200x5 GLMM that
Latte beat 12x (0.192 s ours vs 0.015 s). Settle the step-8 dispute with the per-EM-iteration wall at
p = 200/1000/5000. Then land only the changes the profiles rank, each an identity gated against
origin/main, tests first.

## Invariants (never violated)
- Profile first: no src/ edit until leaf-S4 has passed and its tables are in the checkpoint (gate G4.5).
- Every src/ change is an identity: gradient rtol 1e-8 at 50 random theta (rtol 1e-14 with the cause
  named if a reduction order changes); fitted params and logLik rtol 1e-6 vs origin/main (69a69b0a0);
  logLik vs frozen gllvmTMB 0.7.0 at 1e-6..1e-9 on the ten grid cells; Optim f/g/iteration counts not
  above 31/92/140 at p=5/20/50; marginal loglik rtol 1e-12 and E-step moments rtol 1e-10 on the phylo
  path; never widen a tolerance (repo AGENTS.md).
- The orphaned test/test_poisson_grad_perf.jl (BASELINE_LOGLIK -14604.017303313138, atol 1e-8) is wired
  into test/runtests.jl by S6 and must hold.
- No DifferentiationInterface, no Enzyme, no Mooncake, no ForwardDiff compat lift in this lane.
- Compute: Mac Studio, JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1; each run under 30 min (D-139); the
  full ten-cell paired grid (30-60 min) only if the p=50 cell moves, with the estimate stated first.
- Never push, never merge, never open a PR from inside this lane. Never stage files you did not create.

## Definition of done
- leaf-S4 gates PASS under
  `node ~/shinichi-brain/skills/unlazy/scripts/gate-check.mjs --reverify --timeout 1800 .unlazy/julia-speed-20260919/gates/leaf-S4.md`;
  then leaf-S6 (and leaf-S7 / leaf-S7b if the profiles name them; written before each starts) PASS the
  same way; full `Pkg.test()` green.
- bench/results/ holds the three TSVs with git SHA, Julia version, BLAS config, threads in the header.
- LOOP/lanes/speed78-20260919/checkpoint.md states TRUTH LIVES IN with paths and the branch SHA, and
  names which route owns the Latte 12x.

## Arcs
See arcs.md. Order: S4 (three profiles) -> checkpoint -> S6 (per-site changes the profile ranks) ->
S7b (grouped route, only if S4 puts the 12x there) -> S7 (phylo EM, only what the wall names) -> verify.

## Gates (STOP and surface)
A gate failing twice on the same cause; a proposed edit outside the OWNS list; any compute above 30 min;
a design choice between genuinely different gradient architectures (surface, do not pick).

## Pre-authorised
Scoped edits under bench/, the OWNS src files, new test files and the one runtests.jl include; local
Julia runs under 30 min; `Pkg.test()`; local commits on claude/lane-speed78-20260919; .unlazy ledgers.

## Resume order
LOOP/lanes/speed78-20260919/GOAL.md -> checkpoint.md -> ultra-plan.md -> AGENTS.md (repo) ->
docs/dev-log/core070/poisson-perf-repair-notes.md -> .unlazy/julia-speed-20260919/GATES.md and gates/.
