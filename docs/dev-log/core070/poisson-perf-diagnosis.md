# Poisson dense-Laplace performance diagnosis (read-only profile, 2026-09-01)

Context: bench-prerun-findings.md — Poisson 0.45–0.80x vs frozen R/TMB,
gap widening with p, while Gaussian is 20–103x faster.

## Measured (p=50, n=500, K=2 — the worst cell)
- Marginal value eval: 7.0 ms / 11.3 MB. Gradient eval: 170.5 ms / 543 MB —

> **Re-banked 2026-09-19 (lane speed78-20260919, on 69a69b0a0):** the 170.5 ms gradient wall at p=50 above predates the R2 mode-hoist repair this note prescribes (commit a04c9d26). Measured today on the current code with `bench/profile_laplace_allocs.jl` (twice): value 8.5 ms, gradient 95.6 to 98.0 ms across three runs (the cited TSV `bench/results/laplace_allocs_d807d96a7.tsv` holds 97.99 ms; `laplace_allocs_d56bccea4.tsv` 95.56 ms), of which the ForwardDiff pass ~92 ms with 383 MB of Dual allocations and the hoisted per-site mode solves ~7 ms; 13 chunk passes (chunk 12, n_theta 149). The 543 MB, 24.3x and 48x figures and the p-scaling line below carry the same provenance: pre-R2, kept for the record. The 170.5 ms figure is kept for provenance, not as the baseline.

  24.3x value cost, 48x allocations.
- 42% of gradient samples inside `_laplace_mode` (src/families/laplace.jl:60):
  the CONCRETE per-site Newton mode solve is redone from scratch once per
  ForwardDiff CHUNK pass (chunk size 12, nθ=3p−1 ⇒ 13 chunks at p=50), not
  once per gradient call. Remaining ~58%: the diffable one-Newton-step
  construction + ForwardDiff chunk machinery (src/laplace_grad.jl:62–92).
- Scaling: grad time ∝ p^1.4; grad/value ratio grows ~linearly in p
  (2.0→19.3 for p=5→40) tracking ⌈(3p−1)/12⌉; flat in n. This exactly
  explains the widening R/Julia gap: TMB's reverse tape is one pass, ours is
  ~nθ/12 forward passes each re-paying mode solves.
- LBFGS iterations also rise with p (31/92/140 at p=5/20/50) — secondary.

## Ranked repairs
1. Hand-derived analytic chain-rule gradient (implicit ẑ(θ) derivative, no
   dual chunking) — src/laplace_grad.jl:62–125. Est. 6–10x gradient at p=50,
   growing with p. Matches the roadmap's "hot scalar families → hand-coded
   chain-rule kernels" lane. (Note: docs claim Poisson-log already has a
   hand-coded implicit gradient — the PROFILED default path is the chunked
   ForwardDiff one; reconcile claim vs dispatch during repair.)
2. Hoist the concrete mode solve out of the chunk loop; cache ẑ per (θ,site)
   and thread into `_poisson_site_diffable` — est. ~1.6x at p=50.
3. Pre-allocate per-site workspaces in `_laplace_mode` (543MB churn) —
   est. 15–30% on both value and grad; repo convention.
4. `Optim.only_fg!` combined closure (value+grad passes currently separate,
   src/families/poisson.jl:249–310); likely recurs in sibling families.
5. Iteration growth with p — instrument before touching (no tolerance edits).

## Could not determine
R iteration counts (not in receipts); >p^1 exponent beyond p=50; whether
1+2+3 compounded fully close the gap.
