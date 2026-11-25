# After-task: bridge_fit per-route options allow-list (#936)

## Goal

Each `bridge_fit` route rejects unknown `options` keys with `ArgumentError`
naming the key and the route. Lists derived from keys read in tests on each
route; multivariate precision allow-list unchanged in `bridge_precision_multivariate.jl`.

## Change

- `src/bridge.jl`: `_bridge_assert_options_allowed` and route-specific allow-lists
  for one-part (incl. covariate sub-route), mixed-family, gaussian sources, and
  phylo diagnostic (`phylo_model` only).
- `test/test_bridge_options_allowlist.jl`: regression coverage including `row_eff`,
  `reml`, `ci_nboot` on disallowed routes.

## Evidence

- Head: branch `cursor/fix-936-allowlist-1f20` after commit on 2026-11-22.
- `test/test_bridge_options_allowlist.jl`: **14 / 14** pass.
- Bridge test batch (14 files): **704 / 704** pass (local Julia 1.13.4).
- Full `Pkg.test()` not run (cloud time budget).
- Overlap: independent of draft PR #922 (phylo-arg / reml gate / non-Dict); reml on
  non-Gaussian routes is rejected here via allow-list, not `_bridge_check_reml_route`.

## Rose

Julia-only behavioural gate; does not claim R twin Δ or ledger promotion.