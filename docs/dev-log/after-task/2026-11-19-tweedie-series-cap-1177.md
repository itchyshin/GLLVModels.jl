# After-task: Tweedie series cap (#1177)

**Date:** 2026-11-19. **Branch:** `cursor/tweedie-series-cap-1177-8da0`. **Issue:** #1177 (refs only).

## Goal

`tweedie_logpdf` must agree with the exact Dunn-Smyth series (and, at p = 3/2, the Bessel closed form) at small dispersion instead of returning a value that is too low.

## Change

- `src/families/tweedie.jl`: `_tweedie_series_cap(j)` replaces the fixed `cap = 5000` in `_tweedie_logA`; `clamp(ceil(10 sqrt(j) + 100), 5000, 400000)` with `j` the index of the largest term.
- `src/families/aghq_tweedie.jl`: `_tweedie_logA_ad` (the ForwardDiff copy that mirrors the same window) uses the same cap.
- `test/test_tweedie.jl`: new testset against the p = 3/2 closed form (phi 1e-4 to 1e-7, mu = y and mu != y), AD copy against the Float64 series, and the cap function.

## Evidence

See `docs/dev-log/check-log.md` entry of 2026-11-19. New testset fails 9 of 12 on the old source and passes 12 of 12 on the fix; `test_tweedie.jl` 26/26.

## Not done / limits

- Below phi about 1e-9 the 400000-term bound still cuts the series; this is silent, as before. Raising it makes one density call at extreme `y` cost seconds.
- The reported `loglik` of fits whose dispersion runs to zero changes; those fits were not re-examined.
- Full `Pkg.test()`, JET and Aqua were not run. No tolerance or expectation was changed.
- Rose verdict: not run.
