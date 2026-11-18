# N8: memory design for bootstrap_Sigma_r (keep_draws, large n_boot)

Status: design note, no code change, no row status change. Base: origin/main 3c94a76f7.
Source: src/bootstrap_sigma_r.jl (line numbers below are on that commit).

## 1. Where memory goes (code reading)

- Line 331 `Ys = simulate(rng)`: all `n_boot` simulated datasets are materialised up
  front (R-style single batch). Cost: n_boot * p * n * 8 bytes (GllvmFit, setup at
  lines ~140-160; TwoLevelFit lines ~170-190 uses ntot in place of n).
- Line 333 `draws`: one `Dict{Symbol,Any}` per replicate, ALWAYS built, whatever
  `keep_draws` says. Per replicate, per tier: Sigma (p^2) + R (p^2) + communality (p)
  Float64. GllvmFit (tier B only): (2p^2+p)*8 bytes. TwoLevelFit (B, W, ICC_site):
  (4p^2+3p)*8 bytes.
- Line 396 `keep_draws ? NamedTuple[nt(d, keys_pe) ...] : nothing`: the NamedTuples
  reference the same arrays, so keep_draws=true adds almost no memory; it only
  decides whether the result still holds `draws` after return. keep_draws=false does
  NOT reduce the peak during the run (draws are needed for aggregation at step 7).
- Aggregation (lines ~345-358) builds a fresh `vals` vector per matrix cell by
  scanning all draws through Dict lookups: O(p^2 * n_boot) lookups. Transient only.

## 2. Measured (Mac, 1 thread, OPENBLAS_NUM_THREADS=1, Julia 1.10, local; scripts in the session scratchpad)

Fit: fit_gaussian_gllvm, p=30, n=200, K=2, has_diag=true, n_boot=999, progress=false,
real refits (no _refit seam), 0 failed replicates.

| run              | wall      | s/rep | maxrss before | after | delta | result summarysize |
| keep_draws=false | 1240.2 s  | 1.242 | 730 MiB       | 973   | 243   | 0.04 MiB           |
| keep_draws=true  | 1245.5 s  | 1.247 | 720 MiB       | 982   | 262   | 14.14 MiB (draws 14.09) |

Components, p=30, n=200: Ys batch 45.78 MiB (matches 999*30*200*8); draws 14.09 MiB
(matches 999*1830*8 = 14.6 MB). Peak delta (243-262 MiB) is about 4x the retained
60 MiB: the rest is refit transients and GC slack. maxrss is a high-water mark and
the two runs ran concurrently, so +19 MiB for keep_draws is indicative, not exact.

Wall time is the binding constraint, not memory. A second probe at p=100, n=300,
K=3, n_boot=4: 78.7 s/replicate (fit once: 53.9 s), so 999 replicates would take
about 21.9 h (extrapolation from 4 replicates; one probe, treat as order of magnitude).
Memory delta there was 22 MiB.

## 3. Projection at n_boot = 999 (analytic, same formulas as measured)

| p, n        | Ys        | draws GllvmFit | draws TwoLevelFit |
| 30, 200     | 46 MiB    | 14 MiB         | 28 MiB            |
| 100, 300    | 229 MiB   | 153 MiB        | 307 MiB           |
| 300, 1000   | 2287 MiB  | 1374 MiB       | 2751 MiB          |
| 1000, 2000  | 15244 MiB | 15251 MiB      | 30510 MiB         |

Memory only becomes a problem at p of about 300 or more, where 999 refits would be
far beyond practical wall time anyway (>3 h per fit extrapolates well past a day).
So memory is a secondary risk; do not add complexity for it in v1.

## 4. Proposed defaults and changes (none implemented here)

1. Keep `keep_draws = false` (matches R, result stays small: 0.04 MiB vs 14 MiB).
   Document that keep_draws=true costs about nb*(2p^2+p)*8 bytes per tier set and
   does not lower the peak when false.
2. Keep `n_boot = 999` default (R parity; the warning at line 292-295 already
   protects against small values). Add a one-line docstring note that cost scales
   with refit time: 1.24 s/rep at p=30, 78.7 s/rep at p=100.
3. Optional, behaviour-preserving, low priority: generate each Y lazily inside the
   loop instead of the batch. The refit does not consume `rng`, and both simulate
   functions consume it strictly in replicate order, so the stream and every draw
   would be bit-identical. Saves the Ys term (46 MiB at p=30; 2.3 GiB at p=300).
   Needs a regression test comparing draws (seeded) before and after.
4. Optional: replace the per-replicate Dict by preallocated arrays (n_boot x cell)
   per entry; cuts Dict overhead and the O(p^2 * n_boot) lookup aggregation. Not
   needed at measured sizes.
5. Progress: `progress = true` emits one @info per replicate (line 336), 999 lines.
   Suggest keeping true but logging every max(1, n_boot/20) replicates plus the
   last, or using ProgressMeter. Either changes visible output, so it needs agreement.

## 5. Needs maintainer agreement

- n_boot = 999 as the Julia default (currently mirrors R; also the default any user
  hits on a slow fit; one probe says a p=100 call costs about 22 h).
- progress default (true, per-replicate @info) and whether to thin it (item 5).
- Whether lazy Y generation (item 3) is wanted now or deferred; it touches the
  driver and needs the bit-identical regression test.
- Whether to add a documented wall-time warning for large p (not proposed as code;
  a time estimate before the run is the existing policy for users).
- n_cores > 1 stays refused (line ~278); parallel replicates would change the memory
  profile (n_cores simultaneous refits) and the RNG stream; out of scope.
