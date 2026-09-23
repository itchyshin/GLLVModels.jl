# Latte OFF vs ON walls (measure-only · 2026-09-23)

Tip SHA: `4e976e259` (`origin/main` after #449 / #451).
Host: Totoro (`taskset -c 0-15`; `JULIA_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=1`).
Script: `bench/latte_gap_retime/off_on_wall.jl`.
Default flip: **NO** (this packet does not change `src/`).

## Speedup table (median of 5 warm timed reps)

| cell | wall OFF (s) | wall ON (s) | OFF/ON | ≥1.5× gate |
|---|---:|---:|---:|---|
| `glmm_200x5` | 0.587 | 0.203 | **2.90×** | PASS |
| `glmm_5000x3_g500` | 10.879 | 9.245 | **1.18×** | FAIL |

Timing scope (identical arms): real `fit_gllvm` Poisson,
`GroupingTerm(:unit; mode=:indep)`, `warm_start_inner=true`; arms differ only by
`diag_precision_kernel` (which also drives `diag_precision` +
`reuse_identical_hf_ho`). Fixture md5 `glmm_200x5` =
`60a293d6c45e36cc458adc5df2458b30`.

## Identity

- This run: `|Δll| = 0` on both cells (OFF vs ON).
- Prior S4 certify (#449): **26/26** Totoro
  (`docs/dev-log/simulation-artifacts/2026-09-23-latte-kernel-retime/`).

## Recommendation for Shinichi (G0 still open)

Keep opt-in (default OFF). Small cell clears the ~1.5× gate; the large
certify cell does not (1.18×). Mixed clearance is not enough to flip the
default. Keyword stays available; no README/NEWS speed claim from this packet.

## Files

- `latte_off_on_wall_4e976e259.tsv`
- `wall_4e976e259.log`
- `meta_4e976e259.txt`
