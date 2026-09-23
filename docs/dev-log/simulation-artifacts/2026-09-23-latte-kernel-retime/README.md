# Latte-kernel after #448: identity + A1 retime receipts

Tip for walls: `4e976e259` (`origin/main` after #449 / #451).

## S4 identity

S4 identity TSV + Totoro log (`s4-identity-totoro.log`): **26/26** PASS
(rtol 1e-8). Keyword `diag_precision_kernel` default remains `false`.

Oracle note: `~/local-scratch/latte-prerun-20260918/runs/f3_ours_glmm.jl` is a
scalar 1-D Laplace path; it is not the same problem as the joint
`m`-dimensional CHOLMOD path in `joint_grouped_laplace_loglik`.

## A1 paired OFF vs ON walls (Totoro, 2026-09-23)

Harness: `bench/latte_gap_retime/off_on_wall.jl` (also banked under
`docs/dev-log/evidence/2026-09-23-latte-off-on-wall/`).

| cell | OFF (s) | ON (s) | × | ≥1.5× |
|---|---:|---:|---:|---|
| `glmm_200x5` | 0.587 | 0.203 | 2.90 | PASS |
| `glmm_5000x3_g500` | 10.879 | 9.245 | 1.18 | FAIL |

Identity this run: `|Δll| = 0` both cells. Threads 4 / BLAS 1 / taskset 0–15 /
5 warm timed reps.

**Verdict: KEEP opt-in** (default OFF). Large cell below the dual-fixture 1.5×
bar; no default-ON PR from this packet. No public speed claim.

Files: `latte_off_on_wall_4e976e259.tsv`, `a1-off-on-wall-totoro.log`,
`a1-meta_4e976e259.txt`.
