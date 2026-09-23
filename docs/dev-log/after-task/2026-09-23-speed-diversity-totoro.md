# After-task: GLLVM speed-board diversity cells (2026-09-23)

Active lenses: Shannon, Ada, Rose (perspectives). Spawned subagents: none.
Lane: `cursor/speed-diversity-20260923` · tip measured `4e976e259`.

## Scope

Expand GLLVModels three-package speed board from 10 toward 20 with NEW
distinct kinds not already on the board. Totoro only (≤16 cores; used
`JULIA_NUM_THREADS=4` + `taskset -c 0-3`). No heavy local Julia. No Latte
default-ON. No README/NEWS speed claim.

## Outcome

1. Diversity plan sibling not ready; selected 8 kinds from board + ultra-plan
   hints (ordinal, ZIP, ZINB, phylo Poisson GLM, SPDE spatial, concurrent LV,
   missing-data Gaussian, medium profile CI).
2. Added `bench/speed_board_diversity.jl` (Julia tip-abs walls; no R H2H:
   no cheap paired script for these kinds).
3. Totoro @ `4e976e2`: **8/8** `has_receipt` after one missing-mask retry
   (first attempt zeroed masked Y → PosDefException; fixed by leaving Y and
   using residual σ=0.8).
4. Board plan status **10/10 → 18/18**; two kinds remain to reach 20.

## Walls (median of 3 warm reps; J=4 OB=1)

| cell_id | wall_s |
|---|---:|
| `gllvm-ordinal-unstruct-small` | 0.1443 |
| `gllvm-zip-unstruct-small` | 0.4358 |
| `gllvm-zinb-unstruct-small` | 2.7330 |
| `gllvm-phylo-pois-glm-smoke` | 0.0187 |
| `gllvm-spatial-spde-gauss-smoke` | 0.1407 |
| `gllvm-concurrent-pois-small` | 0.2524 |
| `gllvm-missing-gauss-mask` | 0.0277 |
| `gllvm-profile-ci-medium` | 3.8452 |

Evidence: `docs/dev-log/evidence/2026-09-23-speed-diversity-totoro/`.

## Rose

Tip-abs only. No fake before/after. No cross-host speedup. Latte kernel
stays default OFF. Soft fence: these are board inventory walls, not public
speed claims.

## Next

Two more distinct kinds to hit 20 (e.g. Beta/Gamma analytic unstruct, or
hurdle-Poisson if fast). Latte kernel arc remains separate.
