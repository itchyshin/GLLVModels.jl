# After-task: GLLVM Phase B first-wave Totoro (2026-09-23)

Active lenses: Shannon, Ada, Rose (perspectives). Spawned subagents: none.
Lane: `cursor/speed-diversity-20260923` · tip measured `4e976e259`.

## Scope

Unlock Phase B report fill for GLLVModels: bank plan
`2026-09-23-speed-report-20x3-diversity.md` §4.1 first-wave **10** cell_ids
toward report aim ~20. Totoro only (`JULIA_NUM_THREADS=4`,
`OPENBLAS_NUM_THREADS=1`, staggered under load≈229). Julia abs OK. No Latte
default flip. No heavy local Julia. No README/NEWS speed claim.

## Outcome

1. Ran `bench/speed_board_firstwave.jl` for all 10 plan cell_ids on Totoro.
2. **10/10** `ok` receipts at tip `4e976e2`.
3. Filled report scaffold slots **11–20**; board status → report **20/20**.
4. Honesty fences: `gllvm-profile-ci-glmm` used Wald fallback (profile not
   admitted on grouped fit); `gllvm-latte-gap-200x5` is labelled ratio vs
   banked Latte 0.015 s only (25.53× under load; quiet-host hist ~8×).

## Walls (median of 3 warm reps)

| cell_id | wall_s |
|---|---:|
| `gllvm-binom-glmm-200x5` | 0.2369 |
| `gllvm-gauss-lv-t4-p20n500` | 0.0088 |
| `gllvm-pois-lv-t4-p20n500` | 1.6459 |
| `gllvm-nb2-lv-t4-p20n500` | 4.8812 |
| `gllvm-gauss-unstruct-p50n2k` | 0.5197 |
| `gllvm-gauss-phylo-fit-p200` | 0.0418 |
| `gllvm-profile-ci-glmm` | 0.0091 (Wald) |
| `gllvm-latte-gap-200x5` | 0.3830 (25.53× labelled) |
| `gllvm-pois-phylo-small` | 0.0191 |
| `gllvm-spatial-gauss-small` | 0.0130 |

Evidence: `docs/dev-log/evidence/2026-09-23-speed-firstwave-totoro/`.

## Rose

Tip-abs / labelled only. No fake before/after. No cross-host ×. Latte kernel
stays default OFF. Report aim for GLLVM met at 20; DRM 11–20 still empty.

## Next

**MERGED** as squash `36479c28dde2f44f7e72821e546c3eee5f45c0f3` on
`origin/main` (PR #454, head `1022e3a74`, 2026-09-23T19:08:38Z).
Merge landed before all required Julia shards settled; Frozen-R advisory
settled FAILURE (allowed). Main Documenter green; main CI was still
in progress at receipt write. Latte OFF→ON remains other agent. DRM
Phase B sibling still owns DRM slots 11–20.
