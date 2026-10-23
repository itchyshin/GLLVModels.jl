# After-task: GLLVM speed-board Totoro cells (2026-09-23)

Active lenses: Shannon, Ada, Rose (perspectives). Spawned subagents: none.

## Scope

Fill GLLVModels three-package speed-board cells still `needs_run` after
Latte #449 (`f36def049` on main; tip measured at `8d58a0c94`). Totoro only;
no heavy local Julia. No Latte default-ON.

## Outcome

1. Inventory: four board cells were `needs_run`
   (`gllvm-gauss-unstruct-small`, `gllvm-gauss-unstruct-large`,
   `gllvm-nb2-or-binom-unstruct`, `gllvm-profile-ci-small`). Latte identity
   already PASS on #449; skipped.
2. Sibling check: Latte lane held grouped `warm_identity`, not these
   `speed_bench` cell_ids. Proceeded.
3. Totoro cell 1 (`8,40,1`, PROFILE_CI=0, J=4 OB=1): Gaussian 0.0001 s;
   NB analytic 0.0222 s (9.04x vs finite); Binomial analytic 0.0130 s
   (10.84x vs finite).
4. Totoro cell 2 (`30,100,2`): Gaussian 0.0016 s; NB analytic 1.6035 s;
   Binomial analytic 2.5181 s.
5. Board flipped three cells to `has_receipt` in the first Totoro slice.
6. Follow-up Totoro `PROFILE_CI=1` on `8,40,1` @ tip `8d58a0c`:
   Poisson profileCI **1.5809 s**, NB 2.5044 s, Binomial 1.9279 s.
   `gllvm-profile-ci-small` → `has_receipt`. GLLVM board **10/10**.
7. No README/NEWS speed claim. No `diag_precision_kernel` default flip.

## Checks

- Totoro logs under `docs/dev-log/evidence/2026-09-23-speed-board-totoro/`
- TSV `docs/dev-log/evidence/2026-09-23-speed-board-totoro/board_gllvm_20260923_8d58a0c94.tsv`
- Profile log `board_profile_ci_8d58a0c.log`
- Board plan counts: GLLVM **10** has_receipt / **0** needs_run

## Rose

Absolute tip walls only (no fake before/after). Analytic-vs-finite ratios
are within-run, same host. Cross-host comparisons forbidden. Latte kernel
stays default OFF.

## Next

Soft-owed DRM Aug-24 H2H re-anchor (drmTMB already DRModels-aware on
`origin/main`); H² soft-hist same-DGP optional.
