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
5. Board flipped three cells to `has_receipt`. `gllvm-profile-ci-small`
   remains `needs_run`.
6. No README/NEWS speed claim. No `diag_precision_kernel` default flip.

## Checks

- Totoro logs under `docs/dev-log/evidence/2026-09-23-speed-board-totoro/`
- TSV `bench/results/board_gllvm_20260923_8d58a0c94.tsv`
- Board plan `docs/dev-log/plans/2026-09-23-three-package-speed-board.md`
  counts: GLLVM 9 has_receipt / 1 needs_run

## Rose

Absolute tip walls only (no fake before/after). Analytic-vs-finite ratios
are within-run, same host. Cross-host comparisons forbidden. Latte kernel
stays default OFF.

## Next

Optional: `PROFILE_CI=1` on `8,40,1` for `gllvm-profile-ci-small`.
