# After-task: docs #1487 — ordination vs loading rotation pairing

## Goal

Document that default `ordination` uses principal rotation of centred site scores,
while `getLV` / `getLoadings` / `rotation` use the loading SVD; biplots must pair
coordinates from the same path.

## Files changed

- `docs/src/tutorial.md` — ordination vs post-fit rotation section
- `src/ordination.jl`, `src/postfit.jl` — docstrings (`ordination`, `rotation`,
  `getLoadings`, `getLV` Gaussian + Binomial)
- `test/test_ordination.jl` — pairing regression (#1487)

## Tests

Julia 1.10.14, `JULIA_NUM_THREADS=1`:

```sh
julia --project=. --startup-file=no -e 'include("test/test_ordination.jl")'
```

39 passed, 0 failed, ~45s (ordination + ordiplot + design-forwarding block).

## R-parity / JET / Aqua

Not applicable (docs + docstrings only).

## Rose verdict

PASS — tutorial and accessor docstrings now match implemented rotations; no
user-facing API or default changes.