# After-task — reader-first documentation recovery

## Scope and outcome

Reworked the public GLLVModels.jl learning routes for biology PhD students.
Pages now start with scientific questions and runnable routes, distinguish
community abundance from occurrence data, and make the R/Julia relationship
optional and explicit.

## Boundaries

No model code, formula grammar, public API, numeric parity fixture, or
capability claim changed. Figures and visual redesign remain outside this
text-only slice.

## Evidence

Five reader lenses covered community data, phylogeny, bridge handoff,
uncertainty/model selection, and spatial or repeated-measure designs.

## Verification

- `python3 tools/check_reader_surface.py`
- `python3 -m unittest tools.tests.test_reader_surface`
- `env GLLVM_DOCS_DEPLOY=false julia --project=docs docs/make.jl`
- `git diff --check`

All passed. The fresh site is at `docs/build/1/index.html`.

## Follow-up

Mirror reader journeys with gllvmTMB where useful, without forcing an
identical article inventory.
