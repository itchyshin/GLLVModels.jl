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

## Gate hardening continuation — 2026-09-20

The source gate now derives its checked Markdown routes from the literal
`makedocs(pages = [...])` list in `docs/make.jl`, requires each listed route to
exist, and checks the public `README.md` too. A second mode scans visible text
in the generated HTML after Documenter expands `@docs` blocks, so public
docstrings are covered by an observable rendered check. The Documenter workflow
runs the source gate before Julia setup and dependency installation, builds
without deployment, runs the rendered gate, then deploys only the checked site.

Additional verification:

- `python3 -m unittest tools.tests.test_reader_surface` — 14 tests passed.
- `python3 tools/check_reader_surface.py` — 32 source files passed (README plus
  all navigation routes).
- Local Documenter generated 32 HTML pages under `docs/build/1`; `python3
  tools/check_reader_surface.py --rendered docs/build/1` passed.
- Julia parsed both `docs/make.jl` and `docs/deploy.jl`; `git diff --check`
  passed.
