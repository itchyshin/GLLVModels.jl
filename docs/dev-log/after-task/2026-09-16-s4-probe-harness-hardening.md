# After-task: S4 probe harness hardening (DRAFT #409)

**Date:** 2026-09-16  
**PR:** [#409](https://github.com/itchyshin/GLLVModels.jl/pull/409) (stay DRAFT)  
**Paste:** not set; no live probe

## Scope

- IN: Julia-side paste-gated harness prep for gllvmTMB #1283 isolated runner.
- OUT: probe execution, merge, GOAL complete, gllvmTMB edits.

## Landed

- `--dry-run` on `run_s4_public_phylo_dep_probe.jl` (preflight only; `S4_PUBLIC_PHYLO_DEP_PREFLIGHT_DRY_RUN_OK`).
- `s4_public_phylo_dep_preflight!` + summary; actionable `ArgumentError` hints.
- Receipt template: `docs/dev-log/after-task/TEMPLATE-s4-public-phylo-dep-probe-receipt.md`.
- Runbook checklist items for dry-run + template.

## Checks

- `julia --project=. test/test_destination_b_s4_public_phylo_dep_probe_harness.jl` → 13/13 pass.
- Driver without paste → exit 2 (usage printed).

## Rose

- Paste gate unchanged; no capability promotion claimed.

## Follow-up

- Maintainer paste `S4 probe yes` before live probe.
- CI on push (8/8 Julia; frozen-R advisory OK).
