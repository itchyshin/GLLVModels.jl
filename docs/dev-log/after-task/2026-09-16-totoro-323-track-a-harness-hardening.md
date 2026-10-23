# Totoro #323 Track A harness hardening (DRAFT #410)

**Date:** 2026-09-16  
**PR:** DRAFT [#410](https://github.com/itchyshin/GLLVModels.jl/pull/410)  
**Paste:** not fired (harness only)

## Scope

- IN: mirror #409 harness pattern for Track A (`--dry-run`, preflight report, clearer paste errors, receipt template, runbook checklist, Julia tests).
- OUT: Totoro SSH, oracle build, `runparity.jl`, merge, goal complete.

## Checks

| Command | Result |
|---------|--------|
| `julia --project=. test/test_totoro_323_track_a_harness.jl` | (CI / local) |
| launcher without paste | exit 2 |
| `--dry-run --gllvm-root .` | `TOTORO_323_TRACK_A_PREFLIGHT_DRY_RUN_OK` |

## Files touched

- `tools/totoro323/totoro_323_track_a_harness.jl`
- `tools/totoro323/run_totoro_323_track_a_launcher.jl`
- `tools/totoro_323_track_a_launcher.sh`
- `test/test_totoro_323_track_a_harness.jl`
- `docs/dev-log/after-task/TEMPLATE-totoro-323-track-a-receipt.md`
- `docs/dev-log/plans/2026-09-16-totoro-323-track-a-runbook-paste-gated.md`
- `docs/dev-log/check-log.md`

## Follow-up

- Maintainer paste `ack Totoro D-139 #323 Track A` → Codex launch pack on Totoro; fill receipt template.
- Keep DRAFT until paste + Totoro receipt; do not merge sibling paste DRAFTs without their pastes.
