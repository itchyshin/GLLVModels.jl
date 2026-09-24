# Totoro #323 Track A receipt (paste-gated)

**Date:** YYYY-MM-DD  
**Paste ack:** maintainer pasted `ack Totoro D-139 #323 Track A` (do not claim without evidence)  
**GLLVModels.jl tip:** `<short SHA>`  
**gllvmTMB frozen pin:** `b4d5fee64def88bc768dda1f1f77c29b295edd86`  
**Contract:** `docs/dev-log/core070/frozen-r070-contract.toml`

## Scope boundary

- IN: Track A Frozen R smoke on Totoro per launch pack (full CI mirror or stated subset).
- OUT: goal complete, advisory CI gating, gllvmTMB `src/` edits from GLLVModels.jl, `waive #323`.

## Environment

| Field | Value |
|-------|--------|
| Host | Totoro (D-50) |
| R version | 4.5.3 (`R RHOME`) |
| GLLVM_ROOT | |
| TRACK | A |
| RECEIPT_STAMP | |
| GLLVM_PARITY_RECEIPT_DIR | |
| Wall clock | |
| D-139 band | ~90–150 min (stop if >50% over band) |

## Commands (exact)

```text
export GLLVM_TOTORO_PASTE='ack Totoro D-139 #323 Track A'
# Codex on Totoro — launch pack Runner block (steps 0–5a)
# See docs/dev-log/after-task/2026-09-14-issue-323-totoro-launch-pack.md
```

Local harness preflight only (no Totoro):

```text
julia --project=. tools/totoro323/run_totoro_323_track_a_launcher.jl --dry-run \\
  --gllvm-root $(pwd)
```

## Outcome

| Check | Pass/Fail | Notes |
|-------|-----------|-------|
| Oracle build | | |
| Full runparity (Track A) | | |
| Holdout NATIVE-06-NB2 `r_gradient_max ≤ 1e-4` | | |
| Holdout NATIVE-12-TRUNCATED-NB2 | | |
| Holdout NATIVE-10-STUDENT | | |

## Failure classification (if any)

- [ ] R build / remotes / TMB compile
- [ ] Frozen oracle drift vs pin
- [ ] Julia parity gradient miss (do not widen `@test` rtol)

## Follow-up

- [ ] After-task under `docs/dev-log/after-task/` with receipt paths
- [ ] Optional #323 issue comment (maintainer)
- [ ] GOAL QS323 checkbox (maintainer only; goal may remain incomplete)
