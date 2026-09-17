# S4 public phylo_dep probe receipt (paste-gated)

**Date:** YYYY-MM-DD  
**Paste ack:** maintainer pasted `S4 probe yes` (do not claim without evidence)  
**GLLVM.jl tip:** `<short SHA>`  
**gllvmTMB recorder:** PR #1283 @ `97214679c` (branch `codex/destination-b-s4-phylo-dep-formula-20260910`)  
**Frozen oracle pin (unchanged):** `b4d5fee64def88bc768dda1f1f77c29b295edd86`

## Scope boundary

- IN: isolated public-formula probe receipt vs recorder runner (DestB S4 cell).
- OUT: Arc 0 promotion, capability row `covered`, honest-0.7 FINAL-REVIEW complete, gllvmTMB engine edits from GLLVM.jl.

## Environment

| Field | Value |
|-------|--------|
| Host | |
| Julia | `julia --version` |
| GLLVM.jl project | |
| gllvmTMB root HEAD | |
| Receipt JSON path | |
| Wall clock | |
| Compute tier | Mac-light / Totoro (D-50) |

## Commands (exact)

```text
export GLLVM_S4_PROBE_PASTE='S4 probe yes'
julia --project=. tools/destination_b/run_s4_public_phylo_dep_probe.jl \
  --gllvmtmb-root ... \
  --julia-project ... \
  --julia ... \
  --receipt ...
```

Preflight-only (no R probe):

```text
julia --project=. tools/destination_b/run_s4_public_phylo_dep_probe.jl --dry-run \
  --gllvmtmb-root ... \
  --julia-project ... \
  --julia ... \
  --receipt ...
```

## Outcome

| Check | Pass/Fail | Notes |
|-------|-----------|-------|
| Recorder runner exit 0 | | |
| Receipt JSON present | | |
| Pass/fail table | | |

## Failure classification (if any)

- [ ] Julia surface gap
- [ ] Recorder drift vs `97214679c`
- [ ] R-oracle / frozen-R defect

Do not widen `@test` rtol to green a failure.

## Follow-up

- [ ] Pending board / paste packet update (docs PR only)
- [ ] GOAL QS4 checkbox (maintainer only; goal may remain incomplete)
