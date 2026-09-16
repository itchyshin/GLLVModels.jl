# D3 `loading_profile` Stage 1 — paste-gated runbook (#402)

**Runbook:** ready (docs-only merge **#402**). **Execution:** maintainer paste **`G0 Stage 1`** still required — no `src/` export until then.

**Programme:** true-parity T5 row 8 (`namespace/export/loading_profile` → `BLOCKED_NEEDS_JULIA_SURFACE`).

**Stage 0 on `origin/main`:** [#345](https://github.com/itchyshin/GLLVM.jl/pull/345) — confirmatory substrate only ([`loading-profile-confirmatory-substrate.json`](../core070/loading-profile-confirmatory-substrate.json), `test/test_loading_profile_stage0.jl`).

**Scout authority:** [PR #341](https://github.com/itchyshin/GLLVM.jl/pull/341) (merged); [`after-task/2026-09-14-loading-profile-d3-surface-scout.md`](../after-task/2026-09-14-loading-profile-d3-surface-scout.md).

This file is a **pre-paste runbook**. It is **not** Stage 1 implementation.

---

## Paste unlock

| Paste (exact) | Agent may |
|---------------|-----------|
| `G0 Stage 1` | Merge DRAFT Stage 1 harness when green, then execute the bounded slice below in that PR (or a follow-on). |

Without the paste: **no** `src/` export, **no** ledger rebind, **no** shim removal.

---

## First actions after paste (bounded slice)

1. **Fitter pin path:** wire `lambda_constraint` (or equivalent) on the Gaussian + ordinary latent fitters using Stage 0 fixtures (`MASK-B-PINS`, `MASK-B-UPPER`, `MASK-B-ALLFIXED`).
2. **Public export:** confirmatory `loading_profile(fit; level, entries, n_grid, grid_extent, conf_level, y)` per scout signature — **not** `loading_profile_exploratory`.
3. **Tests:** `test/test_loading_profile_confirmatory.jl` (unit + one R-aligned pin-and-refit grid cell on frozen fixtures; heavy grid on Totoro only with separate D-139 ack).
4. **Ledger:** update `namespace/export/loading_profile` + joint D3 wording **only** after paired fixture evidence (maintainer decision set 2026-09-03).
5. **Docs cascade:** docstrings, reference page, `docs/dev-log/check-log.md`, after-task report.

---

## Explicit fences (unchanged)

- **No** `Project.toml` version bump.
- **No** removal of `loading_profile` deprecation shim in the same PR as first export (separate maintainer-approved API slice).
- **Rose fence:** Stage 1 receipt ≠ full R grid parity ≠ T5 row “covered” until ledger bind + maintainer wording.

---

## Related DRAFTs (sibling paste gates)

| Paste | Scaffold PR |
|-------|-------------|
| `G0 Stage 1` | DRAFT [#411](https://github.com/itchyshin/GLLVM.jl/pull/411) |
| `accept delta dispersion A` | DRAFT [#399](https://github.com/itchyshin/GLLVM.jl/pull/399) |
| `S4 probe yes` | DRAFT [#409](https://github.com/itchyshin/GLLVM.jl/pull/409) |
| `ack Totoro D-139 #323 Track A` | DRAFT [#410](https://github.com/itchyshin/GLLVM.jl/pull/410) |
