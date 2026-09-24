# Totoro #323 Track A — D-139 runbook (paste-gated #402)

**Runbook:** ready (docs-only merge **#402**). **Execution:** paste **`ack Totoro D-139 #323 Track A`** still required — no SSH / oracle launch until then.

**Issue:** [#323](https://github.com/itchyshin/GLLVM.jl/issues/323) (advisory Frozen R 0.7.0 family smoke).

**Disposition context:** Option **(A)** execute Track A after ack ([`2026-09-14-advisory-frozen-r-smoke-323-pending.md`](../decisions/2026-09-14-advisory-frozen-r-smoke-323-pending.md)); programme default **(B) waived** on 2026-09-15 does **not** remove this runbook if maintainer later pastes Track A ack.

**Canonical execution pack:** [`after-task/2026-09-14-issue-323-totoro-launch-pack.md`](../after-task/2026-09-14-issue-323-totoro-launch-pack.md) (Codex executor on Totoro; Cursor lane does **not** SSH).

This file is a **pointer runbook** for the paste gate only. It adds no new commands beyond the launch pack.

---

## Paste unlock

| Paste (exact) | Agent may |
|---------------|-----------|
| `ack Totoro D-139 #323 Track A` | Codex (or maintainer) runs Track A on Totoro per launch pack; records receipts; optional #323 comment. |
| `ack Totoro D-139 #323 Track B` | Narrow oracle + three holdout cells only. |
| `ack Totoro D-139` | Codex defaults to Track A (launch pack table). |

**≠** `waive #323` · **≠** goal complete · **≠** advisory CI becomes gating.

---

## D-139 estimate (reminder)

| Track | Wall clock | Core·hours |
|-------|------------|------------|
| **A** full `runparity.jl` + oracle build | ~90–150 min | ~1.5–2.5 |
| **B** oracle + 3 holdout cells | ~60–90 min | ~1.0–1.5 |

Single Julia process; `OPENBLAS_NUM_THREADS=1`, `JULIA_NUM_THREADS=1`. Stop if runtime exceeds band by >50% and re-report.

---

## Frozen pin (immutable)

| Field | Value |
|-------|-------|
| gllvmTMB ref | `b4d5fee64def88bc768dda1f1f77c29b295edd86` |
| Contract | `docs/dev-log/core070/frozen-r070-contract.toml` |

---

## After paste — executor checklist

1. Maintainer paste captured in chat (exact string).
2. Codex attaches to Totoro workspace; sets `TRACK=A`, receipt stamp, `GLLVM_PARITY_RECEIPT_DIR`.
3. Run launch pack **Runner block** (full CI mirror).
4. Refresh `r_gradient_max` for holdout cells `NATIVE-06-NB2`, `NATIVE-12-TRUNCATED-NB2`, `NATIVE-10-STUDENT`.
5. After-task under `docs/dev-log/after-task/` with receipt paths; update #323 if maintainer wants issue comment.

---

## Julia harness checklist (DRAFT #410 — local only)

- [x] Paste gate: `ENV["GLLVM_TOTORO_PASTE"]` must equal `ack Totoro D-139 #323 Track A` (driver exits **2** without paste).
- [x] Harness entry: `tools/totoro323/run_totoro_323_track_a_launcher.jl` + `tools/totoro323/totoro_323_track_a_harness.jl` (shell: `tools/totoro_323_track_a_launcher.sh`).
- [ ] Harness **dry-run** (no paste, no SSH, no parity):  
  `julia --project=. tools/totoro323/run_totoro_323_track_a_launcher.jl --dry-run --gllvm-root "$(pwd)"`  
  → expect `TOTORO_323_TRACK_A_PREFLIGHT_DRY_RUN_OK`
- [ ] After-Totoro receipt template: `docs/dev-log/after-task/TEMPLATE-totoro-323-track-a-receipt.md`
- [ ] Frozen gllvmTMB pin unchanged: `b4d5fee64def88bc768dda1f1f77c29b295edd86` (printed in dry-run summary; confirm on Totoro)
- [ ] D-139: full Track A only after maintainer paste; Cursor lane does **not** SSH

---

## Fences

- **No** Totoro spend without ack (D-139).
- **No** gllvmTMB `src/` edits from GLLVM.jl lane.
- **No** `Project.toml` bump.
- T4 realistic-size grid needs separate D-139 ack naming that grid (launch pack + maintainer decision set).

---

## Related DRAFTs

| Paste | Scaffold |
|-------|----------|
| `G0 Stage 1` | [`2026-09-16-d3-loading-profile-stage1-paste-gated-scaffold.md`](2026-09-16-d3-loading-profile-stage1-paste-gated-scaffold.md) |
| `S4 probe yes` | [`2026-09-16-s4-probe-julia-checklist-paste-gated.md`](2026-09-16-s4-probe-julia-checklist-paste-gated.md) · DRAFT harness **[#409](https://github.com/itchyshin/GLLVModels.jl/pull/409)** |
| `ack Totoro D-139 #323 Track A` | DRAFT harness **[#410](https://github.com/itchyshin/GLLVModels.jl/pull/410)** |
