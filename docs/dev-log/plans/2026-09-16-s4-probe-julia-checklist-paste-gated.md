# S4 public-formula probe — Julia-side checklist (paste-gated runbook #402)

**Runbook:** ready (docs-only merge **#402**). **Execution:** paste **`S4 probe yes`** still required — no probe run, no engine merge, no gllvmTMB `src/` until then.

**Goal gate:** [`LOOP/GOAL.md`](../../LOOP/GOAL.md) QS4 (second explicit yes only).

**Twin recorder (read-only reference):** gllvmTMB draft [PR #1283](https://github.com/itchyshin/gllvmTMB/pull/1283), commit **`97214679c`** on `origin/codex/destination-b-s4-phylo-dep-formula-20260910` ([G9 push receipt](../after-task/2026-09-14-destb-g9-s4-recorder-push.md)).

This checklist is **Julia-side prep only**. No probe execution from cloud / Cursor without the paste.

---

## Paste unlock

| Paste (exact) | Agent may |
|---------------|-----------|
| `S4 probe yes` | Execute isolated S4 Julia probe vs recorder artifacts; file receipt under `docs/dev-log/after-task/`; update DestB / GOAL checkboxes. |

---

## Pre-flight (before any probe run)

- [ ] Confirm recorder ref fetchable: `git ls-remote origin codex/destination-b-s4-phylo-dep-formula-20260910` → tip `97214679c…` (harness checks when network available)
- [ ] Read gllvmTMB #1283 scope (formula / fixture contract only; **no** TMB edits from GLLVM.jl lane)
- [ ] Frozen oracle pin unchanged: `b4d5fee64def88bc768dda1f1f77c29b295edd86` (printed in dry-run summary; confirm manually)
- [x] Julia entry script (paste-gated harness; DRAFT PR — does not execute without paste):
  `tools/destination_b/run_s4_public_phylo_dep_probe.jl` +
  `tools/destination_b/s4_public_phylo_dep_probe_harness.jl`
- [ ] Harness dry-run (no paste, no R): `julia --project=. tools/destination_b/run_s4_public_phylo_dep_probe.jl --dry-run --gllvmtmb-root … --julia-project … --julia … --receipt …` → expect `S4_PUBLIC_PHYLO_DEP_PREFLIGHT_DRY_RUN_OK`
- [ ] After-task receipt template present: `docs/dev-log/after-task/TEMPLATE-s4-public-phylo-dep-probe-receipt.md`
- [ ] D-50: probe on **local Mac-light** or **Totoro** with maintainer compute ack if wall clock >30 min (separate from #323 Track A ack)
- [x] `GLLVM_S4_JULIA_HOME` wiring fixed (`claude/s4-probe-wiring-20260924`): the harness now passes the
  **directory** holding the `julia` executable, matching the recorder's
  `file.path(julia_home, "julia")` in `s4_public_phylo_dep_clean_julia_probe()` — previously it passed
  the executable file itself, which that `file.path()` call could never resolve.
- [x] `GLLVM_S4_JULIA_ENV` now defaults to a committed probe-only environment,
  `tools/destination_b/probe_env/Project.toml` (`claude/s4-probe-wiring-20260924`): it `develop`s the
  repo root and lists `LogExpFunctions` as a direct dependency, since the recorder's clean-Julia-probe
  does `using LogExpFunctions` before `using GLLVM` and the root `Project.toml` only carries
  `LogExpFunctions` transitively. One-time local instantiate (Manifest is gitignored, like every other
  sub-environment in this repo). Since the option A shim (below), `probe_env/Project.toml` lists the
  unregistered `GLLVM` shim, so the older one-step `Pkg.develop(path="../../.."); Pkg.instantiate()` now
  fails with `expected package GLLVM [530e1681] to be registered` (and from the repo root the relative
  `../../..` does not resolve at all). Use this two-step setup, run from `tools/destination_b/probe_env`:
  `julia --project=GLLVM -e 'using Pkg; Pkg.develop(path="../../.."); Pkg.instantiate()'`
  `julia --project=. -e 'using Pkg; Pkg.develop([PackageSpec(path="../../.."), PackageSpec(path="GLLVM")]); Pkg.instantiate()'`
  `--julia-env` still overrides this default when passed explicitly.
- [x] `using GLLVM` resolved by a probe-only shim (maintainer decision 2026-09-24, option A;
  `claude/s4-probe-run-20260924`): `tools/destination_b/probe_env/GLLVM/` is a local package named
  `GLLVM` that re-exports `GLLVModels.bridge_fit`, developed into the probe env only. It is never
  registered and the repository root project cannot load it. Because the recorder asserts
  `Base.pkgdir(GLLVM) == GLLVM_DESTINATION_B_PROJECT`, the probe passes the shim directory as the
  Julia project. Receipt: `docs/dev-log/after-task/2026-09-24-s4-public-phylo-dep-probe-receipt.md`.
- [ ] **Recorder runner does not attach testthat (OPEN, needs Shinichi).** Under `Rscript --vanilla` the
  frozen runner evaluates `test_that(...)` in `globalenv()` with testthat loaded but not attached, so
  both selected tests error with `could not find function "test_that"` before any fit. The recorder is
  frozen (D-220). A diagnostic run that attached testthat through `R_DEFAULT_PACKAGES` (not a receipt)
  then failed one point-estimate gate (`phylo_covariance`, tolerance 5e-6). See the receipt for both.

---

## Julia-side execution checklist (after paste)

1. Check out GLLVM.jl at programme tip; `Pkg.instantiate()`; single Julia process.
2. Materialize or load S4 fixture bundle aligned with recorder commit (wide/long public formula path for **`phylo_dep()`** cell per DestB scope).
3. Run isolated probe; capture **pass/fail table** + log paths (no GitHub Actions artifacts).
4. Classify each failure: Julia surface gap vs recorder drift vs R-oracle defect (do not widen `@test` rtol).
5. Write after-task: scope boundary (probe receipt ≠ Arc 0 promotion ≠ capability row `covered`).
6. Update pending board + paste packet only via docs PR (no silent GOAL complete).

---

## Fences

- **No** gllvmTMB engine surgery from this repo.
- **No** S4 probe without **`S4 probe yes`** (this document does not substitute for the paste).
- **No** claim that S4 closure implies honest-0.7 FINAL-REVIEW complete.

---

## Related DRAFTs

| Paste | Scaffold |
|-------|----------|
| `S4 probe yes` | DRAFT **[#409](https://github.com/itchyshin/GLLVModels.jl/pull/409)** (Julia harness; merge after paste + probe receipt, not before) |
| `G0 Stage 1` | [`2026-09-16-d3-loading-profile-stage1-paste-gated-scaffold.md`](2026-09-16-d3-loading-profile-stage1-paste-gated-scaffold.md) |
| `ack Totoro D-139 #323 Track A` | [`2026-09-16-totoro-323-track-a-runbook-paste-gated.md`](2026-09-16-totoro-323-track-a-runbook-paste-gated.md) |

After paste, set `export GLLVM_S4_PROBE_PASTE='S4 probe yes'` and run the driver
documented in `s4_public_phylo_dep_scaffold_usage()` (gllvmTMB checkout at
`97214679c`, single Julia process).
