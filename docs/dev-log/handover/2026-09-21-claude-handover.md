# Claude Handover — GLLVModels.jl reader-documentation close

**Date:** 2026-09-21 (America/Edmonton)
**From:** Codex
**To:** Claude Code
**Repository:** `itchyshin/GLLVModels.jl`

## Critical context

This is the GLLVModels.jl part of the four-package reader-first documentation programme.
A reader should learn what a generalized linear latent variable model (GLLVM) is before
meeting matrix orientation, covariance, rotation, or internal implementation language.
The core question is: **which responses vary together, and how much variation is shared
versus response-specific?**

GLLVModels.jl is a Julia companion for multivariate models. It is useful to ecologists,
evolutionary and environmental scientists, and other researchers with related outcomes.
It is not a replacement for richer R-side workflows and must not overpromise parity,
interval coverage, or a general-purpose model route. Exclude figures, badges, releases,
registry work, and engine changes.

**FINDINGS-OF-RECORD: none.**

## Ten-milestone ledger

| Milestone | Status for this handover |
| --- | --- |
| 1. CI recovery and clean merges | Recheck a current green head before every merge. |
| 2. GLLVModels first route | **Primary work here:** landing, plain limits, runnable Gaussian route. |
| 3. drmTMB core learning pages | Owned by drmTMB Claude PR #1418. |
| 4. gllvmTMB safety route | Owned by gllvmTMB Claude PR #1317. |
| 5. DRModels beginner routes | Owned by the DRModels companion handover. |
| 6. GLLVModels follow-on routes | **Primary work here:** R comparison, diagnostics, and interval wording. |
| 7. Cross-site vocabulary | Keep plain definitions consistent across all four sites. |
| 8. Internal-language sweep | Remove process, record, receipt, and unexplained implementation language. |
| 9. Rendered-reader audit | Build and inspect actual navigation, landing, and first code route. |
| 10. Rose close-out | Claims, boundaries, CI, Pages, and only then clean owned merges. |

## Current reader PRs — reconcile, do not duplicate

| PR | Reader contribution | Current instruction |
| --- | --- | --- |
| [#429](https://github.com/itchyshin/GLLVModels.jl/pull/429) | landing page, quickstart, diagnostics, route guard | merged at `51f63db5`; use as base, do not recreate |
| [#428](https://github.com/itchyshin/GLLVModels.jl/pull/428) | reader-first learning routes | dirty after stacked #429 merged; reconcile against main, do not force |
| [#439](https://github.com/itchyshin/GLLVModels.jl/pull/439) | bootstrap-scope wording | draft; inspect then integrate deliberately |
| [#437](https://github.com/itchyshin/GLLVModels.jl/pull/437) | derived loading-interval scope | draft; retain its evidence boundary |
| [#433](https://github.com/itchyshin/GLLVModels.jl/pull/433) | public post-fit docstrings | draft; source edits need owner transfer |

The baseline is `main` at `402f2f8e3` (`docs: make the R comparison a reader guide (#443)`).
The advisory Frozen R 0.7.0 family-smoke failure on #428 is not a reason to weaken wording
or claim a numerical repair.

## Current state and boundaries

- **Working:** #429’s landing work is on main; this branch contains only this durable programme handover.
- **In progress:** #428 needs current-base reconciliation; #439 and #437 must form one honest reader path.
- **Protected:** #430 is an engine/performance lane. Do not change likelihood code, API contracts,
  numerical claims, R oracle, or generated assets.
- **Do not stage:** generated site output or source files owned by #433/#437/#439 without ownership transfer.

## OWED next immediate steps

1. Read `AGENTS.md`, `HANDOVER.md`, and the coordination board; compare it with current GitHub state
   and classify #428/#439/#437/#433 as `OWED`, `DONE`, `RETRACTED`, or `PROTECTED`.
2. Begin public pages with the data and scientific question. Provide a small runnable Gaussian response-matrix
   example before latent dimensions, covariance, rotation, or loading interpretation.
3. Put a plain limit beside each action: state what the route fits, what an interval statement covers,
   and when to choose another route or the R package. Do not suggest complete parity or package-wide calibration.
4. Sweep related pages for jargon and internal-process labels. The R comparison should teach a choice,
   not merely list missing features.
5. Render the site and read landing → quickstart → diagnostics as a novice. Perform Rose’s claim-and-boundary pass.
6. Merge only a current, clean, settled-green PR with explicit ownership. Re-evaluate #428 on current main;
   never force a stale stacked merge.

## Verification

Claude may write/review prose; use Codex for live Julia compilation or rendering if Claude lacks a full toolchain.

```sh
tools/lane_preflight.sh .
git status --short --branch
git diff --check
julia --project=docs docs/make.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

Inspect the rendered preview and deployed `dev` site after a merge. Record the advisory frozen-oracle
result accurately rather than hiding it.

## Linked handovers

| Repository | Handover / PR | Boundary |
| --- | --- | --- |
| drmTMB | [#1418](https://github.com/itchyshin/drmTMB/pull/1418) | designated Claude reader-contract repair |
| gllvmTMB | [#1317](https://github.com/itchyshin/gllvmTMB/pull/1317) | designated Claude first-tutorial rewrite |
| DRModels.jl | `claude/drmodels-reader-arc-handover-20260921` | distributional-regression landing and beginner routes |

## Landing state

| Artifact / branch | Committed | Pushed | PR | State |
| --- | --- | --- | --- | --- |
| `claude/gllvmodels-reader-arc-handover-20260921` | yes | pending | none yet | CARRIED-OVER: push and open a draft PR before a fresh Claude session relies on this note. |

## How to resume

```text
Read AGENTS.md and docs/dev-log/handover/2026-09-21-claude-handover.md. Run the handover rehydration steps, reconcile them with the current git state, then continue only the OWED Next Immediate Steps.
```
