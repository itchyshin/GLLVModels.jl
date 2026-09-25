# After-task — §2 Hessian disputed-default decision draft (docs-only)

**Status 2026-09-25:** superseded. This after-task's deliverable
(`docs/dev-log/decisions/2026-09-15-hessian-disputed-default-pending.md`) was
not the draft that was accepted. The same §2 question (cloglog / Tweedie
grouped Hessian default) was decided through a separate, differently-named
draft, `docs/dev-log/decisions/2026-09-15-second-order-hessian-s2-pending.md`,
which was ACCEPTED (option A) and applied 2026-09-15 (PR #347, merged
`3091fe61`). Do not cite this draft or its deliverable as the operative
decision.

**Date:** 2026-09-15  
**Lane:** Cursor / Ada (true-parity `/goal`; ledger-gap inventory **rank 3**)  
**Branch:** `docs/hessian-disputed-default-20260915` from `origin/main` @ `0da63860`  
**Worktree:** `~/local-scratch/docs/hessian-disputed-default-20260915` (isolated from Dropbox `feat/second-order-delta-followup-20260915`)  
**Scope:** PENDING maintainer decision draft + this report — **no** `src/`, **no** `Project.toml`, **no** gllvmTMB, **no** ledger JSON, **no** Totoro.

## Rose fence (read first)

- **≠** D3 **`loading_profile` Stage 1** — not authorised here.  
- **≠** **S4 probe** second yes — not touched.  
- **≠** Totoro / DRAC / #323 execution.  
- **≠** second-order parity, programme §7 closure, or **`covered`** promotion for cloglog / Tweedie grouped.  
- **=** One pending decision file so Shinichi can paste **`accept §2 hessian A|B|C`**; follow-on cascade is a **separate** slice after acceptance.

## Deliverable

| Artifact | Path |
|----------|------|
| Pending decision | `docs/dev-log/decisions/2026-09-15-hessian-disputed-default-pending.md` |
| Ada default | **(A)** — ratify `:observed` for Binomial/cloglog and Tweedie grouped |

## Evidence read (no new measurements)

- `docs/dev-log/after-task/2026-09-15-true-parity-ledger-gap-inventory.md` (rank 3)  
- `docs/dev-log/core070/true-parity-decision-map.md` (T3 second-order scope)  
- `docs/dev-log/core070/second-order-parity-contract.md` §2, §6  
- `docs/dev-log/core070/cloglog-leaf-notes.md`  
- `docs/dev-log/decisions/2026-08-28-arc-decision-batch.md` (Tweedie flip; stale cloglog line)  
- `docs/dev-log/after-task/2026-09-14-second-order-parity-inventory.md` (holdouts)

## Checks run

```text
# Documenter (local, worktree)
julia --project=docs docs/make.jl
```

- **Not run:** `Pkg.test()`, second-order drivers, RCall parity.

## Follow-up after maintainer acceptance

1. Cascade PR: contract §2 stale flags, holdout table, optional batch-row annotation — engine touch only if **(B)** chosen.  
2. Paired second-order toy cells for cloglog + Tweedie grouped (separate engine lane).  
3. Pending board / LOOP checkpoint — **other lanes** (not edited in this slice).

## Sign-off

| Role | Verdict |
|------|---------|
| Ada | Draft complete; recommend **(A)** |
| Rose | OK for **PENDING** docs-only PR; no capability claim |
| Maintainer | Paste reply phrase to ACCEPT |
