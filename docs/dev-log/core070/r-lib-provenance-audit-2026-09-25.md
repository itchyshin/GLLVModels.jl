# R library provenance audit for the seven migrated core070 tools

Date: 2026-09-25.

## Why this audit exists

PR #474 added `second_order_r_lib(env=ENV)` in `tools/core070_second_order/r_lib.jl`. It
refuses to run when `GLLVM_PARITY_R_LIBS` is set but the named directory has no gllvmTMB,
instead of silently falling back to R's default library. `tools/core070_second_order/common.jl`
already routed its `_require_gllvmtmb!()` through that guard.

The after-task report for that PR, section 8, named seven tools that still called
`library(gllvmTMB)` directly, bypassing the guard:

- `tools/core070_aghq_gaussian_pair_run.jl`
- `tools/core070_aghq_binomial_pair_run.jl`
- `tools/core070_aghq_poisson_pair_run.jl`
- `tools/core070_aghq_public_gaussian_pair_run.jl`
- `tools/core070_source_fixed_residual_pair.jl`
- `tools/core070_covariance_mode_fits.jl`
- `tools/d220_paired_gaussian_cell.jl`

It also flagged a near relative: `test/parity/parity_helpers.jl`'s own
`_parity_prepend_twin_lib!()` read `GLLVM_PARITY_R_LIBS` itself and quietly returned when the
named library had no gllvmTMB, instead of refusing. Every one of the seven tools above, plus
two more found by grep while doing this work (`test/parity/test_gaussian_parity.jl` and
`test/parity/test_ordinal_probit_parity.jl`), called gllvmTMB with no guard in front of it at
all.

Branch `claude/r-lib-guard-tools-20260925` routes all of these through the guard. This
document is the read-only half of that work: a list of every receipt these tools left under
`docs/dev-log/` before the fix, and whether the receipt itself proves which gllvmTMB build
produced it. No receipt listed here was edited or deleted.

## Rule for marking

A receipt is **build-confirmed** when its own content names the R library path that produced
it, or the frozen CORE-070 reference commit `b4d5fee64def88bc768dda1f1f77c29b295edd86`. A
receipt with neither is **provenance-unknown**: it may still be correct, but nothing in the
file itself says which gllvmTMB build produced it.

For the six build-confirmed receipts below, the commit marker is not a label someone typed in.
It comes from `_core070_source_pin!()` (`test/parity/parity_helpers.jl`), which calls
`find.package('gllvmTMB')` in the live R session, then checks the installed package's NAMESPACE
and source-tree SHA-256 against the frozen commit's own hashes, and throws if they do not
match. So each marker is a run-time self-check the tool performed against whatever gllvmTMB
happened to be reachable through R's `.libPaths()` at the moment it ran. Before this branch,
nothing in these tools guaranteed `.libPaths()` pointed at the intended library before that
check ran; this fix guarantees it going forward. The receipts below are
history and were not re-run or re-verified for this audit; they are marked only on what they
already recorded.

## Receipts

| Tool | Receipt | Status | Evidence in the file |
|---|---|---|---|
| `tools/core070_aghq_gaussian_pair_run.jl` | `docs/dev-log/core070/aghq-gaussian-evidence.json` | build-confirmed | `"reference": "b4d5fee64def88bc768dda1f1f77c29b295edd86"` |
| `tools/core070_aghq_binomial_pair_run.jl` | `docs/dev-log/core070/aghq-binomial-evidence.json` | build-confirmed | `"reference": "b4d5fee64def88bc768dda1f1f77c29b295edd86"` |
| `tools/core070_aghq_poisson_pair_run.jl` | `docs/dev-log/core070/aghq-poisson-pair-evidence.json` | build-confirmed | `"reference_commit": "b4d5fee64def88bc768dda1f1f77c29b295edd86"` |
| `tools/core070_aghq_public_gaussian_pair_run.jl` | `docs/dev-log/core070/aghq-public-gaussian-evidence.json` | build-confirmed | `"reference": "b4d5fee64def88bc768dda1f1f77c29b295edd86"` |
| `tools/core070_source_fixed_residual_pair.jl` | `docs/dev-log/core070/source-fixed-residual-evidence.json` | provenance-unknown | no field anywhere in the file names a library path or a reference commit |
| `tools/core070_source_fixed_residual_pair.jl` | `docs/dev-log/core070/source-fixed-residual-final-evidence.json` | provenance-unknown | same; this file adds a `local_versions_script` key but still no library or commit field |
| `tools/core070_covariance_mode_fits.jl` | `docs/dev-log/core070/covariance-mode-fits-evidence.json` | build-confirmed | `"reference_commit": "b4d5fee64def88bc768dda1f1f77c29b295edd86"` |
| `tools/d220_paired_gaussian_cell.jl` | `docs/dev-log/core070/d220-paired-gaussian-cell-receipt-2026-09-05.json` | build-confirmed | `"oracle_ref": "b4d5fee64def88bc768dda1f1f77c29b295edd86"`, plus the live `"r_gllvmTMB_version": "0.7.1"` it recorded separately |

**Totals: 6 build-confirmed, 2 provenance-unknown, across 8 retained receipts from 7 tools.**

## How each receipt was matched to its tool

Each `docs/dev-log/after-task/2026-08-31-core070-*.md` report names both its tool and its
receipt file directly, for example the gaussian AGHQ report's "Commands:
`tools/core070_aghq_gaussian_pair_run.jl`" next to "See `core070/aghq-gaussian-evidence.json`".
`find` and `grep` across `docs/dev-log/` turned up no other JSON or TOML files tied to these
seven tools by name or by the case IDs their fixtures use (`MODE-ORD-INDEP`, `FIT-MODE-ORD-DEP`,
and so on). Some of the six build-confirmed evidence files are a combined receipt for a pair
tool and a sibling non-pair tool that never touches R at all (for example
`aghq-gaussian-evidence.json` covers both `core070_aghq_gaussian_run.jl`, pure Julia, and
`core070_aghq_gaussian_pair_run.jl`, the one in scope here); that does not change the marking,
since the commit field is present in the file either way.

## Why the two source-fixed-residual receipts are the ones with no marker

`tools/core070_source_fixed_residual_pair.jl` was the one tool in the audit whose pre-fix code
called `_core070_source_pin!()` (line 7) *before* it ever touched gllvmTMB in R (the bare
`library(gllvmTMB)` call was line 8). `_core070_source_pin!()`'s `find.package('gllvmTMB')`
check does not need the package to be attached, only reachable on `.libPaths()`, so this
ordering meant the check ran against whatever R's default library already held, not the
intended twin. This branch fixes the order (guard first, then `_core070_source_pin!()`) along
with removing the bare call; see `tools/core070_source_fixed_residual_pair.jl` on
`claude/r-lib-guard-tools-20260925`. Both retained receipts for this tool were written under
the old order and record no library or commit marker at all, consistent with that gap. The
tool's own raw per-run `result.toml` (written to the `ARGS[1]` output directory) was never
retained under `docs/dev-log/`, so it is not part of this audit.

## What this audit does not claim

This is a listing of what the existing receipts already say about themselves, not a fresh
re-verification of any past run. A build-confirmed marking means the tool checked its own
provenance at the time and recorded a match; it does not mean this audit re-ran that check
today. No file listed above was edited, deleted, or regenerated for this document.
