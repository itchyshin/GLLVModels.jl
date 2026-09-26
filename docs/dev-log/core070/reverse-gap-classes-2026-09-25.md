# Reverse-gap classes for the 91 "genuinely ahead, unclassified" names (proposed, unsigned)

Status: **PROPOSED 2026-09-25 -- not signed by the maintainer.** This document writes up the
per-name classes that `tools/parity_ledger.py` already carries (added as a WIP commit on
`claude/reverse-gap-classes-20260925`, `REVERSE_CLASSES` dict) so the class/reason table has a
durable home next to the tool, and so Shinichi can review and sign it. Nothing here has been
approved; every class below is a proposal, and each `REVERSE_CLASSES` string in the tool repeats
that `[PROPOSED 2026-09-25, unsigned -- ...]` tag so a reader hitting the raw tool output sees the
same caveat.

## What this closes

Clause C6 of the true-parity programme required that the 91 Julia-only export names the countdown
was reporting as "genuinely ahead, unclassified" (REVERSE, no written class) each get a written
reason, the same way the existing `AHEAD_EXPLICIT`/`AHEAD_PATTERNS` tables already classify the
larger pattern-matched Julia-only surface. `REVERSE_CLASSES` is a separate, explicit per-name dict
so `classify_ahead()`'s matching logic needed no change -- it is consulted only as a last-resort
class for names nothing else classified.

## Tool re-run: exact output

Re-ran on the rebased branch (`claude/reverse-gap-classes-20260925`, rebased onto
`origin/main` at `d4da31544`, gllvmTMB pinned to the frozen 0.7.0 oracle, default `--gllvmtmb`
path):

```
REVERSE -- GLLVModels.jl EXPORTS WITH NO gllvmTMB TWIN (0) -- genuinely ahead, unclassified

COUNTDOWN: 62 export gaps genuinely owed (0 UNTRACKED of those) · 0 renamed away · 4 not-capability · 6 accounted for · 0 genuinely ahead · 375 ahead-accounted
FORWARD=62 REVERSE=0
```

**Unclassified REVERSE = 0.** All 91 names that used to fall through to "genuinely ahead,
unclassified" now carry a written class in `REVERSE_CLASSES`.

## The classes and counts

91 proposed entries, grouped by class (a name can only carry one class; counts sum to 91):

| Class | Count | What it means |
|---|---:|---|
| julia-only phylogenetic engine substrate | 27 | Same bucket as the existing `^phylo_/^edge_/^branch_/^clade_` `AHEAD_PATTERNS` regexes, but the name's shape does not match those regexes. A GLLVModels.jl capability (phylogenetic latent-variable engine internals) with no gllvmTMB analogue. |
| julia-only diagnostic or extractor | 27 | A named accessor, CI/profile-curve helper, or result-struct/short-name duplicate for something gllvmTMB already exposes generically (through `summary()`, `confint()`, `predict()`, or an already-twinned `extract_*()`), not a separately exported function or type on the R side. |
| internal-but-exported helper | 10 | Dispatch/construction types or generic statistics utilities (e.g. `excess_kurtosis`, `spearman`, `welch_t`, `GllvmModel`, `GroupingTerm`) that are exported but serve internal bookkeeping or validation roles, not a modelling capability; several are flagged as candidates to unexport. |
| julia-only family | 6 | A response-distribution family gllvmTMB does not offer at all (Beta-hurdle, Conway-Maxwell-Poisson, Generalized-Poisson-1, Hurdle-NB2, Hurdle-Poisson, ordered-beta). |
| R has it after 0.7.0 | 5 | The frozen 0.7.0 oracle NAMESPACE predates the export; `origin/main` already has it (`select_lv`, `chibar2_pvalue`, `variance_lrt`, and two more) -- the countdown's frozen-oracle comparator is simply behind, not a genuine gap. |
| R has it after 0.7.0, FLAGGED | 3 | Same "R gained it after 0.7.0" situation, but for the three zero-inflated families (`ZIB`, `ZINegBin`, `ZIPoisson`) bridge.jl's own comments already call the route "Julia-forward / twin-asymmetric" with no twin light RCall Delta -- flagged because the parameterizations differ (shared-z two-part Newton-scoring vs. R's per-trait intercept-only zero-part Laplace) and need maintainer confirmation on whether this is a genuine twin or a documented divergence. |
| julia-only spatial engine substrate | 3 | SPDE/Matern spatial-substrate builders (`spatial_cov` and two more); gllvmTMB's TMB template does not implement this per `docs/src/gllvmtmb-parity.md` "Honest gaps". |
| Julia-first capability awaiting an R decision | 2 | `SourceCovariance` and `cv_gllvm`: capabilities with no R export or capability-status row, where the R side has either already filed the gap (gllvmTMB issue #1192) or has a directly comparable precedent (`select_lv`) that R later ported -- flagged as candidates for the same treatment, not yet decided. |
| R has it under another name | 8 | Renamed-away pairs already tracked in `api-rename-notes.md` (`RENAMED_AWAY`) plus `StudentTFamily` (an alias-matched duplicate export) and `TwoLevelRepeatabilityProfileWithdrawn` (an intentional-refusal exception class mirroring R's own abort class) -- same underlying capability under a different name, not an additional one. |

## D8 hand-off text

Per the true-parity gate-tier table (`docs/dev-log/core070/true-parity-gate-tier-2026-09-05.md`),
row **D8**:

> D8 | `parity/REVERSE-GAP-DISPOSITION` | Reverse list (`parity_ledger.py`) | meta | Tool-produced;
> R lane owns R-side ports

and per `docs/dev-log/after-task/2026-09-14-destb-api-boundary.md`:

> Reverse gaps are tool-dispositioned (`D8`), not Julia lane debt.

This document is that D8 tool-production step: it makes the reverse list's per-name disposition
legible and reviewable. It does **not** discharge D8 -- D8 is R-lane owned, and closing it requires
the maintainer to review and sign these classes (or send back corrections), then decide, for the
"R has it after 0.7.0" / "Julia-first capability awaiting an R decision" rows specifically, whether
gllvmTMB's `main` should port, alias, or explicitly decline each one. Until signed, every class
here stays `[PROPOSED 2026-09-25, unsigned]` and carries no authority beyond documenting the
countdown's current disposition.

## Not covered by this document

- Whether any individual class assignment is *correct* -- that is exactly what needs the
  maintainer's sign-off.
- The FORWARD side (62 export gaps genuinely owed) -- untouched by this pass, tracked separately
  by the same tool's countdown.
- Any Julia-side code change, unexport, or rename -- several classes above name a name as "a
  candidate to unexport" or "a candidate for the same treatment [R]"; none of that has been acted
  on here.
