# A0 briefs (WS0 additive re-pin), sliced so each is one PR

Order: A0a -> A0b -> (A0c, A0d in parallel). Each: Sonnet builder (high), then a fresh Fable reviewer. All draft PRs, "Part of" wording, merge only on Shinichi's word.

## A0a  Tracked ledger and oracle (no R needed)
- New `docs/dev-log/core070/true-parity-latest/GATES.md`: clauses C0 to C8, F-gates, X2, M1 as in ultra-plan.md "Destination", pinned at P1 = 9539352f6, with the D-295 boundary written in (temporal, phylo latent inside; column grammar, spatial outside by signed disposition).
- New `tools/true_parity_check.mjs`, from `~/local-scratch/lanes/GLLVM.jl-gllvm-backlog-20260926/.unlazy/true-parity/check.mjs` (untracked; copy, do not move): row count read from the scoreboard file (no hard-coded 32); receipt resolution (every cited receipt path must exist on the ref: `git cat-file -e REF:path`); modes C0 to C8 plus X2.
- Negative controls as a test (`tools/test_true_parity_check.*` or a node script run in CI): a dangling receipt fails C1; a C8 capability with no scoreboard row fails C2; a name-only match fails C8; a stale carried receipt fails its row. Run against a fixture tree via PARITY_REF=FS.
- The 0.7.0 oracle, ledger and pins stay untouched (additive).

## A0b  Additive P1 pin
- `tools/parity_oracle.py`: add `P1_GLLVMTMB_ORACLE = "9539352f6..."` (full SHA) beside `FROZEN_GLLVMTMB_ORACLE`; `DEFAULT_R_REF` moves to P1 only through an explicit, tested switch; P0 stays callable. Update `tools/test_parity_oracle_defaults.py`.
- Required (non-advisory) CI job for P1 twin tests: `on: [pull_request, workflow_dispatch]`, ubuntu only.
- The pin string appears in about 555 tracked files: do NOT rewrite them; they are P0 evidence.

## A0c  Case-map rows for the +25 exports and +11 S3 methods
- Diff NAMESPACE b4d5fee64..9539352f6 in the gllvmTMB clone; one row per new export or method with a PROPOSED class (twin / excluded / needs-surface / semantic-divergence / outside-boundary) and a one-line reason. Shinichi signs in the PR (D-295 row 3). Column-grammar and spatial exports: outside-boundary with a disposition row.

## A0d  Row-level stale scan and the 20 isdm rows
- Start from `reviews/p1-stale-scan-prelim.md` (all 29 R-pinned contracts stale). Map each ledger row to its receipt/contract and mark `PARTIAL_STALE_AT_P1` where the carry rule fails; output the count by family and structure; re-cite the 20 isdm rows whose receipts point at a gitignored path (reclassify, do not invent receipts).
- Output feeds A3; compute for re-measurement goes to kohaku or DRAC (Totoro down until 2026-09-28).
