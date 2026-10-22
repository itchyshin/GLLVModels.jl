# extract_phylo_signal left on a dirty source tree

Date: 2026-10-21. Overnight file for the 2026-10-20 leave. Worktree `/Users/z3437171/.codex/worktrees/p1-isdm-20261019/GLLVM.jl` at `5cfc6754212588222779ccfd66ad7000375cf8d5`. Read only. No commit, push, merge, or git add. No source, test, receipt, or `GATES.md` edit.

Verdict: **LEFT_DIRTY_TREE**.

## Why this worktree cannot attempt the row

`namespace/export/extract_phylo_signal` stays left. The 2026-10-20 signature says leave the row and keep the dirty source tree untouched (`docs/dev-log/plans/2026-10-20-p1-thirteen-row-signature-drafts.md`). That source is `/Users/z3437171/.codex/worktrees/joint-phylo-extractor/GLLVM.jl` at `9074da8bd57023d0ba0b4c78021ee8534a9accf0`. A read-only recheck on 2026-10-21 still finds that tree dirty. This worktree is a different checkout at `5cfc6754212588222779ccfd66ad7000375cf8d5`. A measurement here would have to bind or edit the owner tree. The signature withholds that release.

The case receipt `docs/dev-log/core070/true-parity-latest/receipts/namespace/cases/CORE070-NAMESPACE-EXTRACT-PHYLO-SIGNAL-NATIVE.json` has `result` PASS and verifier category `registration_twin`. It checks that R exports `extract_phylo_signal` from the pinned NAMESPACE and defines it in `R/extract-omega.R` at the frozen commit `9539352f66f2db2cc26b1c393e67212a359b60c9`. The Julia symbol it checks is `phylo_signal`: exported, a Function, `first_own_method` `src/confint_derived.jl:311`, `julia_commit` `c29c1c0b6d8d985e1edd96426edef2621402a03f`. The receipt meaning line calls this a name match. Semantics and numeric output stay outside that check. No paired R numerical receipt is stored for the row.

Implementation approval covers an explicit species selector, point estimates, refusal of `ci=true`, and shared Gaussian residual mode (`docs/dev-log/plans/2026-10-19-p1-remaining-decision-packet.md`). The same packet names unlanded variants `1a230eb51` and `7d6f088a7`, plus the dirty owner tree at `9074da8bd`. The unsigned plan at `/private/tmp/gllvm-p1-current-main/docs/dev-log/plans/2026-10-07-p1-extract-phylo-signal-contract.md` still leaves the P1 namespace scope and the row decision open. The source audit still requires two positive cases and the wrong-denominator, source-allocation, tip-mapping, and alias controls (`docs/dev-log/audits/2026-10-19-p1-thirteen-row-source-audit.md`). The focused counts 44/44 and 105/105 stay attached to those earlier variants. The untracked owner note `docs/dev-log/decisions/2026-10-10-joint-phylo-extractor-contract.md` says its approval leaves the namespace row unadmitted.

## Dirty files that block the row

Rechecked read-only in `/Users/z3437171/.codex/worktrees/joint-phylo-extractor/GLLVM.jl`. HEAD `9074da8bd57023d0ba0b4c78021ee8534a9accf0`. Branch `codex/true-parity-p1-extract-phylo`.

Modified:

- `README.md`
- `docs/dev-log/check-log.md`
- `docs/src/api.md`
- `docs/src/postfit-extractors.md`
- `docs/src/precision-bridge-development.md`
- `src/families/fit_gllvm.jl`
- `src/joint_covariance_identification.jl`
- `src/joint_phylo_grouped_fit.jl`
- `src/joint_phylo_grouped_postfit.jl`
- `test/test_joint_covariance_dense_gram.jl`
- `test/test_joint_covariance_identification.jl`
- `test/test_joint_phylo_grouped_alias_public.jl`
- `test/test_joint_phylo_grouped_fit.jl`
- `test/test_joint_phylo_grouped_postfit.jl`

Untracked:

- `docs/dev-log/decisions/2026-10-10-joint-phylo-extractor-contract.md`

Those fifteen paths stay as they are. No patch from `1a230eb51` or `7d6f088a7` was copied onto this worktree.

## What stays in place

HEAD of this worktree remains `5cfc6754212588222779ccfd66ad7000375cf8d5`. The receipt stays PASS at the registration tier. The scoreboard row stays REGISTRATION-ONLY. No tolerance changed. No gate box was checked. C2, C3, C4, and X2 stay open on origin/main. P1 stays open.
