# GOAL — users of GLLVModels.jl never have to supply the number of latent dimensions d: the package estimates it from the data, reports what it chose and why, and says honestly what that choice does to later inference

**IMMUTABLE for this run** (drafted 2026-09-26 by the overnight lane from Shinichi's request; the lane confirms it at G0). Re-read this file at the top of EVERY arc, before anything else.

## Definition of done
- [ ] A decision map (destination, decisions so far, not yet specified, out of scope) records the route and the default criterion, and Shinichi has approved it.
- [ ] `d` can be omitted (or set to an explicit "auto" value): the fitter estimates it from the data and returns the chosen d with the comparison behind it (criterion values per candidate d, and the runner-up).
- [ ] Recovery evidence: on simulated data with known d, the default choice recovers the true d at stated rates across n, p and family, with a pre-run test before any campaign over 3 hours (D-287).
- [ ] Post-selection caveat documented where users see it: intervals computed after choosing d are conditional on that choice.
- [ ] Draft PR(s) reviewed independently; merge only on Shinichi's word (public API change).

## Invariants (never violate, even to finish faster)
- Never push, merge, or publish — those are HUMAN GATES. Land work on this branch only.
- Verification means reading the LOG and inspecting the ARTEFACT, never the exit code.
- A narrow or negative search is not proof. "No X exists" usually means the query missed X.
- Destructive or irreversible ⇒ STOP and surface, even if it feels urgent.
- Query the second brain first (`search_notes` with `project: "shinichi-brain"` first; add `search_all_projects: true` for other repos' docs; name no other project). In a code repo also run `python3 ~/shinichi-brain/tools/route.py <repo>` and read the LOAD-FIRST block. This worktree is one branch, not the memory.

## Pre-authorisation (copied from approved ultra-plan)
- Routine scoped edits, local commands, tests, builds, checkpoints, local commits, and listed checks: CONTINUE.
- Optional remote authority: DRAFT (to be confirmed at G0): push branch claude/lane-auto-d-20260926 and open draft PRs; never merge or release.
- Must stop: merge/release/public message or claim; credentials/security changes; destructive work outside this worktree; new compute/cost beyond the estimate; scope-changing evidence.

## Git and GitHub transport
- Use this repository's existing `origin` SSH remote for all Git operations (`git fetch`, `pull`, and an explicitly authorised `push`). Do not change remotes or keys.
- Do not open a browser or run `gh auth login`, device login, or token setup. GitHub API work (Actions dispatch, PR creation/merge, issue/comment writes) is separate from SSH and needs explicit task authority plus an already-working API credential. If it is not already available, report the limitation; do not request or start a login as a workaround.

## Starting facts (measured 2026-09-26 by the overnight lane; verify, do not trust)
- Both packages already have fit-and-compare pieces: GLLVModels.jl `select_lv` (src/model_selection.jl), `cv_gllvm` (src/cv.jl), `chibar2_pvalue`; gllvmTMB exports `select_lv` and `chibar2_pvalue`.
- Candidate routes: (1) automatic fit-and-compare over d = 0..Kmax (AIC, BIC, chi-bar-squared LRT, or CV as the default); (2) one fit with shrinkage of loading columns (ordered-factor-LASSO style; Bayesian analogues are multiplicative gamma process and cumulative shrinkage process priors); (3) a spectral starting guess. Literature citations came from memory and are UNVERIFIED until checked.
- Known hazards: adding a dimension is a boundary test (chi-bar-squared, not chi-squared); AIC tends to over-select and BIC to under-select at small n; runaway loadings can mimic an extra dimension (vault note "Two runaway modes in GLLVM loadings").

## Files this lane owns (the overnight lane is fenced off them)
- src/model_selection.jl, src/cv.jl, and the K/d argument handling in src/families/fit_gllvm.jl, plus new files this lane creates. Claim exact files with LANE_ID=claude:GLLVM.jl:auto-d before editing.

## Out of scope (the fence — do NOT drift here)
- The overnight gllvm-backlog lane's files and PRs (read LOOP/lanes/gllvm-backlog-20260926/OVERNIGHT.md on branch claude/lane-gllvm-backlog-20260926): family kernels, confint_family.jl, grouped_dispersion.jl, the parity docs.
- (Superseded by D-292, 2026-09-26: gllvmTMB is now in this lane. Build auto-d in R and Julia side by side, each cross-checking the other. The first question is feasibility: has anyone chosen the number of latent variables in GLLVMs, and how well does it work?)
- Porting the 25 post-0.7.0 gllvmTMB exports (a separate arc).
