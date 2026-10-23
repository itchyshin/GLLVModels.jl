# Commits on origin/main absent from this HEAD

Worktree: `/Users/z3437171/.codex/worktrees/p1-isdm-20261019/GLLVM.jl`

HEAD is `5cfc67542` (`5cfc6754212588222779ccfd66ad7000375cf8d5`). `origin/main` is `1d8fc465d`. The merge-base is `882292ac6` (`882292ac6539d00bc0a08f5d0d06dc9f471251b3`). `git rev-list --count` gives 34 commits on `origin/main` that are absent from HEAD, and 18 commits on HEAD that are absent from `origin/main`. `origin/main` was already present, so this note did not fetch. Nothing here was merged.

The list below is `git log --oneline HEAD..origin/main`, newest first. Each line is the short hash and the subject only.

```
1d8fc465d Merge pull request #868 from itchyshin/cursor/false-converged-754-753-781-78dc
c74856e86 Merge origin/main into the false-convergence branch.
30416623a Merge pull request #905 from itchyshin/claude/sharp-gates-3ud4k2
c09470676 Merge origin/main into the false-convergence branch.
72d375b93 Merge origin/main into the phylo-latent rounding-floor branch.
dac271f42 Merge pull request #885 from itchyshin/claude/beautiful-hopper-ydytwd
68fbd7dcf Merge pull request #886 from itchyshin/claude/g2-503-damped-mode
8bac66e3f Merge pull request #811 from itchyshin/cursor/oct4-732
7f8a81543 Merge pull request #918 from itchyshin/claude/fervent-fermi-e3zb3p
0fe3f3172 Merge pull request #882 from itchyshin/claude/fix-769-cv-binomial-scale
b6f9d5ab8 Merge pull request #901 from itchyshin/claude/intelligent-gauss-eb1sh2
7e5d793cc Merge pull request #934 from itchyshin/claude/b16-L10-lambda-constraint-wald
227fd66a0 Documenter deploy: build-time token and loud failure when nothing is published
1973d591e Add Documenter deploy-guard test for #719
ba788d061 Beta-binomial log-pmf: remove the phi = 1e6 switch to Binomial (refs #872)
8a9e5059a Tighten the docs of the pinned-fit Wald rule and the lognormal refusal (refs #794, #764)
fa544b9b5 Correct the loading_profile_exploratory docstring on pinned fits (refs #764)
045cdddc6 fit_lognormal_gllvm refuses lambda_constraint (refs #794)
c97e15ae7 Gate pinned and structural loading rows on the free information (refs #794, #764)
28436abf0 Add forcing test for Wald routes on lambda_constraint fits (refs #794, #764)
62dc9ff86 Polish a non-converged pervar EM fit with L-BFGS on the log-variance scale (refs #736)
1aec7ee4f Tweedie grouped: damp the per-site mode search and fail loudly when it does not converge
64ba9d4dc cv_gllvm: Binomial with N stores and scores expected counts (refs #769)
4b4674d61 docs: record Pkg.test tally for the #905 reviewer fix
e23afd520 fix(phylo_latent): never set converged when gradient_norm > g_tol
ec91a3fa4 fit_phylo_latent_gllvm: accept a stop at the objective's rounding floor as converged (refs #822)
eb623e412 Record review after-task and check-log.
38fcc42cd Make variance-boundary floors relative to the data scale.
397fd905e Record CI green on the false-converged PR.
aa4ae9bb5 Record after-task and check-log for the false-converged PR.
34b898ed9 Stop reporting converged when the fit is not an interior optimum.
29b15474f Record main repros for false converged flags.
3b6f77188 fix(postfit): index eligible grid points, not finite_ll slots (#732)
956a50097 fix(postfit): pick profile_cross_rho best_rho only when converged (#732)
```

## Subject scan

Parity subject count: 0. No subject contains `parity` in any letter case.

The token `GATES` appears in 0 subjects. The token `scoreboard` appears in 0 subjects. The phrase `family receipt` appears in 0 subjects.

Nearby wording, recorded so it is visible and left uninterpreted:

- `30416623a` has the branch name `sharp-gates` in the subject. The body subject line is `fit_phylo_latent_gllvm: rounding-floor tag without converged above g_tol (refs #822)`.
- `c97e15ae7` begins with the word `Gate`. The body uses `gates` for the loading confidence-interval rule.
- `29b15474f` has subject `Record main repros for false converged flags.` Its body says `Scripts and receipts` for repro scripts of #754, #753, and #781. The subject does not say receipt, and it does not name a response family.

Family names do appear in other subjects (`Beta-binomial`, `Binomial`, `lognormal`, `Tweedie`, `phylo_latent`). Those lines describe likelihood, scoring, or convergence edits. None of them is worded as a family receipt.

## Clause wording

NONE_CLAIM_MET

No message among these 34 commits says that C2, C3, C4, or X2 became MET. The characters `C3` occur once, inside the session URL on `ba788d061`. That string is part of a session id. The message does not say a clause became MET. This note does not treat any commit as closing C2, C3, C4, or X2.
