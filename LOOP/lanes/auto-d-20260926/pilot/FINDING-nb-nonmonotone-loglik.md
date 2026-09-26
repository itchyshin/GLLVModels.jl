# Finding (2026-09-27 ~00:30Z): NB log-likelihood falls as K rises, while reporting converged = true

Data: pilot.jl grid "heavy", family NB2 (size 2), n = 300 sites, p = 20 species, K_true = 3, seed hash(("nb",300,20,3,1)), origin/main @ 2847b5dbf (lane branch has no src edits).
Fits: fit_gllvm(Y; family = NegativeBinomial(), K = k) (default route coerces disp_group = :species).

| K_fit | loglik | converged | secs |
|---|---|---|---|
| 1 | -19473.13 | true | 27.8 |
| 2 | -18874.03 | true | 26.4 |
| 3 | -19113.13 | true | 20.4 |
| 4 | -20240.21 | true | 46.0 |

A K+1 model nests the K model (zero column), so the maximised loglik must be non-decreasing in K. K=3 and K=4 are
worse optima reported as converged. Effect on selection: every criterion picks K = 2 when truth is 3.
Consequence for auto-d: any fit-and-compare rule must (a) warm-start K+1 from the K solution (append a small column)
and/or multi-start, and (b) flag loglik(K+1) < loglik(K) - tol as a failed fit rather than a valid candidate.
Owner of the fitter: family kernels are the overnight gllvm-backlog lane's files; reported, not fixed here.
Reproduce: julia --project=. LOOP/lanes/auto-d-20260926/pilot/pilot.jl out.csv 1 heavy
