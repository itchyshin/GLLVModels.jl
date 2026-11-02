# Repro: false `converged` (#754, #753, #781)

Recorded on `origin/main` at `e6cfcd4ee7b5bcf1ba4ff2f0b6413bf80f31efe3`
(issue pin was `824d22a4`; the three defects still fire on current main).

```sh
julia --project=. docs/dev-log/repro/2026-10-10-false-converged/repro_754_branch_re.jl
julia --project=. docs/dev-log/repro/2026-10-10-false-converged/repro_753_em_phylo.jl
julia --project=. docs/dev-log/repro/2026-10-10-false-converged/repro_781_zinb_cov.jl
```

Before (main `e6cfcd4ee`, Julia 1.10.12):

```
#754 noise only: s2=6.21e-10  s2eps=1.155  conv=true  iters=55  negll=47.71
#754 constant y: s2=5.0e-324  s2eps=5.0e-324  conv=true  negll=-11870.67
#754 NaN in y:   ErrorException branch-RE optimisation found no finite likelihood point
#753             σ_eps=1.4901161193847656e-8  conv=true  n_iter=5  logLik=1939.83
#781 warmstart:  conv=true  iters=0  loglik=-412.93
#781 absurd:     conv=true  loglik=-12720.22
#781 NaN θ_init: conv=true  loglik=-1.0e12
```

After the fix, noise-only reports `sigma2_boundary=true` and `conv=false`;
constant / NaN `y` raise `ArgumentError` naming `y`; the EM rank-1 cell
reports `sigma_eps_boundary=true` and `conv=false`; ZINB `iterations=0`
returns `conv=false`, and a non-finite `θ_init` is rejected.
