# Audit notes (Class A inner-loop / Class B converged flag) - worktree gllvm-nb2-finite-20260924
Optim 1.13.3: converged = x||f||g conv; x_abstol=f_abstol=0 default => zero-length step => x_converged.
Class B guarded (grad check): grouped_fit.jl:408, grouped_nongaussian_fit.jl:983, source_fit.jl:374,
  precision_multivariate_fit.jl:261, joint_phylo_grouped_fit.jl:250, tweedie _tweedie_verdict (tweedie.jl:214).
SUSPECTS so far:
- covariates.jl:27-49 _laplace_mode_off: undamped, no restart, no backtrack; fit_gllvm_cov (covariates.jl:318 _fit_verdict). Default formula+X for Poisson/Binomial/Exponential; bridge poisson/binomial+X (bridge.jl:1606). A+B HIGH
- grouped_dispersion.jl:1382 _nb1_grouped_loglik_site: same shape as gamma; fit_nb1_gllvm_grouped (1610) + _cov (1758). Default NB1 (fit_gllvm coerce, bridge 1262/1560, formula 258). A+B HIGH
- grouped_dispersion.jl:1806 _tweedie_grouped_loglik_site: same shape; fit_tweedie_gllvm_grouped; B guarded by _tweedie_verdict. A MED (disp_group route only)
- laplace.jl:102 _laplace_mode: backtracking only for opt-in families; TweedieED NOT opted in (comment at laplace.jl:36 is wrong; tweedie.jl:268 says reverted). maxiter exhaustion returns z silently.
- NB2 grouped kernel grouped_dispersion.jl:39 still undamped (excluded fitters, note only)
## Probe results (stress draws: Λ scale up to 3, ±50% perturbation; "non-mode" = |∇logpost|>1e-4 at kernel z while BFGS/backtracked ref converges)
- _laplace_mode_off Poisson 420/4000 (10.5%), Binomial 16/4000; all finite; worst in-band Δll 129 (binomial)
- NB1 grouped kernel 108/3000 finite
- twopart ZIP 616/1500 (480 in-band), ZINB 328/1500, HurdlePoisson 147, DeltaGamma 274
- COMPoisson 19/800; OrderedBeta 212/1500; BetaBinomial 103/1500; StudentT shared 246/1500, grouped 273/1500 (gaps small, multimodal)
- GP1 generic 120/1000 (gap up to 9e4); mixed 42/1200; Tweedie grouped 13/300; ordinal pertrait 0/1500 (clean)
- CONTROL generic backtracked: Poisson 0, Binomial 0; NB2 71 (clamp region), Gamma 196 (maxiter slow Fisher conv; median gap 3e-5), Exponential 224 (tiny), Beta 55 (tiny)
- NB2 grouped kernel (EXCLUDED fitters) 147/1500 still undamped -> note to coordinator
## Class B end-to-end
- poisson_default 4/4 honest (|g|<1e-5); poisson_formula_X (sc=1) 4/4 honest
- nb1_default: seeds 1,2,3 converged=true with |g| 1230, 1.35e11, 542; seed4 honest. Mechanism seed1: x_abschange=0,f_abschange=0 -> x_conv&f_conv, g_residual 5193; seed2: neighbour at 1e-5 returns +9.09e12 (Class A cliff) -> zero-length step
- confint bootstrap: _family_bootstrap confint_family.jl:2904-2935 counts finite θ as n_converged; never reads fb.converged/loglik
- confint profile: _family_profile_refit confint_family.jl:2817-2819 ok=true for any finite nmin incl. 1e12 sentinel; ignores Optim.converged
