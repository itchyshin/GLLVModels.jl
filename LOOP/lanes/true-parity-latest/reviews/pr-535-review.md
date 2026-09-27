# Review of PR #535, temporal port spec (head c121fe12d), 2026-09-27: NOT READY (two misreads), READY after edits

Verified: phi = (1-1e-6) tanh(theta), kappa = exp(theta), AR1 a = phi^gap (integer squaring), OU exp(-kappa elapsed), unit stationary variance with scale in Lambda/psi; V = Z (K_blockdiag kron Sigma_T) Z' + sigma_eps^2 I is R's exact marginal (R's own oracle builds it); modes and ranks (dep = p, latent = 1, indep = 0); packing convention equals unpack_lambda; fit_gaussian_sources cannot carry a parametric kernel, so a new NLL is right; the gllvm() long-data hook (formula.jl:300) is a separate method from the Normal branch (234-250) but still needs the grammar lane. 91 blocks recounted; 48 fenced / 43 twinned arithmetic right.

1. HIGH. compare_temporal is NOT restricted to unreplicated Gaussian temporal_indep (R/temporal-selection.R:5-23): it admits all modes, replicated fits and AR1-vs-OU. AIC = -2 logLik + 2 length(opt$par) exactly.
2. HIGH. Profile walk misdescribed: R uses TMB::tmbprofile (h0 = 1e-4; ystep caps the objective change per step with halve/double rules; stop at |y - y0| > ytol, NA, parm.range or maxit = ceiling(5 ytol / ystep); nlminb step.min 0.001, warm start; starts at last.par.best), then .profile_bounds interpolates on the zeta scale sign(theta - theta_hat) sqrt(2 (nll - nll_hat)) (R/profile-ci.R:249-262), NA-vs-Inf via .profile_terminus_status (157-184), nrow < 3 gives NA (442-444); test uses ystep .25, ytol 1, parm.range theta +- .01. Port all of it; +1-2 days.
3. MEDIUM. Kernel, phylo, animal cross-source pairs (18 blocks) are inside P1 and twinnable in principle: relabel "deferred pending Q1", not fenced. Only the spatial pair is out of boundary.
4. MEDIUM. Two twins (program-bootstrap.R:40-52, program-selection.R:22-34) swap R's kernel fixture for a unit source: mark as partial twins, not kernel evidence.
5. LOW. sigma_eps suppression needs a per-row diagonal term and a non-unreplicated workflow (R/fit-multi.R:6959-6960); when it fires log_sigma_eps is fixed and mapped off, changing df.
6. LOW. Receipts assert names(opt$par), not an assumed set.
7. LOW. forecast est includes the offset (fence offsets); negative-variance refusal only below -1e-8, then clamp.
8. LOW. AR1 powers with integer exponents in Julia (phi^Int(gap)); (-0.6)^2.0 throws.
