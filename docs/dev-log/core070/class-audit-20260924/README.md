# Class-wide audit of silent inner-loop failure and false convergence flags (2026-09-24)

Receipts for the audit that followed #477, #479 and #480. Every script ran read-only against
GLLVModels.jl at d9bc77412 (Julia 1.10.12, Optim 1.13.3, the test/parity environment). They were
first written under /tmp/claude-503/audit/ and copied here unchanged, except that two long file
names were shortened:

- `fit_verdict_classB_probe.jl` (log: `_fit_verdict_run1.log`): the shared `_fit_verdict` check,
  probed through seven default routes (ZIP, NB1, Poisson, Binomial, ordinal, truncated Poisson,
  Gaussian), 21 fits.
- `twopart_classA_probe.jl` (log: `zi_twopart_full.log`): the two-part kernel `_twopart_mode`,
  ten datasets over ZIP, ZINB and ZIB.
- `skeptic/`: two independent checks written from scratch. `zip_skeptic.jl` computes the exact
  marginal likelihood by grid quadrature, with no Laplace step. `skeptic_zip_nb1.jl` re-optimises
  a smooth, fully converged objective. Logs: `skeptic/A.log`, `skeptic/run1.log`.
- `probe_A*`, `probe_B*`, `env/`: the discovery probes of the per-site kernels and the covariate
  routes (`notes.md` summarises them).

Two classes were checked:

- **Class A:** a per-site mode search stops without converging and returns a finite, wrong value.
- **Class B:** `converged = true` is reported at a point that is not stationary. Optim's
  `converged` is `x_converged || f_converged || g_converged`, and `x_abstol` and `f_reltol`
  default to 0, so a zero-length step counts.

Findings are summarised in the issues that cite this folder.
