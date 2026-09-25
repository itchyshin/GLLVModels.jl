# NB2 boundary-stall screens (2026-09-24, #477 and #476)

Scripts and their printed output, run on a Mac Studio with Julia 1.10.12, single-threaded.
The R side is gllvmTMB `b4d5fee64` built with `tools/core070_build_oracle.py` (prepare, build and
verify all passed). Data: the NATIVE-06 design (p = 5, K = 2, per-trait intercepts, loadings
`0.30 .* parity_loadings_p5k2()`, shared true dispersion `r`), drawn from `Random.seed!(seed)`.
Run each script from the repository root with
`julia +1.10.12 --project=test/parity <script>` and `R_LIBS` / `GLLVM_PARITY_R_LIBS` pointing
at that oracle library.

| Script | Code it ran against | What it shows |
|---|---|---|
| `nb2_screen.jl` | before any fix | 9 datasets: Julia at the boundary on every one; gllvmTMB finite on some |
| `nb2_detail.jl` | before any fix | Julia's own objective at gllvmTMB's point beats Julia's returned point (seeds 46 and 45, r = 1 and 2) |
| `nb2_variants.jl` | scratch copy of the objective | start at r = 1 (a) and restart (b) both lift the three stalled fits |
| `nb2_variants2.jl` | scratch copy of the objective | on seeds 51 and 52 a fresh start finds gllvmTMB's point, which is lower than Julia's boundary fit |
| `nb2_screen2.jl` | first version of the restart (all boundary groups reset together) | 32 datasets, R gradient at its own optimum 1e-5 to 2e-3 |
| `nb2_interior_screen.jl` | final restart | n = 200, r = 1: 3 of 8 datasets have an interior maximum that survives pushing each trait to the boundary |
| `nb2_pairs_check.jl` | final restart | the chosen dataset (seed 46, n = 200): every single and pair push is 1.27 to 6.51 lower |
| `nb2_final_screen.jl` | final restart | 16 datasets: old fit, new fit, gllvmTMB and the best single-trait push, one line each |

The final restart resets the boundary groups together and, when there are several, each on its
own, and keeps the best only if it is better by more than 1e-6.

## `aghq/`: Laplace against near-exact integration (seeds 46 and 52, n = 80, r = 1)

An adaptive Gauss-Hermite evaluator built from the package's own NB2 pieces
(`aghq_nb2_eval_lib.jl`). With one node it reproduces the package Laplace value (|Δ| 4.5e-13);
it converges by 15 nodes, and a brute-force trapezoid grid agrees to 3e-8. An independent
skeptic reproduced the ranking and margins with its own evaluator. The scripts use absolute
paths from the run (`/tmp/claude-503/nb2-aghq`); logs are the printed output. Result: on seed 46
the Laplace lead of the boundary point (0.317) is a tie under AGHQ (0.013); on seed 52 the
boundary point really is higher (0.419) and the interior Laplace maximum is an artefact.

## `siblings/`: the same screen on five sibling fitters (Julia only)

One harness per fitter replicates its objective and optimizer exactly (checked: the replicated
objective equals `-fit.loglik` at every returned point) and tries alternative starts on 10
simulated datasets. Every claimed stall was re-checked by a separate skeptic. The scripts use
absolute paths from the run (`/tmp/claude-503/sibling-screen`).

| Fitter | Result |
|---|---|
| `fit_nb_gllvm_grouped_cov` (`nb2-cov*`) | same class as #477: 3 of 10 stall 0.39 to 1.30 below; the package restart recovers all 3 (now applied in the fitter) |
| `fit_gamma_gllvm_grouped` (`gamma*`) | different bug: the per-site mode search diverges at the default start, returns a finite nonsense value, and the fit reports converged after 2 iterations, 43 units below the optimum |
| `fit_beta_gllvm_grouped` (`beta*`) | different bug: converged is reported at a non-stationary point (max gradient 6.0) after a zero-length line-search step; a fresh start is 6.6 higher (3.2 under AGHQ) |
| `fit_nb1_gllvm_grouped` (`nb1*`) | no stall on 8 datasets; boundary hits are genuine |
| `fit_tweedie_gllvm_grouped` (`tweedie*`) | not screened (one fit takes 4 to 73 min on a Mac); by reading, it never flags a dispersion boundary |
