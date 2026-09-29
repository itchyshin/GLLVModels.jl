# phylo_latent twin receipts at gllvmTMB P1

Source pin `9539352f66f2db2cc26b1c393e67212a359b60c9` (gllvmTMB 0.7.1). R ran
from a temporary private library built with `R CMD INSTALL` from a detached
worktree at P1 (R 4.6.0, TMB 1.9.21, ape 5.8.1, Matrix 1.7.5); DLL SHA-256
`cba0f54d5492f0c6d0c5474c19281e3b58d5e2b0709e536684619558aaf55b8d`. The R
package source was never edited. Julia side: `fit_phylo_latent_gllvm` on Julia
1.10.12 (reference platform), from the working tree on `97e11be04` before the
first commit of this branch. `qualified = false` in every receipt until the
maintainer signs the dated promotion block (D-300 answer 9).

Generating scripts: `tools/phylo_latent/r_reference_p1.R` (SHA-256
`b6acbf5e014a5eae39ba47d164def4c1c8461ccaf59ca8ec5ce8cfda3b34ff7f`) and
`tools/phylo_latent/compare_phylo_latent_p1.jl` (the committed Julia receipts were
written by its first version through JSON3; it now uses a dependency-free JSON
codec so the P1 CI job can run the replay with `--project=.`, and a re-fit with
it reproduces the A14 tree receipt's logLik exactly). Order of operations per
case: R `generate` writes the literal fixture; Julia `fit` writes its receipt
from that fixture; R `fit` fits the same fixture, records its own optimum and
evaluates its objective at the Julia optimum. The replay test
`test/test_phylo_latent_paired_p1.jl` pins every file's SHA-256 and recomputes
the Julia side.

| File | SHA-256 |
|---|---|
| `a14-fixture.json` | `9a4c2e1f87a3fbb519400fff8a99fdfb6d66b249d473b0e25833b93b78e7e13c` |
| `struct_phy_tree_rr/r-receipt.json` | `c550ddbab35b4ba952dbda470f7e879898fb43fd665f5a7c144b10ec2c2c91f4` |
| `struct_phy_tree_rr/julia-receipt.json` | `7da419e71edc2268395e84a9687ded92fa466902691374d68e3c53b0f430becd` |
| `struct_phy_dense_rr/r-receipt.json` | `b7d728eb30e27c207ba05d4867730052fc75ebe82db1cac15cc3e738d03fc3d5` |
| `struct_phy_dense_rr/julia-receipt.json` | `700f14677d5498d832d900ea7939d8ed023c90948f1603891029da18401cd288` |
| `a15-fixture.json` | `f3bace851f60f337a8274859644135a84792b9e9ab35c07f757a7ade336706d4` |
| `cov_phylo_latent_rsz/r-receipt.json` | `5121fbfeb20421149e9c535ea9220458d531f8ec0cfaa71c5b010a8597a7e40e` |
| `cov_phylo_latent_rsz/julia-receipt.json` | `9840f0a17840e438f66c5df97b5325188f15d0e05789191a14b6188f04054bab` |

## A14 fixture

The height-four eight-tip tree of the Destination B precursor, three traits,
two replicates per species, `d = 1`, drawn in R with seed `20260927`. Both A14
cases fit the same response. R parameters: `b_fix` x3, `log_sigma_eps`,
`theta_rr_phy` x3; nlminb `rel.tol = 1e-12`, status 0 on both routes.

## A14 results (tolerances: logLik rtol 1e-6, cross objective abs 1e-8, estimates rtol 1e-4, log-det abs 1e-8)

| Quantity | STRUCT-PHY-TREE-RR | STRUCT-PHY-DENSE-RR |
|---|---|---|
| R logLik at own optimum | -8.681920085429528 | -8.68192011899238 |
| Julia logLik at own optimum | -8.68192008542988 | -8.681920118992728 |
| logLik relative difference | 4.1e-14 | 4.0e-14 |
| Julia objective at R optimum minus R objective | 1.1e-14 | 1.1e-14 |
| R objective at Julia optimum minus Julia objective | 7.1e-15 | 3.6e-15 |
| R gradient (max abs) at Julia optimum | 9.3e-6 | 9.3e-6 |
| Sigma_phy relative difference (max abs / max) | 2.1e-7 | 2.1e-7 |
| beta relative difference (elementwise max) | 2.1e-7 | 2.1e-7 |
| sigma_eps^2 relative difference | 2.6e-8 | 2.7e-8 |
| log det A, R minus Julia | 0 | 0 |
| n_aug (R, Julia) | 14, 14 | 8, 8 |
| Julia converged; FD gradient max abs | true; 9.3e-6 | true; 9.3e-6 |
| R nlminb status; AD gradient max abs at own optimum | 0; 1.3e-6 | 0; 1.4e-6 |
| cond(H): R sdreport `cov.fixed`, Julia FD Hessian | 194.974, 194.974 | 194.974, 194.974 |
| R wall time / Julia wall time | 0.58 s / 4.4 s | 0.54 s / 4.9 s |

Julia wall time includes the finite-difference Hessian diagnostic and
first-call compilation.

## A15 fixture (COV-PHYLO-LATENT-RSZ)

The signed shape (D-300 answer 2): `ape::rcoal(100)` with seed `20260928`,
tips relabelled `sp1` to `sp100` and written with 17 significant digits (both
engines read that Newick string), 20 traits, 5 replicates per species
(500 observations, 10 000 long rows), `d = 2`. Truth drawn in R (intercepts
`sd = 0.5`, loadings `sd = 0.6`, residual `sd = 0.5`, all recorded in the
fixture). 60 free parameters on both sides; `n_aug = 198`.

## A15 results

| Quantity | Value | Bar |
|---|---|---|
| R / Julia logLik | -7371.771401586599 / -7371.7714015845095 | |
| logLik relative difference | 2.8e-13 | rtol 1e-6, pass |
| Julia objective at R optimum minus R objective | 3.7e-11 | abs 1e-8, pass |
| R objective at Julia optimum minus Julia objective | 1.7e-11 | abs 1e-8, pass |
| Sigma_phy relative difference (norm; max abs / max) | 1.1e-6; 1.2e-6 | rtol 1e-4, pass |
| beta relative difference (norm; elementwise max) | 1.5e-5; 1.2e-4 | rtol 1e-4, pass in norm |
| sigma_eps^2 relative difference | 4.2e-7 | rtol 1e-4, pass |
| log det A, R minus Julia; n_aug | 0; 198 and 198 | abs 1e-8, pass |
| R AD gradient max abs at Julia optimum | 2.0e-4 | 1e-4, **not met** |
| Julia FD gradient max abs at R optimum | 4.0e-3 | 1e-4, **not met** |
| R nlminb status; AD gradient max abs at own optimum | 0 (relative convergence); 4.0e-3 | |
| Julia `converged`; FD gradient max abs at own optimum | false (`gradient_not_converged`, 338 LBFGS iterations); 2.0e-4 | Julia `g_tol = 1e-5` |
| cond(H): R sdreport `cov.fixed`; Julia FD Hessian | 83 030; 81 735 | recorded, not gated |
| Wall time, fit only: R; Julia (1.10.12, 2 threads) | 3.4 s; 17.7 s | estimate was minutes; far under 1 h |

The elementwise beta maximum is the one near-zero intercept (trait 14,
`b = 0.0476`), absolute difference `5.8e-6`.

**Stationarity gap, recorded, not widened.** Both engines stop on the flat
floor of an objective of size 7372. R's own nlminb optimum has AD gradient
`4.0e-3`; Julia's LBFGS stops at FD gradient `2.0e-4` (R's AD gradient at the
same point is `2.03e-4`, so the finite differences are accurate there), so
Julia reports `converged = false` under its absolute `g_tol = 1e-5`. A
restart from the returned point, or 2000 iterations, does not move it. One
Newton step with the finite-difference Hessian from the Julia point lowers the
objective by `1.8e-10` and the gradient to `6e-6`, so the Julia point is
within about `2e-10` of the optimum in objective. The replay asserts the
primary receipt (cross objectives abs 1e-8, logLik rtol 1e-6) and, for the
gradients, explicit looser bounds with measured headroom: R's AD gradient at
the Julia point `<= 1e-3` (measured 2.0e-4), Julia's FD gradient at R's point
`<= 1e-2` (measured 4.0e-3), Julia `converged = false` with
`stopping_reason = :gradient_not_converged` and `gradient_norm <= 1e-3`
(recorded explicitly in the stored receipt), and the elementwise beta gap
`<= 1e-4` absolute (measured 1.6e-5). A scale-aware stopping rule or a Newton
polish in `fit_precision_multivariate` would close the gap; that belongs to
the convergence lane (decision-packet item 485), not this PR.

**Update (PR #547 CI fix).** The same stall, at a smaller scale, failed CI on
Linux: the 50-species tree fit of `test_phylo_latent_twin.jl` stopped at FD
gradient `1.7e-5` on OpenBLAS (converged on macOS), with logLik within
`7e-10` relative of the vcv route. `fit_phylo_latent_gllvm` (not the frozen
`fit_precision_multivariate`) now takes at most three positive-definite Newton
descent steps from a `gradient_not_converged` stall and refits from the
polished point under the unchanged `g_tol`. On that Linux fit the polish
lowers the objective by `4e-13` and the gradient to `8e-8`. The stored A15
Julia receipt above is unchanged and keeps its recorded stall; the A15 live
refit now reports `converged = true` (gradient `6.2e-6`, logLik within
`3e-13` relative of R's, measured on Linux OpenBLAS), and the replay asserts
that.

## In-keyword `Ainv`

`fit_phylo_latent_gllvm(...; Ainv = inv(C), tip_labels)` on the A14 fixture
reproduces the dense receipt (R's keyword rewrites `Ainv` to
`vcv = solve(as.matrix(Ainv))`): `n_aug = 8`, the same log-det, logLik within
rtol 1e-6 of R's `-8.68192011899238`, and the Julia objective at R's dense
optimum within 1e-8 of R's.
