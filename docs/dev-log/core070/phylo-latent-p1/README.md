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
`tools/phylo_latent/compare_phylo_latent_p1.jl`. Order of operations per
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
