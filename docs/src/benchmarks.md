# Benchmarks

The benchmark grid from the parallel `gllvmTMB-julia-bench/` study
compares R `gllvmTMB` and GLLVModels.jl on the same simulated data per
replicate. Both engines minimise the same Gaussian marginal
log-likelihood with `f_reltol = 1e-10` and `g_tol = 1e-6`. The R engine
uses `gllvmTMB::gllvmTMB` (TMB Laplace + L-BFGS); the Julia engine uses
`GLLVModels.fit_gaussian_gllvm` (closed-form Gaussian marginal + `Optim`
L-BFGS with `ForwardDiff` gradients).

Six cells × three replicates × two engines = 36 fits. All 36 converged.

!!! warning "This grid is Gaussian only — the speedup does NOT generalise"
    Every number on this page comes from the **Gaussian** path, where GLLVModels.jl
    uses a closed-form marginal and R uses a TMB Laplace approximation. That is
    an *algorithmic* difference, not a language one, and it is where the large
    factors come from.

    Non-Gaussian families use a dense Laplace approximation on both sides, and
    the measured speedups are far smaller. In small matched fits comparing the
    two packages:

    | family | measured per-fit speedup vs `gllvmTMB` |
    |---|---|
    | lognormal | ≈ 1280× (also a closed-form path) |
    | truncated_poisson | ≈ 2.2× |
    | Gamma | ≈ 1.6× |

    So do not read the Gaussian headline as a general claim about the package.
    It describes the closed-form path and nothing else.

## Wall-clock (median over reps)

| cell_id        | n_sites | n_species | d | median t\_R (s) | median t\_Julia (s) | median R / Julia |
|----------------|--------:|----------:|--:|----------------:|--------------------:|-----------------:|
| c01_small_noX  |      20 |         5 | 1 |          0.0230 |              0.0001 |          194.9   |
| c02_small_X    |      20 |         5 | 1 |          0.0250 |              0.0001 |          185.3   |
| c03_med_noX    |      80 |        10 | 2 |          0.1020 |              0.0002 |          698.1   |
| c04_med_X      |      80 |        10 | 2 |          0.1120 |              0.0003 |          335.3   |
| c05_large_noX  |     200 |        20 | 2 |          0.5140 |              0.0013 |          398.8   |
| c06_large_X    |     200 |        20 | 2 |          0.6180 |              0.0037 |          161.2   |

Julia is the faster engine in every cell after a per-signature warm-up.
The relative speedup is largest in the smallest cells (where the
R-side TMB Laplace setup dominates a small inner loop) and the
absolute time saved per fit is largest in the larger cells — the
metric that matters for a multi-thousand-fit simulation study.

## Agreement (worst-case across the grid)

| cell_id        | max \|Δ logLik\| | max Σ\_y rel-Frobenius |
|----------------|-----------------:|-----------------------:|
| c01_small_noX  |       4.965e-10  |             5.769e-06  |
| c02_small_X    |       4.443e-10  |             5.519e-06  |
| c03_med_noX    |       3.683e-08  |             1.804e-05  |
| c04_med_X      |       3.839e-08  |             2.810e-05  |
| c05_large_noX  |       1.923e-07  |             3.742e-05  |
| c06_large_X    |       2.343e-07  |             4.424e-05  |

Worst-case `|Δ logLik|` over the full grid is **2.343e-07**, below the
pre-specified limit of `1e-4`. Worst-case relative Frobenius on `Σ_y` is
**4.424e-05**, below its `1e-3` limit. The two engines agree to at least six
significant digits on every fit.

Full per-rep details, the wall-clock log-log plot, and the verification
criteria are available in the source benchmark repository at
`gllvmTMB-julia-bench/report/grid-bench.md`.

## O(p) phylogenetic gradient scaling

The headline of the node-frame analytic gradient (`grad_node_perspecies`) is
that a single **exact** gradient evaluation scales **linearly** in the number
of species `p` on a tree — where R `gllvmTMB` caps near `p ≈ 500`. Timed on a
balanced tree (one evaluation, median over `BenchmarkTools` samples; reproduce
with `julia --project=bench bench/node_gradient_bench.jl`):

| p      | state build (ms) | gradient (ms) | gradient / p (µs) |
|-------:|-----------------:|--------------:|------------------:|
| 100    |            0.088 |        0.0094 |             0.094 |
| 500    |            0.287 |        0.041  |             0.081 |
| 1 000  |            0.583 |        0.077  |             0.077 |
| 5 000  |            2.935 |        0.393  |             0.079 |
| 10 000 |            5.968 |        0.771  |             0.077 |

The per-tip time (last column) is essentially flat, and the log–log scaling
slope is **0.96 for the gradient** and **0.92 for the state build** — both
≈ 1, confirming `O(p)`. A single exact gradient at `p = 10 000` takes
**0.77 ms**. (The older `sparse_phy_grad` path is `O(p²)`, slope ≈ 2; it is
retained as an independent-complexity cross-check, not the production path.)
