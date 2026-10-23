# Analytic gradient of the grouped non-Gaussian Laplace objective

Status: derivation only. No `src/` file is edited by this document and no Julia was run
to produce it. Every claim about current behaviour is a claim about code read at
`file:line`, not about a measurement taken here.

Purpose: write the mathematics before the implementation, and bind every symbol to the
function that will compute it, so that a mismatch between the math and the code is
visible on this page rather than in a silently wrong optimum.

Reading order for a reviewer: section 1 fixes notation against the code, section 3 is the
calculus, section 7 is the part that decides whether this ships correctly.

---

## 1. The objective, exactly as the code computes it

### 1.1 Index sets and data

| symbol | meaning |
|---|---|
| `p` | number of traits |
| `n` | number of units |
| `N = p*n` | number of responses, stacked trait-fast as `vec(Y)` |
| `m = size(W, 2)` | length of the stacked random-effect vector |
| `q = size(D, 2)` | number of mean coefficients |
| `nθ` | length of the outer parameter vector |

`y in R^N` is `vec(data)` and `nobs in R^N` is `vec(trials)`, passed at
`src/grouped_nongaussian_fit.jl:362`.

### 1.2 The outer parameter vector

`theta` is one flat vector, laid out in three contiguous blocks
(`src/grouped_nongaussian_fit.jl:349`, `expected = q + source_coordinates + length(dispersion_indices)`):

1. `gamma = theta[1:q]`, the mean coefficients (`src/grouped_nongaussian_fit.jl:354`).
   For the fixtures in scope, `D = _trait_mean_design(p, n)` (`src/source_fit.jl:137`) so
   `gamma` is one intercept per trait.
2. `psi = theta[q+1 : q+c]`, `c = source_coordinates`, the grouping-term coordinates,
   unpacked by `_grouped_term_unpack` (`src/grouped_fit.jl:125`). For an `:indep` term with
   `common = false` this is `p` coordinates, each a **log standard deviation**: the code
   forms `d = exp.(2 .* theta_block)` (`src/grouped_fit.jl:150`), a **variance**, and the
   design then takes `sqrt(d[trait])` (`src/grouped_nongaussian_fit.jl:147`). So the entry
   that actually enters `W` is `exp(theta_j)`, and `d(entry)/d(theta_j) = exp(theta_j)`,
   which is the entry itself. This is the single most error-prone scale in the whole
   derivation and is revisited in section 7.
3. `rho = theta[q+c+1 : end]`, log dispersion, exponentiated at
   `src/grouped_nongaussian_fit.jl:99` into `Beta(phi, 1.0)` or `NegativeBinomial(r, 0.5)`.
   Empty for Poisson and Binomial (`src/grouped_nongaussian_fit.jl:95-96`).

Write `theta = (gamma, psi, rho)` and let `k` index a single coordinate of `theta`.

### 1.3 Conditional model

`W = W(psi)` is the `N`-by-`m` sparse random-effect design built by
`_grouped_laplace_design` (`src/grouped_nongaussian_fit.jl:164`), block-columns
`kron(Z_s, Lstar_s)` per grouping term (`grouped_trait_design`,
`src/grouped_laplace.jl:111`), where `Lstar_s = [L_s, U_s]` with `U_s` the diagonal
square-root block of the unique variances (`src/grouped_nongaussian_fit.jl:135-153`).

The linear predictor is

```
eta(b, theta) = D * gamma + W(psi) * b            (src/grouped_laplace.jl:172)
```

with `b ~ N(0, I_m)`. Per-response conditional log-density, score and observed curvature:

```
l_i(eta_i; rho) = _glm_logpdf(family_i, linkinv(link, eta_i), nobs_i, y_i)   (src/grouped_laplace.jl:193)
s_i             = d l_i / d eta_i        = _glm_score(...)                   (src/grouped_laplace.jl:211)
w_i             = - d^2 l_i / d eta_i^2  = _glm_obs_weight(...)              (src/grouped_laplace.jl:213, defined src/families/laplace.jl:260)
kappa_i         = d w_i / d eta_i        = - d^3 l_i / d eta_i^3             (does not exist today)
```

`w` is the **observed** curvature, not Fisher. `_joint_grouped_components` computes both
(`fisher` at `src/grouped_laplace.jl:212`, `observed` at `:213`) but only `Ho`, the
observed one, reaches the reported objective; `Hf` is used solely as a positive-definite
fallback step direction (`src/grouped_laplace.jl:481`).

### 1.4 Joint log-posterior, mode, precision, objective

```
Q(b, theta) = sum_i l_i(eta_i(b, theta); rho) - 0.5 * b' b        (src/grouped_laplace.jl:193-206)
g(b, theta) = W' s(eta(b, theta)) - b                             (src/grouped_laplace.jl:451)  [= dQ/db]
A(b, theta) = W' diag(w(eta(b, theta))) W + I_m                   (src/grouped_laplace.jl:217)  [= Ho]
bhat(theta) : g(bhat, theta) = 0                                  (converged at src/grouped_laplace.jl:452)
ld(theta)   = logdet(A(bhat, theta))                              (src/grouped_laplace.jl:460 via the CHOLMOD factor Fo, :453-459)
L(theta)    = Q(bhat, theta) - 0.5 * ld(theta)                    (src/grouped_laplace.jl:462, the `q0 - 0.5 * ld` field)
F(theta)    = - L(theta)                                          (src/grouped_nongaussian_fit.jl:368-369)
```

`F` is what the outer optimiser minimises. **The deliverable of the implementation slice is
`grad F = - grad L`.** Getting this outer sign wrong is a distinct failure from getting the
calculus wrong and is listed separately in section 7.

Identity worth stating once, because it is the reason the grouped route is in some ways
*easier* than the per-site precedent: since `w_i = - d s_i / d eta_i` by definition,

```
dg/db = W' diag(ds/deta) W - I = -(W' diag(w) W + I) = -A .
```

So the matrix `A` that the objective's log-det uses **is exactly** the matrix the implicit
function theorem demands. There is no freedom here and no second matrix to build.

---

## 2. Stationarity and the implicit function theorem

The mode is defined by

```
g(bhat(theta), theta) = W(psi)' s(eta(bhat, theta)) - bhat = 0 .          (*)
```

Differentiate (*) totally in `theta_k`:

```
(dg/db) * dbhat/dtheta_k + dg/dtheta_k|_b = 0
=>  -A * u_k + v_k = 0
=>  u_k := dbhat/dtheta_k = A^{-1} v_k ,
```

where `v_k := dg/dtheta_k` holding `b = bhat` fixed. Expand `v_k` term by term. Write
`dk W := dW/dtheta_k` (nonzero only for `k` in the `psi` block) and
`dk gamma := d gamma/dtheta_k` (a unit vector for `k` in the `gamma` block, else zero):

```
explicit eta perturbation:   e_k := D * dk gamma + (dk W) * bhat                  in R^N
v_k = (dk W)' s  -  W' ( w .* e_k )  +  W' ( ds/drho .* dk rho )
```

The middle sign is not a typo: `ds_i/deta_i = -w_i`. The last term is present only for
`k` in the `rho` block and only for Beta and NB2.

`u_k` is obtained by one triangular solve per coordinate against the **factor already
computed at convergence**, `Fo` (`src/grouped_laplace.jl:453-454`, now carried out of the
inner fit on `result.factor`, `:32`): `u_k = Fo \ v_k`. That is
`nθ` sparse solves, `nθ = 6` on the large fixture. No selected inverse is needed for this
part.

**Transfer note versus `src/laplace_grad.jl`.** The per-site route never forms `u_k`
explicitly; it forms one differentiable Newton step
`z(theta) = zhat + A^{-1}(score(zhat) - zhat)` (`src/laplace_grad.jl:127`, `:305`, `:468`)
and lets ForwardDiff produce `dz/dtheta = dzhat/dtheta` implicitly. That half transfers in
principle and is identical mathematics. What does **not** transfer is the second half:
after forming `z`, the per-site code evaluates `logdet(Az)` on a dense `K`-by-`K` matrix
under duals (`src/laplace_grad.jl:148`). Here `A` is `m`-by-`m` sparse and its log-det is
taken through a CHOLMOD factor (`src/grouped_laplace.jl:460`), which ForwardDiff cannot
traverse. A third thing does not transfer either: `_grouped_laplace_design` is typed
`Vector{SparseMatrixCSC{Float64,Int}}` and calls `Matrix{Float64}(load)`
(`src/grouped_nongaussian_fit.jl:136`, `:164`), so the design cannot carry duals at all
without a new, type-generic builder. Hence the construction below is hand-derived rather
than AD-carried, and `dk W` is an explicit object we must build.

---

## 3. Total derivative, and precisely what the envelope theorem kills

```
dL/dtheta_k = [ dQ/dtheta_k ]_expl                      (A)
            + [ dQ/db ]_{b=bhat} . u_k                  (B)
            - 0.5 * tr( A^{-1} [ dA/dtheta_k ]_expl )   (C)
            - 0.5 * tr( A^{-1} (dA/db) . u_k )          (D)
```

**Only (B) vanishes.** `[dQ/db]_{b=bhat} = g(bhat, theta)' = 0` by (*). That is the whole
content of the envelope theorem here.

**(D) does not vanish and is the heart of this derivation.** `A` is evaluated *at* the
mode, and `logdet A` is not stationary in `b`. A reader who applies the envelope theorem to
`L` as a whole, rather than to `Q` alone, drops (D), never needs `u_k`, never needs a
selected inverse, and obtains a gradient that is wrong by exactly (D). The cheap wrong
answer is the one that skips all of the hard machinery. See section 7.1.

Because `A` depends on `b` only through `eta`, (C) and (D) recombine cleanly. Define the
**total** perturbations

```
edot_k := D * dk gamma + (dk W) * bhat + W * u_k              in R^N   (total d eta / d theta_k)
wdot_k := kappa .* edot_k + (dw/drho) .* dk rho               in R^N   (total d w / d theta_k)
Adot_k := (dk W)' diag(w) W + W' diag(w) (dk W) + W' diag(wdot_k) W
```

and then

```
dL/dtheta_k =   s' * ( D * dk gamma + (dk W) * bhat )                    (A1: explicit mean/design)
              + sum_i ( dl_i/drho ) * dk rho                             (A2: dispersion, Beta/NB2 only)
              - sum_i w_i * ( W A^{-1} (dk W)' )_{ii}                    (C1: design-derivative trace)
              - 0.5 * sum_i wdot_k,i * ( W A^{-1} W' )_{ii}              (C2 + D: curvature trace)
```

Derivation of the trace reductions, using `tr(A^{-1} W' diag(c) M) = sum_i c_i (M A^{-1} W')_{ii}`:

```
tr( A^{-1} (dk W)' diag(w) W ) = tr( A^{-1} W' diag(w) (dk W) ) = sum_i w_i (W A^{-1} (dk W)')_{ii}
```

the two being equal by symmetry of `A^{-1}`, which is where the factor of 2 in `Adot_k`
becomes the coefficient `-1` (not `-0.5`) on C1. Losing that 2 is failure 7.3.

Note that (A) contains **no** `-0.5 b'b` contribution: `bhat` is held fixed in the explicit
derivative and the prior term has no explicit `theta`.

---

## 4. The log-det term as a trace against selected entries of the inverse

`d/dtheta logdet A = tr(A^{-1} dA/dtheta)` is used in the form

```
tr( A^{-1} M ) = sum over (j,l) in pattern(M) of  (A^{-1})_{jl} * M_{lj} ,
```

i.e. a Frobenius inner product that reads `A^{-1}` **only at the sparsity pattern of `M`**.
Every `M` appearing in section 3 is of the form `W' diag(c) W'` or `(dk W)' diag(c) W`, so

```
pattern(M) subset { (j,l) : exists row i with W_{ij} != 0 and W_{il} != 0 } = pattern(W' W)
```

using `supp((dk W)_{i,:}) subset supp(W_{i,:})`, which holds because `dk W` differentiates
the *values* of an existing block of `W` and never introduces a new block column
(`src/grouped_nongaussian_fit.jl:135-153`: the loading block `L` and the unique block `U`
occupy fixed columns; only their entries depend on `psi`).

Since `A = W' diag(w) W + I` (`src/grouped_laplace.jl:217`),

```
pattern(W' W)  subset  pattern(A)  subset  pattern( P' (L + L') P ) = what takahashi_selinv returns
```

(`src/takahashi_selinv.jl:65-66`, `:91-101`), the last inclusion because Cholesky fill-in
only adds structural nonzeros.

**Answer to "exactly the pattern of A or more":** the entries *needed* are exactly
`pattern(A)` (the pattern of `W'W` together with the diagonal, the diagonal being needed
because `j = l` occurs in the row quadratic forms). The entries *supplied* by
`takahashi_selinv` are `pattern(L + L')` mapped back to the original ordering, which is a
**superset**: strictly more than needed whenever the factorisation fills in. So nothing is
missing and some computed entries are discarded. The routine already returns the result in
the original, un-permuted ordering (`src/takahashi_selinv.jl:182-196`), and it already
accepts a `SparseArrays.CHOLMOD.Factor{Float64}` (`src/takahashi_selinv.jl:103`), which is
what `Fo` is (`src/grouped_laplace.jl:453-454`).

Two concrete quantities are extracted from the selected inverse `Sigma`:

```
t_i    := ( W Sigma W' )_{ii}       = sum over j,l in supp(W_{i,:}) of W_{ij} W_{il} Sigma_{jl}
r_i(k) := ( W Sigma (dk W)' )_{ii}  = sum over j in supp(W_{i,:}), l in supp((dk W)_{i,:}) of W_{ij} (dk W)_{il} Sigma_{jl}
```

`t` is computed **once** per gradient call and reused across all `nθ` coordinates; `r(k)` is
per coordinate but touches only the block columns that `theta_k` moves. Cost of both is
`O(sum_i |supp(W_{i,:})|^2)`, with `|supp(W_{i,:})|` equal to the number of grouping terms
times (rank + unique columns), a single-digit number on the fixtures in scope.

**Uncertainty (pattern).** The inclusion `pattern(W'W) subset pattern(A)` is structural and
assumes no exact numerical cancellation in `sparse(W' * spdiagm(0 => observed) * W + ...)`
(`src/grouped_laplace.jl:217`). If a sum of `w_i W_{ij} W_{il}` cancelled to exactly zero
*and* Julia's sparse product dropped the entry, the Cholesky pattern could omit an entry the
gradient still needs, and `takahashi_selinv` would return a structural zero where a nonzero
belongs. I have not verified whether `SparseArrays`' `spmatmul` drops numerically-zero
products; I believe it does not drop them (it allocates on the structural pattern), but I
did not confirm this and did not run Julia. The implementation should assert that every
`(j,l)` it reads from `Sigma` is structurally present rather than silently reading zero.

**Uncertainty (cost).** `takahashi_selinv` is `O(nnz(L))`. For a single non-crossed grouping
term, `A` is block diagonal with `(rank + unique)`-sized blocks per group and `nnz(L)` is
tiny. For crossed terms the fill-in is not bounded by this argument and I make no claim
about the scaling. The `nθ` solves for `u_k` are cheap regardless.

---

## 5. The observed-curvature term in full

The grouped route uses the observed weight `w_i = -d^2 l_i/d eta_i^2`
(`src/grouped_laplace.jl:213`), which for non-canonical families depends on `y_i`. Writing
it out:

```
wdot_k,i = (dw_i/deta_i) * edot_k,i + (dw_i/drho) * dk rho
         = kappa_i * edot_k,i + (dw_i/drho) * dk rho
kappa_i  = - d^3 l_i(eta_i; rho) / d eta_i^3       (evaluated at the mode, with y_i fixed)
edot_k,i = ( D dk gamma )_i + ( (dk W) bhat )_i + ( W u_k )_i
```

**The `y` dependence is not a structural obstacle, and I want to correct the brief on this
point.** `y` is constant in `theta`. The observed weight depends on `theta` only through
`eta` and through `rho`, exactly as a Fisher weight would; `y` changes the *formula* for
`kappa_i`, not the shape of the chain rule. Concretely, for the two fixtures in scope the
observed weight equals the Fisher weight pointwise, because both are canonical:
`_glm_weight_matches_observed(::Poisson, ::LogLink) = true`
(`src/families/poisson.jl:10`) and the same for Binomial with a logit link
(`src/families/binomial.jl:32`). For Poisson with a log link, `w_i = mu_i` and
`kappa_i = mu_i`. For Binomial with a logit link, `w_i = n_i mu_i (1 - mu_i)` and
`kappa_i = n_i mu_i (1 - mu_i)(1 - 2 mu_i)`. The `y`-carrying case arises only for Beta and
NB2, and the per-site precedent **already exercises it** (NB observed weight at
`src/laplace_grad.jl:317`, Beta at `:481`), just by AD rather than in closed form.

**Therefore I do not recommend a Fisher fallback, and I record why.** Two separate reasons.
First, the objective's log-det uses observed curvature, so a Fisher-weight log-det
derivative is not the derivative of this objective: the package has already been burned by
exactly this and left the warning in place at `src/laplace_grad.jl:308-316` and
`:383-390` ("an analytic gradient tuned to a different log-det than the one being reported
is not the gradient of the objective, and it degrades optimisation SILENTLY rather than
erroring"). Second, and independently, the implicit-function matrix has no Fisher variant
available: section 1.4 shows `dg/db = -A` with the observed weight by definition, so using
`Hf` there would be wrong even if the objective used a Fisher log-det.

For the record, the named fallback if `kappa` proves unobtainable for some family is
**`:fd_logdet_direction`**: keep (A1), (A2) and (C1) analytic, and obtain the `(C2 + D)`
scalar per coordinate by central-differencing **only** `ld(theta)` (the value at
`src/grouped_laplace.jl:460`) while holding nothing else fixed. This changes how the
derivative is computed, never what is optimised. Its cost is `2 nθ` extra inner solves, so
it recovers roughly half of the measured 54 percent FD-gradient share rather than all of
it, and it reintroduces the warm-start bias hazard for that half. It is a fallback, not a
plan.

---

## 6. Alignment table

Every symbol in sections 1 to 5, the function or variable that computes it, and where it
lives.

**Reconciled against the implementation on 2026-09-21** (commit `7ae2f8cbf`). This table was
written before any code, so its first version was a plan: every row said `exists` or
**`must be written`**. It is now a map of the shipped code instead, and three things changed
in the move. First, every `must be written` row is written. Second, four rows named functions
the implementation chose to **inline** inside `_grouped_analytic_loglik_gradient` rather than
package separately -- `_grouped_eta_explicit`, `_grouped_mode_rhs`, `_grouped_mode_jacobian`
and `_grouped_eta_total` do not exist under those names, and the rows below now point at the
local variables that carry those quantities, at all three of the sites that compute them
(the mean block, the grouping block, the dispersion block). The computations are present and
in the derived form; only the packaging differs. Third, the derivation **missed a symbol**:
the dispersion block needs `ds/drho` as well as `dw/drho` and `dl/drho`, and its row is added
below rather than left implicit. Line numbers throughout were re-resolved against the current
files, because the S8 change moved most of them.

| Symbol | Meaning | Function / variable that computes it | State |
|---|---|---|---|
| `theta` | outer parameter vector | the `value` argument of the objective closure, `src/grouped_nongaussian_fit.jl:351` | pre-existing |
| `gamma` | mean coefficients, `theta[1:q]` | `gamma`, `src/grouped_nongaussian_fit.jl:354` | pre-existing |
| `psi` | grouping coordinates (loadings, log SDs) | `_grouped_term_unpack`, `src/grouped_fit.jl:125` | pre-existing |
| `rho` | log dispersion | `_grouped_nongaussian_family`, `src/grouped_nongaussian_fit.jl:93-107` | pre-existing |
| `D` | mean design, `N`-by-`q` | `_trait_mean_design`, `src/source_fit.jl:137` | pre-existing |
| `W(psi)` | sparse RE design, `N`-by-`m` | `_grouped_laplace_design`, `src/grouped_nongaussian_fit.jl:164` | pre-existing, Float64 only |
| `dk W` | `dW/dtheta_k`, same pattern as its block | `_grouped_laplace_design_jacobian`, `src/grouped_nongaussian_fit.jl:257` | **written for S8** |
| `bhat` | joint mode | `result.mode`, field of `JointGroupedLaplaceResult`, `src/grouped_laplace.jl:16`, set at `:462` | pre-existing |
| `eta` | linear predictor at the mode | `_joint_grouped_state`, `src/grouped_laplace.jl:178` | pre-existing |
| `s` | score `dl/deta` | `_glm_score` via `_joint_grouped_components`, `src/grouped_laplace.jl:219` | pre-existing |
| `w` | observed weight `-d2l/deta2` | `_glm_obs_weight`, `src/families/laplace.jl:260`, used at `src/grouped_laplace.jl:221` | pre-existing |
| `kappa` | `dw/deta = -d3l/deta3` | `_glm_obs_weight_deta`, `src/grouped_laplace.jl:257` | **written for S8** |
| `dw/drho` | weight sensitivity to log dispersion | `_glm_obs_weight_dphi`, `src/grouped_laplace.jl:280-283` | **written for S8** (Beta, NB2 only) |
| `dl/drho` | log-density sensitivity to log dispersion | `_glm_logpdf_dphi`, `src/grouped_laplace.jl:275-278` | **written for S8** (Beta, NB2 only) |
| `ds/drho` | **score** sensitivity to log dispersion | `_glm_score_dphi`, `src/grouped_laplace.jl:285-288` | **written for S8; MISSING from this table until 2026-09-21** (Beta, NB2 only) |
| `A` (`Ho`) | joint observed precision | `_joint_grouped_components`, `src/grouped_laplace.jl:208`; returned as `result.precision` | pre-existing |
| `Fo` | CHOLMOD factor of `A` | `result.factor`, the field S8 added, `src/grouped_laplace.jl:32`, set at `:462-463` | **exposed for S8**; the pre-S8 code computed and discarded it |
| `ld` | `logdet(A)` | `logdet(Fo)`, `src/grouped_laplace.jl:460`, field `logdet_precision`, `:20` | pre-existing |
| `Sigma` | `A^{-1}` at the selected pattern | `takahashi_selinv`, `src/takahashi_selinv.jl:103`, called from `src/grouped_nongaussian_fit.jl:453` | pre-existing function, **wired to the grouped route for S8** |
| `t_i` | `(W Sigma W')_{ii}` | `_grouped_selinv_row_quadform`, `src/grouped_laplace.jl:314`, called at `src/grouped_nongaussian_fit.jl:459` | **written for S8** |
| `r_i(k)` | `(W Sigma (dk W)')_{ii}` | `_grouped_selinv_row_crossform`, `src/grouped_laplace.jl:342`, called at `src/grouped_nongaussian_fit.jl:484` | **written for S8** |
| `e_k` | explicit `d eta / d theta_k` | INLINED as `e_expl`: `src/grouped_nongaussian_fit.jl:467` (mean block, `D[:,k]`), `:478` (grouping block, `dW * bhat`); identically zero in the dispersion block, so no variable exists there | **inlined, not a named function** |
| `v_k` | `dg/dtheta_k` at fixed `b` | INLINED as `rhs`: `:468`, `:479`, `:501` | **inlined, not a named function** |
| `u_k` | `dbhat/dtheta_k = A^{-1} v_k` | INLINED as `u = Fo \ rhs` (CHOLMOD solve): `:469`, `:480`, `:502` | **inlined; no wrapper written** |
| `edot_k` | total `d eta / d theta_k` | INLINED as `edot`: `:470`, `:481`, `:503` | **inlined, not a named function** |
| `wdot_k` | total `d w / d theta_k` | INLINED as `wdot`: `:471`, `:482`, and `:498` + `:504` in the dispersion block, where the explicit `dw/drho` part is accumulated first and the implicit `kappa .* edot` part added to it | **inlined**, as this table always said |
| `grad L` | gradient of the Laplace marginal | `_grouped_analytic_loglik_gradient`, `src/grouped_nongaussian_fit.jl:403` | **written for S8** |
| `grad F` | gradient of the minimised objective, `-grad L` | `_grouped_analytic_gradient`, `src/grouped_nongaussian_fit.jl:521` | **written for S8** |
| FD reference | central difference of `F` | `_grouped_fd_gradient`, `src/grouped_fit.jl:213` | pre-existing |

Three rows deserved emphasis before the code existed, because they were the ones that could be
satisfied *incorrectly* rather than merely being absent: `dk W` (scale, section 7.4), `kappa`
(availability, section 7.6), and `Fo` (then discarded, so a naive implementation would
refactorise `A` from `result.precision`, which is correct but pays for a second factorisation).
All three were resolved the way the section argued they should be: `dk W` has its own function,
`kappa` is available because all three families differentiate three levels (probed directly,
2026-09-21), and `Fo` is now carried on the result rather than recomputed.

What the reconciliation did NOT change: no equation in sections 1 to 5 moved, and no line of
`src/` was edited to match the table. Where the table and the code disagreed, the code was
taken as correct and the table was rewritten, because the code is what GB.2 tested per
coordinate and what GB.3 compared against origin/main.

---

## 7. What could silently go wrong

Each item below produces a *finite, plausibly-scaled* gradient. None of them errors. They
are ordered by how likely I think they are to survive a careless test.

**7.1 Dropping term (D), the implicit contribution through the log-det.** The most likely
error, because the incorrect version is the one that needs none of the machinery in
sections 2 and 4: no `u_k`, no `Fo` solve, no selected inverse. The resulting gradient is
wrong by `-0.5 tr(A^{-1} (dA/db) u_k)` in every coordinate. It has the right sign and the
right order of magnitude in the mean coordinates, where (A1) dominates. Its signature is
that the optimiser converges to a theta where the *incomplete* gradient vanishes, which is
a genuinely different point from the FD optimum, so a "does the fit still land in the same
place" test at loose tolerance will pass. I have not measured the size of this term and
will not guess at it; the test in section 8 must be able to see it at the FD reference's own
accuracy rather than at an invented bound.

**7.2 Sign slip on the `-0.5 ld` direction, or forgetting `F = -L`.** Two different sign
errors with different signatures. Forgetting the outer negation flips every coordinate and
is caught instantly by anything. Flipping only the log-det direction is the dangerous one:
in the mean coordinates the log-det contribution is a small correction to (A1), so those
coordinates still match FD to a couple of digits, while the log-SD coordinates, which are
almost entirely log-det, are wrong by roughly a factor of `-1`. **Consequence for testing:
the agreement test must be per-coordinate, never on the gradient norm or on a cosine
similarity.**

**7.3 Losing the factor of 2 on the design-derivative trace (C1).** The symmetric pair
`(dk W)' diag(w) W + W' diag(w) (dk W)` collapses to `2 x` one trace, and halving it is an
easy slip. The resulting error is **identically zero in every `gamma` coordinate**, because
`dk W = 0` there. An intercept-only or Poisson-intercept-only FD check passes perfectly and
tells you nothing. This is the strongest argument for testing every coordinate, including a
fixture where at least one loading coordinate is nonzero.

**7.4 The `exp(2 theta)` versus `exp(theta)` scale on the unique-variance block.** `theta_j`
is a log standard deviation, `d_j = exp(2 theta_j)` is a variance
(`src/grouped_fit.jl:150`), and the entry that enters `W` is `sqrt(d_j) = exp(theta_j)`
(`src/grouped_nongaussian_fit.jl:147`). A derivative written against the variance rather
than the design entry is wrong by exactly a factor of 2 on those coordinates and on no
others. This is the classic densities-and-parameterisations mismatch of this codebase's
family (the `meanlog`/`sdlog` and `1/sigma^2` class) and it will not announce itself: a
factor-2 gradient in a subset of coordinates still descends, just along the wrong path, and
BFGS will absorb some of it into the inverse-Hessian estimate.

**7.5 Using `Hf` where `A` belongs.** `joint_grouped_laplace_loglik` keeps two factor
caches, `ff_cache` for the Fisher precision and `ho_cache` for the observed one
(`src/grouped_laplace.jl:290-291`). Picking up `Ff` for the `u_k` solve or for the selected
inverse is a one-character-class error. For Poisson and Binomial the two weights coincide
pointwise (`src/families/poisson.jl:10`, `src/families/binomial.jl:32`) so **a Poisson
regression test cannot detect it at all**; Beta and NB2 would fail. Any FD gate that ships
Poisson-only leaves this live.

**7.6 `kappa` obtained by a third nested ForwardDiff.** `_glm_obs_weight`'s default is
already two nested `ForwardDiff.derivative` calls through the coded log-density
(`src/families/laplace.jl:260-264`), so `kappa` invites a third nesting. I flag two
hazards. (a) Availability: `_glm_logpdf(::Poisson, mu, n, y) = logpdf(Poisson(mu), Int(y))`
(`src/families/poisson.jl:13`), and `src/laplace_grad.jl:28-30` records that the per-site
route needed a hand-written `_pois_logpmf` because of "Distributions' `logpdf(::Poisson,
::Int)` under a Dual mean". Two levels of nesting evidently work today in the grouped route;
I have not verified that three do, and I did not run Julia to check. (b) The clamp
convention: `_glm_obs_weight`'s docstring warns that where `_clamp_mu` binds, the AD
fallback returns the derivative of the *clamped* composition, zero in the saturated region
(`src/families/laplace.jl:253-258`). For this route that hazard is defused, because
`_joint_grouped_state` rejects any saturated point outright rather than clamping through it
(`src/grouped_laplace.jl:174-179`), so the gradient is only ever evaluated in the interior.
I state this as resolved, not as ignored.

**7.7 A loosely converged inner mode.** The derivation assumes `g(bhat) = 0` exactly. At a
finite `tol` the dropped piece is `g(bhat)' u_k`, first order in the residual. The default
`inner_tol = 1e-8` (`src/grouped_nongaussian_fit.jl:571`) should make this negligible
relative to the FD reference's own floor, but the analytic gradient and the FD gradient
degrade *differently* as `tol` loosens, so a test run at a loose `inner_tol` can show a
disagreement that is the fixture's fault rather than the gradient's. Record `inner_tol` in
the test.

**7.8 Reading `Sigma` in the permuted ordering.** `takahashi_selinv` already un-permutes
(`src/takahashi_selinv.jl:182-196`), so this is only a risk if a future optimisation reaches
for `_takahashi_selinv` directly. A permuted `Sigma` gives entirely plausible magnitudes and
a completely wrong trace.

**7.9 Dropping (A2) or `dw/drho`.** Beta and NB2 only. The gradient is exact in every
coordinate except the dispersion ones, so a test that reports a single summary number over a
6-coordinate Poisson fixture never sees it.

The common thread: **every one of 7.3, 7.4, 7.5 and 7.9 is invisible to a Poisson,
intercept-dominated, norm-summarised check.** The test design in section 8 follows from
that, not from a general preference for strictness.

---

## 8. The finite-difference agreement test

### 8.1 Construction

Reference: `_grouped_fd_gradient(objective_cold, theta)` (`src/grouped_fit.jl:213`) against
the **cold** closure (`warm_start_inner = false`,
`src/grouped_nongaussian_fit.jl:632-634`). Not the warm one: the warm closure's inner mode
depends on cache state, which is what biased the FD gradient to 1e-4 against a 1e-4
criterion (`src/grouped_nongaussian_fit.jl:350-370`). Once the analytic gradient exists, the
production path has no FD step to bias, but **the test's reference still does**, so the
reference stays cold permanently.

Assertion form, per coordinate `k`:

```
abs( g_analytic[k] - g_fd[k] )  <=  rtol * abs(g_fd[k]) + atol
```

with `rtol = 1e-6` and `atol = 1e-6 * maximum(abs, g_fd)`. The absolute floor exists because
a coordinate whose true gradient is near zero has no meaningful relative scale.

### 8.2 Why 1e-6 is the reference's accuracy and not a widened bound

`_grouped_fd_gradient` uses a central difference at `h = 1e-5 * max(1, abs(theta_j))`
(`src/grouped_fit.jl:215`). Its two error sources are truncation, `(h^2/6) abs(L''')`, about
`1.7e-11 * abs(L''')`, and cancellation, about `eps * abs(F) / h`, about `2.2e-11 * abs(F)`.
On the large fixture `abs(F)` is of order `1e4`, so cancellation alone puts an absolute floor
near `2e-7` on each component. Independently, this package has already measured the
combined floor: the cold FD gradient norm at a converged optimum sits at about `1e-7`
(`src/grouped_nongaussian_fit.jl:360-364`), which is an empirical read on the same quantity.
So `1e-6` is roughly one order above the instrument's demonstrated resolution, which is the
standard and defensible place to set a gate. Anything looser is a bound chosen to make the
test pass, and in particular a `1e-2`-class tolerance would not reliably see 7.1.

### 8.3 Certify the instrument before trusting it

Before asserting agreement, compute the reference at `h` and at `h/2` and require

```
abs( g_fd_h[k] - g_fd_halfh[k] )  <=  rtol * abs(g_fd_h[k]) + atol
```

for the same `rtol`, `atol`. If the two disagree by more than the tolerance, the FD
reference is not accurate enough to adjudicate at that tolerance and the test must report
*that*, not a gradient failure. This converts the tolerance from an assumption into a
measurement and is the honest answer to "why that rtol".

### 8.4 Which test catches which failure

| Failure | Caught by |
|---|---|
| 7.1 dropped implicit log-det term | per-coordinate FD at 1e-6, **evaluated away from the optimum** (at the optimum the true gradient is near zero and the atol floor masks a small absolute error); plus the check that `g_analytic` vanishes at the FD-located optimum |
| 7.2 log-det sign | per-coordinate FD; the log-SD coordinates fail by roughly 2x their own magnitude while the mean coordinates pass |
| 7.3 lost factor of 2 on (C1) | per-coordinate FD **on a fixture with a nonzero loading coordinate**; invisible on a mean-only fixture |
| 7.4 `exp(2 theta)` scale | per-coordinate FD on the unique-variance coordinates; a clean factor of exactly 2 in exactly that block is the signature |
| 7.5 Fisher instead of observed | per-coordinate FD **on a Beta and an NB2 fixture**; a Poisson or Binomial fixture cannot detect it |
| 7.6 `kappa` unavailable or clamp-truncated | finiteness assertion on `kappa`, plus per-coordinate FD; if `kappa` silently returns zero in some region the log-SD coordinates go wrong |
| 7.7 loose inner mode | re-run the same assertion at `inner_tol = 1e-10`; a disagreement that shrinks is the fixture, one that persists is the gradient |
| 7.8 permuted `Sigma` | a dense cross-check on a small fixture: compare `t = diag(W Sigma W')` against `diag(W * inv(Matrix(A)) * W')` at `rtol = 1e-10`, which isolates the selected-inverse wiring from the calculus |
| 7.9 dropped dispersion terms | per-coordinate FD on Beta and NB2, dispersion coordinate specifically |

Two levels, deliberately: the dense cross-check in the 7.8 row runs at machine precision and
adjudicates the *plumbing*, while the FD comparison runs at 1e-6 and adjudicates the
*calculus*. A single test at a single tolerance cannot separate those two, and when it fails
it will not tell you which one broke.

### 8.5 Fixture requirements implied by the table above

At minimum: one Poisson fixture with at least one nonzero loading coordinate and at least
one unique-variance coordinate; one Beta or NB2 fixture with a dispersion coordinate; one
small fixture (`m` in the tens) where a dense `inv(A)` is affordable. All three evaluated at
a theta **away from the optimum**, and the Poisson one also evaluated at the optimum for the
stationarity check.
