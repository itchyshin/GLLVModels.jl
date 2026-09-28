# Review: PR #576 — fix(twolevel): reject negative quadratic forms in the two-level marginal

- Branch `claude/fix-icc-bootstrap-lower-bound`, head `999c01cc9`, base `cb5688f7e` (parent of head; GitHub's recorded base OID `880cad4c7` is stale but the merge-base with `origin/main` is `cb5688f7e`). One commit, 3 files (+60/-0).
- Reviewer worktrees: `/Users/z3437171/local-scratch/lanes/GLLVM.jl-review-576` (head) and `…-base` (cb5688f7e), both removed after review. Julia 1.10.0 (`~/.juliaup/bin/julialauncher`), `OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=4`. Scripts and logs under the session scratchpad (`review.jl`, `dense_alt.jl`, `runfiles.jl`; `review-{head,base}.log`, `dense_alt.log`, `tests-head.log`, `tests-base-twolevel.log`).
- Date: 2026-09-27.

## Verdict: NON-BLOCKING

The change is safe, strictly monotone (it can only turn an impossible finite value into a rejected evaluation), bit-identical in the valid region, red-then-green as claimed, and it reproduces the claimed bootstrap bounds exactly. It is, however, a **symptom guard, not a root-cause fix**: the Woodbury solve for `Σ_W⁻¹` is unstable well before the quadratic form goes negative, and I measured finite log-likelihoods overstated by up to **+135 nats** with the guard silent. A two-line dense-Cholesky replacement (the repo's own precedent in `src/families/gaussian_pervar.jl`) removes the whole failure class with ≤ 7e-13 error across every point I tried, keeps AD working, and gives equivalent bootstrap bounds. I recommend merging #576 as-is only if the follow-up in Finding 1 is filed and the CHANGELOG wording in Finding 2 is tightened; otherwise fold the dense solve into this PR.

## Findings

### 1. The `-Inf` guard catches only the sign of the failure; finite, overstated log-likelihoods still pass through (should fix — root cause)

Evidence. Sweep of `twolevel_marginal_loglik` (Float64, head) against a BigFloat dense reference, at the fixture's converged point (`fit_twolevel_gaussian(Y_tl, individual; K_B=1, K_W=1)`, loglik -271.95) with one `σ²_W[t]` pushed to `10^-k` (`review.jl`, "Sweep B"; `err = ll_float64 - ll_exact`, positive = overstated; last column = same point through a dense Float64 Cholesky of `Σ_W`):

```
t k   ll_float64      ll_exact        err=ll_f-ll_x   guard   dense64_err
4 12  -1172.132       -1172.1336      0.001608        no      -6.82e-13
4 14  -1171.9754      -1172.1336      0.1581          no      -2.27e-13
4 15  -1171.645       -1172.1336      0.4886          no      -6.82e-13
4 16  -1164.1231      -1172.1336      8.01            no      -4.55e-13
4 17  -1037.0103      -1172.1336      135.1           no      -4.55e-13
4 18  -Inf            -1172.1336      -Inf            -Inf    -4.55e-13
4 25  -1.5358199e+10  -1172.1336      -1.536e+10      no      -2.27e-13
2 15  -297.31446      -298.77037      1.456           no      0
2 16  -293.45472      -298.77037      5.316           no      5.68e-14
2 17  -Inf            -298.77037      -Inf            -Inf    0
2 18  -2611.8434      -298.77037      -2313           no      0
```

Same picture on the PR's own broken-replicate parameters ("Sweep A", `σ²_W[2] = 10^-k` with the other three entries at their replicate values 2.2e15 / 1.5e14 / 2.9e11): the error is already 0.06 at k=14 and -0.57 at k=15, the guard fires at k=16–18 and 25–30 but *not* at k=20 (`-1.15e5`, wrong but finite) or k=37 (`-3.5e21`). Random-parameter sweep ("Sweep C", 200 draws, `|Λ_W| ~ 5`, `σ²_W[2] = 10^-k`, k ∈ 10..40): guard fired 86/200, **9/200 finite and overstated by > 1e-6 relative** (worst +0.41 nats).

Interpretation. The Woodbury form `Σ⁻¹V = D⁻¹V − D⁻¹Λ(I+ΛᵀD⁻¹Λ)⁻¹ΛᵀD⁻¹V` subtracts two terms of size `~Λ²/σ²_W[t]`; the rounding error is `~eps · Λ²/σ²_W[t] · |y|²`, of either sign. The guard rejects the negative-sign half only once the error exceeds the true value. In the band `σ²_W[t] ∈ [1e-17, 1e-13]` (for `|Λ_W| ~ 4`) the error is O(0.01–100) nats, finite, and positive half the time. A bootstrap refit that wanders there can still report `converged = true` at an inflated loglik. I did not observe such a replicate in the seed-11/12/13 runs (no retained replicate has `loglik > 0` after the fix, and the bounds match R), so this is a residual risk rather than a demonstrated regression — but the header comment's claim that the path "stays robust as σ²_W → 0" is measured false.

Suggested fix (propose, not implemented). Replace the Woodbury solve for `Σ_W` with a dense `p×p` Cholesky, exactly as the mean part (`Σ_W + n_i Σ_B`) already does three lines below, and as `src/families/gaussian_pervar.jl` lines 16–18 / 315–316 chose deliberately ("subtractive Woodbury solves can lose the quadratic form near a zero residual variance"). p is small in the two-level regime (the file says so itself for the mean part), so cost is irrelevant. I tested this by monkey-patching `_woodbury_core` (`dense_alt.jl`):

```julia
function G._woodbury_core(Λ, d)
    c = cholesky(Symmetric(G._dense_sigma(Λ, d)))
    return (V -> c \ V), logdet(c)
end
```

Results: GATE 1b point evaluates to `-622.7947834877165` (exact -622.795); error ≤ 7e-13 at every sweep point above including k=30; ForwardDiff gradient finite; point fit `loglik = -271.9517`, repeatability agrees with the Woodbury fit to 1e-8 (not bit-identical — expected from a different solver, so this is a separate PR with its own parity re-check); bootstrap (nsim=200):

```
seed 11 kept=163  lower=[0.600, 0.139, 0.137, 0.418]  upper=[0.820, 0.463, 0.442, 0.726]
seed 12 kept=169  lower=[0.588, 0.123, 0.130, 0.435]  upper=[0.819, 0.434, 0.462, 0.741]
seed 13 kept=160  lower=[0.549, 0.115, 0.135, 0.426]  upper=[0.820, 0.452, 0.432, 0.718]
(R gllvmTMB P1:      lower=[0.571, 0.126, 0.127, 0.414]  upper=[0.829, 0.468, 0.441, 0.720])
```

The ~17% attrition is unchanged under the dense solve, which supports the PR's reading that the dropped replicates are a boundary/`g_tol` issue, not this numerical one. The `-Inf` guard can stay as a belt-and-braces check after the dense swap.

### 2. Comments and CHANGELOG overstate what the guard delivers (should fix — wording)

- `src/twolevel.jl` lines 27–28 and 34–36 still say the Woodbury path "stays robust as σ²_W → 0" / "stays well-conditioned as d → 0". Finding 1 shows the opposite; the PR's new comment (lines 104–108) describes the symptom without correcting the claim above it.
- GATE 1b's comment: "The marginal must never overstate the exact (BigFloat, dense) value." The test pins one point; the property is not delivered in general (Finding 1). Suggest: "…must not overstate the exact value **at this replicate's parameters**".
- CHANGELOG entry: "the solve loses all precision and the quadratic form comes out negative … A negative quadratic form now returns `-Inf`". Accurate for what was changed, but a reader will take it as the bug being closed. Suggest adding one sentence: the Woodbury solve remains inaccurate for `σ²_W[t]` below ~1e-13 relative to `Λ_W²`; a dense-Cholesky replacement is tracked separately.

### 3. Optimiser and AD behaviour through the guard — OK (no action)

- `fit_twolevel_gaussian` (line 234–242) already maps non-finite objective values to the `1e12` sentinel; `_fit_verdict` (`src/fit_verdict.jl:57`) refuses to report the sentinel as converged. So `-Inf` never reaches L-BFGS. Measured at the GATE-1b θ (head): raw `nll = Inf`; `ForwardDiff.gradient` of the raw closure is all `-0.0`, finite; fitter closure value `1.0e12`, gradient all `0.0`. `convert(T, -Inf)` with `T <: ForwardDiff.Dual` is fine.
- The Wald closure (`repeatability_wald_ci`, line 427–430) has no sentinel, but it is only evaluated at the MLE, where the guard cannot fire (both quadratic forms are the converged fit's own finite values). If it did fire, `H` would be non-finite and the existing `all(isfinite, H)` check returns `method = :failed`. Profile is withdrawn by design. No other callers of `_twolevel_loglik` exist (`grep -rn _twolevel_loglik src test`: lines 136, 238, 429 in `twolevel.jl`, plus the tests).
- `quad_mean < 0` is effectively dead: `cMean \ mi` is a dense Cholesky solve whose quadratic form cannot go negative unless `cholesky` throws first (caught → sentinel). Harmless.

### 4. Bit-identity in the valid region — confirmed (no action)

Same script on base and head (`review-base.log` vs `review-head.log`), identical UInt64 bit patterns:

```
theta_hat bits: 3fe5811a1349b6dc,3fd541fc03834356,…,c000ce113d445f2e   (16 params, all equal)
loglik bits:    c070ff3a2d46bf22   loglik=-271.9517033351459 conv=true iter=333
Wald CI (4 traits): identical to the last digit, e.g. trait 2 lower=0.16032540566671577 upper=0.5002249326525043
random fits 1–3: loglik bits c088965337bbc5f9 / c088729e113deebe / c087da14327995ea on both
```

### 5. Red-then-green — confirmed (no action)

```
# base cb5688f7e, head's test/test_twolevel.jl
marginal never overstates exact value at extreme variance ratios: Test Failed at test_twolevel.jl:86
   Evaluated: 7.852801119452123e21 <= -622.7947772597687
Test Summary: | Pass  Fail  Total  ->  75  1  76
# head 999c01cc9, five listed files
Test Summary: | Pass  Total     Time
pr576 files   |  948    948  2m59.9s
```

Base bootstrap (seed 11, nsim 200) reproduces the reported collapse exactly (`kept=153`, lower `1.06e-7` and `4.53e-40` for traits 2 and 4); head reproduces `kept=166`, lower `[0.6005, 0.1538, 0.1292, 0.4001]`, upper `[0.8196, 0.4452, 0.4418, 0.7257]`. Matches the PR table.

### 6. Same subtractive-Woodbury quadratic form elsewhere (list only; not fixed here)

All compute `D⁻¹v − D⁻¹Λ(I+ΛᵀD⁻¹Λ)⁻¹ΛᵀD⁻¹v` (or the `(b − Λy_K)./d` variant) and have the same error mode when any `d[t]` is tiny relative to `Λ[t,:]²`:

| site | diagonal `d` | per-trait ratio risk |
|---|---|---|
| `src/likelihood.jl:179–201` (Gaussian marginal, non-phylo) | `d_total = σ²_eps (+ σ²_B[t] + σ²_W[t])` | scalar-σ path: uniform, low; with per-trait `σ²_B`/`σ²_W`: same risk as here |
| `src/profile.jl:210, 221, 260, 458, 465, 498` (σ_eps profile-out) | `σ²_eps`-based | uniform, low |
| `src/reml.jl:51` (`_gaussian_gls`) | `_gaussian_d_total(...)` | same as likelihood.jl |
| `src/likelihood_sparse_phy.jl:342–357` (`_woodbury_apply[_matrix]`), `src/sparse_phy_grad.jl:231, 235`, `src/em_phylo.jl:245` | `d_total` | same as likelihood.jl |
| `src/lowrank_cholesky.jl:89–160` (`LowRankPlusDiagChol`, `ldiv!`) | user `d` | generic; only exported from `GLLVModels.jl`, no internal callers found |
| `src/families/gaussian_pervar.jl` | per-trait `φ²` | **already dense Cholesky, deliberately** (lines 16–18, 315–316) — the precedent for Finding 1 |

No `< 0` guard exists at any of these sites. Whether any of them is reachable at extreme ratios in practice was not checked.

### 7. CHANGELOG / PR body honesty and hygiene — OK apart from Finding 2

- Numbers in the PR table reproduced exactly for base and head (seed 11); dense-alt seeds 12/13 agree with the PR's seed-12/13 rows to ±0.02.
- "I did not run the full `Pkg.test()` suite locally" is stated; I did not either (five files only, 948/948).
- No agent `@handles` in the commit message, CHANGELOG, or PR body (`git show 999c01cc9 | grep '@'` finds only the `Co-Authored-By` trailer e-mail).
- The commit is one concern; staging is by name (3 files).
- CHANGELOG entry sits under `## Development › ### Fixed`, consistent with its neighbours.

## What I did not check

- Julia 1.13 (only 1.10.0 available on this machine via juliaup default); the PR reports both.
- The full `Pkg.test()` suite, Aqua/JET.
- Whether any *retained* bootstrap replicate after the fix sits in the finite-but-overstated band of Finding 1 (I checked only that none has `loglik > 0`, and that the bounds match R). Proving absence would need a per-replicate BigFloat re-evaluation over all 166 kept fits — about 5 min extra; recommended for the follow-up PR rather than this one.
- The R side (`bootstrap_Sigma.R` in the diag directory) beyond confirming that the drop-non-converged / type-7 percentile logic matches what the PR body says.
- Reachability of the other Woodbury sites in Finding 6 at extreme ratios.
- The regenerated parity receipt on the true-parity branches (out of scope per PR body).
