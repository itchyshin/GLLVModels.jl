# Second-order contract §6 holdouts (frozen list, 2026-09-04)

**Status 2026-09-25:** the Tweedie row below is stale where it says the
species estimated-power second-order cell is "still open" or "not
attempted". PR #391 (merged 2026-09-16) wired both the shared and species
Tweedie estimated-power second-order cells (option A, PARTIAL: β/`b_fix`
block only, plug-in power, not a D1 each-own-optimum pass). The Delta-lognormal
/ Delta-Gamma row is not stale: `accept delta dispersion A` is still listed
as an outstanding maintainer paste in `LOOP/checkpoint.md` as of this date,
so Delta remains pending as written below.

Source: `second-order-parity-contract.md` §6–§7. These families/cells are **out of the
first second-order batch claim** until the listed blocker clears.

| Holdout | Blocker | Disposition |
|---|---|---|
| Binomial-cloglog | ~~§2 disputed~~ **ACCEPTED (A) 2026-09-15** — `:observed` ratified | Each-own-optimum D1 promotable (`binomial_cloglog`); ≠ §7 |
| Tweedie shared + grouped | Default `:observed` ratified (A); **`TweedieGroupedFit` ∈ `_CIFit`** + fixed-power `tweedie_fixed` + estimated-shared `tweedie_shared` (option A, β/`b_fix` only; 2026-09-16) | First-order parity exists; species estimated-power SO cell still not attempted |
| GP-1 | Fisher-retained; parity vs TMB Fisher alternative unsettled | Out of scope pending ruling |
| Student-t fixed-ν | First-order logLik parity; native Wald in `_FamilyFit` (2026-09-15, fixed ν only; #367) | Native fixed-ν Wald + paired SO cell `studentt_fixed_nu` (2026-09-16; β[] block) |
| Student-t free ν | Nonlinear boundary; Wald SE pathology (panel finding 4) | Excluded; `_family_ci` throws ArgumentError |
| Ordinal per-trait cutpoints | **OrdinalPerTraitFit + OrdinalPerTraitCovFit ∈ `_CIFit` 2026-09-15** | Native Wald + paired SO cell `ordinal_pertrait_probit` (2026-09-16; β[] block) |
| Lognormal, Truncated-Poisson, Truncated-NB2 | **LognormalFit + TruncatedPoissonFit + TruncatedNegBin2Fit ∈ `_FamilyFit` 2026-09-15** | Native Wald + paired SO cells `lognormal` / `truncated_poisson` / `truncated_nbinom2` (2026-09-16) |
| Delta-lognormal, Delta-Gamma | Julia shared scalar dispersion vs R per-trait dispersion (PR #347 measured FAIL) | Decision pending: `accept delta dispersion A` |
| Multinomial FE softmax | **MultinomialFit ∈ `_CIFit` 2026-09-15** (FE softmax) | Native Wald + paired SO cell `multinomial_fe` (2026-09-16) |
| BetaBinomial shared-φ | **BetaBinomialFit ∈ `_CIFit`** (Wald pre-existing); **`betabinomial_shared` SO cell** (2026-09-15) | Native Wald + paired toy cell (β block only; φ not R-paired); ≠ §7 |
| Realistic Gaussian grid | Intercept-`X` patch cleared estimand mismatch (2026-09-04) | Repaired: 8/8 cells pass SE D1; outside toy batch-1 |
| Loadings Λ raw entries | Rotation ambiguity (§3) | Compare derived quantities or Procrustes (deferred) |

**Batch 1 (contract §6, five families):** Gaussian, Poisson-log, Binomial-logit, Beta-logit,
NB2-log — covered by toy 20-cell grid (superset) and merged-tip refresh
(`second-order-batch-out-20260904-merged-tip/`).

---

## Disposition pass (Option D, 2026-09-05)

**Scope:** contract §6 holdouts only. No programme-level second-order parity claim
(contract §7). Status legend:

- **OUT** — excluded from any batch claim until blocker clears
- **PARTIAL** — first-order or health-only / each-own-optimum receipts; no second-order bind
- **NOT ATTEMPTED** — no honest paired run in this arc

**Summary:** OUT **5** · PARTIAL **10** · NOT ATTEMPTED **0** (plus batch-1 NB2 noted below; BB shared-φ SO cell 2026-09-15)

| Holdout | Status | Reason (measured) | Evidence |
|---|---|---|---|
| Binomial-cloglog | **PARTIAL (SO promoted)** | §2 **(A)** 2026-09-15; `hessian_selector_disputed=false`; SE max rel Δ ≈ 2.1e-5 each-own-optimum | `docs/dev-log/core070/second-order-batch-out/binomial_cloglog.json` |
| Tweedie shared + grouped | PARTIAL (SO wiring + `tweedie_fixed` + `tweedie_shared`) / PARTIAL (1st) | §2 default signed; Wald + `tweedie_fixed`; estimated shared option A cell 2026-09-16 (β/`b_fix` only; power plug-in). Species SO still open. ≠ D1 until smoke JSON | `test/test_second_order_tweedie_grouped_ci.jl`; `tools/core070_second_order/{common,cells}.jl` |
| GP-1 | OUT | Fisher-retained on Julia side; no ruling on whether R parity compares against a TMB Fisher alternative | Contract §6 lines 205–207; `GP1Fit` in `_CIFit` but no paired SO cell |
| Student-t ν (free) | OUT | Wald SE pathology at ν boundary (§3); `_family_ci` rejects `estimated_nu=true`; no SO pairing | Contract §3 lines 120–123; `test/test_second_order_studentt_ci.jl` |
| Student-t fixed-ν | **PARTIAL (native Wald + SO cell)** | `StudentTFit` Wald (#367); paired `studentt_fixed_nu` cell (species σ, fixed ν; live Δ se_rel ≈ 3e-6); bridge untouched | `test/test_second_order_studentt_ci.jl`; `tools/core070_second_order/cells.jl` |
| Ordinal per-trait cutpoints | **PARTIAL (native Wald + SO cell)** | `_CIFit` wired; paired `ordinal_pertrait_probit` (β[] block; live Δ se_rel ≈ 1e-5); bridge guard held (#357) | `test/test_second_order_ordinal_pertrait_ci.jl` |
| Lognormal | **PARTIAL (native Wald + SO cell)** | `_FamilyFit` Wald; paired `lognormal` (β[] block; σ not paired); live Δ se_rel ≈ 8e-6 | `test/test_second_order_lognormal_ci.jl` |
| Truncated-Poisson | **PARTIAL (native Wald + SO cell)** | paired `truncated_poisson`; live Δ se_rel ≈ 9e-6 | `test/test_second_order_truncpois_ci.jl` |
| Truncated-NB2 | **PARTIAL (native Wald + SO cell)** | paired `truncated_nbinom2` (β[] only; Julia shared `r` vs R per-trait φ — live Δ se_rel ≈ 0.098) | `test/test_second_order_truncnb2_ci.jl` |
| Truncated-NB2 — **2026-09-25 update** | **PARTIAL (native Wald + SO cell, like-with-like)** | Cell re-pointed to `fit_truncated_nbinom2_gllvm_pertrait` (gllvmTMB's `truncated_nbinom2()` has no shared-dispersion mode — it always fits per-trait `log_phi_truncnb2` — so the shared-`r` pairing above compared two different models) on a new screened fixture, `test/parity/fixtures/truncnb2_interior_seed61_n150.toml` (seed=61, n=150; the old seed=58/n=120 shape pushes one trait's dispersion to the Poisson-limit boundary on **both** engines (Julia `r ≈ 2.9e9`, R `phi ≈ 1.2e7`), a shared boundary, not an R-only or `se=TRUE` artefact; corrected 2026-09-26, `truncnb2-interior-screen-20260925.md`). Each-own-optimum, β[] block only: SE max rel Δ = 2.3e-5 (tol 1e-2, `cond(H)_R`=187 so unscaled) **PASS**; vcov Frobenius rel Δ = 2.9e-5 (tol 1e-2) **PASS**; Wald CI endpoint rel-to-half-width Δ = 5.8e-5 (tol 5e-2) **PASS**. The old shared-r/seed=58 pairing's CI-endpoint metric, computed the same way, is 0.37 — a clear **FAIL** against the same 5e-2 bar; its SE/vcov deltas (0.106, 0.317) only looked like passes because `cond(H)_R` there was 163,115, inflating the conditioning-scaled tolerance to 1.63 — an artifact of the mismatch's own ill-conditioning, not agreement. The per-trait Wald adapter's own boundary handling (`_FamilyCI`'s 7th `boundary` argument) and a separate seed-65/n=150 finding (Julia's per-trait fitter itself stalls at an inferior interior optimum while reporting `converged = true`) were caught by an independent review of #493 and are tracked in #499. | `docs/dev-log/core070/second-order-batch-out/truncated_nbinom2.json`; `docs/dev-log/core070/truncnb2-interior-screen-20260925.md` |
| NB2-log (batch-1) | PARTIAL (SO) | In batch-1 each-own-optimum receipts (20/20 SE D1 toy; 5/5 batch-1 smoke); **not** matched-coordinates; vcov full-block skipped on boundary | `second-order-d1-gate-receipt-2026-09-04.json`; `t14-nb2-wald-nan-diagnosis.md` |
| Delta-lognormal, Delta-Gamma | OUT (paired SO) / PARTIAL (native Wald) | SO wiring landed in PR #347; D1 comparison fails on dispersion parameterisation (Julia shared vs R per-trait); pending decision A | `after-task/2026-09-15-second-order-delta-followup.md`; `decisions/2026-09-15-delta-dispersion-alignment-pending.md` |
| Multinomial FE softmax | **PARTIAL (native Wald + SO cell)** | paired `multinomial_fe`; live Δ se_rel ≈ 5e-6 | `test/test_second_order_multinomial_ci.jl`; `tools/core070_second_order/cells.jl` |
| BetaBinomial shared-φ | **PARTIAL (native Wald + SO cell)** | `BetaBinomialFit` Wald was already wired; added `cell_betabinomial_shared` (β[] block vs R per-trait φ; φ not paired) | `test/test_second_order_betabinomial_shared_ci.jl`; `tools/core070_second_order/cells.jl` |
| Realistic Gaussian grid | PARTIAL | **Repaired 2026-09-04:** intercept-`X` patch clears estimand mismatch; 8/8 realistic Gaussian cells SE D1 pass (max rel ΔSE β = 2.2e-5). Still outside toy batch-1 scope | `second-order-gaussian-intercept-disposition-2026-09-04.md`; `realistic-size-pairing-disposition-2026-09-04.md` |
| Loadings Λ raw | OUT | Rotation ambiguity (§3); compare derived Σ_y / communality / correlation instead | Contract §3 lines 108–119 |

**Advisory note (CI + local 0.7.1):** Retained pinned-build oracle remains authority
(`ci-oracle-reproducibility-finding.md`). Local live **0.7.1** smoke is receipt-only —
see `advisory-r071-smoke-2026-09-05.md` (NB2 health gate fails on advisory build; not
a holdout disposition upgrade).

**Claim boundary (§7):** A passing batch-1 or realistic-size D1 gate is wiring +
tolerance read on named fixture shapes only. It is **not** second-order parity,
matched-coordinates parity, coverage, or holdout clearance.
