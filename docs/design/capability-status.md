# GLLVModels.jl capability status (twin of gllvmTMB)

Mission Control input for `/p/gllvmTMB/julia-surface`. **GLLVModels.jl is a twin of
gllvmTMB**: public capability rows use the **same R vocabulary**
(sources × modes, families, intervals, slopes) so the board shows R↔Julia
alignment. The *code behind* a row may be Julia (closed-form / dense Laplace /
sparse phy / SPDE) rather than TMB — that is an engine difference, not a
different product taxonomy.

Status words (MC parser; counts derived at render time — never hand-typed into
`status/*.json`):

- `implemented` — Julia code under `src/` **and** a test (`test/` and/or gated
  `test/parity/`) for this capability row (or its clear twin analogue).
- `rejected` — deliberately refused, fail-loud, or not advertised.
- `planned` — tracked / designed; no promoted twin-complete implementation yet.
- `missing` — no Julia implementation found for this R-parallel row.

**Rose fence.** Intended API similarity ≠ full parity claim.

- Light gllvmTMB logLik: named-route **63/63** + shared site-X
  Gaussian/Binomial/Poisson **18/18** + NB2+X/Beta+X **16/16** (#177) +
  **Gamma+X** light cell (per-trait α, observed Laplace; Δ≈3e-8) +
  **Ordinal+X** light cell (per-trait cutpoints + shared γ, `ordinal_probit`;
  Δ≈5e-9) + **NB1+X** light cell (per-trait φ, observed Laplace; abs Δ≈1.53e-9,
  seed=48; #186) + **Poisson species-XB** light cell (`(0+trait):x` /
  `fit_gllvm_speciescov`; Δ≈4e-9) + **BetaBinomial+X** light cell (per-trait φ,
  trials `N`, finite-difference outer Laplace; abs Δ≈1.50e-8, seed=49). Engine
  Arc 1 lands per-trait NB2/Beta/Gamma/NB1/BetaBinomial+X and Ordinal+X. Not
  full family parity; ADEMP and coverage certificates remain fenced.
- R-bridge (`engine = "julia"`) rows that are live are still **partial** vs the
  public R-user surface even when Status = `implemented`.
- Phylo Model A / source-specific `lv` intervals: **rejected** for advertising.
- Prefer reading this beside R `/p/gllvmTMB/surface` — gaps should stay visible
  as `planned` / `missing` / `rejected`, not renamed away.

## Twin join: DIFFER rows (capability-status cross-walk)

`tools/parity_ledger.R` (twin, read-only) joined to this file reports **57
matched** (22 AGREE, 23 R-NARROWER, 3 J-NARROWER, 9 DIFFER), **23 R-only**,
**34 Julia-only**, CLOSURE FAIL (2 undispositioned rows) as of 2026-09-25, run
with the tool's own default `--r-ref origin/main`. **Corrected 2026-09-25:**
the "@ frozen oracle" label previously on this line was wrong. The tool's own
header comment says plainly that `docs/design/capability-status.md` "did not
exist at the frozen 0.7.0 oracle (`b4d5fee64`)" and that pinned-oracle
comparison instead uses `tools/parity_ledger.py` on `NAMESPACE`; this join
always runs against R's `origin/main`, which moves. Re-running this exact
command later will not reproduce these counts (inventory for the prior
2026-09-15 run: `docs/dev-log/after-task/2026-09-15-true-parity-ledger-gap-inventory.md`,
itself against `origin/main` at that date, not a frozen pin).
**DIFFER ≠ covered twin.** Do not promote these rows, paste light Δ, or describe
them as “aligned capability” on public surfaces until both sides share the same
status word *and* the scope fence is written.

| # | Capability row (join key) | Twin (R) status | Julia (this file) | Honesty fence |
|---|---------------------------|-----------------|-------------------|---------------|
| 1 | `spatial × dep` | scope-limited | `planned` (fail-loud only) | Arc 0 stub only (#329); not structured spatial dep parity |
| 2 | `phylo_latent + lv = ~ x` (Phylo Model A intervals) | planned | `rejected` | Intentional Julia refusal; not a lag to “catch up” |
| 3 | `multinomial / categorical` | scope-limited | `missing` | FE softmax engine ≠ twin latent/phylo/spatial multinomial (Design 123) |
| 4 | Simulation-validated coverage certificate | scope-limited | `missing` | Julia arcG/DRAC diagnostics are **not** this row; out of R↔Julia parity |
| 5 | `AGHQ` estimator | scope-limited (opt-in experimental) | `missing` | Julia internal GH for VA only; no public `aghq=` knob |
| 6 | Mixed-family response vector | scope-limited (native programme validated) | `planned` | **Bridge mixed-family point-fit (`implemented` below) is a transport subset only**, does not close this row |
| 7 | `grouping level × unit` | scope-limited | `planned` | Added 2026-09-25, see the new `## Grouping levels` section; `Row effects fixed` above is ACCOUNTED as a coarser-granularity match, this row stays `planned` rather than inheriting that promotion |
| 8 | `Internal IID column coefficients` (`column_coef()`) | scope-limited | `missing` | Reachable only after the 2026-09-25 header fix below; no Julia `column_coef()`-equivalent construct has been located |
| 9 | `Column-slope covariance helpers incl. diagonal phylogenetic column slopes` | scope-limited | `missing` | Reachable only after the 2026-09-25 header fix below; no Julia implementation located |

**Planned / missing rows that must not read as harness parity:** any row still
`planned` or `missing` here (including Arc 0 Gaussian-only cells marked
`implemented` with scope caveats) stays **out** of “R workflow runs identically
through Julia” claims on `docs/src/gllvmtmb-parity.md`. **R-NARROWER (23):**
Julia `implemented` where R registers `scope-limited` is a **promotion fence**,
not evidence that Julia exceeds the twin, wait for receipts at R’s narrower
claim before advertising.

## Covariance structure grid (sources × modes)

R grammar: source ∈ {none, phylogenetic, animal, spatial, kernel} × mode ∈
{indep, dep, latent} (+ `common = TRUE` / `unique = TRUE` modifiers). Julia
exposes twin capabilities under native fitters; engine notes in parentheses are
implementation detail only.

| Capability | Status |
|---|---|
| none × indep (`indep()` / ordinary independent RE) | implemented |
| none × dep (`dep()` / unstructured trait covariance) | implemented (function API) |
| none × latent (`latent()` / ordinary LV GLLVM) | implemented |
| phylogenetic × indep (`phylo_indep()`) | implemented |
| phylogenetic × dep (`phylo_dep()`) | implemented (Arc 0 Gaussian function API only) |
| phylogenetic × latent (`phylo_latent()`) | implemented |
| animal × indep (`animal_indep()`) | implemented |
| animal × dep (`animal_dep()`) | implemented (Arc 0 Gaussian function API only) |
| animal × latent (`animal_latent()`) | implemented (Arc 0 Gaussian function API only) |
| spatial × indep (`spatial_indep()`) | implemented |
| spatial × dep (`spatial_dep()`) | planned (Arc 0 fail-loud entry only) |
| spatial × latent (`spatial_latent()`) | implemented |
| kernel × indep (`kernel_indep()`) | implemented (Arc 0 Gaussian function API only) |
| kernel × dep (`kernel_dep()`) | implemented (Arc 0 Gaussian function API only) |
| kernel × latent (`kernel_latent()`) | implemented (Arc 0 Gaussian function API only) |
| phylo_latent + `lv = ~ x` (Phylo Model A public intervals) | rejected |

Notes (not status rows): Julia phylo rows share three **equivalent** likelihood
representations (sparse CHOLMOD, contrasts, edge-incidence). Gaussian animal/spatial
today enter via `relatedness_cov` / `spatial_cov` (and SPDE latent for
non-Gaussian `spatial_latent`). `none × indep` maps to random row effects /
per-trait diagonal paths; full unstructured `dep()` without LV is still a gap.

**`none × dep` promotion (2026-08-28, maintainer decision gate 6).** Promoted
from `planned` on the evidence: `fit_dep_gllvm` is implemented
(`src/none_dep.jl`), exported (`src/GLLVModels.jl:226`), included
(`src/GLLVModels.jl:76`), and tested (`test/test_none_dep.jl`, 29 assertions, in
`runtests.jl`). It forces `K = p` (full-rank packed Λ, `p(p+1)/2` free
parameters, Σ = LLᵀ) — the same estimand as the twin's
`latent(0 + trait | g, d = T)`.

**Scope caveat, MEASURED not asserted:** the row is `implemented` for the
**function API only**. There is no `@formula` `dep()` sugar (v1 rejects
`FunctionTerm` / `(… | g)`), the wrapper is **Gaussian-only** (non-Normal
families fail loud), and no phylo/animal/spatial/kernel `dep` variant exists.
Critically, **at `K = p` the Σ / σ_eps split is not separately identified**:
the likelihood pins only the total `Σ_total = ΛΛᵀ + σ²I`. Verified 2026-08-28
(seed 4747, p=4, n=200): the fit reports `σ_eps = 0.9875`, yet the
alternative parameterisation `(Λ = chol(Σ_total).L, σ_eps = 0)` reproduces
`Σ_total` to `4.4e-16`. Read `σ_eps` from this path as one point on a flat
ridge, never as an estimated residual variance.

**Honest-0.7 Arc 0 grid promotion (2026-09-14, DestB G2 / Rose fence).** Seven
covariance cells landed on `main` (#324–#334). This pass updates the matrix only;
it is **not** Destination B `FINAL-REVIEW`, not twin Δ parity, and not formula /
`@formula` / `gllvm()` / bridge admission.

| Cell | PR (main) | Fitter / test | Status after G2 |
|---|---|---|---|
| phylo × dep | #324 | `fit_phylo_dep_gllvm`, `test/test_phylo_dep.jl` | Arc 0 Gaussian function API only |
| animal × dep | #325 | `fit_animal_dep_gllvm`, `test/test_animal_dep.jl` | same |
| animal × latent | #327 | `fit_animal_latent_gllvm`, `test/test_animal_latent.jl` | same; `unique = true` still refused |
| spatial × dep | #329 | `fit_spatial_dep_gllvm`, `test/test_spatial_dep.jl` | **fail-loud only** — row stays `planned` |
| kernel × indep | #331 | `fit_kernel_indep_gllvm`, `test/test_kernel_indep.jl` | Arc 0 Gaussian function API only |
| kernel × dep | #333 | `fit_kernel_dep_gllvm`, `test/test_kernel_dep.jl` | same; `rho ≠ 1` refused |
| kernel × latent | #334 | `fit_kernel_latent_gllvm`, `test/test_kernel_latent.jl` | same |

**Shared scope caveats (all Arc 0 Gaussian rows above):** no `@formula` keyword sugar;
non-Gaussian families fail loud; no R-bridge / light logLik receipt; no realistic-size
(RSZ) or second-order (2SO) receipt; sparse-phy / mesh transport gaps unchanged.
**Do not** read these rows as gllvmTMB `covered` or as honest-0.7 programme complete.

## Response families

Twin family names align with gllvmTMB / gllvm. Status = native Julia engine
(with tests). Bridge partiality is under **R bridge**, not hidden by renaming.

| Capability | Status |
|---|---|
| gaussian | implemented |
| poisson | implemented |
| nbinom2 | implemented |
| nbinom1 | implemented |
| binomial | implemented |
| betabinomial | implemented |
| beta | implemented |
| Gamma | implemented |
| tweedie | implemented |
| ordinal_probit | implemented |
| ordinal_logit | implemented (default link; see note) |
| student | implemented (**parity Δ PAID 2026-08-28**) |
| lognormal | implemented |
| truncated_poisson | implemented |
| truncated_nbinom2 | implemented |
| censored_poisson | implemented |
| multinomial / categorical | missing |
<!-- Row stays `missing` deliberately: engine + parity cell is NOT a surface admit.
     **FE-only light RCall Δ PAID 2026-08-24** — live Δ abs ≈2.27e-12 @ rtol 1e-6
     (seed=57, ncat=4, n=400; gllvmTMB 0.7.0 / R 4.6.0;
     `test/parity/test_multinomial_parity.jl`). Exact concave softmax, no Laplace on
     either side, hence the picoscale Δ. Both engines additionally pinned to the
     closed-form intercept-only MLE `Σ n_c log(n_c/n)`. The 2026-08-18 Identity
     fenced a Δ as FORBIDDEN "until an engine exists" and sanctioned exactly this
     cell once it did (#257). Claim is limited to FE-only softmax logLik parity: it
     does NOT promote this row, does NOT cover the twin's latent/phylo/spatial
     multinomial surface (Design 123 — structurally absent in Julia), and is NOT
     itself a surface admit. ≠ full family parity.

     WORDING CORRECTED 2026-08-25. This clause previously read "does NOT admit
     multinomial to `fit_gllvm`/bridge dispatch", which reads as a statement of
     fact and is half wrong:
       * `fit_gllvm`: a live dispatch arm DOES exist — `fit_gllvm.jl:283-284`
         (`_fit_gllvm(::Multinomial, …) = fit_multinomial_gllvm(…)`), with a
         guarded branch at `:148`, and `test/test_multinomial.jl:136-138` asserts
         `fit_gllvm(Y; family = GLLVM.Multinomial())` returns a `MultinomialFit`
         agreeing with the named fitter to atol 1e-8.
       * bridge: correct — `grep -c multinomial src/bridge.jl` → 0.
     Bundling the two surfaces into one phrase is what made it misread. The row
     status is UNCHANGED (`missing`) and is separately defensible: it tracks the
     twin's latent/phylo/spatial multinomial surface, which is genuinely absent. -->
| delta_gamma | implemented (**parity Δ PAID 2026-08-28**) |
| delta_lognormal | implemented (**parity Δ PAID 2026-08-28**) |
| hurdle_poisson / hurdle_nbinom2 | implemented |
| zip / zinb / zib | implemented |
| ordered_beta / beta_hurdle | implemented |
| exponential (Gamma shape=1 path) | implemented |
| com_poisson | implemented |

Notes (not status rows): `zip` / `zinb` / `zib` are Julia-forward (ZIP+X via
`fit_zip_gllvm_cov`; ZINB+X via `fit_zinb_gllvm_cov`, shared scalar `r`).
**Corrected 2026-09-25** (the "twin cut ZIP/ZINB" line here was stale): gllvmTMB
no longer cuts ZIP/ZINB/ZIB. `zi_poisson()`, `zi_nbinom2()`, and `zi_binomial()`
merged to the twin's `main` as FAM-21/22/23 (PR #1240, Arc D1, maintainer-approved
D-207); R's own ledger lists this row `scope-limited`, not missing. The two sides
still diverge in the NB2 dispersion parameterisation: gllvmTMB reuses the
ordinary per-trait `nbinom2()` dispersion (one `log_phi_nbinom2` per trait),
while GLLVModels.jl's ZINB uses one SHARED SCALAR NB2 dispersion `r` across
every trait (`ZINBCovFit` docstring), so a light Δ between the two
`implemented`-shaped statuses needs a parameterisation decision first, not just
a run. Status cells stay bare MC tokens. `zib` also has a native Julia ZIB+X
fitter (`fit_zib_gllvm_cov`) with dual shared slopes (`γz`, `γc`), `Λ_z = 0`,
and one shared scalar `N::Int`. Since #218 / #220 the **no-X** ZIB is reachable
through `fit_gllvm(Y; family = ZIB(N))` and `@formula(y ~ 1)`, and since the
bridge no-X arc through `bridge_fit(; family = "zib", N = …)` with
Wald/profile/bootstrap CI and one **required shared scalar** trials count `N`
(a uniform `p×n` `N` collapses; a non-uniform one errors, ZIB is deliberately
**out** of `_BRIDGE_TRIALS_FAMILIES`, so `cbind_binomial` stays false). ZIB+X
on any public surface, bridge missing-response masks, `confint` under X, and a
measured gllvmTMB light Δ all remain OWED: the twin's `zi_binomial()` now
exists (same PR #1240, FAM-23), so a Δ is no longer categorically forbidden,
but none has been run yet.
`student` / `com_poisson` promoted on native engine + package
**Student-t parity, PARTIAL (2026-08-28).** The live logLik Δ against
gllvmTMB 0.7.1 is PAID — but only in the **fixed-ν** configuration, and the
scope fence matters:

- **What is paid:** with ν pinned on BOTH sides (`gllvmTMB::student(df = 4)`
  vs Julia `nu = 4.0`) and Julia set to the twin's per-trait scale
  (`disp_group = :species`), Δ logLik = −9.66e-10 (builder, seed 71) and
  **3.34e-9, rel 3.6e-12 on an independent re-measurement at fresh seed 9203**
  — both far inside the 1e-6 light-cell gate. Julia's per-trait σ̂ matches the
  twin's to 4–5 significant figures. Joint estimated-ν parity is also PAID
  (Δ logLik ≈ 6.8e-10, rel 7.3e-13 on seed 72). Test: `test/parity/test_studentt_parity.jl`.
- Related: the twin's df profile CI is off by one (reports df−1 as df; see
  `docs/dev-log/decisions/2026-08-28-studentt-parameterisation.md`), so any
  future ν-interval comparison must account for that before attributing a
  mismatch to GLLVModels.jl.
- Known limitation of `disp_group = :species` on this family: the postfit
  helpers in `link_residual.jl` / `simulate_fit.jl` assume a scalar σ and now
  raise a clean `MethodError` under per-trait σ — a fail-fast boundary, not
  silent misbehaviour, and not extended in this slice.

tests (`test_studentt.jl`, `test_com_poisson.jl`). Since the 2026-08-16 Student-t
surface admit the **no-X** `student` surface is reachable through `fit_gllvm(Y; family = StudentTFamily(ν))`
and `@formula(y ~ 1)`; the FIXED `ν` travels on the marker (a separate `nu`
keyword is rejected) and the marker's `σ` is an inert tag payload. Student-t +X,
`disp_group`, row effects, `bridge.jl` admission, and any gllvmTMB parity claim
remain OWED — the twin's `student` route is not benchmarked here, so **no**
light Δ is invented. Since the 2026-08-16 COM-Poisson no-X surface admit,
`com_poisson` is reachable through `fit_gllvm(Y; family = COMPoisson())` and
`@formula(y ~ 1)`; the marker's `ν` is an inert tag payload (always estimated —
the opposite of Student-t's structural `ν`). COM-Poisson +X, `disp_group`,
row effects, `bridge.jl`, and any twin light Δ remain OWED — the twin has no
CMP family, so a Δ would be invented. Since the 2026-08-16 Delta no-X surface admit,
`delta_lognormal` / `delta_gamma` are reachable through
`fit_gllvm(Y; family = DeltaLogNormal())` / `DeltaGamma()` and `@formula(y ~ 1)`;
marker `σ` / `α` are inert tag payloads (always estimated). Delta +X,
`disp_group`, row effects, `bridge.jl`, and any twin light Δ remain OWED — no
invented twin Δ. Since the 2026-08-16 Hurdle-Poisson no-X surface admit,
`hurdle_poisson` is reachable through `fit_gllvm(Y; family = HurdlePoisson())`
and `@formula(y ~ 1)`; the marker is empty (no payload). Since the 2026-08-17
Hurdle-NB no-X surface admit, `hurdle_nbinom2` is reachable through
`fit_gllvm(Y; family = HurdleNB())` and `@formula(y ~ 1)`; the marker's `r` is
an inert tag payload (always estimated; no `r_init`). Since the 2026-08-17 Beta-hurdle no-X surface admit, `beta_hurdle` is reachable
through `fit_gllvm(Y; family = BetaHurdle())` and `@formula(y ~ 1)`; the
marker's `φ` is an inert tag payload (always estimated; no `φ_init`).
Since the 2026-08-17 Ordered-beta no-X surface admit, `ordered_beta` is
reachable through `fit_gllvm(Y; family = OrderedBeta())` and `@formula(y ~ 1)`;
the marker's `c0`, `c1`, and `φ` are inert tag payloads (always estimated;
not used as inits; not Ordinal's `τ₁ = 0` pin).
Hurdle-Poisson / Hurdle-NB / Beta-hurdle / Ordered-beta +X, `disp_group`,
row effects, `bridge.jl`, and any twin light Δ remain OWED — the twin has no
hurdle / ordered-beta family, so a Δ would be invented. `truncated_poisson` =
zero-truncated Poisson (Identity 2026-08-15; twin fid 10; engine+admit; **bridge no-X paid** via `bridge_fit(; family = "truncated_poisson")`; **light RCall no-X Δ PAID 2026-08-24** — live Δ abs ≈2.71e-9 @ rtol 1e-6 (seed=53, p=5, K=2, n=60; Laplace both sides; gllvmTMB 0.7.0 / R 4.6.0; `test/parity/test_truncated_poisson_parity.jl`; ≠ full family parity)); `truncated_nbinom2`
= zero-truncated NB2 (Identity 2026-08-15; twin fid 11; **light RCall no-X Δ PAID 2026-08-24** — live Δ abs ≈1.58e-6 @ rtol 1e-6, relative ≈1.15e-9 (seed=58, p=5, K=1, n=120, per-trait `r`; gllvmTMB 0.7.0 / R 4.6.0; `test/parity/test_truncated_nbinom2_parity.jl`; pairs with `fit_truncated_nbinom2_gllvm_pertrait`, NEVER the shared-scalar route; requires `hessian=:observed`, the default since 2026-08-24 — the Fisher weight gives relative 1.06e-5 and FAILS; ≠ full family parity); Arc1 shared scalar `r`
≡ twin `φ`; Arc1b 2026-08-18 per-trait pack ≡ twin `log_phi_truncnb2`;
≠ bridge admit ≠ AGHQ). `lognormal` = one-part lognormal (Identity 2026-08-15; twin fid 3; engine+admit; **bridge no-X paid** via `bridge_fit(; family = "lognormal")`; **light RCall no-X Δ PAID 2026-08-24** — live Δ abs ≈2.24e-8 @ rtol 1e-6 (seed=52, p=5, K=2, n=60; exact-vs-exact; shared scalar σ; loglik includes `−Σ log y`, verified structurally **and** by a scale-shift test on `2·Y`; gllvmTMB 0.7.0 / R 4.6.0; `test/parity/test_lognormal_parity.jl`; ≠ full family parity)). `censored_poisson` = right-censored Poisson (Identity 2026-08-15; Julia-forward). **Corrected 2026-09-25**: the twin is no longer constructor-only, `censored_poisson()` got a real engine behind its exported constructor (gllvmTMB PR #1254, id 21, `scope-limited` as FAM-25 on R's own ledger), so a light RCall Δ is no longer categorically forbidden; none has been run yet, so it stays OWED, not PAID.

## Intervals and estimation evidence

| Capability | Status |
|---|---|
| Point extraction (coef / loadings / Σ_y / correlations) | implemented |
| Wald intervals | implemented |
| Profile-likelihood intervals | implemented |
| Parametric bootstrap intervals | implemented |
| Simulation-validated coverage certificate (broad grid) | missing |
| Julia-only arcG / DRAC Wald coverage diagnostics | partial |
| Light gllvmTMB logLik named routes (63/63) | implemented |
| Shared-X light logLik Gauss/Bin/Pois (18/18) | implemented |
| ML default (Gaussian closed-form / non-Gaussian Laplace) | implemented |
| REML (Gaussian pilot twin) | implemented |
| AGHQ estimator | missing |
| VA / ELBO alternative (selected families; not R-default) | implemented |

Notes (not status rows): the arcG / DRAC Wald coverage rows (e.g. 0.932–0.958
across 20 diagnostic cells) record **undercoverage evidence** on Julia's dense
path — not a production coverage certificate and not R's withdrawn 0.7.0
total-variance “0.94 floor” claim (0.7.1: three Wald cells only; gap sheet
§Class-2). Gaussian REML is promoted on `src/reml.jl`
(`gaussian_reml_loglik`, `fit_gaussian_reml`) + the bridge `reml=true` route +
`test/test_reml.jl` (dense-oracle criterion at rtol 1e-8, FD gradient ≤ 1e-6,
span-of-`X` invariance, recovery, bridge route). Twin admits a Gaussian-only
REML pilot and non-Gaussian REML stays `rejected`; the `fit_gaussian_gllvm(reml
= true)` profile engine and phylogenetic REML are **not** on `main` (feature
branch), so this row is the standalone + bridge path only — no twin light Δ, no
coverage certificate.

AGHQ (this row and "Broad AGHQ (Julia)" below) stays `missing`. Julia has no
`aghq` symbol under `src/` or `test/` (probed 2026-08-17 at `origin/main`
`51ffa320`). The VA path's `_gauss_hermite` is physicists' GH for VA `E_q`,
not AGHQ — do not rename it. Twin `gllvmTMB` @ `e3e813f4` ships opt-in
experimental AGHQ via `gllvmTMBcontrol(aghq = FALSE | k | "auto")`
(`R/aghq-control.R`, `aghq-gate.R`, `aghq-auto-ridge.R`, `aghq-report.R`);
default remains Laplace; eligibility is a single ordinary loadings-only
`latent()` unit-tier block; the twin itself makes no capability claim for
quadrature-fitted models. Identity:
`docs/dev-log/decisions/2026-08-17-aghq-identity.md`. No engine in that
slice. No twin light Δ.

## Random slopes and special capabilities

| Capability | Status |
|---|---|
| Fixed-effect covariates `X` (shared site design) | implemented |
| Species-specific environmental coefficients | implemented |
| Fourth-corner / trait–environment | implemented |
| Row effects fixed | implemented |
| Row effects random | implemented |
| Per-species / grouped dispersion (`disp.group`) | implemented |
| Keyworded random slopes (≥1) | planned |
| Uncorrelated slope (R double-bar / uncorrelated RE) | planned |
| Missing responses (NA / mask) | implemented |
| Missing predictor `mi()` | implemented |
| Latent scores on covariates `latent(..., lv = ~ x)` ordinary | implemented |
| Concurrent / constrained / RRR ordination (`num.lv.c` / `num.RR`) | implemented |
| Quadratic response | implemented |
| Mixed-family response vector | planned |
| `@formula` / long+wide data (fixed effects) | implemented |

**Missing predictor `mi()` (T13 receipt, DestB G4 2026-09-14).** Row was already
`implemented` on `origin/main`; this pass **pins test evidence** after
true-parity map T13 drift audit (exports existed; receipt was implicit). Exports:
`fit_gaussian_mi_fiml`, `fit_gaussian_mi_phylo`, `fit_gllvm_mi`, `fit_gllvm_mi_multi`
(`src/GLLVModels.jl`). Focused run (single Julia process, files included in order):

```sh
~/.juliaup/bin/julia --project=. -e 'using Test; include("test/test_mi_fitter.jl"); …'
```

| Test file | Pass / Total |
|---|---|
| `test/test_mi_fitter.jl` | 5 / 5 |
| `test/test_missing_predictor_fiml.jl` | 9 / 9 |
| `test/test_missing_predictor_phylo.jl` | 9 / 9 |
| `test/test_missing_predictor_z.jl` | 6 / 6 |
| `test/test_missing_predictor_poisson.jl` | 6 / 6 |
| `test/test_missing_predictor_dispersion.jl` | 15 / 15 |
| `test/test_missing_predictor_multi.jl` | 7 / 7 |
| **Sum** | **57 / 57** |

**Fence:** native Julia mi() axis only; no R-bridge light Δ pasted here; not
Destination B `FINAL-REVIEW`; not full gllvmTMB 0.7 parity. `@formula` / public
R `mi()` keyword parity is a separate surface (change-control / twin lane).

**NB2 grouped-cov Wald at dispersion boundary (T14, DestB G5 2026-09-14).**
Maintainer-approved fix set landed **2026-09-02** on `main`
(`docs/dev-log/check-log.md` §T14; diagnosis
`docs/dev-log/core070/t14-nb2-wald-nan-diagnosis.md`): **F3** bridge CI helper
(`_bx_ci_max_absdiff`, agreed `Inf` not `NaN` poison); **F2** separate
well-conditioned NB2 grouped-cov Wald cell + explicit seed-523 degenerate cell;
**F1** `dispersion_boundary` on grouped NB/NB1/Beta/Gamma fits + per-parameter
Wald degradation in `_family_wald` (`src/confint_family.jl`). Focused re-run
this slice:

| Test file | Pass / Total |
|---|---|
| `test/test_grouped_dispersion.jl` (incl. F1 boundary flag) | 20 / 20 |
| `test/test_bridge_x.jl` (incl. F2/F3 NB2 Wald cells) | 192 / 192 |

**Open sub-item (not blocking T14 closure):** no single seed gives the *old*
3×70 default-shape fixture well-conditioned on **both** Julia 1.10 and 1.12
(check-log 2026-09-02 F2 note); F2 uses a **different** well-conditioned DGP
(`n = 200`, `nb_r = 2`, intercept 1.5). ≠ DestB FINAL-REVIEW; ≠ second-order
parity certificate for all NB2+X cells.

**Knife-edge fixture audit (T15, DestB G6 2026-09-14).** List-first inventory:
`docs/dev-log/after-task/2026-09-14-destb-g6-t15-knife-edge.md` — **18**
fixtures dispositioned (keep / document / already-retargeted); **0** test edits
this slice. Named degenerate NB2 seed-523 cells stay **by design** post-T14.

## Model selection

**Added 2026-09-25.** R's ledger carries a row for `select_lv()` / `anova()`
under this exact name; GLLVModels.jl's ledger never had a row here at all, so
the two files could not join on this capability.

| Capability | Status |
|---|---|
| select_lv() rank selection + anova() boundary likelihood-ratio test | implemented (partial; R pairing not established, see note) |

Notes (not a status row): `select_lv` (`src/model_selection.jl:57`, tested in
`test/test_model_selection.jl` and `test/test_known_sentinel_defects.jl`) is
Julia's latent-dimension rank-selection routine; `chibar2_pvalue` /
`variance_lrt` (`src/boundary_inference.jl`, exported at `src/GLLVModels.jl:249`)
are Julia's boundary-aware likelihood-ratio test. R's alias for this row lists
exactly these three names (`select_lv / chibar2_pvalue / variance_lrt`), so the
row is named here under R's own primary Capability text to join correctly.
`implemented` reflects code + test under this ledger's own definition only, not
cross-package parity: no light RCall Δ has been run against gllvmTMB's
`select_lv()` / `anova()`. Flagged `partial - R pairing not established`
because whether Julia's rank-selection criterion and boundary LRT are the same
estimand as R's (same test statistic, same boundary correction) has not been
checked function by function; treat the join as a name match only until that
comparison is done.

## Grouping levels

**Added 2026-09-25.** R's ledger has a `## Grouping levels` section (`unit`,
`unit_obs`, `cluster`, `cluster2`: the formula-grouping-level keywords for
random-effect structure); GLLVModels.jl's ledger never carried rows under
these names, so this tool's own `--check-names` "MISSING grouping-level rows"
check could never find them here to report.

| Capability | Status |
|---|---|
| grouping level × unit | planned (partial; R pairing not established, see note) |
| grouping level × unit_obs | planned (partial; R pairing not established, see note) |
| grouping level × cluster | planned (partial; R pairing not established, see note) |
| grouping level × cluster2 | planned (partial; R pairing not established, see note) |

Notes (not a status row): `planned` here is a placeholder, not a verified
capability comparison. GLLVModels.jl has no documentation that maps its own
random-intercept machinery onto this specific four-way R vocabulary. The
`Row effects fixed` / `Row effects random` rows above are already ACCOUNTED in
`tools/parity_ledger.R` as covering `grouping level × unit` /
`grouping level × unit_obs` at a coarser granularity; whether that accounting
also extends to `cluster` / `cluster2` is exactly the pairing these rows leave
open. Flagged `partial - R pairing not established` for the same reason as the
`select_lv()` / `anova()` row above.

## R bridge (`engine = "julia"`)

Same twin surface, transport layer. Status = code + bridge/parity test exist;
**every live bridge family remains partial vs full R-user parity.**

| Capability | Status |
|---|---|
| Bridge capability ledger + drift probe (**probe is RED — see fence below**) | implemented |
| Bridge no-X point fit (core one-part families) | implemented |
| Bridge fixed-effect X (selected families) | implemented |
| Bridge missing-response mask (selected families) | implemented |
| Bridge CI transport Wald/profile/bootstrap (selected) | implemented |
| Bridge predictor-informed `lv` / `X_lv` (selected) | implemented |
| Bridge mixed-family vector | implemented (bridge point-fit only — **DIFFER #6**; full row stays `planned` above) |
| Bridge full family × full structure parity | rejected |
| Bridge phylo / animal / spatial / kernel source parity | planned |
| Bridge Phylo Model A / source-specific `lv` advertising | rejected |

## Bridge drift probe: `implemented` is misleading for this row (2026-08-26)

**This section exists because the row above is a compound of two halves with different
truth values.** The capability-ledger half is genuinely implemented and tested
(`bridge_capabilities()` at `src/bridge.jl:632`, exported `src/GLLVModels.jl:251`, test wired
at `test/runtests.jl:206`). **The drift probe is RED**, and nothing in either repo's CI
reports it.

Measured against the twin at `origin/main`, re-derived from both constant sets:

| surface | R mirror | engine | engine-only |
|---|---|---|---|
| one-part families | 11 (`R/julia-bridge.R:18`) | 17 (`src/bridge.jl:164`) | `zip`, `zinb`, `zib`, `lognormal`, `betabinomial`, `truncated_poisson` |
| fixed-effect `X` | 6 (`:76`) | 12 (`src/bridge.jl:635`) | `zip`, `zinb`, `betabinomial`, `nb1`, `ordinal`, `ordinal_probit` |
| CI under `X` | inherits the stale 6 (`:106`) | all 12 (`_BRIDGE_NO_CI_X_FAMILIES` is empty) | same six |

There is no drift in the other direction: the R mirror never claims a family the engine
lacks.

**Count the X row from `bridge_capabilities()`, not from the constant.**
`_BRIDGE_X_FAMILIES` (`src/bridge.jl:198`) holds **11** and is documented as *"One-part
**NON-Gaussian** families"* — it excludes `gaussian` by design. The engine's actual
advertised surface is built at `src/bridge.jl:635` as
`Set(vcat(["gaussian"], _BRIDGE_X_FAMILIES))` = **12**, and the R mirror's list *includes*
`gaussian`. Citing the constant against the R list compares a gaussian-exclusive count with
a gaussian-inclusive one. The six-family delta is unaffected — `gaussian` is on both sides —
but the totals are 6 vs 12. (Corrected 2026-08-26 after a Rose audit; the first version of
this fence made exactly the error the fence exists to prevent.)

**Why nothing catches it.** `.gllvm_julia_expected_capability_drifts()`
(`R/julia-bridge.R:432`) returns a literal 0-row frame whose comment states as fact that
the two surfaces agree. The only engine-facing assertion
(`tests/testthat/test-julia-bridge.R:2848`) sits behind `skip_if_no_julia()`, so it does
not run in ordinary CI — and would fail if it did. The check that *does* pass compares the
R mirror against itself.

**User-visible consequence.** An R user on `engine = "julia"` cannot reach six families the
Julia engine ships — including the zero-inflated trio and lognormal, which are precisely
the families where GLLVModels.jl is *ahead* of the twin. The bridge that would expose that lead
does not know they exist.

Reported upstream at `gllvmTMB#488`, the issue that predicted this bug class and asked for
the audit. The status word is left `implemented` because the ledger half is real; this
fence is what makes the row honest, in the same style as the Laplace-curvature section
below.

## Laplace curvature: which families match TMB's log-det (2026-08-25)

**This section exists because `implemented` alone is misleading for these rows.**
A row can be fully implemented and tested and still not reproduce `gllvmTMB` to
the precision a reader would assume, for a reason orthogonal to capability.

`gllvmTMB` is built on TMB, whose `MakeADFun(..., random = ...)` differentiates
the coded joint negative log-likelihood — so its Laplace log-determinant uses the
**observed** joint Hessian, structurally, without ever choosing. GLLVModels.jl
hand-codes its Laplace kernels, and several used the **Fisher (expected)**
information in that role. The two coincide at canonical links (Poisson/log,
Binomial/logit) where the curvature is free of `y`, which is why the launch
families were unaffected and the discrepancy went unnoticed for so long.

**Status by family.** "observed" = matches TMB's log-det; "Fisher" = does not,
so a log-likelihood from that family will *not* match `gllvmTMB` to machine
precision even where the row reads `implemented`.

| family / link | log-det curvature | note |
|---|---|---|
| Poisson / log | observed ≡ Fisher | canonical; y-free, unaffected |
| Binomial / logit | observed ≡ Fisher | canonical; y-free, unaffected |
| TruncatedPoisson / log | observed ≡ Fisher | verified by expansion |
| CensoredPoisson / log | observed | hand-derived, already correct |
| Ordinal | observed | correct by construction |
| Gamma / log | **observed** | flipped 2026-08-25 — was the public default path |
| Exponential / log | **observed** | fixed |
| NB1 (grouped route) | **observed** | fixed |
| TruncatedNegBin2 | **observed** | fixed; both entry points agree since 2026-08-25 |
| DeltaGamma | **observed** | fixed |
| **NB2 (shared route)** | **observed** | flipped 2026-08-27 (PR #269) |
| **Beta** | **observed** | flipped 2026-08-28, decision A (PR #270) |
| **NB1 (generic core)** | **observed** | flipped 2026-08-28, decision A |
| **Student-t** | **observed** | flipped 2026-08-28, decision A |
| **Tweedie** | **observed** | flipped 2026-08-28, maintainer gate 1 |
| **Binomial / probit** | **observed** | flipped 2026-08-28, maintainer gate 2 |
| **GP-1** | Fisher | **retained BY DECISION** — evidence against, see below |
| **Binomial / cloglog** | Fisher | intrinsic Laplace saturation pathology, see below |

CORRECTED 2026-08-28: this table had drifted roughly four flips behind the
engine (it still described Beta, NB2, NB1 and Student-t as "not yet decided"
after decision A had already flipped all four). The census structural guard
`test/test_curvature_census.jl` is the machine-checked source of truth; this
table is prose and must be re-read against it whenever a default moves.

`Binomial` is worth calling out: it is clean at **logit** (canonical, the two
weights coincide), a flipped instance at **probit**, and a documented
exception at **cloglog**. These are properties of the *(family, link)* pair,
so any census organised by family alone will miss them.

**Census state:** `KNOWN_OPEN` is EMPTY as of 2026-08-28 — every one-part
family's curvature is adjudicated and declared. Two families are deliberate
exceptions rather than open items: GP-1 sits in `DEFERRED_BY_DECISION` with
its evidence recorded, and Binomial/cloglog's runaway is an intrinsic Laplace
saturation pathology (link FD-verified correct; diagnostic guard shipped in
PR #272), not a weight bug. **Still open: the TWO-PART families** — only
DeltaGamma has a specialised observed count-part weight, so the selector is
currently inert for the other nine (`TWOPART_KNOWN_OPEN`).

**This is a PARITY goal, not an accuracy improvement — and that distinction is
load-bearing.** Measured against high-resolution numerical quadrature over 12
seeds per family:

- **Gamma**: observed is closer **12/12**, with 20–60× smaller error.
- **Beta**: observed is closer only **2/12** — Fisher is usually nearer.
- **GP-1**: observed is measurably **worse** for dispersion recovery
  (α = 0.879 against a truth of 0.4, where Fisher lands inside the test's
  tolerance) — this is why GP-1 was retained on Fisher by decision.

So each family is decided on its own evidence, not on principle. Matching TMB is
the objective; "the numbers get better" would be an overstatement, and for two of
the three families measured it is simply false.

**How the remaining flips were decided (2026-08-28).** The 900-cell
adjudication campaign scored each cell on two metrics against an exact
quadrature oracle: objective error (|Laplace − exact|, which feeds AIC/BIC
honesty) and estimator preference (does the exact marginal prefer the
observed fit's θ̂ or the Fisher fit's?). Where the two metrics disagreed —
Beta, NB1, Student-t — the maintainer chose **estimator quality over
reported-loglik accuracy** (decision A), accepting a measurably more biased
reported loglik for estimates nearer the exact optimum, for TMB parity.
Tweedie was the strongest flip case in the entire table (observed preferred
in 98–100% of cells in every regime). Probit was decided on parity grounds
with a supporting derivation: its observed curvature is provably
non-negative (affine in `y`, endpoints positive under BigFloat; Pratt 1981
proves the probit binomial log-likelihood globally concave), so the flip
carries no indefiniteness risk.

**Where a curvature was corrected, the previous behaviour stays reachable** via
`hessian = :fisher` on the corresponding marginal.

**Not closed, but the AGHQ instance of this fault class is fixed (2026-08-28,
AGHQ unpark Slice 0/1).** A role-separation contract (Fisher-scored mode
search, selectable log-det) now covers 12 kernels; `src/families/aghq_grid.jl`
was the 13th and, until this fix, carried an unconditional Fisher weight at
`aghq_stage1a_loglik_site` for BOTH roles — silently diverging from the same
family's own default Laplace fitter for every family whose `_default_hessian`
is `:observed` (Beta, Gamma, NegativeBinomial, NB1, StudentT, Exponential,
TruncatedNegBin2, TweedieED, Binomial-probit). It now takes a
`hessian::Symbol = _default_hessian(family, link)` keyword mirroring
`laplace_loglik_site`/`covariates.jl` exactly: the Newton mode search stays
Fisher-scored; only the adaptation curvature (log-det AND the per-site
Cholesky reused across every quadrature node) is selectable. `hessian =
:fisher` pinned reproduces the pre-fix value bit-for-bit (verified against an
independent copy of the pre-change unconditional-Fisher formula, all 9
affected families plus Poisson at k=1 and k=3); the family-default k=1
template now equals that family's own default dense Laplace marginal to
1e-10, for all 9 affected families (`test/test_aghq_grid.jl`). **This closes
only the AGHQ instance.** The module remains internal — no `aghq=` public
surface, no capability-status ledger row changes from `missing`/`missing`
(§AGHQ above). The fault class generally (any future kernel that builds its
own `Λ'WΛ + I`) is still not guaranteed closed by this fix.

## gllvmTMB 0.7.1 delta (tracked 2026-08-27)

The twin moved while the Julia campaign ran: `gllvmTMB` origin/main is at
**0.7.1 (release candidate)** — 126 commits past the 0.7.0 snapshot this ledger
was written against. **The parity milestone stays pinned at 0.7.0** (do not
re-baseline mid-campaign); this section makes the new twin surface visible as
tracked debt instead of invisible. Sources: twin `NEWS.md` at origin/main plus
the merged feature PRs (#1192, #1196, #1216, #1217).

New twin capability with **no Julia ledger vocabulary until now**:

| Capability | Status |
|---|---|
| Response-column slope family (`slope()`, `phylo_slope()`, `animal_slope()`, `kernel_slope()`, `spatial_slope()`; Gaussian long-format, predictor-only) | missing |
| Internal IID column coefficients (#1216) | missing |
| Per-source iSDM observation formulas (#1192) | missing |
| Column-slope covariance helpers incl. diagonal phylogenetic column slopes (#1196) | missing |

**Header fixed 2026-09-25** (was `| Capability (twin 0.7.1 vocabulary) | Status |`,
which `tools/parity_ledger.R`'s parser does not recognise as a Capability|Status
table, so all four rows above were silently dropped from every join run before
this fix). Running the join against this corrected header surfaces two of the
four rows as UNDISPOSITIONED (the other two, `Internal IID column coefficients`
and `Column-slope covariance helpers ...`, already join cleanly to R's aliased
rows of the same name). Proposed dispositions for the two that do not join,
**PROPOSED, NOT SIGNED** (a disposition in `tools/parity_ledger.R`'s own
`PORT`/`ACCOUNTED`/`DIVERGENCE` tables requires editing that file, which lives
in gllvmTMB and is out of scope here; the maintainer signs off on the actual
tool edit):

- `Response-column slope family (...; Gaussian long-format, predictor-only)`:
  **PROPOSED PORT.** R's own row of the same name (minus this row's trailing
  scope clause) is `scope-limited` (register FG-19, FG-15, PHY-06, ANI-06,
  SPA-11, KER-04) and carries an Aliases cell that is this exact Julia string,
  so R's own ledger already intends this pairing
  (`dev/gapclose/build-capability-status.R`'s provenance note: "Where the
  R-canonical name differs from GLLVM.jl's spelling of the same concept, the
  Aliases column carries GLLVM.jl's exact string so `tools/parity_ledger.R`
  still joins them"). The join still fails here because that Aliases cell has
  a semicolon INSIDE its parenthetical, and `split_aliases()` splits on every
  semicolon with no parenthesis-awareness, producing two malformed halves that
  match neither this Julia string nor anything sensible. That parser bug lives
  in `tools/parity_ledger.R`, in gllvmTMB, so it is not fixed here.
- `Per-source iSDM observation formulas (#1192)`: **PROPOSED ACCOUNTED.**
  `#1192` is gllvmTMB's own filed issue ("per-source observation models; hand-masked
  bias columns fail silently; 34-48h to build", from gllvmTMB's 2026-08-20
  handover) rather than a built R capability, so there is no register row to
  join to on either side; this is not a Julia-ahead capability owed as a port,
  and not a confirmed shared register row either. Track it via #1192 rather
  than double-counting it here.

Deltas to rows that already exist elsewhere in this ledger (no duplicate rows —
the existing row keeps its status; the twin side moved):

- **Mixed-family response vector** (`planned` above): the twin's named
  mixed-family LV programme is now validated and closed on its main (#1217) —
  the R side of this row strengthened from partial to native-validated.
- **Predictor-informed latent scores** (`implemented` above, partial scope):
  0.7.1 ships an evaluated guide; the twin's interval evidence remains limited
  to named native Gaussian and rank-1 multi-trial binomial cells, so the
  Julia-side scope fence is unchanged.

Not capability (API hygiene in 0.7.1, nothing to mirror): unused grouping-slot
warnings (#1190), `extract_Sigma_*` soft-deprecation (#1194), VA remains opt-in
experimental (#1189).

## Withdrawn and deferred (twin fences)

| Capability | Status |
|---|---|
| Full family R↔Julia parity claim | rejected |
| Phylo Model A public interval promotion | rejected |
| Delta/hurdle latent-scale correlation advertising | rejected |
| Non-Gaussian REML | rejected |
| Broad AGHQ (Julia) | missing |

## Evidence pointers

- R surface (compare side-by-side): `/p/gllvmTMB/surface` ←
  `gllvmTMB/docs/dev-log/capability-surface.html`
- Light logLik 63/63: `docs/dev-log/handover/2026-08-01-cursor-handover.md`
- Shared-X 18/18: `docs/dev-log/after-task/2026-08-02-x-covariate-light-loglik.md`
- NB1+X engine (bridge/`@formula`/`fit_nb1_gllvm_grouped_cov`):
  `docs/dev-log/after-task/2026-08-05-nb1-x-engine-arc12.md` — light RCall
  cell live Δ abs ≈1.53e-9 @ rtol 1e-6 (seed=48; ≠ full family parity)
- NB1 no-X surface admit (`fit_gllvm` + `@formula` with no X; Identity
  `docs/dev-log/decisions/2026-08-16-nb1-betabinom-fit-gllvm-identity.md`):
  `docs/dev-log/after-task/2026-08-16-nb1-nox-surface.md` — exported `NB1` marker,
  per-trait φ via the API-B coerce. Julia-side surface + identity only; **no** new
  twin Δ (bridge behaviour unchanged, `src/bridge.jl` not opened)
- **NB1 no-X light RCall Δ PAID 2026-08-24** — live Δ abs ≈1.34e-8 @ rtol 1e-6
  (seed=55, p=5, K=1, n=120, per-trait φ; gllvmTMB 0.7.0 / R 4.6.0;
  `test/parity/test_nox_dispersion_parity.jl`; ≠ full family parity). **This cell
  first FAILED at Δ ≈ −0.115 and found a real engine defect**, since fixed:
  `fit_nb1_gllvm_grouped` declared no `hessian` keyword, so it silently inherited the
  `:fisher` default of `nb1_grouped_marginal_loglik_laplace` — the
  expected-information Laplace, a *different objective* from TMB's — while its NB2,
  Beta and `_cov` siblings all default to `:observed`. The optimiser was never
  failing; it was converging correctly to the wrong objective. Fix aligns the default;
  `hessian=:fisher` stays reachable explicitly. See `docs/dev-log/check-log.md`
  2026-08-24
- BetaBinomial+X engine (bridge/`@formula`/`fit_beta_binomial_gllvm_grouped_cov`):
  `docs/dev-log/after-task/2026-08-05-betabinomial-x-engine-arc12.md` — light
  RCall cell live Δ abs ≈1.50e-8 @ rtol 1e-6 (seed=49; ≠ full family parity)
- **Gamma + BetaBinomial no-X light RCall Δ PAID 2026-08-24** —
  `test/parity/test_nox_dispersion_parity.jl`, per-trait dispersion via the grouped
  fitters: Gamma (twin fid 4) live Δ abs ≈2.05e-8 @ rtol 1e-6 (seed=54, p=5, K=1,
  n=120); BetaBinomial (twin fid 8, N=8, trials via twin API-B `weights`) live Δ abs
  ≈6.15e-9 @ rtol 1e-6 (seed=56, p=5, K=1, n=120). gllvmTMB 0.7.0 / R 4.6.0.
  These are the **no-X** arms of families that previously had twin evidence only
  under +X; ≠ full family parity
- BetaBinom no-X surface admit (`fit_gllvm` + `@formula` with no X; same Identity
  as the NB1 row above): `docs/dev-log/after-task/2026-08-16-betabinom-nox-surface.md`
  — exported `BetaBinom` marker, per-trait φ via the API-B coerce, p×n trials `N`
  **required** at the entry point (φ unidentifiable at `N = 1`). Julia-side surface
  + identity only; **no** new twin Δ (bridge behaviour unchanged, `src/bridge.jl`
  not opened)
- ZIP+X engine (bridge/`@formula`/`fit_zip_gllvm_cov`; Identity 2026-08-09):
  `docs/dev-log/after-task/2026-08-09-zip-x-engine.md` — Julia identity/FD only;
  **no** twin light Δ (gllvmTMB ZIP cut)
- ZINB+X engine (bridge/`@formula`/`fit_zinb_gllvm_cov`; Identity 2026-08-13):
  `docs/dev-log/after-task/2026-08-14-zinb-x-engine.md` — Julia identity/FD only;
  shared scalar `r`; **no** twin light Δ (gllvmTMB ZINB cut)
- ZINB+X confint under X (`confint(ZINBCovFit)`; FD Hessian; `ci_x_*` true):
  Julia CI claim only ≠ twin Δ ≠ ADEMP
- truncated_poisson Identity + engine + **bridge no-X** (zero-truncated; twin fid 10):
  `docs/dev-log/decisions/2026-08-15-truncated-poisson-identity.md` ·
  `src/families/truncated_poisson.jl` · `src/bridge.jl` (`_BRIDGE_ONEPART_FAMILIES`
  after `zib` / `_bridge_family_key` / `_bridge_fit_onepart`) ·
  `test/test_truncated_poisson.jl` · `test/test_bridge_truncated_poisson.jl` —
  Julia identity/FD + bridge admit; light RCall Δ still OWED (not invented);
  CI / X / X_lv / masks remain follow-ups; y ≥ 1 fail-loud
- truncated_nbinom2 Identity + engine (zero-truncated NB2; twin fid 11; shared-`r` Arc1 + per-trait Arc1b):
  `docs/dev-log/decisions/2026-08-15-truncated-nbinom2-identity.md` ·
  `docs/dev-log/plans/2026-08-15-truncated-nbinom2-identity-engine.md` ·
  `docs/dev-log/after-task/2026-08-18-truncated-nbinom2-arc1b.md` ·
  `src/families/truncated_nbinom2.jl` · `test/test_truncated_nbinom2.jl`
  — Arc1b pack `[β; pack(Λ); log r_1…log r_p]` ≡ twin `log_phi_truncnb2`;
  ≠ bridge admit ≠ AGHQ
- lognormal Identity + engine + admit + **bridge no-X** (one-part lognormal; twin fid 3):
  `docs/dev-log/decisions/2026-08-15-lognormal-identity.md` ·
  `src/families/lognormal.jl` · `src/bridge.jl` (`_BRIDGE_ONEPART_FAMILIES` /
  `_bridge_family_key` / `_bridge_fit_onepart`) · `test/test_lognormal.jl` ·
  `test/test_bridge_lognormal.jl` ·
  `docs/dev-log/handover/2026-08-15-lognormal-ADMIT.md` — Julia identity/FD +
  bridge admit; light RCall Δ still OWED (not invented); CI / X / X_lv / masks
  remain follow-ups
- censored_poisson Identity + engine + admit (right-censored Poisson):
  `docs/dev-log/decisions/2026-08-15-censored-poisson-identity.md` ·
  `src/families/censored_poisson.jl` · `test/test_censored_poisson.jl` ·
  `docs/dev-log/after-task/2026-08-15-censored-poisson-engine.md` — Julia-forward.
  **Corrected 2026-09-25**: gllvmTMB's `censored_poisson()` is no longer
  constructor-only (PR #1254 built the engine behind the exported constructor,
  id 21; R's own ledger now lists it `scope-limited`, FAM-25), so a light
  RCall Δ is no longer forbidden by construction; it is simply not measured
  yet and stays OWED
- Hurdle-Poisson no-X surface admit (`fit_gllvm` + `@formula` fall-through):
  `src/families/fit_gllvm.jl` · `test/test_hurdle_poisson.jl` — empty marker
  `HurdlePoisson()`; +X / bridge remain OWED; twin light Δ
  **forbidden** (no twin hurdle family)
- Hurdle-NB no-X surface admit (`fit_gllvm` + `@formula` fall-through):
  `src/families/fit_gllvm.jl` · `test/test_hurdle_nb.jl` — tag-payload marker
  `HurdleNB()` (`r` never read); +X / bridge remain OWED; twin light Δ
  **forbidden** (no twin hurdle family)
- Beta-hurdle no-X surface admit (`fit_gllvm` + `@formula` fall-through):
  `src/families/fit_gllvm.jl` · `test/test_beta_hurdle.jl` — tag-payload marker
  `BetaHurdle()` (`φ` never read); +X / bridge remain OWED; twin light Δ
  **forbidden** (no twin beta-hurdle family)
- Ordered-beta no-X surface admit (`fit_gllvm` + `@formula` fall-through):
  `src/families/fit_gllvm.jl` · `test/test_ordered_beta.jl` — three-field
  tag-payload marker `OrderedBeta()` (`c0`, `c1`, `φ` never read); +X /
  bridge remain OWED; twin light Δ **forbidden** (no twin ordered-beta
  family; `"ordered"` on the bridge already means ordinal)
- ZIB no-X surface admit (`fit_gllvm` #218, `@formula` #220):
  `src/families/fit_gllvm.jl` · `src/formula.jl` · `test/test_zero_inflated.jl` /
  `test/test_formula.jl` — X on those surfaces remains OWED
- ZIB no-X **bridge** admit (Identity
  `docs/dev-log/decisions/2026-08-16-zib-bridge-identity.md`):
  `src/bridge.jl` · `test/test_bridge_zib.jl` — `"zib"` is a one-part bridge
  family with Wald/profile/bootstrap no-X CI and a required shared scalar `N`;
  ZIB+X on the bridge, `_family_ci(::ZIBCovFit)`, and masks remain OWED; a twin
  light RCall Δ is **forbidden**, not owed (no twin ZIB)
- student / com_poisson ledger promote: `test/test_studentt.jl`,
  `test/test_com_poisson.jl` (code already present)
- REML promote: `src/reml.jl` + bridge `reml=true` + `test/test_reml.jl`
  (dense-oracle criterion rtol 1e-8; FD gradient ≤ 1e-6; span-of-`X` invariance;
  β/σ_eps recovery; bridge `gaussian_reml_rr` route vs the standalone fitter).
  Gaussian-only; `fit_gaussian_gllvm(reml = true)` and phylo REML are not on
  `main`
- AGHQ Identity (estimator absent; twin opt-in experimental; VA GH ≠ AGHQ):
  `docs/dev-log/decisions/2026-08-17-aghq-identity.md` — both AGHQ ledger
  rows stay `missing`; no engine, no stub knob, no twin Δ
- Public catch-up prose: `docs/src/gllvmtmb-parity.md` (Documenter legend ≠ this
  MC vocabulary)
