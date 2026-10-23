# Julia speed arc, session 2026-09-19: the OWED steps of the Fable plan

```
🎯 GOAL
Solo platform: Claude (this session; read from the runtime, not inferred)
Deliverable: (1) HSquared per-section AI-REML profiler + the q=5k fill-471 pre-run number;
             (2) HSquared selected-inverse bench, three arms, Mac rungs only, SelectedInversion.jl in
                 bench/Project.toml only; it enters the package, if at all, as an OPTIONAL EXTENSION behind
                 our own fallback, and only after the bench clears Fable's pre-registered gate (below);
             (3) DRModels q=4 phylo profile at p=1000/5000, then the three hygiene fixes (closed-form
                 logdet P, cholesky! reuse, warm u0) tests-first on a branch;
             (4) GLLVModels Poisson Laplace re-profile, then the four rung-1 changes tests-first on a branch;
             (5) GLLVModels sparse-phylo/EM hoists after (4), same lane.
HEADLINE: the measurements. Nothing in src/ moves until a section table says what owns the time.
IN PARALLEL: HSquared profiler · DRM profile · GLLVM profile (three repos, three worktrees, three leases).
DEFER (fenced): HSquared src/ (steps 4, 5, 9: Szymek's push lands first); the Totoro fill-471 arm
  (pre-run shown, then ask); the ASReml comparison (Szymek runs it); the message to Szymek (Shinichi sends);
  the ForwardDiff compat lift (separate bitwise-gated PR); merging anything; SelectedInversion.jl as a HARD
  dependency of any package (the optional-extension route is step 5, after Szymek, only if the bench gate
  clears); a home-grown supernodal port (dropped unless depend and vendor both fail against a measured win).
DISCIPLINE: verify = every src change gated bitwise or at a stated rtol against origin/main before any speed
  claim; compute = Mac Studio (4 cores per lane, OPENBLAS_NUM_THREADS=1, JULIA_NUM_THREADS=4), each step's
  compute under 30 min (D-139); closure = the acceptance ledger passes, Rose audit, handover written.
```

## Context

The 2026-09-18 session measured that HSquared's "slow REML" was two post-fit dense inverses (fixed in #355,
CI green on `8e9c3ef0`) and that the real REML wall at high fill is our own Takahashi selected inverse
(381 s of a 1,529 s fit at q=20k, fill 471, vs 0.35 s to factorise). A Fable panel wrote a ten-step,
measure-first plan (`memory/Julia speed plan 2026-09-18 (Fable panel)...md`). The handover
(`docs/dev-log/handover/2026-09-19-claude-handover-julia-speed-arc.md`) names five Next Immediate Steps.

### Rehydration, reconciled with git on 2026-09-19 (this session)

| handover step | status | evidence |
|---|---|---|
| 1 Lane preflight, three engines; confirm #355 CI | **DONE** | `gh pr checks 355`: all 4 shards + docs pass on `8e9c3ef0`; preflight run in vault, HSquared worktree, DRM.jl, GLLVM.jl; no live leases anywhere |
| 1b Szymek's push | **PROTECTED** | newest HSquared remote ref is still `origin/main` 09-16 (Szymek); he told Shinichi "maybe today" |
| 2 HSquared step 1 (profiler, sim-only) | **OWED** | no `sim/profile_ai_reml_sections.jl` on any ref (scout confirms) |
| 3 HSquared step 2 (selinv bench, Mac arm) | **OWED**, decision rule changed (below) | no `bench/` on the branch (scout confirms) |
| 4 DRM step 6, GLLVM steps 7-8 | **OWED** | renames merged (`origin/main` DRModels `90fbb0e28`, GLLVModels `69a69b0a0`); local folders still `DRM.jl` / `GLLVM.jl`; DRM local `main` 2 behind; GLLVM checkout on a CI branch |
| 5 Do-not-do list | **PROTECTED** | carried into DEFER |

### What Shinichi told us (2026-09-19, mid-session)

1. **SelectedInversion.jl: inspire, do not depend.** His words: "should we get inspired, rather than using
   these packages??" So step 2 stays a measurement, but its decision rule flips: the package is an oracle in
   `bench/` only; if arm (c) beats the dependency-free hoisted copy (b) by 10x or more at fill 471, step 5
   becomes "port supernodal selected inversion into `takahashi_selinv.jl` under MIT attribution", sized by
   the measured gap. Never in `Project.toml`.
2. **Szymek: asked; "some stuff to do, maybe today".** HSquared `src/` stays his until his branch is on origin.
3. **ASReml: Szymek runs it.** We can estimate from his 09-18 data; not a slice in this plan.
4. **Names.** GitHub says DRModels.jl / GLLVModels.jl; local folders and the brain say DRM.jl / GLLVM.jl.
   Use the old names for paths and brain queries, the new names only for URLs. No folder renames here.
5. **"/ultra-plan with /unlazy first?"** One plan: ultra-plan Phase 2.5 is the unlazy ledger. Gates are
   written below and land in `.unlazy/` before any slice runs.

### Sweep receipt (Phase 0.25)

- **repo git state** → `git fetch --all` in HSquared/DRM/GLLVM; `git worktree list`; `gh pr list` ×3;
  `lane_preflight.sh` ×4 → HSquared: our branch + no Szymek push; DRM: `main` 2 behind origin, Codex
  worktrees `/private/tmp/drmodels-handover-fix-20260918` and `~/local-scratch/lanes/drmjl-d269-reader`
  (scout reports their files); GLLVM: checkout on `ci/documenter-gllvmodels-slug-20260918`, 4 draft PRs
  waiting on pastes (#399/#409/#410/#411), old cursor PRs #363/#314 → **resume nothing; build the gap**.
- **twin / sister repos** → hsquared (R) #235 open, independent; drmTMB #1387 filed; gllvmTMB untouched
  → **n/a for these slices**.
- **brain** (`search_notes`, `search_all_projects: true`, query "Julia speed AI-REML selected inverse
  Takahashi profile per-section timing SelectedInversion cholesky! reuse") → June 2026 notes name the
  per-iteration cost as unmeasured (`native-engine-recon-r2`, `native-engine-arc-design`); no harness,
  no decision on the dependency → **build**.
- **log / decisions / journal** → `grep -in "selectedinversion\|profile_ai_reml\|julia speed\|takahashi"
  memory/AGENT_LOG.md` (09-19 entry only), `memory/DECISIONS.md` (D-269, D-270; no D on this),
  `journal/` (09-18, 09-19), `projects/deep-research/README.md` (nothing) → **nothing to resume**.
- **Verdict** → genuinely new: the profiler, the bench, the DRM/GLLVM profiles and the src hygiene fixes.
  Reuse: DRM `bench/profile_sparse_grad.jl`, `bench/profile_step1.jl`, `bench/head_to_head_q4_scaling.jl`;
  GLLVM `bench/speed_bench.jl`, `bench/sparse_phy_bench.jl`, `bench/em_phylo_bench.jl`,
  `docs/dev-log/core070/poisson-perf-diagnosis.md`; HSquared `sim/drac/f0_adversarial_fill.jl` and the
  dense-inv pins at `test/runtests.jl:3043-3061` (scouts confirm each path).

### Route check (Phase 0.6)
Destination in one sentence: three section tables exist with reproduced baselines, and every src change
that follows is an identity gated against origin/main. Every slice names an output path. No slice waits on
an undecided design (the one open design question, the dependency, is now answered). **Route knowable.**

## Preflight (Phase 0.2, Shannon)

- Vault: `no foreign lane ... no 2nd claude lane in 12h`; 11 foreign uncommitted files, never staged.
- HSquared worktree: `FOREIGN LANE ACTIVE (codex)` = stale PRs #322/#321 (09-07..09-11), not live;
  lane taken: `sim/`, `bench/` on a new branch off `claude/selinv-defaults-350`. `src/` untouched.
- DRM.jl: no live lease; lane taken: `bench/`, the step-6 src files, new test files. Not `src/DRModels.jl`,
  not `NEWS.md` (Codex D-269 lease paths), unless the scout shows that work landed.
- GLLVM.jl: `FOREIGN LANE ACTIVE (cursor)` = PRs #363/#314 (09-07), not live; lane taken: `bench/`,
  `src/laplace_grad.jl`, `src/families/laplace.jl`, `src/families/poisson.jl`, new tests.
- Leases claimed at execution start (writes a file, so not in plan mode):
  `lane_lease.sh --claim <repo> --paths <as above>`; refused → narrow or serialise, never bypass.

## Model routing (Phase 0 / 2)

Session model: Opus 5 (this window). Fable switch is the app's model picker (Shinichi's call; planning
here ran on Opus, recorded per D-151). Usage bars: not readable from this session.

| slice | member | model · effort | dispatch | wall | files | dep |
|---|---|---|---|---|---|---|
| S0 RECON ×3 | scouts | Haiku (HSquared) · Sonnet ×2 (DRM, GLLVM) · low | claude/model-param | 10 min | scratchpad `recon-*.md` | none (running) |
| S1 HSquared step 1 profiler | curie-style builder | Sonnet · high | claude/model-param, `isolation:"worktree"` off (own scratch worktree) | 60-90 min | `sim/profile_ai_reml_sections.jl`, `sim/results/ai_reml_sections_<sha>.tsv` | S0 |
| S2 HSquared step 2 bench, Mac arms | builder | Sonnet · high | same worktree, after S1 | 60-90 min | `bench/Project.toml`, `bench/selinv_arms.jl`, `bench/results/selinv_arms_<sha>.tsv` | S1 (fixture gen + pre-run ratio) |
| S3 DRM step 6 profile | builder | Sonnet · medium | new worktree `~/local-scratch/drm-speed6-20260919`, branch `claude/speed-step6-q4-profile` off origin/main | 45-60 min | `bench/profile_q4_sections.jl` (extends the two existing), `bench/results/q4_sections_<sha>.tsv` | S0 |
| S4 GLLVM step 7 profile (per-site p=20/50 AND grouped 200×5) | builder | Sonnet · medium | new worktree `~/local-scratch/gllvm-speed7-20260919`, branch `claude/speed-step7-laplace-grad` off origin/main | 45-60 min | `bench/profile_laplace_allocs.jl`, `bench/profile_grouped_glmm.jl`, `bench/results/laplace_allocs_<sha>.tsv`, `bench/results/grouped_glmm_<sha>.tsv` | S0 |
| S5 DRM step 6 changes (a)(b)(c) | builder | Sonnet · high | S3 worktree | 4-8 h | src files listed in step 6; `test/test_q4_perf_identities.jl` | S3 table |
| S6 GLLVM step 7, re-scoped: shared mode solve in `fg!`, one `GradientConfig` per fit + chunk sweep, allocation-free `_poisson_site_diffable`; wire `test_poisson_grad_perf.jl` | builder | Sonnet · high | S4 worktree | 4-8 h | `src/laplace_grad.jl`, `src/families/laplace.jl`, `src/families/poisson.jl`, `test/runtests.jl` (+1 include); `test/test_laplace_grad_identity.jl` | S4 table |
| S7 GLLVM step 8 (only what the p=200/1000/5000 wall names) and S7b grouped-route `cholesky!` reuse / analytic gradient if S4 says the 12× lives there | builder | Sonnet · high | S4 worktree, after S6 | 4-8 h | files listed in step 8; `src/families/grouped_laplace.jl` for S7b | S4, S6 |
| S8 MECHANICAL-VERIFY | scout | Haiku · low | claude/model-param | 10 min | `gate-check --reverify` on every leaf; artefacts non-empty | each of S1-S7 |
| S9 VERIFY (judgment) | gauss + Rose | Opus 5 · high (the one ceiling child) | claude/model-param | 30 min | refute one passed gate per slice; audit before any PR | S8 |
| S10 RECONCILE | melissa | Sonnet · low | claude/model-param | 10 min | `docs/dev-log/plan-actual/2026-09-19-julia-speed-arc.md` | S9 |

FAN-OUT BUDGET: checkpoint=G0 · new children 6/6 (S1, S3, S4 in batch 1; S2 reuses S1's agent; S5, S6
fresh in batch 2; S7 reuses S6; S8 Haiku; S9 the one ceiling child) · live ≤ 3 at once (each lane runs
Julia at 4 threads on a 16-P-core machine; three lanes = 12 threads, the D-249 ceiling).
SCOUT SUITABILITY: yes, S0 HSquared recon on Haiku (running), S8 mechanical verify on Haiku.
ULTRA EFFORT: no. CONTEXT BRAKE: parent input ~60k · not reached. COMPACTIONS: parent 0.
LANE RECEIPT: CONTINUE HERE through batch 1; batch 2 (src changes, 12-24 h of builder time) will not fit
this window → run as `/arc-loop`. The goal lives on disk in each lane's committed `LOOP/` kit
(`GOAL.md` immutable, `arcs.md`, `checkpoint.md`, `ultra-plan.md` = this file), scaffolded by
`tools/lane_launch.sh <repo> <lane>` (worktree at `~/local-scratch/lanes/<repo>-<lane>` on
`claude/lane-<lane>`; `git push` / `gh pr merge` / `gh release` DENIED inside the lane, so the approved
pushes and draft PRs are the orchestrator's action after S9, never the loop's). Lanes:
`HSquared.jl-speed12-20260919` (base `claude/selinv-defaults-350`, so the post-fit sections are measured
with #355's fix, not the retired 17 s bug; if `lane_launch.sh` takes no base ref, the worktree is added by
hand from that branch and the kit scaffolded after), `DRM.jl-speed6-20260919` and `GLLVM.jl-speed78-20260919`
(base `origin/main`). Handover written at every checkpoint.
D-43 PANEL: fired once per repo when a PR is proposed (2 Sonnet + 1 Opus), not per commit.
ESTIMATE: batch 1 ≈ 1.5-2 h wall (three builders in parallel, compute < 30 min each); batch 2 ≈ 12-24 h
of builder time across two lanes → needs the arc-loop and at least one handover.

### Scout findings folded in (HSquared, measured on `8e9c3ef0`)

- AI-REML loop body: `src/likelihood.jl:457-533` (the Fable plan's `:434-486` have shifted on the branch).
  Sections: `_sparse_mme_system` at `:459`; CHOLMOD `cholesky` at `:461`; solve `factor \ rhs` at `:462`;
  `selinv_trace_against` at `:466` (→ `src/takahashi_selinv.jl:168-197`; full matrix `:114-165`); AI step
  `:482-487`. Post-fit SE re-factorisations: locate at execution (was `:2790-2818`).
- No timing hook (`verbose`/`callback`/`trace` absent). `_fit_ai_reml_diagnostics` (`:395-562`) returns
  `.diagnostics.factorizations`. So S1 does two things: (i) `Profile.@profile` of the real `fit_ai_reml`
  with samples attributed by function (no reimplementation, the honest share); (ii) a faithful copy of the
  loop body calling the same internals in the same order, `@elapsed`/`@allocated` per section, gated
  bitwise against `fit_ai_reml` in the same process (G1.1). If (ii) is not bitwise, (i) is the deliverable
  and the drift is reported, not hidden.
- **The w4 fixture** is NOT `sim/cpu_fit_benchmark.jl`: it is `projects/H2-twin/repros/w4-speed-gen-fit.jl`
  (vault; `_halfsib_pedigree`, `_sizes_for(q)` = 5% sires / 10% dams, deterministic
  `_make_y_animal_indexed(q, reps=2)`, no RNG). It activates the prunable worktree `/private/tmp/h2camp-jl`;
  S1 copies the three functions into the harness and runs `--project=.`. Pinned σ²ₐ from
  `projects/H2-twin/updates/2026-09-13-speed-w4.md:52-55`: q=1k 0.149778 · 5k 0.098201 · 20k 0.132360 ·
  50k 0.103807 (G1.2 pins all four).
- `sim/drac/f0_adversarial_fill.jl` CLI: `[q1,q2,...] [out.tsv] [nfounder_frac]`; reports fill `nnz(L)/n`
  and ainv/chol/fit/pcg/selinv timings. S1 reuses its pedigree builder for the fill ~75/150/471 rungs.
- `bench/` does not exist; `sim/` has no `Project.toml` (scripts use `--project=.`). Package deps:
  LinearAlgebra, Optim, Random, SHA, SparseArrays; SelectedInversion absent (stays absent).
- Tests pin by `isapprox` with committed tolerances; dense-inv pins for selinv exist in `test/runtests.jl`
  (scout could not confirm `:3043-3061`; S2 locates them by `grep -n "takahashi" test/runtests.jl`).
- Repo AGENTS.md: opt-in benchmarks carry "NOT CI / OPT-IN measurement only" headers; S1/S2 do the same.

### Scout findings folded in (DRM.jl, measured on local `main` 27fc92020; origin/main differs only by the rename)

- "q = 4" is the 4×4 log-Cholesky Λ over {mu1, mu2, sigma1, sigma2} in the bivariate double-hierarchical
  phylo route (`src/coevolution_q.jl:39`). Route: `_fit_bivariate_q4_phylo` (`src/gaussian_bivariate.jl:1160-1246`,
  dispatched at `:180-187`) → `fit_q4_sparse_tmb` (ML, `src/fit_q4_sparse_tmb.jl:473`) or `fit_q4_reml`
  (`src/reml_q4.jl:732`). NLL: `marginal_nll` (`fit_q4_sparse_tmb.jl:249-257`), on `laplace_ll`
  (`src/sparse_aug_plsm.jl:350-367`); exact gradient `marginal_and_exact_grad` (`:288-`).
- logdet P: `cholesky(Symmetric(P) + 1e-10I)` at `sparse_aug_plsm.jl:362-363`, one fresh factorisation per
  NLL call; the closed form `logdetP = const − N·logdet Λ` already exists on the gradient side
  (`fit_q4_sparse_tmb.jl:331`). **Zero `cholesky!` on the route**; `sparse_pd_chol` (`:176-203`) factorises
  fresh at every Newton step in both `_estep_fast` (`:254-279`, warm) and `_estep_robust` (`:289-321`, cold,
  `nit = max(n_newton, 200)`, start `u = zeros(nu)`, `nu = 4·n_total`).
- `_q4_fd_vcov` (`src/gaussian_bivariate.jl:1348-1360`; callers `:1114`, `:1278`): central FD of the exact
  gradient, `2·nθ` calls (nθ = k1+k2+ks1+ks2+kr+10; 34 at nθ=17), **`u0` never passed** → every call is a
  cold robust Newton; the warm `u_hat` is computed afterwards at `:1289`. `src/bridge.jl:526-531` already
  defaults `q4_vcov=false` because it is "expensive at large q4 phylogenetic fits". Warm `u0` into
  `_q4_fd_vcov` is the cheapest win on the route.
- Bench: `bench/Project.toml` exists (deps DRM, DelimitedFiles, LinearAlgebra, Printf, Random, SparseArrays,
  Statistics; the `DRM` dep name must match origin/main after the rename); `bench/profile_sparse_grad.jl`,
  `bench/profile_step1.jl`, `bench/head_to_head_q4_scaling.jl` (`DRM_376_PS` env, default 100/1000/5000/10000),
  `bench/run_sparse_tmb_nd.jl`, `bench/run_scaling.jl` (the 2.18× / logLik −256.51 baseline). No `sim/`.
- Tests: `test/test_parity_biv_q4_phylo_reml.jl` pins `loglik = −219.6139863046289` (n=128, atol 0.03,
  `test/parity/q4-reml/biv-q4-phylo-reml/expected.toml`); `test_q4_reml_vcov.jl`, `test_q4_reml_warm_restart.jl`,
  `test_575_q4_optimum.jl`, `test_gaussian_bivariate_phylo.jl` cover the route. `test/test_qgate_alloc_inner.jl`
  not confirmed by the scout; S5 locates it.
- **Repo AGENTS.md overrides the hub here:** `src/` core engine is "verified; do not touch without the owning
  persona (Noether) + maintainer sign-off; never regress the 2.18× / logLik −256.51 baseline" (`AGENTS.md:103-108`);
  TDD, docstrings, check-log entry, after-task, Rose audit (`:115-131`); pre-edit lane check on the
  coordination board (`:110-111`). So S5 develops on a branch (reversible) but its draft PR carries a Noether
  audit and **Shinichi's sign-off is the landing gate** (added to MUST STOP). `bench/` is Curie's; perf gates
  are Karpinski's (`AGENTS.md:35,40`).
- Other lanes: `/private/tmp/drmodels-handover-fix-20260918` (rename only, clean) and
  `~/local-scratch/lanes/drmjl-d269-reader` (CI only, clean) touch none of the route files. Coordination board's
  last active lane is Cursor's 08-28 lss arc, historical. **No overlap.**
- Note for briefs: `graft` inside a subagent binds to the vault, not the repo; builders run it with the repo
  as cwd or fall back to `rg`.

### Scout findings folded in (GLLVM.jl, measured on `aa21455e1`; `src/` identical to origin/main bar the module name)

- **Three of the Fable step-7 changes already exist** (core070 repair, commit `a04c9d26`;
  `docs/dev-log/core070/poisson-perf-repair-notes.md:23-126`): the per-site mode is solved once, concretely,
  before `ForwardDiff.gradient` (`src/laplace_grad.jl:118-140`, R2); `LaplaceModeWorkspace`
  (`src/families/laplace.jl:77-92`, R3) is wired into `poisson_laplace_grad` (`:134`) and the `negll`
  closure (`src/families/poisson.jl:256`); `Optim.only_fg!` is in place (`poisson.jl:302-325`, R4) and
  **measured zero speed-up (9.30 s → 9.30 s)** because value and gradient each run their own per-site
  Newton pass. NB/Gamma/Beta still re-solve the mode inside the differentiated closure
  (`laplace_grad.jl:166-167, :243-244, :323-324`).
- **What is genuinely left on the per-site route:** no `GradientConfig`/`Chunk` anywhere in `src/`
  (13 chunk passes at p=50, nθ=149); `_poisson_site_diffable` (`laplace_grad.jl:62-96`) allocates `A`, `z`,
  `Az` (Dual-typed) per site per chunk pass (`:82, :83, :89`), flagged unrepaired at repair-notes `:84-94`;
  and the structural one: **share the mode solve between value and gradient at the same θ** inside `fg!`
  (one per-site Newton per θ, not two). Step 7 is re-scoped to these three, plus extending the R2 hoist to
  NB/Gamma/Beta only if the profile ranks it.
- **The Latte 12× fixture runs the GROUPED route**, not the per-site gradient:
  `~/local-scratch/latte-prerun-20260918/runs/f3_ours_glmm.jl` calls
  `fit_gllvm(Y1; family=Poisson(), grouping=[GroupingTerm(:unit; mode=:indep)], unit=group)`; warm min
  **0.192 s** (`f3_ours_glmm.log`; the handover's 0.187 is not in any log there) vs Latte 0.015 s.
  `data/glmm_200x5.csv` has no header: col 1 = count, col 2 = group 1..200. The Fable plan lists the grouped
  Laplace only conditionally in step 8 (`grouped_laplace.jl:219, :231, :242-246`: two fresh symbolic
  analyses per inner Newton iteration under a 2·nθ finite-difference outer gradient). So S4 profiles BOTH
  fixtures (per-site p=20/50 and the grouped 200×5), and the 12× target is assigned to whichever slice the
  profile names (a grouped-route `cholesky!` reuse + analytic gradient slice, S7b, if the grouped route
  owns it).
- **Step 8 is disputed by the scout and settled by the profile.** The scout reads `_estep_sparse`
  (`src/em_phylo.jl:322-`) as O(K_B²·p) with no dense p×p (the dense `cholesky`/`inv` sit in `_estep_dense`
  `:254-299`, the fallback); the Fable plan says a dense p×p Cholesky hides at `em_phylo.jl:399-403` inside
  `_estep_sparse` and that the monotonicity check / SQUAREM map (`likelihood.jl:212-244`,
  `em_phylo.jl:764-765, :789-790`, `em_squarem.jl:84-95`) run two p×p Choleskys per iteration. S7's first
  act is the wall-per-EM-iteration at p=200/1000/5000: p³ scaling confirms the plan, p scaling refutes it.
- `bench/Project.toml` exists (BenchmarkTools harness; `Pkg.develop(path=".")`); relevant scripts
  `bench/speed_bench.jl`, `bench/grad_speedup.jl`, `bench/em_phylo_bench.jl`, `bench/em_squarem_bench.jl`,
  `bench/sparse_phy_grad_bench.jl`, `bench/node_gradient_bench.jl`; results in `bench/results/`.
- **Orphan found (Rose principle):** `test/test_poisson_grad_perf.jl` (pins `BASELINE_LOGLIK =
  -14604.017303313138` at p=20, n=500, K=2, atol 1e-8; FD gate 1e-6) exists but is **not included in
  `test/runtests.jl`**. S6 wires it in as the regression gate for its own changes.
- Tests on the route: `test_poisson_laplace.jl`, `test_laplace_grad.jl` (analytic vs FD rtol 1e-4),
  `test_laplace_alloc_equiv.jl`, `test_laplace_dual_safety.jl`; phylo: `test_em_phylo.jl`,
  `test_em_sparse_estep_default.jl`, `test_takahashi_selinv.jl`, `test_node_gradient.jl`,
  `test_sparse_phy.jl`; four phylo tests deliberately un-wired for CI flakiness (`runtests.jl:155-178`).
- Repo AGENTS.md: tests ship with implementation (`:300-313`); never widen a tolerance (`:28-47`);
  Engine Quality Battery item 5 "Allocs.jl pass, zero allocation in the inner loop" then Florence's
  speed-up plot, after-task, Rose sign-off (`:348-365`). Issue #426 wants SelectedInversion benchmarked
  here too (adopt only >2× at 1e-10 agreement, coordinated with HSquared#353): the same oracle-not-dependency
  rule and the S2 harness answer it (step 10).

### Dependency policy: Fable's verdict (2026-09-19, on Shinichi's "use Fable to think about this")

Full text: scratchpad `fable-dependency-policy.md` (to be filed in the vault as a wiki page at execution).
The rule, tiered by the package's failure mode, applied to SelectedInversion.jl (all thresholds
pre-registered, evaluated at q=20k, fill 471, one BLAS thread):

| branch | trigger | ships |
|---|---|---|
| **Keep ours + hoist** | `S_c < 10`, or `err_c > 1e-10` anywhere, or install adds > 2 non-stdlib packages or any `_jll`, or projected whole-fit gain `G_fit < 2` | arm (b) if bitwise and ≥1.1× on any rung; then the one CliqueTrees contingency measurement |
| **Optional extension** (recommended if the README transfers) | `S_c ≥ 10` AND `G_fit ≥ 2` AND `err_c ≤ 1e-10` on every fixture incl. boundary and both `is_super` branches AND install ≤ 2 packages, no `_jll`, precompile ≤ 60 s | `[weakdeps] SelectedInversion` + `ext/HSquaredSelectedInversionExt.jl` (the CUDA/Makie pattern HSquared already uses); backend `:auto | :takahashi | :package`, `:auto` = package only when loaded, `is_super`, and a dense-`inv` n=50 load canary passed (demote with one `@warn`); caret pin `0.2.1`; CI with and without on 1.10 and 1.12; `hsquared` (R) loads it only on canary-passed minors |
| **Hard dependency** | not reachable from this bench: needs the extension shipped, canary green on 1.10/1.12/next minor, package ≥ 1.0 or public-API-only, and Szymek's pedigree measured above the ~150 fill crossover | move to `[deps]`, keep `:takahashi` one release |
| **Vendor** | extension criteria met AND upstream unresponsive > 90 days to a breakage (or archived, or pin broke two minors running) AND the kernel is ≤ ~600 stdlib-only lines | `src/vendor/…` under MIT with upstream hash, same canary |
| **Port** | dropped; reopened only if `S_c ≥ 10` measured AND extension and vendor both impossible AND Shinichi approves the 20-30 h | a new supernodal kernel with the full ledger |

Fable reconciled the two earlier thresholds ("within 3× of the factorisation" ≈ 360× vs "≥ 10×"): the gate is
`S_c ≥ 10` with `G_fit ≥ 2` because the user's criterion is minutes saved per fit; `R_c = T_c/T_fact` is
reported, not gating, and tells step 10 whether the wall is closed. For DRModels/GLLVModels (step 10) the same
rule applies to their own fixtures; tree precisions are simplicial, so the expected outcome there is "leave,
close #771/#426 with the number".

Draft ledger line (written to `memory/DECISIONS.md` at execution, after Shinichi's yes):
`D-271 | A speed package enters a Julia engine only by a number measured on our own matrices, at the tier its
failure mode allows: a bitwise dependency-free change ships at any gain; a pre-1.0, single-maintainer, or
private-struct package enters only as an optional extension behind our own fallback and a load canary after a
≥10× kernel win with rtol 1e-10 agreement and ≤2 added packages; hard dependency only once ≥1.0 or
public-API-only and canary-green across two Julia minors; vendor the MIT kernel only when upstream stops fixing
a break within 90 days; port a large kernel ourselves only after depend and vendor have both failed against that
measured win. | 2026-09-19 | proposed`

**This replaces the provisional "port on ≥10×" reply from earlier today** (that menu did not offer the
extension route). It is Ada's recommendation; Shinichi's yes at plan approval locks it.

### Decisions locked (2026-09-19)
- SelectedInversion.jl: never in `bench/`-free code before the bench; the branch is chosen by the table above
  once S2 (and the Totoro arm) report; the extension route, if triggered, is step 5, after Szymek lands.
- Remote authority: push the three branches; DRAFT PRs on DRModels.jl / GLLVModels.jl after S9; never merge.
- Scope: batch 1 now; batch 2 under `/arc-loop` with the goal on disk; pause at the Totoro arm and each PR.
- Design source: the Fable panel (09-18) did the design; this plan is its execution slicing, so no second
  Plan agent was run.

## What changes, in plain words

1. **HSquared: measure the REML loop, section by section.** A script in `sim/` fits the same pedigrees
   the package already ships (half-sib at q = 1k / 5k / 20k / 50k, and the adversarial pedigree at fill
   ~75 / 150, plus one q=5k fill-471 pre-run) and prints, per iteration, how long assembly, factorisation,
   solves, the selected inverse, and the AI step take. The fit it times must equal a plain `fit_ai_reml`
   bit for bit, and q=50k must give the banked σ²ₐ = 0.103807. Output: a TSV plus the pre-run ratio.
   No `src/` edit.
2. **HSquared: settle the selected inverse by measurement, not by README.** A `bench/` environment
   (its own `Project.toml`, so the package's five dependencies are untouched) runs three arms on the same
   CHOLMOD factor: our kernel; our kernel with the `sparse(ch.L)` copy hoisted once per factor; and
   SelectedInversion.jl as the oracle. All three must agree to rtol 1e-10 with each other and with dense
   `inv` at n = 50 / 500; `factor.is_super` is recorded so a simplicial fallback is never mistaken for a
   package win. Mac rungs (fill ≤ 150, q ≤ 50k) run now; the fill-471 q=20k arm (~1.5 h) goes to Totoro
   only after the pre-run number is shown and Shinichi says yes. **Decision rule (Fable, pre-registered):**
   ten times faster than our hoisted kernel, same numbers to ten decimals, at most two small packages added,
   and at least a 2× whole-fit gain → it ships as an **optional add-on** (the base install stays at five
   packages; our kernel stays as the fallback; a load canary guards it; the R side switches it on only
   where CI has tested it). Below that → we keep our own code and ship only the hoist. Writing our own
   supernodal version (20-30 h) stays off the table unless both the add-on and a vendored copy fail.
3. **DRModels: profile the q=4 phylogenetic route at p = 1000 / 5000**, splitting the inner Newton's
   repeated factorisations, the second factorisation taken only for logdet P, the kron prior rebuilt three
   times per gradient, and `_q4_fd_vcov`'s 34 cold mode searches. Then, tests first: (a) closed-form
   logdet P (an identity, rtol 1e-12 on random Λ); (b) `cholesky!` with symbolic reuse in the inner
   Newton and `build_M` (pattern asserted on first use, fallback count asserted zero); (c) warm `u0` into
   `_q4_fd_vcov` (the one non-bitwise gate: Wald vcov rtol 1e-8). Benchmark: `bench/head_to_head_q4_scaling.jl`
   paired with drmTMB, nrep 4, 1 BLAS thread, at p = 100 / 1000 / 5000.
4. **GLLVModels: re-profile first, on two fixtures.** The per-site Poisson Laplace gradient at
   p = 20 / 50 (`Profile.Allocs`) and the grouped 200 × 5 GLMM that Latte beat 12× (0.192 s ours vs
   0.015 s). Three of the four Fable changes are already in the code (mode hoisted out of the chunk
   passes, workspace, `only_fg!`), and `only_fg!` bought nothing because value and gradient each still run
   their own per-site Newton. So the changes that remain: share one mode solve per θ between value and
   gradient inside `fg!`; one `GradientConfig` per fit (chunk 12 / 24 / 32 measured); an allocation-free
   `_poisson_site_diffable`. Gate: new gradient = old gradient rtol 1e-8 at 50 random θ on p = 5 / 20 / 50;
   f/g call counts must not grow; logLik vs frozen gllvmTMB 0.7.0 stays at 1e-6..1e-9; the orphaned
   `test_poisson_grad_perf.jl` is wired into the suite as the regression pin. If the profile puts the 12×
   on the grouped route, a grouped-route slice (`cholesky!` reuse across its inner Newton, analytic
   instead of finite-difference outer gradient) takes that target.
5. **GLLVModels: the sparse-phylo / EM hoists, only what the wall names.** First the wall per EM iteration
   at p = 200 / 1000 / 5000: p³ scaling confirms the Fable finding (a dense p×p hiding inside the O(p)
   E-step and the monotonicity check), p scaling refutes it. Then the identities the profile ranks: a
   per-fit workspace holding `Q_cond`, one symbolic then `cholesky!` per evaluation, `takahashi_diag` once
   per gradient, the Woodbury capacitance for the p×p if it is there, the monotonicity check through the
   sparse loglik. Gates: rtol 1e-12 on the marginal loglik, rtol 1e-10 on E-step moments and the EM
   trajectory.

Not changing: HSquared `src/` (Szymek), any `Project.toml` of a package, the public API of any engine,
the R twins.

## Acceptance ledger (Phase 2.5, written before dispatch; lands in `.unlazy/julia-speed-20260919/`)

`.unlazy/` is git-ignored in each worktree (`.git/info/exclude`) before the first run.

```
leaf-S1 OWNS: sim/profile_ai_reml_sections.jl sim/results/ai_reml_sections_*.tsv   (HSquared worktree)
- [ ] G1.1 harness fit == plain fit_ai_reml, bitwise (sigma_a2, sigma_e2, iterations, loglik, factorizations)
  CHECK: julia --project=. sim/profile_ai_reml_sections.jl --gate bitwise --q 5000
  EXPECT: "BITWISE_EQUAL true"
- [ ] G1.2 half-sib q=50k reproduces sigma_a2 = 0.103807
  CHECK: julia --project=. sim/profile_ai_reml_sections.jl --q 50000 --fixture halfsib
  EXPECT: "sigma_a2 = 0.103807"
- [ ] G1.3 section TSV non-empty, one row per (fixture, q, fill, iteration, section)
  CHECK: awk 'END{print NR}' sim/results/ai_reml_sections_*.tsv
  EXPECT: > 20
- [ ] G1.4 q=5k fill-471 pre-run: selinv:factorisation ratio same order as 381:0.35
  CHECK: grep -E "^f0adv\t5000\t471" sim/results/ai_reml_sections_*.tsv (manual read: ratio)
  EXPECT: ratio >= 100  (else the banked record is not reproduced; say so, do not launch Totoro)

leaf-S2 OWNS: bench/Project.toml bench/selinv_arms.jl bench/results/selinv_arms_*.tsv   (HSquared worktree)
- [ ] G2.1 arms a/b/c trace agree rtol 1e-10 with sum(Ainv .* takahashi_selinv(...)) on F0 fill 75/150 + f0_scale
  CHECK: julia --project=bench bench/selinv_arms.jl --gate
  EXPECT: "GATE trace rtol<=1e-10 PASS" for every fixture and arm
- [ ] G2.2 diag vs dense inv at n=50 and n=500, both is_super branches
  EXPECT: "GATE diag dense PASS n=50 super=true|false" ×4
- [ ] G2.3 package Project.toml unchanged
  CHECK: git diff --stat origin/claude/selinv-defaults-350 -- Project.toml
  EXPECT: empty
- [ ] G2.4 results TSV carries every quantity Fable's rule needs, per (fixture, arm, threads):
      q, nfixed, nnz(C), nnz(L), fill, is_super, index eltype, T_fact (numeric refactorisation with
      symbolic reuse), T_a/T_b/T_c (one selinv pass as HSquared uses it: trace + diag; median of >= 3 after
      warm-up, bytes alongside), S_b, S_c, R_c, err_c, bitwise_b; header with Julia, SuiteSparse_jll,
      SelectedInversion, OS, CPU; Manifest diff of the bench env (packages added, any _jll, precompile s)
  EXPECT: header contains "is_super" and "T_fact" and "err_c"; Manifest diff printed
- [ ] G2.5 applicability preconditions checked and printed: is_super == true at the fill-471 fixture
      (else force supernodal and re-run; if it cannot be forced, verdict "leave"); one extra arm forced
      simplicial at fill 471; a near-boundary sigma_a2 -> 0 fixture included
  EXPECT: "PRECONDITIONS is_super=true forced_simplicial_arm=done boundary_fixture=done"
- [ ] G2.6 (Totoro arm, after Shinichi's yes) T_a at q=20k fill 471 reproduces the banked 381 s within 2x
  EXPECT: 190 s <= T_a <= 762 s, else the baseline is re-stated before the rule is applied

leaf-S3 OWNS: bench/profile_q4_sections.jl bench/results/q4_sections_*.tsv   (DRM worktree)
- [ ] G3.1 section table at p=1000 and 5000, both engines, 3 reps, under 20 min
  CHECK: julia --project=. bench/profile_q4_sections.jl --p 1000,5000
  EXPECT: TSV rows for sections {inner_newton_chol, logdetP_chol, kron_prior, fd_vcov_estep, takahashi}
- [ ] G3.2 the p=2000 banked 52.9 s reproduced within 20%
  EXPECT: "p=2000 wall" within 42..64 s

leaf-S4 OWNS: bench/profile_laplace_allocs.jl bench/profile_grouped_glmm.jl bench/results/laplace_allocs_*.tsv
  bench/results/grouped_glmm_*.tsv   (GLLVM worktree)
- [ ] G4.1 Profile.Allocs split at p=20/50 n=500 K=2 on the CURRENT code: value-path mode solve vs
      gradient-path mode solve vs _poisson_site_diffable Dual allocs vs chunk machinery
  EXPECT: shares summing to ~100%; the value+gradient double mode solve quantified
- [ ] G4.2 chunk-pass count recorded (13 at p=50 with chunk 12)
  EXPECT: "chunk_passes=13"
- [ ] G4.3 grouped 200x5 GLMM (Latte fixture, data/glmm_200x5.csv, GroupingTerm(:unit; mode=:indep)):
      warm wall reproduced and split into outer FD gradient evaluations x inner Newton x symbolic analyses
  EXPECT: warm median within 0.15..0.25 s (banked 0.192); count of fresh symbolic analyses per fit printed
- [ ] G4.4 per-EM-iteration wall at p=200/1000/5000 on the phylo EM (bench/em_phylo_bench.jl extended)
  EXPECT: fitted exponent printed; "p^3" or "p^1" verdict on the step-8 dispute

leaf-S5 OWNS: src/sparse_aug_plsm.jl src/fit_q4_sparse_tmb.jl src/coevolution_q.jl src/locscale_marginal.jl
  src/location_only.jl src/sparse_laplace_glmm.jl src/locscale_inner.jl test/test_q4_perf_identities.jl   (DRM)
- [ ] G5.1 marginal NLL at fixed theta == origin/main rtol 1e-12 on the q4 fixtures
- [ ] G5.2 closed-form logdet P == factorised logdet, random Lambda, rtol 1e-12
- [ ] G5.3 Newton iteration counts and ridge lambda sequence identical
- [ ] G5.4 cholesky! fallback count == 0 in the bench harness
- [ ] G5.5 _q4_fd_vcov Wald vcov == origin/main rtol 1e-8
- [ ] G5.6 test/test_qgate_alloc_inner.jl green; full Pkg.test() green
- [ ] G5.7 head_to_head_q4_scaling p=100/1000/5000 nrep=4: factorisations per evaluation 4 -> 2 on the location-only spine
  CHECK for each: julia --project=. test/test_q4_perf_identities.jl ; Pkg.test()
  EXPECT: "Test Summary: ... Pass" with 0 fail

leaf-S6 OWNS: src/laplace_grad.jl src/families/laplace.jl src/families/poisson.jl test/runtests.jl(+1 include)
  test/test_laplace_grad_identity.jl   (GLLVM)
- [ ] G6.1 new gradient == old gradient rtol 1e-8 at 50 random theta, p=5/20/50 (rtol 1e-14 with the cause
      named if a reduction order changes)
- [ ] G6.2 fitted params and logLik == origin/main rtol 1e-6; test_poisson_grad_perf.jl BASELINE_LOGLIK
      -14604.017303313138 atol 1e-8 holds, and the file is now in runtests.jl
- [ ] G6.3 Optim f_calls/g_calls/iterations not above 31/92/140 at p=5/20/50
- [ ] G6.4 logLik vs frozen gllvmTMB 0.7.0 at 1e-6..1e-9 on the ten grid cells
- [ ] G6.5 per-site Newton solves per Optim iteration == 1 (was 2: value + gradient), asserted by a counter
      in the bench harness; bench/speed_bench.jl gradient/value ratio at p=50 below 24.3x
- [ ] G6.6 test/test_laplace_grad.jl and test_laplace_alloc_equiv.jl unchanged and green; Pkg.test() green

leaf-S7b (conditional on G4.3 naming the grouped route) OWNS: src/families/grouped_laplace.jl
  test/test_grouped_laplace_identity.jl   (GLLVM, after S6)
- [ ] G7b.1 grouped Poisson fit == origin/main rtol 1e-8 (params, logLik) on the 200x5 fixture and the
      existing grouped tests; symbolic analyses per inner Newton iteration 2 -> 0 after the first
- [ ] G7b.2 warm wall on the 200x5 fixture recorded against 0.192 s (ours) and 0.015 s (Latte)

leaf-S7 OWNS: src/likelihood_sparse_phy.jl src/sparse_phy_grad.jl src/node_gradient.jl src/fit_phylo.jl src/em_phylo.jl
  src/em_squarem.jl test/test_sparse_phy_identities.jl   (GLLVM, after S6)
- [ ] G7.1 marginal loglik at fixed params == origin/main rtol 1e-12
- [ ] G7.2 sparse-phylo gradient bitwise after once-per-gradient caching
- [ ] G7.3 E-step moments rtol 1e-10 vs _estep_dense at p=200/1000; EM trajectory rtol 1e-10 for 50 iterations
- [ ] G7.4 monotonicity check flags the same iterations as the dense one; SQUAREM decisions identical
- [ ] G7.5 cholesky! fallback count == 0; em_phylo parity fixtures unchanged; Pkg.test() green
```

## Execution order and gates

**Batch 1 (parallel, 3 live):** S1 (HSquared), S3 (DRM), S4 (GLLVM). Each builder: claim the lease,
create its worktree/branch, write the harness, run the smoke (one small cell, confirm non-empty valid
output) before the full ladder, write the TSV, return the path + the section shares.
**Checkpoint A (no ask needed):** S8 Haiku re-verifies leaf-S1/S3/S4. Then S2 in the HSquared lane.
**PAUSE (Shinichi):** the Totoro fill-471 arm, shown with G1.4's number and the ~1.5 h estimate.
**Batch 2 (parallel, 2 live):** S5 (DRM changes) and S6 (GLLVM changes), tests written first, each
change re-profiled after landing. S7 follows S6 in the same lane.
**Close:** S8 on every leaf, S9 (Rose + gauss on Opus) tries to refute one passed gate per slice and
audits each branch before a PR is proposed; S10 Melissa; after-task report; handover; `AGENT_LOG.md`,
journal, DECISIONS (a D-271 line recording "SelectedInversion.jl: oracle, not dependency; port on measured
≥10x") written to the vault after re-claiming the vault lease on `memory/`.

PRE-AUTHORISED AFTER G0: scoped edits in the three worktrees; routine local Julia runs under 30 min
each at 4 threads / 1 BLAS thread; `Pkg.test()`; local commits on the named branches; `.unlazy/` ledgers;
these verification commands: the CHECK lines above.
OPTIONAL REMOTE AUTHORITY: push the three branches; open DRAFT PRs on DRModels.jl and GLLVModels.jl
after S9 passes; never merge, never push `claude/selinv-defaults-350` or any branch of Szymek's.
MUST STOP: the Totoro arm; any src edit in HSquared; any Project.toml dependency change in a package;
a gate that fails twice on the same cause; evidence the banked baseline is not reproduced (G1.4, G3.2).

## Verification (end-to-end)

- `node ~/shinichi-brain/skills/unlazy/scripts/gate-check.mjs --reverify .unlazy/julia-speed-20260919/gates/leaf-S*.md`
  in each worktree; exit 0 required per leaf.
- Full `Pkg.test()` in each touched package worktree (HSquared ~25 min; run once at close, not per edit).
- The three section tables opened and read by a human-shaped check: shares sum to ~100%, banked numbers
  reproduced (G1.4, G3.2, G4.1).
- `tools/slop_check.py` on the after-task report and the handover before they ship.

## Team raised (plan review, attributed)

- **Gauss:** the closed-form logdet P is an identity only if `Q_cond` is fixed across the evaluation; the
  test must vary Λ, not Q. Recommendation: G5.2 draws random Λ at fixed Q. Default: as written.
- **Rose:** S5/S6/S7 will not fit this window; without the arc-loop and a goal on disk the second half
  vanishes at compaction. Recommendation: write the goal file in the first execution action. Default: yes.
- **Karpinski:** hoisting the mode solve out of chunk passes changes when Duals are seeded; "bitwise for
  chunk changes alone" may not hold if the reduction order changes. Recommendation: G6.1 states rtol
  1e-14 as the fallback with the cause named, as the Fable plan already says. Default: as written.
- **Shannon:** three Julia lanes at 4 threads each is the Mac's 12-thread ceiling; do not add a fourth
  live builder. Recommendation: cap live at 3. Default: yes.
- **Fable (dependency judge):** neither "depend" nor "port" is the right frame; the `[weakdeps]`
  extension HSquared already uses for CUDA/Makie gives users the speed at zero base-install cost, keeps our
  kernel as the fallback, and needs no 20-30 h port. Recommendation: pre-register the gate, run the bench,
  let the number pick the tier. Question for Shinichi: yes to the tiered rule as D-271? Default: yes.
- **Ada:** run batch 1 now; the measurements are the deliverable this session cannot lose. Endorses Fable's
  rule: it satisfies "fewer dependencies" (base install unchanged), "make things dependent if a package is
  good" (opt-in, measured), and D-139 (usability does not bend: canary + fallback + one `@warn`).

## Questions still open (answered at approval or by the numbers)
1. D-271 as drafted: yes / edit? (Default: yes.)
2. The Totoro fill-471 arm: asked again with G1.4's number in hand, ~1.5 h estimate.
3. Whether the S5 (DRM `src/`) draft PR may land needs Noether's audit and Shinichi's sign-off per the repo's
   own AGENTS.md; the PR is opened, not merged.
