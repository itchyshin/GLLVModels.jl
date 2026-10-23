# Lane auto-d-20260926 — estimate the number of latent dimensions from data (GLLVModels.jl)

```
🎯 GOAL
Solo platform: Claude Code (this session; lane_preflight PLATFORM: claude)
Lane taken: claude/lane-auto-d-20260926 (worktree ~/local-scratch/lanes/GLLVM.jl-auto-d-20260926, 2 kit commits, 0 behind main)
Deliverable: a CLEARED DECISION MAP (route, default criterion, API shape, caveat wording) grounded in a cited
  literature pass; then, after Shinichi's sign-off, a draft PR where omitting K gives a fit whose K was chosen
  from the data by that rule, with measured recovery rates and an honest post-selection caveat.
HEADLINE: Ranga's grounded literature pass (/notebook) on how GLLVMs / factor models choose the number of
  latent variables, distilled into one decision memo. Everything else waits on it.
IN PARALLEL: a no-src-change recovery pilot of the criteria we already have (select_lv AIC/BIC, cv_gllvm,
  chibar2 sequential LRT, PPCA eigen-gap) on known-K simulated data.
DEFER: shrinkage estimator build (ordered factor LASSO) unless the map picks it; grouped/phylo/row_eff/pervar
  routes; any gllvmTMB change (Cursor lane); porting the 25 post-0.7.0 exports.
DISCIPLINE: verify = unlazy gate ledger re-run (--reverify) + D-43 panel at the build milestone ·
  compute = laptop pre-run test (≤4 threads) then Totoro (≤150 cores) with a stated estimate (D-287) ·
  closure = map approved AND draft PR open with recovery table AND caveat in the docstring + docs page.
```

## Context

Today a user must pick the number of latent dimensions. In Julia, `fit_gllvm(Y)` with no `K` throws
(`K::Integer` has no default in every family fitter, e.g. `src/families/negbin.jl:176`; the variant routes
throw "K (or num_lv) is required", `src/families/fit_gllvm.jl:267-290`). In R, `latent(d = 1)` silently
defaults to one dimension (gllvmTMB `R/brms-sugar.R:607`). Shinichi wants the package to estimate it, and
wants the literature read first (his message: notebook with Ranga, then ultra-plan with unlazy).

## WHAT THE BRAIN AND REPO ALREADY KNOW (sweep receipt)

| surface | evidence it ran | finding | call |
|---|---|---|---|
| repo git state | `git status -sb`; `git worktree list`; `git rev-list --left-right --count origin/main...claude/lane-auto-d-20260926` → `0 2` | lane already scaffolded by the overnight lane at 23:25Z: GOAL.md drafted (confirm at G0), arcs.md empty | **resume** this branch/worktree |
| reuse targets | `git show origin/main:src/model_selection.jl`, `src/cv.jl:304`, `src/boundary_inference.jl:28`, `src/ppca_init.jl:47` | `select_lv(Y; family, Kmax=3, criterion=:bic)` sweeps K=1:Kmax, try/catch per K, AIC/BIC only, no K=0; BIC uses nobs = p·n cells (log(pn) penalty); `cv_gllvm` takes one fixed K; `chibar2_pvalue(LRT,q)` is the Self–Liang variance-boundary mixture; `ppca_init` gives closed-form eigen decomposition (a free spectral guess) | **reuse** all four |
| tests | `test/test_model_selection.jl` (16 @test) | asserts best_k ∈ 1:3 and monotone loglik; **no recovery-to-truth test** | gap |
| twin gllvmTMB | read-only: `R/select-lv.R:156`, `R/brms-sugar.R:607` | R `select_lv` sweeps d=1:d_max, :bic/:aic/:aicc; default d = 1; anova with chi-bar p-values (#1249) | co-opt design; **no writes** (D-220 amendment) |
| brain (vault first) | `search_notes "number of latent dimensions selection select_lv auto d GLLVM"`; `grep -in select_lv/chibar/"latent dimension" memory/AGENT_LOG.md memory/DECISIONS.md projects/deep-research/README.md journal/` | no prior decision on auto-d; dr21 mentions AGHQ as ground truth for "number of latent dimensions"; HMSC ships an MGP shrinkage prior, gllvm/VAST ship none (AGHQ-runaway note); runaway loadings can mimic an extra dimension | reuse the hazards; nothing to resume |
| fence | `LOOP/lanes/gllvm-backlog-20260926/OVERNIGHT.md` (branch claude/lane-gllvm-backlog-20260926) | overnight lane runs to 11:00Z, ≤2 live agents, **hit the account usage limit twice tonight**; fenced off model_selection.jl, cv.jl, K/d handling in fit_gllvm.jl | stay lean while it runs |
| lane preflight | `tools/lane_preflight.sh` | FOREIGN LANE ACTIVE (codex, cursor, direct-to-main) + 19 claude lanes; leases on com_poisson.jl, ordered_beta.jl | claim exact files with `LANE_ID=claude:GLLVM.jl:auto-d` before any edit |
| NotebookLM | MCP: CONNECTION_CLOSED; CLI `notebooklm auth check --test --json` | `token_fetch: false` → **login expired** | Shinichi runs `notebooklm login` (personal account), or fall back (Q1) |

**Verdict:** reuse select_lv / cv_gllvm / chibar2 / ppca_init; resume the scaffolded lane; the genuine gap is
(a) a chosen default rule with evidence, (b) an auto path in the K handling, (c) a recovery test, (d) the caveat.

## Route check (Phase 0.6) → DECISION MAP, not a slice list

Destination is writable, but the build slices depend on "which route" and "which API", so Phase 0 builds a map.

### Destination
A GLLVModels.jl user fits a GLLVM without supplying K. The package chooses K by one documented default rule,
returns the chosen K together with the per-candidate criterion values and the runner-up, the rule's recovery
rate is measured on simulated data across n, p and family, and every place a user reads an interval says it
is conditional on the chosen K.

### Decisions so far
- Reuse, not rebuild: select_lv, cv_gllvm, chibar2_pvalue, ppca_init (Shinichi, lane brief).
- Literature pass first, via Ranga (Shinichi, 2026-09-26 chat). Execution under unlazy gates (same message).
- No writes to gllvmTMB; R-side auto-d is a request to the Cursor lane (D-220 amendment).
- Public API change needs Shinichi's sign-off (lane brief).
- **Omitting K/d means "estimate it"** (Shinichi, 2026-09-26 chat: "OK what if we do not supply d - yes").
  This turns today's Julia error into behaviour; formal sign-off on the full API still at G1.
- **A1's first job is feasibility**: have people estimated the number of latent variables in GLLVMs and
  factor models, by what methods, and with what recovery? (Shinichi, same chat.) HSquared reuse is deferred.

- **gllvmTMB gets the same rule** (Shinichi, 2026-09-26: "not only GLLVModels.jl but also gllvmTMB too").
  Under the D-220 amendment the Cursor lane owns gllvmTMB, so this lane delivers an R-side spec
  (`docs/design/74-*.md` §"R twin": `latent(d = )` omitted ⇒ auto, same criterion, same record, same caveat)
  plus a handover to the Cursor lane (`handover-to-cursor`). No gllvmTMB writes from here unless Shinichi
  moves that work to this lane. A1's research covers both packages; the destination now reads "Julia ships,
  R spec handed over".
- NotebookLM: Shinichi re-logs in (`notebooklm login`, personal account); A1 starts once
  `notebooklm auth check --test --json` shows `token_fetch: true`.

### Plan review (Fisher + Rose, Opus, 2026-09-26) — adopted
- Chi-bar-squared demoted to a diagnostic: K vs K+1 is a Davies-type problem (the direction of the extra
  column is unidentified under the null), so the null is likely heavier than ½χ²₁ and the test would
  over-select (lead, UNVERIFIED: Hayashi–Bentler–Yuan 2007, Drton 2009). Parametric bootstrap LRT is the
  defensible reference. The claim at `src/boundary_inference.jl:8` needs fixing in-lane.
- Pilot adds BIC with log(n) (`bic(fit, n::Integer)`, `src/postfit.jl:599`, no src edit) next to log(p·n);
  K_true ∈ {0,1,2,3}; Kmax ≥ K_true+2; ≥200 reps; metrics = exact rate, P(too few), P(too many), MCSE,
  per-K failure rate.
- Hazards: `select_lv`'s bare catch drops failed K silently and never reads `converged`
  (`src/model_selection.jl:70-83`), biasing down; runaway Laplace loglik (`src/families/binomial.jl:133-141`)
  biases up; K=0 unsupported; NB/Beta/Delta route through `disp_group=:species` (`fit_gllvm.jl:225-230`),
  so "default route only" would drop NB. T6 becomes a decide-with-Shinichi ticket.
- Kmax ceiling: K < p; Ledermann bound for `pervar`.
- Gates tightened: G1 needs a resolved DOI per sub-question or an explicit "no source found"; G2 keeps NA
  and reports failure rate; G4 threshold fixed from the pilot's lower confidence bound BEFORE A4; G5 lists
  failed candidates with reasons; G7 needs MCSE and too-few/too-many columns.
- Caveat extended: chosen K has sampling variability (report runner-up gap); ΛΛ′, variance partitions and
  ordinations are conditional on K; under misspecification K grows with n ("dimensions supported at this n");
  recovery rates hold only for the simulated setups.
- **A4 waits until the overnight lane closes (11:00Z) and its PRs merge**, then rebase and re-run preflight;
  A2 runs against origin/main with no src edits.

### Open tickets (each is a QUESTION)
| id | kind | question | unblocks |
|---|---|---|---|
| T1 | research (Ranga) | How do GLLVM / factor-model papers choose the number of latent variables, and how well does each rule recover it at ecological n, p? Cover: AIC/BIC/AICc/EBIC and which "n" goes in BIC (sites vs cells); LRT K vs K+1 and whether chi-bar-squared is even right (the extra column is not a single variance on a boundary: loadings are unidentified under the null); CV; ordered factor LASSO; MGP / cumulative shrinkage priors; spectral rules (parallel analysis, eigenvalue ratio, Bai–Ng); what gllvm, HMSC, boral, VAST, gllvmTMB do by default; post-selection inference for K. | T2, T5 |
| T2 | decide-with-Shinichi | Route: fit-and-compare, single-fit shrinkage, spectral, or a hybrid (spectral guess sets the candidate window, then fit-and-compare)? And the default criterion. | build |
| T3 | prototype | Measured, on the existing API: which criterion recovers true K best at small/moderate n for Gaussian, Poisson, Binomial, NB? (recovery pilot, no src edits) | T2 |
| T4 | decide-with-Shinichi | API: does omitting K mean auto (turns today's error into behaviour; diverges from R's d = 1), or an explicit `K = :auto`? Include K = 0? Default Kmax rule? Return a GllvmFit carrying a selection record, or the LVSelection? | build |
| T5 | decide-with-Shinichi | Caveat stance: conditional-on-K wording only, or also offer bootstrap that re-selects K per replicate to propagate selection uncertainty? | docs, confint |
| T6 | task | Which fit_gllvm routes get auto in v1 (default family route only?), and does K = 0 fit on each family? | build scope |

### Not yet specified (fog)
- Cost at large p: a sweep costs Kmax fits; whether warm-starting K+1 from K matters.
- How auto interacts with `confint`/bootstrap and with `predict`.
- Whether runaway loadings (AGHQ note) inflate the chosen K, and whether a detector must gate the choice.
- R-parity: a written request to the Cursor lane once the Julia rule is fixed.

### Out of scope
- gllvmTMB edits (Cursor lane, D-220 amendment). Overnight-lane files and PRs (OVERNIGHT.md fence).
- New selective-inference theory; MCMC/Bayesian engine.
- grouped / phylo / row_eff / pervar auto routes (revisit after v1).

## Slices (arcs), roles, models

| # | arc | member | model · effort · dispatch | time | output | dep |
|---|---|---|---|---|---|---|
| A0 | RECON (done in planning) | Ada | Haiku · low · Agent model param | done | receipt above | — |
| A1 | Literature pass (T1) | Ranga via `/notebook` | inline skill on Opus session (Gemini reads corpus); Sonnet sub-agent only for distillation if needed · medium | 1–2 h after login | vault dr-note `projects/deep-research/dr-auto-d-*.md` + repo `docs/design/74-auto-latent-dimension.md` (slot 74 is next free; claim by committing) | Q1 |
| A2 | Recovery pilot (T3) | Curie (sims) | Sonnet · medium · Agent model param | build 1 h; run: estimate after pre-run test | `LOOP/lanes/auto-d-20260926/pilot/` script + CSV + table | none (parallel to A1) |
| A3 | Decision memo + map close (T2, T4, T5, T6) | Ada + Fisher review | Opus 5.5 · high (Fable if the LRT question stays contested) | 45 min | memo in design/74, map tickets closed | A1, A2 |
| G1 | **Shinichi sign-off** on route, criterion, API | Shinichi | — | — | DECISIONS entry | A3 |
| A4 | Build (TDD) | R-package/Julia engineer | Sonnet · medium | 3–5 h | select_lv extensions + auto path in K handling + recovery test | G1 |
| A5 | Recovery campaign | Curie | Sonnet · medium; Totoro | estimate stated first | recovery table in design/74 | A4 |
| A6 | MECHANICAL-VERIFY | scout | Haiku · low | 15 min | gate re-run log | A4, A5 |
| A7 | D-43 panel | 2 Sonnet + 1 Opus (Fisher lens) | fresh contexts | 30 min | verdicts | A6 |
| A8 | Docs + caveat + draft PR | Rose | Sonnet · medium | 1 h | docstring, docs page, draft PR (push only if authorised at G0) | A7 |
| A8b | R-twin spec + Cursor handover | Rose | Sonnet · medium | 45 min | design/74 §R twin + `docs/dev-log/handover/*-cursor-auto-d.md` | G1 |
| A9 | RECONCILE | Melissa | Sonnet · low | 15 min | `docs/dev-log/plan-actual/2026-09-2x-auto-d.md` | A8 |

FAN-OUT BUDGET: ≤2 live while the overnight lane runs (to 11:00Z / 05:00 MDT), because both lanes spend one
account window and it has already run out twice tonight. ≤6 new children per checkpoint, ≤1 Opus child.
SCOUT SUITABILITY: yes (A0 done on Haiku; A6 Haiku).
ESTIMATE: planning-to-G1 ≈ 3–4 h wall (dominated by A1); G1-to-draft-PR ≈ 1 working day. Two sessions; the
lane kit (GOAL/arcs/checkpoint) carries state across compaction via /arc-loop.

## Acceptance ledger (unlazy) — written before dispatch

`.unlazy/auto-d/` in the lane worktree (added to `.git/info/exclude` first). Gates, per arc:
- A1: G1 design/74 exists, has a section per T1 sub-question, every claim carries a citation or the tag UNVERIFIED — CHECK `grep -c UNVERIFIED\|doi\|arXiv docs/design/74-*.md` + manual read by Fisher.
- A2: G2 pilot CSV non-empty, one row per (family, n, p, K_true, criterion, rep), no NA in chosen_K — CHECK a Julia/awk one-liner; EXPECT row count = grid size.
- A4: G3 `julia --project -e 'using Pkg; Pkg.test(test_args=["test_model_selection"])'` passes; G4 new recovery test asserts chosen K == K_true in ≥ stated fraction on a literal seeded fixture; G5 `fit_gllvm(Y; family=Poisson())` (no K) returns a fit whose selection record lists every candidate; G6 explicit-K calls return bit-identical results to origin/main on the existing fixtures (no healthy-fit change).
- A5: G7 recovery table in design/74 with rates by criterion × family × n × p, and the run's estimate vs actual.
- A8: G8 docstring and docs page contain the conditional-on-K caveat; `agent_mention_check.py --text <PR body>` clean.

## Verification
- Re-run the ledger with `gate-check.mjs --reverify` per arc; exit 0 required. Try to refute one passed gate.
- Package tests: `julia --project -e 'using Pkg; Pkg.test()'` (≤4 threads, OPENBLAS_NUM_THREADS=1), full suite before the PR.
- Recovery: known-DGP simulation, pre-run test on the laptop first, Totoro for the campaign if the estimate exceeds laptop scale; any run estimated over 3 h comes back to Shinichi with the pre-run result (D-287).
- D-43 panel before any "done" claim.

## Post-selection caveat (to report honestly throughout)
Choosing K from the same data and then fitting at that K makes every later interval, p-value and
chi-bar-squared test conditional on the choice; they understate uncertainty about K itself. The K vs K+1 LRT
is non-regular (loadings unidentified under the null), so a chi-bar-squared reference is an approximation to
be checked in the pilot, not assumed. Interpretations of individual latent axes change as K changes.

## PRE-AUTHORISED AFTER G0
Scoped edits in the lane worktree; routine local commands; Julia tests/builds (≤4 threads); pilot runs
estimated ≤3 h; checkpoints; local commits on claude/lane-auto-d-20260926; `/notebook` notebook creation on
the personal Google account.
OPTIONAL REMOTE AUTHORITY: none until G1 (then: push branch + open a draft PR; never merge).
MUST STOP: any public API change before G1; merge/release; any gllvmTMB write; edits to overnight-lane files;
compute beyond the stated estimate; evidence that changes scope.

RECONCILE: Melissa at close (A9).
