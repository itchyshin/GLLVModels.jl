# Codex handover: true-parity-latest lane, 2026-10-06

You are Codex, picking up the true-parity-latest lane from a Claude Code session. You did not see that session. This file, `AGENTS.md`, and the files named below are everything you need. Read `AGENTS.md` first (it is native for you), then all of this file before acting.

## 1. Goal and boundaries

**Goal.** GLLVModels.jl at true parity with gllvmTMB main pinned at **P1 = `9539352f6`** (gllvmTMB 0.7.1). Boundary D-295 (signed 2026-09-27): temporal and phylo latent are inside P1; column grammar and spatial are outside, closed by signed disposition and revisited at P2. Every P1 capability inside the boundary gets a scoreboard row with a receipt made at P1, or a 0.7.0 receipt whose source files are byte-identical at P1. After P1 closes, re-pin to the newest gllvmTMB main and repeat (issue #833).

**Done when.** Clauses C0 to C8 and the F-gates reverify MET from tracked files on origin/main (`node tools/true_parity_check.mjs <clause>`), the negative controls fail as expected (`node tools/test_true_parity_check.mjs`), and a handover is on main.

**Stop and ask the maintainer before:** any merge; releases, tags or `Project.toml` (it stays `0.3.0`); public parity claims; edits to the shared Laplace mode code (`src/families/laplace.jl` `_laplace_mode*`) beyond the approved weights change; DRAC jobs (give a time estimate first); any run over 3 hours (pre-run test and approval first); another lane's files; case-map classifications or admission-set changes. **Agents never sign.** Signatures, classifications and dispositions are the maintainer's; you may record a signature he gave, never create one.

## 2. State on origin/main @ `b57b04d9b`

| Clause | Result | Meaning |
|---|---|---|
| C0 | MET | pin and oracle provenance |
| C1 | MET | required-source rows bound (20 numeric, 4 behavioural) |
| C2 | NOT MET, 283 done | scoreboard rows inside the boundary |
| C3 | NOT MET, 6 of 8 | realistic-size campaign rows |
| C4 | NOT MET, 4 of 8 | real-data workflows |
| C5 | MET, 4 of 4 | grouping-level rows |
| C6 | NOT MET | 378 Julia-only exports, 37 still undecided |
| C7, C8 | MET | |
| X2 | 297 of 317 | all scoreboard rows |
| F-gates | informational in GATES.md | they fold into C2 once their ids are scoreboard rows |

Negative controls: all pass. CI (`.github/workflows/true-parity-check.yml`) runs the checker and assembler controls, `true_parity_assemble.py --check`, `tools/test_core070_behaviour_receipts.py`, and the receipt-tool `--check`s for aghq, data, family, inference, isdm, namespace_2, postfit, behaviour, fixture receipts, bridge readback and campaign. It does not run `tools/core070_covariance_p1_receipts.py`, `tools/core070_grouping_p1_receipts.py` (broken, #824) or the C6 classifier (needs Julia, #837).

## 3. The 20 open rows, one by one

| Row | Blocker | Owner | Next action (section 5) |
|---|---|---|---|
| family/FAMILY-11-LOG | numeric cases pass; bridge context case is NOT_EXECUTED, and N1 accepts only an R refusal recorded as `R_BOUNDARY_UNCHANGED` | Codex | 5.1 |
| family/FAMILY-00-IDENTITY | native and formula cases FAIL on receipts made before the `["z_B"]` expectation fix (#821); bridge case NOT_EXECUTED and R's bridge fits a different model (df 5 vs 8), so it cannot be boundary context | Codex re-measures; maintainer decides the bridge leg | 5.2 |
| family/ORDINAL-LOGIT-RSZ | LLt diff 1.30e-4 > 1e-4 | Codex | 5.3 |
| isdm/ISDM-HEADLINE-RSZ | predict link/response diffs 7.7e-5 to 1.5e-4 > 1e-6 | Codex | 5.3 |
| data/RD-SPIDER-NB2 | logLik 1.46e-6 > 1e-6; LLt 1.62e-4 > 1e-4; predicted link 6.7e-5 > 1e-6 | Codex | 5.4 |
| data/RD-FUNGI-BINOMIAL | beta 1.24e-4 > 1e-4; LLt 4.7e-4; predicted link 3.3e-4 | Codex | 5.4 |
| data/RD-URBANISATION-BINOMIAL | beta 2.3e-4; LLt 1.6e-3; predicted link 8.2e-4 (raw files are off the public repo) | Codex | 5.4 |
| data/RD-BEETLE-NB2 | R's Hessian not positive definite; logLik 9.1e-6; beta 2.9e-4; LLt 4.6e-3 | Codex diagnoses; may need the maintainer | 5.4 |
| postfit/POSTFIT-SURFACE-check_auto_residual | in the behavioural scope (N10) but its receipt has no behaviour block | Codex | 5.5 |
| isdm/ISDM-COUNT, -EXTRA-SOURCE, -MISSING-IN-TRAIT, -MISSING-SOURCE, -WRAPPER-LAW | in the behavioural scope (N6) but need Julia public-door runs and behaviour blocks (#836) | Codex | 5.6 |
| namespace/export/extract_phylo_signal | name match only; a bare phylo_latent fit gives H2 = 1, which cannot discriminate; needs the joint phylo plus grouped route (#834) | Codex | 5.7 |
| namespace/export/bootstrap_Sigma, namespace/S3method/simulate,gllvmTMB_multi | name match only; need Monte-Carlo twins under a pre-stated N4 rule | Codex | 5.8 |
| postfit/POST-SIMULATE-DEFAULT | its Monte-Carlo twin fails 7 of 40 alternative seeds; held (#828) | Codex investigates; maintainer decides a new pre-registration | 5.8 |
| postfit/POST-COEF-EMPTY | a 0 = 0 length comparison is non-discriminating under the shared rule | maintainer (a discriminating comparand or a disposition) | none for agents |
| namespace/export/animal_latent | needs a random-slope `animal_latent`; left unbound by ruling (#835) | maintainer | none for agents |

C6 (37 names) is separate: the maintainer reviews `c6-proposed-decisions-2026-10-05.md` (lane kit), with you preparing evidence if he asks (#831). Signed rows go into `SIGNED_OVERRIDES` in `tools/true_parity_c6_classify.jl`, then the classifier regenerates `reverse-gap-decisions.json` (never hand-edit it).

## 4. Live toolchain

```bash
export PATH="$HOME/.juliaup/bin:$PATH"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1   # the Mac is shared and often loaded
export GLLVM_P1_RLIB="$HOME/local-scratch/gllvmTMB-p1-scratch/Rlib"     # frozen gllvmTMB 0.7.1 at P1 (Mac oracle)
export GLLVMTMB_DIR="$HOME/Dropbox/Github Local/gllvmTMB"               # read-only clone; read P1 sources with git show 9539352f6:<path>
export NOT_CRAN=true
```

- Install: `julia --project=. -e 'using Pkg; Pkg.instantiate()'`. Worktrees have no Manifest: copy the git-ignored `Manifest.toml` from the main checkout.
- Full suite: `julia --project=. -e 'using Pkg; Pkg.test()'`. P1-tagged twin tests (first line `# gllvm-parity-tag: P1`) run in CI as `julia --project=. test/<file>` in the root env and may use only root dependencies (SHA, Distributions, ForwardDiff, LinearAlgebra, Optim, Printf, Random, SparseArrays, SpecialFunctions, Statistics, StatsModels, Tables, stdlibs). No StableRNGs, no JSON3; fixtures in TOML.
- **Totoro** (384 cores, no GPU, no Duo): only through the existing socket, `ssh -o ControlPath=~/.ssh/cm-snakagaw@totoro.biology.ualberta.ca:22 -o BatchMode=yes snakagaw@totoro.biology.ualberta.ca`. At most 150 cores; set BLAS/OMP threads explicitly (OpenBLAS defaults to 192 there); kill what you start. Totoro has a P1 oracle build registered as `GLLVM_PARITY_ORACLE_BUILD=totoro` (`build-totoro.json`, PR #814) and a working R-to-Julia bridge set up at `~/hsq_work/gllvm-w2-3` (Julia 1.10.12, TMB 1.9.21, Matrix 1.7.5 in a private library). Linux bridge quirks: preload Julia's libunwind into R for JuliaCall, and clear `LD_LIBRARY_PATH` for Julia children started from R (a `julia` wrapper on PATH works).
- **kohaku** is a GPU box (8 vCPU cap), not for CPU campaigns. **DRAC** only by `sbatch`, never the login node, with an estimate first.
- Estimate before every run. Over 3 hours: pre-run test plus approval.

## 5. Procedures for the open work

Each item is one branch and one PR: implement, regenerate, verify, independent review, then the maintainer's merge. After any ledger change run, in order: the owning receipt tool (`--apply-twins`, `--write` or `--rederive`, as that tool documents), `python3 tools/true_parity_assemble.py`, then every `--check`, `node tools/test_true_parity_check.mjs`, `python3 tools/test_true_parity_assemble.py`, and `PARITY_REF=HEAD node tools/true_parity_check.mjs X2` (plus the clause you touched).

### 5.1 FAMILY-11-LOG (+1)
1. On Totoro, in the bridge environment, run the truncated-NB2 case through gllvmTMB's public bridge (`gllvmTMB(..., engine = "julia")`) at P1 and capture R's refusal (the GJL-GATE-FAMILY gate). Use the fixture behind `CORE070-FAMILY-11-LOG-PUBLIC-R-BRIDGE`.
2. Write it as a receipt with `kind: r_public_bridge_boundary`, verdict `R_BOUNDARY_UNCHANGED`, no comparison block, on the `-PUBLIC-R-BRIDGE` case id, with P1 provenance (library NAMESPACE hash and version, as `tools/core070_family_bridge_p1.R` records).
3. The family tool's `n1_context` (`tools/core070_family_p1_receipts.py`, `FAMILY_N1_CONTEXT`) then binds the row; its note already says so.
4. Acceptance: FAMILY-11-LOG EVIDENCED; X2 +1; the negative control that refuses a NOT_EXECUTED bridge case still passes.

### 5.2 FAMILY-00-IDENTITY (re-measure; may stay partial)
1. Re-run the Gaussian family parity cell at P1 (Totoro with `GLLVM_PARITY_ORACLE_BUILD=totoro`, as PR #814 did) now that `test/parity/test_gaussian_original_required.jl:51` expects `["z_B"]`. Regenerate the family receipts from one clean commit (the tool requires that), or use its `--rederive` mode if only the tool changed.
2. If native and formula now PASS, the row still has a bridge case that is NOT_EXECUTED and whose R bridge fits a different model, so N1 cannot excuse it. Report to the maintainer; do not reclassify.
3. Also check `tools/core070_default_unique_pair.jl:35`, `tools/core070_aghq_admission_verify.py:77-78` and `test/parity/isdm_unique_cases.jl:53`, which still expect `['z_B','s_B']` (#830). The iSDM P1 fixture still shows `s_B`, so decide per model.

### 5.3 ORDINAL-LOGIT-RSZ and ISDM-HEADLINE-RSZ (C3, +2)
Ruling N9 (maintainer, 2026-10-05, read as of 2026-10-06): a C3 row binds on numbers compared at a point where both engines reach max-abs gradient 1e-5, recorded in a `convergence_parity` block; the rule applies only to receipts that carry the block. The checker enforces it (`convergenceParityProblem` in `tools/true_parity_check.mjs`; the assembler mirrors it).
1. Land the polish harness from draft PR #715 (`polish_J.jl`, `run_R.R` changes); read `gh pr view 715` and its diff.
2. Teach `tools/true_parity/campaign/write_receipts.py` to record `compared_point: "newton_polished"`, both engines' max-abs gradients, and the `convergence_parity` block (`bound: 1e-5`).
3. Re-run both campaigns with polish (ORDINAL about 10 to 15 minutes single-threaded; ISDM similar; Totoro is fine with threads capped).
4. Acceptance: each row EVIDENCED only if both gradients are at most 1e-5 and every comparison passes its existing tolerance. Tolerances are not changed. If R cannot reach 1e-5, report it; do not loosen anything.

### 5.4 C4 real-data rows (urbanisation, spider, fungi, beetle)
C4 now accepts direct-engine runs (ruling of 2026-10-05). The four rows miss tolerances by small amounts, consistent with optimisers stopping early, as on the C3 rows. Check each receipt's recorded gradients before any re-run.
1. Apply the 5.3 polish to these campaigns and re-run; tolerances stay as they are.
2. Beetle: R's Hessian is not positive definite. Diagnose it (boundary dispersion, too many factors?) and report before claiming anything.
3. Urbanisation's raw files are off the public repo; its receipt is checked for internal consistency only. Ask the maintainer where the raw run lives before re-running.
4. Acceptance: each row EVIDENCED on passing comparisons at polished points; otherwise a written diagnosis per row.

### 5.5 check_auto_residual (+1)
Add a behaviour block for `postfit/POSTFIT-SURFACE-check_auto_residual` through `tools/core070_behaviour_receipts.py` (the row is already in `BEHAVIOURAL_EXTENDED_SOURCE_IDS`), record the R and Julia routes at P1, and regenerate with the postfit tool. Acceptance: the row binds as behavioural; `tools/test_core070_behaviour_receipts.py` passes.

### 5.6 Five iSDM public-door rows (+5, #836)
For each of ISDM-COUNT, -EXTRA-SOURCE, -MISSING-IN-TRAIT, -MISSING-SOURCE, -WRAPPER-LAW: run the case through Julia's public iSDM door (`fit_isdm_gllvm`), record the refusal or result class, add the behaviour block, and regenerate with `tools/core070_isdm_p1_receipts.py`. ISDM-EXTRA-SOURCE's R side refuses at the earlier `length(family)` guard (`R/fit-multi.R:1440` at P1), so compare that class. The three internal-predicate rows are already closed by signed disposition.

### 5.7 extract_phylo_signal (+1, #834)
Build a P1 twin on a fit with both a phylogenetic and a grouped latent tier so H2 is estimated; compare `extract_phylo_signal` (and its intervals if R gives them) against Julia's `phylo_signal`; bind the namespace row. Use `fit_phylo_latent_gllvm` and the fixture style in `docs/dev-log/core070/phylo-latent-p1/`.

### 5.8 Monte-Carlo rows
For `bootstrap_Sigma` and `simulate,gllvmTMB_multi`: commit the Monte-Carlo rule file before any run (N4; see `receipts/inference/ci-route-011-mc/rule.json` and `test/test_mc_simulate_p1.jl`), include a deliberately different alternative that must fail, then run and bind. For POST-SIMULATE-DEFAULT (#828): find out whether the stored R fixture is an unusual draw or R's default `simulate()` differs from Julia's (read R's default simulate path at P1; draw a larger R sample). A new twin needs a new pre-registration and the maintainer's agreement. Never loosen a pre-registered rule after seeing results.

## 6. What landed this session

| PR | What |
|---|---|
| #806 | profile and bootstrap entry points refuse `lambda_constraint` fits (#804); masked Gaussian fits return `+Inf` at non-PD trial points (#805, closed #716) |
| #807 | stored training offset used by `getLV`/`predict`/`residuals` on 11 shared-dispersion fit types (refs #788) |
| #808 | offsets in `fit_mixed_gllvm` with gllvmTMB's row-wise admission rule; DATA-OFF-MIXED |
| #809 | namespace-2 batch runs at P1; shared discriminating rule applied to its receipts |
| #810 | six covariance twins (COV-ORD-LATENT x3, COV-PHYLO x3); stationarity asserted instead of the `converged` flag |
| #814 | family rows re-measured on Totoro; FAMILY-02, -05, -07, BETA-ALIAS; route-vs-native entries in `route_consistency` |
| #817 | phylo docstrings corrected; postfit and isdm receipt checks repaired; receipt checks added to CI |
| #819, #821 | C6 inputs refreshed; 23 signed C6 names; `predict(...; modes = :training)`; DATA-OFF-PREDICT; FAMILY-00 expectation |
| #838 | the maintainer rulings of 2026-10-05 (four slices integrated; X2 206 to 284) |
| #839 | observation weights on the Poisson Laplace route via the shared mode code; unweighted fits bit-identical (253/253); 13 DATA-W rows (X2 284 to 297) |
| #818, #840 | lane handovers |

Issues filed for future work: #820 (Delta `predictor = :shared` post-fit bug), #822 to #837.

## 7. Decisions you must respect

- **Signed rulings** (vault `memory/DECISIONS.md` D-319, 2026-10-05, and follow-ups of 2026-10-06): every recommendation on the rulings page; D signed yes (phylo-latent promotion); animal_latent stays unbound; C4 text changed to accept direct-engine runs; C6 high-confidence group signed, medium and low held; N9 applies only to receipts carrying the convergence block; N6 internal predicates are ISDM-NO-TRAITS, -WRONG-ID, -WRONG-LINK; the weights edit to the shared Laplace mode approved under review; the six helper-internal weights rows bind on fit-level twins. Rule texts: `docs/dev-log/core070/true-parity-latest/GATES.md`, section "Rulings of 2026-10-05".
- **Shared discriminating rule:** any comparison whose R value is below `DEGENERATE_ABS = 1e-10` marks the receipt `numeric_non_discriminating`, which does not bind (`mark_degenerate` in `tools/core070_postfit_p1_receipts.py`; the data, family, inference and namespace tools use it).
- **Bridge readback** (R returning Julia's numbers) binds only for `fitted`, `predict`, `residuals`; route-vs-native entries never sit in a receipt's `comparison` block.
- **N1 boundary context** excuses only an R public-bridge refusal (`R_BOUNDARY_UNCHANGED` on a `-PUBLIC-R-BRIDGE` case) or an admission-only formula case, never a NOT_EXECUTED case, and at least one other case must be compared.
- **Monte-Carlo rows** bind only on a rule committed before the run, with a must-fail alternative.

## 8. Gotchas found the hard way

- Generated files (`case-map-*.json`, `case-map-assembled.json`, `scoreboard.md`, `reverse-gap*.json`, `behaviour-equivalence.json`) are never hand-merged. On a conflict, write main's version (`git show origin/main:<path> > <path>`) and regenerate with the owning tool.
- A receipt tool's `--check` that is not in CI drifts silently; two broke main before #817.
- An optimiser's `converged` flag can differ between macOS and Linux at the same optimum (#822); assert stationarity (positive-definite Hessian, Newton decrement at most 1e-9, gradient at most 1e-4) instead.
- Seeded RNG streams differ between Julia 1.10 and 1.12+; store drawn data in fixtures rather than regenerating from a seed.
- Run `python3 tools/wrong_close_guard.py --body-file <body> --git-range origin/main..HEAD` before every push; "refs #N" goes on its own line; an intended close needs an `<!-- intended-closes: N -->` comment and `--intended N`. Agents are never named as GitHub @handles.
- GitHub sometimes cancels CI jobs en masse; re-run cancelled jobs once before treating it as a failure.
- Hooks on this machine block `git checkout -- <path>` and `git checkout --theirs`; reconstruct forward with `git show` instead.

## 9. Where truth lives

- Ledger, rules and checker: `docs/dev-log/core070/true-parity-latest/GATES.md`, `tools/true_parity_check.mjs`, `tools/true_parity_assemble.py`, `scoreboard.md`, `case-map-assembled.json`.
- Phylo-latent receipts and the dated promotion block: `docs/dev-log/core070/phylo-latent-p1/README.md`.
- Lane kit (local, not in git): `~/local-scratch/lanes/GLLVM.jl-true-parity-latest/LOOP/lanes/true-parity-latest/` with `GOAL.md`, `checkpoint.md`, `signed-rulings-2026-10-05.md`, `wave-plan-2026-10-03.md`, `weights-design-2026-10-05.md`, `c6-proposed-decisions-2026-10-05.md`, `post-nobs-fallback-evidence.md`, and `scripts/merge_train_v2.sh` (head-pinned merge train; `MERGE_METHOD=--merge` for integration PRs).
- Previous lane handover: `docs/dev-log/handover/2026-10-05-claude-handover-true-parity-latest.md`.

## 10. How to resume

Start Codex in the repository root and paste:

> Rehydrate from docs/dev-log/handover/2026-10-06-codex-handover.md + the AGENTS.md snapshot, then continue with the Next Immediate Steps.

Suggested order: 5.1, 5.5 and 5.6 first (cheap, agent-only), then 5.3 and 5.4 (campaign polish), then 5.2, 5.7 and 5.8. If every agent item succeeds, X2 reaches about 313 of 317; the last four rows (FAMILY-00's bridge leg, POST-COEF-EMPTY, POST-SIMULATE-DEFAULT, animal_latent) and C6 need the maintainer.
