# Initial P1 panel: failed early candidate

These are verbatim independent reviews of aef746495. They are historical findings, not current signoff. Parent repairs are recorded in the reconciliation.

## emmy

**BLOCKED: add both receipt-control files to `test/runtests.jl` before accepting this checkpoint.** They are new Julia tests, but neither is included there, so the package suite will not run them. The include list ends at [test/runtests.jl:589](/Users/z3437171/local-scratch/lanes/GLLVM.jl-true-parity-p1-next/test/runtests.jl:589); the unregistered files are [test/test_family11_p1_boundary.jl](/Users/z3437171/local-scratch/lanes/GLLVM.jl-true-parity-p1-next/test/test_family11_p1_boundary.jl:1) and [test/test_first_seven_behaviour_p1.jl](/Users/z3437171/local-scratch/lanes/GLLVM.jl-true-parity-p1-next/test/test_first_seven_behaviour_p1.jl:1). Confirm with:

```sh
rg -n 'test_family11_p1_boundary|test_first_seven_behaviour_p1' test/runtests.jl
```

The search returned no matches. Both controls are opt-in, so registration alone will not execute them in the normal suite; their enabled path must also be run explicitly.

**What I verified:** all five owning `--check` commands and the assembler check passed under Python 3.14.8. The checkpoint negative-control command printed `P1_CHECKPOINT_CONTROLS_PASS`. The v4 raw TSV hashes independently match `run.json`; its metadata identifies P1 and the recorded Julia launch commit/source tree. These checks support receipt consistency, but do not establish the raw runs independently.

The Python behavior controls did **not** complete: temporary-directory fixtures failed in this read-only environment, and the R-backed panel control also failed. The default `python3` here is 3.10.12, which lacks `tomllib`; the owning checks passed only when explicitly run with Python 3.14. So the opt-in Julia wrappers’ use of `python3` also needs a compatible interpreter in the execution environment. I did not establish whether that causes a failure on the specified Totoro setup.

This verdict is limited to the tooling and provenance scope reviewed; it does not certify numerical parity or the parent’s pending Julia suite. Graft reported approximately 49,705 tokens saved this turn.

## pat-fresh

**BLOCKED for exact candidate `aef7464`.** Its checkpoint checker rejects the postfit receipt, so this commit does not pass its own checkpoint gate.

In `aef7464`, `provenanceProblem` requires hashes at the receipt’s top level ([checker lines 49–53](</Users/z3437171/local-scratch/lanes/GLLVM.jl-true-parity-p1-next/tools/true_parity_checkpoint_check.mjs:49>)). The postfit receipt keeps its hashes under `behaviour.read_from` ([receipt lines 47–55](</Users/z3437171/local-scratch/lanes/GLLVM.jl-true-parity-p1-next/docs/dev-log/core070/true-parity-latest/receipts/postfit/cases/CORE070-WAVE7-CHECK-AUTO-RESIDUAL.json:47>), while the case map cites that receipt at [case-map-postfit.json lines 154–168](</Users/z3437171/local-scratch/lanes/GLLVM.jl-true-parity-p1-next/docs/dev-log/core070/true-parity-latest/case-map-postfit.json:154>). Running the checkpoint command at the requested candidate returned `P1_CHECKPOINT_NOT_MET: postfit/POSTFIT-SURFACE-check_auto_residual: missing raw hashes`.

During review, HEAD advanced to `ce3432b`, which adds support for nested behavior hashes. A subsequent run printed `P1_CHECKPOINT_MET` for ref `aef7464`, but it used the newer checker code; that does not change the result for the exact candidate. The worktree remained clean, and I made no changes.

**Public-case review:** I inspected both recorded runner scripts, the v4 raw observations, and the derived receipts. The six exact executable cases are `CORE070-FIRST7-CHECK-AUTO-RESIDUAL`, `CORE070-FIRST7-ISDM-COUNT`, `CORE070-FIRST7-ISDM-EXTRA-SOURCE`, `CORE070-FIRST7-ISDM-MISSING-IN-TRAIT`, `CORE070-FIRST7-ISDM-MISSING-SOURCE`, and `CORE070-FIRST7-ISDM-WRAPPER-LAW`. The COUNT case is the ordinary all-count route; the runner separately fits a mixed-source positive control. The wrapper case constructs logit standalone and records the collector’s refusal. Family-11 records refusal at both public R bridge routes. These are inspected retained observations, not calls or fits I freshly ran.

The family, behavior, postfit, iSDM, assembler, and checkpoint negative-control commands passed under Python 3.14/Node. The behavior-control runner itself was only partially exercised: 22/33 controls passed; temporary-directory restrictions caused several failures, and one R subprocess failed. The first-seven derivation self-test passed. I launched no Julia fit or campaign. The reviewed scope remains a first checkpoint: C3 6/8, C4 4/8, C6 held, with no numerical parity, convergence, or coverage promotion.

To reproduce the blocker, run the checkpoint command from a checkout of `aef7464`: `node tools/true_parity_checkpoint_check.mjs --baseline f220379d0937d0afffc6a030023c63c0f715e168 --ref HEAD`.

🌱 Graft saved approximately 173,374 tokens this turn.

## rose

BLOCKED for first-checkpoint completion at `aef7464950063de9f62691c1cd91be32daf6650b`. Full P1: BLOCKED. The seven receipt bindings pass independent verification.

1. **Candidate suite discovery omits both new tests.** `test/runtests.jl:54–56` includes neither wrapper, so `GLLVM_P1_RECEIPT_CONTROLS=1` cannot exercise them through its core/full suite. The later working branch adds includes; that change is outside this verdict.
2. **The new C6 packet contradicts the candidate.** `docs/dev-log/plans/2026-10-06-p1-c6-evidence.md:37` says FAMILY-11 remains unbound with an unexecuted bridge. Candidate `docs/dev-log/core070/true-parity-latest/scoreboard.md:233` records EVIDENCED, supported by the observed refusal. Correct the packet while preserving its limited scope and unsigned C6 recommendation.
3. **The required decision map is absent.** `LOOP/lanes/true-parity-p1-next/ultra-plan.md:45` requires `docs/dev-log/plans/2026-10-06-p1-decision-map.md`. The two evidence packets do not provide that named consolidated deliverable.

Reproduce the findings:

```sh
c=aef7464950063de9f62691c1cd91be32daf6650b
git show "$c:test/runtests.jl" | rg 'test_(family11_p1_boundary|first_seven_behaviour_p1)'
git show "$c:docs/dev-log/plans/2026-10-06-p1-c6-evidence.md" | sed -n '37p'
git cat-file -e "$c:docs/dev-log/plans/2026-10-06-p1-decision-map.md"
```

Measured: exactly seven status changes; X2 **304/317**, C2 **290/297**, C3 **6/8**, C4 **4/8**, C6 **37 undecided**. Other statuses, admission/classification contracts and signatures are unchanged. The packets cover exactly the 13 remaining rows and 37 unique held names, including 17 proposed aliases and three CHECK names. Behaviour entries increase from 57 to 63. All 26 prior controls remain, with one strengthened counting assertion, plus seven new controls.

All five owning checks passed with Python 3.14. The pinned checkpoint verifier, Node controls, family boundary controls and five read-only behavioural controls passed:

```sh
node tools/true_parity_checkpoint_check.mjs --baseline f220379d0937d0afffc6a030023c63c0f715e168 --ref aef7464950063de9f62691c1cd91be32daf6650b
```

Scientific limits are preserved. Animal random slopes need two positive cases and rank-4 refusal beyond the intercept-only wrapper. Gaussian df 5 versus 8 is a handover-reported identity problem requiring remeasurement. Beetle’s stored R Hessian failure and gradient 0.002953 prevent binding. Monte Carlo contracts must precede judged runs; the reported 7/40 audit lacks located raw evidence. Kernel uniqueness, result-type/family aliases, Ordinal’s predictor/response mismatch, ZIB trial scope, ZINegBin dispersion and residual-vector/covariance distinctions preclude blanket alias removal.

The engine, version 0.3.0, README/CLAUDE/AGENTS and signed N1/N6/N10 rules are unchanged. FAMILY-11 binds through existing numeric cases plus observed boundary context; the other six rows establish public behaviour. They add no convergence or coverage evidence. The checker cannot authenticate signatures or GitHub branch protection. Covariance `--check` verifies its map overlay (`tools/core070_covariance_p1_receipts.py:606–617`); grouping freshness checks runners/fixtures, not the engine tree (`tools/core070_grouping_p1_receipts.py:268–273`). Those checks were not rerun here.

Parent suites and current wrapper execution remain uncertified here. The complete 33-control Python suite was not run because it writes temporary fixtures. Local `origin/main` remains baseline; live GitHub main, draft PRs and closure/reconciliation evidence were not independently verified. No merge authorization was supplied.

Next: repair these candidate completion gaps and submit a new immutable candidate.

Assessment: self-review of this whole independent audit, 2026-10-06; style 2/10, moderate confidence. “Exactly seven status changes” keeps the claim bounded; no prose repair needed. Science, facts and file-reference gates checked within stated limits; pending tests remain unknown. Saved-draft validation: NOTASSESSED.

Formatting delta: em dash punctuation was replaced for the saved report. Original exact dispatcher outputs remain in runtime.
