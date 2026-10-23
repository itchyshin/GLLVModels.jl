GOAL: auto-d lane on-call (merge trains) + NB re-run. STATE 2026-09-29 10:45Z.
DONE: NB re-run finished (0.934 default rule, 4,794/4,800 datasets, 1,888 core-h); docstring + design/74 updated
  locally (b840533a0). Draft PRs: #608 (NB2 smoke fixture), #610 (Student-t restart + fixtures). #529 test premise
  fixed (09701e985, attributed to #601). #551 at 29bf83a31 pushed; local pre-merge of #529 at 0c5e1f100 (not pushed).
HELD (push only on the train lane's ping): #518 (39 local commits incl. main merge + NB result), #551 refresh, #610 main merge.
OWED: rerun NB tasks 388,389 (`sbatch --array=388,389 --exclude=c63 ...run_nb_rerun_nibi.sh`, Shinichi) -> 6 datasets
  in cell (120,10,3); R side: restore NB figure in gllvmTMB #1324 docs + apply r-pdhess-fix.md once the Codex CRAN
  lease (codex:cran-071-20260927) is released.
OPEN GATES: merges (train lane / Shinichi); DRAC submissions (Shinichi).
RESUME: read this file; `gh pr view 606 --json state`.

DECISION 2026-09-29 (Shinichi): frozen-R smoke truncated-NB2 cell (test_truncated_nbinom2_parity.jl:80) records R r_gradient_max instead of gating, same as NB2 (#608). Relayed to the truncated-NB2 lane (owner of the file).

PREFERENCE 2026-09-29 (Shinichi): "try to use more sonnets". Route mechanical work (check-log refreshes, test runs, CI/cluster watching, harvests, sweeps) to Sonnet 5 sub-agents (Agent model: "sonnet"); keep judgement calls on the main session.

OWED (2026-09-29): gllvmTMB #1324 R/gllvmTMB.R ("Choosing d automatically") still says NB recovery "has not yet been measured"; update to 0.93 and regenerate man/gllvmTMB.Rd AFTER the 0.7.1 CRAN release lane releases R/gllvmTMB.R, man/gllvmTMB.Rd, inst/COPYRIGHTS (exclusive to it per Shinichi). #1324 adds to both files, so expect a merge conflict there after the release.

UPDATE 2026-09-29 (Shinichi via the true-parity lane): gllvmTMB #1324 CLOSED so the 0.7.1 CRAN resubmission goes first. Branch claude/lane-auto-d-r-20260926 kept at 215f9544f (fully pushed, nothing uncommitted): penalised-Hessian fix, NB figures, the d = "auto" section in R/gllvmTMB.R (+135) and man/gllvmTMB.Rd (+23). Hands off NEWS.md, docs/dev-log/check-log.md, R/gllvmTMB.R, man/gllvmTMB.Rd, inst/COPYRIGHTS until the release lane releases them. AFTER the release: merge main into that branch, fix the NB sentence in R/gllvmTMB.R, re-run the select_lv tests, and open a new PR (Shinichi to confirm).
