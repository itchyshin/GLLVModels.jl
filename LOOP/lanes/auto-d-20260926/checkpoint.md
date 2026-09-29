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
