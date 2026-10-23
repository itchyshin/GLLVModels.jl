# P1 F-gate ticks check, 2026-10-21

Read only check. No test was re-run. Julia was not started. GATES.md, source, and receipts were not edited. No box was unticked.

Worktree: `/Users/z3437171/.codex/worktrees/p1-isdm-20261019/GLLVM.jl`. HEAD: `5cfc67542`.

Question per gate: does the note cite an existing test and a pass count?

## Result

| Gate | Result |
| --- | --- |
| F-ISDM-1FO-PREDICT | CITED_PASS |
| F-ORDINAL-LOGIT-1FO | CITED_PASS |
| F-PHYLO-LATENT-1FO | CITED_PASS |
| F-ZI-1FO | CITED_PASS |

All four notes cite a test file that exists in the worktree and a pass count. The counts are the note's own record of a local run. This check did not confirm them by running anything.

## Per gate

### F-ISDM-1FO-PREDICT

- GATES.md line 1167: ticked, and the line links `docs/dev-log/2026-10-20-p1-f-isdm-1fo-predict.md`.
- Test cited: `test/parity/isdm_cases.jl`. The file exists, and it holds the testsets `P1-ISDM-ADMISSION-20` and `iSDM P1 paired twins`. The fixture `test/fixtures/isdm/r_values_p1.toml` exists.
- Pass count cited: 41 of 41 (`P1-ISDM-ADMISSION-20`) and 219 of 219 (`iSDM P1 paired twins`), exit code 0.
- Gap in the note: the first paragraph says "The GATES.md checkbox stays open". The box is ticked. That sentence is stale and contradicts the tick. The 219 count is a file total across four paired cases, and the gate names only the `predict` cell. The note says so itself and backs the cell with a gap table.

### F-ORDINAL-LOGIT-1FO

- GATES.md line 1171: ticked. The line carries no link to a note.
- Test cited: `test/test_ordinal_logit_twin.jl`, testset `ordinal_logit() twin: gllvmTMB P1 (9539352f6) fixed dataset`. The file and that testset exist. The fixture `test/fixtures/ordinal_logit_p1.toml` exists.
- Pass count cited: 19 of 19, plus 7 of 7 and 3 of 3 for the two API testsets, exit code 0.
- Gap in the note: the header still reads `Verdict: BLOCKED_ON_CONTRACT`, and the Contract and Measurement sections say no local fit was run and the checkbox stays open. A later section, "Existing check" and "Run", gives `Verdict: PASS` and says the box is ticked. The note holds two verdicts. The pass count is there, but a reader of the first screen sees BLOCKED.

### F-PHYLO-LATENT-1FO

- GATES.md line 1169: ticked. The line carries no link to a note.
- Test cited: `test/test_phylo_latent_paired_p1.jl`. The file exists. It holds the A14 testsets (tree, dense, routes agree, `Ainv`) and `A15 COV-PHYLO-LATENT-RSZ`. The fixture `docs/dev-log/core070/phylo-latent-p1/a14-fixture.json` exists.
- Pass count cited: A14 78 of 78 and A15 35 of 35, both exit code 0.
- Gap in the note: the summary label `F-PHYLO-LATENT-1FO A14 written replay` is not a testset name in the file. The 78 count came from a process that ran the A14 testsets and printed under that label. The A15 count came from a process that ran only the existing A15 testset, and the note says so. The count is traceable to the file, but it is not a line a plain `Pkg.test` would print.
- A15 is a recorded stationarity gap case (stored receipt not converged). The note records this openly and gives the looser bounds.

### F-ZI-1FO

- GATES.md line 1172: ticked. The line carries no link to a note.
- Test cited: `test/test_zi_twin.jl`. The file exists. The testset `zi_* twins: gllvmTMB P1 (9539352f6) fixed datasets` exists at line 141. The fixture `test/fixtures/zi_p1.toml` exists.
- Pass count cited: 37 of 37 for the twins testset. Also 22 of 22, 5 of 5, and 16 of 16 for the other three testsets, exit code 0.
- Gap in the note: the sections "What the written contract says", "Why this stopped", "Routes left in place", and "Left undone" describe an earlier stop with no run and no pass. The later "Existing test run" section supersedes them. The header says `Verdict: PASS`, so the top is consistent, but the middle still says no Julia process was started.

## Cross-cutting findings

1. The tick edits to GATES.md are uncommitted. `git status` shows GATES.md as modified, and the diff flips five boxes (these four plus F-TEMPORAL-1FO). The five new notes are untracked.
2. Only F-ISDM-1FO-PREDICT and F-TEMPORAL-1FO carry a note link on their GATES.md line. The other three ticks have no pointer from the gate line to the note.
3. The shared lines `CHECK: none yet` and `EVIDENCE: pending` still sit under all five F gates. The ticks do not change them. The notes agree that C2 is not claimed MET and no scoreboard row was admitted.
4. Ordinal logit and ZI notes both began as "blocked on contract" stops and were later extended with a run. A pass count exists in each. The earlier wording was not trimmed.

## Decision needed from Shinichi

None blocks the tick on the CITED_PASS question. Optional cleanup, as a separate slice: fix the stale "checkbox stays open" sentence in the ISDM note, and the stale BLOCKED_ON_CONTRACT header in the ordinal note.
