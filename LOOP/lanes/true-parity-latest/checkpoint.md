# Checkpoint (overwritten every arc)

- DONE: plan approved 2026-09-27; lane kit; rehydration 2026-09-27 (main 1385b0490; #514, #515, #521 review, D1, A0 all OWED; 3 dead leases reaped). D1 DONE: `packet-1.md` signed as recommended (D-295). A1a (iSDM spec) is now unblocked.
- IN PROGRESS: #514 pushed fast-forward to 245889dc4 (S1 kept, decrement-only convergence, S2 latch gated; 0 of 20 healthy fits worse, total time 43 s vs 45 s main; tests 41+364/18 on 1.10, 41+366/18 on 1.13); CI running; r514 (fresh Fable merge-gate review) running. b515 (draft PR for #515) and r521 (Fable review of #521) running.
- NEXT: when b514/b515 report, run a fresh Fable review of each; merge on green + non-blocking review (Shinichi's word, merge_train.sh). Then A0 (additive re-pin) via fix-build-review workflow.
- OPEN GATE: none for now. Totoro is DOWN (Shinichi, 2026-09-27) until later on 2026-09-28: Shinichi: "use DRAC or kohaku" instead. Route A3 re-measurements to kohaku (<=8 vCPU, CPU work ok for this) or DRAC (sbatch arrays, --time/--account, never the login node; each DRAC campaign gets a time estimate first, and a pre-run plus his ack if over 3 h). (Packet 1 + boundary SIGNED 2026-09-27, D-295: temporal and phylo latent inside; column grammar and spatial outside.)
- WHERE TRUTH LIVES: branch claude/lane-true-parity-latest in ~/local-scratch/lanes/GLLVM.jl-true-parity-latest; LOOP/lanes/true-parity-latest/.
- RESUME: read GOAL.md -> checkpoint.md -> HANDOVER.md -> ultra-plan.md in LOOP/lanes/true-parity-latest/, then continue from NEXT.
