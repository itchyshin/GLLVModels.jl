# Packet 1c: temporal port scope questions (2026-09-27)

**SIGNED 2026-09-27 by Shinichi, in chat: "Packet 1c and 1d: accept all as recommended." Vault decision D-298.**

From the temporal port spec (draft PR #535, head 5da58438f, reviewed and revised). Engineering questions the spec settles itself (profile bound NA as `missing`, StableRNG bootstrap, integer AR1 exponents, error type, fit type, receipt names from `opt$par`) are not asked.

| # | Question | Recommendation | Reply to paste |
|---|---|---|---|
| 1 | R tests temporal combined with kernel, phylo and animal sources (18 test blocks, 99 expectations). Twin them in P1? | Defer: build the temporal source alone first (slices 1 and 2), then decide per pair. Kernel is the cheapest (about 2 days); phylo and animal about 3 to 4 days each. The spatial pair stays outside P1 (D-295). | "1c-1: defer the cross-source pairs; decide after slice 2." |
| 2 | Temporal combined with ordinary `unit` and `unit_obs` terms (14 blocks, slice 2). Inside P1? | Yes, as slice 2 (3 to 4 days); it is ordinary composition R supports and users will expect. | "1c-2: unit/unit_obs composition inside P1 as slice 2." |
| 3 | Formula admission. Slice 1 uses a separate door (`fit_temporal_gllvm`) with no edit to `formula.jl`; slice 2 needs a one-line pre-pass hook in the long-data `gllvm()` method (a separate method from the Normal branch, but inside the grammar lane's file). | Approve slice 1 as specified; ask the grammar lane (Gaussian intercepts, #519) for the hook when slice 2 starts, not now. | "1c-3: slice 1 door approved; ask the grammar lane for the hook at slice 2." |
| 4 | R's wide `traits()` input route for temporal. | Fence it; the long form is R's canonical temporal input and all its helpers use it. | "1c-4: fence the wide route." |
| 5 | Reach temporal fits through the gllvmTMB `engine = "julia"` bridge in P1? | No; a later gllvmTMB PR after the Julia receipts exist (as for iSDM). | "1c-5: no bridge route in P1." |
| 6 | Three Julia docs contradict R at P1 (`docs/src/gllvmtmb-parity.md:359`, `r/README_bridge.md:144`, `ROADMAP.md:83-87`). | Fix them in the implementing PR. | "1c-6: fix the stale docs in the build PR." |

One-line reply: "Packet 1c: accept 1 to 6 as recommended."

Estimate: build 9 to 13 days (slice 1 6 to 9 including the tmbprofile port, slice 2 3 to 4), tests and receipts 4 to 5 days.
