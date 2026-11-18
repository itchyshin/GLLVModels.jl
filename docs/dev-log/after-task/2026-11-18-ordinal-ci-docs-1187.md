# After-task: per-trait ordinal CI docs (#1187)

**Claim.** The capability pages now say native `confint` returns Wald and
bootstrap intervals for a per-trait ordinal fit (`beta[t]`, `Lambda[i,k]`,
`tau[t,c]`), and that a profile interval can return an open end. They also
say `bridge_fit` still refuses those intervals. Grouped Tweedie stays
unavailable.

**Files.** `docs/src/gllvmtmb-parity.md`, `docs/src/roadmap.md`,
`docs/src/confidence-intervals.md`, two stale comments in `src/bridge.jl`,
and `test/test_ordinal_ci_docs_1187.jl` (wired in `test/runtests.jl`).

**Tests.** That file on Julia 1.10.13, 1 thread. Before: 0 passed, 7 failed.
After: 7 passed, 0 failed. Full `Pkg.test` was not run. The bridge refusal
itself was not changed.

**Rose verdict:** PASS WITH NOTES — wording only; the bridge section that
says per-trait ordinal CI endpoints remain unavailable through the bridge
is unchanged.
