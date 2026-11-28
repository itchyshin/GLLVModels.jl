# Psi prior-table bridge checkpoint

- Branch: `codex/psi-sd-prior-live-main-20261124`.
- Native Psi source pin: `e6b24e3dd17618bb3fb08253f9e6321b0dea5d2b`.
- Shared-table integration pin: `a2d5267a8fefed301cfd14615acf6e9b53aec9c3`, based on live main `15ca983e543a0391c338c47a00a606374b661e12`.
- Validation worktree: `/Users/z3437171/.codex/worktrees/psi-sd-table-validate-20261124/GLLVModels.jl`, branch `codex/psi-sd-prior-table-validate-20261124`, HEAD `a2d5267a`.
- The shared-table integration worktree remains separate and unchanged. The validation worktree was created from its frozen commit and overlays only Psi-owned files.
- Psi-owned uncommitted files: `src/source_fit.jl`, `src/psi_sd_prior.jl`, `test/auto/test_psi_sd_prior.jl`, and the two frozen fixture files under `test/fixtures/psi_sd_prior/`.
- `git diff --check` passed after the bridge edits. No Julia or R tests were run.
- Totoro process check at 2026-11-24 03:37 UTC: DRModels Julia `Pkg.test()` live, PID 2698092, elapsed 1h43m. Local R process 21585 has an active Julia bridge test; local package check PID 77917 is also active. Do not launch another Julia or R test until owners report the slots clear.
- Next: finish static review of the source-fit adapter; once the live jobs clear, run the focused source-table and Psi tests, the frozen five-point table-derivative script, then the full Julia suite on the exact integrated candidate. Run the current-main gllvmTMB public-table probe and full R suite on its exact integrated pin.
- Remaining acceptance gates: both public-table probes, table-routed derivatives and mutation controls, both full package suites, and an after-task audit. Keep the two-kernel coevolution route documented as unsupported until its required unique source model is independently validated.
