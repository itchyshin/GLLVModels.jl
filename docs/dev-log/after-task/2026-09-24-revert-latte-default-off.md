# After-task: restore Latte diag_precision_kernel default OFF (2026-09-24)

Active lenses: Shannon, Ada, Rose (perspectives). Spawned subagents: none.
Lane: `cursor/revert-latte-default-off-20260924`.

## Scope

Shinichi paste GO: revert #453 so `diag_precision_kernel` defaults are `false`
again. Opt-in `true` and #452 wall evidence stay intact. No default-ON reopen.

## Change

`src/grouped_nongaussian_fit.jl`: all four `diag_precision_kernel::Bool`
defaults set to `false`; public docstring says default `false` and
`Pass true to opt in`. S7b CHOLMOD pins in
`test/test_grouped_laplace_identity.jl` kept (comment wording only).

## Not claimed

No README or NEWS speed claim. #452 OFF vs ON walls remain the evidence
packet; this PR only restores the KEEP-default-OFF behaviour after the
#453 KEEP violation.

## Rose

Claim matches code: default OFF; opt-in path live; no public speed promotion.
