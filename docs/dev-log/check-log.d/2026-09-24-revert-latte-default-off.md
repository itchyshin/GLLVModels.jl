# 2026-09-24: Latte default OFF restore (revert #453 KEEP)

| Field | Value |
|---|---|
| Branch | `cursor/revert-latte-default-off-20260924` |
| Base | `origin/main` @ `1703c54b` (#453 default ON) |
| Change | `diag_precision_kernel::Bool=false` at 4 sites in `src/grouped_nongaussian_fit.jl` |
| Docstring | default `false`; Pass `true` to opt in |
| Opt-in | `diag_precision_kernel=true` unchanged |
| Evidence retained | #452 walls; `test/test_latte_kernel_identity.jl` OFF vs ON |
| Public claim | none |
| Owner | Shinichi paste GO: revert #453 to default false |
