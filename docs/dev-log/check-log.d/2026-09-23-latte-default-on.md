# 2026-09-23: Latte default-ON flip

| Field | Value |
|---|---|
| Branch | `cursor/latte-default-on-20260923` |
| Base evidence | #452 walls tip `4e976e259`; gate PASS (`glmm_200x5` 2.90×) |
| Change | `diag_precision_kernel::Bool=true` defaults in `src/grouped_nongaussian_fit.jl` |
| Opt-out | `diag_precision_kernel=false` still works |
| Public claim | none |
| Merge | **DRAFT only** — awaiting Shinichi G0; do not merge |
