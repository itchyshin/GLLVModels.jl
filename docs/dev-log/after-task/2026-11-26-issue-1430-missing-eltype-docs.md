# After Task: document widening `Y` before `missing` assignment (refs #1430)

## Goal

Document that `Y[i,j] = missing` on a `Matrix{Float64}` is a Base `MethodError`,
not a GLLVModels.jl bug; show widening (`Matrix{Union{Missing,T}}(Y)` or
`allowmissing` via Missings.jl) before the #1107 mask-from-missing workflow.

## Implemented

- `docs/src/response-families.md` — Gaussian `Normal()` section: prose + runnable
  snippet (seed 777, widen then fit).
- `docs/src/api.md` — one-sentence cross-link on eltype requirement.
- `test/test_issue_1430_docs.jl` — doc wording pins + executes the documented fit path.

No `src/` edits.

## Tests

```text
julia --project=. --startup-file=no -e 'include("test/test_issue_1430_docs.jl")'
Test Summary: issue 1430 missing-response docs (#1107 workflow) | 6 passed, 0 failed, 0 errored, 20.3s
```

Before doc change: `occursin("allowmissing", text)` and the MethodError prose would fail.

## R-parity / engine

N/A — documentation only.

## Rose verdict

PASS — docs match Julia 1.10 behaviour; snippet runs; no capability claim change.