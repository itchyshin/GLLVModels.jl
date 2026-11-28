# After-task: #1192 population predict documentation and error gate

**Goal:** Document which fit types support `predict(fit, design)` without responses,
and raise `ArgumentError` (naming supported types) on covariate fits that previously
surfaced a `MethodError`.

**Claim:** Docs and docstrings now distinguish population-level vs conditional
`predict`; unsupported covariate types refuse `predict(fit, X)` with a guided
`ArgumentError`.

**Files:** `src/population_predict_gate.jl`, `src/postfit.jl`, `src/families/rrr.jl`,
`docs/src/working-with-a-fit.md`, `test/test_predict_population_predict_gate.jl`,
`test/test_docs_snippets_734_738.jl`, `test/runtests.jl`, `docs/dev-log/check-log.md`.

**Tests (Julia 1.13.4, 2026-11-22):**

- `test/test_predict_population_predict_gate.jl`: 22 pass / 0 fail (~47s)
- `test/test_docs_snippets_734_738.jl`: 18 pass / 0 fail (~1s)

**R-parity:** Not exercised (docs/error-only slice).

**Rose verdict:** PASS WITH NOTES — population predict remains unimplemented on
several covariate families; PR recommends adding methods as a follow-up (issue option 2).