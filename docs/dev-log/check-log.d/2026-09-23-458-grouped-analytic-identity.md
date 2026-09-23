# 2026-09-23: #458 grouped-analytic identity (GB.3 / G9.1)

| Gate | Evidence | Verdict |
| --- | --- | --- |
| Name FAIL fixture | Pre-fix Julia 1.12 x64 (`julia +1.12~x64`): `--gate identity` → `FAIL poisson_percoord` then `GATE GB.3 FAIL`. Loglik rel `7.3e-14`; max beta rel `2.116e-08` (> `1e-8`); over-rtol theta coords all FLAT under `_s9c_flat_direction_check`. aarch64 1.12 / 1.10 stayed green (beta rel under bare rtol). Matches Totoro tip `8d58a0c94` (Julia 1.12.6 Linux x64) truncated bank ending `PASS dep_term \| GATE GB.3 FAIL`. | PASS (named) |
| Fix | `test/test_grouped_analytic_grad.jl`: drop hard `dbeta <= RTOL_IDENTITY` from `gate_identity` / `gate_mixed` so `_s9c_theta_verdict` flat adjudication is authority for beta⊂theta; `_s9c_theta_verdict` returns only non-flat offenders. No `src/` change; no Latte/NM flip. | PASS |
| Re-verify identity | `JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia +1.12~x64 --project=. test/test_grouped_analytic_grad.jl --gate identity` → exit 0, `GATE GB.3 PASS` (incl. `PASS poisson_percoord`). Same on aarch64 `+1.12`. Compound GB.3 CHECK (analytic identity && laplace `--gate identity`) → exit 0 (`GATE GB.3 PASS` + `GATE G7b.1 PASS`). | PASS |

Rose fence: not an engine likelihood bug (dll ~1e-13; flats prove same point). Not Latte S4. G9.3 / G9c.2 (`calls=0`) remain separate.
