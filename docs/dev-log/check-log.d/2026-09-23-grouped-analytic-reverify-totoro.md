# 2026-09-23: grouped-analytic-20260920 Totoro reverify (post-#449 tip)

| Field | Value |
|---|---|
| Tip verified | `8d58a0c94` (`origin/main` at run start; #449 `f36def049` ancestor; #450 check-log merge) |
| Host | Totoro (`totoro.biology.ualberta.ca`; ControlMaster; ≤4 Julia threads) |
| Checkout | `/home/snakagaw/GLLVModels.jl-grouped-analytic-reverify-20260923` on `main` |
| Command | `node ~/.cursor/skills/unlazy/scripts/gate-check.mjs --root . --cwd . --reverify --approve --scope grouped-analytic-20260920 --jobs 1 --timeout 7200` |
| Wall | ~07:02–13:19 MDT (~6.3 h; three serial `Pkg.test` gates) |
| Result | **NOT ALL MET**: `UNMET: 13` (met: 6, abandoned: 4, reran: 19) |
| Latte S4 identity | Already banked PASS on tip via #449 (26/26); **no duplicate S4 run** |
| Defaults | Latte `diag_precision_kernel` left OFF; NM default unchanged |

## Pass / fail table (live gates)

| Gate | Verdict | Note |
|---|---|---|
| S8:GA.1 | FAIL | EXPECT matcher / section-sum wording vs measured wall |
| S8:GB.1 | FAIL | CHECK invokes `manual` (exit 127); ledger CHECK broken |
| S8:GB.2 | **PASS** | analytic vs FD grad; worst rel 9.17e-8 ≤ 1e-6 |
| S8:GB.3 | FAIL | identity gate printed `GATE GB.3 FAIL` (in-tree proxy) |
| S8:GB.4 | **PASS** | warm-start unconfined; fixture D |
| S8:GB.5 | **PASS** | counts / walls reported |
| S8:GB.6 | FAIL | `Pkg.test` **timed out** 7200s (SIGKILL) |
| S9:G9.1 | FAIL | same identity failure class as GB.3 |
| S9:G9.2 | **PASS** | Hessian SE vs FD rtol 1e-4 |
| S9:G9.3 | FAIL | fallback contract EXPECT not matched |
| S9:G9.7 | **PASS** | final analytic gradient bit-identical |
| S9:G9.8 | FAIL | `Pkg.test` **timed out** 7200s |
| S9:G9.9 | FAIL | EXPECT matcher vs measurement-amendment prose |
| S9c:G9c.1 | **PASS** | holes 1–3 fd_agreement + identity |
| S9c:G9c.2 | FAIL | mixed path: forced failures = 0 |
| S9c:G9c.3 | FAIL | CHECK prose starts with `a` → `/bin/sh: a: not found` |
| S9c:G9c.4 | FAIL | unknown `--gate coverag` errors correctly but EXPECT mismatch |
| S9c:G9c.5 | FAIL | EXPECT wants `GATE GB.2 PASS, GATE GB.3 PASS…`; GB.3 still red |
| S9c:G9c.6 | FAIL | `Pkg.test` **timed out** 7200s |

## Receipts

- `docs/dev-log/simulation-artifacts/2026-09-23-grouped-analytic-reverify/pass-fail.txt`
- `…/totoro-reverify.log`
- `…/watch.log`

## Triage (2026-09-23, post-bank)

**Verdict: REGRESSION** (identity). Not tip-stale for engine; not timeout-only.

| Class | Gates |
|---|---|
| Real tip regression | **GB.3**, **G9.1** (`GATE GB.3 FAIL`); **G9.3** / **G9c.2** (`forced=0` / `calls=0` — same kit, may share root) |
| TIMEOUT starvation | GB.6, G9.8, G9c.6 (`Pkg.test` SIGKILL at 7200s) |
| EXPECT drift / broken CHECK | GA.1, GB.1, G9.9, G9c.3, G9c.4, G9c.5 |
| Wrong tip vs `36479c28`+ | **No** — `8d58a0c94..origin/main` has **0** `src/`/`test/` commits |

Next: [#458](https://github.com/itchyshin/GLLVModels.jl/issues/458) on GB.3 / G9.1 (reproducer: `test/test_grouped_analytic_grad.jl --gate identity`). Do not demote Latte/NM. Do not re-run full kit until identity is green.

## Rose fence

Does **not** close the S8/S9 ledger. Does **not** flip Latte defaults ON.
Timeouts on full-suite gates need a longer per-gate budget or a focused suite command before claiming green.
