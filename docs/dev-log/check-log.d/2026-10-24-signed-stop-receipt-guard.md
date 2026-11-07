# Signed-stop receipt regeneration guard

- Based on current `origin/main` commit `0a4d49eaaad0bd595e0a9e677a18c62a3ddebaf2` in `codex/true-parity-p1-current-main-20261024`.
- `write_receipts.py` now requires the three rows held by the signed 2026-10-21 optimizer rule to remain present, nonbinding, and `FAIL` before it persists candidate raw receipt files. The same guard runs during `--check`.
- Regression coverage includes attempted promotion of all three rows, a missing held row, and the `--apply` write-order control. The real receipts remain `FAIL`; `LLt` remains `1e-4`, and the iSDM prediction tolerances remain `1e-6`.
- `PATH=/Users/z3437171/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin:$PATH PYTHONDONTWRITEBYTECODE=1 python3 tools/test_true_parity_assemble.py`: 535 passed, 0 failed, including the Node-backed controls; final output `ALL PASS`.
- `PYTHONDONTWRITEBYTECODE=1 python3 tools/true_parity/campaign/write_receipts.py --check`: `CAMPAIGN_RECEIPTS_OK`; urbanisation remains summary-only because its raw data are not in the repository.
- `PYTHONDONTWRITEBYTECODE=1 python3 tools/true_parity_assemble.py --check`: `ASSEMBLE_OK 317 rows current`.
- `git diff --check` and Python byte-compilation passed. Julia tests, JET, Allocs, Aqua, and R fits were not run because this is a Python receipt-tool change with no model or likelihood edits.
- The after-task structure checker passed. With bundled Node on `PATH`, repository-wide ledger re-verification failed to parse the existing `.unlazy/totoro-t4-p6-grid/GATES.md` because a gate lacks the required ID and colon. That unrelated ledger was left untouched; task closure remains open.
- No scoreboard row, tolerance, source data, merge, push, or public parity claim changed. The parity goal remains open.
- The central `docs/dev-log/check-log.md` was not edited because an active non-overlapping extractor lease includes that shared file; this fragment records the checkpoint without touching the leased path.
