# POST-COEF-EMPTY receipt bookkeeping (2026-11-01)

Scope: documentation and guard only. No fit run. No row status, tolerance, scoreboard or C2 count change.

- `tools/core070_postfit_p1_receipts.py`: `julia_surface` citation `src/postfit.jl:633` corrected to `:652` (at 681c4c3ca, line 652 is `StatsAPI.coef(fit::GllvmFit)`; 633 sits inside the `nobs` code). `numeric_note` now opens with "DIAGNOSTIC, NON-BINDING" (synthetic R object, 0 = 0 cannot discriminate, D-319 N2).
- Receipt JSON regenerated with `--apply-integer-equality` (idempotent). Its sha256 changes from 72fdb2739730... to 2f108b0a6102...; `docs/dev-log/audits/2026-10-19-p1-thirteen-row-source-audit.md:38` still pins the old bytes and is left untouched as a historical record.
- `verdict: PASS`, `evidence_kind: numeric_r_vs_julia`, `harness_pass: true` are NOT changed (packet step 5b, owner's call, checker effect untested). The receipt file therefore still reads PASS; only the note and the scoreboard say non-binding.
- New `tools/test_postcoef_receipt_guard.py` (7 checks, 6 negative controls). It has no check on `verdict`.
- Contract on origin/main (3c94a76f7) still says the R side is a synthetic mock; the "real fitted R object" clause is not on main. The label above therefore rests on the 0 = 0 fact, not on the contract.
