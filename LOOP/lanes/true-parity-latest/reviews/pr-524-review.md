# Review of PR #524 (head 1172d8096), 2026-09-27: BLOCKING (two small fixes)

Additive contract holds: P0 constant unchanged, DEFAULT_R_REF still P0 with the switch unset, only parity_ledger.py and the test import parity_oracle; ~116 other tools hardcode their own P0 literal. P1 SHA verified (9539352f66f2..., DESCRIPTION 0.7.1). Tests 7/7, --self-test OK.

1. High, blocking. Unknown or mistyped GLLVM_PARITY_PIN (p1, P2, "P1 ") silently falls back to P0, and a test enshrines it. Fix: normalise and fail loudly naming the variable.
2. High, blocking. The P1 job runs Julia-only files with no R or gllvmTMB checkout, so it cannot run an R-backed twin; and discovery passes on zero files, so a typo in the tag or path keeps it green forever; the grep also picks up non-.jl files. Fix: a committed sentinel tagged test, fail on zero files, --include='*.jl', and state that R-backed P1 twins go through test-parity once re-pinned.
3. Medium, coordination. #523's C0 regex wants `DEFAULT_R_REF = P1_GLLVMTMB_ORACLE` literally, which would delete the switch. DECISION (lane, 2026-09-27): #524 uses `_DEFAULT_PIN = "P0"`; #523's C0 checks for `_DEFAULT_PIN = "P1"`.
4. Medium. Exporting GLLVM_PARITY_PIN=P1 breaks the three P0 tests and --self-test without naming the variable; help text formats the P0 constant, not DEFAULT_REF.
5. Minor. P1 job should carry the 1.10 + stable matrix; label says 0.7.1-candidate.
