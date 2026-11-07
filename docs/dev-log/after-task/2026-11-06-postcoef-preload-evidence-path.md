# POST-COEF preload evidence path

The maintainer approved one additional POST-COEF evidence-only descendant path, `postcoef-empty-p1/oracle-verify-preload.log`. The generic historical batch artifact list remains unchanged.

The two new descendant controls failed before the rule edit and passed afterwards. They check that the POST-COEF path is accepted while a neighboring file and the same filename in the historical batch remain rejected. A draft-case test had a pre-existing fixture error: it replaced the live fixture's harness fields with a version lacking fit-health data. The test now carries those fields through the draft.

Checks on the local review checkout:

- `PYTHONPATH=tools python3 -m unittest test_postcoef_receipt_integration.EvidenceDescendantTests test_postcoef_historical_preservation`: 60 passed.
- `PYTHONPATH=tools python3 -m unittest test_core070_postcoef_live_ref test_postcoef_contract_consistency`: 44 passed.
- The draft-case error was reproduced using the unmodified parent receipt module before the fixture correction.

The source tree under `src/` is unchanged. No likelihood, user-facing API, benchmark or numerical parity threshold changed. No paired R–Julia fit was run. The full Julia package and core suites, JET and Aqua were not run for this local tools-only candidate; no full-suite result is claimed.

The trusted launcher still points to an earlier gate. Its binding and the objective and Hessian roundoff criteria remain unresolved. This local review candidate is unpushed and unmerged; no capture or row promotion has occurred.

Rose verdict: FAIL for closure. The focused controls do not establish a trusted live gate or complete package validation.
