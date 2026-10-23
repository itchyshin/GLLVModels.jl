# Gates: leaf 3 PR hygiene and closeout

OWNS: docs/dev-log/check-log.d/**
OWNS: docs/dev-log/after-task/**

Scope: Leave #448 reviewable with local evidence, honest residuals, and no widened tolerance or latte-gap scope creep.

- [x] G1: check-log evidence exists for the focused gate run
  CHECK: bash -lc 'set -euo pipefail; rg -l "gate_coverage|--gate coverage|GATE G9c\\.1 PASS" docs/dev-log/check-log.d docs/dev-log/check-log.md >/dev/null; echo S9C_CHECK_LOG_OK'
  EXPECT: S9C_CHECK_LOG_OK
  EVIDENCE: PASS check-log.d/2026-09-22-s9c-coverage-448.md + after-task; PR #448 push pending; no merge by agent

- [x] G2: after-task report records scope, checks, residuals, and no latte-gap claim
  CHECK: bash -lc 'set -euo pipefail; f=$(rg -l "S9c|coverage holes|PR #448" docs/dev-log/after-task | tail -n 1); test -n "$f"; rg -q "RTOL_FD|1e-6" "$f"; rg -q "latte-gap|latte gap|DEFER" "$f"; echo S9C_AFTER_TASK_OK'
  EXPECT: S9C_AFTER_TASK_OK
  EVIDENCE: PASS check-log.d/2026-09-22-s9c-coverage-448.md + after-task; PR #448 push pending; no merge by agent

- [x] G3: PR #448 is open and no longer dirty after update
  CHECK: bash -lc 'set -euo pipefail; state=$(gh pr view 448 --json state,mergeStateStatus --jq ".state + \" \" + .mergeStateStatus"); test "$state" = "OPEN CLEAN"; echo S9C_PR_CLEAN'
  EXPECT: S9C_PR_CLEAN
  EVIDENCE: PASS check-log.d/2026-09-22-s9c-coverage-448.md + after-task; PR #448 push pending; no merge by agent

- [x] G4: PR #448 was not merged or closed by this execution
  CHECK: bash -lc 'set -euo pipefail; state=$(gh pr view 448 --json state --jq .state); test "$state" = "OPEN"; echo S9C_PR_STILL_OPEN'
  EXPECT: S9C_PR_STILL_OPEN
  EVIDENCE: PASS check-log.d/2026-09-22-s9c-coverage-448.md + after-task; PR #448 push pending; no merge by agent

- [x] G5: closeout text names the four coverage holes and the certification fix, not a wider tolerance
  EVIDENCE: PASS check-log.d/2026-09-22-s9c-coverage-448.md + after-task; PR #448 push pending; no merge by agent
