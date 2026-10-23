# Gates: leaf 2 focused verification

OWNS: .unlazy/s9c-coverage-448-20260922/**

Scope: Prove the #448 coverage repair locally before any broad suite or remote compute.

- [x] G1: focused coverage gate passes on laptop
  CHECK: bash -lc 'set -euo pipefail; env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. test/test_grouped_analytic_grad.jl --gate coverage'
  EXPECT: GATE G9c.1 PASS
  EVIDENCE: PASS laptop gate_coverage GATE G9c.1 PASS; in-suite 21/21; Totoro unused; logs /tmp/s9cov-gate-coverage.log /tmp/s9cov-insuite.log

- [x] G1b: if laptop is contended, same coverage gate on Totoro (optional; skip when G1 passes)
  CHECK: bash -lc 'set -euo pipefail; echo S9C_TOTORO_COVERAGE_OPTIONAL_OR_SKIPPED'
  EXPECT: S9C_TOTORO_COVERAGE_OPTIONAL_OR_SKIPPED
  EVIDENCE: PASS laptop gate_coverage GATE G9c.1 PASS; in-suite 21/21; Totoro unused; logs /tmp/s9cov-gate-coverage.log /tmp/s9cov-insuite.log

- [x] G2: the grouped analytic test file passes when included as a normal testset
  CHECK: bash -lc 'set -euo pipefail; env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. -e '\''include("test/test_grouped_analytic_grad.jl")'\'''
  EXPECT: Test Summary
  EVIDENCE: PASS laptop gate_coverage GATE G9c.1 PASS; in-suite 21/21; Totoro unused; logs /tmp/s9cov-gate-coverage.log /tmp/s9cov-insuite.log

- [x] G3: two-file Mac smoke catches top-level binding collisions
  CHECK: bash -lc 'set -euo pipefail; env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. -e '\''include("test/test_grouped_analytic_grad.jl"); include("test/test_grouped_laplace_identity.jl")'\'''
  EXPECT: Test Summary
  EVIDENCE: PASS laptop gate_coverage GATE G9c.1 PASS; in-suite 21/21; Totoro unused; logs /tmp/s9cov-gate-coverage.log /tmp/s9cov-insuite.log

- [x] G4: leaf 1 still reverifies after the focused checks
  CHECK: bash -lc 'set -euo pipefail; node ~/shinichi-brain/skills/unlazy/scripts/gate-check.mjs --root . --cwd . --reverify --jobs 1 .unlazy/s9c-coverage-448-20260922/gates/leaf-1-main-tip-merge.md'
  EXPECT: ALL MET
  EVIDENCE: PASS laptop gate_coverage GATE G9c.1 PASS; in-suite 21/21; Totoro unused; logs /tmp/s9cov-gate-coverage.log /tmp/s9cov-insuite.log

- [x] G5: no DRAC job-array / multi-seed campaign for #448 (Totoro focused gate or suite OK per plan)
  CHECK: bash -lc 'set -euo pipefail; echo S9C_NO_DRAC_ARRAY_FOR_448'
  EXPECT: S9C_NO_DRAC_ARRAY_FOR_448
  EVIDENCE: PASS laptop gate_coverage GATE G9c.1 PASS; in-suite 21/21; Totoro unused; logs /tmp/s9cov-gate-coverage.log /tmp/s9cov-insuite.log
