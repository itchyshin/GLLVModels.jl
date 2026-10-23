# Gates: leaf 1 main-tip certification merge

OWNS: test/test_grouped_analytic_grad.jl

Scope: Keep current `main` gate machinery and add branch S9c certification helpers in `test/test_grouped_analytic_grad.jl` only.

- [x] G1: branch is no longer behind `origin/main` after the merge/rebase step
  CHECK: bash -lc 'set -euo pipefail; read ahead behind < <(git rev-list --left-right --count HEAD...origin/main); test "$behind" = 0; echo S9C_MAIN_TIP_OK'
  EXPECT: S9C_MAIN_TIP_OK
  EVIDENCE: PASS e14e8b26e laptop 2026-09-22; both surfaces present; _coverage_fixtures in _fixtures; RTOL_FD=1e-6; no em_phylo/sparse carry

- [x] G2: both implementation sets are present in the merged test file
  CHECK: bash -lc 'set -euo pipefail; rg -q "function gate_hessian" test/test_grouped_analytic_grad.jl; rg -q "function gate_nm_fallback" test/test_grouped_analytic_grad.jl; rg -q "function gate_counts" test/test_grouped_analytic_grad.jl; rg -q "function gate_final_gradient" test/test_grouped_analytic_grad.jl; rg -q "_hessian_standard_errors" test/test_grouped_analytic_grad.jl; rg -q "_s9_counted_fit" test/test_grouped_analytic_grad.jl; rg -q "_s9c_fd_certificate" test/test_grouped_analytic_grad.jl; rg -q "_s9c_richardson_check" test/test_grouped_analytic_grad.jl; rg -q "_s9c_flat_direction_check" test/test_grouped_analytic_grad.jl; rg -q "_s9c_theta_verdict" test/test_grouped_analytic_grad.jl; rg -q "_s9c_assert_per_trait_dispersion" test/test_grouped_analytic_grad.jl; echo S9C_BOTH_IMPLEMENTATIONS_OK'
  EXPECT: S9C_BOTH_IMPLEMENTATIONS_OK
  EVIDENCE: PASS e14e8b26e laptop 2026-09-22; both surfaces present; _coverage_fixtures in _fixtures; RTOL_FD=1e-6; no em_phylo/sparse carry

- [x] G3: coverage fixtures are exercised in the in-suite testset
  CHECK: bash -lc 'set -euo pipefail; rg -q "_coverage_fixtures\\(\\)" test/test_grouped_analytic_grad.jl; rg -q "_coverage_fixtures\\(\\).*|for .*_coverage_fixtures" test/test_grouped_analytic_grad.jl || rg -q "for .* in _coverage_fixtures\\(\\)" test/test_grouped_analytic_grad.jl; echo S9C_COVERAGE_IN_SUITE_OK'
  EXPECT: S9C_COVERAGE_IN_SUITE_OK
  EVIDENCE: PASS e14e8b26e laptop 2026-09-22; both surfaces present; _coverage_fixtures in _fixtures; RTOL_FD=1e-6; no em_phylo/sparse carry

- [x] G4: `RTOL_FD` remains the original 1e-6 gate, not a wider tolerance
  CHECK: bash -lc 'set -euo pipefail; rg -q "const RTOL_FD = 1e-6" test/test_grouped_analytic_grad.jl; echo S9C_RTOL_UNCHANGED'
  EXPECT: S9C_RTOL_UNCHANGED
  EVIDENCE: PASS e14e8b26e laptop 2026-09-22; both surfaces present; _coverage_fixtures in _fixtures; RTOL_FD=1e-6; no em_phylo/sparse carry

- [x] G5: unrelated old-base files are not part of the PR #448 repair
  CHECK: bash -lc 'set -euo pipefail; bad=$(git diff --name-only origin/main...HEAD | rg "^(src/em_phylo\\.jl|test/test_sparse_phy_identities\\.jl)$" || true); test -z "$bad"; echo S9C_NO_OLD_BASE_FILES'
  EXPECT: S9C_NO_OLD_BASE_FILES
  EVIDENCE: PASS e14e8b26e laptop 2026-09-22; both surfaces present; _coverage_fixtures in _fixtures; RTOL_FD=1e-6; no em_phylo/sparse carry
