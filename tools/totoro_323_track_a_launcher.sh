#!/usr/bin/env bash
# Totoro #323 Track A — shell entry (paste-gated DRAFT harness).
#
# Runbook: docs/dev-log/plans/2026-09-16-totoro-323-track-a-runbook-paste-gated.md
# Does not SSH; does not start oracle build or runparity.jl.
#
# Usage:
#   tools/totoro_323_track_a_launcher.sh --gllvm-root "$(pwd)"   # exits 2 without paste
#   GLLVM_TOTORO_PASTE='ack Totoro D-139 #323 Track A' \\
#     tools/totoro_323_track_a_launcher.sh --gllvm-root "$(pwd)"

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PASTE_EXACT='ack Totoro D-139 #323 Track A'

if [[ "${GLLVM_TOTORO_PASTE:-}" != "${PASTE_EXACT}" ]]; then
  echo "Totoro #323 Track A launcher: refusing without paste '${PASTE_EXACT}' in GLLVM_TOTORO_PASTE." >&2
  julia --project="${ROOT}" "${ROOT}/tools/totoro323/run_totoro_323_track_a_launcher.jl" --help >/dev/null 2>&1 || true
  exit 2
fi

JULIA_BIN="${JULIA_EXECUTABLE:-julia}"
exec "${JULIA_BIN}" --project="${ROOT}" \
  "${ROOT}/tools/totoro323/run_totoro_323_track_a_launcher.jl" "$@"
