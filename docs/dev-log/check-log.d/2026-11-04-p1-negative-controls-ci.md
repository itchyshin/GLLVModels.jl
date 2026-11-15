# P1 negative-controls Julia setup, 2026-11-04

Base: f34d5c175773729cb5a8f228184b56f877234f1a. Maintainer approved the continuation plan and local CI correction. No push, merge, N1c execution or row admission is authorized by this checkpoint.

Added Julia 1.10 setup to the negative-controls job before all controls. Both failing Python controls execute extracted fixture constructors with Julia Printf only; no model fit or package installation is needed.

Validation: original local environment without Julia produced 35/37 controls, with the two expected FileNotFoundError failures. A Totoro check with Julia 1.10.12 on PATH initially produced 36/37 because its checkout was shallow. After fetching full history to match the workflow's existing fetch-depth 0, the unchanged tools/test_core070_behaviour_receipts.py passed 37/37, exit 0. The two target controls passed in both Julia-present runs.

Ruby's standard YAML parser parsed the patched workflow and verified Julia 1.10 setup precedes every control command. git diff --check passed. PyYAML was unavailable, so no new dependency was installed. actionlint was unavailable. Hosted Actions execution and the entire job's other receipt checks remain unverified. Known receipt-refresh and main-shard failures are separate.

Remote logs: /home/snakagaw/hsq_work/codex-bootstrap-sigma-validation-f34d5c175/p1-negative-controls-julia110{,-fullhistory}.log and corresponding .exit files. Baseline log: /tmp/p1-negative-controls-without-julia-20261104.log. No test, source, receipt, scoreboard, tolerance or signed FAIL was changed.
