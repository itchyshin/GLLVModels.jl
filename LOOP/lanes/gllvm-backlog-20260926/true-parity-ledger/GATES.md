# GATES — true parity, GLLVModels.jl vs frozen gllvmTMB 0.7.0 (b4d5fee64)

SCOPE: the seven-clause destination in docs/dev-log/core070/true-parity-decision-map.md, read from
origin/main (never a branch or working tree). Oracle: .unlazy/true-parity/check.mjs (positive and
negative controls run 2026-09-26: every mode can print MET; one reverted row flips only its clauses).
Clauses C2 to C6 read the gate-tier scoreboard, so they are only as current as its last refresh.
A row counts as done only when its status is EVIDENCED or DISPOSITION-SIGNED.
OWNS: .unlazy/true-parity/**

- [ ] C1: every required ledger row (497) is bound to a receipt or carries a maintainer-signed disposition; BLOCKED_* and PARTIAL_* dispositions do not count as signed
  CHECK: node check.mjs C1
  EXPECT: C1_MET
  EVIDENCE: pending

- [ ] C2: every paired family and covariance cell (scoreboard A-rows, not RSZ) has first- and second-order receipts
  CHECK: node check.mjs C2
  EXPECT: C2_MET
  EVIDENCE: pending

- [ ] C3: one realistic-size cell (p >= 20, n >= 500, condition number recorded) per paired family (RSZ rows)
  CHECK: node check.mjs C3
  EXPECT: C3_MET
  EVIDENCE: pending

- [ ] C4: one real-data workflow per qualified family or structure runs end to end through engine = "julia" and passes the eight acceptance classes (C-rows)
  CHECK: node check.mjs C4
  EXPECT: C4_MET
  EVIDENCE: pending

- [ ] C5: grouping levels unit, unit_obs, cluster, cluster2 exist on both engines under those names and pair (B-rows)
  CHECK: node check.mjs C5
  EXPECT: C5_MET
  EVIDENCE: pending

- [ ] C6: the reverse gap list is tool-produced and every item has a written decision (D8)
  CHECK: node check.mjs C6
  EXPECT: C6_MET
  EVIDENCE: pending

- [ ] C7: docs/src/gllvmtmb-parity.md states in one place what parity does not mean
  CHECK: node check.mjs C7
  EXPECT: C7_MET
  EVIDENCE: pending

- [ ] X1: bridge, extractor and fit-input rows D1 to D7 are done (the gate-tier rows the seven clauses lean on)
  CHECK: node check.mjs EXTRACT
  EXPECT: EXTRACT_MET
  EVIDENCE: pending

- [ ] X2: all 32 gate-tier rows are EVIDENCED or DISPOSITION-SIGNED
  CHECK: node check.mjs ALL32
  EXPECT: ALL32_MET
  EVIDENCE: pending

- [ ] M1: a maintainer-signed joint note exists before Project.toml leaves 0.3.0 (manual; Shinichi signs)
  EVIDENCE: pending
