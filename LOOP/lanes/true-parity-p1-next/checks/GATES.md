# Gates: first seven P1 rows

Scope: seven exact receipts plus decision packet; full P1 remains open.

- [ ] G1: Seven exact rows bind at HEAD without unrelated regressions
  CHECK: node tools/true_parity_checkpoint_check.mjs --baseline f220379d0937d0afffc6a030023c63c0f715e168 --ref HEAD
  EXPECT: P1_CHECKPOINT_MET
  EVIDENCE: pending

- [ ] G2: Generated assembly is current
  CHECK: python3 tools/true_parity_assemble.py --check
  EXPECT: ASSEMBLE_OK
  EVIDENCE: pending

- [ ] G3: Canonical checker negative controls pass
  CHECK: node tools/test_true_parity_check.mjs
  EXPECT: All true-parity negative controls passed.
  EVIDENCE: pending

- [ ] G4: Assembler controls pass
  CHECK: python3 tools/test_true_parity_assemble.py
  EXPECT: ALL PASS
  EVIDENCE: pending

- [ ] G5: Checkpoint verifier controls reject wrong rows and invalid evidence
  CHECK: node tools/test_true_parity_checkpoint_check.mjs
  EXPECT: P1_CHECKPOINT_CONTROLS_PASS
  EVIDENCE: pending

- [ ] G6: Independent review accounts for all remaining 13 scoreboard rows and 37 C6 names
  EVIDENCE: pending

- [ ] G7: All checkpoint branches have explicit maintainer merge approval and evidence on main
  CHECK: node tools/true_parity_checkpoint_check.mjs --baseline f220379d0937d0afffc6a030023c63c0f715e168 --ref origin/main
  EXPECT: P1_CHECKPOINT_MET
  EVIDENCE: pending

- [ ] G8: Fresh completion panel, after-task report and reconciliation complete
  EVIDENCE: pending

