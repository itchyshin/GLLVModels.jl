# GATES — true parity, GLLVModels.jl vs gllvmTMB P1 (main `9539352f66f2db2cc26b1c393e67212a359b60c9`, 0.7.1 candidate)

SCOPE: the C0-C8 destination clauses in
`~/local-scratch/lanes/GLLVM.jl-true-parity-latest/LOOP/lanes/true-parity-latest/ultra-plan.md`
("Destination (the stopping condition)"), read from origin/main (never a branch or working
tree). Oracle: `tools/true_parity_check.mjs`. This file and the oracle are TRACKED IN GIT
(D-294/D-295, Packet 1 row 4, signed 2026-09-27) — the previous ledger lived under the
gitignored `.unlazy/true-parity/` and lost its receipts (20 isdm rows cited a path that no
longer exists). Tracking here is additive: the 0.7.0 ledger, oracle pin, and every receipt
under `docs/dev-log/core070/` that predates this file stay untouched.

PIN: P1 = gllvmTMB `main` at commit `9539352f66f2db2cc26b1c393e67212a359b60c9` (0.7.1
candidate, untagged as of 2026-09-27). P0 (the frozen 0.7.0 oracle, `b4d5fee64def88bc768dda1f1f77c29b295edd86`)
remains the pin for the existing `tools/parity_oracle.py::FROZEN_GLLVMTMB_ORACLE` and every
receipt that cites it; nothing here rewrites those ~555 files or that pin. Re-pointing
`tools/parity_oracle.py::DEFAULT_R_REF` at P1 is A0b's job (a separate PR), not this one.

## The D-295 boundary

Signed 2026-09-27, Packet 1 row 0 ("accept 0 to 13 as recommended"):

- **Inside P1:** temporal (Gaussian, rank-1, AR1/OU — Shinichi asked for it by name) and
  phylo latent (`covariance/COV-PHYLO-LATENT`, A14/A15 — a 0.7.0-era gap owed regardless of
  the re-pin).
- **Outside P1, by signed disposition, revisited at P2:** column-coefficient grammar (a
  Gaussian point model with no intervals in R; 109 R files, ~25,724 lines) and spatial
  (`spatial_dep`, `spatial_*` — Julia keeps its own SPDE/Matérn stack as a documented extra,
  not a twin of R's spatial surface).
- iSDM (R's public door, `gllvmTMB(..., family = isdm_sources(...))`, non-spatial, Laplace,
  first order plus `predict`, no intervals) is the HEADLINE inside P1 (Packet 1 row 2).
- Name twins never count on name alone: `SEMANTIC_DIVERGENCE` capabilities (e.g. R's `zi_*`
  vs Julia's two-part ZI) stay FORWARD until a twin with R's semantics exists or Shinichi
  signs a disposition (Packet 1 row 5).

## The receipt carry rule (Packet 1 row 1)

A P0 receipt counts at P1 only if every file in its `carry.source_pins` is byte-identical
between the gllvmTMB commit the receipt was measured against and P1. `tools/true_parity_check.mjs`
does not re-hash the gllvmTMB tree itself; a case-map row that claims a carry must record
both hashes it was measured against (`sha256_at_p0` and `sha256_at_p1` per source-pin path).
If any pair differs, or the P1 hash is missing (not yet re-measured), the row's effective
status is `PARTIAL_STALE_AT_P1` and it does **not** count as bound for C1, regardless of its
original P0 disposition. 62 R and C++ files changed between the pins (measured 2026-09-27),
so most carried rows are expected to land here until WS0d's stale-row scan re-measures them.

## Clauses

Every clause starts unmet except C7, whose evidence (`docs/src/gllvmtmb-parity.md`) already
holds on `origin/main` independent of the pin. C1 through C6 and C8 need the P1 scoreboard
(`docs/dev-log/core070/true-parity-latest/scoreboard.md`) and case-map
(`docs/dev-log/core070/true-parity-latest/case-map.json`), which this PR does not create
(A0c/A0d build those rows; Shinichi signs classifications there, D-295 row 3) — until then,
`node tools/true_parity_check.mjs <mode>` reports `MEASUREMENT_FAILED` (exit 2) for those
modes on `origin/main`, which is the honest state, not a false pass.

- [ ] C0: the P1 oracle exists alongside P0 (additive) in `tools/parity_oracle.py`
      (`P1_GLLVMTMB_ORACLE` present and equal to the full P1 SHA, `FROZEN_GLLVMTMB_ORACLE`
      unchanged), `DEFAULT_R_REF` points at the P1 constant, and a required (non-advisory,
      no `continue-on-error`) CI job runs the P1 twin tests
  CHECK: node tools/true_parity_check.mjs C0
  EXPECT: C0_MET
  EVIDENCE: pending (A0b re-pins `DEFAULT_R_REF`; no P1 CI job exists yet)

- [ ] C1: every required row (`required_core`, `compatibility_adapter`) at P1 is bound to a
      receipt that resolves on `origin/main`, or carries a maintainer-signed disposition;
      `BLOCKED_*`, `PARTIAL_*` and `PARTIAL_STALE_AT_P1` do not count as signed; every cited
      receipt path must exist at the ref
  CHECK: node tools/true_parity_check.mjs C1
  EXPECT: C1_MET
  EVIDENCE: pending (P1 case-map not yet populated; A0c/A0d)

- [ ] C2: every P1 capability inside the D-295 boundary (isdm 1FO plus `predict`, temporal
      1FO, phylo_latent 1FO, ordinal_logit 1FO, zi_1fo under R semantics, and so on) has a
      scoreboard row with a resolving P1 receipt; row count is read from the scoreboard file,
      never hard-coded
  CHECK: node tools/true_parity_check.mjs C2
  EXPECT: C2_MET
  EVIDENCE: pending

- [ ] C3: one realistic-size cell (p >= 20, n >= 500, condition number recorded) per paired
      family/structure inside the boundary
  CHECK: node tools/true_parity_check.mjs C3
  EXPECT: C3_MET
  EVIDENCE: pending

- [ ] C4: one real-data workflow per qualified family or structure runs end to end through
      `engine = "julia"` and passes the acceptance classes
  CHECK: node tools/true_parity_check.mjs C4
  EXPECT: C4_MET
  EVIDENCE: pending

- [ ] C5: grouping levels `unit`, `unit_obs`, `cluster`, `cluster2` exist on both engines
      under those names and pair (same reading as the 0.7.0 ledger's B-04 disposition:
      same names plus the paired Gaussian receipts that exist today; non-Gaussian numerical
      pairing is out of scope, Packet 1 row 9)
  CHECK: node tools/true_parity_check.mjs C5
  EXPECT: C5_MET
  EVIDENCE: pending

- [ ] C6: the reverse-gap list is tool-produced and every item (including the Julia-only
      extras: `SourceCovariance`, two-part ZI) has a written decision
  CHECK: node tools/true_parity_check.mjs C6
  EXPECT: C6_MET
  EVIDENCE: pending

- [x] C7: `docs/src/gllvmtmb-parity.md` states in one place what parity does not mean
  CHECK: node tools/true_parity_check.mjs C7
  EXPECT: C7_MET
  EVIDENCE: verified 2026-09-27 against `origin/main` (`1385b0490`): the "### What parity
  does not mean" heading is present; `C7 parity_page_has_not_mean_section=true` /
  `C7_MET`. Pin-independent (the page is not gllvmTMB-ref-scoped).

- [ ] C8: every R export at P1 is twinned (case-map row with a Julia receipt) or signed;
      name matches alone never count — a `semantic_divergence` row without a signed
      disposition fails this clause even if `executable_case_ids` is non-empty
  CHECK: node tools/true_parity_check.mjs C8
  EXPECT: C8_MET
  EVIDENCE: pending (A0c classifies the +25 exports / +11 S3 methods since 0.7.0; Shinichi
  signs in that PR, Packet 1 row 3)

- [ ] X2: all scoreboard rows are `EVIDENCED` or `DISPOSITION-SIGNED`; row count read from
      the file, an empty selection is never a pass
  CHECK: node tools/true_parity_check.mjs X2
  EXPECT: X2_MET
  EVIDENCE: pending

## F-gates (per-capability, informational until the scoreboard exists)

One F-gate per capability named in the D-295 boundary; each becomes a scoreboard row under
C2 once A0c/A0d land. Listed here so the boundary is legible without re-reading the plan:

- [ ] F-ISDM-1FO-PREDICT — R's public door, non-spatial, Laplace, first order plus `predict`
- [ ] F-TEMPORAL-1FO — Gaussian rank-1 latent score, AR1/OU, first order
- [ ] F-PHYLO-LATENT-1FO — `phylo_latent()` bare, first-order paired (carries A14/A15 from
  the 0.7.0 scoreboard)
- [ ] F-ORDINAL-LOGIT-1FO — `ordinal_logit`, first order
- [ ] F-ZI-1FO — R-semantics `zi_*`, first order (Julia's own two-part ZI stays a documented
  extra; a name match alone never signs this row, see C8)

CHECK: none yet (informational; folds into C2 once the P1 scoreboard carries these ids)
EVIDENCE: pending

## M1

- [ ] M1: a maintainer-signed joint note exists before `Project.toml` leaves 0.3.0 (manual;
  Shinichi signs; not mechanically checkable)
  EVIDENCE: pending
