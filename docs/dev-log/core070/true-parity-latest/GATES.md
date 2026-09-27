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
`tools/parity_oracle.py::DEFAULT_R_REF` at P1 is A0b's job (a separate PR, `claude/true-parity-p1-pin`
/ PR #524, additive: `P1_GLLVMTMB_ORACLE`, `R_REF_PINS`, the `GLLVM_PARITY_PIN` env switch,
`_DEFAULT_PIN` left at `"P0"` by design until a later PR flips that one token), not this one.

## Post-review hardening (2026-09-27)

An independent review of the first cut of this ledger and tool (PR #523, head `346d01734`)
came back BLOCKING: the oracle accepted labels in place of evidence. Nothing false could be
claimed at the time (the P1 scoreboard/case-map did not exist yet), but every gap below would
have let a future row be signed off without real evidence. All are fixed in
`tools/true_parity_check.mjs`, each with a negative control in `tools/test_true_parity_check.mjs`
that fails on the old behaviour and passes on the fix:

1. A receipt path that resolved to a **directory** counted as present (`git cat-file -e`
   succeeds on trees, and `existsSync` is true for a directory too). Fixed: every receipt path
   must resolve to a **blob** (`git cat-file -t` = `blob`; FS `statSync(...).isFile()`).
2. A scoreboard cell with **no extractable receipt path** (an empty cell, "see PR #501", a
   `.unlazy/...` reference, an unsupported extension) counted as done anyway. Fixed: an
   `EVIDENCED` (or signed) row needs at least one extracted, resolving path; text that plainly
   looks like a path (`looksPathLike`) but that the extractor could not capture is a
   `MEASUREMENT_FAILED`, not a silent pass or a silent not-done.
3. Self-labelled evidence passed: (a) `DISPOSITION-SIGNED` with no signer or date; (b)
   `executable_case_ids` with no `evidence.receipt` still counted as bound/twinned; (c)
   `outside_boundary` with `disposition: null` vanished from C1 and C8 entirely. Fixed: a
   signed disposition needs both `signed_by` (non-empty) and `signed_on` (`YYYY-MM-DD`) on the
   row — **the tool never verifies the signer's identity; that happens in PR review**, by a
   human reading the diff, exactly as any other reviewed change; a bound/twinned row needs a
   receipt that actually resolves, case ids alone are not enough; `outside_boundary` rows are
   evaluated by C8 like any other row and must carry a real signed disposition.
4. `git ls-tree --name-only REF -- .github/workflows` (no trailing slash) returned the
   directory entry itself, not its contents, so C0's CI-job scan silently scanned nothing in
   git mode (`C0_MET` was unreachable, but so was an honest `C0_NOT_MET` reason). C0 also never
   checked `CAPABILITY_LEDGER_REF`, and a workflow merely mentioning "P1" or "true-parity" in
   passing (including this ledger's own CI smoke test) could satisfy it. Fixed: `listDir` adds
   a trailing slash; C0 requires the job id `p1-twin-tests` running the `gllvm-parity-tag: P1`
   discovery convention from PR #524's `parity-p1-twin.yml`, excludes this repo's own
   `true-parity-check.yml` by name as a backstop, and checks `CAPABILITY_LEDGER_REF` is
   present. **GitHub's "required check" status is branch-protection configuration on `main`,
   which no tool running against repo content can read — C0 checks the job exists, runs the
   P1 convention, and is not `continue-on-error`; it does not and cannot confirm branch
   protection marks it required.** A git-mode negative control (a temporary `git init` repo)
   exercises `show`, `existsAsBlob` and `listDir` through real `git show` / `git cat-file` /
   `git ls-tree`, the same code path CI runs against `origin/main`, not only the FS fixture
   fallback.
5. The carry rule compared author-written strings ("abc123"/"abc123" passed, no hash-format
   check) and was opt-in (a row with no `carry` block counted as fresh by default; an empty
   `source_pins` array passed). Fixed: every required row must record `measured_against`
   (`"P1"` or the full P1 sha, or the pin/commit it was actually measured at); a row not
   measured directly against P1 needs a non-empty `carry.source_pins`, each pair a real
   64-lowercase-hex sha256 with `sha256_at_p0 === sha256_at_p1`, or the row reads
   `PARTIAL_STALE_AT_P1` and does not count as bound. **Deferred, stated not silently patched:**
   a `CARRY_VERIFY` mode that takes a local gllvmTMB clone (`GLLVMTMB_DIR`) and re-hashes
   `git show P0:path` / `P1:path` itself, rather than trusting the two hashes recorded on the
   row, does not exist yet. Until it lands, every `carry` entry is trusted input from whoever
   re-measured the row (checked for format and equality only, not independently reproduced).
6. C8's name-only-match rule only catches a row that is **self-labelled**
   `semantic_divergence`; the tool has no way to notice, from the case-map alone, that a row
   classified `required_core` is actually a name twin (e.g. R's `zi_*` mislabeled instead of
   correctly tagged). **Deferred, stated not silently patched:** catching this needs a receipt
   that records an actual R-vs-Julia numeric comparison, not just a classification field; until
   that exists, correct classification at P1 is A0c's signed responsibility (D-295 row 3,
   Shinichi signs), not something this tool can verify on its own.
7. C6 accepted any non-empty `decision` string (`"TBD"` passed); C7 matched any heading
   containing "is not" (an unrelated heading like "This page is not finished" passed). Fixed:
   C6 requires `decision` to be exactly one of a fixed vocabulary (below); C7 requires the
   exact heading `What parity does not mean` at any heading level.
8. This file omitted Packet 1 row 6 (`select_lv`), `CAPABILITY_LEDGER_REF` from C0's own
   description, and the scoreboard id conventions the tool relies on. All added below.
9. CI (`.github/workflows/true-parity-check.yml`) runs only the fixture-based negative
   controls, never the real modes against `origin/main` — this is intentional (there is
   nothing real to check yet) but is not a substitute for `node tools/true_parity_check.mjs
   <mode>` run by hand once A0c/A0d populate the ledger.

**Second round (same day):** the first fix for item 4 still let a false `C0_MET` through. It
treated the presence of the `R_REF_PINS` + `GLLVM_PARITY_PIN` switch as sufficient for "P oracle
points at P", when the plan's own C0 text requires `DEFAULT_R_REF` to actually resolve to P1 --
a branch can add the switch and still leave the live default at P0 by design (exactly PR #524's
real, current state), and the first fix reported that as met. Caught by re-running C0 against
that real sibling branch, not only the fixtures: `PARITY_REF=origin/claude/true-parity-p1-pin
node tools/true_parity_check.mjs C0` printed `C0_MET` when it should not have. Fixed by reading
the live default directly from the single `_DEFAULT_PIN` token `tools/parity_oracle.py` (PR
#524) now exposes for exactly this purpose. A second bug surfaced verifying that fix the same
way: the first version of the `_DEFAULT_PIN` regex was unanchored, so it matched the token where
the module's own docstring mentions it in prose ("flip `` `_DEFAULT_PIN = "P1"` `` below") before
it reached the real assignment further down the file -- again a false `C0_MET` against the same
real branch. Fixed by anchoring the match to the start of a line. Both are now negative controls
(`c0_default_pin_p0`/`c0_default_pin_p1`/`c0_default_pin_mentioned_in_prose_only` in
`tools/test_true_parity_check.mjs`, one of them in git mode). The lesson generalises: a fixture
proves the logic is internally consistent, but only re-running the live modes against a real,
independently-authored branch catches a check that is consistent with itself and still wrong.

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
- **`select_lv` (Packet 1 row 6): Shinichi signs this row.** The auto-d lane (#518, gllvmTMB
  #1324) builds `select_lv` and drafts the receipt; this ledger only records it — no agent
  signs it. The row must carry the known twin difference as a fence: R rejects fits whose
  Hessian is not positive-definite, a behaviour Julia's `select_lv` does not currently
  reproduce. Until Shinichi signs, this row stays open under C1/C8 like any other unsigned row.

## The receipt carry rule (Packet 1 row 1)

Every required row records `measured_against`: `"P1"` or the full P1 sha if it was measured
directly at P1, otherwise the pin/commit it actually predates. A row not measured directly
against P1 counts at P1 only if every file in its `carry.source_pins` is byte-identical
between the gllvmTMB commit the receipt was measured against and P1 — each pin entry is
`{path, sha256_at_p0, sha256_at_p1}`, and both hashes must be real 64-lowercase-hex sha256
strings with `sha256_at_p0 === sha256_at_p1`. `tools/true_parity_check.mjs` does not re-hash
the gllvmTMB tree itself (see gap 5 above, `CARRY_VERIFY` deferred); a case-map row that
claims a carry must record both hashes itself, computed by whoever re-measures the row. If
`measured_against` is missing, if `carry` or `carry.source_pins` is missing or empty, if a
hash is not 64-lowercase-hex, or if the pair differs, the row's effective status is
`PARTIAL_STALE_AT_P1` and it does **not** count as bound for C1, regardless of its original P0
disposition. 62 R and C++ files changed between the pins (measured 2026-09-27), so most
carried rows are expected to land here until WS0d's stale-row scan re-measures them.

## Case-map row schema (what A0c/A0d must write)

A required row is bound only with a resolving receipt (`evidence.receipt`, a path that exists
as a blob at the ref) **and** non-empty `executable_case_ids` — neither alone is enough. A
`DISPOSITION-SIGNED` row additionally needs `signed_by` (a non-empty name) and `signed_on`
(`YYYY-MM-DD`) on the row itself; **the tool checks the fields are present, not that the named
person actually signed — identity is verified in PR review**, by whoever reviews the diff that
adds the row, the same way any other change to this repo is reviewed. `classification` values
this tool understands: `required_core`, `compatibility_adapter` (both feed C1), plus
`semantic_divergence`, `outside_boundary`, `excluded`, `needs_surface` (C8 requires each of
these to be either twinned or carry a real signed disposition — none of them are exempt or
silently skipped). A row may carry an optional `capability` field naming the scoreboard row id
it corresponds to (used by C2's cross-check, control (b) below).

## Scoreboard id conventions (what the tool's C2-C5 filters rely on)

`tools/true_parity_check.mjs` splits scoreboard rows by id prefix/suffix, not by a separate
column: a row whose id ends `-RSZ` is a realistic-size cell (C3); a row whose id starts `RD-`
is a real-data workflow (C4); a row whose id starts `GRP-` is a grouping-level row (C5); every
other row is a plain P1-boundary capability (C2). A0c/A0d must follow this convention when they
add real scoreboard rows, or their rows will silently fall into the wrong clause.

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
      unchanged, `CAPABILITY_LEDGER_REF` present), the explicit `R_REF_PINS` +
      `GLLVM_PARITY_PIN` switch exists, **and `DEFAULT_R_REF` itself actually resolves to P1**
      — read directly from the single `_DEFAULT_PIN` token (`_DEFAULT_PIN = "P1"`), not
      inferred from the switch merely existing — plus a CI job named `p1-twin-tests` that runs
      the `gllvm-parity-tag: P1` discovery convention and is not `continue-on-error`.
      A switch that CAN select P1 on request is necessary but not sufficient: the plan's own
      C0 text requires `DEFAULT_R_REF` to point at P, and a branch that adds the switch while
      leaving the live default at P0 is a real, distinguishable state, not C0_MET.
      "Required" in GitHub's sense is branch-protection configuration on `main`, which cannot
      be read from repo content — this clause checks the job exists and is not advisory, no
      more; the maintainer still marks it required from the GitHub UI once satisfied
  CHECK: node tools/true_parity_check.mjs C0
  EXPECT: C0_MET
  EVIDENCE: pending. `claude/true-parity-p1-pin` (PR #524) adds the P1 pin, the switch, and the
  `parity-p1-twin.yml` job, but keeps `_DEFAULT_PIN = "P0"` by design until a later PR flips
  the one token — `node tools/true_parity_check.mjs C0` against that branch correctly reports
  `C0_NOT_MET` with `default_pin=P0` (verified 2026-09-27; an earlier version of this clause
  accepted the switch alone and produced a false `C0_MET` against that same branch — the exact
  defect this tool exists to catch, caught by re-running the check against the real sibling
  branch rather than only the fixtures). C0 becomes met once a later PR sets
  `_DEFAULT_PIN = "P1"` there and it merges.

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
      extras: `SourceCovariance`, two-part ZI) has a written decision from a fixed vocabulary
      (`KEPT_AS_JULIA_EXTRA`, `PORT_TO_MATCH_R`, `DEPRECATE_AND_REMOVE`,
      `RENAME_TO_AVOID_COLLISION`) — a placeholder like `"TBD"` or an empty string does not
      count as decided just because the field is non-empty or present
  CHECK: node tools/true_parity_check.mjs C6
  EXPECT: C6_MET
  EVIDENCE: pending

- [x] C7: `docs/src/gllvmtmb-parity.md` states in one place what parity does not mean, under
      the exact heading `What parity does not mean` (any heading level) — a heading that merely
      contains "is not" (e.g. "This page is not finished") does not satisfy this clause
  CHECK: node tools/true_parity_check.mjs C7
  EXPECT: C7_MET
  EVIDENCE: verified 2026-09-27 against `origin/main` (`1385b0490`): the exact "### What parity
  does not mean" heading is present; `C7 parity_page_has_exact_not_mean_heading=true` /
  `C7_MET`. Pin-independent (the page is not gllvmTMB-ref-scoped).

- [ ] C8: every R export at P1 is twinned (case-map row with a resolving Julia receipt AND
      non-empty `executable_case_ids`) or carries a real signed disposition (`signed_by` +
      `signed_on`, see the case-map schema above); name matches alone never count — a
      `semantic_divergence` row fails this clause unless it is properly signed, however many
      `executable_case_ids` it has; an `outside_boundary` row is evaluated the same way and
      does **not** leave the ledger silently just by being placed outside the boundary
  CHECK: node tools/true_parity_check.mjs C8
  EXPECT: C8_MET
  EVIDENCE: pending (A0c classifies the +25 exports / +11 S3 methods since 0.7.0; Shinichi
  signs in that PR, Packet 1 row 3). Known gap, stated not silently patched: the tool can only
  catch a name-only match that is self-labelled `semantic_divergence` — it cannot detect from
  the case map alone that a row classified `required_core` is actually a name twin; correct
  classification at P1 is A0c's signed responsibility (see gap 6 above).

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
