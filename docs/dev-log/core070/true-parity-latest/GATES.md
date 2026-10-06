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
9. CI (`.github/workflows/true-parity-check.yml`) runs the fixture-based negative
   controls, the assembler's own controls (`tools/test_true_parity_assemble.py`) and
   `tools/true_parity_assemble.py --check` (so a stale or hand-edited scoreboard, assembled case map
   or reverse-gap fails the PR), never the real modes against `origin/main` — this is intentional
   (there is nothing real to check yet) but is not a substitute for `node tools/true_parity_check.mjs
   <mode>` run by hand once A0c/A0d populate the ledger. The job runs when the tools, the fixtures or
   anything under `docs/dev-log/core070/true-parity-latest/` change.

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
as a blob at the ref) **and** non-empty `executable_case_ids` **and** an evidence tier that binds; no
one of these alone is enough. Two tiers bind. `evidence_tier: "numeric"`, backed by a `comparison`
block in the receipt (see "Evidence tier" below), counts in C1's `bound=` and `bound_numeric=`.
`evidence_tier: "behavioural"`, only on the rows itchyshin/GLLVModels.jl#684 item 2 covers (the
frozen list in "Rulings of 2026-10-02", ruling 2) and backed by a `behaviour` block, counts in
`bound_behavioural=`, never in `bound=` or `bound_numeric=`. Every other tier (`registration` and
the rest) does not bind. A `DISPOSITION-SIGNED` row additionally needs `signed_by` and `signed_on`
on the row itself.
Since the review of #561 the tool checks both: `signed_by` must be exactly `Shinichi Nakagawa` or
`itchyshin` (anything containing `agent`, `Claude`, `Codex`, `Cursor` or `Fable` is refused, as is
any other name; C1 reports `DISPOSITION-SIGNER-NOT-ALLOWED`), and `signed_on` must be a real
calendar date in `YYYY-MM-DD` form that is not in the future (checked against the latest current
date anywhere on earth, UTC+14, so a local date is never refused for time-zone reasons; C1 reports
`DISPOSITION-SIGNED-BAD-DATE`). The same rules apply to `signed_by:` / `signed_on:` tokens in a
scoreboard cell. A properly signed row is counted in C1's `bound_signed=`, never in `bound=`
(`bound=` counts numeric evidence only), so a signed registration row cannot read as a twin.
**The tool still cannot verify that the named person actually signed; an agent can type the
maintainer's name. Identity is verified in PR review**, by whoever reviews the diff that adds the
row, the same way any other change to this repo is reviewed. Negative controls:
`c1_signed_by_agent`, `c1_signed_bad_date` (`9999-99-99`), `c1_signed_future_date`. `classification` values
this tool understands: `required_core`, `compatibility_adapter` (both feed C1), plus
`semantic_divergence`, `outside_boundary`, `excluded`, `needs_surface` (C8 requires each of
these to be either twinned or carry a real signed disposition — none of them are exempt or
silently skipped). A row may carry an optional `capability` field naming the scoreboard row id
it corresponds to (used by C2's cross-check, control (b) below).

## Evidence tier: a registration match is not a twin (D-295 row 5)

Added after an independent review of the namespace re-measure (PR #559): a Tier 0 namespace receipt
shows only that the R export is registered and defined and that a same-named Julia symbol exists,
and the first cut of this tool counted that as a fully bound row, indistinguishable from a numeric
twin. Every case-map row now carries `evidence_tier`:

- `"numeric"`: the receipt records an actual R-vs-Julia output comparison;
- `"registration"`: export/existence registration only (the namespace Tier 0 batch);
- `"behavioural"` (added by itchyshin/GLLVModels.jl#684 item 2, 2026-10-02): the row's R behaviour is a
  refusal, a printed summary, a routing decision or an error class, and the receipt records that both
  engines gave the same one. It is allowed only on the rows in the frozen list (ruling 2 below); on any
  other row the label does not bind. It is not numeric evidence and is counted apart.

C1 prints `bound_numeric=N bound_registration_only=M` and is MET only when `M == 0` (behavioural rows
are counted in `bound_behavioural=B`, and a behavioural row that fails its rule is listed under
`behavioural_label_without_behavioural_receipt=` and blocks C1); a
registration-only row still resolves if it carries a real signed disposition (`signed_by` +
`signed_on`). C8 reports a registration-only row as `REGISTRATION_ONLY_NOT_TWINNED` instead of
twinned. A missing `evidence_tier` is fail-closed (treated as registration-only). Negative controls
in `tools/test_true_parity_check.mjs` (`c1_registration_only`, `c1_evidence_tier_missing`,
`c1_registration_only_signed`, one of them in git mode).

### What the fail-closed default does to rows that carry no `evidence_tier`

Disclosed after the review of #561 (an earlier draft of this PR said "P0 behaviour is unchanged";
that holds for C1 at P0 and for the namespace batch tools, not for C8 at P0):

- **P0.** `docs/dev-log/core070/required-source-case-map.json` (769 rows) has no `evidence_tier`
  on any row. The 14 `aghq/AGHQ-*` rows bound to `aghq-public-policy-bind-receipt-2026-09-04.json`
  were counted as twinned by the checker at `cb5688f7e`; they now read
  `REGISTRATION_ONLY_NOT_TWINNED` in C8 at P0. Measured tally, same case-map, before and after:
  `7 DANGLING_RECEIPT, 748 NOT_TWINNED_NOT_SIGNED` (14 twinned) becomes `7 DANGLING_RECEIPT,
  748 NOT_TWINNED_NOT_SIGNED, 14 REGISTRATION_ONLY_NOT_TWINNED`. The verdict is `C8_NOT_MET` before
  and after; C1 at P0 is unchanged apart from the two new counters. Every P0 row reads as
  registration-only until `evidence_tier` is back-filled. The tiers are **not** back-filled here:
  choosing a row's tier is a classification, and classifications are signed by Shinichi (D-295
  row 3). The aghq receipt records R fits with objective values but no Julia comparison, so it has
  no `comparison` block and would not bind as numeric under the rule below even if relabelled.
- **#533's `case-map.json`.** None of its 38 rows carries `evidence_tier` (head `9c1f55038`).
  After the fold they will read as registration-only (C1 `registration_only=`, C8
  `REGISTRATION_ONLY_NOT_TWINNED`, or `NOT_TWINNED_NOT_SIGNED` where they have no receipt). That is
  the expected effect of the fail-closed default, not a regression; the fold PR, or #533, sets the
  tier per row, and a `"numeric"` tier only binds with a receipt that carries a `comparison` block.

### The numeric tier is verified against the receipt, not trusted (review of #561)

The first cut trusted the label: `"numeric"` was a string on an agent-writable field, and flipping
one word on the real namespace row `namespace/S3method/coef,gllvmTMB_multi` (a registration receipt
with no numbers in it) printed `C1_MET` and `C8_MET`. The label can now only lower a row, never
raise it. A row labelled `"numeric"` counts as numeric only if its cited receipts carry a
machine-readable comparison block:

```json
"comparison": {
  "pin": "P1",
  "cases": [
    { "case_id": "CASE-1", "quantity": "coef", "max_abs_diff": 2.4e-06, "tolerance": 1e-04 }
  ]
}
```

- `comparison` is a top-level key of a JSON receipt; `pin` is `"P1"` or the full P1 sha.
- `cases` is non-empty; each case has a non-empty `case_id`, a `tolerance` that is a finite number
  greater than 0, and one of: `abs_diff`, `max_abs_diff` (finite, >= 0), or `r_value` +
  `julia_value` (finite numbers, or equal-length arrays of finite numbers; the tool computes the
  maximum absolute difference itself).
- When a case carries a comparable `r_value` + `julia_value` pair, the tool's own difference is
  the one judged, and any recorded `abs_diff` / `max_abs_diff` on that case must agree with it to
  1e-12 relative, or the row fails as `NUMERIC_RECORDED_DIFF_MISMATCH` (see below). A recorded
  difference stands on its own only when the case carries no comparable pair.
- Every case must satisfy difference <= tolerance.
- The union of `case_id`s across the row's receipts must cover every `executable_case_id` on the row.
- A malformed block in any cited receipt fails the row; a non-JSON receipt simply carries no block.

The shape follows the per-case records the core070 twin batches already write
(`max_abs_diff`, `tolerance`, `quantity`, e.g.
`docs/dev-log/core070/t5-rebind-out/estimand-rebind-02/julia-results.json`), plus the `pin` and a
`case_id` per case, so a numeric twin batch can emit it with a small wrapper. The R reference
values in a twin fixture such as `test/fixtures/ordinal_logit_p1.toml` are one side of that
comparison; the receipt must record both sides, or the difference, with its tolerance.

A `"numeric"` row whose receipts fail any of these rules is reported as
`NUMERIC_LABEL_WITHOUT_NUMERIC_RECEIPT`: C1 lists it under `numeric_label_without_numeric_receipt=`
(with the reason) and is not MET; C8 fails it with that tag. Negative controls:
`c1_numeric_label_registration_receipt` (numeric label, registration-style receipt; also in git
mode), `c1_numeric_label_malformed_comparison` (tolerance a string, no difference recorded),
`c1_numeric_label_over_tolerance` (`abs_diff` above `tolerance`), and the reviewer's mutation run on
the real `coef,gllvmTMB_multi` row and its real receipts, which went from `C1_MET`/`C8_MET` to
`C1_NOT_MET`/`C8_NOT_MET`. The positive-control `base` fixture's receipts now carry real blocks.

### A receipt that says it failed does not bind (review of #567)

A tamper test on #567's receipts set `verdict` to `"FAIL"` with the comparison block intact, and
the row still bound. The tool now reads the status fields that real core070 receipts write:
`verdict`, `batch_status` and `harness_pass` (case receipts) and `status` (batch receipts). Each
one present at the top level of a cited JSON receipt, or inside its `comparison` block, must hold
a pass value (`"PASS"`, `"pass"` or `true`); anything else, including `"FAIL"`, `null` or an
object, fails the row as `NUMERIC_RECEIPT_NOT_PASSED(<field>=<value> in <receipt>; <why the
exception does not apply>)`. C1 lists it under `numeric_receipt_not_passed=` and is not MET; C8
fails it with that tag. A missing status field is not a failure (older receipts carry none).

The one waiver is a maintainer-signed field on the case-map row:

```json
"receipt_status_exception": {
  "reason": "batch receipt reads FAIL because of one unrelated case; this case passes",
  "signed_by": "Shinichi Nakagawa",
  "signed_on": "2026-09-27"
}
```

Same signer allow-list and date rule as a signed disposition, plus a non-empty `reason`. It waives
only the status check (the comparison block must still be valid, pinned to P1 and within
tolerance), and a row bound this way counts in `bound_signed=`, never in `bound=` or
`bound_numeric=`. Negative controls: `c1_numeric_receipt_verdict_fail` (top-level `verdict:
"FAIL"`), `c1_numeric_receipt_comparison_status_fail` (`batch_status: "FAIL"` inside the
comparison block), `c1_numeric_receipt_fail_exception_by_agent` (exception signed by an agent
name: still not bound); positive control `c1_numeric_receipt_fail_signed_exception`.

### A recorded difference is cross-checked against the two sides (review of #567)

A second tamper test left `max_abs_diff` small while the recorded `r_value` and `julia_value`
disagreed by 1; the row still bound, because the recorded difference was trusted over the vectors.
Now, when a case has both sides, the tool recomputes the (maximum) absolute difference itself and
fails the row if a recorded `abs_diff` or `max_abs_diff` disagrees with it beyond 1e-12 relative,
whether or not either value is within tolerance: C1 lists it under
`numeric_recorded_diff_mismatch=` and is not MET; C8 fails it as
`NUMERIC_RECORDED_DIFF_MISMATCH(<case, recorded, recomputed, receipt>)`. There is no signed
waiver: a receipt whose numbers disagree with each other has to be regenerated. The 1e-12 bound
was checked against the real case receipts on #567 and #569 (21 cases with both sides and a
recorded `max_abs_diff`): none disagree. Negative controls: `c1_numeric_recorded_diff_stale`
(sides differ by 1, recorded 6e-11) and `c1_numeric_recorded_diff_mismatch_within_tol` (recorded
1e-7, recomputed 4e-7, both under tolerance 1e-6).

What this does not do: the tool checks that the receipt records a comparison within tolerance; it
does not re-run the comparison, and it cannot tell whether the tolerance chosen is reasonable. A
receipt that records false numbers passes. That is PR review's job, as for any other receipt.

### Tolerances are author-declared; review them against the harness source

The third tamper test on #567 widened a case's `tolerance` to 1.0 and the row still bound. That is
by design, not an oversight: the tool has no independent source for what the right bound is for a
given quantity, so it does not try to judge tolerances automatically. A tolerance is whatever the
receipt's author (usually the batch harness) declared, and the tool only checks that it is a
finite number greater than 0 and that the difference sits under it. A reviewer must therefore
check each numeric row's tolerances against the harness source that produced the receipt (the
batch contract or script that sets the bound, e.g. a `*-batch-contract-p1.json` or the
`tools/core070_*_batch.*` script named in the receipt), and treat a tolerance that differs from
that source, or one that is loose for the quantity compared, as a blocking finding.

Recommendation (not enforced by the tool): give each numeric case-map row a free-text
`tolerance_source` field naming where its tolerances come from, for example
`"tolerance_source": "wave6-conversion-batch-contract-p1.json, cases[].tolerance"`. The checker
ignores the field; it exists so a reviewer can find the source without reverse-engineering the
receipt.

The namespace Tier 0 batch itself was tightened at the same time: at P1 an executable row passes
only if the Julia symbol is exported and a Function (measured by the Julia child, not typed into
the contract), so an unexported helper or a type no longer passes on its name.

## Namespace rows and the fold into case-map.json (expected totals)

The 71 namespace rows re-measured at P1 (arc A3) live in
`docs/dev-log/core070/true-parity-latest/case-map-namespace.json` until PR #533's `case-map.json`
lands, so none of #533's rows are touched; run the checker on them with
`PARITY_CASEMAP=docs/dev-log/core070/true-parity-latest/case-map-namespace.json`. That file holds
69 rows (the 2 retired exports are listed in its `retired_at_p1` block and map to #533's
`retired/...` rows, not duplicated). The recommended next step is to fold these rows into
`case-map.json` after #533 merges, rather than teach the checker to read several files. **Expected
totals once folded: 38 + 69 = 107 rows, 90 of them required (`required_core` +
`compatibility_adapter`).** A folded file with fewer rows, or fewer required rows, means something
was dropped in the fold.

## Scoreboard id conventions (what the tool's C2-C5 filters rely on)

`tools/true_parity_check.mjs` splits scoreboard rows by id prefix/suffix, not by a separate
column: a row whose id ends `-RSZ` is a realistic-size cell (C3); a row whose id starts `RD-`
is a real-data workflow (C4); a row whose id starts `GRP-` is a grouping-level row (C5); every
other row is a plain P1-boundary capability (C2). A0c/A0d must follow this convention when they
add real scoreboard rows, or their rows will silently fall into the wrong clause.

## Rulings of 2026-10-02 (itchyshin/GLLVModels.jl#684)

On 2026-10-02 the maintainer, Shinichi Nakagawa, signed four rulings in
itchyshin/GLLVModels.jl#684. Cite them by that reference, never as a bare number. This section is
the only place the checker rules for rulings 1 to 3 are written down; ruling 4 (C3 to C5 rows and the
Totoro campaign) adds case-map rows and receipts and changes no rule. Nothing here is a new
signature: the tool still cannot check that the named person signed, and PR review does that by
reading the diff. Each ruling is ported to `tools/true_parity_assemble.py`, so the scoreboard
status and the checker agree.

### Ruling 1: integer equality (item 1)

Rows that compare an exact integer (`POST-LOGLIK-DF`, `POST-LOGLIK-NOBS`, `POST-NOBS-COUNT`,
`POST-NOBS-FALLBACK`) may record tolerance 0.5. The case must say so:

```json
{ "case_id": "CORE070-...", "kind": "integer_equality",
  "r_value": 15, "julia_value": 15, "tolerance": 0.5 }
```

`r_value` and `julia_value` must both be safe integers (`Number.isSafeInteger`, so 2^53 and beyond
are refused in both tools), or equal-length non-empty arrays of them, and `tolerance` must be
exactly 0.5. Then "within tolerance" can only mean
"equal". A case with no `kind` is judged as before. Any other `kind` fails the row. The row keeps
`evidence_tier: "numeric"` and counts in `bound_numeric=`.

Negative controls (`tools/test_true_parity_check.mjs`, group "integer equality"): 15 vs 15 binds;
equal integer vectors bind; off by one (15 vs 16); non-integer values (15.2 vs 15.2); tolerance 1;
vector length mismatch; abs_diff with no values; unknown kind; tolerance 0.4, 0.49, 0.5000001 and 0.51
(only exactly 0.5 means equality; a `> 0.5` test would accept 0.4); unsafe integers
(9007199254740993 vs 9007199254740992, and 2^53 itself) are refused while 9007199254740991 binds; and a
case with no `kind` keeps today's rule. Assembler: `integer_equality_*`, `unknown_comparison_kind_unverified`,
`no_kind_case_keeps_todays_rule`.

### Ruling 2: the behavioural tier (item 2)

A row whose R behaviour is a refusal, a printed summary, a routing decision or an error class
(the 59 inference routing and error-class rows; the C1 rows `print.gllvmTMB_select_lv`,
`print.anova.gllvmTMB_multi`, `update.gllvmTMB_multi`, `extract_latent_scores.default`) has no
number to compare. It closes when a receipt shows both engines giving the same refusal, route,
error class or printed fields. It counts as behavioural, not numeric.

**Scope: a frozen, explicit list.** The tier exists for the rows the ruling names and no others. A row
binds behaviourally only if its `source_id` is exactly one of:

- the 59 inference rows whose `evidence_tier` on `origin/main` (at `5b186bf32`, in
  `case-map-inference.json`) is `routing_control_flow` (45) or `reject_error_class` (14):
  `inference/CI-ROUTE-001` to `-084` with the gaps in the numbering, listed id by id in
  `BEHAVIOURAL_INFERENCE_SOURCE_IDS` (assembler) and in the checker's set of the same name;
- the four named C1 rows `latent-scores/extract_latent_scores.default`,
  `select-lv/print.gllvmTMB_select_lv`, `model-comparison/print.anova.gllvmTMB_multi` and
  `model-comparison/update.gllvmTMB_multi`.

It is not a prefix rule. `inference/CI-ROUTE-008` and `-010` (numeric today, with real R-versus-Julia
comparison blocks) and `-009` and `-011` (partial) are not in the list; neither is a bare `inference/`,
`inference/../isdm/X`, or any new `inference/...` id. Any row outside the list that says
`evidence_tier: "behavioural"` reads `BEHAVIOURAL_LABEL_WITHOUT_BEHAVIOURAL_RECEIPT` with the reason
"source_id not covered by itchyshin/GLLVModels.jl#684 item 2". Adding a row is a new ruling, reviewed as
a diff to both lists. Tests fail if the two lists differ, if the list does not have 59 entries, if it
contains `CI-ROUTE-008` to `-011`, or if it stops tying to `case-map-inference.json` (every listed id is
a row there whose tier is `routing_control_flow`, `reject_error_class` or already `behavioural`; every
`routing_control_flow` or `reject_error_class` row is listed; no row outside the list is behavioural).
On the scoreboard the same list applies by scoreboard id (the `source_id` with every run of other
characters turned into `-`).

A receipt carries a top-level `behaviour` block:

```json
"behaviour": {
  "pin": "P1",
  "cases": [
    { "case_id": "CORE070-...",
      "source_id": "inference/CI-ROUTE-001",
      "kind": "route",
      "r_observed": "<label>",
      "julia_observed": "<label>" }
  ]
}
```

- `pin` is `P1` or the full P1 sha. `kind` is `route`, `refusal`, `error_class` or `printed_fields`.
- `source_id` is optional. When present the entry applies only to the row with that `source_id`.
- `r_observed` and `julia_observed` are visible strings (see "Visible text" below), or non-empty
  arrays of visible strings of the same length (one label per printed field). Each label is what that
  engine produced.

**Trust model.** A label is a typed string. This tool trusts it the way it trusts a typed `r_value` or
`julia_value` in a numeric receipt: it checks shape and agreement, not provenance, and it does not read
the raw run files. What ties a label to the raw files is the receipt writer's own `--check` (for the
core070 writers, such as `tools/core070_inference_p1_receipts.py --check`, it re-hashes every
`read_from` file and re-derives the receipt from the tracked raw files), together with review of the
writer. A hand-typed receipt with two identical invented labels binds if nobody runs a writer's check,
so a behavioural receipt must come from a writer that has one.

**Matching is by class identity, not by canonical string.**
`docs/dev-log/core070/true-parity-latest/behaviour-equivalence.json` lists, per kind, classes of labels
that the two engines use for the same behaviour:

```json
{ "schema": 1, "pin": "P1", "classes": [
  { "kind": "route", "canonical": "wald", "r": [".confint_lambda:wald"], "julia": ["wald_packed"],
    "basis": "one sentence citing the R function and the Julia function that implement the same route" } ] }
```

A label listed in a class (of that kind, on that side) stands for that class. A label in no class
stands only for itself. Two raw labels match iff they are the same string, or both are listed in the
same class. The `canonical` field names a class in reports; it is not a label. So R's raw `wald` (in no
class) does not match Julia's `jl_wald` because a class called `wald` lists `jl_wald`; an unrelated R
`stop` does not match Julia's `throw_X` because a refusal class called `stop` lists `throw_X`; and when
both engines emit the same literal string, for example `r_wald`, they match even if it is listed on one
side only. Arrays are compared element by element. A label listed in two classes of the same kind and
side makes the table ambiguous and the run `MEASUREMENT_FAILED` (exit 2). Every class needs a visible
`basis`. Two classes of one kind may not share a `canonical` (they would be indistinguishable in a
report): add the labels to the existing class. A wrong `schema` or `pin`, a malformed class, an
ambiguous label and a duplicate canonical are all `MEASUREMENT_FAILED` on C1 and C8 even when no
behavioural row exists yet, and the assembler fails on the same table.

**Visible text.** One definition, in both ports (`isVisible` in the checker, `is_visible` in the
assembler): a string is visible if it has at least one character in Unicode category L, N, P or S. A
string of only whitespace or format characters (U+FEFF, U+200B, U+0085, U+00A0, ...) is empty. This
applies to an observed label, a class's `canonical`, `basis` and labels, a C6 `basis` and a C6 ruling
`ref`. JS `trim()` and Python `strip()` disagree on those characters, so neither is used. Each runtime
uses its own Unicode database, so a character assigned in a very recent Unicode version could read
differently in the two; the controls use characters assigned long ago. A one-character visible text such
as `.` is visible: the rule is visibility, and whether a basis is a real reason is for review.
**Strict types.** `schema` must be the JSON number 1 (`true`, `"1"`, `[1]` and `null` are refused in both
ports; the checker cannot tell `1` from `1.0`, so `1.0` is accepted in both).

A row with `evidence_tier: "behavioural"` binds when all of these hold:

1. `executable_case_ids` is non-empty and every `evidence.receipt` resolves to a file.
2. The carry is fresh (`measured_against` P1).
3. Every cited receipt's `behaviour` block is well formed and pinned to P1. A malformed block fails
   the row.
4. Every executable case id has at least one applicable entry. An entry applies when its `case_id`
   matches and either its `source_id` is the row's or it has no `source_id` and **only one row** of the
   case map being checked (any tier) lists that case id. When several rows cite the same case id, an
   entry without `source_id` covers none of them and each row needs its own `source_id`-scoped entry.
   (The inference receipts share case ids across rows, for example
   `CORE070-INFERENCE-LAMBDA-CI-METHOD-ROUTE` serves `CI-ROUTE-001` to `-004`, each a different route, so
   they must scope.)
5. Every applicable entry matches, by class identity.
6. No cited receipt shows a failure. Each of `status`, `verdict`, `batch_status`, `harness_pass` and
   `result` that is present must hold a pass value (`"PASS"`, `"pass"` or `true`), at the top level of
   the receipt, in its `behaviour` block, in its `comparison` block, and on each behaviour case; a
   `batch_verifier`, if present, must be an object whose `status`, if present, is a pass value; a
   behaviour case's `match`, if present, must be `true`. Anything else (`"FAIL"`, `null`, an object,
   `false`, `0`) fails the row. A missing field is not a failure. `receipt_status_exception` does not
   apply to a behavioural row: it waives a failed status for numeric rows only, so a behavioural receipt
   that reads as failed has to be regenerated.
7. The row's `source_id` is in the frozen list above.
8. If a cited receipt also carries a `comparison` block, that block holds under the numeric rule
   (pinned to P1, every case within tolerance). A numeric failure is not hidden by relabelling the
   row.

The numeric tier's status list is deliberately unchanged by this ruling (`status`, `verdict`,
`batch_status`, `harness_pass`, top level and `comparison` block). `result` is read for behavioural
receipts only. Measured on the tracked ledger at the time of the change: adding `result` to the numeric
tier would change no bound numeric row. Six receipts carry a top-level `result` of `FAIL`, all namespace
case receipts; two are cited by registration-only rows (`extract_proportions`, `extract_residual_split`)
and the other four are kept un-cited under `registration_receipts` of numeric rows on purpose.

It counts in the C1 counter `bound_behavioural=`, never in `bound=` or `bound_numeric=`. C8 accepts
it as twinned (behaviourally). A behavioural label that fails the rule is reported as
`BEHAVIOURAL_LABEL_WITHOUT_BEHAVIOURAL_RECEIPT` (C1 list
`behavioural_label_without_behavioural_receipt=`, C8 failing tag of the same name) with the reason.
The scoreboard status is `EVIDENCED-BEHAVIOURAL` (or `BEHAVIOURAL-UNVERIFIED` with a reason). The
checker counts `EVIDENCED-BEHAVIOURAL` as done and prints `done_behavioural=` on C2 to C5 and X2,
with two limits. It counts only on a row in the frozen list (an `inference-` scoreboard id from the
list, or the slug of a named C1 row), and it never counts on a C3 (`-RSZ`), C4 (`RD-`) or C5 (`GRP-`)
row: those are numeric campaign clauses and read `BEHAVIOURAL_NOT_ALLOWED_FOR_THIS_ROW`. The assembler
emits the status only for rows in scope. A scoreboard row is label-only for the checker; what ties it to
the case maps is `python3 tools/true_parity_assemble.py --check`, which CI runs.

Negative controls (group "behavioural"): matching labels bind and never touch `bound_numeric`;
route mismatch with no class; mismatch rescued by a class; a class of another kind does not
rescue; ambiguous table (exit 2); empty basis (exit 2); case id not covered; entry scoped to
another `source_id` does not cover (and one scoped to the row does); a scoped mismatch fails even
beside a matching unscoped entry; receipt verdict FAIL (top level and in the block); tier
behavioural with only a numeric block; wrong pin; empty or blank label; array length mismatch;
invalid kind; stale carry; dangling receipt; no case ids; an unknown tier stays registration-only.
Scope controls (group "behavioural scope"): rows outside the frozen list do not bind (including
`CI-ROUTE-008` to `-011`, a bare `inference/`, `inference/../isdm/X`, an unlisted new id, a trailing
space and a case variant of a listed id, on the checker and on the scoreboard), each named row and
listed row does, all 59 listed rows bind in one assembler run while the four others do not; a cited
receipt with an out-of-tolerance `comparison` does not bind a relabelled row; a case-level `verdict`
FAIL does not bind. Review-round controls: class identity (the four external-review cases F8a, F8b,
F8e and F8f, two classes, arrays); visible text (U+FEFF, U+200B, U+0085, U+00A0, U+2060, blanks, on
labels, table fields and C6 text, with the same verdict in checker and assembler, and a probe-character
agreement test between the two definitions); strict `schema`; failure detection (`result`, nested
`batch_verifier.status`, `comparison` status fields, case-level `status` and `match`); unscoped entries
(two rows sharing a case id, one scoped, a numeric row as a second citer); and the numeric tier
unchanged. Mutation controls: the carry check inside the behavioural check (C8, row outside C1
scope), side-specific matching (an R label in the Julia slot), and the equivalence table's `schema`,
`pin`, ambiguity, duplicate-canonical and empty-basis checks with no behavioural row present.
Scoreboard controls: `EVIDENCED-BEHAVIOURAL` counts as done with `done_behavioural=1` on a listed
`inference-` row; it does not close C3, C4 or C5, and does not count on a row outside the list;
`BEHAVIOURAL-UNVERIFIED` does not count. Assembler: `behavioural_*`, `scope_*`, `label_identity_*`,
`visible_*`, `schema_*`, `unscoped_*`.

What this does not do: no row changes tier in the PR that adds the rule. Receipts and the
equivalence classes arrive with the PRs that measure the rows.

### Ruling 3: C6 decisions (item 3)

Julia-only exports that are internal helpers are excluded from C6 (`EXCLUDED_INTERNAL_HELPER`).
Julia-only exports that are documented user-facing extras are signed as documented Julia extras
(`KEPT_AS_JULIA_EXTRA`). The input is
`docs/dev-log/core070/true-parity-latest/reverse-gap-decisions.json`:

```json
{ "schema": 1,
  "ruling": { "ref": "itchyshin/GLLVModels.jl#684 item 3", "signed_by": "Shinichi Nakagawa", "signed_on": "2026-10-02" },
  "criterion": "the mechanical rule used, in one paragraph",
  "generator": "path of the committed script that produced the file",
  "decisions": { "<julia export name>": { "decision": "KEPT_AS_JULIA_EXTRA", "basis": "evidence" } } }
```

`tools/true_parity_assemble.py` reads it (optional file), copies `decision`, `basis` and the ruling
(`ref`, `signed_by`, `signed_on`) onto each matching item of `reverse-gap.json`, and sets its
`status` to `decided`. A name in `decisions` that is not a reverse-gap item fails the run (stale).
Names with no clear classification stay out of the file and remain undecided.

The checker's C6 vocabulary is `KEPT_AS_JULIA_EXTRA`, `PORT_TO_MATCH_R`, `DEPRECATE_AND_REMOVE`,
`RENAME_TO_AVOID_COLLISION`, `EXCLUDED_INTERNAL_HELPER`. A decided item also needs a `basis` that is
visible text (see "Visible text" in ruling 2) and a `ruling`. The ruling `ref` must be `itchyshin/GLLVModels.jl#684 item 3`, the only
ruling the tool recognises (`C6_RULINGS` in the checker, copied into the assembler); `signed_on`
must be `2026-10-02`, its date; and `signed_by` must pass the signature rule (allow-list, real past
date). That ruling covers `KEPT_AS_JULIA_EXTRA` and `EXCLUDED_INTERNAL_HELPER` only. A decision
`PORT_TO_MATCH_R`, `DEPRECATE_AND_REMOVE` or `RENAME_TO_AVOID_COLLISION` is listed under
`unsigned_decision=` (no signed ruling covers it) and C6 fails. A new ruling is a new entry in
`C6_RULINGS`, added in review. The assembler refuses to copy a ruling it does not recognise (unknown
`ref`, wrong date, signer outside the allow-list), a decision word the ruling does not cover, and a
file with an empty `criterion`, a `generator` that is not a file inside the tree (an absolute path, a path
with a `..` segment anywhere, such as `tools/../tools/gen.py`, and a directory are refused; only the
assembler reads the generator, the checker does not), a `basis` with no visible character, or a
`KEPT_AS_JULIA_EXTRA` basis that fails the documented-extra rule below. C6 also prints
`decision_counts=` per vocabulary word.

Basis rules. An `EXCLUDED_INTERNAL_HELPER` basis must be visible text. A `KEPT_AS_JULIA_EXTRA` basis must also
cite at least one page under `docs/src`, and every page it cites must be an existing `.md` file written exactly as
`docs/src/<path>.md`. The basis is read as tokens: it is split at ASCII whitespace and at `( ) [ ] { } < > " ' `
`` ` `` `, ; : ! ? #` (so `(docs/src/a.md)`, `docs/src/a.md#sec` and `docs/src/a.md:12` cite `docs/src/a.md`), trailing
dots are dropped, and every token that contains `docs/src` must then be a page path: it starts with `docs/src/`, has
no `..`, `.`, dot-leading or empty segment, ends in `.md` and has nothing after that. A token that is not such a path
fails the basis even beside a good citation: `docs/src/page.md.bak`, `docs/src/page.md~`, `./docs/src/page.md`,
`other/docs/src/page.md`, a URL, `docs/src/` alone, and an existing `.json`, `.txt`, `.jl` or `.toml` file under
`docs/src`. Each cited page must resolve to a file at the ref (the checker, `git cat-file`) or in the tree (the
assembler). The delimiter set and the page pattern are the same text in both tools (`DOCS_SRC_SPLIT_RE`,
`DOCS_SRC_PAGE_RE`) and a control runs both on one corpus. The reasons print as
`unsigned_decision=<id>(KEPT_AS_JULIA_EXTRA basis must cite a docs/src/... file[; not an exact docs/src/<path>.md page: <tokens>])`,
`(KEPT_AS_JULIA_EXTRA basis cites <tokens>, which is not an exact docs/src/<path>.md page)` and
`(KEPT_AS_JULIA_EXTRA basis cites <path>, which does not resolve at the ref)`. Case is not checked beyond what the
file system or git gives: on a case-insensitive file system the assembler can accept a page whose case differs.

Reverse-gap matching. `reverse-gap.json` lists the Julia exports that have no gllvmTMB counterpart. A
Julia export has one when its name equals an R export or S3 generic name after removing `_` and `.`
and lower-casing (the `norm()` of `tools/parity_ledger.py`), or when it is the Julia side of a tracked
namespace receipt. `ZiPoisson` and `zi_poisson`, `Lognormal` and `lognormal` are the same name. The
hand-made `ALIASES` table of `tools/parity_ledger.py` (for example `betabinomial` to `BetaBinom`) is
not applied here: those are mapping decisions, not name matches.

Negative controls (group "C6"): `EXCLUDED_INTERNAL_HELPER` with basis and ruling is valid; no
ruling; ruling without a `ref`; ruling signed by an agent name; signer outside the allow-list;
future date; no signer; no basis; unknown decision word (`EXCLUDED_HELPER`, a case variant); an
undecided item still fails. Scope controls (group "C6 scope"): unrecognised ruling ref, a bare
`#684`, `PORT_TO_MATCH_R`, `DEPRECATE_AND_REMOVE` and `RENAME_TO_AVOID_COLLISION` under the item 3
ruling, a `signed_on` other than 2026-10-02. Basis controls (group "C6"): a `KEPT_AS_JULIA_EXTRA` basis
with no path, `.`, a path outside `docs/src`, a `..` path, a dot-leading segment, a missing file, one
present and one missing file, and a directory; a `KEPT_AS_JULIA_EXTRA` basis citing an existing file
binds, also in git mode; an `EXCLUDED_INTERNAL_HELPER` basis, a `KEPT_AS_JULIA_EXTRA` basis and a ruling
`ref` made only of U+FEFF, U+200B, U+0085 or blanks. Assembler: a generator that does not exist, is a
directory, is outside the tree, is an absolute path or has a `..` segment that resolves back into the tree; the same basis controls; reverse-gap names that
match after `norm()`; `decisions_*`,
`decision_for_non_item_is_stale_and_fails`, `decisions_file_*_fails`, and `c6_*_matches_checker`
(the two tools' copies of the vocabulary, ruling table and named rows must not drift).

### Ruling 4 (item 4)

Add the C3 to C5 rows proposed in PR #650 (the beetle row included now that #662 is merged) and run
the campaign on Totoro. This is data, not a rule: those PRs add rows and receipts, and the checker
selects them by the existing `-RSZ`, `RD-` and `GRP-` id conventions below.

### Hardening after the independent review of #687

The review of the merged PR #687 left the follow-ups below. Each has controls that fail on the code before the
change. None is a new ruling or a new signature.

**A disposition is never a status word.** The Status column of `scoreboard.md` is written by
`tools/true_parity_assemble.py`; a case-map row's `disposition` is free text. Before this change a non-null
`disposition` that was not a valid signature was copied into the Status column as it stood, so a row with
`"disposition": "EVIDENCED"` and a receipt that exists read as done, and X2 counted it. Now
(`disposition_status`) a disposition that equals a reserved status word reads `DISPOSITION-UNVERIFIED`, with the
reason `reserved status word`. The reserved words (`RESERVED_STATUS_WORDS`) are every status the assembler writes:
the entries of `STATUS_ORDER` and the buckets of `TIER_BUCKET`. The comparison is made after trimming (with the
characters of both JS `trim()` and Python `strip()`) and upper-casing, so `" evidenced "`, `"EVIDENCED"` followed by
U+FEFF and `"EVIDENCED-BEHAVIOURAL"` all match. A disposition also reads `DISPOSITION-UNVERIFIED` when it is not a
string, is blank, or contains `|`, a line feed or a carriage return (a `|` shifts the columns the checker splits a
row into). Text that contains `NEEDS_JULIA_SURFACE` still reads `NEEDS-SURFACE`, and any other plain string is still
copied into the column as it is; the checker counts none of those as done. The statuses `DISPOSITION-SIGNED`,
`EVIDENCED` and `EVIDENCED-BEHAVIOURAL` come only from the signature path and the binding rules.

**The checker agrees.** C2 to C5 and X2 read the Status word of a scoreboard row, and `--check` is what ties the word
to the case maps. The checker now also refuses a done word (`EVIDENCED`, `EVIDENCED-BEHAVIOURAL`,
`DISPOSITION-SIGNED`) on a row whose receipt cell begins with `not bound`, which is how the assembler writes every row
it did not bind; the row reads `STATUS_NOT_BOUND`. A scoreboard written by hand with another receipt cell is not
affected. The tracked scoreboard has no done row with such a cell, so no count changes.

**A ruling ref is looked up by own property.** The checker finds a C6 ruling with
`Object.prototype.hasOwnProperty.call(C6_RULINGS, ref)`. A lookup with `in` would accept the names every object
inherits (`constructor`, `__proto__`, `toString`) and then crash with exit 1 and no verdict. Each such ref is
reported as `ruling ref "<ref>" is not a recognised signed ruling` with `C6_NOT_MET` and exit 0, in the checker
(group "C6 scope") and in the assembler (`c6_ruling_ref_*_is_not_a_recognised_ruling`). The control was checked by
replacing the lookup with `in`: it fails, and the 29 other C6 controls that existed before it all still passed.

**The generator path refuses `..` segments.** `reverse-gap-decisions.json` names the committed script that wrote
it in `generator`. The assembler refuses the path when any segment, split at a slash or a backslash, is `..`, even when the path
resolves to a file inside the tree (`tools/../tools/gen.py`). It still refuses an absolute path, a path that resolves
outside the tree, a directory and a missing file. Controls: `c6_generator_with_a_dotdot_segment_is_refused_*`, and
`c6_generator_with_a_plain_in_tree_path_still_ok` for the positive case.

**The cited `docs/src` page is matched exactly.** See "Basis rules" under ruling 3. Before this change a basis
cited a page when the text merely contained one: `docs/src/page.md.bak` cited `page.md`, `other/docs/src/page.md`
counted, and an existing `docs/src/assets/x.json` counted. Controls: checker group "C6 exact page" (also in git mode),
assembler `c6_kept_basis_exact_page_*`, and `kept_basis_accept_and_refuse_agree_between_checker_and_assembler`.
The tracked decisions (261 `KEPT_AS_JULIA_EXTRA`, 57 `EXCLUDED_INTERNAL_HELPER`) all cite plain `docs/src/*.md`
pages, so C6 reads the same decided count as before.

Controls. Assembler: `disposition_status_word_*`, `disposition_pipe_forged_*`, `disposition_with_*line_break*`,
`every_reserved_status_word_*`, `non_string_or_blank_disposition_*`, `plain_dispositions_*`,
`reserved_status_words_cover_*`, `checker_done_set_is_a_subset_*`. Checker (group "scoreboard"): a done word on a
`not bound` row is not done for each of the three words (X2 and C2), the same words with a bound cell are done, and
a non-done word stays not done.

## Rulings of 2026-10-05 (maintainer ruling 2026-10-05, D-319)

On 2026-10-05 the maintainer, Shinichi Nakagawa, approved every recommendation on the rulings page of the
true-parity lane (vault decision D-319). Verbatim: "Approve all recommendations on the rulings page and the C6
high-confidence group. D: [yes]. animal_latent: leave unbound. C4: change the text. Review C6 medium and low rows
with me later." The signature source is `signed-rulings-2026-10-05.md` in the lane kit
(`~/local-scratch/lanes/GLLVM.jl-true-parity-latest/LOOP/lanes/true-parity-latest/`), which quotes each
recommendation. Cite this section as "maintainer ruling 2026-10-05 (D-319)", with the item name. As for the rulings of
2026-10-02, nothing here is a new signature: the tool cannot check that the named person signed, and PR review does
that by reading the diff. Each rule that changes a check is ported to `tools/true_parity_assemble.py`, so the
scoreboard status and the checker agree. The C6 items are recorded with the C6 decisions, not here.

### Ruling 1: bridge readback (the 8 `*-JULIA-BRIDGE-COMPARE` rows)

Question signed: may "R's `gllvmTMB_julia` method returns exactly what Julia computed" bind the eight
`namespace/S3method/<method>,gllvmTMB_julia` rows (`coef`, `fitted`, `logLik`, `predict`, `residuals`, `summary`,
`confint`, `simulate`)? The live readback receipt is
`receipts/julia-twins/namespace-numeric/bridge_readback.json` (evidence kind `live_bridge_readback`, PR #790). The
signed answer differs by method, because the strength of the comparison differs:

- `fitted`, `predict` and `residuals`: yes, as numeric evidence. Each R method does its own arithmetic on what Julia
  returned (intercepts plus the linear predictor, the response scale, the residual types), so comparing it with
  GLLVModels on the same inputs is a real test of R's adapter. The row binds on the readback cases of its own method
  only: `P1-BRIDGE-READBACK-FITTED-*`, `-PREDICT-*`, `-RESIDUALS-*`.
- `coef`, `logLik`, `summary`, `confint` and `simulate`: no numeric binding. These R methods copy what Julia
  returned, so the comparison is circular (the 2026-09-30 audit and #561). Each closes by a signed disposition that
  says the R value comes from Julia (`disposition` `DISPOSITION-SIGNED`, `signed_by` the maintainer, `signed_on`
  2026-10-05, and the reason in `disposition_basis`). `simulate` has no readback case and closes the same way.

The checker and the assembler enforce the split. A numeric row that cites a receipt whose `evidence_kind` is
`live_bridge_readback` binds only if its `source_id` is one of `namespace/S3method/fitted,gllvmTMB_julia`,
`predict,gllvmTMB_julia`, `residuals,gllvmTMB_julia` (and `namespace/export/gllvm_julia_fit`, which already binds on
the same receipt through cases that compare against the TMB engine), and only if every executable case id starts
with that row's prefix (`P1-BRIDGE-READBACK-FITTED-`, `-PREDICT-`, `-RESIDUALS-`, `-GJF-`). Otherwise the row reads
`NUMERIC_LABEL_WITHOUT_NUMERIC_RECEIPT` with the reason "bridge readback binds only ...". So relabelling the `coef` row
numeric on the readback receipt does not bind it, and a `fitted` row cannot borrow the `COEF` cases.
(`BRIDGE_READBACK_ROW_PREFIX` in both tools.)

Row-level edits are in `case-map-namespace.json` and belong to the covariance and namespace slice: three rows to
`evidence_tier` numeric citing the readback receipt with their own cases, five rows to a signed disposition.

Negative controls (checker, group "bridge readback"): `fitted` on its own cases binds; `coef` relabelled numeric on the
readback receipt does not; `fitted` citing a `COEF` case does not; a row outside the four does not; a receipt of
another evidence kind is not affected. Assembler: `bridge_readback_*`.

### Ruling 2: namespace keyword rows may bind without the bridge leg

Signed yes: a namespace export row for a covariance keyword may bind on its native-model and formula-interface fits
while its covariance row stays unbound at the public-R-bridge boundary. Rows named in the ruling: `animal_dep`,
`animal_indep`, `kernel_dep`, `kernel_indep` (receipts from #786), and probably `Beta` and `dep`. Reason, from the
ruling page: these exports are formula markers, so the fit is the estimand, and the bridge refusal is an R-side
route. This reverses the rule of the 2026-09-30 audit (`audit-unwired-evidence-2026-09-30.md`, row
`namespace-export-Beta`): "an export-constructor row should not borrow a case its own family row cannot yet bind". No
tool enforced that rule, so no code changes; it was applied by hand when rows were mapped, and this ruling withdraws
it. The binding itself still needs the numeric rule: a receipt with a comparison block, pinned to P1, every case
within tolerance. The row edits belong to the covariance and namespace slice.

### Ruling 3: CI-ROUTE-029, Julia's default rho interval is R's `fisher-z` default

Signed yes: for rho, Julia's default `confint` route (`method = :wald`, which for a correlation computes a Fisher-z
transformed Wald interval, label `rho:wald_derived`) is the same default route as R's (`.confint_rho:fisher-z`).
"Same interval, different name." It is recorded as one equivalence class in `behaviour-equivalence.json` (route
`rho:fisher-z`, R `[".confint_rho:fisher-z"]`, Julia `["rho:wald_derived"]`), written by
`tools/core070_behaviour_receipts.py` (`CLASSES`), whose `default_class_unconfirmed` hold on CI-ROUTE-029 is removed.
R's own `method = "wald"` for rho (`.confint_rho:wald`, probe row CI-ROUTE-031) is a different interval and is not in
the class, so R's plain Wald still does not match Julia's Fisher-z. `inference/CI-ROUTE-034` (an explicit `fisher-z`
request, which Julia's public `confint` refuses at the run commit) joins the same class once the inference slice adds
a `fisher_z` method alias (ruling page item "CI-ROUTE-034", recommend the alias, signed): its Julia label is then added
to this class, never a new class. CI-ROUTE-029 binds once `case-map-inference.json` is regenerated.

Negative control (checker, "ruling 3 (tracked table)"): with the tracked `behaviour-equivalence.json`, R
`.confint_rho:fisher-z` against Julia `rho:wald_derived` binds on CI-ROUTE-029, and R `.confint_rho:wald` against the
same Julia label does not ("R: no class; Julia: class rho:fisher-z").

### N1: admission-only and PUBLIC-R-BRIDGE boundary cases are non-binding context

Signed yes ("about 13 rows"). Some rows carry a case that can never produce an R-versus-Julia number: R refuses at
its public Julia bridge before any Julia call (the capability guard and kernel guard of `R/julia-bridge.R`, the
truncated-NB2 gate; receipts with evidence kind `r_public_bridge_boundary`, verdict `R_BOUNDARY_UNCHANGED`), the
bridge case was not executed for that reason (`not_executed`, `NOT_EXECUTED`), or the R case only admits a formula
grammar (`r_only_formula_grammar`, `R_ONLY_PASS`). Under the numeric rule every executable case id must be compared,
so such a case blocked its row even when every fit case agreed. The rows it affects: the 7 covariance rows
`COV-ANIMAL-DEP`, `COV-ANIMAL-INDEP`, `COV-KERNEL-DEP`, `COV-KERNEL-INDEP`, `COV-ORD-DEP`, `COV-ORD-INDEP`,
`COV-ORD-INDEP-COMMON` (9 bridge cases), `family/FAMILY-00-IDENTITY` and `family/FAMILY-11-LOG`, plus rows whose
formula-grammar case was already moved to `non_binding_receipts` by hand (six covariance rows in #810).

Rule (checker `boundaryContext`, assembler `boundary_context`). A numeric row may list such case ids under
`boundary_context_case_ids` and cite their receipts under `evidence.boundary_context_receipts`. A listed case is
exempt from the coverage requirement only if:

- it is also in `executable_case_ids` (it stays visible on the row), listed once;
- its own receipt resolves to a file, is JSON, has that `case_id`, and its `evidence_kind` and `verdict` are one of
  `r_public_bridge_boundary` + `R_BOUNDARY_UNCHANGED`, `not_executed` + `NOT_EXECUTED`, `r_only_formula_grammar` +
  `R_ONLY_PASS`; the two bridge kinds only on a case id ending `-PUBLIC-R-BRIDGE`;
- its receipt carries no `comparison` block (a case that has a number is compared, not set aside);
- at least one other executable case binds numerically under the unchanged rule.

Otherwise the row reads `NUMERIC_LABEL_WITHOUT_NUMERIC_RECEIPT` with a reason that starts "boundary context
(maintainer ruling 2026-10-05 (D-319), N1)". A row without the field is judged exactly as before. The context receipts
are not read by the receipt-status check, because their verdicts are not pass values by design. The row still counts
in `bound=` and `bound_numeric=`: what binds it is the compared cases, and the context case is recorded, not
compared. Note on FAMILY-00: its R bridge silently fits a different model (df 5 against 8); the rule sets the case
aside but does not make it a twin, and the row also needs its formula-interface case to pass.

Row-level edits belong to the slices that own `case-map-covariance.json` and `case-map-family.json`: add the field and
move each boundary receipt from `non_binding_receipts` or `receipt` to `boundary_context_receipts`, and cite the
fit-case receipts under `evidence.receipt` with `evidence_tier` numeric.

Negative controls (checker, group "N1 boundary context"): a bridge refusal, a not-executed bridge case and an
admission-only case each bind as context; without the field the uncompared case still blocks the row; a context
receipt of another kind, a wrong verdict, a bridge kind on a non-bridge case, a context receipt with a comparison
block, every case set aside, a context id outside `executable_case_ids`, a missing context receipt and a missing
`boundary_context_receipts` each fail. Assembler: `boundary_context_*`.

### N9: convergence parity for C3 campaign rows

Signed yes, as drafted in PR #715 (`docs/dev-log/w1-11-convergence-parity-rule-DRAFT.md` on that branch): a C3
comparison counts only when both engines reach gradient max-abs 1e-5 at the point whose outputs are compared. The
draft's reason: on the ordinal cell R's `nlminb` stops at gradient max-abs 4.3e-4 and Julia at 7.8e-6 on the same
objective, so the row measured how early each optimiser stopped; after one Newton step each (R 4.7e-7, Julia 2.4e-6)
the loadings product agrees to 1.85e-7 against the unchanged tolerance 1e-4. No tolerance changes.

Rule (checker `convergenceParityProblem`, assembler `convergence_parity_problem`). A receipt records the condition in a
top-level block:

```json
"convergence_parity": {
  "gradient_bound": 1e-5,
  "compared_point": "returned" | "newton_polished",
  "engines": { "R": { "max_abs_gradient": 4.7e-7 }, "julia": { "max_abs_gradient": 2.4e-6 } }
}
```

When a cited receipt carries the block, the bound must be exactly 1e-5, `compared_point` one of the two words, and
both gradients finite, at least 0 and at most 1e-5; otherwise the row does not bind (reason "convergence parity
(maintainer ruling 2026-10-05 (D-319), N9)"). Each gradient is on that engine's own objective in its own coordinates,
as the draft says.

Two readings, stated rather than chosen silently:

- **Applied (this change): the rule governs receipts that record it.** A C3 receipt without the block is judged as
  before. Reason: the ruling page says N9 "binds ORDINAL-LOGIT-RSZ and, after a re-run, ISDM-HEADLINE-RSZ", that is, it
  was signed to let the stopped-early rows be compared at a converged point, not to reopen the rows that bind.
- **Not applied: the rule as a precondition for every C3 row.** Measured on the tracked receipts, all five C3 rows that
  bind today have an R gradient above 1e-5 (Gaussian 1.97e-2, Poisson 3.80e-3, NB2 3.76e-3, binomial 9.86e-4,
  temporal 4.35e-2) and four record no Julia gradient at all. Reading the rule strictly would unbind all five (C3 from
  5/8 to 0/8) until the campaign is re-run with the polish. That is the maintainer's call; the change to the checker
  is one line (require the block on every `-RSZ` row).
- The draft's step-3 alternative (a Newton step size `max |H^-1 g|` at most 1e-5 when an engine's gradient noise
  floor sits above the bound) is not part of the signed text and is not accepted by the checker.

The campaign receipts do not carry the block yet. `tools/true_parity/campaign/write_receipts.py` (and `run_J.jl`,
which must record Julia's gradient) belong to the campaign slice: ORDINAL-LOGIT-RSZ binds once its receipt records the
polished comparison with the block (the PR #715 pre-run numbers pass), and ISDM-HEADLINE-RSZ after a re-run.

Negative controls (checker, group "N9 convergence parity"): the ordinal pre-run numbers bind, as do exactly 1e-5 and a
`returned` point; R's unpolished 4.3e-4, Julia above the bound, a declared bound of 1e-4, a missing Julia gradient, an
unknown compared point, a negative gradient and a non-object block each fail. Assembler: `convergence_parity_*`.

### C4: the clause accepts direct-engine runs

Signed: "C4: change the text." C4 no longer requires the `engine = "julia"` bridge leg; a real-data workflow run end to
end on both engines directly (gllvmTMB at P1 and GLLVModels.jl on the same data bytes) satisfies it. The ruling page's
reason: the bridge leg tests R's adapter rather than the model, and the adapter is covered by ruling 1. The clause text
under "Clauses" is changed accordingly.

Checker change: an `EVIDENCED` real-data (`RD-`) scoreboard row is done only if a receipt it cites records both
engines (a JSON receipt with an `engines` block holding an `R` and a `julia` object), else `C4_NOT_A_DIRECT_ENGINE_RUN`
on C4 and X2. That makes "end to end on both engines" a checked condition instead of a word in a cell. A
`DISPOSITION-SIGNED` real-data row is not a run and is judged as before. Every tracked campaign receipt already has
the block.

What still keeps the real-data rows open is the receipt, not the checker: `write_receipts.py` writes a pass-rule leg
`engine_julia_bridge_route_and_eight_acceptance_classes_run` that is false on every C4 receipt. Dropping that leg is
the campaign slice's edit; then `data/RD-CRABS-GAUSSIAN` (every number inside tolerance today) binds. Spider, beetle,
fungi and urbanisation still fail on numbers or convergence, and the three disposition rows wait on their signed
dispositions (item F, owned by the covariance and namespace slice).

Negative controls (checker, group "C4 direct engine"): the base fixture's RD row (receipt with both engines) is done; a
receipt with no engines block, only the R engine, or an array in place of the block is not done on C4 and X2; a signed
RD disposition row is done; a C3 row with a plain receipt is unaffected. Every fixture copy of the RD receipt gained
the engines block.

### D: phylo-latent promotion (D-300 answer 9) is signed

Signed: "D: [yes]". The dated promotion block of D-300 answer 9, which the PR #547 receipts wait on
(`phylo-latent-p1/cov_phylo_latent_rsz/r-receipt.json` records `qualified = false` until it is signed), is signed on
2026-10-05 under maintainer ruling 2026-10-05 (D-319), item D. This file records the signature only. Flipping the
receipts' qualification and the pass-rule leg `R_side_receipt_qualified_by_maintainer` (so that
`covariance/COV-PHYLO-LATENT-RSZ` and the phylo twins can bind) belongs to the covariance and namespace slice; the
"Phylo row does not bind" paragraph under "C3 to C5 campaign rows" describes the state before that flip.

### N3: known-V meta rows are a documented gap, revisited at P2

Signed: "documented gap, revisit at P2". `covariance/COV-META-EXACT` and `covariance/COV-META-LEGACY` (R's known
sampling-covariance `meta` term and its deprecated spelling) have no Julia surface at P1 (grep check recorded in the wave plan of 2026-10-03). They are a
documented gap of P1, revisited when the pin moves to P2; no Julia alias for the deprecated spelling is added, and the
build (estimated 18 to 24 hours, an API change) is not scheduled. The rows close by a signed disposition that says so,
written by the slice that owns `case-map-covariance.json`; until then they read `NON-NUMERIC` (tier `r_only`). Their
R cases are admission-only, so N1 does not bind them: a row needs at least one compared case.

### Item A, N6 and N10: the behavioural scope is extended by 14 listed rows

Ruling 2 of 2026-10-02 froze the behavioural tier to 63 rows. Three signed items extend that list, and only by
explicit ids, never by prefix:

- **Item A** (signed yes): the 7 aghq control rows `aghq/AGHQ-CTRL-AUTO`, `-FALSE`, `-NINE`, `-NULL`, `-ONE`, `-TRUE`,
  `-TWO` and `inference/CI-ROUTE-009`. Each already had a behaviour block, kept as non-binding evidence because the
  row was outside the frozen list.
- **N6** (signed yes: "iSDM behavioural ruling, 3 internal predicates, D-296 written onto ISDM-LEGACY"). The 9 iSDM
  rows whose R side is an admission or refusal predicate split three ways, read from each R case's `r_expression`:
  - behavioural scope, 5 rows whose R case is reachable through `gllvmTMB(..., family = isdm_sources(...))`:
    `isdm/ISDM-COUNT`, `-EXTRA-SOURCE`, `-MISSING-IN-TRAIT`, `-MISSING-SOURCE`, `-WRAPPER-LAW`. Each binds only when a
    receipt shows Julia's public door giving the same refusal or admission (slice W3-2); none does yet.
  - signed disposition, 3 internal predicates: `isdm/ISDM-NO-TRAITS` (calls the predicate with `traits = NULL`),
    `-WRONG-ID` and `-WRONG-LINK` (tamper the family-id or link-id column of the internal row-id matrix). No public
    call on either engine can set those arguments. The ruling names the count, not the ids; the three ids are the
    lane's reading of the receipts (AGENT-INFERRED) and are for the reviewer of this change to confirm.
  - signed disposition, `isdm/ISDM-LEGACY`: the R-only backward-compatibility disposition already signed in Packet 1b
    item 2 (2026-09-27, D-296) is now written onto the row (`signed_on` 2026-09-27, the date of that signature).
  The four dispositions are written by `tools/core070_isdm_p1_receipts.py` (`SIGNED_DISPOSITIONS`), so its
  `--check` keeps re-deriving the rows; each row carries the reason in `disposition_basis`.
- **N10** (signed yes): `postfit/POSTFIT-SURFACE-check_auto_residual`. It binds when a behaviour block shows the same
  verdict from both engines (slice W3-9); its receipt has none yet.

The 14 ids live in `BEHAVIOURAL_EXTENDED_SOURCE_IDS` in the checker and in the assembler, a separate set so the
59-row inference list stays tied to `case-map-inference.json`. Every other rule of ruling 2 is unchanged: a row in
scope still needs a matching, well-formed, passing behaviour block, and being listed binds nothing by itself.
`inference/CI-ROUTE-008`, `-010` and `-011` stay outside. Tests fail if the two copies drift, if the set does not
have 14 unique ids, if it overlaps the other two lists or names one of the four signed-disposition iSDM rows.

Rows that bind on this change: the 7 aghq control rows (`core070_aghq_p1_receipts.py --apply-twins` relabels them,
because their blocks already match). `inference/CI-ROUTE-009` binds once `case-map-inference.json` is regenerated
(`core070_inference_p1_receipts.py --write`), which belongs to the inference slice. Until then that tool's `--check`
reports CI-ROUTE-009 as differing from its re-derivation.

Negative controls (checker, group "behavioural scope"): each of the 14 rows binds with a matching block and fails on a
mismatched label; near misses do not bind (`aghq/AGHQ-CTRL-THREE`, `aghq/AGHQ-AUTO-K-BINOMIAL`, the four
signed-disposition iSDM rows, `postfit/POSTFIT-SURFACE-check_auto`, a trailing space); on the scoreboard,
`EVIDENCED-BEHAVIOURAL` counts on `inference-CI-ROUTE-009`, `aghq-AGHQ-CTRL-AUTO`, `isdm-ISDM-WRAPPER-LAW` and
`postfit-POSTFIT-SURFACE-check_auto_residual` and not on `aghq-AGHQ-CTRL-THREE`, `isdm-ISDM-WRONG-ID` or
`isdm-ISDM-LEGACY`. Assembler: `behavioural_scope_named_row_*` (now over the extended ids too),
`scope_extended_list_has_14_unique_ids_disjoint_from_the_other_lists`, `behavioural_extended_ids_match_checker`,
`scope_list_ties_to_case_map_inference_json` (CI-ROUTE-009 may be partial or behavioural).

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
      receipt that resolves on `origin/main` (numeric evidence, counted in `bound_numeric=`; or, for
      the frozen list of rows itchyshin/GLLVModels.jl#684 item 2 names, behavioural evidence, counted
      in `bound_behavioural=`), or carries a maintainer-signed disposition; `BLOCKED_*`, `PARTIAL_*`
      and `PARTIAL_STALE_AT_P1` do not count as signed; every cited receipt path must exist at the ref
  CHECK: node tools/true_parity_check.mjs C1
  EXPECT: C1_MET
  EVIDENCE: pending (P1 case-map not yet populated; A0c/A0d)

- [ ] C2: every P1 capability inside the D-295 boundary (isdm 1FO plus `predict`, temporal
      1FO, phylo_latent 1FO, ordinal_logit 1FO, zi_1fo under R semantics, and so on) has a
      scoreboard row with a resolving P1 receipt (a row at `EVIDENCED-BEHAVIOURAL` counts as done
      only inside the frozen behavioural list); row count is read from the scoreboard file,
      never hard-coded
  CHECK: node tools/true_parity_check.mjs C2
  EXPECT: C2_MET
  EVIDENCE: pending

- [ ] C3: one realistic-size cell (p >= 20, n >= 500, condition number recorded) per paired
      family/structure inside the boundary
  CHECK: node tools/true_parity_check.mjs C3
  EXPECT: C3_MET
  EVIDENCE: pending

- [ ] C4: one real-data workflow per qualified family or structure runs end to end on both engines
      (gllvmTMB at P1 and GLLVModels.jl, directly, on the same data bytes; the `engine = "julia"` bridge leg is not
      required, maintainer ruling 2026-10-05 (D-319), C4) and passes the campaign pass rule; an `EVIDENCED` row
      must cite a receipt that records both engines
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
      `RENAME_TO_AVOID_COLLISION`, `EXCLUDED_INTERNAL_HELPER`) — a placeholder like `"TBD"` or
      an empty string does not count as decided just because the field is non-empty or present;
      a decided item also needs a visible basis and a signed ruling, and a `KEPT_AS_JULIA_EXTRA`
      basis must cite a `docs/src/...` file that resolves at the ref (see "Rulings of 2026-10-02",
      ruling 3)
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

- [ ] X2: all scoreboard rows are `EVIDENCED`, `EVIDENCED-BEHAVIOURAL` or `DISPOSITION-SIGNED`;
      `EVIDENCED-BEHAVIOURAL` counts only on a row in the frozen behavioural list and never on a C3,
      C4 or C5 row, and is printed as `done_behavioural=`; row count read from the file, an empty
      selection is never a pass
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

## C3 to C5 campaign rows (itchyshin/GLLVModels.jl#684 item 4)

The maintainer's ruling (itchyshin/GLLVModels.jl#684 item 4, 2026-10-02) is to add the rows proposed in PR #650
(beetle included) and run the campaign on Totoro. Twenty rows were added, 8 for C3 (`-RSZ`), 8 for C4
(`data/RD-*`) and 4 for C5 (`fit-input/GRP-*`). The ruling does not quote the plan's tolerances, pass rule, row
placement or licence handling. They are carried here AS PROPOSED in PR #650 and still wait for the maintainer's
confirmation; every receipt says so in `tolerance_status`.

- **`clause` marker.** A case-map row may carry `"clause": "C3" | "C4" | "C5"`. The assembler then requires
  that the row's scoreboard id is selected by exactly that clause's checker rule, so a deliberate campaign id
  (for example `family-NB2-LOG-RSZ`) is accepted, while an unmarked row whose id would silently move into
  C3/C4/C5 still fails.
- **Receipts** live under `receipts/<family>/campaign/`, one per row, with the raw R and Julia outputs beside
  them (`raw/*.gz`). They are written only by `tools/true_parity/campaign/write_receipts.py`, from those raw
  outputs. `write_receipts.py --check` rebuilds each receipt's comparison block, pass-rule legs, verdict and engine
  blocks, and each campaign case-map row, from the committed raw files, and fails on any difference (a hand-edited
  value, tolerance or flag fails). The seven synthetic data files are committed (gzip) under
  `tools/true_parity/campaign/data/` with their sha256, and `--check` verifies the receipts' data pin against them.
  One row is the exception: the urbanisation matrix is unpublished, so `data/RD-URBANISATION-BINOMIAL` has no committed
  raw outputs (see **Urbanisation** below), and `--check` says so on every run instead of re-deriving it.
- **Pass rule.** Both engines converged (R convergence 0 with a positive-definite Hessian; Julia `converged`
  true) and every listed quantity inside its tolerance, on the same data bytes, with the ten named gllvmTMB entry
  points that `run_R.R` calls (`gllvmTMB`, `gllvmTMBcontrol`, `nbinom2`, `ordinal_logit`, `isdm_sources`, `extract_Sigma`,
  `extract_cutpoints`, `predict.gllvmTMB_multi`, `extract_temporal`, `temporal_latent`) deparsing identically to their
  P1 source. That is all the guard checks by deparse. Internal gllvmTMB functions (for example the fitting code in
  `fit-multi.R`) and the compiled library are not deparse-checked: they are trusted by the recorded version 0.7.1, the
  library path and the sha256 of the P1 source files listed in `p1_source_sha256.json`. A drift in an internal function or
  in the compiled code of a lane library would therefore not be caught by the guard. A C4 row also needs the
  `engine = "julia"` bridge route and the plan's eight acceptance classes (plan section 1.3); neither has been run, so
  no C4 row binds yet. (Superseded for C4 by maintainer ruling 2026-10-05 (D-319), C4: a direct-engine run satisfies C4 and
  the bridge leg is no longer required; see "Rulings of 2026-10-05".) A row that meets the whole rule binds (`evidence_tier` numeric).
  A row that does not is cited under `non_binding_receipts` with every reason, and stays open. Its tier says which kind of
  failure it is. If every number is inside tolerance and the only failing leg is a required step that was not run or not
  signed (the C4 bridge leg, the phylo qualification below), the tier is `partial_case_not_executed` (scoreboard PARTIAL)
  and the `evidence.tier` sentence names the missing leg. If a number or a convergence flag also fails, the tier is
  `numeric_fail` (FAIL) and the sentence lists every failing leg and number. No tolerance is widened and nothing is
  re-run to get a pass.
- **C5 rule.** The four grouping rows follow their own rule, taken from the merged PR #593 receipts: name parity PASS
  with the misspelt-keyword negative control rejected by both engines, the #593 receipt's own verdict PASS, R convergence 0
  and Julia `converged` true in the paired fit, the paired logLik within 1e-6, and a replay on current main that verifies
  the fixture hashes, passes name parity again and reproduces the receipt's logLik within 1e-6. There is no R
  positive-definite-Hessian leg for C5: the #593 receipts do not record one.
- **Phylo row (`covariance/COV-PHYLO-LATENT-RSZ`) does not bind.** Its R side is the tracked PR #547 receipt
  (`phylo-latent-p1/cov_phylo_latent_rsz/r-receipt.json`), which records `qualified = false`; the README there says every
  receipt stays unqualified until the maintainer signs the dated promotion block (D-300 answer 9). The measurement is kept
  as a non-binding receipt (both numbers are inside tolerance), the pass rule has a leg
  `R_side_receipt_qualified_by_maintainer` that is false, and the row reads PARTIAL. No agent may sign the block. (The block was signed on 2026-10-05, maintainer ruling 2026-10-05 (D-319), item D; the receipt flip
  follows in the covariance slice. See "Rulings of 2026-10-05".)
- **Relative tolerances** (standard errors, NB2 dispersion) are carried as a discrepancy against zero (`r_value` 0,
  `julia_value` the relative difference), because the checker compares absolute differences. The `r_value` is therefore
  not an R measurement: the case's quantity name says "relative difference", a `convention` field says so, and
  `raw_values_location` names where the raw R and Julia values are (`r_raw` and `julia_raw` in the same case block, and
  the committed raw files). Whether to keep this encoding is a question for the maintainer.
- **cond(H).** R's is `kappa(solve(sdr$cov.fixed), exact = FALSE)`, a 1-norm condition-number estimate of the inverse of
  TMB's sdreport covariance, in gllvmTMB's own parameter coordinates. Julia's is the exact 2-norm condition number
  (eigenvalue ratio) of `vcov(fit, Y)`, the inverse observed information of the Julia fit, in the Julia fit's own
  coordinates. For the temporal row Julia's is the exact eigenvalue ratio of the ForwardDiff Hessian of the temporal
  negative log-likelihood at `fit.parameters` (optimiser coordinates), rebuilt in `run_J.jl`. For the phylo row R's is
  `kappa(sd$cov.fixed, exact = TRUE)` from the #547 script and Julia's is the fit's own finite-difference Hessian
  diagnostic. They are different estimators in different parameter bases: recorded, never compared. Each receipt says so
  in `cond_H_statement` and in `engines.*.cond_H_method`.
- **Disposition rows.** `data/RD-PHYLO-DISPOSITION`, `data/RD-TEMPORAL-DISPOSITION` and
  `data/RD-ISDM-DISPOSITION` carry `disposition: null` and a `proposed_disposition` object. Item 4 of the
  ruling does not quote them, and the plan says each needs the maintainer's own signature, so nothing is
  signed on them. Their classification stays `required_core` (the plan proposes `outside_boundary`, but a classification
  change is the maintainer's to sign), so on the assembled fold C1 still requires them.
- **Licence.** GPL datasets (MASS, gllvm) are loaded by name at run time by `gen_data.R`; the repository tracks
  that loader, each dataset's sha256 (in the receipt) and the receipts, never the tables. The GPL datasets' raw R and
  Julia outputs are committed, as they are for the other public data.
- **Urbanisation.** The urbanisation matrix is the maintainer's own unpublished data, and its redistribution status is
  unconfirmed. It is not tracked, and neither are the per-observation raw outputs of its row (linear predictors,
  loadings). `receipts/data/campaign/RD-URBANISATION-BINOMIAL.json` keeps only summary values (logLik, the fixed effects,
  the maximum differences, wall times, the data file's sha256 and the sha256 of the two uncommitted raw files).
  `write_receipts.py --check` prints "raw outputs kept off the public repo; re-derive locally with URBMAP_ROOT set" for
  this row and checks only the receipt's internal consistency and its case-map row; with the retained raw files,
  `--check --local-raw DIR` rebuilds it in full. `gen_data.R urban` requires the `URBMAP_ROOT` environment variable and
  stops with a message when it is unset: it has no default path. The row's raw outputs were first pushed in an earlier
  commit of this branch; they are removed from the branch tip but remain in the branch history on GitHub.
