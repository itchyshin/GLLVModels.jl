# PR #589 review — P1 ledger assembly (`tools/true_parity_assemble.py`)

Reviewer: independent adversarial pass (Claude, review lane). Head `fa56e49d3`, base
`claude/true-parity-p1-aghq` (`8f4427d4a`). Reviewed the three non-merge assembler commits
(`744f93c53`, `8dca4f9d1`, `fa56e49d3`) in a detached worktree from `fa56e49d3`. Nothing edited,
pushed, commented or merged. Date 2026-09-28.

## Verdict: NON-BLOCKING

Every headline claim reproduces on the tree (13/13 tests, `--check` current, checker verdicts
identical to the PR table, rows byte-identical to the source maps, 372 reverse-gap items all
`unsigned`/`null`, no class/disposition/case-id changes, no src/Project.toml/.github/GATES.md/P0
evidence touched in the assembler commits, no agent handles). Two narrow divergences between the
Python port and the checker's C1 exist (findings 1 and 2); neither is triggered by any row or
receipt on the tree today, and each is a one-line fix. They should be fixed before the PR leaves
draft, because both sit in the direction where `scoreboard.md` would read done and C1 would not.

## Findings

### 1. Port divergence, inflating direction: numeric `1` counts as a pass in Python, not in the checker

`derive_status` → `numeric_receipt_problem` tests `obj[f] not in ("PASS", "pass", True)`
(`tools/true_parity_assemble.py`, `numeric_receipt_problem`). In Python `1 == True` and
`1.0 == True`, so a receipt writing `harness_pass: 1` (or `batch_status: 1`) is a pass for the
assembler. The checker's `isPassValue` is `v === 'PASS' || v === 'pass' || v === true`, so `1`
is NUMERIC_RECEIPT_NOT_PASSED.

Evidence (tamper G: set `harness_pass: 1` in
`receipts/family/cases/CORE070-FAMILY-01-CLOGLOG-NATIVE-MODEL.json`):

```
python derive_status: ('EVIDENCED', '')
checker C1: bound=51 bound_numeric=51 ... numeric_receipt_not_passed:family/FAMILY-01-CLOGLOG(harness_pass=1
checker C8 row: NUMERIC_RECEIPT_NOT_PASSED
```

Consequence: the scoreboard row reads `EVIDENCED` with a resolving path, so C2/X2 count it done
while C1 does not. `real_tree_evidenced_equals_checker_c1_bound` would not catch it on a tree
where such a receipt exists only if the tree is re-tested; it holds today only because no receipt
writes a numeric status. Fix: `isinstance(v, bool) and v` or an explicit `v is True` check.

### 2. Port divergence, inflating direction: a signed disposition is reported before dangling/stale

`derive_status` returns `DISPOSITION-SIGNED` first, before the dangling-receipt and carry checks.
The checker's C1 runs dangling and carry first and `continue`s, so a validly signed row whose
`evidence.receipt` dangles, or whose `measured_against` is not P1 with no carry, is counted in
`dangling_receipts=` / `stale_carries=` and not in `bound_signed=`. The port's docstring says
"same rule as the checker".

Evidence (tamper E, valid signature + nonexistent receipt; tamper F, valid signature +
`measured_against: "P0"`, no carry):

```
E  python: ('DISPOSITION-SIGNED', '')   checker C1: dangling_receipts:family/FAMILY-01-CLOGLOG:...does-not-exist.json
F  python: ('DISPOSITION-SIGNED', '')   checker C1: stale_carries:family/FAMILY-01-CLOGLOG:PARTIAL_STALE_AT_P1(no carry.source_pins)
```

The scoreboard cell carries `signed_by:`/`signed_on:` tokens, so C2 accepts the row as done via
`hasSignedTokens` while C1 does not resolve it. Note the checker is itself inconsistent here:
C8's `dispositionSignedProperly` short-circuits before the dangling check, so the port matches C8
and not C1. Not reachable today (no signed rows in any map). Fix: run the dangling/carry checks
before the signature branch when `receipt_paths(row)` is non-empty, or state in the docstring
which clause's order the port follows.

### 3. Port omission, conservative direction: `receipt_status_exception` is not honoured

The checker lets a maintainer-signed `receipt_status_exception` waive a failed status field and
counts the row in `bound_signed=`. The port has no such branch: the row reads
`NUMERIC-UNVERIFIED`. Tamper D (FAIL receipt + valid signed exception): python
`NUMERIC-UNVERIFIED`, checker `bound_signed=1`, C8 resolved. The scoreboard under-reports, so the
`EVIDENCED == bound=` invariant still holds (`bound=` excludes signed rows), but the docstring's
"same rule as the checker" is incomplete. Non-blocking.

### 4. Malformed `carry` crashes the assembler instead of reporting

`carry_problem` does `(row.get("carry") or {}).get("source_pins")` and `sp.get(...)`; a `carry`
that is a string, or a `source_pins` entry that is a string, raises `AttributeError` (tampers H,
H2). The checker reports `PARTIAL_STALE_AT_P1(...)` for both. The crash is fail-closed (non-zero
exit, no output written) so it is not exploitable, but it is an uncaught traceback rather than an
`ASSEMBLE_FAIL` line. Non-blocking.

### 5. C2 does count a row as done from the scoreboard word; only `--check` stops it, and `--check` is not wired anywhere

Tamper: hand-edit row `aghq-AGHQ-AUTO-K-BINOMIAL` to `EVIDENCED` with an existing receipt path
(or to `DISPOSITION-SIGNED` with signer tokens), no case-map change.

```
checker C2: rows=297 done=53      (was 52)
assembler --check: ASSEMBLE_STALE scoreboard.md
```

So the checker's C2/X2 trust the scoreboard text by design (that predates this PR), and the
assembler's `--check` is the only thing that ties the word back to the evidence. `--check` is not
referenced by any workflow, by `tools/test_true_parity_check.mjs`, or by anything outside its own
two files; it runs only if someone runs `python3 tools/test_true_parity_assemble.py` or the
command by hand. The proposed GATES.md text says "`--check` in review", which is manual. Until it
is wired (a CI step, or a call from the checker's own test), C2's reading of `scoreboard.md` can
drift from the maps without any tool noticing. Non-blocking for this PR; a follow-up.

### 6. Scoreboard id scheme makes C4 and C5 structurally empty

`scoreboard_id` prefixes every id with its family (`data/RD-01` → `data-RD-01`,
`data/GRP-1` → `data-GRP-1`). C4 selects `/^RD-/` and C5 `/^GRP-/`, so no row produced by this
assembler can ever be selected by C4 or C5; only C3's `-RSZ$` suffix survives the prefix. The
collision guard in `scoreboard_id` (`^(RD|GRP)-`) is therefore dead code for prefixed ids (it
fires only for the `-RSZ$` case; verified: `data/X-RSZ` raises, `data/RD-01` does not). The PR's
"C4/C5 NOT_MET: EMPTY_SELECTION, no RD-/GRP- rows exist" is true but incomplete: adding such rows
to a map would not change it. Needs a decision (drop the family prefix for those ids, or change
the checker's C4/C5 selectors) before any RD/GRP row is measured. Non-blocking here.

### 7. `--check --extra-map` without `--out-dir` always reports STALE

`main` permits `--extra-map` with `--check` alone, but the stale comparison then runs against the
tracked outputs, which do not contain the extra rows, so it prints `ASSEMBLE_STALE scoreboard.md,
case-map-assembled.json` and exits 1 even when there is no conflict (the conflict scan does run
first and would `ASSEMBLE_FAIL` on a real conflict). The documented route
`--extra-map X --out-dir D` and `--check --extra-map X --out-dir D` both work: #533's
`case-map.json` (head `9c1f55038`) folds to 335 rows, 318 required, `ASSEMBLE_OK`, matching the
PR body. Usability nit.

### 8. Reverse-gap wording in the review brief vs the PR

The brief asked whether "semantic-divergence names (e.g. zero-inflated) are excluded as claimed".
The PR claims the opposite and the code does the opposite: the list is by name only, so
`ZIP`, `ZIB`, `ZINB`, `ZIPoisson`, `zip_marginal_loglik_laplace` and so on ARE in the gap (no
same-named R export), and a same-named symbol whose semantics differ (`simulate`, `predict`,
`coef`, `select_lv`, 72 such names) is NOT in the gap. The PR body states this as the reason the
list is a lower bound. Consistent; no finding against the PR.

Verified: `gap == julia_exports − r_names − receipt_named` exactly (372). `r_names` (202) is the
export()/S3method set from gllvmTMB `NAMESPACE` at P1; a local gllvmTMB clone gives sha256
`c1e91cd7…0f0c` for that blob, matching `NAMESPACE_SHA256_P1` and `reverse-gap-inputs.json`.
Spot checks `welch_t`, `AnBSparseSolver`, `twolevel_marginal_loglik`: present in `src/`, absent
from `r_names`. Fifteen Julia symbols are excluded only because a #561 namespace receipt names
them as the Julia side of an R export (`OrdinalFit`, `TwoLevelFit`, `fit_beta_gllvm`,
`fit_gaussian_mi_fiml`, `coef_table`, ...); that rule is stated in the PR body and the
`derived_from` text. Schema matches C6: array, each item with `source_id`, `name`, `decision`
(C6 reads `it.decision` and `it.source_id || it.name`); the checker prints all 372 as
`invalid_decision=...:null`, as claimed.

## What holds (evidence)

- `python3 tools/test_true_parity_assemble.py`: 13 PASS lines, `ALL PASS`.
  `node tools/test_true_parity_check.mjs`: all negative controls pass.
- `python3 tools/true_parity_assemble.py --check`: `ASSEMBLE_OK 297 rows current`.
- Checker, `PARITY_REF=HEAD` (git mode) and `PARITY_REF=FS` both, `PARITY_CASEMAP=case-map-assembled.json`:
  C0 NOT_MET (`default_pin=P0`), C1 required=297 bound=52 bound_numeric=52
  bound_registration_only=44 bound_signed=0 free=197 unsigned_or_blocked=4, no dangling, no stale;
  C2 rows=297 done=52; C3/C4/C5 EMPTY_SELECTION; C6 items=372 all null; C7 MET; C8 failing=245;
  X2 done=52. Identical to the PR table.
- `case-map-assembled.json` rows: 297, equal (`==` on the parsed dicts, in order) to the union of
  the nine source maps; no duplicate `source_id` across maps; classifications only
  `required_core` (219) and `compatibility_adapter` (78); dispositions only `null` (293) and
  `BLOCKED_NEEDS_JULIA_SURFACE` (4). No `semantic_divergence` or `outside_boundary` rows.
- Scoreboard parses to exactly 297 rows in the checker (totals table skipped via the backtick
  first cell). 52 `EVIDENCED`, 0 `DISPOSITION-SIGNED`; the 52 are the same rows the checker binds.
- Tampers where port and checker agree (both not bound): `batch_status: "FAIL"` (B), stale
  `max_abs_diff` with r/julia values unchanged (C), r/julia drifted beyond tolerance with a stale
  recorded diff (C2), agent signer (F2), future date (F3), `batch_status: "Pass"` (G2), comparison
  pinned P0 (I), uncovered case id (J), tier `registration` with a numeric receipt (K),
  NaN literal in the receipt (M; Python parses it and fails the case, JS rejects the file and
  reports no comparison block — same verdict). Agree (both bound): boolean r/julia values fall
  back to the recorded diff (L); a second non-JSON receipt is skipped (N).
- Assembler commits touch only: `tools/true_parity_assemble.py`, `tools/test_true_parity_assemble.py`,
  `docs/dev-log/check-log.md`, and the three generated files plus `reverse-gap-inputs.json` under
  `docs/dev-log/core070/true-parity-latest/`. All additions; no src/, Project.toml, .github/,
  GATES.md or P0 evidence. (The two merge commits carry #587 and #561 content, including #561's
  checker and a GATES.md edit; those are the merged PRs' reviews, not this one's.)
- PR body: draft, "nothing is a signature", per-clause table, "not covered" section, the
  `--extra-map` scratch fold (335/318) and the `_DEFAULT_PIN` note are all accurate. No `@handle`
  for any agent in the PR body, commit messages, tool, or check-log. Co-author trailers are the
  standard `noreply@anthropic.com` form.

## What I did not check

- `names(GLLVModels)` at `cba170c28` really has 458 non-module names (would need a Julia
  precompile; the `julia_exports` list is taken as recorded in `reverse-gap-inputs.json`).
- That the #561 namespace receipts' `julia_check.symbol` values are correct counterparts (that is
  #561's review).
- The content of the merged branches (#587, #561) beyond confirming they do not touch src/.
- The checker's git-mode `existsAsBlob` against a pushed ref other than HEAD of this worktree.
- Whether `hasSignedTokens` in C2 could be satisfied by a `DISPOSITION-SIGNED` row whose map
  signature is valid but whose scoreboard cell is malformed (the assembler writes the cell; only
  hand edits would produce that, and `--check` catches hand edits).
- Any behaviour of `--refresh-reverse-gap-inputs` (needs `GLLVMTMB_DIR` and a Julia names TSV;
  not exercised).
