# Review of PR #561, namespace rows re-measured at P1 with an evidence tier (head 2d0cf3387, base cb5688f7e), 2026-09-27

Reviewer stance: independent, adversarial. Worked in a detached worktree at 2d0cf3387 (`/Users/z3437171/local-scratch/lanes/GLLVM.jl-review-561`, removed afterwards). Nothing on the branch was edited, pushed, or commented on.

## Verdict: BLOCKING (one item), otherwise sound

The PR does what it says for the namespace rows: every row is `registration`, nothing binds C1, the six mismatches are real, the receipts reproduce byte-for-byte. The blocking item is that the new `evidence_tier` guard is a self-declared string the checker never checks against the receipt, so a one-word edit (`"registration"` -> `"numeric"`) on a namespace row makes C1_MET and C8_MET. That is the #559 hole moved one field to the right. The PR must not be the basis of any EVIDENCED claim until the tier is derived from receipt content, or at minimum a negative control pins the hole as a known gap in GATES.md.

## Findings

### 1. BLOCKING. `evidence_tier: "numeric"` is trusted without a numeric receipt

The checker reads the label only (`isNumericTier(row) { return row.evidence_tier === 'numeric'; }`, `tools/true_parity_check.mjs:172`). It never opens the receipt to see whether an R-vs-Julia comparison is recorded. Mutation test on the real namespace row `namespace/S3method/coef,gllvmTMB_multi` (a Tier 0 registration receipt, no numbers anywhere), one row in an otherwise untouched copy of `case-map-namespace.json`, FS mode:

```
$ PARITY_REF=FS PARITY_CASEMAP=<copy with evidence_tier: "numeric"> node tools/true_parity_check.mjs C1
C1 required=1 bound=1 bound_numeric=1 bound_registration_only=0 free=0 unsigned_or_blocked=0 {} dangling_receipts=none stale_carries=none registration_only=none
C1_MET
$ ... C8
C8 rows=1 failing=none
C8_MET
```

Other odd values behave as documented (fail-closed): removed field, `"Numeric"`, `""`, `null`, `["numeric"]` all give `bound_registration_only=1`, `C1_NOT_MET`, `REGISTRATION_ONLY_NOT_TWINNED`. So the tier is fail-closed for everything except the one value that matters, and that value is checked by string equality on an agent-writable field.

The positive control has the same shape: the `base` fixture's "numeric" rows resolve to `receipts/r1.json`, whose whole content is `{"case_id": "R1", "result": "PASS"}`. The `c1_registration_only` fixture uses the same `r1.json` for its registration row. The fixtures therefore prove that the tool cannot tell a numeric receipt from a registration receipt by content; only the label differs.

GATES.md ("Evidence tier" section) says `"numeric"` means "the receipt records an actual R-vs-Julia output comparison". The tool does not enforce that sentence.

Suggested fix (either closes the hole; the first is the real one):
- Derive the tier from the receipt. A row counts as `numeric` only if every resolving receipt (or the case receipt named by `executable_case_ids`) carries a machine-readable comparison block, e.g. `schema` in an allow-list of twin schemas and, per case, `r_value`/`julia_value`/`tolerance`/`abs_diff` (or `comparison: {...}`) with finite numbers. A row labelled `numeric` whose receipt lacks that block is reported as `NUMERIC_LABEL_WITHOUT_NUMERIC_RECEIPT` and counted as registration-only. Add two negative controls: (a) `numeric` label + namespace-style receipt -> `C1_NOT_MET`/`C8_NOT_MET`; (b) `numeric` label + a receipt whose comparison block has `abs_diff > tolerance` -> not bound. Update the `base` fixture receipts to carry the block so the positive control is real.
- Until that lands, write in GATES.md, in the "Evidence tier" section, that the tier is author-declared and that PR review must diff any row whose tier changes to `numeric` against its receipt; and add `tools/test_true_parity_check.mjs` control (a) as an `expected failure` that documents the gap rather than passing silently.

### 2. NON-BLOCKING (pre-existing, but this PR widens its reach). The signed-disposition escape hatch accepts any string and any date-shaped string

`dispositionSignedProperly` (`true_parity_check.mjs:180`) needs only `disposition === 'DISPOSITION-SIGNED'`, a non-blank `signed_by`, and `signed_on` matching `\d{4}-\d{2}-\d{2}`. Same row as above:

```
signed_by: "Claude Fable (agent)", signed_on: "2026-09-27"  ->  C1_MET, C8_MET
signed_by: " x ",                  signed_on: "9999-99-99"  ->  C1_MET, C8_MET
```

An agent-written field satisfies it; the date is not even a valid date. GATES.md is candid that identity is verified in PR review, not by the tool, and the check pre-dates this PR. What this PR adds is a new use of the hatch: a registration-only row now *resolves C1 and C8* if signed (test `c1_registration_only_signed`), so the hatch is now the only way a name match can ever satisfy the ledger. Two cheap tightenings: (i) a signer allow-list in the tool (the maintainer's name; anything else -> `DISPOSITION-SIGNED-UNVERIFIED`), and a real date check; (ii) report signed rows in their own counter (`bound_signed=N`) so a signed registration row never appears inside `bound=`. Neither replaces human review; both stop an agent from producing a MET line.

### 3. NON-BLOCKING. "P0 behaviour is unchanged" is not true for C8 at P0

Commit c29c1c0b6's message and the PR body say P0 behaviour is unchanged. That holds for the namespace batch tools (P0 self-test: `rejected_mutations=5`, as before) and for C1 at P0 (line identical modulo the two new counters). It does not hold for C8 at P0. `docs/dev-log/core070/required-source-case-map.json` (769 rows) has no `evidence_tier` on any row, so the fail-closed default demotes every row that was twinned:

```
old checker (cb5688f7e) P0 C8 tally:   7 DANGLING_RECEIPT, 748 NOT_TWINNED_NOT_SIGNED  (14 rows twinned)
new checker (2d0cf3387) P0 C8 tally:   7 DANGLING_RECEIPT, 748 NOT_TWINNED_NOT_SIGNED, 14 REGISTRATION_ONLY_NOT_TWINNED
```

The 14 are the `aghq/AGHQ-*` rows bound to `aghq-public-policy-bind-receipt-2026-09-04.json`. The overall verdict was C8_NOT_MET before and after, and fail-closed is the documented design, so this is not wrong, but it is undisclosed. Fix: say it in the PR body and in GATES.md ("at P0 every row reads as registration-only until `evidence_tier` is back-filled"), or back-fill the 14 rows if their receipt is judged numeric (it records R fits with objective values, but no Julia comparison, so `registration` may actually be the honest tier; that judgement is a separate decision).

### 4. NON-BLOCKING. The `evidence.tier` prose on the six FAIL rows is a template that asserts the opposite of the measurement

All 52 measured rows carry the same `evidence.tier` string: "registration: R export registered+defined at P1 and a same-named exported Julia Function; ...". On `extract_loadings`, `extract_proportions`, `extract_residual_split`, `extract_Sigma_B`, `extract_Sigma_W`, `extract_cutpoints` the measured result is `registration_mismatch` (symbol unexported or a type). The `measured_result.reason` and the case receipt are correct; the `evidence.tier` text is not. A reader who greps `evidence.tier` gets six false statements. Fix: make the string conditional on the verdict, or drop the free-text field and rely on `measured_result`.

### 5. NON-BLOCKING. Checker command in the PR body omits `PARITY_REF=FS`

"run both in FS mode and against HEAD": FS mode needs `PARITY_REF=FS` (default is `origin/main`, which does not have the file -> `MEASUREMENT_FAILED ... not on origin/main`). With that set, the reported lines reproduce exactly:

```
C1 required=69 bound=0 bound_numeric=0 bound_registration_only=44 free=23 unsigned_or_blocked=2 {"BLOCKED_NEEDS_JULIA_SURFACE":2} dangling_receipts=none stale_carries=none registration_only=<44 rows>
C1_NOT_MET
C8 rows=69 failing=<25 NOT_TWINNED_NOT_SIGNED; 44 REGISTRATION_ONLY_NOT_TWINNED>
C8_NOT_MET
```

Same in git mode (`PARITY_REF=HEAD`). `node tools/test_true_parity_check.mjs`: all controls pass, including the five new tier controls and the git-mode registration control.

### 6. OK. `deviance` does not count toward C1, and the case-map row is worded correctly

Case-map row `namespace/S3method/deviance,gllvmTMB_multi`: `evidence_tier: "registration"`, category `registration_twin`, reason ends "Name match only." It therefore cannot bind C1 (verified: it is one of the 44 in `registration_only=`). The "genuine semantic twin by definition" phrasing lives in the contract's `julia_note` and the PR body, not in the ledger row, and it is the phrasing the #559 review itself used. Confirmed both sides: R at P1 `deviance.gllvmTMB_multi <- function(object, ...) -2 * as.numeric(stats::logLik(object, ...))` (`git show 9539352f6:R/methods-gllvmTMB.R:3355`); Julia `StatsAPI.deviance(fit::AnyGllvmFit) = -2 * StatsAPI.loglikelihood(fit)` (`src/postfit_tables.jl:21`). Same definition; numeric equality inherits entirely from logLik parity, which is not compared. Not a D-295 row 5 violation, because it is not counted. Suggest softening "genuine semantic twin" to "same definition on both sides; parity is exactly logLik parity, not measured here" so no later reader upgrades it.

### 7. OK. Julia facts verified directly

With the builder worktree's `Manifest.toml` (the review worktree had none), `OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=4`, Julia 1.10.0, `using GLLVModels` at 2d0cf3387:

```
coef        exported=true  isaFunction=true  kind=Function ownmethods=34
deviance    exported=true  isaFunction=true  kind=Function ownmethods=1
predict     exported=true  isaFunction=true  kind=Function ownmethods=43
nobs        exported=true  isaFunction=true  kind=Function ownmethods=13
TwoLevelFit exported=true  isaFunction=false kind=DataType
OrdinalFit  exported=true  isaFunction=false kind=DataType
unpack_lambda exported=false isaFunction=true
proportions   exported=false isaFunction=true
```

Matches the receipts and the six mismatches exactly. `src` tree at 2d0cf3387 equals the receipt's `julia_src_tree` (`cb2652e71...`), so the facts measured at c29c1c0b6 hold at head.

### 8. OK. Receipt integrity and reproducibility

- `receipt.json`: `pin: "P1"`, `reference_commit` = P1, `status: "FAIL"` (honest: six rows fail), `source_access: git-show`, `julia_src_dirty: false`.
- sha256 guards recomputed on disk: contract `fe645d39...`, julia-facts `119680c4...`, r-facts `d1bf555c...`, results.tsv `c0c03338...` all equal the receipt. NAMESPACE sha at P1 from the local gllvmTMB clone (`git show 9539352f6:NAMESPACE | shasum -a 256`) = `c1e91cd7...` as pinned.
- `GLLVM_PARITY_PIN=P1 python3 tools/core070_verify_namespace_1_batch.py <batch>` -> `CORE070_NAMESPACE_1_STATE_OK`; `--self-test` P1 `rejected_mutations=8`, P0 `rejected_mutations=5`; `--write-case-receipts` into a scratch dir produced 54 files byte-identical to the committed `receipts/namespace/cases/` (`diff -r` clean). `core070_namespace_1_p1_contract.py --check` -> `CURRENT` (needs `GLLVMTMB_DIR`); `test_core070_namespace_1_pin.py` OK.
- Generators committed: `tools/core070_namespace_1_batch.{R,jl}`, `core070_verify_namespace_1_batch.py`, `core070_namespace_1_p1_contract.py`.

### 9. OK. C8 change and GATES.md totals are consistent

C8 now emits `REGISTRATION_ONLY_NOT_TWINNED` after the twinned test, matching GATES.md. Fold totals: #533 head (9c1f55038) `case-map.json` has 38 rows, 21 `required_core`; namespace file has 69 rows (61 `required_core` + 8 `compatibility_adapter`); 38+69=107, 21+69=90. Checks out. One coordination note for the fold: none of #533's 38 rows carry `evidence_tier`, so after the fold they all read as registration-only under the fail-closed default; #533 (or the fold PR) must set the tier per row.

### 10. OK. Scope and hygiene

- `git diff --name-only cb5688f7e..2d0cf3387 | grep -E '^(src/|Project.toml|.github/)'` -> none. Files touched: `docs/`, `tools/`, `test/fixtures/true_parity/**` only. P0 evidence files untouched (`required-source-case-map.json`, P0 scoreboard, P0 receipts). The `base` fixture edit adds `evidence_tier: "numeric"` to two fixture rows; it is a test fixture, not P0 evidence.
- PR body: no `@handle` anywhere (`grep -nE "@[A-Za-z]"` on the body -> none; commit trailers only `Co-Authored-By`). Parity claims are not overstated: "Numeric twin 0", "no row binds C1", "a PASS is not a twin" in every receipt. The one inaccurate sentence is finding 3 ("P0 behaviour is unchanged").
- The eight `*-JULIA-BRIDGE-COMPARE` rows keep `original_classification: required_core` and are still counted in `required=` (checker includes `compatibility_adapter`), so the reclassification does not shrink the required set.

## What I did not check

- The R side of the batch was not re-run (no `Rscript` invocation here); I verified the R facts only through the recorded sha256s and by reading `deviance.gllvmTMB_multi` and NAMESPACE at P1 from the local gllvmTMB clone.
- The 17 not-re-measured rows: accepted as free (no receipts, `measured_against` null); did not audit whether each genuinely needs an R oracle.
- Whether `registration` is the right tier for the 14 P0 aghq rows (finding 3) is a judgement I did not make.
- `test_core070_build_oracle_pin.py` and `test_parity_oracle_defaults.py` were not run.
- No check that the branch-protection "required check" for the P1 twin job exists on GitHub (C0 itself says it cannot read that).
- Did not review the cherry-picked #559 commits (48f4d93af, 0e19e7619) beyond confirming the #559 review's blocking item is addressed at the batch level (exported + Function measured) and re-opened at the checker level (finding 1).
