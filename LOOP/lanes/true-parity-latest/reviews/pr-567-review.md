# Review: draft PR #567 — covariance family re-measured at gllvmTMB P1

- Branch `claude/true-parity-p1-covariance`, head `508d297d0`, base `main` `cb5688f7e`.
- Reviewer: independent adversarial pass, own detached worktree at
  `/Users/z3437171/local-scratch/lanes/GLLVM.jl-review-567` (removed after this review).
- Nothing on the branch was edited, pushed, commented or merged. This file is not committed.
- Compute: local Mac, `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JULIA_NUM_THREADS=4`, about 25 min total.

## Verdict: BLOCKING (one decision gate; the evidence itself reproduces)

The measurements are real, from the P1 oracle, and reproduce bit-for-bit on my machine. The
comparison blocks faithfully mirror the harness bounds. Classifications are untouched. Pin
switches are strict. The single blocking item is not a defect in the numbers: the two rows the PR
marks `evidence_tier: "numeric"` (and which bind, `bound=2`, under both checkers) come from a
batch whose own verifier rejects the run (`receipt status is not PASS`). The builder flagged this
for the maintainer but tracked the rows as bound anyway. Until the maintainer decides, the tracked
case map should not present them as bound. Everything else is non-blocking.

## Findings

### 1. BLOCKING (decision gate) — the 2 numeric rows bind on a batch whose verifier rejects the run

Both numeric rows (`covariance/COV-KERNEL-FOLDED-UNIQUE`, `covariance/COV-KERNEL-LATENT`) have
exactly one executable case each, and both cases live in the 10-case wave6 batch:

```
$ python3 -c "... case-map-covariance.json ..."
covariance/COV-KERNEL-FOLDED-UNIQUE numeric ['CORE070-WAVE6-KERNEL-LATENT-SINGLE-PSI-COVARIANCE'] PASS
covariance/COV-KERNEL-LATENT        numeric ['CORE070-WAVE6-KERNEL-LATENT-MULTI-NAMESPACE']       PASS
```

The tracked batch receipt and verifier log:

```
receipts/covariance/wave6-conversion-p1/receipt.json:  "status": "FAIL", "julia_exit_code": 1
receipts/covariance/wave6-conversion-p1/verify.log:
  ValueError: receipt status is not PASS        (tools/core070_verify_wave6_conversion_batch.py:159)
```

What fails, exactly (from `julia-results.json`):

```
CORE070-WAVE6-POSTFIT-NOBS-MULTI {"kind": "own_receipt_defect", "r_nobs": 400, "julia_nobs": 400.0,
  "pass": false, "r_expected_p_times_n": 400, "julia_expected_n": 80.0, "known_defect_pending_decision": true}
```

Is the nobs case genuinely unrelated to covariance? Yes. Its source row is
`postfit/POSTFIT-SURFACE-nobs.gllvmTMB_multi`, fixture `gaussian_small`, and the case does not
assert R == Julia; it asserts a recorded Julia defect (nobs = n = 80). That defect was fixed on
`main` before this PR's base:

```
$ git merge-base --is-ancestor 1c5fbd353 cb5688f7e && echo on-base
on-base
1c5fbd353 2026-09-01 fix(nobs): adopt R's p·n observed-cell nobs/bic convention
```

So the FAIL is a stale P0 expectation carried verbatim into `wave6-conversion-batch-contract-p1.json`
(the generator's "cases verbatim" rule), not a covariance disagreement. Both engines now agree
(400 = 400). The two covariance cases pass on their own numbers, and I reproduced them
(finding 3).

Why it still blocks: the programme's discipline is "receipt + verifier, or it is not evidence".
The receipt tool (`tools/core070_covariance_p1_receipts.py`) writes `batch_status: "FAIL"` and a
`batch_status_note` into each case receipt, which is honest, but the row still gets
`evidence_tier: "numeric"` and binds. Neither checker reads `batch_status`, so a verifier-rejected
batch produces bound rows silently. That decision is the maintainer's, and the PR itself says so
("Needs the maintainer's decision, item 1"). A PR that asks for the decision should not pre-empt it
in the tracked case map.

Suggested fix (either, maintainer's choice):
- (a) Hold the two rows back: `evidence_tier` other than `"numeric"`, receipts under
  `non_binding_receipts`, `disposition: null`, note "pending maintainer ruling on wave6 nobs
  contract staleness". Counts become 0 numeric / 9 partial / 8 R-only. Or
- (b) Fix the cause: update the nobs case in `wave6-conversion-batch-contract-p1.json` (a
  documented P1 adaptation in the generator, with the reason and the fixing commit `1c5fbd353`),
  re-run the batch to a PASS receipt, and regenerate. This is a contract edit, so it needs the
  maintainer's explicit OK; the generator docstring currently forbids it.

Either way, the receipt tool should refuse to mark a row `numeric` when the batch verifier did
not pass, unless a maintainer-signed exception is recorded on the row.

### 2. NON-BLOCKING — P1 runner manifest carries 30 P0 line citations that point at the wrong code at P1

`frozen-r070-contract-p1.toml` header is fully recomputed at P1 (verified: reference_commit,
NAMESPACE blob/sha, archive and tree sha all match my independent recomputation, finding 3). The
header comment discloses that the citations below are not re-anchored. But the files they cite
are NOT byte-identical between P0 and P1, and the cited line ranges now hold different code:

```
$ (gllvmTMB) for f in NAMESPACE R/aghq-control.R R/fit-multi.R R/julia-bridge.R: compare blob at b4d5fee64 vs 9539352f6
DIFF  NAMESPACE
SAME  R/aghq-control.R
DIFF  R/fit-multi.R
DIFF  R/julia-bridge.R

$ git show 9539352f6:R/julia-bridge.R | sed -n '3521,3526p'     # cited "R/julia-bridge.R:3521-3589" for all 9 *-PUBLIC-R-BRIDGE obligations
    )
  )
  class(out) <- "summary.gllvmTMB_julia"
  out
}
$ git show 9539352f6:R/julia-bridge.R | grep -n GJL-GATE-STRUCTURED-TERMS
147: ...  3678: ...  3689: ...        # the gate moved from 3583 (P0) to ~3678-3689 (P1)
```

Obligations affected: `BRIDGE-01` (cites NAMESPACE blob `dd2d012e`, the P0 blob, lines 131-132),
`reference_admission` (`R/fit-multi.R:1020-1051`, moved), the 18 `MODE-*`/`FIT-MODE-*` rows
(`b4d5fee64:test/parity/fixtures/...`, a GLLVModels path under a gllvmTMB sha, odd but P0-inherited),
and the 9 `*-PUBLIC-R-BRIDGE` rows (`R/julia-bridge.R:3521-3589`, now the summary method). The 9
bridge rows are the ones this PR re-measures at P1 and whose message changed at P1 (PR body, item
3), so a P1 reader following the citation lands on unrelated code.

This does not make any numeric claim wrong (the runner validates the manifest's header pins and
case registries, not the free-text `source` lines), but a file named `-p1` whose evidence rows cite
P0 lines is misleading, and the maintainer's signed rule ("a P0 receipt counts at P1 only if its
source files are byte-identical") is exactly the test these citations fail.

Suggested fix: extend `tools/core070_covariance_p1_contract.py` to re-anchor the `source` lines it
can (`R/<file>:<lines> @ <sha>` -> recompute the line range at P1 by searching the cited text, or
fall back to `R/<file> @ <P1 blob sha>` file-level pins) and to rewrite `NAMESPACE:<blob>:<lines>`
with the P1 blob. Where re-anchoring is not mechanical, replace the line range with the P1 blob
sha and a note. Keep `--check` covering the result.

### 3. CLEAN — R values are from the P1 oracle, not P0 and not Julia self-values; re-run reproduces

Oracle provenance, recomputed independently from the local clone (never checked out):

```
$ git -C gllvmTMB archive --format=tar 9539352f6 > p1.tar; shasum -a 256 p1.tar
73b665b22c1497d43dc0674eecd5657f2659a726fcad13004de899833e2d5423     == receipts/covariance/oracle/{source,build}.json archive_sha256
$ tree_hash(extracted archive, tool's own recipe)
86fa00f05ee2ccb8ffef65a82cd59c47b1b2a256a5bad86957a92461a6dbe1db (9808 files)  == source_tree_sha256
$ GLLVM_PARITY_PIN=P1 python3 tools/core070_build_oracle.py verify --destination /Users/z3437171/local-scratch/a3cov-oracle/build
CORE070_ORACLE_VERIFY_PASS
$ cat .../library/gllvmTMB/CORE070_SOURCE_PIN.toml
reference_commit = "9539352f66f2db2cc26b1c393e67212a359b60c9"; source_tree_sha256 = "86fa00f0..."; installed_tree_sha256 = "5d487719..."
$ .../library/gllvmTMB/DESCRIPTION -> Version: 0.7.1 (== git show 9539352f6:DESCRIPTION)
```

The runparity run copied the same build receipt into its run directory
(`runparity-covariance-18/build.json` is byte-identical to `oracle/build.json`), and
`parity_helpers.jl` validates the installed marker's `source_tree_sha256` against the selected pin
before any required case runs, so the 18 native/formula cases were fitted against the P1 library.

Re-run of the two numeric rows' cases (whole wave6 batch, `GLLVM_PARITY_PIN=P1`, same library,
46.6 s wall):

```
CORE070-WAVE6-KERNEL-LATENT-SINGLE-PSI-COVARIANCE: tracked max_abs_diff=7.25e-08 rerun=7.25e-08 pass=True |julia tracked-rerun|=0
CORE070-WAVE6-KERNEL-LATENT-MULTI-NAMESPACE:       tracked max_abs_diff=1.26e-06 rerun=1.26e-06 pass=True |julia tracked-rerun|=0
CORE070-WAVE6-POSTFIT-NOBS-MULTI tracked 400.0 False rerun 400.0 False
R oracle values: all 10 cases max|tracked - rerun| = 0; fixtures identical
```

The R and Julia values differ from each other (7e-8, 1e-6), so they are not self-values.

Partial re-run of the runparity harness at P1 (required mode, same library, scratch receipt dir):
the fixed-residual group (`MODE-ORD-INDEP`, `MODE-ORD-COMMON`) completed, 27/27 assertions:

```
MODE-ORD-INDEP:  |R loglik tracked-rerun|=0 |native loglik tracked-rerun|=0 |R beta|=0 checks_all=True |R-native| = 2.56e-11
MODE-ORD-COMMON: |R loglik tracked-rerun|=0 |native loglik tracked-rerun|=0 |R beta|=0 checks_all=True |R-native| = 7.11e-15
```

The tight-control modes group (7 `FIT-MODE-*` cases) and the formula groups errored at
`tools/core070_covariance_mode_fits.jl:15` ("retained baseline required": a `baseline/` directory
holding a default-control run must exist in the working directory), which I did not set up within
the compute budget. See finding 13 for what it took to get this far.

### 4. CLEAN — comparison blocks faithfully mirror the harness; nothing loosened

Harness bounds (read from source): `abs(loglik diff) <= 1e-6`; `isapprox(...; atol=1e-5, rtol=1e-5)`
for beta / covariance / residual variance (`tools/core070_covariance_mode_fits.jl:294-312`,
`tools/core070_source_fixed_residual_pair.jl:50-53`, `test/parity/covariance_formula_cases.jl:175-198`,
`_PARITY_TOL = 1e-5`); wave6 `maximum(abs.(jl .- r)) <= tolerance` (`core070_wave6_conversion_batch.jl:299`).
Julia's array `isapprox` is `norm(x - y) <= max(atol, rtol*max(norm(x), norm(y)))`, which is what
the receipt tool records; scalars are wrapped as 1-vectors (norm = abs), so nothing is loosened.

Independent recomputation of every comparison entry from its own `r_value`/`julia_value`:

```
entries 101  bad 0  worst diff/tol ratio 0.124
loglik worst: 2.5551e-11 (MODE-ORD-INDEP-FORMULA-INTERFACE)     # matches the PR body's 2.56e-11
```

Contract-to-receipt hash ties all hold (`run.toml`, wave6 `receipt.json`, cov-batch results each
carry the sha256 of the tracked P1 contract they ran under; fixture sha256s match the tree).

### 5. NON-BLOCKING (for #561, not #567) — what #561's checker catches and what it does not

Both checkers on this branch's case map: `C1 required=17 bound=2 free=15`, `C1_NOT_MET`,
`C8 ... 15 NOT_TWINNED_NOT_SIGNED`, exactly as the PR body states. Tamper tests on scratch copies
of one numeric receipt, #561 checker (`1598b2142`), `C1`:

```
baseline copy                              bound=2
max_abs_diff above tol                     bound=1  (caught)
comparison.pin = P0                        bound=1  (caught)
tolerance = 0                              bound=1  (caught)
wrong case_id                              bound=1  (caught)
receipt file deleted                       bound=1  (caught)
drop max_abs_diff, keep vectors            bound=2  (checker recomputes from vectors: fine)
stale max_abs_diff, vectors disagree by 1  bound=2  NOT caught (recorded diff trusted over vectors)
verdict = "FAIL", comparison intact        bound=2  NOT caught (verdict not read)
tolerance widened to 1.0                   bound=2  NOT caught (checker cannot know the harness bound)
```

For #567 this is moot because finding 4 verified the receipts independently, but the last three
rows are worth a note on #561: prefer recomputing the diff from vectors when both are present,
and treat `verdict != PASS` or `batch_status != PASS` as not bound.

### 6. NON-BLOCKING — the two R-only batches do not tie their receipts to the library marker

`tools/core070_wave6_conversion_batch.R` and `tools/core070_covariance_batch.R` record only
`frozen_library` (a path) and `gllvmTMB_version`; unlike `parity_helpers.jl`, they never read
`CORE070_SOURCE_PIN.toml` or compare its `source_tree_sha256` with the selected pin. With
`GLLVM_PARITY_PIN=P1` and argv[1] pointing at a P0 library, the run would proceed and the receipt
would say P1 (the P1 contract's reference_commit check is against the contract file, not the
library). Today the path in the receipt resolves to the verified P1 build, so the claim holds, but
it holds by inspection, not by check.

Suggested fix: in both R scripts read the marker under `frozen_library/gllvmTMB/`, `stopifnot` its
`reference_commit` equals the pin's, and write `source_tree_sha256` into the receipt; have the two
verifiers check it against `tools/core070_oracle_pins.toml`.

### 7. NON-BLOCKING — provenance nit: `glvmodels_commit` records the base, not the harness that ran

Every case receipt says `"glvmodels_commit": "cb5688f7e..."` (the base). The P1 run needs this PR's
pin switches (`295ac5aa5`), so the runs were made from a dirty tree at `cb5688f7e`. Harmless
here (the switches only select paths), but a receipt should name a commit that contains the code
that produced it. Suggested fix: record `git describe --dirty` or the branch head after committing
the tool changes, or re-run the receipt generator once (cheap: it only rewrites JSON).

### 8. CLEAN — pin switches are strict; P0 default paths unchanged

```
GLLVM_PARITY_PIN=P2 python3 -c "import parity_oracle"   -> SystemExit: 'P2' is not a recognized pin ... ['P0','P1']
GLLVM_PARITY_PIN=P2 Rscript tools/core070_covariance_batch.R ...   -> Error: parity_pin %in% c("P0","P1") is not TRUE
GLLVM_PARITY_PIN=P1 Rscript tools/core070_covariance_batch.R ... (no CORE070_READBACK_DIR) -> Error: ... needs CORE070_READBACK_DIR
GLLVM_PARITY_PIN=P2 julia -e 'include("test/parity/core070_pin.jl")'  -> ERROR: "P2" is not a recognized pin
GLLVM_PARITY_PIN=""  julia -e 'include("test/parity/core070_pin.jl")'  -> ERROR: "" is not a recognized pin
python3 tools/test_parity_oracle_defaults.py -> OK (default P0, b4d5fee64)
julia --project=. test/parity/test_core070_pin.jl -> 16/16 pass, 13.1 s
node tools/test_true_parity_check.mjs -> all negative controls passed
GLLVMTMB_DIR=... python3 tools/core070_covariance_p1_contract.py --check -> CORE070_COVARIANCE_P1_CONTRACTS_CURRENT
```

Two nits, no action required: (i) `_core070_frozen_contract_rel()` in `core070_pin.jl` has a
`get(..., P0 path)` fallback for unknown pins; it is unreachable today because the pin lookup
errors first, and a future pin added to `core070_oracle_pins.toml` without a manifest entry would
be refused by `_core070_check_frozen_contract_pin` (error, not silent P0). A hard error there
would read more clearly. (ii) The two R scripts hard-code the P1 sha as a string literal instead
of reading `tools/core070_oracle_pins.toml`; `parity_oracle.py` is meant to be the single reader.

I did not run a P0 batch end to end: no P0 oracle library exists on this machine
(`.unlazy/core070-aghq/` is absent). P0 behaviour was checked by reading every switch (default
branch is the pre-PR path, byte for byte) and by the two P0-default self-tests above.

### 9. CLEAN — scope: no reclassification, no src/, Project.toml, .github/ or P0 evidence changes

```
$ git diff --name-only cb5688f7e 508d297d0 | grep -E '^(src/|Project.toml|\.github/)'   -> (none)
$ git diff --stat cb5688f7e 508d297d0 | tail -1
92 files changed, 21504 insertions(+), 18 deletions(-)
```

All 17 rows: `classification` and `executable_case_ids` identical to
`docs/dev-log/core070/required-source-case-map.json` (checked row by row). The 18 deletions are
all in the five harness files (path selection), none in `docs/dev-log/core070/*` P0 files.

### 10. NON-BLOCKING — tracked size: `oracle/source.json` is 1.25 MB of per-file hashes

Largest new tracked files: `receipts/covariance/oracle/source.json` 1,247,418 B (9,808 file
hashes for the whole gllvmTMB tree, including `.agents/skills/...`), `frozen-r070-contract-p1.toml`
220 KB, two raw mode-fit `result.toml` at ~105 KB, `covariance-formula-modes-raw__result.toml`
153 KB. No `.rds`, no install log, no Julia stdout beyond one line. The P0 equivalent inventory was
never tracked (it lived under `.unlazy/`). The inventory is reproducible from the pinned commit
(`git archive` + the tool's hash recipe reproduced it exactly, finding 3), and its tree hash is
already in `tools/core070_oracle_pins.toml`, so tracking the full list buys little. Judgement
call: either drop it and point `reference_source_inventory` at the tree hash + regeneration
command, or keep it deliberately as the cited inventory. The raw `result.toml` files are the
values the comparison blocks were computed from, so they should stay.

### 11. The `*-PUBLIC-R-BRIDGE` split proposal is a classification change — maintainer's call

Splitting the 9 bridge case ids out of the 7 partial rows changes those rows' `executable_case_ids`
in the maintainer-owned `required-source-case-map.json` and creates new rows that need a
classification and a disposition. Under the signed rules (classifications are the maintainer's
only; a name match never counts) that is not something the lane can do, and the PR correctly does
not do it. The PR's framing ("proposal only, nothing reclassified") is accurate.

### 12. CLEAN — PR body

No `@handle` anywhere in the body (grep `@[A-Za-z]` empty). Claims checked against evidence: 2/0/7/8/0
counts, 450/450, worst logLik gap 2.56e-11, wave6 7.3e-8 and 1.3e-6, 9 bridge refusals with
`GJL-GATE-STRUCTURED-TERMS` (8) and one early generic error (ORD-DEP), checker lines. All match
what I re-derived. No parity claim broader than the case level; "Draft. Not for merge until the
maintainer has read it" is present. One wording point: "The two covariance cases in that batch
pass on their own numbers, and I marked their rows numeric" should be read together with finding 1.

### 13. NON-BLOCKING — the runner's pin switch is incomplete: two oracle-receipt paths stay hardcoded and untracked

`test/parity/parity_helpers.jl:27-28` still reads the oracle receipts from fixed P0-era paths,
copies them verbatim into every required run, and does not validate their contents:

```
const _CORE070_ORACLE_BUILD_RECEIPT  = ".unlazy/core070-aghq/oracle-receipts/build.json"
const _CORE070_ORACLE_SOURCE_RECEIPT = ".unlazy/core070-aghq/oracle-source/source.json"
...  cp(source, joinpath(receipt_dir, basename(source)); force = false)      # no hash check
```

Reproduction: with `GLLVM_PARITY_PIN=P1` and a fresh worktree, `runparity.jl` fails
`required oracle receipt is missing: .unlazy/core070-aghq/oracle-receipts/build.json`, and then
again for `oracle-source/source.json`. It ran only after I copied the P1 `build.json` and
`source.json` to those untracked paths by hand. The tracked
`runparity-covariance-18/build.json` therefore came from whatever sat at that path in the
builder's tree (it is the P1 receipt, byte-identical to `oracle/build.json`, so the claim is
fine), but a stale P0 `build.json` at the same path would be copied into a P1 run's receipts with
no error, because the marker validation (finding 3) reads the installed library, not this file.

Two smaller run-environment notes from the same attempt, for whoever re-runs this: the R child
must see both the oracle library and the library holding `TMB` (`R_LIBS=<oracle>:<user lib>`;
`GLLVM_PARITY_R_LIBS` alone did not put the oracle first for `find.package`), and case subsets must
be whole fixture groups (`a Gaussian covariance fixture must be requested as its complete group`).
None of this is documented in the PR's "How it was run".

Suggested fix: key the two paths on the pin (or on `GLLVM_PARITY_ORACLE_DIR`) and, in
`_core070_copy_oracle_receipts!`, check the copied `build.json`'s `reference_commit` /
`source_tree_sha256` / `archive_sha256` against the selected pin before copying. Add the exact
environment (R_LIBS, marker, receipt dir, `.unlazy` staging, `baseline/`) to the PR body or a
`docs/dev-log/core070/true-parity-latest/README`.

## What I did not check

- The 16 remaining native/formula runparity cases (tight-control modes and both formula groups):
  blocked on the `baseline/` default-control staging (finding 3, finding 13). Their tracked
  receipts were verified by hash ties and by recomputing every comparison entry (finding 4), and
  the two fixed-group cases reproduced bit-for-bit.
- P0 batches end to end (no P0 library on this machine); P0 was checked by code reading and the
  two P0-default self-tests only.
- The R-only grammar batch and the bridge-boundary runner were not re-executed (their verifier log
  and TSV were read; `covariance-batch-p1/verify.log` reads `CORE070_COVARIANCE_BATCH_VERIFIED`).
- The default-control mode-fit baseline (`mode-fits-default-control/result.toml`) was not re-run.
- The generator's behaviour on a different `GLLVMTMB_DIR` (only `--check` was run).
- Whether `1598b2142` is the final head of #561 (the checker I ran is that commit's copy).
- Nothing on GitHub beyond `gh pr view 567`.
