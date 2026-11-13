# P1 signed known-FAIL exception: beetle (RD-BEETLE-NB2)

Drafted 2026-11-10 by an agent from `drafts-w19/2026-11-09-p1-beetle-signed-exception-fixed-20261109.md`, corrected, and signed on 2026-11-13 by Shinichi's card answer (ruling SHINICHI-1408). The signature is recorded in the JSON file named below and in the Signature section at the end.

Tags: [ran] I computed or ran it this session. [read] I read it in a named file. [guess] my inference.

## What a signature would say

Row `data/RD-BEETLE-NB2` (scoreboard id `data-RD-BEETLE-NB2`, clause C4, also counted in X2) is ruled out of P1 scope as a signed known FAIL.

This is not a pass. The row stays FAIL in its receipt. A signed exception adds nothing to the pass count: the checker keeps `done_excluding_signed` at what it was, and the row is listed under `signed=`.

Beetle has never had a ruling of its own before this one: the 2026-10-21 stopping rule names four rows (spider, fungi, ordinal, iSDM) and says of beetle only that it "stays blocked because R records pdHess false" [read]. The case for excusing beetle is weaker than for spider, because its cause is not diagnosed. Shinichi signed it on 13 Nov 2026 by card answer ("Beetle: sign", ruling SHINICHI-1408), transcribed by the lane, on the strength of this sentence as the reason:

**R reports a non-positive-definite Hessian, cause not diagnosed.**

## Measured numbers (all FAIL, none changed)

Receipt `docs/dev-log/core070/true-parity-latest/receipts/data/campaign/RD-BEETLE-NB2.json`, sha256 `89c1f28d53cc3c004fd24525639ba867801470b79c6c8dfc9dd6ebe4de1c1286` [ran: hash and every figure below re-read from the file at PR 1105 head `fcbea7f1`]. NB2, 68 species, 87 sites.

| Quantity | Max abs gap | Bar | Result |
|---|---|---|---|
| logLik | 9.0807e-06 | 1e-6 | outside (9.1 times the bar) |
| beta (71 fixed effects) | 2.8714e-04 | 1e-4 | outside (2.9 times); 6 of 71 exceed [ran] |
| Lambda Lambda' | 4.6329e-03 | 1e-4 | outside (46 times) |
| linear predictor | 8.6945e-04 | 1e-6 | outside (869 times) |

Engine blocks [ran, from the receipt]: R (gllvmTMB 0.7.1) convergence 0 ("relative convergence (4)"), pdHess FALSE, max absolute gradient 2.9531e-03, cond(H) 3.3930e+08, with the warning `In sqrt(diag(sdr$cov.fixed)) : NaNs produced`. Julia 1.12.6 converged after 398 iterations in about 191 s. The receipt records no Julia gradient and `pd_hessian` is null.

Four species (sp13, sp28, sp30, sp37) have dispersion above 1e6 and Julia flags them in the receipt [ran]. R's values are 1.7e5, 2.8e8, 8.1e8 and 4.2e7 at R's point; Julia's are far larger (for sp37 about 9.7e74) [read, `wave19-matched-point.md`, `wave24-beetle-pathology.md`, and the 2026-10-20 note]. The row does not score dispersion.

## Reason, as far as measured

R reports a non-positive-definite Hessian (pdHess FALSE), cause not diagnosed.

What is measured about R: pdHess is FALSE, the standard errors give NaNs, and the cond(H) estimate is 3.4e8. R's Hessian failure alone keeps the row from counting under the signed stopping rule and under the receipt's own pass rule [read, receipt `pass_rule.legs.R_pdHess_true` = false].

What was measured afterwards on the Julia objective only (wave 19 and wave 24, Totoro Julia 1.10.12, finite-difference Hessian; read, not rerun by me) [read]:
- Julia's objective at R's own point reproduces R's recorded logLik to 4.9e-9, so the two engines score the same likelihood there. This corrects the old draft, which said nothing showed that.
- Julia's gradient at its own receipt point is about 7.4e-6, below 1e-5. The receipt itself never recorded it.
- The Julia-objective Hessian has 4 near-zero or slightly negative eigenvalues at Julia's optimum, so Julia's own Hessian check would also fail. The null directions are the log-dispersion coordinates of sp13, sp28, sp30 and sp37. The fourth direction (sp13) is identified by inference only.
- Dropping those four coordinates leaves a positive definite Hessian (smallest eigenvalue 5.1e-3).

What is not measured: why R's `sdreport` Hessian fails. The four Poisson-limit species are the likely cause [guess], and the old rule against claiming it stays: this file does not claim that they cause R's pdHess to be FALSE. No R-side test was run.

Record discrepancy about the data file, now narrowed: the 2026-10-21 stopping rule says the local file hash does not match. The 2026-10-20 note explains it as a local rebuild (hash `5ea37a22...`), whose file the later pass did not re-open [read]. Wave 19 recomputed the sha256 of `beetle_wide.csv` on Totoro as `1d7603dd...`, equal to the hash in both the R and the Julia receipts [read; the run was not repeated by me]. I take the mismatch as settled by that one recomputation, but it is still one source.

## What does not change

- No numeric bar changes: logLik 1e-6, beta 1e-4, Lambda Lambda' 1e-4, linear predictor 1e-6, predict 1e-6.
- The stopping rule of 2026-10-21 stays in force for every other row.
- No refit, no new capture, no receipt edit.
- This is not a precedent for waiving other rows. It covers this one row and this one measured state.

## Revisit

At P2 planning, or earlier if the optional matched-point diagnostic with an R half (R's objective at Julia's optimum, plus R's Hessian eigenvalues) is run and assigns a cause. The exception ends when the row passes under the signed standard.

## Revoke path (corrected)

The signature lives in one file, `docs/dev-log/core070/true-parity-latest/signed-exceptions/data-RD-BEETLE-NB2.json`, not in the scoreboard row. The old draft said to delete `signed_by` and `signed_on` from the row. That would not work, because the assembler reads the file.

1. Shinichi says "revoke beetle exception" in chat.
2. An agent opens a reviewed change that deletes that JSON file (or sets its `state` to something other than `SIGNED`) and runs `python3 tools/true_parity_assemble.py`. The row returns to FAIL with the cell "not bound; cited: ...".
3. The checker then prints `signed=0` for the row. No receipt is rewritten, because the receipt was never changed.

A changed receipt also voids the exception: the file pins the receipt by sha256 [ran in the checker's own tests, see README].

## Conditions before a signature takes effect

1. The checker patch (`signed-exception-checker-20261109.patch`) must be on the branch that carries the signed file. Without it the checker ignores the file and the row stays FAIL.
2. The signed file must be committed, and the scoreboard regenerated with `python3 tools/true_parity_assemble.py` [ran in a throwaway copy].
3. The old open item "hash discrepancy" is narrowed above; Shinichi should read that paragraph before signing.
4. Public pages that quote counters are found and updated with Shinichi's approval.

## Required public wording (corrected)

The old text said "316 of 317" for the one-signed case. That is wrong. Counting a signed exception as matched would make 316, and the rules forbid that. The most the rules allow is 315 of 317 matched. Today the checker prints X2 304 of 317 [ran], so 315 is a ceiling that needs all 11 other open rows closed by evidence, not a current figure.

Use the number the checker prints as `done_excluding_signed=`, never the `done=` figure (with a signed exception `done=` rises by one per signed row [ran]).

> N of 317 matched, one real-data case (beetle) excluded by signed exception, the rest as listed in the checker output.

With N the checker's `done_excluding_signed`. At the ceiling (N = 315) and spider still failing: 315 matched + 1 excluded + 1 still failing = 317. If spider is signed too: "315 of 317 matched, two real-data cases excluded by signed exception" (315 + 2 = 317).

"True parity" never appears without the exception clause. Plain `X2_MET` cannot print under these rules; the checker prints `X2_MET_WITH_SIGNED_EXCEPTIONS` at best.

## Signature

Signed. The checker does not read this block. It reads the JSON file named above.

signed_by: itchyshin

signed_on: 2026-11-13

signature_source: Shinichi's card answer of 13 Nov 2026, 2:08 PM MT (ruling SHINICHI-1408 in the lane decision file: "Spider: sign" and "Beetle: sign"). Transcribed by the lane, not typed by him.
