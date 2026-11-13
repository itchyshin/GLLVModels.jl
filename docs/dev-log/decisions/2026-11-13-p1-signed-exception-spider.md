# P1 signed known-FAIL exception: spider (RD-SPIDER-NB2)

Drafted 2026-11-10 by an agent from `drafts-w19/2026-11-09-p1-spider-signed-exception-fixed-20261109.md`, corrected, and signed on 2026-11-13 by Shinichi's card answer (ruling SHINICHI-1408). The signature is recorded in the JSON file named below and in the Signature section at the end.

Tags: [ran] I computed or ran it this session. [read] I read it in a named file. [guess] my inference.

## What a signature would say

Row `data/RD-SPIDER-NB2` (scoreboard id `data-RD-SPIDER-NB2`, clause C4, also counted in X2) is ruled out of P1 scope as a signed known FAIL.

This is not a pass. The row stays FAIL in its receipt. A signed exception adds nothing to the pass count: the checker keeps `done_excluding_signed` at what it was, and the row is listed under `signed=`. The signature records that P1 is judged without this row, and why.

The exception was signed by Shinichi's card answer of 13 Nov 2026 ("Spider: sign", ruling SHINICHI-1408), transcribed by the lane. Beetle has its own file and its own signature.

## Measured numbers (all FAIL, none changed)

Receipt `docs/dev-log/core070/true-parity-latest/receipts/data/campaign/RD-SPIDER-NB2.json`, sha256 `343e9f683a565f699c2814ada7112de88c99c9e31c105ef10e775aaf7e4fb877` [ran: hash and every figure below re-read from the file at PR 1105 head `fcbea7f1`]. NB2, 12 species, 28 sites.

| Quantity | Max abs gap | Bar | Result |
|---|---|---|---|
| logLik | 1.4573e-06 | 1e-6 | outside (1.5 times the bar) |
| beta (15 fixed effects) | 4.9511e-05 | 1e-4 | inside |
| Lambda Lambda' | 1.6173e-04 | 1e-4 | outside (1.6 times) |
| linear predictor | 6.7399e-05 | 1e-6 | outside (67 times) |

Engine blocks in that receipt [ran]: R (gllvmTMB 0.7.1) convergence 0, pdHess TRUE, max absolute gradient 2.9438e-03, cond(H) 7.87e+08. Julia 1.12.6 converged after 127 iterations, `pd_hessian` FALSE. The receipt's own pass rule fails on "every quantity within tolerance" only; its R pdHess leg is true.

Tighter R refit in the signed stopping rule of 2026-10-21 (gllvmTMB 0.7.1, nlminb rel.tol 1e-14, 53 iterations) [read, `decisions/2026-10-21-p1-optimizer-stopping-rule.md`]: max absolute gradient 5.7424e-06, pdHess FALSE, logLik gap 1.4440e-06 (outside), Lambda Lambda' gap 1.6564e-05 (inside), linear-predictor gap 5.4459e-06 (outside).

The two R runs disagree on pdHess (TRUE in the committed receipt, FALSE after the tighter refit). They are different runs. Any quote must name the run. Neither is chosen here.

## Same-point agreement: the two numbers, reconciled by source

Quantity: the Julia grouped-NB2 Laplace objective (`nb_grouped_marginal_loglik_laplace`, observed curvature) evaluated at R's stored parameters, minus R's stored logLik of -686.167996628694. It shows that both engines score the same likelihood at one point. It is supporting evidence, not a bar. It is a Julia-only half: R was never asked to evaluate its objective at Julia's point [read, `wave19-matched-point.md` section 6].

| Figure | Julia objective at R's point | Difference | Run | Source |
|---|---|---|---|---|
| 8.05e-10 | -686.167996629499 | -8.05e-10 | Julia 1.10.0, local Mac, loadings = eigenfactor of stored Lambda Lambda', inner tol 1e-9, maxiter 100 | committed note `docs/dev-log/2026-10-20-p1-overnight-spider-beetle.md` line 19; repeated in the 2026-10-21 stopping rule and spider audit note [read; subtraction redone by hand: [ran]] |
| 9.5e-10 | -686.1679966296413 | -9.47e-10 | Julia 1.10.12, Totoro, point rebuilt from the raw receipt with a lower-triangular rank-2 factor | `wave19-matched-point.md` section 2 [read; subtraction redone by hand: [ran]. The Totoro run itself was not repeated, and its raw output is not in the saved copy] |

Verdict: both are real and both trace to a source. They are the same quantity measured twice, by two Julia versions and two ways of building the loading factor. The two Julia objective values differ from each other by 1.4e-10, and no source says why (version, factor rebuild and inner tolerance all differ). So neither is "the" figure, and I did not pick one. The sentence that belongs in a signed text is: "below 1e-9 in both runs (8.05e-10 on Julia 1.10.0, 9.5e-10 on Julia 1.10.12)". Either value is about 1000 times under the 1e-6 logLik bar.

## Reason, as far as measured

1. Under the signed stopping rule (a comparison counts only when both engines report a positive-definite Hessian and a max absolute gradient below 1e-5), the row cannot count. Julia's `pd_hessian` is false [read, receipt]. R's gradient in the committed receipt is 2.9e-3, above 1e-5. After the tighter refit R's gradient is 5.7e-6 but its pdHess is FALSE.
2. Three species (sp01, sp05, sp11) sit at the Poisson limit on both engines (dispersion above 1e6; Julia warns for groups 1, 5 and 11 in the receipt) [ran: read from the receipt's `engine_messages`; read, wave 19 and wave 24 for the R-side sizes 5.6e8, 9.8e5 and 1.5e9].
3. Wave 19 and wave 24 measured, on the Julia objective only, that the Hessian is singular in exactly the log-dispersion directions of those three species, and that dropping those coordinates leaves a positive definite Hessian with smallest eigenvalue 0.0411 [read; finite differences, Julia objective, not rerun by me]. This is a measurement of the likelihood surface. It is not a measurement of R's `sdreport` Hessian.
4. The source audit of 2026-10-21 (`2026-10-21-p1-spider-gpt.md`) found no new bug in dispersion grouping, covariate order or species order [read].

Not measured: why R's pdHess is FALSE after the tighter refit. The Julia-side finding makes the Poisson-limit species the likely cause [guess]; no R-side test was run.

Reason on record if signed: "Julia pdHess false; three species at the Poisson limit on both engines; the Julia-objective Hessian is singular in exactly those directions; R's own pdHess cause not measured."

## What does not change

- No numeric bar changes: logLik 1e-6, beta 1e-4, Lambda Lambda' 1e-4, linear predictor 1e-6, predict 1e-6.
- The stopping rule of 2026-10-21 stays in force for every other row.
- No refit, no new capture, no receipt edit.
- This is not a precedent for waiving other rows. It covers this one row and this one measured state.

## Revisit

At P2 planning, or earlier if a matched-point diagnostic with an R half (R's objective at Julia's optimum, plus Hessian eigenvalues) is run and shows a cause. The exception ends when the row passes under the signed standard.

## Revoke path (corrected)

The signature lives in one file, `docs/dev-log/core070/true-parity-latest/signed-exceptions/data-RD-SPIDER-NB2.json`, not in the scoreboard row. The old draft said to delete `signed_by` and `signed_on` from the row. That would not work, because the assembler reads the file.

1. Shinichi says "revoke spider exception" in chat.
2. An agent opens a reviewed change that deletes that JSON file (or sets its `state` to something other than `SIGNED`) and runs `python3 tools/true_parity_assemble.py`. The row returns to FAIL with the cell "not bound; cited: ...".
3. The checker then prints `signed=0` for the row. No receipt is rewritten, because the receipt was never changed.

A changed receipt also voids the exception: the file pins the receipt by sha256, and the checker refuses a stale hash [ran in the checker's own tests, see README].

## Conditions before a signature takes effect

1. The checker patch (`signed-exception-checker-20261109.patch`) must be on the branch that carries the signed file. Without it the checker ignores the file and the row stays FAIL. Reviewed in `README.md`.
2. The signed file must be committed, and the scoreboard regenerated with `python3 tools/true_parity_assemble.py` (the assembler reads the file) [ran in a throwaway copy].
3. The old draft's open item "assembler not read" is closed: the patch adds the assembler path, and `--check` reports `ASSEMBLE_OK` after the rehearsal [ran].
4. Public pages that quote counters are found and updated with Shinichi's approval.

## Required public wording (corrected)

The old text said "316 of 317". That is wrong. Counting a signed exception as matched would make 316, and the rules forbid that. The most the rules allow is 315 of 317 matched. Today the checker prints X2 304 of 317 [ran, `PARITY_REF=FS node tools/true_parity_check.mjs X2` on `fcbea7f1`], so 315 is a ceiling that needs all 11 other open rows closed by evidence, not a current figure.

Use the number the checker prints as `done_excluding_signed=`, never the `done=` figure (with a signed exception `done=` rises by one per signed row [ran]).

> N of 317 matched, one real-data case (spider) excluded by signed exception, the rest as listed in the checker output.

With N the checker's `done_excluding_signed`. At the ceiling (N = 315) and beetle still failing: 315 matched + 1 excluded + 1 still failing = 317. If beetle is signed too: "315 of 317 matched, two real-data cases excluded by signed exception" (315 + 2 = 317).

"True parity" never appears without the exception clause. Nothing says P1 is met on the numbers alone. Plain `X2_MET` cannot print under these rules; the checker prints `X2_MET_WITH_SIGNED_EXCEPTIONS` at best.

## Signature

Signed. The checker does not read this block. It reads the JSON file named above.

signed_by: itchyshin

signed_on: 2026-11-13

signature_source: Shinichi's card answer of 13 Nov 2026, 2:08 PM MT (ruling SHINICHI-1408 in the lane decision file: "Spider: sign" and "Beetle: sign"). Transcribed by the lane, not typed by him.
