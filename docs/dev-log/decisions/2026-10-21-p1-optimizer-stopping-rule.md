# P1 optimizer stopping rule

Date: 2026-10-21

Status: SIGNED

Shinichi Nakagawa signed this stopping rule on 2026-10-21. The four campaign
receipts stay FAIL. The signature adopts the stopping standard below and
leaves the existing numeric bars unchanged. Receipts, gate files, and the
scoreboard stay untouched, and no row is marked as passing.

## Claim

On the four campaign rows below, Julia and R evaluate the same likelihood.
The leftover gaps are two nearby optimizer stops on a flat or nearly
separated surface. A tighter R refit did not bring every existing bar inside.

The likelihood is the score that says how well a set of parameters fits the
data. On a flat surface, many nearby parameter values score almost the same,
so two optimizers can stop a short step apart. On a nearly separated surface
the score stretches, which happens when a species is almost always absent or
almost always present, and the search can stop while a small slope remains.
The species associations and the fitted values can then differ by a small
amount while both programs are scoring the same model.

Where a matched-point check was recorded, the two Laplace scores already
agree much more closely than the comparison bars: about 8.05e-10 for the
spider fit, about 2e-9 for the ordinal fit, and about 1e-10 when the iSDM
predictor is rebuilt at the stored parameters. The refits below still leave
the existing bars unmet.

The adopted rule is a shared stopping standard. The existing numeric bars
stay at log-likelihood 1e-6, Lambda Lambda' 0.0001, linear predictor 1e-6,
and predict 1e-6.

## Adopted rule

Label: ADOPTED. Shinichi Nakagawa signed this rule on 2026-10-21.

A comparison counts only when both engines report a positive-definite Hessian
and a max absolute gradient below 1e-5. The Hessian is the curvature at the
reported fit; positive definite means the fit sits at a peak. The max
absolute gradient is the largest remaining slope of the likelihood. Points
that fail that standard stay outside the parity receipt even if some
parameter gaps are small.

Do not change the existing numeric bars (logLik 1e-6, Lambda Lambda' 0.0001,
linear predictor 1e-6, predict 1e-6).

## Spider, negative binomial

Wave 0 library, gllvmTMB 0.7.1, nlminb rel.tol 1e-14. Note:
`/tmp/p1-spider-reltol-wave0-20261021.md`.

Convergence code 0, relative convergence (4), 53 iterations, pdHess FALSE,
max absolute gradient 5.7423624730823519e-06, max Lambda Lambda' gap
1.6564122642392931e-05 (inside 0.0001), logLik gap versus Julia
1.4440465747611597e-06 (outside 1e-6), linear-predictor gap
5.4459055682798407e-06 (outside 1e-6). Same-point Laplace agreement from the
earlier note was about 8.05e-10. STILL_OUTSIDE.

## Fungi, binomial

P1 library, R 4.6.1, gllvmTMB 0.7.1. Note:
`/tmp/p1-fungi-reltol-20261021.md`.

logLik -5792.186252389, max absolute gradient 7.66689532527565e-05,
convergence code 1, false convergence (8), pdHess true, beta gap
3.413854359202162e-06, Lambda Lambda' gap 1.225400863269499e-05,
linear-predictor gap 1.056249763919936e-05. sp54 still flagged (prevalence
0.0433, max_loading 2.62). STILL_OUTSIDE.

## Ordinal logit

Wave 0 library, commit 9539352f. Note:
`/tmp/p1-ordinal-reltol-20261021.md`.

nlminb rel.tol 1e-14 stopped after 1 iteration, singular convergence (7),
convergence code 1, pdHess true, max absolute gradient
0.0004301884504441800, Lambda Lambda' (t01, t15) unchanged at
0.2439067508813110, gap versus Julia 0.000129891, above 0.0001. Same-point
logLik match was about 2e-9. STILL_OUTSIDE.

## iSDM headline

Note: `/tmp/p1-isdm-reltol-20261021.md`.

new logLik -23699.326170254481, max absolute gradient
7.1598713482323717e-4, convergence code 1, false convergence (8), 73
iterations, pdHess TRUE, beta gap 5.4255175069295447e-7, predict-link gap
1.3474597717944015e-5. Gradient still above 1e-5. PREDICT_STILL_OUTSIDE.
Rebuilding the predictor at stored parameters matched each engine to about
1e-10.

## Receipts

These four receipts stay FAIL.

Beetle stays blocked because R records pdHess false and the local file hash
does not match.

## Rows outside this stopping rule

Identity, animal_latent, bootstrap_Sigma, extract_phylo_signal, simulate,
and the two post-fit rows sit outside this stopping rule. Keep them on their
own evidence. Do not treat them as optimizer misses.

## Scoreboard

304 of 317 scoreboard rows are already bound. This note does not claim C0-C8
or X2 are MET on origin/main. Pull request 987 is separate and does not
close these rows.

## Signature

Name: Shinichi Nakagawa

Date: 2026-10-21

Signed text: a comparison counts only when both engines report a
positive-definite Hessian and a max absolute gradient below 1e-5. The
existing numeric bars stay unchanged. The four campaign receipts stay FAIL.
