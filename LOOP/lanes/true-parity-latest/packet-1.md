# Packet 1: true parity at P1, the decisions only you can make (2026-09-27)

Pin P1 = gllvmTMB main 9539352f6 (0.7.1, untagged). Plan: `ultra-plan.md` in this folder. Each row has a recommendation and a reply you can paste. Reply with the row numbers you accept; anything you do not mention stays open, and the lane keeps building what does not depend on it.

Four items were answered on 2026-09-24 and are not asked again: the S4 probe, D3 Stage 1, Totoro #323 Track A, delta dispersion A. The older packet (`docs/dev-log/owed/2026-09-25-true-parity-decision-packet.md`, 36 rows) still stands for the 0.7.0 ledger. Its row 1 (B-06, which R version) is answered by your re-pin decision; Packet 2 (week 2) carries the rest forward.

## The claim boundary (answer this first; it sets the estimate)

| # | Question | Recommendation | Cost | Reply to paste |
|---|---|---|---|---|
| 0 | Which of temporal, column-coefficient grammar, spatial (`spatial_dep`, `spatial_*`) and phylo latent (A14, A15) are inside P1? | **Inside:** temporal (you asked for it by name) and phylo latent (a 0.7.0 gap we owe anyway). **Outside, by signed disposition, revisited at P2:** column grammar (a Gaussian point model with no intervals in R, 109 R files, 4 to 6 weeks) and spatial (Julia keeps its own SPDE stack as a documented extra). | About 12 to 14 weeks with this split, against 18 to 22 with everything inside, or 10 to 12 with only temporal inside. | "Boundary: temporal and phylo latent IN; column grammar and spatial OUT for P1, revisit at P2." |

## New rules for P1

| # | Question | Recommendation | Reply to paste |
|---|---|---|---|
| 1 | **Receipt carry rule.** Does a 0.7.0 receipt still count at P1? | Only when every file in its source pins is byte-identical at P1. Otherwise the row reads `PARTIAL_STALE_AT_P1` until it is re-measured or you sign it. 62 R and C++ files changed between the pins, so expect many stale rows; WS0's first output is that count. | "1: approve the carry rule." |
| 2 | **iSDM twin target and scope.** | R's public door, `gllvmTMB(..., family = isdm_sources(...))`: non-spatial, Laplace, first order plus `predict`. No intervals, because R's own capability ledger claims none (ISDM-01 to 03 are point-fit recovery). Julia's `SourceCovariance` stays as a documented Julia-only extra. Spatial iSDM waits until spatial is inside a boundary. | "2: approve the iSDM target as scoped." |
| 3 | **The +25 exports and +11 S3 methods since 0.7.0.** Who classifies them (twin, excluded, needs surface)? | The WS0 pull request proposes a class and a one-line reason for each; you sign them there in one pass. Nothing is signed by an agent. | "3: classify in the WS0 PR; I sign there." |
| 4 | **Ledgers tracked in git.** | Yes. Ledger, receipts and oracle move to `docs/dev-log/core070/true-parity-latest/` and `tools/true_parity_check.mjs`. The gitignored `.unlazy/` folder has already lost receipts: 20 isdm rows cite a path that no longer exists. | "4: track the ledgers." |
| 5 | **Name twins that mean different things** (for example R's `zi_*` versus Julia's two-part ZI). | A name match never counts. Such exports sit in a `SEMANTIC_DIVERGENCE` set and stay FORWARD until a twin with R's semantics exists or you sign a disposition. Julia's own design stays as a documented extra (your 2026-09-27 rule). | "5: approve the semantic-divergence rule." |
| 6 | **Who signs the `select_lv` row?** | You. The auto-d lane (#518, gllvmTMB #1324) builds and drafts the receipt; this lane only records it. The known twin difference (R rejects fits with a non-positive-definite Hessian) goes in the row as a fence. | "6: I sign select_lv; auto-d drafts the receipt." |

## Carried from the 0.7.0 programme

| # | Question | Recommendation | Reply to paste |
|---|---|---|---|
| 7 | **T11.** 38 API-alignment collisions, judged R-side defects, handed to the gllvmTMB lane and blocked since. | Re-measure the 38 at P1 in WS0 (some may be fixed in 0.7.1). Those still defective become signed R-side dispositions, outside C8, each linked to a gllvmTMB issue. | "7: re-measure T11 at P1; the rest become signed R-side dispositions." |
| 8 | **T12.** `unit_obs` exists only for the Gaussian two-level fit; the B2 row is PARTIAL. | Keep `unit_obs` for non-Gaussian families out of P1 and write it as a fence. It is new engine work, not a twin gap in the 0.7.0 sense. | "8: non-Gaussian unit_obs out of P1, fenced." |
| 9 | **B-04.** What does "pair" mean for the grouping levels `unit`, `unit_obs`, `cluster`, `cluster2` (C5)? | Same names on both engines plus the paired Gaussian receipts that exist today. Non-Gaussian numerical pairing leaves the claim (reopening it is 3 to 7 days). | "9: C5 = same names plus Gaussian paired receipts." |
| 10 | **T14.** NB2 Wald NaN at a degenerate optimum. Fixes F1 to F3 are on main; one cross-Julia-version seed gap is still named open. | Accept receipt-only closure for the F1 to F3 fix set; keep the cross-version seed as a named open sub-item. With B-10 of the older packet: the #478 restart only, no AGHQ-backed NB2. | "10: T14 closed for F1-F3; cross-version seed stays named open." |
| 11 | **T15.** Knife-edge single-seed fixtures: 18 were audited and dispositioned on 2026-09-14, with no test edits. | Ratify the audit. The lane rule already requires new tests to use literal or hash-verified data, never one seed's outcome. | "11: ratify the T15 audit." |
| 12 | **T5.** Eight partial rows: 7 re-bound on 2026-09-03 Totoro receipts; `loading_profile` needs a Julia surface (D3). | Re-check the 7 under the carry rule (row 1). `loading_profile` stays needs-surface and follows D3 Stage 1, already answered. | "12: re-check T5's 7 under the carry rule; loading_profile follows D3." |
| 13 | **Ledger-gap ranks** (inventory of 2026-09-15). | Superseded by the P1 workstream order in the plan. Keep the inventory as history. | "13: plan order replaces the ranks." |

## One-line reply if you accept every recommendation

"Packet 1: accept 0 to 13 as recommended."
