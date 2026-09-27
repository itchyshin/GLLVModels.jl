# Packet 1b: iSDM scope questions before the build (2026-09-27)

**SIGNED 2026-09-27 by Shinichi, in chat: "Packet 1b: accept 1 to 5 as recommended." Vault decision D-296.**

From the iSDM port spec (PR #525, head 032d90284, reviewed and revised). Each row has a recommendation and a reply to paste. Engineering questions the spec settles itself (cloglog kernel copy, name-based coefficient pairing, cross-objective pass criterion, gradient method) are not asked.

| # | Question | Recommendation | Reply to paste |
|---|---|---|---|
| 1 | R's in-sample `predict(..., se.fit = TRUE)` returns a fixed-effect-only delta SE (3 R assertions). Twin it in P1? | No. Fence it on the ISDM-03 predict row; twin only R's refusal of `se.fit` with newdata. It is an interval-like quantity and R's ledger claims no intervals for iSDM. | "1b-1: fence in-sample se.fit." |
| 2 | R's legacy two-source route (`family_var = "isdm_family"`, fixed names `gbif`/`survey_pa`) predates the public door. Twin it? | No. Record an R-only backward-compatibility disposition on ISDM-01; the shared code (offset gate, core predicate) is twinned anyway. | "1b-2: legacy route R-only disposition." |
| 3 | A declaration where every source is a count source needs no iSDM admission in R and keeps likelihood weights. | Accept it at the Julia door, file its receipt under the mixed-family rows, and fence weights (the Julia door has no likelihood weights in P1). | "1b-3: all-count accepted, weights fenced." |
| 4 | Missing responses in the long table. | Refuse `missing` in P1 with a clear error, and port R's observed-arm check as a contract test. | "1b-4: refuse missing in P1." |
| 5 | R's source-formula fits can have no `latent()` term. | Admit zero or one `latent()`; with none, the fit runs as a GLM through the same kernel (K = 0), so those R tests are twinned too. | "1b-5: admit K = 0." |

One-line reply: "Packet 1b: accept 1 to 5 as recommended."

Build estimate after the review: 1b 8 to 10 days (starts after #514 lands), 1c 4 to 5 days, 1d 2 to 3 days as a gllvmTMB PR.
