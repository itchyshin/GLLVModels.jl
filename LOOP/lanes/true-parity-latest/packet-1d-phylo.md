# Packet 1d: phylo latent (A14, A15) scope questions (2026-09-27)

From the phylo latent port spec (draft PR #545, head c0fd5f5be, reviewed and revised). Estimate about 8.5 agent-days; the twin builds on the R-shaped `PrecisionPhy` / `fit_precision_multivariate` path that already pairs with 0.7.0 to 1e-14. Counts: 26 R blocks twinned, 16 deferred, 23 owned by the temporal spec (#535), 261 fenced with stated reasons.

| # | Question | Recommendation | Reply to paste |
|---|---|---|---|
| 1 | Ratify the phylo transport defaults Q1 to Q4 from the 0.7.0 programme | Ratify; the twin path uses `correlation = true`, native constructors stay opt-in | "1d-1: ratify Q1-Q4." |
| 2 | A15 test shape | 100 species x 5 replicates x 20 traits, d = 2 (replication identifies the residual) | "1d-2: A15 shape as proposed." |
| 3 | Species matching on the twin entry | By label, as R does | "1d-3: match by label." |
| 4 | R's 1e-8 ridge on a dense vcv | Replicate it exactly (pairing is the point) | "1d-4: replicate the ridge." |
| 5 | `extract_phylo_signal` on a bare phylo fit | Return H2 = 1 with V_eta, as R does, instead of refusing; refuse `ci = true` | "1d-5: port H2 = 1; refuse ci." |
| 6 | Entry point | Named `fit_phylo_latent_gllvm`; `fit_gllvm(; phylo = ...)` untouched | "1d-6: named entry." |
| 7 | The #129 profile scale-tag bug | Separate PR after the Gaussian-intercepts lane (#519) lands | "1d-7: #129 after #519." |
| 8 | Gradient | Finite differences now; a selected-inverse gradient only if A15 takes over an hour | "1d-8: FD now." |
| 9 | Promotion sign-off for A14/A15 | A dated maintainer block in the receipt PR body | "1d-9: dated block." |
| 10 | `STRUCT-PHY-TREE-PROPTO` (`phylo_scalar`) is a planned case in A14's own gate row | Fence it; A14 promotes on its two `phylo_latent` cases with this one UNPAID (a different, scalar-variance model; a small later slice if wanted) | "1d-10: fence PROPTO." |
| 11 | Is non-Gaussian `phylo_latent()` inside P1? | Inside by D-295's wording, but a later slice via `src/phylo_glm.jl`, not this spec (R's own evidence is heavy-gated) | "1d-11: non-Gaussian phylo_latent is a later slice." |
| 12 | R accepts `rho` on a bare `phylo_latent()`; the twin would refuse `rho != 1` with a labelled scope fence | Accept the fence (separate estimand, own R tests, no A14/A15 case names it) | "1d-12: accept the rho fence." |

One-line reply: "Packet 1d: accept 1 to 12 as recommended."
