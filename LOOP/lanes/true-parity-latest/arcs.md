# Arcs (status: TODO / IN PROGRESS / DONE / PAUSED; gates marked)

- [~] A0 WS0 (PRs #523 ledger+oracle, #524 pin, #526 case-map rows, #527 carry scan; all draft, reviewed where noted; landing needs Shinichi) WS0 additive re-pin to P1: P1 oracle beside P0, CAPABILITY_LEDGER_REF pinned, required P1 CI job, case-map rows for +25 exports and +11 S3 methods (proposed; Shinichi signs), stale-row scan under the carry rule, 20 isdm rows reclassified, `tools/true_parity_check.mjs` with dynamic row count, receipt resolution, negative controls; tracked ledger. (4 to 5 days)
- [x] A0r Recon of each new export at P1. DONE: reviews/p1-export-recon.md (slope family corrected to column grammar)
- [x] D1 GATE: Packet 1 and the P1 claim boundary. DONE: signed as recommended 2026-09-27 (D-295, packet-1.md)
- [x] A1a iSDM spec: DONE as draft PR #525 (032d90284), reviewed and revised; Packet 1b open
- [ ] A1b iSDM kernel and fitter in new files: IN PROGRESS (#514 landed 855542118)
- [ ] A1c iSDM twin tests and receipts
- [ ] A1d iSDM engine = "julia" route in gllvmTMB
- [ ] A2 small twins: ordinal_logit, extract_latent_scores; medium: zi_* (R semantics), meta/meta_V, multinomial LV, Wald ordination_uncertainty
- [ ] A3 re-measure at P1 (#527: 0 of 306 required P0 rows carry; 278 dangling, 21 stale); size the harness and compute first; T9 bindings; D3 to D5, D8; grouping pairing
- [ ] A4a gllvmTMB #1236 bridge rebase and finish; #1283 recorder
- [ ] A4b real-data workflows C1 to C5
- [~] A5 silent-failure backlog: #514 landed 855542118, #515 via #522 landed 52ed4281b; remaining #504 adapters, #505, Tweedie grouped
- [ ] A6 temporal at R's scope (INSIDE P1): spec DONE (#535); slice 1 = draft PR #543 under review; slice 2 next (needs the grammar lane's hook); A9 phylo latent A14/A15 (INSIDE P1): spec #545 reviewed, Packet 1d signed (D-300), build IN PROGRESS
- [ ] A7 column-coefficient grammar and A8 spatial_dep/spatial_*: OUTSIDE P1 (D-295). Write their signed-disposition rows in WS0; revisit at P2
- [ ] D2 GATE: Packet 2 (47 pending-decision rows, 22 spec-defect rows), week 2
- [ ] D3 GATE: Packet 3 (91 reverse-gap classes, Julia-only extras), week 5
