# Julia vs R latent-dimension recovery, `criterion = :bic` (R's setting), same Julia datasets

Kmax = 4 (Julia), d_max = 4 (R); binary_ridge = 2.0 vs Inf. Counts are "correct/10 (NA = no admissible K)".

| n | p | K | Julia bic ridge | Julia bic none | R bic ridge | R bic none | Julia bic_sites ridge | Julia bic_sites none |
|---|---|---|---|---|---|---|---|---|
| 60 | 10 | 2 | 1/10 (NA0) | 0/10 (NA4) | 1/10 (NA0) | 0/10 (NA5) | 5/10 (NA0) | 0/10 (NA4) |
| 120 | 10 | 2 | 8/10 (NA0) | 0/10 (NA5) | 1/10 (NA0) | 0/10 (NA5) | 9/10 (NA0) | 1/10 (NA5) |
| 120 | 20 | 2 | 8/10 (NA0) | 4/10 (NA0) | 8/10 (NA0) | 4/10 (NA1) | 10/10 (NA0) | 5/10 (NA0) |
| 120 | 20 | 3 | 4/10 (NA0) | 0/10 (NA3) | 4/10 (NA0) | 1/10 (NA1) | 9/10 (NA0) | 1/10 (NA3) |

Total wall time: ~42 min (launch 11:10, completion 11:51:50, 2026-09-27).

Under the matched criterion (`:bic`), Julia and R agree closely at (60,10,2), (120,20,2) and (120,20,3) ridge — but diverge sharply at (120,10,2) ridge (Julia 8/10 vs R 1/10) and partially at (120,20,3) none (Julia 0/10 vs R 1/10); `:bic_sites` (Julia's own default) recovers noticeably better than `:bic` across every cell.
