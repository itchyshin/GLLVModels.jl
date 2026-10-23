# P1 Behaviour and Postfit Open-Row Check (2026-10-21)

## Question

After C1 behaviour was reported MET with its four named rows receipted, check `docs/dev-log/core070/true-parity-latest` for any behaviour or postfit row that remains open outside the thirteen already-known unresolved rows:

1. `RD-URBANISATION-BINOMIAL` (`data/RD-URBANISATION-BINOMIAL`)
2. `RD-SPIDER-NB2` (`data/RD-SPIDER-NB2`)
3. `RD-BEETLE-NB2` (`data/RD-BEETLE-NB2`)
4. `RD-FUNGI-BINOMIAL` (`data/RD-FUNGI-BINOMIAL`)
5. `FAMILY-00-IDENTITY` (`family/FAMILY-00-IDENTITY`)
6. `ORDINAL-LOGIT-RSZ` (`family/ORDINAL-LOGIT-RSZ`)
7. `ISDM-HEADLINE-RSZ` (`isdm/ISDM-HEADLINE-RSZ`)
8. `simulate gllvmTMB_multi` (`namespace/S3method/simulate,gllvmTMB_multi`)
9. `animal_latent` (`namespace/export/animal_latent`)
10. `bootstrap_Sigma` (`namespace/export/bootstrap_Sigma`)
11. `extract_phylo_signal` (`namespace/export/extract_phylo_signal`)
12. `POST-COEF-EMPTY` (`postfit-policy/POST-COEF-EMPTY`)
13. `POST-SIMULATE-DEFAULT` (`postfit-policy/POST-SIMULATE-DEFAULT`)

## Verdict

`NONE_OPEN`

Every behaviour row and every postfit row in `docs/dev-log/core070/true-parity-latest` outside the thirteen excluded rows is already closed (`EVIDENCED`, `EVIDENCED-BEHAVIOURAL`, or `DISPOSITION-SIGNED`).

## Ledger Accounting in `docs/dev-log/core070/true-parity-latest`

### 1. Assembled P1 Ledger (`scoreboard.md` and `case-map-assembled.json`)

`scoreboard.md` and `case-map-assembled.json` contain 317 required rows across 9 families:

- 206 `EVIDENCED` (numeric)
- 63 `EVIDENCED-BEHAVIOURAL`
- 35 `DISPOSITION-SIGNED`
- 13 unresolved rows (3 `REGISTRATION-ONLY`, 2 `PARTIAL`, 1 `NON-DISCRIMINATING`, 6 `FAIL`, 1 `NOT-MEASURED`)

The 304 closed rows plus the 13 excluded rows account for all 317 rows in `case-map-assembled.json`. Zero additional rows are open in any family.

### 2. C1 Headline Case Map (`case-map.json`)

`case-map.json` tracks 38 headline rows:

- 22 `numeric` (`EVIDENCED`)
- 4 `behavioural` (`EVIDENCED-BEHAVIOURAL` via `test/test_c1_behaviour_p1.jl` and `receipts/c1-behaviour-p1/`):
  - `latent-scores/extract_latent_scores.default`
  - `select-lv/print.gllvmTMB_select_lv`
  - `model-comparison/print.anova.gllvmTMB_multi`
  - `model-comparison/update.gllvmTMB_multi`
- 12 `DISPOSITION-SIGNED` (`excluded`, `semantic_divergence`, `outside_boundary`, `retired`)

Zero rows in `case-map.json` are open.

### 3. Behavioural Scope Rows Across All Case Maps

Across the frozen Ruling 2 behavioural scope and its signed extensions (Rulings A, N6, and N10 in `GATES.md`), every behavioural row is closed:

- **C1 headline (`case-map.json`)**: 4 of 4 bound as `behavioural`.
- **Inference (`case-map-inference.json`, 63 rows total)**:
  - 50 rows bound as `EVIDENCED-BEHAVIOURAL` (49 from the initial 59 inference behavioural rows plus `inference/CI-ROUTE-009` added under Item A).
  - 10 rows closed as `DISPOSITION-SIGNED` under Item B (`CI-ROUTE-015`, `CI-ROUTE-043`, `CI-ROUTE-046`, `CI-ROUTE-055`, `CI-ROUTE-058`, `CI-ROUTE-061`, `CI-ROUTE-065`, `CI-ROUTE-067`, `CI-ROUTE-068`, `CI-ROUTE-070`).
  - 3 rows bound as numeric `EVIDENCED` (`CI-ROUTE-008`, `CI-ROUTE-010`, `CI-ROUTE-011`).
  - 0 rows open.
- **AGHQ control (`case-map-aghq.json`, 27 rows total)**:
  - 7 control rows (`aghq-control/AGHQ-CTRL-AUTO`, `AGHQ-CTRL-FALSE`, `AGHQ-CTRL-NINE`, `AGHQ-CTRL-NULL`, `AGHQ-CTRL-ONE`, `AGHQ-CTRL-TRUE`, `AGHQ-CTRL-TWO`) bound as `EVIDENCED-BEHAVIOURAL`.
  - 19 rows bound as numeric `EVIDENCED` and 1 row closed as `DISPOSITION-SIGNED` (`aghq-helper/AGHQ-HELPER-REML-GUARD`).
  - 0 rows open.
- **iSDM public-door (`case-map-isdm.json`, 7 rows total)**:
  - 5 public-door rows (`isdm/ISDM-COUNT`, `isdm/ISDM-EXTRA-SOURCE`, `isdm/ISDM-MISSING-IN-TRAIT`, `isdm/ISDM-MISSING-SOURCE`, `isdm/ISDM-WRAPPER-LAW`) bound as `EVIDENCED-BEHAVIOURAL` (with `test/test_isdm_public_door_p1.jl` and `receipts/isdm/first-seven-p1/`).
  - 1 row bound as numeric `EVIDENCED` (`isdm/ISDM-POISSON-BINOMIAL`).
  - 1 row open (`isdm/ISDM-HEADLINE-RSZ`, `FAIL`), which is in the thirteen excluded rows.
- **Postfit (`case-map-postfit.json`, 52 rows total)**:
  - 1 behavioural row (`postfit/POSTFIT-SURFACE-check_auto_residual`) bound as `EVIDENCED-BEHAVIOURAL` (with `test/test_check_auto_residual_p1.jl` and `receipts/postfit/first-seven-p1/`).

The 63 `EVIDENCED-BEHAVIOURAL` rows on `scoreboard.md` are exact: 50 (`inference`) + 7 (`aghq`) + 5 (`isdm`) + 1 (`postfit`) = 63, plus the 4 `behavioural` rows in `case-map.json`.

### 4. Postfit Rows (`case-map-postfit.json`)

`case-map-postfit.json` contains 52 rows (36 `postfit/POSTFIT-SURFACE-*` and 16 `postfit-policy/POST-*`):

- 45 `EVIDENCED` (numeric): 33 `postfit/POSTFIT-SURFACE-*` rows and 12 `postfit-policy/POST-*` rows.
- 1 `EVIDENCED-BEHAVIOURAL`: `postfit/POSTFIT-SURFACE-check_auto_residual`.
- 4 `DISPOSITION-SIGNED`:
  - `postfit/POSTFIT-SURFACE-.proportions_bootstrap_ci` (Ruling N2)
  - `postfit/POSTFIT-SURFACE-.proportions_wald_ci` (Ruling N2)
  - `postfit-policy/POST-NOBS-FALLBACK` (Ruling N4)
  - `postfit-policy/POST-PREDICT-DEFAULT` (Ruling N4)
- 2 open rows:
  - `postfit-policy/POST-COEF-EMPTY` (`NON-DISCRIMINATING`)
  - `postfit-policy/POST-SIMULATE-DEFAULT` (`PARTIAL`)

Both open postfit rows are in the thirteen excluded rows. No other postfit row is open.
