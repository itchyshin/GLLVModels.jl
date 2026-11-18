# bootstrap_Sigma_r at the phy tier: n_boot = 999 descriptive runs (2026-11-02)

Evidence note. Julia only, no R comparison, no status, scoreboard, receipt or tolerance change. Each case is one dataset and one seed (`seed = 20261102`), so these numbers describe behaviour, not coverage, bias or calibration.

## Setup

- Call: `bootstrap_Sigma_r(fit; n_boot = 999, level = :phy, what = [:Sigma, :communality], seed = 20261102, progress = false)`.
- Totoro, one thread per process, four processes at once, load average about 235 (times are probably inflated). Longest chain about 29 minutes.
- Cases `t16_unique` and `t32_unique` (`JointPhyloGroupedGaussianFit`, 16 and 32 tips, d = 3, explicit unique, one `:unit` term) use the design function from the Stage 2 pre-run script.
- Cases `s1_bare`, `s1_unique` (`PrecisionMultivariateFit`, 16 tips, d = 3, rank 1, n = 64) and `two_term` (`t16_unique` plus a crossing `:cluster` term with 5 groups) are the runner's own choices, not an original protocol. The original design file was lost.
- For `PrecisionMultivariateFit` the failure breakdown and the non-PD count come from a wrapper around the refit call, because that class has no breakdown field and no identification check.

## Results

| case | n_failed | thrown / non-converged / non-identified | converged, non-PD Hessian | refit time mean / median / max (s) |
|---|---|---|---|---|
| s1_bare | 0 | 0 / 0 / n.a. | 0 of 999 | 0.013 / 0.012 / 0.24 |
| s1_unique | 3 | 0 / 3 / n.a. | 414 of 996 | 0.153 / 0.158 / 0.60 |
| t16_unique | 19 | 0 / 16 / 3 | 692 of 983 | 0.559 / 0.516 / 2.03 |
| t32_unique | 9 | 1 / 7 / 1 | 303 of 991 | 1.258 / 1.124 / 4.51 |
| two_term | 36 | 1 / 35 / 0 | 447 of 963 | 1.450 / 1.332 / 3.88 |

Interval widths (95% percentile), min / median / max over the three traits:

| case | Sigma_phy diagonal | communality_phy |
|---|---|---|
| s1_bare | 0.237 / 0.264 / 0.464 | 0 / 0 / 0 (all [1, 1]) |
| s1_unique | 0.241 / 0.699 / 1.205 | 0.704 / 0.831 / 0.989 |
| t16_unique | 0.874 / 1.003 / 1.234 | 0.880 / 0.942 / 0.997 |
| t32_unique | 0.447 / 0.787 / 0.961 | 0.842 / 0.965 / 0.986 |
| two_term | 0.463 / 0.825 / 1.578 | 0.948 / 0.964 / 0.983 |

- Every run warned that replicates were lost. The `JointPhyloGroupedGaussianFit` runs also gave the non-PD Hessian message. `t16_unique` warned that the original Hessian is not PD (smallest eigenvalue -4.45e-8).
- No `Sigma_phy` bound sits at or near 0. The `communality_phy` upper bound equals 1 in every unique-component case (all three traits in `t16_unique`, `t32_unique` and `two_term`; traits 1 and 2 in `s1_unique`). In `s1_unique` the original communality for trait 2 is already 1.0 (fitted unique variance at zero).

## What this shows

The phy-tier bootstrap completes at n_boot = 999 for all five cases. Between 0% and 3.6% of refits are lost, almost all non-converged. The loss rate looks higher for the larger or more boundary-prone fits, but that is five single cases, not a tested relationship.

## What this does not show

- Coverage, bias or calibration (one dataset and one seed per case).
- Whether the upper end at 1 is a faithful percentile or a boundary artefact (not checked).
- Agreement with R (no R run).
- Anything about the cases the runner chose itself beyond that they run.

## Process note

After a failed script edit the four jobs briefly ran twice (up to 8 processes against a cap of 4) for about a minute. The first set was killed by PID and only the second set's outputs are used.
