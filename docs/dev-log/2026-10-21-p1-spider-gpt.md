# RD-SPIDER-NB2 focused source audit

Date: 2026-10-21. This was a read-only audit except for this new note. No Julia
process was started. No source, test, receipt, gate, tolerance, or existing note
was changed. No pathology split was run.

## Verdict

RD-SPIDER-NB2: NO_NEW_BUG

RD-BEETLE-NB2: BLOCKED_BY_R_PDHESS

## Spider evidence

The grouped-NB2 likelihood does not expose a second defect in dispersion
grouping, covariate column order, or species order.

Dispersion grouping is species aligned. The formula route sends
`NegativeBinomial` with covariates to `fit_nb_gllvm_grouped_cov` at
`src/formula.jl:359-360`. That fitter defaults `group` to `1:p`, sorts the
labels, and builds `gidx` in trait order at
`src/families/grouped_dispersion.jl:936-955`. It then expands the fitted
group dispersions with `rg[gidx[t]]` at
`src/families/grouped_dispersion.jl:970-979`. The raw files contain twelve
dispersions on each side, one for each of `sp01` through `sp12`. The three
Poisson-limit traits are the same species on both sides: `sp01`, `sp05`, and
`sp11`. The finite dispersions also match by position. For example, `sp02` is
1.575178494 in R and 1.575177798 in Julia. This is incompatible with a
dispersion-group permutation.

Covariate columns are aligned. The Julia campaign declares the covariates as
`ConWate`, `BareSand`, and `CovMoss` in that order and fills the third axis of
`X` in the same order at `tools/true_parity/campaign/run_J.jl:194-202`.
The formula front end obtains the StatsModels columns in formula order and
broadcasts each site column over species at `src/formula.jl:300-339`. The R
formula names the same three shared slopes in the same order at
`tools/true_parity/campaign/run_R.R:94-99`. The raw R coefficient names end
with `ConWate`, `BareSand`, `CovMoss`, while Julia stores `gamma[1]`,
`gamma[2]`, `gamma[3]` after the twelve intercepts. Their paired values are
close: the maximum difference over all fifteen fixed effects is
4.9511286195569326e-05. A swapped covariate column would not produce that
coordinatewise agreement.

Species order is aligned. The generator writes species columns as `sp01`
through `sp12` at `tools/true_parity/campaign/gen_data.R:121-127`. Julia
preserves the CSV header order when it builds `Y` and records
`trait_levels` at `tools/true_parity/campaign/run_J.jl:196-200`. R constructs
the long data with the same wide-column order and fixes the factor levels to
that order at `tools/true_parity/campaign/run_R.R:65-71`. The raw R
`trait_levels` and `LLt_trait_names`, and the raw Julia `trait_levels`, are
identical: `sp01` through `sp12`. The close paired intercepts and the
trait-fast predictor agreement provide a second check against a species
permutation.

The first-pass objective result therefore remains decisive. The Julia
grouped-NB2 Laplace objective evaluated at R's parameters differs from R's
stored log-likelihood by only 8.05e-10. The receipt's 1.46e-6 log-likelihood
gap, the `sp03` `Lambda Lambda'` gap of 0.000161728, and the linear-predictor
gap of 6.74e-5 compare two points on that same objective. R stopped with a
maximum absolute gradient of about 0.00294, while Julia's was about 6.8e-6.
The source and receipt evidence supports an optimizer stopping-point
difference. This audit found no new Julia mapping defect.

## Beetle boundary

RD-BEETLE-NB2 is BLOCKED_BY_R_PDHESS because the R receipt records
`pdHess = false`. In addition, the locally rebuilt beetle file does not match
the receipt hash
`1d7603ddceac61ea7179ed9519cb87bc5b7557ec1dd30f9267bd6224cd952c9b`.
This audit does not treat beetle as fixable in Julia.
