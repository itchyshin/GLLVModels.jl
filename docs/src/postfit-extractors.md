# Post-fit extractors

These `extract_*`/`get*` functions are read-only accessors on an already-fitted
model. Every one is a thin reader over a quantity the fit already computed —
none of them re-optimises anything — and most are `gllvmTMB`-style snake_case
mirrors of an accessor Reader may already know from the R package. They are
grouped here by what they read: the implied covariance structure (`Σ_y` at a
tier), loadings, and the tier-scoped summaries (communality, correlation,
proportions, Ω, ICC, repeatability).

## Tiers, and what "level" means

The `level` keyword selects a source of variation, also called a **tier**.
For a Gaussian `GllvmFit`, `:unit` selects the `Λ_B` source and its diagonal
variance; `:unit_obs` selects the `Λ_W` source and its diagonal variance.
Supported levels depend on the fit type: `extract_Sigma` for `GllvmFit`
does not accept `:phy`.

Choose the denominator to match your biological question:

- `extract_communality`, `extract_correlations`, and `extract_proportions`
  default to `level = :unit`. They describe the selected source alone and
  exclude Gaussian observation noise `σ_eps²`.
- `communality(fit)` and `extract_communality(fit; level = :total)` describe
  the unit latent fraction of `sigma_y_site(fit)`. That denominator includes
  the other site-specific variances and observation noise, but excludes the
  structured phylogenetic block. The analogous `:total` option is available
  for `extract_correlations` and `extract_proportions`.
- `extract_Omega` defaults to `level = :auto`, combining the sources present
  in the fit without observation noise. Its `level = :total` option adds
  observation noise; it is a different summary from `sigma_y_site`.

For example, with one latent source and no source-specific diagonal
variance, `extract_communality(fit)` is `1.0` wherever that source has
positive variance. This means all variation **within that source** is
shared. It does not mean all response variation is shared: if observation
noise is positive, `communality(fit)` is smaller than `1.0`. The two
denominators coincide only when the additional variance contributions
are zero. A source with zero total variance has an undefined fraction.

Check the individual accessor's definition when comparing outputs:
`extract_Sigma(fit; level = :unit_obs)` includes observation noise, unlike
the source-only `extract_communality` denominator at that level.
`extract_Sigma(fit; level = :site)` returns `sigma_y_site(fit)`.

```@docs
extract_Sigma
extract_Sigma_table
extract_loadings
extract_rotated_loadings
extract_residual_cov
extract_residual_cor
getResidualCov
getResidualCor
extract_communality
extract_correlations
extract_cross_correlations
extract_proportions
extract_Omega
extract_ICC_site
extract_repeatability
extract_cutpoints
extract_ordination
extract_phylo_signal
```

## What is not here

`extract_residual_split` (an OLRE-specific σ²_d/σ²_e/σ²_total decomposition)
and `extract_coevolution_modules`'s companion accessor are not implemented —
see [Post-fit tables and prediction](postfit-tables.md) for
`extract_coevolution_modules` itself, and
[Diagnostics and model comparison](diagnostics.md) for `getREsd`'s replacement
(`latent_score_sd`, on the [SE and profile machinery](se-profile-machinery.md)
page), which needed a rename rather than a straight port — R's `getREsd`
reads TMB random-effect blocks that GLLVModels.jl does not expose the same way.

See also: [Covariance & correlation](covariance-correlation.md) ·
[Confidence intervals](confidence-intervals.md) ·
[Derived confidence intervals](derived-confidence-intervals.md).
