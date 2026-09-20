# Phylogenetic analysis: one continuous trait

```@raw html
<div class="gllvm-route gllvm-route--applied">
  <div>
    <span class="gllvm-route__eyebrow">Applied phylogenetic route</span>
    <p>Match one trait value to each tree tip and estimate phylogenetic and residual variance.</p>
  </div>
</div>
```

`fit_phylo_gaussian` fits a Gaussian model for **one continuous trait measured
once per tree tip**. This example aligns named observations with a Newick
tree and inspects the fitted mean and two variance components. This is a
univariate phylogenetic model; it does not add the latent community axes used
in the [community abundance example](community-abundance.md).

## The model

For a trait vector ``y`` over ``p`` species,

```math
y\sim\mathcal{N}\left(\mu\mathbf{1},\;
\sigma^2_{\mathrm{phy}}C+\sigma^2_{\mathrm{eps}}I\right).
```

Here ``C`` is the Brownian-motion tip covariance implied by the supplied
tree. The fitter estimates a common mean ``\mu``, the multiplier
``\sigma^2_{\mathrm{phy}}`` of that covariance, and independent residual
variance ``\sigma^2_{\mathrm{eps}}`` by maximum likelihood. There is no
multivariate response matrix or `K` argument in this interface.

## Align the trait values with the tree

```julia
using GLLVModels

newick = "((sp1:1,sp2:1):1,(sp3:1,sp4:1):1);"
phy = augmented_phy(newick)

# Labels, rather than the input table's row order, determine the match.
species = ["sp3", "sp1", "sp4", "sp2"]
trait = [0.9, 1.2, 1.1, 1.6]

@assert length(species) == length(trait)
@assert length(unique(species)) == length(species)
@assert length(unique(phy.leaf_names)) == phy.n_leaves
@assert Set(species) == Set(phy.leaf_names)
trait_by_species = Dict(zip(species, trait))
y = [trait_by_species[name] for name in phy.leaf_names]
@assert all(isfinite, y)

collect(zip(phy.leaf_names, y))
```

The response vector follows `phy.leaf_names`, the parser's tip order. A
length check alone cannot detect a biological mismatch. The assertions catch
duplicate labels and missing or extra species before fitting. With real data,
resolve those differences deliberately and record any pruning or exclusions.

`augmented_phy` accepts binary trees with positive non-root branch lengths.
The default `correlation = false` retains the branch-length scale. The
optional `correlation = true` rescales an ultrametric tree to unit height;
it changes the interpretation of the fitted phylogenetic variance multiplier.
Keep the tree and its scaling with your results. Four tips are used here to
show the interface, not to support reliable variance estimation.

## Fit and inspect the point estimates

```julia
fit = fit_phylo_gaussian(phy, y)

(
    converged = fit.converged,
    mean = fit.μ,
    phylogenetic_variance_multiplier = fit.σ²_phy,
    residual_variance = fit.σ²_eps,
    negative_loglikelihood = fit.negll,
    iterations = fit.iterations,
)
```

The phylogeny comes first and the trait vector second. The result stores
variances directly, not standard deviations. `fit.negll` is the **negative**
log-likelihood, so the log-likelihood is `-fit.negll`.

Check `fit.converged` and the scale and plausibility of the estimates. If
convergence fails, revisit the data and fitting settings before interpreting
the components. Even a converged result does not establish that Brownian
motion, the supplied tree, or the observation model is appropriate.

## What the variance components mean

The phylogenetic contribution to species ``i``'s marginal variance is
``\sigma^2_{\mathrm{phy}}C_{ii}``. Consequently, the multiplier's magnitude
depends on the tree's branch-length scale: it is not automatically a marginal
variance or a proportion of total variance. On a non-ultrametric tree,
``C_{ii}`` can differ among tips.

The residual component describes independent variation under this model;
it does not distinguish measurement error from other unmodelled causes.
Neither component alone establishes a causal evolutionary or environmental
explanation.

This example supplies **point estimates only**. It does not construct
confidence intervals, test phylogenetic signal, or reconstruct ancestral
states. The fitted object shown here contains the mean, variance estimates,
likelihood, and optimization diagnostics; those outputs do not by themselves
provide uncertainty estimates for this small dataset.

## Next steps

For multiple traits, repeated observations, or non-Gaussian responses, consult
[Structured dependence](../structured-dependence.md) and
[Choose R, Julia, or the bridge](../choose-r-julia-bridge.md) to select an
interface that covers your design. Those models require their own data layout
and interpretation; changing this vector to a matrix does not fit them.
