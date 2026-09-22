# A first phylogenetic Gaussian model

```@raw html
<div class="gllvm-route gllvm-route--applied">
  <div>
    <span class="gllvm-route__eyebrow">Applied phylogenetic route</span>
    <p>Fit one continuous trait across a tree first; use the result as a variance-component description, not a complete evolutionary explanation.</p>
  </div>
</div>
```

This is the supported starting point for phylogenetic analysis in
`GLLVModels.jl`: `fit_phylo_gaussian` fits a Gaussian model for **one
continuous trait measured once per tree tip**, with the tree supplied as
Newick text. This example aligns named observations with the tree and
inspects the fitted mean and two variance components, phylogenetic and
residual, in a Gaussian Brownian-motion model. It is a univariate phylogenetic
model: it does not fit a multivariate trait matrix, the latent community axes
used in the [community abundance example](community-abundance.md), latent
functional modules, phylogenetic signal intervals, or ancestral states.

!!! warning "Tip order is part of the data"
    The response vector must have one value per tip and be in the tree's tip
    order (`phy.leaf_names`). Check that alignment before fitting; matching
    dimensions alone does not verify biological identity.

## The model and its scope

For a trait vector ``y`` over ``p`` species, the fitter uses

```math
y\sim\mathcal{N}\left(\mu\mathbf{1},\;
\sigma^2_{\mathrm{phy}}C+\sigma^2_{\mathrm{eps}}I\right),
```

where ``C`` is the Brownian-motion tip covariance implied by the supplied
tree. The fitter estimates a common mean ``\mu``, the multiplier
``\sigma^2_{\mathrm{phy}}`` of that covariance, and an independent residual
variance ``\sigma^2_{\mathrm{eps}}`` by maximum likelihood. There is no
multivariate response matrix or `K` argument in this interface. The two
variance components describe variation compatible with this model. They do
not prove that a trait is evolutionarily conserved, that a particular
ecological process caused the variation, or that another phylogenetic model
would give the same answer.

## Build a small tree and align the trait vector

```julia
using GLLVModels

# Four named tips and their branch lengths.
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
duplicate labels and missing or extra species before fitting. For a real
dataset, preserve the original tree, resolve those differences deliberately,
and document any pruning or exclusions rather than relying only on the checks
above.

`augmented_phy` accepts Newick text for binary trees with positive non-root
branch lengths and constructs the sparse phylogenetic representation used by
the fitter. The default `correlation = false` retains the branch-length
scale. The optional `correlation = true` rescales an ultrametric tree to unit
height; it changes the interpretation of the fitted phylogenetic variance
multiplier. Keep the tree and its scaling with your results. Four tips are
used here to show the interface, not to support reliable variance estimation.

## Fit the supported Gaussian model

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

`fit_phylo_gaussian` is a univariate model: its arguments are the phylogeny
first and the length-`p` trait vector second. The result stores variances
directly, not standard deviations. `fit.negll` is the **negative**
log-likelihood, so the log-likelihood is `-fit.negll`.

Check `fit.converged` and inspect the scale and plausibility of the estimates
before reporting them. If convergence fails, revisit the data and fitting
settings before interpreting the components. Convergence is a numerical
diagnostic, not evidence that Brownian motion, the supplied tree, or the
sampling process is correct.

## Interpret variance components carefully

`fit.σ²_phy` is the estimated variance attached to the supplied Brownian
phylogenetic covariance, while `fit.σ²_eps` is the model's independent
residual variance. The phylogenetic contribution to species ``i``'s marginal
variance is ``\sigma^2_{\mathrm{phy}}C_{ii}``, so the multiplier's magnitude
depends on the tree's branch-length scale: it is not automatically a marginal
variance or a proportion of total variance. On a non-ultrametric tree,
``C_{ii}`` can differ among tips. The residual component describes
independent variation under this model; it does not distinguish measurement
error from other unmodelled causes.

Their relative size is a compact description conditional on this trait, tree,
and Gaussian model. This vignette provides point estimates only: it does not
provide confidence intervals, a formal test of phylogenetic signal, ancestral
states, or a causal interpretation of either component. The fitted object
shown here contains the mean, variance estimates, likelihood, and optimization
diagnostics; those outputs do not by themselves provide uncertainty estimates
for this small dataset.

That boundary matters especially for small trees, uncertain branch lengths,
measurement error, non-Brownian evolution, and traits measured in different
environments. Those choices can materially change the fitted components.

## Next step

Use this route to establish a correctly aligned tree and trait vector, then fit
and inspect the basic Gaussian variance-component model. If the scientific
question requires multiple traits, repeated observations, latent covariance,
non-Gaussian responses, ancestral-state reconstruction, or uncertainty
statements, do not promote this example into that workflow. Specify the target
estimand, then consult [Structured dependence](../structured-dependence.md) and
[Choose R, Julia, or the bridge](../choose-r-julia-bridge.md) to select an
interface whose public surface and uncertainty support cover your design.
Those models require their own data layout and interpretation; changing this
vector to a matrix does not fit them.

For a basic community count ordination, see the
[Community abundance vignette](community-abundance.md).
