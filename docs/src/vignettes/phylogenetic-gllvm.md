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
`GLLVModels.jl`: a single continuous trait measured once for each tip and a
tree supplied as Newick text. It estimates phylogenetic and residual variance
in a Gaussian Brownian-motion model. It does not fit a multivariate trait
matrix, latent functional modules, phylogenetic signal intervals, or ancestral
states.

!!! warning "Tip order is part of the data"
    The response vector must have one value per tip and be in the tree's tip
    order. Check that alignment before fitting; matching dimensions alone does
    not verify biological identity.

## The model and its scope

For a trait vector `y` over `p` species, the fitter uses

```math
y \sim \mathcal{N}\left(\mu\mathbf{1},\; \sigma^2_{\mathrm{eps}} I +
\sigma^2_{\mathrm{phy}}\Sigma_{\mathrm{phy}}\right),
```

where `\Sigma_phy` is the Brownian-motion tip covariance implied by the
supplied tree. The two variance components describe variation compatible with
this model. They do not prove that a trait is evolutionarily conserved, that a
particular ecological process caused the variation, or that another
phylogenetic model would give the same answer.

## Build a small tree and trait vector

```julia
using GLLVModels, Random

Random.seed!(24)

# Four named tips and their branch lengths. Put your trait vector in this tip order.
phy = augmented_phy("((sp1:1,sp2:1):1,(sp3:1,sp4:1):1);")
y = [1.2, 1.6, 0.9, 1.1]

length(y) == phy.n_leaves
```

`augmented_phy` accepts Newick text and constructs the sparse phylogenetic
representation used by the fitter. For a real dataset, preserve the original
tree, document any pruning, and make the tip-to-trait match explicit rather
than relying only on the last line above.

## Fit the supported Gaussian model

```julia
fit = fit_phylo_gaussian(phy, y)

fit.converged
fit.μ
fit.σ²_phy
fit.σ²_eps
```

`fit_phylo_gaussian` is a univariate model: its arguments are the phylogeny
first and the length-`p` trait vector second. Check `fit.converged` and inspect
the scale and plausibility of the estimates before reporting them. Convergence
is a numerical diagnostic, not evidence that Brownian motion, the tree, or the
sampling process is correct.

## Interpret variance components carefully

`fit.σ²_phy` is the estimated variance attached to the supplied Brownian
phylogenetic covariance, while `fit.σ²_eps` is the model's independent
residual variance. Their relative size is a compact description conditional on
this trait, tree, and Gaussian model. This vignette provides point estimates
only: it does not provide confidence intervals, a formal test of phylogenetic
signal, or a causal interpretation of either component.

That boundary matters especially for small trees, uncertain branch lengths,
measurement error, non-Brownian evolution, and traits measured in different
environments. Those choices can materially change the fitted components.

## Next step

Use this route to establish a correctly aligned tree and trait vector, then fit
and inspect the basic Gaussian variance-component model. If the scientific
question requires multiple traits, latent covariance, non-Gaussian responses,
ancestral-state reconstruction, or uncertainty statements, do not promote this
example into that workflow. Specify the target estimand and choose a method
whose public interface and uncertainty support cover it.

For a basic community count ordination, see the
[Community abundance vignette](community-abundance.md).
