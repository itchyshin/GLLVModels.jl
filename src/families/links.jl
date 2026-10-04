# Response families + link functions.
#
# Response families reuse Distributions.jl distribution types as markers — the
# GLM.jl convention: `family = Normal()` (Gaussian, the default) or `Binomial()`
# (binary), `Poisson()`, … The marginal log-likelihood dispatches on the family
# type via Julia multiple dispatch (Gaussian → closed-form marginal;
# non-Gaussian → Laplace), with no hardcoded family switch. GLLVModels defines only
# the link types below — Distributions provides the distributions, not the links.

abstract type Link end
"""Logit link: `g(μ) = log(μ / (1 - μ))`, inverse `g⁻¹(η) = 1 / (1 + exp(-η))`. Canonical link for `Binomial`. Conventional default (not canonical) for `Beta`."""
struct LogitLink    <: Link end
"""Probit link: `g(μ) = Φ⁻¹(μ)`, inverse `g⁻¹(η) = Φ(η)` (standard-normal CDF)."""
struct ProbitLink   <: Link end
"""Complementary log-log link: `g(μ) = log(-log(1 - μ))`, inverse `g⁻¹(η) = 1 - exp(-exp(η))`."""
struct CLogLogLink  <: Link end
"""Identity link: `g(μ) = μ`. Canonical link for `Normal`."""
struct IdentityLink <: Link end
"""Log link: `g(μ) = log(μ)`, inverse `g⁻¹(η) = exp(η)`. Canonical link for `Poisson`. Default (not canonical) for `Gamma` and `NegativeBinomial`."""
struct LogLink      <: Link end

"""
    linkinv(link, η) -> μ

Inverse link `g⁻¹`: map the linear predictor `η` to the mean `μ`.
"""
linkinv(::LogitLink, η)    = inv(one(η) + exp(-η))
linkinv(::ProbitLink, η)   = cdf(Normal(), η)
linkinv(::CLogLogLink, η)  = -expm1(-exp(η))
linkinv(::IdentityLink, η) = η
linkinv(::LogLink, η)      = exp(η)

"""
    mu_eta(link, η) -> dμ/dη

Derivative of the mean with respect to the linear predictor (numerically safe
at large |η|).
"""
mu_eta(::LogitLink, η)    = (e = exp(-abs(η)); e / (one(η) + e)^2)
mu_eta(::ProbitLink, η)   = pdf(Normal(), η)
mu_eta(::CLogLogLink, η)  = exp(η - exp(η))
mu_eta(::IdentityLink, η) = one(η)
mu_eta(::LogLink, η)      = exp(η)

"""
    linkfun(link, μ) -> η

Link `g`: map the mean `μ` to the linear predictor `η` (used for initialisation).
"""
linkfun(::LogitLink, μ)    = log(μ / (one(μ) - μ))
linkfun(::ProbitLink, μ)   = quantile(Normal(), μ)
linkfun(::CLogLogLink, μ)  = log(-log1p(-μ))
linkfun(::IdentityLink, μ) = μ
linkfun(::LogLink, μ)      = log(μ)

"""
    default_link(family) -> Link

Default link for a response family. Methods in this file:

- `Normal` → `IdentityLink` (canonical)
- `Binomial` → `LogitLink` (canonical)
- `Poisson` → `LogLink` (canonical)
- `NegativeBinomial` → `LogLink` (default; not canonical)
- `Beta` → `LogitLink` (conventional default; not a canonical-link GLM)
- `Gamma` → `LogLink` (default; not canonical)
"""
default_link(::Normal)   = IdentityLink()
default_link(::Binomial) = LogitLink()
default_link(::Poisson)  = LogLink()
default_link(::NegativeBinomial) = LogLink()
default_link(::Beta)     = LogitLink()
default_link(::Gamma)    = LogLink()
