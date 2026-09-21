# Choose R, Julia, or the bridge

```@raw html
<div class="gllvm-route gllvm-route--start">
  <div>
    <span class="gllvm-route__eyebrow">Pick a route</span>
    <p>Choose the package that best matches your data and the way you want to work. The packages overlap, but they do not offer the same models.</p>
  </div>
</div>
```

Both packages ask which responses vary together. `gllvmTMB` is an R package
with a formula-based interface and a larger set of applied guides.
GLLVModels.jl is a Julia package that starts from a response matrix. They are
related tools, not drop-in replacements for one another.

## Use R (`gllvmTMB`)

Start in R when you want the formula-first teaching route or the applied
article set.

- Get started: [R get-started guide](https://itchyshin.github.io/gllvmTMB/articles/gllvmTMB.html)
- What that route currently supports: [current limits](https://itchyshin.github.io/gllvmTMB/articles/current-limits.html)
- Spatially structured continuous traits: [multivariate spatial models](https://itchyshin.github.io/gllvmTMB/articles/spatial-models.html)
- Repeated multivariate measurements: [temporal covariance](https://itchyshin.github.io/gllvmTMB/articles/temporal-ar1.html)
- Repeated survey visits in the experimental integrated-SDM route: [what repeated visits add](https://itchyshin.github.io/gllvmTMB/articles/integrated-repeated-visits.html)

Those limits belong to the R package. Calling Julia does not lift them.

## Use Julia (`GLLVModels.jl`)

Start in Julia when you already have a response matrix and want the
matrix-first companion. Responses are rows and sites are columns
($p \times n$). Only the families and post-fit tools described in the
documentation should be treated as available; overlap with R does not make the
two workflows interchangeable. Begin with the [Quick start](quickstart.md), or
use [What can I fit today?](what-can-i-fit-today.md) to choose a documented
route.

### Move one R matrix into Julia

1. Install Julia 1.10 or later, then run `using Pkg` and
   `Pkg.add(url = "https://github.com/itchyshin/GLLVModels.jl")`.
2. If your R response matrix has sites in rows and species in columns
   (`n × p`), transpose it before fitting in Julia (`Y'` gives responses in
   rows and observations in columns).
3. Run the [Quick start](quickstart.md) before adapting the example to your
   data.

- What is currently available in both packages, and a detailed R–Julia
  capability comparison when you need to compare a particular model:
  [Capability parity](gllvmtmb-parity.md)

## Use the bridge (one-way R → Julia only)

The bridge is optional: the default `gllvmTMB` fitting workflow runs in R
without Julia. To use the bridge, set `engine = "julia"` in `gllvmTMB(...)`.
It sends a subset of
cross-sectional reduced-rank models from R into Julia through JuliaCall. It
is one-way: **R → Julia**. It does not run Julia models back through R, and
it does not cover phylogeny, spatial, animal, kernel, or iSDM structure, nor
the full `traits()` formula grammar.

For the admitted bridge surface versus the wider engine, see
[Capability parity](gllvmtmb-parity.md). The bridge rejects unsupported
structures explicitly rather than silently changing the model.

## What this page does not claim

- Universal parity, or that every R workflow has an identical Julia counterpart.
- Calibrated interval coverage on either side.

For planned capability work, see the package [Roadmap](roadmap.md).
