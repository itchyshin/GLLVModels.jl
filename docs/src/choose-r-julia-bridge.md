# Choose R, Julia, or the bridge

```@raw html
<div class="gllvm-route gllvm-route--start">
  <div>
    <span class="gllvm-route__eyebrow">Pick a route</span>
    <p>Choose the language that fits your data and workflow. The packages overlap, but they do not offer the same models.</p>
  </div>
</div>
```

GLLVModels.jl is the matrix-first Julia companion to R `gllvmTMB`. The R package
remains the formula-first model surface and the richer applied article set.
The packages share core estimands; they do not offer identical workflows, and
they are not a menu of interchangeable Julia optimisers.

## Use R (`gllvmTMB`)

Start in R when you want the formula-first teaching route or the applied
article set.

- Get started: [R get-started guide](https://itchyshin.github.io/gllvmTMB/articles/gllvmTMB.html)
- What that route currently supports: [current limits](https://itchyshin.github.io/gllvmTMB/articles/current-limits.html)

Those limits belong to the R package. Calling Julia does not lift them.

## Use Julia (`GLLVModels.jl`)

Start in Julia when you already have a response matrix and want the
matrix-first companion. Responses are rows and sites are columns
($p \times n$). Only the families and post-fit tools described in the
documentation should be treated as available; overlap with R does not make the
two workflows interchangeable.

- First fit and the R ⟷ Julia conversion table: [Quick start](quickstart.md)
- What is currently available in both packages: [Capability parity](gllvmtmb-parity.md)

## Use the bridge (one-way R → Julia only)

The bridge is `gllvmTMB(..., engine = "julia")`. It sends a subset of
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
- R-site navigation, which is maintained separately.

For planned capability work, see the package [Roadmap](roadmap.md).
