using Documenter
using DocumenterVitepress
using GLLVModels

# Documenter validates local image links before DocumenterVitepress moves the
# generated source tree into `build/.documenter`. Seed its normal build asset
# path so that validation sees the same static images VitePress later serves.
let source_assets = joinpath(@__DIR__, "src", "assets"),
    build_assets = joinpath(@__DIR__, "build", "assets")
    mkpath(build_assets)
    for asset in readdir(source_assets; join = true)
        cp(asset, joinpath(build_assets, basename(asset)); force = true)
    end
end

makedocs(;
    root = @__DIR__,
    remotes = "--local" in ARGS ? nothing : Dict(),
    sitename = "GLLVModels.jl",
    authors  = "Shinichi Nakagawa",
    modules  = [GLLVModels],
    format   = MarkdownVitepress(
        repo      = "github.com/itchyshin/GLLVModels.jl",
        devbranch = "main",
        devurl    = "dev",
    ),
    pages    = [
        "Start: choose an analysis route" => [
            "Overview: three biology questions" => "index.md",
            "Choose R, Julia, or the bridge"   => "choose-r-julia-bridge.md",
            "General latent-variable first fit" => "quickstart.md",
            "Tutorial: model-interface tour"    => "tutorial.md",
            "Common Pitfalls"                   => "pitfalls.md",
        ],
        "Phylogenetic comparative models" => [
            "First phylogenetic Gaussian model" => "vignettes/phylogenetic-gllvm.md",
            "Morphometrics"                     => "morphometrics.md",
        ],
        "Community and species-distribution models" => [
            "First community abundance model" => "vignettes/community-abundance.md",
            "Structured Dependence"         => "structured-dependence.md",
            "Joint Named Grouping"          => "grouped-models.md",
        ],
        "General latent-variable analysis" => [
            "Mathematical Model"       => "model.md",
            "Response Families"        => "response-families.md",
            "Tweedie Power"            => "tweedie-power.md",
            "Student-t Parity Limits"  => "studentt-parity.md",
            "Working with a Fit"       => "working-with-a-fit.md",
            "Covariance & Correlation" => "covariance-correlation.md",
            "Structured-Term Fitting"  => "structured-term-fitting.md",
            "Confidence Intervals"     => "confidence-intervals.md",
            "Derived Confidence Intervals" => "derived-confidence-intervals.md",
        ],
        "Reference, development, and benchmarks" => [
            "API Reference"                  => "api.md",
            "Post-Fit Extractors"            => "postfit-extractors.md",
            "Post-Fit Tables & Prediction"   => "postfit-tables.md",
            "Diagnostics & Model Comparison" => "diagnostics.md",
            "Confidence-interval machinery (technical)" => "se-profile-machinery.md",
            "Precision bridge (development)"             => "precision-bridge-development.md",
            "Low-level Reference (technical)"            => "low-level-reference.md",
            "Benchmarks"                                  => "benchmarks.md",
            "Comparison vs gllvmTMB"                      => "comparison.md",
            "Capability Parity"                           => "gllvmtmb-parity.md",
            "Roadmap"                                     => "roadmap.md",
            "Changelog"                                   => "changelog.md",
        ],
    ],
    warnonly = false,
)

# A standalone local preview still requests this script from the version picker.
# Deployment supplies its own version index; do not write one for deployed builds.
if "--local" in ARGS
    local_sites = filter(path -> isdir(path) && isfile(joinpath(path, "index.html")),
                         readdir(joinpath(@__DIR__, "build"); join=true))
    isempty(local_sites) && error("No rendered local documentation site found")
    for site in local_sites
        write(joinpath(site, "versions.js"), "var DOC_VERSIONS = [\"dev\"];\n")
    end
end

# Use DocumenterVitepress.deploydocs (NOT Documenter's): it flattens the Vitepress
# build output (build/1/*) into the version root on gh-pages and rewrites the
# site `base`. Plain Documenter.deploydocs deploys build/ verbatim, which lands
# the site under dev/1/ with base=/dev/ — every asset/nav link then 404s.
if !("--local" in ARGS)
    DocumenterVitepress.deploydocs(;
    repo         = "github.com/itchyshin/GLLVModels.jl.git",
    target       = joinpath(@__DIR__, "build"),
    devbranch    = "main",
    branch       = "gh-pages",
    push_preview = true,
    )
end # --local builds never call deploydocs
