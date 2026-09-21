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
        "Start here" => [
            "What is a GLLVM?" => "index.md",
            "Fit your first model" => "quickstart.md",
            "Coming from R?" => "choose-r-julia-bridge.md",
            "Common pitfalls" => "pitfalls.md",
        ],
        "Choose a scientific question" => [
            "Traits and repeated outcomes" => "tutorial.md",
            "First phylogenetic Gaussian model" => "vignettes/phylogenetic-gllvm.md",
            "First community abundance model" => "vignettes/community-abundance.md",
            "Morphometrics" => "morphometrics.md",
        ],
        "Understand your results" => [
            "Working with a fit" => "working-with-a-fit.md",
            "Covariance & Correlation" => "covariance-correlation.md",
            "Confidence intervals" => "confidence-intervals.md",
            "Diagnostics and model comparison" => "diagnostics.md",
        ],
        "Tested models and limits" => [
            "What can I fit today?" => "what-can-i-fit-today.md",
            "Response families" => "response-families.md",
        ],
        "Function reference" => [
            "API reference" => "api.md",
            "Post-fit extractors" => "postfit-extractors.md",
            "Post-fit tables and prediction" => "postfit-tables.md",
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

# CI builds with GLLVM_DOCS_DEPLOY=false, audits the generated reader surface,
# then runs docs/deploy.jl. Keeping deployment separate prevents an unchecked
# rendered page (including expanded public docstrings) from reaching gh-pages.
if !("--local" in ARGS) && get(ENV, "GLLVM_DOCS_DEPLOY", "true") == "true"
    include("deploy.jl")
end # --local builds never call deploydocs
