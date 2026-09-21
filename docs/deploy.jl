using DocumenterVitepress

# Use DocumenterVitepress.deploydocs (NOT Documenter's): it flattens the
# VitePress build output (build/1/*) into the version root on gh-pages and
# rewrites the site base. This script is deliberately separate from make.jl so
# CI can audit generated reader text before any deployment.
DocumenterVitepress.deploydocs(;
    repo         = "github.com/itchyshin/GLLVModels.jl.git",
    target       = joinpath(@__DIR__, "build"),
    devbranch    = "main",
    branch       = "gh-pages",
    push_preview = true,
)
