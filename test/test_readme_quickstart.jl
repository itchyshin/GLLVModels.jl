# Runs the julia code blocks under the README's "Quick start" and
# "Confidence intervals" headings exactly as written (#536). The blocks are
# extracted from README.md, so the test cannot drift from the README.
# #1007: every ```julia example except the Install `Pkg.add` block must also
# run in a fresh module (a fresh session).
using Test
using GLLVModels

function _readme_blocks(readme::AbstractString, heading::AbstractString)
    lines = readlines(readme)
    start = findfirst(==("## " * heading), lines)
    start === nothing && error("README heading not found: $heading")
    blocks = String[]
    inblock = false
    buf = String[]
    for line in lines[(start + 1):end]
        if !inblock && startswith(line, "## ")
            break
        elseif !inblock && startswith(line, "```julia")
            inblock = true
            empty!(buf)
        elseif inblock && startswith(line, "```")
            inblock = false
            push!(blocks, join(buf, "\n"))
        elseif inblock
            # Package installation is covered by the Install section; skip it here.
            occursin(r"^\s*(using Pkg|Pkg\.add)", line) || push!(buf, line)
        end
    end
    return blocks
end

function _readme_julia_blocks(readme::AbstractString)
    lines = readlines(readme)
    blocks = String[]
    inblock = false
    buf = String[]
    skip_install = false
    for line in lines
        if !inblock && startswith(line, "```julia")
            inblock = true
            skip_install = false
            empty!(buf)
        elseif inblock && startswith(line, "```")
            inblock = false
            skip_install || push!(blocks, join(buf, "\n"))
        elseif inblock
            if occursin(r"^\s*(using Pkg|Pkg\.add)", line)
                skip_install = true
            else
                push!(buf, line)
            end
        end
    end
    return blocks
end

@testset "#536 README Quick start runs" begin
    readme = joinpath(@__DIR__, "..", "README.md")
    mod = Module(:ReadmeQuickstart)
    for heading in ("Quick start", "Confidence intervals")
        blocks = _readme_blocks(readme, heading)
        @test !isempty(blocks)
        for code in blocks
            @test (Base.include_string(mod, code); true)
        end
    end
    fit = getfield(mod, :fit)
    @test fit.converged
    @test size(fit.Λ) == (20, 2)
    @test length(fit.ψ²) == 20
    @test isfinite(fit.loglik)
end

@testset "#1007 README julia blocks run in a fresh session" begin
    readme = joinpath(@__DIR__, "..", "README.md")
    blocks = _readme_julia_blocks(readme)
    @test length(blocks) >= 4
    for (i, code) in enumerate(blocks)
        mod = Module(Symbol("ReadmeFresh", i))
        @test (Base.include_string(mod, code); true)
    end
end
