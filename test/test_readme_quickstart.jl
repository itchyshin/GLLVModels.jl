# Runs the julia code blocks under the README's "Quick start" and
# "Confidence intervals" headings exactly as written (#536). The blocks are
# extracted from README.md, so the test cannot drift from the README.
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
