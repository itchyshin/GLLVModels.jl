using Test, GLLVModels

# #734 / #738: the docs snippets named in the issues must be self-contained, and the
# Gaussian `confint` non-PD wording must match `_gaussian_wald_covariance_from_hessian`.
const _DOCS = joinpath(@__DIR__, "..", "docs", "src")

function _julia_blocks(file)
    txt = read(joinpath(_DOCS, file), String)
    return [m.captures[1] for m in eachmatch(r"```julia\n(.*?)```"s, txt)]
end

@testset "docs snippets are self-contained (#734)" begin
    wf = join(_julia_blocks("working-with-a-fit.md"), "\n")
    @test occursin("using Distributions", wf)
    @test occursin(r"^\s*Yb\s*="m, wf)

    sd = join(_julia_blocks("structured-dependence.md"), "\n")
    @test !occursin(r"=\s*\.\.\.\s*(#.*)?$"m, sd)
    @test !occursin("my_tree.tre", sd)

    tp = _julia_blocks("temporal.md")
    @test any(b -> occursin(r"^\s*data\s*="m, b), tp)
    first_use = findfirst(b -> occursin("fit_temporal_gllvm(data", b), tp)
    first_def = findfirst(b -> occursin(r"^\s*data\s*="m, b), tp)
    @test first_def !== nothing && first_use !== nothing && first_def < first_use

    pb = _julia_blocks("precision-bridge-development.md")
    j = findfirst(b -> occursin("fit_gllvm(Y;", b), pb)
    @test j !== nothing
    for name in ("Y", "precision", "species_id", "cluster")
        @test occursin(Regex("^\\s*$name\\s*=", "m"), pb[j])
    end
end

@testset "Gaussian confint non-PD wording (#738)" begin
    f = GLLVModels._gaussian_wald_covariance_from_hessian
    # Indefinite but invertible Hessian: Cholesky fails, so every SE and Sigma are NaN.
    V, se, pd = f([2.0 0.0; 0.0 -1.0], 2)
    @test !pd && all(isnan, se) && all(isnan, V)
    txt = read(joinpath(_DOCS, "confidence-intervals.md"), String)
    @test occursin("fails a Cholesky factorisation", txt)
    @test !occursin("is unchanged and still reports all-NaN on a non-PD Hessian", txt)
end
