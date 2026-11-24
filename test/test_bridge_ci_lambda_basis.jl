using GLLVModels, Test, Random, LinearAlgebra

@testset "bridge CI Lambda basis vs loadings (refs #1158)" begin
    Random.seed!(33)
    p, n, d = 6, 80, 2
    Y = Float64.(rand(0:6, p, n)) .+ 0.0
    b = GLLVModels.bridge_fit(; y = Y, family = "poisson", d = d, options = Dict("ci_method" => "wald"))
    L = zeros(p, d)
    for i in findall(s -> occursin("ambda", s), b.ci_param_names)
        m = match(r"\[(\d+),(\d+)\]", b.ci_param_names[i])
        L[parse(Int, m[1]), parse(Int, m[2])] = b.ci_estimate[i]
    end
  # rotation-invariant block matches; raw Λ may differ
    @test maximum(abs.(L * L' .- b.loadings * b.loadings')) < 1e-10
end