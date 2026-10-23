# Low-rank block route of `temporal_marginal_nll` (latent, no unique, no
# composed terms): V = A (K ⊗ I_d) A' + sigma^2 I is evaluated through a
# (d x occasions) Cholesky. It must equal the dense block value and gradient.
using Test, GLLVModels, LinearAlgebra, ForwardDiff, StatsModels, Random
const GML = GLLVModels

@testset "temporal low-rank block route equals the dense block" begin
    rng = MersenneTwister(20261001)
    ns, nt, p = 3, 7, 5
    cols = (series=String[], trait=String[], occasion=Float64[], value=Float64[])
    for s in 1:ns, t in 1:nt, j in 1:p
        # series 3 drops a whole occasion so blocks differ in size
        s == 3 && t == 4 && continue
        push!(cols.series, "s$s"); push!(cols.trait, "t$j")
        push!(cols.occasion, Float64(t)); push!(cols.value, randn(rng))
    end
    y = Float64.(cols.value)
    for structure in (:ar1, :ou), d in (1,)
        term = temporal_latent(:(0 + trait | series), :occasion; d=d, structure=structure, unique=false)
        spec0 = GML._parse_temporal_term(term, cols; trait=:trait, response=:value, structure=Expr[])
        spec = GML._temporal_with_composition(spec0,
            GML._temporal_composition(spec0, Expr[], cols, y; unit=nothing, unit_obs=nothing))
        _, X, _ = GML._temporal_design(@formula(value ~ 0 + trait), cols)
        L = GML.TemporalLayout(size(X, 2), spec)
        theta = GML._temporal_start(y, X, L) .+ 0.1 .* randn(rng, L.total)
        rows = GML._temporal_series_rows(spec)
        # dense reference, one block at a time
        function dense(th)
            P = GML._temporal_cov_pieces(th, spec, L)
            res = y .- X * th[L.beta]
            nll = zero(eltype(th))
            for r in rows
                m = length(r)
                V = [GML._temporal_cov_entry(spec, P, r[i], r[j]) + (i == j ? P.sigma2 : 0.0)
                     for i in 1:m, j in 1:m]
                F = cholesky(Symmetric(V))
                z = F.L \ res[r]
                nll += (m * log(2pi) + logdet(F) + dot(z, z)) / 2
            end
            return nll
        end
        f(th) = GML.temporal_marginal_nll(th, y, X, spec; series_rows=rows)
        @test f(theta) ≈ dense(theta) atol = 1e-10 rtol = 1e-12
        @test ForwardDiff.gradient(f, theta) ≈ ForwardDiff.gradient(dense, theta) atol = 1e-8 rtol = 1e-9
    end
end
